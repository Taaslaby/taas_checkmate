use std::collections::HashMap;

use crate::api::neuroml::parse::{Ir, SpikeGenKind, Synapse};
use crate::api::neuroml::StimulusKind;

/// Regular spike train for `spikeGenerator`: first spike at `period_ms`, then
/// every `period_ms` until `duration_ms` (the NeuroML2 semantics).
fn regular_spike_times(period_ms: f32, duration_ms: f32) -> Vec<f32> {
    if !period_ms.is_finite() || period_ms <= 0.0 || !duration_ms.is_finite() {
        return Vec::new();
    }
    let mut times = Vec::new();
    let mut t = period_ms;
    while t < duration_ms {
        times.push(t);
        t += period_ms;
    }
    times
}

/// Deterministic SplitMix64 generator — the whole ISI schedule is precomputed
/// at compile time from a seed, so the runtime stays stateless (no interior
/// mutability in `injections_at`), the same seed reproduces the same train,
/// and CPU/GPU/reruns agree.
struct SeededRng(u64);

impl SeededRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Spike train for `spikeGeneratorRandom`: uniform ISI in `[min_isi, max_isi]`
/// (LEMS `isi = minISI + MSEC·random((maxISI−minISI)/MSEC)`), each next spike
/// `isi` after the **last planned** time, not the current one. Seeded → the
/// same model + seed always yields the same deterministic train.
fn random_isi_spike_times(
    min_isi_ms: f32,
    max_isi_ms: f32,
    duration_ms: f32,
    seed: u64,
) -> Result<Vec<f32>, String> {
    if !min_isi_ms.is_finite()
        || !max_isi_ms.is_finite()
        || min_isi_ms < 0.0
        || max_isi_ms < min_isi_ms
    {
        return Err(format!(
            "spikeGeneratorRandom: need 0 <= minISI <= maxISI, got [{min_isi_ms}, {max_isi_ms}]"
        ));
    }
    let span = max_isi_ms - min_isi_ms;
    let mut rng = SeededRng::new(seed);
    let mut times = Vec::new();
    let mut t = 0.0f32;
    for _ in 0..10_000_000 {
        let isi = min_isi_ms + span * rng.unit();
        if isi <= 0.0 || !isi.is_finite() {
            break; // pathological zero-ISI guard (minISI == 0 draws)
        }
        t += isi;
        if t >= duration_ms {
            break;
        }
        times.push(t);
    }
    Ok(times)
}

/// Spike train for `spikeGeneratorRefPoisson`: maximum-entropy ISI over
/// `[minimum_isi, ∞)` with mean `1/rate_hz` (LEMS `isi = minimumISI −
/// (averageIsi−minimumISI)·log(random(1))`). Requires `averageIsi > minimumISI`
/// (the mean must exceed the support bound). Seeded → deterministic.
fn ref_poisson_spike_times(
    rate_hz: f32,
    min_isi_ms: f32,
    duration_ms: f32,
    seed: u64,
) -> Result<Vec<f32>, String> {
    if !rate_hz.is_finite() || rate_hz <= 0.0 || !min_isi_ms.is_finite() || min_isi_ms < 0.0 {
        return Err(format!(
            "spikeGeneratorRefPoisson: need rate > 0 and minimumISI >= 0, got {rate_hz} Hz, {min_isi_ms} ms"
        ));
    }
    let avg_isi_ms = 1000.0 / rate_hz;
    if avg_isi_ms <= min_isi_ms {
        return Err(format!(
            "spikeGeneratorRefPoisson: average ISI ({avg_isi_ms:.2} ms) must exceed minimumISI ({min_isi_ms} ms)"
        ));
    }
    let mut rng = SeededRng::new(seed);
    let mut times = Vec::new();
    let mut t = 0.0f32;
    for _ in 0..10_000_000 {
        // ln(1−u) with u ∈ [0,1) avoids log(0) = +∞.
        let isi = min_isi_ms - (avg_isi_ms - min_isi_ms) * (1.0 - rng.unit()).ln();
        if isi <= 0.0 || !isi.is_finite() {
            break;
        }
        t += isi;
        if t >= duration_ms {
            break;
        }
        times.push(t);
    }
    Ok(times)
}

/// Resolve one input-generator id into `(kind, start_ms, end_ms, amplitude)`
/// stimuli (LEMS formulas from `NeuroML2CoreTypes/Inputs.xml`).
///
/// * `pulseGenerator` → one `Current` window;
/// * `sineGenerator` → `[Sine]` window, `i = w·A·sin(phase + 2π(t−delay)/period)`;
/// * `rampGenerator` → the `[Ramp]` window plus, when `baselineAmplitude ≠ 0`,
///   constant-current stimuli for `[0, delay)` and `[delay+duration, ∞)` (LEMS
///   holds the baseline outside the window);
/// * `compoundInput` → recurses into its children; each child becomes its own
///   stimulus, scaled by the compound's `weight`;
/// * `timedSynapticInput` → `SpikeTimes` pulses of `weight·gbase` at each
///   listed spike time (the synapse's kinetics are not simulated for direct
///   inputs — delta-pulse lowering, like `spikeArray`).
///
/// Returns an empty `Vec` for unknown ids (the caller turns it into an error).
pub(crate) fn resolve_input_gen(
    gen_id: &str,
    ir: &Ir,
    synapses: &HashMap<&str, &Synapse>,
    duration_ms: f32,
    scale: f32,
    seed: &mut usize,
    depth: usize,
) -> Result<Vec<(StimulusKind, f32, f32, f32)>, String> {
    if depth > 8 {
        return Err("compoundInput nesting too deep (cycle?)".to_string());
    }
    let mut next_seed = || {
        let s = *seed;
        *seed += 1;
        s as u32
    };
    if let Some(pg) = ir.pulse_gens.iter().find(|g| g.id == gen_id) {
        Ok(vec![(
            StimulusKind::Current,
            pg.delay_ms,
            pg.delay_ms + pg.duration_ms,
            pg.amplitude * scale,
        )])
    } else if let Some(ps) = ir.poisson_synapses.iter().find(|p| p.id == gen_id) {
        Ok(vec![(
            StimulusKind::Poisson {
                rate_hz: ps.rate_hz,
                seed: next_seed(),
            },
            0.0,
            f32::INFINITY,
            scale,
        )])
    } else if let Some(tp) = ir
        .transient_poisson_synapses
        .iter()
        .find(|p| p.id == gen_id)
    {
        Ok(vec![(
            StimulusKind::Poisson {
                rate_hz: tp.rate_hz,
                seed: next_seed(),
            },
            tp.delay_ms,
            tp.delay_ms + tp.duration_ms,
            scale,
        )])
    } else if let Some(sa) = ir.spike_arrays.iter().find(|g| g.id == gen_id) {
        Ok(vec![(
            StimulusKind::SpikeTimes {
                times_ms: sa.times_ms.clone(),
            },
            0.0,
            f32::INFINITY,
            scale,
        )])
    } else if let Some(r) = ir.spike_array_refs.iter().find(|g| g.id == gen_id) {
        let arr = ir
            .spike_arrays
            .iter()
            .find(|g| g.id == r.array)
            .ok_or_else(|| {
                format!(
                    "spikeArrayRef `{}` references unknown spikeArray `{}`",
                    r.id, r.array
                )
            })?;
        Ok(vec![(
            StimulusKind::SpikeTimes {
                times_ms: arr.times_ms.clone(),
            },
            0.0,
            f32::INFINITY,
            scale,
        )])
    } else if let Some(sg) = ir.spike_gens.iter().find(|g| g.id == gen_id) {
        match sg.kind {
            SpikeGenKind::Regular { period_ms } => Ok(vec![(
                StimulusKind::SpikeTimes {
                    times_ms: regular_spike_times(period_ms, duration_ms),
                },
                0.0,
                f32::INFINITY,
                scale,
            )]),
            SpikeGenKind::Poisson { rate_hz } => Ok(vec![(
                StimulusKind::Poisson {
                    rate_hz,
                    seed: next_seed(),
                },
                0.0,
                f32::INFINITY,
                scale,
            )]),
            // Random/refractory-ISI generators: the full schedule is
            // precomputed from the seed at compile time → deterministic
            // `SpikeTimes` (runtime stays stateless).
            SpikeGenKind::Random {
                min_isi_ms,
                max_isi_ms,
            } => Ok(vec![(
                StimulusKind::SpikeTimes {
                    times_ms: random_isi_spike_times(
                        min_isi_ms,
                        max_isi_ms,
                        duration_ms,
                        next_seed() as u64,
                    )?,
                },
                0.0,
                f32::INFINITY,
                scale,
            )]),
            SpikeGenKind::RefPoisson {
                rate_hz,
                min_isi_ms,
            } => Ok(vec![(
                StimulusKind::SpikeTimes {
                    times_ms: ref_poisson_spike_times(
                        rate_hz,
                        min_isi_ms,
                        duration_ms,
                        next_seed() as u64,
                    )?,
                },
                0.0,
                f32::INFINITY,
                scale,
            )]),
        }
    } else if let Some(s) = ir.sine_gens.iter().find(|g| g.id == gen_id) {
        Ok(vec![(
            StimulusKind::Sine {
                period_ms: s.period_ms,
                phase: s.phase,
            },
            s.delay_ms,
            s.delay_ms + s.duration_ms,
            s.amplitude * s.weight * scale,
        )])
    } else if let Some(r) = ir.ramp_gens.iter().find(|g| g.id == gen_id) {
        let w = r.weight * scale;
        let mut v = Vec::with_capacity(3);
        if r.baseline_amp != 0.0 {
            v.push((StimulusKind::Current, 0.0, r.delay_ms, r.baseline_amp * w));
            v.push((
                StimulusKind::Current,
                r.delay_ms + r.duration_ms,
                f32::INFINITY,
                r.baseline_amp * w,
            ));
        }
        v.push((
            StimulusKind::Ramp {
                start_amp: r.start_amp * w,
                finish_amp: r.finish_amp * w,
            },
            r.delay_ms,
            r.delay_ms + r.duration_ms,
            0.0,
        ));
        Ok(v)
    } else if let Some(c) = ir.compounds.iter().find(|g| g.id == gen_id) {
        let mut v = Vec::new();
        for child in &c.children {
            for (kind, start, end, amp) in
                resolve_input_gen(child, ir, synapses, duration_ms, scale, seed, depth + 1)?
            {
                v.push((kind, start, end, amp * c.weight));
            }
        }
        Ok(v)
    } else if let Some(t) = ir.timed_syn_inputs.iter().find(|g| g.id == gen_id) {
        let gbase = synapses
            .get(t.synapse.as_str())
            .and_then(|s| s.gbase)
            .unwrap_or(1.0);
        Ok(vec![(
            StimulusKind::SpikeTimes {
                times_ms: t.times_ms.clone(),
            },
            0.0,
            f32::INFINITY,
            t.weight * gbase * scale,
        )])
    } else {
        Ok(Vec::new())
    }
}
