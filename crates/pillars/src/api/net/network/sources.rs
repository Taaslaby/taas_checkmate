/// Deterministic, stateless pseudo-Poisson roll: a uniform `[0, 1)` value
/// hashed from `(seed, stream, step)`, so the same seed reproduces the same
/// spike train every run without mutable RNG state (CPU/GPU/repeated runs).
/// SplitMix64 finaliser over a mixed key.
pub(crate) fn poisson_spike(seed: u32, stream: u32, step: u64, p: f32) -> bool {
    let mut z = ((seed as u64) << 32) | (stream as u64);
    z ^= step.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let u = (z >> 40) as f32 / (1u64 << 24) as f32;
    u < p
}

/// Runtime state of one external Poisson spike source (e.g. NeuroML
/// `poissonFiringSynapse` used as a presynaptic population). Fires at per-step
/// probability `p = rate·dt` and scatters its outgoing synapses through the
/// same delay/kinetics machinery as a real neuron's spikes.
pub(crate) struct PoissonSourceRuntime {
    /// Per-step spike probability (`rate_hz · dt_s`), clamped to `[0, 1]`.
    pub(crate) p: f32,
    pub(crate) seed: u32,
    /// Outgoing synapses: `(post_global, weight, delay, syn_type, plast_type)`.
    pub(crate) outgoing: Vec<(u32, f32, u8, u8, u8)>,
    /// Per-outgoing Tsodyks–Markram state (parallel to `outgoing`).
    pub(crate) r: Vec<f32>,
    pub(crate) u: Vec<f32>,
    pub(crate) last: Vec<u32>,
}

/// A sigmoid `gradedSynapse` profile — a **stateful** analog conductance whose
/// activation follows the presynaptic membrane potential:
///
/// ```text
/// inf(vpre) = 1 / (1 + exp(-(vpre − Vth)/delta))
/// tau(vpre) = 1 / (k · (1 + exp((vpre − Vth)/delta)))
/// ds/dt     = (inf − s) / tau
/// I_post    = weight · conductance · s · (erev − V_post)
/// ```
///
/// This is the NeuroML2 `gradedSynapse` (a one-way, continuously-transmitting
/// analogue synapse; the presynaptic neuron is unaffected).
#[derive(Debug, Clone, Copy)]
pub struct SigmoidGradedSynapse {
    /// Half-activation voltage (mV).
    pub vth: f32,
    /// Activation-curve slope (mV); must be non-zero.
    pub delta: f32,
    /// Transmitter–receptor dissociation rate (per ms); must be `> 0`.
    pub k: f32,
    /// Reversal potential (mV).
    pub erev: f32,
    /// Conductance (raw nS, as the other synapse weights).
    pub conductance: f32,
}

impl SigmoidGradedSynapse {
    pub fn new(vth: f32, delta: f32, k: f32, erev: f32, conductance: f32) -> Self {
        Self {
            vth,
            delta,
            k,
            erev,
            conductance,
        }
    }

    /// Validate the profile: finite parameters, `delta != 0`, `k > 0`,
    /// non-negative `conductance`.
    pub fn validate(&self) -> Result<(), String> {
        let p = [self.vth, self.delta, self.k, self.erev, self.conductance];
        if p.iter().any(|x| !x.is_finite()) {
            return Err("SigmoidGradedSynapse: parameters must be finite".into());
        }
        if self.delta == 0.0 {
            return Err("SigmoidGradedSynapse: delta must be non-zero".into());
        }
        if self.k <= 0.0 {
            return Err("SigmoidGradedSynapse: k must be > 0".into());
        }
        if self.conductance < 0.0 {
            return Err("SigmoidGradedSynapse: conductance must be >= 0".into());
        }
        Ok(())
    }
}

/// Runtime state of one registered [`SigmoidGradedSynapse`] profile: shared
/// parameters plus a per-connection activation state `s`.
pub(crate) struct SigmoidGradedRuntime {
    pub(crate) vth: f32,
    pub(crate) inv_delta: f32,
    pub(crate) k: f32,
    pub(crate) erev: f32,
    pub(crate) conductance: f32,
    pub(crate) dt_ms: f32,
    /// `(pre_global, post_global, weight, s)` — `s` is the activation state.
    pub(crate) conns: Vec<(u32, u32, f32, f32)>,
}

impl SigmoidGradedRuntime {
    pub(crate) fn new(spec: SigmoidGradedSynapse, dt_ms: f32) -> Self {
        Self {
            vth: spec.vth,
            inv_delta: 1.0 / spec.delta,
            k: spec.k,
            erev: spec.erev,
            conductance: spec.conductance,
            dt_ms,
            conns: Vec::new(),
        }
    }

    /// Integrate every connection one step (exponential Euler on `s`) and add
    /// the resulting current `weight·conductance·s·(erev − V_post)` to `i_in`.
    pub(crate) fn step(&mut self, v: &[f32], i_in: &mut [f32]) {
        for (pre, post, weight, s) in &mut self.conns {
            let vpre = v[*pre as usize];
            let vpost = v[*post as usize];
            let arg = (vpre - self.vth) * self.inv_delta;
            let inf = 1.0 / (1.0 + (-arg).exp());
            let tau = 1.0 / (self.k * (1.0 + arg.exp()));
            let decay = (-self.dt_ms / tau).exp();
            *s += (inf - *s) * (1.0 - decay);
            let current = *weight * self.conductance * *s * (self.erev - vpost);
            i_in[*post as usize] += current;
        }
    }
}
