//! Neuron population classes.

// ═══════════════════════════════════════════════════════════
//  Short-Term Plasticity (Tsodyks-Markram) + LFP proxy
// ═══════════════════════════════════════════════════════════

/// Short-Term Plasticity synapse state per connection.
///
/// Based on: Tsodyks & Markram (1997) "The neural code between
/// neocortical pyramidal neurons depends on neurotransmitter release
/// probability" — PNAS.
///
/// Канон — `crate::ops::plasticity::stp::stp_step` (lane-функция):
/// Exponential Euler, SIMD-цикл + scalar tail — тот же lane-канон на f32
/// (нейрон 0 == нейрон tail бит-в-бит).
pub struct ShortTermPlasticity {
    pub x: Vec<f32>,
    pub u: Vec<f32>,
    pub tau_d: Vec<f32>,
    pub tau_f: Vec<f32>,
    pub u0: Vec<f32>,
    pub a: Vec<f32>,
    pub n: usize,
    pub dt: f32,
}

impl ShortTermPlasticity {
    /// Create STP with n synapses.
    pub fn new(n: usize, dt: f32) -> Self {
        let n_pad = n.div_ceil(8) * 8;
        Self {
            x: vec![1.0; n_pad],
            u: vec![0.2; n_pad],
            tau_d: vec![200.0; n_pad],
            tau_f: vec![600.0; n_pad],
            u0: vec![0.2; n_pad],
            a: vec![1.0; n_pad],
            n: n_pad,
            dt,
        }
    }

    /// Create STP with heterogeneous (randomized) parameters.
    pub fn new_heterogeneous(n: usize, dt: f32, seed: u32) -> Self {
        let mut rng = FastRng::new(seed);
        let n_pad = n.div_ceil(8) * 8;
        let mut stp = Self::new(n_pad, dt);
        for i in 0..n_pad {
            stp.tau_d[i] = 100.0 + rng.next_f32() * 400.0;
            stp.tau_f[i] = 300.0 + rng.next_f32() * 700.0;
            stp.u0[i] = 0.05 + rng.next_f32() * 0.45;
            stp.a[i] = 0.5 + rng.next_f32() * 1.5;
        }
        stp
    }

    /// One step: update STP state and compute effective synaptic current.
    pub fn step(&mut self, pre_spikes: &[i32], I_out: &mut [f32]) {
        let n = self.n.min(pre_spikes.len()).min(I_out.len());
        if n == 0 {
            return;
        }
        // Batch SIMD kernel lives in `kernels::synapse` (the batch layer);
        // `api` only adapts the population buffers to it.
        crate::kernels::synapse::stp_step_batch(
            &mut self.u[..n],
            &mut self.x[..n],
            &pre_spikes[..n],
            &self.tau_d[..n],
            &self.tau_f[..n],
            &self.u0[..n],
            &self.a[..n],
            &mut I_out[..n],
            self.dt,
        );
    }

    pub fn reset(&mut self) {
        self.x.fill(1.0);
        self.u
            .iter_mut()
            .zip(self.u0.iter())
            .for_each(|(u, &u0)| *u = u0);
    }
}

/// Fast LCG RNG for initialization.
struct FastRng {
    state: u32,
}

impl FastRng {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.state >> 8) as f32 * crate::kernels::RNG_SCALE_24
    }
}

/// LFP proxy: aggregate synaptic activity into local field potential.
pub fn compute_lfp(
    spikes: &[i32],
    electrode_map: &[i32],
    lfp_out: &mut [f32],
    n_electrodes: usize,
) {
    let n_out = n_electrodes.min(lfp_out.len());
    lfp_out[..n_out].fill(0.0);
    let n = spikes.len().min(electrode_map.len());
    for i in 0..n {
        if spikes[i] != 0 {
            let e = electrode_map[i] as usize;
            if e < n_electrodes {
                lfp_out[e] += 1.0;
            }
        }
    }
}
