//! GLIF3/GLIF4 (Allen Institute) stateful populations.

// ═══════════════════════════════════════════════════════════
//  GLIF3 / GLIF4 (Allen Institute)
// ═══════════════════════════════════════════════════════════

/// 1/x для x > 0, иначе 0 — сторож легаси-семантики τ ≤ 0 → «без эволюции».
#[inline(always)]
fn inv_or_zero(x: f32) -> f32 {
    if x > 0.0 {
        1.0 / x
    } else {
        0.0
    }
}

/// Repack legacy 8-col GLIF3 params → GLIF3Model 9-col (кэшируется в state).
fn pack_glif3(params: &[f32], tau_ref: f32, n: usize) -> Vec<f32> {
    let mut p = vec![0.0f32; n * 9];
    for i in 0..n {
        p[i] = params[i];
        for c in 1..5 {
            p[c * n + i] = params[c * n + i];
        }
        p[5 * n + i] = inv_or_zero(params[5 * n + i]);
        p[6 * n + i] = params[6 * n + i];
        p[7 * n + i] = params[7 * n + i];
        p[8 * n + i] = tau_ref;
    }
    p
}

/// Repack legacy 10-col GLIF4 params → GLIF4Model 11-col (кэшируется в state).
fn pack_glif4(params: &[f32], tau_ref: f32, n: usize) -> Vec<f32> {
    let mut p = vec![0.0f32; n * 11];
    for i in 0..n {
        p[i] = params[i];
        for c in 1..5 {
            p[c * n + i] = params[c * n + i];
        }
        p[5 * n + i] = inv_or_zero(params[5 * n + i]);
        p[6 * n + i] = params[6 * n + i];
        p[7 * n + i] = params[7 * n + i];
        p[8 * n + i] = inv_or_zero(params[8 * n + i]);
        p[9 * n + i] = params[9 * n + i];
        p[10 * n + i] = tau_ref;
    }
    p
}

/// GLIF3 state — LIF with after-spike current (I_ahp).
#[derive(Debug, Clone)]
pub struct Glif3State {
    pub n: usize,
    /// Membrane potential `[n]` (mV)
    pub v: Vec<f32>,
    /// After-spike conductance `[n]`
    pub g_ahp: Vec<f32>,
    /// Total input current `[n]`
    pub Ii: Vec<f32>,
    /// Spike output `[n]`, 0/1
    pub spikes: Vec<i32>,
    /// Refractory counters `[n]`
    pub refractory: Vec<f32>,
    /// Parameters: [dt, g_leak, E_leak, v_thresh, v_reset, τ_ahp, Δg_ahp, E_ahp]
    pub params: Vec<f32>,
    /// Refractory period (ms)
    pub tau_ref: f32,
    /// Кэш pillars-лейаута (9-col GLIF3Model) — пересобирается при set_params.
    packed: Vec<f32>,
}

impl Glif3State {
    /// Create a new GLIF3 population of `n_neurons` with uniform parameters.
    pub fn new(
        n_neurons: usize,
        dt: f32,
        g_leak: f32,
        E_leak: f32,
        v_thresh: f32,
        v_reset: f32,
        tau_ahp: f32,
        delta_g_ahp: f32,
        E_ahp: f32,
        tau_ref: f32,
    ) -> Self {
        let n = n_neurons.div_ceil(8) * 8;
        let mut params = vec![0.0f32; n * 8];
        for j in 0..n {
            params[j] = dt;
            params[n + j] = g_leak;
            params[2 * n + j] = E_leak;
            params[3 * n + j] = v_thresh;
            params[4 * n + j] = v_reset;
            params[5 * n + j] = tau_ahp;
            params[6 * n + j] = delta_g_ahp;
            params[7 * n + j] = E_ahp;
        }
        let packed = pack_glif3(&params, tau_ref, n);
        Self {
            n,
            v: vec![E_leak; n],
            g_ahp: vec![0.0; n],
            Ii: vec![0.0; n],
            spikes: vec![0; n],
            refractory: vec![0.0; n],
            params,
            tau_ref,
            packed,
        }
    }

    /// Step all neurons.
    pub fn step(&mut self, I_ext: Option<&[f32]>) {
        if let Some(ext) = I_ext {
            let min_len = self.n.min(ext.len());
            self.Ii[..min_len].copy_from_slice(&ext[..min_len]);
        }
        let n = self.n;
        unsafe {
            crate::core::runner::run_neuron::<crate::zoo::neurons::GLIF3Model>(
                self.v.as_mut_ptr(),
                self.g_ahp.as_mut_ptr(),
                self.refractory.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                self.Ii.as_ptr(),
                self.spikes.as_mut_ptr(),
                self.packed.as_ptr(),
                n,
            );
        }
    }

    /// Reset all state variables.
    pub fn reset(&mut self) {
        self.v.fill(self.params[2 * self.n]); // E_leak at column 2
        self.g_ahp.fill(0.0);
        self.refractory.fill(0.0);
        self.spikes.fill(0);
    }

    /// Set an individual neuron's parameters.
    pub fn set_params(
        &mut self,
        idx: usize,
        g_leak: f32,
        E_leak: f32,
        v_thresh: f32,
        v_reset: f32,
        tau_ahp: f32,
        delta_g_ahp: f32,
        E_ahp: f32,
    ) {
        if idx < self.n {
            let n = self.n;
            self.params[n + idx] = g_leak;
            self.params[2 * n + idx] = E_leak;
            self.params[3 * n + idx] = v_thresh;
            self.params[4 * n + idx] = v_reset;
            self.params[5 * n + idx] = tau_ahp;
            self.params[6 * n + idx] = delta_g_ahp;
            self.params[7 * n + idx] = E_ahp;
            self.packed = pack_glif3(&self.params, self.tau_ref, n);
        }
    }
}

/// GLIF4 state — GLIF3 + threshold adaptation (θ).
#[derive(Debug, Clone)]
pub struct Glif4State {
    pub n: usize,
    /// Membrane potential `[n]` (mV)
    pub v: Vec<f32>,
    /// Threshold offset `[n]` (effective threshold = v_thresh_0 + θ)
    pub theta: Vec<f32>,
    /// After-spike conductance `[n]`
    pub g_ahp: Vec<f32>,
    /// Total input current `[n]`
    pub Ii: Vec<f32>,
    /// Spike output `[n]`, 0/1
    pub spikes: Vec<i32>,
    /// Refractory counters `[n]`
    pub refractory: Vec<f32>,
    /// Parameters: [dt, g_leak, E_leak, v_thresh_0, v_reset, τ_ahp, Δg_ahp, E_ahp, τ_θ, Δθ]
    pub params: Vec<f32>,
    /// Refractory period (ms)
    pub tau_ref: f32,
    /// Кэш pillars-лейаута (11-col GLIF4Model) — пересобирается при set_params.
    packed: Vec<f32>,
}

impl Glif4State {
    /// Create a new GLIF4 population of `n_neurons` with uniform parameters.
    pub fn new(
        n_neurons: usize,
        dt: f32,
        g_leak: f32,
        E_leak: f32,
        v_thresh_0: f32,
        v_reset: f32,
        tau_ahp: f32,
        delta_g_ahp: f32,
        E_ahp: f32,
        tau_theta: f32,
        delta_theta: f32,
        tau_ref: f32,
    ) -> Self {
        let n = n_neurons.div_ceil(8) * 8;
        let mut params = vec![0.0f32; n * 10];
        for j in 0..n {
            params[j] = dt;
            params[n + j] = g_leak;
            params[2 * n + j] = E_leak;
            params[3 * n + j] = v_thresh_0;
            params[4 * n + j] = v_reset;
            params[5 * n + j] = tau_ahp;
            params[6 * n + j] = delta_g_ahp;
            params[7 * n + j] = E_ahp;
            params[8 * n + j] = tau_theta;
            params[9 * n + j] = delta_theta;
        }
        let packed = pack_glif4(&params, tau_ref, n);
        Self {
            n,
            v: vec![E_leak; n],
            theta: vec![0.0; n],
            g_ahp: vec![0.0; n],
            Ii: vec![0.0; n],
            spikes: vec![0; n],
            refractory: vec![0.0; n],
            params,
            tau_ref,
            packed,
        }
    }

    /// Step all neurons.
    pub fn step(&mut self, I_ext: Option<&[f32]>) {
        if let Some(ext) = I_ext {
            let min_len = self.n.min(ext.len());
            self.Ii[..min_len].copy_from_slice(&ext[..min_len]);
        }
        let n = self.n;
        unsafe {
            crate::core::runner::run_neuron::<crate::zoo::neurons::GLIF4Model>(
                self.v.as_mut_ptr(),
                self.theta.as_mut_ptr(),
                self.g_ahp.as_mut_ptr(),
                self.refractory.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                self.Ii.as_ptr(),
                self.spikes.as_mut_ptr(),
                self.packed.as_ptr(),
                n,
            );
        }
    }

    /// Reset all state variables.
    pub fn reset(&mut self) {
        self.v.fill(self.params[2 * self.n]); // E_leak
        self.theta.fill(0.0);
        self.g_ahp.fill(0.0);
        self.refractory.fill(0.0);
        self.spikes.fill(0);
    }

    /// Set an individual neuron's GLIF4 parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn set_params(
        &mut self,
        idx: usize,
        g_leak: f32,
        E_leak: f32,
        v_thresh_0: f32,
        v_reset: f32,
        tau_ahp: f32,
        delta_g_ahp: f32,
        E_ahp: f32,
        tau_theta: f32,
        delta_theta: f32,
    ) {
        if idx < self.n {
            let n = self.n;
            self.params[n + idx] = g_leak;
            self.params[2 * n + idx] = E_leak;
            self.params[3 * n + idx] = v_thresh_0;
            self.params[4 * n + idx] = v_reset;
            self.params[5 * n + idx] = tau_ahp;
            self.params[6 * n + idx] = delta_g_ahp;
            self.params[7 * n + idx] = E_ahp;
            self.params[8 * n + idx] = tau_theta;
            self.params[9 * n + idx] = delta_theta;
            self.packed = pack_glif4(&self.params, self.tau_ref, n);
        }
    }
}

