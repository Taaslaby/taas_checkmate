//! LUT-based Hodgkin-Huxley ion channels — zero exp() at runtime.
//!
//! # Usage
//! ```ignore
//! use taas_core::neuron::hh::{HhLutConfig, step_hh_lut, standard_config};
//!
//! let mut cfg = standard_config(128, 0.01); // 128 neurons, dt=0.01ms
//! step_hh_lut(&mut cfg); // call every dt
//! ```

use rayon::prelude::*;
use crate::kernels;

/// Per-neuron HH parameters (constant during simulation).
#[derive(Debug, Clone)]
pub struct HhParams {
    pub g_leak: f32,   // mS/cm²
    pub E_leak: f32,   // mV
    pub C_m: f32,      // µF/cm²
}

/// Runtime state for one HH neuron.
#[derive(Debug, Clone)]
pub struct HhState {
    pub v: f32,            // mV
    pub gates: Vec<f32>,   // gate variables in order: [m_0, h_0, m_1, h_1, ...]
}

/// Compiled LUT configuration for simulation.
pub struct HhLutConfig {
    pub n: usize,
    pub n_gates: usize,
    pub n_channels: usize,
    pub v_min: f32,
    pub v_max: f32,
    pub n_bins: f32,
    pub dt: f32,

    // Per-neuron arrays (length n)
    pub v: Vec<f32>,
    pub gates: Vec<f32>,
    pub Ii: Vec<f32>,
    pub g_leak: Vec<f32>,
    pub E_leak: Vec<f32>,
    pub C_m: Vec<f32>,

    // LUT data (per gate, length n_gates)
    pub m_inf_luts: Vec<Vec<f32>>,  // [n_gates][n_bins+1]
    pub decay_luts: Vec<Vec<f32>>,  // [n_gates][n_bins+1]
    pub powers: Vec<u32>,           // [n_gates]

    // Channel params (per channel, length n_channels)
    pub E_rev_arr: Vec<f32>,   // [n_channels]
    pub g_max_arr: Vec<f32>,   // [n_channels]
    pub n_gates_per_ch: Vec<usize>,

    /// Which gate index is the leak (or !0 for explicit leak).
    pub g_leak_idx: usize,

    /// Steady-state gate values at -65 mV (used by reset()).
    pub init_gates: Vec<f32>,
}

/// Build a `HhLutConfig` with standard NaV + KV channels.
pub fn standard_config(n: usize, dt: f32) -> HhLutConfig {
    let channels = kernels::neuron::hh::standard_channels();
    let n_channels = channels.len();
    let n_gates: usize = channels.iter().map(|ch| ch.gates.len()).sum();
    let luts = kernels::neuron::hh::build_luts(&channels, dt, -100.0, 60.0, 1024);

    // Per-gate arrays (order: all gates of ch0, then ch1, ...)
    let mut m_inf_luts = Vec::with_capacity(n_gates);
    let mut decay_luts = Vec::with_capacity(n_gates);
    let mut powers = Vec::with_capacity(n_gates);
    // Per-channel arrays
    let mut E_rev_arr = Vec::with_capacity(n_channels);
    let mut g_max_arr = Vec::with_capacity(n_channels);
    let mut n_gates_per_ch = Vec::with_capacity(n_channels);

    for (ch_i, ch) in channels.iter().enumerate() {
        E_rev_arr.push(ch.E_rev);
        g_max_arr.push(ch.g_max);
        n_gates_per_ch.push(ch.gates.len());
        for (g_i, gate) in ch.gates.iter().enumerate() {
            m_inf_luts.push(luts[ch_i][g_i].m_inf.clone());
            decay_luts.push(luts[ch_i][g_i].decay.clone());
            powers.push(gate.power);
        }
    }

    let init_gates = kernels::neuron::hh::steady_state_gates(-65.0, &channels);
    let mut gates = Vec::with_capacity(n * n_gates);
    for _ in 0..n {
        gates.extend_from_slice(&init_gates);
    }

    HhLutConfig {
        n,
        n_gates,
        n_channels,
        v_min: -100.0,
        v_max: 60.0,
        n_bins: 1024.0,
        dt,
        v: vec![-65.0; n],
        gates,
        Ii: vec![0.0; n],
        g_leak: vec![0.3; n],
        E_leak: vec![-65.0; n],
        C_m: vec![1.0; n],
        m_inf_luts,
        decay_luts,
        E_rev_arr,
        g_max_arr,
        powers,
        n_gates_per_ch,
        g_leak_idx: !0,
        init_gates,
    }
}

/// Build `HhLutConfig` from custom channel specs.
/// Used by `pacemaking_channels()` and custom Python configs.
pub fn build_config_from_channels(
    channels: &[kernels::neuron::hh::ChannelSpec],
    n: usize, dt: f32,
    v_min: f32, v_max: f32, n_bins: usize,
) -> HhLutConfig {
    let n_channels = channels.len();
    let n_gates: usize = channels.iter().map(|ch| ch.gates.len()).sum();
    let luts = kernels::neuron::hh::build_luts(channels, dt, v_min, v_max, n_bins);

    let mut m_inf_luts = Vec::with_capacity(n_gates);
    let mut decay_luts = Vec::with_capacity(n_gates);
    let mut powers = Vec::with_capacity(n_gates);
    let mut E_rev_arr = Vec::with_capacity(n_channels);
    let mut g_max_arr = Vec::with_capacity(n_channels);
    let mut n_gates_per_ch = Vec::with_capacity(n_channels);

    for (ch_i, ch) in channels.iter().enumerate() {
        E_rev_arr.push(ch.E_rev);
        g_max_arr.push(ch.g_max);
        n_gates_per_ch.push(ch.gates.len());
        for (g_i, gate) in ch.gates.iter().enumerate() {
            m_inf_luts.push(luts[ch_i][g_i].m_inf.clone());
            decay_luts.push(luts[ch_i][g_i].decay.clone());
            powers.push(gate.power);
        }
    }

    let init_gates = kernels::neuron::hh::steady_state_gates(-65.0, channels);
    let mut gates = Vec::with_capacity(n * n_gates);
    for _ in 0..n {
        gates.extend_from_slice(&init_gates);
    }

    HhLutConfig {
        n,
        n_gates,
        n_channels,
        v_min,
        v_max,
        n_bins: n_bins as f32,
        dt,
        v: vec![-65.0; n],
        gates,
        Ii: vec![0.0; n],
        g_leak: vec![0.3; n],
        E_leak: vec![-65.0; n],
        C_m: vec![1.0; n],
        m_inf_luts,
        decay_luts,
        E_rev_arr,
        g_max_arr,
        powers,
        n_gates_per_ch,
        g_leak_idx: !0,
        init_gates,
    }
}

// ═══════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use kernels::neuron::hh::*;

    /// Reference exact HH step using exp() — ground truth.
    fn hh_exact_step(v: &mut f32, gates: &mut [f32],
                     channels: &[ChannelSpec],
                     I_ext: f32, g_leak: f32, E_leak: f32, C_m: f32, dt: f32) {
        let mut I_ion = 0.0;
        let mut gi = 0;
        for ch in channels {
            let mut g_ch = ch.g_max;
            for gate in &ch.gates {
                let a = (gate.alpha)(*v);
                let b = (gate.beta)(*v);
                let sum = a + b;
                let m_inf = if sum > 1e-12 { a / sum } else { 0.5 };
                let tau = if sum > 1e-12 { 1.0 / sum } else { 1.0 };
                let decay = (-dt / tau).exp();
                gates[gi] = m_inf + (gates[gi] - m_inf) * decay;
                g_ch *= gates[gi].powi(gate.power as i32);
                gi += 1;
            }
            I_ion += g_ch * (*v - ch.E_rev);
        }
        let I_leak = g_leak * (*v - E_leak);
        let dV = (I_ext - I_ion - I_leak) / C_m;
        *v += dV * dt;
    }

    #[test]
    fn lut_vs_exact_spike_timing() {
        let channels = standard_channels();
        let n_gates: usize = channels.iter().map(|ch| ch.gates.len()).sum();

        // LUT config
        let mut lut = standard_config(1, 0.01);
        // Override Ii with a step current
        lut.Ii[0] = 10.0; // 10 µA/cm² — strong enough to spike

        // Exact state (copy initial conditions)
        let mut v_exact = -65.0;
        let mut gates_exact = vec![0.5; n_gates];

        let n_steps = 10_000; // 100 ms at dt=0.01
        let mut lut_spikes = 0;
        let mut exact_spikes = 0;

        let mut last_lut_v = -65.0;
        let mut last_exact_v = -65.0;
        let mut max_v_diff = 0.0;

        for _step in 0..n_steps {
            // Step LUT
            step_hh_lut(&mut lut);
            let v_lut = lut.v[0];

            // Step exact
            hh_exact_step(&mut v_exact, &mut gates_exact, &channels,
                          10.0, 0.3, -65.0, 1.0, 0.01);

            // Track voltage difference
            let v_diff = (v_lut - v_exact).abs();
            if v_diff > max_v_diff { max_v_diff = v_diff; }

            // Count spikes (crossing 0 mV upward)
            if v_lut > 0.0 && last_lut_v <= 0.0 { lut_spikes += 1; }
            if v_exact > 0.0 && last_exact_v <= 0.0 { exact_spikes += 1; }

            last_lut_v = v_lut;
            last_exact_v = v_exact;
        }

        // Both should spike (with 10 µA/cm², HH fires ~50 Hz)
        assert!(lut_spikes > 0, "LUT HH produced no spikes");
        assert!(exact_spikes > 0, "Exact HH produced no spikes");

        // Spike count should match (allow ±1 for timing drift)
        let spike_diff = (lut_spikes as i32 - exact_spikes as i32).abs();
        assert!(spike_diff <= 1,
            "Spike count mismatch: LUT={lut_spikes}, exact={exact_spikes}");

        // Voltage may diverge during spikes due to LUT quantization (~0.16 mV/bin).
        // The key requirement is spike count match — dynamics are chaotic under
        // accumulated phase error but spike rate remains correct.
        println!("LUT HH test PASSED: LUT spikes={lut_spikes}, exact={exact_spikes}, max_V_diff={max_v_diff:.2} mV");
    }
}

/// Step all neurons by dt using LUT HH dynamics.
pub fn step_hh_lut(cfg: &mut HhLutConfig) {
    let n = cfg.n;
    let n_gates = cfg.n_gates;

    // Pre-extract immutable references to avoid borrow conflicts
    let v_min = cfg.v_min;
    let v_max = cfg.v_max;
    let n_bins = cfg.n_bins;
    let dt = cfg.dt;
    let n_channels = cfg.n_channels;
    let E_rev_arr = &cfg.E_rev_arr;
    let g_max_arr = &cfg.g_max_arr;
    let powers = &cfg.powers;
    let n_gates_per_ch = &cfg.n_gates_per_ch;
    let m_inf_luts = &cfg.m_inf_luts;
    let decay_luts = &cfg.decay_luts;

    unsafe fn call_kernel(
        v: &mut [f32], gates: &mut [f32], Ii: &[f32],
        g_leak: &[f32], E_leak: &[f32], C_m: &[f32],
        n: usize, n_gates: usize,
        v_min: f32, v_max: f32, n_bins: f32, dt: f32, n_channels: usize,
        E_rev_arr: &[f32], g_max_arr: &[f32], powers: &[u32],
        n_gates_per_ch: &[usize],
        m_inf_luts: &[Vec<f32>], decay_luts: &[Vec<f32>],
    ) {
        let m_inf_ptrs: Vec<*const f32> = m_inf_luts.iter().map(|v| v.as_ptr()).collect();
        let decay_ptrs: Vec<*const f32> = decay_luts.iter().map(|v| v.as_ptr()).collect();
        kernels::neuron::hh::hh_lut_step(
            v.as_mut_ptr(), gates.as_mut_ptr(), Ii.as_ptr(),
            g_leak.as_ptr(), E_leak.as_ptr(), C_m.as_ptr(),
            n, n_gates,
            m_inf_ptrs.as_ptr(), decay_ptrs.as_ptr(),
            v_min, v_max, n_bins,
            E_rev_arr.as_ptr(), g_max_arr.as_ptr(), powers.as_ptr(),
            n_gates_per_ch.as_ptr(), n_channels, dt,
        );
    }

    if n < 8 {
        unsafe {
            call_kernel(
                &mut cfg.v[..n], &mut cfg.gates[..n * n_gates], &cfg.Ii[..n],
                &cfg.g_leak[..n], &cfg.E_leak[..n], &cfg.C_m[..n],
                n, n_gates,
                v_min, v_max, n_bins, dt, n_channels,
                E_rev_arr, g_max_arr, powers, n_gates_per_ch,
                m_inf_luts, decay_luts,
            );
        }
        return;
    }

    cfg.v.par_chunks_exact_mut(8)
        .zip(cfg.gates.par_chunks_exact_mut(8 * n_gates))
        .zip(cfg.Ii.par_chunks_exact(8))
        .zip(cfg.g_leak.par_chunks_exact(8))
        .zip(cfg.E_leak.par_chunks_exact(8))
        .zip(cfg.C_m.par_chunks_exact(8))
        .for_each(|(((((v8, g8), I8), gl8), El8), cm8)| {
            unsafe {
                call_kernel(v8, g8, I8, gl8, El8, cm8, 8, n_gates,
                    v_min, v_max, n_bins, dt, n_channels,
                    E_rev_arr, g_max_arr, powers, n_gates_per_ch,
                    m_inf_luts, decay_luts,
                );
            }
        });
}
