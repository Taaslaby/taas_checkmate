//! LUT-based Hodgkin-Huxley ion channels — zero exp() calls at runtime.
//!
//! # Idea
//! Standard HH: α(V), β(V) via exp() → m∞, τ → m(t+dt) = m∞ + (m-m∞)·exp(-dt/τ)
//! LUT HH:   precompute m∞(V), decay(V) = exp(-dt/τ(V)) for 1024 V-levels
//!            → runtime: quantize V, LUT → m∞, decay, FMA update. Zero exp().
//!
//! # LUT size
//! 1024 V-levels × (m∞, decay) × N_gates × 4 bytes = 8KB per gate.
//! Typical HH (3 gates: Na-m, Na-h, K-n) = 24KB. L1 cache.
//!
//! # Channel types
//! - `NaV`: m³h — Na⁺ spike
//! - `KV`:  n⁴  — K⁺ rectifier
//! - `Leak`: passive (no gates, just G_leak · (E_leak − V))
//! - Extensible: CaV, KA, KM, HCN, etc.

#![allow(non_snake_case)]
use wide::f32x8;

// ═══════════════════════════════════════════════════════
//  Helpers (same as mod.rs: load8, store8, ...)
// ═══════════════════════════════════════════════════════

#[inline(always)]
unsafe fn load8(ptr: *const f32, off: usize) -> f32x8 {
    let p = ptr.add(off);
    f32x8::from([*p, *p.add(1), *p.add(2), *p.add(3), *p.add(4), *p.add(5), *p.add(6), *p.add(7)])
}
#[inline(always)]
unsafe fn load8_i32(ptr: *const i32, off: usize) -> f32x8 {
    let p = ptr.add(off);
    f32x8::from([*p as f32, *p.add(1) as f32, *p.add(2) as f32, *p.add(3) as f32,
                 *p.add(4) as f32, *p.add(5) as f32, *p.add(6) as f32, *p.add(7) as f32])
}
#[inline(always)]
unsafe fn store8(ptr: *mut f32, off: usize, v: f32x8) {
    let a = v.to_array();
    let p = ptr.add(off);
    *p = a[0]; *p.add(1) = a[1]; *p.add(2) = a[2]; *p.add(3) = a[3];
    *p.add(4) = a[4]; *p.add(5) = a[5]; *p.add(6) = a[6]; *p.add(7) = a[7];
}

// ═══════════════════════════════════════════════════════
//  LUT Generation (call at init, offline)
// ═══════════════════════════════════════════════════════

/// Rate-function closure: α(V) or β(V) in [1/ms].
pub type RateFn = fn(f32) -> f32;

/// A single gate (e.g. Na-m, Na-h, K-n) with its LUT.
#[derive(Debug, Clone)]
pub struct GateLUT {
    pub m_inf: Vec<f32>,   // [N_BINS] steady-state
    pub decay: Vec<f32>,   // [N_BINS] exp(-dt / τ(V))
}

/// Channel specification: one or more gates + reversal + conductance.
#[derive(Debug, Clone)]
pub struct ChannelSpec {
    pub name: &'static str,
    pub gates: Vec<GateSpec>,
    pub E_rev: f32,       // mV
    pub g_max: f32,        // mS/cm²
}

#[derive(Debug, Clone)]
pub struct GateSpec {
    pub label: &'static str,
    pub alpha: RateFn,
    pub beta: RateFn,
    pub power: u32,         // exponent in conductance: m^power
}

/// Precompute LUTs for all channels at a given dt and V-range.
pub fn build_luts(channels: &[ChannelSpec], dt: f32,
                  v_min: f32, v_max: f32, n_bins: usize) -> Vec<Vec<GateLUT>> {
    channels.iter().map(|ch| {
        ch.gates.iter().map(|g| {
            let mut m_inf = Vec::with_capacity(n_bins + 1);
            let mut decay = Vec::with_capacity(n_bins + 1);
            for bi in 0..=n_bins {
                let v = v_min + (v_max - v_min) * (bi as f32) / (n_bins as f32);
                let a = (g.alpha)(v);
                let b = (g.beta)(v);
                let sum = a + b;
                let minf = if sum > 1e-12 { a / sum } else { 0.5 };
                let tau = if sum > 1e-12 { 1.0 / sum } else { 1.0 };
                m_inf.push(minf);
                decay.push((-dt / tau).exp());
            }
            GateLUT { m_inf, decay }
        }).collect()
    }).collect()
}

/// Quantize voltage to LUT index, clamped.
#[inline(always)]
pub fn v_to_idx(v: f32, v_min: f32, inv_dv: f32, n_bins: usize) -> usize {
    let idx = ((v - v_min) * inv_dv) as isize;
    idx.clamp(0, n_bins as isize) as usize
}

// ═══════════════════════════════════════════════════════
//  Standard HH rate functions (reference)
// ═══════════════════════════════════════════════════════

/// Na activation (m-gate) — Hodgkin & Huxley 1952, squid axon @ 6.3°C
pub fn na_m_alpha(v: f32) -> f32 {
    let v_ = v + 40.0;
    if v_.abs() < 1e-6 { 1.0 } else { 0.1 * v_ / (1.0 - (-v_ / 10.0).exp()) }
}
pub fn na_m_beta(v: f32) -> f32 {
    4.0 * (-(v + 65.0) / 18.0).exp()
}

/// Na inactivation (h-gate)
pub fn na_h_alpha(v: f32) -> f32 {
    0.07 * (-(v + 65.0) / 20.0).exp()
}
pub fn na_h_beta(v: f32) -> f32 {
    1.0 / (1.0 + (-(v + 35.0) / 10.0).exp())
}

/// K activation (n-gate)
pub fn k_n_alpha(v: f32) -> f32 {
    let v_ = v + 55.0;
    if v_.abs() < 1e-6 { 0.1 } else { 0.01 * v_ / (1.0 - (-v_ / 10.0).exp()) }
}
pub fn k_n_beta(v: f32) -> f32 {
    0.125 * (-(v + 65.0) / 80.0).exp()
}

/// CaV-T (T-type calcium) — low-threshold, pacemaking.
/// Hill & Tononi 2005, Wang & Rinzel 1993.
/// Single activation gate (c), constant tau_c = 0.6ms.
pub fn cat_c_alpha(v: f32) -> f32 {
    let m_inf = 1.0 / (1.0 + (-(v + 56.0) / 6.2).exp());
    m_inf / 0.6
}
pub fn cat_c_beta(v: f32) -> f32 {
    let m_inf = 1.0 / (1.0 + (-(v + 56.0) / 6.2).exp());
    (1.0 - m_inf) / 0.6
}

/// Standard HH channels: NaV + KV.
/// These produce fast spiking (like Izhikevich RS).
pub fn standard_channels() -> Vec<ChannelSpec> {
    vec![
        ChannelSpec {
            name: "NaV",
            gates: vec![
                GateSpec { label: "m", alpha: na_m_alpha, beta: na_m_beta, power: 3 },
                GateSpec { label: "h", alpha: na_h_alpha, beta: na_h_beta, power: 1 },
            ],
            E_rev: 50.0,
            g_max: 120.0,
        },
        ChannelSpec {
            name: "KV",
            gates: vec![
                GateSpec { label: "n", alpha: k_n_alpha, beta: k_n_beta, power: 4 },
            ],
            E_rev: -77.0,
            g_max: 36.0,
        },
    ]
}

/// Compute steady-state gate values at a given membrane potential.
/// Uses alpha/beta rate functions: m_inf = α(V) / (α(V) + β(V))
pub fn steady_state_gates(v: f32, channels: &[ChannelSpec]) -> Vec<f32> {
    let mut gates = Vec::new();
    for ch in channels {
        for gate in &ch.gates {
            let a = (gate.alpha)(v);
            let b = (gate.beta)(v);
            let sum = a + b;
            let minf = if sum > 1e-12 { a / sum } else { 0.5 };
            gates.push(minf);
        }
    }
    gates
}

/// HH with CaV-T (T-type calcium) for pacemaking.
/// Adds low-threshold Ca²⁺ current — neurons fire spontaneously without input.
/// 4 gates: NaV-m, NaV-h, KV-n, CaT-c
pub fn pacemaking_channels() -> Vec<ChannelSpec> {
    vec![
        ChannelSpec {
            name: "NaV",
            gates: vec![
                GateSpec { label: "m", alpha: na_m_alpha, beta: na_m_beta, power: 3 },
                GateSpec { label: "h", alpha: na_h_alpha, beta: na_h_beta, power: 1 },
            ],
            E_rev: 50.0,
            g_max: 120.0,
        },
        ChannelSpec {
            name: "KV",
            gates: vec![
                GateSpec { label: "n", alpha: k_n_alpha, beta: k_n_beta, power: 4 },
            ],
            E_rev: -77.0,
            g_max: 36.0,
        },
        ChannelSpec {
            name: "CaT",
            gates: vec![
                GateSpec { label: "c", alpha: cat_c_alpha, beta: cat_c_beta, power: 1 },
            ],
            E_rev: 120.0,
            g_max: 0.5,
        },
    ]
}

// ═══════════════════════════════════════════════════════
//  SIMD LUT-HH step kernel
// ═══════════════════════════════════════════════════════

/// Scalar HH step using LUTs.
/// Channels are interleaved in params: for each channel (E_rev, g_max, n_gates_in_ch).
/// gates array: column-major, [gate_0(nrn 0..n), gate_1(nrn 0..n), ...]
pub(crate) unsafe fn hh_lut_step_scalar(
    v: *mut f32, gates: *mut f32,
    Ii: *const f32, g_leak: *const f32, E_leak: *const f32, C_m: *const f32,
    n: usize,
    _n_gates: usize,
    m_inf_luts: *const *const f32, decay_luts: *const *const f32,
    v_min: f32, v_max: f32, n_bins: usize,
    E_rev_arr: *const f32, g_max_arr: *const f32, powers: *const u32,
    n_gates_per_ch: *const usize, n_channels: usize,
    _g_leak_idx: usize, dt: f32,
) {
    let inv_dv = (n_bins as f32) / (v_max - v_min);
    for i in 0..n {
        let vi = *v.add(i);
        let idx = v_to_idx(vi, v_min, inv_dv, n_bins);
        let mut I_ion = 0.0_f32;
        let mut gi = 0;  // global gate index
        
        for ch in 0..n_channels {
            let ng = *n_gates_per_ch.add(ch);
            let e_rev = *E_rev_arr.add(ch);
            let g_max = *g_max_arr.add(ch);
            let mut g_ch = g_max;
            
            for _ in 0..ng {
                let m_inf_arr = *m_inf_luts.add(gi);
                let decay_arr = *decay_luts.add(gi);
                let power = *powers.add(gi);
                let p = gates.add(gi * n + i);
                let m_inf = *m_inf_arr.add(idx);
                let decay = *decay_arr.add(idx);
                *p = m_inf + (*p - m_inf) * decay;
                g_ch *= (*p).powi(power as i32);
                gi += 1;
            }
            I_ion += g_ch * (vi - e_rev);
        }
        let I_leak = *g_leak.add(i) * (vi - *E_leak.add(i));
        let cm = *C_m.add(i);
        let dV = (*Ii.add(i) - I_ion - I_leak) / cm;
        *v.add(i) = vi + dV * dt;
    }
}

// ═══════════════════════════════════════════════════════
//  SIMD (f32x8) HH step — 8 neurons at once
// ═══════════════════════════════════════════════════════

pub(crate) unsafe fn hh_lut_step_simd(
    v: *mut f32, gates: *mut f32,
    Ii: *const f32, g_leak: *const f32, E_leak: *const f32, C_m: *const f32,
    n: usize, _n_gates: usize,
    m_inf_luts: *const *const f32, decay_luts: *const *const f32,
    v_min: f32, v_max: f32, n_bins: f32,
    E_rev_arr: *const f32, g_max_arr: *const f32, powers: *const u32,
    n_gates_per_ch: *const usize, n_channels: usize,
    dt: f32,
) {
    let inv_dv = n_bins / (v_max - v_min);
    let v_min_v = f32x8::splat(v_min);
    let inv_dv_v = f32x8::splat(inv_dv);
    let zero = f32x8::splat(0.0);
    let n_bins_f = f32x8::splat(n_bins);
    
    for i in (0..n).step_by(8) {
        if i + 8 > n { break; }
        let vi = load8(v, i);
        let idx_f = ((vi - v_min_v) * inv_dv_v).max(zero).min(n_bins_f);
        let mut I_ion = f32x8::splat(0.0);
        let mut gi = 0;  // global gate index
        
        for ch in 0..n_channels {
            let ng = *n_gates_per_ch.add(ch);
            let e_rev_v = f32x8::splat(*E_rev_arr.add(ch));
            let g_max_v = f32x8::splat(*g_max_arr.add(ch));
            let mut g_ch = g_max_v;
            
            for _ in 0..ng {
                let m_inf_arr = *m_inf_luts.add(gi);
                let decay_arr = *decay_luts.add(gi);
                let power = *powers.add(gi);
                
                let mut m_inf_a = [0.0f32; 8];
                let mut decay_a = [0.0f32; 8];
                let idx_a = idx_f.to_array();
                for k in 0..8 {
                    let fi = (idx_a[k] as usize).min(n_bins as usize - 1);
                    m_inf_a[k] = *m_inf_arr.add(fi);
                    decay_a[k] = *decay_arr.add(fi);
                }
                let m_inf_v = f32x8::from(m_inf_a);
                let decay_v = f32x8::from(decay_a);
                
                let gi_ptr = gates.add(gi * n + i);
                let gv = load8(gi_ptr, 0);
                let new_gv = m_inf_v + (gv - m_inf_v) * decay_v;
                store8(gi_ptr, 0, new_gv);
                g_ch *= new_gv.powf(power as f32);
                gi += 1;
            }
            I_ion += g_ch * (vi - e_rev_v);
        }
        
        let I_leak = load8(g_leak, i) * (vi - load8(E_leak, i));
        let cm = load8(C_m, i);
        let dV = (load8(Ii, i) - I_ion - I_leak) / cm;
        store8(v, i, vi + dV * f32x8::splat(dt));
    }
}

/// Public entry: auto-chooses SIMD or scalar.
pub unsafe fn hh_lut_step(
    v: *mut f32, gates: *mut f32, Ii: *const f32,
    g_leak: *const f32, E_leak: *const f32, C_m: *const f32,
    n: usize, n_gates: usize,
    m_inf_luts: *const *const f32, decay_luts: *const *const f32,
    v_min: f32, v_max: f32, n_bins: f32,
    E_rev_arr: *const f32, g_max_arr: *const f32, powers: *const u32,
    n_gates_per_ch: *const usize, n_channels: usize, dt: f32,
) {
    if n >= 8 {
        hh_lut_step_simd(v, gates, Ii, g_leak, E_leak, C_m,
            n, n_gates, m_inf_luts, decay_luts, v_min, v_max, n_bins,
            E_rev_arr, g_max_arr, powers, n_gates_per_ch, n_channels, dt);
    }
    let remainder = n % 8;
    if remainder > 0 {
        let simd_end = n - remainder;
        hh_lut_step_scalar(
            v.add(simd_end), gates.add(simd_end), Ii.add(simd_end),
            g_leak.add(simd_end), E_leak.add(simd_end), C_m.add(simd_end),
            remainder, n_gates,
            m_inf_luts, decay_luts, v_min, v_max, n_bins as usize,
            E_rev_arr, g_max_arr, powers,
            n_gates_per_ch, n_channels, !0, dt);
    }
}
