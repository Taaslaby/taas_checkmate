use crate::core::util::{load8, load8_i32, store8};
use wide::{f32x8, CmpGt};

// ═══════════════════════════════════════════════════════════
//  Build receptor array from nt_types (CSR)
// ═══════════════════════════════════════════════════════════

/// Build a per-synapse `receptor_types` array from per-neuron `nt_types` using
/// the CSR structure. Each synapse gets the receptor type of its pre-neuron's NT.
///
/// `nt > 0` → AMPA (exc), `nt ≤ 0` → GABA_A (inh).
pub fn build_receptor_types(row_ptr: &[i32], n_synapses: usize, nt_types: &[f32]) -> Vec<u8> {
    let mut receptors = vec![0u8; n_synapses.max(1)];
    let n = row_ptr.len().saturating_sub(1);
    let mut syn_idx = 0usize;
    for pre in 0..n {
        let nt = nt_types.get(pre).copied().unwrap_or(1.0);
        let rec = if nt > 0.0 { 0u8 } else { 2u8 }; // AMPA or GABA_A
        let start = row_ptr.get(pre).copied().unwrap_or(0).max(0) as usize;
        let end = row_ptr.get(pre + 1).copied().unwrap_or(0).max(0) as usize;
        let end = end.min(n_synapses);
        for _ in start..end {
            if syn_idx < receptors.len() {
                receptors[syn_idx] = rec;
            }
            syn_idx += 1;
        }
    }
    receptors
}

/// Batch Tsodyks–Markram short-term plasticity step — SIMD blocks + scalar
/// tail over the per-lane canon [`crate::ops::plasticity::stp::stp_step`].
///
/// Updates `u`/`x` in place and accumulates `i_out[i] += s·a·u'·x'` where
/// `s = 1` for a pre-spike at `i`. All buffers share one length `n`.
///
/// # Panics
/// If any buffer is shorter than `u`.
pub fn stp_step_batch(
    u: &mut [f32],
    x: &mut [f32],
    pre_spikes: &[i32],
    tau_d: &[f32],
    tau_f: &[f32],
    u0: &[f32],
    a: &[f32],
    i_out: &mut [f32],
    dt: f32,
) {
    let n = u.len();
    assert!(
        x.len() >= n
            && pre_spikes.len() >= n
            && tau_d.len() >= n
            && tau_f.len() >= n
            && u0.len() >= n
            && a.len() >= n
            && i_out.len() >= n,
        "stp_step_batch: buffer shorter than u ({n})"
    );
    let n8 = n & !7;
    let dt_s = f32x8::splat(dt);
    let zero_s = f32x8::splat(0.0);
    let one_s = f32x8::splat(1.0);

    for off in (0..n8).step_by(8) {
        // SAFETY: `off + 8 ≤ n` for every loop iteration, so all loads/stores
        // below are in bounds of the (≥ n) slices.
        unsafe {
            let uu = load8(u.as_ptr(), off);
            let xx = load8(x.as_ptr(), off);
            let sp = load8_i32(pre_spikes.as_ptr(), off);
            let td = load8(tau_d.as_ptr(), off).max(f32x8::splat(1e-10));
            let tf = load8(tau_f.as_ptr(), off).max(f32x8::splat(1e-10));
            let u0v = load8(u0.as_ptr(), off);
            let av = load8(a.as_ptr(), off);
            let (un, xn) = crate::ops::plasticity::stp::stp_step(
                uu,
                xx,
                sp,
                u0v,
                one_s / td,
                one_s / tf,
                dt_s,
            );
            // s = 1 где pre_spike, иначе 0; I_out += s·a·u_new·x_new.
            let s = sp.cmp_gt(zero_s).blend(one_s, zero_s);
            let mut io = load8(i_out.as_ptr(), off);
            io += s * av * un * xn;
            store8(u.as_mut_ptr(), off, un);
            store8(x.as_mut_ptr(), off, xn);
            store8(i_out.as_mut_ptr(), off, io);
        }
    }
    // Скалярный tail — тот же lane-канон на f32.
    for i in n8..n {
        let s = if pre_spikes[i] != 0 { 1.0f32 } else { 0.0f32 };
        let (un, xn) = crate::ops::plasticity::stp::stp_step(
            u[i],
            x[i],
            s,
            u0[i],
            1.0 / tau_d[i].max(1e-10),
            1.0 / tau_f[i].max(1e-10),
            dt,
        );
        u[i] = un;
        x[i] = xn;
        i_out[i] += s * a[i] * un * xn;
    }
}
