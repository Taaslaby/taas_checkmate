//! Neuron population classes.

use crate::core::lane::SimdLane;
use crate::core::util::{load8, load8_i32, store8};
use crate::ops::plasticity::stp as pillars_stp;
use wide::f32x8;

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
        let (x, u) = (&mut self.x[..n], &mut self.u[..n]);
        let (tau_d, tau_f, u0, a) = (
            &self.tau_d[..n],
            &self.tau_f[..n],
            &self.u0[..n],
            &self.a[..n],
        );
        let i_out = &mut I_out[..n];
        let pre = &pre_spikes[..n];

        let n8 = n & !7;
        let dt_s = f32x8::splat(self.dt);
        let zero_s = f32x8::splat(0.0);
        let one_s = f32x8::splat(1.0);

        for off in (0..n8).step_by(8) {
            unsafe {
                let uu = load8(u.as_ptr(), off);
                let xx = load8(x.as_ptr(), off);
                let sp = load8_i32(pre.as_ptr(), off);
                let td = load8(tau_d.as_ptr(), off).max(f32x8::splat(1e-10));
                let tf = load8(tau_f.as_ptr(), off).max(f32x8::splat(1e-10));
                let u0v = load8(u0.as_ptr(), off);
                let av = load8(a.as_ptr(), off);
                let (un, xn) = pillars_stp::stp_step(uu, xx, sp, u0v, one_s / td, one_s / tf, dt_s);
                // s = 1 где pre_spike, иначе 0; I_out += s·a·u_new·x_new
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
            let s = if pre[i] != 0 { 1.0f32 } else { 0.0f32 };
            let (un, xn) = pillars_stp::stp_step(
                u[i],
                x[i],
                s,
                u0[i],
                1.0 / tau_d[i].max(1e-10),
                1.0 / tau_f[i].max(1e-10),
                self.dt,
            );
            u[i] = un;
            x[i] = xn;
            i_out[i] += s * a[i] * un * xn;
        }
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

