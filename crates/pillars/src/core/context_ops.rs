//! # Biological Opcodes — читаемые имена для нейробиологов
//!
//! Impl-блок `SimdContext<S>` с биологическими опкодами. Вынесен из
//! `context.rs`, чтобы держать ядро (state/params/шины) отдельно от
//! готовых биологических примитивов.
//!
//! Все методы используют [`SimdLane`] внутри — одинаковый код для SIMD
//! и скаляра. Имена соответствуют TNMT vm/ опкодам (op_leak_integrate,
//! op_fire_reset, ...).
//!
//! ⚠️ Операнды — индексы (`usize`), а не типизированные параметры:
//! этот уровень — низкоуровневый «индексный» API. Для типизированного
//! написания моделей используйте `#[derive(NeuronModel)]` + `ComputeNeuron`.

use super::context::SimdContext;
use super::lane::SimdLane;
use std::marker::PhantomData;
use std::ptr::NonNull;



impl<'a, S: SimdLane> SimdContext<'a, S> {
    /// Forward Euler: `v += (v_rest - v + I_syn) · inv_tau · dt`
    ///
    /// # Params
    /// - `v_idx`: state index for membrane potential
    /// - `inv_tau_idx`: param index for `1/τ` (precomputed — zero divisions in hot path)
    /// - `v_rest_idx`: param index for resting potential
    #[inline(always)]
    pub fn leak_integrate(&mut self, v_idx: impl Into<usize>, inv_tau_idx: impl Into<usize>, v_rest_idx: impl Into<usize>) {
        let v_idx = v_idx.into();
        let inv_tau_idx = inv_tau_idx.into();
        let v_rest_idx = v_rest_idx.into();
        let v = self.s(v_idx);
        let dv = (self.p(v_rest_idx) - v + self.i()) * self.p(inv_tau_idx) * self.dt();
        self.set_s(v_idx, v + dv);
    }

    /// Spike detection + voltage reset. Returns spike mask.
    ///
    /// If `v >= v_thresh`: sets `v = v_reset` and returns true.
    /// Otherwise: v unchanged, returns false.
    ///
    /// # Params
    /// - `v_idx`: state index for membrane potential
    /// - `thresh_idx`: param index for spike threshold
    /// - `reset_idx`: param index for reset potential
    #[inline(always)]
    pub fn fire_reset(&mut self, v_idx: impl Into<usize>, thresh_idx: impl Into<usize>, reset_idx: impl Into<usize>) -> S {
        let v_idx = v_idx.into();
        let thresh_idx = thresh_idx.into();
        let reset_idx = reset_idx.into();
        let v = self.s(v_idx);
        let spike = v.cmp_ge(self.p(thresh_idx));
        self.set_s(v_idx, spike.cond().then(self.p(reset_idx)).else_(v));
        spike
    }

    /// Clamp membrane potential to reset during refractory period.
    ///
    /// Если `refr > 0`: v = v_reset (нейрон в рефрактерности).
    /// Иначе: v не меняется.
    #[inline(always)]
    pub fn refractory_clamp(&mut self, v_idx: impl Into<usize>, refr_idx: impl Into<usize>, v_reset_idx: impl Into<usize>) {
        let v_idx = v_idx.into(); let refr_idx = refr_idx.into(); let v_reset_idx = v_reset_idx.into();
        let is_refr = self.s(refr_idx).cmp_gt(S::zero()).cond();
        self.set_s(v_idx, is_refr.then(self.p(v_reset_idx)).else_(self.s(v_idx)));
    }

    /// Decrement refractory timer by dt, or set to tau_ref on spike.
    #[inline(always)]
    pub fn refractory_timer(&mut self, refr_idx: impl Into<usize>, spike: S, tau_ref_idx: impl Into<usize>) {
        let refr_idx = refr_idx.into(); let tau_ref_idx = tau_ref_idx.into();
        let refr = self.s(refr_idx);
        let refr_dec = refr - self.dt();
        let refr_set = spike.cond().then(self.p(tau_ref_idx)).else_(refr_dec);
        self.set_s(refr_idx, refr_set.cmp_gt(S::zero()).cond().then(refr_set).else_(S::zero()));
    }

    /// Leak + fire + refractory — полный шаг LIF-подобного нейрона.
    ///
    /// 🛡️ Re-clamp to v_reset after integration but before spike detection.
    /// Prevents refractory breakthrough at high input currents.
    #[inline(always)]
    pub fn lif_step(
        &mut self, v_idx: impl Into<usize>, refr_idx: impl Into<usize>,
        inv_tau_idx: impl Into<usize>, v_rest_idx: impl Into<usize>,
        v_thresh_idx: impl Into<usize>, v_reset_idx: impl Into<usize>, tau_ref_idx: impl Into<usize>,
    ) {
        let v_idx = v_idx.into(); let refr_idx = refr_idx.into();
        let v_reset_idx = v_reset_idx.into();
        let v_thresh_idx = v_thresh_idx.into();
        self.refractory_clamp(v_idx, refr_idx, v_reset_idx);
        self.leak_integrate(v_idx, inv_tau_idx, v_rest_idx);
        // 🛡️ Re-clamp after integration: prevents v jumping from v_reset to above
        // v_thresh in one dt when I_syn is very large.
        let is_refr = self.s(refr_idx).cmp_gt(S::zero()).cond();
        let v_new = self.s(v_idx);
        self.set_s(v_idx, is_refr.then(self.p(v_reset_idx)).else_(v_new));
        let spike = self.fire_reset(v_idx, v_thresh_idx, v_reset_idx);
        self.spike(spike);
        self.refractory_timer(refr_idx, spike, tau_ref_idx);
    }

    /// Ohmic ionic current: `I = g · (V - E_rev)` (NEURON convention).
    ///
    /// gating — произведение гейтов. Если канал без гейтинга, передайте `S::one()`.
    #[inline(always)]
    pub fn ohmic_current(&self, g_idx: impl Into<usize>, e_rev_idx: impl Into<usize>, v: S, gating: S) -> S {
        let g_idx = g_idx.into(); let e_rev_idx = e_rev_idx.into();
        let g = self.p(g_idx) * gating;
        g.mul_add(v, -g * self.p(e_rev_idx))
    }

    /// Ohmic без гейтинга: `I = g · (V - E_rev)` (leak).
    #[inline(always)]
    pub fn ohmic_const(&self, g_idx: impl Into<usize>, e_rev_idx: impl Into<usize>, v: S) -> S {
        let g_idx = g_idx.into(); let e_rev_idx = e_rev_idx.into();
        let g = self.p(g_idx);
        g.mul_add(v, -g * self.p(e_rev_idx))
    }

    /// Мембранная деполяризация: `dv = (I_inj - ΣI_ion) · inv_Cm · dt`
    ///
    /// Все ионные токи вычитаются из входящего (NEURON convention).
    #[inline(always)]
    pub fn membrane_depolarize(&mut self, v_idx: impl Into<usize>, inv_cm_idx: impl Into<usize>, i_ion: S) {
        let v_idx = v_idx.into(); let inv_cm_idx = inv_cm_idx.into();
        let v = self.s(v_idx);
        let dv = (self.i() - i_ion) * self.p(inv_cm_idx) * self.dt();
        self.set_s(v_idx, v + dv);
    }

    /// Edge-triggered spike detection + was_below update (один проход cmp_ge).
    ///
    /// Спайк = rising edge: v ≥ v_thresh И v_prev < v_thresh.
    /// Автоматически обновляет was_below_flag для следующего шага.
    /// Использует `is_above.not()`, а не `spike.not()` — предотвращает ложные спайки
    /// на устойчивом надпороговом напряжении.
    ///
    /// Returns spike mask.
    #[inline(always)]
    pub fn edge_spike(
        &mut self, v_idx: impl Into<usize>, v_thresh_idx: impl Into<usize>, was_below_idx: impl Into<usize>,
    ) -> S {
        let v_idx = v_idx.into(); let v_thresh_idx = v_thresh_idx.into(); let was_below_idx = was_below_idx.into();
        let is_above = self.s(v_idx).cmp_ge(self.p(v_thresh_idx));
        let was_below = self.s(was_below_idx).cmp_gt(S::zero());
        let spike = is_above.and(was_below);
        // was_below = 1 if v < v_thresh (below), 0 if above
        // ⚠️ MUST use is_above.not(), NOT spike.not() — otherwise sustained
        //    above-threshold voltages incorrectly set flag=1, causing
        //    false spikes on the next step.
        let flag = is_above.cond().then(S::zero()).else_(S::one());
        self.set_s(was_below_idx, flag);
        spike
    }
}

// ═══════════════════════════════════════════════════════════════
//  Tests (перенесены из context.rs вместе с опкодами)
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::context::MAX_STATES;
    use wide::f32x8;

    /// Create a minimal SimdContext with 8 neurons, 6 params (dt, inv_tau, v_rest, v_thresh, v_reset, tau_ref)
    fn make_context<S: SimdLane>(off: usize, params: &[f32], n: usize) -> SimdContext<'_, S> {

        SimdContext {
            state: [S::zero(); MAX_STATES],
            params_ptr: NonNull::new(params.as_ptr() as *mut f32).expect("params_ptr must be non-null"),
            n,
            off,
            i_syn: S::zero(),
            i_pre: S::zero(),
            i_post: S::zero(),
            dt_val: params[0],
            spike_mask: S::zero(),
            _marker: PhantomData,
        }
    }

    fn make_lif_params(n: usize) -> Vec<f32> {
        let mut p = vec![0.0f32; n * 6];
        for i in 0..n {
            p[i] = 0.5;            // dt
            p[n + i] = 0.1;         // inv_tau_m
            p[2*n + i] = -70.0;     // v_rest
            p[3*n + i] = -54.0;     // v_thresh
            p[4*n + i] = -70.0;     // v_reset
            p[5*n + i] = 2.0;       // tau_ref
        }
        p
    }

    #[test]
    fn test_context_leak_integrate() {
        let params = make_lif_params(8);
        let mut ctx = make_context::<f32x8>(0, &params, 8);
        ctx.set_s(0, f32x8::splat(-70.0));
        ctx.i_syn = f32x8::splat(30.0); // strong depolarizing current

        ctx.leak_integrate(0usize, 1usize, 2usize);

        let v_new = ctx.s(0);
        // Expected: v = -70 + (-70 - (-70) + 30) * 0.1 * 0.5
        //                  = -70 + 30 * 0.1 * 0.5
        //                  = -70 + 1.5 = -68.5
        assert!((v_new.to_array()[0] - (-68.5)).abs() < 1e-5,
            "v_new: expected -68.5, got {:.6}", v_new.to_array()[0]);
    }

    #[test]
    fn test_context_fire_reset() {
        let params = make_lif_params(8);
        let mut ctx = make_context::<f32x8>(0, &params, 8);
        // Set v above threshold
        ctx.set_s(0, f32x8::splat(-50.0)); // v > v_thresh (-54)
        let spike = ctx.fire_reset(0usize, 3usize, 4usize);

        // Should have spiked
        assert_ne!(spike.to_array()[0].to_bits(), 0, "should spike at v=-50");
        // v should be reset to v_reset = -70
        let v = ctx.s(0);
        assert!((v.to_array()[0] - (-70.0)).abs() < 1e-6,
            "v after reset: expected -70, got {:.6}", v.to_array()[0]);
    }

    #[test]
    fn test_context_fire_reset_no_spike() {
        let params = make_lif_params(8);
        let mut ctx = make_context::<f32x8>(0, &params, 8);
        ctx.set_s(0, f32x8::splat(-60.0)); // below threshold
        let spike = ctx.fire_reset(0usize, 3usize, 4usize);
        assert_eq!(spike.to_array()[0].to_bits(), 0, "should NOT spike at v=-60");
        // v unchanged
        let v = ctx.s(0);
        assert!((v.to_array()[0] - (-60.0)).abs() < 1e-6);
    }

    #[test]
    fn test_context_refractory_clamp() {
        let params = make_lif_params(8);
        let mut ctx = make_context::<f32x8>(0, &params, 8);
        ctx.set_s(0, f32x8::splat(100.0)); // high v
        ctx.set_s(1, f32x8::splat(1.0));  // refractory > 0

        ctx.refractory_clamp(0usize, 1usize, 4usize);

        // v should be clamped to v_reset = -70
        let v = ctx.s(0);
        assert!((v.to_array()[0] - (-70.0)).abs() < 1e-6,
            "refractory clamp: expected -70, got {:.6}", v.to_array()[0]);
    }

    #[test]
    fn test_context_refractory_timer_decay() {
        let params = make_lif_params(8);
        let mut ctx = make_context::<f32x8>(0, &params, 8);
        ctx.set_s(1, f32x8::splat(2.0)); // refr = 2.0 ms
        let no_spike = f32x8::zero();

        // dt = 0.5, so refr should decrement by 0.5
        ctx.refractory_timer(1usize, no_spike, 5usize);

        let refr = ctx.s(1);
        assert!((refr.to_array()[0] - 1.5).abs() < 1e-6,
            "refr after decay: expected 1.5, got {:.6}", refr.to_array()[0]);
    }

    #[test]
    fn test_context_refractory_timer_spike_reset() {
        let params = make_lif_params(8);
        let mut ctx = make_context::<f32x8>(0, &params, 8);
        ctx.set_s(1, f32x8::splat(0.0));  // refr = 0
        // Generate proper SIMD mask via cmp_ge
        let spike = f32x8::splat(1.0).cmp_ge(f32x8::splat(0.0));

        ctx.refractory_timer(1usize, spike, 5usize);

        // refr should be set to tau_ref = 2.0
        let refr = ctx.s(1);
        assert!((refr.to_array()[0] - 2.0).abs() < 1e-6,
            "refr after spike: expected 2.0, got {:.6}", refr.to_array()[0]);
    }

    #[test]
    fn test_context_lif_step_full_cycle() {
        let params = make_lif_params(8);
        let mut ctx = make_context::<f32x8>(0, &params, 8);
        ctx.set_s(0, f32x8::splat(-70.0)); // v
        ctx.set_s(1, f32x8::splat(0.0));  // refr
        ctx.i_syn = f32x8::splat(30.0);

        // Step 1: integrate, spike expected
        ctx.lif_step(0usize, 1usize, 1usize, 2usize, 3usize, 4usize, 5usize);

        // Reset state and run multiple times
        ctx.set_s(0, f32x8::splat(-70.0));
        ctx.set_s(1, f32x8::splat(0.0));

        for _ in 0..10 {
            ctx.lif_step(0usize, 1usize, 1usize, 2usize, 3usize, 4usize, 5usize);
        }

        let v_after = ctx.s(0);
        assert!(v_after.to_array()[0].is_finite());
    }

    #[test]
    fn test_context_ohmic_current() {
        let n = 8;
        // Add g_Na and E_Na param slots at indices 6, 7
        let mut p8 = vec![0.0f32; n * 8];
        for i in 0..n {
            p8[i] = 0.5;            // dt
            p8[n + i] = 0.1;         // inv_tau_m
            p8[2*n + i] = -70.0;     // v_rest
            p8[3*n + i] = -54.0;     // v_thresh
            p8[4*n + i] = -70.0;     // v_reset
            p8[5*n + i] = 2.0;       // tau_ref
            p8[6*n + i] = 120.0;     // g_Na
            p8[7*n + i] = 50.0;      // E_Na
        }

        let mut ctx = make_context::<f32x8>(0, &p8, n);
        ctx.set_s(0, f32x8::splat(-65.0)); // v

        // I_Na = g_Na * (v - E_Na) = 120 * (-65 - 50) = 120 * (-115) = -13800
        let i_na = ctx.ohmic_current(6usize, 7usize, ctx.s(0), f32x8::splat(1.0));

        assert!((i_na.to_array()[0] - (-13800.0)).abs() < 1.0,
            "ohmic Na current: expected -13800, got {:.0}", i_na.to_array()[0]);
    }

    #[test]
    fn test_context_ohmic_scalar() {
        let n = 8;
        let mut flat = Vec::with_capacity(n * 8);
        for col in 0..8 {
            for _ in 0..n {
                flat.push(match col {
                    6 => 0.5,       // g
                    7 => -70.0,     // E_rev
                    _ => 0.0,
                });
            }
        }

        let ctx = make_context::<f32>(0, &flat, n);
        let i = ctx.ohmic_const(6usize, 7usize, -65.0);
        // I = 0.5 * (-65 - (-70)) = 0.5 * 5 = 2.5
        assert!((i - 2.5).abs() < 1e-6, "ohmic scalar: expected 2.5, got {:.6}", i);
    }

    #[test]
    fn test_context_edge_spike() {
        let n = 8;
        let mut params = vec![0.0f32; 4 * n];
        for i in 0..n {
            params[i] = 0.5;        // dt
            params[n + i] = -54.0;   // v_thresh
            params[2*n + i] = 0.0;   // unused
            params[3*n + i] = 0.0;   // unused
        }

        let mut ctx = make_context::<f32x8>(0, &params, n);
        ctx.set_s(0, f32x8::splat(-60.0)); // v below thresh
        ctx.set_s(1, f32x8::splat(1.0));  // was_below = 1

        // Now set v above thresh
        ctx.set_s(0, f32x8::splat(-50.0));
        let spike = ctx.edge_spike(0usize, 1usize, 1usize);

        // Should detect rising edge: was_below=1, now above → spike
        assert_ne!(spike.to_array()[0].to_bits(), 0, "edge should detect spike");
        // was_below flag should now be 0
        let flag = ctx.s(1);
        assert_eq!(flag.to_array()[0].to_bits(), 0, "was_below flag should clear");
    }

    #[test]
    fn test_context_edge_spike_no_false_on_sustained() {
        let n = 8;
        let mut params = vec![0.0f32; 4 * n];
        for i in 0..n {
            params[i] = 0.5;
            params[n + i] = -54.0;
            params[2*n + i] = 0.0;
            params[3*n + i] = 0.0;
        }

        let mut ctx = make_context::<f32x8>(0, &params, n);
        // v was already above threshold: was_below = 0
        ctx.set_s(0, f32x8::splat(-50.0));
        ctx.set_s(1, f32x8::splat(0.0));   // was_below = 0

        let spike = ctx.edge_spike(0usize, 1usize, 1usize);

        // Should NOT spike on sustained above-threshold
        assert_eq!(spike.to_array()[0].to_bits(), 0, "no spike on sustained above-threshold");
    }

    #[test]
    fn test_context_membrane_depolarize() {
        let n = 8;
        let mut params = vec![0.0f32; 3 * n];
        for i in 0..n {
            params[i] = 0.5;          // dt
            params[n + i] = 0.1;       // inv_Cm
            params[2*n + i] = 0.0;     // unused
        }

        let mut ctx = make_context::<f32x8>(0, &params, n);
        ctx.set_s(0, f32x8::splat(-70.0));
        ctx.i_syn = f32x8::splat(10.0);

        // dv = (I_inj - I_ion) * inv_Cm * dt = (10 - 0) * 0.1 * 0.5 = 0.5
        ctx.membrane_depolarize(0usize, 1usize, f32x8::splat(0.0));

        let v = ctx.s(0);
        assert!((v.to_array()[0] - (-69.5)).abs() < 1e-6,
            "membrane: expected -69.5, got {:.6}", v.to_array()[0]);

        // With ion current subtracted:
        ctx.set_s(0, f32x8::splat(-70.0));
        ctx.membrane_depolarize(0usize, 1usize, f32x8::splat(5.0));
        // dv = (10 - 5) * 0.1 * 0.5 = 0.25
        let v2 = ctx.s(0);
        assert!((v2.to_array()[0] - (-69.75)).abs() < 1e-6,
            "membrane with I_ion: expected -69.75, got {:.6}", v2.to_array()[0]);
    }
}
