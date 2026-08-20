use super::lane::SimdLane;
use std::marker::PhantomData;
use std::ptr::NonNull;

// ═══════════════════════════════════════════════════════════════
//  MAX_STATES — the one place that caps model state count
// ═══════════════════════════════════════════════════════════════

/// Maximum number of state channels a model may declare.
///
/// The `state` array is fixed-size to stay on the stack (no heap in the hot
/// loop). Raising this constant is the ONLY change needed to support more
/// states — nothing else in the engine hardcodes `6` anymore:
///
/// - `load_state_scalar!` / `save_state_scalar!` (runner.rs) loop up to
///   `M::N_STATE` at compile time;
/// - `load_state_simd!` / `save_state_simd!` likewise;
/// - the legacy raw runners `run_neuron*` accept exactly 6 state pointers by
///   signature — models with more states must use `NeuralPopulation`.
///
/// 32 covers multi-compartment dendrites, glia, learnable-transformer (25 states) and future models.
pub const MAX_STATES: usize = 32;

/// Heuristic cap for debug-build param-index checks in [`SimdContext::p`] /
/// [`SimdContext::pf`]. Not a hard engine limit — no model uses more than a
/// handful of columns; raise if a model ever needs more.
pub const MAX_PARAM_COLS: usize = 32;

// ═══════════════════════════════════════════════════════════════
//  SimdContext — CPU-side execution context for one SIMD chunk
// ═══════════════════════════════════════════════════════════════
//
//  Architecture:
//
//    Zoo Model::step<S: SimdLane>(ctx: &mut SimdContext<S>)
//      ↑ reads .s(), .p(), .i(), .dt() …
//      ↓ writes .set_s(), .spike()
//
//    Core population.rs
//      ↑ allocates SimdContext for each SIMD chunk (8-wide or scalar)
//      ↑ loads state from SoA memory via read_unaligned
//      ↓ saves state back after step completes
//
//  Principle: Clean Bus
//  All external events (synaptic current, pre/post spikes) enter
//  through ctx only. No manual inject() calls.
//
//  Buses:
//  - `i_syn`:  Post-synaptic input current (sum of all weighted spikes)
//  - `i_pre`:  Pre-synaptic spike mask (for STDP/plasticity)
//  - `i_post`: Post-synaptic spike mask (for STDP/plasticity)
// ═══════════════════════════════════════════════════════════════

/// SIMD execution context — one chunk of `S::SIZE` neurons.
///
/// # Memory Model
///
/// ```text
///                        params_ptr
///                            │
///  SoA layout: [dt₀..dtₙ] [inv_tau₀..inv_tauₙ] [v_rest₀..] …
///                ↑param[0]   ↑param[1]             ↑param[2]
///                stride = n (capacity)
///
///  .p(idx)  → load_param(params_ptr, idx, n, off)
///              = SIMD vector of 8 consecutive values from column `idx`
///  .pf(idx) → scalar *(params_ptr + idx * n + off)
/// ```
///
/// # Lifetime Safety
///
/// The `'a` lifetime ensures that `params_ptr` cannot outlive the parameter
/// buffer it points to. This prevents use-after-free bugs when the underlying
/// `NeuralPopulation` SoA buffer is reallocated (e.g., via `.reserve()`).
///
/// ```rust,ignore
/// let mut pop = NeuralPopulation::new(...);
/// let ctx = pop.make_context();  // ctx borrows pop for lifetime 'a
/// pop.reserve(1000);             // ❌ Compile error: pop is borrowed
/// ```
///
/// # Safety
/// All pointer-based access is **unsafe free** — it's the runner's
/// responsibility to provide valid `params_ptr` and correct `n`/`off`.
///
/// **Note**: All fields are `pub(crate)` — only the runner (population.rs)
/// and zoo models may access them. External users interact through methods.
pub struct SimdContext<'a, S: SimdLane> {
    pub(crate) state: [S; MAX_STATES],
    pub(crate) params_ptr: NonNull<f32>,
    /// Number of parameter columns in the concrete model.
    /// Every parameter accessor checks against this value before doing raw
    /// pointer arithmetic, so safe model code cannot read past the SoA buffer.
    pub(crate) param_count: usize,
    pub(crate) n: usize,
    pub(crate) off: usize,
    pub(crate) i_syn: S,
    pub(crate) i_pre: S,
    pub(crate) i_post: S,
    pub(crate) dt_val: f32,
    pub(crate) spike_mask: S,
    pub(crate) _marker: PhantomData<&'a [f32]>,
}

impl<'a, S: SimdLane> SimdContext<'a, S> {
    #[inline(always)]
    pub fn s(&self, i: usize) -> S {
        assert!(
            i < MAX_STATES,
            "SimdContext::s: state index {i} >= MAX_STATES ({MAX_STATES})"
        );
        self.state[i]
    }
    #[inline(always)]
    pub fn set_s(&mut self, i: usize, v: S) {
        assert!(
            i < MAX_STATES,
            "SimdContext::set_s: state index {i} >= MAX_STATES ({MAX_STATES})"
        );
        self.state[i] = v;
    }
    /// Parameters: SIMD (SoA vector load) / scalar
    ///
    /// # Safety (caller contract)
    /// `i` must be a valid param column index (`< M::N_PARAMS`); the runner
    /// guarantees `params_ptr`/`n`/`off` describe a live padded buffer, so a
    /// valid index keeps the read in bounds. Debug builds additionally check
    /// a heuristic cap of [`MAX_PARAM_COLS`] (no model uses more; raise if
    /// ever needed).
    #[inline(always)]
    pub fn p(&self, i: usize) -> S {
        self.assert_param_index(i);
        // SAFETY: `assert_param_index` proves that `i` addresses one of the
        // model's allocated parameter columns; the runner supplies valid
        // `n`/`off` for this SIMD chunk.
        unsafe { S::load_param(self.params_ptr.as_ptr(), i, self.n, self.off) }
    }

    /// Parameter access via `Into<usize>` enum.
    ///
    /// Convenience that removes the `as usize` cast noise, but it is NOT
    /// type-checked against the model's param columns — `E` is any
    /// `Into<usize>`. Same bounds contract as [`SimdContext::p`]: the
    /// converted index must be `< M::N_PARAMS`.
    #[inline(always)]
    pub fn p_enum<E: Into<usize> + Copy>(&self, e: E) -> S {
        self.p(e.into())
    }
    /// Scalar parameter access via raw index.
    ///
    /// # Safety (caller contract)
    /// `i` must be a valid param column index (`< M::N_PARAMS`).
    #[inline(always)]
    pub fn pf(&self, i: usize) -> f32 {
        self.assert_param_index(i);
        // SAFETY: `assert_param_index` proves that `i` addresses one of the
        // model's allocated parameter columns; the runner supplies valid
        // `n`/`off` for this scalar lane.
        unsafe { *self.params_ptr.as_ptr().add(i * self.n + self.off) }
    }

    /// Scalar parameter access via `Into<usize>` enum — same bounds contract
    /// as [`SimdContext::pf`] (NOT type-checked against param columns).
    #[inline(always)]
    pub fn pf_enum<E: Into<usize> + Copy>(&self, e: E) -> f32 {
        self.pf(e.into())
    }

    /// Broadcast parameter access — read the scalar at column `i` element 0
    /// and splat it across the lane.
    ///
    /// For parameters that are constant across the population (marked
    /// `#[broadcast]` in the model), this loads **4 bytes** instead of the
    /// full `S::SIZE × 4`-byte column — an ~8× cut in param read traffic in
    /// the hot loop. The value is assumed identical for every neuron; the
    /// caller (the `#[broadcast]` contract) guarantees element 0 is the
    /// canonical value.
    ///
    /// # Safety (caller contract)
    /// `i` must be a valid param column index (`< M::N_PARAMS`).
    #[inline(always)]
    pub fn pb(&self, i: usize) -> S {
        self.assert_param_index(i);
        // SAFETY: `assert_param_index` proves `i` addresses an allocated
        // column; element 0 is within bounds (runner guarantees `n >= 1`).
        unsafe { S::splat(*self.params_ptr.as_ptr().add(i * self.n)) }
    }

    #[inline(always)]
    fn assert_param_index(&self, i: usize) {
        assert!(
            i < self.param_count,
            "SimdContext: parameter index {i} out of range (model has {} columns)",
            self.param_count
        );
    }

    #[inline(always)]
    pub fn i(&self) -> S {
        self.i_syn
    }
    #[inline(always)]
    pub fn i_syn_ref(&self) -> &S {
        &self.i_syn
    }
    #[inline(always)]
    pub fn pre(&self) -> S {
        self.i_pre
    }
    #[inline(always)]
    pub fn post(&self) -> S {
        self.i_post
    }
    /// Per-lane timestep (SoA params column 0): each SIMD lane reads its own
    /// `dt[off + lane]` instead of a broadcast scalar.
    ///
    /// This is the SIMD path used by the biological opcodes
    /// ([`leak_integrate`](crate::ops::membrane::leak_integrate),
    /// [`refractory_timer`](crate::ops::refractory::refractory_timer),
    /// [`membrane_depolarize`](crate::ops::membrane::membrane_depolarize),
    /// and the formula VM) — they take
    /// `dt` as a lane vector. The `ComputeNeuron` bridge instead feeds the
    /// broadcast [`SimdContext::dt_scalar`] into `StepContext::dt_simd()`.
    ///
    /// For **uniform dt** (the norm) both paths agree. If you make dt
    /// per-lane, the SIMD chunks and the scalar tail (which uses `dt_scalar()`)
    /// will diverge for the same neuron — add a parity test
    /// (`test_population_simd_tail_parity`) when you do.
    #[inline(always)]
    pub fn dt(&self) -> S {
        assert!(
            self.param_count > 0,
            "SimdContext::dt: model has no parameter columns"
        );
        // SAFETY: params column 0 exists (every model has >= 1 param), buffer
        // live, off + SIZE <= capacity (runner contract).
        unsafe { S::load_param(self.params_ptr.as_ptr(), 0, self.n, self.off) }
    }

    /// Scalar timestep — the broadcast `dt` the runner loaded from `params[0]`
    /// (neuron 0's dt) before stepping. Used by the `ComputeNeuron` bridge
    /// (`StepContext::dt()`); all zoo models go through it.
    #[inline(always)]
    pub fn dt_scalar(&self) -> f32 {
        self.dt_val
    }

    /// Create a raw context for StorageBuffer-based execution (no load_i needed).
    ///
    /// # Safety
    /// `params_ptr` must point to a valid parameter buffer that outlives the
    /// returned context for lifetime `'a`; `offset + S::SIZE <= capacity`.
    #[inline(always)]
    pub unsafe fn new_raw(
        params_ptr: *const f32,
        param_count: usize,
        capacity: usize,
        offset: usize,
        dt: f32,
    ) -> Self {
        // SAFETY: documented contract — params_ptr is non-null and live for
        // 'a; the pointer is stored as-is (raw contexts don't own memory).
        Self {
            state: [S::zero(); MAX_STATES],
            params_ptr: NonNull::new_unchecked(params_ptr as *mut f32),
            // Raw contexts have no model type from which to infer the number
            // of columns, so the caller supplies it explicitly.
            param_count,
            n: capacity,
            off: offset,
            i_syn: S::zero(),
            i_pre: S::zero(),
            i_post: S::zero(),
            dt_val: dt,
            spike_mask: S::zero(),
            _marker: PhantomData,
        }
    }

    /// Load i_syn from a slice (length must equal lane SIZE).
    ///
    /// # Safety
    /// `slice` must have at least `S::SIZE` elements, and all elements must be readable.
    #[inline(always)]
    pub fn load_i(&mut self, slice: &[f32]) {
        assert!(
            slice.len() >= S::SIZE,
            "SimdContext::load_i: slice has {}, needs at least {} elements",
            slice.len(),
            S::SIZE
        );
        // SAFETY: the assertion above establishes that the slice contains a
        // full SIMD lane; stride n = S::SIZE keeps the vector in bounds.
        unsafe {
            self.i_syn = S::load_param(slice.as_ptr(), 0, S::SIZE, 0);
        }
    }

    /// Load one state channel from a flat slice (e.g. post-synaptic spikes into CaPlasticity s(3)).
    ///
    /// # Safety
    /// `slice` must have at least `S::SIZE` elements. `state_idx` must be < [`MAX_STATES`].
    #[inline(always)]
    pub fn load_state_channel(&mut self, state_idx: usize, slice: &[f32]) {
        assert!(
            state_idx < MAX_STATES,
            "SimdContext::load_state_channel: state index {state_idx} >= MAX_STATES ({MAX_STATES})"
        );
        assert!(
            slice.len() >= S::SIZE,
            "SimdContext::load_state_channel: slice has {}, needs at least {} elements",
            slice.len(),
            S::SIZE
        );
        // SAFETY: both assertions above establish the documented contracts.
        // read) and state_idx < MAX_STATES (in-bounds array index; the const
        // guard in the load/save macros makes M::N_STATE <= MAX_STATES).
        unsafe {
            self.state[state_idx] = S::load_param(slice.as_ptr(), 0, S::SIZE, 0);
        }
    }

    /// Save one state channel back to a SoA column pointer.
    ///
    /// # Safety
    /// - `ptr` must point to a valid `capacity`-sized column (so that
    ///   `ptr.add(self.off)` is valid for `S::SIZE` elements) with the same
    ///   stride as this context's buffers;
    /// - `state_idx` must be < [`MAX_STATES`].
    #[inline(always)]
    pub unsafe fn save_state_channel(&self, state_idx: usize, ptr: *mut f32) {
        assert!(
            state_idx < MAX_STATES,
            "SimdContext::save_state_channel: state index {state_idx} >= MAX_STATES ({MAX_STATES})"
        );
        // SAFETY: documented contracts above — the raw slice is valid for
        // S::SIZE elements and write_to_slice writes exactly SIZE lanes.
        self.state[state_idx]
            .write_to_slice(std::slice::from_raw_parts_mut(ptr.add(self.off), S::SIZE));
    }

    // ─── Spike bus ─────────────────────────────────────────────────

    #[inline(always)]
    pub fn spike(&mut self, cond: S) {
        self.spike_mask = cond;
    }
    #[inline(always)]
    pub fn spike_mask(&self) -> S {
        self.spike_mask
    }

    // Biological opcodes (`leak_integrate`, `fire_reset`, `lif_step`, …)
    // удалены: модели перешли на типизированный `ComputeNeuron` + свободные
    // функции `ops::*`; этот файл отвечает только за state/params/шины.
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use wide::f32x8;

    /// Create a minimal SimdContext with 8 neurons, 6 params (dt, inv_tau, v_rest, v_thresh, v_reset, tau_ref)
    fn make_context<S: SimdLane>(off: usize, params: &[f32], n: usize) -> SimdContext<'_, S> {
        SimdContext {
            state: [S::zero(); MAX_STATES],
            params_ptr: NonNull::new(params.as_ptr() as *mut f32)
                .expect("params_ptr must be non-null"),
            param_count: params.len() / n,
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
            p[i] = 0.5; // dt
            p[n + i] = 0.1; // inv_tau_m
            p[2 * n + i] = -70.0; // v_rest
            p[3 * n + i] = -54.0; // v_thresh
            p[4 * n + i] = -70.0; // v_reset
            p[5 * n + i] = 2.0; // tau_ref
        }
        p
    }

    #[test]
    fn test_context_s_set_s() {
        let params = make_lif_params(8);
        let mut ctx = make_context::<f32x8>(0, &params, 8);
        ctx.set_s(0, f32x8::splat(-65.0));
        let s0 = ctx.s(0);
        assert!((s0.to_array()[0] - (-65.0)).abs() < 1e-6);
    }

    #[test]
    fn test_context_p_load() {
        let params = make_lif_params(8);
        let ctx = make_context::<f32x8>(0, &params, 8);
        // p(0) = dt = 0.5
        let dt = ctx.p(0);
        assert!((dt.to_array()[0] - 0.5).abs() < 1e-6);
        // p(2) = v_rest = -70.0
        let v_rest = ctx.p(2);
        assert!((v_rest.to_array()[0] - (-70.0)).abs() < 1e-6);
    }

    #[test]
    fn test_context_pf_scalar() {
        let params = make_lif_params(8);
        let ctx = make_context::<f32x8>(0, &params, 8);
        let dt = ctx.pf(0);
        assert!((dt - 0.5).abs() < 1e-6);
        let v_rest = ctx.pf(2);
        assert!((v_rest - (-70.0)).abs() < 1e-6);
    }

    #[test]
    fn test_context_load_i() {
        let params = make_lif_params(8);
        let mut ctx = make_context::<f32x8>(0, &params, 8);
        let i_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        ctx.load_i(&i_vals);
        let i = ctx.i();
        assert!((i.to_array()[0] - 1.0).abs() < 1e-6);
        assert!((i.to_array()[7] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_context_new_raw() {
        let params = make_lif_params(8);
        let ctx = unsafe { SimdContext::<f32x8>::new_raw(params.as_ptr(), 6, 8, 0, 0.5) };
        assert!((ctx.dt_val - 0.5).abs() < 1e-6);
        for i in 0..MAX_STATES {
            assert_eq!(ctx.state[i].to_array()[0], 0.0, "state[{i}] should be 0");
        }
    }

    #[test]
    #[should_panic(expected = "needs at least")]
    fn test_context_load_i_rejects_short_slice() {
        let params = make_lif_params(8);
        let mut ctx = make_context::<f32x8>(0, &params, 8);
        ctx.load_i(&[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "parameter index")]
    fn test_context_param_index_is_checked_in_release() {
        let params = make_lif_params(8);
        let ctx = make_context::<f32x8>(0, &params, 8);
        let _ = ctx.p(6);
    }

    #[test]
    #[should_panic(expected = "state index")]
    fn test_context_state_index_is_checked() {
        let params = make_lif_params(8);
        let mut ctx = make_context::<f32x8>(0, &params, 8);
        ctx.set_s(MAX_STATES, f32x8::zero());
    }

    #[test]
    fn test_context_p_enum() {
        let params = make_lif_params(8);
        let ctx = make_context::<f32x8>(0, &params, 8);

        // Test p_enum with a simple Into<usize> type
        #[derive(Copy, Clone)]
        enum ParamIdx {
            Dt,
            InvTau,
            VRest,
        }
        impl From<ParamIdx> for usize {
            fn from(p: ParamIdx) -> usize {
                match p {
                    ParamIdx::Dt => 0,
                    ParamIdx::InvTau => 1,
                    ParamIdx::VRest => 2,
                }
            }
        }

        let dt = ctx.p_enum(ParamIdx::Dt);
        assert!((dt.to_array()[0] - 0.5).abs() < 1e-6);
        let inv_tau = ctx.p_enum(ParamIdx::InvTau);
        assert!((inv_tau.to_array()[0] - 0.1).abs() < 1e-6);
        let v_rest = ctx.p_enum(ParamIdx::VRest);
        assert!((v_rest.to_array()[0] - (-70.0)).abs() < 1e-6);
    }
}
