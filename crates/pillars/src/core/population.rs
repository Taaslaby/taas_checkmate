//! # NeuralPopulation — Safe SIMD Neuron Population
//!
//! === BOUNDARY: Safe Rust → unsafe SoA ===
//! Manages aligned SIMD memory for a homogeneous population of neurons.
//! Low-level runner functions live in [`super::runner`].
//!
//! ## Architecture
//!
//! ```text
//!   NeuralPopulation::step(i_in, spikes_out)  — SAFE
//!     ├── n ≤ 8:  step_scalar()  — f32 context, 1 neuron at a time
//!     └── n > 8:  step_simd()    — f32x8 context, 8 neurons at a time
//!           └── tail loop (n8..n): scalar fallback for remaining neurons
//! ```
//!
//! ## Memory Layout
//!
//! ```text
//!   capacity = n.div_ceil(8) * 8   (padded to SIMD width)
//!
//!   params: [dt×cap] [inv_tau×cap] [v_rest×cap] …
//!   state:  [v×cap] [u×cap] [refr×cap] …
//!
//!   Column stride = capacity — ensures SIMD chunks are contiguous.
//! ```
//!
//! ## Clean Bus
//! - `i_in` → ctx.i_syn (post-synaptic current, clamped to ±100)
//! - `i_pre` / `i_post` → ctx.i_pre / ctx.i_post (for STDP, optional)

#![deny(unsafe_op_in_unsafe_fn)]

use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::core::arch::SimdType;
use crate::core::buf::AlignedVec;
use crate::core::context::SimdContext;
use crate::core::lane::SimdLane;
use crate::zoo::Model;

// Import SIMD state load/save macros from runner
use crate::{load_state_scalar, load_state_simd, save_state_scalar, save_state_simd};

// ═══════════════════════════════════════════════════════════════
//  Input Current Clamping
// ═══════════════════════════════════════════════════════════════

/// Default range for input current clamping (nA or pA depending on model).
///
/// # Rationale
/// Clamps input current to ±100 to prevent:
/// 1. **Numerical overflow** in exponential models (AdEx, HH)
/// 2. **Unrealistic firing rates** (>1000 Hz) that break refractory logic
/// 3. **NaN propagation** from extreme exp() arguments
///
/// # Biological context
/// - Typical cortical neuron: I_syn ∈ [-50, +50] pA
/// - Strong synaptic input: ~100 pA (saturating)
/// - Values >100 pA are non-physiological for point neuron models
///
/// # Override
/// To disable clamping (e.g., for abstract models), modify the
/// `.clamp()` calls in `step_scalar()` and `step_simd()` methods.
///
/// **Note:** This is a **silent operation** — values outside this range
/// are clamped without warning. For models requiring larger currents
/// (e.g., HH with 500 pA), consider removing the clamp or adjusting the range.
pub const I_SYN_CLAMP_RANGE: (f32, f32) = (-100.0, 100.0);

// ═══════════════════════════════════════════════════════════════
//  NeuralPopulation — High-Level Safe API
// ═══════════════════════════════════════════════════════════════

/// A homogeneous population of `M` neurons with aligned SIMD storage.
pub struct NeuralPopulation<M: Model> {
    params: AlignedVec,
    state: AlignedVec,
    n: usize,
    capacity: usize,
    pre_bus: Option<AlignedVec>,
    post_bus: Option<AlignedVec>,
    _marker: std::marker::PhantomData<M>,
}

impl<M: Model> NeuralPopulation<M> {
    /// Create a new population. Memory is padded to 8-wide SIMD alignment.
    #[must_use]
    pub fn new(count: usize) -> Self {
        let capacity = count
            .checked_add(7)
            .map(|x| x / 8 * 8)
            .expect("NeuralPopulation: count overflows when rounding to SIMD width");
        let param_count = capacity
            .checked_mul(M::N_PARAMS)
            .expect("NeuralPopulation: params size overflow");
        let state_count = capacity
            .checked_mul(M::N_STATE)
            .expect("NeuralPopulation: state size overflow");

        Self {
            params: AlignedVec::new(param_count, 0.0),
            state: AlignedVec::new(state_count, 0.0),
            n: count,
            capacity,
            pre_bus: None,
            post_bus: None,
            _marker: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.n
    }
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Enable pre/post spike buses for STDP/plasticity.
    pub fn enable_buses(&mut self) {
        self.pre_bus = Some(AlignedVec::new(self.capacity, 0.0));
        self.post_bus = Some(AlignedVec::new(self.capacity, 0.0));
    }

    /// Disable and free pre/post spike buses.
    pub fn disable_buses(&mut self) {
        self.pre_bus = None;
        self.post_bus = None;
    }

    /// Check if pre/post buses are enabled.
    pub fn buses_enabled(&self) -> bool {
        self.pre_bus.is_some()
    }

    // ─── Fill helpers ──────────────────────────────────────────

    /// Fill a state column with a value (all neurons).
    ///
    /// # Panics
    /// If `col >= M::N_STATE` — an out-of-range column would wrap
    /// `col * capacity` in release mode and write outside the allocation.
    pub fn fill_state(&mut self, col: usize, value: f32) {
        assert!(
            col < M::N_STATE,
            "fill_state: column {col} out of range for model with N_STATE={}",
            M::N_STATE
        );
        let start = col
            .checked_mul(self.capacity)
            .expect("fill_state: capacity overflow");
        for v in &mut self.state.as_mut_slice()[start..start + self.n] {
            *v = value;
        }
    }

    /// Fill a param column with a value (all neurons).
    ///
    /// # Panics
    /// If `col >= M::N_PARAMS` — an out-of-range column would wrap
    /// `col * capacity` in release mode and write outside the allocation.
    pub fn fill_param(&mut self, col: usize, value: f32) {
        assert!(
            col < M::N_PARAMS,
            "fill_param: column {col} out of range for model with N_PARAMS={}",
            M::N_PARAMS
        );
        let start = col
            .checked_mul(self.capacity)
            .expect("fill_param: capacity overflow");
        for v in &mut self.params.as_mut_slice()[start..start + self.n] {
            *v = value;
        }
    }

    // ─── Slice access ───────────────────────────────────────────

    /// Read-only access to a state column.
    pub fn state_slice(&self, col: usize) -> &[f32] {
        self.assert_state_col(col);
        let start = col * self.capacity;
        &self.state.as_slice()[start..start + self.n]
    }

    /// Mutable access to a state column (safe slice).
    pub fn state_slice_mut(&mut self, col: usize) -> &mut [f32] {
        self.assert_state_col(col);
        let start = col * self.capacity;
        &mut self.state.as_mut_slice()[start..start + self.n]
    }

    /// Mutable access to a param column (safe slice).
    pub fn param_slice_mut(&mut self, col: usize) -> &mut [f32] {
        self.assert_param_col(col);
        let start = col * self.capacity;
        &mut self.params.as_mut_slice()[start..start + self.n]
    }

    /// Unsafe mutable pointer to a state column (internal use only).
    ///
    /// # Safety
    /// `col` must be < `M::N_STATE`; the returned pointer is valid for
    /// `self.n` elements and is 32-byte aligned. Writes must stay within the
    /// column — no adjacent-column overflow (stride = capacity).
    ///
    /// Note: unused by the crate today; kept `pub(crate)` (not public) so
    /// the unsafe surface stays out of the API until a real user needs it.
    #[allow(dead_code)]
    pub(crate) unsafe fn state_ptr_mut(&mut self, col: usize) -> *mut f32 {
        self.assert_state_col(col);
        // SAFETY: `col < N_STATE` (asserted above), stride = capacity, so
        // the offset stays within the allocation.
        unsafe { self.state.as_mut_ptr().add(col * self.capacity) }
    }

    /// Unsafe mutable pointer to a param column (internal use only).
    ///
    /// # Safety
    /// `col` must be < `M::N_PARAMS`; the returned pointer is valid for
    /// `self.n` elements and is 32-byte aligned. Writes must stay within the
    /// column — no adjacent-column overflow (stride = capacity).
    ///
    /// Note: unused by the crate today; kept `pub(crate)` (not public) so
    /// the unsafe surface stays out of the API until a real user needs it.
    #[allow(dead_code)]
    pub(crate) unsafe fn param_ptr_mut(&mut self, col: usize) -> *mut f32 {
        self.assert_param_col(col);
        // SAFETY: `col < N_PARAMS` (asserted above), stride = capacity, so
        // the offset stays within the allocation.
        unsafe { self.params.as_mut_ptr().add(col * self.capacity) }
    }

    /// Read-only access to a param column.
    pub fn param_slice(&self, col: usize) -> &[f32] {
        self.assert_param_col(col);
        let start = col * self.capacity;
        &self.params.as_slice()[start..start + self.n]
    }

    #[inline(always)]
    fn assert_state_col(&self, col: usize) {
        assert!(
            col < M::N_STATE,
            "state_slice: column {col} out of range for model with N_STATE={}",
            M::N_STATE
        );
    }

    #[inline(always)]
    fn assert_param_col(&self, col: usize) {
        assert!(
            col < M::N_PARAMS,
            "param_slice: column {col} out of range for model with N_PARAMS={}",
            M::N_PARAMS
        );
    }

    // ─── Step ──────────────────────────────────────────────────

    /// Advance one time step. Writes spikes to `spikes_out` (0 or 1).
    pub fn step(&mut self, i_in: &[f32], spikes_out: &mut [i32]) {
        assert_eq!(i_in.len(), self.n);
        assert_eq!(spikes_out.len(), self.n);
        if self.n == 0 {
            return;
        }
        if self.n <= 8 {
            self.step_scalar(i_in, spikes_out);
        } else {
            // SAFETY: `step_simd` requires n > 8 (guaranteed above) and
            // padded-to-capacity buffers (guaranteed by `NeuralPopulation::new`).
            unsafe {
                self.step_simd(i_in, spikes_out);
            }
        }
    }

    fn step_scalar(&mut self, i_in: &[f32], spikes_out: &mut [i32]) {
        let cap = self.capacity;
        // SAFETY: `params` is an allocated AlignedVec (never dangling when
        // n > 0); element 0 is within bounds. This is the broadcast dt used
        // by `dt_scalar()`/`dt_simd()` — see `SimdContext::dt` for the per-lane
        // variant that no current model uses.
        let dt = unsafe { *self.params.as_ptr() };
        // SAFETY: NonNull from a live AlignedVec allocation — infallible;
        // hoisted out of the loop (it was re-checked on every neuron).
        let params_ptr = unsafe { NonNull::new_unchecked(self.params.as_ptr() as *mut f32) };

        for i in 0..self.n {
            let ii = i_in[i].clamp(I_SYN_CLAMP_RANGE.0, I_SYN_CLAMP_RANGE.1);
            let sp = self.state.as_ptr();
            let state = load_state_scalar!(sp, cap, i);

            // SAFETY: buses (when enabled) are AlignedVecs of `capacity`;
            // i < self.n <= capacity.
            let pre = self
                .pre_bus
                .as_ref()
                .map_or(0.0, |b| unsafe { *b.as_ptr().add(i) });
            let post = self
                .post_bus
                .as_ref()
                .map_or(0.0, |b| unsafe { *b.as_ptr().add(i) });

            let mut ctx = SimdContext::<f32> {
                state,
                params_ptr,
                param_count: M::N_PARAMS,
                n: cap,
                off: i,
                i_syn: ii,
                i_pre: pre,
                i_post: post,
                dt_val: dt,
                spike_mask: 0.0,
                _marker: PhantomData,
            };

            M::step(&mut ctx);

            let sm = self.state.as_mut_ptr();
            save_state_scalar!(sm, cap, i, ctx);
            spikes_out[i] = ctx.spike_mask().move_mask() as i32;
        }
    }

    /// SIMD step for `n > 8`. SAFETY contract (invariants this method relies
    /// on, all guaranteed by `step()` + `AlignedVec`):
    /// - `n8 = n & !7` is the largest multiple of 8 ≤ n, so every chunk
    ///   satisfies `off + 8 ≤ n` — the 8-wide reads of `i_in` stay in bounds;
    /// - state/params/pre/post buses are padded to `capacity` (multiple of 8),
    ///   so `off + 7 < capacity` for every lane;
    /// - column stride is `capacity` in both the SIMD and the scalar tail,
    ///   so chunk and tail writes never overlap or overflow a column.
    unsafe fn step_simd(&mut self, i_in: &[f32], spikes_out: &mut [i32]) {
        let n8 = self.n & !7;
        // SAFETY: same as step_scalar — live AlignedVec, element 0 in bounds.
        let dt = unsafe { *self.params.as_ptr() };
        // SAFETY: NonNull from a live AlignedVec allocation — infallible;
        // hoisted once (was rebuilt with `.expect()` on every chunk).
        let params_ptr = unsafe { NonNull::new_unchecked(self.params.as_ptr() as *mut f32) };
        let cap = self.capacity;

        for off in (0..n8).step_by(8) {
            if off + 64 < n8 {
                // SAFETY: off + 64 < n8 <= capacity — prefetch stays in bounds.
                unsafe {
                    crate::core::util::prefetch_read(self.state.as_ptr(), off + 64);
                    crate::core::util::prefetch_read(self.params.as_ptr(), off + 64);
                }
            }

            let sp = self.state.as_ptr();
            let state = load_state_simd!(sp, cap, off, SimdType);

            // SAFETY: off + 8 <= n (see contract above); `i_in` may be any
            // user slice, so read_unaligned is required (i_in is NOT
            // guaranteed 32-byte aligned).
            let i_syn = unsafe { (i_in.as_ptr().add(off) as *const SimdType).read_unaligned() }
                .max(SimdType::splat(I_SYN_CLAMP_RANGE.0))
                .min(SimdType::splat(I_SYN_CLAMP_RANGE.1));

            // SAFETY: buses are AlignedVecs of `capacity` — off + 8 <= n8
            //     <= capacity, so all 8 lanes are within the padded buffer.
            let i_pre = self
                .pre_bus
                .as_ref()
                .map_or(SimdType::splat(0.0), |b| unsafe {
                    (b.as_ptr().add(off) as *const SimdType).read_unaligned()
                });
            let i_post = self
                .post_bus
                .as_ref()
                .map_or(SimdType::splat(0.0), |b| unsafe {
                    (b.as_ptr().add(off) as *const SimdType).read_unaligned()
                });

            let mut ctx = SimdContext::<SimdType> {
                state,
                params_ptr,
                param_count: M::N_PARAMS,
                n: cap,
                off,
                i_syn,
                i_pre,
                i_post,
                dt_val: dt,
                spike_mask: SimdType::splat(0.0),
                _marker: PhantomData,
            };

            M::step(&mut ctx);

            let sm = self.state.as_mut_ptr();
            save_state_simd!(sm, cap, off, ctx, SimdType);

            let m = ctx.spike_mask().move_mask() as u32;
            for j in 0..8 {
                spikes_out[off + j] = ((m >> j) & 1) as i32;
            }
        }

        // Scalar tail (unpadded neurons, n8 < self.n). Identical math to
        // step_scalar — a fresh scalar context per neuron.
        if n8 < self.n {
            for i in n8..self.n {
                let ii = i_in[i].clamp(I_SYN_CLAMP_RANGE.0, I_SYN_CLAMP_RANGE.1);
                let sp = self.state.as_ptr();
                let state = load_state_scalar!(sp, cap, i);

                // SAFETY: buses are AlignedVecs of `capacity`; i < n <= cap.
                let pre = self
                    .pre_bus
                    .as_ref()
                    .map_or(0.0, |b| unsafe { *b.as_ptr().add(i) });
                let post = self
                    .post_bus
                    .as_ref()
                    .map_or(0.0, |b| unsafe { *b.as_ptr().add(i) });

                let mut ctx = SimdContext::<f32> {
                    state,
                    params_ptr,
                    param_count: M::N_PARAMS,
                    n: cap,
                    off: i,
                    i_syn: ii,
                    i_pre: pre,
                    i_post: post,
                    dt_val: dt,
                    spike_mask: 0.0,
                    _marker: PhantomData,
                };

                M::step(&mut ctx);

                let sm = self.state.as_mut_ptr();
                save_state_scalar!(sm, cap, i, ctx);
                spikes_out[i] = ctx.spike_mask().move_mask() as i32;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  Backward-compat re-exports (canonical path: crate::core::runner::run_neuron)
// ═══════════════════════════════════════════════════════════════

#[cfg(feature = "rayon")]
pub use crate::core::runner::run_neuron_par;
pub use crate::core::runner::{run_neuron, run_neuron_ext};

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zoo::neurons::LIFModel;

    #[test]
    fn test_population_create() {
        let pop = NeuralPopulation::<LIFModel>::new(100);
        assert_eq!(pop.len(), 100);
        assert!(pop.capacity().is_multiple_of(8));
    }

    #[test]
    fn test_population_empty() {
        let pop = NeuralPopulation::<LIFModel>::new(0);
        assert!(pop.is_empty());
    }

    #[test]
    fn test_population_step_basic() {
        let mut pop = NeuralPopulation::<LIFModel>::new(8);
        pop.fill_param(0, 0.5);
        pop.fill_param(1, 0.1);
        pop.fill_param(2, -70.0);
        pop.fill_param(3, -54.0);
        pop.fill_param(4, -70.0);
        pop.fill_param(5, 2.0);
        pop.fill_state(0, -70.0);

        let i_in = vec![30.0; 8];
        let mut spikes = vec![0i32; 8];
        pop.step(&i_in, &mut spikes);

        assert!(pop.state_slice(0)[0] > -70.0, "Membrane should depolarize");
    }

    #[test]
    fn test_population_step_non_aligned() {
        let mut pop = NeuralPopulation::<LIFModel>::new(10);
        pop.fill_param(0, 0.5);
        pop.fill_param(1, 0.1);
        pop.fill_param(2, -70.0);
        pop.fill_param(3, -54.0);
        pop.fill_param(4, -70.0);
        pop.fill_param(5, 2.0);
        pop.fill_state(0, -70.0);

        let i_in = vec![30.0; 10];
        let mut spikes = vec![0i32; 10];

        for _ in 0..20 {
            pop.step(&i_in, &mut spikes);
        }

        let v8 = pop.state_slice(0)[8];
        let v0 = pop.state_slice(0)[0];
        assert!(
            (v8 - v0).abs() < 0.001,
            "Tail v[8]={:.3} != SIMD v[0]={:.3}",
            v8,
            v0
        );
    }

    #[test]
    fn test_population_buses() {
        let mut pop = NeuralPopulation::<LIFModel>::new(8);
        pop.enable_buses();
        pop.fill_param(0, 0.5);
        pop.fill_param(1, 0.1);
        pop.fill_param(2, -70.0);
        pop.fill_param(3, -54.0);
        pop.fill_param(4, -70.0);
        pop.fill_param(5, 2.0);
        pop.fill_state(0, -70.0);

        let i_in = vec![30.0; 8];
        let mut spikes = vec![0i32; 8];
        pop.step(&i_in, &mut spikes);
    }

    /// n = 13 (not a multiple of 8): neurons 0..8 run through the SIMD path,
    /// neurons 8..13 through the scalar tail — they must produce bit-identical
    /// dynamics (same params, same input, same code). Guards against any
    /// SIMD/tail divergence sneaking in.
    #[test]
    fn test_population_simd_tail_parity() {
        let mut pop = NeuralPopulation::<LIFModel>::new(13);
        pop.fill_param(0, 0.5);
        pop.fill_param(1, 0.1);
        pop.fill_param(2, -70.0);
        pop.fill_param(3, -54.0);
        pop.fill_param(4, -70.0);
        pop.fill_param(5, 2.0);
        pop.fill_state(0, -70.0);

        // Same input for every neuron → v[8] (tail) must equal v[0] (SIMD)
        // at every step, bit for bit.
        let i_in = vec![30.0; 13];
        let mut spikes = vec![0i32; 13];
        for _ in 0..100 {
            pop.step(&i_in, &mut spikes);
            for i in 1..13 {
                assert_eq!(
                    pop.state_slice(0)[i].to_bits(),
                    pop.state_slice(0)[0].to_bits(),
                    "neuron {i} (SIMD if <8, tail otherwise) diverged from neuron 0"
                );
            }
        }
    }

    #[test]
    fn test_population_spike_count() {
        let mut pop = NeuralPopulation::<LIFModel>::new(8);
        pop.fill_param(0, 0.5);
        pop.fill_param(1, 0.1);
        pop.fill_param(2, -70.0);
        pop.fill_param(3, -54.0);
        pop.fill_param(4, -70.0);
        pop.fill_param(5, 2.0);
        pop.fill_state(0, -70.0);

        let i_strong = vec![100.0; 8];
        let mut spikes = vec![0i32; 8];
        let mut total = 0i32;

        for _ in 0..100 {
            pop.step(&i_strong, &mut spikes);
            total += spikes.iter().sum::<i32>();
        }
        assert!(total > 0, "LIF should spike with I=100, got {total}");
    }

    #[test]
    fn test_population_stress_silence() {
        let mut pop = NeuralPopulation::<LIFModel>::new(8);
        pop.fill_param(0, 0.5);
        pop.fill_param(1, 0.1);
        pop.fill_param(2, -70.0);
        pop.fill_param(3, -54.0);
        pop.fill_param(4, -70.0);
        pop.fill_param(5, 2.0);
        pop.fill_state(0, -70.0);

        let i_zero = vec![0.0; 8];
        let mut spikes = vec![0i32; 8];

        for step in 0..10000 {
            pop.step(&i_zero, &mut spikes);
            for i in 0..8 {
                assert!(!pop.state_slice(0)[i].is_nan(), "v[{i}] NaN at step {step}");
            }
        }
        assert_eq!(spikes.iter().sum::<i32>(), 0, "No spikes at I=0");
    }

    #[test]
    fn test_population_stress_explosion() {
        let mut pop = NeuralPopulation::<LIFModel>::new(8);
        pop.fill_param(0, 0.5);
        pop.fill_param(1, 0.1);
        pop.fill_param(2, -70.0);
        pop.fill_param(3, -54.0);
        pop.fill_param(4, -70.0);
        pop.fill_param(5, 2.0);
        pop.fill_state(0, -70.0);

        let i_big = vec![1000.0; 8];
        let mut spikes = vec![0i32; 8];

        // Accumulate over the whole run — the per-step buffer is phase-dependent
        // (with frozen refractory, spikes land on a fixed cadence), so checking
        // only the final step's buffer would be flaky.
        let mut total = 0i64;
        for step in 0..10000 {
            pop.step(&i_big, &mut spikes);
            total += spikes.iter().map(|&s| s as i64).sum::<i64>();
            for i in 0..8 {
                assert!(!pop.state_slice(0)[i].is_nan(), "v[{i}] NaN at step {step}");
                assert!(pop.state_slice(0)[i].is_finite(), "v[{i}] non-finite");
            }
        }
        assert!(total > 0, "Should spike with I=1000");
    }
}
