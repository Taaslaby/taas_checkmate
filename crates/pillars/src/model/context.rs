use super::lane::SimdLane;

/// Zero-Copy SIMD Execution Context for Model Step Iterations.
///
/// # Buses
/// - `i_syn`: Synaptic input current (post-synaptic)
/// - `i_pre`: Pre-synaptic spike bus (for STDP, synaptic plasticity)
/// - `i_post`: Post-synaptic spike bus (for STDP, Hebbian learning)
///
/// # Principle: Clean Bus
/// All external events enter through ctx only.
/// No manual inject() calls — pre/post spikes flow through dedicated buses.
pub struct SimdContext<S: SimdLane> {
    pub state: [S; 6],
    pub params_ptr: *const f32,
    pub n: usize,
    pub off: usize,
    pub i_syn: S,
    pub i_pre: S,
    pub i_post: S,
    pub dt_val: f32,
    pub spike_mask: S,
}

impl<S: SimdLane> SimdContext<S> {
    #[inline(always)] pub fn s(&self, i: usize) -> S { self.state[i] }
    #[inline(always)] pub fn set_s(&mut self, i: usize, v: S) { self.state[i] = v; }

    /// Parameters: SIMD (SoA vector load) / scalar
    #[inline(always)]
    pub fn p(&self, i: usize) -> S {
        unsafe { S::load_param(self.params_ptr, i, self.n, self.off) }
    }

    /// Type-safe parameter access via usize-convertible enum.
    /// Removes the `as usize` cast noise.
    #[inline(always)]
    pub fn p_enum<E: Into<usize> + Copy>(&self, e: E) -> S {
        unsafe { S::load_param(self.params_ptr, e.into(), self.n, self.off) }
    }

    #[inline(always)]
    pub fn pf(&self, i: usize) -> f32 {
        unsafe { *self.params_ptr.add(i * self.n + self.off) }
    }

    #[inline(always)] pub fn i(&self) -> S { self.i_syn }
    #[inline(always)] pub fn i_syn_ref(&self) -> &S { &self.i_syn }
    #[inline(always)] pub fn pre(&self) -> S { self.i_pre }
    #[inline(always)] pub fn post(&self) -> S { self.i_post }
    #[inline(always)] pub fn dt(&self) -> S { S::splat(self.dt_val) }
    #[inline(always)] pub fn dtf(&self) -> f32 { self.dt_val }

    /// Create a raw context for StorageBuffer-based execution (no load_i needed).
    #[inline(always)]
    pub fn new_raw(params_ptr: *const f32, capacity: usize, offset: usize, dt: f32) -> Self {
        Self {
            state: [S::zero(); 6],
            params_ptr,
            n: capacity,
            off: offset,
            i_syn: S::zero(),
            i_pre: S::zero(),
            i_post: S::zero(),
            dt_val: dt,
            spike_mask: S::zero(),
        }
    }

    /// Load i_syn from a slice (length must equal lane SIZE).
    ///
    /// # Safety
    /// `slice` must have at least `S::SIZE` elements, and all elements must be readable.
    #[inline(always)]
    pub fn load_i(&mut self, slice: &[f32]) {
        unsafe { self.i_syn = S::load_param(slice.as_ptr(), 0, S::SIZE, 0); }
    }

    /// Load one state channel from a flat slice (e.g. post-synaptic spikes into CaPlasticity s(3)).
    ///
    /// # Safety
    /// `slice` must have at least `S::SIZE` elements. `state_idx` must be < 6.
    #[inline(always)]
    pub fn load_state_channel(&mut self, state_idx: usize, slice: &[f32]) {
        unsafe {
            self.state[state_idx] = S::load_param(slice.as_ptr(), 0, S::SIZE, 0);
        }
    }

    /// Save one state channel back to a SoA column pointer.
    ///
    /// # Safety
    /// `ptr` must point to a valid `capacity`-sized column. `state_idx` must be < 6.
    #[inline(always)]
    pub unsafe fn save_state_channel(&self, state_idx: usize, ptr: *mut f32) {
        self.state[state_idx].write_to_slice(std::slice::from_raw_parts_mut(ptr.add(self.off), S::SIZE));
    }

    #[inline(always)] pub fn spike(&mut self, cond: S) {
        self.spike_mask = cond;
    }
    #[inline(always)] pub fn spike_mask(&self) -> S { self.spike_mask }
}
