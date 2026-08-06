//! # Triangulated LUT — Compile Brain Models into Lookup Tables
//!
//! Converts expensive ODE-based neuron models into a 65536-entry lookup table
//! keyed by `(quantized_V, binary_input_history)`.
//!
//! ## Why
//!
//! HH: 7 exp() + FMA = ~170 cycles/step. LUT: 1 load + 1 add = ~5 cycles/step.
//!
//! ## Calibration
//!
//! 1. Run reference `Model` with random Poisson inputs
//! 2. At each step: `(V_q, history) → (delta_v, spike)`
//! 3. Average all observations → fill LUT
//! 4. Interpolate missing entries
//!
//! ## Inference
//!
//! Each step = 1 bit shift + 1 quantize + 1 table load + 1 f32 add.
//! ~10–30× faster than ODE integration.

/// Number of LUT entries: 256 V-levels × 16 history states = 4096.
///
/// 4096 × 8 bytes = 32 KB — fits in L1 cache on modern CPUs.
/// (Previous 65536 × 8 = 512 KB exceeded L2 causing cache-miss overhead.)
pub const LUT_SIZE: usize = 4096;

/// Number of history bits (4-bit = 16 states, 5+ bit = more cache misses).
pub const HISTORY_BITS: usize = 4;
pub const HISTORY_MASK: u8 = 0x0F;

/// A single LUT entry.
#[derive(Copy, Clone, Debug, Default)]
pub struct LutEntry {
    /// Change in membrane potential (mV).
    pub delta_v: f32,
    /// 1.0 if neuron spikes this step, 0.0 otherwise.
    pub spike: f32,
}

/// Pre-computed lookup table for a single neuron type.
///
/// 4096 × 8 bytes = 32 KB — fits in L1 cache on modern CPUs.
#[derive(Clone)]
pub struct TriangLUT {
    pub entries: Vec<LutEntry>,
    pub v_min: f32,
    pub v_max: f32,
}

impl TriangLUT {
    /// Create a new LUT with zeroed entries (heap allocated).
    #[inline(always)]
    pub fn new(v_min: f32, v_max: f32) -> Self {
        let entries = vec![LutEntry::default(); LUT_SIZE];
        Self {
            entries,
            v_min,
            v_max,
        }
    }

    /// Quantize a voltage value to 8 bits (0..255).
    #[inline(always)]
    pub fn quantize_v(&self, v: f32) -> u16 {
        let clamped = v.clamp(self.v_min, self.v_max);
        let range = self.v_max - self.v_min;
        ((clamped - self.v_min) * (255.0 / range)) as u16
    }

    /// Build 12-bit key from V-quantized (high 8 bits) and history (low 4 bits).
    ///
    /// 8-bit V × 4-bit history = 4096 entries = 32 KB (L1 cache).
    #[inline(always)]
    pub fn key(v_q: u16, history: u8) -> usize {
        (v_q as usize) << HISTORY_BITS | (history & HISTORY_MASK) as usize
    }

    /// Look up a single entry.
    #[inline(always)]
    pub fn lookup(&self, v_q: u16, history: u8) -> LutEntry {
        self.entries[Self::key(v_q, history)]
    }
}

/// A single input mode for calibration: (rate, weight, steps_per_cycle).
///
/// Cycles through all modes sequentially, each lasting `steps_per_cycle` steps.
pub type InputMode = (f32, f32, usize);

/// Configuration for LUT calibration.
#[derive(Clone)]
pub struct LutConfig {
    /// Number of calibration steps.
    pub n_steps: usize,
    /// Input profiles to cycle through (rate, weight, steps_per_cycle).
    ///
    /// Example: `[(1.0, 50.0, 50), (0.0, 0.0, 50)]` cycles 50 steps at I=50,
    /// then 50 steps at I=0 — mixed calibration for wider coverage.
    pub input_modes: Vec<InputMode>,
    /// Membrane voltage range for quantization.
    pub v_min: f32,
    pub v_max: f32,
    /// Simulation time step (ms).
    pub dt: f32,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for LutConfig {
    fn default() -> Self {
        Self {
            n_steps: 5_000_000,
            input_modes: vec![(0.03, 30.0, 1)],
            v_min: -100.0,
            v_max: 60.0,
            dt: 0.5,
            seed: 42,
        }
    }
}

/// Build a LUT from a reference `Model` by running calibration simulations.
pub struct LutBuilder {
    config: LutConfig,
}

impl LutBuilder {
    pub fn new(config: LutConfig) -> Self {
        Self { config }
    }

    /// Run calibration and produce a `TriangLUT`.
    ///
    /// `make_params` creates a param vector for the reference model
    /// with `n` neurons (SoA layout). `n` should be 1 (scalar calibration).
    pub fn build<M: crate::zoo::Model + Sized>(
        &self,
        params: &[f32],
        state_init: &[f32],
    ) -> TriangLUT {
        let cfg = &self.config;
        let n = 1; // calibrate 1 neuron at a time

        // Allocate state buffers (run_neuron needs up to 6)
        let mut v: Vec<f32> = vec![state_init[0]]; // s0
        let mut u: Vec<f32> = vec![if state_init.len() > 1 {
            state_init[1]
        } else {
            0.0
        }];
        let mut h: Vec<f32> = vec![if state_init.len() > 2 {
            state_init[2]
        } else {
            0.0
        }];
        let mut r: Vec<f32> = vec![if state_init.len() > 3 {
            state_init[3]
        } else {
            0.0
        }];
        let mut s: Vec<f32> = vec![if state_init.len() > 4 {
            state_init[4]
        } else {
            0.0
        }];
        let mut t: Vec<f32> = vec![if state_init.len() > 5 {
            state_init[5]
        } else {
            0.0
        }];

        // Don't pass pointers to empty vecs — use null-safe run_neuron
        // Actually run_neuron reads all 6 pointers. We need to pass valid ones.
        // Ensure all vecs have at least n elements
        while u.len() < n {
            u.push(0.0);
        }
        while h.len() < n {
            h.push(0.0);
        }
        while r.len() < n {
            r.push(0.0);
        }
        while s.len() < n {
            s.push(0.0);
        }
        while t.len() < n {
            t.push(0.0);
        }

        let mut i_in = vec![0.0f32; n];
        let mut spikes = vec![0i32; n];

        // Build LUT with double-precision accumulators
        let mut sum_dv = vec![0.0f64; LUT_SIZE];
        let mut sum_spk = vec![0u64; LUT_SIZE];
        let mut count = vec![0u64; LUT_SIZE];

        let lut_tmp = TriangLUT::new(cfg.v_min, cfg.v_max);

        // Simple LCG random (no external dep)
        let mut rng_state = cfg.seed;

        // Calibration loop
        let mut history: u8 = 0;

        for _step in 0..cfg.n_steps {
            let v_before = v[0];
            let v_q = lut_tmp.quantize_v(v_before); // Select input mode for this step (cycle through modes)
            let mode_idx = _step % cfg.input_modes.len();
            let (mode_rate, mode_weight, _mode_dur) = cfg.input_modes[mode_idx];

            // Poisson spike within current mode
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let rnd = ((rng_state >> 33) as u32) as f32 / 4294967296.0;
            let input_bit = if rnd < mode_rate { 1u8 } else { 0u8 };
            i_in[0] = if input_bit != 0 { mode_weight } else { 0.0 };

            // Build key from state BEFORE step
            let key = TriangLUT::key(v_q, history);

            // Step reference model
            unsafe {
                crate::core::population::run_neuron::<M>(
                    v.as_mut_ptr(),
                    u.as_mut_ptr(),
                    h.as_mut_ptr(),
                    r.as_mut_ptr(),
                    s.as_mut_ptr(),
                    t.as_mut_ptr(),
                    i_in.as_ptr(),
                    spikes.as_mut_ptr(),
                    params.as_ptr(),
                    n,
                );
            }

            let delta_v = v[0] - v_before;
            let spiked = spikes[0] != 0;

            sum_dv[key] += delta_v as f64;
            if spiked {
                sum_spk[key] += 1;
            }
            count[key] += 1;

            // Shift history (4-bit: mask to 0x0F)
            history = (history << 1 | input_bit) & HISTORY_MASK;
        }

        // Build final LUT by averaging
        let mut lut = TriangLUT::new(cfg.v_min, cfg.v_max);
        for k in 0..LUT_SIZE {
            if count[k] > 0 {
                let c = count[k] as f64;
                lut.entries[k] = LutEntry {
                    delta_v: (sum_dv[k] / c) as f32,
                    spike: (sum_spk[k] as f64 / c) as f32,
                };
            }
        }

        // Fill missing entries via nearest-neighbor interpolation
        fill_missing(&mut lut);

        lut
    }
}

/// Fill missing LUT entries by scanning for nearest valid neighbor.
fn fill_missing(lut: &mut TriangLUT) {
    for k in 0..LUT_SIZE {
        if lut.entries[k].count_is_zero() {
            let v_q = (k >> HISTORY_BITS) as u16;
            let hist = (k & HISTORY_MASK as usize) as u8;
            lut.entries[k] = nearest_valid(lut, v_q, hist);
        }
    }
}

trait CountIsZero {
    fn count_is_zero(&self) -> bool;
}

impl CountIsZero for LutEntry {
    #[inline(always)]
    fn count_is_zero(&self) -> bool {
        self.delta_v == 0.0 && self.spike == 0.0
    }
}

fn nearest_valid(lut: &TriangLUT, v_q: u16, hist: u8) -> LutEntry {
    // Linear scan by increasing Manhattan distance
    // hist is 4-bit (0..15), v_q is 8-bit (0..255)
    let dmax = 255u16;
    for d in 1u16..=dmax {
        // Same V, adjacent histories (4-bit: 0..15)
        let dh_lim = d.min(15);
        for dh in 0u16..=dh_lim {
            let dh_u8 = dh as u8;
            for &h in &[hist.wrapping_add(dh_u8), hist.wrapping_sub(dh_u8)] {
                if h == hist {
                    continue;
                }
                let e = lut.entries[TriangLUT::key(v_q, h)];
                if !e.count_is_zero() {
                    return e;
                }
            }
        }
        // Adjacent V, same history
        for &v in &[v_q.saturating_add(d), v_q.saturating_sub(d)] {
            if v > 255 {
                continue;
            }
            let e = lut.entries[TriangLUT::key(v, hist)];
            if !e.count_is_zero() {
                return e;
            }
        }
    }
    LutEntry::default()
}

// ═══════════════════════════════════════════════════════════════
//  AVX2 Native Gather (eliminates to_array/from_array roundtrip)
// ═══════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    //! Native SIMD LUT gather using `_mm256_i32gather_ps`.
    //! Bypasses to_array/from_array — processes 8 neurons entirely in registers.
    use super::TriangLUT;
    use core::arch::x86_64::*;

    /// Single-step inference for 8 neurons using raw __m256 AVX2 gather.
    ///
    /// Returns (v_new, hist_new, spike_mask).
    ///
    /// # Safety
    /// Must be called on x86_64 with AVX2 support.
    #[inline(always)]
    pub unsafe fn triang_step_avx2_raw(
        v: __m256,
        history: __m256,
        i_syn: __m256,
        lut: &TriangLUT,
        v_thresh: f32,
        input_thresh: f32,
    ) -> (__m256, __m256, __m256) {
        let v_min = _mm256_set1_ps(lut.v_min);
        let v_max = _mm256_set1_ps(lut.v_max);
        let range = lut.v_max - lut.v_min;
        let scale_v = _mm256_set1_ps(255.0 / range);
        let one_epi32 = _mm256_set1_epi32(1);
        let mask_4bit = _mm256_set1_epi32(0x0F); // 4-bit history
                                                 // 1. Clamp V, quantize to 0..255
        let v_clamped = _mm256_min_ps(_mm256_max_ps(v, v_min), v_max);
        let v_scaled = _mm256_mul_ps(_mm256_sub_ps(v_clamped, v_min), scale_v);
        let v_q = _mm256_cvttps_epi32(v_scaled); // truncate f32→i32

        // 2. History as i32, mask to 4 bits
        let hist_i32 = _mm256_cvttps_epi32(history);
        let hist_masked = _mm256_and_si256(hist_i32, mask_4bit);

        // 3. Index = (v_q << 4) | hist_masked  (12-bit key, max 4095)
        // v_q ≤ 255 (8-bit), hist_masked ≤ 15 (4-bit) → index ≤ 255*16+15 = 4095
        // No AND needed — the 12-bit range is provably satisfied.
        let v_q_sh4 = _mm256_slli_epi32(v_q, 4);
        let index = _mm256_or_si256(v_q_sh4, hist_masked);

        // 4. Gather delta_v (LutEntry stride=8, delta_v at +0)
        let entries_ptr = lut.entries.as_ptr() as *const f32;
        let delta_v = _mm256_i32gather_ps(entries_ptr, index, 8);

        // 5. Gather spike (LutEntry stride=8, spike at +4)
        let spike_base = (lut.entries.as_ptr() as *const u8).add(4) as *const f32;
        let spike_f32 = _mm256_i32gather_ps(spike_base, index, 8);

        // 6. V += delta_v
        let v_new = _mm256_add_ps(v, delta_v);

        // 7. New history = (old_hist << 1) | input_bit
        // input_bit = i_syn > input_thresh (bitmask 0 or 1)
        let input_cmp = _mm256_cmp_ps(i_syn, _mm256_set1_ps(input_thresh), _CMP_GT_OQ);
        let input_cmp_i32 = _mm256_castps_si256(input_cmp);
        let input_bit = _mm256_and_si256(input_cmp_i32, one_epi32);
        let new_hist_i32 = _mm256_or_si256(_mm256_slli_epi32(hist_masked, 1), input_bit);
        let new_hist = _mm256_cvtepi32_ps(new_hist_i32);

        // 8. Edge-triggered spike: (v_new >= thresh) AND (v < thresh)
        let above_new = _mm256_cmp_ps(v_new, _mm256_set1_ps(v_thresh), _CMP_GE_OQ);
        let above_old = _mm256_cmp_ps(v, _mm256_set1_ps(v_thresh), _CMP_GE_OQ);
        // NOT(above_old) via ANDNOT: ~above_old AND all_ones
        let all_ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1i32));
        let was_below = _mm256_andnot_ps(above_old, all_ones);
        let edge_spike = _mm256_and_ps(above_new, was_below);

        // 9. LUT spike: spike > 0.5
        let half = _mm256_set1_ps(0.5);
        let lut_spike = _mm256_cmp_ps(spike_f32, half, _CMP_GE_OQ);

        // 10. Combined spike: edge OR LUT (add propagates NaN, not max!)
        let combined_spike = _mm256_add_ps(edge_spike, lut_spike);

        (v_new, new_hist, combined_spike)
    }

    /// Single-step inference for 8 neurons using `wide::f32x8` (safe wrapper).
    ///
    /// # Safety
    /// Must be called on x86_64 with AVX2 support.
    #[inline(always)]
    pub unsafe fn triang_step_f32x8(
        v: wide::f32x8,
        history: wide::f32x8,
        i_syn: wide::f32x8,
        lut: &TriangLUT,
        v_thresh: f32,
        input_thresh: f32,
    ) -> (wide::f32x8, wide::f32x8, wide::f32x8) {
        // INLINE-CRITICAL: transmute(32B) + call + transmute(32B)
        // MUST be #[inline(always)] — function call overhead (save/restore
        // 8 YMM registers) dominates the ~50-100 cycle gather.
        #[allow(clippy::missing_transmute_annotations)]
        {
            let v_mm: __m256 = core::mem::transmute(v);
            let h_mm: __m256 = core::mem::transmute(history);
            let i_mm: __m256 = core::mem::transmute(i_syn);
            let (v2, h2, s) = triang_step_avx2_raw(v_mm, h_mm, i_mm, lut, v_thresh, input_thresh);
            (
                core::mem::transmute(v2),
                core::mem::transmute(h2),
                core::mem::transmute(s),
            )
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_quantize_v() {
        let lut = TriangLUT::new(-100.0, 60.0);
        assert_eq!(lut.quantize_v(-100.0), 0);
        assert_eq!(lut.quantize_v(60.0), 255);
        assert!(lut.quantize_v(-65.0) > 0);
        assert!(lut.quantize_v(-65.0) < 255);
    }

    #[test]
    fn test_lut_key_12bit() {
        // 12-bit key: V_q << 4 | (history & 0x0F)
        let v_q: u16 = 128;
        let hist: u8 = 0x0A; // 4-bit valid
        let key = TriangLUT::key(v_q, hist);
        assert_eq!(key >> HISTORY_BITS, v_q as usize);
        assert_eq!(key & HISTORY_MASK as usize, 0x0A);
        // History should be masked to 4 bits
        let hist_wide: u8 = 0xFF;
        let key2 = TriangLUT::key(v_q, hist_wide);
        assert_eq!(key2 & HISTORY_MASK as usize, 0x0F);
        // Max key: V_q=255, hist=15 → 255*16+15 = 4095
        assert_eq!(TriangLUT::key(255, 15), 4095);
    }

    #[test]
    fn test_lut_nearest_valid() {
        let mut lut = TriangLUT::new(-100.0, 60.0);
        // Insert one known entry
        lut.entries[TriangLUT::key(128, 0)] = LutEntry {
            delta_v: 0.5,
            spike: 0.0,
        };
        let entry = nearest_valid(&lut, 128, 42);
        assert!(
            (entry.delta_v - 0.5).abs() < 1e-6,
            "Should find nearest: {}",
            entry.delta_v
        );
    }

    #[test]
    fn test_lut_fill_missing() {
        let mut lut = TriangLUT::new(-100.0, 60.0);
        // Fill some entries (4-bit history: 0..15)
        for v_q in 0u16..256 {
            for hist in 0u8..=15 {
                if hist % 3 == 0 {
                    let k = TriangLUT::key(v_q, hist);
                    lut.entries[k] = LutEntry {
                        delta_v: 0.1 * hist as f32,
                        spike: 0.0,
                    };
                }
            }
        }
        fill_missing(&mut lut);
        // Check all entries are filled
        for k in 0..LUT_SIZE {
            assert!(
                !lut.entries[k].count_is_zero(),
                "Entry {k} still zero after fill_missing"
            );
        }
    }
}
