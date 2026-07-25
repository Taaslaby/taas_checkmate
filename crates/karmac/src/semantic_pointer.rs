//! Semantic Pointers — Vector Symbolic Architecture (SPA) operations.
//!
//! Implements the core SPA operations from Chris Eliasmith's Neural Engineering
//! Framework (NEF), which underpin SPAUN's cognitive representations.
//!
//! # Key Concepts
//!
//! Semantic Pointers are high-dimensional vectors that encode structured information.
//! They support three core operations:
//!
//! 1. **Bind** (⊗): Associate two concepts (e.g., bind(Color, Red))
//!    - Uses circular convolution in frequency domain
//! 2. **Unbind** (⊗⁻¹): Retrieve a bound component
//!    - Uses circular correlation
//! 3. **Superpose** (+): Merge multiple concepts into one vector
//!    - Simple addition (with normalization)
//!
//! # Usage
//! ```rust
//! use karmac::semantic_pointer::SemanticPointer;
//!
//! let sp = SemanticPointer::new(512);
//! let color = sp.random_vector();
//! let red = sp.random_vector();
//! let bound = sp.bind(&color, &red); // Color ⊗ Red
//! let retrieved = sp.unbind(&bound, &color); // ≈ Red
//! ```
//!
//! # Math
//!
//! Circular convolution (frequency domain):
//! ```text
//! C = IFFT(FFT(a) ⊙ FFT(b))
//! ```
//!
//! Circular correlation (unbinding):
//! ```text
//! a ⊗⁻¹ c = IFFT(FFT(c) ⊙ conj(FFT(a)))
//! ```

use rand::Rng;

/// Semantic Pointer engine with FFT-based bind/unbind.
pub struct SemanticPointer {
    /// Dimensionality (must be power of 2 for FFT).
    pub dim: usize,
    /// Precomputed twiddle factors for FFT.
    twiddles: Vec<(f32, f32)>,
    /// Bit-reversal permutation table.
    bit_rev: Vec<usize>,
}

impl SemanticPointer {
    /// Create a new Semantic Pointer engine.
    /// `dim` should be a power of 2 (e.g., 256, 512, 1024).
    pub fn new(dim: usize) -> Self {
        assert!(dim.is_power_of_two(), "dim must be power of 2, got {}", dim);
        let twiddles = Self::compute_twiddles(dim);
        let bit_rev = Self::compute_bit_rev(dim);
        Self {
            dim,
            twiddles,
            bit_rev,
        }
    }

    /// Generate a random unit vector (semantic pointer).
    pub fn random_vector(&self) -> Vec<f32> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(rand::thread_rng().gen());
        let v: Vec<f32> = (0..self.dim)
            .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
            .collect();
        self.normalize(&v);
        v
    }

    /// Create a semantic pointer from integer index (one-hot-like).
    pub fn from_index(&self, index: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        v[index % self.dim] = 1.0;
        v
    }

    /// Bind two vectors using circular convolution.
    /// `result = a ⊗ b` (bind)
    pub fn bind(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), self.dim);
        assert_eq!(b.len(), self.dim);

        // FFT both
        let fa = self.fft(a);
        let fb = self.fft(b);

        // Element-wise complex multiply
        let fc: Vec<(f32, f32)> = fa
            .iter()
            .zip(fb.iter())
            .map(|(&(ar, ai), &(br, bi))| (ar * br - ai * bi, ar * bi + ai * br))
            .collect();

        // IFFT
        let result = self.ifft(&fc);
        self.normalize(&result)
    }

    /// Unbind: retrieve a component from a bound vector.
    /// `result = c ⊗⁻¹ a` (unbind a from c, where c = a ⊗ b → result ≈ b)
    pub fn unbind(&self, c: &[f32], a: &[f32]) -> Vec<f32> {
        assert_eq!(c.len(), self.dim);
        assert_eq!(a.len(), self.dim);

        let fc = self.fft(c);
        let fa = self.fft(a);

        // Circular correlation: F(c) ⊙ conj(F(a))
        let fr: Vec<(f32, f32)> = fc
            .iter()
            .zip(fa.iter())
            .map(|(&(cr, ci), &(ar, ai))| {
                // (cr + i·ci) · (ar - i·ai) = (cr·ar + ci·ai) + i·(ci·ar - cr·ai)
                (cr * ar + ci * ai, ci * ar - cr * ai)
            })
            .collect();

        let result = self.ifft(&fr);
        self.normalize(&result)
    }

    /// Superpose (bundle) multiple vectors: sum + normalize.
    pub fn superpose(&self, vectors: &[&[f32]]) -> Vec<f32> {
        assert!(!vectors.is_empty());
        let dim = vectors[0].len();
        let mut result = vec![0.0f32; dim];
        for v in vectors {
            assert_eq!(v.len(), dim);
            for i in 0..dim {
                result[i] += v[i];
            }
        }
        self.normalize(&result)
    }

    /// Similarity between two vectors (cosine similarity).
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Normalize vector to unit length.
    pub fn normalize(&self, v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / norm).collect()
        }
    }

    // ─── FFT Implementation (radix-2 DIT) ────────────────────────────────────

    fn compute_twiddles(n: usize) -> Vec<(f32, f32)> {
        let mut twiddles = Vec::with_capacity(n / 2);
        for k in 0..n / 2 {
            let angle = -2.0 * std::f32::consts::PI * k as f32 / n as f32;
            twiddles.push((angle.cos(), angle.sin()));
        }
        twiddles
    }

    fn compute_bit_rev(n: usize) -> Vec<usize> {
        let bits = (n as f32).log2() as u32;
        (0..n)
            .map(|i| {
                let mut r = 0u32;
                let mut x = i as u32;
                for _ in 0..bits {
                    r = (r << 1) | (x & 1);
                    x >>= 1;
                }
                r as usize
            })
            .collect()
    }

    fn fft(&self, input: &[f32]) -> Vec<(f32, f32)> {
        let n = self.dim;
        assert_eq!(input.len(), n);

        // Bit-reverse reorder
        let mut buf: Vec<(f32, f32)> = self.bit_rev.iter().map(|&i| (input[i], 0.0)).collect();

        // Cooley-Tukey butterfly
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let tw = self.twiddles[k * n / len];
                    let even_idx = start + k;
                    let odd_idx = start + k + half;
                    let (er, ei) = buf[even_idx];
                    let (or, oi) = buf[odd_idx];
                    //蝴蝶: even + twiddle * odd
                    let tr = or * tw.0 - oi * tw.1;
                    let ti = or * tw.1 + oi * tw.0;
                    buf[even_idx] = (er + tr, ei + ti);
                    buf[odd_idx] = (er - tr, ei - ti);
                }
            }
            len <<= 1;
        }

        buf
    }

    fn ifft(&self, freq: &[(f32, f32)]) -> Vec<f32> {
        let n = self.dim;
        assert_eq!(freq.len(), n);

        // Conjugate → FFT → conjugate → scale
        let conj: Vec<(f32, f32)> = freq.iter().map(|&(r, i)| (r, -i)).collect();
        let mut time = vec![(0.0f32, 0.0f32); n];

        // Bit-reverse
        for (i, &bi) in self.bit_rev.iter().enumerate() {
            time[i] = conj[bi];
        }

        // Cooley-Tukey butterfly
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let tw = self.twiddles[k * n / len];
                    let even_idx = start + k;
                    let odd_idx = start + k + half;
                    let (er, ei) = time[even_idx];
                    let (or, oi) = time[odd_idx];
                    let tr = or * tw.0 - oi * tw.1;
                    let ti = or * tw.1 + oi * tw.0;
                    time[even_idx] = (er + tr, ei + ti);
                    time[odd_idx] = (er - tr, ei - ti);
                }
            }
            len <<= 1;
        }

        // Scale by 1/n
        let inv_n = 1.0 / n as f32;
        time.iter().map(|(r, _)| r * inv_n).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn sp_bind_unbind_recovers_original() {
        let sp = SemanticPointer::new(256);
        let a = sp.random_vector();
        let b = sp.random_vector();

        let bound = sp.bind(&a, &b);
        let recovered = sp.unbind(&bound, &a);

        // recovered ≈ b (cosine similarity should be high)
        let sim = sp.similarity(&recovered, &b);
        assert!(sim > 0.3, "unbind should recover b, similarity={}", sim);
    }

    #[test]
    fn sp_superpose_and_distinguish() {
        let sp = SemanticPointer::new(256);
        let a = sp.random_vector();
        let b = sp.random_vector();
        let c = sp.random_vector();

        let bundle = sp.superpose(&[&a, &b, &c]);

        // Each component should be recoverable by similarity
        let sim_a = sp.similarity(&bundle, &a);
        let sim_b = sp.similarity(&bundle, &b);
        let sim_bogus = sp.similarity(&bundle, &sp.random_vector());

        assert!(sim_a > sim_bogus, "a should be in bundle");
        assert!(sim_b > sim_bogus, "b should be in bundle");
    }

    #[test]
    fn sp_bind_deterministic() {
        let sp = SemanticPointer::new(128);
        let a = sp.random_vector();
        let b = sp.random_vector();
        let c1 = sp.bind(&a, &b);
        let c2 = sp.bind(&a, &b);
        assert_eq!(c1, c2);
    }

    #[test]
    fn sp_from_index() {
        let sp = SemanticPointer::new(8);
        let v = sp.from_index(3);
        assert_eq!(v[3], 1.0);
        assert!(v.iter().enumerate().all(|(i, &x)| i == 3 || x == 0.0));
    }

    #[test]
    fn sp_similarity_range() {
        let sp = SemanticPointer::new(256);
        let a = sp.random_vector();
        let sim_self = sp.similarity(&a, &a);
        assert!(
            close(sim_self, 1.0, 1e-5),
            "self-similarity should be 1.0, got {}",
            sim_self
        );
    }

    #[test]
    fn sp_random_vectors_orthogonal() {
        let sp = SemanticPointer::new(512);
        let a = sp.random_vector();
        let b = sp.random_vector();
        let sim = sp.similarity(&a, &b);
        // Random vectors in high-dim should be nearly orthogonal
        assert!(
            sim.abs() < 0.2,
            "random vectors should be near-orthogonal, sim={}",
            sim
        );
    }

    #[test]
    fn sp_bind_commutes() {
        let sp = SemanticPointer::new(256);
        let a = sp.random_vector();
        let b = sp.random_vector();
        let ab = sp.bind(&a, &b);
        let ba = sp.bind(&b, &a);
        let sim = sp.similarity(&ab, &ba);
        assert!(sim > 0.9, "bind should be commutative, sim={}", sim);
    }

    #[test]
    fn sp_normalize_unit() {
        let sp = SemanticPointer::new(128);
        let v = sp.normalize(&vec![3.0; 128]);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(close(norm, 1.0, 1e-5));
    }
}
