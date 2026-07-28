# 📚 API Reference — taas-checkmate

> Rust API документация для `taas-checkmate` v0.1.0.

---

## Crate-level Re-exports

```rust
// top-level: taas_checkmate::
pub use io::{safety_clamp, LinearDecoder, PwmEncoder};
pub use kuramoto::{kuramoto_step_dense, kuramoto_step_sparse, t3_step_sparse, ...};
pub use neuron::{step_izhikevich, step_saw, step_psi_wave, step_lif, step_lif_at,
    step_adex, stdp_update, stdp_homeostatic, ...};
pub use sparse::{propagate_spikes, compute_hamiltonian, step_psi_wave_full, apply_stdp_ltd};
pub use t3::{TriOscillator, NlmsDecoder, KalmanDecoder, InverseKuramotoDecoder,
             StdpDecoder, PstDecoder, RidgeSolver, TrajectoryBuffer,
             Decoder, T3Backend, Avx2Backend,
             pst_features, pst_linear_features, pst_linear_dim,
             compute_pearson, normalize_features};
pub use neuron::{EPropLIF, eprop_update, surrogate_gradient, PoissonGenerator,
    ShortTermPlasticity, compute_lfp};
pub use neuron::{TtfsEncoder, population_rate_decode};
pub use neuron::{SornNetwork, intrinsic_plasticity, synaptic_normalization};
pub use taas_format::{TaasWriter, TaasReader, TAAS_MAGIC, TAAS_VERSION};
```

---

## 1. T3 — Temporal Triplet Engine

### 1.1 T3Backend trait

```rust
pub trait T3Backend: Send + Sync {
    fn prepare(&self, phases: &mut [f32], cos_out: &mut [f32], sin_out: &mut [f32]);
    fn update(&self, phases: &mut [f32], amplitudes: &mut [f32],
              omegas: &[f32], coupling: &[f32], drives: &[f32],
              dt: f32, alpha: f32, gain: f32);
    fn nlms_train(&self, weights: &mut [f32], bias: &mut [f32],
                  states: &[f32], error: f32, lr: f32, eps: f32);
    fn kalman_update(&self, weights: &mut [f32], cov: &mut [f32],
                     states: &[f32], error: f32, s: f32);
}
```

**Implementations:**
- `Avx2Backend` — ZST, вызывает AVX2 ASM через FFI
- `CudaBackend` — (planned) GPU backend

### 1.2 TriOscillator

```rust
pub struct TriOscillator<B: T3Backend = Avx2Backend> {
    pub n: usize,
    pub backend: B,
    pub th1: Vec<f32>, pub th2: Vec<f32>, pub th3: Vec<f32>,        // phases
    pub prev_th1: Vec<f32>, pub prev_th2: Vec<f32>, pub prev_th3: Vec<f32>, // prev
    pub r1: Vec<f32>, pub r2: Vec<f32>, pub r3: Vec<f32>,            // amplitudes
    pub prev_r1: Vec<f32>, pub prev_r2: Vec<f32>, pub prev_r3: Vec<f32>,
    pub omega1: Vec<f32>, pub omega2: Vec<f32>, pub omega3: Vec<f32>, // frequencies
    pub drive1: Vec<f32>, pub drive2: Vec<f32>, pub drive3: Vec<f32>, // drives
    pub sin1: Vec<f32>, pub cos1: Vec<f32>,
    pub sin2: Vec<f32>, pub cos2: Vec<f32>,
    pub sin3: Vec<f32>, pub cos3: Vec<f32>,
    pub coupling1: Vec<f32>, pub coupling2: Vec<f32>, pub coupling3: Vec<f32>,
    pub cross_w12: Vec<f32>, pub cross_w13: Vec<f32>, pub cross_w23: Vec<f32>,
    pub alpha: f32, pub drive_gain: f32, pub epsilon: f32,
    pub phase_gate_gain: f32,
    pub gamma1: f32, pub gamma2: f32, pub gamma3: f32,
    pub initialised: bool,
}
```

**Methods:**

| Method | Signature | Description |
|:-------|:----------|:------------|
| `new` | `(n, freq_min, freq_max, backend) -> Self` | Create 3 populations × N oscillators |
| `set_uniform_coupling` | `(&mut self, w12, w13, w23)` | Set uniform cross-coupling weights |
| `step` | `(&mut self, dt: f32)` | One time step (dynamics + coupling + update) |
| `order_parameter` | `(&self, phases) -> f32` | Compute R for one population |
| `order_parameter_cross` | `(&self, ph1, ph2) -> f32` | Cross-R between two populations |
| `phases` | `(&self) -> (&[f32], &[f32], &[f32])` | Get θ₁, θ₂, θ₃ |
| `amplitudes` | `(&self) -> (&[f32], &[f32], &[f32])` | Get r₁, r₂, r₃ |
| `omegas` | `(&self) -> (&[f32], &[f32], &[f32])` | Get ω₁, ω₂, ω₃ |

**Public fields (all accessible):**
- `n` — oscillators per population
- `backend` — compute backend (e.g. `Avx2Backend`)
- `th1..th3` — phases [N] each
- `prev_th1..prev_th3`, `prev_r1..prev_r3` — previous state for dθ/dt
- `r1..r3` — Stuart-Landau amplitudes [N]
- `omega1..omega3` — natural frequencies [N]
- `drive1..drive3` — external drive inputs [N]
- `sin1/sin2/sin3`, `cos1/cos2/cos3` — sin/cos buffers
- `coupling1..coupling3` — computed coupling [N]
- `cross_w12/w13/w23` — cross-coupling weights [N×N flat]
- `alpha, drive_gain, epsilon` — Stuart-Landau params
- `phase_gate_gain, gamma1/2/3` — Phase Gate params
- `initialised` — flag after first step

### 1.3 Phase Gate

```rust
pub fn phase_gate(r: f32, gamma: f32, n: usize, gain: f32) -> f32;
pub fn gate_coupling(w: f32, r_source: f32, gamma: f32, n_source: usize, gain: f32) -> f32;
```

### 1.4 Cochlea

```rust
pub struct Biquad { b0, b1, b2, a1, a2, x1, x2: f32 }
impl Biquad {
    pub fn new() -> Self;                                  // passthrough
    pub fn bandpass(fc: f32, q: f32, fs: f32) -> Self;    // BPF
    pub fn lowpass(fc: f32, q: f32, fs: f32) -> Self;     // LPF
    pub fn tick(&mut self, x: f32) -> f32;                 // process one sample
    pub fn reset(&mut self);
}

pub struct Cochlea { filters, envelopes, center_freqs, n_bands, sample_rate, smoothing, q }
impl Cochlea {
    pub fn new(n_bands, low_hz, high_hz, sample_rate) -> Self;
    pub fn tick(&mut self, sample: f32) -> &[f32];        // returns envelope slice
    pub fn reset(&mut self);
    pub fn detect_spikes(&self, threshold: f32) -> Vec<usize>;
    pub fn get_envelopes(&self) -> &[f32];
}
```

### 1.5 PST Features

```rust
/// O(N²) pairwise: returns (sin_matrix, cos_matrix) each n1×n2
pub fn pst_features(ph1: &[f32], ph2: &[f32]) -> (Vec<f32>, Vec<f32>);

/// O(N) linear: 3·N·6 + 12 features
pub fn pst_linear_features<B: T3Backend>(tri: &TriOscillator<B>) -> Vec<f32>;

/// Feature dimension for given N
pub fn pst_linear_dim(n: usize) -> usize;
```

### 1.6 TrajectoryBuffer

```rust
pub struct TrajectoryBuffer { buf, capacity, feat_dim, pos, count }
impl TrajectoryBuffer {
    pub fn new(capacity: usize, feat_dim: usize) -> Self;
    pub fn push(&mut self, features: &[f32]);
    pub fn trajectory(&self, L: usize) -> Vec<f32>;   // L-frame concatenated
    pub fn is_full(&self) -> bool;
    pub fn clear(&mut self);
    pub fn len(&self) -> usize;
    pub fn capacity(&self) -> usize;
}
```

### 1.7 Decoder trait + Implementations

```rust
pub trait Decoder {
    fn dim(&self) -> usize;
    fn predict(&self, states: &[f32]) -> f32;
    fn train(&mut self, states: &[f32], target: f32) -> f32;
    fn reset(&mut self);
}
```

| Struct | Constructor | Notes |
|:-------|:-----------|:------|
| `NlmsDecoder` | `new(n, lr, eps)` | Normalized LMS |
| `KalmanDecoder` | `new(n, q, r)` | Diagonal RLS/Kalman |
| `InverseKuramotoDecoder` | `new(n, alpha, drive_gain)` | Training-free, needs `decode()` |
| `StdpDecoder` | `new(n, lr, decay, sparsity)` | Anti-Hebbian STDP |
| `PstDecoder` | `new(feat_dim, L, alpha)` | Batch Ridge, needs `train_batch()` |

**InverseKuramotoDecoder extra:**
```rust
impl InverseKuramotoDecoder {
    pub fn decode(&mut self, amps: &[f32]) -> f32;
    pub fn decode_batch(&mut self, amps: &[f32]) -> Vec<f32>;
}
```

**PstDecoder extra:**
```rust
impl PstDecoder {
    pub fn train_batch(&mut self, X: &[f32], y: &[f32], n_samples: usize) -> f32;
    pub fn predict(&self, trajectory: &[f32]) -> f32;
}
```

### 1.8 Ridge Solver

```rust
pub struct RidgeSolver { alpha: f32 }
impl RidgeSolver {
    pub fn new(alpha: f32) -> Self;
    pub fn solve(&self, x: &[f32], y: &[f32], n: usize, d: usize) -> Vec<f32>;
}

pub fn solve_cg(a: &[f32], b: &[f32], n: usize, max_iter: usize, tol: f32) -> Vec<f32>;
pub fn compute_pearson(a: &[f32], b: &[f32]) -> f32;
pub fn normalize_features(features: &mut [f32], n_samples: usize, dim: usize);
```

---

## 2. Kuramoto

```rust
// Standard Kuramoto step (dense, N×N matrix)
pub fn kuramoto_step_dense(
    phases: &mut [f32], omegas: &[f32], coupling_matrix: &[f32],
    sin_buf: &mut [f32], cos_buf: &mut [f32], coupling_buf: &mut [f32],
    dt: f32,
);

// Sparse Kuramoto step (CSR format)
pub fn kuramoto_step_sparse(
    phases: &mut [f32], omegas: &[f32],
    row_ptr: &[i32], col_indices: &[i32], weights: &[f32],
    sin_buf: &mut [f32], cos_buf: &mut [f32], coupling_buf: &mut [f32],
    dt: f32,
);

// T3 sparse step (Stuart-Landau + Kuramoto)
pub fn t3_step_sparse(
    phases: &mut [f32], amplitudes: &mut [f32], omegas: &[f32],
    row_ptr: &[i32], col_indices: &[i32], weights: &[f32],
    drives: &[f32],
    sin_buf: &mut [f32], cos_buf: &mut [f32], coupling_buf: &mut [f32],
    dt: f32, decay: f32, alpha: f32, gain: f32,
);

// LMS / NLMS trainer
pub fn nlms_train(weights: &mut [f32], bias: &mut [f32], states: &[f32],
    error: f32, lr: f32, eps: f32);
pub fn t3_lms_train(weights: &mut [f32], bias: &mut [f32], states: &[f32],
    error: f32, lr: f32);

// Kalman update
pub fn kalman_update(weights: &mut [f32], cov: &mut [f32], states: &[f32],
    error: f32, s: f32);

// Adam optimizer
pub fn adam_train(weights: &mut [f32], m: &mut [f32], v: &mut [f32],
    states: &[f32], error: f32, lr: f32,
    beta1: f32, beta2: f32, bc1: f32, bc2: f32, eps: f32);

// Interference
pub fn interference(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]);
pub fn interference_mean(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]);

// Coherence
pub fn phase_coherence(output: &mut [f32], sin_buf: &[f32], cos_buf: &[f32],
    amplitudes: &[f32]);

// Band energy
pub fn band_energy(band_energies: &mut [f32], amplitudes: &[f32], omegas: &[f32],
    band_edges: &[f32]);

// Other
pub fn lazy_sincos_update(sin_out: &mut [f32], cos_out: &mut [f32],
    phases: &[f32], phases_prev: &mut [f32], eps: f32);
pub fn batch_coherence_sweep(output: &mut [f32], areas: &[f32], areas_sin: &[f32],
    areas_cos: &[f32], n_areas: usize, n_elem: usize);
pub fn winding_matrix(output: &mut [f32], phases_a: &[f32], phases_b: &[f32]);
pub fn pst_decode(output: &mut [f32], buffer: &[f32], weights: &[f32],
    feat_dim: usize, L: usize, pos: usize);
pub fn ring_synthesis(output: &mut [f32], amplitudes: &[f32], trig_buf: &[f32],
    weights: &[f32]);
```

---

## 3. Neuron Models

```rust
pub fn step_izhikevich(v: &mut [f32], u: &mut [f32], trace: &mut [f32],
    a: &[f32], b: &[f32], i_ext: &mut [f32], dt: f32,
    trace_decay: f32, spikes: &mut Vec<u32>, c: &[f32], d: &[f32]);

pub fn step_saw(v: &mut [f32], u: &mut [f32], i_ext: &mut [f32],
    dt: f32, spikes: &mut Vec<u32>);

pub fn step_psi_wave(v: &mut [f32], u: &mut [f32], psi: &mut [f32],
    i_ext: &mut [f32], dt: f32, lambda: f32);

pub fn step_lif(v: &mut [f32], i_ext: &mut [f32], dt: f32,
    v_rest: f32, v_thresh: f32, r_mem: f32, tau: f32, v_reset: f32,
    refrac_steps: &mut [u32], spikes: &mut Vec<u32>);

pub fn step_adex(v: &mut [f32], w: &mut [f32], i_ext: &mut [f32], dt: f32,
    g_l: f32, e_l: f32, v_t: f32, delta_t: f32,
    a: f32, tau_w: f32, b: f32, v_reset: f32, w_reset: f32, spikes: &mut Vec<u32>);

pub fn stdp_update(weights: &mut [f32], pre_trace: &[f32], post_trace: &[f32],
    lr: f32, tau_plus: f32, tau_minus: f32, a_plus: f32, a_minus: f32);

pub fn stdp_homeostatic(weights: &mut [f32], target_rate: f32, lr: f32);

pub struct DelayBuffer { ... }
impl DelayBuffer {
    pub fn new(capacity: usize) -> Self;
    pub fn push(&mut self, value: f32);
    pub fn get(&self, delay: usize) -> f32;
    pub fn reset(&mut self);
}
```

---

## 4. I/O

```rust
pub fn safety_clamp(x: &mut [f32], max_abs: f32);

pub struct LinearDecoder { ... }
impl LinearDecoder {
    pub fn new(n: usize) -> Self;
    pub fn predict(&self, states: &[f32]) -> f32;
    pub fn train(&mut self, states: &[f32], target: f32, lr: f32);
    pub fn reset(&mut self);
}

pub struct PwmEncoder { ... }
impl PwmEncoder {
    pub fn new(n: usize, scale: f32) -> Self;
    pub fn encode(&self, signal: &[f32]) -> Vec<f32>;
    pub fn decode(&self, pwm: &[u8]) -> Vec<f32>;
}
```

---

## 5. Sparse (CSR)

```rust
pub fn propagate_spikes(
    spikes: &[u32], col_indices: &[u32], row_ptrs: &[u32],
    weights: &[f32], target: &mut [f32], n_pre: usize, n_post: usize,
);

pub fn compute_hamiltonian(
    theta: &[f32], amplitudes: &[f32], coupling: &[f32],
) -> f32;

pub fn step_psi_wave_full(
    real: &mut [f32], imag: &mut [f32],
    drive: &[f32], dt: f32, omega: f32, gamma: f32,
);

pub fn apply_stdp_ltd(
    weights: &mut [f32], pre_spikes: &[u32], post_spikes: &[u32],
    lr: f32, tau: f32,
);
```

---

## 6. Constants

```rust
// t3 module
pub const PI: f32 = std::f32::consts::PI;        // 3.1415927
pub const TWO_PI: f32 = 2.0 * std::f32::consts::PI; // 6.2831855
```

---

## Usage Examples

### Basic T3 + Decoder

```rust
use taas_checkmate::{TriOscillator, Avx2Backend, Decoder, NlmsDecoder, pst_linear_features};

let mut tri = TriOscillator::new(64, 0.5, 1.5, Avx2Backend);
tri.set_uniform_coupling(0.1, 0.05, 0.02);

// Audio processing loop
let mut dec = NlmsDecoder::new(pst_linear_dim(64), 0.01, 1e-8);

for sample in audio_samples.chunks(64) {
    tri.drive1.copy_from_slice(sample);
    tri.step(0.001);
    
    let fv = pst_linear_features(&tri);
    let output = dec.predict(&fv);
    dec.train(&fv, target);
}
```

### Cochlea + TriOscillator

```rust
use taas_checkmate::t3::{Cochlea, TriOscillator, Avx2Backend};

let mut cochlea = Cochlea::new(16, 80.0, 8000.0, 16000.0);
let mut tri = TriOscillator::new(16, 0.5, 1.5, Avx2Backend);

for &sample in audio_buffer.iter() {
    let env = cochlea.tick(sample);
    tri.drive1.copy_from_slice(env);
    tri.step(0.001);
}
```

### Ridge Regression + Trajectory

```rust
use taas_checkmate::{RidgeSolver, TrajectoryBuffer, compute_pearson};

let mut buf = TrajectoryBuffer::new(16, 10);  // 16 frames × 10 features
for frame in data.chunks(10) {
    buf.push(frame);
}

let mut solver = RidgeSolver::new(0.01);
let w = solver.solve(&X, &y, n_samples, feat_dim);
```
