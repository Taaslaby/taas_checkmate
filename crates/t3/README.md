# taas-t3 — Temporal Triplet Oscillator Engine

3-population Kuramoto oscillator with Stuart-Landau amplitudes, PST features, decoders, and spike encoders. Generic over compute backends (AVX2, scalar, and optional CUDA via `pillars-cuda`).

## Architecture

```
Drive → TriOscillator → PST Features → Decoder → Output
  ↑                                        │
  └────── Inverse Kuramoto (zero-shot) ────┘
```

## Core Components

### TriOscillator

Three coupled Kuramoto populations with Stuart-Landau amplitude dynamics. Cross-coupling is **O(N)** (not O(N²)) via uniform weight sums.

```rust
use taas_t3::{TriOscillator, Avx2Backend};

let mut tri = TriOscillator::new(64, 0.5, 1.5, Avx2Backend);
tri.set_uniform_coupling(0.3, 0.15, 0.08);

// Drive input (first 4 oscillators)
for i in 0..4 { tri.drive1[i] = 5.0; }

tri.step(0.01);  // one simulation step

// Read out
let r = tri.order_parameter(&tri.th1);
let (sin1, cos1, sin2, cos2, sin3, cos3) = (&tri.sin1, &tri.cos1, &tri.sin2, &tri.cos2, &tri.sin3, &tri.cos3);
```

### Backend Trait

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

- `Avx2Backend` (default) — SIMD through canonical `pillars` kernels
- `ScalarBackend` — portable, for debugging/verification
- `CudaBackend` — optional GPU backend from `pillars-cuda` (requires the `cuda` feature and CUDA/NVRTC runtime)

### Decoders

| Decoder | Type | Description |
|---|---|---|
| `NlmsDecoder` | Adaptive | Normalized LMS, fast convergence |
| `KalmanDecoder` | Adaptive | RLS/Kalman filter with state covariance |
| `InverseKuramotoDecoder` | Zero-shot | Drive recovery from amplitude derivatives, no training |
| `StdpDecoder` | Hebbian | Anti-Hebbian STDP with trace and sparsity |
| `PstDecoder` | Batch | Ridge regression from trajectory buffer |
| `HybridDecoder` | Combined | PST + Inverse Kuramoto + NLMS (zero-shot init → fine-tune) |

```rust
let mut dec = NlmsDecoder::new(64, 0.01, 1e-8);
let error = dec.train(&states, target);
let pred = dec.predict(&states);
```

### PST Features

Phase Space Topology features — 0 trig calls (uses pre-computed sin/cos from `TriOscillator`).

```rust
let fv = pst_linear_features(&tri);
// dim = 3N·6 + 12
let dim = pst_linear_dim(n);
```

### Spiker

Spike → hologram pipeline for connecting spiking networks to T3:

```rust
let positions = [[0.5, 0.5], [-0.5, 0.5]];
let mut spiker = Spiker::new(&positions, 50.0, 1.0);
spiker.feed(&[1.0, 0.0, 0.0, 0.0]);
let h = spiker.hologram();  // 3-channel FM drive
```

Variants: `NormalizedSpiker` (online normalization), `HybridSpiker` (adaptive gain).

### Cochlea

Auditory filter bank for audio → envelope → spike conversion:

```rust
let mut cochlea = Cochlea::new(16, 100.0, 8000.0, 44100.0);
let envelope = cochlea.tick(audio_sample);
```

## PST Control

Forward + inverse control loop: train Ridge from PST features, then find drive for target class.

```rust
let config = PstControlConfig::default();
let mut ctrl = PstControl::new(config);
ctrl.fit_ridge(&x, &y, n_samples, alpha);
let (drive, pred_class) = ctrl.drive_for(&mut tri, target_class, "gradient");
```

## Dependencies

- `pillars` (canonical SIMD kernels and backend primitives)
- `wide`, `serde`, `serde_json`, `rand`
- Optional: `pillars-cuda` through the root `cuda` feature (CUDA/NVRTC required for runtime execution)
