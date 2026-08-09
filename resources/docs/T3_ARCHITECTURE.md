# 🧬 T3 Architecture — Temporal Triplet Engine

> Полное описание архитектуры T3 (Temporal Triplet) — 3-population oscillator system.
> Версия: neuron-engine 0.1.0

---

## 1. Overview

T3 — это **супер-модуль** из 8 подмодулей для realtime обработки сигналов через Kuramoto-осцилляторы. Спроектирован как **generic-система** с pluggable backend (AVX2 / CUDA).

### 1.1 Философия

```
Физическая AI-модель вместо Transformer:

  Transformers:  billions params × GPU ← дорого, латентно, не приватно
  T3:            84 осциллятора × CPU  ← бесплатно, realtime, локально
```

Осцилляторы — естественная модель для **ритмических сигналов** (аудио, EEG, вибрации, биржевые данные). Фазовая динамика кодирует временные соотношения без обучения.

### 1.2 Структура модуля

```
src/t3/
├── mod.rs          — корень + реэкспорты
├── backend.rs      — T3Backend trait + Avx2Backend implementation
├── backend_cuda.rs — CudaBackend (feature `cuda`, через pillars-cuda::gpu)
├── oscillator.rs   — TriOscillator<B: T3Backend> — основная структура
├── phase_gate.rs   — Sigmoid Phase Gate
├── cochlea.rs      — Biquad filter bank + envelope follower
├── pst.rs          — Phase Space Topology features
├── trajectory.rs   — Ring buffer for L-frame trajectories
├── decoders.rs     — Decoder trait + 5 decoder implementations
└── ridge.rs        — Ridge regression solver + CG
```

### 1.3 Data Flow

```text
Audio / Signal
    │
    ▼
┌──────────────────┐
│    Cochlea        │  ← Biquad Bandpass × N
│  (Biquad IIR)     │  → envelope follower
│  + envelope       │  → spikes (optional)
└──────┬───────────┘
       │ envelopes
       ▼
┌──────────────────────────────────────┐
│      TriOscillator<B: T3Backend>     │
│                                      │
│  Population 1 (Fast)  ──────→ Pop 2 │
│       · ω = [f_min..f_max]   · ω ↑  │
│       · Phase Gate · r₁       · r₂  │
│       · Stuart-Landau amp           │
│                                      │
│  Population 3 (Slow)  ←────── Pop 2 │
│       · ω lowest layer              │
│       · Phase Gate · r₃             │
└──────┬─────────────────────────────┘
       │ phases (θ₁,θ₂,θ₃) + amps (r₁,r₂,r₃)
       ▼
┌──────────────────────────────────────┐
│        Feature Extraction            │
│                                      │
│  • PST features  (pairwise O(N²))   │
│  • PST linear    (O(N) combined)    │
│  • Cross order parameters           │
│  • Trajectory buffer (L-frame)      │
└──────┬─────────────────────────────┘
       │ feature vector
       ▼
┌──────────────────────────────────────┐
│           Decoder                    │
│                                      │
│  ┌──────────┐ ┌──────────┐          │
│  │  NLMS    │ │  Kalman  │          │
│  ├──────────┤ ├──────────┤          │
│  │ Inverse  │ │  STDP    │          │
│  │ Kuramoto │ │          │          │
│  └──────────┘ └──────────┘          │
│                                      │
│  output = W · state + bias          │
└──────┬─────────────────────────────┘
       │ decoded signal
       ▼
    Audio / Control / Prediction
```

---

## 2. Core Architecture

### 2.1 Backend Pattern (Trait-based)

Модуль использует **strategy pattern** через trait `T3Backend`:

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

**Avx2Backend** — ZST (zero-sized type). Все методы напрямую вызывают `extern "C"` ASM через FFI:

```text
T3Backend::prepare → kuramoto_prepare_avx  (sin/cos + wrap phases)
T3Backend::update  → t3_update_avx          (Stuart-Landau + phase step)
T3Backend::nlms_train  → nlms_train_avx     (Normalized LMS)
T3Backend::kalman_update → kalman_update_avx (Kalman filter)
```

**CudaBackend** — реализован в `backend_cuda.rs` (feature `cuda`, через
`pillars-cuda::gpu`): `TriOscillator<CudaBackend>` переносит фазы/амплитуды на GPU.

### 2.2 TriOscillator — 3-Population State

```rust
pub struct TriOscillator<B: T3Backend = Avx2Backend> {
    pub n: usize,             // oscillators per population
    pub backend: B,           // compute backend
    
    // Phases [N] × 3
    pub th1: Vec<f32>, pub th2: Vec<f32>, pub th3: Vec<f32>,
    // Previous phases (for dθ/dt in PST)
    pub prev_th1: Vec<f32>, pub prev_th2: Vec<f32>, pub prev_th3: Vec<f32>,
    
    // Amplitudes [N] × 3
    pub r1: Vec<f32>, pub r2: Vec<f32>, pub r3: Vec<f32>,
    // Natural frequencies [N] × 3
    pub omega1: Vec<f32>, pub omega2: Vec<f32>, pub omega3: Vec<f32>,
    // Drive inputs [N] × 3
    pub drive1: Vec<f32>, pub drive2: Vec<f32>, pub drive3: Vec<f32>,
    
    // Sin/cos buffers [N] × 3
    pub sin1: Vec<f32>, pub cos1: Vec<f32>, ...
    // Coupling [N] × 3
    pub coupling1: Vec<f32>, ...
    // Cross-weights [N×N] × 3
    pub cross_w12: Vec<f32>, pub cross_w13: Vec<f32>, pub cross_w23: Vec<f32>,
    
    // Stuart-Landau params
    pub alpha: f32,          // nonlinearity (≈1.0)
    pub drive_gain: f32,     // drive gain (≈10.0)
    pub epsilon: f32,        // amplitude floor (0.1)
    
    // Phase Gate params
    pub phase_gate_gain: f32, // sigmoid steepness (4.0)
    pub gamma1, gamma2, gamma3: f32,  // layer thresholds
}
```

### 2.3 Step Function

Каждый вызов `TriOscillator::step(dt)` выполняет 6 шагов:

```text
1. Save previous state     → prev_th[1..3] = th[1..3]
2. Compute sin/cos         → backend::prepare() (AVX2)
3. Order parameters        → R₁, R₂, R₃
4. Phase gates             → g₁ = σ(R₁, γ₁), g₂ = σ(R₂, γ₂), g₃ = σ(R₃, γ₃)
5. Cross-coupling          → coupling[i] = Σ w_ij · r_j · sin(θⱼ-θᵢ) · gate
6. Backend update          → backend::update() (AVX2: phases + amplitudes)
7. Wrap phases             → θ ∈ [-π, π]
8. Clamp amplitudes        → r ∈ [0, 10]
```

---

## 3. Component Details

### 3.1 Phase Gate

Сигмоидальный гейт из оригинального Nakajima NeuroBrain:

```math
G(r) = 1 / (1 + exp(-gain · (r - γ/√N)))
```

- **Назначение:** защищает медленные слои от шума быстрых
- При низкой когерентности (R < γ/√N) гейт закрыт (≈0)
- При высокой когерентности гейт открыт (≈1)

**Конфигурация по умолчанию:**
- `gain = 4.0`, `γ₁ = 0.5`, `γ₂ = 1.0`, `γ₃ = 1.12`

### 3.2 Cochlea (Auditory Filter Bank)

Аудио-препроцессор для преобразования звука в features:

- **Biquad:**
  - `bandpass(fc, q, fs)` — Audio EQ Cookbook BPF
  - `lowpass(fc, q, fs)` — Audio EQ Cookbook LPF
  - Transposed Direct Form II implementation
- **Cochlea:**
  - N логарифмически распределённых фильтров
  - Envelope follower (leaky integrator, smoothing=0.995)
  - Spike detection (threshold crossing)

### 3.3 PST Features

**Pairwise PST** (O(N²)):
```math
sin(θ₁ₚ - θ₂_q) = sinₚ·cos_q - cosₚ·sin_q
cos(θ₁ₚ - θ₂_q) = cosₚ·cos_q + sinₚ·sin_q
```

**Linear PST** (O(N)) — 6N + 12 компонент на популяцию:
1. sin(θ), cos(θ) — 2N
2. dθ/dt — N (фазовые скорости)
3. Амплитуды r — N
4. sin²(θ), cos²(θ) — 2N (квадратичные)
5. Cross order params R₁₂, R₂₃, R₁₃ — 3
6. Mean sin/cos/r — 3 × 3 = 9

Full dimension: `3N·6 + 12`

### 3.4 TrajectoryBuffer

Ring buffer для построения L-кадровых траекторий:

```rust
TrajectoryBuffer {
    buf: [capacity × feat_dim],  // flat ring buffer
    pos: usize,                   // current write position
    count: usize,                 // frames written
}
```

- `push(features)` — запись кадра
- `trajectory(L)` — конкатенация L последних кадров
- Используется в `PstDecoder` для ridge-регрессии

### 3.5 Decoders

5 реализаций `Decoder` trait:

| Decoder | Update Rule | Use Case |
|:--------|:-----------|:---------|
| **NLMS** | `w += lr·e·x/(‖x‖²+ε)` | General online learning |
| **Kalman** | `k = cov·x/s; w += k·e; cov -= k·cov` | Adaptive filtering |
| **InverseKuramoto** | `drive ≈ (ṙ - (α-r²)·r)/(gain·(r+ε))` | Zero-shot inference |
| **STDP** | `trace = decay·trace + \|ds/dt\|; Δw = lr·error·trace` | Biologically plausible |
| **PST** | `w = (XᵀX + αI)⁻¹Xᵀy` via CG | Batch ridge regression |

### 3.6 Ridge Solver

Conjugate Gradient для Ridge регрессии:

```math
min_w ‖Xw - y‖² + α·‖w‖²
→ (XᵀX + αI)w = Xᵀy
→ solve via CG (max 100 iterations, tol = 1e-6)
```

Дополнительно: `compute_pearson()` — correlation, `normalize_features()` — z-score.

---

## 4. Backend Comparison

| Feature | Avx2Backend (default) | CudaBackend (feature `cuda`) |
|:--------|:---------------------|:---------------------|
| Hardware | Any x86-64 with AVX2 | NVIDIA GPU |
| Vectorization | 8× f32 per instruction | 32–1024× threads |
| Memory | Host RAM | Device VRAM |
| Latency | < 1µs | 10–100µs (PCIe) |
| Power | 15–65W | 150–400W |

---

## 5. Integration with Python

Via PyO3 (`--features python`):

```rust
// python.rs — PyO3 bindings
#[pyclass]
struct T3Engine { inner: TriOscillator<Avx2Backend> }

#[pymethods]
impl T3Engine {
    fn step(&mut self, dt: f32) -> Vec<f32> { ... }
    fn get_phases(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) { ... }
}
```

---

## 6. Performance Characteristics

| Configuration | Oscillators | Features | Decoder | Speed (×RT) |
|:-------------|:-----------|:---------|:-------|:-----------:|
| Minimal | 3 × 8 | linear | NLMS | **2,500×** |
| Small | 3 × 64 | linear | Kalman | **388×** |
| Medium | 3 × 128 | linear | NLMS | **240×** |
| Large | 3 × 256 | pairwise | Ridge | **45×** |

*Measured on Intel i7-12700H, single core, 48kHz sample rate.*

---

## 7. Module Map

```text
                                    ┌─────────────┐
                                    │    mod.rs   │
                                    │ (re-exports)│
                                    └──────┬──────┘
            ┌────────────┬──────────────┬───┼───┬──────────────┬────────────┐
            ▼            ▼              ▼       ▼              ▼            ▼
      ┌─────────┐ ┌────────────┐ ┌────────┐ ┌────────┐ ┌──────────┐ ┌────────┐
      │ backend │ │ oscillator │ │ phase_ │ │cochlea │ │   pst    │ │traject-│
      │  .rs    │ │   .rs      │ │ gate.rs│ │  .rs   │ │   .rs    │ │ory.rs  │
      └─────────┘ └────────────┘ └────────┘ └────────┘ └──────────┘ └────────┘
                                                         │            │
                                                         ▼            ▼
                                                   ┌──────────┐ ┌──────────┐
                                                   │ decoders │ │  ridge   │
                                                   │   .rs    │ │   .rs    │
                                                   └──────────┘ └──────────┘
```

---

## 8. Testing Strategy

- **unit tests** — each sub-module has its own `#[cfg(test)]`
- **integration** — `t3::oscillator::tests::test_tri_synchronization`
- **backend** — `t3::backend::tests::test_avx2_backend_prepare`
- **decoder polymorphism** — `t3::decoders::tests::test_decoder_trait` (Box<dyn Decoder>)

```
34 tests in t3 module:
├── oscillator  — 5 tests (step, gate, sync, order param)
├── backend     — 3 tests (prepare, update, nlms)
├── phase_gate  — 4 tests (high R, low R, threshold, gate coupling)
├── cochlea     — 4 tests (biquad, tick, spikes, reset)
├── pst         — 3 tests (size, shape, dim)
├── trajectory  — 4 tests (push, trajectory, wrap, clear)
├── decoders    — 6 tests (nlms, kalman, inverse, stdp, pst, trait)
└── ridge       — 5 tests (ridge, pearson ×2, normalize, cg)
```
