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

// connectome (единый крейт, волна 12): корневой alias = pillars_connectome
pub use pillars_connectome as connectome;
// pillars_connectome::ConnectomeData, Brain, SynapticBrain, sim_*, BrainRunner,
// SpikeRecorder, rcm::*, SpikeQueue, BrainState, LifParams, ...

// cuda (волна 13, feature "cuda"): единый pillars-cuda, alias в root/t3/train
pub use pillars_cuda::gpu as cuda;
// GpuContext, DevicePtr, csr_spmv_tanh, csr_stdp_dopamine_devptr, get_device_count,
// set_device, pst_jacobian_gpu, t3_fused_forward_bptt, interference_matrix, ...
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
- `CudaBackend` — GPU backend (feature `cuda`, через `pillars-cuda::gpu`)

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

## 7. Connectome — единый `pillars-connectome` (волна 12)

> Легаси `taas-connectome` слит в `pillars-connectome` (1 крейт) и удалён.
> Корневой `connectome::` = `pillars_connectome::` — публичный API легаси
> сохранён 1:1. Движок — 100% pillars (`Brain`/`SynapticBrain` строят
> `SparseGraph` внутри себя).

### 7.1 Данные: `ConnectomeData`

```rust
// i32-CSR + NT types + receptor types + группы — единый канон
let conn = pillars_connectome::ConnectomeData::new(
    row_ptr,      // Vec<i32>, [n_neurons + 1]
    col_indices,  // Vec<i32>, [n_synapses]
    values,       // Vec<f32>, [n_synapses]
    nt_types,     // Vec<f32>, per-neuron: +1 exc, -1 inh (или пусто → все exc)
    groups,       // Vec<(&str, Vec<usize>)> — именованные группы
);
conn.build_receptor_types();            // nt-знаки → AMPA(0)/GABA_A(2)
conn.build_receptor_types_flywire(&m);  // Drosophila-маппинг (+2=GLUT→GluCl(5) и т.д.)
conn.reorder_rcm();                     // Reverse Cuthill–McKee in-place
conn.save_baked(&mut w) / load_baked(&mut r); // бинарный .taas (magic "TAAS")
```

Плюс `BrainIO` + `InputChannel`/`OutputChannel` (Direct/Poisson encoding, Linear readout).

### 7.2 Движки: `Brain<M>` и `SynapticBrain<M>`

```rust
// Instant-current: каждый спайк добавляет вес ребра во входной ток цели
let mut brain = pillars_connectome::Brain::<pillars::zoo::LIFModel>::from_connectome(&conn);
// или с per-synapse axonal delays (python-фасад):
let g = pillars::api::SparseGraph::from_csr(row_ptr_u32, col_u32, &weights, &delays);
let mut brain = pillars_connectome::Brain::<M>::from_sparse_graph(g);

brain.configure(|p| { /* fill_param / fill_state */ });
brain.inject(&[0], &[50.0]);   // внешний ток
brain.step();
let spikes = brain.spikes();   // &[i32] mask 0/1

// Conductance: AMPA/NMDA/GABA kinetics (рецепторы из conn.receptor_types)
let mut sb = pillars_connectome::SynapticBrain::<M>::from_connectome(&conn, dt_ms);
sb.set_kinetics(&k);           // carry-over между вызовами
let s = sb.step(&external_i);  // Vec<i32> спайки за шаг
```

### 7.3 Высокоуровневый API

| Символ | Назначение |
|--------|------------|
| `sim_lif`, `sim_izhikevich`, `sim_izhikevich_dual`, `sim_izhikevich_synaptic`, `sim_lif_synaptic`, `sim_rate`, `sim_gco`, `sim_gco_record` | batch-симуляция на коннектоме (делегаты `crate::Brain`/`SynapticBrain`; `scale` свёрнут в weights) |
| `sim_hh_pillars`, `sim_hh_synaptic_pillars`, `hh_population_standard`, `hh_population_pacemaking` | HH на `NeuralPopulation<HHModel>` (эталон — `bench_hh_parity`) |
| `BrainRunner` + `BrainConfig` (lif/izh/hh/rate/gco), `PoissonGroup`, `LearningRule` | конфигурируемый цикл |
| `SpikeRecorder`, `SpikeRecordings` | DN/MN rate extraction |
| `csr_propagate_from_queue`, `csr_propagate_conductance_from_queue`, `SpikeQueue` | низкоуровневая CSR-пропагация |
| `rcm::{build_adjacency, reverse_cuthill_mckee, spectral_ordering, bandwidth, apply_permutation_to_csr, ...}` | переупорядочение коннектома (10–50× cache misses) |
| `sim::gpu::{sim_lif_gpu, sim_izhikevich_gpu}` | CUDA (feature `cuda`) |

### 7.4 Python

```python
from taas_checkmate.pillars import ConnectomeBrain, SynapticConnectomeBrain

b = ConnectomeBrain(row_ptr, col_idx, weights, delays)   # u32 CSR, delays u8
b.dt = 0.5; b.inject([0], [30.0]); b.step()
spikes = b.spikes          # numpy (n,) 0/1

sb = SynapticConnectomeBrain(row_ptr, col_idx, weights, delays, receptor_types, dt)
sb.step(i_in, spikes_out)  # conductance path
```

Легаси-классы (`ConnectomeData`, `BrainSimulator`, `BrainRunner`, `SpikeRecorder`) —
в `taas_checkmate.connectome` через корневой alias.

---

## 8. CUDA — единый `pillars-cuda` (волна 13)

> Легаси `cuda-kernels` (`crates/cuda`, 11.5K строк) слит в `pillars-cuda` и
> удалён. Workspace: 9 → **7 крейтов**. Корневой `cuda::` (и `taas_t3::cuda`,
> `taas_train::cuda`) = `pillars_cuda::gpu::` — все python-биндинги и
> `CudaBackend` работают без изменений.

### 8.1 Два GPU-слоя

1. **NVRTC (JIT)** — `#[dual_neuron]`-модели из `pillars::zoo::neurons`
   (17 моделей) транспилируются в CUDA C++ строку и компилируются в PTX
   через NVRTC на лету. `CompiledModel` + `*_CUDA_SOURCE`, parity-тесты
   `lif/izhikevich/hh/zoo_parity`.
2. **AOT-микроядра** — `kernels/*.cu` (по одному микроядру на файл:
   `gpu_context`, `csr_spmv_tanh`, `stdp_dopamine`, `t3_update`, `decoders`,
   `kuramoto`, `gco`, `fused`, `gates`, `hh`, `izhikevich`, …) компилируются
   nvcc в статическую либу. Модуль `gpu` — FFI-обвязка.

### 8.2 `gpu`-модуль (бывший `cuda_kernels`)

```rust
// Persistent GPU-контекст: все нейроны/синапсы остаются на устройстве
let mut ctx = pillars_cuda::gpu::GpuContext::new(n_neurons, n_synapses)?;
ctx.upload_connectome(&row_ptr, &col, &values, &nt_types, &external_i)?;
ctx.upload_lif_params(dt_ms, &tau_m, &v_rest, &v_thresh, &v_reset)?;
ctx.batch_lif_steps(&external_i, dev_spike_buf, &mut spikes, &mut v, n_steps)?;

// RAII device-память
let d_v = pillars_cuda::gpu::DevicePtr::<f32>::new(n)?;
d_v.copy_from_host(&data)?;

// GPU SPAUN-кернелы (device-указатели)
pillars_cuda::gpu::csr_spmv_tanh(out, in, rp, ci, values, input, scale, n, None)?;
pillars_cuda::gpu::csr_stdp_dopamine_devptr(d_w, d_rp, d_ci, d_tr, lr, delta, w_min, w_max, n)?;
```

- `GpuContext` — upload/download/batch LIF+Izh+HH, connectome CSR propagate,
  CUDA Graphs, T3/Kuramoto/GCO/Aberton, LIF-AT/STDP, HH-движок (прямой `__expf()`).
- Standalone-обёртки (`safety_clamp`, `stdp_update`, `nlms_train`, `kuramoto_prepare`,
  `echo_tap`, `t3_*`, `interference_matrix`, `band_energy`, `ring_synthesis`, …) —
  **host → device → host** round-trip (волна 13: исправлен баг с host-указателями).

### 8.3 Python

```python
from taas_checkmate import cuda           # feature "cuda"
from taas_checkmate.spaun_gpu import SpaunPipelineGPU   # GPU SPAUN
from taas_checkmate.hh_gpu import GpuHhLut
```

### 8.4 Валидация

- `pillars-cuda` tests: **38** (17 gpu-module + 21 NVRTC-parity) — зелёные на RTX 4060 Ti
- `neuron_reference::verify_vs_neuron_gpu` — HH GPU vs NEURON ground-truth ✓
- CI исключает `pillars-cuda` (нужен nvcc) как ранее `cuda-kernels`

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
