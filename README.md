# ═══════════════════════════════════════════════════════════
# TaasLab yakutia \\ ТаасЛаб якутия \\
# ═════════════════════════════════════

# 🧠 taas-checkmate — Pure Rust SIMD Neural Engine

[![Rust](https://img.shields.io/badge/rust-1.78%2B-blue)]()
[![SIMD](https://img.shields.io/badge/SIMD-f32x8-orange)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![CI](https://github.com/USER/taas-checkmate/actions/workflows/ci.yml/badge.svg)]()

> **Pure Rust wavelet neural engine** 

## 🏗 Architecture

```
Python (PyO3)  ⇄  Rust safe wrappers  ⇄  pure Rust SIMD kernels (wide::f32x8)
                                                       ↕
                                             AVX2 / NEON / WASM SIMD
```


## 📦 Module Structure (60+ source files)

```
taas_checkmate/src/
├── lib.rs                ← re-exports + missing_docs lint
│
├── core/                 ← core AI types   [feature = "autograd"]
│   ├── tensor.rs         ← Tensor (nd-array, matmul, broadcasting)
│   └── autograd.rs       ← Var + backward (w/ computational graph)
│
├── models/               ← neural models
│   ├── neuron/           ← 12 files (см. ниже)
│   ├── nn.rs             ← Dense, Softmax, Dropout, BatchNorm   [autograd]
│   └── kuramoto.rs       ← Kuramoto, interference, coherence
│
├── ops/                  ← operations
│   ├── sparse.rs         ← CSR matrix, spike propagation, Hamiltonian
│   └── koit.rs           ← XOR attention, hardware RNG
│
├── training/             ← training utilities
│   ├── optim.rs          ← SGD, Adam optimizer   [autograd]
│   ├── loss.rs           ← CrossEntropy, MSE     [autograd]
│   └── rl.rs             ← Actor-Critic agent w/ dopamine-STDP
│
├── kernels/              ← SIMD kernels (all models)
│   ├── signal.rs         ← linear_decoder, load8/store8
│   ├── neuron.rs         ← izhikevich, lif, adex, saw, eprop, stdp, stp
│   ├── decode.rs         ← decoders (kalman, nlms, adam)
│   ├── attention.rs      ← sparse attention
│   ├── phase.rs          ← phase wrapping
│   ├── rng.rs            ← fast RNG
│   └── measure.rs        ← performance counters
│
├── t3/                   ← T3 oscillator engine
│   ├── oscillator.rs     ← TriOscillator<B: T3Backend>
│   ├── backend.rs        ← Avx2Backend, ScalarBackend
│   ├── backend_cuda.rs   ← CudaBackend   [cuda]
│   ├── phase_gate.rs     ← sigmoid gate
│   ├── pst.rs            ← PST features
│   ├── cochlea.rs        ← biquad filter bank
│   ├── trajectory.rs     ← ring buffer
│   └── decoders.rs       ← Nlms, Kalman, Stdp, Pst, Hybrid
│
├── bindings/python/      ← PyO3 bindings   [feature = "python"]
│   ├── neuron.rs         ← step_izhikevich, saw, lif, adex, stdp
│   ├── kuramoto.rs       ← kuramoto, t3, kalman, nlms, coherence
│   ├── io.rs             ← gates, encoders, decoders
│   ├── sparse.rs         ← CSR ops + sim_step
│   ├── koit.rs           ← XOR attention, rdrand
│   ├── t3.rs             ← PyTriOsc, PyNlmsDec, PyKalmanDec + PST
│   ├── snn.rs            ← PyLifNeuron, SORN, TTFS + plasticity
│   ├── eprop.rs          ← PyEPropLIF, surrogate, eprop_update
│   ├── poisson_stp.rs    ← Poisson, STP, LFP
│   ├── rl_agent.rs       ← ActorCriticAgent
│   └── taas_format.rs    ← save/load .taas
│
├── io.rs                 ← PWM encoder, gates, safety clamp
└── taas_format.rs        ← binary .taas snapshot format
```

### 🧬 neuron/ — 12 model files

```
models/neuron/
├── mod.rs           ← re-exports + constants (I_CLAMP, DEFAULT_TAU_REF)
├── izhikevich.rs    ← Izhikevich, EPropIzhikevich, EPropIzhikevichMulti
├── lif.rs           ← LIF, LIF-AT, EPropLIF
├── adex.rs          ← AdEx, EPropAdExMulti
├── saw.rs           ← SAW, ψ-wave, EPropSaw
├── stdp.rs          ← STDP, homeostatic STDP, DopamineSTDP
├── stp.rs           ← Short-Term Plasticity (Tsodyks-Markram), LFP
├── sorn.rs          ← SORN, intrinsic plasticity, synaptic normalization
├── encoding.rs      ← BSA, Phase, TTFS spike encoding
├── poisson.rs       ← PoissonGenerator
├── delay.rs         ← DelayBuffer (circular)
├── stats.rs         ← SpikeRaster, firing rates, ISI, CV
└── eprop.rs         ← surrogate_gradient, eprop_update (free fn)
```

## 🔧 Features

| Feature    | Description | Dependencies |
|:-----------|:------------|:-------------|
| `autograd` | Tensor + Var + backward + nn + optim + loss | `rand` |
| `python`   | PyO3 + numpy zero-copy bindings | `pyo3`, `numpy` |
| `cuda`     | CUDA backend for T3 engine | `cuda-kernels` (local) |

```bash
# Minimal: just SIMD kernels + neuron models
cargo build --no-default-features

# With autograd (Tensor + backprop)
cargo build --features autograd

# Full stack
cargo build --features autograd,python,cuda
```

## 🤖 Spiking Actor-Critic RL Agent

Pure Rust spiking Actor-Critic with dopamine-modulated STDP (3-factor plasticity).  
Based on Izhikevich (2007) *"Solving the distal reward problem through linkage of STDP and dopamine signaling"*.

```rust
use taas_checkmate::rl::{ActorCriticAgent, test_mdp_episode};

let mut agent = ActorCriticAgent::new(
    4,      // n_input
    16,     // n_hidden (padded to ×8)
    2,      // n_actions
    30,     // n_steps internal simulation
    1.0,    // dt (ms)
    42,     // seed
);

// Forward pass (Poisson → Izhikevich → LIF → softmax)
let (action, value) = agent.act(&[0.5, -0.3, 0.0, 0.8]);

// Learn from reward
let td_error = agent.learn(1.0, 0.95, false);
```

### SIMD Optimization Journey

| Step | Optimization | act() | Speedup |
|:-----|:------------|:-----:|:-------:|
| 0 | Baseline | 417 µs | 1.0× |
| 1 | Fused weight→Izhikevich SIMD kernel | 289 µs | 1.44× |
| 2 | Poisson RNG fused into SIMD pass | 260 µs | 1.60× |
| 3 | Fast-exp in softmax | 254 µs | **1.64×** |

### Fused SIMD Kernels

| Kernel | Purpose |
|:-------|:--------|
| `poisson_fused_hidden_step` | LCG RNG → weights → Izhikevich in 1 pass |
| `fused_output_lif_step` | hidden spikes → weights → LIF in 1 pass |
| `dopamine_weight_update` | Δw = η·δ·e with SIMD clamp |

## 🔬 Spiking Neuron Models

All neuron models are split by family — each file contains the step function + E-Prop variants:

| Model | File | Type | SIMD Kernel |
|:------|:-----|:----:|:-----------:|
| Izhikevich | `izhikevich.rs` | Regular spiking, bursting, chattering | `izhikevich_step` |
| SAW | `saw.rs` | Spike-Adaptive-Width | `saw_step` |
| ψ-wave | `saw.rs` | Schrödinger-like wave neuron | `psi_wave_step` |
| LIF | `lif.rs` | Leaky Integrate-and-Fire | `lif_step` |
| LIF-AT | `lif.rs` | LIF + adaptive threshold | `lif_at_step` |
| AdEx | `adex.rs` | Adaptive Exponential | `adex_step` |

### Plasticity

| Rule | Description |
|:-----|:------------|
| STDP | Spike-timing dependent plasticity (stdp_update, stdp_homeostatic) |
| Dopamine-STDP | 3-factor: STDP × dopamine reward signal |
| STP | Tsodyks-Markram short-term (facilitation + depression) |
| E-Prop | LIF, Izhikevich, SAW, AdEx with eligibility traces |

## ⚡ SIMD Kernels (20+)

All kernels use `wide::f32x8` for 8-wide portable SIMD:

| Kernel | SIMD ops |
|:-------|:---------|
| `izhikevich_step` | dv/dt, du/dt, spike detect, trace, refractory |
| `lif_step` | RC dynamics, spike detect |
| `lif_at_step` | RC + adaptive θ, β update |
| `adex_step` | fast_exp approximation, dv/dt, dw/dt |
| `stdp_update` | weight update + clamp |
| `kalman_update` | K-gain + covariance update |
| `nlms_train` | norm² + weight update |
| `dopamine_traces_step` | pre/post traces + eligibility |

## 🚀 T3 Super-module (PhaseFlow)

The **T3 (Temporal Triplet)** engine — 3-population Kuramoto oscillator system with Stuart-Landau amplitudes.

```rust
use taas_checkmate::{TriOscillator, Avx2Backend, Decoder, NlmsDecoder, pst_linear_features};

let mut tri = TriOscillator::new(128, 0.5, 1.5, Avx2Backend);
tri.set_uniform_coupling(0.1, 0.05, 0.02);
tri.step(0.001);

let fv = pst_linear_features(&tri);
let mut dec = NlmsDecoder::new(fv.len(), 0.01, 1e-8);
let output = dec.predict(&fv);
```

## 🐍 Python Bridge

```bash
cargo build --features python
```

Creates `libtaas_checkmate.so` — zero-copy numpy bindings for all functions.

```python
import taas_checkmate as tc
tc.step_izhikevich(v, u, trace, a, b, I, dt, 0.5, spikes, c, d)
# or use stateful classes:
snn = tc.PyLifNeuron(1024)
snn.step(I, spikes)
```

## 🐍 Python API (Русский)

Подробная документация Python API на русском с примерами для каждого под-модуля:
👉 **[PYTHON_API_RU.md](PYTHON_API_RU.md)**

```python
from taas_checkmate.neurons import step_psi_wave
from taas_checkmate.t3 import PyTriOscillator

agent.v       # свойство вместо copy_v(buf)
agent.weights # свойство вместо copy_weights(out)
```

## 📊 Performance

| Pipeline | Speed |
|:---------|:-----:|
| 3 × 64 oscillators + PST + NLMS | **388× realtime** @ 48kHz |
| 3 × 128 oscillators + Kalman | **240× realtime** @ 44.1kHz |
| Izhikevich 8K neurons | **1M neur/s** (×5 faster than NEST) |
| Sigmoid (Phase Gate) 8K | **8.2 ns/elem** (×3 faster than scipy) |

Benchmarks: `cargo bench --no-default-features` generates HTML reports in `target/criterion/`.

## 🔧 Quick Start

```bash
# Clone + build
cd taas_checkmate
cargo build --features autograd

# Run tests
cargo test --features autograd

# Benchmarks
cargo bench --no-default-features

# Documentation
cargo doc --no-deps --open
```

## 📖 Documentation

- [T3 Architecture](docs/T3_ARCHITECTURE.md)
- [Mathematical Audit](docs/MATHEMATICAL_AUDIT.md)
- [API Reference](docs/API_REFERENCE.md)
- `cargo doc --no-deps --open`

## 📜 License

GNU
