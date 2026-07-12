# ═══════════════════════════════════════════════════════════
# TaasLab yakutia \\ ТаасЛаб якутия \\
# ═══════════════════════════════

# 🧠 taas-checkmate — Pure Rust SIMD Neural Engine

[![Rust](https://img.shields.io/badge/rust-1.78%2B-blue)]()
[![SIMD](https://img.shields.io/badge/SIMD-f32x8-orange)]()
[![License](https://img.shields.io/badge/license-GPL--3.0-blue)]()
[![CI](https://github.com/anomalyco/taas-checkmate/actions/workflows/ci.yml/badge.svg)]()

> **Pure Rust spiking neural engine** — SIMD kernels, Kuramoto oscillators, FlyWire connectome, cognitive architecture

## 🏗 Architecture

```
Python (PyO3)  ⇄  Rust safe wrappers  ⇄  pure Rust SIMD kernels (wide::f32x8)
                                                       ↕
                                             AVX2 / NEON / WASM SIMD / CUDA
```

## 📦 Workspace (12 crates, 90+ source files)

```
taas_checkmate/
├── Cargo.toml              ← workspace root (lib = cdylib + lib)
├── src/
│   ├── lib.rs              ← re-exports all crates
│   ├── io.rs               ← PWM encoder, gates, safety clamp
│   └── python/             ← PyO3 bindings   [feature = "python"]
│
├── crates/
│   ├── kernels/            ← SIMD kernels (raw, unsafe)           19 files
│   │   └── src/neuron/     ← izhikevich, lif, adex, saw, hh, stp, dopamine, fused
│   │
│   ├── core/               ← Safe wrappers + neuron models        20 files
│   │   └── src/neuron/     ← 16 files: models + encoding + stats + eprop + autotune
│   │
│   ├── t3/                 ← T3 oscillator engine (Kuramoto + Stuart-Landau)  12 files
│   │   └── src/            ← oscillator, backends, decoders, pst, cochlea, spiker
│   │
│   ├── train/              ← RL + BPTT + SPIGRAD                  11 files
│   │   └── src/spike/      ← forward, backward, pst_jacobian, trajectory
│   │
│   ├── connectome/         ← FlyWire connectome simulator          5 files
│   │
│   ├── karmac/             ← Cognitive architecture (GNW, BasalGanglia, LOD)  6 files
│   │
│   ├── connectome-gpt/     ← Graph VAE + decoder for connectome    7 files
│   │
│   ├── cpe/                ← Coherence, Dopamine, STDP, Hybrid     6 files
│   │
│   ├── taas-format/        ← Binary .taas snapshot format          6 files
│   │
│   ├── cuda-kernels/       ← CUDA backend                         1 file
│   │
│   ├── unity/              ← C ABI for Unity                       1 file
│   │
│   └── gpu-eval/           ← GPU evaluation                        1 file
│
├── tests/                  ← Integration tests (Rust + Python)
├── benches/                ← Criterion benchmarks
├── examples/rust/          ← 7 examples (MNIST, connectome, hologram)
├── examples/python/        ← 9 Python examples (ψ-resonance, VNC, FlyWire)
└── docs/                   ← 10 documentation files
```

## 🔧 Features

| Feature | Description | Dependencies |
|:--------|:------------|:-------------|
| `t3` | T3 oscillator engine (default) | `taas-t3` |
| `train` | BPTT, SPIGRAD, connectome chase | `taas-train/t3` |
| `connectome` | Connectome LIF brain simulator | `taas-connectome`, `taas-train/connectome` |
| `karmac` | Cognitive architecture (GNW, LOD, Basal) | `karmac` |
| `cuda` | GPU acceleration (implies t3 + connectome) | `cuda-kernels` + CUDA backends |
| `python` | PyO3 + numpy zero-copy bindings | `pyo3`, `numpy` |
| `unity` | C ABI for Unity (implies connectome) | `taas-unity` |
| `all` | Everything | all of the above |

```bash
# Minimal: just SIMD kernels + T3 oscillator
cargo build --no-default-features

# With training
cargo build --features train

# Connectome brain simulator
cargo build --features connectome

# Full stack
cargo build --features all
```

## 🔬 Spiking Neuron Models

### SIMD Kernels (`crates/kernels/src/neuron/`) — 10 files

| File | Models | Integration | SIMD |
|:-----|:-------|:------------|:-----|
| `izhikevich.rs` | `izhikevich_step`, `eprop_izhikevich_step`, `eprop_izhikevich_step_multi` | Forward Euler (quadratic v²) | f32x8 |
| `lif.rs` | `lif_step`, `lif_at_step`, `eprop_lif_step` | Exponential Euler (exact for const I) | f32x8 |
| `adex.rs` | `adex_step`, `eprop_adex_step_multi` | Forward Euler + fast_exp | f32x8 |
| `saw.rs` | `saw_step`, `eprop_saw_step`, `psi_wave_step` | Forward Euler + noise / unit circle | f32x8 |
| `hh.rs` | `hh_lut_step` (scalar + SIMD) | LUT-interpolated exponential Euler | f32x8 |
| `fused.rs` | `fused_hidden_izhikevich_step`, `fused_output_lif_step`, `poisson_fused_hidden_step` | Fused: RNG → weights → dynamics in 1 pass | f32x8 |
| `dopamine.rs` | `dopamine_traces_step`, `dopamine_weight_update` | 3-factor STDP × dopamine (Izhikevich 2007) | f32x8 |
| `stp.rs` | `stp_step`, `stdp_update`, `stdp_homeostatic` | Tsodyks-Markram STP + homeostatic STDP | f32x8 |
| `delay.rs` | `delay_buffer_circular`, `delay_buffer_multi_tap` | Circular buffer for delays | f32x8 |

### Safe Wrappers (`crates/core/src/neuron/`) — 16 files

| File | Models |
|:-----|:-------|
| `izhikevich.rs` | `step_izhikevich`, `EPropIzhikevich`, `EPropIzhikevichMulti` |
| `lif.rs` | `step_lif`, `step_lif_at`, `EPropLIF` |
| `adex.rs` | `step_adex`, `EPropAdExMulti` |
| `saw.rs` | `step_saw`, `step_psi_wave`, `EPropSaw` |
| `hh.rs` | `HhLutConfig`, `HhState`, `step_hh_lut` |
| `stdp.rs` | `DopamineSTDP` — 3-factor plasticity |
| `stp.rs` | `ShortTermPlasticity`, `compute_lfp` |
| `sorn.rs` | `SornNetwork` — intrinsic plasticity + synaptic normalization |
| `encoding.rs` | BSA, Phase, TTFS spike encoders |
| `poisson.rs` | `PoissonGenerator` |
| `delay.rs` | `DelayBuffer` (circular) |
| `stats.rs` | SpikeRaster, firing rates, ISI, CV |
| `eprop.rs` | `surrogate_gradient`, `eprop_update` |
| `autotune.rs` | `probe_fr`, `suggest_config` |
| `trainer.rs` | `EPropTrainer` |

### Plasticity Rules

| Rule | Description |
|:-----|:------------|
| STDP | Spike-timing dependent plasticity (`stdp_update`, `stdp_homeostatic`) |
| Dopamine-STDP | 3-factor: STDP × dopamine reward signal (Izhikevich 2007) |
| STP | Tsodyks-Markram short-term (facilitation + depression) |
| E-Prop | Eligibility traces + surrogate gradient for LIF, Izhikevich, SAW, AdEx |
| Intrinsic Plasticity | Adaptive thresholds for target firing rate |
| Synaptic Normalization | Weight normalization to target sum |

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
| 0 | Baseline | 417 us | 1.0x |
| 1 | Fused weight→Izhikevich SIMD kernel | 289 us | 1.44x |
| 2 | Poisson RNG fused into SIMD pass | 260 us | 1.60x |
| 3 | Fast-exp in softmax | 254 us | **1.64x** |

### Fused SIMD Kernels

| Kernel | Purpose |
|:-------|:--------|
| `poisson_fused_hidden_step` | LCG RNG → weights → Izhikevich in 1 pass |
| `fused_output_lif_step` | hidden spikes → weights → LIF in 1 pass |
| `fused_hidden_izhikevich_step` | input spikes → weights → Izhikevich in 1 pass |
| `dopamine_weight_update` | Dw = n*D*e with SIMD clamp |

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

## 🧠 Cognitive Architecture (karmac)

| Module | Description |
|:-------|:------------|
| `gnw.rs` | Global Neuronal Workspace — conscious ignition |
| `basal_ganglia.rs` | Action selection via competitive inhibition |
| `working_memory.rs` | Persistent activity working memory |
| `lod.rs` | Level of Description — multi-scale interface |
| `semantic_pointer.rs` | Semantic Pointer Hypothesis (SPAUN-like) |

## 🧬 Connectome Brain Simulator

FlyWire connectome simulation with LIF neurons and spike propagation.

```rust
use taas_checkmate::connectome;

// Load FlyWire connectome, run simulation
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
👉 **[PYTHON_API_RU.md](docs/PYTHON_API_RU.md)**

```python
from taas_checkmate.neurons import step_psi_wave
from taas_checkmate.t3 import PyTriOscillator

agent.v       # свойство вместо copy_v(buf)
agent.weights # свойство вместо copy_weights(out)
```

## 📊 Performance

| Pipeline | Speed |
|:---------|:-----:|
| 3 x 64 oscillators + PST + NLMS | **388x realtime** @ 48kHz |
| 3 x 128 oscillators + Kalman | **240x realtime** @ 44.1kHz |
| Izhikevich 8K neurons | **1M neur/s** (x5 faster than NEST) |
| Sigmoid (Phase Gate) 8K | **8.2 ns/elem** (x3 faster than scipy) |

Benchmarks: `cargo bench --no-default-features` generates HTML reports in `target/criterion/`.

## 📚 Documentation

| File | Description |
|:-----|:------------|
| [T3 Architecture](docs/T3_ARCHITECTURE.md) | T3 oscillator engine internals |
| [Mathematical Audit](docs/MATHEMATICAL_AUDIT.md) | Equation verification |
| [API Reference](docs/API_REFERENCE.md) | Full API docs |
| [Python API (RU)](docs/PYTHON_API_RU.md) | Python bindings documentation |
| [RL API](docs/RL_API.md) | Reinforcement learning API |
| [VNC Ring Oscillator](docs/VNC_RING_OSCILLATOR.md) | VNC ring oscillator design |
| [GPT Inspector](docs/GPT_INSPECTOR.md) | Connectome-GPT analysis |
| [ArXiv Paper](docs/ARXIV_PAPER.md) | Research paper draft |

## 🚀 Quick Start

```bash
# Clone + build
cd taas_checkmate
cargo build --no-default-features

# Run tests
cargo test --no-default-features

# With features
cargo build --features t3,train,connectome

# Benchmarks
cargo bench --no-default-features

# Documentation
cargo doc --no-deps --open
```

## 📜 License

GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
