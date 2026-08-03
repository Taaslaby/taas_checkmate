# 🏛️ Pillars — Zero-Cost SIMD Engine for Spiking Neural Networks

Pillars is a **SIMD neural dynamics engine** that compiles the same Rust code into
AVX2/NEON vector instructions and scalar fallback — without interpreters, JITs, or
runtime dispatch.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     🚀 API (high-level)                      │
│  Network, NeuralPopulation, SparseGraph                      │
│  Safe Rust — no raw pointers, no unsafe                      │
├──────────────────────────────────────────────────────────────┤
│                     🦁 ZOO (models)                          │
│  15 point neurons + 4 synapses + sensors + glia              │
│  Each model: impl Model + impl ComputeNeuron                 │
├──────────────────────────────────────────────────────────────┤
│                🧬 OPS (biological primitives)                │
│  spike, ohmic, gate, calcium, plasticity, neurochemistry     │
│  Pure S: SimdLane → {...} — same code for SIMD & scalar     │
├──────────────────────────────────────────────────────────────┤
│          ⚙️ CORE (SIMD engine)                               │
│  SimdLane trait  →  f32x8 (AVX2)  │  f32 (scalar tail)      │
│  ─────────────────────────────────────────────────────────── │
│  Each method = 1 hardware instruction (cmp_ge → vcmpps,     │
│  exp → _mm256_exp_ps, etc.)                                  │
├──────────────────────────────────────────────────────────────┤
│                      🖥️ HARDWARE                             │
│  x86_64 (AVX2) / aarch64 (NEON) / wasm32 (SIMD128) / any    │
└──────────────────────────────────────────────────────────────┘
```

## Layer Boundaries — Where Math Becomes Binary

| Layer | Abstraction | What lives here | Safe? |
|-------|-------------|-----------------|:-----:|
| **`api/`** | Network orchestrator | Population wiring, spike routing, delays | ✅ Safe |
| **`zoo/`** | Neuron models | LIF, HH, Izhikevich, ... | ✅ Safe |
| **`ops/`** | Math primitives | `ohmic(g, e, v)`, `gate_integrate(...)` | ✅ Safe |
| **`core/lane.rs`** | SimdLane trait | `cmp_ge` → `vcmpps`, `exp` → `vexpps` | ⚠️ unsafe boundary |
| **`core/lane_f32x8.rs`** | AVX2 instructions | `_mm256_*` intrinsics | 🔒 unsafe |
| **`core/lane_f32.rs`** | Scalar fallback | `f32::` operations | ✅ Safe |

### Compilation Pipeline

```
User code:  v.cmp_ge(v_thresh)
    ↓
SimdLane trait (generics)
    ↓
Monomorphization for S = f32x8  ─or─  S = f32
    ↓                           ↓
_mm256_cmp_ps(v, thresh, GE)   f32::cmp(&v, &thresh)
    ↓                           ↓
AVX2 instruction: vcmpps       scalar: ucomiss + setae
```

The compiler resolves the entire SIMD abstraction at **compile time**.
Zero runtime overhead. Zero interpretive layers.

## Quick Start

```rust
use pillars::prelude::*;

// 1. Create a population of 138k LIF neurons
let mut pop = NeuralPopulation::<LIFModel>::new(138_000);

// 2. Set parameters (SIMD-aware, all neurons homogeneous)
pop.fill_param(0, 0.5);     // dt
pop.fill_param(1, 0.1);     // inv_tau_m
pop.fill_param(2, -70.0);   // v_rest
pop.fill_param(3, -54.0);   // v_thresh
pop.fill_param(4, -70.0);   // v_reset
pop.fill_param(5, 2.0);     // tau_ref
pop.fill_state(0, -70.0);   // v

// 3. Run
let i_in = vec![30.0; 138_000];
let mut spikes = vec![0i32; 138_000];
pop.step(&i_in, &mut spikes);
```

## Performance

| Model | 138k × 100 steps | Throughput |
|-------|:----------------:|:----------:|
| HH (raw scalar) | 528 ms | 271 Kelem/s |
| HH (SIMD ODE) | 204 ms | 675 Kelem/s |
| HH (LUT 32KB L1) | 210 ms | 658 Kelem/s |
| LIF (SIMD ODE) | ~30 ms | ~4.6 Melem/s |

Benchmarks: `cargo bench --package pillars --bench bench_hh`

## Project Structure

```
crates/pillars/
├── README.md
├── Cargo.toml
├── src/
│   ├── lib.rs          Crate-level exports, Q10, E_rev constants
│   ├── prelude.rs      Single import: use pillars::prelude::*
│   ├── core/           SIMD engine (SimdLane, context, runner)
│   │   ├── arch.rs     Compile-time SIMD width detection (SimdType, LANES)
│   │   ├── buf.rs      AlignedVec — 32-byte aligned heap buffers
│   │   ├── lane*.rs    SimdLane trait → f32x8 (AVX2/NEON) | f32 (scalar)
│   │   ├── context.rs  SimdContext — state/params/buses, MAX_STATES=16
│   │   ├── compute.rs  ComputeNeuron — typed model API
│   │   ├── population.rs NeuralPopulation — safe SoA runner
│   │   └── runner.rs   Legacy raw-pointer runners run_neuron*
│   ├── math/           Canonical per-lane primitives (sigmoid, vtrap, fast_exp)
│   ├── ops/            Biological LEGO blocks (~40 ops)
│   │   ├── channels/   Ion channels (Kir, HCN, Gap, GluCl, KM)
│   │   ├── learning/   Hebbian, SGD, surrogate gradients
│   │   ├── neurochemistry/ GPCR cascades, receptors, signaling
│   │   ├── physics/    Diffusion, ephaptic fields, metabolism
│   │   └── plasticity/ STDP, STP, homeostatic, trace
│   ├── zoo/            Model library
│   │   ├── neurons/    Point neurons (LIF, HH, Izhikevich, AdEx, GLIF, …)
│   │   ├── synapses/   Synaptic dynamics (NMDA, Tsodyks-Markram, CaPlasticity, Exp2Syn)
│   │   ├── sensors/    Bio-inspired sensors
│   │   └── glia/       Glial modulation (astrocyte)
│   ├── api/            High-level API (Network, SparseGraph, Population)
│   ├── kernels/        Batch f32x8 microkernels (sibling of ops/)
│   └── prelude.rs      Single import
├── benches/            Benchmarks (bench_hh, bench_lif, bench_glif4_haap)
└── tests/              Integration tests (vs NEURON reference, network delays)
```
