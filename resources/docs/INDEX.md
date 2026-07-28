# 🧠 taas-checkmate — Project Index

> **Pure Rust wavelet neural engine — SIMD, CUDA, FlyWire connectome.**  
> Repository: [github.com/anomalyco/taas-checkmate](https://github.com/anomalyco/taas-checkmate)

---

## 📦 Workspace Structure (10 crates)

```
taas_checkmate/
├── Cargo.toml          ← Workspace root + Python bindings (cdylib)
├── build.rs            ← CUDA nvcc compilation
├── src/
│   ├── lib.rs          ← Main crate (Python bindings via PyO3)
│   └── python/
│       ├── mod.rs      ← PyO3 module registration
│       ├── neuron.rs   ← Neuron step functions (LIF, Izhikevich, HH, GCO, Atlasov)
│       ├── connectome.rs     ← Connectome benchmarks (CPU + GPU)
│       ├── connectome_brain.rs ← BrainSimulator PyO3 class
│       ├── sparse.rs   ← CSR sparse operations
│       ├── hh.rs       ← HH LUT PyO3 class
│       ├── karmac.rs   ← CarmackBSP PyO3 class
│       ├── t3.rs       ← T3 oscillator + SPIGRAD
│       ├── eprop.rs    ← E-Prop learning
│       ├── ...         ← 20+ additional Python modules
│
├── crates/
│   ├── kernels/         ← [core] SIMD neuron kernels (f32x8)
│   │   ├── src/neuron/  ← LIF, Izhikevich, HH, AdEx, FHN, HR, WC, SAW,
│   │   │                  Dendrite, Synapse, STP, Dopamine, Astrocyte,
│   │   │                  Atlasov Neuromotor, DepthWave/GCO, Fused kernels
│   │   ├── src/safe.rs  ← Safe wrappers for SIMD kernels
│   │   ├── src/lib.rs   ← Public API re-exports
│   │   └── tests/       ← Integration tests (kernel_progon.rs)
│   │
│   ├── core/            ← [core] Safe wrappers + high-level neuron models
│   │   ├── src/neuron/  ← Wrappers: LIF, Izhikevich, HH (LUT), AdEx,
│   │   │                  AdEx, FHN, HR, WC, SAW, Atlasov, STP, STDP,
│   │   │                  E-Prop, SORN, Encoding, Autotune, Stats, Delay
│   │   ├── src/sparse.rs  ← GCO + CSR propagation
│   │   ├── src/kuramoto.rs ← T3 oscillator engine
│   │   └── src/koit.rs  ← XOR attention RNG
│   │
│   ├── connectome/      ← [optional] Connectome brain simulator
│   │   ├── src/sim.rs   ← sim_lif, sim_hh, sim_izhikevich, sim_gco
│   │   ├── src/data.rs  ← ConnectomeData (CSR + NT types)
│   │   ├── src/rcm.rs   ← Reverse Cuthill-McKee ordering
│   │   ├── src/cortical.rs ← Laminar cortical model
│   │   └── src/record.rs← SpikeRecorder
│   │
│   ├── t3/              ← [optional] T3 oscillator + PST + decoders
│   │   ├── src/oscillator.rs  ← TriOscillator
│   │   ├── src/pst.rs   ← Phase Space Topology features
│   │   ├── src/decoders.rs ← NLMS, Kalman, IK decoders
│   │   ├── src/trajectory.rs ← T2SpikeGrad (BPTT)
│   │   └── src/backend.rs / backend_cuda.rs
│   │
│   ├── train/           ← [optional] Training (RL, E-Prop, SPIGRAD)
│   │   ├── src/rl.rs    ← Actor-Critic agent
│   │   ├── src/spike/   ← Forward/backward BPTT
│   │   └── src/connectome_chase.rs ← Connectome chase training
│   │
│   ├── cuda-kernels/    ← [optional] CUDA GPU acceleration
│   │   ├── kernels/*.cu ← 25+ CUDA kernels
│   │   ├── src/lib.rs   ← Rust FFI bindings
│   │   ├── src/context.rs ← GpuContext
│   │   ├── src/ffi.rs   ← extern "C" declarations
│   │   └── src/kernels.rs ← Safe wrappers
│   │
│   ├── karmac/          ← [optional] Cognitive architecture
│   │   ├── src/lod.rs   ← Carmack LOD (octree)
│   │   ├── src/gnw.rs   ← Global Neuronal Workspace
│   │   ├── src/basal_ganglia.rs ← Action selection
│   │   ├── src/semantic_pointer.rs ← Semantic pointers
│   │   └── src/working_memory.rs ← Working memory
│   │
│   ├── taas-format/     ← [core] .taas file format (save/load)
│   │   ├── src/writer.rs / reader.rs ← Binary serialization
│   │   └── src/components.rs ← BrainState save/load
│   │
│   ├── taas-embodied/   ← [optional] Embodied simulation
│   │   └── src/fly_embodied.rs ← Fly + MuJoCo integration
│   │
│   ├── cpe/             ← [core] Chaos Pattern Engine
│   │   ├── src/coherence.rs, stdp.rs, hybrid.rs
│   │   └── src/dopamine.rs ← 3-factor plasticity
│   │
│   ├── iit/             ← [core] Integrated Information Theory
│   │   └── src/lib.rs   ← Phi computation
│   │
│   ├── gpu-eval/        ← [core] GPU evaluation helpers
│   │
│   └── unity/           ← [optional] Unity C ABI bindings
│       └── cs/TaaSBrain.cs ← C# bridge
│
├── examples/
│   ├── core/            ← Core engine examples
│   ├── connectome/      ← FlyWire connectome baking
│   ├── mnist/           ← MNIST training (eprop, t3)
│   ├── hh/              ← HH calibration + visualization
│   ├── fly/             ← MuJoCo fly simulations
│   │   ├── scripts/     ← Python scripts (VNC, CPG, walking)
│   │   └── models/      ← MuJoCo XML + OBJ assets
│   └── python/          ← Legacy Python scripts
│
├── tests/               ← Integration tests
├── benches/             ← Criterion benchmarks (bench_all.rs)
└── docs/                ← Documentation
    ├── INDEX.md         ← THIS FILE — project map
    ├── README_SAKHA.md  ← Yakut language README
    ├── ARCHITECTURE.md  ← System architecture
    ├── NEURON_MODELS.md ← Neuron model reference
    ├── ATLASOV_NEUROMOTOR.md ← Atlasov muscle engine
    ├── RECEPTORS_AND_NEUROTRANSMITTERS.md ← Synaptic physiology
    ├── VNC_BIOMECHANICS.md ← VNC + MuJoCo biomechanics
    ├── PYTHON_API_RU.md ← Python API reference (Russian)
    ├── RL_API.md        ← Reinforcement learning API
    ├── T3_ARCHITECTURE.md ← T3 oscillator architecture
    └── MATHEMATICAL_AUDIT.md ← Mathematical formula audit
```

---

## 🔧 Crate Dependency Graph

```
taas-checkmate (root)
├── taas-kernels        ← NO DEPENDENCIES (standalone SIMD kernels)
├── taas-core           ← depends on: kernels
├── taas-format         ← depends on: kernels, core, connectome, t3, train
├── cpe                ← depends on: kernels, core
├── iit                ← depends on: kernels, core
├── gpu-eval           ← depends on: (standalone)
│
├── t3 [optional]      ← depends on: kernels, core
├── train [optional]   ← depends on: kernels, core, t3
├── connectome [optional] ← depends on: kernels, core, t3
├── cuda-kernels [optional] ← depends on: kernels, core, t3, connectome
├── karmac [optional]  ← depends on: kernels, core
├── taas-embodied [optional] ← depends on: kernels, core, connectome
└── unity [optional]   ← depends on: connectome
```

---

## 🏷️ Feature Flags

| Feature | Includes | Purpose |
|---------|----------|---------|
| `default` | `t3` | T3 oscillator engine |
| `t3` | `taas-t3` | Oscillator + PST + decoders |
| `train` | `taas-train/t3` | BPTT, SPIGRAD, connectome chase |
| `connectome` | `taas-connectome` + `taas-train/connectome` | Connectome brain simulator |
| `karmac` | `karmac` | Cognitive architecture (GNW, LOD) |
| `cuda` | `t3` + `connectome` + `cuda-kernels` | GPU acceleration |
| `python` | `pyo3` + `numpy` + `connectome` + `t3` + `karmac` | Python bindings |
| `unity` | `connectome` | Unity C ABI |
| `all` | Everything | Full build |

---

## 🧪 Test Inventory

| Package | Tests | Scope |
|---------|-------|-------|
| `taas-kernels` | **155** | LIF (5), Izhikevich (13), HH (SIMD), AdEx, FHN, Dendrite, Synapse, Astrocyte, Atlasov (7), DepthWave/GCO (3), E-Prop, Fused |
| `taas-core` | **40** | Neuron wrappers, HH LUT (14), E-Prop, Poisson, Autotune, SORN, Encoding, STDP |
| `taas-connectome` | **42** | ConnectomeData, sim_lif, sim_hh, RCM, Cortical, Record |
| `taas-t3` | **77** | Oscillator, PST, Backend (CPU+CUDA), Trajectory, Decoders |
| `cuda-kernels` | **17** | GPU kernels (requires nvcc) |
| `karmac` | **29** | LOD (octree), GNW, Basal Ganglia, Semantic Pointer, Working Memory |
| `taas-iit` | **5** | Phi computation |
| `taas-embodied` | **2** | Fly embodied pipeline |
| **Total** | **~367+** | |

---

## 📊 Key Validation Results

| Model | Tests | Industry Standard Verified |
|-------|-------|--------------------------|
| **LIF** | 5 | ✅ Analytical F-I curve, refractory period, exponential Euler accuracy |
| **Izhikevich** | 13 | ✅ 5 canonical firing patterns (RS, FS, IB, LTS, CH) per Izhikevich 2003 |
| **HH (LUT)** | 14 | ✅ H&H 1952: gating variables, AP shape, threshold, I-F curve, input resistance |
| **Atlasov** | 7 | ✅ Ca²⁺ kinetics, Hill-type f(L) & f(V), agonist-antagonist |
| **GCO** | 3 | ✅ Phase coherence maintainance |

---

## 🔄 CI/CD

See `.github/workflows/ci.yml`:
- Rust stable build
- `cargo test --all-features`
- Clippy linting

---

*Last updated: 2026-07-26*
