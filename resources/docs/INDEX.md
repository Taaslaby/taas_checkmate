# 🧠 taas-checkmate — Project Index

> **Pure Rust wavelet neural engine — SIMD, CUDA, FlyWire connectome.**  
> Repository: [github.com/anomalyco/taas-checkmate](https://github.com/anomalyco/taas-checkmate)

---

## 📦 Workspace Structure (8 crates)

```
taas_checkmate/
├── Cargo.toml          ← Workspace root + Python bindings (cdylib)
├── build.rs            ← CUDA nvcc compilation
├── src/
│   ├── lib.rs          ← Main crate: re-exports pillars, connectome (= pillars-connectome),
│   │                      python bindings, unity ABI, karmac (inline), iit (inline)
│   ├── python/
│   │   ├── mod.rs      ← PyO3 module registration (taas_checkmate.pillars facade)
│   │   ├── neuron.rs   ← step_* нейронов (LIF, Izhikevich, HH, GCO, ...) — скаляр-или-массив
│   │   ├── pillars.rs  ← NeuralPopulation, Network, ConnectomeBrain, SynapticConnectomeBrain
│   │   ├── connectome_brain.rs ← BrainSimulator / BrainRunner PyO3 классы
│   │   ├── connectome.rs       ← Connectome + BrainIO (legacy ABI)
│   │   ├── sparse.rs   ← CSR sparse operations
│   │   ├── karmac.rs   ← Cognitive architecture PyO3 классы
│   │   ├── t3.rs       ← T3 oscillator + SPIGRAD
│   │   ├── ...         ← 20+ additional Python modules
│   ├── unity.rs        ← C ABI для Unity
│   └── karmac/         ← [feature karmac] GNW, LOD, BasalGanglia (inline)
│
├── crates/
│   ├── pillars/          ← [core] ЕДИНЫЙ ДВИЖОК (api + zoo + kernels + core)
│   │   ├── src/api/      ← NeuralPopulation, Network, SparseGraph, builder,
│   │   │                  stateful-классы (EPropLIF, SornNetwork, DelayBuffer, ...)
│   │   ├── src/zoo/      ← NeuronModel'ы: LIF, Izhikevich (+dual), HH, AdEx, FHN, HR,
│   │   │                  WC, SAW, GCO, Rate, ... (#[neuron] derive)
│   │   ├── src/kernels/  ← signal, decode, input, synapse, learning, phase,
│   │   │                  depth_wave, astrocyte, wc, io, stats
│   │   ├── src/core/     ← SIMD runner (run_neuron), lane, buf, population, network
│   │   └── tests/        ← 18+ parity-тестов против легаси-эталонов
│   │
│   ├── pillars-macros/   ← #[neuron] / #[dual_neuron] derive-макросы моделей
│   ├── pillars-cuda/     ← CUDA-ускорители pillars (warp kernels)
│   │
│   ├── pillars-connectome/ ← [optional] ЕДИНЫЙ connectome-крейт (волна 12: легаси
│   │   │                      taas-connectome слит сюда и удалён)
│   │   ├── src/data.rs   ← ConnectomeData (i32-CSR + NT + receptor_types + groups),
│   │   │                  BrainIO/каналы, .taas save/load, рецептор-билдеры
│   │   ├── src/brain.rs  ← Brain<M> (pillars::api::Network + SparseGraph внутри)
│   │   ├── src/synaptic.rs ← SynapticBrain<M> (conductance: AMPA/NMDA/GABA kinetics)
│   │   ├── src/sim.rs    ← sim_lif, sim_izhikevich(+dual/synaptic), sim_rate, sim_gco,
│   │   │                  sim_hh_*, hh_population_* (делегаты crate::Brain/SynapticBrain)
│   │   ├── src/runner.rs ← BrainRunner (конфигурируемый runner, Poisson-группы, LearningRule)
│   │   ├── src/record.rs ← SpikeRecorder
│   │   ├── src/rcm/      ← Reverse Cuthill–McKee ordering (bfs/spectral/permute)
│   │   ├── src/types.rs / propagate.rs ← SpikeQueue, BrainState, params, CSR propagation
│   │   ├── tests/hh_parity.rs ← SIMD/scalar + pacemaking/standard HH гарантии
│   │   └── benches/bench_rcm.rs ← RCM bandwidth + propagate до/после (5% и 10%)
│   │
│   ├── t3/              ← [optional] T3 oscillator + PST + decoders
│   ├── train/           ← [optional] Training (RL, E-Prop, SPIGRAD, connectome chase)
│   └── taas-format/     ← [core] .taas file format (save/load, BrainIO, ConnectomeData)
│
├── resources/
│   ├── docs/            ← Документация (INDEX, API_REFERENCE, PYTHON_API_RU, ...)
│   ├── examples/        ← Примеры (spiking_gpt, spaun_pipeline, embodied_creature, ...)
│   ├── validate/        ← NEURON-валидация (hh/izh/adex/fhn .mod + compare_all.py)
│   └── neurofly/        ← FlyWire/MANC pipeline (подмодуль)
│
├── tests/               ← Integration tests (smoke API, neuron, kuramoto, sparse, ...)
└── benches/             ← Criterion benchmarks (bench.rs, bench_all.rs, bench_lod_gnw.rs)
```

---

## 🔧 Crate Dependency Graph

> ⚠️ Актуально (Волна 12, 2026-08-08): `taas-core`, `taas-kernels`,
> `taas-connectome` **удалены**. `pillars` — единый канон
> (api/zoo/kernels/core); `pillars-connectome` — единственный connectome-крейт
> (легаси API сохранён 1:1, корневой `connectome::` = `pillars_connectome::`).
> См. `UNIFIED_API.md`.

```
taas-checkmate (root)
├── pillars              ← единый канон (зависит только от pillars-macros)
├── pillars-macros       ← derive-макросы (без зависимостей)
├── pillars-cuda         ← depends on: pillars, pillars-macros; gpu-модуль (GpuContext + 23 AOT-микроядра kernels/*.cu)
├── pillars-connectome   ← depends on: pillars (+ pillars-cuda, feature "cuda")
├── t3 [optional]        ← depends on: pillars (+ pillars-cuda, feature "cuda")
├── train [optional]     ← depends on: pillars, t3, pillars-connectome, pillars-cuda
└── taas-format          ← depends on: pillars, pillars-connectome, t3 (+ train)
```

---

## 🏷️ Feature Flags

| Feature | Includes | Purpose |
|---------|----------|---------|
| `default` | `t3` | T3 oscillator engine |
| `t3` | `taas-t3` | Oscillator + PST + decoders |
| `train` | `taas-train/t3` | BPTT, SPIGRAD, connectome chase |
| `connectome` | `pillars-connectome` + `taas-train/connectome` | Connectome brain simulator |
| `karmac` | `karmac` | Cognitive architecture (GNW, LOD) |
| `cuda` | `t3` + `connectome` + `pillars-cuda` + `pillars-connectome/cuda` | GPU acceleration |
| `python` | `pyo3` + `numpy` + `pillars-connectome` + `connectome` + `t3` + `karmac` | Python bindings |
| `unity` | `connectome` | Unity C ABI |
| `all` | Everything | Full build |

---

## 🧪 Test Inventory

> Числа — по последнему прогону; свежие значения — `cargo test` / CI.

| Package | Tests | Scope |
|---------|-------|-------|
| `pillars` | 18+ parity-бинарей + lib | run_neuron, zoo-модели, api (Network/Graph/Build), parity vs легаси |
| `pillars-connectome` | **55** | ConnectomeData, Brain/SynapticBrain, sim_lif/izh/rate/gco/hh, RCM, Runner, hh_parity (4) |
| `taas-t3` | **77** | Oscillator, PST, Backend (CPU+CUDA), Trajectory, Decoders |
| `taas-train` | **24** | RL agent, connectome chase, E-Prop |
| `pillars-cuda` | **38** | NVRTC zoo-модели + gpu-модуль (GpuContext, микроядра, parity) (requires nvcc) |
| `taas-format` | **18** | .taas writer/reader, BrainIO, ConnectomeData roundtrip |
| **Total** | **~390+** | |

---

## 📊 Key Validation Results

| Model | Verified | Notes |
|-------|----------|-------|
| **LIF** | ✅ | Analytical F-I curve, refractory period, exponential Euler accuracy |
| **Izhikevich** | ✅ | 5 canonical firing patterns (RS, FS, IB, LTS, CH) per Izhikevich 2003 |
| **HH** | ✅ | H&H 1952 via `pillars::HHModel` (SIMD/scalar parity, pacemaking CaV-T) |
| **AdEx / FHN / HR / WC** | ✅ | Parity vs NEURON (resources/validate) |
| **GCO** | ✅ | Phase coherence maintenance |

---

## 🔄 CI/CD

See `.github/workflows/ci.yml`:
- Rust stable build
- `cargo clippy --features "t3,train,connectome,karmac,unity" -- -D warnings`
- `cargo build` + `cargo test` с теми же фичами

---

*Last updated: 2026-08-08 (Волна 12 — единый pillars-connectome)*
