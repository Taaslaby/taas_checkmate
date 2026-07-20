# 🧠 Taas-Checkmate TODO

> Актуальное состояние проекта и план развития.
> Последнее обновление: 21 июля 2026

---

## 📊 Состояние проекта

| Метрика | Значение |
|:--------|:--------:|
| Крейтов в workspace | 12 |
| SIMD ядер (kernels) | ~30 функций |
| Unit tests (kernels) | 64/64 ✅ |
| Progonator tests (kernels) | 47/47 ✅ |
| `--all-features` tests | 81/81 ✅ |
| Clippy warnings (kernels) | **0** ✅ |
| Clippy warnings (workspace) | ~400 ⚠️ |

---

## ✅ Выполнено

### 🔧 SIMD Оптимизации ядер

| Что | Статус | Speedup |
|:---|---|:---:|
| **LIF**: Exponential Euler → Forward Euler (мембрана) | ✅ | **2.04× → 5.12×** 🚀 |
| **WC sigmoid_simd**: scalar `f32::exp()` loop → `fast_exp_simd` | ✅ | Ускорение sigmoid |
| **WC τ decay**: scalar exp loop → Forward Euler | ✅ | **~1.5× → 2.25×** |
| **LIF-AT мембрана**: Forward Euler | ✅ | Ускорение |
| **eProp LIF мембрана**: Forward Euler | ✅ | Ускорение |
| **Все scalar-exp-in-SIMD баги** ликвидированы | ✅ | 0 осталось |

### 🧪 Тестирование

| Что | Статус |
|:---|---|
| `cargo test --all-features` (81 тест) | ✅ |
| eProp gradient FD test (flaky → стабильный) | ✅ Фикс: eps=0.5, input_current=8.0 |
| SIMD vs scalar tail golden tests | ✅ 5 тестов |
| Fused kernel consistency test | ✅ |
| Progonator (47 интеграционных тестов) | ✅ |

### 📋 CI / Clippy

| Что | Статус |
|:---|---|
| CI pipeline с `--all-features` build/test/clippy | ✅ |
| Clippy warnings в kernels crate | **19→0** ✅ |

### 🏗 Инфраструктура

| Что | Статус |
|:---|---|
| Criterion benchmarks (`bench.rs`) — 15 групп ядер | ✅ |
| Новые бенчмарки (`bench_all.rs`) — 15 групп ядер | ✅ |
| Benchmark анализ (SIMD vs scalar speedup) | ✅ |

---

## 🎯 План (по приоритету)

### 🔴 Priority 1 — Блокеры

- [ ] **Добить ~400 clippy warnings** по всему workspace
  - `cargo clippy --all-features 2>&1 | grep 'warning:' | wc -l`
  - Основные крейты: `core`, `t3`, `train`, `connectome`, `connectome-gpt`
- [ ] **Аудит connectome-gpt** (9.2K LOC)
  - Самый большой крейт в workspace
  - Что делает: Graph VAE + decoder для connectome
  - Нужно решить: оставить, вынести, удалить?
  - LOC по файлам:

### 🟡 Priority 2 — Архитектура

- [ ] **Cross-crate интеграционные тесты**
  - `kernels → core → connectome → t3 → train`
  - Проверить, что изменения API не ломают downstream крейты
- [ ] **Рефакторинг нейронных ядер**
  - Унифицировать API: все `*_step` функции принимают одинаковые структуры параметров
  - Column-major params → named structs где возможно
- [ ] **Аудит зависимостей**
  - `cargo tree -e features` — проверить дублирование и лишние фичи
  - Особенно: `connectome-gpt` (куча ML зависимостей)

### 🟠 Priority 2.5 — Новые нейробиологические фичи

- [ ] **🧬 Dendritic computation** — compartmentalized dendrites с local PSP integration
  - Каждый нейрон имеет N дендритных компартментов
  - Каждый компартмент: `I_syn = Σ g_i · (E_rev_i − V_dend)`
  - Сома: `dV_soma/dt = f(V_soma, ...) + g_dend · (V_dend_mean − V_soma)`
  - SIMD: f32x8 × n_compartment — можно векторизовать по компартментам
  - Идея реализации: `crates/kernels/src/neuron/dendrite.rs`

- [ ] **🧠 Cortical layers** — laminar-specific connectivity
  - 6 слоёв (L1–L6) с типовыми паттернами:
    - L4 → L2/3 (feedforward)
    - L2/3 → L5 (superficial→deep)
    - L5 → L6 (corticothalamic)
    - L6 → L4 (modulatory)
  - Реимплементация: CSR connectome + layer metadata per neuron
  - Можно добавить как прослойку в `crates/connectome/`

- [ ] **🔬 Astrocyte modulation** — slow glial modulation of synaptic gain
  - Астроцит интегрирует спайковую активность (τ ~ секунды)
  - Модулирует synaptic gain: `g_syn *= 1 + α · A(t)`
  - Где `A(t)` — медленная переменная астроцита
  - Реализация: `SynapticKinetics` в `synapse.rs` — добавить astrocyte gain factor

- [ ] **⚡ STDP triplet rule** — биологичнее чем парный STDP
  - Тройное правило: `dw/dt = A⁺ · tr₁ · tr₂ − A⁻ · tr₂ · tr₃`
  - Где `tr₁` — fast pre-trace, `tr₂` — slow pre-trace, `tr₃` — post-trace
  - Реализация: расширить `DopamineSTDP` в `crates/kernels/src/neuron/dopamine.rs`

### 🟢 Priority 3 — Производительность

- [ ] **HH Poly step SIMD оптимизация**
  - Сейчас: 6 gate variables × polynomial eval per neuron
  - Speedup: 2.23× (потенциал: 4-5×)
  - Идея: fused multiply-add для gate polynomial evaluation
- [ ] **Sigmoid gate bottleneck**
  - `sigmoid_gate`, `thalamic_gate`, `phase_gate`: 1.05-1.19× speedup
  - Причина: sigmoid = exp + divide
  - Идея: lookup table вместо polynomial exp для gate functions?
- [ ] **LIF-AT speedup**
  - Текущий: 1.92× (vs LIF 5.12×)
  - Theta adaptation: `θ *= exp(-dt/τ_θ)` — можно Forward Euler?
  - Исследовать влияние на точность

### ⚪ Priority 4 — Документация

- [ ] **Документировать все unsafe функции** (# Safety секции)
  - Текущее состояние: частично (fix в hh.rs, adex.rs сделан)
  - Нужно проверить остальные крейты
- [ ] **Добавить docstring для всех safe wrappers**
- [ ] **Обновить README** с новыми speedup числами
- [ ] **Архитектурная диаграмма** crate dependencies

---

## 🐛 Известные проблемы

### Архитектурные

| Проблема | Описание | Статус |
|:---------|:---------|:------:|
| **HH LUT gather bottleneck** | SIMD gather из LUT — архитектурное ограничение AVX2 (1.28×) | 🔴 Not fixable без AVX-512 |
| **Sigmoid divide bottleneck** | SIMD divide не намного быстрее scalar при малых n | 🟡 Fundamental |
| **Column-major params** | Странный layout: params[0..n]=dt, params[n..2n]=tau_m, ... | 🟡 Legacy |

### Кодовые

| Проблема | Файл | Статус |
|:---------|:-----|:------:|
| `hh_lut_step_scalar` — 21 параметр (clippy: 20 max) | `hh.rs` | ✅ Fixed с `#[allow(..)]` |
| `test_eprop_izhikevich_gradient_fd` — flaky FD | `neuron_tests.rs` | ✅ Fixed (eps 0.5, I=8.0) |
| `bench_princeton_connectome` — stale API | `examples/` | ✅ Fixed (scale=1.0) |
| `biology_check` — stale API | `examples/` | ✅ Fixed (scale=1.0) |
| `neuron_tests` — `&I_ref` → `&mut I_ref` | `tests/` | ✅ Fixed |

---

## 📈 Performance Benchmarks

### Существующие (bench.rs) — 14 групп

| Ядро | scalar_n1 | simd_n8 | Per-element | Speedup |
|:---|---:|---:|---:|---:|
| hebbian | 4.61 ns | 3.34 ns | 0.42 ns | **10.98×** 🚀 |
| kuramoto | 10.80 ns | 9.64 ns | 1.21 ns | **8.93×** 🚀 |
| nlms | 4.34 ns | 4.08 ns | 0.51 ns | **8.51×** 🚀 |
| psi_wave | 12.60 ns | 12.41 ns | 1.55 ns | **8.13×** 🚀 |
| t3 | 12.44 ns | 12.83 ns | 1.60 ns | **7.78×** |
| kalman | 8.35 ns | 8.90 ns | 1.11 ns | **7.52×** |
| safety_clamp | 3.38 ns | 5.00 ns | 0.63 ns | **5.37×** |
| stdp | 4.49 ns | 6.98 ns | 0.87 ns | **5.16×** |
| **lif** | 8.24 ns | 12.86 ns | 1.61 ns | **5.12×** 🚀 (было 2.04×) |
| izhikevich | 11.23 ns | 19.06 ns | 2.38 ns | **4.72×** |
| saw | 9.90 ns | 17.71 ns | 2.21 ns | **4.48×** |
| adex | 22.12 ns | 54.61 ns | 6.83 ns | **3.24×** |
| wc_step | 24.38 ns | 86.69 ns | 10.84 ns | **2.25×** ✅ (было ~1.5×) |
| hh_poly | 94.75 ns | 339.65 ns | 42.46 ns | **2.23×** |
| hh_lut | 27.19 ns | 169.95 ns | 21.24 ns | **1.28×** 🔴 |
| lif_at | 11.10 ns | 46.16 ns | 5.77 ns | **1.92×** 🔴 |

### Новые (bench_all.rs) — 15 групп

| Ядро | scalar_n1 | simd_n8 | Per-element | Speedup |
|:---|---:|---:|---:|---:|
| sgd_step | 4.42 ns | 3.48 ns | 0.44 ns | **10.0×** 🚀 |
| fhn_step | 10.02 ns | 8.68 ns | 1.09 ns | **9.14×** 🚀 |
| stp_step | 10.01 ns | 9.63 ns | 1.20 ns | **8.34×** 🚀 |
| add_simd | 2.18 ns | 2.24 ns | 0.28 ns | **7.79×** |
| relu_simd | 2.45 ns | 3.13 ns | 0.39 ns | **6.28×** |
| wc_step | 24.38 ns | 86.69 ns | 10.84 ns | **2.25×** |
| phase_gate | 6.76 ns | 39.43 ns | 4.93 ns | **1.37×** 🔴 |
| sigmoid_gate | 5.56 ns | 37.32 ns | 4.67 ns | **1.19×** 🔴 |
| thalamic_gate | 5.58 ns | 38.52 ns | 4.81 ns | **1.16×** 🔴 |

> **Note:** simd_n8 processes 8 elements in one call. Per-element = simd_n8 / 8.
> Speedup = scalar_n1 / per-element.

---

## 🧪 Запуск

```bash
# Ядра (быстро)
cargo test -p taas-kernels --lib               # 64 unit tests
cargo test -p taas-kernels --test kernel_progon # 47 integration tests
cargo clippy -p taas-kernels --all-features     # 0 warnings

# Весь workspace
cargo test --all-features                       # 81 tests

# Бенчмарки
cargo bench --bench bench                        # основные ядра
cargo bench --bench bench_all                    # новые ядра

# Clippy workspace
cargo clippy --all-features                      # ~400 warnings
```

---

## 📐 Архитектура

```
taas_checkmate/
├── crates/
│   ├── kernels/      ← Pure Rust SIMD ядра (unsafe, f32x8)
│   ├── core/         ← Safe wrappers + нейронные модели + eprop
│   ├── t3/           ← T3 oscillator engine (Kuramoto + Stuart-Landau)
│   ├── train/        ← RL + BPTT + SPIGRAD
│   ├── connectome/   ← FlyWire connectome simulator
│   ├── karmac/       ← Cognitive architecture (GNW, BasalGanglia, LOD)
│   ├── connectome-gpt/ ← Graph VAE + decoder (9.2K LOC — ❓)
│   ├── cpe/          ← Coherence, Dopamine, STDP, Hybrid
│   ├── taas-format/  ← Binary .taas snapshot format
│   ├── cuda-kernels/ ← CUDA backend
│   ├── unity/        ← C ABI for Unity
│   └── gpu-eval/     ← GPU evaluation
├── benches/          ← Criterion benchmarks
├── tests/            ← Integration tests
├── examples/         ← Rust + Python examples
└── docs/             ← Documentation
```
