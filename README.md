# taas-checkmate — симулятор мозга на Rust

[![Rust](https://img.shields.io/badge/rust-1.78%2B-blue)]()
[![SIMD](https://img.shields.io/badge/SIMD-f32x8-orange)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green)]()
[![License](https://img.shields.io/badge/license-GPL--3.0-blue)]()
[![Сахалыы](https://img.shields.io/badge/тыл-Сахалыы-green)](resources/docs/README_SAKHA.md)

👉 [Документация на якутском (README_SAKHA.md)](resources/docs/README_SAKHA.md)

## Что это

**taas-checkmate** — быстрый симулятор биологических нейронов и нейронных сетей на Rust с ускорением на GPU (CUDA).

В отличие от обычного машинного обучения, где нейроны — просто числа в матрице, здесь нейроны ведут себя как настоящие: накапливают заряд, выдают спайки (электрические импульсы), общаются через химические рецепторы и учатся прямо во время работы.

## Как это работает — простыми словами

### Нейрон — стакан с дыркой
Представьте стакан с маленькой дырочкой на дне.
- В него постоянно льётся вода (ток от соседних нейронов).
- Если лить понемногу — вода вытекает через дырочку, ничего не происходит.
- Если влить резко — уровень дойдёт до края (**порог**) и стакан «перевернётся»: это **спайк**, электрический импульс.
- После импульса стакан опустошается (**сброс**) и немного отдыхает (**рефрактерный период**).

### Дендриты — ветки дерева
Нейрон разветвляется на дендриты — по ним сигнал течёт к центру клетки. По пути часть сигнала теряется. В этом проекте дендриты **активные**: на их стенках стоят ионные каналы (Na⁺, K⁺, Ca²⁺), которые могут усилить сигнал и выдать локальный **дендритный спайк**.

### Синапсы — замки и ключи
Спайк одного нейрона — это пузырёк химии для соседа. На стенках соседа стоят рецепторы-замки:
- **AMPA** — быстрый: открылся сразу, ток пошёл.
- **NMDA** — умный: открывается только если нейрон уже немного заряжен (пробка из магния). Основа памяти.
- **GABA** — тормозной: не наливает, а сливает, защищая сеть от перегрева.

### Скорость — SIMD и GPU
Миллионы таких «стаканов» нельзя считать по одному.
- **SIMD (AVX2/AVX-512)**: 8–16 нейронов обновляются за один такт процессора.
- **GPU (CUDA)**: тысячи ядер видеокарты считают весь коннектом (например, мозг мухи) в реальном времени.

### Обучение — Хебб и дофамин
- **STDP**: если нейрон A сработал перед нейроном B — связь между ними усиливается («сработали вместе — связаны вместе»).
- **Дофамин**: мозг хранит след недавних действий; если приходит награда, следы превращаются в прочные связи.

## Структура проекта

```
taas_checkmate/
├── crates/
│   ├── pillars/            ← основной движок: SIMD-ядра + модели нейронов
│   │                          (LIF, Izhikevich, HH, AdEx, FHN…)
│   ├── pillars-macros/     ← процедурные макросы (#[neuron])
│   ├── pillars-cuda/       ← CUDA: NVRTC-компиляция моделей + GPU-контекст
│   ├── pillars-connectome/ ← симулятор связей мозга (коннектом мухи FlyWire)
│   ├── prometheus-grow/    ← движок развития: нейроны растут из «генома»
│   │                          (порт CX3D + модели NETMORPH)
│   ├── t3/                 ← осцилляторы и ритмы мозга (Курамото)
│   ├── train/              ← обучение (RL Actor-Critic, E-Prop, SPIGRAD)
│   ├── taas-format/        ← формат .taas (сохранение/загрузка)
│   └── connscope/          ← GUI-просмотрщик коннектомов
├── resources/examples/     ← примеры и данные
└── tests/                  ← 370+ автоматических тестов
```

## Быстрый старт

### Rust

```bash
git clone https://github.com/Taaslaby/taas_checkmate.git
cd taas_checkmate
cargo build --no-default-features

# тесты
cargo test --all-features
```

### Python (PyO3)

```bash
cargo build --features python
```

```python
import taas_checkmate as tc

# шаг симуляции 1000 нейронов Izhikevich
v, u, spikes = tc.step_izhikevich(v, u, I, dt=0.5)

# коннектом мухи FlyWire
conn = tc.ConnectomeData(rp, ci, vals, nt_types)
brain = tc.BrainRunner(conn, model="lif", dt_ms=0.5)
spikes = brain.run(n_steps=100)
```

### Рост сети из генома (prometheus-grow)

```bash
# вырастить арборы по моделям NETMORPH и вывести морфометрию
cargo run -p prometheus-grow --example netmorph_validation
```

## Производительность

| Компонент | Скорость |
|:---|:---:|
| Izhikevich (8K нейронов) | ~1M нейронов/сек |
| Осцилляторы T3 | 388× реального времени @ 48kHz |
| GPU-коннектом (CUDA) | 130K нейронов FlyWire в реальном времени |
| Growth engine | 8.9M клетко-шагов/с на 6 ядрах |

Подробности: [docs/PERFORMANCE_GROW.md](docs/PERFORMANCE_GROW.md),
эксперимент «из генома — в гамма-ритм»:
[docs/EXPERIMENT_GENOME_TO_GAMMA.md](docs/EXPERIMENT_GENOME_TO_GAMMA.md).

## Сборка и SIMD

Проект собирается с `-C target-cpu=native` (`.cargo/config.toml`) — включаются AVX2/FMA. Без этого `f32x8` молча эмулируется парой `f32x4`.

⚠️ Бинари получаются непереносимыми между машинами. Портабельная сборка:

```bash
RUSTFLAGS="-C target-cpu=x86-64" cargo build
```

Проверить, что AVX2 реально используется:

```bash
cargo rustc --release --lib -p pillars -- --emit=asm
# в target/release/deps/pillars-*.s ищем ymm-регистры и vfmadd
```

## Большие данные

Крупные датасеты (коннектомы, MNIST) не хранятся в репозитории — положите их вручную:

- `resources/examples/data/fafb.bin` — коннектом мухи FAFB
- `resources/examples/data/manc_viz.bin` — VNC MANC
- `resources/data/*.taas` — снапшоты сетей

## Документация

- [PYTHON_API_RU.md](resources/docs/PYTHON_API_RU.md) — справочник Python API
- [T3_ARCHITECTURE.md](resources/docs/T3_ARCHITECTURE.md) — архитектура ритмов
- [RL_API.md](resources/docs/RL_API.md) — обучение с подкреплением
- [UNIFIED_API.md](resources/docs/UNIFIED_API.md) — единая схема pillars-API

## Лицензия

GNU General Public License v3.0 — см. [LICENSE](LICENSE).
