# 🦀 Rust Examples — `taas-checkmate`

Rust примеры, которые компилируются через `cargo run --example <name>`.
Все примеры используют `taas-checkmate` как библиотеку (без Python).

---

## Примеры

| Пример | Описание | Запуск |
|:-------|:---------|:-------|
| `bench_connectome` | Бенчмарк коннектома FlyWire-scale: Izhikevich + CSR propagate на CPU (SIMD + rayon). 138K нейронов, ~5M синапсов, RT factor, scaling projection. | `cargo run --example bench_connectome --release` |
| `mnist_eprop` | MNIST классификация через E-Prop Izhikevich (Bellec et al. 2020). Per-synapse eligibility, Poisson encoding, feedback alignment. | `cargo run --example mnist_eprop --release` |
| `mnist_t3` | MNIST через T3 — HybridDecoder (InverseKuramoto + PST + NLMS). Zero-shot physics prediction + online NLMS learning. | `cargo run --example mnist_t3 --release` |
| `mnist_train` | MNIST через spiking Actor-Critic (contextual bandit). Dopamine-STDP, без backprop. | `cargo run --example mnist_train --release` |
| `learning_bench` | Бенчмарк скорости обучения Actor-Critic агента на простом MDP. | `cargo run --example learning_bench` |

## Данные

MNIST IDX файлы — `examples/data/`:

```bash
python3 examples/python/download_mnist.py
```

## Быстрый старт

```bash
cd taas_checkmate
python3 examples/python/download_mnist.py   # скачать данные
cargo run --example mnist_eprop --release    # запустить E-Prop MNIST
```

## Architecture

```
Input (784 px) ──→ Poisson ──→ Processing (Izhikevich/T3/RL) ──→ Readout ──→ Digit
```

Все примеры работают на **CPU-only** (SIMD f32x8 + rayon). Без GPU, без ASM, без FFI.
