# ⚙️ Core Engine

Базовые бенчмарки и демо движка — работают без дополнительных фич (кроме отмеченных).

| Пример | Описание | Запуск |
|:-------|:---------|:-------|
| `bench_lif` | LIF throughput benchmark, SIMD + rayon | `cargo run --example bench_lif --release` |
| `bench_connectome` | Izhikevich + CSR propagate, FlyWire-scale 138K | `cargo run --example bench_connectome --release` |
| `learning_bench` | Actor-Critic обучение на MDP | `cargo run --example learning_bench` |
| `hologram_bridge` | Position → phase → HH → STDP → motor pipeline | `cargo run --example hologram_bridge` |
| `bench_cuda_lif` | LIF на GPU | `cargo run --features cuda --example bench_cuda_lif --release` |
| `gpu_kuramoto_bench` | Kuramoto на GPU | `cargo run --features cuda --example gpu_kuramoto_bench --release` |
| `gpu_bptt_refine` | BPTT refinement на GPU | `cargo run --features cuda --example gpu_bptt_refine --release` |
| `rcm_vs_spectral` | RCM vs Spectral reordering comparison | `cargo run --features connectome --example rcm_vs_spectral --release` |
