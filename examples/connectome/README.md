# 🧬 Connectome Brain

Симуляция на реальных данных FlyWire, Princeton, Drosophila.

Требуют `--features connectome` или `--features cuda` (CUDA подразумевает connectome).

| Пример | Описание | Запуск |
|:-------|:---------|:-------|
| `bake_fafb` | Bake FlyWire connectome → `.taas` | `cargo run --features connectome --example bake_fafb --release` |
| `bake_fafb_dros` | Bake Drosophila connectome → `.taas` | `cargo run --features connectome --example bake_fafb_dros --release` |
| `fafb_nt` | Neurotransmitter analysis FA FB | `cargo run --features connectome --example fafb_nt --release` |
| `biology_check` | Biological plausibility: HH vs LIF | `cargo run --features connectome --example biology_check --release` |
| `bench_fly_brain` | Fly brain benchmark (CUDA) | `cargo run --features cuda --example bench_fly_brain --release` |
| `bench_fly_hh` | Fly brain HH benchmark (CUDA) | `cargo run --features cuda --example bench_fly_hh --release` |
| `bench_mouse_brain` | Mouse brain GPU benchmark | `cargo run --features cuda --example bench_mouse_brain --release` |
| `bench_princeton_connectome` | Princeton connectome benchmark | `cargo run --features cuda --example bench_princeton_connectome --release` |
| `viz_fly_brain` | Fly brain visualization | `cargo run --features cuda --example viz_fly_brain --release` |
| `t2_vision` | T2 vision processing | `cargo run --features cuda --example t2_vision --release` |
| `t3_bci_fly` | T3 BCI на fly connectome | `cargo run --features cuda --example t3_bci_fly --release` |
