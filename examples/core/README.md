# ⚙️ Core Engine

Демо движка — работают без дополнительных фич (кроме отмеченных).

| Пример | Описание | Запуск |
|:-------|:---------|:-------|
| `hologram_bridge` | Position → phase → HH → STDP → motor pipeline | `cargo run --example hologram_bridge` |
| `gpu_bptt_refine` | BPTT refinement на GPU | `cargo run --features cuda --example gpu_bptt_refine --release` |
