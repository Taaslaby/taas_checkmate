# 🧠 `taas-checkmate` Examples

```
examples/
├── core/          — Core engine: LIF, GPU, Kuramoto, RCM, hologram
├── hh/            — Hodgkin-Huxley: real, self, viz, diag, calibrate
├── connectome/    — Connectome brain: bake, bench, biology, viz, T3
├── mnist/         — MNIST training: eprop, T3, train
├── python/        — Python scripts: psi resonance, fly tools, viz
│   ├── psi/       —   PSI resonance analysis
│   └── fly/       —   Fly connectome tools
├── fly/           — Fly assets: models, configs, simulation scripts
└── data/          — Baked connectome (`.taas`) + reference files
```

## Быстрый старт

```bash
# Connectome (requires connectome feature)
cargo run --features connectome --example bake_fafb --release

# HH (requires cuda feature)
cargo run --features cuda --example hh_self --release

# MNIST (requires train feature)
cargo run --features train --example mnist_eprop --release
```

> 💡 `--features all` включает всё сразу.
