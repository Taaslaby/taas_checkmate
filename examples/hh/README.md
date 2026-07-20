# ⚡ Hodgkin-Huxley

HH симуляции на GPU (CUDA LUT). Кроме `calibrate_hh` — он использует `taas-connectome`.

| Пример | Описание | Запуск |
|:-------|:---------|:-------|
| `hh_real` | Реалистичный HH с гетерогенными параметрами | `cargo run --features cuda --example hh_real --release` |
| `hh_self` | HH self-connection, автокорреляция | `cargo run --features cuda --example hh_self --release` |
| `hh_spike_diag` | HH spike diagnostics: ISI, bursts | `cargo run --features cuda --example hh_spike_diag --release` |
| `hh_viz` | HH visualization export | `cargo run --features cuda --example hh_viz --release` |
| `calibrate_hh` | HH parameter calibration на коннектоме | `cargo run --features connectome --example calibrate_hh --release` |
