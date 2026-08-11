# TAAS Checkmate — TODO.md

## ✅ Done

- [x] Pillars crate: 17 neuron models (LIF, HH, Izhikevich, AdEx, HAAP, FHN, WC, SAW, GLIF3/4, VSN, HR, MorrisLecar, Astrocyte, CaPlasticity, Exp2Syn, NMDA, TsodyksMarkram)
- [x] `validate_model!` compile-time checker for all models
- [x] Pillars Checker — borrow-checker-like bio-invariant validation
- [x] clippy: 0 errors, 0 warnings across all targets
- [x] NEURON 9.0.1 validation: HH (3 tests), Izhikevich (2 tests)
- [x] LIF empirical validation: forward Euler vs Python reference (2 tests)
- [x] `ohmic()` convention: `g·(E-V)` → `g·(V-E)` (NEURON standard)
- [x] `gate_integrate` removed from SimdLane trait
- [x] `PhantomData` / dead fields cleaned from SimdContext
- [x] **Перенос всех моделей из `kernels` в `pillars`** — Izhikevich, LIF, AdEx, FHN, WC, SAW и др. живут в `pillars::zoo::neurons` (Волна 12)
- [x] **Удаление легаси-крейтов** — `kernels/src/neuron/` и `crates/core/` полностью удалены, консолидировано в `pillars`
- [x] **bench_rcm** — зарегистрирован в `crates/pillars-connectome/Cargo.toml` (`[[bench]] name = "bench_rcm"`)
- [x] **Pillars → Python bindings** — фасад `taas_checkmate.pillars` (71 функция + 13 классов, проверяется `smoke_pillars.py`)
- [x] **CI pipeline** — `.github/workflows/ci.yml` (build + clippy `-D warnings` + test)
- [x] **Pillars benchmarks** — `bench_lif`, `bench_hh`, `bench_glif4_haap`, `bench_formula`, `bench_lif_par` в `crates/pillars/benches/`
- [x] **Spike Ring Buffer** — `kernels/delay.rs` (`delay_buffer_circular` ring buffer) + `api/delay.rs` (DelayBuffer) + VNC delay lines в `kernels/vnc.rs`
- [x] **NEURON `.mod` файлы в репозитории** — `hh_taas.mod`, `izh_taas.mod`, `lif_taas.mod`, `adex_taas.mod`, `fhn_taas.mod`, `ml_taas.mod` в `resources/validate/neuron_models/`

## 🔴 High Priority

- [ ] **AdEx NEURON validation** — `adex_vs_neuron.rs` тест (`.mod` файл уже в `resources/validate/neuron_models/adex_taas.mod`, не хватает Rust-теста)

## 🟡 Medium Priority

- [ ] **HAAP model vs NEURON** — validate HAAP channel kinetics (M-current, HCN); `haap_comparison.png` есть, но `hcn.mod` / `km.mod` (из tnmt/validate) в репозиторий не добавлены
- [ ] **GLIF3/4 vs Allen Institute reference** — validate against SONATA/NeuroML standard

## 🟢 Nice to Have

- [ ] **NeuroML/SONATA importer** — load models from Allen Institute / HBP

## 🧪 NEURON Validation Status

| Model   | Test File                | NEURON Ref | Status |
|---------|--------------------------|------------|--------|
| HH      | `hh_vs_neuron.rs`        | 9.0.1      | ✅ 3 tests |
| Izhikevich | `izhikevich_vs_neuron.rs` | 2003 paper | ✅ 2 tests |
| LIF     | `lif_vs_neuron.rs`       | Empirical  | ✅ 2 tests |
| AdEx    | `adex_vs_neuron.rs`      | ❌ Missing  | ❌ (`.mod` готов) |
| HAAP    | —                        | ❌ Missing  | ❌ |
| GLIF3/4 | —                        | ❌ Missing  | ❌ |
