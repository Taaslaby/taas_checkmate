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

## 🔴 High Priority

- [ ] **AdEx NEURON validation** — `adex_vs_neuron.rs` test
- [ ] **Port all models from `kernels` to `pillars`** — Izhikevich, LIF, AdEx, FHN, WC, SAW are duplicated
- [ ] **Remove `kernels/src/neuron/` legacy SIMD** after porting is complete
- [ ] **Remove `crates/core/src/neuron/`** — delegates to kernels, should delegate to pillars

## 🟡 Medium Priority

- [ ] **bench_rcm registration** — added to Cargo.toml with `required-features = ["connectome"]`
- [ ] **NEURON `.mod` files in repo** — `hcn.mod`, `km.mod` from tnmt/validate/ should be in pillars/tests/
- [ ] **HAAP model vs NEURON hh.mod** — validate HAAP channel kinetics (M-current, HCN)
- [ ] **GLIF3/4 vs Allen Institute reference** — validate against SONATA/NeuroML standard
- [ ] **Pillars vs kernels speed benchmark** — compare throughput on 138K neurons

## 🟢 Nice to Have

- [ ] **NeuroML/SONATA importer** — load models from Allen Institute / HBP
- [ ] **Spike Ring Buffer** — routing pillar for VNC delay lines
- [ ] **Pillars → Python bindings** — expose pillars models via PyO3
- [ ] **CI pipeline** — GitHub Actions with `cargo clippy --all-targets + cargo test --all`
- [ ] **Pillars benchmarks** — bench_lif, bench_hh, bench_izh against old kernels

## 🧪 NEURON Validation Status

| Model   | Test File                | NEURON Ref | Status |
|---------|--------------------------|------------|--------|
| HH      | `hh_vs_neuron.rs`        | 9.0.1      | ✅ 3 tests |
| Izhikevich | `izhikevich_vs_neuron.rs` | 2003 paper | ✅ 2 tests |
| LIF     | `lif_vs_neuron.rs`       | Empirical  | ✅ 2 tests |
| AdEx    | `adex_vs_neuron.rs`      | ❌ Missing  | ❌ |
| HAAP    | —                        | ❌ Missing  | ❌ |
| GLIF3/4 | —                        | ❌ Missing  | ❌ |
