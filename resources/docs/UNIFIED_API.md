# Единый API: всё вокруг pillars

> Фаза 4 (финальная) плана миграции. Цель: **один публичный фасад** —
> `pillars::prelude::*`. `taas-kernels` и `taas-core` становятся тонкими
> адаптерами (или исчезают), `connectome`/`unity`/python ходят в pillars напрямую.
> Дата: 2026-08-08.

---

## 1. Целевая архитектура

```
pillars :: prelude::*   ← ЕДИНСТВЕННЫЙ ПУБЛИЧНЫЙ ФАСАД (уже полон)
  ├── api/     NeuralPopulation, Network, SparseGraph, builders, formula
  ├── zoo/     модели: LIF, Izh, HH, AdEx, GLIF, HAAP, FHN, SAW, WC, ML, HR, ...
  ├── core/    run_neuron, SimdLane, ComputeNeuron (движок)
  ├── kernels/ batch-микрокернелы: signal, decode, measure, phase, learning,
  │            input, stats, dopamine, delay, hh_channels, astrocyte, dendrite,
  │            depth_wave, synapse, fused, eprop, wc, vnc
  └── ops/     per-lane био-примитивы (ohmic, gate, plasticity, calcium)

taas-kernels  → compat-шим: safe.rs/io.rs/neuron/* реэкспортируют pillars
taas-core     → тонкие адаптеры: neuron/* обёртки на run_neuron,
                cpe/iit — высокоуровневые алгоритмы поверх pillars::kernels
taas-connectome → pillars напрямую (runner/sim), pillars-connectome::Brain
root (unity)  → pillars напрямую
python        → единый модуль `taas.pillars` на pillars::api + zoo (фаза 5)
```

**Правило:** ни один крейт, кроме pillars, не содержит реализации
математики нейронов/синапсов/декодеров. Всё, что реализует формулу, живёт
в pillars (`zoo`/`kernels`/`ops`/`core`).

**Пути доступа:** потребитель с прямой зависимостью на pillars зовёт
`pillars::kernels::*` напрямую; потребитель без неё (напр. t3) — через
compat-фасад `kernels::phase::*` (это тот же pillars-код, реэкспорт).

**Известные исключения (целевые для поздних траншей):** `core/cpe/mod.rs`
строит CSR-coupling скалярным нативным `sin()` (не pillars `fast_sincos`) —
оставлено как есть; кандидат на pillars-канон в транше 3d/4.

---

## 2. Текущее состояние (факт, 08.08)

### 2.1. Внешние вызовы `kernels::safe::*` — УДАЛЕНЫ (транши 3a+3b)

Все 7 живых внешних мостов переведены на pillars напрямую и удалены из
`safe.rs`:

| Мост (удалён) | Потребитель | Канон pillars |
|---|---|---|
| `gco_step` | connectome runner/sim | `pillars::kernels::phase::gco_step` |
| `hebbian_csr_step` | connectome runner | `pillars::kernels::learning::hebbian_csr_step` |
| `kuramoto_prepare` | core cpe | `pillars::kernels::phase::kuramoto_prepare` |
| `kuramoto_update` | core cpe | `pillars::kernels::phase::kuramoto_update` |
| `lif_step` | connectome runner, unity, core, бенчи | `run_neuron::<LIFModel>` |
| `izhikevich_step` | connectome runner, unity, core, бенчи | `run_neuron::<IzhikevichModel>` |
| `izhikevich_dual_step` | connectome runner | `run_neuron::<IzhikevichDualModel>` |

Оставшиеся функции safe.rs — либо вызываются только изнутри kernels
(тесты, другие мосты), либо уже являются однострочными `pillars::...`
реэкспортами (measure/phase/decode/signal/learning/dopamine/fused/eprop).

> **⚠ Субстеппинг izhikevich (важно для будущих портов):** легаси
> `kernels::neuron::izhikevich` (до 35356c1) использовал простой Forward
> Euler (один полный шаг `dt`). Канон `IzhikevichModel` в pillars — **2
> полушага 0.5·dt** (субстеппинг для численной стабильности, см. комментарий
> в `crates/pillars/src/zoo/neurons/izhikevich.rs`). Потребители уже были на
> каноне через мост (волна 2+), поэтому пользовательского поведения ничего
> не изменилось. Parity-тест `safe_vs_pillars.rs::ref_izh` воспроизводит
> КАНОН (не легаси) — это фиксация канона, а не гарантия legacy-идентичности.

### 2.2. Что уже готово

- `pillars::prelude::*` — полный фасад: api + zoo + core + math + kernels
- `kernels::neuron/*` — чистые compat-реэкспорты pillars (волна 8)
- `kernels::io.rs` — чистый compat на pillars (транш 2c)
- `kernels::{measure, phase, vnc}` — реэкспорты pillars (транш 2a)
- `core::neuron/{lif,izhikevich,soma_dendrite}.rs` — напрямую `run_neuron` (транш 3b)
- `kernels/src/safe.rs` — **УДАЛЁН** (транш 3e): lib.rs — чистый compat-реэкспорт pillars
- `core/*` — все вызовы на pillars напрямую, ноль `kernels::safe::*` (транш 3d)
- `connectome::sim.rs` — helpers `step_*_pillars` — **тонкие делегаты** в `pillars::api::neurons` (волна 12, дубль тел устранён); симуляции уже на `pillars-connectome::Brain`

### Волна 12 (финал): слияние connectome-крейтов

Крейты `taas-connectome` (легаси, 6053 строки) и `pillars-connectome` **слиты в один**
`pillars-connectome`. Публичный API легаси сохранён 1:1 — корневой `connectome::` теперь
`pillars_connectome::`:

| Было (taas-connectome) | Стало (pillars-connectome) |
|---|---|
| `ConnectomeData` (i32-CSR + NT + receptor_types + BrainIO + .taas) | `data::ConnectomeData` — канон (движок конвертит в `SparseGraph`) |
| `BrainRunner`, `SpikeRecorder`, `rcm::*` | те же модули, тот же API |
| `sim::{sim_lif, sim_izhikevich, sim_rate, sim_hh_*, sim_gco, …}` | `sim::*` — делегаты `crate::Brain`/`crate::SynapticBrain` (scale свёрнут в weights) |
| `hh_population_standard/pacemaking` | `sim::hh_population_*` (не тронуты) |
| `cortical.rs` (1210 строк) | **удалён** — 0 внешних потребителей |
| `pillars-connectome::ConnectomeData` (SparseGraph) | `Brain::from_sparse_graph` / `SynapticBrain::from_sparse_graph` (delays сохраняются) |

Воркспейс: 10 крейтов. `cargo build --all-targets` 0 warnings, `clippy -D warnings` зелёный.

---

## 3. План траншей

### Транш 3a — лёгкие мосты 1:1 → pillars напрямую
Перевести потребителей на `pillars::kernels::{phase, learning}` напрямую:
- `core/cpe/mod.rs`: `kernels::safe::kuramoto_prepare` → `pillars::kernels::phase::kuramoto_prepare` (+ update)
- `connectome/runner.rs` + `connectome/sim.rs`: `safe::gco_step` → `pillars::kernels::phase::gco_step`, `safe::hebbian_csr_step` → `pillars::kernels::learning::hebbian_csr_step`
- После перевода мосты `safe::{gco_step, hebbian_csr_step, kuramoto_prepare, kuramoto_update}` удалить (внутри kernels никто не зовёт).

### Транш 3b — нейронные шаги → run_neuron ✅
Связка `lif_step`/`izhikevich_step`/`izhikevich_dual_step` переведена:
- `connectome/sim.rs`: helpers `step_lif_pillars`/`step_izhikevich_pillars`/`step_izhikevich_dual_pillars` (run_neuron + перепаковка) — реэкспортированы из lib.rs.
- `connectome/runner.rs`, `connectome/sim.rs`, `unity.rs`: все вызовы на helpers.
- `unity.rs` (Unity C ABI): модели `lif`/`izhikevich`/`rate`/`gco`/`hh` — `rate`/`gco`/`hh` добавлены в волну Unity-харденинга; C#-мост: `resources/unity/TaasEngine.cs` (+ `BrainExampleBehaviour.cs`).
- `core/neuron/{izhikevich,lif,soma_dendrite}.rs`: прямые `run_neuron::<IzhikevichModel>`/`<LIFModel>` (soma_dendrite: layout params 5-col→6-col LIFModel).
- `benches/bench.rs`, `tests/memory_bench.rs`: на `run_neuron` (pack_n helpers).
- `safe::{lif_step, izhikevich_step, izhikevich_dual_step}` удалены.
- Parity: `safe_vs_pillars.rs` — `ref_lif`/`ref_izh` inline-эталоны (канон; izh — субстеппинг, см. §2.1).

### Транш 3c — safe.rs → compat-реэкспорт ✅
- 43 функции с 1:1 pillars-сигнатурами → `pub use pillars::...` (signal 14, decode 8, measure 7, phase 2, learning 4, dopamine 2, delay 2, wc 1, fused 3). Тела и assert'ы не дублируются — проверки живут в pillars-каноне.
- 19 **сигнатурных адаптеров** остаются (каждый помечен `(адаптер, 3c)`): переупаковка legacy-лейаутов (column-major ↔ SoA), `Option<I_dend>`-семантика eprop, fallback для нестандартных HH-лейаутов. Зовутся только из тестов kernels.
- Удалены мёртвые: `build_hh_channel_meta` (kernel_progon → pillars-путь `neuron::hh::build_channel_meta`) и `dendrite_step` (никто не звал).
- Итог: safe.rs **2603 → 1334 строк**. Внешних вызовов `kernels::safe::*` в crates/src/tests/benches нет (проверено grep'ом по всем 64 функциям).

### Транш 3d — core → тонкие адаптеры ✅
- `core/kuramoto.rs` — 14 вызовов → `pillars::kernels::{phase,decode,measure}` напрямую (`batch_coherence_sweep` — прямой вызов с выводом `n_areas`).
- `core/iit.rs` — `spike_entropy` → `pillars::kernels::measure`; `core/cpe/mod.rs` — удалён `pub use crate::kernels` (никто не звал).
- `core/neuron/{bcm,delay,stdp,wc}.rs` — прямые pillars-вызовы (`learning`/`delay`/`dopamine`/`wc`).
- eprop-вызовы (`izhikevich`, `lif`, `saw`, `adex`) → `pillars::kernels::eprop` напрямую (core всегда передавал `I_dend=None`, поэтому dendritic-прослойка не нужна).
- Адаптеры с repacking перенесены в core: `adex.rs` (9-col→12-col `AdExModel`), `glif.rs` (8→9 `GLIF3Model`, 10→11 `GLIF4Model`), `haap.rs` (17-col `HAAPModel`), `saw.rs` (8-col `SAWModel` + LCG-noise), `stp.rs` (SIMD-цикл поверх `pillars::ops::plasticity::stp`).
- Итог: в core **ноль** вызовов `kernels::safe::*` — только pillars (grep подтверждён).
- `core/Cargo.toml` — добавлен `wide = "0.7"` (для `f32x8` в stp).

### Транш 3e — финальная чистка ✅
- `kernels/src/safe.rs` **удалён целиком** (было 1334 строки). Все живые сигнатурные адаптеры перенесены в тесты/бенчи как локальные pillars-мосты:
  - `lifat_parity.rs` — локальная `lif_at_step`; `cuda/signal.rs` — `lif_at_pillars` (7-col → 8-col `LIFAtModel`); `bench.rs` — `lif_at_run`/`adex_run`/`saw_run`/`hh_poly_run`.
  - `bench_all.rs` — `fhn_run`/`stp_run`; `plasticity_parity.rs` — SIMD-цикл `stp_step`; `kernel_progon.rs` — `stp_step`/`hh_poly_step`/`fhn_step`/`adex_step`/`eprop_adex_step_multi`.
  - `python/neuron.rs` — `nmda_mg_block_step` → `pillars::zoo::synapses` напрямую; `safe_vs_pillars.rs` — `old_fused_hh` на raw-кернел `hh::fused_hh_poly_step` (настоящий легаси-референт vs `run_neuron::<HHModel>`).
- `kernels/src/lib.rs` — **чистый compat-реэкспорт pillars**: 9 групповых `pub use` (signal 14, decode 8, measure 7, phase 2, learning 4, dopamine 2, delay 2, wc 1, fused 3) + `neuron` (run_neuron, zoo-модели, hh) + `io`. `pub use safe::*` удалён.
- `neuron/mod.rs` — добавлены реэкспорты `run_neuron` + `{AdEx, FHN, GLIF3, GLIF4, HAAP, HH, LIFAt, LIF, MorrisLecar, SAW}Model` для тестов/бенчей.
- `measure_phase_parity.rs` — `safe::X` → прямые pillars-имена; `synapse_parity.rs` — delay-тест приведён к канону (clamp к [0, n-1], а не модулю).
- Итог: **0 warnings** (--all-targets), kernels 20/20 зелёных, core 83, connectome 53, t3 81, python ✓, unity 0.

### Фаза 5 (отдельно) — python на pillars ✅ (транши 5a-5c)

### Фаза 5 (отдельно) — python на pillars ✅ (транши 5a-5c)
Все 1:1 pillars-прокси переведены на прямые pillars-вызовы:
- `kuramoto.rs` — 12 из 14 функций → `pillars::kernels::{measure, decode}` (interference,
  phase_coherence, band_energy, kalman/adam/nlms_train, pst_decode, ring_synthesis и др.);
  `interference_mean`/`phase_coherence` → `&mut out[0]` (pillars-сигнатура `&mut f32`);
  `feat_dim` в pst_decode сохранён как compat-аргумент (Python-ABI не меняется).
  Оставшиеся `kuramoto_step_sparse`/`t3_step_sparse` — высокоуровневые CSR-шаги core
  (rayon + разреженная матрица; в pillars таких нет — документировано).
- `vnc.rs` — `kernels::vnc` → `pillars::kernels::vnc`.
- `hh_gpu.rs` — `kernels::neuron::hh` → `pillars::kernels::hh_channels`.
- `neuron.rs` — `stdp_update`/`stdp_homeostatic` → `pillars::kernels::learning`;
  `step_gco` → `pillars::kernels::phase::gco_step`; `SynapticKinetics` →
  `pillars::kernels::synapse`. repacking-адаптеры (step_izh/lif/adex/wc/haap/hr и
  stateful-классы EPropLIF/SornNetwork/Glif3/4/Astrocyte/Chordotonal/cpe) остаются на
  core — это законный тонкий pillars-слой (в pillars их нет 1:1).
- Бонус: `benches/bench_rcm.rs` починен (pre-existing: rand 0.9 API → 0.8
  gen_range/gen_bool, `receptor_types` поле, StdRng вместо SmallRng) — бенч
  компилировался только с connectome, был тихо сломан.
- `taas_checkmate.pillars` — единый фасад популяций (уже был); остальные подмодули —
  compat-поверхность. Итог: python --all-targets 0 warnings, ABI не изменён.

### Фаза 6 — stateful-классы core → pillars ✅ (транши 6a-6d)

Все stateful-классы перенесены в pillars-канон, core стал тонким compat-реэкспортом:
- **`pillars::api::neurons`** — новый канон (файл `crates/pillars/src/api/neurons.rs`):
  - Перенесены: `EPropLIF` (ядро `kernels::eprop::eprop_lif_step`), `Glif3State`/`Glif4State`
    (`run_neuron::<GLIF3/4Model>` + кэш packed-лейаута), `ShortTermPlasticity` +
    `compute_lfp` (`ops::plasticity::stp::stp_step`, SIMD+tail), `SomatoDendriticNeuron`
    (`run_neuron::<LIFModel>` + `kernels::dendrite`), `ChordotonalState`/`Profile`/`Builder`
    (`zoo::sensors::chordotonal_step`).
  - Реэкспорты kernels-канона: `PoissonGenerator` (input), `SornNetwork` +
    `intrinsic_plasticity`/`synaptic_normalization` (learning), `eprop_update`/`surrogate_gradient`
    (learning), `AstrocyteState` (astrocyte).
- **`pillars::api::cpe`** — ChaosPatternEngine перенесён целиком (config/coherence/dopamine/stdp/
  hybrid + `kuramoto_step` на `kernels::phase::{kuramoto_prepare, kuramoto_update}` + `CpeEngine`).
- **core — мёртвый слой по этим классам**: `neuron/{poisson,sorn,eprop,glif,stp,soma_dendrite,
  chordotonal,lif}.rs` → чистые `pub use pillars::api::neurons::*`; `cpe/*.rs` (5 файлов, 688 строк)
  удалены, `cpe/mod.rs` → единый реэкспорт. Python ABI сохранён (pyclass/pyfunction имена те же).
- Тесты переехали в pillars (api::neurons +10, api::cpe +14).
- Итог: pillars 0 warnings, **pillars lib 565 passed** (+25), core 55+5, kernels 20/20,
  connectome 53, t3 81, python ✓, --all-targets 0 warnings.

### Фаза 7 — оставшиеся stateful-классы → pillars ✅ (транши 7a-7d)

- **`pillars::api::neurons` расширен подмодулями** (канон, Фаза 7):
  - `api/neurons/eprop.rs` — `EPropIzhikevich`, `EPropIzhikevichMulti`, `EPropSaw`,
    `EPropAdExMulti` (ядра `kernels::eprop::eprop_{izhikevich,saw,adex}_step[_multi]`;
    rayon cfg-gated + serial fallback).
  - `api/neurons/haap.rs` — `HaapState`/`HaapProfile`/`HaapStateBuilder`
    (`run_neuron::<HAAPModel>`).
  - `api/neurons/atlasov.rs` — `step_atlasov_neuromotor` + `AtlasovState`/`AtlasovStateBuilder`/
    `BiomechanicsProfile`/`FiberPoolConfig`/`DrosophilaLegConfig` (`run_neuron::<AtlasovNeuromotorModel>`;
    чанк-обработка вынесена в `process_chunk`, joint-логика в zip-ветки, rayon cfg-gated).
  - Прямо в `neurons.rs`: `DopamineSTDP` (`kernels::dopamine`), `DelayBuffer`
    (push/read/read_multi, `kernels::delay`), реэкспорты `TtfsEncoder`/`bsa_*`/`phase_encode`
    (`kernels::input`) и `SpikeRaster`/`firing_rates`/`compute_isi`/`compute_cvs`
    (`kernels::stats`).
- **`pillars::api::trainer`** — `EPropTrainer` + `WorkerState` (Hogwild! батч-тренер;
  rand 0.8 добавлен в pillars Cargo.toml; rayon cfg-gated через `worker_batch`).
- **core — мёртвый слой по этим классам**: `neuron/{delay,encoding,stats,stdp,trainer,
  atlasov_neuromotor}.rs` → чистые реэкспорты pillars; `{haap,izhikevich,saw,adex}.rs` —
  repacking-адаптеры `step_*` остаются, stateful-классы → реэкспорт. Python ABI сохранён.
- Тесты переехали в pillars (+25).
- Итог: pillars 0 warnings (590 passed), core 36, kernels 20/20, connectome/t3 ✓,
  --all-targets 0 warnings, python ✓, no-default-features компилируется.

### Фаза 8 — последние живые модули core → pillars ✅ (транши 8a-8d)

- **`pillars::api::autotune`** — `EPropTunerConfig`/`FrProbePoint`/`probe_fr`/`suggest_config`
  (SNR-тюнер поверх `api::neurons::eprop::EPropIzhikevichMulti`, rand StdRng).
- **`pillars::api::neurons::bcm`** — `BcmSynapse`/`HomeostaticScale`
  (`kernels::learning::{bcm_step, homeostatic_scale}`).
- **`pillars::api::neurons::hr`** — `step_hr` (repacking `run_neuron::<HRModel>`,
  rayon cfg-gated через `process_chunk` + serial-fallback).
- **`pillars::api::embodied`** — `FlyEmbodiedState`/`JointMapping`/`FlyPose`/`SimStepOutput`
  (NMJ `DelayBuffer` + `AtlasovState` + joint-динамика; serde добавлен в pillars).
- **core — мёртвый слой**: `neuron/{autotune,bcm,hr,embodied}.rs` → чистые реэкспорты
  pillars (по 1-4 строки); тесты переехали в pillars (+13). Python ABI сохранён
  (`probe_fr`/`suggest_config` через `core::neuron::autotune`, `step_hr` — `core::neuron::hr`).
- Итог: pillars **603 passed** (+13), core 26, kernels 20/20, connectome 53, t3 81,
  --all-targets 0 warnings, python ✓, no-default-features ✓ (3 pre-existing runner.rs).

### Фаза 9 — python-биндинги на pillars напрямую + единый фасад ✅ (транши 9a-9f)

- **step_* → pillars::api::neurons**: 9 repacking-адаптеров из core перенесены
  (подмодули `lif`/`izhikevich`/`saw`/`adex`/`fhn`/`wc`, `step_haap` в `haap.rs`) —
  rayon cfg-gated через `process_chunk` (паттерн hr/atlasov). `core::neuron` —
  полностью мёртвый слой (все файлы — чистые реэкспорты).
- **core::{kuramoto, sparse} → pillars**: `kuramoto_step_sparse/dense`,
  `t3_step_sparse` → `pillars::kernels::phase`; CSR-утилиты
  (propagate/hamiltonian/gco_full/rate_step/stdp_ltd) → новый
  `pillars::kernels::csr`. `core::{kuramoto, sparse}` — реэкспорты
  (+2 slice-адаптера interference_mean/phase_coherence для тестов).
- **python-биндинги → pillars напрямую**: все `crate::core::*` импорты заменены
  на `pillars::api::neurons` / `pillars::api::{autotune, trainer, cpe}` /
  `pillars::kernels::{phase, csr, decode, measure}` (11 файлов).
- **`taas_checkmate.pillars` — единый фасад**: `pillars::register` возвращает
  модуль; в него добавлены все функции/классы подмодулей (neurons, kuramoto,
  snn, eprop, autotune, sparse, io, cpe, vnc, lif_connectome + EPropLIF,
  SornNetwork, TtfsEncoder, PoisssonGenerator, ShortTermPlasticity, GLIF3/4,
  AstrocyteState, EPropTrainer, ActorCriticAgent…). `from
  taas_checkmate.pillars import *` покрывает всё.
- **src/lib.rs**: корневые реэкспорты — напрямую из pillars::api/kernels;
  `core::{neuron, sparse, kuramoto}` оставлены как legacy-алиасы для тестов/бенчей.
- Итог: pillars **615 passed** (+12), core 23, kernels 20/20, root-тесты
  (neuron 82, kuramoto 7, sparse 10, izh 7), connectome 53, t3 81,
  --all-targets 0 warnings, python ✓, no-default-features ✓.

### Фаза 10 — крейт taas-core удалён из workspace ✅

- **iit.rs** (1066 строк, единственный живой код в core) перенесён в корень
  `src/iit.rs` (самодостаточен, `pub mod iit`).
- **src/lib.rs**: `pub use taas_core as core` удалён; legacy-модули
  `neuron`/`sparse`/`kuramoto` — inline-реэкспорты pillars
  (`pillars::api::neurons::*`, `pillars::kernels::csr::*`,
  `pillars::kernels::{phase,decode,measure}` + 2 slice-адаптера
  interference_mean/phase_coherence для тестов).
- **train**: `use pillars::api::neurons as neuron` (rl.rs использует
  PoissonGenerator/step_lif), dep `taas-core` удалён, добавлен `pillars`.
- **connectome**: dep `taas-core` удалён (использовался только в комментарии).
- **crates/core/** удалён (34 файла), workspace members обновлён.
- Итог: **0 warnings** (--all-targets), pillars 615, train ✓, connectome 53,
  t3 81, root-тесты (neuron 82, kuramoto 7, sparse 10, izh 7), python ✓,
  unity ✓, no-default-features ✓. Рабочий слой: pillars + kernels + train +
  t3 + connectome + cuda. **core-зависимостей больше нет.**

### Фаза 11 — крейт `taas-kernels` удалён, compat-поверхность в pillars ✅

- **Реальные тела перенесены в pillars**: 11 min-truncating io-адаптеров
  (stochastic_encoder, t3_drive_encoder, t3_complex_decoder, t3_pst_features,
  t3_tact_generator, t3_inverse_kuramoto, echo_tap, thalamic_gate,
  sigmoid_gate, phase_gate_apply, hebbian_update) → **`pillars::kernels::io`**;
  `fast_sincos_simd`/`wrap_phase`/`clamp_amp` — уже были в
  `pillars::kernels` (дубликаты удалены вместе с крейтом).
- **Flat compat-поверхность** — верхний уровень `pillars::kernels`:
  `t3_update`, `nlms_train`, `kalman_update`, `stdp_update`, `wc_step`,
  fused-семейство, signal/decode/measure/dopamine/delay-имена без префикса
  подмодуля + `fast_exp`/`fast_exp_simd` из `pillars::math`.
- **Потребители переведены напрямую**: t3/train — `pub use
  pillars::kernels as kernels` (было `taas_kernels`); cuda-тест —
  `pillars::core::runner::run_neuron::<pillars::zoo::neurons::LIFAtModel>`;
  root `kernels` — inline-модуль (glob + `neuron::hh` → `hh_channels`);
  benches — `pillars::kernels::hh_channels`.
- **18 parity-тестов переехали** из `kernels/tests` в `pillars/tests`
  (импорты переписаны на pillars-пути; kernel_progon — без glob).
- **crates/kernels/** удалён (999 строк), workspace members + deps (root,
  t3, train, cuda, connectome) вычищены, CI: `cargo test -p pillars --test
  kernel_progon`.
- Root `kernels` — inline-модуль с **явным** списком имён (без glob): новые
  имена в pillars не протекают в compat-поверхность автоматически.
- ⚠️ cuda-тест (`signal.rs`, LIF-AT мост через `run_neuron`) переведён на
  pillars, но требует nvcc — проверить на GPU-машине.
- Итог: **0 warnings** (--all-targets), pillars 615 lib + 18 parity-бинарников
  (~160 тестов) ✓, t3 81, connectome 53, root-тесты (neuron 82, memory_bench
  2, kuramoto 7, sparse 10, izh 7, io 12) ✓, python ✓, no-default-features ✓
  (3 pre-existing). **Workspace: pillars + pillars-* + t3 + cuda + connectome
  + train + taas-format — весь канон в pillars, мёртвых крейтов нет.**

---

## 4. Критерий завершения

1. `grep -rn 'safe::' crates src tests benches` → только внутри `kernels/src/safe.rs` (или пусто).
2. `grep -rn 'kernels::' crates/core crates/connectome src/unity.rs` → только pillars-пути.
3. `cargo build` — 0 warnings; все тесты зелёные (kernels, pillars, core, connectome, root).
4. `pillars::prelude::*` — единственный импорт, нужный новому коду.
