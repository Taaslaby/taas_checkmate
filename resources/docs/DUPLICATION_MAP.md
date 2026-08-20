# Карта дублирования: `taas-core` × `taas-kernels` × `pillars`

> ⚠️ **ИСТОРИЧЕСКИЙ ДОКУМЕНТ (архив)** — миграция завершена.
> В **Фазе 10** (2026-08-08) крейт `taas-core` удалён из workspace (34 файла),
> `iit.rs` переехал в корень (`taas_checkmate::iit`). В **Фазе 11** (2026-08-08)
> удалён и крейт `taas-kernels` (999 строк): io-адаптеры → `pillars::kernels::io`,
> flat compat-поверхность → верхний уровень `pillars::kernels`, 18 parity-тестов
> → `pillars/tests`. В **Волне 12** (2026-08-08) легаси `taas-connectome` слит в
> `pillars-connectome` (1 крейт): дубли ядер `step_*_pillars` стали тонкими
> делегатами в `pillars::api::neurons`, `cortical.rs` (мёртвый) удалён,
> `ConnectomeData` (i32-CSR) — единый канон. Единая точка входа — **pillars**;
> актуальная схема — в `UNIFIED_API.md`.

> Дата: 2026-08-03 · Аудит по факту grep по воркспейсу
> Вопрос: **какой крейт становится единственным источником SIMD-нейронов?**

## 0. Консолидация крейтов (13 → 9)

Мелкие крейты-спутники свернуты в родительские (2026-08-03):

| Было | Стало | Обоснование |
|------|-------|-------------|
| `taas-embodied` (228 строк) | `core::neuron::embodied` | 0 потребителей, мёртвый код — встроен рядом с `AtlasovState` |
| `cpe` (690 строк) | `core::cpe` | единственный потребитель — python-биндинги |
| `iit` (1 065 строк) | `core::iit` | единственный потребитель — python-биндинги + реэкспорт из корня |
| `karmac` (3 074 строки) | корневой `src/karmac/` (feature `karmac`) | единственный потребитель — python-биндинги (spaun) |

После сворачивания воркспейс: **pillars, pillars-macros, kernels, t3, cuda,
connectome, core, train, taas-format** + корневой крейт. Bенч `bench_lod_gnw`
и example `spaun_pipeline` перенесены в корень (`benches/`, `examples/`).

**Ответ сразу: `pillars`.** Он уже стоит внизу графа зависимостей (никто из трёх
на него не влияет — `pillars` зависит только от `pillars-macros`), владеет
типизированным API (`ComputeNeuron` + `#[derive(NeuronModel)]`), безопасным
`NeuralPopulation` и полным набором ops/math. `taas-kernels` и `taas-core`
должны стать тонкими прослойками-адаптерами над ним.

---

## 1. Граф зависимостей (факт)

```
pillars ──→ pillars-macros            (низ, ни от кого не зависит)
taas-kernels ──→ pillars              (средний слой, уже зависит от pillars!)
taas-core ──→ pillars, taas-kernels   (верх)
root ──→ всё вышеперечисленное
```

Ключевой факт: **`taas-kernels` уже зависит от `pillars`**. Значит дублирующие
реализации внутри kernels — не «параллельный движок», а легаси-копии, которые
можно заменять ре-экспортами без изменения направления зависимостей.

## 2. Количественная картина (строки кода)

| Крейт | Суммарно | Крупнейшие файлы |
|------:|---------:|------------------|
| `pillars` | **20 557** | api/network.rs 738, api/formula.rs 704, kernels/dendrite/cable.rs 679, zoo/neurons/hh.rs 631 |
| `kernels` | **~16 900** (после Фазы 3: −1 124) | safe.rs 2 901 (обёртки pillars), neuron/hh.rs 1 197, neuron/izhikevich.rs 844 (было 1 394), neuron/lif.rs 447 (было 868), neuron/adex.rs 213 (было 366) |
| `core` | **6 838** | neuron/atlasov_neuromotor.rs 824, neuron/trainer.rs 529, neuron/haap.rs 494, neuron/autotune.rs 486 |

## 3. Матрица дублирования

### 3.1 Математические примитивы

| Примитив | `kernels` | `pillars` | Комментарий |
|----------|-----------|-----------|-------------|
| `fast_exp` (скаляр) | `src/neuron/mod.rs:126` | `src/math/fast_exp.rs:15` | Коэффициенты c1..c6 в Horner, НО это **две независимые копии** полинома — бит-идентичность НЕ гарантирована, нужен парити-тест перед ре-экспортом |
| `fast_exp_simd` (f32x8) | `src/neuron/mod.rs:155` | `src/math/fast_exp.rs:72` + `fast_exp_poly_simd:87` | Дублирование полинома |
| `fast_exp` как метод лейна | — | `core/lane.rs:104`, `lane_f32x8.rs:33`, `lane_f32.rs:32` | Третья копия в виде трейта |
| `sigmoid` | `src/signal.rs:13` + инлайн в wc/synapse/dendrite/haap | `math/sigmoid.rs`, `kernels/signal.rs:19`, `ops/gate.rs` | 3–4 независимые копии |
| `safety_clamp` | **`src/safe.rs:18` + `src/io.rs:11`** | `src/kernels/signal.rs:19` | **3 CPU-копии одной функции** + 2 в cuda (`cuda/src/signal.rs:276`, `context.rs:795`) |
| `wrap_phase` / `clamp_amp` | `src/lib.rs` | `src/kernels/mod.rs` | Полные копии |
| `vtrap` | ~~в `neuron/hh.rs`~~ (**удалён 07.08, транш 1e**) | `math/` + `kernels/hh_channels.rs:63` | ✅ единый канон |

### 3.2 Шаги нейронов (batch SIMD) — главный дубль

`taas-kernels/src/safe.rs` (2 623 строки) содержит batch-функции `lif_step`,
`izhikevich_step`, `saw_step`, `adex_step`, `hh_poly_step`, `fused_hh_poly_step`,
`fhn_step`, `wc_step`… — это ровно та роль, которую в pillars играют
`core/population.rs` + `core/runner.rs` + `zoo/neurons/*`.

| Модель | `kernels` (легаси SIMD) | `pillars` (канон) |
|--------|-------------------------|-------------------|
| LIF | `neuron/lif.rs` (**удалён 07.08, волна 6** — `lif_at_step` → pillars; файл целиком) | `zoo/neurons/lif.rs` + **`kernels/eprop.rs` (✅ 07.08, e-prop-канон)** |
| Izhikevich | `neuron/izhikevich.rs` (**удалён 07.08, волна 5** — `eprop_izhikevich_step(_multi)`/`hebbian_csr_step` → pillars) | `zoo/neurons/izhikevich.rs` + **`zoo/neurons/izhikevich_dual.rs` (✅ dual-канон)** + **`kernels/eprop.rs` (✅ e-prop-канон)** + **`kernels/learning.rs::hebbian_csr_step` (✅)** |
| HH | `neuron/hh.rs` (**удалён 07.08, транш 1e — целиком**; движок `ChannelMeta`/`hh_poly_step`/`fused_hh_poly_step` → pillars) | `zoo/neurons/hh.rs` + `kernels/hh_channels.rs` (✅ канон: spec-слой + полиномный движок; `neuron::hh` — compat-реэкспорт; parity `hh_parity.rs` 11/11) |
| AdEx | `neuron/adex.rs` (**удалён 07.08, волна 5** — `eprop_adex_step_multi` → pillars) | `zoo/neurons/adex.rs` + **`kernels/eprop.rs` (✅ e-prop-канон)** |
| GLIF | `neuron/glif.rs` (**удалён 08.08, волна 7** — мост на `GLIF3Model`/`GLIF4Model`) | `zoo/neurons/glif3.rs`, `glif4.rs` (✅ канон; Forward Euler vs легаси exp) |
| HAAP | `neuron/haap.rs` (**удалён 08.08, волна 7** — мост на `HAAPModel`) | `zoo/neurons/haap.rs` (✅ канон; Forward Euler vs легаси exp) |
| HR / ML | `neuron/hr.rs`, `neuron/morris_lecar.rs` (**удалены 08.08, волна 7** — HR 9-col мост; ML interleaved 16/нейрон → 14-col) | `zoo/neurons/hr.rs`, `morris_lecar.rs` (✅ канон; ML edge-спайки vs легаси level) |
| SAW | `neuron/saw.rs` (**удалён 08.08, волна 7** — шум на границе safe-моста) | `zoo/neurons/saw.rs` + **`kernels/eprop.rs` (✅ e-prop-канон)** |
| **Fused-слои** (hidden Izh, output LIF, poisson-fused) | `neuron/fused.rs` (**удалён 07.08**, −426 строк) | **`kernels/fused.rs`** (✅ 07.08, канон; hot path `train/rl.rs` идёт через него) | `fused_hh_poly_step` уже мостился на pillars HHModel для standard-лейаута (легаси только для кастомных channel_meta + бенчи); `fused_hidden_izhikevich_step`/`fused_output_lif_step`/`poisson_fused_hidden_step` портированы бит-в-бит, parity — `kernels/tests/fused_parity.rs` (нюанс `wide::mul_add` зависит от FMA целевой CPU — разобрано в «Волне 2»; кернелы 22.08 переведены на явное `* +`) |
| WC | `neuron/wc.rs` (**удалён 08.08, волна 7**) | **`kernels/wc.rs` (✅ 08.08, бит-в-бит порт; gain-сигмоида `exp(−βx)`) + `zoo/neurons/wc.rs` (width-сигмоида — задокументированное расхождение zoo-варианта)** |
| FHN | `neuron/fhn.rs` (**удалён 08.08, волна 7**) | `zoo/neurons/fhn.rs` (✅ канон; edge-спайки) |
| Atlasov | `neuron/atlasov_neuromotor.rs` (**удалён 08.08, волна 7** — мост на zoo-модель) | `zoo/neurons/atlasov_neuromotor.rs` (✅ канон; f_l гауссиана vs легаси-парабола) |

### 3.3 Синапсы, дендриты, глия, нейрохимия

| Домен | `kernels` | `pillars` |
|-------|-----------|-----------|
| 7 типов рецепторов | `neuron/synapse.rs` (**удалён 08.08, волна 8, транш 1c** — мост на pillars) | `kernels/synapse.rs` (548, ✅ канон) + `zoo/synapses/exp2syn.rs` |
| Кабельные дендриты | `neuron/dendrite.rs` (1 302) | `kernels/dendrite/cable.rs` (679) |
| Tsodyks–Markram / STP | `neuron/stp.rs` | `zoo/synapses/tsodyks_markram.rs` + `ops/plasticity/` |
| NMDA / TM-модель / delay | `neuron/nmda.rs`, `neuron/tsodyks_markram.rs`, `neuron/delay.rs` (**удалены 08.08, волна 8, транш 1a** — мост на pillars) | `zoo/synapses/nmda.rs`, `zoo/synapses/tsodyks_markram.rs`, `kernels/delay.rs` (✅ канон; tsodyks: Exp Euler vs легаси Forward Euler — задокументировано; nmda: `SimdLane::exp` vs легаси всегда-полином) |
| Дофамин | `neuron/dopamine.rs` (**удалён 08.08, волна 8, транш 1b** — мост на pillars) | `kernels/dopamine.rs` (✅ канон; tail: fast_exp vs легаси нативный exp — задокументировано) |
| BCM / STDP / обучение | `neuron/bcm.rs`, `neuron/stp.rs` (**удалены 08.08, волна 8, транш 1b** — мост на pillars) | `kernels/learning.rs` (✅ канон; `stdp_homeostatic` портирован бит-в-бит), `ops/plasticity/stdp.rs`, `ops/plasticity/stp.rs` (✅ lane-канон; stp мост: Exp Euler + post-facilitation u + clamp [0,1] vs легаси Euler + pre-spike u — задокументировано) |
| Астроцит | `neuron/astrocyte.rs` (**удалён 08.08, волна 8, транш 1c**) ← `core/neuron/astrocyte.rs` (реэкспорт pillars) | `kernels/astrocyte.rs` (✅ канон — stateful-порт легаси, бит-в-бит; `zoo/glia/astrocyte.rs` — lane-модель `#[no_spike]`) | Две абстракции: stateful (kernels, для VNC/python-обёртки) и lane (zoo, NeuronModel). Легаси-дубль удалён; `core` — реэкспорт pillars. Convenience-метод `SynapticKinetics::apply_astrocyte_gain` (legacy-only, без потребителей) не портирован — канон: `AstrocyteState::apply_to_kinetics` |
| `signal.rs`, `measure.rs`, `decode.rs`, `phase.rs`, `vnc.rs`, `ion.rs` | `src/*.rs` | `kernels/signal.rs`, `measure.rs`, `decode.rs`, `phase.rs`, `vnc.rs` | |

### 3.4 Внутреннее дублирование в самом pillars

| Легаси-путь | Современный путь | Комментарий |
|-------------|------------------|-------------|
| `core/context_ops.rs` (431 стр., `lif_step` на стр. 81, compose-машинерия) | `core/compute.rs` + `core/population.rs` (ComputeNeuron) | Две параллельные парадигмы шага живут в одном крейте; `context_ops` — кандидат на удаление после полного перехода zoo на `#[derive(NeuronModel)]` |

### 3.5 «Интеллект» — то, что НЕ надо трогать

`taas-core` уникален в алгоритмах: `trainer.rs`, `autotune.rs`, `stats.rs`,
`encoding.rs`, `poisson.rs`, `soma_dendrite.rs`, `eprop.rs`, `sorn.rs`.
Это не SIMD-нейроны, а обучение/анализ — им место в core (или отдельном крейте),
дублей нет.

---

## 4. Что уже сделано (факты из кода)

- **Волна 7 (08.08): «zero legacy» — все batch-кернелы нейронов удалены из `kernels/neuron/*`; канон — pillars.**
  - **Портирован бит-в-бит:** `wc_step` → **`pillars::kernels::wc`** (новый модуль; gain-сигмоида `1/(1+exp(−βx))`, SIMD-кламп ±10 + `fast_exp_poly_simd`, tail `f32::exp` без клампа — задокументированное расхождение SIMD/tail сохранено). Parity: `crates/kernels/tests/wc_parity.rs` — **2/2 бит-в-бит** (n=8 SIMD + n=13 с хвостом).
  - **Мосты safe → zoo через `run_neuron`:** `fhn_step` → `FHNModel` (5-col, порог 0.5, edge-спайки); `haap_step` → `HAAPModel` (17-col, inv_cm/inv_tau); `glif3_step` → `GLIF3Model` (9-col), `glif4_step` → `GLIF4Model` (11-col, inv_tau_ahp/inv_tau_theta + tau_ref колонкой); `morris_lecar_step` → `MorrisLecarModel` (легаси-лейаут параметров — **interleaved 16/нейрон**, читаем по stride 16; edge-спайки через was_below); `saw_step` шумный путь — детерминированный `SAWModel` + пост-инъекция `noise_amp·(rand−0.5)` тем же LCG на границе (легаси-кернел удалён).
  - **core мигрирован напрямую:** `core::step_hr` → `run_neuron::<HRModel>` (9-col, v_thresh=1.0); `core::step_atlasov_neuromotor` → `run_neuron::<AtlasovNeuromotorModel>` (ca_decay_const=`exp(−dt/τ_pump)`, inv_vmax=`1/v_max`, inv_width=`1/(l0·0.5)`, длина/скорость — в слоты состояний 3/4; f_l гауссиана — задокументировано).
  - **Удалены файлы:** `neuron/{fhn,glif,haap,hr,morris_lecar,saw,wc,atlasov_neuromotor}.rs` (≈3 000 строк) + parity-тест `fhn_parity.rs` (легаси-эталон удалён вместе с кернелом). LCG-шум перенесён из `neuron/mod.rs` в `safe.rs`. Заголовки `gco_parity.rs`/`safe_vs_pillars.rs` приведены к реальности (обёртка vs прямой pillars-вызов). **0 warnings.**
  - В `kernels/neuron/*` остались только синапсы/пластичность/сенсорика/каналы HH (их канон живёт здесь — pillars-аналога нет).
- **Волна 6 (07.08): мост `safe::lif_at_step` → `run_neuron::<LIFAtModel>` — последняя живая легаси-обёртка с готовым pillars-каноном закрыта.**
  - `safe::lif_at_step` (единственная оставшаяся легаси-обёртка; канон LIFAtModel существует с волны 1) переведён на `run_neuron::<LIFAtModel>` по образцу `safe::lif_step` (Фаза 2): легаси 7-col `[dt, τ_m, v_rest, v_thresh, v_reset, τ_θ, β]` → pillars 8-col `[dt, inv_τ_m, v_rest, v_thresh, v_reset, inv_τ_θ, β, τ_ref]` через thread-local scratch; v→slot 0, θ→slot 1, refr→slot 2; `τ_m ≤ 0` → inv=0 (без интеграции), `τ_θ ≤ 0` → inv=1e9 (θ→0, легаси-SIMD-семантика); I-clamp ±100 и скалярный хвост n%8 — внутри `run_neuron_ext`.
  - Parity: `crates/kernels/tests/lifat_parity.rs` переписан — **бит-в-бит** мост ≡ прямой `run_neuron::<LIFAtModel>` (n=8 SIMD-путь + n=13 с хвостом) + семантика vs скалярный эталон легаси (θ в пределах 1e-3 rel — fast_exp vs точный exp, число спайков равно, refr в пределах одного спайкового шага). Тест намеренно использует τ_θ > 0: при τ_θ ≤ 0 легаси-SIMD давал θ→0, легаси-tail оставлял θ без изменений — мост через `inv=1e9` даёт θ→0 в обоих путях (расхождение унаследовано от волны 1, см. `core::step_lif_at`).
  - Легаси удалён: **`neuron/lif.rs` — файл целиком** (`lif_at_step` + тесты, ~294 строки); `pub mod lif` и doc-строка убраны из `neuron/mod.rs`. **0 warnings.**
  - Потребители без правок: `bench.rs` `bench_lif_at` (n=1/n=8), `cuda/signal.rs` CPU-референс, `tests/neuron_tests.rs::test_lif_at_step` (через `core::step_lif_at`).
- **Волна 5 (07.08): e-prop-семейство + Hebbian CSR портированы — последние живые batch-кернелы без pillars-канона.**
  - **`pillars::kernels::eprop`** (новый модуль, safe API + SIMD f32x8) — бит-в-бит порт 5 функций: `eprop_lif_step`, `eprop_izhikevich_step`, `eprop_izhikevich_step_multi`, `eprop_adex_step_multi`, `eprop_saw_step`. Сохранена вся легаси-семантика: clamp I ±100, clamp exp-аргумента ±10 с маской `v > v_th−20`, NaN-спайк в AdEx, заморозка eligibility/ψ в рефрактерности LIF, ψ=0 в рефрактерности (SIMD-путь), floor diff 1e-6 (только LIF), 10× weight-конвенция multi-версий, полиномная `fast_exp_poly_simd` в SIMD-пути vs `f32::exp` в tail.
  - **`hebbian_csr_step` → `pillars::kernels::learning`** (CSR-пластичность `dW = η·m_pre·m_post − α·W·dt`, clamp [−1,1]) — бит-в-бит; живые потребители `connectome/sim.rs` + `connectome/runner.rs` работают без правок.
  - safe-обёртки (`eprop_lif_step`, `eprop_izhikevich_step(_multi)`, `eprop_adex_step_multi`, `eprop_saw_step`, `hebbian_csr_step`, а также `gco_step` → `pillars::kernels::phase::gco_step`) стали тонкими делегатами на pillars.
  - Parity: `crates/kernels/tests/eprop_parity.rs` — **7/7 бит-в-бит** vs скалярные эталоны (SIMD-путь n=8 + гибридный n=13 для tail: 0..7 exp_poly, 8..12 f32::exp — задокументированное расхождение SIMD/tail легаси).
  - Легаси удалён: **`neuron/izhikevich.rs` и `neuron/adex.rs` — файлы целиком удалены**; из `lif.rs` убран `eprop_lif_step` (+тест), из `saw.rs` — `eprop_saw_step`/`gco_step`; мёртвый `neuron/mod.rs::surrogate_gradient` удалён. **0 warnings.**
  - Живые проверки: корневые `test_eprop_*` **34/34** (вкл. `test_eprop_multi_learns_xor`, `test_eprop_learns_and` — обучение идёт через pillars), golden-eprop 4/4, kernels 121/121 + интеграционные 84/84, connectome 53+4 ✓.
  - **Не портированы (осознанно, на момент волны 5):** `wc_step`, `glif3/4_step`, `haap_step`, шумный `saw_step` — **все закрыты волной 7** (см. ниже): WC портирован бит-в-бит в `pillars::kernels::wc`, GLIF/HAAP переведены на zoo-модели (Forward Euler — задокументированная смена интеграции), SAW-шум вынесен на границу safe-моста.
- **Волна 4 (07.08): легаси LUT-HH удалён — канонический LUT — pillars `TriangLUT`.**
  - Легаси `GateLut`/`build_luts`/`v_to_idx`/`hh_lut_step`(+scalar/simd) из `neuron/hh.rs` **удалены** (−283 строки: 1 323 → 1 040). Их единственным потребителем был `bench_hh_lut` в `benches/bench.rs`.
  - `bench_hh_lut` **переведён на pillars-канон**: `LutBuilder::new(config).build::<HHModel>` (калибровка 2M шагов, 4096×8B = 32KB L1) + `TriangulatedModel`-популяция (n=1/8/1024). Тот же LEVEL 4-паттерн, что в `pillars/benches/bench_hh.rs`.
  - `ChannelSpec`/`GateSpec`/`RateFn`/`standard_channels`/`steady_state_gates` остаются (их используют `build_channel_meta`-мост в safe.rs и тесты `kernel_progon`/`safe_vs_pillars`).
  - Обновлены: таблица моделей в `neuron/mod.rs` (строка `hh_lut_step` убрана) и doc-header hh.rs (указатель на `ops::triangulated`).
- **Волна 3 перехода (07.08): `izhikevich_dual_step` портирован — последний neuron-кернел без pillars-канона.**
  - **`IzhikevichDualModel`** (Sun et al. 2026) в `zoo/neurons/izhikevich_dual.rs` (`#[dual_neuron]`, 18-я модель зоопарка; состояния v/u/refr/trace/m, 10 параметров). Ключевое: **single-step Forward Euler** (НЕ substepping, в отличие от pillars IzhikevichModel) — точная легаси-семантика dual; `m` обновляется всегда (1.0 на спайке / 0.5 иначе, clamp [0,1]), trace заморожен в рефрактерности, модуляция `I += w_m·m` после клампа ±100.
  - safe-обёртка `izhikevich_dual_step` → pillars; живые потребители `connectome/sim.rs` + `connectome/runner.rs` работают без правок (57 тестов connectome ✓).
  - Parity: `kernels/tests/izhikevich_dual_parity.rs` — бит-в-бит vs скалярный эталон. Зафиксировано **внутреннее расхождение легаси SIMD/tail для trace**: SIMD `tr·td + (1−tr)` (при tr=0 → 1.0) vs tail `tr += td·(1−tr)` (→ 0.5) — канон следует SIMD.
  - Легаси-кернел + 3 его теста **удалены** из `neuron/izhikevich.rs` (−340 строк, было 844 → 504). Зарегистрирован в pillars-cuda (`IZHIKEVICH_DUAL_CUDA_SOURCE`, `zoo_parity` 14/14 ✓, tol 1e-2 — nvcc FMA-контракция).
- **Волна 2 перехода (07.08): fused-семейство разобрано и портировано.**
  - Диагноз «живости»: `poisson_fused_hidden_step` и `fused_output_lif_step` — **живые** (hot path `train/rl.rs`, Actor-Critic RL); `fused_hidden_izhikevich_step` — ядро динамики poisson-версии + `test_fused_sparse_consistency`; `fused_hh_poly_step` — уже мостился на pillars HHModel (легаси только для кастомных channel_meta + бенчи).
  - Три функции **портированы бит-в-бит** в **`pillars::kernels::fused`** (safe slice API, SIMD `f32x8` внутри, активный список без аллокаций); safe-обёртки в `kernels/src/safe.rs` стали тонкими делегатами на pillars (паттерн Фазы 2). Легаси `kernels/neuron/fused.rs` **удалён сразу** (−426 строк; страховка — `fused_parity.rs` на скалярном эталоне), `RNG_SCALE_24` из `neuron/mod.rs` осиротел и тоже удалён.
  - Parity: `crates/kernels/tests/fused_parity.rs` — бит-в-бит vs скалярный эталон (скрытый слой, выходной LIF через полином `fast_exp`, poisson-fused с детерминированным LCG). Ключевой нюанс (обновлено 22.08): `wide::f32x8::mul_add` под `target-cpu=native` + FMA становится **аппаратно fused** (одно округление), а на машинах без FMA — mul+add (два округления); расхождение давало 1 ULP в parity-тестах. Исправление: кернелы eprop/fused больше не используют `mul_add` — считают явным `* +` (как эталоны), так что бит-идентичность не зависит от FMA-фич целевой машины.
  - Потребители: `train/rl.rs` работает без изменений (через обёртки), `rl_tests` 9/9 ✓, `test_fused_sparse_consistency` ✓.
- **Волна 1 перехода (07.08):**
  - `gco_step` → **портирован** в `pillars::kernels::phase::gco_step` (бит-идентичный перенос, `crates/kernels/tests/gco_parity.rs`); `core::step_gco` (Python `step_gco`) переведён на pillars.
  - `lif_at_step` → **портирован** как **`LIFAtModel` (GLIF2-эквивалент)** в `zoo/neurons/lifat.rs` (`#[dual_neuron]`, 17-я модель зоопарка); `core::step_lif_at` (Python `LifAtState`) переведён на `run_neuron::<LIFAtModel>`. **Волна 6 (07.08):** мост `safe::lif_at_step` → `run_neuron::<LIFAtModel>` — последняя легаси-обёртка закрыта, легаси-кернел удалён вместе с файлом `neuron/lif.rs`. Parity: `crates/kernels/tests/lifat_parity.rs` — мост **бит-в-бит** ≡ прямой `run_neuron` (n=8 SIMD + n=13 хвост) + семантика vs скалярный эталон (θ в пределах 1e-3 rel — fast_exp vs точный exp, число спайков равно). Также **зарегистрирован в `pillars-cuda`** (`LIFAT_CUDA_SOURCE` + `lifat_parity` в `zoo_parity.rs`, 13/13 ✓ — транспиляция `.exp()` проверена NVRTC на GPU).
- **Актуализация 07.08 (по факту аудита):**
  - WC/FHN/астроцит **портированы в zoo** — `zoo/neurons/wc.rs`, `fhn.rs`, `zoo/glia/astrocyte.rs` (все через `#[derive(NeuronModel)]`, WC/FHN — `#[dual_neuron]`).
  - `fast_exp`/`fast_exp_simd` **единый источник** — `kernels/src/neuron/mod.rs` делает `pub use pillars::math::fast_exp::*`; бит-идентичность подтверждена `tests/parity_pillars.rs` на сетке [-87, 87].
  - `core/neuron/astrocyte.rs` стал реэкспортом kernels (не копией).
  - **FHN-расхождение закрыто:** `core::step_fhn` (Python API) переведён на `run_neuron::<FHNModel>` — edge-детекция спайков (правило №6 AGENTS.md) вместо легаси level-детекции с бурстами. Добавлены `kernels/tests/fhn_parity.rs` (бит-идентичность v/w + задокументированная разница семантики спайков) и регрессионный тест `fhn_wrapper_no_back_to_back_spikes` в core.
  - `kernels/src/safe.rs` (3 134 строки) — по факту не «обёртки pillars» (это была оценка карты), а bucket IO/декодеров/T3 (stochastic_encoder, t3_lms/kalman/adam, hebbian, simd-утилиты) + тонкие pillars-обёртки (`lif_step`, `izhikevich_step`, `saw_step`).

- **Фаза 3 (выполнена): мёртвые batch-ядра удалены из `kernels/src/neuron/*`.**
  Удалены: `lif::lif_step` (вместе с неиспользуемым `lif::lif_cn_step`),
  `izhikevich::izhikevich_step`, `adex::adex_step` — вместе с осиротевшими тестами
  (суммарно **−1 124 строки** по git diff). Их роль полностью взяли на себя
  pillars-обёртки в `kernels/src/safe.rs` (`run_neuron::<LIFModel/IzhikevichModel/
  AdExModel>` с переупаковкой лейаутов через thread-local scratch). Актуальные
  размеры: `lif.rs` удалён целиком (**волна 6, 07.08** — `lif_at_step` → pillars),
  `izhikevich.rs`/`adex.rs` удалены (**волна 5**), eprop/dual/hebbian-варианты —
  в pillars. Проверены
  все потребители (connectome, unity, train, python-биндинги, benches) — все
  используют safe-обёртки; легаси LUT-HH (`build_luts`/`hh_lut_step`/`GateLut`)
  **удалён (волна 4, 07.08)** — канон — `pillars::ops::triangulated` (`TriangLUT`
  + `TriangulatedModel`), `benches/bench.rs` переведён на него.
- **`core/neuron/mod.rs` уже делегирует pillars**: реэкспортирует `run_neuron`,
  `run_neuron_par`, `SimdLane`, `Model`, `LIFModel`, `IzhikevichModel`,
  `AdExModel`, `HHModel`, `SAWModel`, `chordotonal_step`, `nmda_mg_block_step`,
  `tsodyks_markram_step` из `pillars::zoo`/`pillars::core`. Core хранит только
  собственные safe-обёртки и обучение.
- **HH переехал целиком** (предыдущая миграция): `crates/core/src/neuron/hh.rs`
  удалён, `HhLutConfig`/`step_hh_lut` мертвы, connectome-раннер гоняет
  `NeuralPopulation<HHModel>` (включая CaV-T pacemaking).
- **Крупные блоки kernels уже перенесены в pillars**: `kernels/synapse.rs`,
  `kernels/dendrite/cable.rs`, `kernels/dopamine.rs` (канон — транш 1b), `kernels/measure.rs`,
  `kernels/decode.rs`, `kernels/phase.rs`, `kernels/vnc.rs` имеют pillars-двойники.
  **Транш 2a (08.08): raw-дубли `kernels/src/{measure,phase,vnc}.rs` удалены** —
  safe-обёртки делегируют в `pillars::kernels::{measure, phase}`; `kernels::*` —
  compat-реэкспорты; `measure_phase_parity.rs` 7/7.

---

## 5. План консолидации (по фазам)

### Фаза 0 — Заморозка
Никаких новых моделей/шагов в `taas-kernels/src/neuron/*` и `core/src/neuron/*`.
Новые нейроны — только в `pillars::zoo` (ComputeNeuron).

### Фаза 1 — Математика: единственный источник в `pillars::math`
- ✅ `fast_exp`/`fast_exp_simd` — **сделано (07.08)**: реэкспорт из `pillars::math`, бит-идентичность на сетке [-87, 87] в `tests/parity_pillars.rs`.
- Перенести/оставить канонические `sigmoid`, `vtrap` только в `pillars::math`.
- В `kernels/src/neuron/mod.rs` заменить тела на `pub use pillars::math::fast_exp::*;`
  (обратная совместимость для 9 крейтов-потребителей). **Сначала** прогнать
  парити-тест `fast_exp_poly_simd` vs `kernels::fast_exp_simd` на сетке [-87, 87]:
  если расхождение >1e-5 rel, зафиксировать каноном pillars и обновить тесты
  kernels (тест `test_fast_exp_scalar_matches_simd` уже есть в pillars).
- `safety_clamp`: оставить одну реализацию (в `pillars::kernels::signal` или
  перенести в `pillars::math::safe`), из `kernels/src/safe.rs` и `io.rs` сделать
  ре-экспорт. В `cuda` — собственный GPU-вариант, он не дубль CPU.

### Фаза 2 — Batch-шаги: safe.rs → pillars
- Крупнейший файл `kernels/src/safe.rs` (2 623 строки): все `*_step` переписать
  как тонкие обёртки над `NeuralPopulation`/`run_neuron` ИЛИ удалить после
  перевода потребителей (connectome/sim.rs, connectome/runner.rs, unity.rs,
  cuda/examples) на pillars.
- Потребители `taas_kernels::safe::*`: `crates/connectome` (sim.rs, runner.rs),
  `src/unity.rs`, `crates/pillars-cuda/examples`, `crates/train`, `crates/t3`,
  `crates/cpe`, `crates/iit`, `crates/karmac`, `src/io.rs`, `crates/taas-embodied`.
  Каждый — кандидат на pillars. Заметка: `taas-core` делает
  `pub use taas_kernels as kernels;` — фасад kernels автоматически сохранит
  работоспособность core и его потребителей без правок.

### Фаза 3 — Модели: kernels/neuron/* → pillars::zoo
- ✅ WC, FHN, astrocyte, **LIF-AT (LIFAtModel/GLIF2)**, **gco_step**, **fused-слои (3 кернела → `pillars::kernels::fused`)**, **`izhikevich_dual_step` → `IzhikevichDualModel`** — **портированы (07.08)** (см. §4). Все batch-нейроны в `kernels/neuron/*` теперь имеют pillars-канон.
- Осталось: астроцит-унификация (kernels vs zoo). Волна 8 (транши 1a–1f) — **завершена**; `kernels::neuron` — чистый compat-слой. LUT-HH-легаси **удалён (волна 4)**. e-prop/Hebbian-CSR — **портированы (волна 5)**. **Мост `safe::lif_at_step` — закрыт (волна 6).** **Волна 7 (08.08): «zero legacy» — все batch-кернелы нейронов удалены** (WC → `pillars::kernels::wc` бит-в-бит; FHN/GLIF/HAAP/HR/ML/SAW/Atlasov → zoo-модели через `run_neuron`). **Волна 8 (08.08): синапсы на pillars** — транш 1a: `neuron/{delay,nmda,tsodyks_markram}.rs` удалены, safe-обёртки делегируют в pillars (`synapse_parity.rs` 4/4). **Транш 1b: пластичность на pillars** — `neuron/{stp,bcm,dopamine}.rs` удалены, safe-обёртки → `kernels/learning` + `kernels/dopamine` + lane-канон `ops/plasticity/stp`; `stdp_homeostatic` портирован в pillars (новый канон); core/neuron/{bcm,stp,stdp} переведены на full-n канон (ручные tail удалены); `plasticity_parity.rs` 10/10. **Транш 1c: рецепторы/сенсорика/глия на pillars** — `neuron/{synapse,chordotonal,astrocyte}.rs` (2 051 строки) удалены; connectome переведён на `pillars::kernels::synapse` напрямую (конвертеры legacy↔pillars **удалены**); `core/neuron/astrocyte.rs` — реэкспорт pillars; compat-реэкспорты в `kernels::neuron` сохраняют Python ABI; `receptor_parity.rs` 7/7. **Транш 1d: дендрит на pillars** — `neuron/dendrite.rs` (1 300) удалён; канон `pillars::kernels::dendrite/cable.rs` (842, SIMD; исправлен легаси-баг C_m в электротоническом вкладе — SIMD≠tail); compat-реэкспорт; `dendrite_parity.rs` 4/4 с inline-эталоном из git HEAD. **Транш 1e: HH-каналы на pillars** — `neuron/hh.rs` (1 040) удалён целиком; движок (`ChannelMeta`/`build_channel_meta`/`hh_poly_step`/`fused_hh_poly_step` + q10/vtrap_simd) портирован бит-в-бит в `pillars::kernels::hh_channels`; сохранены SIMD/scalar расхождения (explicit vs implicit Euler, fused-scalar clamp v, q10 guard-зона); `neuron::hh` — compat-реэкспорт; `hh_parity.rs` 11/11. **Транш 1f: depth_wave на pillars** — `neuron/depth_wave.rs` (264) и мёртвый дубль `crates/kernels/depth_wave.rs` удалены; канон `pillars::kernels::depth_wave::GcoState` (детерминированная L2-норма vs недетерминированный легаси-rayon); `neuron::depth_wave` — compat-реэкспорт; `depth_wave_parity.rs` 6/6. **Волна 8 завершена — `kernels::neuron` содержит только compat-реэкспорты pillars.** План: `resources/docs/PILLARS_MIGRATION.md`. Задокументированные семантические развилки zoo vs легаси: WC width-сигмоида zoo-модели, GLIF/HAAP Euler vs exp, Atlasov f_l гауссиана vs парабола, ML/FHN edge-спайки vs level, SAW-шум на границе моста, DepthGCO L2-норма детерминированная последовательная vs недетерминированный легаси-rayon (транш 1f). **Транш 2a: raw-дубли `{measure,phase,vnc}.rs` удалены** — safe-обёртки → тонкие мосты в `pillars::kernels::{measure,phase}`; мёртвые локальные load8/store8 удалены; `measure_phase_parity.rs` 7/7 (бит-в-бит, SIMD-ветка обязана использовать настоящий `fast_sincos_simd` — скалярный round 1 ULP расходится). **Транш 2b (аудит): `{signal,decode}.rs` отсутствуют** — удалены в Фазе 1; 24 safe-обёртки — мосты на `pillars::kernels::{signal,decode}`; parity `decode_anchor.rs` 8/8 + `parity_pillars.rs` + `kernel_progon.rs`; `io.rs` — живой compat-слой (PwmEncoder/LinearDecoder уникальны, не дубль pillars, python ABI); мёртвый `ion.rs` (type-state PoC, 0 потребителей) удалён. **Транш 2c: `PwmEncoder`/`LinearDecoder` переехали в канон `pillars::kernels::input`** (бит-в-бит; `LinearDecoder::decode` → `signal::linear_decoder`); `kernels/src/io.rs` — чистый compat: структуры реэкспортируются из pillars, функции — min-адаптеры на pillars напрямую (python ABI сохранён, двойная косвенность через safe устранена); `io_parity.rs` 5/5 (inline-эталоны из git HEAD + io≡pillars). **Фаза 4 «Единый API»** (план: `resources/docs/UNIFIED_API.md`). **Транш 3a: `safe::{gco_step, hebbian_csr_step, kuramoto_prepare, kuramoto_update}` удалены** — потребители (core cpe/kuramoto, connectome runner/sim, t3 backend, benches, kernel_progon, kernels lib.rs тесты) переведены на `pillars::kernels::{phase, learning}` напрямую; `gco_parity.rs` переписан на inline-эталон из git HEAD (бит-в-бит). В safe.rs остались только 3 нейронных моста (`lif_step`/`izhikevich_step`/`izhikevich_dual_step` для connectome+unity) — транш 3b.

### Фаза 3.5 — ops-адопция (новое, 07.08)
- Проблема: только 4/20 файлов zoo используют ops (~100 операндов); LIF/Izh/AdEx/FHN/GLIF/HAAP и др. пишут ту же математику инлайн через SimdLane → формулы в двух местах внутри pillars.
- План: перевести LIF и Izhikevich на ops (`leak_integrate`, `horner_izhi`, `fire_reset`, `refractory_*`) по образцу HH (с бит-паритетом через существующие lane-тесты); составить карту dead-ops (ноль потребителей во всём workspace) → удалить или задокументировать как публичный API для внешних авторов.
- `kernels/src/neuron/hh.rs` → тонкий `pub use` на `pillars::zoo::HHModel`
  и `pillars::kernels::hh_channels` (вся математика уже там).

### Фаза 4 — Удаление / фасад
- После Фаз 1–3 `taas-kernels` сводится к фасаду: `pub use pillars::*;` —
  и либо живёт как compat-слой, либо удаляется, а 9 потребителей переводятся
  на `pillars` напрямую.
- `taas-core` остаётся крейтом алгоритмов (trainer, autotune, stats, eprop,
  sorn, encoding, poisson) + реэкспорты pillars; скалярные `step_*`-обёртки
  удаляются.

### Фаза 5 — Документация и границы
- Зафиксировать слои: `pillars::math` (чистая математика) → `pillars::kernels`
  (batch-микроядра, flat-срезы) → `pillars::ops` (био-LEGO) → `pillars::zoo`
  (готовые модели) → `pillars::api` (NeuralPopulation/Network).
- Проверка: `cargo tree -i pillars` должен показать, что pillars не тянет ни
  kernels, ни core (сегодня так и есть).

---

## 6. Критерий «сделано»

1. `taas-kernels/src/neuron/*` и `taas-kernels/src/safe.rs` не содержат
   собственных реализаций шагов — только ре-экспорты/фасад.
2. `fast_exp`/`sigmoid`/`safety_clamp` определены ровно один раз (в pillars).
3. Все модели мухи (HH/GLIF/HAAP/Izh/AdEx) считаются через
   `NeuralPopulation<M>` и проходят парити-тесты (образец: `tests/hh_parity.rs`).
4. `cargo clippy --workspace` = 0, все тесты зелёные.
