# NEURON Catch-Up Plan — догоняем по каналам и NeuroML

> План работ по закрытию разрывов из
> [NEURON_GAP_ANALYSIS.md](NEURON_GAP_ANALYSIS.md). Фазы упорядочены по
> соотношению «рычаг / стоимость». Каждая фаза — отдельный PR-размерный шаг
> с критериями приёмки и parity-защитой.

## Цель

Научить pillars загружать и гонять **произвольные conductance-based каналы**
из NeuroML2 (и, через них, сверяться с NEURON), вместо нынешнего enum из 4
гейтов `{NaM, NaH, Kn, CatC}`.

---

## Сводка NEURON-parity (доказательная база)

Итоговая сверка pillars против NEURON 9.0.2 (эталоны — `resources/validate/neuron_models/*_ref.py`, значения хардкодятся в Rust-тесты, поэтому CI не требует NEURON):

| Механизм | Эталон | Результат |
|---|---|---|
| Пассивный кабель (ball-and-stick) | `gen_cable_ref.py` (`pas`, Hines) | max \|ΔV\| ≈ 3.7e-4 мВ (машинная точность) |
| Ветвящийся кабель (Y) | `gen_cable_branch_ref.py` | max \|ΔV\| ≈ 3–5e-4 мВ (4 точки) |
| Активный кабель (NaV/KV) | `gen_cable_hh_ref.py` (`hh`) | **8 спайков vs 8** (точное совпадение) |
| Адаптивный шаг (CVODE-аналог) | `gen_cable_ref.py` (финал) | < 0.01 мВ, `dt` реально меняется |
| Марков/kschan в кабеле | `gen_cable_ks_ref.py` (`ks3.mod`) | max \|ΔV\| ≈ 0.014 мВ (интегратор) |
| Точечный kschan | `ks_vs_neuron.rs` | ≈ 6e-4 мВ |
| `channelPopulation` (стохастика, mean-field) | переиспользует `REF_V_KS3` | ≈ 0.041 мВ (среднее по ансамблю N=2048×16) |
| `voltageClamp` | `gen_clamp_ref.py` (`SEClamp`) | равновесие < 0.001 мВ; max ≈ 0.47 мВ (быстрый фронт) |
| Tsodyks–Markram STP | `gen_tm_ref.py` (`tm.mod`) | < 1e-4 (delivered weights) |
| Ca²⁺ (Nernst + decayingPool) | `gen_ca_ref.py` (`cachan_simple.mod`+`cacum.mod`) | max \|ΔV\| ≈ 0.08 мВ; max rel \|ΔCa\| ≈ 0.9% (оболочка) |
| Ca²⁺ `channelPopulationNernst` (mean-field) | транзитивно через `channelDensityNernst` | < 0.001 мВ / 0.3% (mean по ансамблю) |
| HAAP (M-ток + HCN) | `validate_haap_neuron.py` (`hcn.mod`/`km.mod`) | субпорог (spiking-модель не сравнивается выше порога) |
| **Poirazi 2003 CA1 сома (реальная OSB, Ih + M), 35 °C + per-channel Q10** | `gen_poirazi_ref.py` (`poirazi_h/km/na/kdr.mod` + Q10) | **max \|ΔV\| ≈ 0.0007 мВ** (машинная точность) |

> `voltageClampTriple` держит `returnVoltage` навсегда (LEMS) — на NEURON `SEClamp`
> не мапится (тот выключается после 2-й ступени), сверяется аналитически.

---

## Фаза 0 — Фикстуры и ground truth (маленькая)

**Что:** завести канонический NeuroML2 squid-axon файл как тест-фикстуру.

- Создать `resources/validate/neuroml/hh_squid.nml` — полный `ionChannelHH`
  (Na: m³h, K: n⁴) + `leak` + `hhCell`/`channelDensity` + `pulseGenerator`.
- Вынести `HH`-константу из теста `neuroml::tests` в `load_file` по фикстуре
  (сейчас она инлайн-строка).

**Приёмка:** `load_file("hh_squid.nml")` даёт тот же результат, что и инлайн-тест
`hh_cell_spikes_under_pulse`.

---

## Фаза 1 — Обобщение каналов (data-driven)  ← ГЛАВНЫЙ ШАГ

**Что:** заменить хардкод `GateType` на замкнутые формы NeuroML2. Точка входа —
`crates/pillars/src/kernels/hh_channels.rs` (generic-движок `hh_poly_step` /
`fused_hh_poly_step`). Канонический `zoo::HHModel` не трогаем — он остаётся
быстрым путём.

**Шаги:**

1. Ввести в `hh_channels.rs` data-driven типы:

   ```rust
   enum RateForm { Exp, Sigmoid, ExpLinear }          // forwardRate/reverseRate
   enum VarForm  { Exp, Sigmoid, ExpLinear }          // steadyState
   struct ClosedForm { form: RateForm, rate: f32, midpoint: f32, scale: f32 }

   enum GateKinetics {
       Rates        { forward: ClosedForm, reverse: ClosedForm },          // gateHHrates
       TauInf       { time_course: ClosedForm, steady_state: ClosedForm }, // gateHHtauInf
       Instantaneous{ steady_state: ClosedForm },                          // gateHHInstantaneous
       RatesTau     { forward: ClosedForm, reverse: ClosedForm, time_course: ClosedForm },
       RatesInf     { forward: ClosedForm, reverse: ClosedForm, steady_state: ClosedForm },
       RatesTauInf  { forward: ClosedForm, reverse: ClosedForm, time_course: ClosedForm, steady_state: ClosedForm },
   }
   ```

2. `GateSpec`: `power` (instances) + `GateKinetics` вместо `alpha`/`beta` fn-указателей.
   `ChannelSpec` остаётся `{ name, gates, E_rev, g_max }`.

3. Удалить `GateType`-маппинг **для обобщённого пути**: `ChannelMeta` несёт
   data-driven `kinetics: Vec<GateKinetics>`; α/β-вычисление в `hh_poly_step`
   (SIMD + scalar) переведено на `GateKinetics::alpha_beta(_simd)` (exponential
   Euler `q = inf + (q−inf)·exp(−dt·sum·q10)`). **`GateType` сохранён** — он
   остаётся ключом fused fast-path (`fused_hh_poly_step`), который факторизует
   общий `exp(−v/10)` и завязан на 4 канонических семейства; fused-путь —
   оптимизация над стандартным набором, NeuroML-каналы идут через non-fused.

4. Перевыразить `standard_channels()` / `pacemaking_channels()` через те же
   `ClosedForm`-параметры (Na-m = ExpLinear(1,−40,10) + Exp(4,−65,−18) и т.д.).

**Приёмка:**
- parity: data-driven `standard_channels()` даёт **бит-в-бит** те же m/h/n, что
  текущий `GateType`-путь (зафиксировать тестом до удаления старого кода).
- существующие тесты `hh_channels` + `hh_parity` + `fused_parity` зелёные;
- `pacemaking_fires_spontaneously` и `standard_channels_quiescent_without_input`
  не регрессируют.

---

## Фаза 2 — NeuroML: ionChannelHH / gate* в парсер  ← ✅ полностью

**Что:** перестать скипать `ionChannel`/`gateHH*`; парсить каналы и строить
произвольный conductance-based путь.

**Статус:** **сделано и покрыто тестами.** parse.rs (шаги 1–4) парсит каналы
в data-driven IR; compile.rs (шаги 5–6) запускает произвольные наборы каналов
на новом `ConductancePopulation` (generic-движок поверх `hh_poly_step`).
Стандартные NaV m³h / KV n⁴ по-прежнему идут через `HHModel` (бит-в-бит);
нестандартные каналы (HCN/M-стиль `gateHHtauInf`, произвольные `gateHHrates`)
— через `ConductancePopulation`. Неподдерживаемые гейты (`gateHHInstantaneous`
и т.п.) дают честную ошибку загрузки.

**Шаги (parse.rs) — ✅:**
1. Парсить `<ionChannel>` / `<ionChannelHH>`: атрибуты `id`, `conductance`,
   `species`; дети `gateHHrates` / `gateHHtauInf` / `gateHHInstantaneous` /
   `gateHHratesTau` / `gateHHratesInf` / `gateHHratesTauInf`.
2. Парсить `forwardRate`/`reverseRate`/`steadyState`/`timeCourse` с
   `type` ∈ `{HHExpRate, HHSigmoidRate, HHExpLinearRate, HHExpVariable,
   HHSigmoidVariable, HHExpLinearVariable, fixedTimeCourse}` и параметрами
   `rate`/`midpoint`/`scale`/`tau` (юниты: `per_ms`, `mV`, `ms`).
3. Парсить `q10Settings` (`q10Fixed`, `q10ExpTemp`).
4. В IR добавить `ChannelSpec`-структуры; убрать `ionChannel`/`ionChannelHH`/
   `gateHH*` из `skipped_element_note()` (`gateKS`/`gateFractional` остаются
   «признанными, но неподдержанными» — warning).

**Шаги (compile.rs) — ✅:**
5. `CellSpec::ConductanceBased` — новый вариант: `channelDensity`/
   `channelPopulation` резолвит `ionChannel` → `ChannelMeta`, берёт
   `condDensity`/`erev` как g_max/E_rev.
6. `ConductancePopulation` (новый `api::conductance`) — `Population` поверх
   generic-движка `hh_poly_step` (не `HHModel`), с edge-детекцией спайков
   (0 мВ), steady-state init гейтов и Q10 через `celsius`. CPU-only: для
   data-driven каналов нет `#[dual_neuron]` CUDA-источника (честно помечается
   пустым `GpuPopMeta.source`).

**Приёмка:**
- ✅ `hh_squid.nml` (gateHHrates m³h/n⁴) грузится и спайкает как
  `hh_cell_spikes_under_pulse` (тест `hh_squid_nml_gatehh_rates_spikes_under_pulse`);
- ✅ m/h/n rate-функции round-trip бит-в-бит (тест
  `squid_ion_channel_rates_roundtrip_bit_exact`);
- ✅ `gateHHtauInf`-пример (km/HCN-стиль) грузится без warning (тест
  `gate_hh_tau_inf_parses_without_warning`);
- ✅ смешанная клетка (Na/K + HCN `gateHHtauInf`) запускается generic-движком
  без warning и спайкает (тест `hh_cell_mixed_channels_run_on_generic_engine`);
- ✅ `ConductancePopulation` воспроизводит канонические инварианты: standard
  молчит без входа, pacemaking спайкает спонтанно (тесты в `api::conductance`);
- ✅ неподдерживаемый гейт (`gateHHInstantaneous`) → ошибка загрузки, не skip
  (тест `hh_cell_unsupported_gate_errors`).

---

## Фаза 3 — HAAP против NEURON (hcn / km)

**Что:** закрыть TODO-пробел «HAAP vs NEURON».

**Статус (частично):** найден и устранён критичный баг в pillars HAAP — гейты
были переписаны ИИ-кодерами: `k` превращён из slope-фактора (делитель) в
множитель. **Научный вывод** (модель по имени «Hyperpolarization-Activated
Adaptive» и по назначению — rebound после торможения + адаптация):
- M-ток (`u`) открывается на **деполяризации**: `u_∞ = 1/(1+exp(−(V−V_half_u)/k_u))`;
- HCN (`h`) открывается на **гиперполяризации**: `h_∞ = 1/(1+exp(+(V−V_half_h)/k_h))`;
- `k` — slope-фактор (**делитель**), стандартная Больцмановская конвенция.

ВНИМАНИЕ: `km.mod`/`hcn.mod`/`haap_ref.py` (TNMT VM) содержат формулы с
**инвертированными знаками** относительно собственных комментариев и имени
модели — `hcn.mod` пишет «activates on hyperpolarization», а формула
`1/(1+exp(−(v−vhalf)/k))` открывает его на деполяризации. Поэтому TNMT-формулы
НЕ являются эталоном для сверки: эталон — научно корректная конвенция выше.

**Сделано:**
- `zoo/neurons/haap.rs` — сигмоиды приведены к `k`-делителю + биологические
  знаки (был `k`-множитель — причина «взрыва exp», который «чинили» клампом ±50;
  при делителе |arg| ≲ ±40 и переполнения нет).
- `api/neurons/haap.rs` — `sigmoid` и steady-state init уже были корректны
  (делитель + биологические знаки); оставлены, комментарии уточнены.
- Тест `sigmoid_matches_biological_convention` — лочит конвенцию.

**⚠ Открытый вопрос — init-конвенция.** Сейчас два разных init: steady-state
(u≈0.0025, h≈0.16 при V=−65 — физически согласованный, как INITIAL в NEURON) и
`zoo`-тесты/`connectome` (u=0, h=1 — произвольный, вносит фиктивный транзиент).
Их нужно свести к steady-state.

**Сделано дополнительно:** init сведён к steady-state (`zoo`-тесты + `connectome`
+ `compare_all.py`); добавлен rebound-тест (`haap_hcn_opens_on_hyperpolarization_`
`and_rebounds`); `km.mod`/`hcn.mod` занесены в репо с исправленными знаками.

**✅ Завершено:** моды скомпилированы (`mknrndll` в ASCII temp-директории — обход
кириллицы), `taas_checkmate.pyd` пересобран (maturin), RMSE прогнан.
Канонический скрипт — `resources/validate/neuron_models/validate_haap_neuron.py`.

**Результат (subthreshold, где модели эквивалентны):**
| Протокол | I (µA/cm²) | RMSE |
|---|---|---|
| деполяризующий (leak + M открытие + HCN закрытие) | +1.5 | **0.011 mV** |
| гиперполяризующий rebound (HCN открытие → rebound) | −3.0 | **0.017 mV** |

RMSE ≈ 0.01–0.02 mV — машинная точность: единицы (`g/1000` → S/cm²,
`I·area·1000` → nA) и знаки гейтов подтверждены. Спайковый режим (I=8) **не
сравнивается по построению**: pillars HAAP — спайковая модель (V≥V_thresh →
reset + Δu), а NEURON `pas+km+hcn` — пассивные проводимости без порога. Там
NEURON интегрирует до −17.9 mV, pillars ресетится на −40 → RMSE ~10 mV это не
ошибка, а разная семантика выше порога.

**Попутно найден и исправлен баг в харнессах:** `compare_all.py`/`validate_all.py`
считали `ROOT = <project>/resources` (2 уровня вверх от `resources/validate/neuron_models/`)
и не добавляли корень в `sys.path`, из-за чего `import taas_checkmate` молча брал
**устаревшую** копию из site-packages (старая сигмоида). Добавлен `PROJECT_ROOT`
(3 уровня вверх) + `sys.path.insert(0, …)`. `validate_all.py::validate_haap` всё ещё
использует старый init (`u=0, h=1`) и I=8 — канон теперь в `validate_haap_neuron.py`.

**Приёмка:** `validate_haap_neuron.py` → `RESULT: PASS` (RMSE < 0.5 mV);
`haap_comparison.png` при `--plot`.

---

## Фаза 4 — gateKS (кинетические схемы)  ← ✅ сделано

Марковские схемы (`KSState` / `KSTransition` / `gateKS`). Нужны для многих
реальных каналов (Ca-динамика, полный HCN).

**Статус:** сделано и покрыто тестами.

**Сделано:**
- `kernels::hh_channels` — типы `MarkovGateSpec` / `MarkovTransition` /
  `MarkovRate` (Directed / TauInfForward / TauInfReverse) + интегратор master
  equation через **точную матричную экспоненту** `exp(Q·dt)` (scaling-and-
  squaring + ряд Тейлора степени 8), стационарное распределение (Гаусс с
  частичным выбором ведущего элемента), открытая доля. `MAX_MARKOV_STATES = 16`.
- `gate_kinetics_to_markov` — опускает HH-гейт до эквивалентной 2-состоянийной
  схемы (closed ⇌ open, forward=α, reverse=β), что позволяет одному движку
  гонять смесь HH- и KS-каналов.
- `parse.rs` — парсинг `ionChannelKS`/`gateKS`/`closedState`/`openState`/
  `forwardTransition`/`reverseTransition`/`tauInfTransition` в `MarkovGateSpec`;
  state-id резолвятся в индексы, неизвестное состояние → честная ошибка.
  `ionChannelKS`/`gateKS` убраны из «пропущенных».
- `gateFractional` (`subGate` + `fractionalConductance`) — тоже поддержан:
  парсится в `FractionalGateSpec` (взвешенная сумма независимых tau-inf
  субгейтов) и опускается в марковскую схему над **декартовым произведением**
  состояний субгейтов (`fractional_gate_to_markov`, 2^K состояний, K ≤ 4);
  `MarkovGateSpec.open` стал `Vec<f32>` (вес состояния, а не флаг).
- `api::conductance` — `MarkovPopulation` (скалярный master-equation движок):
  гейты в gate-major state-major раскладке, steady-state init, spike-detection
  по 0 мВ, Q10 через `celsius`.
- `compile.rs` — `hhCell` с любым `ionChannelKS` `channelDensity` →
  `MarkovPopulation` (HH-каналы той же клетки опускаются до Маркова). Без
  KS-каналов — прежний SIMD `ConductancePopulation`.

**Приёмка:**
- ✅ 2-состоянийная марковская схема совпадает с аналитикой
  `q_inf + (q−q_inf)·exp(−(α+β)·dt)` (тест `markov_two_state_matches_analytic_`
  `exponential`, 1e-6);
- ✅ стационарное распределение сходится к `q_inf` и сохраняет Σp=1;
- ✅ HH-гейт, опущенный до Маркова, совпадает с HH-движком (`gate_kinetics_`
  `lowered_to_markov_matches_hh`);
- ✅ parse roundtrip `gateKS` (forward/reverse + tauInfTransition) без warning;
- ✅ end-to-end: `hhCell` + `ionChannelKS` (`tauInfTransition` HCN) + Na/K —
  спайкает под импульсом через `MarkovPopulation`
  (`hh_cell_kinetic_scheme_channel_runs_on_markov_engine`);
- ✅ `MarkovPopulation` с опущенным стандартным набором молчит без входа
  (`markov_population_standard_channels_quiescent`);
- ✅ трёхсостоянийная обратимая цепь C0 ⇌ C1(open) ⇌ C2: стационар по
  детальному балансу + сохранение Σp (`markov_three_state_chain_steady_state_`
  `and_conservation`) — якорь на N>2 состояний;
- ✅ `gateFractional` из двух субгейтов → 4 состояния, открытая доля =
  взвешенной сумме субгейтов (`fractional_gate_to_markov_matches_weighted_sum`);
- ✅ parse roundtrip `gateFractional` (веса + tau-inf субгейты) без warning
  (`gate_fractional_parses_without_warning`);
- ✅ parse roundtrip `gateKS` трёх состояний (closed ⇌ open ⇌ inactivated)
  (`gate_ks_three_state_parses_without_warning`);
- ✅ end-to-end: `hhCell` + `gateFractional` (2 субгейта) + Na/K спайкает через
  `MarkovPopulation` (`hh_cell_fractional_channel_runs_on_markov_engine`);
- ✅ **NEURON parity** (`tests/ks_vs_neuron.rs`): `MarkovPopulation` под
  subthreshold current-clamp совпадает с NEURON 9.0.2 до машинной точности:
  **2-состояния** (`ks2.mod`, Na-m гейт) max |ΔV| ≈ 1e-4 mV, **3-состояния**
  (`ks3.mod`, closed ⇌ open ⇌ inactivated, Na-m + Na-h) max |ΔV| ≈ 6e-4 mV.
  Эталон генерится `resources/validate/neuron_models/gen_ks_ref.py` (общий
  для обеих схем). Примечание: `INITIAL` в `ks3.mod` обязан быть точным
  стационаром цепи (детальный баланс), а не «независимыми» m·h — иначе
  NEURON стартует из другого состояния и расходится с `markov_steady_state`.

**Ограничение (честно):** `MarkovPopulation` — скалярный (матричная экспонента
на нейрон), поэтому медленнее SIMD-движка; для HH-наборов сохраняется быстрый
`ConductancePopulation`. SIMD-версия master equation — отдельная оптимизация.

---

## Фаза 5 (долгий горизонт) — морфология + интеграторы

**Подфаза 5a (парсинг морфологии) — ✅ сделано.**

- `parse.rs` парсит `morphology`/`segment`/`segmentGroup`/`segmentParent`/
  `biophysicalProperties`/`membraneProperties`/`intracellularProperties`/
  `channelDensity`/`resistivity`/`initMembPotential` и `<cell>` (ссылки на
  морфологию + биофизику) в IR: `MorphoIR`/`Segment`/`SegmentGroup`/
  `BiophysicalProperties`/`MorphoCell`. `cell`/`morphology`/`segment`/… убраны
  из `skipped_element_note` (больше не warning).
- Сегмент: `id`, `name`, `proximal` (опц. — наследуется от родителя),
  `distal` (x/y/z/diameter), `parent` (`segmentParent` + `fractionalLength`,
  по умолчанию 1.0). `segmentGroup`: `member` + `include`.
- Покрыто тестами (4 новых): парсинг дерева сома→дендрит без warning;
  `proximal` отсутствует → `None`; `cell` + `biophysicalProperties` резолвятся
  по id; `channelDensity` без `erev` → честная ошибка.
**Подфаза 5b (импорт SWC) — ✅ сделано.**

- `api/neuroml/swc.rs` — читает SWC point-list (`n T x y z R P` на строку) и
  опускает его в тот же `MorphoIR`, что и NeuroML `<morphology>`: каждый
  не-корневой сэмпл → `Segment` (`proximal` = точка родителя, `parent` +
  `fractionalLength=1.0`), корень (`P=-1`) → `parent=None`/`proximal=None`;
  радиус SWC конвертируется в диаметр (`2R`). `segmentGroup` автогенерируются
  по типу сэмпла (soma/axon/dendrite/apical).
- `parse_swc(text, id)` + `load_swc_file(path)` (id = имя файла без расширения).
- Покрыто тестами (5): дерево сома→дендрит (parent/root/proximal/диаметры),
  комментарии/пустые строки, apical-группа, неизвестный parent → ошибка,
  дубликат id → ошибка, малформат/пустой → ошибка.

**Подфаза 5c (интегратор мембраны) — ✅ частично.**

- `MarkovPopulation` переведён с **explicit Euler** на **implicit Euler
  (derivimplicit)** для мембраны: `V' = (C/dt·V + I + ΣgE) / (C/dt + Σg)`,
  с проводимостями/гейтами на старом V (semi-implicit) — тот же метод, что
  `HHModel` и NEURON по умолчанию. Снимает explicit-Euler ограничение
  устойчивости по dt для жёстких KS-каналов (Ca/HCN); master equation уже
  точная (матричная экспонента).
- Тест `markov_membrane_implicit_stable_at_large_dt`: на чистой RC-мембране
  при dt=10 мс (≈3τ, где explicit Euler осциллирует/расходится) implicit
  монотонно садится на покой без перескока.
- `ks_vs_neuron.rs` по-прежнему зелёный (max |ΔV| ≈ 1e-4 / 6e-4 mV).
- **Позже унифицирован** `hh_poly_step` (generic SIMD): расхождение
  «SIMD explicit Euler / скалярный implicit Euler» устранено — оба пути теперь
  derivimplicit (тот же метод, что `HHModel`/NEURON), кламп fused v_new тоже
  выровнен. Inline legacy-эталон в `hh_parity.rs` обновлён до той же семантики,
  и добавлены parity-тесты «SIMD-блок vs скалярный хвост» для идентичных лейнов
  (n=9/15): `hh_poly_step` бит-в-бит, `fused` ~4e-5 мВ. Один и тот же нейрон
  больше не меняет динамику от размера популяции.

**Подфаза 5d-1 (d_λ-дискретизация) — ✅ сделано.**

- `api/neuroml/discretize.rs` — мост `MorphoIR → CompartmentTree`:
  каждый сегмент режется на `n = max(1, ⌈L/(d_λ·λ)⌉)` компартментов
  (λ = √(d·Rm/4·Ra), DC space constant), первый компартмент цепляется к
  последнему компартменту родителя, остальные — линейной цепочкой; сома
  (корень) — один компартмент, одиночная точка → `L = diameter`.
- Пассивные кабельные параметры на компартмент (NEURON-конвенция единиц:
  µm, Ω·cm, µF/cm², Ω·cm²): площадь `π·d·L`, ёмкость `cm·A`, лик `A/Rm`,
  аксиальное сопротивление `4·Ra·L/(π·d²)`.
- Покрыто тестами (5): d_λ режет 1000 µm-дендрит (λ=1000 µm) ровно на 10;
  R_a/C_m/g_leak против аналитики; короткий сегмент → 1 компартмент;
  несколько корней → ошибка.

**Подфаза 5d-2 (пассивный кабельный солвер) — ✅ сделано.**

- `CableSolver` в `discretize.rs` — **implicit Euler** пассивного кабеля над
  `CompartmentTree`, линейная система `A·V = b` решается за O(n) **алгоритмом
  Hines** (восходящая элиминация + нисходящая подстановка). Безусловно
  устойчив — нет explicit-Euler ограничения по dt.
- **Единицы сведены к NEURON-конвенции** (как `kernels::dendrite::cable`):
  V[mV], t[ms], C[nF], g[µS], I[nA], R[MΩ] — проверено размерным тестом
  τ = C/g = 20 мс. Компартмент: C = cm·A (nF), g_leak = A/Rm (µS),
  R_axial = 4·Ra·L/(π·d²) (MΩ).
- Покрыто тестами (4): покой — точная неподвижная точка; τ = 20 мс
  (V(τ) ≈ E + 0.632·ΔV, ошибка < 0.05 мВ); ball-and-stick затухает монотонно
  с расстоянием; dt=100 мс (5τ) устойчив без перескока.

**Подфаза 5d-3 (активные HH-каналы в кабеле) — ✅ сделано.**

- `ActiveChannel` (HH-гейты `GateKinetics` + `e_rev` + абсолютная
  проводимость µS) и `ActiveCableSolver` (обёртка над `CableSolver`):
  полу-неявный шаг — экспоненциальный Euler гейтов на старом V, затем
  Hines-решение с активной проводимостью в диагонали. `CableSolver` получил
  `step_with_extra(g_extra, g_e_extra)` — вклад активного тока
  `g·(V−E_rev)` входит в неявную линейную систему (устойчиво).
- Тест: один компартмент со стандартными NaV/KV + HH-лик (Rm=3333 Ω·cm² =
  1/0.3 mS/cm²) спайкает под надпороговым током — активные каналы реально
  входят в кабельный солвер.
- **Ещё не сделано**: привязка `ionChannelKS` (Марков) к компартментам,
  assignment **активных** каналов по segmentGroup из `biophysicalProperties`,
  NEURON-parity кабеля, адаптивный шаг (CVODE-аналог).

**Подфаза 5d-4 (пассивная биофизика → кабель) — ✅ сделано (polish).**

- `parse_channel_density`/`specificCapacitance` теперь **unit-aware**:
  `condDensity` (`S_per_m2`/`mS_per_cm2`/…) → S/cm², `specificCapacitance`
  (`F_per_m2`/`uF_per_cm2`/…) → µF/cm² (больше не «raw strip» с потерей
  порядка величины).
- `CableParams::from_biophysics(bp)` + `passive_leak_reversal(bp)` — мост из
  `biophysicalProperties` в кабель: `resistivity` → `ra_ohm_cm`,
  `specificCapacitance` → `cm_uf_per_cm2`, пассивный лик (`ion non_specific`/
  `pas`/`leak`) → `rm_ohm_cm2 = 1/condDensity`, а `erev` лика (фолбэк —
  `initMembPotential`) как e_leak.

**Подфаза 5e (кабель → Network) — ✅ сделано.**

- `CablePopulation` в `discretize.rs` — boxed-популяция (реализует `Population`)
  из `n` идентичных кабельных нейронов: ток (nA) подаётся в сому (корень),
  наблюдаемое состояние — напряжение сомы (колонка 0), спайк — пересечение
  0 мВ на восходящем фронте.
- `compile.rs` теперь резолвит `population` → `<cell>` (морфология + биофизика):
  дискретизация d_λ (`CableParams::from_biophysics` + `passive_leak_reversal`),
  затем `CablePopulation` в `Network`. Пассивный кабель; масштаб тока nA = 1.
- Покрыто тестами (2): end-to-end `load()` → морфо-клетка с сомой+дендритом
  деполяризуется под импульсом в сому (покой = `erev` pas-канала); ссылки на
  неизвестную морфологию/биофизику → честная ошибка.

**Подфаза 5f (активные каналы по segmentGroup) — ✅ сделано.**

- `MorphoIR::resolve_segments(group)` — резолв `segmentGroup` → id сегментов,
  `include`-цепи транзитивно (cycle-safe), спец-id `all` = все сегменты.
- `CompartmentTree::group_compartments(segments)` — сегменты → индексы
  компартментов после дискретизации.
- `MarkovActiveChannel` (кабель-версия `gateKS`/`gateFractional`: master
  equation) + `set_markov_channels` в `ActiveCableSolver`/`CablePopulation`;
  HH- и Марков-каналы интегрируются в одном полу-неявном Hines-шаге.
- `compile.rs::resolve_cable_channel` — `channelDensity` → `ionChannelHH`
  (HH-гейты или Марков для fractional) / `ionChannelKS` (Марков), масштаб
  «density × membrane area» → абсолютная µS, привязка к компартментам по
  segmentGroup. Пассивный `pas`/`leak` по-прежнему уходит в `CableParams`.
- Покрыто тестами: морфо-клетка с NaV/KV на соме **спайкает**; HCN
  (`ionChannelKS`) в кабеле гоняется без NaN; неизвестная segmentGroup → ошибка.

**Подфаза 5g (NEURON-parity кабеля, пассивный) — ✅ сделано.**

- `gen_cable_ref.py` — эталон NEURON 9.0.2: пассивный ball-and-stick
  (сома L=20/d=20 + дендрит L=1000/d=2), Ra=100, cm=1, Rm=20000, E=-65,
  I=+0.01 nA в сому (10–60 мс), dt=0.025, secondorder=0 (backward Euler).
- `passive_ball_and_stick_matches_neuron` (в `discretize.rs`) — сверка Hines-
  солвера с эталоном по V(soma) и V(tip) на сетке 5 мс.
- **Результат: max |ΔV| ≈ 3.7e-4 мВ** — машинная точность (остаток — f32
  pillars против f64 NEURON за 4000 шагов). Подтверждены d_λ-дискретизация,
  осевые сопротивления (junction = (R_parent+R_this)/2), ёмкость/лик и сам
  Hines-солвер.

**Подфаза 5h (активная кабельная parity) — ✅ сделано (spike-count).**

- `gen_cable_hh_ref.py` — эталон NEURON 9.0.2: сома L=20/d=20 с `hh`
  (NaV/KV + лик 0.3 mS/cm², E=-54.3), I=+0.3 nA (10–90 мс), dt=0.025,
  secondorder=0, tstop=100.
- `active_hh_soma_matches_neuron_spike_count` (в `discretize.rs`) — морфо-сома
  с NaV/KV в `ActiveCableSolver`, тот же протокол.
- **Результат: 8 спайков против 8 у NEURON** (точное совпадение). Гейты
  pillars — exponential-Euler, NEURON — backward-Euler, поэтому сверяется
  число спайков, а не трасса (тайминг спайков чувствителен к методу).

**Подфаза 5i (адаптивный шаг, CVODE-аналог) — ✅ сделано.**

- `AdaptiveCableSolver` в `discretize.rs` — адаптивный драйвер над
  `ActiveCableSolver`. Оценка локальной ошибки усечения — step doubling (один
  полный шаг `dt` vs два шага `dt/2`); контроллер подстраивает `dt`
  (`factor = 0.9·err^(−1/(p+1))`, кламп `[0.1, 2.0]`, `dt` в `[dt_min, dt_max]`).
- Ошибка — max-norm по компартментам с пер-компартментным скейлом
  `atol + rtol·|V|` (mV). При отказе состояние восстанавливается из снимка
  (`ActiveCableSolver` — `Clone`) и шаг повторяется с меньшим `dt`.
- Те же идеи, что у CVODE (жёсткие каналы/спайки → авто-уменьшение `dt`,
  пологие фазы → рост `dt`).
- Тесты (2): пассивный ball-and-stick — финал V(сома)/V(кончик) совпадает с
  эталоном NEURON (−64.6701 мВ, <0.01 мВ), `dt` реально меняется;
  активный HH — то же число спайков (8±2), `dt` сжимается на спайках.

**Подфаза 5k (переменный порядок, CVODE-уровень) — ✅ сделано.**

- `AdaptiveCableSolver` теперь **переменного порядка 1 ↔ 2**, как CVODE:
  - **Порядок 1** — чистый implicit Euler (BDF1) через step-doubling;
  - **Порядок 2** — принятая траектория получает **Richardson-экстраполяцию**
    `v = 2·v_half − v_full`, которая сокращает ведущий член ошибки `h²` —
    одностшаговый эквивалент **BDF2** (для линейной кабельной системы они
    совпадают). Гейты/кинетика остаются в состоянии полу-шаговой траектории
    (semi-implicit, как в фикс.-dt солвере).
  - Контроллер: `factor = 0.9·err^(−1/3)` на порядке 2 (ошибка ~ dt³), рестарт
    на **порядок 1 после отказа шага** (как CVODE), промоция обратно на 2 при
    запасе (`err < 0.5`).
- Тесты (3): пассивный/активный (5i) сохранены без изменений допусков;
  `adaptive_variable_order_transitions` — порядок реально поднимается 1 → 2 и
  сбрасывается на фронте сильного импульса (отказ шага), а траектория при
  рабочем допуске 1e-3 сходится к эталону 1e-5 (< 2 мВ на размахе ~100 мВ).

**Подфаза 5j (ветвящийся кабель + branch-point fix) — ✅ сделано.**

- `gen_cable_branch_ref.py` — эталон NEURON 9.0.2 для **Y-образного** пассивного
  кабеля: сома L=20/d=20 → ствол L=1000/d=2 → две асимметричные дочери
  (da L=500/d=1 — тонкая/длинная, затухает; db L=200/d=2 — толстая/короткая,
  проводит), обе от кончика ствола; пассивный `pas`, dt=0.025, secondorder=0.
- **Найден и исправлен баг в ветвлении**: при ≥2 детей полу-сопротивление
  родителя дублировалось на каждого ребёнка (пере-связь дочерей, ошибка ~1%).
  NEURON моделирует branch point как отдельный узел, где родительская
  «дистальная половина» — **один общий** резистор. Теперь `discretize`
  вставляет zero-volume branch-узел (`r_axial=0`, `area=0`, `cm=0`, `g_leak=0`)
  на дистальном конце сегмента с ≥2 детьми; дети цепляются к нему →
  `junction = R_parent/2 + R_child/2` с общей родительской половиной, ровно как
  NEURON.
- `passive_branching_cable_matches_neuron` (в `discretize.rs`) — сверка V(сома),
  V(branch-узел = `trunk(1.0)`), V(кончик da), V(кончик db).
- **Результат: max |ΔV| ≈ 3–5e-4 мВ** на всех четырёх точках — машинная
  точность, как и линейный ball-and-stick. Асимметрия da/db сохранена точно.

**Подфаза 5l (Марков/kschan в кабеле vs NEURON) — ✅ сделано.**

- `gen_cable_ks_ref.py` — эталон NEURON 9.0.2 для ball-and-stick (сома
  L=20/d=20 + дендрит L=1000/d=2) с 3-состоянием `ks3.mod` (Na-m/Na-h rates)
  на **соме**: pas 0.3 mS/cm² / E=-54.3 везде, ks3 gbar=36 mS/cm² / ek=-77,
  V0=-65, I=+0.125664 nA (10 µA/cm²) в сому (10–60 мс), dt=0.025,
  secondorder=0. K-шунт сомы (m1≈0.05 при −65) держит её у покоя, пассивный
  дендрит (e_pas=−54.3) заряжается — канал реально работает в кабеле.
- `markov_cable_matches_neuron_ks3` (в `discretize.rs`) — та же морфология/
  протокол через `ActiveCableSolver` + `MarkovActiveChannel` (expm, steady-state
  init при V0), сверка V(сома)/V(кончик) на сетке 5 мс.
- **Результат: max |ΔV| ≈ 0.014 мВ** (сома и кончик; допуск 0.03). NEURON
  интегрирует KINETIC backward-Euler (METHOD sparse), pillars — точной
  матричной экспонентой → расхождение на уровне интегратора; против ~6e-4 у
  точечного `ks_vs_neuron.rs` — добавка от аксиального пути 1000 мкм.

**Подфаза 5m (fully-implicit гейты — исследовано, отклонено) — ✅ закрыто с доказательством.**

- Пилот: Picard-итерация в `step_dt` (снапшот гейтов на старте шага, гейты при
  текущем `v`, Hines-решение, повтор до сходимости < 1e-4 мВ, ≤ 8 итераций)
  — «встроенная якобианная линеаризация» из старого «Дальше».
- **Результат эксперимента на ks3-кабеле (5l): fully-implicit УХУДШАЕТ parity**
  — max |ΔV| вырос с 0.014 до 0.155 мВ (сома) / 1.11 мВ (кончик). Физика:
  неявные гейты открывают K-шунт сильнее на новом (более деполяризованном) V
  → занижают V; NEURON fixed-step так НЕ делает.
- **Вывод (зафиксирован в доке `step_dt`)**: NEURON `secondorder=0` оценивает
  rates на V **начала шага** — states semi-implicit, ровно как pillars.
  Fully-implicit гейты имели бы смысл только против CVODE-эталона (которого у
  нас нет), поэтому пункт закрыт отрицательным результатом, солвер оставлен
  semi-implicit (0.014 мВ — согласие на уровне интегратора).

**Подфаза 5n (Hines-солвер: масштабирование + 1000-компартментный parity) — ✅ сделано.**

- **Оптимизация `CableSolver`**: топология кэшируется один раз при
  конструировании (`g_parent` — проводимость к родителю, `g_children_sum` —
  сумма по детям; раньше `1/junction_resistance` с делениями пересчитывался
  на каждом шаге); буферы `diag`/`rhs`/`v_new` переиспользуются (ноль
  аллокаций на шаг), результат — zero-copy `swap(v, v_new)`. Тот же профит в
  `ActiveCableSolver` (буферы `g_extra`/`g_e_extra`). Результаты бит-в-бит не
  изменились (все parity-тесты на месте).
- `gen_cable_long_ref.py` + `passive_long_cable_matches_neuron` —
  **1001-компартментный** пассивный кабель (сома + дендрит L=100000 мкм,
  nseg=1000 при d_λ=0.1): **max |ΔV| = 3.9e-4 мВ на соме, 0.000000 на
  середине/кончике** (50–100 λ от сомы — ни дрейфа, ни накопления ошибки
  f32-элиминации). Масштаб NEURON закрыт.
- **Найденный численный факт**: слишком мелкая сетка (d_λ=0.001 → сегменты
  1 мкм) патологична — осевая проводимость в ~550× больше мембранной,
  тридиагональная матрица плохо обусловлена, f32-элиминация дрейфует
  (~2e-3 мВ/шаг). Сетка λ/10 — хорошо обусловлена. Задокументировано в
  `gen_cable_long_ref.py` и тесте.

- **Бенчмарк-покрытие**: `benches/bench_cable.rs` — end-to-end
  `load()` → `run()` (dt=0.025, 5 мс = 200 шагов, импульс 0.1 nA в сому),
  `harness = false` в Cargo.toml:

  | Бенч | Время/run | Пропускная | Примечание |
  |------|-----------|-----------|------------|
  | `cable_passive_1k` (1001 комп.) | 7.58 мс | 26.4 Melem/s | |
  | `cable_passive_10k` (10001 комп.) | 77.4 мс | 25.9 Melem/s | **линейно** (10× комп. → 10.2× время) |
  | `cable_active_1k` (NaV/KV на соме) | 7.62 мс | 26.3 Melem/s | +0.5% к пассивному (гейты амортизированы) |

  Hines-элиминация — узкое место (как и должно быть); активные каналы на
  соме при 1000 компартментов добавляют <1%.

**Дальше:**

- SIMD-элиминация Hines по уровням BFS (параллельность на уровне дерева) —
  опционально, текущий O(n) уже без аллокаций (~26 Melem/s).
- Произвольная морфология масштаба NEURON (тысячи сегментов, non-uniform) —
  корректность закрыта (5j — ветвление, 5n — масштаб), бенчмарк-покрытие
  добавлено.

**Подфаза 5o (неравномерные плотности: `channelDensityNonUniform`) — ✅ сделано.**

- `parse.rs`: `channelDensityNonUniform` + `<variableParameter segmentGroup>` с
  `<proximal>`/`<distal>` (S/cm², unit-aware) → `NonUniformChannelDensity`;
  убран из skip-list.
- `compile.rs`: плотность линейно интерполируется вдоль пути группы
  (`path_um`: proximal у корня → distal на конце, нормализация в диапазоне
  группы), затем × площадь → g_max (µS). Тот же резолв канала (HH/Markov).
- `channelPopulation` (stochastic Markov с числом каналов) — **сделано**
  отдельно, см. Фазу 9c.

**Приёмка:**
- ✅ `morpho_cell_non_uniform_density_loads_and_spikes` — без warning, канал
  реально входит в кабель (спайк), V конечен;
- ✅ `morpho_cell_non_uniform_density_gradient_matters` — смена proximal/distal
  меняет детерминированную траекторию (градиент действительно применён).

---

## Фаза 6 — синаптические кинетики ← ✅ сделано

**Что:** `expOneSynapse`/`expTwoSynapse`/`alphaSynapse`/`expThreeSynapse`
больше не сбрасываются в статический delta-синапс — их тау-константы дают
правильную временную форму, а `erev` — правильную V-зависимость (проводимость).

**Сделано:**

- `api::synapse::SynapseKinetics` — профиль expOne (одиночная экспонента) /
  expTwo (двойная, peak-нормирована к амплитуде по LEMS-формуле
  `t_peak = τ_r·τ_d/(τ_d−τ_r)·ln(τ_d/τ_r)`), `erev` (реверс-потенциал, default
  0 мВ) + `with_erev`; `validate()` (конечные/положительные τ,
  `τ_rise < τ_decay`, конечный `erev`).
- `SparseGraph` несёт per-synapse `syn_types` (`0` = static, `1..` = kinetic
  id); `Edge::with_synapse` + `Network::register_kinetics(profile, dt)`;
  `push_spikes` (seq + par) роутит прыжок в нужный delay-ring по типу.
- `Network::step()` интегрирует кинетику канонической двух-ODE формой:
  `g *= e^(−dt/τ_d)`, `x *= e^(−dt/τ_r)`, прыжок `A = weight·gbase` добавляется
  в обе; проводимость `norm·(g−x)` (expTwo) или `g` (expOne), а ток —
  `g(t)·(erev − V)` (NEURON-конвенция), `V` — мембрана на старте шага
  (semi-implicit, как derivimplicit-путь HH). Наливается на `i_in` поверх
  статического слота.
- `compile.rs` регистрирует профиль для каждого синапса с тау-константой и
  назначает `syn_type` рёбрам; `erev` парсится (`erev="…mV"`, default 0);
  синапс без `tauDecay` (как в exc↔inh-примере) остаётся статическим delta —
  обратная совместимость.
- `SynapseKinetics` — enum (expOne/expTwo/alpha/expThree). `alpha` — альфа-
  функция (`g(t) = A·(t/τ)·e^(1−t/τ)`, peak `A` на `t=τ`, двух-ODE форма
  `dg/dt=(e·a−g)/τ`, `da/dt=−a/τ`); `expThree` — два decay-компонента
  (`gbase1`/`gbase2` с `tauDecay1`/`tauDecay2`) на общем `tauRise`, по
  NeuroML2 `g = gbase1·(B−A) + gbase2·(C−A)` с wave-факторами. Интеграция —
  точная (экспоненциальный Эйлер на линейных ODE, прыжок на границе шага).

**Приёмка:**
- ✅ `kinetic_exp_one_decays_exponentially` — expOne даёт ровно `A·e^(−t/τ)`;
- ✅ `kinetic_exp_two_rises_then_decays` — expTwo стартует с 0, peak ≈ A у
  `t_peak`, затем затухает к нулю;
- ✅ `kinetic_conductance_respects_driving_force` — ток `g·(erev−V)`: 0 на
  `V=erev` (шунт), деполяризация ниже, гиперполяризация выше `erev`;
- ✅ `register_kinetics_rejects_invalid_profile` — некорректные τ отвергаются;
- ✅ `exp_two_synapse_routes_to_kinetic_engine` — end-to-end: syn_type в графе
  > 0 для expTwo, 0 для expOne-без-τ, без warning, прогон без NaN;
- ✅ `kinetic_alpha_peaks_at_tau` — alpha-функция стартует с 0, peak ≈ `A` у
  `t=τ`, и совпадает с замкнутой формой `A·(t/τ)·e^(1−t/τ)` на каждом шаге;
- ✅ `kinetic_exp_three_matches_closed_form` — expThree совпадает с замкнутой
  формой `gbase1·(B−A) + gbase2·(C−A)` на каждом шаге, стартует с 0;
- ✅ `kinetic_exp_three_zero_second_component_matches_exp_two` — expThree с
  `gbase2=0` побитово равен expTwo с `gbase=gbase1`;
- ✅ `alpha_and_exp_three_synapses_route_to_kinetic_engine` — end-to-end:
  alpha/expThree получают kinetic id, без warning, прогон без NaN;
- ✅ `test_push_spikes_routes_kinetic` — seq-push роутит по типу синапса.

**Ограничение (честно):** `erev` задаёт V-зависимость, но абсолютная шкала
проводимости — «в единицах модели» (точечные нейроны уже принимают `weight·
gbase` как raw-масштаб тока, без строгого nS→nA). `stdpSynapse` остаётся
«распознан-но-пропущен» (в самом NeuroML помечен «EXAMPLE NOT YET WORKING»).

### Фаза 6c — `doubleSynapse` (два механизма на связи) ← ✅ сделано

**Что:** `doubleSynapse` (AMPA + NMDA и т.п.) больше не пропускается — одна
связь несёт два независимых синаптических механизма.

**Сделано:**

- `parse.rs`: `DoubleSynapseSpec { id, synapse1, synapse2 }`, парсинг
  `synapse1`/`synapse2` ссылок;
- `compile.rs`: проекция с `synapse="ds"` опускается в **два ребра** на
  connection — по одному на каждый дочерний синапс, каждый со своим
  `weight·gbase`, кинетикой (`syn_type`) и пластикой (`plast_type`);
  `doubleSynapse` как синапс poisson-проекции — честная ошибка (два механизма
  на одном источнике бессмысленны);
- skip-list очищен от `doubleSynapse`.

**Приёмка:**
- ✅ `double_synapse_lowers_to_two_edges` — одна связь → два ребра
  (static-AMPA + kinetic-NMDA), оба pre→post, прогон без NaN;
- ✅ `double_synapse_as_poisson_synapse_errors` — poisson-источник + ds →
  ошибка с именем синапса.

После этого из синапсов «распознано-но-пропущено» остался только
`stdpSynapse` (сломан в самом NeuroML).

### Фаза 6b — коротко-временная пластика (`blockingPlasticSynapse`) ← ✅ сделано

**Что:** `blockingPlasticSynapse` с дочерними `tsodyksMarkramDepMechanism` /
`tsodyksMarkramDepFacMechanism` больше не пропускается — Tsodyks–Markram
депрессия/фасилитация работает на каждом edge.

**Сделано:**

- `api::synapse::PlasticityProfile` — `depression(U0, τ_rec)` /
  `depression_facilitation(U0, τ_rec, τ_fac)` + `validate()`;
- `Network::register_plasticity(profile, dt)` → 1-based id; рёбра несут
  `plast_type` (`Edge::with_synapse_plasticity`); per-edge состояние `R/U/last`
  в `build()`;
- `push_spikes` (seq + par) и poisson-scatter масштабируют доставленный вес
  на `R·U`. **Порядок ровно по LEMS:** `plasticityFactor = R·U` — derived
  variable, поэтому родительский `OnEvent` читает его **до** релеевого
  события, обнуляющего `R = R·(1−U)` — текущий спайк видит ресурс, который
  он потребляет (доставка `U0·(1−U0)ⁿ` при отсутствии восстановления).
  Восстановление — lazy: `R` релаксирует экспоненциально между спайками,
  один `powi(n)` на elapsed-шаги точен.
- `parse.rs`/`compile.rs`: синапс парсится (kind = expTwo-основа + профиль
  пластики), профиль регистрируется, рёбра получают `plast_type`; poisson-
  источники тоже несут пластику (per-source `R/U/last`).

**Приёмка:**
- ✅ `test_tm_depression_geometric_decay` — delivered = `0.5, 0.25, 0.125…`
  (U0 = 0.5, без восстановления);
- ✅ `test_tm_depression_recovers_between_spikes` — короткий `τ_rec`
  возвращает доставку к ≈ U0 между редкими спайками;
- ✅ `test_tm_facilitation_second_spike_stronger` — с `τ_fac` второй спайк
  сильнее первого (U растёт 0.1 → 0.19 → …);
- ✅ `plastic_synapse_depresses_repeated_drive` — end-to-end: статический
  пост-нейрон стреляет, депрессированный — молчит при том же пре-драйве;
- ✅ `blocking_plastic_synapse_registers_tm_profile` — end-to-end NeuroML:
  без warning, `plast_types` и `syn_types` непусты, прогон без NaN.
- ✅ `test_tm_matches_neuron_delivered_weights` — **сверка с NEURON**: та же
  последовательность доставленных весов (депрессия 0.5→0.295→…→0.153 и
  фасилитация 0.1→0.144→0.149→…) против канонического `tm.mod` (NetStim →
  `TsodyksMarkram` synapse), совпадение до <1e-4 (f32 vs f64). Эталон
  генерится `gen_tm_ref.py`.

**Методология сверки с NEURON.** У NetCon нет встроенного Tsodyks–Markram —
TM в NEURON реализуется кастомным `tm.mod` (`NET_RECEIVE` + `METHOD cnexp`
для точной экспоненты, как у pillars `powi(a_rec)`). `tm.mod` писался по
литературе (Tsodyks & Markram 1997/1998), независимо от pillars, и совпадает
с LEMS `tsodyksMarkram*Mechanism` — тройная сверка (литература ↔ LEMS ↔
NEURON-интегратор) подтверждает порядок `deliver R·U → deplete → facilitate`.

---

## Фаза 7 — Poisson/spike-массив входы ← ✅ сделано

**Что:** `spikeArray`/`spikeArrayRef`, `poissonFiringSynapse` и
`transientPoissonFiringSynapse`, а также LEMS `spikeGenerator`/
`spikeGeneratorPoisson` больше не пропускаются с warning — они работают как
внешние входы через `explicitInput`.

**Сделано:**

- `StimulusKind` (в `Stimulus::kind`): `Current` (pulseGenerator, как раньше),
  `SpikeTimes { times_ms }` (spikeArray — детерминированные времена) и
  `Poisson { rate_hz, seed }` (poissonFiringSynapse).
- `LoadedNetwork::injections_at(t_ms)` — единая точка расписания: constant-
  окна, одиночный `dt`-импульс на каждый перечисленный спайк и seeded
  pseudo-Poisson (`P(spike) = rate·dt`). Stateless: хэш
  `splitmix64(seed, stream, step) < p`, поэтому один и тот же прогон
  воспроизводим и на CPU, и на GPU, и при повторе — без мутабельного RNG.
- `compile.rs` резолвит `explicitInput` по всем типам генераторов
  (spikeArrayRef → spikeArray через `ref`; битая ссылка — честная ошибка).
  Амплитуда спайкового входа — 1.0 raw nA за спайк (`scale` — как ток).
- `transientPoissonFiringSynapse` → `Poisson` в окне `[delay, delay+duration]`;
  `spikeGenerator` → регулярный трейн (`SpikeTimes`, первый спайк на `period`);
  `spikeGeneratorPoisson` → seeded pseudo-Poisson. Все три — через тот же
  `explicitInput`-резолв.
- CPU (`run`), трассировка (`run_traced`) и GPU (`run_gpu`) применяют входы
  через общий `injections_at` — три цикла больше не дублируют расписание.

**Приёмка:**
- ✅ `to_hz_parses_rate_units` — `Hz`/`per_s`/`kHz`/`per_ms`;
- ✅ `spike_array_input_schedules_deterministic_pulses` — реф резолвится,
  импульсы только в перечисленные моменты;
- ✅ `poisson_input_deterministic_and_bounded` — p=0.5 стреляет, но не каждый
  шаг, воспроизводимо;
- ✅ `poisson_zero_rate_never_spikes` — 0 Гц молчит;
- ✅ `spike_array_ref_to_unknown_array_errors` — битый `ref` падает с ошибкой;
- ✅ `transient_poisson_input_respects_window` — Poisson только внутри окна;
- ✅ `spike_generator_inputs` — регулярный период + Poisson (bounded/детерминирован).

**Ограничение (честно):** входы инжектят одиночный `dt`-импульс тока (current-
clamp-интерпретация). Рандомные ISI-генераторы — см. Фаза 7d.

### Фаза 7b — Poisson как пресинаптический источник ← ✅ сделано

`poissonFiringSynapse` больше не ограничен `explicitInput`: в `<projection>`
с `presynapticPopulation="pois"` его спайки идут **через синапсы** (weight ×
gbase, аксональная задержка, опциональная кинетика) на пост-популяцию.

- `Network::add_poisson_source(rate_hz, dt_ms, seed)` + `connect_poisson_source`
  — внешний источник с прекомпилированным списком исходящих синапсов; каждый
  шаг `step()` делает seeded-бросок (`p = rate·dt`) и скаттерит спайк тем же
  правилом `(step + max(delay,1)) % 32`, что и `push_spikes` (static и kinetic).
- `compile.rs` резолвит `presynapticPopulation` на `poissonFiringSynapse` (один
  источник, `preCellId` индекс должен быть 0; битый — ошибка), регистрирует
  источник по id (разные проекции шарят один источник) и направляет связи.

**Приёмка:**
- ✅ `poisson_source_routes_spikes_through_synapses` — p=1 ведёт пост-нейрон к
  спайку, 0 Гц молчит;
- ✅ `poisson_source_after_build_panics` — регистрация после `build()` падает;
- ✅ `poisson_presynaptic_projection_drives_post_neuron` — end-to-end NeuroML;
- ✅ `poisson_projection_rejects_nonzero_source_index` — `preCellId != 0` → ошибка.

### Фаза 7c — динамические входы (`sineGenerator`/`rampGenerator`/`compoundInput`/`timedSynapticInput`) ← ✅ сделано

**Что:** LEMS-генераторы токов больше не игнорируются молча — работают по
точным формулам `NeuroML2CoreTypes/Inputs.xml`.

**Сделано:**

- `StimulusKind::Sine { period_ms, phase }` — `i = w·A·sin(phase + 2π·(t−delay)/period)`
  внутри окна (phase в радианах), 0 вне;
- `StimulusKind::Ramp { start_amp, finish_amp }` — линейно `start→finish` по
  окну; при `baselineAmplitude ≠ 0` компилятор дополнительно эмитит два
  `Current`-стимула (до delay и после окна) — baseline держится ровно по LEMS;
- `compoundInput` — рекурсивно резолвит детей (pulse/sine/ramp, вложенность
  до 8 уровней с защитой от циклов), каждый ребёнок становится отдельным
  стимулом, сумма × вес компаунда;
- `timedSynapticInput` — спайки по `<spike time>` через ссылку `synapse`:
  импульсы `weight·gbase` в перечисленные моменты;
- skip-list пополнен: `pulseGeneratorDL`/`sineGeneratorDL`/`rampGeneratorDL`/
  `compoundInputDL` (безразмерные), `voltageClamp`/`voltageClampTriple`
  (V-обратная связь), `doubleSynapse` — теперь честно светятся в warnings,
  а не теряются молча. Резолвер входов вынесен в `resolve_input_gen`.

**Приёмка:**
- ✅ `sine_generator_injects_sinusoid` — t=0→0, t=1→+5, t=2→0, t=3→−5, окно
  [0,10);
- ✅ `ramp_generator_injects_linear_ramp` — 0→10 за окно, вне окна тишина;
- ✅ `ramp_generator_holds_baseline_outside_window` — baseline 3 до/после, рампа
  без baseline;
- ✅ `compound_input_sums_child_generators` — pulse+sine складываются;
- ✅ `timed_synaptic_input_pulses_with_gbase` — 2·0.5 = 1.0 в t=5/15 мс.

**Ограничение (честно):** `timedSynapticInput` — delta-импульсы `weight·gbase`
(кинетика ссылаемого синапса для прямых входов не симулируется, как у
`spikeArray`). `voltageClamp` — см. Фаза 7e; `voltageClampTriple` (3 ступени)
остаётся в warnings.

### Фаза 7d — рандомные ISI-генераторы (`spikeGeneratorRandom`/`spikeGeneratorRefPoisson`) ← ✅ сделано

**Что:** последние LEMS spike-источники вышли из skip-list — входной слой
NeuroML закрыт целиком.

**Сделано:**

- `SpikeGenKind::Random { min_isi_ms, max_isi_ms }` — равномерный ISI
  `min + (max−min)·u` (LEMS `isi = minISI + MSEC·random((maxISI−minISI)/MSEC)`),
  каждый следующий спайк — `isi` от **последнего запланированного** времени;
- `SpikeGenKind::RefPoisson { rate_hz, min_isi_ms }` — максимально-энтропийное
  распределение на `[minimumISI, ∞)` со средним `1/rate` (LEMS `isi =
  minimumISI − (averageIsi−minimumISI)·log(random(1))`); при `averageIsi ≤
  minimumISI` (среднее < минимума носителя — физически невозможно) — честная
  ошибка;
- **Расписание целиком генерируется на этапе компиляции** seeded-SplitMix64
  (`SeededRng`) в `compile.rs` → готовый `SpikeTimes`. Runtime остаётся
  stateless (без interior mutability в `injections_at`), один seed → побитово
  тот же трейн, CPU/GPU/reruns сходятся. Защита от нулевых ISI и потолок
  в 10M спайков.

**Приёмка:**
- ✅ `spike_generator_random_isi_respects_bounds_and_seed` — все ISIs ∈ [5, 20]
  мс, десятки спайков за 1s, повторная загрузка побитово идентична;
- ✅ `spike_generator_ref_poisson_isi_respects_minimum_and_mean` — все ISIs ≥
  2 мс, среднее ≈ 10 мс (1/100Hz), ~500 спайков за 5s, детерминировано;
- ✅ `spike_generator_ref_poisson_mean_below_minimum_errors` — avg 1ms ≤ min
  5ms → ошибка с объяснением.

### Фаза 7e — voltage clamp (патч-кламп) ← ✅ сделано

**Что:** LEMS `voltageClamp` вышел из skip-list — первый **V-зависимый** вход
(не ток по расписанию, а обратная связь по мембране).

**Сделано:**

- `LoadedNetwork::clamps: Vec<VoltageClamp>` — `{pop, indices, schedule,
  series_resistance_mohm, weight, delay_ms, duration_ms}`;
- `LoadedNetwork::clamp_currents(t)` — читает живое V (`state(pop, 0)`, сома
  для кабеля) и возвращает `i = weight·(target(t) − V)/R` (mV/MΩ = nA, масштаб
  популяции уже в weight); `run()` применяет их до `step()` — V виден на
  старте шага (semi-implicit, как кинетика). Пропорциональный контроллер:
  при `R ≪ R_in` мембрана удерживается в ~долях мВ от target;
- `ClampSchedule::Window { target }` — простой зажим на окне;
  `ClampSchedule::Triple { conditioning, testing, return, active }` — три
  ступени: conditioning до delay, testing в окне, return **держится навсегда**;
  `active=0` выключает зажим целиком (LEMS-семантика);
- `parse.rs`: `to_mohm` (ohm/kohm/Mohm), парсинг `voltageClamp` и
  `voltageClampTriple` (conditioning/testing/returnVoltage, active);
- `compile.rs`: explicitInput на clamp → запись в `clamps` (не в `Stimulus`);
  `voltageClampTriple` убран из skip-list — входной слой закрыт полностью.

**Приёмка:**
- ✅ `voltage_clamp_pulls_membrane_to_target` — до окна покой (−70), в окне
  V ≈ target (−60, допуск 1.5 мВ), после окна релаксация к −70;
- ✅ `voltage_clamp_triple_steps_and_holds_return` — conditioning −70, testing
  −60 в окне, return −70 держится навсегда; `active=0` → покой всю дорогу.
- ✅ `voltage_clamp_matches_neuron_seclamp` — **сверка с NEURON**: plain
  `voltageClamp` (Window) против `SEClamp` (rs=200 МΩ, пас-компартмент
  C=3.2 пФ / 0.2 нС): равновесие −60.38 мВ совпадает до <0.001 мВ, max |ΔV|
  ≈ 0.47 мВ — только на быстром фронте (τ_eff≈0.6 мс vs dt=0.5 мс, полу-неявный
  vs неявный контроллер). Эталон генерится `gen_clamp_ref.py`.

**Методология сверки с NEURON.** NEURON `SEClamp` — двухступенчатый конечный
усилитель (`amp1`/`dur1`, `amp2`/`dur2`), который **выключается** после
`dur1+dur2` (`amp3` игнорируется — проверено эмпирически: amp3=−70 и −80 дают
одинаковый след). Это 1:1 с LEMS `voltageClamp` (i=0 до `delay`, ток в окне,
i=0 после). `voltageClampTriple` держит `returnVoltage` **навсегда** (LEMS
`Inputs.xml`), поэтому на SEClamp не мапится — его return-фаза сверяется
аналитически (тот же контроллер, другое расписание).

**Ограничение (честно):** GPU-цикл connscope (`run_gpu`) пока применяет
только `injections_at` — зажимы нужно продублировать там же (CPU/`run()`
уже работают); clamp — один шаг запаздывания (ток из V на старте шага).

---

## Фаза 8 — gap junctions (электрические синапсы) ← ✅ сделано

**Что:** `gapJunction` + `electricalProjection` больше не пропускаются с warning
— компилируются в симметричную омическую связь.

**Сделано:**

- `Network::connect_gap_junction(from, from_idx, to, to_idx, g)` — симметричная
  пара; в `step()` ток `I_i += g·(V_j − V_i)`, `I_j += g·(V_i − V_j)` считается
  из V на старте шага (явный по V, стабилен при `g·dt ≪ 1`). V-скратч теперь
  заполняется, если есть кинетика **или** gap junctions.
- `parse.rs` — `gapJunction` (conductance), `electricalProjection`
  (presynaptic/postsynapticPopulation, `gapJunction` ref), и связи
  `electricalConnection`/`…Instance`/`…InstanceW` (`preCellId`/`postCellId` или
  `preCell`/`postCell`, опциональный `weight`).
- `compile.rs` — резолвит проекцию в `connect_gap_junction`:
  `g = weight × conductance` (conductance из `gapJunction`, default 1.0).

**Приёмка:**
- ✅ `gap_junction_couples_two_neurons` — деполяризованный нейрон тянет второй
  вверх, сам падает (`g·ΔV` в обе стороны);
- ✅ `gap_junction_rejects_non_finite_conductance` — NaN/отрицательная g → panic;
- ✅ `electrical_projection_couples_neurons` — end-to-end NeuroML;
- ✅ `gap_junction_parses_without_warning` — больше не в warnings.

**Ограничение (честно):** проводимость — raw-единицы (как веса синапсов, без
строгого nS→nA); связь явная по V (не неявная линейная система), поэтому для
больших `g·dt` нужен мелкий шаг.

---

## Фаза 9 — graded (аналоговые) синапсы ← ✅ сделано

**Что:** `linearGradedSynapse`/`silentSynapse` + `continuousProjection` больше не
пропускаются с warning — компилируются в одностороннюю омическую связь
(`linearGradedSynapse` в NeuroML2 — дословно «behaves just like a one way gap
junction»: `i = weight·conductance·(vpeer − v)`).

**Сделано:**

- `Network::connect_graded_synapse(from, from_idx, to, to_idx, g)` — **одно-
  сторонняя** связь: в `step()` пост-нейрон получает `I += g·(V_pre − V_post)`, а
  пре-нейрон не затрагивается.
- `parse.rs` — `linearGradedSynapse` (conductance), `silentSynapse` (conductance
  0), `continuousProjection` (pre/post + `synapse` ref) и связи
  `continuousConnection`/`continuousConnectionInstance` (`preCellId`/`postCellId`
  или `preCell`/`postCell`, опциональный `weight`).
- `compile.rs` — `continuousProjection` → `connect_graded_synapse`,
  `g = weight × conductance` (conductance из синапса, default 1.0).

**Приёмка:**
- ✅ `graded_synapse_is_one_way_coupling` — пост-нейрон поднимается, пре-нейрон
  получает только leak (не тянется связью);
- ✅ `graded_synapse_rejects_non_finite_conductance` — NaN g → panic;
- ✅ `continuous_projection_is_one_way` — end-to-end NeuroML.

**Ограничение (честно):** проводимость — raw-единицы.

### Фаза 9b — сигмоидный gradedSynapse ← ✅ сделано

`gradedSynapse` (stateful) больше не «распознан-но-пропущен» — компилируется в
собственный движок. Точные формулы NeuroML2:

```text
inf(vpre) = 1 / (1 + exp(-(vpre − Vth)/delta))
tau(vpre) = 1 / (k · (1 + exp((vpre − Vth)/delta)))
ds/dt     = (inf − s) / tau            (экспоненциальный Эйлер)
I_post    = weight · conductance · s · (erev − V_post)
```

- `SigmoidGradedSynapse` (публичный профиль) + `validate()` (конечность,
  `delta != 0`, `k > 0`, `conductance >= 0`);
- `Network::register_sigmoid_graded(spec, dt_ms)` + `connect_sigmoid_graded`
  (per-connection state `s`); в `step()` интегрируется по V_pre и наливает
  ток `w·g·s·(erev − V_post)` в пост-нейрон (односторонняя).
- `parse.rs` — `gradedSynapse` (`conductance`, `delta`/`Vth`/`erev` в mV, `k`
  per_ms); `compile.rs` — `continuousProjection` с `synapse="gradedSynapse"`
  регистрирует профиль и соединяет связи.

**Приёмка:**
- ✅ `sigmoid_graded_runtime_activates_and_drives_current` — `s` растёт к `inf`,
  ток положительный, пре-нейрон не затрагивается;
- ✅ `register_sigmoid_graded_rejects_invalid` — `delta=0`/`k=0` отвергаются;
- ✅ `sigmoid_graded_synapse_drives_post` — end-to-end NeuroML.

---

## Фаза 9c — дискретные популяции каналов (`channelPopulation`) ← ✅ сделано

**Что:** `channelPopulation` (популяция из `number` дискретных ионных каналов
вместо плотности) вышел из skip-list — поддержан как **стохастический** канал.

**Семантика (сверено с LEMS `Cells.xml`):**

```text
p_open    = Π_gates q_gate^power          (детерминированная открытая доля)
n_open    ~ Binomial(number, p_open)      (биномиальная выборка)
g_eff     = single_channel_conductance · n_open
I         = g_eff · (erev − V)
```

master equation (`markov_step`, точная матричная экспонента) даёт mean-field
занятость, а открытая доля в каждом шаге — биномиальная выборка по числу
каналов. Среднее совпадает с детерминированным `channelDensity`-аналогом,
дисперсия `p_open·(1−p_open)/number` — канальный шум (∝ 1/√N), существенный
для мелких нейронов/аксонов.

**Сделано:**

- `discretize.rs`: `sample_binomial` (точный «second waiting time» / геометрический
  метод — O(успехов) для малых p); `StochasticMarkovActiveChannel` (seeded
  `StdRng`); `CablePopulation::set_stochastic_channels` ресидит каждый нейрон
  (независимые, но воспроизводимые потоки); интеграция в `step_dt` кабеля;
- `parse.rs`: `to_us` (S/mS/uS/nS/pS → µS), `IrChannelPopulation`,
  `channel_populations` в `MembraneProperties`, парсинг `channelPopulation`;
  `channelPopulationNernst` (Ca — hardcoded) → честный warning;
- `compile.rs`: резолв `channelPopulation` (`ionChannelKS` канонично,
  `ionChannelHH` опускается до Маркова) → канал на **соме** (нет
  `segmentGroup` — point current), `g_max_us = single_cond · number`, seed из
  FNV-1a хэша id.

**Приёмка:**
- ✅ `sample_binomial_mean_variance_and_determinism` — среднее/дисперсия +
  детерминизм по сиду;
- ✅ `stochastic_channel_sampling_is_noisy_and_reproducible` — p_open≈0.5,
  разные сиды дают разный шум;
- ✅ `channel_population_parses_and_nernst_warns` — парсинг + warning Nernst;
- ✅ `morpho_cell_channel_population_stochastic_noise` — без warning, два
  нейрона расходятся (шум), тот же сид → побитово тот же результат.
- ✅ `channel_population_mean_matches_neuron` — **сверка с NEURON**: среднее
  по ансамблю (N=2048 каналов × 16 seed'ов) сходится к детерминистическому
  эталону `REF_V_KS3_*` (NEURON `KINETIC` backward-Euler, генерится
  `gen_cable_ks_ref.py`): max |ΔV| ≈ 0.041 мВ (допуск 0.1).

**Методология сверки с NEURON/jNeuroML.** Детерминистический предел
`channelPopulation` совпадает с `channelDensity`: та же master equation
(тот же `p_open`), та же суммарная проводимость `single_cond·N = gbar·area`.
jNeuroML экспортирует `channelPopulation` **детерминистически** (mean-field,
`number × single_cond`), не реализуя канальный шум для произвольного KS — поэтому
эталоном служит NEURON `KINETIC`, а биномиальная выборка `n_open` — наше
расширение, валидированное по среднему (этот тест) и по дисперсии
(`sample_binomial_mean_variance_and_determinism`).

---

## Фаза 10 — Python-мост / pyNeuroML-интероп  ← ✅ ядро

**Что:** экспортировать NeuroML2-ядро pillars в Python и замкнуть интероп-цикл
«pyNeuroML → .nml → Rust-движок → сверка с NEURON» без пересборки Rust.

**Сделано:**

- `src/python/python/neuroml.rs` — PyO3-подмодуль `taas_checkmate.neuroml`:
  класс `NeuromlNetwork` (`run`, `step`, `run_trace` — трасса V(t) как
  `[steps+1, n]` float32, `state`, `spikes`, `num_populations`, `dt`,
  `duration`, `warnings`) + функции `load(xml, dt_ms, duration_ms)` и
  `load_file(path)` (LEMS `<include>` резолвится относительно файла);
- `scripts/neuroml_bridge_demo.py` — полный интероп-цикл из Python: сборка
  модели (XML или pyNeuroML-объекты, если пакет установлен) → прогон на
  Rust-движке → **сверка с NEURON `SEClamp`** (та же геометрия, что
  `gen_clamp_ref.py`): равновесие −60.38 мВ совпадает, settled max |ΔV| ≈
  0.47 мВ (та же сетка, что Rust-тест `voltage_clamp_matches_neuron_seclamp`).

**Статус моста:** компилируется, `run_trace`/`load_file`/`spikes`/`state`/`ca`/
`run_ca_trace` проверены из Python, сверка с NEURON пройдена (settled < 0.6 мВ).
Генерация `.nml` из pyNeuroML-объектов (`NeuroMLDocument` →
`NeuroMLWriter.write`) работает — `pip install pyneuroml` (1.3.x), демо
проходит полный цикл и выдаёт тот же след, что literal XML.

**Smoke-тест:** `tests/test_neuroml_bridge.py` (запускается в CI рядом с
`test_smoke_api.py`) покрывает `ca`/`run_ca_trace`/`run_trace`/контракт ошибок
`ValueError`/`channelPopulationNernst`/`load_file`.

**Осталось (опционально):** круговая автосверка всех фикстур
(`resources/validate/neuroml/*.nml`) против jNeuroML из CI-скрипта.

---

## Фаза 11 — Ca²⁺ подсистема (`channelDensityNernst` + `species`) ← ✅ сделано

**Что:** последняя реальная семантика NeuroML, закрывающая Ca-зависимые токи:
`channelDensityNernst` (обращение = Nernst от внутриклеточного [Ca²⁺]) +
`species` + `decayingPoolConcentrationModel` (тонкая оболочка с релаксацией).

**Семантика (LEMS `Cells.xml`):** концентрация = моль/м³ = мМ; Nernst
`E = (R·T/(z·F))·ln(ext/int)` (z=2, `ext <= 0` → 0 — LEMS guard); пул
`d[Ca]/dt = iCa/(2F·V_shell) − (Ca−rest)/τ`, объём оболочки — **точная
сферическая** `(4π/3)(R³−(R−d)³)` с `R = √(area/4π)`.

**Сделано:**

- `parse.rs`: `IrSpecies`/`IrConcentrationModel` (+ `initialConcentration`/
  `initialExtConcentration`), парсинг `species` и `decayingPoolConcentrationModel`;
  `channelDensityNernst` → `nernst_densities`; Nernst-warning убран из skip-list;
- `discretize.rs`: `nernst_reversal_mv` (Nernst, z=2), `NernstActiveChannel`
  (HH-gate с обращением от текущего [Ca²⁺], `ca_ext` из species),
  `CaPool` (точная экспонента ODE, кламп ≥ 0), `decaying_pool_flux_scale`
  (сферическая оболочка), поля `ca`/`ca_pools`/`nernst_channels`/`i_ca` в
  `ActiveCableSolver` и их обновление в `step_dt` (полу-неявно: iCa на старом
  V/[Ca], как гейты);
- `compile.rs`: резолв species → `CaPool` + init [Ca], `channelDensityNernst`
  → `NernstActiveChannel` на сегменты (`condDensity·area` → µS);
- тесты: аналитика (`nernst_reversal_sanity`, `decaying_pool_steady_state_and_relaxation`)
  + end-to-end (`morpho_cell_ca_nernst_and_decaying_pool`) + **NEURON-parity**
  (`ca_subsystem_matches_neuron_cacum`);
- публичный доступ к [Ca²⁺] через `Population::cable_ca` → `Network::cable_ca`
  (зеркало `CableCa`, neuron-major flat, mM — аналог `cable_voltage`), и
  Python-мост: `NeuromlNetwork.ca(pop)` (сома [Ca²⁺] по нейронам) +
  `run_ca_trace(pop)` (трасса `[steps+1, n]` float32).

**Методология сверки с NEURON.** `cachan_simple.mod` (ohmic Ca-канал,
`ica = gbar·(v−eca)`, `eca` = Nernst от глобального `cai`) + `cacum.mod`
(накопление в оболочку `area·depth` с релаксацией τ) — канонический паттерн
NEURON для Ca-субсистемы. `secondorder=0` (backward Euler) = Hines-солвер.
Единственная осознанная расходимость — **геометрия оболочки**: NEURON тонкая
`area·depth`, LEMS сферическая. При depth/R = 0.1/10 = 1e-2 объёмы различаются
на ~1% (flux-масштаб pillars на ~1% больше) — допуск по [Ca] 1.5% и по V 0.5 мВ
покрывает это с запасом (измерено: 0.9% / 0.08 мВ).

**`channelPopulationNernst`** (дискретная популяция Ca-каналов с Nernst-
обращением) поддержан: `StochasticNernstActiveChannel` (та же биномиальная
выборка `n_open ~ Binomial(N, p_open)`, что `channelPopulation`, но обращение
= Nernst и ток питает Ca-пул). Mean-field-лимит — тот же
`channelDensityNernst`, поэтому сверка **транзитивна**: mean по ансамблю
(N=2048, 16 seed'ов) сходится к детерминистике до <0.001 мВ / 0.3%
(`channel_population_nernst_mean_matches_density`); стохастическая дисперсия
`p(1−p)/N` покрыта отдельным `sample_binomial`-тестом. Полноценная
NEURON-сверка **дисперсии** остаётся открытой (нужен `StochKv`-подобный
Ca-механизм — общая стохастика произвольных KINETIC-схем в NEURON отсутствует,
как и для `channelPopulation`).

---

## Фаза 12 — OSB / экосистемные модели (`NML2_FullCell`) ← ✅ сделано

**Что:** переход от синтетических фикстур к **реальным моделям из экосистемы
NeuroML2** (шаг, который превращает «движок не врёт» в «грузит опубликованные
модели из коробки»).

- Скачан канонический пример `NML2_FullCell.nml` (мультикомпартментная
  CA1-подобная клетка: сома + 2 дендрита + шипик) + включаемый
  `NML2_SimpleIonChannel.nml` (NaV m³h) из репозитория `NeuroML/NeuroML2`
  (`examples/`) в `resources/examples/neuroml/`.
- **Найден и закрыт пробел парсера**: `<channelPopulation segment="2">` —
  одиночный сегмент (id), а не `segmentGroup`. Раньше атрибут молча
  игнорировался и популяция всегда вешалась на сому; теперь `segment`
  парсится (default `"0"`), резолвится `MorphoIR::resolve_segment` и точечная
  популяция (`channelPopulation`/`channelPopulationNernst`) крепится к
  **этому** сегменту (его проксимальному компартменту при d_λ-дроблении),
  а не к соме.
- `NML2_FullCell.nml` — библиотека компонентов (инлайн `<morphology>`/
  `<biophysicalProperties>` внутри `<cell>`, без `<network>`; это же сказано в
  комментарии файла — «не может использоваться для симуляций jLEMS/jNeuroML
  as-is»). Парсер уже понимает инлайн-форму; тест дописывает сеть во
  временную копию и гоняет весь конвейер (include-резолв + инлайн-морфология +
  `pas` + NaV density + `channelPopulation segment="2"`).

**Проверки:**
- `morpho_cell_channel_population_segment_placement` — точечная популяция на
  `segment="2"` деполяризует дистальный компартмент сильнее сомы (регрессия на
  «всегда сома»);
- `loads_neuroml_example_full_cell` — реальный файл грузится с диска и
  запускается, все 4 сегмента дискретизованы, V конечен.

---

## Фаза 13 — реальная OSB-модель CA1 с Ih: Poirazi 2003, сверка с NEURON ← ✅ сделано

**Что:** прогнать **настоящую опубликованную CA1-клетку с HCN/M-током** из
экосистемы (Poirazi–Brannon–Mel 2003, `OpenSourceBrain/Poirazi2003-CA1PyramidalCell`,
NeuroML2-экспорт) и сверить с NEURON. Это закрывает последний открытый пункт
фазы B: «HAAP/HCN на реальных OSB-моделях».

**Скачано в `resources/validate/osb/`:** реальные каналы `h.channel.nml`
(Ih, кастомные LEMS-ComponentType `Poirazi_h_tau`/`Poirazi_h_ss`),
`km.channel.nml` (M-ток, `gateHHrates`/`HHExpLinearRate` + q10ExpTemp),
`kdr.channel.nml`, `na.channel.nml`, `leak_chan.channel.nml` (без изменений) +
редуцированная клетка `poirazi_ca1_soma.nml` — **реальная морфология сомы**
(сегмент 0: d=4.58 µm, L=0.01 µm) + **реальные сомальные плотности**
(na 7, kdr 1.4, leak 5e-3, hs 1.872e-2, km 60 mS/cm²; Cm=1, Ra=0.05 kΩ·cm,
V0=−70, thresh 0).

**Найден и закрыт настоящий пробел парсера — кастомные LEMS-`ComponentType`:**
реальные модели определяют кинетику гейтов произвольными LEMS-выражениями
вместо стандартных `HHExpRate`/`HHSigmoidVariable`/`fixedTimeCourse`. Раньше
`timeCourse type="Poirazi_h_tau"` падал с «unsupported timeCourse type».
Теперь:

- `kernels::hh_channels` — `LemsExpr` (Const/V/арифметика/`exp`/`Piecewise`
  с `Case`-условиями), `VarForm = Closed(ClosedForm) | Expr(LemsExpr)`,
  `TauForm = Fixed(f32) | Expr(LemsExpr)`; `GateKinetics::TauInf` несёт
  `VarForm`/`TauForm` (SIMD-путь оценивает выражения по-лейново);
  `MarkovRate::TauInfForward/Reverse` — те же формы (lowering HH→Марков
  сохраняется для кастомного tau);
- `api/neuroml/lems.rs` (новый) — токенизатор + рекурсивный парсер
  LEMS-выражений (числа с экспонентой, `exp()`, унарный минус, Fortran-
  сравнения `.lt.`/`.le.`/`.gt.`/`.ge.`/`.eq.`) и IR `ComponentType`
  (`<Constant>`/`<Parameter>`/`<Requirement>`/`<DerivedVariable>`/
  `<ConditionalDerivedVariable>`/`<Case>`); параметры (`midpoint`/`scale`)
  биндятся из атрибутов гейта, `Requirement` в теле — честная ошибка;
- `parse.rs` — pre-pass собирает `ComponentType` (могут быть объявлены после
  канала), `parse_time_course`→`TauForm`, `parse_closed_form`→`VarForm`;
  `gateFractional`/`gateKS`-rate с кастомными формами — честная ошибка.

**Редукция (честно задокументирована в NML):** полная модель (5160 сегментов
дендритного дерева, GHK-каналы, Ca-зависимые K) не грузится движком, поэтому
взята сома с реальными каналами. Na `s`-гейт опущен: публикуемое
`no_inact="1"` делает его steady-state тождественно 1 (гейт не вносит
вклад). **Per-channel `q10Settings` теперь применяется** (Фаза 13.5):
`IrQ10::factor_at(celsius)` даёт множитель скорости гейта — `q10Fixed`
якорится в 6.3 °C, `q10ExpTemp` в свой `experimentalTemp` (jNeuroML-
семантика, совпадает с NEURON-экспортом в .mod). Температура — параметр
компиляции (`load_at`/`load_file_at`, по умолчанию 6.3 °C).

**Сверка с NEURON при физиологической 35 °C с Q10** (`gen_poirazi_ref.py` →
`poirazi_na/kdr/km/h.mod` + Q10-множитель из `celsius` в `rates()`,
`cnexp`, `secondorder=0`):

| Протокол | I (nA) | max \|ΔV\| |
|---|---|---|
| гиперполяризация (Ih sag) | −1e-7, 50–100 мс | **0.0008 мВ** |
| деполяризация (M-ток доминирует в покое) | +2e-7, 50–100 мс | **0.0007 мВ** |

**max |ΔV| ≈ 0.0007–0.0008 мВ — машинная точность**: кастомный LEMS-tau Ih,
`HHExpLinearRate` M-ток **и per-channel Q10** сводятся к NEURON cnexp до
младших знаков (f32 vs f64 за 6000 шагов). Научный бонус: M-ток (60 mS/cm²,
Q10(35 °C) ≈ 2.71) доминирует в покое сомы — клетка сидит на ≈−79 мВ, оба
протокола субпороговые (чистое V(t)-сравнение, без спайков).

**Проверки:** `loads_poirazi_ca1_soma_osb_model` (грузится без warning,
1 компартмент, V конечен), `poirazi_ca1_soma_matches_neuron` (оба протокола,
допуск 0.05 мВ — запас против 0.0007), `custom_component_type_gate_lowers_to_`
`lems_expr` (аналитика ss(−90)=0.7128, tau(−90)=46.34 мс, piecewise V≥−30→1).

---

## Фаза 13.5 — per-channel Q10 и температура компиляции ✅ сделано

**Что:** применять `q10Settings` (`q10Fixed`/`q10ExpTemp`) на уровне гейтов в
cable-пути и сделать температуру параметром компиляции, чтобы сверка с
NEURON шла при физиологической 35 °C, а не в якоре 6.3 °C.

**Реализация:**

- `parse.rs` — `IrQ10 { Fixed, ExpTemp }` + `factor_at(celsius)`
  (jNeuroML-семантика: `q10Fixed` якорится в 6.3 °C, `q10ExpTemp` в свой
  `experimentalTemp`; множитель `q10^((T−T0)/10)` — ровно то, что jNeuroML
  вставляет в `.mod`); `ResolvedChannel`/`ResolvedMarkovChannel` несут
  `q10s: Vec<f32>` параллельно гейтам; `resolve_channel`/
  `resolve_markov_channel` принимают `celsius`;
- `compile.rs` — `CableChannel` несёт `q10s`, `compile()`/
  `add_population`/`add_morpho_population` принимают `celsius` и прокидывают
  в каналы;
- `discretize.rs` — per-gate `q10s` в `ActiveChannel`/`MarkovActiveChannel`/
  `StochasticMarkovActiveChannel`/`NernstActiveChannel`/
  `StochasticNernstActiveChannel`; `step_dt` использует per-gate множители
  (глобальный `q10` из `ActiveCableSolver` убран);
- `mod.rs` — `load_at(xml, dt, duration, celsius)` / `load_file_at(path,
  celsius)`; `load`/`load_with`/`load_file` без температуры сохраняют
  прежнее поведение (6.3 °C, Q10=1) — точка-путь не задет;
- точка-путь (HHModel/ConductancePopulation) остаётся на глобальном Q10.

**Сверка:** `poirazi_ca1_soma_matches_neuron` теперь грузит модель при
`load_file_at(..., 35.0)`; NEURON-эталон (`poirazi_km.mod` + Q10 из `celsius`
в `rates()`) перегенерирован при 35 °C — M-ток ускоряется в
`2.3^((35−23)/10) ≈ 2.71` раза, сома сидит чуть глубже (−79.21 мВ).
**max |ΔV| ≈ 0.0007–0.0008 мВ — машинная точность сохраняется при честной
температуре.**

**Тесты:** `ir_q10_factor_at_convention` (якоря 6.3/23 °C → 1.0, множители
при 35 °C), parity-тест при 35 °C. Проверки: `cargo test -p pillars --lib`
(869 тестов), clippy 0, fmt, workspace + python-фича.

---

## Риски

| Риск | Митигация |
|---|---|
| fast_exp SIMD vs scalar даёт разные младшие биты | parity-тесты бит-в-бит (уже есть паттерн в `hh_channels` tests) |
| `ChannelSpec` должен жить в `'static`/Copy-контексте нейронов | хранить только `f32`-параметры и `&'static str`, без heap |
| Q10 семантика разъезжается (guard-зона vs точная) | ✅ Фаза 13.5: `IrQ10::factor_at` + тест конвенции (`ir_q10_factor_at_convention`); jNeuroML-семантика, сверка при 35 °C |
| gateKS/gateFractional | ✅ покрыты (Фаза 4): gateKS → MarkovGateSpec, gateFractional → произведение субгейтов |
| кастомные LEMS-ComponentType в реальных моделях | ✅ Фаза 13: `LemsExpr` (арифметика/exp/piecewise); `Requirement`-символы — честная ошибка |
