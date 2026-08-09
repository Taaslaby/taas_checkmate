# План миграции: всё вокруг pillars

> Статус: **ЭТАП 2 (в работе)** · Обновляется по мере прохождения фаз.
> Цель: **pillars — единственный двигатель**. kernels/core/t3/connectome/train —
> тонкие обёртки или удалены. Python переписан на единый pillars-API.

---

## Эндшпиль (целевая архитектура)

```
pillars ── единственный источник (модели, ядро, ops, math, kernels, api)
   │
   ├── core      → тонкие адаптеры (или влит в pillars)
   ├── t3        → только на pillars (слой kernels убран)
   ├── connectome→ pillars::core + pillars-connectome (legacy kinetics удалены)
   ├── train     → pillars напрямую (fused-путь)
   └── python    → один coherent модуль `taas.pillars` на pillars::api + zoo
                   + compat-алиасы для старых скриптов
kernels ── сведён к compat-шим (ре-экспорты ради Python ABI) или удалён
cuda    ── GPU-порты pillars-формул (без собственных расхождений)
```

**Принцип каждого транша:** мост → parity-тест → **удаление легаси**. Мост —
переходный приём (strangler fig), не пункт назначения.

---

## Фаза 1 — все кернелы в pillars (ВОЛНА 8)

У pillars уже есть канонические двойники почти для всего. Работа = перемост
safe-обёрток на pillars, удаление легаси-файлов, parity-тесты.

### 1a. Синапсы малые — ✅ СДЕЛАНО (2026-08-08)
| Модуль kernels | Строк | Канон pillars | Статус |
|---|---|---|---|
| `delay.rs` | 47 | `kernels/delay.rs` | ✅ мост + удалён |
| `nmda.rs` | 61 | `zoo/synapses/nmda.rs` | ✅ мост + удалён |
| `tsodyks_markram.rs` | 115 | `zoo/synapses/tsodyks_markram.rs` | ✅ мост + удалён (exp-Euler vs Euler — задокументировано) |

> ⚠️ CUDA-порты `cuda_nmda_mg_block_step` / `cuda_tsodyks_markram_step` (crates/cuda)
> — отдельные GPU-имплементации, не выровнены с каноном (GPU tsodyks, вероятно,
> всё ещё Forward Euler). Выровнять в фазе 6 вместе с NLMS-расхождением.

### 1b. Пластичность — ✅ СДЕЛАНО (2026-08-08)
| Модуль kernels | Строк | Канон pillars | Развилки |
|---|---|---|---|
| `stp.rs` (stp_step + stdp + stdp_homeostatic) | 204 | `ops/plasticity/stp.rs` (lane) + **`kernels/learning.rs::stdp_homeostatic` (новый канон)** | stp: Exp Euler + post-facilitation u + clamp [0,1] vs легаси Forward Euler + pre-spike u (документировано); stdp_homeostatic портирован бит-в-бит |
| `bcm.rs` | 424 | `kernels/learning.rs` | сигнатуры совпали, мост + удалён |
| `dopamine.rs` | 174 | `kernels/dopamine.rs` | сигнатуры совпали, мост + удалён (tail: fast_exp vs нативный exp — задокументировано) |

> Parity: `crates/kernels/tests/plasticity_parity.rs` (10 тестов) — safe == pillars
> бит-в-бит для bcm/homeo/stdp/stdp_homeo/dopamine, stp мост SIMD==tail бит-в-бит.

### 1c. Рецепторы/сенсорика/глия — ✅ СДЕЛАНО (2026-08-08)
| Модуль kernels | Строк | Канон pillars | Статус |
|---|---|---|---|
| `synapse.rs` (мульти-рецепторы) | 883 | `kernels/synapse.rs` | ✅ мост + удалён (реэкспорт `pub use pillars::kernels::synapse`) |
| `chordotonal.rs` | 369 | `zoo/sensors/chordotonal.rs` | ✅ мост + удалён (реэкспорт `pub use pillars::zoo::sensors::chordotonal`) |
| `astrocyte.rs` | 799 | `kernels/astrocyte.rs` (stateful-порт) | ✅ мост + удалён (реэкспорт `pub use pillars::kernels::astrocyte`) |

> Connectome переведён на `pillars::kernels::synapse` напрямую — конвертеры
> `kinetics_legacy_to_pillars`/`kinetics_pillars_to_legacy` **удалены** (типы теперь
> один источник). `core/neuron/astrocyte.rs` — реэкспорт pillars. Python ABI сохранён
> через реэкспорты в `kernels::neuron` (пути `neuron::synapse::*`, `neuron::astrocyte::*`
> живы). Parity: `receptor_parity.rs` 7/7 (бит-в-бит оба пути + числовые инварианты).

### 1d. Дендрит
`dendrite.rs` (1 300) → `kernels/dendrite/cable.rs` (679). Крупнейший транш.

### 1e. HH-каналы — ✅ 07.08
`hh.rs` (1 040) **удалён целиком** — весь движок (`q10_from_celsius`,
`vtrap_simd`, `ChannelMeta`/`GateType`/`build_channel_meta`, `hh_poly_step`
SIMD+scalar, `fused_hh_poly_step` SIMD+scalar) портирован бит-в-бит в
`pillars::kernels::hh_channels` (теперь spec-слой + полиномный движок).
`kernels::neuron::hh` — compat-реэкспорт `pub use pillars::kernels::hh_channels as hh`.
Parity: `kernels/tests/hh_parity.rs` 11/11 — inline legacy-эталон из git HEAD
(SIMD explicit-Euler vs scalar implicit-Euler, fused-scalar clamp v к [-100,50],
fused-SIMD clamp только Ii ±100, q10 guard-зона |c−6.3|≤0.1).
Мёртвый `spike_mask_to_i32` из `neuron/mod.rs` удалён.

### 1f. depth_wave — ✅ 08.08 (последний транш волны 8)
`neuron/depth_wave.rs` (264) **удалён целиком** (и не подключённый дубль
`crates/kernels/depth_wave.rs`). Канон — `pillars::kernels::depth_wave::GcoState`
(CSR SpMV + depth-модулированная эволюция + L2-норма). Единственная развилка:
легаси `rayon::par_iter().sum()` для нормы был **недетерминирован** — канон
считает детерминированным последовательным циклом. `neuron::depth_wave` —
compat-реэкспорт; parity `depth_wave_parity.rs` 6/6 (бит-в-бит v/u/h_real/
h_imag после каждого шага + coherence + motor_amplitudes + reset/new + speed).
**Модуль `kernels::neuron` больше не содержит ни одного канона — только
compat-реэкспорты pillars. Волна 8 завершена.**

### 2a. measure/phase/vnc — raw-дубли → pillars (08.08, этап 2)
`kernels/src/{measure,phase,vnc}.rs` (raw-pointer API, ~800 строк) **удалены
целиком** — safe-обёртки `safe::{interference, interference_mean,
phase_coherence, band_energy, winding_matrix, spike_entropy, kuramoto_prepare,
kuramoto_update, kuramoto_update_viscous, t3_update, lazy_sincos_update}`
переписаны на тонкие мосты в `pillars::kernels::{measure, phase}`.
`kernels::{measure, phase, vnc}` — compat-реэкспорты pillars.
Мёртвые локальные `load8`/`store8` из `neuron/mod.rs` удалены.
Parity: `measure_phase_parity.rs` 7/7 бит-в-бит vs inline-эталоны из git HEAD
(SIMD-ветка обязана использовать настоящий `fast_sincos_simd`: скалярный
`f32::round()` away-from-zero vs SIMD nearest-even — 1 ULP).

### 2b. signal/decode — уже pillars (08.08, этап 2 — аудит)
`kernels/src/{signal,decode}.rs` **отсутствуют** (удалены в Фазе 1 консолидации;
пути не существуют даже в git HEAD). 24 safe-обёртки — тонкие мосты с assert-ами
на `pillars::kernels::{signal, decode}` (signal: stochastic_encoder, linear_decoder,
thalamic_gate, sigmoid_gate, phase_gate_apply, t3_drive_encoder, t3_tact_generator,
echo_tap, t3_inverse_kuramoto, relu_simd, add_simd, sgd/adam_step_simd,
hebbian_update; decode: t3_complex_decoder, t3_pst_features, t3_lms_train,
nlms_train, kalman_update, adam_train, pst_decode, ring_synthesis).
Parity уже есть: `decode_anchor.rs` 8/8 (независимые скалярные эталоны),
`parity_pillars.rs` (safety_clamp, sigmoid_gate), `kernel_progon.rs`, lib.rs тесты.
Мёртвый `ion.rs` (type-state PoC, 0 потребителей) **удалён**.

### 2c. io.rs → pillars (08.08, этап 2)
`PwmEncoder`/`LinearDecoder` **перенесены в канон `pillars::kernels::input`**
(бит-в-бит; `LinearDecoder::decode` на `crate::kernels::signal::linear_decoder`;
4 новых pillars-теста). `kernels/src/io.rs` — чистый compat-слой: структуры
реэкспортируются из pillars (`pub use pillars::kernels::input::{PwmEncoder,
LinearDecoder}`), функции остаются min-адаптерами (контракт python ABI — pillars
там assert-семантика) и делегируют напрямую в `pillars::kernels::{signal, decode}`
(двойная косвенность через safe устранена). Parity: `io_parity.rs` 5/5
(бит-в-бит vs inline-эталоны из git HEAD + io-адаптеры ≡ pillars при равных
длинах).

### Фаза 4 — Единый API (08.08, этап 3)
Дизайн: `resources/docs/UNIFIED_API.md` — один фасад `pillars::prelude::*`;
kernels/core → тонкие адаптеры; потребители (connectome/t3/core/unity) на pillars
напрямую.

### 3a. Лёгкие мосты 1:1 → pillars (08.08)
`safe::{gco_step, hebbian_csr_step, kuramoto_prepare, kuramoto_update}`
**удалены** — все потребители переведены на pillars напрямую:
- `core/cpe/mod.rs` + `core/kuramoto.rs` → `pillars::kernels::phase::kuramoto_*`
- `connectome/{runner,sim}.rs` → `pillars::kernels::phase::gco_step` +
  `pillars::kernels::learning::hebbian_csr_step`
- `t3/backend.rs`, `benches/{bench,bench_all}.rs`, `kernel_progon.rs`,
  `measure_phase_parity.rs`, kernels lib.rs тесты → pillars-пути
Parity: `gco_parity.rs` переписан на inline-эталон из git HEAD (2 теста,
бит-в-бит n=8/13). Осталось в safe.rs: только `lif_step`/`izhikevich_step`/
`izhikevich_dual_step` (connectome + unity) — транш 3b.
Parity: `measure_phase_parity.rs` 7/7 — inline-эталон из git HEAD, включая
SIMD-ветку через настоящий `fast_sincos_simd` (скалярный `f32::round` даёт
1 ULP расхождение vs SIMD nearest-even — задокументировано в тесте).

---

## Фаза 2 — верхнеуровневые kernels → pillars

`ion.rs` (299), `measure.rs` (334), `phase.rs` (243), `vnc.rs` (109) → ре-экспорты
из pillars `kernels/{measure, phase, vnc}` + удаление своих реализаций.

---

## Фаза 3 — потребители на pillars напрямую

| Кто | Что сделать |
|---|---|
| `core/neuron/*` (adex, glif, haap, izh, saw, wc) | `kernels::safe::*` → `run_neuron::<Model>` напрямую (как lif/fhn/hr/atlasov) |
| `train` (6 575, 0 pillars-ссылок) | fused-путь → pillars `kernels/fused.rs` |
| `t3` | убрать промежуточный слой kernels (останется только GPU-путь) |
| `connectome` | legacy-кинектика → pillars, удалить `kinetics_legacy_to_pillars` |
| бенчи (67 сайтов) | `kernels::` → pillars напрямую |

---

## Фаза 4 — единый API

Инвентаризация `pillars::prelude` + `api/` + `zoo` → один фасад.
Карта «старое имя → новое» для всех публичных имён (kernels::safe::*, core::neuron::*, t3::*).
Никаких «ещё одного слоя поверх слоёв» — единый API **заменяет** старые поверхности.

---

## Фаза 5 — python почти заново

Сейчас: 23 активных модуля биндингов, **14 703 строки**, только 4 файла на pillars
(pillars.rs — уже прообраз: `define_population!`, zero-copy numpy, per-neuron массивы).

Цель: один coherent модуль `taas.pillars` (2-3k строк) на pillars::api + zoo,
+ compat-алиасы старых имён. `kernels/core/t3` из python убираются полностью.

---

## Фаза 6 — финал

- `kernels` → compat-шим (`pub use pillars::…`) или удалён.
- `core` → тонкие адаптеры или влит в pillars.
- CUDA-кернелы — выровнять все формулы с pillars (kalman ✅, осталось NLMS-расхождение).
- Обновить DUPLICATION_MAP.md до полного нуля дублей.
