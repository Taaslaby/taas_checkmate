# Карта дублирования: `taas-core` × `taas-kernels` × `pillars`

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
| `vtrap` | в `neuron/hh.rs` | `math/` + `kernels/hh_channels.rs:63` | Дублирование |

### 3.2 Шаги нейронов (batch SIMD) — главный дубль

`taas-kernels/src/safe.rs` (2 623 строки) содержит batch-функции `lif_step`,
`izhikevich_step`, `saw_step`, `adex_step`, `hh_poly_step`, `fused_hh_poly_step`,
`fhn_step`, `wc_step`… — это ровно та роль, которую в pillars играют
`core/population.rs` + `core/runner.rs` + `zoo/neurons/*`.

| Модель | `kernels` (легаси SIMD) | `pillars` (канон) |
|--------|-------------------------|-------------------|
| LIF | `neuron/lif.rs` (447; `lif_step`+`lif_cn_step` удалены, только `lif_at_step`+eprop) | `zoo/neurons/lif.rs` + `core/population.rs` |
| Izhikevich | `neuron/izhikevich.rs` (844; `izhikevich_step` удалён, только dual/eprop/hebbian) | `zoo/neurons/izhikevich.rs` |
| HH | `neuron/hh.rs` (1 197; LUT-инфраструктура живёт для benches) | `zoo/neurons/hh.rs` + `kernels/hh_channels.rs` |
| AdEx | `neuron/adex.rs` (213; `adex_step` удалён, только `eprop_adex_step_multi`) | `zoo/neurons/adex.rs` |
| GLIF | `neuron/glif.rs` (737) | `zoo/neurons/glif3.rs`, `glif4.rs` |
| HAAP | `neuron/haap.rs` (491) | `zoo/neurons/haap.rs` |
| SAW | `neuron/saw.rs` (362) | `zoo/neurons/saw.rs` (реэкспорт в core) |
| WC / FHN | `neuron/wc.rs`, `fhn.rs` | — (ещё не портированы) |

### 3.3 Синапсы, дендриты, глия, нейрохимия

| Домен | `kernels` | `pillars` |
|-------|-----------|-----------|
| 7 типов рецепторов | `neuron/synapse.rs` (883) | `kernels/synapse.rs` (548) + `zoo/synapses/exp2syn.rs` |
| Кабельные дендриты | `neuron/dendrite.rs` (1 302) | `kernels/dendrite/cable.rs` (679) |
| Tsodyks–Markram / STP | `neuron/stp.rs` | `zoo/synapses/tsodyks_markram.rs` + `ops/plasticity/` |
| Дофамин | `neuron/dopamine.rs` | `kernels/dopamine.rs` (219) |
| BCM / STDP / обучение | `neuron/bcm.rs`, `signal.rs` | `kernels/learning.rs` (512), `ops/learning/hebbian.rs`, `ops/plasticity/stdp.rs` |
| Астроцит | `neuron/astrocyte.rs` (799) **и** `core/neuron/astrocyte.rs` | — (в pillars нет) | Два независимых файла в двух крейтах — самая вопиющая копия |
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

- **Фаза 3 (выполнена): мёртвые batch-ядра удалены из `kernels/src/neuron/*`.**
  Удалены: `lif::lif_step` (вместе с неиспользуемым `lif::lif_cn_step`),
  `izhikevich::izhikevich_step`, `adex::adex_step` — вместе с осиротевшими тестами
  (суммарно **−1 124 строки** по git diff). Их роль полностью взяли на себя
  pillars-обёртки в `kernels/src/safe.rs` (`run_neuron::<LIFModel/IzhikevichModel/
  AdExModel>` с переупаковкой лейаутов через thread-local scratch). Актуальные
  размеры: `lif.rs` 447, `izhikevich.rs` 844, `adex.rs` 213 строк — остались только
  `lif_at_step`, eprop/dual/hebbian-варианты и `eprop_adex_step_multi`. Проверены
  все потребители (connectome, unity, train, python-биндинги, benches) — все
  используют safe-обёртки; LUT-HH-инфраструктура (`build_luts`/`hh_lut_step`/
  `GateLut`) сохранена — её используют `benches/bench.rs` и resources-примеры.
- **`core/neuron/mod.rs` уже делегирует pillars**: реэкспортирует `run_neuron`,
  `run_neuron_par`, `SimdLane`, `Model`, `LIFModel`, `IzhikevichModel`,
  `AdExModel`, `HHModel`, `SAWModel`, `chordotonal_step`, `nmda_mg_block_step`,
  `tsodyks_markram_step` из `pillars::zoo`/`pillars::core`. Core хранит только
  собственные safe-обёртки и обучение.
- **HH переехал целиком** (предыдущая миграция): `crates/core/src/neuron/hh.rs`
  удалён, `HhLutConfig`/`step_hh_lut` мертвы, connectome-раннер гоняет
  `NeuralPopulation<HHModel>` (включая CaV-T pacemaking).
- **Крупные блоки kernels уже перенесены в pillars**: `kernels/synapse.rs`,
  `kernels/dendrite/cable.rs`, `kernels/dopamine.rs`, `kernels/measure.rs`,
  `kernels/decode.rs`, `kernels/phase.rs`, `kernels/vnc.rs` имеют pillars-двойники.

---

## 5. План консолидации (по фазам)

### Фаза 0 — Заморозка
Никаких новых моделей/шагов в `taas-kernels/src/neuron/*` и `core/src/neuron/*`.
Новые нейроны — только в `pillars::zoo` (ComputeNeuron).

### Фаза 1 — Математика: единственный источник в `pillars::math`
- Перенести/оставить канонические `fast_exp`, `fast_exp_poly_simd`, `sigmoid`,
  `vtrap` только в `pillars::math`.
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
  `src/unity.rs`, `crates/cuda/examples`, `crates/train`, `crates/t3`,
  `crates/cpe`, `crates/iit`, `crates/karmac`, `src/io.rs`, `crates/taas-embodied`.
  Каждый — кандидат на pillars. Заметка: `taas-core` делает
  `pub use taas_kernels as kernels;` — фасад kernels автоматически сохранит
  работоспособность core и его потребителей без правок.

### Фаза 3 — Модели: kernels/neuron/* → pillars::zoo
- Портировать оставшиеся без pillars-двойника: WC, FHN, astrocyte, dendrite,
  atlasov_neuromotor, chordotonal (частично уже есть в pillars `zoo/sensors`).
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
