# IMPROVEMENTS.md — карта улучшений (архитектурный обзор 2026-08-17)

> Док для ИИ-агента и автора. Словарь: **depth** (интерфейс прост, реализация насыщена),
> **deletion test** (удаление концентрирует сложность — хорошо), **seam/adapter**
> (шов и тонкая обёртка над ним), **leverage** (одна правка улучшает многое),
> **locality** (понять концепцию можно не выходя из модуля).
> HTML-версия с диаграммами: `%TEMP%\architecture-review-20260817-024449.html`.

## Приоритетный порядок работ

## ✅ connscope (2026-08-17)

- **Фасовка**: `console.rs` → `app/console.rs`; `io/raw.rs` (598) → `io/raw/{mod,graphml,fafb_npz}.rs`;
  18 тестов из main.rs разложены по модулям (фикстура `tiny_data` — `#[cfg(test)] pub(crate)`
  в io/data.rs); main.rs 349→49 строк. 31→31 тестов (позже 33).
- **Код**: `pillars-cuda` опционален за фичей `cuda` (дефолтная сборка без CUDA-toolchain;
  алиас `GpuNetwork` — единственная точка типа; UI без cuda честно показывает CPU-only).
  Парсер GraphML отрефакторен в `from_graphml_str` + 2 юнит-теста (33 passed total).
- Осталось по коду: тесты npz-лоадера (нужен zip-writer фикстура — zip уже в зависимостях);
  при желании — разбор крупных GUI-файлов (plots.rs 771, inspector.rs 594).
- **Модульность (воркбенч)**: `app/` переструктурирован в `shell/` (top/menu/status/
  console/tabs/shortcuts/profiler) + режимы `connectome/` (canvas/ego/inspector/sim_ui/
  plots/state/camera/picking/plotlib) + `neuroml/` + общий `tokens.rs`. ConnApp пока
  god-объект — при добавлении первого нового режима (scripts/nodes) извлечь трейт `Mode`
  (`fn show(&mut self, ctx, &mut ShellServices)`), не раньше.
- **Токены**: аудит показал — UI уже полностью токенизирован (spacing/padding/icons/
  widths/swatch). Оставшиеся литералы контентные (FontId painter-подписей, ширины
  StatCard, размеры окон) — легитимно вне шкалы. Цветовых повторов 3+ нет.
- **Дизайн (вёрстка профессиональнее)**: новая `shell/theme.rs` — полная тема:
  IBM Plex Sans Regular/SemiBold (кириллица) + JetBrains Mono (данные) из
  `resources/fonts/`; SemiBold — отдельная FontFamily `heading` (в egui 0.35 нет
  весов, `.strong()` = лишь яркий цвет), хелперы `tokens::heading/eyebrow`;
  палитра тёмного «прибора»: панели ступенью яркости, полупрозрачные рамки,
  corner_radius 6, один сигнальный акцент #45C99B; типошкала Heading 16/Body
  13.5/Small 12/Mono 12.5; мягкие тени. Заголовки инспектора/статуса — на
  `tokens::heading`. Clippy 0, 33 теста OK. Осталось: перевести остальные
  `.strong()`-заголовки (top/menu/sim_ui) и моно для body_id по мере касания.
- **Проход 2026-08-17 (рендер/вьювер, A–D)**:
  - **A. 2D-атлас**: `build_atlas` переписан — float-накопление (Σ цвет·вес,
    Σ вес) вместо аддитивного RGB-clamp; гауссов сплат-кёрнел (σ=0.9 текселя,
    табулированное ядро), одиночная точка = круглая точка; тонмаппинг плотности
    `alpha = 1 − exp(−0.8·Σw)`, цвет = средневзвешенный, композит над тем же
    фоном (14,14,24). Новый параметр `splat_radius` (тексели). Canvas: перепек
    атласа при пересечении power-of-two бакетов магнификации тексела,
    `radius = clamp(2.5/2^bucket, 1.0, 6.0)` — экранная точка ~постоянна;
    при bucket ≥ 2 (тексел ≥ 4 px) включаются векторные круги (старое
    условие `zoom*BASE > 3.0` срабатывало всегда и рисовало 100K кругов
    даже в обзорном зуме). Тесты: старый `test_build_atlas_size` адаптирован
    под новый параметр, добавлен `test_build_atlas_gaussian_dot_and_density`.
  - **B. Вьювер**: плавный зум к курсору (~120 мс, экспо-лерп `zoom_target`,
    якорь «мир под курсором» держится каждый кадр анимации); hover-тултип 2D —
    ближайший нейрон в 8 экранных px (`nearest_screen_2d`, линейный скан),
    текст «тип / #id / body_id / NT / deg»; double-click по нейрону = плавное
    центрирование камеры (`pan_target`). Новые хоткеи: **F** = fit/reset
    (zoom=1, pan по умолчанию, в 3D — стартовый ракурс), **+/=** и **−** =
    зум к центру; не срабатывают при фокусе в текстовом поле. Кнопки Fit view
    в top/меню переведены на общий `reset_camera()`.
  - **C. Ego-граф**: силовая раскладка стартует с реальной анатомии
    (data.world → 2D по ego-bbox), кольцо — фолбэк для нейронов без позиции;
    `labels_always(true→false)` — подписи только у hovered/selected/dragged
    (проверено по egui_graphs 0.31: метка видна при hover); толщина рёбер ~
    весу через `with_edge_stroke_hook` (0.8–3.2 px + подсветка выбранного);
    FR-параметры спокойнее: dt 0.04, damping 0.25, max_step 8.0.
  - **D. 3D-шейдер (splat)**: дистанционный туман к цвету фона
    (`fog = (1−depth)²·0.55`) — глубинный кью; rim смягчен (0.35→0.22,
    степень 2.5→2.0). trace/sel/glow/boost не тронуты.
  - Верификация: `cargo check/test/clippy -p connscope` — 0 ошибок/ворнингов,
    34 passed (+1 новый тест).
- **Проход 2026-08-17 (UX-раскладка Blender/Adobe)**: вьюпорт освобождён от
  «тулкита сверху» — рамка как в модульных редакторах:
  ```text
  ┌──────────────────────────────────────────────────────┐
  │ меню · ТОП-БАР (1 строка): логотип · режимы · поиск  │
  │                    · Fit/3D/справка                  │
  ├──┬───────────────────────────────────────────┬───────┤
  │R │                                           │ ИНСПЕ │
  │A │            VIEWPORT (атлас / ego)         │ КТОР  │
  │I │                                           │ (рес.,│
  │L │───────────────────────────────────────────│ свор.)│
  │  │ ДОК: Плоттер / Консоль / Симуляция       │       │
  ├──┴───────────────────────────────────────────┴───────┤
  │ СТАТУС: контекст-подсказка · hover · fps · счётчики  │
  └──────────────────────────────────────────────────────┘
  ```
  Новые файлы: `shell/rail.rs` (левый рейл 44 px: XZ/XY/YZ, 3D, попапы
  окраски/фильтров/3D-камеры, Fit; активная кнопка — акцентом #45C99B,
  тултипы с хоткеями), `shell/panels.rs` (нижний док: вкладки Плоттер/
  Консоль/Симуляция, сворачивается в полоску 26 px, ресайз персистентен;
  `section_header` — переиспользуемый сворачиваемый заголовок секции).
  Карта переездов: проекции+ракурсы+орто/GPU/точки → рейл (попап «3D»);
  НТ-свачи+класс+мин.степень → рейл (попап «Фильтры»); окраска → рейл
  (попап «🎨»); вкладки Атлас/Ego + поиск → топ-бар; консоль (было окно) →
  док-вкладка; графики сима (была панель при результате) → док-вкладка
  «Плоттер»; статус загрузки → статус-бар (теперь: контекст-подсказка по
  режиму 2D/3D/ego/сим, справа статус+zoom+n/s+fps EMA). Меню: консоль и
  библиотека плоттеров — пункты «Вид», новая модалка «Горячие клавиши».
  Новые хоткеи: **H** = рейл вкл/выкл, **[** / **]** = вкладки дока,
  **?**/**F1** = модалка хоткеев. Верификация: check/test/clippy зелёные,
  34 passed + 1 ignored (3 старых ворнинга в plots.rs/sim — не из этого прохода).

## ✅ Выполнено 2026-08-17

- **№1 (частично)**: удалены `core/context_ops.rs` (534 стр.) и `ops/synapse.rs`
  (160 стр.) — мёртвые пути, 0 потребителей. Осталось: zoo-синапсы как адаптеры
  над `kernels::SynapticKinetics` (3 реализации → 2, полная унификация — дизайн-задача).
- **№2**: создан `tools/taas_io.py` (load_taas/write_taas/load_haap/load_hhparams/load_mod,
  все с проверкой magic). Мигрированы 13 скриптов, `py_compile` OK. Пропущены
  `bake_haap.py`/`bake_physio.py` (load_mod вложена в main() — миграция не окупается).
- **№3**: все 16 ручных ParamIdx/StateIdx-энумов удалены, ~35 файлов мигрированы на
  генерируемые `<Model>ParamIdx`/`<Model>StateIdx` (варианты = имена полей). Порядок
  колонок везде совпал. Попутно: `examples/vnc_indices.inc.rs` → `examples/support/`
  (невалидное имя crate'а ломало `--all-targets`).
- **№4 (частично)**: `pillars.rs` 2231 → 1479 строк (−34%): 8 ручных Py-классов
  переведены на существующий макрос `define_population!` (расширен extra:-блоком и
  enum-индексами), двойной список регистрации классов дедуплицирован. VSN/Transformer
  оставлены ручными (уникальные конструкторы). Осталось: скелеты `py_step_*` в
  `neuron.rs` и root/facade списки в `mod.rs` (содержательно разные, слияние рискованно).
- Верификация: `cargo check --workspace --all-targets` — 0 ошибок;
  `cargo test -p pillars --lib` — 680 passed; `python smoke_pillars.py` — PASS.
- **№5, фаза 1 (инкрементальный вариант)**: удалён `src/unity.rs` (1839 стр.,
  0 потребителей) + фича `unity` из Cargo.toml/CI/INDEX.md; `step_izhikevich*`
  переехали из sim/lif.rs в sim/izhikevich.rs (пути реэкспорта стабильны);
  heap-репак в `step_lif_pillars` заменён на стековые чанки (численно идентично).
  Починен падающий тест `worker_runs_chain_with_delay` (30 шагов = 3 мс не хватало
  на 2 задержки + нарастание τ_syn; воркер корректен, тест увеличен до 100 шагов).
  `hetero_brain` не тронут (торчит в Python API). Осталось: фаза 2 — python-обёртки
  (connectome_brain/hetero_brain) на BrainRunner; karmac/brain.rs; двойной стек в
  runner.rs (legacy BrainState vs pillars-native Brain).
- **№5, фаза 2**: удалён `PyBrainSimulator` (−801 стр.; connectome_brain.rs 1477→902).
  Уникальные методы (step_batch_dual, gco_record, step_batch_gpu{_izhikevich})
  перенесены на PyBrainRunner как тонкие делегаты к conn::sim_* (GPU — под тем же
  cfg cuda). Все 13 скриптов уже сидели на BrainRunner — ничего не сломалось.
  README/README_SAKHA/INDEX/API_REFERENCE обновлены. Smoke PASS, .pyd обновлён.
  Python теперь имеет один симулятор мозга. Осталось по №5: hetero_brain (736,
  python-виден, скриптами не используется), karmac/brain.rs (1057), legacy
  BrainState-стек внутри runner.rs.
- **№5, фаза 2b**: удалён `hetero_brain.rs` (736 стр.) + `make_brain`/класс из
  регистрации (скриптами/доками не использовался; проверено grep по .py/.md).
  Smoke PASS. `karmac/brain.rs` НЕ трогаем: `SpikingBrain` — живой внутренний
  интерфейс трёх karmac-арен (arena/hide_seek/platformer), легитимный адаптер
  в изолированном модуле.
- **Гигиена скриптов**: все исследовательские скрипты собраны в `scripts/` (18 шт.,
  включая bake_*), в корне только `setup.py` + `smoke_pillars.py`. Во все импортеры
  `tools.taas_io` добавлен path-bootstrap (после переезда `from tools.taas_io import`
  ломался — sys.path указывал на scripts/, а не на корень). Удалён `read_results.py`
  (анализ внешнего fly-brain-main CSV, которого в репо нет). `bake_manc_viz` оставлен —
  на него ссылаются тесты connscope (формат MNCV 1:1); `bake_generic` — универсальный
  пекарь, тоже живой. Двоичные halfcenter_* — разные эксперименты (CPG-антифаза vs
  load-reset), оставлены как исследовательский журнал VNC-линии.

- **№5 остаток (отдельная кампания)**: legacy `BrainState` внутри runner.rs —
  85 использований self.state, семейства run_lif/run_izh/run_gco/run_rate/
  run_lif_alpha держат состояние в BrainState, а HH/HAAP — в NeuralPopulation.
  План: модель за моделью переводить run_* на NeuralPopulation (как HH/HAAP),
  после каждого — parity-тест «BrainState vs Population» на одинаковых весах
  (паттерн: sim_lif_delegate_matches_legacy_counts). НЕ делать одним махом —
  калибровки vnc_*.py завязаны на численное поведение.

### 1. [Strong] Удалить мёртвые пути + одна синаптическая кинетика  ← НАЧАТЬ ОТСЮДА

### 5. [Worth exploring] Четыре «симулятора мозга» → один канон
- `unity.rs` (1839), `python/connectome_brain.rs` (1477), `karmac/brain.rs` (1057),
  `python/hetero_brain.rs` (736) — каждый по-своему оборачивает pillars-connectome,
  с дублированной HH/LIF-инициализацией и калибровкой.
- Внутри pillars-connectome два стека: legacy `sim::*`/BrainState vs pillars-native
  `Brain`/`SynapticBrain`, сшитые монолитом `runner.rs` (2258). `sim/lif.rs` репакует
  legacy params во временные Vec на каждый шаг (признано «переходным» в комментарии);
  туда же misplaced Izhikevich-функции.
- Канон: `BrainRunner` над `Brain`/`SynapticBrain`; unity/pyo3/karmac/hetero — тонкие
  адаптеры. Parity/NEURON-тесты защищают один симулятор вместо четырёх.

### 6. [Worth exploring] Двойной CUDA-FFI-стек в `pillars-cuda`
- Старый `ffi.rs/kernel.rs/engine.rs` (1726) vs новый `gpu/` (~5000: context 1690,
  signal 2206, kernels 305, ffi 818). Новый потребитель (connectome `sim/gpu.rs`)
  сидит только на `gpu::GpuContext`. Проверить, кто на старом, мигрировать, удалить.

### 7. [Speculative] Feature-гигиена фасада
- `src/iit.rs` (1066) — безусловный `pub mod iit` в lib.rs:74, python-биндинг мёртв
  (закомментирован). За фичу или удалить.
- `python`/`wasm` фичи тянут karmac целиком, включая игровые арены
  (`hide_seek.rs` 2072, `platformer.rs` 1035). Сплит: `karmac-core` vs `karmac-arenas`.

## Тестовые дыры (попутно)
- `api/builder/` — 0 тестов в 7 из 8 файлов.
- `api/neuroml/` — 1 из 3 файлов с тестами.
- `kernels/`: hh_channels (953), cable (842), learning (735), astrocyte (629),
  phase (645), signal (594) — без встроенных тестов.

## Не трогать (проверено, это сильные места)
- `SimdLane` (1 метод = 1 инструкция, SIMD/scalar consistency-тесты) — ядро в порядке.
- Дисциплина unsafe (6 известных файлов, всё выше — safe).
- `#[dual_neuron]` → NVRTC: дублирования математики CPU/GPU нет, parity-тесты есть.
- `karmac/` — изолирован и хорошо протестирован (~80 тестов).
- `api/compact.rs`, `api/push.rs`, `api/network.rs` — глубокие модули, не shallow.
