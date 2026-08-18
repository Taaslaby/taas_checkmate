# neoMyelin (connscope): дизайн-система и API

Справочник для контрибьюторов. Кодовая часть — это `crates/connscope` (бинарь
`connscope`, продукт **neoMyelin**); `lib.rs` пустой, поэтому «API» ниже — это
внутренние границы модулей (`pub(crate)`), контракт режима `Mode` и CLI.

---

## Часть 1. Дизайн-система

### 1.1. Философия

Интерфейс — **научный прибор** (осциллограф/микроскоп), а не сайт:

- **Один сигнальный акцент** — холодный «люминофорный» зелёный. Он тратится
  только на активное/живое: выделение, активную вкладку, идущую симуляцию.
  Всё остальное — тихие нейтральные серые. Никакой конкурирующей палитры.
- **Разделение панелей яркостью, а не рамками** — поверхности различаются
  ступенью фона (`panel_fill` → `window_fill` → inset), жёстких границ нет.
- **Типографика = иерархия**: SemiBold-заголовки живут отдельным
  `FontFamily` («heading»), потому что в egui нет весов, а `.strong()` — это
  лишь более яркий цвет, не настоящий вес.

### 1.2. Токены (`app::tokens`)

Единая шкала вместо разбросанных magic-чисел. База сетки — **4 px**.

**Спейсинг:**

| Токен | px | Назначение |
|---|---|---|
| `SP_XS` | 4 | минимальный зазор (иконка ↔ текст) |
| `SP_SM` | 6 | тесный вертикальный зазор (label ↔ plot) |
| `SP_MD` | 8 | базовый: `item_spacing`, зазор между карточками |
| `SP_LG` | 12 | паддинг карточки, зазор между крупными блоками |

**Правила применения:**

- в **строке** (чипы, кнопки) — `SP_MD` (`item_spacing`);
- **между** секциями/карточками — `SP_MD..SP_LG`;
- паддинг карточки — `PAD_CARD` (12) / `PAD_CARD_SM` (8);
- **иконки Phosphor — только через `icon()`/`icon_label()`** с размером из
  `ICON_*` (13/15/20), никогда инлайном глифа в текст: глифы несут внутренний
  паддинг, и «склеиваются» с текстом без двойного пробела из `icon_label`.

**Геометрия оболочки:** `STATUS_BAR_H=24`, `DOCK_DEFAULT_H=256`,
`DOCK_MIN_H=120`, `INSPECTOR_DEFAULT_W=360` (min 280 / max 640).

**Тулбар:** `TOP_BAR_H=40`, ширина комбо `TOOL_COMBO_W=110`, слайдер
`TOOL_SLIDER_W/H=120/15`, поиск `TOOL_SEARCH_W=200`, горизонтальный зазор
`TOP_GAP_X=SP_SM`.

**Радиусы:** `RADIUS_XS=1` (сигнальная метка активного инструмента) ·
`RADIUS_SM=3` (свотч легенды) · `RADIUS_TB=4` (тулбарные кнопки) ·
`RADIUS_MD=6` (карточки, кнопки-действия, стандартные виджеты).

**Типографика:** `TEXT_HEADING=16` (SemiBold) · `TEXT_BODY=13.5` (тело и
кнопки) · `TEXT_SMALL=12` (подписи) · `TEXT_MONO=12.5` (данные). Единый
источник размеров для `TextStyle`-шкалы в `theme` и паинтерных подписей
виджетов — не хардкодить `13.5` по файлам.

**Контролы:** `CONTROL_H=30` — высота кнопки-действия (`action_button`).

**Хелперы:** `spacing(ui)` — применить `item_spacing=8×6` первой строкой в
каждом `Panel/Window::show`; `heading(rich_text)` — перевести заголовок в
SemiBold-семейство; `eyebrow(text)` — маленький приглушённый ярлык секции
(категория данных, **не** заголовок — без SemiBold).

### 1.3. Тема (`app::shell::theme`)

**Палитра (тёмный «прибор»):**

| Токен | Значение | Роль |
|---|---|---|
| `ACCENT` | `#5AD6B2` | сигнальный акцент (выделение/живое) |
| `BG_PANEL` | `#151718` | базовая поверхность |
| `BG_CARD` | `#1C2021` | карточки, доки, окна |
| `BG_INSET` | `#0E1011` | поля ввода, «утопленные» области |
| `STROKE_SOFT` | `#313738` | тихая рамка-разделитель |
| `TEXT` / `TEXT_DIM` | `#E6E6E6` / `#999999` | основной / второстепенный текст |
| `BG_CTRL` / `BG_CTRL_HOVER` / `BG_CTRL_ACTIVE` / `BG_CTRL_OPEN` | `#252A2B` / `#313839` / `#3A4343` / `#2A3031` | состояния стандартных контролов (кнопка/комбо: покой/hover/нажатие/открыт) |
| `DANGER` | `#A84444` | приглушённый красный — ошибки и «стоп» (единственное цветное исключение, источник `ActionTone::Danger`) |
| `WARNING` | `#E0B84A` | приглушённый янтарный — предупреждения (валидация, пропущенные элементы) |

Фон — нейтральный тёмно-серый (**R=G=B**, без синевы: синий подтон «дёшевил»).

**Типографика:**

- IBM Plex Sans Regular — тело (`Proportional`);
- IBM Plex Sans SemiBold — заголовки, семейство `HEADING_FAMILY="heading"`;
- JetBrains Mono — данные (консоль, body_id, числа, индексы).

Шкала (`tokens::TEXT_*`): Heading 16 / Body 13.5 / Button 13.5 / Small 12 / Monospace 12.5.

**Установка:** `theme::install(ctx)` вызывается один раз из `app::run` после
Phosphor. Вставляет шрифты поверх дефолтных (не `set_fonts` — ничего не
затирается) и собирает `Visuals`/`Style` на `egui::Theme::Dark`.
Наборы глифов шрифтов не пересекаются — порядок вставки не важен.

### 1.4. Акценты (`AccentPreset`)

6 пресетов поверх любой из тем elegance: `Blue`, `Violet`, `Gold`, `Green`,
`Rose`, `Cyan`. Механика — перекраска трёх полей `Palette`:

- `focus` — структурный акцент (активные табы, рамки фокуса, слайдеры);
- `blue` / `blue_hover` — семантический акцент главных кнопок.

`Blue` — возврат к родному виду (палитру не трогает). Дешёвая операция:
`theme.install` пропускает повторное применение, если тема не менялась.

### 1.5. Виджет-кит (`app::widgets`)

Примитивы, не знающие про `ConnApp`, — работают только с `ui` + данными:

- `section_card(ui, icon, title, body)` — карточка с SemiBold-заголовком и
  `PAD_CARD` (единственная карточка во всём приложении);
- `compact_card(ui, icon, title, sub, body)` — плоты: `PAD_CARD_SM`, заголовок
  без SemiBold, опциональный подзаголовок;
- `key_value_fields(ui, &[(key, value, tooltip)])` — сетка «ключ: значение»
  одним неразрывным `LayoutJob`; ключ — `strong_text_color` (не сливается с
  фоном ни на одной теме);
- `badge_grid(ui, cols, badges)` — бейджи ровной сеткой, каждый растянут на
  равную долю ширины;
- `legend_swatch(ui, name, color, count)` — цветной квадрат рисуется краской
  (символа «██» нет в шрифте);
- `action_button` / `full_width_action` — единая кнопка-действие, тон
  `ActionTone::{Primary, Neutral, Danger}`. Рисуется **нашим** паинтером и
  совпадает с акцентом приложения: Primary = заливка `ACCENT`, Neutral =
  тихий фон + рамка, Danger = приглушённый красный (только «стоп»);
- `tool_button` / `icon_button` — Blender-style «инструментные» кнопки рейла/
  тулбара: активное состояние = рамка + узкая сигнальная метка слева (видна
  периферийным зрением), не яркая заливка.

**Правило акцента в действиях:** Primary резервируется под явный запуск/
подтверждение — в карточке не больше одного CTA.

---

## Часть 2. API

### 2.1. Карта модулей

```
src/
├── main.rs     вход: GUI (app::run) или CLI (cli::run), panic-хук в .log
├── app/        GUI (egui/wgpu): рамка (shell) + режимы (connectome/neuroml)
│   ├── mode.rs    трейт Mode + ShellServices — контракт режима
│   ├── tokens.rs  дизайн-токены (часть 1)
│   ├── widgets/   виджет-кит (часть 1.5)
│   ├── shell/     menu, status, shortcuts, profiler, theme, console(_ui)
│   ├── connectome/  ConnectomeMode: camera, canvas, picking, ego, inspector,
│   │                sim_ui, plots, plotlib, state
│   └── neuroml/    NeuromlMode: ui (модалка), cable (морфология), editor (IDE)
├── render/     atlas (CPU-запечка 2D/3D) · lod (LOD) · splat (GPU-облако)
├── io/         data (MancData + .bin/.taas) · raw (GraphML/npz/csv.gz)
├── core/       nt (НТ/классы) · color (режимы окраски) · pharma (препараты)
├── sim/        models (плагины нейронов) + прогоны · neuroml (NeuroML-мост)
└── cli.rs      CLI-комбайн (отчёты переиспользуются консолью GUI)
```

Зависимости односторонние: `io/core/render/sim` не знают про `app`.

### 2.2. Контракт режима (`app::mode`)

Трейт `Mode` — единственная точка расширения: **новый режим = новая структура
+ `impl Mode`, оболочка не меняется.** Режим владеет своим состоянием и рисует
свои поверхности; оболочка (`ConnApp`) вызывает хуки в фиксированном порядке
кадра:

```text
frame_tick → view_menu → top_bar → rail
→ status_left/status_right → windows → dock_surface
```

Хуки (пустые реализации по умолчанию почти у всех):

| Хук | Назначение |
|---|---|
| `frame_tick(ctx)` | камера, живые потоки, `request_repaint` |
| `top_bar(ui, services)` | одна строка под меню (вкладки, поиск, действия) |
| `rail(ui)` | левый рейл инструментов |
| `dock_surface(ui, surface, services)` | контент вкладки дерева egui_dock |
| `status_left() -> String` | левая часть статус-бара (hover-инфо) |
| `status_right(ui)` | правая (счётчики, зум, статус сима) |
| `windows(ctx, frame, services)` | плавающие окна/модалки режима |
| `view_menu(ui) -> Option<Surface>` | режимная секция меню «Вид» + запрос фокуса |
| `handle_shortcuts(ctx) -> bool` | горячие клавиши режима |
| `load_data(path)` | Файл → Открыть / Ctrl+R |
| `data_stats() -> (n, s)` | статистика для «О программе» |
| `set_gpu_target_format(fmt)` | формат поверхности wgpu (для splat) |

Рабочее пространство — **единое дерево `egui_dock`**, которым владеет оболочка:
`Surface::{Inspector, Atlas, Ego, Plots, Console, Sim, Morphology}`. Инспектор /
атлас / ego / плоттер / консоль / сим рисует `ConnectomeMode`, «Морфологию» —
оболочка из `NeuromlMode`. Пресеты раскладки — меню «Вид → Раскладка».

`ShellServices` — канал «режим → оболочка»: флаг `show_shortcuts` (модалка
«Горячие клавиши») + `focus_surface` (запрос фокуса вкладки). Режим не видит
`ConnApp`.

Разделение хоткеев: **оболочка** — `F1/?` (справка), `Ctrl+O/R` (файлы),
`Ctrl+Shift+D` (профайлер), `F12` (консоль), `[ ]` (цикл вкладок дока);
**режим** — `F` (fit), `+/−` (зум), `H` (рейл), `Ctrl+S` (поиск).

### 2.3. Режим «Коннектом» (`app::connectome`)

`ConnectomeMode` — состояние вьюера, разбитое на компоненты с `Default`:

| Поле | Тип | Ответственность |
|---|---|---|
| `data` | `Option<MancData>` | загруженный коннектом + `status` |
| `camera` | `CameraState` | камера/ракурс/GPU-рендер (`atlas`, `atlas3d`, `lod`, `splat_cache`) |
| `filters` | `Filters` | видимость (`show_nt[7]`, `class_filter`, `min_deg`, `color_mode`) |
| `sel` | `Selection` | выбор, hover, ego-граф, поиск |
| `sim` | `SimState` | параметры/результат локального и полного прогона |
| `pharma` | `PharmaState` | препарат + концентрация (Hill) |
| `live` | `LiveState` | живой поток (glow, stream, speed) |
| `disin` | `DisinState` | растормаживание |

Ключевые механики состояния:

- **`SplatKey`** — ключ кеша GPU-облака: квантованные `yaw/pitch/persp`
  (u32-биты), `lod_level`, фильтры, адрес слайса трассы, выбор. Пан/зум/размер
  точки в ключ **не** входят (живут в uniform'ах) — «тихий» кадр пропускает
  пересборку и CPU-сортировку до 120K точек.
- **`Render3d`** — `Spheres | Points | Neon` (объёмные сферы / точки / неон).
- **`PharmaState::conc_um()`** — `10^log_conc` µM, `0` когда препарат выключен
  (тогда `scale = 1.0` — «нет препарата»).

### 2.4. Режим «NeuroML» (`app::neuroml`)

`NeuromlMode` — загрузка и прогон NeuroML (CPU SIMD / GPU CUDA). Переопределяет
только `windows()`. Состояние: `loaded: Option<LoadedNetwork>`,
`result: Option<RunResult>`, `backend`, `cable: Option<CableState>` (геометрия
морфологии + `CableTraces`), `editor: EditorState` (подсветка XML + валидация).

### 2.5. Данные (`io::data::MancData`)

Центральная модель в памяти. Поля: `xs/ys/zs` (нормализованные `[0,1]³`,
`has_pos`), `nt/klass/leg/side` (метаданные), `nt_types`, `body_id`,
`type_idx/subclass_idx/target_idx` + таблицы строк, `row_ptr/col/val` — CSR
синапсов, опциональный обратный `rev_*`.

**Методы:**

- `MancData::detect(path)` — выбор формата: по расширению `.graphml`/`.npz`,
  по magic `MNCV` (.bin) / `TAAS` (.taas);
- `load(path)` / `load_taas(path)` — запечённые форматы;
- `n()` / `s()` — нейроны / синапсы;
- `nt(i)` / `klass(i)` / `type_name(i)` / `subclass(i)` / `target(i)` /
  `leg_name(i)` / `side_name(i)` — метаданные;
- `world(i, Proj)` — экранные координаты проекции (`Proj::{Xy,Xz,Yz}`);
- `out_deg(i)` / `in_deg(i)` — степени (обратный CSR строится лениво в
  `ensure_reverse`);
- `ego_edges(node, k) -> Vec<(target, weight, out)>` и `ego_edges_agg(...) ->
  Vec<(target, weight, out, n_synapses)>` — топ-K соседей; agg-версия
  суммирует мультисинапсы одной пары.

**Сырые форматы (`io::raw`):** `from_graphml(path)` (networkx/igraph,
опционально x/y/z; без координат — синтетическая сфера Фибоначчи) и
`from_fafb_npz(path)` (zip/npy CSR + `*_positions.npy` + flywire-аннотации
`csv.gz`, джойн по root-id).

### 2.6. Домен (`core`)

- **`nt::Nt`** — `Unknown/Ach/Gaba/Glut/Da/Ser/Oct`; `color()`, `name()`,
  `inhibitory()` (GABA + GLUT — у мухи GluCl тормозной Cl⁻-канал);
- **`nt::Klass`** — `Other/Motor/Sensory/Ascending/Descending/Intrinsic/Glia/
  Efferent` (коды 0..7);
- **`color::ColorMode`** — `Nt | Klass | Type | Degree`; `auto(d)` выбирает
  первый режим с данными (НТ → тип → класс → степень); `color_for(d, i, mode)`.
  `Type` — золотой угол от индекса типа (соседние типы не сливаются);
  `Degree` — лог-тепловая карта (единственный режим, который никогда не
  «весь серый»);
- **`pharma`** — `Target` (каналы `NaV/KV/CaT/Leak/Hcn/M` + рецепторы
  `GabaA/Nach/GluCl`), `Effect` (`Block/Open/Antagonize/Agonize`), `Drug`
  (Hill: `occupancy(c)` → `scale(c)`, `scale(0)==1`), `DrugLibrary` (загрузка
  встроенного `resources/pharma/drugs.toml`, `get`/`by_target`).

### 2.7. Рендер (`render`)

Три пути от дешёвого к дорогому; мировая система координат — `[0,1]³`,
камера — `yaw` (Z) + `pitch` (X) + перспектива:

- `atlas` — CPU-запечка 2D/3D в `ColorImage` (гауссовы сплаты,
  `alpha = 1−exp(−0.8·Σw)` против выжигания плотных скоплений);
- `lod` — Potree-стиль: ~15K представителей вместо 139K;
- `splat` — GPU (wgpu): биллборд-квады + объёмные сферы, CPU-сортировка
  far→near, diffuse/Phong/rim + туман. `PointData` — ровно 64 байта с
  compile-time `assert!` на рассинхрон раскладки Rust↔WGSL (vec3 выравнивается
  на 16 байт).

### 2.8. Симуляция (`sim`)

**Плагины нейронов (`sim::models`):** трейт `NeuronModel`
(`short_name/name/config/tune/has_h_current`), реестр
`MODELS = [Lif, Izh, Hh, Haap]`, `index_of_name`. Новый нейрон = один файл +
одна строка в `MODELS` (индекс хранится в состоянии, никакого enum/match).

**Прогоны:**

- `run_simulation(data, nodes, drive, model, bias, steps, pulse_frac, dt_ms,
  g_h) -> SimResult` — локальный контур; `SimResult{idx, raster, rates,
  vtraces, steps, dt_ms, chunk, drive_local, model}`;
- `run_full_sim(...) -> FullSimResult` — вся сеть; `FullSimResult{activity,
  class_activity[8], activated, total_spikes, active_neurons, first_fire,
  spike_counts, spike_times}` (последние два — для ISI-гистограммы и фазового
  анализа моторов);
- `run_full_sim_stream(...) -> SimStream` — тот же прогон потоком:
  `StreamMsg::{Fired{step, fired}, Done(Box<FullSimResult>)}`, отмена через
  `SimStream::stop()` (флаг `cancel` + join);
- `find_disinhibition(data, src, top) -> Vec<(mid, y, w1, w2)>` — цепи
  `тормоз →⊣ тормоз →⊣ нейрон`;
- `induced_subgraph(data, nodes) -> (idx, ConnectomeData)` — CSR-подграф.

`SimOutcome::{Local, Full}` — суммарный итог (хранится в `SimState`).

### 2.9. CLI и консоль

CLI-комбайн (`cli::run`) и встроенная консоль GUI (F12) используют **одни и те
же отчёты-строки**:

```
connscope stats   <file>
connscope search  <file> <запрос>
connscope ego     <file> <idx> [--k 10]
connscope sim     <file> --drive <idx> [--model lif|izh|hh|haap] [--steps 200] [--bias 20] [--pulse 0.6] [--gh 0.3]
connscope fullsim <file> --drive <idx> [--model lif] [--steps 200] [--bias 30] [--pulse 0.6]
connscope disin   <file> <idx> [--top 10]
connscope help
```

`cli::flag_val/flag_drive/parse_model/class_name` — общие парсеры флагов,
переиспользуемые консолью.

### 2.10. Как добавить новый режим

```rust
struct ScriptsMode { /* своё состояние */ }
impl Default for ScriptsMode { /* ... */ }
impl Mode for ScriptsMode {
    fn central(&mut self, ui: &mut egui::Ui) { /* ... */ }
    // остальные поверхности — по необходимости
}
// в ConnApp::default:
mode: Box::new(ScriptsMode::default()),
```

Оболочка (`shell/`) не меняется вообще.

### 2.11. Как добавить новый нейрон

`crates/connscope/src/sim/models/foo.rs` (реализовать `NeuronModel`) + одна
строка в `MODELS`. Всё: UI-переключатель, CLI `--model`, консоль и строки
статуса подхватят модель автоматически.
