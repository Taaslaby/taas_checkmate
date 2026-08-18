//! # Состояние режима «Коннектом» по компонентам (`app::state`)
//!
//! Компоненты состояния режима — по одному на зону ответственности. Каждый
//! самодостаточен, со своим `Default`, а [`ConnectomeMode`](super::ConnectomeMode)
//! держит их вместе. UI режима (вкладки, док, рейл) тоже здесь.
//!
//! ```text
//! ConnectomeMode
//! ├── data: Option<MancData>      загруженный коннектом + строка статуса
//! ├── camera: CameraState         камера/ракурс/GPU-рендер облака
//! ├── filters: Filters            видимость (NT/класс/степень/окраска)
//! ├── sel: Selection              выбор нейрона + ego-граф + поиск
//! ├── sim: SimState               параметры и результат симуляции
//! ├── pharma: PharmaState         препарат + концентрация (Хилл-маппинг)
//! ├── live: LiveState             живой поток (вспышки в 3D)
//! └── disin: DisinState           детектор растормаживания
//! ```

use eframe::egui::{TextureHandle, Vec2};

use crate::core::color::ColorMode;
use crate::core::pharma::DrugLibrary;
use crate::io::data::Proj;
use crate::render::lod::Lod;
use crate::render::splat::PointData;
use crate::sim::{SimOutcome, SimStream};

use crate::app::connectome::ego::EgoGraph;

// ─── Камера ────────────────────────────────────────────────────
/// Камера, ракурс и шейдерный рендер облака точек.
pub(crate) struct CameraState {
    pub pan: Vec2,
    /// Last viewport size, used by search/focus actions outside the canvas.
    pub viewport: Vec2,
    pub zoom: f32,
    /// Целевой pan для плавного центрирования (double-click по нейрону).
    pub pan_target: Option<Vec2>,
    /// Разрешение запечки 2D-атласа `[w, h]` — соразмерно экранному размеру
    /// bbox данных; при его смене атлас перепекается.
    pub atlas_res: [usize; 2],
    pub proj: Proj,
    /// `true` = орбитальное 3D-облако, `false` = плоские 2D-проекции.
    pub view3d: bool,
    pub ortho: bool,
    pub yaw: f32,
    pub pitch: f32,
    pub persp: f32,
    pub spin3d: bool,
    /// Инерция орбиты: скорость вращения после отпускания мыши.
    pub yaw_vel: f32,
    pub pitch_vel: f32,
    /// Запечённые текстуры атласов (перепекаются при смене ракурса/фильтров).
    pub atlas: Option<TextureHandle>,
    pub atlas3d: Option<TextureHandle>,
    /// LOD облака: уровни детализации (все точки / представители ячеек).
    pub lod: Lod,
    /// Шейдерный 3D-рендер (wgpu-биллборды) вместо программной запечки.
    pub use_gpu3d: bool,
    /// Visual profile for the GPU cloud.
    pub render3d: Render3d,
    /// Формат поверхности (из `RenderState`) — нужен для создания пайплайна.
    pub gpu_target_format: Option<wgpu::TextureFormat>,
    /// Pixels-per-point (HiDPI).
    pub ppp: f32,
    /// Размер точек шейдерного облака (в точках).
    pub pt_size: f32,
    /// Непрозрачность точек шейдерного облака [0..1].
    pub pt_alpha: f32,
    /// Переиспользуемый буфер сортировки GPU-облака (far → near): ёмкость
    /// удерживается между кадрами — орбита/спин не пересоздают его каждую
    /// секунду (раньше ~8 МБ аллокаций × 60 кадров/с при вращении).
    pub splat_scratch: Vec<(f32, PointData)>,
    /// Кеш GPU-облака: ключ последней сборки. Пока ключ не изменился и нет
    /// живого сигнала, пересортировка и пересборка до 120K точек пропускаются
    /// («тихий» кадр тратит только memcpy точек в callback).
    pub splat_cache: Option<SplatKey>,
    /// На следующем кадре канваса сделать мгновенный «fit» (центр мозга по
    /// фактическому размеру канваса). Ставится при загрузке данных.
    pub fit_pending: bool,
}

/// Ключ кеша GPU-облака: всё, что влияет на **набор точек** (проекция, LOD,
/// фильтры, трасса, выбор). Пан/зум/размер/непрозрачность точки в ключ не
/// входят — они живут в uniform'ах и не требуют пересборки точек.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct SplatKey {
    pub yaw: u32,
    pub pitch: u32,
    pub persp: u32,
    pub lod_level: usize,
    pub show_nt: [bool; 7],
    pub class_filter: i8,
    pub min_deg: usize,
    pub color_mode: ColorMode,
    /// 0 = трассы нет, иначе — адрес слайса `first_fire` (меняется при новом симе).
    pub trace_ptr: usize,
    pub trace_steps: usize,
    pub selected: Option<usize>,
}

impl Default for CameraState {
    fn default() -> Self {
        Self {
            pan: Vec2::ZERO,
            viewport: Vec2::new(800.0, 600.0),
            zoom: 1.0,
            pan_target: None,
            atlas_res: [0, 0],
            proj: Proj::Xz,
            view3d: false,
            ortho: true,
            yaw: 0.6,
            pitch: -0.3,
            persp: 0.6,
            spin3d: false,
            yaw_vel: 0.0,
            pitch_vel: 0.0,
            atlas: None,
            atlas3d: None,
            lod: Lod::empty(),
            use_gpu3d: true,
            render3d: Render3d::Spheres,
            gpu_target_format: None,
            ppp: 1.0,
            pt_size: 7.0,
            pt_alpha: 0.85,
            splat_scratch: Vec::new(),
            splat_cache: None,
            fit_pending: false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Render3d {
    Spheres,
    Points,
    Neon,
}

impl Render3d {
    pub(crate) const ALL: [Self; 3] = [Self::Spheres, Self::Points, Self::Neon];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Spheres => "Сферы",
            Self::Points => "Точки",
            Self::Neon => "Neon",
        }
    }
}

// ─── Фильтры ───────────────────────────────────────────────────
/// Видимость нейронов на атласе (NT-свачи, класс, минимальная степень, окраска).
pub(crate) struct Filters {
    pub show_nt: [bool; 7],
    pub class_filter: i8,
    pub min_deg: usize,
    pub color_mode: ColorMode,
}

impl Default for Filters {
    fn default() -> Self {
        Self {
            show_nt: [true; 7],
            class_filter: -1,
            min_deg: 0,
            color_mode: ColorMode::Nt,
        }
    }
}

// ─── Выбор ─────────────────────────────────────────────────────
/// Выбранный нейрон, его ego-граф и поиск.
pub(crate) struct Selection {
    pub hover: Option<usize>,
    pub selected: Option<usize>,
    /// Сырые рёбра ego: (индекс соседа, вес, направление out=true).
    pub ego: Vec<(usize, f32, bool)>,
    /// Интерактивный ego-граф (egui_graphs) для инспектора.
    pub ego_graph: EgoGraph,
    /// Для какого выбора ego-граф уже показан — при смене выбора
    /// сбрасываем persist-состояние раскладки (иначе FR «переезжает»).
    pub ego_shown_key: Option<usize>,
    /// Режим ego-графа: `false` = силовая раскладка (Fruchterman–Reingold),
    /// `true` = реальные 2D-координаты.
    pub ego_layout_coords: bool,
    /// Для какой проекции собран ego-граф (режим координат).
    pub ego_shown_proj: Proj,
    pub search: String,
    pub search_status: String,
    pub minimap_drag: bool,
}

impl Default for Selection {
    fn default() -> Self {
        Self {
            hover: None,
            selected: None,
            ego: Vec::new(),
            ego_graph: egui_graphs::Graph::new(petgraph::stable_graph::StableGraph::default()),
            ego_shown_key: None,
            // Anatomical coordinates are deterministic and readable by
            // default. Force layout remains available as an explicit mode.
            ego_layout_coords: true,
            ego_shown_proj: Proj::Xz,
            search: String::new(),
            search_status: String::new(),
            minimap_drag: false,
        }
    }
}

// ─── Симуляция ─────────────────────────────────────────────────
/// Параметры и результат симуляции (локальной или полного VNC).
pub(crate) struct SimState {
    /// Индекс модели в `sim::models::MODELS` (0 = LIF, первая в реестре).
    pub model: usize,
    pub bias: f32,
    pub steps: usize,
    pub pulse: f32,
    pub gh: f32,
    pub result: Option<SimOutcome>,
    pub status: String,
    /// Полный VNC (вся сеть) вместо локального ego-контура.
    pub full: bool,
    /// Отрисовывать heatmap-трассу первого спайка.
    pub show_trace: bool,
}

impl Default for SimState {
    fn default() -> Self {
        Self {
            model: 0, // LIF — первая в sim::models::MODELS
            bias: 20.0,
            steps: 800,
            pulse: 0.6,
            gh: 0.3,
            result: None,
            status: String::new(),
            full: false,
            show_trace: false,
        }
    }
}

// ─── Фармакология ──────────────────────────────────────────────
/// Выбранный препарат и его концентрация (Хилл-маппинг в `core::pharma`).
pub(crate) struct PharmaState {
    /// Разобранная библиотека препаратов (загружается один раз).
    pub library: DrugLibrary,
    /// Индекс выбранного препарата в `library.drugs`.
    pub drug: usize,
    /// Активен ли препарат (иначе концентрация = 0, масштаб = 1.0).
    pub active: bool,
    /// log₁₀ концентрации в µM: −2 (0.01 µM) .. 3 (1000 µM).
    pub log_conc: f32,
}

impl PharmaState {
    /// Концентрация в µM: `0.0` если препарат выключен.
    pub fn conc_um(&self) -> f32 {
        if self.active {
            10f32.powf(self.log_conc)
        } else {
            0.0
        }
    }
}

impl Default for PharmaState {
    fn default() -> Self {
        Self {
            library: DrugLibrary::load(),
            drug: 0,
            active: false,
            log_conc: 0.0, // 1 µM
        }
    }
}

// ─── Живой сигнал ──────────────────────────────────────────────
/// Потоковая симуляция («живой сигнал»): фоновый поток + свечение нейронов.
pub(crate) struct LiveState {
    /// Фоновый поток симуляции + приёмник событий (см. `sim::SimStream`).
    pub stream: Option<SimStream>,
    /// Текущее «свечение» нейронов [0..1] — спарк от потоковой симуляции.
    pub glow: Vec<f32>,
    /// Сколько шагов потока уже отыграно (для счётчика в UI).
    pub steps_done: usize,
    /// Темп показа анимации: шагов в секунду.
    pub speed: f32,
    pub status: String,
}

impl Default for LiveState {
    fn default() -> Self {
        Self {
            stream: None,
            glow: Vec::new(),
            steps_done: 0,
            speed: 90.0,
            status: String::new(),
        }
    }
}

// ─── Растормаживание ───────────────────────────────────────────
/// Результаты поиска цепей `тормоз →⊣ тормоз →⊣ нейрон`.
#[derive(Default)]
pub(crate) struct DisinState {
    pub chains: Vec<(usize, usize, f32, f32)>,
    pub status: String,
}

#[cfg(test)]
mod tests {
    use super::PharmaState;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    /// Выключенный препарат всегда даёт нулевую концентрацию — масштаб 1.0.
    #[test]
    fn conc_um_is_zero_when_inactive() {
        let p = PharmaState {
            active: false,
            log_conc: 3.0, // даже при максимуме шкалы
            ..Default::default()
        };
        assert_eq!(p.conc_um(), 0.0);
    }

    /// Активный препарат: log₁₀ → µM (log 0 = 1 µM, log 3 = 1000 µM).
    #[test]
    fn conc_um_converts_log_scale() {
        let p = PharmaState {
            active: true,
            log_conc: 0.0,
            ..Default::default()
        };
        assert!(approx(p.conc_um(), 1.0), "log 0 → 1 µM");

        let p = PharmaState {
            active: true,
            log_conc: 3.0,
            ..Default::default()
        };
        assert!(approx(p.conc_um(), 1000.0), "log 3 → 1000 µM");

        let p = PharmaState {
            active: true,
            log_conc: -2.0,
            ..Default::default()
        };
        assert!(approx(p.conc_um(), 0.01), "log −2 → 0.01 µM");
    }
}
