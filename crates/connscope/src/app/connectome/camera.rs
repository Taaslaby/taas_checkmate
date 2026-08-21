//! # Камера и проекции (`app::camera`)
//!
//! Всё, что переводит «мир» коннектома в экранные координаты и обратно:
//!
//! * **pan/zoom** — `to_screen` / `screen_to_world` / `world_rect` (2D-канвас);
//! * **3D-ракурс** — `proj_point` (проекция нейрона через `atlas::project3d`),
//!   `set_camera` (пресеты Дорсаль/Вентраль/…), `eff_persp` (орто vs перспектива);
//! * **навигация** — `focus_on` (центрировать камеру на нейроне),
//!   `minimap_rect` / `minimap_pan` (клик по миникарте = прыжок).

use eframe::egui::{self, pos2, vec2, Rect, Vec2};

use crate::app::connectome::ConnectomeMode;
use crate::render::atlas::project3d;

impl ConnectomeMode {
    /// Мировые координаты → экранные (с учётом zoom и pan).
    pub(crate) fn to_screen(&self, w: Vec2) -> Vec2 {
        w * (self.camera.zoom * crate::app::BASE) + self.camera.pan
    }

    /// Экранные координаты → мировые.
    pub(crate) fn screen_to_world(&self, s: Vec2) -> Vec2 {
        (s - self.camera.pan) / (self.camera.zoom * crate::app::BASE)
    }

    /// Прямоугольник видимого «мира» для канваса данного размера.
    pub(crate) fn world_rect(&self, canvas: Vec2) -> Rect {
        let a = self.screen_to_world(Vec2::ZERO);
        let b = self.screen_to_world(canvas);
        Rect::from_min_max(pos2(a.x, a.y), pos2(b.x, b.y))
    }

    /// Экранная позиция нейрона в текущем виде: 3D-орбита или осевая проекция.
    pub(crate) fn proj_point(&self, i: usize) -> Vec2 {
        if self.camera.view3d {
            let d = self.data.as_ref().unwrap();
            project3d(
                d.xs[i],
                d.ys[i],
                d.zs[i],
                self.camera.yaw,
                self.camera.pitch,
                self.eff_persp(),
            )
            .0
        } else {
            self.data.as_ref().unwrap().world(i, self.camera.proj)
        }
    }

    /// Прыжок камеры в стандартный ракурс (3D); сбрасывает кеш текстуры.
    pub(crate) fn set_camera(&mut self, yaw: f32, pitch: f32) {
        self.camera.yaw = yaw;
        self.camera.pitch = pitch;
        self.camera.atlas3d = None;
    }

    /// Общий переключатель 2D↔3D (топбар, рейл, меню «Вид»): при входе в 3D
    /// кадрирует мозг по экрану и сбрасывает кеши текстур. Раньше три места
    /// дублировали эту логику и успели разъехаться (меню кадрировало по
    /// размеру попапа, а не окна).
    pub(crate) fn toggle_3d(&mut self, ctx: &egui::Context) {
        self.camera.view3d = !self.camera.view3d;
        if self.camera.view3d {
            let size = ctx.input(|i| {
                i.raw
                    .screen_rect
                    .map_or(egui::vec2(1560.0, 960.0), |r| r.size())
            });
            self.enter_3d(size);
        }
        self.camera.atlas = None;
        self.camera.atlas3d = None;
    }

    /// Вход в 3D с автокадрированием: центрирует мозг на канвасе и подбирает
    /// zoom под меньшую сторону, чтобы облако целиком было в окне. Без этого
    /// pan/zoom из 2D переносятся в 3D как есть — после фокуса на нейроне
    /// облако оказывается «за окном», а ego-стрелки рисуются по пустоте.
    pub(crate) fn enter_3d(&mut self, canvas: Vec2) {
        self.camera.view3d = true;
        let side = canvas.x.min(canvas.y).max(200.0);
        self.camera.zoom = (0.82 * side / crate::app::BASE).clamp(0.3, 60.0);
        // Центр мозга (0.5, 0.5) → центр канваса.
        self.camera.pan = canvas * 0.5 - vec2(0.5, 0.5) * (self.camera.zoom * crate::app::BASE);
        self.camera.atlas = None;
        self.camera.atlas3d = None;
    }

    /// Шаг зума хоткеями «+/−»: мгновенно × factor, якорь — центр канваса.
    /// Размер канваса нужен, чтобы точка под центром оставалась на месте.
    pub(crate) fn zoom_step(&mut self, factor: f32, canvas: Vec2) {
        let anchor_world = self.screen_to_world(canvas * 0.5);
        let anchor_screen = canvas * 0.5;
        self.camera.zoom = (self.camera.zoom * factor).clamp(0.3, 60.0);
        self.camera.pan = anchor_screen - anchor_world * (self.camera.zoom * crate::app::BASE);
    }

    /// Сброс камеры «Fit/reset» (кнопка и хоткей F). В 2D — фит по bbox мозга
    /// (`fit_now`), в 3D — мир [0,1]³ по меньшей стороне + стартовый ракурс.
    /// Мгновенно (без «догоняющей» анимации — вид предсказуем).
    pub(crate) fn reset_camera(&mut self) {
        let canvas = self.camera.viewport.max(Vec2::new(400.0, 300.0));
        if self.camera.view3d {
            let zoom = (0.84 * canvas.x.min(canvas.y) / crate::app::BASE).clamp(0.3, 60.0);
            self.camera.zoom = zoom;
            // Центр мозга (0.5, 0.5) → центр канваса.
            self.camera.pan = canvas * 0.5 - vec2(0.5, 0.5) * (zoom * crate::app::BASE);
            self.camera.yaw = 0.6;
            self.camera.pitch = -0.3;
            self.camera.yaw_vel = 0.0;
            self.camera.pitch_vel = 0.0;
            self.camera.atlas3d = None;
        } else {
            self.fit_now(canvas);
        }
        self.camera.pan_target = None;
    }

    /// 2D-bbox видимых точек в текущей проекции; если данных/точек нет —
    /// весь мир [0,1]². Нужен, чтобы «fit» центрировал сам мозг, а не пустой
    /// квадрат мира: после изотропной нормализации мозг занимает лишь часть
    /// [0,1]² (0.51:0.63:1), и фит по [0,1]² оставлял его мелким и «кривым»
    /// в центре пустого канваса.
    fn proj_bbox(&self) -> Rect {
        let Some(d) = self.data.as_ref() else {
            return Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
        };
        let mut min = Vec2::splat(f32::MAX);
        let mut max = Vec2::splat(f32::MIN);
        let mut any = false;
        for i in 0..d.n {
            if !d.has_pos(i) {
                continue;
            }
            let w = d.world(i, self.camera.proj);
            min = min.min(w);
            max = max.max(w);
            any = true;
        }
        if any {
            Rect::from_min_max(pos2(min.x, min.y), pos2(max.x, max.y))
        } else {
            Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0))
        }
    }

    /// Bbox данных с полями (~4%) для запечки 2D-атласа: крайние точки не
    /// обрезаются на границе текстуры. Считается по всем позициям независимо
    /// от фильтров — атлас не «прыгает» при смене NT/класса/степени.
    pub(crate) fn atlas_bbox(&self) -> Rect {
        let bbox = self.proj_bbox();
        let side = bbox.size().x.max(bbox.size().y);
        let pad = (side * 0.04).max(0.02);
        Rect::from_min_max(
            pos2(bbox.min.x - pad, bbox.min.y - pad),
            pos2(bbox.max.x + pad, bbox.max.y + pad),
        )
    }

    /// Мгновенный «fit» (без анимации): вписывает bbox мозга в канвас с
    /// полями, сохраняя пропорции (единый масштаб — без растягивания).
    /// Используется на первом кадре после загрузки данных и в «Fit/reset».
    pub(crate) fn fit_now(&mut self, canvas: Vec2) {
        let bbox = self.proj_bbox();
        let span = bbox.size().x.max(bbox.size().y).max(1e-3);
        let zoom = (0.84 * canvas.x.min(canvas.y) / (span * crate::app::BASE)).clamp(0.3, 60.0);
        let center = bbox.center().to_vec2();
        self.camera.zoom = zoom;
        self.camera.pan = canvas * 0.5 - center * (zoom * crate::app::BASE);
        self.camera.pan_target = None;
    }

    /// Эффективная сила перспективы: ортографический режим гасит её полностью.
    pub(crate) fn eff_persp(&self) -> f32 {
        if self.camera.ortho {
            0.0
        } else {
            self.camera.persp
        }
    }

    /// Центрирует камеру на нейроне (если у него есть 3D-позиция).
    /// Используется поиском и консолью: `select <idx>` ведёт глаз к нейрону.
    pub(crate) fn focus_on(&mut self, i: usize) {
        let Some(d) = self.data.as_ref() else { return };
        if !d.has_pos(i) {
            return;
        }
        let w = d.world(i, self.camera.proj);
        let center = self.camera.viewport * 0.5;
        let c = self.screen_to_world(center);
        self.camera.pan += (w - c) * (self.camera.zoom * crate::app::BASE);
        self.camera.pan_target = None;
    }

    /// Наводит камеру атласа на окрестность выбранного нейрона (ego-bbox):
    /// центрирует и подбирает zoom, чтобы все ego-соседи были видны.
    /// «Координация» ego-графа и канваса — одна и та же проекция.
    pub(crate) fn focus_ego(&mut self) {
        let Some(d) = self.data.as_ref() else { return };
        let Some(sel) = self.sel.selected else { return };
        let mut ids: Vec<usize> = vec![sel];
        for &(t, _, _, _) in &self.sel.ego {
            if !ids.contains(&t) {
                ids.push(t);
            }
        }
        let (mut minx, mut maxx, mut miny, mut maxy) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        let mut any = false;
        for &i in &ids {
            if !d.has_pos(i) {
                continue;
            }
            let w = d.world(i, self.camera.proj);
            minx = minx.min(w.x);
            maxx = maxx.max(w.x);
            miny = miny.min(w.y);
            maxy = maxy.max(w.y);
            any = true;
        }
        if !any {
            return;
        }
        let cx = (minx + maxx) / 2.0;
        let cy = (miny + maxy) / 2.0;
        let span = (maxx - minx).max(maxy - miny).max(1e-3);
        // ~28% ширины канваса занимает окрестность.
        self.camera.zoom = (0.28 / span).clamp(0.3, 60.0);
        self.camera.pan =
            self.camera.viewport * 0.5 - vec2(cx, cy) * (self.camera.zoom * crate::app::BASE);
    }

    /// Прямоугольник миникарты в правом нижнем углу канваса.
    pub(crate) fn minimap_rect(&self, canvas: Vec2) -> Rect {
        Rect::from_min_size(
            vec2(
                canvas.x - crate::app::MINIMAP.x - 14.0,
                canvas.y - crate::app::MINIMAP.y - 14.0,
            )
            .to_pos2(),
            crate::app::MINIMAP,
        )
    }

    /// Прыжок камеры по позиции на миникарте (мировой коэффициент 0..1).
    pub(crate) fn minimap_pan(&mut self, canvas: Vec2, pointer_screen: Vec2) {
        let mm = self.minimap_rect(canvas);
        let wp =
            ((pointer_screen - mm.min.to_vec2()) / mm.size()).clamp(Vec2::ZERO, vec2(1.0, 1.0));
        self.camera.pan = canvas * 0.5 - wp * (self.camera.zoom * crate::app::BASE);
    }
}
