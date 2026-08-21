//! # Отрисовка атласа (`app::canvas`)
//!
//! Центральный канвас: нейроны рисуются не попиксельно каждый кадр, а
//! **запекаются в текстуру** (`build_atlas` / `build_atlas_3d*`) и
//! перепекаются только при смене ракурса/фильтров/трассы.
//!
//! * `draw_canvas` — диспетчер: 2D-проекция или 3D-облако, поверх — стрелки
//!   ego-графа, точки при зуме, маркеры hover/выбора, миникарта, гизмо;
//! * `draw_grid` — координатная сетка (только 2D);
//! * `draw_minimap` — миниатюра всего мозга + рамка видимой области;
//! * `draw_gizmo` — оси X/Y/Z, вращающиеся вместе с камерой (3D);
//! * `draw_arrow` — кривая Безье-стрелка связи ego-графа.

use eframe::egui::{
    self, pos2, vec2, Align2, Color32, CornerRadius, FontId, PointerButton, Pos2, Rect, Sense,
    Stroke, Vec2,
};
use egui_wgpu::Callback as WgpuCallback;

use crate::render::atlas::{build_atlas, build_atlas_3d, build_atlas_3d_trace, rotate_dir};
use crate::render::splat::{PointData, SplatCallback, Uniforms};
use crate::sim::SimOutcome;

use crate::app::connectome::state::SplatKey;
use crate::app::connectome::ConnectomeMode;

impl ConnectomeMode {
    /// Центральный канвас атласа: обработка ввода (зум/пан/клик/вращение 3D),
    /// hover, затем отрисовка. Живёт во вкладке «Атлас».
    pub(crate) fn atlas_canvas(&mut self, ui: &mut egui::Ui) {
        let avail = ui.available_size();
        let canvas_h = (avail.y - 60.0).max(200.0);
        let (resp, painter) = ui.allocate_painter(vec2(avail.x, canvas_h), Sense::click_and_drag());
        let origin = resp.rect.min;
        let size = resp.rect.size();
        self.camera.viewport = size;
        // Первый кадр после загрузки данных: мгновенный фит по реальному
        // размеру канваса (иначе стартовый вид — «угол» в мире [0,1]²).
        if self.camera.fit_pending {
            self.fit_now(size);
            self.camera.fit_pending = false;
        }
        let pointer_screen = resp.interact_pointer_pos().map(|p| p - origin);
        let pointer_world = pointer_screen.map(|s| self.screen_to_world(s));

        // Зум под курсором (scroll): мгновенный, с точным якорем «мир под
        // курсором». Без сглаживания — сглаженный зум «догонял» цель после
        // скрола, и вид жил своей жизнью. Каждый кадр скролла умножает zoom
        // ровно на дельту, а панорама пересчитывается так, чтобы якорь не
        // смещался.
        let zoom_delta = ui.input(|i| i.zoom_delta());
        if (zoom_delta - 1.0).abs() > 1e-4 {
            let anchor_world =
                pointer_world.unwrap_or_else(|| self.screen_to_world(size * 0.5));
            let anchor_screen = pointer_screen.unwrap_or(size * 0.5);
            self.camera.zoom = (self.camera.zoom * zoom_delta).clamp(0.3, 60.0);
            self.camera.pan =
                anchor_screen - anchor_world * (self.camera.zoom * crate::app::BASE);
            // Ручное управление отменяет плавное центрирование.
            self.camera.pan_target = None;
        }
        // Плавное центрирование (double-click по нейрону) — единственная
        // анимация: зум уже мгновенный, а панорама мягко едет к цели.
        if let Some(target) = self.camera.pan_target {
            let dt = ui.input(|i| i.stable_dt).clamp(0.001, 0.1);
            let t = 1.0 - (-dt / 0.04).exp();
            self.camera.pan += (target - self.camera.pan) * t;
            if (self.camera.pan - target).length() < 0.5 {
                self.camera.pan = target;
                self.camera.pan_target = None;
            }
            ui.ctx().request_repaint();
        }
        // Пан средней кнопкой мыши.
        if resp.dragged_by(PointerButton::Middle) {
            self.camera.pan += resp.drag_delta();
            self.camera.pan_target = None;
        }

        // Миникарта: drag/клик по ней = прыжок камеры.
        let mm = self.minimap_rect(size);
        let over_mm = pointer_screen.is_some_and(|s| mm.contains(pos2(s.x, s.y)));
        if resp.drag_started() {
            self.sel.minimap_drag = over_mm;
        }
        if resp.drag_stopped() {
            self.sel.minimap_drag = false;
        }
        // Миникарта реагирует одинаково на drag (после старта на ней) и
        // на клик — общее действие вынесено в одно условие.
        if (self.sel.minimap_drag && resp.dragged_by(PointerButton::Primary))
            || (over_mm && resp.clicked())
        {
            if let Some(s) = pointer_screen {
                self.minimap_pan(size, s);
            }
        } else if !over_mm {
            if self.camera.view3d {
                // 3D-навигация: LMB/MMB-drag = орбита, Shift+Любая = пан,
                // чистый клик (без сдвига) = выбор нейрона. Это интуитивно:
                // тянешь облако левой — оно вертится; кликнул по точке — выбор.
                let shift = ui.input(|i| i.modifiers.shift);
                if shift
                    && (resp.dragged_by(PointerButton::Middle)
                        || resp.dragged_by(PointerButton::Primary))
                {
                    self.camera.pan += resp.drag_delta();
                    self.camera.atlas3d = None;
                } else if resp.dragged_by(PointerButton::Middle)
                    || resp.dragged_by(PointerButton::Primary)
                {
                    let d = resp.drag_delta();
                    self.camera.yaw_vel = d.x * 0.006;
                    self.camera.pitch_vel = d.y * 0.006;
                    self.camera.yaw += self.camera.yaw_vel;
                    self.camera.pitch += self.camera.pitch_vel;
                    self.camera.pitch = self.camera.pitch.clamp(-1.5, 1.5);
                    self.camera.atlas3d = None;
                } else if !resp.dragged_by(PointerButton::Primary) {
                    let dt = ui.input(|i| i.stable_dt).clamp(0.001, 0.1);
                    let dec = (-dt * 8.0).exp(); // ~0.4 с на затухание
                    self.camera.yaw += self.camera.yaw_vel;
                    self.camera.pitch += self.camera.pitch_vel;
                    self.camera.pitch = self.camera.pitch.clamp(-1.5, 1.5);
                    self.camera.yaw_vel *= dec;
                    self.camera.pitch_vel *= dec;
                    if self.camera.yaw_vel.abs() > 1e-4 || self.camera.pitch_vel.abs() > 1e-4 {
                        self.camera.atlas3d = None;
                        ui.ctx().request_repaint(); // докрутка идёт без мыши
                    }
                }
                if resp.clicked() {
                    if let Some(s) = pointer_screen {
                        if let Some(n) = self.nearest_screen(s) {
                            self.select(n);
                        }
                    }
                }
            } else {
                // 2D: drag по пустому месту = пан; клик по нейрону = выбор.
                if resp.dragged_by(PointerButton::Primary) {
                    let drag_on_node =
                        pointer_world
                            .and_then(|w| self.nearest(w))
                            .is_some_and(|n| {
                                let np = self.data.as_ref().unwrap().world(n, self.camera.proj);
                                let pp =
                                    pointer_world.map(|w| self.to_screen(w)).unwrap_or_default();
                                (self.to_screen(np) - pp).length() < 12.0
                            });
                    if !drag_on_node {
                        self.camera.pan += resp.drag_delta();
                        self.camera.pan_target = None;
                    }
                }
                if resp.clicked() {
                    if let Some(w) = pointer_world {
                        if let Some(n) = self.nearest(w) {
                            let np = self.data.as_ref().unwrap().world(n, self.camera.proj);
                            if (self.to_screen(np) - self.to_screen(w)).length() < 10.0 {
                                self.select(n);
                            }
                        }
                    }
                }
                // Двойной клик по нейрону = плавно отцентрировать камеру на нём.
                if resp.double_clicked() {
                    if let Some(w) = pointer_world {
                        if let Some(n) = self.nearest(w) {
                            let np = self.data.as_ref().unwrap().world(n, self.camera.proj);
                            if (self.to_screen(np) - self.to_screen(w)).length() < 12.0 {
                                self.camera.pan_target =
                                    Some(size * 0.5 - np * (self.camera.zoom * crate::app::BASE));
                                ui.ctx().request_repaint();
                            }
                        }
                    }
                }
            }
        }

        // Hover для статус-бара и подсказки: в 3D — быстрый хит по
        // представителям LOD (~15K вместо полного скана 139K каждый кадр),
        // клик остаётся точным (полный скан).
        self.sel.hover = if over_mm {
            None
        } else if self.camera.view3d {
            pointer_screen.and_then(|s| self.nearest_rep_screen(s))
        } else {
            // 2D: ближайший видимый нейрон в ~8 экранных пикселях от курсора.
            pointer_screen.and_then(|s| self.nearest_screen_2d(s, 8.0))
        };
        if let Some(h) = self.sel.hover {
            if let Some(d) = self.data.as_ref() {
                if h < d.n {
                    let _ = resp.on_hover_text(format!(
                        "{}\n#{} · body {} · {} · deg={}",
                        if d.type_name(h).is_empty() {
                            "?"
                        } else {
                            d.type_name(h)
                        },
                        h,
                        d.body_id[h],
                        d.nt(h).name(),
                        d.out_deg(h)
                    ));
                }
            }
        }

        self.draw_canvas(&painter, origin, size);
        ui.separator();
        ui.label(egui::RichText::new(&self.status).weak());
    }

    /// Рисует весь канвас: атлас (текстура), ego-стрелки, точки, маркеры.
    pub(crate) fn draw_canvas(&mut self, painter: &egui::Painter, origin: Pos2, size: Vec2) {
        let Some(data) = self.data.as_ref() else {
            painter.text(
                origin + size * 0.5,
                Align2::CENTER_CENTER,
                &self.status,
                FontId::proportional(16.0),
                Color32::from_rgb(200, 200, 220),
            );
            return;
        };
        if !data.has_any_pos() {
            painter.text(
                origin + size * 0.5,
                Align2::CENTER_CENTER,
                "нет 3D-позиций: граф доступен через поиск, ego-граф и симуляцию",
                FontId::proportional(15.0),
                Color32::from_rgb(190, 190, 210),
            );
            return;
        }
        if !self.camera.view3d {
            self.draw_grid(painter, origin, size);
        }

        // Прямоугольник атласа в экранных координатах.
        let atlas_rect = Rect::from_min_max(
            origin + self.to_screen(Vec2::ZERO),
            origin + self.to_screen(vec2(1.0, 1.0)),
        );
        // Магнификация 2D-атласа (экранные px на тексель) — считается в ветке
        // 2D ниже, а используется в блоке векторных точек после if/else.
        let mut tex_mag = 0.0f32;
        if self.camera.view3d {
            if self.camera.use_gpu3d {
                // ⚡ Шейдерный рендер (wgpu): гауссовы сплэты, отсортированные
                // по глубине far→near — ближние перекрывают дальние, без каши.
                let rect = painter.clip_rect();
                // Scratch-буфер сортировки живёт в камере между кадрами —
                // берём, наполняем, возвращаем. Поля разные: заимствование
                // данных и камеры не пересекаются (данные передаём отдельно,
                // чтобы камера могла писать кеш, пока `data` заимствован).
                let cb = {
                    let scratch = std::mem::take(&mut self.camera.splat_scratch);
                    let (cb, scratch, cache_key) = self.make_splat_callback(data, rect, scratch);
                    self.camera.splat_scratch = scratch;
                    if let Some(k) = cache_key {
                        self.camera.splat_cache = Some(k);
                    }
                    cb
                };
                painter.add(egui::Shape::Callback(WgpuCallback::new_paint_callback(
                    rect, cb,
                )));
            } else {
                // CPU-fallback: программная запечка облака в текстуру.
                if self.camera.atlas3d.is_none() {
                    let trace = match (&self.sim.result, self.sim.show_trace) {
                        (Some(SimOutcome::Full(r)), true) => {
                            Some((r.first_fire.as_slice(), r.spike_counts.as_slice()))
                        }
                        _ => None,
                    };
                    let px = match trace {
                        Some((ff, sc)) => build_atlas_3d_trace(
                            data,
                            self.camera.yaw,
                            self.camera.pitch,
                            self.eff_persp(),
                            crate::app::ATLAS_3D,
                            &self.filters.show_nt,
                            self.filters.class_filter,
                            self.filters.min_deg,
                            ff,
                            sc,
                            self.filters.color_mode,
                        ),
                        None => build_atlas_3d(
                            data,
                            self.camera.yaw,
                            self.camera.pitch,
                            self.eff_persp(),
                            crate::app::ATLAS_3D,
                            &self.filters.show_nt,
                            self.filters.class_filter,
                            self.filters.min_deg,
                            self.filters.color_mode,
                        ),
                    };
                    let img =
                        egui::ColorImage::new([crate::app::ATLAS_3D, crate::app::ATLAS_3D], px);
                    self.camera.atlas3d = Some(painter.ctx().load_texture(
                        "manc_atlas3d",
                        img,
                        egui::TextureOptions::LINEAR,
                    ));
                }
                if let Some(tex) = &self.camera.atlas3d {
                    painter.image(
                        tex.id(),
                        atlas_rect,
                        Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                        Color32::WHITE,
                    );
                }
            }
        } else {
            // 2D-проекция: текстура печётся по bbox данных (с полями), а не по
            // всему миру [0,1]² — мозг заполняет атлас целиком, без пустых
            // полей и без растяжения (единый масштаб по обеим осям).
            let bbox = self.atlas_bbox();
            let world_scale = self.camera.zoom * crate::app::BASE;
            let screen_w = bbox.width() * world_scale;
            let screen_h = bbox.height() * world_scale;
            let long_side = screen_w.max(screen_h).clamp(256.0, 2048.0);
            let res_long = (long_side as u32).next_power_of_two().min(2048) as usize;
            // Стороны текстуры пропорциональны bbox — сплэты остаются
            // круглыми (а не эллипсами), и вид не растягивается при отрисовке.
            let (res_x, res_y) = if screen_w >= screen_h {
                (
                    res_long,
                    ((res_long as f32 * screen_h / screen_w.max(1.0)).round() as usize)
                        .clamp(64, 2048),
                )
            } else {
                (
                    ((res_long as f32 * screen_w / screen_h.max(1.0)).round() as usize)
                        .clamp(64, 2048),
                    res_long,
                )
            };
            let res = [res_x, res_y];
            if self.camera.atlas.is_none() || self.camera.atlas_res != res {
                let px = build_atlas(
                    data,
                    self.camera.proj,
                    res,
                    bbox,
                    &self.filters.show_nt,
                    self.filters.class_filter,
                    self.filters.min_deg,
                    self.filters.color_mode,
                    2.5,
                );
                let img = egui::ColorImage::new(res, px);
                self.camera.atlas = Some(painter.ctx().load_texture(
                    "manc_atlas",
                    img,
                    egui::TextureOptions::LINEAR,
                ));
                self.camera.atlas_res = res;
            }
            // Экранный прямоугольник bbox: рисуем полный UV 1:1 — без
            // UV-обрезки, из-за которой вид растягивался/плыл.
            let atlas_screen = Rect::from_min_max(
                origin + self.to_screen(bbox.min.to_vec2()),
                origin + self.to_screen(bbox.max.to_vec2()),
            );
            // Магнификация = экранные px длинной стороны на тексель.
            tex_mag = long_side / res_long as f32;
            if let Some(tex) = &self.camera.atlas {
                // Гасим текстуру при магнификации (tex_mag > 1): она уже
                // размыта, её сменяют резкие векторные точки ниже.
                let tex_alpha = (255.0 * (1.0 - (tex_mag - 1.0).clamp(0.0, 1.0))) as u8;
                painter.image(
                    tex.id(),
                    atlas_screen,
                    Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                    Color32::from_white_alpha(tex_alpha),
                );
            }
        }

        // Стрелки ego-графа вокруг выбранного нейрона (в обеих проекциях).
        if let Some(sel) = self.sel.selected {
            if sel < data.n {
                let a = origin + self.to_screen(self.proj_point(sel));
                for &(t, w, out, _) in &self.sel.ego {
                    if t >= data.n {
                        continue;
                    }
                    let b = origin + self.to_screen(self.proj_point(t));
                    let base = crate::core::color::color_for(data, t, self.filters.color_mode);
                    let alpha = (0.25 + (w.abs() / 40.0).clamp(0.0, 0.7)) as u8;
                    let color = Color32::from_rgba_unmultiplied(
                        base.r(),
                        base.g(),
                        base.b(),
                        if out { 255 } else { alpha },
                    );
                    if out {
                        draw_arrow(painter, a, b, color);
                    } else {
                        draw_arrow(painter, b, a, color.gamma_multiply(0.7));
                    }
                }
            }
        }

        // Точки при зуме (только 2D — в 3D облако рисует текстура):
        // включаются, как только тексел атласа растянут ≥1 экранный пиксель —
        // ровно тогда же начинает гаситься текстура выше. Векторные круги
        // остаются резкими на любом зуме, текстура им не мешает.
        if !self.camera.view3d && tex_mag >= 1.0 {
            let wr = self.world_rect(size);
            let pad = 0.02;
            let r = (1.35 * self.camera.zoom).clamp(0.8, 3.0);
            for i in 0..data.n {
                if !self.visible(i) {
                    continue;
                }
                let w = data.world(i, self.camera.proj);
                if w.x < wr.min.x - pad
                    || w.x > wr.max.x + pad
                    || w.y < wr.min.y - pad
                    || w.y > wr.max.y + pad
                {
                    continue;
                }
                let c = origin + self.to_screen(w);
                let col = crate::core::color::color_for(data, i, self.filters.color_mode);
                draw_soft_splat(painter, c, r, col);
            }
        }

        // Маркеры hover и выбора.
        if let Some(h) = self.sel.hover {
            if h < data.n {
                let c = origin + self.to_screen(self.proj_point(h));
                painter.circle_stroke(
                    c,
                    (7.0 * self.camera.zoom).clamp(7.0, 14.0),
                    Stroke::new(1.8_f32, Color32::from_rgb(255, 224, 126)),
                );
            }
        }
        if let Some(sel) = self.sel.selected {
            if sel < data.n {
                let c = origin + self.to_screen(self.proj_point(sel));
                painter.circle_stroke(
                    c,
                    10.0,
                    Stroke::new(2.2_f32, Color32::from_rgb(245, 255, 252)),
                );
            }
        }

        self.draw_minimap(painter, origin, size);
        if self.camera.view3d {
            self.draw_gizmo(painter, origin);
        }
        // Легенда-оверлей в вьюпорте: дублирует карточку легенды, но не
        // требует открывать инспектор. 4 строки, tabular-nums, полупрозрачный фон.
        self.draw_legend_overlay(painter, origin, size);
        // Keep the data surface clear. Mode, filters and render controls live
        // in the rail rather than covering the connectome.
    }

    /// Собирает GPU-точки для шейдерного рендера: те же фильтры, что у
    /// программного атласа (NT/класс/степень), + heatmap трассы полного сима
    /// (время первого спайка → синий→красный) и подсветка выбора.
    ///
    /// **Сортировка far → near**: точки проецируются (та же математика, что
    /// в `project3d`), и дальние кладутся в буфер первыми — GPU рисует их
    /// в порядке буфера, поэтому ближние аккуратно перекрывают дальние.
    ///
    /// **Кеш**: пока ключ `SplatKey` не изменился и нет живого сигнала,
    /// пересборка (проекция + фильтры до 120K точек) и сортировка
    /// пропускаются — «тихий» кадр делает только memcpy точек в callback.
    /// Возвращает `(callback, scratch, ключ_кеша)`: ключ `Some` — когда точки
    /// пересобраны и их можно закешировать (вызывающий пишет `splat_cache`);
    /// `None` — кеш не трогать (попали в кеш или идёт живой сигнал).
    fn make_splat_callback(
        &self,
        data: &crate::io::data::MancData,
        rect: Rect,
        mut scratch: Vec<(f32, PointData)>,
    ) -> (SplatCallback, Vec<(f32, PointData)>, Option<SplatKey>) {
        let trace = match (&self.sim.result, self.sim.show_trace) {
            (Some(SimOutcome::Full(r)), true) => {
                Some((r.first_fire.as_slice(), r.spike_counts.as_slice(), r.steps))
            }
            _ => None,
        };
        let persp = self.eff_persp();
        // LOD (как Potree): на дальнем плане рисуем представителей ячеек
        // сетки — облако выглядит так же, а точек в разы меньше. Точки
        // грубых уровней крупнее (компенсация разрежения).
        const MAX_GPU_POINTS: usize = 120_000;
        let scale = self.camera.zoom * crate::app::BASE;
        let mut lvl = self.camera.lod.level_for(scale);
        while lvl + 1 < self.camera.lod.levels.len()
            && self.camera.lod.levels[lvl].len() > MAX_GPU_POINTS
        {
            lvl += 1;
        }
        let level_pts = &self.camera.lod.levels[lvl];
        let lod_boost = self.camera.lod.boosts[lvl];
        let live_active = self.live.stream.is_some();
        // Ключ кеша: всё, что влияет на набор точек. Живой сигнал меняет
        // свечение каждый шаг — при нём кеш не используется и не пишется
        // (после остановки кадр честно вернётся к кешированному до-свечению).
        let trace_ptr = trace.map(|(ff, _, _)| ff.as_ptr() as usize).unwrap_or(0);
        let key = SplatKey {
            yaw: self.camera.yaw.to_bits(),
            pitch: self.camera.pitch.to_bits(),
            persp: persp.to_bits(),
            lod_level: lvl,
            show_nt: self.filters.show_nt,
            class_filter: self.filters.class_filter,
            min_deg: self.filters.min_deg,
            color_mode: self.filters.color_mode,
            trace_ptr,
            trace_steps: trace.map(|(_, _, s)| s).unwrap_or(0),
            selected: self.sel.selected,
        };
        let cache_hit = !live_active && self.camera.splat_cache == Some(key);
        if !cache_hit {
            // Переиспользуем ёмкость буфера между кадрами: орбита/спин не
            // должны аллоцировать ~8 МБ отсортированного массива каждую секунду.
            scratch.clear();
            scratch.reserve(level_pts.len().min(MAX_GPU_POINTS));
            // sin_cos — константа кадра, а не точки: выносим из цикла
            // (проекция нужна только для глубины сортировки).
            let (sy, cy) = self.camera.yaw.sin_cos();
            let (sp, cp) = self.camera.pitch.sin_cos();
            for &i in level_pts.iter().take(MAX_GPU_POINTS) {
                if !data.has_pos(i)
                    || !self.filters.show_nt[data.nt[i] as usize]
                    || data.out_deg(i) < self.filters.min_deg
                    || (self.filters.class_filter >= 0
                        && data.klass[i] != self.filters.class_filter as u8)
                {
                    continue;
                }
                // Глубина z2 из той же математики, что в `project3d` —
                // экранная проекция для сортировки не нужна.
                let x = data.xs[i] - 0.5;
                let y = data.ys[i] - 0.5;
                let z = data.zs[i] - 0.5;
                let y1 = x * sy + y * cy;
                let depth = y1 * sp + z * cp;
                let c = crate::core::color::color_for(data, i, self.filters.color_mode);
                let mut trace_v = 0.0f32;
                let mut boost = lod_boost - 1.0;
                if let Some((ff, _sc, steps)) = trace {
                    if steps > 0 && ff[i] > 0 {
                        trace_v = (ff[i] as f32 / steps as f32).clamp(0.0, 1.0);
                        boost = 1.4; // точки трассы крупнее
                    }
                }
                let sel = if self.sel.selected == Some(i) {
                    1.0
                } else {
                    0.0
                };
                if sel > 0.5 {
                    boost = boost.max(0.6);
                }
                // Живое свечение от потоковой симуляции (спарк в 3D).
                let glow = self.live.glow.get(i).copied().unwrap_or(0.0);
                scratch.push((
                    depth,
                    PointData {
                        pos: [data.xs[i], data.ys[i], data.zs[i]],
                        _pad0: 0.0,
                        col: [
                            c.r() as f32 / 255.0,
                            c.g() as f32 / 255.0,
                            c.b() as f32 / 255.0,
                            0.9,
                        ],
                        trace: trace_v,
                        sel,
                        boost,
                        glow,
                        _pad1: 0.0,
                        _pad2: [0.0; 3],
                    },
                ));
            }
            // Дальние первыми — ближние рисуются последними (поверх). total_cmp
            // быстрее partial_cmp+unwrap_or и не зависит от NaN.
            scratch.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        }
        let cache_key = if cache_hit {
            None
        } else {
            (!live_active).then_some(key)
        };
        let points: Vec<PointData> = scratch.iter().map(|(_, p)| *p).collect();
        let uniforms = Uniforms {
            yaw: self.camera.yaw,
            pitch: self.camera.pitch,
            persp,
            ppp: self.camera.ppp,
            scale: self.camera.zoom * crate::app::BASE,
            base_size: self.camera.pt_size,
            alpha: self.camera.pt_alpha,
            render_mode: match self.camera.render3d {
                crate::app::connectome::state::Render3d::Spheres => 0.0,
                crate::app::connectome::state::Render3d::Points => 1.0,
                crate::app::connectome::state::Render3d::Neon => 2.0,
            },
            pan: [self.camera.pan.x, self.camera.pan.y],
            rect: [rect.width(), rect.height()],
            _pad: [0.0; 4],
        };
        (
            SplatCallback::new(points, uniforms, self.camera.gpu_target_format),
            scratch,
            cache_key,
        )
    }

    /// Координатная сетка (только в 2D-режиме).
    fn draw_grid(&self, painter: &egui::Painter, origin: Pos2, size: Vec2) {
        let color = Color32::from_rgba_unmultiplied(93, 115, 113, 38);
        let major_color = Color32::from_rgba_unmultiplied(122, 154, 149, 72);
        let wr = self.world_rect(size);
        let mut x = (wr.min.x / crate::app::GRID).floor() * crate::app::GRID;
        while x <= wr.max.x {
            let sx = origin.x + self.to_screen(vec2(x, 0.0)).x;
            let major = (x / (crate::app::GRID * 5.0)).round() == x / (crate::app::GRID * 5.0);
            painter.line_segment(
                [pos2(sx, origin.y), pos2(sx, origin.y + size.y)],
                Stroke::new(
                    if major { 1.0 } else { 0.5 },
                    if major { major_color } else { color },
                ),
            );
            if major && sx >= origin.x && sx <= origin.x + size.x {
                painter.text(
                    pos2(sx + 4.0, origin.y + size.y - 8.0),
                    Align2::LEFT_BOTTOM,
                    format!("{x:.1}"),
                    FontId::monospace(10.0),
                    Color32::from_rgba_unmultiplied(178, 199, 195, 120),
                );
            }
            x += crate::app::GRID;
        }
        let mut y = (wr.min.y / crate::app::GRID).floor() * crate::app::GRID;
        while y <= wr.max.y {
            let sy = origin.y + self.to_screen(vec2(0.0, y)).y;
            let major = (y / (crate::app::GRID * 5.0)).round() == y / (crate::app::GRID * 5.0);
            painter.line_segment(
                [pos2(origin.x, sy), pos2(origin.x + size.x, sy)],
                Stroke::new(
                    if major { 1.0 } else { 0.5 },
                    if major { major_color } else { color },
                ),
            );
            if major && sy >= origin.y && sy <= origin.y + size.y {
                painter.text(
                    pos2(origin.x + 4.0, sy - 4.0),
                    Align2::LEFT_BOTTOM,
                    format!("{y:.1}"),
                    FontId::monospace(10.0),
                    Color32::from_rgba_unmultiplied(178, 199, 195, 120),
                );
            }
            y += crate::app::GRID;
        }
    }

    /// Миникарта в углу: все нейроны (прореженные), рамка видимой области.
    fn draw_minimap(&self, painter: &egui::Painter, origin: Pos2, canvas: Vec2) {
        let mm = self.minimap_rect(canvas);
        let r = Rect::from_min_size(origin + mm.min.to_vec2(), mm.size());
        painter.rect_filled(
            r,
            CornerRadius::same(5),
            Color32::from_rgba_unmultiplied(10, 16, 17, 232),
        );
        painter.rect_stroke(
            r,
            CornerRadius::same(5),
            Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(90, 90, 130, 140)),
            egui::StrokeKind::Inside,
        );
        painter.text(
            r.left_top() + vec2(8.0, 7.0),
            Align2::LEFT_TOP,
            "OVERVIEW",
            FontId::monospace(9.0),
            Color32::from_rgba_unmultiplied(184, 211, 205, 190),
        );
        let Some(data) = self.data.as_ref() else {
            return;
        };
        let step = (data.n / 4000).max(1);
        for i in (0..data.n).step_by(step) {
            if !data.has_pos(i) {
                continue;
            }
            let w = data.world(i, self.camera.proj);
            let p = r.min + w * mm.size();
            let c = crate::core::color::color_for(data, i, self.filters.color_mode);
            painter.rect_filled(
                Rect::from_min_size(p, vec2(1.5, 1.5)),
                CornerRadius::ZERO,
                c,
            );
        }
        // Рамка видимой области.
        let wr = self.world_rect(canvas);
        let a = r.min + vec2(wr.min.x, wr.min.y).clamp(Vec2::ZERO, vec2(1.0, 1.0)) * mm.size();
        let b = r.min + vec2(wr.max.x, wr.max.y).clamp(Vec2::ZERO, vec2(1.0, 1.0)) * mm.size();
        painter.rect_stroke(
            Rect::from_min_max(a, b),
            CornerRadius::ZERO,
            Stroke::new(1.0_f32, Color32::from_rgb(255, 255, 255)),
            egui::StrokeKind::Inside,
        );
    }

    /// Ориентационный гизмо (оси X/Y/Z), вращается вместе с камерой.
    fn draw_legend_overlay(&self, painter: &egui::Painter, origin: Pos2, _size: Vec2) {
        let Some(data) = self.data.as_ref() else { return };
        let (title, _hint, rows) =
            crate::app::connectome::cards::legend::legend_data(Some(data), self.filters.color_mode);
        if rows.is_empty() {
            return;
        }
        // Компактный оверлей — до 4 строк, чтобы не закрывать мозг.
        let n = rows.len().min(4);
        let w = 152.0;
        let h = 20.0 + n as f32 * 16.0;
        let r = Rect::from_min_size(origin + vec2(8.0, 8.0), vec2(w, h));
        painter.rect_filled(r, 6.0, Color32::from_rgba_unmultiplied(28, 32, 33, 220));
        painter.rect_stroke(
            r,
            6.0,
            Stroke::new(1.0, Color32::from_rgba_unmultiplied(49, 55, 56, 180)),
            egui::StrokeKind::Inside,
        );
        painter.text(
            r.min + vec2(8.0, 6.0),
            Align2::LEFT_TOP,
            title,
            FontId::proportional(10.0),
            Color32::from_rgb(230, 230, 230),
        );
        for (i, (name, col, _cnt)) in rows.iter().take(n).enumerate() {
            let y = r.min.y + 20.0 + i as f32 * 16.0;
            let sw = Rect::from_min_size(pos2(r.min.x + 8.0, y), vec2(10.0, 10.0));
            painter.rect_filled(sw, 2.0, *col);
            // truncate long names to ~14 chars
            let label = if name.chars().count() > 14 {
                let t: String = name.chars().take(14).collect();
                format!("{t}…")
            } else {
                name.clone()
            };
            painter.text(
                pos2(sw.max.x + 6.0, y + 5.0),
                Align2::LEFT_CENTER,
                label,
                FontId::proportional(10.0),
                Color32::from_rgb(190, 190, 210),
            );
        }
    }

    fn draw_gizmo(&self, painter: &egui::Painter, origin: Pos2) {
        let center = origin + vec2(46.0, 46.0);
        let r = 30.0;
        painter.circle_filled(
            center,
            r + 10.0,
            Color32::from_rgba_unmultiplied(16, 16, 28, 210),
        );
        painter.circle_stroke(
            center,
            r + 10.0,
            Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(90, 90, 130, 160)),
        );
        let axes: [(&str, f32, f32, f32, Color32); 3] = [
            ("X", 1.0, 0.0, 0.0, Color32::from_rgb(255, 95, 95)),
            ("Y", 0.0, 1.0, 0.0, Color32::from_rgb(90, 225, 120)),
            ("Z", 0.0, 0.0, 1.0, Color32::from_rgb(95, 155, 255)),
        ];
        for (name, x, y, z, col) in axes {
            // Ось, «смотрящая» на зрителя, ярче; уходящая вглубь — тусклее.
            let (d, depth) = rotate_dir(x, y, z, self.camera.yaw, self.camera.pitch);
            let end = center + d * r;
            let bright = if depth > 0.0 { 1.0 } else { 0.45 };
            let c = Color32::from_rgb(
                (col.r() as f32 * bright) as u8,
                (col.g() as f32 * bright) as u8,
                (col.b() as f32 * bright) as u8,
            );
            painter.line_segment([center, end], Stroke::new(2.0_f32, c));
            painter.circle_filled(end, 2.5, c);
            painter.text(
                end + vec2(5.0, 0.0),
                Align2::LEFT_CENTER,
                name,
                FontId::monospace(11.0),
                c,
            );
        }
    }
}

/// Мягкий гауссов сплат: концентрические круги с падающей альфой — как
/// запечённая точка атласа, а не жёсткий «шар» с обводкой. Внешний слой
/// почти прозрачный (гало), ядро плотное.
fn draw_soft_splat(painter: &egui::Painter, center: Pos2, r: f32, color: Color32) {
    // (доля радиуса, альфа) от внешнего края к центру. Плотное ядро + короткий
    // спад — как у запечённого гауссова сплата (σ≈0.9), без широкого «размытого» гало.
    const LAYERS: [(f32, u8); 4] = [(1.0, 55), (0.6, 150), (0.32, 225), (0.14, 250)];
    for (k, a) in LAYERS {
        painter.circle_filled(
            center,
            r * k,
            Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), a),
        );
    }
}

/// Стрелка-кривая Безье от a к b (связь ego-графа на канвасе).
fn draw_arrow(painter: &egui::Painter, a: Pos2, b: Pos2, color: Color32) {
    if (b - a).length() < 1.0 {
        return;
    }
    let dx = (b.x - a.x).abs().max(30.0) * 0.5;
    let c1 = pos2(a.x + dx, a.y);
    let c2 = pos2(b.x - dx, b.y);
    let pts: Vec<Pos2> = (0..=16)
        .map(|i| {
            let t = i as f32 / 16.0;
            let u = 1.0 - t;
            pos2(
                u * u * u * a.x + 3.0 * u * u * t * c1.x + 3.0 * u * t * t * c2.x + t * t * t * b.x,
                u * u * u * a.y + 3.0 * u * u * t * c1.y + 3.0 * u * t * t * c2.y + t * t * t * b.y,
            )
        })
        .collect();
    painter.add(egui::Shape::line(pts, Stroke::new(1.2_f32, color)));
    painter.circle_filled(b, 1.6, color);
}
