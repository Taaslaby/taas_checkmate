//! # Viz Fly Graph — интерактивный атлас мозга мухи (FAFB v783)
//!
//! Рендерит **настоящий коннектом плодовой мушки** (FlyWire v783):
//! 139 266 нейронов, 15 млн синапсов, каждый нейрон — с реальными 3D
//! координатами и нейромедиатором из `flywire_annotations.tsv`.
//!
//! ```text
//!    ┌───────────────────────────────────────────┐
//!    │  ▓▓ ACh (возб.)  ▓▓ GABA (торм.)          │
//!    │  ▓▓ GLUT (GluCl) ▓▓ DA / SER / OCT        │  ← легенда NT
//!    └───────────────────────────────────────────┘
//!      Оптические доли / центральный мозг / VNC
//!    ┌───────────────────────────────────────────┐
//!    │        ◌ ◌ ◌  · · ·  ◌ ◌ ◌                │
//!    │     ◌ ◌ ◌  ······   ◌ ◌ ◌                 │  ← атлас: каждый
//!    │      ◌ ◌ · · · · · ◌ ◌ ◌                  │    нейрон = точка
//!    │        ◌ ◌ ◌  · · ·  ◌ ◌ ◌                │    цвета NT
//!    └───────────────────────────────────────────┘
//!                    ↕ scroll = zoom · drag = pan
//! ```
//!
//! ## Взаимодействие
//! - **Скролл** = зум вокруг курсора · **drag по пустому / средней кнопкой** = пан.
//! - **Hover** — тултип нейрона (root_id, cell_type, NT, степень).
//! - **Клик** по нейрону — ego-граф: его входящие/исходящие связи (топ по весу).
//! - **Миникарта** (внизу-справа) — клик/драг для навигации.
//! - **Панель**: переключение проекции (XY/XZ/YZ), фильтры по NT и стороне,
//!   порог степени, поиск по cell_type / root_id.
//!
//! ## Данные
//! Требуется `resources/examples/data/fly_viz.bin` — соберите один раз:
//! ```text
//! python3 resources/scripts/fafb/fly_viz_prep.py
//! cargo run -p pillars --example viz_fly_graph --release
//! ```
//! Бинарник самодостаточен: координаты (нормированы [0,1]), NT-коды, типы
//! клеток и CSR-матрица связей — без парсера parquet в рантайме.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use eframe::egui::{
    self, pos2, vec2, Align2, Color32, FontId, Key, PointerButton, Pos2, Rect, Rounding, Sense,
    Stroke, Vec2,
};

// ─── NT palette (цвета — как в инсектной нейрохимии) ──────────
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Nt {
    Ach = 0,
    Gaba = 1,
    Glut = 2,
    Da = 3,
    Ser = 4,
    Oct = 5,
    Unknown = 6,
}

impl Nt {
    pub fn from_code(c: u8) -> Nt {
        match c {
            0 => Nt::Ach,
            1 => Nt::Gaba,
            2 => Nt::Glut,
            3 => Nt::Da,
            4 => Nt::Ser,
            5 => Nt::Oct,
            _ => Nt::Unknown,
        }
    }

    pub fn color(self) -> Color32 {
        match self {
            Nt::Ach => Color32::from_rgb(80, 210, 140), // возб. в инсектах
            Nt::Gaba => Color32::from_rgb(225, 85, 85), // торм.
            Nt::Glut => Color32::from_rgb(85, 145, 235), // GluCl — торм. Cl⁻
            Nt::Da => Color32::from_rgb(215, 120, 225), // модул.
            Nt::Ser => Color32::from_rgb(245, 165, 85),
            Nt::Oct => Color32::from_rgb(235, 215, 95),
            Nt::Unknown => Color32::from_rgb(125, 125, 135),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Nt::Ach => "ACh (возб.)",
            Nt::Gaba => "GABA (торм.)",
            Nt::Glut => "GLUT (GluCl)",
            Nt::Da => "DA (модул.)",
            Nt::Ser => "SER (модул.)",
            Nt::Oct => "OCT (модул.)",
            Nt::Unknown => "unknown",
        }
    }
}

// ─── Проекция ─────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Proj {
    Xy,
    Xz,
    Yz,
}

impl Proj {
    pub const ALL: [Proj; 3] = [Proj::Xy, Proj::Xz, Proj::Yz];
    pub fn name(self) -> &'static str {
        match self {
            Proj::Xy => "XY (дорсаль)",
            Proj::Xz => "XZ (сагиттал)",
            Proj::Yz => "YZ (фронтал)",
        }
    }
    fn pick(self, x: f32, y: f32, z: f32) -> Vec2 {
        match self {
            Proj::Xy => vec2(x, y),
            Proj::Xz => vec2(x, z),
            Proj::Yz => vec2(y, z),
        }
    }
}

// ─── Данные коннектома (самодостаточный бинарник) ─────────────
pub struct FlyData {
    n: usize,
    s: usize,
    xs: Vec<f32>,
    ys: Vec<f32>,
    zs: Vec<f32>,
    nt: Vec<u8>,
    side: Vec<u8>,
    roots: Vec<u64>,
    ct_idx: Vec<u16>,
    cell_types: Vec<String>,
    row_ptr: Vec<i32>,
    col: Vec<i32>,
    val: Vec<f32>,
    /// Лениво строится при первом клике: обратный CSR (пост → пре).
    rev_row_ptr: Option<Vec<i32>>,
    rev_col: Option<Vec<i32>>,
    rev_val: Option<Vec<f32>>,
}

impl FlyData {
    /// Парсит `fly_viz.bin`. Возвращает `None` (с понятной причиной), если файла нет.
    pub fn load(path: &Path) -> Result<FlyData, String> {
        let mut buf = Vec::new();
        File::open(path)
            .map_err(|e| format!("нет данных {path:?}: {e} (запустите fly_viz_prep.py)"))?
            .read_to_end(&mut buf)
            .map_err(|e| e.to_string())?;
        let mut p = 0usize;
        macro_rules! take {
            ($n:expr) => {{
                let b = buf.get(p..p + $n).ok_or("файл повреждён (обрезка)")?;
                p += $n;
                b
            }};
        }
        let magic = take!(4);
        if magic != b"FLYV" {
            return Err("не тот формат (ожидался FLYV)".into());
        }
        let ver = u32::from_le_bytes(take!(4).try_into().unwrap());
        if ver != 1 {
            return Err(format!("неподдерживаемая версия {ver}"));
        }
        let n = u32::from_le_bytes(take!(4).try_into().unwrap()) as usize;
        let s = u32::from_le_bytes(take!(4).try_into().unwrap()) as usize;
        let n_ct = u32::from_le_bytes(take!(4).try_into().unwrap()) as usize;

        let read_f32s = |b: &[u8]| {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<f32>>()
        };
        let read_u8s = |b: &[u8]| b.to_vec();
        let read_u64s = |b: &[u8]| {
            b.chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<u64>>()
        };
        let read_u16s = |b: &[u8]| {
            b.chunks_exact(2)
                .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<u16>>()
        };
        let read_i32s = |b: &[u8]| {
            b.chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<i32>>()
        };

        let xs = read_f32s(take!(n * 4));
        let ys = read_f32s(take!(n * 4));
        let zs = read_f32s(take!(n * 4));
        let nt = read_u8s(take!(n));
        let side = read_u8s(take!(n));
        let roots = read_u64s(take!(n * 8));
        let ct_idx = read_u16s(take!(n * 2));
        let mut cell_types = Vec::with_capacity(n_ct);
        for _ in 0..n_ct {
            let len = u16::from_le_bytes(take!(2).try_into().unwrap()) as usize;
            let bytes = take!(len);
            cell_types.push(String::from_utf8_lossy(bytes).into_owned());
        }
        let row_ptr = read_i32s(take!((n + 1) * 4));
        let col = read_i32s(take!(s * 4));
        let val = read_f32s(take!(s * 4));
        let _ = p; // финальная позиция курсора (тишит unused_assignments в макросе take!)

        Ok(FlyData {
            n,
            s,
            xs,
            ys,
            zs,
            nt,
            side,
            roots,
            ct_idx,
            cell_types,
            row_ptr,
            col,
            val,
            rev_row_ptr: None,
            rev_col: None,
            rev_val: None,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn world(&self, i: usize, proj: Proj) -> Vec2 {
        proj.pick(self.xs[i], self.ys[i], self.zs[i])
    }

    pub fn nt(&self, i: usize) -> Nt {
        Nt::from_code(self.nt[i])
    }

    pub fn cell_type(&self, i: usize) -> &str {
        let idx = self.ct_idx[i] as usize;
        self.cell_types.get(idx).map(|s| s.as_str()).unwrap_or("")
    }

    /// Построить обратный CSR (post → список pre) — один раз, лениво.
    pub fn ensure_reverse(&mut self) {
        if self.rev_row_ptr.is_some() {
            return;
        }
        let n = self.n;
        let s = self.s;
        let mut counts = vec![0i32; n];
        for &c in &self.col {
            if c >= 0 {
                counts[c as usize] += 1;
            }
        }
        let mut rp = vec![0i32; n + 1];
        for i in 0..n {
            rp[i + 1] = rp[i] + counts[i];
        }
        let mut rev_col = vec![0i32; s];
        let mut rev_val = vec![0.0f32; s];
        let mut cursor = rp[..n].to_vec();
        // Идём по рёбрам в порядке прямого CSR. Каждое ребро e принадлежит
        // строке pre, которую отслеживаем указателем: row_ptr[pre] <= e < row_ptr[pre+1].
        // В rev_col храним ПРЕ-нейрон (а не индекс ребра!), в rev_val — его вес.
        let mut pre: usize = 0;
        for e in 0..s {
            while pre + 1 < n && (self.row_ptr[pre + 1].max(0) as usize) <= e {
                pre += 1;
            }
            let c = self.col[e];
            if c >= 0 {
                let ci = c as usize;
                rev_col[cursor[ci] as usize] = pre as i32;
                rev_val[cursor[ci] as usize] = self.val[e];
                cursor[ci] += 1;
            }
        }
        self.rev_row_ptr = Some(rp);
        self.rev_col = Some(rev_col);
        self.rev_val = Some(rev_val);
    }

    /// Топ-K соседей (по весу) для ego-графа: out + in.
    /// Возвращает (target_idx, вес, направление: true = исходящий).
    pub fn ego_edges(&mut self, node: usize, k: usize) -> Vec<(usize, f32, bool)> {
        self.ensure_reverse();
        let mut edges: Vec<(usize, f32, bool)> = Vec::new();
        let lo = self.row_ptr[node].max(0) as usize;
        let hi = self.row_ptr[node + 1].max(0) as usize;
        for kk in lo..hi.min(self.s) {
            let t = self.col[kk];
            if t >= 0 && (t as usize) < self.n {
                edges.push((t as usize, self.val[kk], true));
            }
        }
        if let Some(rp) = &self.rev_row_ptr {
            if let Some(rc) = &self.rev_col {
                if let Some(rv) = &self.rev_val {
                    let lo = rp[node].max(0) as usize;
                    let hi = rp[node + 1].max(0) as usize;
                    for kk in lo..hi.min(self.s) {
                        let src = rc[kk];
                        if src >= 0 && (src as usize) < self.n {
                            edges.push((src as usize, rv[kk], false));
                        }
                    }
                }
            }
        }
        edges.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        edges.truncate(k);
        edges
    }

    pub fn degree(&self, node: usize) -> usize {
        if node >= self.n {
            return 0;
        }
        (self.row_ptr[node + 1].max(0) - self.row_ptr[node].max(0)).max(0) as usize
    }
}

// ─── Построение атласа (текстура) ─────────────────────────────
/// Рисует все нейроны в RGBA-буфер `size×size`: каждый нейрон — точка
/// цвета NT, аддитивно (плотные области светятся ярче). Уважает фильтры:
/// скрытые NT/стороны и нейроны ниже порога степени пропускаются.
pub fn build_atlas(
    data: &FlyData,
    proj: Proj,
    size: usize,
    show_nt: &[bool; 7],
    show_side: &[bool; 4],
    min_deg: usize,
) -> Vec<Color32> {
    let mut px = vec![Color32::from_rgb(14, 14, 24); size * size];
    let mut add = |x: usize, y: usize, c: Color32| {
        let i = y * size + x;
        if i >= px.len() {
            return;
        }
        let old = px[i];
        px[i] = Color32::from_rgb(
            (old.r() as u16 + c.r() as u16).min(255) as u8,
            (old.g() as u16 + c.g() as u16).min(255) as u8,
            (old.b() as u16 + c.b() as u16).min(255) as u8,
        );
    };
    for i in 0..data.n {
        if !show_nt[data.nt(i) as usize]
            || !show_side[(data.side[i] as usize).min(3)]
            || data.degree(i) < min_deg
        {
            continue;
        }
        let w = data.world(i, proj);
        let c = data.nt(i).color();
        let cx = (w.x.clamp(0.0, 1.0) * (size as f32 - 1.0)) as usize;
        let cy = (w.y.clamp(0.0, 1.0) * (size as f32 - 1.0)) as usize;
        // 3×3 пятно — мягче и плотнее
        for dx in 0..3 {
            for dy in 0..3 {
                if cx + dx < size && cy + dy < size {
                    add(cx + dx, cy + dy, c);
                }
            }
        }
    }
    px
}

// ─── Приложение ───────────────────────────────────────────────
const BASE: f32 = 720.0; // полный мозг [0,1]² ≈ 720 px при zoom=1
const ATLAS: usize = 2048;
const MINIMAP: Vec2 = vec2(180.0, 132.0);
const GRID: f32 = 0.1;

struct FlyApp {
    data: Option<FlyData>,
    status: String,
    // камера
    pan: Vec2,
    zoom: f32,
    proj: Proj,
    // атлас
    atlas: Option<egui::TextureHandle>,
    // фильтры
    show_nt: [bool; 7],
    min_deg: usize,
    show_side: [bool; 4],
    // выбор
    hover: Option<usize>,
    selected: Option<usize>,
    ego: Vec<(usize, f32, bool)>,
    search: String,
    search_status: String,
    // true, если текущий драг начался на миникарте (навигация, а не пан)
    minimap_drag: bool,
}

impl Default for FlyApp {
    fn default() -> Self {
        let mut app = Self {
            data: None,
            status: String::new(),
            pan: Vec2::ZERO,
            zoom: 1.0,
            proj: Proj::Xz,
            atlas: None,
            show_nt: [true; 7],
            min_deg: 0,
            show_side: [true; 4],
            hover: None,
            selected: None,
            ego: Vec::new(),
            search: String::new(),
            search_status: String::new(),
            minimap_drag: false,
        };
        // путь к бинарнику: CARGO_MANIFEST_DIR = crates/pillars → ../../ = корень репо
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../../resources/examples/data/fly_viz.bin");
        let path = if let Some(a) = std::env::args().nth(1) {
            PathBuf::from(a)
        } else {
            p
        };
        match FlyData::load(&path) {
            Ok(d) => {
                app.status = format!(
                    "✓ {} нейронов, {} синапсов, {} типов клеток",
                    d.n(),
                    d.s,
                    d.cell_types.len()
                );
                app.data = Some(d);
            }
            Err(e) => app.status = format!("⚠ {e}"),
        }
        // центрируем мозг
        app.pan = vec2(80.0, 40.0);
        app
    }
}

impl FlyApp {
    // ── camera ──────────────────────────────────────────────
    fn to_screen(&self, w: Vec2) -> Vec2 {
        w * (self.zoom * BASE) + self.pan
    }
    fn screen_to_world(&self, s: Vec2) -> Vec2 {
        (s - self.pan) / (self.zoom * BASE)
    }
    fn world_rect(&self, canvas: Vec2) -> Rect {
        let a = self.screen_to_world(Vec2::ZERO);
        let b = self.screen_to_world(canvas);
        Rect::from_min_max(pos2(a.x, a.y), pos2(b.x, b.y))
    }

    // ── выбор / hover ───────────────────────────────────────
    /// Ближайший нейрон к точке в мировых координатах (проекция).
    fn nearest(&self, w: Vec2) -> Option<usize> {
        let d = self.data.as_ref()?;
        let mut best: Option<(usize, f32)> = None;
        for i in 0..d.n {
            if !self.visible_nt(i) || !self.visible_side(i) || d.degree(i) < self.min_deg {
                continue;
            }
            let p = d.world(i, self.proj);
            let dist = (p - w).length_sq();
            if best.is_none_or(|(_, bd)| dist < bd) {
                best = Some((i, dist));
            }
        }
        best.map(|(i, _)| i)
    }

    fn visible_nt(&self, i: usize) -> bool {
        let d = self.data.as_ref().unwrap();
        self.show_nt[d.nt(i) as usize]
    }
    fn visible_side(&self, i: usize) -> bool {
        let d = self.data.as_ref().unwrap();
        self.show_side[(d.side[i] as usize).min(3)]
    }

    fn select(&mut self, i: usize) {
        self.selected = Some(i);
        self.ego = if let Some(d) = self.data.as_mut() {
            d.ego_edges(i, 240)
        } else {
            Vec::new()
        };
    }

    fn search_neuron(&mut self) {
        let q = self.search.trim().to_lowercase();
        if q.is_empty() {
            self.search_status.clear();
            return;
        }
        let Some(d) = self.data.as_ref() else {
            self.search_status = "нет данных".into();
            return;
        };
        let mut found = None;
        for i in 0..d.n {
            let ct = d.cell_type(i).to_lowercase();
            if ct.contains(&q) {
                found = Some(i);
                break;
            }
            // поиск по root_id
            let root = format!("{}", d.roots[i]);
            if root.starts_with(&q) {
                found = Some(i);
                break;
            }
        }
        match found {
            Some(i) => {
                // собираем все owned-данные ДО мутабельного self.select (NLL)
                let w = d.world(i, self.proj);
                let cell = d.cell_type(i).to_string();
                let nt_name = Nt::from_code(d.nt[i]).name().to_string();
                self.select(i);
                // центрируем камеру на найденном нейроне
                let c = self.screen_to_world(vec2(400.0, 300.0));
                self.pan += (w - c) * (self.zoom * BASE);
                self.search_status = format!("→ #{i} {cell} · {nt_name}");
            }
            None => self.search_status = format!("ничего не найдено: {q}"),
        }
    }

    // ── rendering ───────────────────────────────────────────
    fn draw_canvas(&mut self, painter: &egui::Painter, origin: Pos2, size: Vec2) {
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
        // фон-сетка
        self.draw_grid(painter, origin, size);

        // атлас — текстура всего мозга (уважает фильтры NT/сторона/степень)
        if self.atlas.is_none() {
            // строится заново при смене проекции или фильтров (atlas = None)
            let px = build_atlas(
                data,
                self.proj,
                ATLAS,
                &self.show_nt,
                &self.show_side,
                self.min_deg,
            );
            let img = egui::ColorImage {
                size: [ATLAS, ATLAS],
                pixels: px,
            };
            self.atlas = Some(painter.ctx().load_texture(
                "fly_atlas",
                img,
                egui::TextureOptions::LINEAR,
            ));
        }
        let world = self.world_rect(size);
        let atlas_rect = Rect::from_min_max(
            origin + self.to_screen(Vec2::ZERO),
            origin + self.to_screen(vec2(1.0, 1.0)),
        );
        if let Some(tex) = &self.atlas {
            // только видимая часть атласа (uv-обрезка)
            let u0 = (world.min.x / 1.0).clamp(0.0, 1.0);
            let u1 = (world.max.x / 1.0).clamp(0.0, 1.0);
            let v0 = 1.0 - (world.max.y / 1.0).clamp(0.0, 1.0);
            let v1 = 1.0 - (world.min.y / 1.0).clamp(0.0, 1.0);
            let uv = Rect::from_min_max(pos2(u0, v0), pos2(u1, v1));
            painter.image(tex.id(), atlas_rect, uv, Color32::WHITE);
        }

        // ego-граф выбранного нейрона (поверх атласа)
        if let Some(sel) = self.selected {
            if sel < data.n {
                let a = origin + self.to_screen(data.world(sel, self.proj));
                for &(t, w, out) in &self.ego {
                    if t >= data.n {
                        continue;
                    }
                    let b = origin + self.to_screen(data.world(t, self.proj));
                    let base = data.nt(t).color();
                    let alpha = (0.25 + (w.abs() / 40.0).clamp(0.0, 0.7)) as u8;
                    let color = Color32::from_rgba_unmultiplied(
                        base.r(),
                        base.g(),
                        base.b(),
                        if out { 255 } else { alpha },
                    );
                    // стрелка по направлению связи
                    if out {
                        draw_arrow(painter, a, b, color);
                    } else {
                        draw_arrow(painter, b, a, color.gamma_multiply(0.7));
                    }
                }
            }
        }

        // точки при сильном зуме (детализация поверх текстуры)
        if self.zoom * BASE > 3.0 {
            let wr = self.world_rect(size);
            let pad = 0.02;
            let r = (1.6 * self.zoom).clamp(0.6, 3.5);
            for i in 0..data.n {
                if !self.visible_nt(i) || !self.visible_side(i) || data.degree(i) < self.min_deg {
                    continue;
                }
                let w = data.world(i, self.proj);
                if w.x < wr.min.x - pad
                    || w.x > wr.max.x + pad
                    || w.y < wr.min.y - pad
                    || w.y > wr.max.y + pad
                {
                    continue;
                }
                let c = origin + self.to_screen(w);
                let col = data.nt(i).color();
                painter.circle_filled(
                    c,
                    r,
                    Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 235),
                );
            }
        }

        // hover-подсветка
        if let Some(h) = self.hover {
            if h < data.n {
                let c = origin + self.to_screen(data.world(h, self.proj));
                painter.circle_stroke(
                    c,
                    6.0 * self.zoom + 2.0,
                    Stroke::new(1.5_f32, Color32::from_rgb(255, 240, 150)),
                );
            }
        }
        // выбранный нейрон
        if let Some(sel) = self.selected {
            if sel < data.n {
                let c = origin + self.to_screen(data.world(sel, self.proj));
                painter.circle_stroke(
                    c,
                    8.0,
                    Stroke::new(2.0_f32, Color32::from_rgb(255, 255, 255)),
                );
            }
        }

        // миникарта
        self.draw_minimap(painter, origin, size);
    }

    fn draw_grid(&self, painter: &egui::Painter, origin: Pos2, size: Vec2) {
        let color = Color32::from_rgba_unmultiplied(70, 70, 90, 45);
        let wr = self.world_rect(size);
        let mut x = (wr.min.x / GRID).floor() * GRID;
        while x <= wr.max.x {
            let sx = origin.x + self.to_screen(vec2(x, 0.0)).x;
            painter.line_segment(
                [pos2(sx, origin.y), pos2(sx, origin.y + size.y)],
                Stroke::new(1.0_f32, color),
            );
            x += GRID;
        }
        let mut y = (wr.min.y / GRID).floor() * GRID;
        while y <= wr.max.y {
            let sy = origin.y + self.to_screen(vec2(0.0, y)).y;
            painter.line_segment(
                [pos2(origin.x, sy), pos2(origin.x + size.x, sy)],
                Stroke::new(1.0_f32, color),
            );
            y += GRID;
        }
    }

    fn minimap_rect(&self, canvas: Vec2) -> Rect {
        Rect::from_min_size(
            vec2(canvas.x - MINIMAP.x - 14.0, canvas.y - MINIMAP.y - 14.0).to_pos2(),
            MINIMAP,
        )
    }

    fn draw_minimap(&self, painter: &egui::Painter, origin: Pos2, canvas: Vec2) {
        let mm = self.minimap_rect(canvas);
        let r = Rect::from_min_size(origin + mm.min.to_vec2(), mm.size());
        painter.rect_filled(
            r,
            Rounding::same(5.0),
            Color32::from_rgba_unmultiplied(12, 12, 22, 220),
        );
        painter.rect_stroke(
            r,
            Rounding::same(5.0),
            Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(90, 90, 130, 140)),
        );
        let Some(data) = self.data.as_ref() else {
            return;
        };
        // точки нейронов (прореженные) — дешёвый обзор
        let step = (data.n / 4000).max(1);
        for i in (0..data.n).step_by(step) {
            let w = data.world(i, self.proj);
            let p = r.min + w * mm.size();
            let c = data.nt(i).color();
            painter.rect_filled(Rect::from_min_size(p, vec2(1.5, 1.5)), Rounding::ZERO, c);
        }
        // рамка вьюпорта
        let wr = self.world_rect(canvas);
        let a = r.min + vec2(wr.min.x, wr.min.y).clamp(Vec2::ZERO, vec2(1.0, 1.0)) * mm.size();
        let b = r.min + vec2(wr.max.x, wr.max.y).clamp(Vec2::ZERO, vec2(1.0, 1.0)) * mm.size();
        painter.rect_stroke(
            Rect::from_min_max(a, b),
            Rounding::ZERO,
            Stroke::new(1.0_f32, Color32::from_rgb(255, 255, 255)),
        );
        painter.text(
            r.min + vec2(5.0, 3.0),
            Align2::LEFT_TOP,
            "🗺 клик = навигация",
            FontId::proportional(9.0),
            Color32::from_rgba_unmultiplied(170, 170, 200, 160),
        );
    }

    fn minimap_pan(&mut self, canvas: Vec2, pointer_screen: Vec2) {
        let mm = self.minimap_rect(canvas);
        let wp =
            ((pointer_screen - mm.min.to_vec2()) / mm.size()).clamp(Vec2::ZERO, vec2(1.0, 1.0));
        self.pan = canvas * 0.5 - wp * (self.zoom * BASE);
    }

    // ── panels ──────────────────────────────────────────────
    fn top_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    egui::RichText::new("🪰 Fly Brain Atlas — FAFB v783")
                        .strong()
                        .size(16.0),
                );
                ui.separator();
                ui.label(&self.status);
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("Проекция:");
                for p in Proj::ALL {
                    // виджет рисуем всегда (иначе выбранная кнопка исчезает из UI);
                    // защищаем только мутацию — не меняем проекцию на текущую
                    if ui.selectable_label(self.proj == p, p.name()).clicked() && self.proj != p {
                        self.proj = p;
                        self.atlas = None; // пересборка текстуры
                    }
                }
                ui.separator();
                ui.label("NT:");
                for nt in 0..7 {
                    let n = Nt::from_code(nt as u8);
                    let mut on = self.show_nt[nt];
                    let txt = ui.checkbox(&mut on, n.name());
                    if txt.changed() {
                        self.show_nt[nt] = on;
                        self.atlas = None; // фильтр влияет на атлас
                    }
                }
                ui.separator();
                ui.label("Сторона:");
                for (i, name) in ["?", "L", "R", "C"].iter().enumerate() {
                    let mut on = self.show_side[i];
                    if ui.checkbox(&mut on, *name).changed() {
                        self.show_side[i] = on;
                        self.atlas = None;
                    }
                }
            });
            ui.horizontal_wrapped(|ui| {
                if ui
                    .add(egui::Slider::new(&mut self.min_deg, 0..=200).text("мин. степень"))
                    .changed()
                {
                    self.atlas = None;
                }
                ui.separator();
                ui.label("Поиск:");
                let resp = ui.add(
                    egui::TextEdit::singleline(&mut self.search)
                        .desired_width(200.0)
                        .hint_text("cell_type / root_id"),
                );
                let go = resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter));
                if go {
                    self.search_neuron();
                }
                if ui.button("🔍").clicked() {
                    self.search_neuron();
                }
                if !self.search_status.is_empty() {
                    ui.label(egui::RichText::new(&self.search_status).weak());
                }
                ui.separator();
                if ui.button("⟳ Fit view").clicked() {
                    self.pan = vec2(80.0, 40.0);
                    self.zoom = 1.0;
                }
            });
        });
    }

    fn right_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("inspector")
            .default_width(320.0)
            .show(ctx, |ui| {
                ui.heading("Inspector");
                let Some(data) = self.data.as_ref() else {
                    ui.weak("данные не загружены");
                    return;
                };
                // собираем все owned-данные панели пока data заимствован, чтобы
                // в замыкании ScrollArea можно было мутабельно вызвать self.select
                enum Panel {
                    None,
                    OutOfRange(usize),
                    Info { items: Vec<(usize, String)> },
                }
                let panel = match self.selected {
                    Some(i) if i < data.n => {
                        let mut items = Vec::with_capacity(self.ego.len());
                        for &(t, w, out) in &self.ego {
                            if t >= data.n {
                                continue;
                            }
                            let dir = if out { "→" } else { "←" };
                            let ct = if data.cell_type(t).is_empty() {
                                "?"
                            } else {
                                data.cell_type(t)
                            };
                            items.push((
                                t,
                                format!("{dir} #{t}  {ct}  w={w:.1}  {}", data.nt(t).name()),
                            ));
                        }
                        Panel::Info { items }
                    }
                    None => Panel::None,
                    Some(i) => Panel::OutOfRange(i),
                };
                match panel {
                    Panel::Info { items } => {
                        let Some(i) = self.selected else { return };
                        let Some(data) = self.data.as_ref() else {
                            return;
                        };
                        ui.label(egui::RichText::new(format!("#{i}")).strong().size(18.0));
                        ui.label(format!("root_id: {}", data.roots[i]));
                        ui.label(format!(
                            "cell_type: {}",
                            if data.cell_type(i).is_empty() {
                                "—"
                            } else {
                                data.cell_type(i)
                            }
                        ));
                        ui.label(format!("NT: {}", data.nt(i).name()));
                        let side = ["?", "left", "right", "center"][data.side[i].min(3) as usize];
                        ui.label(format!("side: {side}"));
                        ui.label(format!("degree (out): {}", data.degree(i)));
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!("Ego-граф ({} связей):", items.len()))
                                .strong(),
                        );
                        egui::ScrollArea::vertical()
                            .max_height(ui.available_height() - 30.0)
                            .show(ui, |ui| {
                                for (t, label) in items {
                                    let sel = self.selected == Some(t);
                                    if ui.selectable_label(sel, label).clicked() {
                                        self.select(t);
                                    }
                                }
                            });
                    }
                    Panel::None => {
                        ui.weak("Кликните по нейрону — увидите его связи (ego-граф).");
                    }
                    Panel::OutOfRange(i) => {
                        ui.weak(format!("#{i} вне диапазона"));
                    }
                }
                ui.separator();
                egui::CollapsingHeader::new("Легенда")
                    .default_open(true)
                    .show(ui, |ui| {
                        for nt in 0..7 {
                            let n = Nt::from_code(nt as u8);
                            ui.horizontal(|ui| {
                                let (r, g, b) = (n.color().r(), n.color().g(), n.color().b());
                                ui.add(egui::Label::new(
                                    egui::RichText::new("██").color(Color32::from_rgb(r, g, b)),
                                ));
                                ui.label(n.name());
                            });
                        }
                    });
            });
    }

    fn bottom_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(h) = self.hover {
                    if let Some(d) = self.data.as_ref() {
                        if h < d.n {
                            ui.label(format!(
                                "#{h} · {} · {} · deg={}",
                                if d.cell_type(h).is_empty() {
                                    "?"
                                } else {
                                    d.cell_type(h)
                                },
                                d.nt(h).name(),
                                d.degree(h)
                            ));
                        }
                    }
                } else {
                    ui.label(
                        egui::RichText::new("scroll = зум · drag = пан · клик = ego-граф").weak(),
                    );
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("zoom {:.0}%", self.zoom * 100.0));
                    if let Some(d) = &self.data {
                        ui.label(format!("{}/{}", d.n, d.s));
                    }
                });
            });
        });
    }
}

/// Стрелка/дуга между двумя точками (мировые → уже экранные координаты).
fn draw_arrow(painter: &egui::Painter, a: Pos2, b: Pos2, color: Color32) {
    if (b - a).length() < 1.0 {
        return; // self-loop / дубликат — не рисуем вырожденную петлю
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

impl eframe::App for FlyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.top_panel(ctx);
        self.right_panel(ctx);
        self.bottom_panel(ctx);
        // нет анимаций — egui репейнтит по событиям ввода (hover/zoom/pan) сам

        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            let canvas_h = (avail.y - 60.0).max(200.0);
            let (resp, painter) =
                ui.allocate_painter(vec2(avail.x, canvas_h), Sense::click_and_drag());
            let origin = resp.rect.min;
            let size = resp.rect.size();
            let pointer_screen = resp.interact_pointer_pos().map(|p| p - origin);
            let pointer_world = pointer_screen.map(|s| self.screen_to_world(s));

            // ── zoom around cursor ──
            let zoom_delta = ui.input(|i| i.zoom_delta());
            if (zoom_delta - 1.0).abs() > 1e-4 {
                let anchor = pointer_world.unwrap_or(self.screen_to_world(size * 0.5));
                self.zoom = (self.zoom * zoom_delta).clamp(0.3, 60.0);
                if let Some(s) = pointer_screen {
                    self.pan = s - anchor * (self.zoom * BASE);
                }
            }
            if resp.dragged_by(PointerButton::Middle) {
                self.pan += resp.drag_delta();
            }

            // ── миникарта-навигация: только по клику или драгу, не по hover ──
            let mm = self.minimap_rect(size);
            let over_mm = pointer_screen.is_some_and(|s| mm.contains(pos2(s.x, s.y)));
            if resp.drag_started() {
                self.minimap_drag = over_mm;
            }
            if resp.drag_stopped() {
                self.minimap_drag = false; // не даём флагу «залипнуть» — иначе канвас мёртв
            }
            if self.minimap_drag && resp.dragged_by(PointerButton::Primary) {
                // драг, начавшийся на миникарте, продолжает навигацию и вне её
                if let Some(s) = pointer_screen {
                    self.minimap_pan(size, s);
                }
            } else if over_mm && resp.clicked() {
                // одиночный клик по миникарте — центрируем
                if let Some(s) = pointer_screen {
                    self.minimap_pan(size, s);
                }
            } else if !over_mm {
                // pan / drag по канвасу
                if resp.dragged_by(PointerButton::Primary) {
                    let drag_on_node =
                        pointer_world
                            .and_then(|w| self.nearest(w))
                            .is_some_and(|n| {
                                let np = self.data.as_ref().unwrap().world(n, self.proj);
                                let pp =
                                    pointer_world.map(|w| self.to_screen(w)).unwrap_or_default();
                                (self.to_screen(np) - pp).length() < 12.0
                            });
                    if !drag_on_node {
                        self.pan += resp.drag_delta();
                    }
                }
                if resp.clicked() {
                    if let Some(w) = pointer_world {
                        if let Some(n) = self.nearest(w) {
                            let np = self.data.as_ref().unwrap().world(n, self.proj);
                            if (self.to_screen(np) - self.to_screen(w)).length() < 10.0 {
                                self.select(n);
                            }
                        }
                    }
                }
            }

            // ── hover ──
            self.hover = if over_mm {
                None
            } else {
                pointer_world.and_then(|w| self.nearest(w))
            };
            if let Some(h) = self.hover {
                if let Some(d) = self.data.as_ref() {
                    if h < d.n {
                        let _ = resp.on_hover_text(format!(
                            "#{h} · {} · {} · deg={}",
                            if d.cell_type(h).is_empty() {
                                "?"
                            } else {
                                d.cell_type(h)
                            },
                            d.nt(h).name(),
                            d.degree(h)
                        ));
                    }
                }
            }

            self.draw_canvas(&painter, origin, size);
            ui.separator();
            ui.label(egui::RichText::new(&self.status).weak());
        });
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1500.0, 950.0])
            .with_title("Pillars — Fly Brain Atlas (FAFB v783)"),
        ..Default::default()
    };
    eframe::run_native(
        "Fly Brain Atlas",
        options,
        Box::new(|_cc| Box::new(FlyApp::default())),
    )
}

// ─── Тесты ─────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_data() -> FlyData {
        // 4 нейрона: 0→1, 1→2, 2→0, 3 изолирован
        FlyData {
            n: 4,
            s: 3,
            xs: vec![0.1, 0.5, 0.9, 0.2],
            ys: vec![0.2, 0.6, 0.3, 0.8],
            zs: vec![0.0, 0.5, 1.0, 0.4],
            nt: vec![0, 1, 2, 6],
            side: vec![1, 2, 0, 3],
            roots: vec![10, 20, 30, 40],
            ct_idx: vec![0, 0, 1, 0],
            cell_types: vec!["A".into(), "B".into()],
            row_ptr: vec![0, 1, 2, 3, 3],
            col: vec![1, 2, 0],
            val: vec![0.5, 0.7, 0.9],
            rev_row_ptr: None,
            rev_col: None,
            rev_val: None,
        }
    }

    #[test]
    fn test_proj() {
        let d = tiny_data();
        assert_eq!(d.world(0, Proj::Xy), vec2(0.1, 0.2));
        assert_eq!(d.world(0, Proj::Xz), vec2(0.1, 0.0));
        assert_eq!(d.world(0, Proj::Yz), vec2(0.2, 0.0));
    }

    #[test]
    fn test_ego_edges_out_in() {
        let mut d = tiny_data();
        let edges = d.ego_edges(0, 10);
        // out: 0→1; in: 2→0
        assert!(edges.iter().any(|&(t, _, out)| t == 1 && out));
        assert!(edges.iter().any(|&(t, _, out)| t == 2 && !out));
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_ego_edges_sorted_by_weight() {
        let mut d = tiny_data();
        d.ego_edges(0, 10);
        // вес 2→0 = 0.9 > 0→1 = 0.5 — входящая связь тяжелее
        let edges = d.ego_edges(0, 10);
        assert_eq!(edges[0], (2, 0.9, false));
    }

    #[test]
    fn test_build_atlas_size() {
        let d = tiny_data();
        let px = build_atlas(&d, Proj::Xy, 32, &[true; 7], &[true; 4], 0);
        assert_eq!(px.len(), 32 * 32);
        // фон не пустой и не белый
        assert_ne!(px[0], Color32::from_rgb(255, 255, 255));
    }

    #[test]
    fn test_build_atlas_filters() {
        let d = tiny_data();
        // все NT выключены → атлас = чистый фон
        let px = build_atlas(&d, Proj::Xy, 32, &[false; 7], &[true; 4], 0);
        assert_eq!(px[0], Color32::from_rgb(14, 14, 24));
        // min_deg выше максимальной степени → фон
        let px2 = build_atlas(&d, Proj::Xy, 32, &[true; 7], &[true; 4], 100);
        assert_eq!(px2[0], Color32::from_rgb(14, 14, 24));
    }

    #[test]
    fn test_nearest() {
        let mut app = FlyApp::default();
        app.data = Some(tiny_data());
        app.proj = Proj::Xy;
        // точка у (0.5, 0.6) — ближе всего нейрон 1
        let n = app.nearest(vec2(0.51, 0.59));
        assert_eq!(n, Some(1));
    }

    #[test]
    fn test_load_missing_file() {
        let r = FlyData::load(Path::new("/nonexistent/fly_viz.bin"));
        assert!(r.is_err());
    }
}
