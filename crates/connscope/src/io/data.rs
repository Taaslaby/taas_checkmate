//! # Данные коннектома (`io::data`)
//!
//! Центральная модель в памяти — [`MancData`] — плюс загрузка из двух
//! форматов (выбор по magic-байтам в [`MancData::detect`]):
//!
//! * **`.bin`** — самодостаточный снапшот (печётся один раз `bake_manc_viz.py`):
//!   позиции `[0,1]³`, нейромедиаторы, классы, типы/подклассы/мишени, CSR-граф;
//! * **`.taas`** — нативный формат checkmate (`crates/taas-format`).
//!
//! Поля: `xs/ys/zs` (нормализованные voxel-координаты, `has_pos` — есть ли
//! позиция у нейрона), `nt` (нейромедиатор), `klass`/`leg`/`side` (метаданные),
//! `row_ptr/col/val` — CSR-матрица синапсов (плюс опциональный обратный CSR
//! `rev_*` для быстрого «кто в меня входит»). Осевые проекции — [`Proj`].
//!
//! Полезные хелперы: `n()`/`s()`, `nt(i)`/`klass(i)`, `type_name`/`subclass_name`,
//! `in_deg`/`out_deg`, `ego_edges` (рёбра выбранного нейрона), `world(i, proj)`
//! (экранные координаты проекции).

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use eframe::egui::{vec2, Vec2};

use crate::core::nt::{Klass, Nt};

/// Нормализует путь для отображения: убирает сегменты `..`/`.` и делает
/// абсолютным (иначе в поле пути видно `crates\connscope\..\..\resources…`).
/// ⚠ Не `canonicalize()`: он на Windows возвращает `\\?\`-префикс
/// (extended-length path) — в интерфейсе выглядит как мусор в начале пути.
/// `std::path::absolute` (GetFullPathNameW) разворачивает `..` без префикса.
pub fn clean_path(p: &Path) -> PathBuf {
    let p = std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    // Страховка: если путь пришёл из диалога ОС уже с `\\?\` — срезаем.
    match p.to_string_lossy().strip_prefix(r"\\?\") {
        Some(rest) => PathBuf::from(rest),
        None => p,
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

// ─── Данные коннектома ────────────────────────────────────────
pub struct MancData {
    pub n: usize,
    pub s: usize,
    pub xs: Vec<f32>,
    pub ys: Vec<f32>,
    pub zs: Vec<f32>,
    pub has_pos: Vec<u8>,
    pub nt: Vec<u8>,
    pub klass: Vec<u8>,
    pub leg: Vec<u8>,
    pub side: Vec<u8>,
    pub nt_types: Vec<f32>,
    pub body_id: Vec<u64>,
    pub type_idx: Vec<u16>,
    pub subclass_idx: Vec<u16>,
    pub target_idx: Vec<u16>,
    pub types: Vec<String>,
    pub subclasses: Vec<String>,
    pub targets: Vec<String>,
    pub row_ptr: Vec<i32>,
    pub col: Vec<i32>,
    pub val: Vec<f32>,
    pub rev_row_ptr: Option<Vec<i32>>,
    pub rev_col: Option<Vec<i32>>,
    pub rev_val: Option<Vec<f32>>,
}

impl MancData {
    /// Парсит `manc_viz.bin`.
    pub fn load(path: &Path) -> Result<MancData, String> {
        let mut buf = Vec::new();
        File::open(path)
            .map_err(|e| format!("нет данных {path:?}: {e} (запустите bake_manc_viz.py)"))?
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
        let magic = u32::from_le_bytes(take!(4).try_into().unwrap());
        if magic != 0x4D4E4356 {
            return Err(format!("не тот формат (magic {magic:#x}, ожидался MNCV)"));
        }
        let ver = u32::from_le_bytes(take!(4).try_into().unwrap());
        if ver != 1 {
            return Err(format!("неподдерживаемая версия {ver}"));
        }
        let n = u64::from_le_bytes(take!(8).try_into().unwrap()) as usize;
        let s = u64::from_le_bytes(take!(8).try_into().unwrap()) as usize;
        let n_types = u32::from_le_bytes(take!(4).try_into().unwrap()) as usize;
        let n_sub = u32::from_le_bytes(take!(4).try_into().unwrap()) as usize;
        let n_tgt = u32::from_le_bytes(take!(4).try_into().unwrap()) as usize;

        let f32s = |b: &[u8]| {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<f32>>()
        };
        let u8s = |b: &[u8]| b.to_vec();
        let u64s = |b: &[u8]| {
            b.chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<u64>>()
        };
        let u16s = |b: &[u8]| {
            b.chunks_exact(2)
                .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<u16>>()
        };
        let i32s = |b: &[u8]| {
            b.chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<i32>>()
        };
        macro_rules! read_strings {
            ($count:expr) => {{
                let mut out = Vec::with_capacity($count);
                for _ in 0..$count {
                    let len_bytes = buf.get(p..p + 2).expect("файл повреждён (обрезка)");
                    p += 2;
                    let len = u16::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
                    let bytes = buf.get(p..p + len).expect("файл повреждён (обрезка)");
                    p += len;
                    out.push(String::from_utf8_lossy(bytes).into_owned());
                }
                out
            }};
        }

        let xs = f32s(take!(n * 4));
        let ys = f32s(take!(n * 4));
        let zs = f32s(take!(n * 4));
        let has_pos = u8s(take!(n));
        let nt = u8s(take!(n));
        let klass = u8s(take!(n));
        let leg = u8s(take!(n));
        let side = u8s(take!(n));
        let nt_types = f32s(take!(n * 4));
        let body_id = u64s(take!(n * 8));
        let type_idx = u16s(take!(n * 2));
        let subclass_idx = u16s(take!(n * 2));
        let target_idx = u16s(take!(n * 2));
        let types = read_strings!(n_types);
        let subclasses = read_strings!(n_sub);
        let targets = read_strings!(n_tgt);
        let row_ptr = i32s(take!((n + 1) * 4));
        let col = i32s(take!(s * 4));
        let val = f32s(take!(s * 4));
        let _ = p;

        Ok(MancData {
            n,
            s,
            xs,
            ys,
            zs,
            has_pos,
            nt,
            klass,
            leg,
            side,
            nt_types,
            body_id,
            type_idx,
            subclass_idx,
            target_idx,
            types,
            subclasses,
            targets,
            row_ptr,
            col,
            val,
            rev_row_ptr: None,
            rev_col: None,
            rev_val: None,
        })
    }

    /// Парсит `.taas` (baked-формат pillars-connectome): CSR-граф без 3D-позиций.
    /// Формат: [magic(4) | ver(4) | n(8) | s(8) | row_ptr(4·(n+1)) | col(4·s) |
    ///          val(4·s) | nt_types(4·n) | has_receptors(1) | receptor_types(1·s)?]
    pub fn load_taas(path: &Path) -> Result<MancData, String> {
        let mut buf = Vec::new();
        File::open(path)
            .map_err(|e| format!("нет данных {path:?}: {e}"))?
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
        let magic = u32::from_le_bytes(take!(4).try_into().unwrap());
        if magic != 0x54414153 {
            return Err(format!("не тот формат (magic {magic:#x}, ожидался TAAS)"));
        }
        let _ver = u32::from_le_bytes(take!(4).try_into().unwrap());
        let n = u64::from_le_bytes(take!(8).try_into().unwrap()) as usize;
        let s = u64::from_le_bytes(take!(8).try_into().unwrap()) as usize;

        let f32s = |b: &[u8]| {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<f32>>()
        };
        let i32s = |b: &[u8]| {
            b.chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<i32>>()
        };

        let row_ptr = i32s(take!((n + 1) * 4));
        let col = i32s(take!(s * 4));
        let val = f32s(take!(s * 4));
        let nt_types = f32s(take!(n * 4));
        // receptor_types (опционально) — не нужны для просмотра/сима
        if let Some(&has_rc) = buf.get(p) {
            debug_assert!(has_rc <= 1, "неожиданный флаг receptor_types");
        }

        // у .taas только знак НТ: +1 → возб. (ACh), −1 → торм. (GABA)
        let nt: Vec<u8> = nt_types
            .iter()
            .map(|&v| {
                if v > 0.0 {
                    1
                } else if v < 0.0 {
                    2
                } else {
                    0
                }
            })
            .collect();

        Ok(MancData {
            n,
            s,
            xs: vec![0.0; n],
            ys: vec![0.0; n],
            zs: vec![0.0; n],
            has_pos: vec![0; n],
            nt,
            klass: vec![0; n],
            leg: vec![0; n],
            side: vec![0; n],
            nt_types,
            body_id: (0..n as u64).collect(),
            type_idx: vec![0; n],
            subclass_idx: vec![0; n],
            target_idx: vec![0; n],
            types: Vec::new(),
            subclasses: Vec::new(),
            targets: Vec::new(),
            row_ptr,
            col,
            val,
            rev_row_ptr: None,
            rev_col: None,
            rev_val: None,
        })
    }

    /// Определяет формат: по расширению — сырые коннектомы без конвертации
    /// (`.graphml` / `.npz`), по magic — запечённые (`.bin` MNCV / `.taas`).
    pub fn detect(path: &Path) -> Result<MancData, String> {
        // Сырые форматы — без конвертации: GraphML (networkx) и FAFB npz
        // (координаты вторым файлом + csv.gz аннотации flywire рядом).
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref()
        {
            Some("graphml") => return crate::io::raw::from_graphml(path),
            Some("npz") => return crate::io::raw::from_fafb_npz(path),
            _ => {}
        }
        let mut head = [0u8; 4];
        if let Ok(mut f) = File::open(path) {
            let _ = f.read(&mut head);
        } else {
            return Err(format!("нет данных {path:?}"));
        }
        let magic = u32::from_le_bytes(head);
        match magic {
            0x4D4E4356 => MancData::load(path),
            0x54414153 => MancData::load_taas(path),
            _ => Err(format!(
                "неизвестный формат (magic {magic:#x}): ожидался MNCV (.bin), TAAS (.taas), GraphML (.graphml) или FAFB npz (.npz)"
            )),
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }
    pub fn s(&self) -> usize {
        self.s
    }
    pub fn world(&self, i: usize, proj: Proj) -> Vec2 {
        proj.pick(self.xs[i], self.ys[i], self.zs[i])
    }
    pub fn nt(&self, i: usize) -> Nt {
        Nt::from_code(self.nt[i])
    }
    pub fn klass(&self, i: usize) -> Klass {
        Klass::from_code(self.klass[i])
    }
    pub fn type_name(&self, i: usize) -> &str {
        // «None» в данных — это отсутствие типа, а не название: возвращаем
        // пустую строку, чтобы UI показывал «#idx»/«?», а не «None».
        self.types
            .get(self.type_idx[i] as usize)
            .map(|s| if s == "None" { "" } else { s.as_str() })
            .unwrap_or("")
    }
    pub fn subclass(&self, i: usize) -> &str {
        self.subclasses
            .get(self.subclass_idx[i] as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }
    pub fn target(&self, i: usize) -> &str {
        self.targets
            .get(self.target_idx[i] as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }
    pub fn leg_name(&self, i: usize) -> &'static str {
        match self.leg[i] {
            1 => "T1",
            2 => "T2",
            3 => "T3",
            4 => "abdomen",
            _ => "other",
        }
    }
    pub fn side_name(&self, i: usize) -> &'static str {
        match self.side[i] {
            1 => "L",
            2 => "R",
            _ => "other",
        }
    }
    pub fn has_pos(&self, i: usize) -> bool {
        i < self.n && self.has_pos[i] != 0
    }
    /// Есть ли вообще 3D-позиции (иначе атлас пустой, доступны только граф/сим).
    pub fn has_any_pos(&self) -> bool {
        self.has_pos.iter().any(|&x| x != 0)
    }
    pub fn out_deg(&self, i: usize) -> usize {
        if i >= self.n {
            return 0;
        }
        (self.row_ptr[i + 1].max(0) - self.row_ptr[i].max(0)).max(0) as usize
    }

    /// Построить обратный CSR (post → pre) — лениво, один раз.
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

    pub fn in_deg(&mut self, node: usize) -> usize {
        self.ensure_reverse();
        if node >= self.n {
            return 0;
        }
        (self.rev_row_ptr.as_ref().unwrap()[node + 1].max(0)
            - self.rev_row_ptr.as_ref().unwrap()[node].max(0))
        .max(0) as usize
    }

    /// Топ-K соседей ego-графа: (target, вес, out=true исходящий).
    pub fn ego_edges(&mut self, node: usize, k: usize) -> Vec<(usize, f32, bool)> {
        self.ensure_reverse();
        let mut edges: Vec<(usize, f32, bool)> = Vec::new();
        let lo = self.row_ptr[node].max(0) as usize;
        let hi = self.row_ptr[node + 1].max(0) as usize;
        for e in lo..hi.min(self.s) {
            let t = self.col[e];
            if t >= 0 && (t as usize) < self.n {
                edges.push((t as usize, self.val[e], true));
            }
        }
        if let (Some(rp), Some(rc), Some(rv)) = (&self.rev_row_ptr, &self.rev_col, &self.rev_val) {
            let lo = rp[node].max(0) as usize;
            let hi = rp[node + 1].max(0) as usize;
            for e in lo..hi.min(self.s) {
                let src = rc[e];
                if src >= 0 && (src as usize) < self.n {
                    edges.push((src as usize, rv[e], false));
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

    /// Топ-K соседей ego **с агрегацией мультисинапсов**: одна запись на
    /// пару нейронов — (цель, сумма весов, направление, число синапсов).
    /// Между двумя нейронами бывают десятки синапсов — сырой список
    /// дублирует пару, а здесь видно настоящую силу связи.
    pub fn ego_edges_agg(&mut self, node: usize, k: usize) -> Vec<(usize, f32, bool, usize)> {
        self.ensure_reverse();
        // (цель, направление) → (сумма весов, число синапсов)
        let mut agg: std::collections::HashMap<(usize, bool), (f32, usize)> =
            std::collections::HashMap::new();
        let lo = self.row_ptr[node].max(0) as usize;
        let hi = self.row_ptr[node + 1].max(0) as usize;
        for e in lo..hi.min(self.s) {
            let t = self.col[e];
            if t >= 0 && (t as usize) < self.n {
                let e_ = agg.entry((t as usize, true)).or_insert((0.0, 0));
                e_.0 += self.val[e];
                e_.1 += 1;
            }
        }
        if let (Some(rp), Some(rc), Some(rv)) = (&self.rev_row_ptr, &self.rev_col, &self.rev_val) {
            let lo = rp[node].max(0) as usize;
            let hi = rp[node + 1].max(0) as usize;
            for e in lo..hi.min(self.s) {
                let src = rc[e];
                if src >= 0 && (src as usize) < self.n {
                    let e_ = agg.entry((src as usize, false)).or_insert((0.0, 0));
                    e_.0 += rv[e];
                    e_.1 += 1;
                }
            }
        }
        let mut rows: Vec<(usize, f32, bool, usize)> = agg
            .into_iter()
            .map(|((t, out), (w, n))| (t, w, out, n))
            .collect();
        rows.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rows.truncate(k);
        rows
    }
}

// ── Тесты ─────────────────────────────────────────────────────
/// Минимальный граф из 4 нейронов для тестов (общая фикстура крейта):
/// 0→1 (w=5), 1→2 (w=7), 2→0 (w=9), 3 изолирован.
#[cfg(test)]
pub(crate) fn tiny_data() -> MancData {
    MancData {
        n: 4,
        s: 3,
        xs: vec![0.1, 0.5, 0.9, 0.2],
        ys: vec![0.2, 0.6, 0.3, 0.8],
        zs: vec![0.0, 0.5, 1.0, 0.4],
        has_pos: vec![1, 1, 1, 1],
        nt: vec![1, 2, 3, 0],
        klass: vec![1, 5, 2, 0],
        leg: vec![0, 1, 2, 3],
        side: vec![1, 2, 0, 1],
        nt_types: vec![1.0, -1.0, -1.0, 1.0],
        body_id: vec![10, 20, 30, 40],
        type_idx: vec![0, 0, 1, 0],
        subclass_idx: vec![0, 0, 0, 0],
        target_idx: vec![0, 0, 0, 0],
        types: vec!["A".into(), "B".into()],
        subclasses: vec!["s".into()],
        targets: vec!["t".into()],
        row_ptr: vec![0, 1, 2, 3, 3],
        col: vec![1, 2, 0],
        val: vec![5.0, 7.0, 9.0],
        rev_row_ptr: None,
        rev_col: None,
        rev_val: None,
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use eframe::egui::vec2;

    use super::{tiny_data, MancData, Proj};
    use crate::core::nt::Nt;

    #[test]
    fn test_world_proj() {
        let d = tiny_data();
        assert_eq!(d.world(0, Proj::Xy), vec2(0.1, 0.2));
        assert_eq!(d.world(0, Proj::Xz), vec2(0.1, 0.0));
    }

    #[test]
    fn test_ego_edges_direction_and_sort() {
        let mut d = tiny_data();
        let edges = d.ego_edges(0, 10);
        // in 2→0 (9.0) heavy; out 0→1 (5.0)
        assert!(edges.iter().any(|&(t, _, out)| t == 1 && out));
        assert!(edges.iter().any(|&(t, _, out)| t == 2 && !out));
        assert_eq!(edges[0], (2, 9.0, false));
    }

    #[test]
    fn test_in_degree() {
        let mut d = tiny_data();
        assert_eq!(d.in_deg(0), 1); // 2→0
        assert_eq!(d.in_deg(2), 1); // 1→2
    }

    #[test]
    fn test_load_missing_file() {
        let r = MancData::load(Path::new("/nonexistent/manc_viz.bin"));
        assert!(r.is_err());
    }

    #[test]
    fn test_load_taas() {
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(&0x54414153u32.to_le_bytes()); // magic TAAS
        b.extend_from_slice(&1u32.to_le_bytes()); // version
        b.extend_from_slice(&3u64.to_le_bytes()); // n
        b.extend_from_slice(&3u64.to_le_bytes()); // s
        for v in [0i32, 1, 2, 3] {
            b.extend_from_slice(&v.to_le_bytes()); // row_ptr
        }
        for v in [1i32, 2, 0] {
            b.extend_from_slice(&v.to_le_bytes()); // col
        }
        for v in [5.0f32, 7.0, 9.0] {
            b.extend_from_slice(&v.to_le_bytes()); // val
        }
        for v in [1.0f32, -1.0, -1.0] {
            b.extend_from_slice(&v.to_le_bytes()); // nt_types
        }
        b.push(0); // has_receptors = false
        let path = std::env::temp_dir().join("connscope_test.taas");
        std::fs::write(&path, &b).unwrap();
        let mut d = MancData::load_taas(&path).expect("taas должен парситься");
        std::fs::remove_file(&path).ok();
        assert_eq!(d.n(), 3);
        assert_eq!(d.s(), 3);
        assert!(!d.has_any_pos());
        assert_eq!(d.nt(0), Nt::Ach); // +1 → возб.
        assert_eq!(d.nt(1), Nt::Gaba); // −1 → торм.
        let edges = d.ego_edges(0, 10);
        assert!(edges.iter().any(|&(t, w, out)| t == 1 && w == 5.0 && out)); // 0→1
        assert!(edges.iter().any(|&(t, _, out)| t == 2 && !out)); // 2→0
    }

    /// Проверяет, что Rust-парсер совпадает с бинарником bake_manc_viz.py 1:1.
    #[test]
    fn test_load_real_manc_bin() {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../../resources/examples/data/manc_viz.bin");
        if !p.exists() {
            eprintln!("skip: {p:?} не найден (запустите bake_manc_viz.py)");
            return;
        }
        let mut d = MancData::load(&p).expect("реальный manc_viz.bin должен парситься");
        assert_eq!(d.n(), 102369);
        assert_eq!(d.s(), 5243574);
        assert_eq!(d.type_name(158), "IN19A002");
        assert_eq!(d.nt(158), Nt::Gaba);
        assert!(d.has_pos(158));
        // обратный CSR строится без паники на реальном графе
        let od = d.out_deg(158);
        let id_ = d.in_deg(158);
        assert!(od > 0 || id_ > 0);
    }
}
