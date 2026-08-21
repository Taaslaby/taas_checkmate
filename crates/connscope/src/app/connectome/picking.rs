//! # Выбор и поиск нейронов (`app::picking`)
//!
//! * **Фильтры видимости** — `visible` (есть 3D-позиция + НТ включён +
//!   класс подходит + степень ≥ мин.);
//! * **Хит-тест** — `nearest` (2D, по мировым координатам) и `nearest_screen`
//!   (3D, по экранным) — линейный скан по 102K нейронов, в release это
//!   доли миллисекунды;
//! * **`select`** — сердце стенда: клик по нейрону тянет ego-рёбра,
//!   пересобирает интерактивный ego-граф и сбрасывает растормаживание;
//! * **`search_neuron`** — поиск по типу / классу / НТ / ноге / стороне / body_id.

use eframe::egui::Vec2;

use crate::app::connectome::ConnectomeMode;

impl ConnectomeMode {
    /// Проходит ли нейрон текущие фильтры (позиция, NT, класс, мин. степень).
    pub(crate) fn visible(&self, i: usize) -> bool {
        let d = self.data.as_ref().unwrap();
        d.has_pos(i)
            && self.filters.show_nt[d.nt[i] as usize]
            && (self.filters.class_filter < 0 || d.klass[i] == self.filters.class_filter as u8)
            && d.out_deg(i) >= self.filters.min_deg
    }

    /// Ближайший видимый нейрон к мировым координатам (для 2D-вида).
    pub(crate) fn nearest(&self, w: Vec2) -> Option<usize> {
        let d = self.data.as_ref()?;
        let mut best: Option<(usize, f32)> = None;
        for i in 0..d.n {
            if !self.visible(i) {
                continue;
            }
            let p = d.world(i, self.camera.proj);
            let dist = (p - w).length_sq();
            if best.is_none_or(|(_, bd)| dist < bd) {
                best = Some((i, dist));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Ближайший видимый нейрон к экранным координатам (для 3D-вида).
    /// Полный скан — точный, для клика (разовый, не на кадр).
    pub(crate) fn nearest_screen(&self, s: Vec2) -> Option<usize> {
        let d = self.data.as_ref()?;
        let mut best: Option<(usize, f32)> = None;
        for i in 0..d.n {
            if !self.visible(i) {
                continue;
            }
            let sp = self.to_screen(self.proj_point(i));
            let dist = (sp - s).length_sq();
            if best.is_none_or(|(_, bd)| dist < bd) {
                best = Some((i, dist));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Ближайший видимый нейрон к экранным координатам в 2D-виде, не дальше
    /// `max_px` пикселей (hover-тултип: линейный скан по ~100K — доли мс
    /// в release, только на pointer-move).
    pub(crate) fn nearest_screen_2d(&self, s: Vec2, max_px: f32) -> Option<usize> {
        let d = self.data.as_ref()?;
        let max_sq = max_px * max_px;
        let mut best: Option<(usize, f32)> = None;
        for i in 0..d.n {
            if !self.visible(i) {
                continue;
            }
            let sp = self.to_screen(d.world(i, self.camera.proj));
            let dist = (sp - s).length_sq();
            if dist <= max_sq && best.is_none_or(|(_, bd)| dist < bd) {
                best = Some((i, dist));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Быстрый hover-хит (3D): проекция только **представителей LOD** —
    /// ~15K точек вместо полного скана 139K каждый кадр, пока мышь над
    /// канвасом. Точность до ячейки сетки (≈1.5% ширины мозга) — для
    /// тултипа/статуса этого достаточно, клик остаётся точным.
    pub(crate) fn nearest_rep_screen(&self, s: Vec2) -> Option<usize> {
        let level = if self.camera.lod.levels.len() > 1 {
            1
        } else {
            0
        };
        let mut best: Option<(usize, f32)> = None;
        for &i in &self.camera.lod.levels[level] {
            if !self.visible(i) {
                continue;
            }
            let sp = self.to_screen(self.proj_point(i));
            let dist = (sp - s).length_sq();
            if best.is_none_or(|(_, bd)| dist < bd) {
                best = Some((i, dist));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Выбрать нейрон: заполнить ego-рёбра (топ по весу), пересобрать
    /// интерактивный ego-граф, очистить результаты растормаживания.
    pub(crate) fn select(&mut self, i: usize) {
        self.sel.selected = Some(i);
        self.sel.ego = if let Some(d) = self.data.as_mut() {
            // The UI needs one readable edge per neighbour/direction, not a
            // raw list of parallel synapses. Aggregation also keeps the force
            // layout bounded for dense connectomes.
            d.ego_edges_agg(i, 64)
        } else {
            Vec::new()
        };
        self.rebuild_ego_graph();
        self.disin.chains = Vec::new();
        self.disin.status.clear();
    }

    /// Поиск первого нейрона, у которого тип/класс/НТ/нога/сторона/body_id
    /// содержит запрос; найденный — выбирается и камера наводится на него.
    pub(crate) fn search_neuron(&mut self) {
        let q = self.sel.search.trim().to_lowercase();
        if q.is_empty() {
            self.sel.search_status.clear();
            return;
        }
        let Some(d) = self.data.as_ref() else {
            self.sel.search_status = "нет данных".into();
            return;
        };
        // «haystack» — все текстовые поля нейрона, склеенные в одну строку.
        let mut found = None;
        for i in 0..d.n {
            let hay = format!(
                "{} {} {} {} {} {}",
                d.type_name(i),
                d.klass(i).name(),
                d.nt(i).name(),
                d.leg_name(i),
                d.side_name(i),
                d.body_id[i]
            )
            .to_lowercase();
            if hay.contains(&q) {
                found = Some(i);
                break;
            }
        }
        match found {
            Some(i) => {
                let info = d.type_name(i).to_string();
                let nt_name = d.nt(i).name().to_string();
                self.select(i);
                self.focus_on(i);
                self.sel.search_status = format!(
                    "→ #{i} {} · {nt_name}",
                    if info.is_empty() { "?" } else { &info }
                );
            }
            None => self.sel.search_status = format!("ничего не найдено: {q}"),
        }
    }
}
