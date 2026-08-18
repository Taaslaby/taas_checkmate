//! # Уровни детализации облака (`lod`)
//!
//! Как в движках для облаков точек (Potree / Blender / deck.gl): на дальнем
//! плане рисуем по одному **представителю** на ячейку сетки — визуально то
//! же самое (точки и так перекрываются), а отрисовка в разы дешевле.
//! При приближении уровень растёт — появляются все нейроны.
//!
//! Уровни (строятся один раз при загрузке данных):
//!
//! | уровень | что рисуем | порог `scale` (zoom·BASE) |
//! |---|---|---|
//! | 0 | все точки | ≥ 900 (приблизился) |
//! | 1 | представители сетки 1/64 | 320 … 900 |
//! | 2 | представители сетки 1/32 | < 320 (общий план) |
//!
//! Компенсация разрежения: точки грубых уровней рисуются крупнее
//! (`boosts`), чтобы облако не «дырявело».

use crate::io::data::MancData;

/// Уровни детализации облака нейронов.
pub struct Lod {
    /// `levels[0]` = все точки; дальше — представители всё более крупных ячеек.
    pub levels: Vec<Vec<usize>>,
    /// Пороги `scale` (zoom·BASE): уровень 0 при `scale ≥ thresholds[0]`,
    /// уровень i при `thresholds[i] > scale ≥ thresholds[i+1]`.
    pub thresholds: Vec<f32>,
    /// Компенсация разрежения: во сколько раз крупнее рисовать точки уровня.
    pub boosts: Vec<f32>,
}

impl Lod {
    /// Пустой LOD — до загрузки данных (всё рисуется «как есть»).
    pub fn empty() -> Self {
        Self {
            levels: vec![Vec::new()],
            thresholds: Vec::new(),
            boosts: vec![1.0],
        }
    }

    /// Уровень для данного `scale` (zoom·BASE): чем меньше, тем грубее.
    pub fn level_for(&self, scale: f32) -> usize {
        let mut lvl = 0;
        for (i, &th) in self.thresholds.iter().enumerate() {
            if scale < th {
                lvl = i + 1;
            } else {
                break;
            }
        }
        lvl.min(self.levels.len().saturating_sub(1))
    }
}

/// Строит LOD для мозга: уровень 0 = все точки, 1 = представители сетки
/// 1/64, 2 = представители сетки 1/32.
pub fn build_lod(data: &MancData) -> Lod {
    let all: Vec<usize> = (0..data.n).filter(|&i| data.has_pos(i)).collect();
    Lod {
        levels: vec![all, reps_by_grid(data, 64), reps_by_grid(data, 32)],
        thresholds: vec![900.0, 320.0],
        boosts: vec![1.0, 1.7, 2.6],
    }
}

/// Представители: первый нейрон с позицией в каждой ячейке сетки `grid³`.
fn reps_by_grid(data: &MancData, grid: usize) -> Vec<usize> {
    let g = grid as f32;
    let maxc = (grid - 1) as f32;
    let mut seen: std::collections::HashSet<(usize, usize, usize)> =
        std::collections::HashSet::new();
    let mut reps: Vec<usize> = Vec::new();
    for i in 0..data.n {
        if !data.has_pos(i) {
            continue;
        }
        let k = (
            (data.xs[i] * g).floor().clamp(0.0, maxc) as usize,
            (data.ys[i] * g).floor().clamp(0.0, maxc) as usize,
            (data.zs[i] * g).floor().clamp(0.0, maxc) as usize,
        );
        if seen.insert(k) {
            reps.push(i);
        }
    }
    reps
}

// ── Тесты ─────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::build_lod;
    use crate::io::data::tiny_data;

    /// LOD: уровни детализации строятся и переключаются по scale.
    #[test]
    fn test_lod_levels() {
        let d = tiny_data();
        let lod = build_lod(&d);
        // уровень 0 = все точки с позицией (у tiny_data все 4 имеют позицию)
        assert_eq!(lod.levels[0].len(), 4);
        // представители — непусто и не больше полного уровня
        assert!(!lod.levels[1].is_empty() && lod.levels[1].len() <= 4);
        assert!(!lod.levels[2].is_empty() && lod.levels[2].len() <= 4);
        // пороги: далеко — грубый уровень, близко — полный
        assert_eq!(lod.level_for(1200.0), 0);
        assert_eq!(lod.level_for(500.0), 1);
        assert_eq!(lod.level_for(100.0), 2);
        // представители — валидные индексы нейронов
        for &i in &lod.levels[1] {
            assert!(i < d.n);
        }
    }
}
