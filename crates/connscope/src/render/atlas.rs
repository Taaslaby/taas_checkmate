//! Атласы: 2D-текстура по осевой проекции и 3D-орбитальное облако точек.

use eframe::egui::{vec2, Color32, Vec2};

use crate::core::color::{color_for, ColorMode};
use crate::io::data::{MancData, Proj};

/// Фон атласа (тот же тёмный индиго, что был раньше).
const BG: (f32, f32, f32) = (14.0 / 255.0, 14.0 / 255.0, 24.0 / 255.0);

/// 2D-атлас: проекция точек в текстуру `size×size`.
///
/// Каждая точка — **гауссов сплат** радиусом `splat_radius` текселей
/// (σ = 0.9 текселя): одиночный нейрон остаётся круглой точкой, а не
/// квадратом 3×3. Накопление — float: `Σ цвет·вес` и `Σ вес` по всем
/// точкам, затем тонмаппинг плотности:
///
/// * `alpha = 1 − exp(−k·Σвес)`, k = 0.8 — плотные скопления плавно
///   насыщаются до непрозрачного, а не выжигаются в белую кашу;
/// * `color = Σцвет·вес / max(Σвес, ε)` — средневзвешенный цвет пятна;
/// * композит поверх тёмного фона.
pub fn build_atlas(
    data: &MancData,
    proj: Proj,
    size: usize,
    show_nt: &[bool; 7],
    class_filter: i8,
    min_deg: usize,
    mode: ColorMode,
    splat_radius: f32,
) -> Vec<Color32> {
    // Float-накопители: premultiplied color·weight и weight.
    let mut c_sum = vec![[0.0f32; 3]; size * size];
    let mut w_sum = vec![0.0f32; size * size];

    let sigma = 0.9f32;
    let radius = splat_radius.max(0.5);
    let reach = radius.ceil() as isize;
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    // Гауссово ядро, табулированное по целым смещениям (симметрично).
    let mut kernel = [0.0f32; 8];
    for (d, k) in kernel.iter_mut().enumerate() {
        let r = d as f32;
        *k = (-(r * r) * inv_2s2).exp();
    }
    let center_w = 1.0f32;

    for i in 0..data.n {
        if !data.has_pos(i)
            || !show_nt[data.nt[i] as usize]
            || data.out_deg(i) < min_deg
            || (class_filter >= 0 && data.klass[i] != class_filter as u8)
        {
            continue;
        }
        let w = data.world(i, proj);
        let c = color_for(data, i, mode);
        let col = [
            c.r() as f32 / 255.0,
            c.g() as f32 / 255.0,
            c.b() as f32 / 255.0,
        ];
        let cx = (w.x.clamp(0.0, 1.0) * (size as f32 - 1.0)).round() as isize;
        let cy = (w.y.clamp(0.0, 1.0) * (size as f32 - 1.0)).round() as isize;
        for dy in -reach..=reach {
            let py = cy + dy;
            if py < 0 || py as usize >= size {
                continue;
            }
            let wy = kernel[dy.unsigned_abs() % kernel.len()];
            for dx in -reach..=reach {
                let px_x = cx + dx;
                if px_x < 0 || px_x as usize >= size {
                    continue;
                }
                let wx = kernel[dx.unsigned_abs() % kernel.len()];
                let g = if dx == 0 && dy == 0 {
                    center_w
                } else {
                    wx * wy
                };
                if g < 1e-4 {
                    continue;
                }
                let idx = py as usize * size + px_x as usize;
                w_sum[idx] += g;
                c_sum[idx][0] += col[0] * g;
                c_sum[idx][1] += col[1] * g;
                c_sum[idx][2] += col[2] * g;
            }
        }
    }

    // Тонмаппинг плотности + композит над фоном.
    let k = 0.8f32;
    let eps = 1e-6;
    c_sum
        .iter()
        .zip(&w_sum)
        .map(|(c, &w)| {
            if w <= eps {
                return Color32::from_rgb(14, 14, 24);
            }
            let alpha = 1.0 - (-k * w).exp();
            let inv = 1.0 / w.max(eps);
            let r = (c[0] * inv * alpha + BG.0 * (1.0 - alpha)) * 255.0;
            let g = (c[1] * inv * alpha + BG.1 * (1.0 - alpha)) * 255.0;
            let b = (c[2] * inv * alpha + BG.2 * (1.0 - alpha)) * 255.0;
            Color32::from_rgb(r as u8, g as u8, b as u8)
        })
        .collect()
}

/// 3D-проекция точки (voxel `[0,1]³`) → экран `[0,1]²` + глубина.
/// `yaw` — вращение вокруг Z, `pitch` — вокруг X, `persp` — сила перспективы (0 = орто).
pub fn project3d(x: f32, y: f32, z: f32, yaw: f32, pitch: f32, persp: f32) -> (Vec2, f32) {
    let x = x - 0.5;
    let y = y - 0.5;
    let z = z - 0.5;
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let x1 = x * cy - y * sy;
    let y1 = x * sy + y * cy;
    let y2 = y1 * cp - z * sp;
    let z2 = y1 * sp + z * cp;
    let denom = (1.0 - z2 * persp).max(0.18);
    let s = (1.0 / denom).min(3.0);
    (vec2(x1 * s + 0.5, y2 * s + 0.5), z2)
}

/// Поворот единичного направления (для ориентационного гизмо) — без сдвига и перспективы.
/// Возвращает экранное смещение и глубину (z2 > 0 = ось смотрит к зрителю).
pub fn rotate_dir(x: f32, y: f32, z: f32, yaw: f32, pitch: f32) -> (Vec2, f32) {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let x1 = x * cy - y * sy;
    let y1 = x * sy + y * cy;
    let y2 = y1 * cp - z * sp;
    let z2 = y1 * sp + z * cp;
    (vec2(x1, y2), z2)
}

/// Орбитальный 3D-атлас: вращение + перспектива + глубинное затенение.
pub fn build_atlas_3d(
    data: &MancData,
    yaw: f32,
    pitch: f32,
    persp: f32,
    size: usize,
    show_nt: &[bool; 7],
    class_filter: i8,
    min_deg: usize,
    mode: ColorMode,
) -> Vec<Color32> {
    let mut px = vec![Color32::from_rgb(14, 14, 24); size * size];
    let mut pts: Vec<(f32, f32, f32, Color32)> = Vec::new();
    for i in 0..data.n {
        if !data.has_pos(i)
            || !show_nt[data.nt[i] as usize]
            || data.out_deg(i) < min_deg
            || (class_filter >= 0 && data.klass[i] != class_filter as u8)
        {
            continue;
        }
        let (w, depth) = project3d(data.xs[i], data.ys[i], data.zs[i], yaw, pitch, persp);
        pts.push((depth, w.x, w.y, color_for(data, i, mode)));
    }
    // дальние → ближние (ближние рисуются поверх)
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut dmin = 0.0f32;
    let mut dmax = 0.0f32;
    if let Some(first) = pts.first() {
        dmin = first.0;
        dmax = first.0;
    }
    for p in &pts {
        dmin = dmin.min(p.0);
        dmax = dmax.max(p.0);
    }
    let drange = (dmax - dmin).max(1e-4);
    for &(depth, sx, sy, c) in &pts {
        let t = ((depth - dmin) / drange).clamp(0.0, 1.0); // 0=далеко, 1=близко
        let bright = 0.4 + 0.6 * t;
        let col = Color32::from_rgb(
            (c.r() as f32 * bright) as u8,
            (c.g() as f32 * bright) as u8,
            (c.b() as f32 * bright) as u8,
        );
        let cx = (sx.clamp(0.0, 1.0) * (size as f32 - 1.0)) as usize;
        let cy = (sy.clamp(0.0, 1.0) * (size as f32 - 1.0)) as usize;
        // ближние точки крупнее — перспективный размерный кью (2×2 → 4×4)
        let blk = 2 + (t * 2.0) as usize;
        let h = blk as isize / 2;
        for dy in 0..blk {
            for dx in 0..blk {
                let px_x = cx as isize + dx as isize - h;
                let py_y = cy as isize + dy as isize - h;
                if px_x >= 0 && py_y >= 0 && (px_x as usize) < size && (py_y as usize) < size {
                    px[py_y as usize * size + px_x as usize] = col;
                }
            }
        }
    }
    px
}

/// HSV → Color32 (h: 0..360).
pub fn hsv(h: f32, s: f32, v: f32) -> Color32 {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Color32::from_rgb(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// Заливка квадрата `blk×blk` по центру точки (для атласов).
pub fn blit(px: &mut [Color32], size: usize, sx: f32, sy: f32, col: Color32, blk: usize) {
    let cx = (sx.clamp(0.0, 1.0) * (size as f32 - 1.0)) as usize;
    let cy = (sy.clamp(0.0, 1.0) * (size as f32 - 1.0)) as usize;
    let h = blk as isize / 2;
    for dy in 0..blk {
        for dx in 0..blk {
            let px_x = cx as isize + dx as isize - h;
            let py_y = cy as isize + dy as isize - h;
            if px_x >= 0 && py_y >= 0 && (px_x as usize) < size && (py_y as usize) < size {
                px[py_y as usize * size + px_x as usize] = col;
            }
        }
    }
}

/// 3D-атлас с трассой сигнала: обычное облако (тусклый контекст) + поверх него
/// сработавшие нейроны, окрашенные по времени первого спайка (волна: синий→красный)
/// и увеличенные по числу спайков.
pub fn build_atlas_3d_trace(
    data: &MancData,
    yaw: f32,
    pitch: f32,
    persp: f32,
    size: usize,
    show_nt: &[bool; 7],
    class_filter: i8,
    min_deg: usize,
    first_fire: &[u32],
    spike_counts: &[u32],
    mode: ColorMode,
) -> Vec<Color32> {
    let mut px = vec![Color32::from_rgb(14, 14, 24); size * size];
    let n = data.n;
    let mut max_step = 1u32;
    let mut max_spikes = 1u32;
    let mut fired_any = false;
    for i in 0..n {
        let f = first_fire.get(i).copied().unwrap_or(u32::MAX);
        if f != u32::MAX {
            fired_any = true;
            max_step = max_step.max(f);
            max_spikes = max_spikes.max(spike_counts.get(i).copied().unwrap_or(0));
        }
    }
    let mut ctx_pts: Vec<(f32, f32, f32, Color32, usize)> = Vec::new();
    let mut hot_pts: Vec<(f32, f32, f32, Color32, usize)> = Vec::new();
    for i in 0..n {
        if !data.has_pos(i)
            || !show_nt[data.nt[i] as usize]
            || data.out_deg(i) < min_deg
            || (class_filter >= 0 && data.klass[i] != class_filter as u8)
        {
            continue;
        }
        let (w, depth) = project3d(data.xs[i], data.ys[i], data.zs[i], yaw, pitch, persp);
        let f = first_fire.get(i).copied().unwrap_or(u32::MAX);
        if fired_any && f != u32::MAX {
            let t = f as f32 / max_step as f32;
            let s = spike_counts.get(i).copied().unwrap_or(0) as f32 / max_spikes as f32;
            let col = hsv(200.0 - 200.0 * t, 1.0, 1.0); // синий (рано) → красный (поздно)
            let blk = 3 + (s * 2.0) as usize; // чем чаще спайкал, тем крупнее
            hot_pts.push((depth, w.x, w.y, col, blk));
        } else {
            let c = color_for(data, i, mode);
            let dim = Color32::from_rgb(
                (c.r() as f32 * 0.32) as u8,
                (c.g() as f32 * 0.32) as u8,
                (c.b() as f32 * 0.32) as u8,
            );
            ctx_pts.push((depth, w.x, w.y, dim, 2));
        }
    }
    // контекст: дальние → ближние
    ctx_pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for &(_, sx, sy, col, blk) in &ctx_pts {
        blit(&mut px, size, sx, sy, col, blk);
    }
    // трасса: поверх, тоже по глубине
    hot_pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for &(_, sx, sy, col, blk) in &hot_pts {
        blit(&mut px, size, sx, sy, col, blk);
    }
    px
}

// ── Тесты ─────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use eframe::egui::Color32;

    use super::{build_atlas, build_atlas_3d, build_atlas_3d_trace, hsv, project3d, rotate_dir};
    use crate::core::color::ColorMode;
    use crate::io::data::{tiny_data, Proj};

    #[test]
    fn test_build_atlas_size() {
        let d = tiny_data();
        // Новый сигнатурный параметр — радиус гауссова сплата в текселях.
        let px = build_atlas(&d, Proj::Xy, 32, &[true; 7], -1, 0, ColorMode::Nt, 2.5);
        assert_eq!(px.len(), 32 * 32);
        assert_ne!(px[0], Color32::from_rgb(255, 255, 255));
    }

    #[test]
    fn test_build_atlas_gaussian_dot_and_density() {
        let d = tiny_data();
        let px = build_atlas(&d, Proj::Xy, 64, &[true; 7], -1, 0, ColorMode::Nt, 2.5);
        // Есть непустые (ярче фона) пиксели — атлас не пустой.
        let any_lit = px.iter().any(|&c| c.r() > 20 || c.g() > 20 || c.b() > 30);
        assert!(any_lit, "атлас должен содержать точки ярче фона");
        // Ни один пиксель не выжигается в чистый белый: тонмаппинг
        // плотности ограничивает альфу < 1 при разумных весах.
        assert!(px.iter().all(|&c| c != Color32::from_rgb(255, 255, 255)));
    }

    #[test]
    fn test_project3d_ortho_center() {
        let (p, depth) = project3d(0.5, 0.5, 0.5, 0.0, 0.0, 0.0);
        assert!((p.x - 0.5).abs() < 1e-5);
        assert!((p.y - 0.5).abs() < 1e-5);
        assert!(depth.abs() < 1e-5);
    }

    #[test]
    fn test_rotate_dir_identity() {
        let (d, depth) = rotate_dir(1.0, 0.0, 0.0, 0.0, 0.0);
        assert!((d.x - 1.0).abs() < 1e-5);
        assert!(d.y.abs() < 1e-5);
        assert!(depth.abs() < 1e-5);
    }

    #[test]
    fn test_build_atlas_3d_size() {
        let d = tiny_data();
        let px = build_atlas_3d(&d, 0.6, -0.3, 0.6, 32, &[true; 7], -1, 0, ColorMode::Nt);
        assert_eq!(px.len(), 32 * 32);
    }

    #[test]
    fn test_build_atlas_3d_trace_size() {
        let d = tiny_data();
        let ff = vec![0u32, 1, u32::MAX, 2]; // нейрон 2 не сработал
        let sc = vec![5u32, 3, 0, 1];
        let px = build_atlas_3d_trace(
            &d,
            0.6,
            -0.3,
            0.6,
            32,
            &[true; 7],
            -1,
            0,
            &ff,
            &sc,
            ColorMode::Nt,
        );
        assert_eq!(px.len(), 32 * 32);
        // hsv(200,1,1) = чистый синий (начало волны)
        let hot = hsv(200.0, 1.0, 1.0);
        assert_eq!(hot.b(), 255);
    }
}
