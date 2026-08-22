//! Метрики расхождения симуляция ↔ эталон.

/// Результат сравнения двух трасс V(t).
#[derive(Debug, Clone, Copy)]
pub struct TraceMetrics {
    /// Максимальная |ΔV| (мВ).
    pub max_abs_error: f32,
    /// Среднеквадратичная ошибка (мВ).
    pub rmse: f32,
}

/// Посчитать метрики двух трасс равной длины.
pub fn compare(sim: &[f32], reference: &[f32]) -> TraceMetrics {
    debug_assert_eq!(sim.len(), reference.len(), "compare: длины должны совпадать");
    let mut max_abs = 0.0f32;
    let mut sq_sum = 0.0f64;
    for (&s, &r) in sim.iter().zip(reference.iter()) {
        let d = (s - r).abs();
        max_abs = max_abs.max(d);
        sq_sum += (d as f64) * (d as f64);
    }
    TraceMetrics {
        max_abs_error: max_abs,
        rmse: (sq_sum / sim.len().max(1) as f64).sqrt() as f32,
    }
}

/// Порог пересечения (мВ) для подсчёта спайков — классический −20 мВ AdEx.
pub const SPIKE_THRESHOLD_MV: f32 = -20.0;

/// Посчитать спайки как восходящие пересечения порога.
pub fn count_spikes(v_mv: &[f32], threshold: f32) -> usize {
    let mut below = v_mv.first().map(|v| *v <= threshold).unwrap_or(true);
    let mut count = 0;
    for &v in v_mv {
        if v > threshold && below {
            count += 1;
        }
        below = v <= threshold;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_identical_traces() {
        let v = vec![-65.0, -60.0, -10.0, -65.0];
        let m = compare(&v, &v);
        assert_eq!(m.max_abs_error, 0.0);
        assert_eq!(m.rmse, 0.0);
    }

    #[test]
    fn metrics_constant_offset() {
        let sim = vec![-65.0f32, -64.0, -63.0];
        let r = vec![-64.0f32, -63.0, -62.0];
        let m = compare(&sim, &r);
        assert_eq!(m.max_abs_error, 1.0);
        assert!((m.rmse - 1.0).abs() < 1e-6);
    }

    #[test]
    fn spikes_count_upward_crossings() {
        let v = vec![-65.0, -19.0, -65.0, -19.0, -19.0, -65.0];
        assert_eq!(count_spikes(&v, SPIKE_THRESHOLD_MV), 2);
    }
}
