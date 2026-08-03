//! T3 trajectory recording + PST matrix computation.
//!
//! # Trajectory layout (flat Vec)
//!
//! `data` — единый `Vec<f32>` со stride-based доступом:
//!
//! ```text
//! Index                     | Content          | Size
//! 0..n_steps*n              | th1_pre          | n_steps × n
//! n_steps*n .. 2*n_steps*n  | th2_pre          | n_steps × n
//! 2*n_steps*n .. 3*n_steps*n| th3_pre          | n_steps × n
//! 3*n_steps*n .. 4*n_steps*n| r1_pre           | n_steps × n
//! 4*n_steps*n .. 5*n_steps*n| r2_pre           | n_steps × n
//! 5*n_steps*n .. 6*n_steps*n| r3_pre           | n_steps × n
//! 6*n_steps*n ..            | drive1..3        | n_steps × 3n
//! ```
//!
//! + 6 отдельных Vec для final state.

use crate::t3::backend::T3Backend;
use crate::t3::TriOscillator;

/// Offsets within flat Vec
pub const OFF_TH1: usize = 0;
pub fn off_th2(n_steps: usize, n: usize) -> usize {
    n_steps * n
}
pub fn off_th3(n_steps: usize, n: usize) -> usize {
    2 * n_steps * n
}
pub fn off_r1(n_steps: usize, n: usize) -> usize {
    3 * n_steps * n
}
pub fn off_r2(n_steps: usize, n: usize) -> usize {
    4 * n_steps * n
}
pub fn off_r3(n_steps: usize, n: usize) -> usize {
    5 * n_steps * n
}
pub fn off_drive1(n_steps: usize, n: usize) -> usize {
    6 * n_steps * n
}
pub fn off_drive2(n_steps: usize, n: usize) -> usize {
    7 * n_steps * n
}
pub fn off_drive3(n_steps: usize, n: usize) -> usize {
    8 * n_steps * n
}

/// Pre-allocated buffer for PST matrix (avoids per-step Vec allocations).
pub struct PstMatrixBuf {
    pub data: Vec<f32>,
    pub sin1: Vec<f32>,
    pub cos1: Vec<f32>,
    pub sin2: Vec<f32>,
    pub cos2: Vec<f32>,
    pub sin3: Vec<f32>,
    pub cos3: Vec<f32>,
    pub n_steps_alloc: usize,
    pub n_alloc: usize,
}

impl PstMatrixBuf {
    pub fn new(max_steps: usize, n: usize) -> Self {
        let pst_dim = crate::t3::pst_linear_dim(n);
        Self {
            data: Vec::with_capacity(max_steps * pst_dim),
            sin1: vec![0.0; n],
            cos1: vec![0.0; n],
            sin2: vec![0.0; n],
            cos2: vec![0.0; n],
            sin3: vec![0.0; n],
            cos3: vec![0.0; n],
            n_steps_alloc: max_steps,
            n_alloc: n,
        }
    }

    pub fn resize(&mut self, max_steps: usize, n: usize) {
        let pst_dim = crate::t3::pst_linear_dim(n);
        if n > self.n_alloc {
            self.sin1.resize(n, 0.0);
            self.cos1.resize(n, 0.0);
            self.sin2.resize(n, 0.0);
            self.cos2.resize(n, 0.0);
            self.sin3.resize(n, 0.0);
            self.cos3.resize(n, 0.0);
            self.n_alloc = n;
        }
        self.data.reserve(max_steps * pst_dim);
        self.n_steps_alloc = max_steps;
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }
}

/// Записанная траектория T3 для backward pass.
///
/// Единый flat Vec для минимальных копий и хорошей cache locality.
pub struct T3SpikeTrajectory {
    pub n_osc: usize,
    pub n_steps: usize,
    pub dt: f32,
    pub drive_gain: f32,
    pub alpha: f32,
    pub epsilon: f32,
    pub freq_min: f32,
    pub freq_max: f32,
    pub w12: f32,
    pub w13: f32,
    pub w23: f32,

    /// Flat storage: [th1_pre, th2_pre, th3_pre, r1_pre, r2_pre, r3_pre, drive1, drive2, drive3]
    /// Каждый block — n_steps × n_osc.
    pub flat: Vec<f32>,

    // Final state (последний POST-update state)
    pub th1_final: Vec<f32>,
    pub th2_final: Vec<f32>,
    pub th3_final: Vec<f32>,
    pub r1_final: Vec<f32>,
    pub r2_final: Vec<f32>,
    pub r3_final: Vec<f32>,
}

impl T3SpikeTrajectory {
    /// Accessor helpers for backward pass
    pub fn th1_pre(&self) -> &[f32] {
        &self.flat[OFF_TH1..off_th2(self.n_steps, self.n_osc)]
    }
    pub fn th2_pre(&self) -> &[f32] {
        &self.flat[off_th2(self.n_steps, self.n_osc)..off_th3(self.n_steps, self.n_osc)]
    }
    pub fn th3_pre(&self) -> &[f32] {
        &self.flat[off_th3(self.n_steps, self.n_osc)..off_r1(self.n_steps, self.n_osc)]
    }
    pub fn r1_pre(&self) -> &[f32] {
        &self.flat[off_r1(self.n_steps, self.n_osc)..off_r2(self.n_steps, self.n_osc)]
    }
    pub fn r2_pre(&self) -> &[f32] {
        &self.flat[off_r2(self.n_steps, self.n_osc)..off_r3(self.n_steps, self.n_osc)]
    }
    pub fn r3_pre(&self) -> &[f32] {
        &self.flat[off_r3(self.n_steps, self.n_osc)..off_drive1(self.n_steps, self.n_osc)]
    }
    pub fn drive1(&self) -> &[f32] {
        &self.flat[off_drive1(self.n_steps, self.n_osc)..off_drive2(self.n_steps, self.n_osc)]
    }
    pub fn drive2(&self) -> &[f32] {
        &self.flat[off_drive2(self.n_steps, self.n_osc)..off_drive3(self.n_steps, self.n_osc)]
    }
    pub fn drive3(&self) -> &[f32] {
        &self.flat[off_drive3(self.n_steps, self.n_osc)..]
    }

    /// Mutable accessors (for forward pass to fill trajectory)
    pub fn th1_pre_mut(&mut self) -> &mut [f32] {
        let end = off_th2(self.n_steps, self.n_osc);
        &mut self.flat[OFF_TH1..end]
    }
    pub fn th2_pre_mut(&mut self) -> &mut [f32] {
        let (start, end) = (
            off_th2(self.n_steps, self.n_osc),
            off_th3(self.n_steps, self.n_osc),
        );
        &mut self.flat[start..end]
    }
    pub fn th3_pre_mut(&mut self) -> &mut [f32] {
        let (start, end) = (
            off_th3(self.n_steps, self.n_osc),
            off_r1(self.n_steps, self.n_osc),
        );
        &mut self.flat[start..end]
    }
    pub fn r1_pre_mut(&mut self) -> &mut [f32] {
        let (start, end) = (
            off_r1(self.n_steps, self.n_osc),
            off_r2(self.n_steps, self.n_osc),
        );
        &mut self.flat[start..end]
    }
    pub fn r2_pre_mut(&mut self) -> &mut [f32] {
        let (start, end) = (
            off_r2(self.n_steps, self.n_osc),
            off_r3(self.n_steps, self.n_osc),
        );
        &mut self.flat[start..end]
    }
    pub fn r3_pre_mut(&mut self) -> &mut [f32] {
        let (start, end) = (
            off_r3(self.n_steps, self.n_osc),
            off_drive1(self.n_steps, self.n_osc),
        );
        &mut self.flat[start..end]
    }
    pub fn drive1_mut(&mut self) -> &mut [f32] {
        let (start, end) = (
            off_drive1(self.n_steps, self.n_osc),
            off_drive2(self.n_steps, self.n_osc),
        );
        &mut self.flat[start..end]
    }
    pub fn drive2_mut(&mut self) -> &mut [f32] {
        let (start, end) = (
            off_drive2(self.n_steps, self.n_osc),
            off_drive3(self.n_steps, self.n_osc),
        );
        &mut self.flat[start..end]
    }
    pub fn drive3_mut(&mut self) -> &mut [f32] {
        let start = off_drive3(self.n_steps, self.n_osc);
        &mut self.flat[start..]
    }
}

impl T3SpikeTrajectory {
    /// Create a new trajectory with flat storage pre-allocated.
    pub fn new(
        n_osc: usize,
        n_steps: usize,
        dt: f32,
        drive_gain: f32,
        alpha: f32,
        epsilon: f32,
        freq_min: f32,
        freq_max: f32,
        w12: f32,
        w13: f32,
        w23: f32,
    ) -> Self {
        let flat_len = 9 * n_steps * n_osc;
        Self {
            n_osc,
            n_steps,
            dt,
            drive_gain,
            alpha,
            epsilon,
            freq_min,
            freq_max,
            w12,
            w13,
            w23,
            flat: vec![0.0; flat_len],
            th1_final: vec![0.0; n_osc],
            th2_final: vec![0.0; n_osc],
            th3_final: vec![0.0; n_osc],
            r1_final: vec![0.0; n_osc],
            r2_final: vec![0.0; n_osc],
            r3_final: vec![0.0; n_osc],
        }
    }
}

pub fn copy_slice(dst: &mut [f32], src: &[f32]) {
    assert_eq!(dst.len(), src.len(), "copy_slice: length mismatch");
    dst.copy_from_slice(src);
}

/// Compute full PST feature matrix from trajectory in one pass.
/// Uses pre-allocated `PstMatrixBuf` for zero per-step allocations.
pub fn t3spike_compute_pst_matrix_into<'a>(
    traj: &T3SpikeTrajectory,
    buf: &'a mut PstMatrixBuf,
) -> &'a [f32] {
    let n = traj.n_osc;
    let n_steps = traj.n_steps;
    let _pst_dim = crate::t3::pst_linear_dim(n);
    let nf = n as f32;

    buf.clear();
    buf.resize(n_steps, n);

    let sin1 = &mut buf.sin1[..n];
    let cos1 = &mut buf.cos1[..n];
    let sin2 = &mut buf.sin2[..n];
    let cos2 = &mut buf.cos2[..n];
    let sin3 = &mut buf.sin3[..n];
    let cos3 = &mut buf.cos3[..n];

    for step in 0..n_steps {
        let (th1, r1, th2, r2, th3, r3): (&[f32], &[f32], &[f32], &[f32], &[f32], &[f32]);
        if step == n_steps - 1 {
            th1 = &traj.th1_final;
            r1 = &traj.r1_final;
            th2 = &traj.th2_final;
            r2 = &traj.r2_final;
            th3 = &traj.th3_final;
            r3 = &traj.r3_final;
        } else {
            let s = |off: usize, step: usize| &traj.flat[off + step * n..off + (step + 1) * n];
            th1 = s(OFF_TH1, step + 1);
            r1 = s(off_r1(n_steps, n), step + 1);
            th2 = s(off_th2(n_steps, n), step + 1);
            r2 = s(off_r2(n_steps, n), step + 1);
            th3 = s(off_th3(n_steps, n), step + 1);
            r3 = s(off_r3(n_steps, n), step + 1);
        }

        for i in 0..n {
            sin1[i] = th1[i].sin();
            cos1[i] = th1[i].cos();
            sin2[i] = th2[i].sin();
            cos2[i] = th2[i].cos();
            sin3[i] = th3[i].sin();
            cos3[i] = th3[i].cos();
        }

        let s = |off: usize, step: usize| &traj.flat[off + step * n..off + (step + 1) * n];
        let prev_th1 = s(OFF_TH1, step);
        let prev_th2 = s(off_th2(n_steps, n), step);
        let prev_th3 = s(off_th3(n_steps, n), step);

        for i in 0..n {
            buf.data.push(sin1[i]);
            buf.data.push(cos1[i]);
        }
        for i in 0..n {
            buf.data.push(sin2[i]);
            buf.data.push(cos2[i]);
        }
        for i in 0..n {
            buf.data.push(sin3[i]);
            buf.data.push(cos3[i]);
        }

        for (ph, prev) in [(th1, prev_th1), (th2, prev_th2), (th3, prev_th3)] {
            for i in 0..n {
                let mut d = ph[i] - prev[i];
                if d > std::f32::consts::PI {
                    d -= 2.0 * std::f32::consts::PI;
                } else if d < -std::f32::consts::PI {
                    d += 2.0 * std::f32::consts::PI;
                }
                buf.data.push(d);
            }
        }

        buf.data.extend_from_slice(r1);
        buf.data.extend_from_slice(r2);
        buf.data.extend_from_slice(r3);

        for i in 0..n {
            buf.data.push(sin1[i] * sin1[i]);
            buf.data.push(cos1[i] * cos1[i]);
        }
        for i in 0..n {
            buf.data.push(sin2[i] * sin2[i]);
            buf.data.push(cos2[i] * cos2[i]);
        }
        for i in 0..n {
            buf.data.push(sin3[i] * sin3[i]);
            buf.data.push(cos3[i] * cos3[i]);
        }

        let cross = |sa: &[f32], ca: &[f32], sb: &[f32], cb: &[f32]| -> f32 {
            let (ss, cc): (f64, f64) = (0..n).fold((0.0, 0.0), |(s, c), i| {
                (
                    s + (sa[i] as f64 * cb[i] as f64 - ca[i] as f64 * sb[i] as f64),
                    c + (ca[i] as f64 * cb[i] as f64 + sa[i] as f64 * sb[i] as f64),
                )
            });
            ((ss * ss + cc * cc).sqrt() / nf as f64) as f32
        };
        buf.data.push(cross(sin1, cos1, sin2, cos2));
        buf.data.push(cross(sin2, cos2, sin3, cos3));
        buf.data.push(cross(sin1, cos1, sin3, cos3));

        let mu = |arr: &[f32]| arr.iter().sum::<f32>() / nf;
        buf.data.push(mu(sin1));
        buf.data.push(mu(cos1));
        buf.data.push(mu(r1));
        buf.data.push(mu(sin2));
        buf.data.push(mu(cos2));
        buf.data.push(mu(r2));
        buf.data.push(mu(sin3));
        buf.data.push(mu(cos3));
        buf.data.push(mu(r3));
    }
    &buf.data
}

/// Allocating wrapper for PST matrix (uses internal pre-allocated buffer).
pub fn t3spike_compute_pst_matrix(traj: &T3SpikeTrajectory) -> Vec<f32> {
    let n = traj.n_osc;
    let n_steps = traj.n_steps;
    let pst_dim = crate::t3::pst_linear_dim(n);
    let mut out = Vec::with_capacity(n_steps * pst_dim);
    let nf = n as f32;

    for step in 0..n_steps {
        let (th1, r1, th2, r2, th3, r3): (&[f32], &[f32], &[f32], &[f32], &[f32], &[f32]);
        if step == n_steps - 1 {
            th1 = &traj.th1_final;
            r1 = &traj.r1_final;
            th2 = &traj.th2_final;
            r2 = &traj.r2_final;
            th3 = &traj.th3_final;
            r3 = &traj.r3_final;
        } else {
            let _next = (step + 1) * n;
            let s = |off: usize, step: usize| &traj.flat[off + step * n..off + (step + 1) * n];
            th1 = s(OFF_TH1, step + 1);
            r1 = s(off_r1(traj.n_steps, n), step + 1);
            th2 = s(off_th2(traj.n_steps, n), step + 1);
            r2 = s(off_r2(traj.n_steps, n), step + 1);
            th3 = s(off_th3(traj.n_steps, n), step + 1);
            r3 = s(off_r3(traj.n_steps, n), step + 1);
        }
        let s = |off: usize, step: usize| &traj.flat[off + step * n..off + (step + 1) * n];
        let prev_th1 = s(OFF_TH1, step);
        let prev_th2 = s(off_th2(traj.n_steps, n), step);
        let prev_th3 = s(off_th3(traj.n_steps, n), step);

        let mut sin1 = Vec::with_capacity(n);
        let mut cos1 = Vec::with_capacity(n);
        let mut sin2 = Vec::with_capacity(n);
        let mut cos2 = Vec::with_capacity(n);
        let mut sin3 = Vec::with_capacity(n);
        let mut cos3 = Vec::with_capacity(n);
        for i in 0..n {
            sin1.push(th1[i].sin());
            cos1.push(th1[i].cos());
            sin2.push(th2[i].sin());
            cos2.push(th2[i].cos());
            sin3.push(th3[i].sin());
            cos3.push(th3[i].cos());
        }

        for i in 0..n {
            out.push(sin1[i]);
            out.push(cos1[i]);
        }
        for i in 0..n {
            out.push(sin2[i]);
            out.push(cos2[i]);
        }
        for i in 0..n {
            out.push(sin3[i]);
            out.push(cos3[i]);
        }

        for (ph, prev) in [(th1, prev_th1), (th2, prev_th2), (th3, prev_th3)] {
            for i in 0..n {
                let mut d = ph[i] - prev[i];
                if d > std::f32::consts::PI {
                    d -= 2.0 * std::f32::consts::PI;
                } else if d < -std::f32::consts::PI {
                    d += 2.0 * std::f32::consts::PI;
                }
                out.push(d);
            }
        }

        out.extend_from_slice(r1);
        out.extend_from_slice(r2);
        out.extend_from_slice(r3);

        for i in 0..n {
            out.push(sin1[i] * sin1[i]);
            out.push(cos1[i] * cos1[i]);
        }
        for i in 0..n {
            out.push(sin2[i] * sin2[i]);
            out.push(cos2[i] * cos2[i]);
        }
        for i in 0..n {
            out.push(sin3[i] * sin3[i]);
            out.push(cos3[i] * cos3[i]);
        }

        let cross = |sa: &[f32], ca: &[f32], sb: &[f32], cb: &[f32]| {
            let (ss, cc): (f64, f64) = (0..n).fold((0.0, 0.0), |(s, c), i| {
                (
                    s + (sa[i] as f64 * cb[i] as f64 - ca[i] as f64 * sb[i] as f64),
                    c + (ca[i] as f64 * cb[i] as f64 + sa[i] as f64 * sb[i] as f64),
                )
            });
            ((ss * ss + cc * cc).sqrt() / nf as f64) as f32
        };
        out.push(cross(&sin1, &cos1, &sin2, &cos2));
        out.push(cross(&sin2, &cos2, &sin3, &cos3));
        out.push(cross(&sin1, &cos1, &sin3, &cos3));

        let mu = |arr: &[f32]| arr.iter().sum::<f32>() / nf;
        out.push(mu(&sin1));
        out.push(mu(&cos1));
        out.push(mu(r1));
        out.push(mu(&sin2));
        out.push(mu(&cos2));
        out.push(mu(r2));
        out.push(mu(&sin3));
        out.push(mu(&cos3));
        out.push(mu(r3));
    }
    out
}

// ═══════════════════════════════════════════════════════════
//  Gradient Checkpointing
// ═══════════════════════════════════════════════════════════

/// Configuration for gradient checkpointing.
///
/// Trade-off: память ÷ K, compute × (1 + 1/(K-1)).
pub struct CheckpointConfig {
    /// Сохранять каждый K-тый шаг (K=1 = full trajectory, K=∞ = recompute all)
    pub stride: usize,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self { stride: 10 }
    }
}

/// Checkpointed trajectory — хранит каждый K-тый pre-state + все drives.
///
/// Для backward: recompute участки между checkpoint-ами.
pub struct CheckpointedTrajectory {
    pub n_osc: usize,
    pub n_steps: usize,
    pub config: CheckpointConfig,

    /// Checkpoint-ы: pre-state на шагах 0, K, 2K, ...
    pub checkpoints: Vec<f32>,

    /// Все drives [n_steps × 3 × n_osc] — нужны для recompute
    pub drives: Vec<f32>,

    // Параметры для recompute
    pub dt: f32,
    pub drive_gain: f32,
    pub alpha: f32,
    pub epsilon: f32,
    pub freq_min: f32,
    pub freq_max: f32,
    pub w12: f32,
    pub w13: f32,
    pub w23: f32,

    // Final state (после последнего шага)
    pub th1_final: Vec<f32>,
    pub th2_final: Vec<f32>,
    pub th3_final: Vec<f32>,
    pub r1_final: Vec<f32>,
    pub r2_final: Vec<f32>,
    pub r3_final: Vec<f32>,
}

impl CheckpointedTrajectory {
    /// Forward pass с checkpointing.
    ///
    /// Сохраняет pre-state на каждый K-тый шаг и все drives.
    /// После forward финальный state доступен в th1_final..r3_final.
    ///
    /// `freq_min`/`freq_max` передаются явно (не восстанавливаются приблизительно из omega1).
    /// Если переданы как `None`, вычисляются приблизительно (как раньше, обратная совместимость).
    pub fn from_forward<B: T3Backend>(
        tri: &mut TriOscillator<B>,
        drives: &[f32],
        n_steps: usize,
        dt: f32,
        config: CheckpointConfig,
        freq_min_orig: Option<f32>,
        freq_max_orig: Option<f32>,
    ) -> Self {
        let n = tri.n;
        let stride = config.stride.max(1);

        // Checkpoints: каждый stride-тый шаг
        let n_checkpoints = n_steps / stride + 1;
        // На каждый checkpoint сохраняем [th1, th2, th3, r1, r2, r3] — 6n floats
        let mut checkpoints = Vec::with_capacity(n_checkpoints * 6 * n);

        // Сохраняем checkpoint 0 (pre-state первого шага)
        checkpoints.extend_from_slice(&tri.th1[..n]);
        checkpoints.extend_from_slice(&tri.th2[..n]);
        checkpoints.extend_from_slice(&tri.th3[..n]);
        checkpoints.extend_from_slice(&tri.r1[..n]);
        checkpoints.extend_from_slice(&tri.r2[..n]);
        checkpoints.extend_from_slice(&tri.r3[..n]);

        let drives_vec = drives.to_vec();

        for step in 0..n_steps {
            let d_off = step * 3 * n;
            tri.drive1[..n].copy_from_slice(&drives[d_off..d_off + n]);
            tri.drive2[..n].copy_from_slice(&drives[d_off + n..d_off + 2 * n]);
            tri.drive3[..n].copy_from_slice(&drives[d_off + 2 * n..d_off + 3 * n]);

            tri.step(dt);

            // Сохраняем checkpoint после шага (pre-state следующего шага)
            if (step + 1) % stride == 0 && step + 1 < n_steps {
                checkpoints.extend_from_slice(&tri.th1[..n]);
                checkpoints.extend_from_slice(&tri.th2[..n]);
                checkpoints.extend_from_slice(&tri.th3[..n]);
                checkpoints.extend_from_slice(&tri.r1[..n]);
                checkpoints.extend_from_slice(&tri.r2[..n]);
                checkpoints.extend_from_slice(&tri.r3[..n]);
            }
        }

        Self {
            n_osc: n,
            n_steps,
            config,
            checkpoints,
            drives: drives_vec,
            dt,
            drive_gain: tri.drive_gain,
            alpha: tri.alpha,
            epsilon: tri.epsilon,
            freq_min: freq_min_orig.unwrap_or_else(|| tri.omega1[0] - 0.5),
            freq_max: freq_max_orig.unwrap_or_else(|| tri.omega1[n - 1] + 0.5),
            w12: tri.w12,
            w13: tri.w13,
            w23: tri.w23,
            th1_final: tri.th1[..n].to_vec(),
            th2_final: tri.th2[..n].to_vec(),
            th3_final: tri.th3[..n].to_vec(),
            r1_final: tri.r1[..n].to_vec(),
            r2_final: tri.r2[..n].to_vec(),
            r3_final: tri.r3[..n].to_vec(),
        }
    }

    /// Recompute pre-state для steps в [from, to) и записать в flat-буфер.
    ///
    /// `tri`: TriOscillator (будет перезаписан, начиная с checkpoint-состояния)
    /// `flat_buf`: [n_steps × 6n] — сюда запишутся pre-state шагов [from, to)
    ///   th1_pre, th2_pre, th3_pre, r1_pre, r2_pre, r3_pre
    pub fn recompute_segment<B: T3Backend>(
        &self,
        tri: &mut TriOscillator<B>,
        from_step: usize,
        to_step: usize,
        flat_buf: &mut [f32],
    ) {
        let n = self.n_osc;
        let stride = self.config.stride;

        // Найти последний checkpoint ≤ from_step
        let ckpt_step = (from_step / stride) * stride;
        let ckpt_idx = (ckpt_step / stride) * 6 * n;

        // Восстановить состояние из checkpoint
        tri.th1[..n].copy_from_slice(&self.checkpoints[ckpt_idx..ckpt_idx + n]);
        tri.th2[..n].copy_from_slice(&self.checkpoints[ckpt_idx + n..ckpt_idx + 2 * n]);
        tri.th3[..n].copy_from_slice(&self.checkpoints[ckpt_idx + 2 * n..ckpt_idx + 3 * n]);
        tri.r1[..n].copy_from_slice(&self.checkpoints[ckpt_idx + 3 * n..ckpt_idx + 4 * n]);
        tri.r2[..n].copy_from_slice(&self.checkpoints[ckpt_idx + 4 * n..ckpt_idx + 5 * n]);
        tri.r3[..n].copy_from_slice(&self.checkpoints[ckpt_idx + 5 * n..ckpt_idx + 6 * n]);

        for step in ckpt_step..to_step {
            // Сохранить pre-state в flat_buf
            if step >= from_step {
                let off = (step - from_step) * 6 * n;
                flat_buf[off..off + n].copy_from_slice(&tri.th1[..n]);
                flat_buf[off + n..off + 2 * n].copy_from_slice(&tri.th2[..n]);
                flat_buf[off + 2 * n..off + 3 * n].copy_from_slice(&tri.th3[..n]);
                flat_buf[off + 3 * n..off + 4 * n].copy_from_slice(&tri.r1[..n]);
                flat_buf[off + 4 * n..off + 5 * n].copy_from_slice(&tri.r2[..n]);
                flat_buf[off + 5 * n..off + 6 * n].copy_from_slice(&tri.r3[..n]);
            }

            if step < self.n_steps {
                let d_off = step * 3 * n;
                tri.drive1[..n].copy_from_slice(&self.drives[d_off..d_off + n]);
                tri.drive2[..n].copy_from_slice(&self.drives[d_off + n..d_off + 2 * n]);
                tri.drive3[..n].copy_from_slice(&self.drives[d_off + 2 * n..d_off + 3 * n]);
                tri.step(self.dt);
            }
        }
    }
}

/// Fast PST feature matrix — sin(θ) + r per step (6N per step, no trig calls).
pub fn t3spike_compute_pst_fast_matrix(traj: &T3SpikeTrajectory) -> Vec<f32> {
    let n = traj.n_osc;
    let n_steps = traj.n_steps;
    let dim = crate::t3::pst_fast_dim(n);
    let mut out = Vec::with_capacity(n_steps * dim);

    for step in 0..n_steps {
        let (th1, r1, th2, r2, th3, r3): (&[f32], &[f32], &[f32], &[f32], &[f32], &[f32]);
        if step == n_steps - 1 {
            th1 = &traj.th1_final;
            r1 = &traj.r1_final;
            th2 = &traj.th2_final;
            r2 = &traj.r2_final;
            th3 = &traj.th3_final;
            r3 = &traj.r3_final;
        } else {
            let _next = (step + 1) * n;
            let s = |off: usize, step: usize| &traj.flat[off + step * n..off + (step + 1) * n];
            th1 = s(OFF_TH1, step + 1);
            r1 = s(off_r1(traj.n_steps, n), step + 1);
            th2 = s(off_th2(traj.n_steps, n), step + 1);
            r2 = s(off_r2(traj.n_steps, n), step + 1);
            th3 = s(off_th3(traj.n_steps, n), step + 1);
            r3 = s(off_r3(traj.n_steps, n), step + 1);
        }

        for i in 0..n {
            out.push(th1[i].sin());
        }
        for i in 0..n {
            out.push(th2[i].sin());
        }
        for i in 0..n {
            out.push(th3[i].sin());
        }
        out.extend_from_slice(r1);
        out.extend_from_slice(r2);
        out.extend_from_slice(r3);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::t3::backend::{Avx2Backend, ScalarBackend};

    #[test]
    fn test_backend_consistency_avx2_vs_scalar() {
        // Avx2Backend и ScalarBackend должны давать одинаковую траекторию
        let n = 8;
        let n_steps = 100;
        let dt = 0.05;

        let n_drives = n_steps * 3 * n;
        let mut drives = Vec::with_capacity(n_drives);
        for i in 0..n_drives {
            drives.push(((i as f32) * 0.7).sin() * 1.5);
        }

        let (_, traj_avx) = crate::spike::forward::t3spike_forward_with_backend(
            Avx2Backend,
            n,
            n_steps,
            dt,
            10.0,
            1.0,
            0.5,
            3.0,
            0.3,
            0.15,
            0.08,
            &drives,
        );
        let (_, traj_scalar) = crate::spike::forward::t3spike_forward_with_backend(
            ScalarBackend,
            n,
            n_steps,
            dt,
            10.0,
            1.0,
            0.5,
            3.0,
            0.3,
            0.15,
            0.08,
            &drives,
        );

        // Compare final state (should match closely)
        let max_diff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        // Minimax sin/cos ~1e-4 error → tolerable for 100 steps
        let tol = 5e-3;
        assert!(
            max_diff(&traj_avx.th1_final, &traj_scalar.th1_final) < tol,
            "th1_final mismatch"
        );
        assert!(
            max_diff(&traj_avx.r1_final, &traj_scalar.r1_final) < tol,
            "r1_final mismatch"
        );
        assert!(
            max_diff(&traj_avx.th2_final, &traj_scalar.th2_final) < tol,
            "th2_final mismatch"
        );
        assert!(
            max_diff(&traj_avx.r2_final, &traj_scalar.r2_final) < tol,
            "r2_final mismatch"
        );
        assert!(
            max_diff(&traj_avx.th3_final, &traj_scalar.th3_final) < tol,
            "th3_final mismatch"
        );
        assert!(
            max_diff(&traj_avx.r3_final, &traj_scalar.r3_final) < tol,
            "r3_final mismatch"
        );

        // Also check PST features match
        let pst_avx = crate::t3::pst_linear_features(&{
            let mut tri_avx = crate::t3::TriOscillator::new(n, 0.5, 3.0, Avx2Backend);
            tri_avx.drive_gain = 10.0;
            tri_avx.alpha = 1.0;
            tri_avx.set_uniform_coupling(0.3, 0.15, 0.08);
            for step in 0..n_steps {
                let d_off = step * 3 * n;
                tri_avx.drive1[..n].copy_from_slice(&drives[d_off..d_off + n]);
                tri_avx.drive2[..n].copy_from_slice(&drives[d_off + n..d_off + 2 * n]);
                tri_avx.drive3[..n].copy_from_slice(&drives[d_off + 2 * n..d_off + 3 * n]);
                tri_avx.step(dt);
            }
            tri_avx.prepare();
            tri_avx
        });

        let pst_scalar = crate::t3::pst_linear_features(&{
            let mut tri_scalar = crate::t3::TriOscillator::new(n, 0.5, 3.0, ScalarBackend);
            tri_scalar.drive_gain = 10.0;
            tri_scalar.alpha = 1.0;
            tri_scalar.set_uniform_coupling(0.3, 0.15, 0.08);
            for step in 0..n_steps {
                let d_off = step * 3 * n;
                tri_scalar.drive1[..n].copy_from_slice(&drives[d_off..d_off + n]);
                tri_scalar.drive2[..n].copy_from_slice(&drives[d_off + n..d_off + 2 * n]);
                tri_scalar.drive3[..n].copy_from_slice(&drives[d_off + 2 * n..d_off + 3 * n]);
                tri_scalar.step(dt);
            }
            tri_scalar.prepare();
            tri_scalar
        });

        assert!(
            max_diff(&pst_avx, &pst_scalar) < 1e-2,
            "PST features mismatch between Avx2 and Scalar backends"
        );
    }

    #[test]
    fn test_checkpoint_forward_roundtrip() {
        let n = 8;
        let n_steps = 100;
        let stride = 10;
        let dt = 0.05;

        let mut tri = TriOscillator::new(n, 0.5, 3.0, Avx2Backend);
        tri.drive_gain = 10.0;
        tri.alpha = 1.0;
        tri.set_uniform_coupling(0.3, 0.15, 0.08);

        // Create drives
        let n_drives = n_steps * 3 * n;
        let mut drives = Vec::with_capacity(n_drives);
        for i in 0..n_drives {
            drives.push(((i as f32) * 0.3).cos() * 1.5);
        }

        // Full forward (reference)
        let mut tri_ref = TriOscillator::new(n, 0.5, 3.0, Avx2Backend);
        tri_ref.drive_gain = 10.0;
        tri_ref.alpha = 1.0;
        tri_ref.set_uniform_coupling(0.3, 0.15, 0.08);
        for step in 0..n_steps {
            let d_off = step * 3 * n;
            tri_ref.drive1[..n].copy_from_slice(&drives[d_off..d_off + n]);
            tri_ref.drive2[..n].copy_from_slice(&drives[d_off + n..d_off + 2 * n]);
            tri_ref.drive3[..n].copy_from_slice(&drives[d_off + 2 * n..d_off + 3 * n]);
            tri_ref.step(dt);
        }

        // Checkpointed forward (pass freq_min/freq_max explicitly)
        let config = CheckpointConfig { stride };
        let ckpt = CheckpointedTrajectory::from_forward(
            &mut tri,
            &drives,
            n_steps,
            dt,
            config,
            Some(0.5),
            Some(3.0),
        );

        // Verify final state matches
        let max_diff = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(
            max_diff(&ckpt.th1_final, &tri_ref.th1) < 1e-5,
            "th1_final mismatch"
        );
        assert!(
            max_diff(&ckpt.r1_final, &tri_ref.r1) < 1e-5,
            "r1_final mismatch"
        );

        // Recompute segment and compare with reference
        let from = 30;
        let to = 50;
        let mut flat_buf = vec![0.0f32; (to - from) * 6 * n];
        let mut tri_redo = TriOscillator::new(n, 0.5, 3.0, Avx2Backend);
        tri_redo.drive_gain = 10.0;
        tri_redo.alpha = 1.0;
        tri_redo.set_uniform_coupling(0.3, 0.15, 0.08);
        ckpt.recompute_segment(&mut tri_redo, from, to, &mut flat_buf);

        // Compare recomputed pre-states with reference trajectory
        let mut flat_ref = vec![0.0f32; (to - from) * 6 * n];
        let mut tri_ref2 = TriOscillator::new(n, 0.5, 3.0, Avx2Backend);
        tri_ref2.drive_gain = 10.0;
        tri_ref2.alpha = 1.0;
        tri_ref2.set_uniform_coupling(0.3, 0.15, 0.08);
        for step in 0..to {
            if step >= from {
                let off = (step - from) * 6 * n;
                flat_ref[off..off + n].copy_from_slice(&tri_ref2.th1[..n]);
                flat_ref[off + n..off + 2 * n].copy_from_slice(&tri_ref2.th2[..n]);
                flat_ref[off + 2 * n..off + 3 * n].copy_from_slice(&tri_ref2.th3[..n]);
                flat_ref[off + 3 * n..off + 4 * n].copy_from_slice(&tri_ref2.r1[..n]);
                flat_ref[off + 4 * n..off + 5 * n].copy_from_slice(&tri_ref2.r2[..n]);
                flat_ref[off + 5 * n..off + 6 * n].copy_from_slice(&tri_ref2.r3[..n]);
            }
            if step < n_steps {
                let d_off = step * 3 * n;
                tri_ref2.drive1[..n].copy_from_slice(&drives[d_off..d_off + n]);
                tri_ref2.drive2[..n].copy_from_slice(&drives[d_off + n..d_off + 2 * n]);
                tri_ref2.drive3[..n].copy_from_slice(&drives[d_off + 2 * n..d_off + 3 * n]);
                tri_ref2.step(dt);
            }
        }

        let buf_diff = max_diff(&flat_buf, &flat_ref);
        assert!(
            buf_diff < 1e-5,
            "recomputed segment mismatch: {:.2e}",
            buf_diff
        );
    }
}
