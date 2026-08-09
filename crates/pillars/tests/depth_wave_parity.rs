//! # DepthGCO parity — pillars `kernels::depth_wave::GcoState` vs inline legacy
//!
//! Волна 8, транш 1f: `neuron/depth_wave.rs` (264 строки) удалён, канон —
//! `pillars::kernels::depth_wave::GcoState`. Путь `kernels::neuron::depth_wave`
//! — compat-реэкспорт pillars.
//!
//! Методология (wc_parity/dendrite_parity/hh_parity-паттерн): inline-эталон
//! извлечён из git HEAD удалённого файла. Единственное документированное
//! расхождение: легаси использовал `rayon::par_iter().sum()` для глобальной
//! L2-нормы — порядок редукции недетерминирован. Эталон здесь использует
//! детерминированный последовательный цикл (тот же порядок индексов), что и
//! pillars-канон — при rayon-пуле в 1 поток легаси давал тот же бит-результат.
//!
//! Поскольку `compute_h_psi`/`evolve` пишут каждый элемент независимо,
//! паралеллизм не влияет на v/u/h_real/h_imag по-элементно — только на norm_sq.
//! Для единообразия эталон повторяет последовательную свёртку канона.

use pillars::kernels::depth_wave::GcoState;

// ═══════════════════════════════════════════════════════════
//  Inline legacy reference — verbatim из git HEAD
//  crates/kernels/src/neuron/depth_wave.rs (удалён в транше 1f),
//  rayon-редукции заменены детерминированными циклами.
// ═══════════════════════════════════════════════════════════
mod legacy {
    #[derive(Debug, Clone)]
    pub struct GcoState {
        pub v: Vec<f32>,
        pub u: Vec<f32>,
        pub depth: Vec<f32>,
        pub speed: Vec<f32>,
        pub h_real: Vec<f32>,
        pub h_imag: Vec<f32>,
        pub n: usize,
        pub dt: f32,
        #[allow(dead_code)]
        pub mod_range: f32,
    }

    impl GcoState {
        pub fn new(n: usize, depth: Vec<f32>, mod_range: f32, dt: f32, seed: u64) -> Self {
            assert_eq!(depth.len(), n);

            let mut v = vec![0.0f32; n];
            let mut u = vec![0.0f32; n];
            let mut rng = seed;
            for i in 0..n {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let phase = (rng as f32) / (u64::MAX as f32) * std::f32::consts::TAU;
                v[i] = phase.cos();
                u[i] = phase.sin();
            }

            let speed: Vec<f32> = depth.iter().map(|&d| 1.0 + d * mod_range).collect();

            Self {
                v,
                u,
                depth,
                speed,
                h_real: vec![0.0; n],
                h_imag: vec![0.0; n],
                n,
                dt,
                mod_range,
            }
        }

        pub fn compute_h_psi(
            &mut self,
            row_ptr: &[u32],
            col_idx: &[u32],
            vals: &[f32],
            nt_signs: &[f32],
        ) {
            let u = &self.u;
            let v = &self.v;

            for i in 0..self.n {
                let s = row_ptr[i] as usize;
                let e = row_ptr[i + 1] as usize;
                let sign_i = nt_signs[i];
                let mut sum_u = 0.0f32;
                let mut sum_v = 0.0f32;
                for idx in s..e {
                    let j = col_idx[idx] as usize;
                    let w = vals[idx] * sign_i;
                    sum_u += w * u[j];
                    sum_v += w * v[j];
                }
                self.h_real[i] = sum_u;
                self.h_imag[i] = -sum_v;
            }
        }

        pub fn evolve(&mut self, i_ext: Option<&[f32]>) {
            let dt = self.dt;
            for i in 0..self.n {
                let sp = self.speed[i];
                self.v[i] += dt * sp * self.h_real[i];
                self.u[i] += dt * sp * self.h_imag[i];

                if let Some(ext) = i_ext {
                    if i < ext.len() {
                        let sensory_w = 1.0 - self.depth[i];
                        self.v[i] += dt * sensory_w * ext[i];
                    }
                }
            }

            let norm_sq: f32 = self
                .v
                .iter()
                .zip(self.u.iter())
                .map(|(v, u)| v * v + u * u)
                .sum();
            let inv_norm = 1.0 / (norm_sq + 1e-9).sqrt();
            for i in 0..self.n {
                self.v[i] *= inv_norm;
                self.u[i] *= inv_norm;
            }
        }

        pub fn step(
            &mut self,
            row_ptr: &[u32],
            col_idx: &[u32],
            vals: &[f32],
            nt_signs: &[f32],
            i_ext: Option<&[f32]>,
        ) {
            self.compute_h_psi(row_ptr, col_idx, vals, nt_signs);
            self.evolve(i_ext);
        }

        pub fn coherence(&self) -> f32 {
            let n = self.n as f32;
            let mean_re: f32 = self.v.iter().sum::<f32>() / n;
            let mean_im: f32 = self.u.iter().sum::<f32>() / n;
            (mean_re * mean_re + mean_im * mean_im).sqrt()
        }

        pub fn motor_amplitudes(&self) -> Vec<f32> {
            (0..self.n)
                .filter(|&i| self.depth[i] < 0.3)
                .map(|i| (self.v[i] * self.v[i] + self.u[i] * self.u[i]).sqrt())
                .collect()
        }

        pub fn reset(&mut self, seed: u64) {
            let mut rng = seed;
            for i in 0..self.n {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let phase = (rng as f32) / (u64::MAX as f32) * std::f32::consts::TAU;
                self.v[i] = phase.cos();
                self.u[i] = phase.sin();
            }
        }
    }
}

fn assert_bits_eq(a: f32, b: f32, ctx: &str) {
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "{ctx}: pillars {a} (0x{:08x}) != legacy {b} (0x{:08x})",
        a.to_bits(),
        b.to_bits()
    );
}

/// CSR-кольцо из легаси-теста + вариант с внешним током.
fn ring_graph(n: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>) {
    let row_ptr: Vec<u32> = (0..=n as u32).collect();
    let col_idx: Vec<u32> = (1..n as u32).chain(std::iter::once(0)).collect();
    let vals = vec![1.0f32; n];
    let nt_signs: Vec<f32> = (0..n)
        .map(|i| if i % 3 == 2 { -1.0 } else { 1.0 })
        .collect();
    (row_ptr, col_idx, vals, nt_signs)
}

fn run_both(
    n: usize,
    depth: Vec<f32>,
    mod_range: f32,
    dt: f32,
    seed: u64,
    steps: usize,
    use_ext: bool,
) {
    let (row_ptr, col_idx, vals, nt_signs) = ring_graph(n);

    let mut p = GcoState::new(n, depth.clone(), mod_range, dt, seed);
    let mut l = legacy::GcoState::new(n, depth, mod_range, dt, seed);

    let ext: Vec<f32> = if use_ext {
        (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect()
    } else {
        Vec::new()
    };

    for _ in 0..steps {
        let ext_ref = if use_ext { Some(ext.as_slice()) } else { None };
        p.step(&row_ptr, &col_idx, &vals, &nt_signs, ext_ref);
        l.step(&row_ptr, &col_idx, &vals, &nt_signs, ext_ref);

        // По-элементная сверка после каждого шага — ловит первый же дрейф
        for i in 0..n {
            assert_bits_eq(p.v[i], l.v[i], &format!("v[{i}] step n={n}"));
            assert_bits_eq(p.u[i], l.u[i], &format!("u[{i}] step n={n}"));
            assert_bits_eq(p.h_real[i], l.h_real[i], &format!("h_real[{i}] n={n}"));
            assert_bits_eq(p.h_imag[i], l.h_imag[i], &format!("h_imag[{i}] n={n}"));
        }
    }

    assert_bits_eq(p.coherence(), l.coherence(), "coherence");
    assert_eq!(
        p.motor_amplitudes(),
        l.motor_amplitudes(),
        "motor_amplitudes"
    );
}

#[test]
fn gco_parity_ring_n4() {
    run_both(4, vec![0.9, 0.5, 0.5, 0.1], 2.0, 0.5, 42, 100, false);
}

#[test]
fn gco_parity_ring_n8_with_ext() {
    run_both(8, vec![0.5; 8], 1.5, 0.25, 7, 200, true);
}

#[test]
fn gco_parity_n16_no_modulation() {
    // mod_range = 0 → plain GCO
    let depth: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
    run_both(16, depth, 0.0, 0.5, 99, 50, false);
}

#[test]
fn gco_parity_n3_motor_amplitudes() {
    run_both(3, vec![0.9, 0.2, 0.05], 2.0, 0.5, 42, 10, false);
}

#[test]
fn gco_parity_reset_and_new() {
    // new() и reset() дают одинаковые начальные фазы (тот же LCG-хэш)
    let depth = vec![0.9, 0.5, 0.1];
    let mut p = GcoState::new(3, depth.clone(), 2.0, 0.5, 42);
    let mut l = legacy::GcoState::new(3, depth, 2.0, 0.5, 42);
    for i in 0..3 {
        assert_bits_eq(p.v[i], l.v[i], &format!("new v[{i}]"));
        assert_bits_eq(p.u[i], l.u[i], &format!("new u[{i}]"));
    }
    p.reset(123);
    l.reset(123);
    for i in 0..3 {
        assert_bits_eq(p.v[i], l.v[i], &format!("reset v[{i}]"));
        assert_bits_eq(p.u[i], l.u[i], &format!("reset u[{i}]"));
    }
}

#[test]
fn gco_parity_speed_layout() {
    // speed[i] = 1 + depth[i]·mod_range — бит-в-бит
    let depth = vec![1.0, 0.0, 0.5, 0.75];
    let p = GcoState::new(4, depth.clone(), 2.5, 0.1, 1);
    let l = legacy::GcoState::new(4, depth, 2.5, 0.1, 1);
    for i in 0..4 {
        assert_bits_eq(p.speed[i], l.speed[i], &format!("speed[{i}]"));
        assert_bits_eq(p.depth[i], l.depth[i], &format!("depth[{i}]"));
    }
}
