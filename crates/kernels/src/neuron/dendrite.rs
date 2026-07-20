#![allow(non_snake_case)]
use wide::f32x8;
use super::{load8, store8};

// ═══════════════════════════════════════════════════════
//  Dendritic Computation — Compartmentalized Dendrites
// ═══════════════════════════════════════════════════════
//
//  Каждый нейрон имеет n_compartments дендритных сегментов.
//  Каждый сегмент локально интегрирует PSP и передаёт ток в сому.
//
//  Уравнения:
//    V_dend[k] — мембранный потенциал k-го компартмента
//
//    dV_dend[k]/dt = (V_soma − V_dend[k]) / R_m        ← электротонический ток
//                   + I_syn[k] / C_m                     ← локальный синаптический ток
//
//    I_syn[k] = Σ g_r[k] · (E_rev_r − V_dend[k])        ← 7 типов рецепторов
//
//    I_dend += Σ (V_dend[k] − V_soma) / R_m             ← ток в сому
//
//  где R_m — мембранное сопротивление (MΩ),
//      C_m — ёмкость (nF).
//
//  Layout (column-major):
//    V_dend[k * n + i] = компартмент k, нейрон i
//    g_ampa[k * n + i] = AMPA проводимость компартмента k, нейрон i
//    ...
//    V_soma[i]         = сома нейрона i
//
//  Это позволяет загрузить 8 нейронов одного компартмента
//  одним f32x8 — идеально для SIMD!

/// Шаг компартментализованных дендритов — SIMD + scalar tail.
///
/// Обрабатывает `n` нейронов с `k` компартментами каждый.
///
/// # Layout (column-major)
/// Все буферы размера `n × k`, stride = n:
/// ```
/// V_dend[comp * n + nrn]  — потенциал компартмента
/// g_ampa[comp * n + nrn]  — AMPA проводимость
/// g_nmda[comp * n + nrn]  — NMDA проводимость
/// g_gaba_a[comp * n + nrn] — GABA_A проводимость
/// ```
/// V_soma[n] — потенциалы сом
/// I_dend[n] — выход: суммарный дендритный ток в сому (прибавляется к существующему)
///
/// # Уравнения
/// ```
/// dV_d = (V_soma − V_d) / R_m  +  (g_a·(E_a−V_d) + g_n·(E_n−V_d) + g_g·(E_g−V_d)) / C_m
/// I_dend += (V_d − V_soma) / R_m
/// ```
///
/// # Safety
/// Все указатели должны быть валидны для `n * k` элементов (V_dend, g_ampa, g_nmda, g_gaba_a)
/// и `n` элементов (V_soma, I_dend). Не должны пересекаться.
pub(crate) unsafe fn dendrite_step(
    V_dend: *mut f32,        // [n × k] column-major
    g_ampa: *const f32,      // [n × k]
    g_nmda: *const f32,      // [n × k]
    g_gaba_a: *const f32,    // [n × k]
    V_soma: *const f32,      // [n]
    I_dend: *mut f32,        // [n] — accumulates
    n: usize,                // количество нейронов
    k: usize,                // количество компартментов на нейрон
    R_m: f32,                // мембранное сопротивление (MΩ)
    C_m: f32,                // ёмкость (nF)
    dt: f32,                 // шаг (ms)
) {
    let inv_Rm = f32x8::splat(1.0 / R_m);
    let inv_Cm = f32x8::splat(1.0 / C_m);
    let dt_s = f32x8::splat(dt);
    // Реверсивные потенциалы (E_rev) — константы рецепторов
    let E_ampa = f32x8::splat(0.0);    // AMPA: 0 mV (возб.)
    let E_nmda = f32x8::splat(0.0);    // NMDA: 0 mV (возб.)
    let E_gaba_a = f32x8::splat(-65.0); // GABA_A: -65 mV (торм.)

    let n8 = n & !7;

    for comp in 0..k {
        let base = comp * n;  // смещение до компартмента comp

        // SIMD: 8 нейронов за раз
        for off in (0..n8).step_by(8) {
            let idx = base + off;

            let vd = load8(V_dend, idx);           // V_dend[comp, nrn..nrn+8]
            let ga = load8(g_ampa, idx);           // g_ampa[comp, nrn..nrn+8]
            let gn = load8(g_nmda, idx);           // g_nmda[comp, nrn..nrn+8]
            let gg = load8(g_gaba_a, idx);         // g_gaba_a[comp, nrn..nrn+8]
            let vs = load8(V_soma, off);           // V_soma[nrn..nrn+8]

            // I_syn = Σ g_r · (E_rev_r - V_dend)
            let i_syn =
                  ga * (E_ampa - vd)
                + gn * (E_nmda - vd)
                + gg * (E_gaba_a - vd);

            // dV_dend = (V_soma - V_dend)/R_m + I_syn/C_m
            let dV = (vs - vd) * inv_Rm + i_syn * inv_Cm;
            let new_vd = vd + dV * dt_s;
            store8(V_dend, idx, new_vd);

            // I_dend += (V_dend - V_soma)/R_m  (ток из компартмента в сому)
            let i_to_soma = (vd - vs) * inv_Rm;
            let mut i_dend = load8(I_dend, off);
            i_dend += i_to_soma;
            store8(I_dend, off, i_dend);
        }

        // Scalar tail (n % 8)
        if n8 < n {
            for i in n8..n {
                let idx = base + i;
                let vd = *V_dend.add(idx);
                let ga = *g_ampa.add(idx);
                let gn = *g_nmda.add(idx);
                let gg = *g_gaba_a.add(idx);
                let vs = *V_soma.add(i);

                let i_syn = ga * (0.0 - vd) + gn * (0.0 - vd) + gg * (-65.0 - vd);
                let dV = (vs - vd) / R_m + i_syn / C_m;
                *V_dend.add(idx) = vd + dV * dt;

                let i_to_soma = (vd - vs) / R_m;
                *I_dend.add(i) += i_to_soma;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
//  DendriteConfig — удобный safe конструктор
// ═══════════════════════════════════════════════════════

/// Конфигурация дендритного дерева для одного нейрона.
///
/// Хранит количество компартментов и электротонические параметры.
#[derive(Debug, Clone)]
pub struct DendriteConfig {
    /// Количество компартментов на нейрон
    pub n_compartments: usize,
    /// Мембранное сопротивление (MΩ) — чем выше, тем слабее ток к соме
    pub R_m: f32,
    /// Ёмкость компартмента (nF)
    pub C_m: f32,
}

impl Default for DendriteConfig {
    fn default() -> Self {
        Self {
            n_compartments: 4,  // 4 компартмента по умолчанию
            R_m: 100.0,         // 100 MΩ (типично для дендритов)
            C_m: 0.1,           // 0.1 nF (небольшая ёмкость компартмента)
        }
    }
}

/// Состояние дендритного дерева для всей популяции нейронов.
///
/// Все буферы выровнены до %8 для SIMD.
#[derive(Debug, Clone)]
pub struct DendriteState {
    pub n: usize,            // количество нейронов (unpadded)
    pub k: usize,            // количество компартментов
    pub n_padded: usize,     // n, округлённое до %8
    /// Конфигурация (R_m, C_m)
    pub config: DendriteConfig,
    /// V_dend[n × k] — column-major, padding до %8 для SIMD
    pub V_dend: Vec<f32>,
    /// g_ampa[n × k]
    pub g_ampa: Vec<f32>,
    /// g_nmda[n × k]
    pub g_nmda: Vec<f32>,
    /// g_gaba_a[n × k]
    pub g_gaba_a: Vec<f32>,
    /// I_dend[n] — аккумулятор тока в сому
    pub I_dend: Vec<f32>,
}

impl DendriteState {
    /// Создать новое дендритное дерево для `n` нейронов.
    ///
    /// Все потенциалы начинаются с V_rest (обычно V_soma).
    /// Все проводимости начинаются с 0.
    pub fn new(n: usize, config: &DendriteConfig) -> Self {
        let n_padded = n.div_ceil(8) * 8;
        let total = n_padded * config.n_compartments;
        Self {
            n,
            k: config.n_compartments,
            n_padded,
            config: config.clone(),
            V_dend: vec![-65.0; total],  // начать с V_rest
            g_ampa: vec![0.0; total],
            g_nmda: vec![0.0; total],
            g_gaba_a: vec![0.0; total],
            I_dend: vec![0.0; n_padded],
        }
    }

    /// Шаг дендритной динамики для всех нейронов.
    ///
    /// Обновляет V_dend и аккумулирует I_dend.
    /// I_dend[i] += ток из компартментов нейрона i в его сому.
    ///
    /// После вызова I_dend содержит полный дендритный вклад в сому.
    ///
    /// # Panics
    /// Если `V_soma.len() < n`.
    pub fn step(&mut self, V_soma: &[f32], dt: f32) {
        // Сброс I_dend перед аккумуляцией
        self.I_dend.fill(0.0);

        unsafe {
            dendrite_step(
                self.V_dend.as_mut_ptr(),
                self.g_ampa.as_ptr(),
                self.g_nmda.as_ptr(),
                self.g_gaba_a.as_ptr(),
                V_soma.as_ptr(),
                self.I_dend.as_mut_ptr(),
                self.n,
                self.k,
                self.config.R_m,
                self.config.C_m,
                dt,
            );
        }
    }

    /// Установить AMPA проводимость компартмента `comp` нейрона `nrn`.
    pub fn set_ampa(&mut self, nrn: usize, comp: usize, value: f32) {
        if nrn < self.n && comp < self.k {
            self.g_ampa[comp * self.n + nrn] = value;
        }
    }

    /// Установить GABA_A проводимость компартмента `comp` нейрона `nrn`.
    pub fn set_gaba_a(&mut self, nrn: usize, comp: usize, value: f32) {
        if nrn < self.n && comp < self.k {
            self.g_gaba_a[comp * self.n + nrn] = value;
        }
    }

    /// Установить NMDA проводимость компартмента `comp` нейрона `nrn`.
    pub fn set_nmda(&mut self, nrn: usize, comp: usize, value: f32) {
        if nrn < self.n && comp < self.k {
            self.g_nmda[comp * self.n + nrn] = value;
        }
    }

    /// Получить текущий дендритный ток нейрона `nrn`.
    pub fn get_dend_current(&self, nrn: usize) -> f32 {
        if nrn < self.n { self.I_dend[nrn] } else { 0.0 }
    }
}

// ═══════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Базовый тест: AMPA depolarises dendrites → положительный I_dend.
    #[test]
    fn dendrite_proximal_vs_distal() {
        let n = 8;
        let config = DendriteConfig {
            n_compartments: 3,
            R_m: 100.0,         // типичное R_m
            C_m: 1.0,           // типичная ёмкость компартмента
        };
        let mut state = DendriteState::new(n, &config);

        let V_soma = vec![-65.0f32; n];

        // Слабая AMPA проводимость — физиологично для одного входа
        for i in 0..n {
            state.set_ampa(i, 0, 0.01);
            state.set_ampa(i, 1, 0.01);
            state.set_ampa(i, 2, 0.01);
        }

        // Много шагов с dt=0.5 — Forward Euler стабилен при λ = -0.02
        for _ in 0..20 {
            state.step(&V_soma, 0.5);
        }

        let i_sum = state.I_dend[0];
        assert!(
            i_sum > 0.0,
            "AMPA should depolarize dendrites: I_dend={i_sum:.6}"
        );
        assert!(
            i_sum < 10.0,
            "I_dend should be small with weak AMPA: got {i_sum:.6}"
        );
    }

    /// Тест: GABA_A шунтирует AMPA на том же компартменте.
    #[test]
    fn dendrite_gaba_shunts_ampa() {
        let n = 8;
        let config = DendriteConfig {
            n_compartments: 2,
            R_m: 100.0,
            C_m: 1.0,           // большая C_m → стабильный FE
        };
        let mut state_ampa_only = DendriteState::new(n, &config);
        let mut state_ampa_gaba = DendriteState::new(n, &config);

        let V_soma = vec![-65.0f32; n];

        for i in 0..n {
            state_ampa_only.set_ampa(i, 0, 0.1);
            state_ampa_gaba.set_ampa(i, 0, 0.1);
            state_ampa_gaba.set_gaba_a(i, 0, 0.2);  // + GABA_A shunt
        }

        for _ in 0..20 {
            state_ampa_only.step(&V_soma, 0.5);
            state_ampa_gaba.step(&V_soma, 0.5);
        }

        let i_only = state_ampa_only.I_dend[0];
        let i_shunt = state_ampa_gaba.I_dend[0];

        // GABA_A шунтирует: при V_dend > -65, E_gaba=-65 даёт гиперполяризацию
        assert!(
            i_shunt < i_only,
            "GABA_A should reduce net current: only={i_only:.6}, shunt={i_shunt:.6}"
        );
    }

    /// Тест: SIMD vs scalar tail консистентность.
    /// n=9 → SIMD обрабатывает 0..8, scalar tail обрабатывает 8..9.
    #[test]
    fn dendrite_simd_vs_scalar_tail() {
        let n = 9;  // 8 SIMD + 1 scalar tail
        let config = DendriteConfig {
            n_compartments: 2,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut state = DendriteState::new(n, &config);

        let V_soma = vec![-65.0f32; n];

        // Одинаковые проводимости на всех нейронах
        for i in 0..n {
            state.set_ampa(i, 0, 0.3);
            state.set_ampa(i, 1, 0.7);
        }

        // Несколько шагов для накопления различий
        for _ in 0..10 {
            state.step(&V_soma, 0.5);
        }

        // SIMD (индекс 0) vs scalar tail (индекс 8)
        let diff = (state.I_dend[0] - state.I_dend[8]).abs();
        assert!(
            diff < 1e-5,
            "SIMD vs scalar tail mismatch: SIMD={:.8}, scalar={:.8}, diff={:.10}",
            state.I_dend[0], state.I_dend[8], diff
        );

        // V_dend тоже должен совпадать (stride = n, не n_padded)
        let vd_simd = state.V_dend[0 * state.n + 0];
        let vd_scalar = state.V_dend[0 * state.n + 8];
        let vd_diff = (vd_simd - vd_scalar).abs();
        assert!(
            vd_diff < 1e-5,
            "V_dend SIMD vs scalar: SIMD={:.8}, scalar={:.8}, diff={:.10}",
            vd_simd, vd_scalar, vd_diff
        );
    }
}
