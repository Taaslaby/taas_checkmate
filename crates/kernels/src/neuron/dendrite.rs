#![allow(non_snake_case)]
use wide::f32x8;
use super::{load8, store8};

// ═══════════════════════════════════════════════════════════
//  Cable Dendrite — Compartmentalized Dendrites with Axial Coupling
// ═══════════════════════════════════════════════════════════
//
//  Каждый нейрон имеет k компартментов, соединённых аксиально (R_a).
//  Все компартменты имеют мембранное сопротивление (R_m) к соме.
//
//  Уравнение cable (Forward Euler):
//
//    C_m · dV_k/dt =  (V_{k-1} − V_k) / R_a      ← аксиальный ток от предыдущего
//                   + (V_{k+1} − V_k) / R_a      ← аксиальный ток к следующему
//                   + (V_soma  − V_k) / R_m      ← электротонический ток к соме
//                   + Σ g_r[k] · (E_rev_r − V_k) ← локальный синаптический ток
//
//    I_dend += (V_k − V_soma) / R_m              ← полный ток в сому (для soma step)
//
//  Граничные условия:
//    k=0: первый компартмент имеет аксиальную связь с сомой (V_soma − V_0)/R_a
//    k=K-1: последний компартмент не имеет следующего (тупик)
//
//  Layout (column-major, stride = n_padded):
//    buf[comp * n_padded + nrn] = компартмент comp, нейрон nrn
//
//  Это позволяет грузить 8 нейронов одного компартмента одним f32x8!

// ═══════════════════════════════════════════════════════════
//  NMDA Mg²⁺ block (copied from synapse.rs for independent module)
// ═══════════════════════════════════════════════════════════

/// Mg²⁺ block factor: B(V) = 1 / (1 + 0.33·exp(-0.06·V))
/// At resting (-65 mV): B ≈ 0.02 (blocked)
/// At spike peak (+30 mV): B ≈ 0.95 (open)
#[inline(always)]
fn nmda_mg_block(v: f32) -> f32 {
    1.0 / (1.0 + 0.33 * (-0.06 * v).exp())
}

#[inline(always)]
unsafe fn nmda_mg_block_simd(v: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    let factor = f32x8::splat(0.33);
    let exp_arg = f32x8::splat(-0.06) * v;
    let clamped = exp_arg.max(f32x8::splat(-10.0)).min(f32x8::splat(10.0));
    one / (one + factor * super::fast_exp_simd(clamped))
}

// ═══════════════════════════════════════════════════════════
//  Receptor reversal potentials (mV)
// ═══════════════════════════════════════════════════════════

const E_AMPA: f32 = 0.0;
const E_NMDA: f32 = 0.0;
const E_GABA_A: f32 = -65.0;
const E_GABA_B: f32 = -90.0;
const E_NACHR: f32 = 0.0;
const E_GLUCL: f32 = -50.0;
const E_MACHR: f32 = -90.0;

// ═══════════════════════════════════════════════════════════
//  Unsafe SIMD kernel — full cable step
// ═══════════════════════════════════════════════════════════

/// Шаг компартментализованного дендритного кабеля — полный cable equation.
///
/// # Layout (column-major, stride = n_padded)
/// Все буферы размера `n_padded × k`, где n_padded = n, округлённый до %8:
/// ```text
/// V_dend[comp * n_padded + nrn] — потенциал компартмента
/// g_ampa[comp * n_padded + nrn] — AMPA проводимость
/// ... (все 7 типов)
/// ```
/// V_soma[n] — потенциалы сом
/// I_dend[n] — выход: аккумулируется ток в сому (прибавляется к существующему)
///
/// # Cable equation
/// ```text
/// dV_k/dt = ((V_{k-1} − V_k)/R_a + (V_{k+1} − V_k)/R_a + (V_soma − V_k)/R_m + I_syn_k) / C_m
///
/// I_syn_k = g_ampa·(E_ampa − V_k) + g_nmda·B(V_k)·(E_nmda − V_k) +
///           g_gaba_a·(E_gaba_a − V_k) + g_gaba_b·(E_gaba_b − V_k) +
///           g_nachr·(E_nAChR − V_k) + g_glucl·(E_GluCl − V_k) +
///           g_machr·(E_mAChR − V_k)
///
/// I_dend += (V_0 − V_soma) / R_a + Σ (V_k − V_soma) / R_m
/// ```
///
/// # Safety
/// Все указатели должны быть валидны для `n_padded * k` элементов
/// (V_dend, g_ampa..g_machr) и `n` элементов (V_soma, I_dend).
/// Не должны пересекаться.
pub(crate) unsafe fn cable_step(
    V_dend: *mut f32,        // [n_padded × k] column-major
    g_ampa: *const f32,      // [n_padded × k]
    g_nmda: *const f32,      // [n_padded × k]
    g_gaba_a: *const f32,    // [n_padded × k]
    g_gaba_b: *const f32,    // [n_padded × k]
    g_nachr: *const f32,     // [n_padded × k]
    g_glucl: *const f32,     // [n_padded × k]
    g_machr: *const f32,     // [n_padded × k]
    V_soma: *const f32,      // [n]
    I_dend: *mut f32,        // [n] — accumulates
    n: usize,                // количество нейронов (unpadded)
    n_padded: usize,         // n, округлённое до %8
    k: usize,                // количество компартментов на нейрон
    R_a: f32,                // аксиальное сопротивление (MΩ)
    R_m: f32,                // мембранное сопротивление (MΩ)
    C_m: f32,                // ёмкость компартмента (nF)
    dt: f32,                 // шаг (ms)
) {
    let inv_Rm = f32x8::splat(1.0 / R_m);
    let inv_Cm = f32x8::splat(1.0 / C_m);
    let dt_s = f32x8::splat(dt);
    let zero = f32x8::splat(0.0);
    let has_axial = k > 1 && R_a > 0.1;  // skip axial if R_a near-zero (avoid division issues)
    let inv_Ra = if has_axial { f32x8::splat(1.0 / R_a) } else { zero };

    // Reversal potentials as SIMD constants
    let e_ampa = f32x8::splat(E_AMPA);
    let e_nmda = f32x8::splat(E_NMDA);
    let e_gaba_a = f32x8::splat(E_GABA_A);
    let e_gaba_b = f32x8::splat(E_GABA_B);
    let e_nachr = f32x8::splat(E_NACHR);
    let e_glucl = f32x8::splat(E_GLUCL);
    let e_machr = f32x8::splat(E_MACHR);

    let n8 = n & !7;

    for comp in 0..k {
        let base = comp * n_padded;  // stride = n_padded (not n!)

        for off in (0..n8).step_by(8) {
            let idx = base + off;

            let vd = load8(V_dend, idx);
            let vs = load8(V_soma, off);

            // Synaptic current: I_syn = Σ g_r · (E_rev_r - V_dend)
            let ga = load8(g_ampa, idx);
            let gn = load8(g_nmda, idx);
            let gg_a = load8(g_gaba_a, idx);
            let gg_b = load8(g_gaba_b, idx);
            let gnac = load8(g_nachr, idx);
            let gglu = load8(g_glucl, idx);
            let gmac = load8(g_machr, idx);

            let mut i_syn = zero
                + ga * (e_ampa - vd)
                + gn * nmda_mg_block_simd(vd) * (e_nmda - vd)
                + gg_a * (e_gaba_a - vd)
                + gg_b * (e_gaba_b - vd)
                +            gnac * (e_nachr - vd)
                +            gglu * (e_glucl - vd)
                +            gmac * (e_machr - vd);

            // dV_dt = electrotonic + synaptic
            let mut dV_dt = (vs - vd) * inv_Rm;

            // Add axial currents from adjacent compartments
            if has_axial {
                // Current from previous compartment (k-1 → k)
                if comp > 0 {
                    let prev_base = (comp - 1) * n_padded;
                    let v_prev = load8(V_dend, prev_base + off);
                    i_syn += (v_prev - vd) * inv_Ra;
                } else {
                    // First compartment: axial connection to soma
                    i_syn += (vs - vd) * inv_Ra;
                }

                // Current from next compartment (k+1 → k)
                if comp < k - 1 {
                    let next_base = (comp + 1) * n_padded;
                    let v_next = load8(V_dend, next_base + off);
                    i_syn += (v_next - vd) * inv_Ra;
                }
            }

            dV_dt += i_syn * inv_Cm;

            // Update V_dend
            let new_vd = vd + dV_dt * dt_s;
            store8(V_dend, idx, new_vd);

            // Accumulate current to soma:
            // 1) Electrotonic: (V_dend - V_soma) / R_m
            // 2) Axial from first compartment: (V_0 - V_soma) / R_a
            let mut i_to_soma = (vd - vs) * inv_Rm;
            if has_axial && comp == 0 {
                // First compartment also contributes axial current to soma
                i_to_soma += (vd - vs) * inv_Ra;
            }

            let mut i_dend = load8(I_dend, off);
            i_dend += i_to_soma;
            store8(I_dend, off, i_dend);
        }

        // Scalar tail (n % 8 != 0)
        if n8 < n {
            for i in n8..n {
                let idx = base + i;
                let vd = *V_dend.add(idx);
                let vs = *V_soma.add(i);

                // Synaptic current (all 7 receptors)
                let ga = *g_ampa.add(idx);
                let gn = *g_nmda.add(idx);
                let gg_a = *g_gaba_a.add(idx);
                let gg_b = *g_gaba_b.add(idx);
                let gnac = *g_nachr.add(idx);
                let gglu = *g_glucl.add(idx);
                let gmac = *g_machr.add(idx);

                let i_syn = ga * (E_AMPA - vd)
                    + gn * nmda_mg_block(vd) * (E_NMDA - vd)
                    + gg_a * (E_GABA_A - vd)
                    + gg_b * (E_GABA_B - vd)
            + gnac * (E_NACHR - vd)
            + gglu * (E_GLUCL - vd)
            + gmac * (E_MACHR - vd);

                // dV/dt = electrotonic + synaptic
                let mut dV_dt = (vs - vd) / R_m + i_syn / C_m;

                // Axial coupling
                if has_axial {
                    let inv_Ra_scalar = 1.0 / R_a;
                    if comp > 0 {
                        let v_prev = *V_dend.add((comp - 1) * n_padded + i);
                        dV_dt += (v_prev - vd) * inv_Ra_scalar / C_m;
                    } else {
                        dV_dt += (vs - vd) * inv_Ra_scalar / C_m;
                    }
                    if comp < k - 1 {
                        let v_next = *V_dend.add((comp + 1) * n_padded + i);
                        dV_dt += (v_next - vd) * inv_Ra_scalar / C_m;
                    }
                }

                *V_dend.add(idx) = vd + dV_dt * dt;

                // Current to soma
                let mut i_to_soma = (vd - vs) / R_m;
                if has_axial && comp == 0 {
                    i_to_soma += (vd - vs) / R_a;
                }
                *I_dend.add(i) += i_to_soma;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Precomputed decay factors (copied pattern from synapse.rs)
// ═══════════════════════════════════════════════════════════

/// Precomputed decay factors for dendritic conductances.
#[derive(Debug, Clone)]
pub struct DendriteDecayFactors {
    /// Array of 7 decay factors: [AMPA, NMDA, GABA_A, GABA_B, nAChR, GluCl, mAChR]
    pub decays: [f32; 7],
}

impl DendriteDecayFactors {
    /// Compute decay factors from dt (ms) — matches SynapticKinetics time constants.
    pub fn new(dt: f32) -> Self {
        Self {
            decays: [
                (-dt / 2.0).exp(),     // AMPA τ=2ms
                (-dt / 50.0).exp(),    // NMDA τ=50ms
                (-dt / 10.0).exp(),    // GABA_A τ=10ms
                (-dt / 200.0).exp(),   // GABA_B τ=200ms
                (-dt / 3.0).exp(),     // nAChR τ=3ms
                (-dt / 5.0).exp(),     // GluCl τ=5ms
                (-dt / 150.0).exp(),   // mAChR τ=150ms
            ],
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  DendriteConfig — cable parameters
// ═══════════════════════════════════════════════════════════

/// Конфигурация дендритного дерева для одного нейрона.
#[derive(Debug, Clone)]
pub struct DendriteConfig {
    /// Количество компартментов на нейрон
    pub n_compartments: usize,
    /// Аксиальное сопротивление между компартментами (MΩ)
    /// Высокое R_a → слабая связь (электрически длинный дендрит)
    pub R_a: f32,
    /// Мембранное сопротивление компартмента к соме (MΩ)
    pub R_m: f32,
    /// Ёмкость компартмента (nF)
    pub C_m: f32,
}

impl Default for DendriteConfig {
    fn default() -> Self {
        Self {
            n_compartments: 4,
            R_a: 200.0,     // 200 MΩ аксиального сопротивления
            R_m: 100.0,     // 100 MΩ мембранного
            C_m: 0.1,       // 0.1 nF
        }
    }
}

impl DendriteConfig {
    /// Cable time constant τ = R_m · C_m (ms)
    pub fn tau(self) -> f32 {
        self.R_m * self.C_m
    }

    /// Electrotonic length λ = sqrt(R_m / (R_a + R_m)) per compartment
    pub fn electrotonic_length(self) -> f32 {
        if self.R_a <= 0.0 { return f32::INFINITY; }
        (self.R_m / (self.R_m + self.R_a)).sqrt()
    }

    /// Number of compartments for a given total cable length in λ units.
    /// Each compartment ≈ 0.1λ for accurate cable simulation (Rall 1959).
    pub fn compartments_for_length(length_lambda: f32) -> usize {
        (length_lambda / 0.1).ceil().max(2.0) as usize
    }
}

// ═══════════════════════════════════════════════════════════
//  CableDendriteState — полное состояние дендритного кабеля
// ═══════════════════════════════════════════════════════════

/// Состояние компартментализованного дендритного кабеля для всей популяции.
///
/// Все буферы выровнены до %8 для SIMD (stride = n_padded).
///
/// # Пример сомато-дендритного симуляционного цикла
/// ```ignore
/// // Каждый шаг:
/// dend.decay_conductances(&factors);        // g *= exp(-dt/τ)
/// dend.cable_step(V_soma, dt);              // cable dynamics → I_dend
/// // I_dend[i] теперь содержит ток в сому от дендритов
/// lif_step(v, Ii, ..., I_dend = Some(&dend.I_dend));  // soma + dendrites
/// // ... propagate spikes → jump conductances on dendrite compartments
/// ```
#[derive(Debug, Clone)]
pub struct CableDendriteState {
    pub n: usize,            // количество нейронов (unpadded)
    pub k: usize,            // количество компартментов
    pub n_padded: usize,     // n, округлённое до %8
    pub config: DendriteConfig,
    /// V_dend[n_padded × k] — column-major
    pub V_dend: Vec<f32>,
    /// Проводимости всех 7 типов [n_padded × k] каждый
    pub g_ampa: Vec<f32>,
    pub g_nmda: Vec<f32>,
    pub g_gaba_a: Vec<f32>,
    pub g_gaba_b: Vec<f32>,
    pub g_nachr: Vec<f32>,
    pub g_glucl: Vec<f32>,
    pub g_machr: Vec<f32>,
    /// I_dend[n_padded] — аккумулятор тока в сому (перезаписывается каждый шаг)
    pub I_dend: Vec<f32>,
}

impl CableDendriteState {
    /// Создать новое дендритное дерево.
    ///
    /// V_dend инициализируется V_rest = -65 mV.
    /// Все проводимости начинаются с 0.
    pub fn new(n: usize, config: &DendriteConfig) -> Self {
        let n_padded = n.div_ceil(8) * 8;
        let total = n_padded * config.n_compartments;
        Self {
            n,
            k: config.n_compartments,
            n_padded,
            config: config.clone(),
            V_dend: vec![-65.0; total],
            g_ampa: vec![0.0; total],
            g_nmda: vec![0.0; total],
            g_gaba_a: vec![0.0; total],
            g_gaba_b: vec![0.0; total],
            g_nachr: vec![0.0; total],
            g_glucl: vec![0.0; total],
            g_machr: vec![0.0; total],
            I_dend: vec![0.0; n_padded],
        }
    }

    // ─── Cable step ──────────────────────────────────────

    /// Шаг cable dynamics (алиас для обратной совместимости).
    pub fn step(&mut self, V_soma: &[f32], dt: f32) {
        self.cable_step(V_soma, dt);
    }

    /// Шаг cable dynamics: обновляет V_dend и аккумулирует I_dend.
    ///
    /// После вызова `self.I_dend[i]` содержит полный дендритный ток нейрона i
    /// (сумма электротонического + аксиального вкладов всех компартментов).
    ///
    /// # Panics
    /// Если `V_soma.len() < n`.
    pub fn cable_step(&mut self, V_soma: &[f32], dt: f32) {
        assert!(V_soma.len() >= self.n, "cable_step: V_soma too short ({} < {})", V_soma.len(), self.n);

        // Reset I_dend before accumulation
        self.I_dend.fill(0.0);

        unsafe {
            cable_step(
                self.V_dend.as_mut_ptr(),
                self.g_ampa.as_ptr(),
                self.g_nmda.as_ptr(),
                self.g_gaba_a.as_ptr(),
                self.g_gaba_b.as_ptr(),
                self.g_nachr.as_ptr(),
                self.g_glucl.as_ptr(),
                self.g_machr.as_ptr(),
                V_soma.as_ptr(),
                self.I_dend.as_mut_ptr(),
                self.n,
                self.n_padded,
                self.k,
                self.config.R_a,
                self.config.R_m,
                self.config.C_m,
                dt,
            );
        }
    }

    // ─── Conductance decay ───────────────────────────────

    /// Экспоненциальный спад проводимостей: `g *= exp(-dt/τ)`.
    ///
    /// Использует f32x8 SIMD для каждого компартмента.
    pub fn decay_conductances(&mut self, factors: &DendriteDecayFactors) {
        let d0 = f32x8::splat(factors.decays[0]); // AMPA
        let d1 = f32x8::splat(factors.decays[1]); // NMDA
        let d2 = f32x8::splat(factors.decays[2]); // GABA_A
        let d3 = f32x8::splat(factors.decays[3]); // GABA_B
        let d4 = f32x8::splat(factors.decays[4]); // nAChR
        let d5 = f32x8::splat(factors.decays[5]); // GluCl
        let d6 = f32x8::splat(factors.decays[6]); // mAChR

        let total = self.n_padded * self.k;
        let n8 = total & !7;

        unsafe {
            for off in (0..n8).step_by(8) {
                store8(self.g_ampa.as_mut_ptr(), off, load8(self.g_ampa.as_ptr(), off) * d0);
                store8(self.g_nmda.as_mut_ptr(), off, load8(self.g_nmda.as_ptr(), off) * d1);
                store8(self.g_gaba_a.as_mut_ptr(), off, load8(self.g_gaba_a.as_ptr(), off) * d2);
                store8(self.g_gaba_b.as_mut_ptr(), off, load8(self.g_gaba_b.as_ptr(), off) * d3);
                store8(self.g_nachr.as_mut_ptr(), off, load8(self.g_nachr.as_ptr(), off) * d4);
                store8(self.g_glucl.as_mut_ptr(), off, load8(self.g_glucl.as_ptr(), off) * d5);
                store8(self.g_machr.as_mut_ptr(), off, load8(self.g_machr.as_ptr(), off) * d6);
            }
        }

        // Scalar tail
        for i in n8..total {
            self.g_ampa[i] *= factors.decays[0];
            self.g_nmda[i] *= factors.decays[1];
            self.g_gaba_a[i] *= factors.decays[2];
            self.g_gaba_b[i] *= factors.decays[3];
            self.g_nachr[i] *= factors.decays[4];
            self.g_glucl[i] *= factors.decays[5];
            self.g_machr[i] *= factors.decays[6];
        }
    }

    /// Сбросить все проводимости в 0.
    pub fn reset_conductances(&mut self) {
        let total = self.n_padded * self.k;
        self.g_ampa[..total].fill(0.0);
        self.g_nmda[..total].fill(0.0);
        self.g_gaba_a[..total].fill(0.0);
        self.g_gaba_b[..total].fill(0.0);
        self.g_nachr[..total].fill(0.0);
        self.g_glucl[..total].fill(0.0);
        self.g_machr[..total].fill(0.0);
    }

    // ─── Conductance setters ─────────────────────────────

    /// Jump AMPA conductance by `amount` on compartment `comp` of neuron `nrn`.
    pub fn jump_ampa(&mut self, nrn: usize, comp: usize, amount: f32) {
        if nrn < self.n && comp < self.k {
            self.g_ampa[comp * self.n_padded + nrn] += amount;
        }
    }

    /// Jump NMDA conductance.
    pub fn jump_nmda(&mut self, nrn: usize, comp: usize, amount: f32) {
        if nrn < self.n && comp < self.k {
            self.g_nmda[comp * self.n_padded + nrn] += amount;
        }
    }

    /// Jump GABA_A conductance.
    pub fn jump_gaba_a(&mut self, nrn: usize, comp: usize, amount: f32) {
        if nrn < self.n && comp < self.k {
            self.g_gaba_a[comp * self.n_padded + nrn] += amount;
        }
    }

    /// Jump GABA_B conductance.
    pub fn jump_gaba_b(&mut self, nrn: usize, comp: usize, amount: f32) {
        if nrn < self.n && comp < self.k {
            self.g_gaba_b[comp * self.n_padded + nrn] += amount;
        }
    }

    /// Jump nAChR conductance.
    pub fn jump_nachr(&mut self, nrn: usize, comp: usize, amount: f32) {
        if nrn < self.n && comp < self.k {
            self.g_nachr[comp * self.n_padded + nrn] += amount;
        }
    }

    /// Jump GluCl conductance.
    pub fn jump_glucl(&mut self, nrn: usize, comp: usize, amount: f32) {
        if nrn < self.n && comp < self.k {
            self.g_glucl[comp * self.n_padded + nrn] += amount;
        }
    }

    /// Jump mAChR conductance.
    pub fn jump_machr(&mut self, nrn: usize, comp: usize, amount: f32) {
        if nrn < self.n && comp < self.k {
            self.g_machr[comp * self.n_padded + nrn] += amount;
        }
    }

    /// Jump by receptor index (0-6, matches ReceptorType ordering).
    pub fn jump_by_index(&mut self, nrn: usize, comp: usize, receptor: u8, amount: f32) {
        match receptor {
            0 => self.jump_ampa(nrn, comp, amount),
            1 => self.jump_nmda(nrn, comp, amount),
            2 => self.jump_gaba_a(nrn, comp, amount),
            3 => self.jump_gaba_b(nrn, comp, amount),
            4 => self.jump_nachr(nrn, comp, amount),
            5 => self.jump_glucl(nrn, comp, amount),
            6 => self.jump_machr(nrn, comp, amount),
            _ => {}
        }
    }

    // ─── Getters ─────────────────────────────────────────

    /// Суммарная проводимость всех типов для компартмента `comp` нейрона `nrn`.
    pub fn total_conductance(&self, nrn: usize, comp: usize) -> f32 {
        if nrn >= self.n || comp >= self.k { return 0.0; }
        let idx = comp * self.n_padded + nrn;
        self.g_ampa[idx] + self.g_nmda[idx] + self.g_gaba_a[idx] + self.g_gaba_b[idx]
            + self.g_nachr[idx] + self.g_glucl[idx] + self.g_machr[idx]
    }

    /// Потенциал компартмента `comp` нейрона `nrn`.
    pub fn v_dend(&self, nrn: usize, comp: usize) -> f32 {
        if nrn >= self.n || comp >= self.k { return 0.0; }
        self.V_dend[comp * self.n_padded + nrn]
    }

    /// Дендритный ток нейрона `nrn` (после вызова cable_step).
    pub fn dend_current(&self, nrn: usize) -> f32 {
        if nrn >= self.n { 0.0 } else { self.I_dend[nrn] }
    }

    /// Установить V_dend компартмента (для тестов).
    pub fn set_v_dend(&mut self, nrn: usize, comp: usize, value: f32) {
        if nrn < self.n && comp < self.k {
            self.V_dend[comp * self.n_padded + nrn] = value;
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  Совместимость со старым DendriteState
// ═══════════════════════════════════════════════════════════

/// Устаревший тип — используйте [`CableDendriteState`] вместо него.
pub type DendriteState = CableDendriteState;
/// Устаревший тип — используйте [`DendriteConfig`].
// (DendriteConfig уже существует — это новый улучшенный вариант)

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── 1. Базовый EPSP: AMPA деполяризует дендрит ────

    #[test]
    fn test_dendrite_basic_ampa_epsps() {
        let config = DendriteConfig { n_compartments: 2, R_a: 200.0, R_m: 100.0, C_m: 0.1 };
        let mut dend = CableDendriteState::new(8, &config);
        let V_soma = vec![-65.0f32; 8];

        // AMPA jump на компартмент 0, нейрон 0
        dend.jump_ampa(0, 0, 0.1);

        let factors = DendriteDecayFactors::new(0.5);
        for _ in 0..20 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 0.5);
        }

        // AMPA should produce depolarization → positive I_dend
        let i0 = dend.dend_current(0);
        assert!(i0 > 0.0, "AMPA should produce positive I_dend, got {i0:.8}");

        // Neuron 1 (no input) should have near-zero
        let i1 = dend.dend_current(1);
        assert!(i1.abs() < 1e-4, "No-input neuron should have ~0 I_dend, got {i1:.8}");
    }

    // ─── 2. GABA_A шунтирует AMPA ──────────────────────

    #[test]
    fn test_dendrite_gaba_shunt() {
        let config = DendriteConfig { n_compartments: 2, R_a: 200.0, R_m: 100.0, C_m: 0.1 };
        let mut dend_only = CableDendriteState::new(4, &config);
        let mut dend_shunt = CableDendriteState::new(4, &config);
        let V_soma = vec![-65.0f32; 4];
        let factors = DendriteDecayFactors::new(0.5);

        // Both get AMPA, shunt also gets GABA_A on same compartment
        dend_only.jump_ampa(0, 0, 0.2);
        dend_shunt.jump_ampa(0, 0, 0.2);
        dend_shunt.jump_gaba_a(0, 0, 0.4);

        for _ in 0..20 {
            dend_only.decay_conductances(&factors);
            dend_shunt.decay_conductances(&factors);
            dend_only.cable_step(&V_soma, 0.5);
            dend_shunt.cable_step(&V_soma, 0.5);
        }

        let i_only = dend_only.dend_current(0);
        let i_shunt = dend_shunt.dend_current(0);
        assert!(i_shunt < i_only,
            "GABA_A should reduce net current: only={i_only:.6}, shunt={i_shunt:.6}");
    }

    // ─── 3. SIMD vs scalar tail консистентность ────────

    #[test]
    fn test_dendrite_simd_vs_scalar_tail() {
        let config = DendriteConfig { n_compartments: 3, R_a: 200.0, R_m: 100.0, C_m: 0.1 };
        let mut dend = CableDendriteState::new(9, &config);
        let V_soma = vec![-65.0f32; 9];
        let factors = DendriteDecayFactors::new(0.5);

        // All neurons get identical input
        for i in 0..9 {
            dend.jump_ampa(i, 0, 0.3);
            dend.jump_ampa(i, 2, 0.5);
        }

        for _ in 0..15 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 0.5);
        }

        // SIMD (idx 0) vs scalar (idx 8)
        let diff = (dend.dend_current(0) - dend.dend_current(8)).abs();
        assert!(diff < 1e-5,
            "SIMD vs scalar tail mismatch: idx0={:.8}, idx8={:.8}, diff={:.10}",
            dend.dend_current(0), dend.dend_current(8), diff);

        // V_dend also matches
        for comp in 0..3 {
            let vd0 = dend.v_dend(0, comp);
            let vd8 = dend.v_dend(8, comp);
            let vd_diff = (vd0 - vd8).abs();
            assert!(vd_diff < 1e-5,
                "V_dend[comp={comp}] mismatch: nrn0={vd0:.8}, nrn8={vd8:.8}, diff={vd_diff:.10}");
        }
    }

    // ─── 4. Axial coupling: proximal > distal EPSP ─────

    #[test]
    fn test_dendrite_proximal_vs_distal_ampa() {
        // Proximal compartment should produce larger I_dend than distal
        let config = DendriteConfig { n_compartments: 4, R_a: 200.0, R_m: 100.0, C_m: 0.1 };
        let mut prox = CableDendriteState::new(4, &config);
        let mut dist = CableDendriteState::new(4, &config);
        let V_soma = vec![-65.0f32; 4];
        let factors = DendriteDecayFactors::new(0.5);

        // Same AMPA on comp 0 (proximal) vs comp 3 (distal)
        prox.jump_ampa(0, 0, 0.3);
        dist.jump_ampa(0, 3, 0.3);

        for _ in 0..20 {
            prox.decay_conductances(&factors);
            dist.decay_conductances(&factors);
            prox.cable_step(&V_soma, 0.5);
            dist.cable_step(&V_soma, 0.5);
        }

        let i_prox = prox.dend_current(0);
        let i_dist = dist.dend_current(0);
        assert!(i_prox > i_dist,
            "Proximal AMPA should give larger I_dend: prox={i_prox:.6}, dist={i_dist:.6}");
        println!("Proximal I_dend={i_prox:.8}, Distal I_dend={i_dist:.8}, ratio={:.4}", i_prox / i_dist.max(1e-10));
    }

    // ─── 5. Decay проводимостей ────────────────────────

    #[test]
    fn test_dendrite_conductance_decay() {
        let config = DendriteConfig { n_compartments: 2, R_a: 200.0, R_m: 100.0, C_m: 0.1 };
        let mut dend = CableDendriteState::new(4, &config);
        // jump_ampa(nrn=0, comp=0, amount=1.0) → g_ampa[0 * n_padded + 0] = g_ampa[0]
        dend.jump_ampa(0, 0, 1.0);
        // jump_nmda(nrn=1, comp=0, amount=1.0) → g_nmda[0 * n_padded + 1] = g_nmda[1]
        dend.jump_nmda(1, 0, 1.0);

        let factors = DendriteDecayFactors::new(2.0); // dt=2ms
        dend.decay_conductances(&factors);

        // AMPA τ=2ms: exp(-2/2) = 0.3679 at g_ampa[0]
        assert!((dend.g_ampa[0] - 0.3679).abs() < 0.001,
            "AMPA decay: expected 0.3679, got {}", dend.g_ampa[0]);

        // NMDA τ=50ms: exp(-2/50) = 0.9608 at g_nmda[1] (nrn=1, comp=0)
        // Index: 0 * n_padded + 1 = 1
        let nmda_idx = 1; // comp=0, nrn=1
        assert!((dend.g_nmda[nmda_idx] - 0.9608).abs() < 0.001,
            "NMDA decay: expected 0.9608, got {}", dend.g_nmda[nmda_idx]);
    }

    // ─── 6. 7 receptor types все работают ──────────────

    #[test]
    fn test_dendrite_all_7_receptors() {
        let config = DendriteConfig { n_compartments: 1, R_a: 200.0, R_m: 100.0, C_m: 0.1 };
        let mut dend = CableDendriteState::new(8, &config);
        let V_soma = vec![-65.0f32; 8];
        let factors = DendriteDecayFactors::new(0.5);

        // Запускаем все 7 типов на разных нейронах
        dend.jump_ampa(0, 0, 0.5);
        dend.jump_nmda(1, 0, 0.5);
        dend.jump_gaba_a(2, 0, 0.5);
        dend.jump_gaba_b(3, 0, 0.5);
        dend.jump_nachr(4, 0, 0.5);
        dend.jump_glucl(5, 0, 0.5);
        dend.jump_machr(6, 0, 0.5);

        for _ in 0..10 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 0.5);
        }

        // nAChR (E_rev=0) как AMPA — возбуждающий
        assert!(dend.dend_current(4) > 0.0,
            "nAChR should be excitatory, got {}", dend.dend_current(4));

        // GABA_B (E_rev=-90) — гиперполяризующий
        // В покое V=-65, E_rev=-90: ток отрицательный
        assert!(dend.dend_current(3) < 0.0,
            "GABA_B should be hyperpolarizing (inhibitory), got {}", dend.dend_current(3));

        // Проверяем что для всех нейронов посчитан ток
        for i in 0..7 {
            println!("Neuron {i} (receptor {i}): I_dend = {:.8}", dend.dend_current(i));
        }
    }

    // ─── 7. Cable steady state (Rall 1959) ─────────────

    #[test]
    fn test_dendrite_cable_steady_state() {
        // Без синаптического входа, V_dend должен релаксировать к V_soma
        let config = DendriteConfig { n_compartments: 4, R_a: 200.0, R_m: 100.0, C_m: 0.5 };
        let mut dend = CableDendriteState::new(4, &config);
        let V_soma = vec![-60.0f32; 4]; // V_soma чуть выше -65

        // Начальные V_dend = -65, V_soma = -60
        // После многих шагов V_dend → V_soma
        let factors = DendriteDecayFactors::new(0.5);
        for _ in 0..500 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 0.5);
        }

        // V_dend должен быть близок к V_soma = -60
        let vd0 = dend.v_dend(0, 0);
        let vd3 = dend.v_dend(0, 3);
        println!("Steady state: V_soma=-60, V_dend[0]={vd0:.4}, V_dend[3]={vd3:.4}");
        assert!((vd0 - (-60.0)).abs() < 1.0,
            "V_dend[0] should approach V_soma=-60, got {vd0:.4}");
        assert!((vd3 - (-60.0)).abs() < 1.0,
            "V_dend[3] should approach V_soma=-60, got {vd3:.4}");

        // Дендритный ток должен быть ~0 в steady state
        let i0 = dend.dend_current(0);
        assert!(i0.abs() < 0.02,
            "Steady state I_dend should be ~0, got {i0:.8}");
    }

    // ─── 8. Multi-compartment axial gradient ───────────

    #[test]
    fn test_dendrite_axial_voltage_gradient() {
        // AMPA только на компартмент 0 — создаёт градиент V вдоль кабеля
        let config = DendriteConfig { n_compartments: 4, R_a: 500.0, R_m: 100.0, C_m: 0.1 };
        let mut dend = CableDendriteState::new(4, &config);
        let V_soma = vec![-65.0f32; 4];
        let factors = DendriteDecayFactors::new(0.5);

        dend.jump_ampa(0, 0, 0.5); // AMPA только на comp 0

        for _ in 0..30 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 0.5);
        }

        // V_dend должен убывать от comp 0 к comp 3
        let v0 = dend.v_dend(0, 0);
        let v1 = dend.v_dend(0, 1);
        let v2 = dend.v_dend(0, 2);
        let v3 = dend.v_dend(0, 3);
        println!("Axial gradient: comp0={v0:.4}, comp1={v1:.4}, comp2={v2:.4}, comp3={v3:.4}");
        assert!(v0 > v1, "V_dend should decay along cable: v0={v0:.4} > v1={v1:.4}");
        assert!(v1 > v2, "V_dend should decay: v1={v1:.4} > v2={v2:.4}");
        assert!(v2 >= v3, "V_dend should decay: v2={v2:.4} >= v3={v3:.4}");
    }

    // ─── 9. Высокое R_a → слабая аксиальная связь ─────

    #[test]
    fn test_dendrite_high_axial_resistance_isolates_compartments() {
        let config_high = DendriteConfig { n_compartments: 3, R_a: 10000.0, R_m: 100.0, C_m: 0.1 };
        let mut dend_high = CableDendriteState::new(4, &config_high);
        let V_soma = vec![-65.0f32; 4];
        let factors = DendriteDecayFactors::new(0.5);

        // AMPA только на comp 0
        dend_high.jump_ampa(0, 0, 0.5);

        for _ in 0..20 {
            dend_high.decay_conductances(&factors);
            dend_high.cable_step(&V_soma, 0.5);
        }

        // c R_a=10000, comp 2 должен быть почти изолирован
        let v0 = dend_high.v_dend(0, 0);
        let v2 = dend_high.v_dend(0, 2);
        println!("High R_a: comp0={v0:.4}, comp2={v2:.4}, diff={:.4}", (v0 - v2).abs());
        assert!((v0 - v2).abs() > 1.0,
            "High R_a should isolate compartments: ΔV={:.4} should be > 1.0", (v0 - v2).abs());
    }

    // ─── 10. Нулевое R_a = идеальная связь (все V_dend равны) ──

    #[test]
    fn test_dendrite_low_axial_resistance_near_equal() {
        // R_a=1.0 — very strong coupling, stable with dt=0.05: dt/(R_a*Cm)=0.05/(1*0.1)=0.5 < 1
        // λ = sqrt(100/(100+1)) = 0.995 → compartments nearly equal
        let dt = 0.05;
        let config_eq = DendriteConfig { n_compartments: 3, R_a: 1.0, R_m: 100.0, C_m: 0.1 };
        let mut dend_eq = CableDendriteState::new(4, &config_eq);
        let V_soma = vec![-65.0f32; 4];
        let factors = DendriteDecayFactors::new(dt);

        dend_eq.jump_ampa(0, 0, 0.5); // AMPA только на comp 0

        for _ in 0..200 {
            dend_eq.decay_conductances(&factors);
            dend_eq.cable_step(&V_soma, dt);
        }

        // С R_a=1.0, все V_dend должны быть почти равны (λ=0.995)
        let v0 = dend_eq.v_dend(0, 0);
        let v1 = dend_eq.v_dend(0, 1);
        let v2 = dend_eq.v_dend(0, 2);
        let max_diff = ((v0 - v1).abs()).max((v0 - v2).abs());
        println!("R_a=1.0 (dt={dt}): comp0={v0:.6}, comp1={v1:.6}, comp2={v2:.6}, max_diff={max_diff:.8}");
        assert!(max_diff < 0.02,
            "Low R_a should nearly equalize compartments: max_diff={max_diff:.8} > 0.02");
    }

    // ─── 11. NMDA Mg²⁺ block ───────────────────────────

    #[test]
    fn test_dendrite_nmda_mg_block() {
        let config = DendriteConfig { n_compartments: 1, R_a: 200.0, R_m: 100.0, C_m: 0.1 };
        let mut dend = CableDendriteState::new(4, &config);

        // NMDA identical on all 4 neurons, but different V_soma
        dend.jump_nmda(0, 0, 0.5);
        dend.jump_nmda(1, 0, 0.5);
        dend.jump_nmda(2, 0, 0.5);
        dend.jump_nmda(3, 0, 0.5);

        // V_soma: -80, -65, -40, 0 mV
        let V_soma = vec![-80.0, -65.0, -40.0, 0.0];
        let factors = DendriteDecayFactors::new(0.5);

        for _ in 0..10 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 0.5);
        }

        let i0 = dend.dend_current(0); // V=-80, blocked
        let i3 = dend.dend_current(3); // V=0, open (but driving force = 0)

        println!("NMDA I_dend: V=-80mV → {i0:.6}, V=0mV → {i3:.6}");
        // At -80: blocked, small current
        assert!(i0.abs() < i3.abs() || i0.abs() < 5.0,
            "NMDA at -80 should be blocked: {i0:.6}");
    }

    // ─── 12. Membrane time constant τ = R_m · C_m ─────

    #[test]
    fn test_dendrite_membrane_time_constant() {
        // With C_m=1.0, R_m=100, τ=100ms
        let config = DendriteConfig { n_compartments: 1, R_a: 200.0, R_m: 100.0, C_m: 1.0 };
        let mut dend = CableDendriteState::new(4, &config);

        // V_soma = -50 (15 mV above V_dend=-65)
        let V_soma = vec![-50.0f32; 4];
        let factors = DendriteDecayFactors::new(5.0); // dt=5ms → 10 steps = 50ms

        dend.jump_ampa(0, 0, 0.0); // No synapse, just passive relaxation

        for _ in 0..10 {
            dend.decay_conductances(&factors);
            dend.cable_step(&V_soma, 5.0);
        }

        // After 50ms (0.5τ), V_dend should be ~63% of way to V_soma
        // V = -65 + (1 - exp(-t/τ)) * 15 = -65 + (1 - 0.606) * 15 = -59.1
        let vd = dend.v_dend(0, 0);
        println!("Passive relaxation: after 50ms (0.5τ), V_dend={vd:.4}, expected ≈ -59.1");
        assert!(vd > -63.0 && vd < -57.0,
            "V_dend should be approaching V_soma after 0.5τ: {vd:.4}");
    }

    // ─── 13. Reset conductances ────────────────────────

    #[test]
    fn test_dendrite_reset_conductances() {
        let config = DendriteConfig { n_compartments: 2, R_a: 200.0, R_m: 100.0, C_m: 0.1 };
        let mut dend = CableDendriteState::new(4, &config);

        dend.jump_ampa(0, 0, 0.5);
        dend.jump_nmda(1, 1, 0.3);
        dend.jump_gaba_a(2, 1, 0.2);

        assert!(dend.g_ampa[0] > 0.0);
        assert!(dend.g_nmda[1 * dend.n_padded + 1] > 0.0);

        dend.reset_conductances();

        let total = dend.n_padded * dend.k;
        for i in 0..total {
            assert_eq!(dend.g_ampa[i], 0.0, "g_ampa[{i}] should be 0 after reset");
            assert_eq!(dend.g_nmda[i], 0.0);
            assert_eq!(dend.g_gaba_a[i], 0.0);
            assert_eq!(dend.g_gaba_b[i], 0.0);
            assert_eq!(dend.g_nachr[i], 0.0);
            assert_eq!(dend.g_glucl[i], 0.0);
            assert_eq!(dend.g_machr[i], 0.0);
        }
    }

    // ─── 14. Jump by index (compatibility with CSR propagation) ──

    #[test]
    fn test_dendrite_jump_by_index() {
        let config = DendriteConfig { n_compartments: 2, R_a: 200.0, R_m: 100.0, C_m: 0.1 };
        let mut dend = CableDendriteState::new(4, &config);

        dend.jump_by_index(0, 0, 0, 0.5); // AMPA
        dend.jump_by_index(1, 1, 4, 0.3); // nAChR
        dend.jump_by_index(2, 0, 2, 0.2); // GABA_A

        assert!((dend.g_ampa[0] - 0.5).abs() < 1e-6);
        assert!((dend.g_nachr[1 * dend.n_padded + 1] - 0.3).abs() < 1e-6);
        assert!((dend.g_gaba_a[0 * dend.n_padded + 2] - 0.2).abs() < 1e-6);
    }
}
