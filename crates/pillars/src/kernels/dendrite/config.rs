#![allow(non_snake_case)] // R_a/R_m/C_m, V_dend — физическая нотация

//! # Dendrite Config — геометрия и пассивные/активные свойства дерева.
//!
//! Вынесено из единого `dendrite.rs` (год-файл): [`DendriteDecayFactors`],
//! [`DendriteConfig`], [`ActiveDendriteChannels`] — конфигурация без логики
//! кабельной динамики (она в [`super::cable`]).

/// Precomputed decay factors for dendritic conductances.
///
/// `decays = exp(−dt/τ)` for each of the 7 receptor types.
#[derive(Debug, Clone)]
pub struct DendriteDecayFactors {
    /// [AMPA, NMDA, GABA_A, GABA_B, nAChR, GluCl, mAChR]
    pub decays: [f32; 7],
}

impl DendriteDecayFactors {
    /// Compute decay factors from `dt` (ms) — matches SynapticKinetics τ.
    pub fn new(dt: f32) -> Self {
        Self {
            decays: [
                (-dt / 2.0).exp(),   // AMPA τ=2ms
                (-dt / 50.0).exp(),  // NMDA τ=50ms
                (-dt / 10.0).exp(),  // GABA_A τ=10ms
                (-dt / 200.0).exp(), // GABA_B τ=200ms
                (-dt / 3.0).exp(),   // nAChR τ=3ms
                (-dt / 5.0).exp(),   // GluCl τ=5ms
                (-dt / 150.0).exp(), // mAChR τ=150ms
            ],
        }
    }
}

/// Конфигурация дендритного дерева для одного нейрона.
#[derive(Debug, Clone)]
pub struct DendriteConfig {
    /// Number of compartments per neuron
    pub n_compartments: usize,
    /// Axial resistance between compartments (MΩ). High R_a → weak coupling.
    pub R_a: f32,
    /// Membrane resistance of compartment to soma (MΩ)
    pub R_m: f32,
    /// Compartment capacitance (nF)
    pub C_m: f32,
}

impl Default for DendriteConfig {
    fn default() -> Self {
        Self {
            n_compartments: 4,
            R_a: 200.0,
            R_m: 100.0,
            C_m: 0.1,
        }
    }
}

impl DendriteConfig {
    /// Cable time constant `τ = R_m·C_m` (ms)
    pub fn tau(&self) -> f32 {
        self.R_m * self.C_m
    }

    /// Electrotonic length `λ = sqrt(R_m / (R_m + R_a))` per compartment
    pub fn electrotonic_length(&self) -> f32 {
        if self.R_a <= 0.0 {
            return f32::INFINITY;
        }
        (self.R_m / (self.R_m + self.R_a)).sqrt()
    }

    /// Number of compartments for a given cable length in λ units.
    /// Each compartment ≈ 0.1λ for accurate cable simulation (Rall 1959).
    pub fn compartments_for_length(length_lambda: f32) -> usize {
        (length_lambda / 0.1).ceil().max(2.0) as usize
    }
}

/// Активные ионные каналы (Na⁺, K⁺, Ca²⁺) для дендритных компартментов.
#[derive(Debug, Clone)]
pub struct ActiveDendriteChannels {
    /// Max conductances (mS/cm²)
    pub g_na: f32,
    pub g_kv: f32,
    pub g_ca: f32,
    /// Gate variables [n_padded × k]
    pub m_na: Vec<f32>,
    pub h_na: Vec<f32>,
    pub n_kv: Vec<f32>,
    pub m_ca: Vec<f32>,
}

impl ActiveDendriteChannels {
    /// Initialize active channels for all compartments.
    pub fn new(total_elements: usize, g_na: f32, g_kv: f32, g_ca: f32) -> Self {
        Self {
            g_na,
            g_kv,
            g_ca,
            m_na: vec![0.05; total_elements],
            h_na: vec![0.60; total_elements],
            n_kv: vec![0.315; total_elements],
            m_ca: vec![0.01; total_elements],
        }
    }

    /// Gate kinetics + active ionic currents (exponential Euler).
    pub fn step(&mut self, V_dend: &mut [f32], C_m: f32, dt: f32) {
        let n_elements = V_dend.len();
        let e_na = 50.0f32;
        let e_kv = -77.0f32;
        let e_ca = 120.0f32;
        let inv_cm = 1.0 / C_m.max(1e-5);

        let decay_m_na = (-dt / 0.5f32).exp();
        let decay_h_na = (-dt / 2.0f32).exp();
        let decay_n_kv = (-dt / 2.5f32).exp();
        let decay_m_ca = (-dt / 2.0f32).exp();

        for i in 0..n_elements {
            let v = V_dend[i].clamp(-100.0, 100.0);

            // Na_m gate (exponential Euler)
            let exp_m = (-(v + 40.0) / 10.0).clamp(-20.0, 20.0).exp();
            let m_inf = 1.0 / (1.0 + exp_m);
            self.m_na[i] = (self.m_na[i] * decay_m_na + m_inf * (1.0 - decay_m_na)).clamp(0.0, 1.0);

            // Na_h gate
            let exp_h = ((v + 65.0) / 10.0).clamp(-20.0, 20.0).exp();
            let h_inf = 1.0 / (1.0 + exp_h);
            self.h_na[i] = (self.h_na[i] * decay_h_na + h_inf * (1.0 - decay_h_na)).clamp(0.0, 1.0);

            // K_n gate
            let exp_n = (-(v + 55.0) / 10.0).clamp(-20.0, 20.0).exp();
            let n_inf = 1.0 / (1.0 + exp_n);
            self.n_kv[i] = (self.n_kv[i] * decay_n_kv + n_inf * (1.0 - decay_n_kv)).clamp(0.0, 1.0);

            // Ca_m gate
            let exp_ca = (-(v + 20.0) / 9.0).clamp(-20.0, 20.0).exp();
            let m_ca_inf = 1.0 / (1.0 + exp_ca);
            self.m_ca[i] =
                (self.m_ca[i] * decay_m_ca + m_ca_inf * (1.0 - decay_m_ca)).clamp(0.0, 1.0);

            // Active currents (m³h, n⁴, m²)
            let m = self.m_na[i];
            let h = self.h_na[i];
            let n = self.n_kv[i];
            let m_ca = self.m_ca[i];

            let i_na = self.g_na * m * m * m * h * (e_na - v);
            let i_kv = self.g_kv * n * n * n * n * (e_kv - v);
            let i_ca = self.g_ca * m_ca * m_ca * (e_ca - v);

            let i_active = (i_na + i_kv + i_ca).clamp(-100.0, 100.0);
            V_dend[i] = (v + dt * i_active * inv_cm).clamp(-100.0, 100.0);
        }
    }
}
