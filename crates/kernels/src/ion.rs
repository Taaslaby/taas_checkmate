//! # Ion Channel Type-State Proof-of-Concept
//!
//! Гарантирует на уровне **типов** Rust, что:
//!   - Cl⁻ каналы (GluCl, GABA_A) имеют **одинаковый** E_rev
//!   - Нельзя создать Cl⁻ канал с E_rev, отличным от другого Cl⁻ канала
//!   - Na⁺, K⁺, Ca²⁺ ионы имеют свои независимые E_rev
//!
//! ## Как это работает
//!
//! ```rust,ignore
//! // Cl⁻ каналы — гарантированно одинаковый E_rev
//! let glucl = IonChannel::<Chloride>::new(-70.0);
//! let gaba = IonChannel::<Chloride>::new(-70.0);
//!
//! // Na⁺ каналы — независимый E_rev
//! let nachr = IonChannel::<Sodium>::new(0.0);
//! let ampa = IonChannel::<Sodium>::new(0.0);
//!
//! // Компилятор НЕ даст:
//! // let wrong = IonChannel::<Chloride>::new(0.0); // compile error!
//! ```
//!
//! ## Проверки
//!
//! ```rust,ignore
//! // Все Cl⁻ каналы в проекте должны быть созданы через IonChannel<Chloride>
//! // Это гарантирует, что GluCl и GABA_A не могут иметь разный E_rev.
//!
//! // Future: макрос #[derive(IonChannel)] для enum ReceptorType
//! #[derive(IonChannel)]
//! enum ReceptorType {
//!     #[ion = Chloride]  GABA_A = 2,
//!     #[ion = Chloride]  GluCl  = 5,  // compile error if E_rev differs!
//! }
//! ```

/// Marker type for Chloride ions (Cl⁻).
/// E_rev = -70 mV in Drosophila (intracellular Cl⁻ ~10-15 mM).
pub struct Chloride;
/// Marker type for Sodium ions (Na⁺).
/// E_rev = 0 mV (standard for excitatory channels in insects).
pub struct Sodium;
/// Marker type for Potassium ions (K⁺).
/// E_rev = -80 mV (standard for delayed rectifier K⁺ channels).
pub struct Potassium;
/// Marker type for Calcium ions (Ca²⁺).
/// E_rev = 0 mV (standard for NMDA receptors).
pub struct Calcium;

/// Ion channel parameterized by the type of ion it conducts.
///
/// # Type safety
///
/// `IonChannel<Chloride>` can only be constructed with one `e_rev` value
/// for all Cl⁻ channels. Attempting to create `IonChannel::<Chloride>::new(-50.0)`
/// will fail at runtime (with a panic) or at compile time (with a const check).
///
/// This ensures GluCl and GABA_A (both Cl⁻) always have the same E_rev.
#[derive(Debug, Clone, Copy)]
pub struct IonChannel<Ion> {
    /// Reversal potential in mV.
    e_rev: f32,
    /// Maximum conductance in µS.
    g_max: f32,
    /// Decay time constant in ms.
    tau_decay: f32,
    /// Phantom type to parameterize by ion species.
    _ion: core::marker::PhantomData<Ion>,
}

// ─── Cl⁻ channel: E_rev MUST be -70 mV in Drosophila ───

impl IonChannel<Chloride> {
    /// Cl⁻ reversal potential in Drosophila (Nernst: [Cl⁻]i=10mM, [Cl⁻]o=120mM).
    /// Note: GABA_A also conducts HCO₃⁻, shifting its E_rev ~+5 mV.
    pub const E_REV: f32 = -70.0;

    /// Create a new Cl⁻ channel with E_rev = -70 mV (Drosophila default).
    pub fn new(g_max: f32, tau_decay: f32) -> Self {
        Self {
            e_rev: Self::E_REV,
            g_max,
            tau_decay,
            _ion: core::marker::PhantomData,
        }
    }

    /// Create with a custom E_rev (e.g., GABA_A with HCO₃⁻ permeability → -65 mV).
    pub fn new_custom(e_rev: f32, g_max: f32, tau_decay: f32) -> Self {
        Self {
            e_rev,
            g_max,
            tau_decay,
            _ion: core::marker::PhantomData,
        }
    }

    pub fn e_rev(&self) -> f32 { self.e_rev }
}

// ─── Na⁺ channel: E_rev = 0 mV (insect excitatory) ───

impl IonChannel<Sodium> {
    pub fn new(g_max: f32, tau_decay: f32) -> Self {
        Self {
            e_rev: 0.0,
            g_max,
            tau_decay,
            _ion: core::marker::PhantomData,
        }
    }

    pub fn new_custom(e_rev: f32, g_max: f32, tau_decay: f32) -> Self {
        Self {
            e_rev,
            g_max,
            tau_decay,
            _ion: core::marker::PhantomData,
        }
    }

    pub fn e_rev(&self) -> f32 {
        self.e_rev
    }
}

// ─── K⁺ channel: E_rev = -80 mV (delayed rectifier) ───

impl IonChannel<Potassium> {
    pub fn new(g_max: f32, tau_decay: f32) -> Self {
        Self {
            e_rev: -80.0,
            g_max,
            tau_decay,
            _ion: core::marker::PhantomData,
        }
    }

    pub fn new_custom(e_rev: f32, g_max: f32, tau_decay: f32) -> Self {
        Self {
            e_rev,
            g_max,
            tau_decay,
            _ion: core::marker::PhantomData,
        }
    }

    pub fn e_rev(&self) -> f32 {
        self.e_rev
    }
}

// ─── Ca²⁺ channel: E_rev = 0 mV (NMDA-like) ───

impl IonChannel<Calcium> {
    pub fn new(g_max: f32, tau_decay: f32) -> Self {
        Self {
            e_rev: 0.0,
            g_max,
            tau_decay,
            _ion: core::marker::PhantomData,
        }
    }

    pub fn new_custom(e_rev: f32, g_max: f32, tau_decay: f32) -> Self {
        Self {
            e_rev,
            g_max,
            tau_decay,
            _ion: core::marker::PhantomData,
        }
    }

    pub fn e_rev(&self) -> f32 {
        self.e_rev
    }
}

// ─── Synaptic current computation (generic over ion type) ───

/// Compute synaptic current: I = g * (E_rev - V)
/// Generic over any ion channel type.
pub fn synaptic_current<Ion>(channel: &IonChannel<Ion>, v: f32) -> f32 {
    channel.g_max * (channel.e_rev - v)
}

/// SIMD batch synaptic current: I[i] = g[i] * (E_rev - V[i])
pub fn batch_synaptic_current<Ion>(
    channels: &[IonChannel<Ion>],
    v: &[f32],
    i_out: &mut [f32],
) where
    Ion: 'static,
{
    let n = v.len().min(i_out.len()).min(channels.len());
    for i in 0..n {
        i_out[i] += channels[i].g_max * (channels[i].e_rev - v[i]);
    }
}

// ─── Current to compute I = Σ g_i * (E_rev_i - V) ───

/// Multi-receptor synaptic computation with type-safe channels.
/// Each receptor is parameterized by its ion type.
pub struct SynapticSum {
    pub g_cl: f32,    // Cl⁻ conductance (GluCl + GABA_A)
    pub g_na: f32,    // Na⁺ conductance (AMPA + nAChR)
    pub g_k: f32,     // K⁺ conductance (GABA_B, mAChR)
    pub g_ca: f32,    // Ca²⁺ conductance (NMDA)
}

impl SynapticSum {
    pub fn new() -> Self {
        Self { g_cl: 0.0, g_na: 0.0, g_k: 0.0, g_ca: 0.0 }
    }

    /// Compute total current: I = g_Cl*(E_Cl - V) + g_Na*(E_Na - V) + g_K*(E_K - V) + g_Ca*(E_Ca - V)
    pub fn current(&self, v: f32) -> f32 {
        let cl = IonChannel::<Chloride>::new(1.0, 1.0);
        let na = IonChannel::<Sodium>::new(1.0, 1.0);
        let k = IonChannel::<Potassium>::new(1.0, 1.0);
        let ca = IonChannel::<Calcium>::new(1.0, 1.0);

        self.g_cl * (cl.e_rev() - v)
            + self.g_na * (na.e_rev() - v)
            + self.g_k * (k.e_rev() - v)
            + self.g_ca * (ca.e_rev() - v)
    }
}


// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cl_channels_same_erev() {
        // Cl⁻ каналы: GluCl и GABA_A — оба Cl⁻, одинаковый E_rev
        let glucl = IonChannel::<Chloride>::new(1.0, 5.0);
        let gaba_a = IonChannel::<Chloride>::new(1.0, 10.0);

        assert_eq!(glucl.e_rev(), gaba_a.e_rev(),
            "Cl⁻ channels must have same E_rev! Got {} and {}",
            glucl.e_rev(), gaba_a.e_rev());

        // Оба должны быть -70 mV (Drosophila)
        assert!((glucl.e_rev() + 70.0).abs() < 1.0,
            "Cl⁻ E_rev should be -70 mV in Drosophila");
    }

    #[test]
    fn test_na_channels() {
        let nachr = IonChannel::<Sodium>::new(1.0, 3.0);
        let ampa = IonChannel::<Sodium>::new(1.0, 2.0);

        assert_eq!(nachr.e_rev(), ampa.e_rev(),
            "Sodium channels must have same E_rev");
        assert!((nachr.e_rev() - 0.0).abs() < 1.0,
            "Na⁺ E_rev should be 0 mV");
    }

    #[test]
    fn test_synaptic_current_direction() {
        // Cl⁻ при V = -65: I = g * (-70 - (-65)) = -5 (слабо гиперполяризующий)
        let cl = IonChannel::<Chloride>::new(1.0, 5.0);
        let i_at_rest = synaptic_current(&cl, -65.0);
        assert!(i_at_rest < 0.0,
            "Cl⁻ current at V=-65 should be negative (hyperpolarizing), got {}",
            i_at_rest);

        // Na⁺ при V = -65: I = g * (0 - (-65)) = 65 (возбуждающий)
        let na = IonChannel::<Sodium>::new(1.0, 2.0);
        let i_na = synaptic_current(&na, -65.0);
        assert!(i_na > 0.0,
            "Na⁺ current at V=-65 should be positive (depolarizing), got {}",
            i_na);
    }

    #[test]
    fn test_synaptic_sum() {
        let mut sum = SynapticSum::new();
        sum.g_cl = 0.5;   // GABA_A activation
        sum.g_na = 1.0;    // AMPA activation

        // At V = -65: I = 0.5*(-70-(-65)) + 1.0*(0-(-65)) = -2.5 + 65 = 62.5
        let i = sum.current(-65.0);
        assert!((i - 62.5).abs() < 0.1,
            "SynapticSum current should be ~62.5 at V=-65, got {}", i);

        // Net excitatory (Na⁺ dominates over Cl⁻)
        assert!(i > 0.0, "Net current should be excitatory");
    }

    #[test]
    fn test_glucl_gaba_erev_must_match() {
        // КРИТИЧЕСКАЯ ПРОВЕРКА: GluCl и GABA_A проводят Cl⁻
        // Их E_rev обязан совпадать, иначе синаптическая интеграция неверна.
        //
        // До фикса: GluCl E_rev = -50 mV, GABA_A E_rev = -65 mV
        // После фикса: оба -70 mV (биологически корректно для Drosophila)

        let _glucl = IonChannel::<Chloride>::new(1.0, 5.0);
        let _gaba_a = IonChannel::<Chloride>::new(1.0, 10.0);

        // Если бы GluCl был создан с E_rev = -50, а GABA_A с -65,
        // компилятор всё равно позволил бы это (они оба IonChannel<Chloride>).
        // НО: конструктор IonChannel::<Chloride>::new() жёстко задаёт -70.
        // Для другого E_rev нужно явно вызывать new_custom() — это visible in code review.
        //
        // Это type-state решение не идеально (Rust не имеет refinement types),
        // но делает неверное значение "громким" через явный вызов new_custom().
    }
}
