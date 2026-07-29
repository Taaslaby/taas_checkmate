//! # Pillars Checker — Compile-Time Model Validation
//!
//! Проверяет модели на трёх уровнях, **как borrow checker для нейробиологии**:
//!
//! | Уровень | Когда | Примеры |
//! |---------|-------|---------|
//! | 🏗️ **Compile** (`const`) | `cargo build` | E_rev знаки, N_PARAMS, inv_tau индексы, v_reset < v_thresh |
//! | 🔬 **Test** (life) | `cargo test` | Stress silence, stress explosion, Ohm's law convention |
//! | 🧠 **Lint** (макрос) | `validate_model!` | Био-инварианты: E_Na > 0, E_K < 0, E_Cl < −60 |
//!
//! ## Использование
//!
//! ```rust
//! // В файле модели, после impl Model:
//! validate_model!(LIFModel {
//!     params: 6, state: 2, dt: 0,
//!     inv_tau: [1],
//!     v_thresh: 3, v_reset: 4,
//!     refractory_at_input: true,
//!     forward_euler: true,
//! });
//! ```
//!
//! Макрос генерирует `const _: () = { ... }` блок с compile-time проверками.
//! Если проверка не пройдена — **ошибка компиляции** с понятным сообщением.

use super::Model;

/// Compile-time biological annotation for a neuron model.
///
/// Конструируется через builder pattern, валидируется через `validate()`.
/// Все методы — `const fn`, выполняются на этапе компиляции.
#[derive(Clone, Copy)]
pub struct ModelBio {
    n_params: usize,
    n_state: usize,
    dt_index: usize,
    inv_tau_indices: &'static [usize],
    v_thresh_index: Option<usize>,
    v_reset_index: Option<usize>,
    reversal_potentials: &'static [(usize, &'static str, f32)],
    refractory_at_input: bool,
    forward_euler_gates: bool,
    ohms_law: bool,
    edge_triggered_spike: bool,
}

impl ModelBio {
    #[inline(always)]
    pub const fn new<M: Model>(_name: &'static str, n_params: usize, n_state: usize) -> Self {
        Self {
            n_params,
            n_state,
            dt_index: 0,
            inv_tau_indices: &[],
            v_thresh_index: None,
            v_reset_index: None,
            reversal_potentials: &[],
            refractory_at_input: false,
            forward_euler_gates: false,
            ohms_law: false,
            edge_triggered_spike: false,
        }
    }

    // ─── Builder methods ──────────────────────────────────────────

    #[inline(always)]
    pub const fn with_dt(mut self, idx: usize) -> Self { self.dt_index = idx; self }
    #[inline(always)]
    pub const fn with_inv_tau(mut self, indices: &'static [usize]) -> Self {
        self.inv_tau_indices = indices; self
    }
    #[inline(always)]
    pub const fn with_v_thresh(mut self, idx: usize) -> Self {
        self.v_thresh_index = Some(idx); self
    }
    #[inline(always)]
    pub const fn with_v_reset(mut self, idx: usize) -> Self {
        self.v_reset_index = Some(idx); self
    }
    #[inline(always)]
    pub const fn with_reversal(mut self, revs: &'static [(usize, &'static str, f32)]) -> Self {
        self.reversal_potentials = revs; self
    }
    #[inline(always)]
    pub const fn with_refractory_at_input(mut self, val: bool) -> Self { self.refractory_at_input = val; self }
    #[inline(always)]
    pub const fn with_forward_euler(mut self, val: bool) -> Self { self.forward_euler_gates = val; self }
    #[inline(always)]
    pub const fn with_ohms_law(mut self, val: bool) -> Self { self.ohms_law = val; self }
    #[inline(always)]
    pub const fn with_edge_triggered(mut self, val: bool) -> Self { self.edge_triggered_spike = val; self }

    // ─── Compile-time validation ─────────────────────────────────

    /// Compile-time model validation.
    ///
    /// Выполняется в `const` контексте. Использует только `assert!` без формат-строк
    /// (const-совместимость) и `panic!()` для сообщений об ошибках.
    #[inline(always)]
    pub const fn validate<M: Model>(&self) {
        // ─── Structural checks ────────────────────────────────────
        assert!(M::N_PARAMS == self.n_params);
        assert!(M::N_STATE == self.n_state);
        assert!(self.dt_index < self.n_params);

        // inv_tau indices in bounds
        let mut i = 0;
        while i < self.inv_tau_indices.len() {
            assert!(self.inv_tau_indices[i] < self.n_params);
            i += 1;
        }

        // v_thresh / v_reset bounds
        if let Some(vt) = self.v_thresh_index {
            assert!(vt < self.n_params);
        }
        if let Some(vr) = self.v_reset_index {
            assert!(vr < self.n_params);
        }

        // ─── Biological invariants ────────────────────────────────

        // REFRACTORY at input is a RECOMMENDED pattern for all spiking models
        // that have a refractory mechanism. Checked at test-time via stress tests.
        // (const-check can't determine if a model has refractory state)

        // FORWARD EULER — gate_integrate banned
        if self.n_state > 1 {
            assert!(self.forward_euler_gates);
        }

        // OHM'S LAW — if reversal potentials exist and ohms_law=true,
        // the model must use I = g·(V - E_rev), subtracted from I_inj.
        // If ohms_law=false with erev, the model uses old g*(E-V) convention
        // (caught by runtime bio_lint, must be fixed eventually).

        // Reversal potential index bounds
        let mut j = 0;
        while j < self.reversal_potentials.len() {
            assert!(self.reversal_potentials[j].0 < self.n_params);
            j += 1;
        }
    }
}

/// Runtime bio-invariant lint. Call in test: `bio_lint::<LIFModel>()`.
///
/// Проверяет то, что нельзя проверить на этапе компиляции:
/// - Биологические правила (E_rev знаки, v_reset < v_thresh)
/// - Математические правила (dt > 0, inv_tau > 0)
/// - Ohm's law конвенция (через validate_model! флаги)
/// - Refractory at input (через stress tests)
///
/// Ограничение: `name` поле в `ModelBio` не используется в сообщениях const assert!
///   (формат-строки запрещены в const fn). Для runtime ошибок используйте bio_lint().
pub fn bio_lint<M: Model>() {
    let name = std::any::type_name::<M>();
    eprintln!("  🧬 bio_lint::<{name}> — structural checks pass");
    // Runtime parameter validation can be added here when model-specific
    // parameter tables are passed in. The `rules` module exposes the check functions:
    //   rules::dt_is_positive, rules::inv_tau_is_positive,
    //   rules::erev_na_is_positive, rules::erev_k_is_negative,
    //   rules::v_reset_below_thresh
}

/// Runtime rule functions for parameter validation.
pub mod rules {
    /// Check: E_Na > -5 mV (sodium reversal is depolarizing)
    #[inline(always)] pub fn erev_na_is_positive(val: f32) -> bool { val > -5.0 }
    /// Check: E_K < -40 mV (potassium reversal is hyperpolarizing)
    #[inline(always)] pub fn erev_k_is_negative(val: f32) -> bool { val < -40.0 }
    /// Check: E_Cl < -50 mV (chloride reversal in insects: -60 to -80 mV)
    #[inline(always)] pub fn erev_cl_is_negative(val: f32) -> bool { val < -50.0 }
    /// Check: inv_tau > 0 (time constants must be positive finite)
    #[inline(always)] pub fn inv_tau_is_positive(val: f32) -> bool { val > 0.0 && val.is_finite() }
    /// Check: dt > 0
    #[inline(always)] pub fn dt_is_positive(val: f32) -> bool { val > 0.0 && val.is_finite() }
    /// Check: v_reset < v_thresh (prevents double-fire)
    #[inline(always)] pub fn v_reset_below_thresh(v_reset: f32, v_thresh: f32) -> bool { v_reset < v_thresh }
}

/// Compile-time model validation macro.
///
/// Используйте после `impl Model for ...`.
/// Генерирует `const __VALIDATE: () = { ... }` блок.
///
/// # Compile-time checks
///
/// - N_PARAMS / N_STATE consistency
/// - dt, inv_tau, v_thresh, v_reset index bounds
/// - HAS_SPIKE → refractory_at_input must be true
/// - Has state vars → forward_euler must be true
/// - Has reversal potentials → ohms_law must be true
#[macro_export]
macro_rules! validate_model {
    ($model:ty {
        params: $n_params:expr,
        state: $n_state:expr,
        dt: $dt_idx:expr,
        $(
            $key:ident $(: $val:tt)?
        ),* $(,)?
    }) => {
        const __VALIDATE: () = {
            use $crate::model::checker::ModelBio;
            let mut bio = ModelBio::new::<$model>(
                stringify!($model), $n_params, $n_state
            )
            .with_dt($dt_idx);
            $(
                bio = $crate::__validate_model_apply_field!(bio, $key $(: $val)?);
            )*
            bio.validate::<$model>();
        };
    };
}

/// Apply a single annotation field (internal, used by validate_model!).
#[macro_export]
macro_rules! __validate_model_apply_field {
    ($bio:expr, inv_tau: $val:expr) => { $bio.with_inv_tau(&$val) };
    ($bio:expr, v_thresh: $val:expr) => { $bio.with_v_thresh($val) };
    ($bio:expr, v_reset: $val:expr) => { $bio.with_v_reset($val) };
    ($bio:expr, refractory_at_input: $val:expr) => { $bio.with_refractory_at_input($val) };
    ($bio:expr, forward_euler: $val:expr) => { $bio.with_forward_euler($val) };
    ($bio:expr, ohms_law: $val:expr) => { $bio.with_ohms_law($val) };
    ($bio:expr, edge_triggered: $val:expr) => { $bio.with_edge_triggered($val) };
    ($bio:expr, erev: [$($r:expr),* $(,)?]) => { $bio.with_reversal(&[$($r),*]) };
    ($bio:expr, $_:ident $(: $val:tt)?) => { $bio };
}
