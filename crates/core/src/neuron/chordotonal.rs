//! # Chordotonal Sensory Organ (Drosophila mechanoreceptor)
//!
//! Safe Rust wrapper over `kernels::neuron::chordotonal::chordotonal_step`.
//!
//! Models Drosophila chordotonal (scolopidial) mechanoreceptors that sense
//! load and stretch in leg joints. Converts mechanical input → firing rate
//! via receptor potential with adaptation.
//!
//! ## Equations
//! ```text
//! τ_sens · dV/dt = -V + w_load·F + w_stretch·S + V_baseline
//! τ_adapt · dA/dt = -A + b_adapt · rate
//! rate = max_rate · σ(V - A - threshold)
//! σ(x) = 1 / (1 + exp(-x))
//! ```
//!
//! ## Profiles
//! - `phasic()` — rapid adapting (velocity-sensitive), for chordotonal organs
//! - `tonic()` — slow adapting (load-steady), for campaniform sensilla
//! - `mixed()` — phasic-tonic (default), biological chordotonal response
//!
//! ## Usage
//! ```rust
//! use taas_core::neuron::ChordotonalState;
//! let mut sensor = ChordotonalState::new(8);
//! sensor.step(&[1.0; 8], &[0.0; 8], None);
//! // sensor.rates[0] ≈ firing rate in Hz
//! ```

use pillars::zoo::sensors::chordotonal_step;

/// Chordotonal sensory organ state — load/stretch → firing rate.
#[derive(Debug, Clone)]
pub struct ChordotonalState {
    pub n: usize,
    /// Receptor potential (mV), state variable
    pub v_sens: Vec<f32>,
    /// Adaptation variable, state variable
    pub adapt: Vec<f32>,
    /// Output firing rates (Hz)
    pub rates: Vec<f32>,
    /// Per-channel params (9 × n_padded)
    pub params: Vec<f32>,
    n_padded: usize,
    dt: f32,
}

/// Physiological profile for chordotonal sensor tuning.
#[derive(Debug, Clone, Copy)]
pub struct ChordotonalProfile {
    pub tau_sens: f32,
    pub tau_adapt: f32,
    pub b_adapt: f32,
    pub load_gain: f32,
    pub stretch_gain: f32,
    pub baseline: f32,
    pub threshold: f32,
    pub max_rate: f32,
}

impl ChordotonalProfile {
    /// Rapid-adapting phasic sensor — fires at onset of load, then decays.
    /// Typical for chordotonal organs detecting震动/velocity.
    pub const fn phasic() -> Self {
        Self {
            tau_sens: 2.0,
            tau_adapt: 30.0,
            b_adapt: 0.8,
            load_gain: 5.0,
            stretch_gain: 2.0,
            baseline: -3.0,
            threshold: 0.0,
            max_rate: 200.0,
        }
    }

    /// Slow-adapting tonic sensor — steady firing proportional to load.
    /// Typical for campaniform sensilla detecting sustained load.
    pub const fn tonic() -> Self {
        Self {
            tau_sens: 10.0,
            tau_adapt: 500.0,
            b_adapt: 0.1,
            load_gain: 3.0,
            stretch_gain: 1.0,
            baseline: -2.0,
            threshold: 0.0,
            max_rate: 100.0,
        }
    }

    /// Mixed phasic-tonic (biological chordotonal response).
    /// Default for Drosophila leg chordotonal organs.
    pub const fn mixed() -> Self {
        Self {
            tau_sens: 5.0,
            tau_adapt: 100.0,
            b_adapt: 0.3,
            load_gain: 3.0,
            stretch_gain: 1.5,
            baseline: -2.0,
            threshold: 0.0,
            max_rate: 200.0,
        }
    }
}

impl Default for ChordotonalProfile {
    fn default() -> Self {
        Self::mixed()
    }
}

/// Builder for `ChordotonalState`.
#[derive(Debug, Clone)]
pub struct ChordotonalBuilder {
    n: usize,
    profile: ChordotonalProfile,
    dt: f32,
}

impl ChordotonalBuilder {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            profile: ChordotonalProfile::default(),
            dt: 1.0,
        }
    }

    pub fn with_profile(mut self, profile: ChordotonalProfile) -> Self {
        self.profile = profile;
        self
    }

    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    pub fn build(self) -> ChordotonalState {
        let n_padded = self.n.div_ceil(8) * 8;
        let mut params = vec![0.0f32; n_padded * 9];
        let p = &self.profile;

        for i in 0..n_padded {
            params[i] = self.dt;
            params[n_padded + i] = p.tau_sens;
            params[2 * n_padded + i] = p.tau_adapt;
            params[3 * n_padded + i] = p.b_adapt;
            params[4 * n_padded + i] = p.load_gain;
            params[5 * n_padded + i] = p.stretch_gain;
            params[6 * n_padded + i] = p.baseline;
            params[7 * n_padded + i] = p.threshold;
            params[8 * n_padded + i] = p.max_rate;
        }

        ChordotonalState {
            n: self.n,
            v_sens: vec![0.0; n_padded],
            adapt: vec![0.0; n_padded],
            rates: vec![0.0; n_padded],
            params,
            n_padded,
            dt: self.dt,
        }
    }
}

impl ChordotonalState {
    /// Create new chordotonal sensor state with default (mixed) profile.
    pub fn new(n: usize) -> Self {
        ChordotonalBuilder::new(n).build()
    }

    /// Step sensor: convert loads/stretches to firing rates.
    ///
    /// * `loads` — contact forces, length `n` (normalized 0..1)
    /// * `stretches` — joint angle deviations (normalized -1..1)
    /// * `d_loads` — optional velocity (derivative of load), length `n`
    pub fn step(
        &mut self,
        loads: &[f32],
        stretches: &[f32],
        d_loads: Option<&[f32]>,
    ) {
        assert!(loads.len() >= self.n, "loads too short");
        assert!(stretches.len() >= self.n, "stretches too short");

        let dt_old = self.params[0];
        let dt_new = self.dt;
        if (dt_old - dt_new).abs() > 1e-6 {
            for i in 0..self.n_padded {
                self.params[i] = dt_new;
            }
            self.dt = dt_new;
        }

        // Pad inputs to n_padded
        let mut loads_padded = loads.to_vec();
        loads_padded.resize(self.n_padded, 0.0);
        let mut stretches_padded = stretches.to_vec();
        stretches_padded.resize(self.n_padded, 0.0);

        let (_d_loads_padded, d_loads_ptr) = if let Some(dl) = d_loads {
            let mut padded = dl.to_vec();
            padded.resize(self.n_padded, 0.0);
            let ptr = padded.as_ptr();
            (padded, ptr)
        } else {
            (vec![], std::ptr::null())
        };

        unsafe {
            chordotonal_step(
                self.v_sens.as_mut_ptr(),
                self.adapt.as_mut_ptr(),
                self.rates.as_mut_ptr(),
                loads_padded.as_ptr(),
                stretches_padded.as_ptr(),
                d_loads_ptr,
                self.params.as_ptr(),
                self.n_padded,
            );
        }
    }

    /// Get firing rates for the first `n` sensors.
    pub fn firing_rates(&self) -> &[f32] {
        &self.rates[..self.n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chordotonal_state_basic() {
        let mut s = ChordotonalState::new(8);
        let loads = vec![0.5f32; 8];
        let stretches = vec![0.0f32; 8];
        s.step(&loads, &stretches, None);
        assert!(s.rates[0] > 0.0, "Load should produce firing rate");
    }

    #[test]
    fn test_chordotonal_phasic_stronger_than_tonic() {
        let mut phasic = ChordotonalBuilder::new(8)
            .with_profile(ChordotonalProfile::phasic())
            .build();
        let mut tonic = ChordotonalBuilder::new(8)
            .with_profile(ChordotonalProfile::tonic())
            .build();

        let loads = vec![1.0f32; 8];
        let stretches = vec![0.0f32; 8];

        // Step load on: phasic should spike higher initially
        phasic.step(&loads, &stretches, None);
        tonic.step(&loads, &stretches, None);
        let phasic_initial = phasic.rates[0];
        let tonic_initial = tonic.rates[0];

        // After sustained load, tonic should be higher (phasic adapts more)
        for _ in 0..50 {
            phasic.step(&loads, &stretches, None);
            tonic.step(&loads, &stretches, None);
        }
        let phasic_sustained = phasic.rates[0];
        let tonic_sustained = tonic.rates[0];

        // Phasic should drop more than tonic
        let phasic_drop = phasic_initial - phasic_sustained;
        let tonic_drop = tonic_initial - tonic_sustained;
        assert!(phasic_drop > tonic_drop,
            "Phasic should adapt more than tonic");
    }

    #[test]
    fn test_chordotonal_builder_custom() {
        let s = ChordotonalBuilder::new(16)
            .with_profile(ChordotonalProfile::tonic())
            .with_dt(0.5)
            .build();
        assert_eq!(s.n_padded, 16);
        assert!((s.params[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_chordotonal_velocity_response() {
        let mut s = ChordotonalState::new(8);
        let loads = vec![0.0f32; 8];
        let stretches = vec![0.0f32; 8];
        let d_loads = vec![5.0f32; 8]; // rapid force onset

        // With velocity, rate should be higher than without
        s.step(&loads, &stretches, Some(&d_loads));
        let rate_with_vel = s.rates[0];

        let mut s2 = ChordotonalState::new(8);
        s2.step(&loads, &stretches, None);
        let rate_without_vel = s2.rates[0];

        assert!(rate_with_vel > rate_without_vel,
            "Velocity should boost firing rate");
    }
}
