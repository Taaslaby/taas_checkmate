#![allow(non_snake_case)] // I_soma, E_leak, V_thresh — физическая нотация

//! -------------------------------------------------------------------------
//! Filename:    src/neurons/mod.rs
//! Class:       neurons
//! Purpose:     Stateful Neuron Populations
//! Description: Высокоуровневые stateful-классы поверх run_neuron/kernels/ops —
//!              единый источник правды для API и python. Каждый класс
//!              делегирует в pillars-ядро (run_neuron::<Model>, kernels::eprop,
//!              ops::plasticity::stp, kernels::dendrite, zoo::sensors::chordotonal).
//!              Классы: EPropLIF (kernels::eprop) | Glif3/4 (GLIF3/4Model) |
//!              ShortTermPlasticity (stp) | SomatoDendriticNeuron (LIF+cable) |
//!              ChordotonalState (mechanoreceptor) | реэкспорты: PoissonGenerator,
//!              SornNetwork, AstrocyteState, TtfsEncoder, SpikeRaster и др.
//!              Подмодули: eprop, haap, neuromotor, bcm, hr, lif, izhikevich, saw, adex, fhn, wc.
//!
//! author: Atlasov Anton
//! -------------------------------------------------------------------------

// ─── Подмодули ─────────────────────────────────────────────

pub mod adex;
pub mod bcm;
pub mod eprop;
pub mod fhn;
pub mod haap;
pub mod hr;
pub mod izhikevich;
pub mod lif;
pub mod neuromotor;
pub mod saw;
pub mod wc;

pub use adex::step_adex;
pub use bcm::{BcmSynapse, HomeostaticScale};
pub use eprop::{EPropAdExMulti, EPropIzhikevich, EPropIzhikevichMulti, EPropSaw};
pub use fhn::step_fhn;
pub use haap::{step_haap, HaapProfile, HaapState, HaapStateBuilder};
pub use hr::step_hr;
pub use izhikevich::{step_izhikevich, step_izhikevich_dual};
pub use lif::{step_lif, step_lif_at};
pub use neuromotor::{
    step_neuromotor, BiomechanicsProfile, DrosophilaLegConfig, FiberPoolConfig, NeuromotorState,
    NeuromotorStateBuilder,
};
pub use saw::{step_gco, step_saw};
pub use wc::step_wc;

// ─── Реэкспорты kernels-канона ─────────────────────────────

pub use crate::api::autotune::{probe_fr, suggest_config, EPropTunerConfig, FrProbePoint};
pub use crate::kernels::astrocyte::AstrocyteState;
pub use crate::kernels::dendrite::{CableDendriteState, DendriteConfig, DendriteDecayFactors};
pub use crate::kernels::input::{
    bsa_encode, bsa_kernel_hamming, phase_encode, PoissonGenerator, TtfsEncoder,
};
pub use crate::kernels::learning::{
    eprop_update, intrinsic_plasticity, stdp_homeostatic, stdp_update, surrogate_gradient,
    synaptic_normalization, SornNetwork,
};
pub use crate::kernels::stats::{
    compute_cvs, compute_isi, firing_rates, mean_firing_rate, population_rate_decode,
    spike_raster_to_csv, SpikeRaster,
};

// ─── Константы ─────────────────────────────────────────────

/// Default refractory period (ms).
/// Shiu et al. 2024: 2.2ms for Drosophila LIF.
pub const DEFAULT_TAU_REF: f32 = 2.2;

mod chordotonal;
mod delay;
mod dopamine;
mod eprop_lif;
mod glif;
mod somato_dendritic;
mod stp;

pub use chordotonal::{ChordotonalBuilder, ChordotonalProfile, ChordotonalState};
pub use delay::DelayBuffer;
pub use dopamine::DopamineSTDP;
pub use eprop_lif::EPropLIF;
pub use glif::{Glif3State, Glif4State};
pub use somato_dendritic::SomatoDendriticNeuron;
pub use stp::{compute_lfp, ShortTermPlasticity};

// ═══════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glif3_state_basic() {
        let mut state = Glif3State::new(8, 1.0, 0.1, -65.0, -50.0, -65.0, 30.0, 0.01, -80.0, 2.0);
        let i_ext = vec![30.0f32; 8];

        let mut total = 0i32;
        for _ in 0..300 {
            state.step(Some(&i_ext));
            total += state.spikes.iter().sum::<i32>();
        }

        assert!(total > 0, "GLIF3 state: should fire with I=30");
        assert!(state.g_ahp[0] > 0.0, "GLIF3 state: g_ahp should accumulate");
    }

    #[test]
    fn glif4_state_basic() {
        let mut state = Glif4State::new(
            8, 1.0, 0.1, -65.0, -50.0, -65.0, 30.0, 0.01, -80.0, 20.0, 3.0, 2.0,
        );
        let i_ext = vec![30.0f32; 8];

        let mut total = 0i32;
        for _ in 0..300 {
            state.step(Some(&i_ext));
            total += state.spikes.iter().sum::<i32>();
        }

        assert!(total > 0, "GLIF4 state: should fire with I=30");
        assert!(state.theta[0] > 0.0, "GLIF4 state: theta should accumulate");
        assert!(state.g_ahp[0] > 0.0, "GLIF4 state: g_ahp should accumulate");
    }

    #[test]
    fn glif3_state_reset() {
        let mut state = Glif3State::new(8, 1.0, 0.1, -65.0, -50.0, -65.0, 30.0, 0.01, -80.0, 2.0);
        let i_ext = vec![30.0f32; 8];

        for _ in 0..100 {
            state.step(Some(&i_ext));
        }
        assert!(state.g_ahp[0] > 0.0);

        state.reset();
        assert_eq!(state.v[0], -65.0);
        assert_eq!(state.g_ahp[0], 0.0);
        assert_eq!(state.refractory[0], 0.0);
    }

    #[test]
    fn eprop_lif_fires_with_input() {
        let mut pop = EPropLIF::new(8, 1.0, 10.0, 20.0);
        let i_ext = vec![30.0f32; 8];
        let mut spikes = vec![0i32; 8];
        let pre = vec![1i32; 8];

        let mut total = 0usize;
        for _ in 0..200 {
            total += pop.step(&i_ext, &mut spikes, &pre);
        }
        assert!(total > 0, "EPropLIF should fire with I=30");
        assert!(
            pop.eligibility.iter().any(|&e| e > 0.0),
            "eligibility should accumulate"
        );
    }

    #[test]
    fn eprop_update_weights() {
        let pop = EPropLIF::new(8, 1.0, 10.0, 20.0);
        let mut w = vec![0.5f32; 8];
        let l = vec![1.0f32; 8];
        pop.update_weights(&mut w, &l, 0.1);
        assert!((w[0] - 0.5).abs() < 1e-6, "no eligibility → no change");
    }

    #[test]
    fn test_soma_dendrite_ampa_drives_spike() {
        let config = DendriteConfig {
            n_compartments: 4,
            R_a: 10.0,
            R_m: 100.0,
            C_m: 0.1,
        };
        let mut pop = SomatoDendriticNeuron::new(8, &config, 0.5, 20.0, -65.0, -54.0, -65.0, 3.0);
        pop.set_lif_params(0, 20.0, -65.0, -60.0, -65.0);
        let i_ext = vec![0.0f32; 8];

        let mut spike_count = 0i32;
        for _ in 0..300 {
            pop.jump_ampa(0, 0, 0.20);
            pop.step(&i_ext, 0.1);
            spike_count += pop.spikes[0];
        }
        assert!(
            spike_count > 0,
            "Нейрон 0 должен спайковать от дендритного AMPA, got {spike_count}"
        );
        assert_eq!(pop.spikes[1], 0, "Нейрон 1 без входа не должен спайковать");
    }

    #[test]
    fn test_chordotonal_state_basic() {
        let mut s = ChordotonalState::new(8);
        let loads = vec![0.5f32; 8];
        let stretches = vec![0.0f32; 8];
        s.step(&loads, &stretches, None);
        assert!(s.rates[0] > 0.0, "Load should produce firing rate");
    }

    #[test]
    fn test_stp_step_updates_state() {
        let mut stp = ShortTermPlasticity::new(8, 1.0);
        let pre = vec![1i32; 8];
        let mut out = vec![0.0f32; 8];
        stp.step(&pre, &mut out);
        assert!(out[0] > 0.0, "spike should produce current");
        assert!(stp.x[0] < 1.0, "x should deplete after spike");
    }

    #[test]
    fn dopamine_stdp_basic() {
        let mut stdp = DopamineSTDP::new(8, 1.0);
        // Co-activity with simultaneous pre+post: ltd (0.12) > ltp (0.1) →
        // отрицательный eligibility — канон dopamine_traces_step.
        let pre = vec![1i32; 8];
        let post = vec![1i32; 8];
        stdp.step(&pre, &post);
        assert!(
            stdp.eligibility[0] != 0.0,
            "co-activity should create eligibility"
        );

        let mut w = vec![0.5f32; 8];
        stdp.apply_dopamine(&mut w, 1.0, 0.1);
        assert!(w[0] != 0.5, "dopamine should update weight");
    }

    #[test]
    fn delay_buffer_push_read() {
        let mut buf = DelayBuffer::new(8);
        buf.push(&[1.0f32; 8]);
        let delayed = buf.read(2);
        assert!(
            (delayed[0] - 1.0).abs() < 1e-6,
            "read(2) should return pushed data"
        );
    }


    #[test]
    fn delay_buffer_latest_matches_read1() {
        let mut buf = DelayBuffer::new(4);
        buf.push(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(buf.latest(), buf.read(1)[0]);
        buf.push(&[9.0, 8.0, 7.0, 6.0]);
        assert_eq!(buf.latest(), buf.read(1)[0]);
    }

    #[test]
    fn ttfs_encoder_basic() {
        let mut enc = TtfsEncoder::new(4, 10);
        let mut spikes = vec![0i32; 4];
        assert!(
            enc.encode_step(&[1.0f32; 4], &mut spikes),
            "value=1 → fire at step 0"
        );
        assert_eq!(spikes[0], 1);
    }

    #[test]
    fn spike_raster_record() {
        let mut raster = SpikeRaster::new();
        let spikes = [1i32, 0, 0, 1];
        raster.record(0, &spikes, 4);
        let (dense, steps) = raster.to_dense(4);
        assert_eq!(steps, 1);
        assert_eq!(dense[0], 1);
        assert_eq!(dense[3], 1);
        assert_eq!(raster.total_spikes(), 2);
    }
}
