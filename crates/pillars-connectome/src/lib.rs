//! # pillars-connectome — universal connectome brain simulator
//!
//! Единственный connectome-крейт (волна 12: слияние легаси `taas-connectome` и
//! pillars-нативного оркестратора). Движок — 100% pillars:
//!
//! ```text
//! ConnectomeData (i32-CSR + NT types + receptor types + groups)
//!      │
//!      ├── Brain<M>          → pillars::api::Network (spike routing)
//!      ├── SynapticBrain<M>  → pillars::kernels::synapse (conductance kinetics)
//!      └── sim::sim_*        → универсальный API поверх Brain/SynapticBrain
//!      │
//!      ├── BrainRunner       (высокоуровневый конфигурируемый runner)
//!      ├── SpikeRecorder     (rate extraction)
//!      └── rcm              (Reverse Cuthill–McKee reordering)
//! ```
//!
//! Легаси `taas-connectome` удалён; его публичный API сохранён 1:1 здесь
//! (`connectome::` в корневом крейте = `pillars_connectome::`).

// Унаследованные от легаси-крейта соглашения об именах (внешний API).
#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::manual_memcpy)]

pub mod brain;
pub mod data;
pub mod grow_engine;
pub mod hetero;
pub mod propagate;
pub mod rcm;
pub mod record;
pub mod runner;
pub mod sim;
pub mod synaptic;
pub mod types;
pub mod worker;

pub use brain::Brain;
pub use data::{
    BrainIO, ConnectomeData, InputChannel, InputEncoding, NeuronGroup, NtPhenotype, OutputChannel,
    OutputReadout,
};
pub use pillars::kernels::synapse::{DecayFactors, ReceptorType, SynapticKinetics};
pub use rcm::{
    apply_permutation_to_csr, apply_permutation_to_groups, apply_permutation_to_state, bandwidth,
    build_adjacency, fiedler_vector, mean_bandwidth, reverse_cuthill_mckee, spectral_ordering,
};
pub use hetero::{HeteroGroup, HeteroGroupState, HeteroRunner, HeteroSnapshot};
pub use record::{SpikeRecorder, SpikeRecordings};
pub use runner::{BrainConfig, BrainRunner, LearningRule, NeuronConfig, PoissonGroup};
pub use sim::{
    csr_propagate, csr_propagate_conductance_from_queue, csr_propagate_from_queue, haap_population,
    hh_population_pacemaking, hh_population_standard, pad_to, push_spikes_to_queue, sim_gco,
    sim_gco_record, sim_haap_pillars, sim_hh_pillars, sim_hh_synaptic_pillars, sim_izhikevich,
    sim_izhikevich_dual, sim_izhikevich_synaptic, sim_lif, sim_lif_synaptic, sim_rate,
    step_izhikevich_dual_pillars, step_izhikevich_pillars, step_lif_pillars, BrainState, GcoParams,
    IzhikevichDualParams, IzhikevichParams, LifParams, NeuronModel, RateParams, SpikeQueue,
};
pub use synaptic::SynapticBrain;
pub use worker::{AlphaLifOutput, AlphaLifWorker, ShiuParams};

#[cfg(feature = "cuda")]
pub use sim::gpu::{sim_izhikevich_gpu, sim_lif_gpu};

#[cfg(test)]
mod tests {
    use super::*;
    use pillars::zoo::LIFModel;

    /// A 3-neuron chain 0→1→2: injecting current into neuron 0 must make the
    /// signal propagate all the way to neuron 2 through the CSR graph.
    #[test]
    fn signal_propagates_through_chain() {
        // CSR for 0→1, 1→2 (weights 50, no delay).
        let conn = ConnectomeData::new(
            vec![0i32, 1, 2, 2], // row_ptr (n_neurons = 3)
            vec![1i32, 2],       // col_idx
            vec![50.0f32, 50.0], // weights
            vec![],              // nt_types → auto-fill 1.0
            vec![],              // groups
        );
        assert_eq!(conn.n_neurons, 3);
        assert_eq!(conn.n_synapses, 2);

        let mut brain = Brain::<LIFModel>::from_connectome(&conn);
        // Fast neurons (tau = 1 ms) so a single 50-weight pulse crosses the
        // threshold before the leak decays it — makes routing observable.
        // Uses the derive-generated index enums instead of magic columns.
        use pillars::zoo::{LIFModelParamIdx as P, LIFModelStateIdx as S};
        brain.configure(|pop| {
            pop.fill_param(P::dt.into(), 0.5); // dt
            pop.fill_param(P::inv_tau_m.into(), 1.0); // inv_tau_m
            pop.fill_param(P::v_rest.into(), -70.0); // v_rest
            pop.fill_param(P::v_thresh.into(), -54.0); // v_thresh
            pop.fill_param(P::v_reset.into(), -70.0); // v_reset
            pop.fill_param(P::tau_ref.into(), 2.0); // tau_ref
            pop.fill_state(S::v.into(), -70.0); // v
        });

        let mut neuron2_spikes = 0;
        for _ in 0..200 {
            brain.inject(&[0], &[50.0]);
            brain.step();
            neuron2_spikes += brain.spikes()[2];
        }
        assert!(
            neuron2_spikes > 0,
            "signal must propagate 0→1→2; neuron 2 never spiked"
        );
    }

    /// Isolate the Network itself (not my Brain adapter): 2 neurons, edge 0→1.
    #[test]
    fn network_routes_spike_directly() {
        use pillars::api::Network;
        let mut net = Network::new();
        let pop = net.add_population::<LIFModel>(2);
        net.connect(pop, &[0], pop, &[1], &[50.0], &[0]);
        net.build();
        if let Some(p) = net.pop_typed_mut::<LIFModel>(pop) {
            p.fill_param(0, 0.5);
            p.fill_param(1, 1.0); // fast tau so a 50 pulse crosses threshold
            p.fill_param(2, -70.0);
            p.fill_param(3, -54.0);
            p.fill_param(4, -70.0);
            p.fill_param(5, 2.0);
            p.fill_state(0, -70.0);
        }
        let mut counts = [0i32; 2];
        for _ in 0..200 {
            net.inject_current(pop, &[0], &[50.0]);
            net.step();
            let s = net.spikes(pop);
            counts[0] += s[0];
            counts[1] += s[1];
        }
        assert!(
            counts[1] > 0,
            "direct network: neuron 1 should spike, got {counts:?}"
        );
    }

    /// Group mode: "sensory" (neurons 0,1) → "motor" (neurons 2,3) via edge 0→2.
    #[test]
    fn groups_route_cross_group() {
        let mut conn = ConnectomeData::new(
            vec![0i32, 1, 1, 1, 1], // n = 4: neuron 0 has one outgoing edge
            vec![2i32],             // 0 → 2 (sensory → motor)
            vec![50.0f32],
            vec![],
            vec![],
        );
        conn.groups = vec![
            NeuronGroup {
                name: "sensory".into(),
                indices: vec![0, 1],
            },
            NeuronGroup {
                name: "motor".into(),
                indices: vec![2, 3],
            },
        ];

        let mut brain = Brain::<LIFModel>::from_groups(&conn);
        let fast = |p: &mut pillars::NeuralPopulation<LIFModel>| {
            p.fill_param(0, 0.5);
            p.fill_param(1, 1.0); // fast tau
            p.fill_param(2, -70.0);
            p.fill_param(3, -54.0);
            p.fill_param(4, -70.0);
            p.fill_param(5, 2.0);
            p.fill_state(0, -70.0);
        };
        brain.configure_group("sensory", fast);
        brain.configure_group("motor", fast);

        let mut motor_spikes = 0;
        for _ in 0..200 {
            brain.group_inject("sensory", &[0], &[50.0]);
            brain.step();
            motor_spikes += brain.group_spikes("motor")[0];
        }
        assert!(
            motor_spikes > 0,
            "sensory spike must route across groups to motor"
        );
    }

    /// Conductance-based AMPA synapse: neuron 0's spike jumps g_ampa at neuron 1,
    /// whose synaptic current g·(E_rev − V) drives it to spike.
    #[test]
    fn synaptic_ampa_drives_post() {
        // receptor type 0 = AMPA (E_rev = 0 mV) — built from nt_types.
        let mut conn = ConnectomeData::new(
            vec![0i32, 1, 1], // n = 2: neuron 0 has one outgoing edge
            vec![1i32],       // 0 → 1
            vec![1.0f32],     // conductance weight
            vec![1.0, 1.0],   // nt_types (excitatory)
            vec![],
        );
        conn.build_receptor_types(); // nt > 0 → AMPA (0)
        assert_eq!(conn.receptor_types, vec![0u8]);
        let mut brain = SynapticBrain::<LIFModel>::from_connectome(&conn, 0.5);
        brain.configure(|p| {
            p.fill_param(0, 0.5); // dt
            p.fill_param(1, 1.0); // fast tau
            p.fill_param(2, -70.0);
            p.fill_param(3, -54.0);
            p.fill_param(4, -70.0);
            p.fill_param(5, 2.0);
            p.fill_state(0, -70.0);
        });

        let mut neuron1_spikes = 0;
        for _ in 0..200 {
            let s = brain.step(&[50.0, 0.0]);
            neuron1_spikes += s[1];
        }
        assert!(
            neuron1_spikes > 0,
            "AMPA conductance must drive neuron 1 to spike"
        );
    }

    /// The delegated `sim_lif` (pillars-connectome Brain) must reproduce the
    /// legacy parity counts. Measured 40 spikes per neuron when the legacy
    /// `sim_lif` was still the oracle — a regression guard for the delegate.
    #[test]
    fn sim_lif_delegate_matches_legacy_counts() {
        let conn = ConnectomeData::new(
            vec![0i32, 1, 2, 2],
            vec![1i32, 2],
            vec![50.0, 50.0],
            vec![1.0, 1.0, 1.0],
            vec![],
        );
        let params = LifParams::new(
            &[1.0, 1.0, 1.0],       // tau_m
            &[-70.0, -70.0, -70.0], // v_rest
            &[-54.0, -54.0, -54.0], // v_thresh
            &[-70.0, -70.0, -70.0], // v_reset
            0.5,                    // dt_ms
            2.0,                    // tau_ref
        );
        let mut state = BrainState::new(3, NeuronModel::LIF);
        state.v = vec![-70.0; state.v.len()];

        let n_steps = 200;
        let spikes = sim_lif(&conn, &mut state, &params, &[50.0, 0.0, 0.0], n_steps, 1.0);
        let mut counts = [0i32; 3];
        for step in 0..n_steps {
            for i in 0..3 {
                counts[i] += spikes[step * 3 + i];
            }
        }
        assert_eq!(
            counts,
            [40, 40, 40],
            "sim_lif delegate must reproduce the legacy parity counts"
        );

        // Scale-folding lock-in: `scale` is folded into the edge weights. In a
        // subthreshold regime (weights 20 — a single pulse does not cross the
        // threshold) doubling the scale must strictly increase the propagated
        // drive, hence the total spike count.
        let weak = ConnectomeData::new(
            vec![0i32, 1, 2, 2],
            vec![1i32, 2],
            vec![20.0f32, 20.0],
            vec![1.0, 1.0, 1.0],
            vec![],
        );
        let mut s1 = BrainState::new(3, NeuronModel::LIF);
        s1.v = vec![-70.0; s1.v.len()];
        let sp1 = sim_lif(&weak, &mut s1, &params, &[50.0, 0.0, 0.0], n_steps, 1.0);
        let mut s2 = BrainState::new(3, NeuronModel::LIF);
        s2.v = vec![-70.0; s2.v.len()];
        let sp2 = sim_lif(&weak, &mut s2, &params, &[50.0, 0.0, 0.0], n_steps, 2.0);
        let total1: i32 = sp1.iter().sum();
        let total2: i32 = sp2.iter().sum();
        assert!(
            total2 > total1,
            "scale=2.0 must fire more than scale=1.0 (got {total1} vs {total2})"
        );
    }

    /// Izhikevich delegate must genuinely propagate: chain 0→1→2, drive
    /// neuron 0, neuron 2 spikes.
    #[test]
    fn izhikevich_propagates_through_chain() {
        use pillars::zoo::{
            IzhikevichModel, IzhikevichModelParamIdx as P, IzhikevichModelStateIdx as S,
        };

        let conn = ConnectomeData::new(
            vec![0i32, 1, 2, 2],
            vec![1i32, 2],
            vec![50.0f32, 50.0],
            vec![],
            vec![],
        );
        let mut brain = Brain::<IzhikevichModel>::from_connectome(&conn);
        brain.configure(|p| {
            p.fill_param(P::dt.into(), 0.5);
            p.fill_param(P::a.into(), 0.02);
            p.fill_param(P::b.into(), 0.2);
            p.fill_param(P::c.into(), -65.0);
            p.fill_param(P::d.into(), 8.0);
            p.fill_param(P::tau_ref.into(), 2.0);
            p.fill_param(P::v_thresh.into(), 30.0);
            p.fill_state(S::v.into(), -65.0);
            p.fill_state(S::u.into(), 0.0);
            p.fill_state(S::refr.into(), 0.0);
        });

        let mut neuron2_spikes = 0;
        for _ in 0..300 {
            brain.inject(&[0], &[10.0]);
            brain.step();
            neuron2_spikes += brain.spikes()[2];
        }
        assert!(neuron2_spikes > 0, "Izhikevich signal must propagate 0→1→2");
    }

    /// Empty connectome (no synapses) — Brain must still build and step.
    #[test]
    fn empty_connectome_steps() {
        let conn = ConnectomeData::new(vec![0i32, 0, 0], vec![], vec![], vec![], vec![]);
        let mut brain = Brain::<LIFModel>::from_connectome(&conn);
        brain.configure(|pop| {
            pop.fill_param(0, 0.5);
            pop.fill_param(1, 0.1);
            pop.fill_param(2, -70.0);
            pop.fill_param(3, -54.0);
            pop.fill_param(4, -70.0);
            pop.fill_param(5, 2.0);
        });
        for _ in 0..10 {
            brain.step();
        }
        assert_eq!(brain.step_counter(), 10);
        assert_eq!(brain.spikes().len(), 2);
    }
}
