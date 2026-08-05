//! # pillars-connectome — pillars-native connectome orchestrator
//!
//! A clean replacement for the legacy `taas-connectome`, which was assembled
//! from several corners (legacy safe kernels + pillars + cuda-kernels). This
//! crate speaks **only** pillars:
//!
//! ```text
//! ConnectomeData  →  SparseGraph (adjacency, the source of truth)
//!        │
//!        ▼
//! Brain<M>  →  Network (NeuralPopulation<M> + delays + spike routing)
//!        │
//!        ▼
//!      step() / inject() / spikes()
//! ```
//!
//! No legacy adapters, no duplicated equations — the engine is pillars itself.

pub mod brain;
pub mod data;
pub mod synaptic;

pub use brain::Brain;
pub use data::{ConnectomeData, NeuronGroup};
pub use synaptic::SynapticBrain;

#[cfg(test)]
mod tests {
    use super::*;
    use pillars::zoo::LIFModel;

    /// A 3-neuron chain 0→1→2: injecting current into neuron 0 must make the
    /// signal propagate all the way to neuron 2 through the CSR graph.
    #[test]
    fn signal_propagates_through_chain() {
        // CSR for 0→1, 1→2 (weights 50, no delay).
        let conn = ConnectomeData::from_csr(
            vec![0u32, 1, 2, 2], // row_ptr (n_neurons = 3)
            vec![1u32, 2],       // col_idx
            vec![50.0f32, 50.0], // weights
            vec![0u8, 0],        // delays
            vec![0u8, 0],        // receptor_types
        );
        assert_eq!(conn.n_neurons(), 3);
        assert_eq!(conn.n_synapses(), 2);

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
        assert!(counts[1] > 0, "direct network: neuron 1 should spike, got {counts:?}");
    }

    /// Group mode: "sensory" (neurons 0,1) → "motor" (neurons 2,3) via edge 0→2.
    #[test]
    fn groups_route_cross_group() {
        let mut conn = ConnectomeData::from_csr(
            vec![0u32, 1, 1, 1, 1], // n = 4: neuron 0 has one outgoing edge
            vec![2u32],             // 0 → 2 (sensory → motor)
            vec![50.0f32],
            vec![0u8],
            vec![0u8],
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
        // receptor type 0 = AMPA (E_rev = 0 mV).
        let conn = ConnectomeData::from_csr(
            vec![0u32, 1, 1], // n = 2: neuron 0 has one outgoing edge
            vec![1u32],       // 0 → 1
            vec![1.0f32],     // conductance weight
            vec![0u8],
            vec![0u8], // AMPA
        );
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

    /// Parity vs the legacy `taas_connectome::sim::sim_lif` on the same chain
    /// 0→1→2. Both run the SAME pillars LIFModel (the legacy `safe::lif_step`
    /// is already a pillars bridge), so the spike counts must agree closely;
    /// the only difference is propagation (legacy queue vs Network ring buffer).
    #[test]
    fn parity_with_legacy_sim_lif() {
        use pillars::zoo::{LIFModelParamIdx as P, LIFModelStateIdx as S};

        // ── Legacy side: i32 CSR + nt_types (+1 excitatory per neuron). ──
        let conn = taas_connectome::ConnectomeData::new(
            vec![0i32, 1, 2, 2],
            vec![1i32, 2],
            vec![50.0, 50.0],
            vec![1.0, 1.0, 1.0],
            vec![],
        );
        // Legacy params (LifParams::new builds the padded 5-col SoA packet).
        let params = taas_connectome::sim::LifParams::new(
            &[1.0, 1.0, 1.0],       // tau_m
            &[-70.0, -70.0, -70.0], // v_rest
            &[-54.0, -54.0, -54.0], // v_thresh
            &[-70.0, -70.0, -70.0], // v_reset
            0.5,                    // dt_ms
            2.0,                    // tau_ref
        );
        let mut state =
            taas_connectome::sim::BrainState::new(3, taas_connectome::sim::NeuronModel::LIF);
        state.v = vec![-70.0; state.v.len()];

        let n_steps = 200;
        let legacy_spikes =
            taas_connectome::sim::sim_lif(&conn, &mut state, &params, &[50.0, 0.0, 0.0], n_steps, 1.0);
        let mut legacy_counts = [0i32; 3];
        for step in 0..n_steps {
            for i in 0..3 {
                legacy_counts[i] += legacy_spikes[step * 3 + i];
            }
        }

        // ── New pillars-native Brain, same graph + same LIF params. ──
        let new_conn = ConnectomeData::from_csr(
            vec![0u32, 1, 2, 2],
            vec![1u32, 2],
            vec![50.0, 50.0],
            vec![0u8, 0],
            vec![0u8, 0],
        );
        let mut brain = Brain::<LIFModel>::from_connectome(&new_conn);
        brain.configure(|p| {
            p.fill_param(P::dt.into(), 0.5);
            p.fill_param(P::inv_tau_m.into(), 1.0); // 1/tau_m
            p.fill_param(P::v_rest.into(), -70.0);
            p.fill_param(P::v_thresh.into(), -54.0);
            p.fill_param(P::v_reset.into(), -70.0);
            p.fill_param(P::tau_ref.into(), 2.0);
            p.fill_state(S::v.into(), -70.0);
        });
        let mut new_counts = [0i32; 3];
        for _ in 0..n_steps {
            brain.inject(&[0], &[50.0]);
            brain.step();
            let s = brain.spikes();
            for i in 0..3 {
                new_counts[i] += s[i];
            }
        }

        eprintln!("legacy={legacy_counts:?} new={new_counts:?}");
        for i in 0..3 {
            let diff = (new_counts[i] - legacy_counts[i]).abs();
            assert!(
                diff <= 3,
                "neuron {i}: new={} legacy={}",
                new_counts[i],
                legacy_counts[i]
            );
        }
    }

    /// Empty connectome (no synapses) — Brain must still build and step.
    #[test]
    fn empty_connectome_steps() {
        let conn = ConnectomeData::from_csr(
            vec![0u32, 0, 0],
            vec![],
            vec![],
            vec![],
            vec![],
        );
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
