//! Parity test — the full network on GPU vs CPU SIMD.
//!
//! Builds the SAME two-population network (LIF → Izhikevich + a recurrent
//! Izh → LIF feedback with axonal delays) on:
//!   1. `pillars::api::Network` (CPU SIMD path), and
//!   2. `pillars_cuda::CompiledNetwork` (GPU, atomic CSR scatter),
//! drives both identically, and compares per-neuron spike counts over the run.
//!
//! The scatter is atomic (non-bit-exact vs the CPU's serialization order), so
//! exact *state* parity is not guaranteed — but with a dominant external drive
//! the discrete spike counts are robust and must match exactly.

use pillars::api::{Edge, Network, SparseGraph};
use pillars::core::population::NeuralPopulation;
use pillars::zoo::Model;
use pillars::zoo::neurons::{IzhikevichModel, IzhikevichModelParamIdx, LIFModel};
use pillars_cuda::{
    CompiledNetwork, IZHIKEVICH_CUDA_SOURCE, LIF_CUDA_SOURCE, NetworkPop,
};

const N: usize = 8; // per population
const DT: f32 = 0.5;
const N_STEPS: usize = 200;
const I_DRIVE: f32 = 30.0;

/// Fill a LIF population with the shared params/state (dt, inv_tau_m, v_rest,
/// v_thresh, v_reset, tau_ref; state v, refr).
fn setup_lif(pop: &mut NeuralPopulation<LIFModel>) {
    pop.fill_param(0, DT);
    pop.fill_param(1, 0.1); // inv_tau_m
    pop.fill_param(2, -70.0); // v_rest
    pop.fill_param(3, -54.0); // v_thresh
    pop.fill_param(4, -70.0); // v_reset
    pop.fill_param(5, 2.0); // tau_ref
    pop.fill_state(0, -70.0); // v
    pop.fill_state(1, 0.0); // refr
}

/// Fill an Izhikevich (2003) population with regular-spiking params.
fn setup_izh(pop: &mut NeuralPopulation<IzhikevichModel>) {
    pop.fill_param(IzhikevichModelParamIdx::dt as usize, DT);
    pop.fill_param(IzhikevichModelParamIdx::a as usize, 0.02);
    pop.fill_param(IzhikevichModelParamIdx::b as usize, 0.2);
    pop.fill_param(IzhikevichModelParamIdx::c as usize, -65.0);
    pop.fill_param(IzhikevichModelParamIdx::d as usize, 8.0);
    pop.fill_param(IzhikevichModelParamIdx::tau_ref as usize, 2.0);
    pop.fill_param(IzhikevichModelParamIdx::v_thresh as usize, 30.0);
    pop.fill_state(0, -65.0); // v
    pop.fill_state(1, -13.0); // u = b·v0
    pop.fill_state(2, 0.0); // refr
}

/// Build the CSR arrays for the 2-pop network (global indices: LIF = 0..8,
/// Izh = 8..16). LIF[i] → Izh[i] (w=0.5, delay 1) + Izh[i] → LIF[i] (w=0.1, d=0).
fn build_csr() -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<u8>) {
    let mut from = Vec::new();
    let mut to = Vec::new();
    let mut w = Vec::new();
    let mut d = Vec::new();
    for i in 0..N as u32 {
        from.push(i);
        to.push(N as u32 + i);
        w.push(0.5);
        d.push(1u8);
        from.push(N as u32 + i);
        to.push(i);
        w.push(0.1);
        d.push(0u8);
    }
    (from, to, w, d)
}

#[test]
fn compiled_network_cpu_vs_gpu_parity() {
    // ── CPU side ───────────────────────────────────────────────────
    let mut net = Network::new();
    let a = net.add_population::<LIFModel>(N);
    let b = net.add_population::<IzhikevichModel>(N);
    net.configure::<LIFModel>(a, setup_lif);
    net.configure::<IzhikevichModel>(b, setup_izh);

    let mut ab = Vec::new();
    let mut ba = Vec::new();
    for i in 0..N as u32 {
        ab.push(Edge::new(i, i, 0.5, 1));
        ba.push(Edge::new(i, i, 0.1, 0));
    }
    net.connect_edges(a, b, &ab);
    net.connect_edges(b, a, &ba);
    net.build();

    let drive: Vec<u32> = (0..N as u32).collect();
    let drive_vals = vec![I_DRIVE; N];

    // ── GPU side ───────────────────────────────────────────────────
    let mut lif_pop = NeuralPopulation::<LIFModel>::new(N);
    setup_lif(&mut lif_pop);
    let mut izh_pop = NeuralPopulation::<IzhikevichModel>::new(N);
    setup_izh(&mut izh_pop);

    let lif_states: Vec<f32> = (0..LIFModel::N_STATE)
        .flat_map(|c| lif_pop.state_slice(c).to_vec())
        .collect();
    let lif_params: Vec<f32> = (0..LIFModel::N_PARAMS)
        .flat_map(|c| lif_pop.param_slice(c).to_vec())
        .collect();
    let izh_states: Vec<f32> = (0..IzhikevichModel::N_STATE)
        .flat_map(|c| izh_pop.state_slice(c).to_vec())
        .collect();
    let izh_params: Vec<f32> = (0..IzhikevichModel::N_PARAMS)
        .flat_map(|c| izh_pop.param_slice(c).to_vec())
        .collect();

    let (from, to, w, d) = build_csr();
    let g = SparseGraph::from_triplets(2 * N, &from, &to, &w, &d);
    let n_syn = g.num_synapses();

    let mut cn = CompiledNetwork::new(
        &[
            NetworkPop {
                source: LIF_CUDA_SOURCE,
                kernel_name: "lif_kernel",
                ns: LIFModel::N_STATE,
                np: LIFModel::N_PARAMS,
                n: N,
                states: lif_states,
                params: lif_params,
            },
            NetworkPop {
                source: IZHIKEVICH_CUDA_SOURCE,
                kernel_name: "izhikevich_kernel",
                ns: IzhikevichModel::N_STATE,
                np: IzhikevichModel::N_PARAMS,
                n: N,
                states: izh_states,
                params: izh_params,
            },
        ],
        g.row_ptr(),
        g.col_idx(),
        &g.weights()[..n_syn],
        g.delays(),
        DT,
    )
    .expect("failed to compile the network (CUDA device required)");

    // ── Run both, one step at a time, comparing spikes ────────────
    let mut cpu_counts = vec![0i64; 2 * N];
    let mut gpu_counts = vec![0i64; 2 * N];
    let mut gpu_spikes = vec![0i32; 2 * N];

    for _ in 0..N_STEPS {
        // CPU
        net.inject_current(a, &drive, &drive_vals);
        net.step();
        for (pop, off) in [(a, 0usize), (b, N)] {
            for (i, &s) in net.spikes(pop).iter().enumerate() {
                cpu_counts[off + i] += s as i64;
            }
        }
        // GPU
        cn.inject_current(0, &drive, &drive_vals).unwrap();
        cn.step().unwrap();
        cn.download_spikes(&mut gpu_spikes).unwrap();
        for (i, &s) in gpu_spikes.iter().enumerate() {
            gpu_counts[i] += s as i64;
        }
    }

    let cpu_total: i64 = cpu_counts.iter().sum();
    assert!(cpu_total > 0, "network must spike under the drive");

    for i in 0..2 * N {
        assert_eq!(
            cpu_counts[i], gpu_counts[i],
            "neuron {i} spike count diverged: CPU={} GPU={} (all CPU={cpu_counts:?}, GPU={gpu_counts:?})",
            cpu_counts[i], gpu_counts[i]
        );
    }
    eprintln!(
        "[parity] compiled_network: total_spikes={cpu_total} over {N_STEPS} steps (CPU ≡ GPU)"
    );
}

/// The network must also run a single homogeneous population with recurrent
/// self-connections (the minimal "projection on GPU" smoke test).
#[test]
fn compiled_network_single_pop_recurrent() {
    let mut pop = NeuralPopulation::<LIFModel>::new(N);
    setup_lif(&mut pop);
    let states: Vec<f32> = (0..LIFModel::N_STATE)
        .flat_map(|c| pop.state_slice(c).to_vec())
        .collect();
    let params: Vec<f32> = (0..LIFModel::N_PARAMS)
        .flat_map(|c| pop.param_slice(c).to_vec())
        .collect();

    // All-to-all recurrent (excluding self-loops), small weights.
    let mut from = Vec::new();
    let mut to = Vec::new();
    let mut w = Vec::new();
    let mut d = Vec::new();
    for i in 0..N as u32 {
        for j in 0..N as u32 {
            if i == j {
                continue;
            }
            from.push(i);
            to.push(j);
            w.push(0.2);
            d.push(1);
        }
    }
    let g = SparseGraph::from_triplets(N, &from, &to, &w, &d);
    let n_syn = g.num_synapses();

    let mut cn = CompiledNetwork::new(
        &[NetworkPop {
            source: LIF_CUDA_SOURCE,
            kernel_name: "lif_kernel",
            ns: LIFModel::N_STATE,
            np: LIFModel::N_PARAMS,
            n: N,
            states,
            params,
        }],
        g.row_ptr(),
        g.col_idx(),
        &g.weights()[..n_syn],
        g.delays(),
        DT,
    )
    .unwrap();

    let drive: Vec<u32> = (0..N as u32).collect();
    let drive_vals = vec![I_DRIVE; N];
    let mut total = 0i64;
    let mut spikes = vec![0i32; N];
    for _ in 0..N_STEPS {
        cn.inject_current(0, &drive, &drive_vals).unwrap();
        cn.step().unwrap();
        cn.download_spikes(&mut spikes).unwrap();
        total += spikes.iter().sum::<i32>() as i64;
    }
    assert!(total > 0, "recurrent LIF network must spike under drive");
}
