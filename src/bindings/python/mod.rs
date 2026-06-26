//! PyO3 bindings for `taas-checkmate` — zero-copy numpy bindings.
//!
//! Python sub-modules:
//! - `taas_checkmate.neurons` — neuron step functions
//! - `taas_checkmate.t3` — T3 classes + spigrad
//! - `taas_checkmate.io` — IO gates
//! - `taas_checkmate.sparse` — CSR/sparse ops
//! - `taas_checkmate.kuramoto` — Kuramoto functions
//! - `taas_checkmate.snn` — SNN classes
//! - `taas_checkmate.eprop` — E-Prop
//! - `taas_checkmate.connectome` — connectome benchmarks
//! - `taas_checkmate.stp` — Poisson + STP + LFP
//! - `taas_checkmate.rl` — RL agent
//! - `taas_checkmate.koit` — XOR attention + RNG
//! - `taas_checkmate.format` — .taas save/load
//!
//! Root-level aliases (backward compat) maintained for all public functions + classes.

pub mod neuron;
pub mod kuramoto;
pub mod io;
pub mod sparse;
pub mod koit;
pub mod t3;
pub mod snn;
pub mod eprop;
pub mod autotune;
pub mod trainer;
pub mod connectome;
pub mod lif_connectome;
pub mod poisson_stp;
pub mod rl_agent;
pub mod taas_format;

use pyo3::prelude::*;
use pyo3::types::PyCFunction;
use pyo3::wrap_pyfunction;

/// Helper: add a function to a sub-module AND the root module (backward compat).
fn export_fn(m: &Bound<'_, PyModule>, sub: &Bound<'_, PyModule>, func: &Bound<'_, PyCFunction>) -> PyResult<()> {
    sub.add_function(func.clone())?;
    m.add_function(func.clone())?;
    Ok(())
}

/// Create a sub-module and return it.
fn make_sub<'py>(py: Python<'py>, parent: &Bound<'py, PyModule>, name: &str) -> PyResult<Bound<'py, PyModule>> {
    let sub = PyModule::new(py, name)?;
    parent.add_submodule(&sub)?;
    Ok(sub)
}

/// Add a class to both sub-module and root module (backward compat).
macro_rules! export_class {
    ($m:expr, $sub:expr, $cls:ty) => {{
        let m_ref: &Bound<'_, PyModule> = &$m;
        let sub_ref: &Bound<'_, PyModule> = &$sub;
        sub_ref.add_class::<$cls>()?;
        m_ref.add_class::<$cls>()?;
    }};
}

/// taas-checkmate Python module — zero-copy numpy bindings.
#[pymodule]
fn taas_checkmate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ─── neurons ──────────────────────────────────────
    let neurons = make_sub(m.py(), m, "neurons")?;
    macro_rules! add_fn { ($mod:expr, $func:path) => {
        let __f = wrap_pyfunction!($func, &$mod)?;
        export_fn(m, &$mod, &__f)?;
    }}

    add_fn!(neurons, neuron::py_step_izhikevich);
    add_fn!(neurons, neuron::py_step_saw);
    add_fn!(neurons, neuron::py_step_psi_wave);
    add_fn!(neurons, neuron::py_step_lif);
    add_fn!(neurons, neuron::py_step_adex);
    add_fn!(neurons, neuron::py_stdp_update);
    add_fn!(neurons, neuron::py_stdp_homeostatic);

    // ─── kuramoto ──────────────────────────────────────
    let kuramoto = make_sub(m.py(), m, "kuramoto")?;
    add_fn!(kuramoto, kuramoto::py_kuramoto_step_sparse);
    add_fn!(kuramoto, kuramoto::py_t3_step_sparse);
    add_fn!(kuramoto, kuramoto::py_interference);
    add_fn!(kuramoto, kuramoto::py_interference_mean);
    add_fn!(kuramoto, kuramoto::py_batch_coherence_sweep);
    add_fn!(kuramoto, kuramoto::py_phase_coherence);
    add_fn!(kuramoto, kuramoto::py_band_energy);
    add_fn!(kuramoto, kuramoto::py_lazy_sincos_update);
    add_fn!(kuramoto, kuramoto::py_kalman_update);
    add_fn!(kuramoto, kuramoto::py_adam_train);
    add_fn!(kuramoto, kuramoto::py_nlms_train);
    add_fn!(kuramoto, kuramoto::py_pst_decode);
    add_fn!(kuramoto, kuramoto::py_ring_synthesis);
    add_fn!(kuramoto, kuramoto::py_winding_matrix);

    // ─── io ────────────────────────────────────────────
    let io = make_sub(m.py(), m, "io")?;
    add_fn!(io, io::py_safety_clamp);
    add_fn!(io, io::py_sigmoid_gate);
    add_fn!(io, io::py_phase_gate_apply);
    add_fn!(io, io::py_t3_drive_encoder);
    add_fn!(io, io::py_t3_complex_decoder);
    add_fn!(io, io::py_echo_tap);
    add_fn!(io, io::py_hebbian_update);
    add_fn!(io, io::py_stochastic_encoder);
    add_fn!(io, io::py_thalamic_gate);
    add_fn!(io, io::py_t3_inverse_kuramoto);
    add_fn!(io, io::py_t3_pst_features);
    add_fn!(io, io::py_t3_tact_generator);

    // ─── sparse ────────────────────────────────────────
    let sparse = make_sub(m.py(), m, "sparse")?;
    add_fn!(sparse, sparse::py_compute_hamiltonian);
    add_fn!(sparse, sparse::py_step_psi_wave_full);
    add_fn!(sparse, sparse::py_propagate_spikes);
    add_fn!(sparse, sparse::py_sim_step);
    add_fn!(sparse, sparse::py_rate_step);
    add_fn!(sparse, sparse::py_apply_stdp_ltd);

    // ─── koit ──────────────────────────────────────────
    let koit = make_sub(m.py(), m, "koit")?;
    add_fn!(koit, koit::py_xor_attention);
    add_fn!(koit, koit::py_rdrand_fill);

    // ─── t3 ────────────────────────────────────────────
    let t3 = make_sub(m.py(), m, "t3")?;
    export_class!(m, t3, t3::PyTriOscillator);
    export_class!(m, t3, t3::PyNlmsDecoder);
    export_class!(m, t3, t3::PyKalmanDecoder);
    export_class!(m, t3, t3::PyT2SpigradTrajectory);
    export_class!(m, t3, t3::PyPstControl);
    export_class!(m, t3, t3::PySparker);
    export_class!(m, t3, t3::PyHologramEncoder);
    export_class!(m, t3, t3::PyNormalizedSparker);
    add_fn!(t3, t3::py_pst_features);
    add_fn!(t3, t3::py_pst_linear_features);
    add_fn!(t3, t3::py_pst_linear_dim);
    add_fn!(t3, t3::py_compute_pearson);
    add_fn!(t3, t3::py_normalize_features);
    add_fn!(t3, t3::py_t2spigrad_forward);
    add_fn!(t3, t3::py_t2spigrad_backward);
    #[cfg(feature = "cuda")]
    add_fn!(t3, t3::py_t2spigrad_forward_cuda);
    #[cfg(feature = "cuda")]
    add_fn!(t3, t3::py_t2spigrad_backward_cuda);

    // ─── lif_connectome ─────────────────────────────────
    let lif_connectome = make_sub(m.py(), m, "lif_connectome")?;
    add_fn!(lif_connectome, lif_connectome::py_lif_sim_cpu_recording);
    add_fn!(lif_connectome, lif_connectome::py_lif_sim_gpu_recording);

    // ─── snn ───────────────────────────────────────────
    let snn = make_sub(m.py(), m, "snn")?;
    export_class!(m, snn, snn::PyLifNeuron);
    export_class!(m, snn, snn::PySornNetwork);
    export_class!(m, snn, snn::PyTtfsEncoder);
    add_fn!(snn, snn::py_intrinsic_plasticity);
    add_fn!(snn, snn::py_synaptic_normalization);
    add_fn!(snn, snn::py_bsa_encode);
    add_fn!(snn, snn::py_phase_encode);
    add_fn!(snn, snn::py_bsa_kernel_hamming);
    add_fn!(snn, snn::py_population_rate_decode);
    add_fn!(snn, snn::py_spike_raster_to_csv);

    // ─── eprop ─────────────────────────────────────────
    let eprop = make_sub(m.py(), m, "eprop")?;
    export_class!(m, eprop, eprop::PyEPropLIF);
    export_class!(m, eprop, eprop::PyEPropIzhikevichMulti);
    add_fn!(eprop, eprop::py_surrogate_gradient);
    add_fn!(eprop, eprop::py_eprop_update);

    // ─── autotune ──────────────────────────────────────
    let autotune = make_sub(m.py(), m, "autotune")?;
    add_fn!(autotune, autotune::probe_eprop_fr);
    add_fn!(autotune, autotune::suggest_eprop_config);

    // ─── trainer ───────────────────────────────────────
    let trainer = make_sub(m.py(), m, "trainer")?;
    export_class!(m, trainer, trainer::PyEPropTrainer);

    // ─── connectome ────────────────────────────────────
    let connectome = make_sub(m.py(), m, "connectome")?;
    add_fn!(connectome, connectome::py_bench_connectome_cpu);
    add_fn!(connectome, connectome::py_bench_connectome_cpu_recording);
    add_fn!(connectome, connectome::py_bench_connectome_gpu);
    add_fn!(connectome, connectome::py_bench_connectome_gpu_recording);
    add_fn!(connectome, connectome::py_bench_connectome_gpu_stdp);
    #[cfg(feature = "cuda")]
    add_fn!(connectome, connectome::py_bench_psi_wave_gpu);
    add_fn!(connectome, connectome::py_bench_psi_aberton_gpu);

    // ─── stp ───────────────────────────────────────────
    let stp = make_sub(m.py(), m, "stp")?;
    export_class!(m, stp, poisson_stp::PyPoissonGenerator);
    export_class!(m, stp, poisson_stp::PyShortTermPlasticity);
    add_fn!(stp, poisson_stp::py_compute_lfp);

    // ─── rl ────────────────────────────────────────────
    let rl = make_sub(m.py(), m, "rl")?;
    export_class!(m, rl, rl_agent::PyActorCriticAgent);

    // ─── format ────────────────────────────────────────
    let format = make_sub(m.py(), m, "format")?;
    add_fn!(format, taas_format::py_save_taas);
    add_fn!(format, taas_format::py_load_taas);
    add_fn!(format, taas_format::py_inspect_taas);

    Ok(())
}
