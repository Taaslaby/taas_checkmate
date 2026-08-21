//! # PyO3 bindings for `taas-checkmate` — zero-copy numpy.
//!
//! ## Recommended modern API — `taas_checkmate.pillars`
//!
//! The pillars-native SIMD engine (the architecture everything is moving to):
//! ```python
//! from taas_checkmate.pillars import LIFPopulation, IzhikevichPopulation, \
//!     HHPopulation, VSNPopulation, RatePopulation
//! pop = LIFPopulation(1000)
//! pop.dt = 0.5; pop.v_thresh = -54.0
//! pop.step(i_in, spikes)   # zero-copy numpy
//! ```
//! These are the **point-neuron populations** — prefer them for new code.
//!
//! ## Namespaced sub-modules (kept for specific use)
//! | Sub-module | Contents |
//! |------------|----------|
//! | `pillars` | **единый фасад**: Population-классы + все функции/классы подмодулей|
//! | `neurons` | raw per-model step functions (legacy low-level) |
//! | `connectome` / `connectome_brain` / `lif_connectome` | connectome data + brain sim |
//! | `t3` | T3 oscillators + decoders + spigrad |
//! | `kuramoto` | Kuramoto / phase / interference functions |
//! | `snn` / `neuron` / `eprop` / `poisson_stp` | legacy SNN classes + learning |
//! | `karmac` | cognitive architecture (GNW, BasalGanglia, MinBrain) |
//! | `spaun` / `spaun_gpu` | SPAUN pipeline |
//! | `trainer` / `autotune` / `rl_agent` | training + RL |
//! | `taas_format` | .taas save/load |
//! | `hh` / `hh_gpu` | HH bindings |
//! | `io` / `sparse` / `hologram` / `gpu_refiner` / `vnc` / `cpe` | utilities |
//!
//! ## Backward compatibility
//! Every public function/class is also aliased at the root
//! (`taas_checkmate.py_step_lif`, `taas_checkmate.LIFPopulation`, …) so old
//! scripts keep working. New code should import from the sub-modules.

pub mod autotune;
pub mod connectome;
pub mod connectome_brain;
pub mod cpe;
pub mod eprop;
pub mod formula;
pub mod hh;
#[cfg(feature = "cuda")]
pub mod hh_gpu;
pub mod io;
pub mod karmac;
pub mod kuramoto;
pub mod lif_connectome;
pub mod neuroml;
pub mod neuron;
pub mod pillars;
pub mod poisson_stp;
pub mod rl_agent;
pub mod snn;
pub mod sparse;
pub mod t3;
pub mod taas_format;
pub mod trainer;
pub mod vnc;
// pub mod iit;
#[cfg(feature = "cuda")]
pub mod cuda_kernel;
#[cfg(feature = "cuda")]
pub mod gpu_refiner;
pub mod hologram;
pub mod spaun;
#[cfg(feature = "cuda")]
pub mod spaun_gpu;

use pyo3::prelude::*;
use pyo3::types::PyCFunction;
use pyo3::wrap_pyfunction;

/// Helper: add a function to a sub-module AND the root module (backward compat).
fn export_fn(
    m: &Bound<'_, PyModule>,
    sub: &Bound<'_, PyModule>,
    func: &Bound<'_, PyCFunction>,
) -> PyResult<()> {
    sub.add_function(func.clone())?;
    m.add_function(func.clone())?;
    Ok(())
}

/// Helper: add a function ONLY to the facade sub-module (pillars) — no root
/// re-export (эти функции уже на root через свои подмодули).
fn export_fn_facade(sub: &Bound<'_, PyModule>, func: &Bound<'_, PyCFunction>) -> PyResult<()> {
    sub.add_function(func.clone())?;
    Ok(())
}

/// Create a sub-module and return it.
///
/// Registers the sub-module in `sys.modules` under its *public* name so both
/// `taas_checkmate.pillars.X` (attribute) and `from taas_checkmate.pillars import X`
/// (import machinery) work. Without this, pyo3 sub-modules are only reachable as
/// attributes and `from ... import` raises ModuleNotFoundError.
///
/// The public root is the package the user imports. In a normal wheel the
/// extension's `__name__` is `taas_checkmate`; under an editable install the
/// shim loads it as `taas_checkmate.taas_checkmate`, so we strip that suffix.
fn make_sub<'py>(
    py: Python<'py>,
    parent: &Bound<'py, PyModule>,
    name: &str,
) -> PyResult<Bound<'py, PyModule>> {
    let sub = PyModule::new(py, name)?;
    let parent_name = parent.name()?;
    let parent_name = parent_name.to_str()?;
    // extension module loaded as <pkg>.<ext>? strip the <ext> suffix
    let public_root = match parent_name.rsplit_once('.') {
        Some((root, "taas_checkmate")) => root,
        _ => parent_name,
    };
    let fq = format!("{public_root}.{name}");
    let sysmod = py.import("sys")?;
    sub.setattr("__name__", fq.clone())?;
    // Атрибут на родителе: `taas_checkmate.autotune`, `taas_checkmate.neurons`,
    // `taas_checkmate.kuramoto` … (add_submodule регистрирует только в sys.modules
    // и НЕ вешает атрибут — без этого `tcm.autotune` падал с AttributeError).
    parent.add(name, &sub)?;
    parent.add_submodule(&sub)?;
    sysmod.getattr("modules")?.set_item(&fq, sub.clone())?;
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
    macro_rules! add_fn {
        ($mod:expr, $func:path) => {
            let __f = wrap_pyfunction!($func, &$mod)?;
            export_fn(m, &$mod, &__f)?;
        };
    }

    add_fn!(neurons, neuron::py_step_izhikevich);
    add_fn!(neurons, neuron::py_step_saw);
    add_fn!(neurons, neuron::py_step_gco);
    add_fn!(neurons, neuron::py_step_lif);
    add_fn!(neurons, neuron::py_step_adex);
    add_fn!(neurons, neuron::py_step_fhn);
    add_fn!(neurons, neuron::py_step_haap);
    add_fn!(neurons, neuron::py_step_nmda);
    add_fn!(neurons, neuron::py_step_hr);
    add_fn!(neurons, neuron::py_step_wc);
    add_fn!(neurons, neuron::py_step_neuromotor);
    add_fn!(neurons, neuron::py_stdp_update);
    add_fn!(neurons, neuron::py_stdp_homeostatic);
    add_fn!(neurons, neuron::py_firing_rates);
    add_fn!(neurons, neuron::py_mean_firing_rate);
    add_fn!(neurons, neuron::py_compute_isi);
    add_fn!(neurons, neuron::py_compute_cvs);
    add_fn!(neurons, neuron::step_eprop);
    export_class!(m, neurons, neuron::PySomatoDendriticNeuron);
    export_class!(m, neurons, neuron::PyGlif3State);
    export_class!(m, neurons, neuron::PyGlif4State);
    export_class!(m, neurons, neuron::PyChordotonalSensor); // 🦗 механорецептор
                                                            // ─── neuroml ──────────────────────────────────────
    let neuroml_sub = make_sub(m.py(), m, "neuroml")?;
    let f_load = wrap_pyfunction!(neuroml::py_load_neuroml, &neuroml_sub)?;
    export_fn_facade(&neuroml_sub, &f_load)?;
    let f_load_file = wrap_pyfunction!(neuroml::py_load_neuroml_file, &neuroml_sub)?;
    export_fn_facade(&neuroml_sub, &f_load_file)?;
    neuroml_sub.add_class::<neuroml::NeuromlNetwork>()?;

    // ─── karmac ────────────────────────────────────────
    let karmac_sub = make_sub(m.py(), m, "karmac")?;
    export_class!(m, karmac_sub, karmac::PyCarmackBsp);
    export_class!(m, karmac_sub, karmac::PyMinBrain);
    export_class!(m, karmac_sub, karmac::PyBasalGanglia);
    export_class!(m, karmac_sub, karmac::PyGlobalWorkspace);

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
    #[cfg(feature = "cuda")]
    {
        add_fn!(kuramoto, kuramoto::py_ring_synthesis_cuda);
        add_fn!(kuramoto, kuramoto::py_interference_matrix_cuda);
        add_fn!(kuramoto, kuramoto::py_band_energy_cuda);
    }

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
    #[cfg(feature = "cuda")]
    add_fn!(io, io::py_t3_tact_generator_cuda);

    // ─── sparse ────────────────────────────────────────
    let sparse = make_sub(m.py(), m, "sparse")?;
    add_fn!(sparse, sparse::py_compute_hamiltonian);
    add_fn!(sparse, sparse::py_step_gco_full);
    add_fn!(sparse, sparse::py_propagate_spikes);
    add_fn!(sparse, sparse::py_sim_step);
    add_fn!(sparse, sparse::py_rate_step);
    add_fn!(sparse, sparse::py_apply_stdp_ltd);

    // ─── t3 ────────────────────────────────────────────
    let t3 = make_sub(m.py(), m, "t3")?;
    export_class!(m, t3, t3::PyTriOscillator);
    export_class!(m, t3, t3::PyNlmsDecoder);
    export_class!(m, t3, t3::PyKalmanDecoder);
    export_class!(m, t3, t3::PyT2SpikeGradTrajectory);
    export_class!(m, t3, t3::PyPstControl);
    export_class!(m, t3, t3::PyAttentionCoupling);
    export_class!(m, t3, t3::PySpiker);
    export_class!(m, t3, t3::PyHologramEncoder);
    export_class!(m, t3, t3::PyNormalizedSpiker);
    export_class!(m, t3, t3::PySpikeGradBridge);
    export_class!(m, t3, t3::PyTorusGrid);
    export_class!(m, t3, t3::PyTorusTrajectory);
    add_fn!(t3, t3::py_pst_features);
    add_fn!(t3, t3::py_pst_linear_features);
    add_fn!(t3, t3::py_pst_linear_dim);
    add_fn!(t3, t3::py_pst_fast_dim);
    add_fn!(t3, t3::py_compute_pearson);
    add_fn!(t3, t3::py_normalize_features);
    add_fn!(t3, t3::py_t2spike_grad_forward);
    add_fn!(t3, t3::py_t2spike_grad_backward);
    add_fn!(t3, t3::py_t2spike_grad_pst_matrix);
    add_fn!(t3, t3::py_t2spike_grad_pst_fast_matrix);
    #[cfg(feature = "cuda")]
    add_fn!(t3, t3::py_t2spike_grad_forward_cuda);
    #[cfg(feature = "cuda")]
    add_fn!(t3, t3::py_t2spike_grad_backward_cuda);
    #[cfg(feature = "connectome")]
    add_fn!(t3, t3::py_train_connectome_chase);

    // ─── lif_connectome ─────────────────────────────────
    let lif_connectome = make_sub(m.py(), m, "lif_connectome")?;
    add_fn!(lif_connectome, lif_connectome::py_lif_sim_cpu_recording);
    add_fn!(lif_connectome, lif_connectome::py_lif_sim_gpu_recording);

    // ─── snn ───────────────────────────────────────────
    let snn = make_sub(m.py(), m, "snn")?;
    export_class!(m, snn, snn::PyLifNeuron);
    export_class!(m, snn, neuron::PySomatoDendriticNeuron);
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
    add_fn!(connectome, connectome::py_bench_gco_gpu);
    add_fn!(connectome, connectome::py_bench_gco_aberton_gpu);

    // ─── stp ───────────────────────────────────────────
    let stp = make_sub(m.py(), m, "stp")?;
    export_class!(m, stp, poisson_stp::PyPoissonGenerator);
    export_class!(m, stp, poisson_stp::PyShortTermPlasticity);
    add_fn!(stp, poisson_stp::py_compute_lfp);
    export_class!(m, neurons, neuron::PyAstrocyteState); // 🧫 Astrocyte + Gap Junctions

    // ─── rl ────────────────────────────────────────────
    let rl = make_sub(m.py(), m, "rl")?;
    export_class!(m, rl, rl_agent::PyActorCriticAgent);

    // ─── vnc ────────────────────────────────────────────
    let vnc = make_sub(m.py(), m, "vnc")?;
    add_fn!(vnc, vnc::py_vnc_step);

    // ─── connectome_brain ─────────────────────────────────
    let connectome_brain = make_sub(m.py(), m, "connectome_brain")?;
    export_class!(m, connectome_brain, connectome_brain::PyConnectomeData);
    export_class!(m, connectome_brain, connectome_brain::PyBrainRunner);
    export_class!(m, connectome_brain, connectome_brain::PyAlphaLifWorker);
    export_class!(m, connectome_brain, connectome_brain::PyAlphaLifOutput);
    export_class!(m, connectome_brain, connectome_brain::PySpikeRecorder);

    // ─── format ────────────────────────────────────────
    let format = make_sub(m.py(), m, "format")?;
    add_fn!(format, taas_format::py_save_taas);
    add_fn!(format, taas_format::py_load_taas);
    add_fn!(format, taas_format::py_inspect_taas);
    add_fn!(format, taas_format::py_save_connectome);
    add_fn!(format, taas_format::py_load_connectome);
    add_fn!(format, taas_format::py_save_model);
    export_class!(m, format, taas_format::PyTaasReader);

    // ─── cpe ────────────────────────────────────────────
    let cpe = make_sub(m.py(), m, "cpe")?;
    add_fn!(cpe, cpe::py_cpe_kuramoto_step);
    add_fn!(cpe, cpe::py_cpe_phase_coherence);
    add_fn!(cpe, cpe::py_cpe_spike_entropy);
    add_fn!(cpe, cpe::py_cpe_detect_spikes);
    add_fn!(cpe, cpe::py_cpe_dopamine_delta);
    add_fn!(cpe, cpe::py_cpe_dopamine_timing_gate);
    add_fn!(cpe, cpe::py_cpe_eligibility_update);
    add_fn!(cpe, cpe::py_cpe_stdp_update);

    // ─── iit ─────────────────────────────────────────────
    // let iit = make_sub(m.py(), m, "iit")?;
    // export_class!(m, iit, iit::PyIITConsciousness);

    // ─── formula — реалтайм-сборка своего нейрона из ops ──
    let formula = make_sub(m.py(), m, "formula")?;
    export_class!(m, formula, formula::PyNeuronBuilder);
    export_class!(m, formula, formula::PyFormula);
    export_class!(m, formula, formula::PyFormulaPopulation);
    export_class!(m, formula, formula::PyTrace);
    export_class!(m, formula, formula::PyOpKind);
    export_class!(m, formula, formula::PyOpNode);
    export_class!(m, formula, formula::PyReg);
    // функции lif / run_program регистрируются в formula::register —
    // #[pyfunction(name = …)] недоступен извне модуля (приватный mod)

    // ─── hh ─────────────────────────────────────────────
    let hh = make_sub(m.py(), m, "hh")?;
    export_class!(m, hh, hh::PyHhLut);
    #[cfg(feature = "cuda")]
    export_class!(m, hh, hh_gpu::PyGpuHhLut);

    // ─── pillars — единый фасад──────────────────
    let pillars = pillars::register(m)?;
    // Единый фасад: каждый публичный символ подмодулей также доступен через
    // `taas_checkmate.pillars` (без повторного root-экспорта — он уже есть).
    macro_rules! add_fn_facade {
        ($mod:expr, $func:path) => {
            let __f = wrap_pyfunction!($func, &$mod)?;
            export_fn_facade(&$mod, &__f)?;
        };
    }
    macro_rules! export_class_facade {
        ($sub:expr, $cls:ty) => {{
            let sub_ref: &Bound<'_, PyModule> = &$sub;
            sub_ref.add_class::<$cls>()?;
        }};
    }
    add_fn_facade!(pillars, neuron::py_step_izhikevich);
    add_fn_facade!(pillars, neuron::py_step_saw);
    add_fn_facade!(pillars, neuron::py_step_gco);
    add_fn_facade!(pillars, neuron::py_step_lif);
    add_fn_facade!(pillars, neuron::py_step_adex);
    add_fn_facade!(pillars, neuron::py_step_fhn);
    add_fn_facade!(pillars, neuron::py_step_haap);
    add_fn_facade!(pillars, neuron::py_step_nmda);
    add_fn_facade!(pillars, neuron::py_step_hr);
    add_fn_facade!(pillars, neuron::py_step_wc);
    add_fn_facade!(pillars, neuron::py_step_neuromotor);
    add_fn_facade!(pillars, neuron::py_stdp_update);
    add_fn_facade!(pillars, neuron::py_stdp_homeostatic);
    add_fn_facade!(pillars, neuron::py_firing_rates);
    add_fn_facade!(pillars, neuron::py_mean_firing_rate);
    add_fn_facade!(pillars, neuron::py_compute_isi);
    add_fn_facade!(pillars, neuron::py_compute_cvs);
    add_fn_facade!(pillars, neuron::step_eprop);
    add_fn_facade!(pillars, kuramoto::py_kuramoto_step_sparse);
    add_fn_facade!(pillars, kuramoto::py_t3_step_sparse);
    add_fn_facade!(pillars, kuramoto::py_interference);
    add_fn_facade!(pillars, kuramoto::py_interference_mean);
    add_fn_facade!(pillars, kuramoto::py_batch_coherence_sweep);
    add_fn_facade!(pillars, kuramoto::py_phase_coherence);
    add_fn_facade!(pillars, kuramoto::py_band_energy);
    add_fn_facade!(pillars, kuramoto::py_lazy_sincos_update);
    add_fn_facade!(pillars, kuramoto::py_kalman_update);
    add_fn_facade!(pillars, kuramoto::py_adam_train);
    add_fn_facade!(pillars, kuramoto::py_nlms_train);
    add_fn_facade!(pillars, kuramoto::py_pst_decode);
    add_fn_facade!(pillars, kuramoto::py_ring_synthesis);
    add_fn_facade!(pillars, kuramoto::py_winding_matrix);
    add_fn_facade!(pillars, io::py_safety_clamp);
    add_fn_facade!(pillars, io::py_sigmoid_gate);
    add_fn_facade!(pillars, io::py_phase_gate_apply);
    add_fn_facade!(pillars, io::py_stochastic_encoder);
    add_fn_facade!(pillars, io::py_thalamic_gate);
    add_fn_facade!(pillars, io::py_hebbian_update);
    add_fn_facade!(pillars, io::py_echo_tap);
    add_fn_facade!(pillars, io::py_t3_drive_encoder);
    add_fn_facade!(pillars, io::py_t3_complex_decoder);
    add_fn_facade!(pillars, io::py_t3_inverse_kuramoto);
    add_fn_facade!(pillars, io::py_t3_pst_features);
    add_fn_facade!(pillars, io::py_t3_tact_generator);
    add_fn_facade!(pillars, sparse::py_compute_hamiltonian);
    add_fn_facade!(pillars, sparse::py_step_gco_full);
    add_fn_facade!(pillars, sparse::py_propagate_spikes);
    add_fn_facade!(pillars, sparse::py_sim_step);
    add_fn_facade!(pillars, sparse::py_rate_step);
    add_fn_facade!(pillars, sparse::py_apply_stdp_ltd);
    add_fn_facade!(pillars, snn::py_intrinsic_plasticity);
    add_fn_facade!(pillars, snn::py_synaptic_normalization);
    add_fn_facade!(pillars, snn::py_bsa_encode);
    add_fn_facade!(pillars, snn::py_phase_encode);
    add_fn_facade!(pillars, snn::py_bsa_kernel_hamming);
    add_fn_facade!(pillars, snn::py_population_rate_decode);
    add_fn_facade!(pillars, snn::py_spike_raster_to_csv);
    add_fn_facade!(pillars, eprop::py_surrogate_gradient);
    add_fn_facade!(pillars, eprop::py_eprop_update);
    add_fn_facade!(pillars, autotune::probe_eprop_fr);
    add_fn_facade!(pillars, autotune::suggest_eprop_config);
    add_fn_facade!(pillars, cpe::py_cpe_kuramoto_step);
    add_fn_facade!(pillars, cpe::py_cpe_phase_coherence);
    add_fn_facade!(pillars, cpe::py_cpe_spike_entropy);
    add_fn_facade!(pillars, cpe::py_cpe_detect_spikes);
    add_fn_facade!(pillars, cpe::py_cpe_dopamine_delta);
    add_fn_facade!(pillars, cpe::py_cpe_dopamine_timing_gate);
    add_fn_facade!(pillars, cpe::py_cpe_eligibility_update);
    add_fn_facade!(pillars, cpe::py_cpe_stdp_update);
    add_fn_facade!(pillars, vnc::py_vnc_step);
    add_fn_facade!(pillars, lif_connectome::py_lif_sim_cpu_recording);
    #[cfg(feature = "cuda")]
    add_fn_facade!(pillars, lif_connectome::py_lif_sim_gpu_recording);
    #[cfg(feature = "cuda")]
    {
        add_fn_facade!(pillars, kuramoto::py_ring_synthesis_cuda);
        add_fn_facade!(pillars, kuramoto::py_interference_matrix_cuda);
        add_fn_facade!(pillars, kuramoto::py_band_energy_cuda);
        export_class_facade!(pillars, cuda_kernel::PyCudaKernel);
    }
    export_class_facade!(pillars, eprop::PyEPropLIF);
    export_class_facade!(pillars, eprop::PyEPropIzhikevichMulti);
    export_class_facade!(pillars, snn::PySornNetwork);
    export_class_facade!(pillars, snn::PyTtfsEncoder);
    export_class_facade!(pillars, snn::PyLifNeuron);
    export_class_facade!(pillars, poisson_stp::PyPoissonGenerator);
    export_class_facade!(pillars, poisson_stp::PyShortTermPlasticity);
    export_class_facade!(pillars, neuron::PyGlif3State);
    export_class_facade!(pillars, neuron::PyGlif4State);
    export_class_facade!(pillars, neuron::PySomatoDendriticNeuron);
    export_class_facade!(pillars, neuron::PyAstrocyteState);
    export_class_facade!(pillars, neuron::PyChordotonalSensor);
    export_class_facade!(pillars, trainer::PyEPropTrainer);
    export_class_facade!(pillars, rl_agent::PyActorCriticAgent);
    // formula — фасад pillars (единый вход для нового кода)
    export_class_facade!(pillars, formula::PyNeuronBuilder);
    export_class_facade!(pillars, formula::PyFormula);
    export_class_facade!(pillars, formula::PyFormulaPopulation);
    export_class_facade!(pillars, formula::PyTrace);
    export_class_facade!(pillars, formula::PyOpKind);
    export_class_facade!(pillars, formula::PyOpNode);
    export_class_facade!(pillars, formula::PyReg);
    formula::register(&formula, Some(&pillars), Some(m))?;

    // ─── hologram ─────────────────────────────────────────
    let hologram = make_sub(m.py(), m, "hologram")?;
    export_class!(m, hologram, hologram::PyHologramBridge);

    // ─── spaun ─────────────────────────────────────────
    let spaun = make_sub(m.py(), m, "spaun")?;
    export_class!(m, spaun, spaun::PySpaunPipeline);

    // ─── spaun_gpu ─────────────────────────────────────
    #[cfg(feature = "cuda")]
    {
        let spaun_gpu = make_sub(m.py(), m, "spaun_gpu")?;
        export_class!(m, spaun_gpu, spaun_gpu::PySpaunPipelineGPU);
    }

    // ─── gpu_refiner ────────────────────────────────────
    #[cfg(feature = "cuda")]
    {
        let gpu_refiner = make_sub(m.py(), m, "gpu_refiner")?;
        export_class!(m, gpu_refiner, gpu_refiner::PyGpuConnectomeRefiner);
        export_class!(m, gpu_refiner, gpu_refiner::PyRefinerHistory);
    }

    Ok(())
}
