//! Component serialization helpers for .taas format.
//!
//! Each save/load pair handles one model type. Following the pattern established
//! in the original monolithic lib.rs, every physical quantity is stored as a
//! named tensor component (f32[], i32[], i64[], u64[]) inside the .taas archive.

use crate::writer::TaasWriter;
use crate::reader::TaasReader;

// ─── BrainState ─────────────────────────────────────────

/// Save CPU brain simulation state to a .taas archive.
pub fn save_brain_state(state: &taas_connectome::sim::BrainState, writer: &mut TaasWriter) {
    writer.add_metadata("type", "BrainState");
    writer.add_metadata("n_neurons_padded", &state.n_padded.to_string());
    writer.add_f32("brain_v", "BrainState", &state.v);
    writer.add_f32("brain_u", "BrainState", &state.u);
    writer.add_f32("brain_trace", "BrainState", &state.trace);
    writer.add_f32("brain_Ibuf", "BrainState", &state.I_buf);
    writer.add_i32("brain_spikes", "BrainState", &state.spikes);
    writer.add_f32("brain_refractory", "BrainState", &state.refractory);
    writer.add_f32("brain_slow_state", "BrainState", &state.slow_state);
}

/// Load CPU brain state from a .taas reader.
pub fn load_brain_state(reader: &TaasReader) -> Result<taas_connectome::sim::BrainState, String> {
    let n_padded = reader.metadata().get("n_neurons_padded")
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or_else(|| "n_neurons_padded not found".to_string())?;
    // slow_state: backwards-compatible — old archives (pre slow_state) lack this
    // component; reconstruct it from n_padded as 0.5 (neutral baseline, matches
    // BrainState::new()). See BrainState::slow_state doc.
    let slow_state = reader.get_f32("brain_slow_state")
        .unwrap_or_else(|_| vec![0.5; n_padded]);
    let state = taas_connectome::sim::BrainState {
        v: reader.get_f32("brain_v")?,
        u: reader.get_f32("brain_u")?,
        trace: reader.get_f32("brain_trace")?,
        I_buf: reader.get_f32("brain_Ibuf")?,
        spikes: reader.get_i32("brain_spikes")?,
        refractory: reader.get_f32("brain_refractory")?,
        slow_state,
        n_padded,
        n_neurons: 0,
        queue: Default::default(),
    };
    Ok(state)
}

/// Save Izhikevich parameters.
pub fn save_izhikevich_params(params: &taas_connectome::sim::IzhikevichParams, writer: &mut TaasWriter) {
    writer.add_metadata("type", "IzhikevichParams");
    writer.add_metadata("n", &params.a.len().to_string());
    writer.add_f32("iz_a", "IzhikevichParams", &params.a);
    writer.add_f32("iz_b", "IzhikevichParams", &params.b);
    writer.add_f32("iz_c", "IzhikevichParams", &params.c);
    writer.add_f32("iz_d", "IzhikevichParams", &params.d);
    writer.add_scalar_f32("iz_dt_ms", "IzhikevichParams", params.dt_ms);
    writer.add_scalar_f32("iz_tau_ref", "IzhikevichParams", params.tau_ref);
}

/// Load Izhikevich parameters.
pub fn load_izhikevich_params(reader: &TaasReader) -> Result<taas_connectome::sim::IzhikevichParams, String> {
    Ok(taas_connectome::sim::IzhikevichParams {
        a: reader.get_f32("iz_a")?,
        b: reader.get_f32("iz_b")?,
        c: reader.get_f32("iz_c")?,
        d: reader.get_f32("iz_d")?,
        dt_ms: reader.get_scalar_f32("iz_dt_ms")?,
        tau_ref: reader.get_scalar_f32("iz_tau_ref")?,
    })
}

/// Save GPU brain state: downloads v, u from GPU, serialises as tensors.
///
/// Requires the `cuda-kernels` feature (the `gpu` module provides the wrapper).
#[cfg(feature = "cuda")]
pub fn save_gpu_state(gpu: &cuda_kernels::GpuContext, n_neurons: usize, writer: &mut TaasWriter) -> Result<(), String> {
    let mut v = vec![0.0f32; n_neurons];
    let mut u = vec![0.0f32; n_neurons];
    gpu.download_v(&mut v)?;
    gpu.download_u(&mut u)?;
    writer.add_metadata("type", "GpuState");
    writer.add_metadata("n_neurons", &n_neurons.to_string());
    writer.add_f32("gpu_v", "GpuState", &v);
    writer.add_f32("gpu_u", "GpuState", &u);
    Ok(())
}

/// Load GPU brain state: creates GpuContext, uploads v, u.
#[cfg(feature = "cuda")]
pub fn load_gpu_state(reader: &TaasReader, n_synapses: usize) -> Result<cuda_kernels::GpuContext, String> {
    let n_neurons = reader.metadata().get("n_neurons")
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or_else(|| "n_neurons not found".to_string())?;
    let v = reader.get_f32("gpu_v")?;
    let u = reader.get_f32("gpu_u")?;
    let gpu = cuda_kernels::GpuContext::new(n_neurons, n_synapses)?;
    // Upload v, u; use default for a,b,c,d
    let default = vec![0.0f32; n_neurons];
    gpu.upload_neurons(&v, &u, &default, &default, &default, &default)?;
    Ok(gpu)
}

// ─── Brain IO ────────────────────────────────────────────

/// Save BrainIO as JSON metadata.
pub fn save_brain_io(io: &taas_connectome::BrainIO, writer: &mut TaasWriter) {
    writer.add_metadata("type", "BrainIO");
    let json = serde_json::to_string(io).unwrap_or_else(|_| "null".to_string());
    writer.add_metadata("brain_io", &json);
}

/// Load BrainIO from JSON metadata.
pub fn load_brain_io(reader: &TaasReader) -> Result<taas_connectome::BrainIO, String> {
    reader.metadata().get("brain_io")
        .and_then(|j| serde_json::from_str(j).ok())
        .ok_or_else(|| "brain_io not found in metadata".to_string())
}

// ─── Connectome ─────────────────────────────────────────

/// Save a full connectome to a .taas archive.
///
/// Components:
/// - `row_ptr` (i32[]) — CSR row pointer
/// - `col_indices` (i32[]) — CSR column indices
/// - `values` (f32[]) — synaptic weights
/// - `nt_types` (f32[]) — neurotransmitter types (+1 exc, −1 inh)
/// - `receptor_types` (u8[]) — per-synapse receptor (optional; empty if unset)
/// - `groups_json` (metadata) — groups serialised as JSON
pub fn save_connectome(conn: &taas_connectome::ConnectomeData, writer: &mut TaasWriter) {
    writer.add_metadata("type", "Connectome");
    writer.add_metadata("n_neurons", &conn.n_neurons.to_string());
    writer.add_metadata("n_synapses", &conn.n_synapses.to_string());
    writer.add_i32("row_ptr", "Connectome", &conn.row_ptr);
    writer.add_i32("col_indices", "Connectome", &conn.col_indices);
    writer.add_f32("values", "Connectome", &conn.values);
    writer.add_f32("nt_types", "Connectome", &conn.nt_types);
    writer.add_u8("receptor_types", "Connectome", &conn.receptor_types);
    // Serialize groups as JSON in metadata
    let groups_json = serde_json::to_string(&conn.groups).unwrap_or_else(|_| "[]".to_string());
    writer.add_metadata("groups", &groups_json);
}

/// Load a connectome from a .taas reader.
pub fn load_connectome(reader: &TaasReader) -> Result<taas_connectome::ConnectomeData, String> {
    let _n_neurons = reader.metadata().get("n_neurons")
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or_else(|| "n_neurons not found in metadata".to_string())?;
    let row_ptr = reader.get_i32("row_ptr")?;
    let col_indices = reader.get_i32("col_indices")?;
    let values = reader.get_f32("values")?;
    let nt_types = reader.get_f32("nt_types")?;
    // receptor_types is optional — old archives (pre receptor_types) lack it,
    // returning an empty Vec (matches ConnectomeData::new default).
    let receptor_types = reader.get_u8("receptor_types").unwrap_or_default();
    let groups: Vec<taas_connectome::NeuronGroup> = reader.metadata()
        .get("groups")
        .and_then(|g| serde_json::from_str(g).ok())
        .unwrap_or_default();
    let group_refs: Vec<(&str, Vec<usize>)> = groups.iter()
        .map(|g| (g.name.as_str(), g.indices.clone()))
        .collect();
    let mut conn = taas_connectome::ConnectomeData::new(row_ptr, col_indices, values, nt_types, group_refs);
    conn.receptor_types = receptor_types;
    Ok(conn)
}

// ─── TriOscillator ──────────────────────────────────────

pub fn save_trioscilator(tri: &taas_t3::TriOscillator, writer: &mut TaasWriter) {
    writer.add_metadata("type", "TriOscillator");
    writer.add_metadata("n", &tri.n.to_string());
    writer.add_f32("th1", "TriOscillator", &tri.th1);
    writer.add_f32("th2", "TriOscillator", &tri.th2);
    writer.add_f32("th3", "TriOscillator", &tri.th3);
    writer.add_f32("prev_th1", "TriOscillator", &tri.prev_th1);
    writer.add_f32("prev_th2", "TriOscillator", &tri.prev_th2);
    writer.add_f32("prev_th3", "TriOscillator", &tri.prev_th3);
    writer.add_f32("r1", "TriOscillator", &tri.r1);
    writer.add_f32("r2", "TriOscillator", &tri.r2);
    writer.add_f32("r3", "TriOscillator", &tri.r3);
    writer.add_f32("prev_r1", "TriOscillator", &tri.prev_r1);
    writer.add_f32("prev_r2", "TriOscillator", &tri.prev_r2);
    writer.add_f32("prev_r3", "TriOscillator", &tri.prev_r3);
    writer.add_f32("omega1", "TriOscillator", &tri.omega1);
    writer.add_f32("omega2", "TriOscillator", &tri.omega2);
    writer.add_f32("omega3", "TriOscillator", &tri.omega3);
    writer.add_f32("drive1", "TriOscillator", &tri.drive1);
    writer.add_f32("drive2", "TriOscillator", &tri.drive2);
    writer.add_f32("drive3", "TriOscillator", &tri.drive3);
    writer.add_f32("sin1", "TriOscillator", &tri.sin1);
    writer.add_f32("cos1", "TriOscillator", &tri.cos1);
    writer.add_f32("sin2", "TriOscillator", &tri.sin2);
    writer.add_f32("cos2", "TriOscillator", &tri.cos2);
    writer.add_f32("sin3", "TriOscillator", &tri.sin3);
    writer.add_f32("cos3", "TriOscillator", &tri.cos3);
    writer.add_f32("coupling1", "TriOscillator", &tri.coupling1);
    writer.add_f32("coupling2", "TriOscillator", &tri.coupling2);
    writer.add_f32("coupling3", "TriOscillator", &tri.coupling3);
    writer.add_scalar_f32("w12", "TriOscillator", tri.w12);
    writer.add_scalar_f32("w13", "TriOscillator", tri.w13);
    writer.add_scalar_f32("w23", "TriOscillator", tri.w23);
    writer.add_scalar_f32("alpha", "TriOscillator", tri.alpha);
    writer.add_scalar_f32("drive_gain", "TriOscillator", tri.drive_gain);
    writer.add_scalar_f32("epsilon", "TriOscillator", tri.epsilon);
    writer.add_scalar_f32("phase_gate_gain", "TriOscillator", tri.phase_gate_gain);
    writer.add_scalar_f32("gamma1", "TriOscillator", tri.gamma1);
    writer.add_scalar_f32("gamma2", "TriOscillator", tri.gamma2);
    writer.add_scalar_f32("gamma3", "TriOscillator", tri.gamma3);
    writer.add_scalar_bool("initialised", "TriOscillator", tri.initialised);
}

pub fn load_trioscilator(reader: &TaasReader, n: usize) -> Result<taas_t3::TriOscillator<taas_t3::Avx2Backend>, String> {
    use taas_t3::{TriOscillator, Avx2Backend};
    let mut tri = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);
    macro_rules! restore {
        ($field:ident, $name:expr) => {
            if let Ok(v) = reader.get_f32($name) {
                let n = v.len().min(tri.$field.len());
                tri.$field[..n].copy_from_slice(&v[..n]);
            }
        };
    }
    restore!(th1, "th1"); restore!(th2, "th2"); restore!(th3, "th3");
    restore!(prev_th1, "prev_th1"); restore!(prev_th2, "prev_th2"); restore!(prev_th3, "prev_th3");
    restore!(r1, "r1"); restore!(r2, "r2"); restore!(r3, "r3");
    restore!(prev_r1, "prev_r1"); restore!(prev_r2, "prev_r2"); restore!(prev_r3, "prev_r3");
    restore!(omega1, "omega1"); restore!(omega2, "omega2"); restore!(omega3, "omega3");
    restore!(drive1, "drive1"); restore!(drive2, "drive2"); restore!(drive3, "drive3");
    restore!(sin1, "sin1"); restore!(cos1, "cos1");
    restore!(sin2, "sin2"); restore!(cos2, "cos2");
    restore!(sin3, "sin3"); restore!(cos3, "cos3");
    restore!(coupling1, "coupling1"); restore!(coupling2, "coupling2"); restore!(coupling3, "coupling3");
    tri.w12 = reader.get_scalar_f32("w12").unwrap_or(0.0);
    tri.w13 = reader.get_scalar_f32("w13").unwrap_or(0.0);
    tri.w23 = reader.get_scalar_f32("w23").unwrap_or(0.0);
    tri.alpha = reader.get_scalar_f32("alpha").unwrap_or(1.0);
    tri.drive_gain = reader.get_scalar_f32("drive_gain").unwrap_or(10.0);
    tri.epsilon = reader.get_scalar_f32("epsilon").unwrap_or(0.1);
    tri.phase_gate_gain = reader.get_scalar_f32("phase_gate_gain").unwrap_or(4.0);
    tri.gamma1 = reader.get_scalar_f32("gamma1").unwrap_or(0.5);
    tri.gamma2 = reader.get_scalar_f32("gamma2").unwrap_or(1.0);
    tri.gamma3 = reader.get_scalar_f32("gamma3").unwrap_or(1.12);
    tri.initialised = reader.get_scalar_bool("initialised").unwrap_or(true);
    Ok(tri)
}

// ─── NlmsDecoder ────────────────────────────────────────

pub fn save_nlms_decoder(dec: &taas_t3::NlmsDecoder, writer: &mut TaasWriter) {
    writer.add_metadata("decoder_type", "NlmsDecoder");
    writer.add_f32("nlms_weights", "NlmsDecoder", &dec.weights);
    writer.add_f32("nlms_bias", "NlmsDecoder", &dec.bias);
    writer.add_scalar_f32("nlms_lr", "NlmsDecoder", dec.lr);
    writer.add_scalar_f32("nlms_eps", "NlmsDecoder", dec.eps);
}

pub fn load_nlms_decoder(reader: &TaasReader, n: usize) -> Result<taas_t3::NlmsDecoder, String> {
    let mut dec = taas_t3::NlmsDecoder::new(n, 0.01, 1e-8);
    let weights = reader.get_f32("nlms_weights")?;
    let n_w = weights.len().min(dec.weights.len());
    dec.weights[..n_w].copy_from_slice(&weights[..n_w]);
    if let Ok(bias) = reader.get_f32("nlms_bias") {
        if !bias.is_empty() { dec.bias[0] = bias[0]; }
    }
    dec.lr = reader.get_scalar_f32("nlms_lr").unwrap_or(0.01);
    dec.eps = reader.get_scalar_f32("nlms_eps").unwrap_or(1e-8);
    Ok(dec)
}

// ─── KalmanDecoder ──────────────────────────────────────

pub fn save_kalman_decoder(dec: &taas_t3::KalmanDecoder, writer: &mut TaasWriter) {
    writer.add_metadata("decoder_type", "KalmanDecoder");
    writer.add_f32("kalman_weights", "KalmanDecoder", &dec.weights);
    writer.add_f32("kalman_cov", "KalmanDecoder", &dec.cov);
    writer.add_scalar_f32("kalman_bias", "KalmanDecoder", dec.bias);
    writer.add_scalar_f32("kalman_q", "KalmanDecoder", dec.q);
    writer.add_scalar_f32("kalman_r", "KalmanDecoder", dec.r);
}

pub fn load_kalman_decoder(reader: &TaasReader, n: usize) -> Result<taas_t3::KalmanDecoder, String> {
    let mut dec = taas_t3::KalmanDecoder::new(n, 0.001, 0.1);
    let weights = reader.get_f32("kalman_weights")?;
    let n_w = weights.len().min(dec.weights.len());
    dec.weights[..n_w].copy_from_slice(&weights[..n_w]);
    if let Ok(cov) = reader.get_f32("kalman_cov") {
        let n_c = cov.len().min(dec.cov.len());
        dec.cov[..n_c].copy_from_slice(&cov[..n_c]);
    }
    dec.bias = reader.get_scalar_f32("kalman_bias").unwrap_or(0.0);
    dec.q = reader.get_scalar_f32("kalman_q").unwrap_or(0.001);
    dec.r = reader.get_scalar_f32("kalman_r").unwrap_or(0.1);
    Ok(dec)
}

// ─── SornNetwork ────────────────────────────────────────

pub fn save_sorn_network(sorn: &taas_core::neuron::SornNetwork, writer: &mut TaasWriter) {
    writer.add_metadata("type", "SornNetwork");
    writer.add_metadata("n", &sorn.n.to_string());
    let n = sorn.n;
    writer.add_f32_shape("sorn_weights", "SornNetwork", &sorn.weights, vec![n, n]);
    writer.add_f32("sorn_theta", "SornNetwork", &sorn.theta);
    writer.add_f32("sorn_rates", "SornNetwork", &sorn.rates);
    writer.add_scalar_f32("sorn_lr_stdp_ltd", "SornNetwork", sorn.lr_stdp_ltd);
    writer.add_scalar_f32("sorn_lr_stdp_ltp", "SornNetwork", sorn.lr_stdp_ltp);
    writer.add_scalar_f32("sorn_w_min", "SornNetwork", sorn.w_min);
    writer.add_scalar_f32("sorn_w_max", "SornNetwork", sorn.w_max);
    writer.add_scalar_f32("sorn_w_target_sum", "SornNetwork", sorn.w_target_sum);
    writer.add_scalar_f32("sorn_lr_ip", "SornNetwork", sorn.lr_ip);
    writer.add_scalar_f32("sorn_target_rate", "SornNetwork", sorn.target_rate);
    writer.add_scalar_f32("sorn_ip_decay", "SornNetwork", sorn.ip_decay);
}

pub fn load_sorn_network(reader: &TaasReader, n: usize) -> Result<taas_core::neuron::SornNetwork, String> {
    let mut sorn = taas_core::neuron::SornNetwork::new(n, 0.1, 10.0);
    macro_rules! restore {
        ($field:ident, $name:expr) => {
            if let Ok(v) = reader.get_f32($name) {
                let n = v.len().min(sorn.$field.len());
                sorn.$field[..n].copy_from_slice(&v[..n]);
            }
        };
    }
    restore!(weights, "sorn_weights");
    restore!(theta, "sorn_theta");
    restore!(rates, "sorn_rates");
    sorn.lr_stdp_ltd = reader.get_scalar_f32("sorn_lr_stdp_ltd").unwrap_or(0.01);
    sorn.lr_stdp_ltp = reader.get_scalar_f32("sorn_lr_stdp_ltp").unwrap_or(0.005);
    sorn.w_min = reader.get_scalar_f32("sorn_w_min").unwrap_or(0.0);
    sorn.w_max = reader.get_scalar_f32("sorn_w_max").unwrap_or(2.0);
    sorn.w_target_sum = reader.get_scalar_f32("sorn_w_target_sum").unwrap_or(10.0);
    sorn.lr_ip = reader.get_scalar_f32("sorn_lr_ip").unwrap_or(0.001);
    sorn.target_rate = reader.get_scalar_f32("sorn_target_rate").unwrap_or(0.01);
    sorn.ip_decay = reader.get_scalar_f32("sorn_ip_decay").unwrap_or(0.999);
    Ok(sorn)
}

// ─── EPropLIF ────────────────────────────────────────────

pub fn save_eprop_lif(eprop: &taas_core::neuron::EPropLIF, writer: &mut TaasWriter) {
    writer.add_metadata("type", "EPropLIF");
    writer.add_metadata("n", &eprop.n.to_string());
    writer.add_f32("eprop_v", "EPropLIF", &eprop.v);
    writer.add_f32("eprop_eligibility", "EPropLIF", &eprop.eligibility);
    writer.add_f32("eprop_psi", "EPropLIF", &eprop.psi);
    writer.add_f32("eprop_refractory", "EPropLIF", &eprop.refractory);
    writer.add_scalar_f32("eprop_dt", "EPropLIF", eprop.dt);
    writer.add_scalar_f32("eprop_tau_m", "EPropLIF", eprop.tau_m);
    writer.add_scalar_f32("eprop_tau_e", "EPropLIF", eprop.tau_e);
    writer.add_scalar_f32("eprop_tau_ref", "EPropLIF", eprop.tau_ref);
    writer.add_scalar_f32("eprop_v_rest", "EPropLIF", eprop.v_rest);
    writer.add_scalar_f32("eprop_v_thresh", "EPropLIF", eprop.v_thresh);
    writer.add_scalar_f32("eprop_v_reset", "EPropLIF", eprop.v_reset);
    writer.add_scalar_f32("eprop_tau_surrogate", "EPropLIF", eprop.tau_surrogate);
}

pub fn load_eprop_lif(reader: &TaasReader, n: usize) -> Result<taas_core::neuron::EPropLIF, String> {
    let mut eprop = taas_core::neuron::EPropLIF::new(n, 1.0, 10.0, 20.0);
    macro_rules! restore {
        ($field:ident, $name:expr) => {
            if let Ok(v) = reader.get_f32($name) {
                let n = v.len().min(eprop.$field.len());
                eprop.$field[..n].copy_from_slice(&v[..n]);
            }
        };
    }
    restore!(v, "eprop_v");
    restore!(eligibility, "eprop_eligibility"); restore!(psi, "eprop_psi");
    restore!(refractory, "eprop_refractory");
    eprop.dt = reader.get_scalar_f32("eprop_dt").unwrap_or(1.0);
    eprop.tau_m = reader.get_scalar_f32("eprop_tau_m").unwrap_or(10.0);
    eprop.tau_e = reader.get_scalar_f32("eprop_tau_e").unwrap_or(20.0);
    eprop.tau_ref = reader.get_scalar_f32("eprop_tau_ref").unwrap_or(3.0);
    eprop.v_rest = reader.get_scalar_f32("eprop_v_rest").unwrap_or(-65.0);
    eprop.v_thresh = reader.get_scalar_f32("eprop_v_thresh").unwrap_or(-50.0);
    eprop.v_reset = reader.get_scalar_f32("eprop_v_reset").unwrap_or(-70.0);
    eprop.tau_surrogate = reader.get_scalar_f32("eprop_tau_surrogate").unwrap_or(2.0);
    Ok(eprop)
}

// ─── ShortTermPlasticity ─────────────────────────────────

pub fn save_stp(stp: &taas_core::neuron::ShortTermPlasticity, writer: &mut TaasWriter) {
    writer.add_metadata("type", "ShortTermPlasticity");
    writer.add_metadata("n", &stp.n.to_string());
    writer.add_f32("stp_x", "ShortTermPlasticity", &stp.x);
    writer.add_f32("stp_u", "ShortTermPlasticity", &stp.u);
    writer.add_f32("stp_tau_d", "ShortTermPlasticity", &stp.tau_d);
    writer.add_f32("stp_tau_f", "ShortTermPlasticity", &stp.tau_f);
    writer.add_f32("stp_u0", "ShortTermPlasticity", &stp.u0);
    writer.add_f32("stp_a", "ShortTermPlasticity", &stp.a);
    writer.add_scalar_f32("stp_dt", "ShortTermPlasticity", stp.dt);
}

pub fn load_stp(reader: &TaasReader, n: usize) -> Result<taas_core::neuron::ShortTermPlasticity, String> {
    let mut stp = taas_core::neuron::ShortTermPlasticity::new(n, 1.0);
    macro_rules! restore {
        ($field:ident, $name:expr) => {
            if let Ok(v) = reader.get_f32($name) {
                let n = v.len().min(stp.$field.len());
                stp.$field[..n].copy_from_slice(&v[..n]);
            }
        };
    }
    restore!(x, "stp_x"); restore!(u, "stp_u");
    restore!(tau_d, "stp_tau_d"); restore!(tau_f, "stp_tau_f");
    restore!(u0, "stp_u0"); restore!(a, "stp_a");
    stp.dt = reader.get_scalar_f32("stp_dt").unwrap_or(1.0);
    Ok(stp)
}

// ─── ActorCriticAgent ────────────────────────────────────

#[cfg(feature = "train")]
pub fn save_actor_critic_agent(agent: &taas_train::rl::ActorCriticAgent, writer: &mut TaasWriter) {
    writer.add_metadata("type", "ActorCriticAgent");
    writer.add_metadata("n_input", &agent.n_input.to_string());
    writer.add_metadata("n_hidden", &agent.n_hidden.to_string());
    writer.add_metadata("n_actions", &agent.n_actions_real.to_string());
    writer.add_metadata("n_steps", &agent.n_steps.to_string());
    writer.add_f32("rl_w_ih", "ActorCriticAgent", &agent.w_ih);
    writer.add_f32("rl_w_ho", "ActorCriticAgent", &agent.w_ho);
    writer.add_f32("rl_w_ic", "ActorCriticAgent", &agent.w_ic);
    writer.add_f32("rl_elig_ih", "ActorCriticAgent", &agent.elig_ih);
    writer.add_f32("rl_elig_ho", "ActorCriticAgent", &agent.elig_ho);
    writer.add_f32("rl_elig_ic", "ActorCriticAgent", &agent.elig_ic);
    writer.add_f32("rl_trace_input", "ActorCriticAgent", &agent.trace_input);
    writer.add_f32("rl_trace_hidden", "ActorCriticAgent", &agent.trace_hidden);
    writer.add_f32("rl_trace_output", "ActorCriticAgent", &agent.trace_output);
    writer.add_f32("rl_trace_critic", "ActorCriticAgent", &agent.trace_critic_val);
    writer.add_scalar_f32("rl_dt", "ActorCriticAgent", agent.dt);
    writer.add_scalar_f32("rl_gamma", "ActorCriticAgent", agent.gamma);
    writer.add_scalar_f32("rl_lr_actor", "ActorCriticAgent", agent.lr_actor);
    writer.add_scalar_f32("rl_lr_critic", "ActorCriticAgent", agent.lr_critic);
    writer.add_scalar_f32("rl_tau_trace", "ActorCriticAgent", agent.tau_trace);
    writer.add_scalar_f32("rl_tau_e", "ActorCriticAgent", agent.tau_e);
    writer.add_scalar_f32("rl_a_ltp", "ActorCriticAgent", agent.a_ltp);
    writer.add_scalar_f32("rl_a_ltd", "ActorCriticAgent", agent.a_ltd);
    writer.add_scalar_f32("rl_w_min", "ActorCriticAgent", agent.w_min);
    writer.add_scalar_f32("rl_w_max", "ActorCriticAgent", agent.w_max);
    writer.add_scalar_f32("rl_tau_ref", "ActorCriticAgent", agent.tau_ref);
    writer.add_scalar_f32("rl_last_value", "ActorCriticAgent", agent.last_value);
}

#[cfg(feature = "train")]
pub fn load_actor_critic_agent(reader: &TaasReader) -> Result<taas_train::rl::ActorCriticAgent, String> {
    let n_input = reader.metadata().get("n_input").and_then(|s| s.parse().ok()).ok_or("n_input not found")?;
    let n_hidden = reader.metadata().get("n_hidden").and_then(|s| s.parse().ok()).ok_or("n_hidden not found")?;
    let n_actions = reader.metadata().get("n_actions").and_then(|s| s.parse().ok()).ok_or("n_actions not found")?;
    let n_steps = reader.metadata().get("n_steps").and_then(|s| s.parse().ok()).ok_or("n_steps not found")?;
    let dt = reader.get_scalar_f32("rl_dt").unwrap_or(1.0);
    let mut agent = taas_train::rl::ActorCriticAgent::new(n_input, n_hidden, n_actions, n_steps, dt, 42);
    macro_rules! restore {
        ($field:ident, $name:expr) => {
            if let Ok(v) = reader.get_f32($name) {
                let n = v.len().min(agent.$field.len());
                agent.$field[..n].copy_from_slice(&v[..n]);
            }
        };
    }
    restore!(w_ih, "rl_w_ih"); restore!(w_ho, "rl_w_ho"); restore!(w_ic, "rl_w_ic");
    restore!(elig_ih, "rl_elig_ih"); restore!(elig_ho, "rl_elig_ho"); restore!(elig_ic, "rl_elig_ic");
    restore!(trace_input, "rl_trace_input"); restore!(trace_hidden, "rl_trace_hidden");
    restore!(trace_output, "rl_trace_output"); restore!(trace_critic_val, "rl_trace_critic");
    agent.gamma = reader.get_scalar_f32("rl_gamma").unwrap_or(0.95);
    agent.lr_actor = reader.get_scalar_f32("rl_lr_actor").unwrap_or(0.01);
    agent.lr_critic = reader.get_scalar_f32("rl_lr_critic").unwrap_or(0.01);
    agent.tau_trace = reader.get_scalar_f32("rl_tau_trace").unwrap_or(20.0);
    agent.tau_e = reader.get_scalar_f32("rl_tau_e").unwrap_or(500.0);
    agent.a_ltp = reader.get_scalar_f32("rl_a_ltp").unwrap_or(0.1);
    agent.a_ltd = reader.get_scalar_f32("rl_a_ltd").unwrap_or(0.12);
    agent.w_min = reader.get_scalar_f32("rl_w_min").unwrap_or(0.0);
    agent.w_max = reader.get_scalar_f32("rl_w_max").unwrap_or(2.0);
    agent.tau_ref = reader.get_scalar_f32("rl_tau_ref").unwrap_or(3.0);
    agent.last_value = reader.get_scalar_f32("rl_last_value").unwrap_or(0.0);
    Ok(agent)
}

// ─── PoissonGenerator ────────────────────────────────────

pub fn save_poisson(poisson: &taas_core::neuron::PoissonGenerator, writer: &mut TaasWriter) {
    writer.add_metadata("type", "PoissonGenerator");
    writer.add_metadata("n", &poisson.n_neurons().to_string());
    writer.add_f32("poisson_rates", "PoissonGenerator", poisson.rates());
    writer.add_scalar_f32("poisson_dt", "PoissonGenerator", poisson.dt());
}

// ─── TTFS encoder ────────────────────────────────────────

pub fn save_ttfs(ttfs: &taas_core::neuron::TtfsEncoder, writer: &mut TaasWriter, _n: usize) {
    writer.add_metadata("type", "TtfsEncoder");
    let fire_times: Vec<u64> = ttfs.fire_times.iter().map(|&x| x as u64).collect();
    let fired: Vec<i32> = ttfs.fired.iter().map(|&x| if x { 1 } else { 0 }).collect();
    writer.add_u64("ttfs_fire_times", "TtfsEncoder", &fire_times);
    writer.add_i32("ttfs_fired", "TtfsEncoder", &fired);
    writer.add_u64("ttfs_current_step", "TtfsEncoder", &[ttfs.current_step as u64]);
    writer.add_u64("ttfs_max_delay", "TtfsEncoder", &[ttfs.max_delay as u64]);
}

// ─── LIF-AT state ────────────────────────────────────────

pub fn save_lif_at_state(v: &[f32], theta: &[f32], n: usize, writer: &mut TaasWriter) {
    let n_save = n.min(v.len());
    writer.add_metadata("type", "LifNeuronState");
    writer.add_f32("lif_v", "LifNeuronState", &v[..n_save]);
    writer.add_f32("lif_theta", "LifNeuronState", &theta[..n_save]);
}

pub fn load_lif_at_state(reader: &TaasReader) -> Result<(Vec<f32>, Vec<f32>), String> {
    Ok((reader.get_f32("lif_v")?, reader.get_f32("lif_theta")?))
}

// ─── STDP weights ────────────────────────────────────────

pub fn save_stdp_weights(weights: &[f32], n: usize, writer: &mut TaasWriter) {
    writer.add_metadata("type", "STDPWeights");
    let n_save = n.min(weights.len());
    writer.add_f32("stdp_weights", "STDPWeights", &weights[..n_save]);
}

pub fn load_stdp_weights(reader: &TaasReader) -> Result<Vec<f32>, String> {
    reader.get_f32("stdp_weights")
}

// ─── CSR sparse matrix ───────────────────────────────────

pub fn save_csr(row_ptr: &[i32], col_indices: &[i32], values: &[f32], n_neurons: usize, writer: &mut TaasWriter) {
    writer.add_metadata("type", "CSRMatrix");
    writer.add_metadata("n_neurons", &n_neurons.to_string());
    writer.add_i32("csr_row_ptr", "CSRMatrix", row_ptr);
    writer.add_i32("csr_col_indices", "CSRMatrix", col_indices);
    writer.add_f32("csr_values", "CSRMatrix", values);
}

pub fn load_csr(reader: &TaasReader) -> Result<(Vec<i32>, Vec<i32>, Vec<f32>), String> {
    Ok((reader.get_i32("csr_row_ptr")?, reader.get_i32("csr_col_indices")?, reader.get_f32("csr_values")?))
}

// ─── InverseKuramotoDecoder ─────────────────────────────

pub fn save_inverse_kuramoto(dec: &taas_t3::InverseKuramotoDecoder, writer: &mut TaasWriter) {
    writer.add_metadata("decoder_type", "InverseKuramotoDecoder");
    writer.add_metadata("n", &dec.n.to_string());
    writer.add_f32("ikd_prev_amps", "InverseKuramotoDecoder", &dec.prev_amps);
    writer.add_f32("ikd_weights", "InverseKuramotoDecoder", &dec.weights);
    writer.add_scalar_f32("ikd_alpha", "InverseKuramotoDecoder", dec.alpha);
    writer.add_scalar_f32("ikd_drive_gain", "InverseKuramotoDecoder", dec.drive_gain);
    writer.add_scalar_f32("ikd_epsilon", "InverseKuramotoDecoder", dec.epsilon);
    writer.add_scalar_bool("ikd_initialised", "InverseKuramotoDecoder", dec.initialised);
}

pub fn load_inverse_kuramoto(reader: &TaasReader, n: usize) -> Result<taas_t3::InverseKuramotoDecoder, String> {
    let mut dec = taas_t3::InverseKuramotoDecoder::new(n, 1.0, 1.0);
    macro_rules! restore {
        ($field:ident, $name:expr) => {
            if let Ok(v) = reader.get_f32($name) {
                let n = v.len().min(dec.$field.len());
                dec.$field[..n].copy_from_slice(&v[..n]);
            }
        };
    }
    restore!(prev_amps, "ikd_prev_amps"); restore!(weights, "ikd_weights");
    dec.alpha = reader.get_scalar_f32("ikd_alpha").unwrap_or(1.0);
    dec.drive_gain = reader.get_scalar_f32("ikd_drive_gain").unwrap_or(1.0);
    dec.epsilon = reader.get_scalar_f32("ikd_epsilon").unwrap_or(0.1);
    dec.initialised = reader.get_scalar_bool("ikd_initialised").unwrap_or(false);
    Ok(dec)
}

// ─── StdpDecoder ─────────────────────────────────────────

pub fn save_stdp_decoder(dec: &taas_t3::StdpDecoder, writer: &mut TaasWriter) {
    writer.add_metadata("decoder_type", "StdpDecoder");
    writer.add_metadata("n", &dec.weights.len().to_string());
    writer.add_f32("sd_weights", "StdpDecoder", &dec.weights);
    writer.add_f32("sd_traces", "StdpDecoder", &dec.traces);
    writer.add_f32("sd_prev_states", "StdpDecoder", &dec.prev_states);
    writer.add_scalar_f32("sd_bias", "StdpDecoder", dec.bias);
    writer.add_scalar_f32("sd_lr", "StdpDecoder", dec.lr);
    writer.add_scalar_f32("sd_decay", "StdpDecoder", dec.decay);
    writer.add_scalar_f32("sd_sparsity", "StdpDecoder", dec.sparsity);
    writer.add_scalar_bool("sd_initialised", "StdpDecoder", dec.initialised);
}

pub fn load_stdp_decoder(reader: &TaasReader, n: usize) -> Result<taas_t3::StdpDecoder, String> {
    let mut dec = taas_t3::StdpDecoder::new(n, 0.01, 0.9, 1e-5);
    macro_rules! restore {
        ($field:ident, $name:expr) => {
            if let Ok(v) = reader.get_f32($name) {
                let n = v.len().min(dec.$field.len());
                dec.$field[..n].copy_from_slice(&v[..n]);
            }
        };
    }
    restore!(weights, "sd_weights"); restore!(traces, "sd_traces"); restore!(prev_states, "sd_prev_states");
    dec.bias = reader.get_scalar_f32("sd_bias").unwrap_or(0.0);
    dec.lr = reader.get_scalar_f32("sd_lr").unwrap_or(0.01);
    dec.decay = reader.get_scalar_f32("sd_decay").unwrap_or(0.9);
    dec.sparsity = reader.get_scalar_f32("sd_sparsity").unwrap_or(1e-5);
    dec.initialised = reader.get_scalar_bool("sd_initialised").unwrap_or(false);
    Ok(dec)
}

// ─── PstDecoder ──────────────────────────────────────────

pub fn save_pst_decoder(dec: &taas_t3::PstDecoder, writer: &mut TaasWriter) {
    writer.add_metadata("decoder_type", "PstDecoder");
    writer.add_metadata("feat_dim", &dec.feat_dim.to_string());
    writer.add_metadata("L", &dec.L.to_string());
    writer.add_f32("pst_weights", "PstDecoder", &dec.weights);
    writer.add_scalar_f32("pst_bias", "PstDecoder", dec.bias);
    writer.add_scalar_f32("pst_alpha", "PstDecoder", dec.alpha);
}

pub fn load_pst_decoder(reader: &TaasReader, feat_dim: usize, l: usize) -> Result<taas_t3::PstDecoder, String> {
    let mut dec = taas_t3::PstDecoder::new(feat_dim, l, 0.01);
    if let Ok(v) = reader.get_f32("pst_weights") {
        let n = v.len().min(dec.weights.len());
        dec.weights[..n].copy_from_slice(&v[..n]);
    }
    dec.bias = reader.get_scalar_f32("pst_bias").unwrap_or(0.0);
    dec.alpha = reader.get_scalar_f32("pst_alpha").unwrap_or(0.01);
    Ok(dec)
}


