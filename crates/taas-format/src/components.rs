//! Component serialization helpers for .taas format.
//!
//! Each save/load pair handles one model type. Following the pattern established
//! in the original monolithic lib.rs, every physical quantity is stored as a
//! named tensor component (f32[], i32[], i64[], u64[]) inside the .taas archive.

use crate::reader::TaasReader;
use crate::writer::TaasWriter;

// ─── Brain IO ────────────────────────────────────────────

/// Save BrainIO as JSON metadata.
pub fn save_brain_io(io: &pillars_connectome::BrainIO, writer: &mut TaasWriter) {
    writer.add_metadata("type", "BrainIO");
    let json = serde_json::to_string(io).unwrap_or_else(|_| "null".to_string());
    writer.add_metadata("brain_io", &json);
}

/// Load BrainIO from JSON metadata.
pub fn load_brain_io(reader: &TaasReader) -> Result<pillars_connectome::BrainIO, String> {
    reader
        .metadata()
        .get("brain_io")
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
pub fn save_connectome(conn: &pillars_connectome::ConnectomeData, writer: &mut TaasWriter) {
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
pub fn load_connectome(reader: &TaasReader) -> Result<pillars_connectome::ConnectomeData, String> {
    let _n_neurons = reader
        .metadata()
        .get("n_neurons")
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or_else(|| "n_neurons not found in metadata".to_string())?;
    let row_ptr = reader.get_i32("row_ptr")?;
    let col_indices = reader.get_i32("col_indices")?;
    let values = reader.get_f32("values")?;
    let nt_types = reader.get_f32("nt_types")?;
    // receptor_types is optional — old archives (pre receptor_types) lack it,
    // returning an empty Vec (matches ConnectomeData::new default).
    let receptor_types = reader.get_u8("receptor_types").unwrap_or_default();
    let groups: Vec<pillars_connectome::NeuronGroup> = reader
        .metadata()
        .get("groups")
        .and_then(|g| serde_json::from_str(g).ok())
        .unwrap_or_default();
    let group_refs: Vec<(&str, Vec<usize>)> = groups
        .iter()
        .map(|g| (g.name.as_str(), g.indices.clone()))
        .collect();
    let mut conn =
        pillars_connectome::ConnectomeData::new(row_ptr, col_indices, values, nt_types, group_refs);
    conn.receptor_types = receptor_types;
    Ok(conn)
}

// ─── TriOscillator ──────────────────────────────────────

pub fn save_trioscilator(tri: &taas_t3::TriOscillator, writer: &mut TaasWriter) {
    writer.add_metadata("type", "TriOscillator");
    writer.add_metadata("n", &tri.n.to_string());
    writer.add_metadata("format", "v2_grouped");

    // Group all 27 Vec<f32> fields into one tensor [27, n]:
    // [th1..th3, prev_th1..prev_th3, r1..r3, prev_r1..prev_r3,
    //  omega1..omega3, drive1..drive3, sin1..cos3, coupling1..coupling3]
    let n = tri.n;
    let mut buf = Vec::with_capacity(27 * n);
    macro_rules! push_all {
        ($($field:ident),*) => {
            $(buf.extend_from_slice(&tri.$field[..n]);)*
        };
    }
    push_all!(
        th1, th2, th3, prev_th1, prev_th2, prev_th3, r1, r2, r3, prev_r1, prev_r2, prev_r3, omega1,
        omega2, omega3, drive1, drive2, drive3, sin1, cos1, sin2, cos2, sin3, cos3, coupling1,
        coupling2, coupling3
    );
    writer.add_f32_shape("tri_state", "TriOscillator", &buf, vec![27, n]);

    // Scalars stay as individual components
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

/// Restore a TriOscillator field from the grouped tensor.
/// `buf` is flat [27 × n], each chunk has `n` elements.
macro_rules! restore_tri_field {
    ($tri:expr, $buf:expr, $n:expr, $idx:expr, $field:ident) => {
        let start = $idx * $n;
        $tri.$field[..$n].copy_from_slice(&$buf[start..start + $n]);
    };
}

pub fn load_trioscilator(
    reader: &TaasReader,
    n: usize,
) -> Result<taas_t3::TriOscillator<taas_t3::Avx2Backend>, String> {
    use taas_t3::{Avx2Backend, TriOscillator};
    let mut tri = TriOscillator::new(n, 0.5, 1.5, Avx2Backend);

    // Try new grouped format first (v2_grouped)
    if let Ok(grouped) = reader.get_f32("tri_state") {
        if grouped.len() >= 27 * n {
            restore_tri_field!(tri, grouped, n, 0, th1);
            restore_tri_field!(tri, grouped, n, 1, th2);
            restore_tri_field!(tri, grouped, n, 2, th3);
            restore_tri_field!(tri, grouped, n, 3, prev_th1);
            restore_tri_field!(tri, grouped, n, 4, prev_th2);
            restore_tri_field!(tri, grouped, n, 5, prev_th3);
            restore_tri_field!(tri, grouped, n, 6, r1);
            restore_tri_field!(tri, grouped, n, 7, r2);
            restore_tri_field!(tri, grouped, n, 8, r3);
            restore_tri_field!(tri, grouped, n, 9, prev_r1);
            restore_tri_field!(tri, grouped, n, 10, prev_r2);
            restore_tri_field!(tri, grouped, n, 11, prev_r3);
            restore_tri_field!(tri, grouped, n, 12, omega1);
            restore_tri_field!(tri, grouped, n, 13, omega2);
            restore_tri_field!(tri, grouped, n, 14, omega3);
            restore_tri_field!(tri, grouped, n, 15, drive1);
            restore_tri_field!(tri, grouped, n, 16, drive2);
            restore_tri_field!(tri, grouped, n, 17, drive3);
            restore_tri_field!(tri, grouped, n, 18, sin1);
            restore_tri_field!(tri, grouped, n, 19, cos1);
            restore_tri_field!(tri, grouped, n, 20, sin2);
            restore_tri_field!(tri, grouped, n, 21, cos2);
            restore_tri_field!(tri, grouped, n, 22, sin3);
            restore_tri_field!(tri, grouped, n, 23, cos3);
            restore_tri_field!(tri, grouped, n, 24, coupling1);
            restore_tri_field!(tri, grouped, n, 25, coupling2);
            restore_tri_field!(tri, grouped, n, 26, coupling3);
        }
    } else {
        // Fallback: load old per-field format
        macro_rules! restore {
            ($field:ident, $name:expr) => {
                if let Ok(v) = reader.get_f32($name) {
                    let n = v.len().min(tri.$field.len());
                    tri.$field[..n].copy_from_slice(&v[..n]);
                }
            };
        }
        restore!(th1, "th1");
        restore!(th2, "th2");
        restore!(th3, "th3");
        restore!(prev_th1, "prev_th1");
        restore!(prev_th2, "prev_th2");
        restore!(prev_th3, "prev_th3");
        restore!(r1, "r1");
        restore!(r2, "r2");
        restore!(r3, "r3");
        restore!(prev_r1, "prev_r1");
        restore!(prev_r2, "prev_r2");
        restore!(prev_r3, "prev_r3");
        restore!(omega1, "omega1");
        restore!(omega2, "omega2");
        restore!(omega3, "omega3");
        restore!(drive1, "drive1");
        restore!(drive2, "drive2");
        restore!(drive3, "drive3");
        restore!(sin1, "sin1");
        restore!(cos1, "cos1");
        restore!(sin2, "sin2");
        restore!(cos2, "cos2");
        restore!(sin3, "sin3");
        restore!(cos3, "cos3");
        restore!(coupling1, "coupling1");
        restore!(coupling2, "coupling2");
        restore!(coupling3, "coupling3");
    }

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
        if !bias.is_empty() {
            dec.bias[0] = bias[0];
        }
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

pub fn load_kalman_decoder(
    reader: &TaasReader,
    n: usize,
) -> Result<taas_t3::KalmanDecoder, String> {
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
    writer.add_f32(
        "rl_trace_critic",
        "ActorCriticAgent",
        &agent.trace_critic_val,
    );
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
pub fn load_actor_critic_agent(
    reader: &TaasReader,
) -> Result<taas_train::rl::ActorCriticAgent, String> {
    let n_input = reader
        .metadata()
        .get("n_input")
        .and_then(|s| s.parse().ok())
        .ok_or("n_input not found")?;
    let n_hidden = reader
        .metadata()
        .get("n_hidden")
        .and_then(|s| s.parse().ok())
        .ok_or("n_hidden not found")?;
    let n_actions = reader
        .metadata()
        .get("n_actions")
        .and_then(|s| s.parse().ok())
        .ok_or("n_actions not found")?;
    let n_steps = reader
        .metadata()
        .get("n_steps")
        .and_then(|s| s.parse().ok())
        .ok_or("n_steps not found")?;
    let dt = reader.get_scalar_f32("rl_dt").unwrap_or(1.0);
    let mut agent =
        taas_train::rl::ActorCriticAgent::new(n_input, n_hidden, n_actions, n_steps, dt, 42);
    macro_rules! restore {
        ($field:ident, $name:expr) => {
            if let Ok(v) = reader.get_f32($name) {
                let n = v.len().min(agent.$field.len());
                agent.$field[..n].copy_from_slice(&v[..n]);
            }
        };
    }
    restore!(w_ih, "rl_w_ih");
    restore!(w_ho, "rl_w_ho");
    restore!(w_ic, "rl_w_ic");
    restore!(elig_ih, "rl_elig_ih");
    restore!(elig_ho, "rl_elig_ho");
    restore!(elig_ic, "rl_elig_ic");
    restore!(trace_input, "rl_trace_input");
    restore!(trace_hidden, "rl_trace_hidden");
    restore!(trace_output, "rl_trace_output");
    restore!(trace_critic_val, "rl_trace_critic");
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

pub fn save_csr(
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    n_neurons: usize,
    writer: &mut TaasWriter,
) {
    writer.add_metadata("type", "CSRMatrix");
    writer.add_metadata("n_neurons", &n_neurons.to_string());
    writer.add_i32("csr_row_ptr", "CSRMatrix", row_ptr);
    writer.add_i32("csr_col_indices", "CSRMatrix", col_indices);
    writer.add_f32("csr_values", "CSRMatrix", values);
}

pub fn load_csr(reader: &TaasReader) -> Result<(Vec<i32>, Vec<i32>, Vec<f32>), String> {
    Ok((
        reader.get_i32("csr_row_ptr")?,
        reader.get_i32("csr_col_indices")?,
        reader.get_f32("csr_values")?,
    ))
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

pub fn load_inverse_kuramoto(
    reader: &TaasReader,
    n: usize,
) -> Result<taas_t3::InverseKuramotoDecoder, String> {
    let mut dec = taas_t3::InverseKuramotoDecoder::new(n, 1.0, 1.0);
    macro_rules! restore {
        ($field:ident, $name:expr) => {
            if let Ok(v) = reader.get_f32($name) {
                let n = v.len().min(dec.$field.len());
                dec.$field[..n].copy_from_slice(&v[..n]);
            }
        };
    }
    restore!(prev_amps, "ikd_prev_amps");
    restore!(weights, "ikd_weights");
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
    restore!(weights, "sd_weights");
    restore!(traces, "sd_traces");
    restore!(prev_states, "sd_prev_states");
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

pub fn load_pst_decoder(
    reader: &TaasReader,
    feat_dim: usize,
    l: usize,
) -> Result<taas_t3::PstDecoder, String> {
    let mut dec = taas_t3::PstDecoder::new(feat_dim, l, 0.01);
    if let Ok(v) = reader.get_f32("pst_weights") {
        let n = v.len().min(dec.weights.len());
        dec.weights[..n].copy_from_slice(&v[..n]);
    }
    dec.bias = reader.get_scalar_f32("pst_bias").unwrap_or(0.0);
    dec.alpha = reader.get_scalar_f32("pst_alpha").unwrap_or(0.01);
    Ok(dec)
}

// ─── Pillars-native snapshot: NeuralPopulation<M> ─────────

/// Save a pillars `NeuralPopulation<M>` to a `.taas` archive.
///
/// Components: `model` (type name), `n`, `capacity`, `N_PARAMS`, `N_STATE`
/// as metadata; one f32 tensor per param column (`p_{c}`) and state column
/// (`s_{c}`) — the SoA layout pillars uses, so `load_population` restores it
/// exactly.
pub fn save_population<M: pillars::zoo::Model + 'static>(
    pop: &pillars::NeuralPopulation<M>,
    writer: &mut TaasWriter,
) {
    writer.add_metadata("type", "Population");
    writer.add_metadata("model", std::any::type_name::<M>());
    writer.add_metadata("n", &pop.len().to_string());
    writer.add_metadata("capacity", &pop.capacity().to_string());
    writer.add_metadata("n_params", &M::N_PARAMS.to_string());
    writer.add_metadata("n_state", &M::N_STATE.to_string());
    for c in 0..M::N_PARAMS {
        writer.add_f32(&format!("p_{c}"), "Population", pop.param_slice(c));
    }
    for c in 0..M::N_STATE {
        writer.add_f32(&format!("s_{c}"), "Population", pop.state_slice(c));
    }
}

/// Load a pillars `NeuralPopulation<M>` from a `.taas` archive.
///
/// `M` must match what was saved (`n_params`/`n_state` metadata are checked).
pub fn load_population<M: pillars::zoo::Model + 'static>(
    reader: &TaasReader,
) -> Result<pillars::NeuralPopulation<M>, String> {
    let n: usize = reader
        .metadata()
        .get("n")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "n not found".to_string())?;
    let np: usize = reader
        .metadata()
        .get("n_params")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "n_params not found".to_string())?;
    let ns: usize = reader
        .metadata()
        .get("n_state")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "n_state not found".to_string())?;
    if np != M::N_PARAMS {
        return Err(format!(
            "n_params mismatch: archive={np}, model={}",
            M::N_PARAMS
        ));
    }
    if ns != M::N_STATE {
        return Err(format!(
            "n_state mismatch: archive={ns}, model={}",
            M::N_STATE
        ));
    }
    let mut pop = pillars::NeuralPopulation::<M>::new(n);
    for c in 0..M::N_PARAMS {
        let v = reader.get_f32(&format!("p_{c}"))?;
        pop.param_slice_mut(c).copy_from_slice(&v);
    }
    for c in 0..M::N_STATE {
        let v = reader.get_f32(&format!("s_{c}"))?;
        pop.state_slice_mut(c).copy_from_slice(&v);
    }
    Ok(pop)
}

// ─── Pillars-native snapshot: SparseGraph ────────────────

/// Save a pillars `SparseGraph` (CSR) to a `.taas` archive.
pub fn save_sparse_graph(graph: &pillars::api::SparseGraph, writer: &mut TaasWriter) {
    writer.add_metadata("type", "SparseGraph");
    writer.add_metadata("n_neurons", &graph.num_neurons().to_string());
    writer.add_metadata("n_synapses", &graph.num_synapses().to_string());
    let row_ptr: Vec<i64> = graph.row_ptr().iter().map(|&v| v as i64).collect();
    let col_idx: Vec<i64> = graph.col_idx().iter().map(|&v| v as i64).collect();
    writer.add_i64("sg_row_ptr", "SparseGraph", &row_ptr);
    writer.add_i64("sg_col_idx", "SparseGraph", &col_idx);
    writer.add_f32("sg_weights", "SparseGraph", graph.weights());
    writer.add_u8("sg_delays", "SparseGraph", graph.delays());
}

/// Load a pillars `SparseGraph` (CSR) from a `.taas` archive.
pub fn load_sparse_graph(reader: &TaasReader) -> Result<pillars::api::SparseGraph, String> {
    let row_ptr: Vec<u32> = reader
        .get_i64("sg_row_ptr")?
        .iter()
        .map(|&v| v as u32)
        .collect();
    let col_idx: Vec<u32> = reader
        .get_i64("sg_col_idx")?
        .iter()
        .map(|&v| v as u32)
        .collect();
    let weights = reader.get_f32("sg_weights")?;
    let delays = reader.get_u8("sg_delays")?;
    Ok(pillars::api::SparseGraph::from_csr(
        row_ptr, col_idx, &weights, &delays,
    ))
}
