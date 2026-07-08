//! Connectome + Bridge + T3 → chase training pipeline.
//!
//! Полный цикл на Rust (с опциональным GPU):
//! ```
//! Connectome (LIF) → спайки → Bridge → T3 → PST → decoder → 2D позиция
//!                                          ↑
//!                                  BPTT + SGD (bridge + decoder)
//! ```
//!
//! Feature gate: `--features connectome` (CPU) or `--features cuda` (GPU).

use crate::spigrad::SpigradBridge;
use crate::spike::{t3spike_forward, t3spike_backward_per_step};
use crate::t3::pst_linear_dim;

/// Configuration for connectome chase training.
pub struct ChaseConfig {
    // T3 oscillator params
    pub n_osc: usize,
    pub n_steps: usize,
    pub dt: f32,
    pub drive_gain: f32,
    pub alpha: f32,
    pub freq_min: f32,
    pub freq_max: f32,
    pub coupling_w12: f32,
    pub coupling_w13: f32,
    pub coupling_w23: f32,

    // Bridge
    pub spike_dim: usize,
    pub lr_bridge: f32,
    pub lr_decoder: f32,
    pub n_epochs: usize,
    pub grad_clip: f32,

    // Connectome
    pub dt_ms: f32,
    pub readout_idx: Vec<usize>,

    // LIF params
    pub tau_m: f32,
    pub v_rest: f32,
    pub v_thresh: f32,
    pub v_reset: f32,
}

impl ChaseConfig {
    pub fn pst_dim(&self) -> usize {
        pst_linear_dim(self.n_osc)
    }

    pub fn drive_dim(&self) -> usize {
        3 * self.n_osc
    }
}

/// Result of connectome chase training.
pub struct ChaseResult {
    pub bridge: SpigradBridge,
    pub decoder_w: Vec<f32>,
    pub decoder_b: Vec<f32>,
    pub loss_history: Vec<f32>,
}

/// Train a SpigradBridge to chase a moving target using connectome spikes.
///
/// `conn`: CSR connectome with LIF dynamics
/// `external_I_fn`: function that returns external input current for each episode
/// `seed`: RNG seed for target random walk
///
/// Returns trained bridge + linear decoder.
pub fn train_connectome_chase<F>(
    conn: &taas_connectome::data::ConnectomeData,
    config: &ChaseConfig,
    mut external_I_fn: F,
    seed: u64,
) -> ChaseResult
where
    F: FnMut() -> Vec<f32>,
{
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let n = conn.n_neurons;
    let pd = config.pst_dim();
    let dd = config.drive_dim();

    // ── Init bridge ──
    let mut bridge = crate::spigrad::SpigradBridge::new(
        config.spike_dim, config.n_osc, config.dt,
        config.drive_gain, config.alpha,
        config.freq_min, config.freq_max,
        config.coupling_w12, config.coupling_w13, config.coupling_w23,
    );

    // ── Init decoder ──
    let pst_dim = config.pst_dim();
    let mut decode_w: Vec<f32> = (0..pst_dim * 2).map(|_| (rng.gen::<f32>() - 0.5) * 0.2).collect();
    let mut decode_b = vec![0.0; 2];

    // ── Init LIF state ──
    let mut state = taas_connectome::sim::BrainState::new(n, taas_connectome::sim::NeuronModel::LIF);
    let lif_params = taas_connectome::sim::LifParams::new(
        &vec![config.tau_m; n], &vec![config.v_rest; n],
        &vec![config.v_thresh; n], &vec![config.v_reset; n],
        config.dt_ms, 2.0,
    );

    let mut loss_history = Vec::with_capacity(config.n_epochs);
    let mut target = [0.0f32; 2];
    let mut target_vel = [0.0f32; 2];

    for ep in 0..config.n_epochs {
        // ── External input ──
        let external_I = external_I_fn();

        // ── Simulate connectome → spikes ──
        #[cfg(feature = "cuda")]
        let spike_buf = {
            use taas_connectome::sim::gpu;
            let tau_m_v = vec![config.tau_m; n];
            let v_rest_v = vec![config.v_rest; n];
            let v_thresh_v = vec![config.v_thresh; n];
            let v_reset_v = vec![config.v_reset; n];
            gpu::sim_lif_gpu(conn, &mut state, config.dt_ms, &external_I,
                             config.n_steps, &tau_m_v, &v_rest_v, &v_thresh_v, &v_reset_v)
        };
        #[cfg(not(feature = "cuda"))]
        let spike_buf = taas_connectome::sim::sim_lif(
            conn, &mut state, &lif_params, &external_I, config.n_steps
        );

        // ── Extract readout spikes → bridge input ──
        let n_steps = config.n_steps;
        let sd = config.spike_dim;
        let mut spike_in = vec![0.0f32; n_steps * sd];
        for t in 0..n_steps {
            let base = t * n;
            for (j, &idx) in config.readout_idx.iter().enumerate() {
                spike_in[t * sd + j] = spike_buf[base + idx] as f32;
            }
        }

        // ── Bridge forward → T3 → PST → decode position ──
        let (_pst_final, traj) = t3spike_forward(
            config.n_osc, n_steps, config.dt,
            config.drive_gain, config.alpha,
            config.freq_min, config.freq_max,
            config.coupling_w12, config.coupling_w13, config.coupling_w23,
            &bridge.compute_drives(&spike_in, n_steps),
        );

        // Full per-step PST matrix from trajectory
        let pst = crate::spike::t3spike_compute_pst_matrix(&traj);

        // Decode: pst [n_steps × pst_dim] → position [n_steps × 2]
        let mut pos = vec![0.0f32; n_steps * 2];
        for t in 0..n_steps {
            let base_p = t * pd;
            let base_m = t * 2;
            for j in 0..2 {
                let mut acc = decode_b[j];
                for i in 0..pd {
                    acc += pst[base_p + i] * decode_w[i * 2 + j];
                }
                pos[base_m + j] = acc;
            }
        }

        // ── Target random walk ──
        target_vel[0] += (rng.gen::<f32>() - 0.5) * 0.12;
        target_vel[1] += (rng.gen::<f32>() - 0.5) * 0.12;
        target_vel[0] *= 0.92;
        target_vel[1] *= 0.92;
        target[0] = (target[0] + target_vel[0] * 0.15).clamp(-1.8, 1.8);
        target[1] = (target[1] + target_vel[1] * 0.15).clamp(-1.8, 1.8);

        // ── Loss: MSE over all steps ──
        let mut loss = 0.0f32;
        let mut err = vec![0.0f32; n_steps * 2];
        for t in 0..n_steps {
            let base = t * 2;
            err[base] = pos[base] - target[0];
            err[base + 1] = pos[base + 1] - target[1];
            loss += err[base] * err[base] + err[base + 1] * err[base + 1];
        }
        loss *= 0.5 / n_steps as f32;
        loss_history.push(loss);

        // ── Per-step gradient for backward ──
        let inv_n = 1.0 / n_steps as f32;
        let mut per_step_grad = vec![0.0f32; n_steps * 2];
        let mut per_step_pst = vec![0.0f32; n_steps * pd];
        for t in 0..n_steps {
            let base_e = t * 2;
            let base_p = t * pd;
            let ex = err[base_e] * inv_n;
            let ey = err[base_e + 1] * inv_n;
            per_step_grad[base_e] = ex;
            per_step_grad[base_e + 1] = ey;
            for i in 0..pd {
                per_step_pst[base_p + i] = ex * decode_w[i * 2] + ey * decode_w[i * 2 + 1];
            }
        }

        // ── Backward through T3 → grad_drives ──
        let grad_pst_final = vec![0.0f32; pd];
        let grad_drives = t3spike_backward_per_step(
            &grad_pst_final, &traj, Some(&per_step_pst),
        );

        // ── Project grad_drives → bridge weights ──
        let mut grad_w = vec![0.0f32; sd * dd];
        let mut grad_b = vec![0.0f32; dd];
        for t in 0..n_steps {
            let base_d = t * dd;
            let base_s = t * sd;
            for j in 0..dd {
                let g = grad_drives[base_d + j];
                grad_b[j] += g;
                for i in 0..sd {
                    grad_w[i * dd + j] += g * spike_in[base_s + i];
                }
            }
        }

        // ── Clip gradients ──
        let clip = config.grad_clip;
        for w in grad_w.iter_mut() { *w = w.clamp(-clip, clip); }
        for b in grad_b.iter_mut() { *b = b.clamp(-clip, clip); }

        // ── Update bridge ──
        bridge.update(&grad_w, &grad_b, config.lr_bridge);

        // ── Update decoder ──
        for t in 0..n_steps {
            let base_p = t * pd;
            let base_m = t * 2;
            let ex = per_step_grad[base_m];
            let ey = per_step_grad[base_m + 1];
            for i in 0..pd {
                decode_w[i * 2] -= config.lr_decoder * ex * pst[base_p + i];
                decode_w[i * 2 + 1] -= config.lr_decoder * ey * pst[base_p + i];
            }
            decode_b[0] -= config.lr_decoder * ex;
            decode_b[1] -= config.lr_decoder * ey;
        }
    }

    ChaseResult { bridge, decoder_w: decode_w, decoder_b: decode_b, loss_history }
}
