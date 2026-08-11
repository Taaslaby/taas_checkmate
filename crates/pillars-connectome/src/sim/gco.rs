//! Brain simulator module.

use super::BrainState;
use crate::data::ConnectomeData;
use crate::types::GcoParams;

// ─── GCO simulation ──────────────────────────────────

/// Simulate `n_steps` of graph-coupled oscillator dynamics on a connectome.
///
/// Канон: `pillars::kernels::phase::gco_step(v, u, h_real, h_imag, i_ext, speed, dt)`
pub fn sim_gco(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &GcoParams,
    external_I: &[f32],
    n_steps: usize,
) -> Vec<i32> {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let spike_buf_len = n_steps * n;
    let mut spike_buffer = vec![0i32; spike_buf_len];
    let dt = params.dt_ms / 1000.0;

    // H*psi scratch buffers (must match v size)
    let mut h_real = vec![0.0; state.n_padded];
    let mut h_imag = vec![0.0; state.n_padded];
    let speed = vec![1.0; state.n_padded];

    for step in 0..n_steps {
        // 1. Build Hamiltonian from CSR + state
        // h_real[i] = Σ W[j,i] * u[j], h_imag[i] = -Σ W[j,i] * v[j]
        h_real.fill(0.0);
        h_imag.fill(0.0);
        let max_syn = connectome.col_indices.len().min(connectome.values.len());
        for pre in 0..n_neurons {
            let start = connectome.row_ptr[pre].max(0) as usize;
            let end = (connectome.row_ptr[pre + 1] as usize).min(max_syn);
            let mult = if pre < connectome.nt_types.len() {
                connectome.nt_types[pre]
            } else {
                1.0
            };
            for j in start..end {
                let post = connectome.col_indices[j] as usize;
                let w = connectome.values[j] * mult;
                if post < n {
                    h_real[post] += w * state.u[pre];
                    h_imag[post] -= w * state.v[pre];
                }
            }
        }

        // 2. Set external input
        for i in 0..n.min(external_I.len()) {
            state.I_buf[i] = external_I[i];
        }

        // 3. GCO step
        pillars::kernels::phase::gco_step(
            &mut state.v,
            &mut state.u,
            &h_real,
            &h_imag,
            &state.I_buf,
            &speed,
            dt,
        );

        // 4. "Spike" if |v+iu|² > threshold
        for i in 0..n_neurons {
            let prob = state.v[i] * state.v[i] + state.u[i] * state.u[i];
            if prob > 0.5 {
                let idx = step * n_neurons + i;
                if idx < spike_buffer.len() {
                    spike_buffer[idx] = 1;
                }
            }
        }
    }

    spike_buffer
}

pub fn sim_gco_record(
    connectome: &ConnectomeData,
    state: &mut BrainState,
    params: &GcoParams,
    external_I: &[f32],
    n_changes: usize,
    steps_per_input: usize,
    record_every: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = connectome.n_neurons;
    let n_neurons = connectome.row_ptr.len() - 1;
    let dt = params.dt_ms / 1000.0;
    let re = if record_every == 0 {
        steps_per_input
    } else {
        record_every
    };
    let recs_per_input = steps_per_input / re;
    let n_records = n_changes * recs_per_input;

    let mut v_traj = vec![0.0f32; n_records * n];
    let mut u_traj = vec![0.0f32; n_records * n];

    let mut h_real = vec![0.0; state.n_padded];
    let mut h_imag = vec![0.0; state.n_padded];
    let speed = vec![1.0; state.n_padded];

    let mut rec_idx = 0;
    for change in 0..n_changes {
        let base = change * n;
        for i in 0..n.min(external_I.len().saturating_sub(base)) {
            state.I_buf[i] = external_I[base + i];
        }

        for step in 0..steps_per_input {
            h_real.fill(0.0);
            h_imag.fill(0.0);
            let max_syn = connectome.col_indices.len().min(connectome.values.len());
            for pre in 0..n_neurons {
                let start = connectome.row_ptr[pre].max(0) as usize;
                let end = (connectome.row_ptr[pre + 1] as usize).min(max_syn);
                let mult = if pre < connectome.nt_types.len() {
                    connectome.nt_types[pre]
                } else {
                    1.0
                };
                for j in start..end {
                    let post = connectome.col_indices[j] as usize;
                    let w = connectome.values[j] * mult;
                    if post < n {
                        h_real[post] += w * state.u[pre];
                        h_imag[post] -= w * state.v[pre];
                    }
                }
            }

            pillars::kernels::phase::gco_step(
                &mut state.v,
                &mut state.u,
                &h_real,
                &h_imag,
                &state.I_buf,
                &speed,
                dt,
            );

            if (step + 1) % re == 0 {
                let out_base = rec_idx * n;
                for i in 0..n {
                    v_traj[out_base + i] = state.v[i];
                    u_traj[out_base + i] = state.u[i];
                }
                rec_idx += 1;
            }
        }
    }

    (v_traj, u_traj)
}

