use wide::f32x8;

// ═══════════════════════════════════════════════════════════
//  Hebbian CSR plasticity (волна 5 — canonical port of the legacy
//  `izhikevich::hebbian_csr_step`, bit-identical)
// ═══════════════════════════════════════════════════════════

/// Hebbian plasticity update on CSR synaptic weights using slow states.
///
/// `dW = η · m_pre · m_post − α · W · dt`, clamped to [-1, 1].
///
/// `values` — CSR synaptic weights (mutated in-place), `col_indices` — CSR
/// column indices, `row_ptr` — CSR row pointer `[n_neurons + 1]`, `m` — slow
/// state per neuron (recent firing-rate proxy, e.g. from `IzhikevichDualModel`).
///
/// # Reference
/// Sun et al. (2026) — Hebbian plasticity on 15M synapses leads to emergent
/// individuality (used by `connectome/sim.rs` + `connectome/runner.rs`).
pub fn hebbian_csr_step(
    values: &mut [f32],
    col_indices: &[i32],
    row_ptr: &[i32],
    m: &[f32],
    n_neurons: usize,
    lr: f32,
    decay: f32,
    dt: f32,
) {
    let n_synapses = values.len();
    assert_eq!(
        col_indices.len(),
        n_synapses,
        "hebbian_csr_step: col_indices length mismatch"
    );
    assert!(
        row_ptr.len() > n_neurons,
        "hebbian_csr_step: row_ptr too short"
    );
    assert!(m.len() >= n_neurons, "hebbian_csr_step: m too short");

    let lr_s = f32x8::splat(lr);
    let decay_dt_s = f32x8::splat(decay * dt);
    let min_w = f32x8::splat(-1.0);
    let max_w = f32x8::splat(1.0);

    for pre in 0..n_neurons {
        let start = row_ptr[pre].max(0) as usize;
        let end = (row_ptr[pre + 1] as usize).min(n_synapses);
        if start >= end {
            continue;
        }

        let m_pre = f32x8::splat(m[pre]);

        for off in (start..end).step_by(8) {
            let batch_end = (off + 8).min(end);
            let batch_n = batch_end - off;

            // Build post_m vector via plain array (8× faster than match-arms per element)
            let mut post_arr = [0.0f32; 8];
            let mut w_arr = [0.0f32; 8];
            for j in 0..batch_n {
                let post_idx = col_indices[off + j] as usize;
                post_arr[j] = if post_idx < n_neurons {
                    m[post_idx]
                } else {
                    0.0
                };
                w_arr[j] = values[off + j];
            }
            // Fill remaining with last valid value (avoids zero-bias on partial blocks)
            if batch_n < 8 && batch_n > 0 {
                let last_m = post_arr[batch_n - 1];
                let last_w = w_arr[batch_n - 1];
                for j in batch_n..8 {
                    post_arr[j] = last_m;
                    w_arr[j] = last_w;
                }
            }
            let post_m = f32x8::from(post_arr);
            let ww = f32x8::from(w_arr);

            // Hebbian update: dw = lr * m_pre * m_post - decay * w * dt
            let hebb = m_pre * post_m;
            let dw = lr_s * hebb - decay_dt_s * ww;
            let new_w = (ww + dw).max(min_w).min(max_w);

            let new_arr = new_w.to_array();
            values[off..off + batch_n].copy_from_slice(&new_arr[..batch_n]);
        }
    }
}
