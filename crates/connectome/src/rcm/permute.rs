/// Вычислить bandwidth матрицы: max |i - j| для всех ненулевых W(i,j).
pub fn bandwidth(row_ptr: &[i32], col_indices: &[i32]) -> usize {
    if row_ptr.len() < 2 {
        return 0;
    }
    let n = row_ptr.len() - 1;
    let max_syn = col_indices
        .len()
        .min(row_ptr.last().copied().unwrap_or(0).max(0) as usize);
    let mut bw = 0usize;
    for i in 0..n {
        let start = row_ptr[i].max(0) as usize;
        let end = (row_ptr[i + 1] as usize).min(max_syn);
        for j in start..end {
            let post = col_indices[j].max(0) as usize;
            if post < n {
                let diff = i.abs_diff(post);
                if diff > bw {
                    bw = diff;
                }
            }
        }
    }
    bw
}

/// Средняя ширина полосы (mean bandwidth) для оценки.
pub fn mean_bandwidth(row_ptr: &[i32], col_indices: &[i32]) -> f64 {
    if row_ptr.len() < 2 {
        return 0.0;
    }
    let n = row_ptr.len() - 1;
    let max_syn = col_indices
        .len()
        .min(row_ptr.last().copied().unwrap_or(0).max(0) as usize);
    let mut total = 0u64;
    let mut count = 0u64;
    for i in 0..n {
        let start = row_ptr[i].max(0) as usize;
        let end = (row_ptr[i + 1] as usize).min(max_syn);
        for j in start..end {
            let post = col_indices[j].max(0) as usize;
            if post < n {
                let diff = i.abs_diff(post);
                total += diff as u64;
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}

/// Применить перестановку к CSR матрице + nt_types.
pub fn apply_permutation_to_csr(
    row_ptr: &[i32],
    col_indices: &[i32],
    values: &[f32],
    nt_types: &[f32],
    perm: &[usize],
) -> (Vec<i32>, Vec<i32>, Vec<f32>, Vec<f32>) {
    let n = row_ptr.len() - 1;
    let n_syn = values.len();
    assert_eq!(perm.len(), n, "perm length must match n_neurons");

    let mut new_row_ptr = vec![0i32; n + 1];

    for old_i in 0..n {
        let new_i = perm[old_i];
        let start = row_ptr[old_i].max(0) as usize;
        let end = (row_ptr[old_i + 1].max(0) as usize).min(n_syn);
        new_row_ptr[new_i + 1] += (end - start) as i32;
    }

    let mut offset = 0i32;
    for cnt in new_row_ptr.iter_mut().skip(1) {
        let c = *cnt;
        *cnt = offset + c;
        offset = *cnt;
    }
    let total_syn = offset as usize;
    let mut new_col_indices = vec![0i32; total_syn];
    let mut new_values = vec![0.0f32; total_syn];

    let mut cursor: Vec<usize> = new_row_ptr.iter().map(|&rp| rp.max(0) as usize).collect();

    for old_i in 0..n {
        let new_i = perm[old_i];
        let start = row_ptr[old_i].max(0) as usize;
        let end = (row_ptr[old_i + 1].max(0) as usize).min(n_syn);
        let pos = &mut cursor[new_i];
        for j in start..end {
            let old_post = col_indices[j].max(0) as usize;
            let new_post = if old_post < n {
                perm[old_post]
            } else {
                old_post
            };
            new_col_indices[*pos] = new_post as i32;
            new_values[*pos] = values[j];
            *pos += 1;
        }
    }

    let mut new_nt_types = vec![1.0f32; n];
    for old_i in 0..n.min(nt_types.len()) {
        let new_i = perm[old_i];
        new_nt_types[new_i] = nt_types[old_i];
    }

    (new_row_ptr, new_col_indices, new_values, new_nt_types)
}

/// Применить перестановку к массиву состояния.
pub fn apply_permutation_to_state<T: Copy + Default>(state: &mut [T], perm: &[usize]) {
    let n = perm.len();
    if state.len() < n {
        return;
    }
    let old = state[..n].to_vec();
    for (old_i, &new_i) in perm.iter().enumerate() {
        state[new_i] = old[old_i];
    }
}

/// Перестановка neuron groups.
pub fn apply_permutation_to_groups(
    groups: &[crate::data::NeuronGroup],
    perm: &[usize],
) -> Vec<crate::data::NeuronGroup> {
    groups
        .iter()
        .map(|g| {
            let new_indices: Vec<usize> = g
                .indices
                .iter()
                .filter(|&&old| old < perm.len())
                .map(|&old| perm[old])
                .collect();
            crate::data::NeuronGroup {
                name: g.name.clone(),
                indices: new_indices,
            }
        })
        .collect()
}
