//! RCM (Reverse Cuthill-McKee) — переупорядочивание коннектома для минимизации cache misses.
//!
//! Алгоритм:
//! 1. Строим неориентированный граф из CSR (i→j если есть ребро i→j или j→i)
//! 2. Находим псевдо-периферийные узлы (Gibbs-Poole-Stockmeyer)
//! 3. BFS с сортировкой по степени (Cuthill-McKee)
//! 4. Разворачиваем порядок → Reverse Cuthill-McKee
//!
//! Результат: перестановка `perm[i] = new_index`. Применяется к CSR + массивам нейронов.
//! После RCM соседние нейроны в памяти имеют связанные синапсы → prefetcher работает,
//! cache line utilisation растёт.

use crate::data::ConnectomeData;

/// Построить неориентированный список смежности из CSR.
///
/// Если есть ребро i→j (i пре-синаптический, j пост-синаптический),
/// то в неориентированном графе i и j — соседи.
pub fn build_adjacency(connectome: &ConnectomeData) -> Vec<Vec<usize>> {
    let n = connectome.n_neurons;
    let mut adj = vec![Vec::new(); n];

    // Двухпроходная стратегия: сначала собираем все рёбра, потом строим граф.
    // Это избегает одновременного mutable borrow в adj[pre] и adj[post].
    let mut edges = Vec::new();
    for i in 0..n {
        let start = connectome.row_ptr[i].max(0) as usize;
        let end = (connectome.row_ptr[i + 1] as usize).min(connectome.n_synapses);
        for j in start..end {
            let post = connectome.col_indices[j].max(0) as usize;
            if post < n && i != post {
                edges.push((i, post));
            }
        }
    }

    for &(pre, post) in &edges {
        if !adj[pre].contains(&post) {
            adj[pre].push(post);
        }
        if !adj[post].contains(&pre) {
            adj[post].push(pre);
        }
    }

    adj
}

/// BFS-level structure: для каждого уровня храним список вершин.
fn bfs_levels(adj: &[Vec<usize>], root: usize) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut visited = vec![false; n];
    let mut levels = Vec::new();

    visited[root] = true;
    levels.push(vec![root]);

    while let Some(current_level) = levels.last() {
        let mut next_level = Vec::new();
        for &v in current_level {
            for &u in &adj[v] {
                if !visited[u] {
                    visited[u] = true;
                    next_level.push(u);
                }
            }
        }
        if next_level.is_empty() {
            break;
        }
        // Сортируем по степени (чем меньше степень, тем раньше)
        next_level.sort_by_key(|&v| adj[v].len());
        levels.push(next_level);
    }

    levels
}

/// Найти псевдо-периферийный узел (Gibbs-Poole-Stockmeyer).
///
/// Стратегия:
/// 1. Выбрать произвольный стартовый узел (s) с минимальной степенью
/// 2. Запустить BFS от s, найти узел (e) на последнем уровне с минимальной степенью
/// 3. Запустить BFS от e, найти узел (s') на последнем уровне с минимальной степенью
/// 4. Если eccentricity(s') > eccentricity(s), повторить с s = s'
fn pseudo_peripheral_node(adj: &[Vec<usize>]) -> usize {
    let n = adj.len();
    if n == 0 {
        return 0;
    }

    // Шаг 1: стартуем с узла минимальной степени
    let s = (0..n)
        .min_by_key(|&i| adj[i].len())
        .unwrap_or(0);

    let mut current_root = s;
    loop {
        let levels = bfs_levels(adj, current_root);
        if levels.is_empty() {
            break;
        }
        // Последний уровень
        let last_level = levels.last().unwrap();
        // Выбираем узел с минимальной степенью на последнем уровне
        let e = *last_level
            .iter()
            .min_by_key(|&&v| adj[v].len())
            .unwrap_or(&current_root);

        if e == current_root {
            break;
        }

        // BFS от e
        let levels_e = bfs_levels(adj, e);
        if levels_e.len() <= levels.len() {
            // eccentricity не увеличился — сошлось
            break;
        }
        current_root = e;
    }

    current_root
}

/// Cuthill-McKee ordering без reverse.
///
/// Возвращает порядок: `cm_order[i]` — ID нейрона на i-той позиции.
fn cuthill_mckee(adj: &[Vec<usize>], root: usize) -> Vec<usize> {
    let n = adj.len();
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);

    visited[root] = true;
    order.push(root);

    let mut idx = 0;
    while idx < order.len() {
        let v = order[idx];
        idx += 1;

        // Собираем непосещённых соседей v
        let mut neighbors: Vec<usize> = adj[v]
            .iter()
            .filter(|&&u| !visited[u])
            .copied()
            .collect();

        if neighbors.is_empty() {
            continue;
        }

        // Сортируем по степени (возрастание)
        neighbors.sort_by_key(|&u| adj[u].len());

        for &u in &neighbors {
            visited[u] = true;
            order.push(u);
        }
    }

    // На случай несвязного графа: добавляем непосещённые узлы
    for v in 0..n {
        if !visited[v] {
            visited[v] = true;
            order.push(v);
        }
    }

    order
}

/// Reverse Cuthill-McKee: найти псевдо-периферийный узел, CM, reverse.
///
/// Возвращает `permutation[i] = new_index` — новый индекс для нейрона i.
pub fn reverse_cuthill_mckee(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    if n <= 2 {
        return (0..n).collect();
    }

    let root = pseudo_peripheral_node(adj);
    let cm = cuthill_mckee(adj, root);

    // Reverse: RCM = reverse of CM
    // cm[i] — ID нейрона на i-той позиции
    // permutation[cm[i]] = n - 1 - i
    let mut perm = vec![0usize; n];
    for (i, &node) in cm.iter().enumerate() {
        perm[node] = n - 1 - i;
    }

    perm
}

/// Вычислить bandwidth матрицы: max |i - j| для всех ненулевых W(i,j).
///
/// Используется для оценки качества переупорядочивания.
/// Меньше bandwidth → лучше.
pub fn bandwidth(row_ptr: &[i32], col_indices: &[i32]) -> usize {
    if row_ptr.len() < 2 {
        return 0;
    }
    let n = row_ptr.len() - 1;
    let max_syn = col_indices.len().min(
        row_ptr.last().copied().unwrap_or(0).max(0) as usize
    );
    let mut bw = 0usize;
    for i in 0..n {
        let start = row_ptr[i].max(0) as usize;
        let end = (row_ptr[i + 1] as usize).min(max_syn);
        for j in start..end {
            let post = col_indices[j].max(0) as usize;
            if post < n {
                let diff = if i > post { i - post } else { post - i };
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
    let max_syn = col_indices.len().min(
        row_ptr.last().copied().unwrap_or(0).max(0) as usize
    );
    let mut total = 0u64;
    let mut count = 0u64;
    for i in 0..n {
        let start = row_ptr[i].max(0) as usize;
        let end = (row_ptr[i + 1] as usize).min(max_syn);
        for j in start..end {
            let post = col_indices[j].max(0) as usize;
            if post < n {
                let diff = if i > post { i - post } else { post - i };
                total += diff as u64;
                count += 1;
            }
        }
    }
    if count == 0 { 0.0 } else { total as f64 / count as f64 }
}

/// Применить перестановку к CSR матрице + nt_types.
///
/// `perm[i] = new_index` — новый индекс для нейрона i.
/// Создаёт новые row_ptr, col_indices, values, nt_types с переставленными индексами.
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

    // Новые row_ptr — сначала считаем количество синапсов для каждого нового пре-нейрона
    let mut new_row_ptr = vec![0i32; n + 1];

    for old_i in 0..n {
        let new_i = perm[old_i];
        let start = row_ptr[old_i].max(0) as usize;
        let end = (row_ptr[old_i + 1].max(0) as usize).min(n_syn);
        new_row_ptr[new_i + 1] += (end - start) as i32;
    }

    // Префиксная сумма → row_ptr
    let mut offset = 0i32;
    for cnt in new_row_ptr.iter_mut().skip(1) {
        let c = *cnt;
        *cnt = offset + c;
        offset = *cnt;
    }
    let total_syn = offset as usize;
    let mut new_col_indices = vec![0i32; total_syn];
    let mut new_values = vec![0.0f32; total_syn];

    // Заполняем col_indices и values
    let mut cursor: Vec<usize> = new_row_ptr.iter()
        .map(|&rp| rp.max(0) as usize)
        .collect();
    // cursor[n] = total_syn (не трогаем)

    for old_i in 0..n {
        let new_i = perm[old_i];
        let start = row_ptr[old_i].max(0) as usize;
        let end = (row_ptr[old_i + 1].max(0) as usize).min(n_syn);
        let pos = &mut cursor[new_i];
        for j in start..end {
            let old_post = col_indices[j].max(0) as usize;
            let new_post = if old_post < n { perm[old_post] } else { old_post };
            new_col_indices[*pos] = new_post as i32;
            new_values[*pos] = values[j];
            *pos += 1;
        }
    }

    // nt_types: переставляем по perm
    let mut new_nt_types = vec![1.0f32; n];
    for old_i in 0..n.min(nt_types.len()) {
        let new_i = perm[old_i];
        new_nt_types[new_i] = nt_types[old_i];
    }

    (new_row_ptr, new_col_indices, new_values, new_nt_types)
}

/// Применить перестановку к массиву состояния (v, u, I_buf, spikes, refractory).
///
/// `perm[i] = new_index`.
pub fn apply_permutation_to_state<T: Copy + Default>(
    state: &mut [T],
    perm: &[usize],
) {
    let n = perm.len();
    if state.len() < n {
        return;
    }
    let old = state[..n].to_vec();
    for (old_i, &new_i) in perm.iter().enumerate() {
        state[new_i] = old[old_i];
    }
}

/// Перестановка neuron groups (индексы внутри групп тоже переставляются).
pub fn apply_permutation_to_groups(
    groups: &[crate::data::NeuronGroup],
    perm: &[usize],
) -> Vec<crate::data::NeuronGroup> {
    groups.iter().map(|g| {
        let new_indices: Vec<usize> = g.indices.iter()
            .filter(|&&old| old < perm.len())
            .map(|&old| perm[old])
            .collect();
        crate::data::NeuronGroup {
            name: g.name.clone(),
            indices: new_indices,
        }
    }).collect()
}

// ═══════════════════════════════════════════════════════════
//  Тесты
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pseudo_peripheral() {
        let n = 6;
        let row_ptr = vec![0i32, 2, 3, 4, 5, 6, 7];
        let col_indices = vec![1, 5, 2, 3, 4, 0, 0];
        let n_syn = col_indices.len();
        let dat = ConnectomeData {
            n_neurons: n,
            n_synapses: n_syn,
            row_ptr,
            col_indices,
            values: vec![1.0; n_syn],
            nt_types: vec![1.0; n],
            groups: vec![],
        };
        let adj = build_adjacency(&dat);
        let root = pseudo_peripheral_node(&adj);
        println!("pseudo_peripheral root = {root}");
        assert!(root < n, "root {root} out of bounds");
    }

    #[test]
    fn test_rcm_ordering() {
        let n = 6;
        let row_ptr = vec![0i32, 2, 3, 4, 5, 6, 7];
        let col_indices = vec![1, 5, 2, 3, 4, 0, 0];
        let n_syn = col_indices.len();
        let dat = ConnectomeData {
            n_neurons: n,
            n_synapses: n_syn,
            row_ptr,
            col_indices,
            values: vec![1.0; n_syn],
            nt_types: vec![1.0; n],
            groups: vec![],
        };
        let adj = build_adjacency(&dat);
        let perm = reverse_cuthill_mckee(&adj);

        // Проверяем что это перестановка (все индексы 0..n-1)
        let mut sorted = perm.clone();
        sorted.sort();
        for (i, &p) in sorted.iter().enumerate() {
            assert_eq!(p, i, "perm is not a valid permutation at {i}");
        }
        println!("RCM perm: {:?}", perm);
    }

    #[test]
    fn test_bandwidth_reduction() {
        let n = 32;
        let mut row_ptr = vec![0i32; n + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // Кольцо: каждый i → i+1 (mod n)
        for i in 0..n {
            let post = (i + 1) % n;
            col_indices.push(post as i32);
            values.push(1.0);
            row_ptr[i + 1] = i as i32 + 1;
        }

        let n_syn = values.len();
        let dat = ConnectomeData {
            n_neurons: n,
            n_synapses: n_syn,
            row_ptr,
            col_indices,
            values,
            nt_types: vec![1.0; n],
            groups: vec![],
        };

        let bw_before = bandwidth(&dat.row_ptr, &dat.col_indices);
        println!("Bandwidth BEFORE: {bw_before}");

        let adj = build_adjacency(&dat);
        let perm = reverse_cuthill_mckee(&adj);
        let (new_rp, new_ci, _, _) = apply_permutation_to_csr(
            &dat.row_ptr, &dat.col_indices, &dat.values,
            &dat.nt_types, &perm,
        );
        let bw_after = bandwidth(&new_rp, &new_ci);
        println!("Bandwidth AFTER:  {bw_after}");
        println!("Mean BW before:  {:.2}", mean_bandwidth(&dat.row_ptr, &dat.col_indices));
        println!("Mean BW after:   {:.2}", mean_bandwidth(&new_rp, &new_ci));

        assert!(bw_after <= bw_before, "bandwidth should not increase");
    }

    #[test]
    fn test_rcm_preserves_connectome_semantics() {
        let n = 10;
        let mut row_ptr = vec![0i32; n + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let mut rng: u64 = 12345;

        for i in 0..n {
            let n_syn = 2 + ((rng.wrapping_mul(6364136223846793005).wrapping_add(1)) % 3) as usize;
            let start = row_ptr[i];
            for _ in 0..n_syn {
                let post = ((rng.wrapping_mul(6364136223846793005).wrapping_add(1)) % n as u64) as i32;
                col_indices.push(post);
                values.push(((rng.wrapping_mul(6364136223846793005).wrapping_add(1) as f32) % 10.0 - 5.0) / 10.0);
            }
            row_ptr[i + 1] = start + n_syn as i32;
        }
        let n_syn = values.len();
        let nt_types: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

        let dat = ConnectomeData {
            n_neurons: n,
            n_synapses: n_syn,
            row_ptr: row_ptr.clone(),
            col_indices: col_indices.clone(),
            values: values.clone(),
            nt_types: nt_types.clone(),
            groups: vec![],
        };

        let adj = build_adjacency(&dat);
        let perm = reverse_cuthill_mckee(&adj);
        let (new_rp, new_ci, new_vals, new_nt) = apply_permutation_to_csr(
            &dat.row_ptr, &dat.col_indices, &dat.values,
            &dat.nt_types, &perm,
        );

        // Строим мультимножество из нового CSR
        let mut edge_set: std::collections::HashMap<(i32, i32), f32> = std::collections::HashMap::new();
        for i in 0..n {
            let start = new_rp[i].max(0) as usize;
            let end = (new_rp[i + 1] as usize).min(new_vals.len());
            for j in start..end {
                edge_set.insert((i as i32, new_ci[j]), new_vals[j]);
            }
        }

        // Проверяем старые рёбра в новом графе
        for i in 0..n {
            let start = row_ptr[i].max(0) as usize;
            let end = (row_ptr[i + 1] as usize).min(values.len());
            for j in start..end {
                let new_i = perm[i] as i32;
                let new_j = perm[col_indices[j] as usize] as i32;
                let v = edge_set.get(&(new_i, new_j));
                assert!(v.is_some(), "Edge ({new_i}→{new_j}) from ({i}→{}) not found after RCM", col_indices[j]);
                assert!((*v.unwrap() - values[j]).abs() < 1e-6,
                    "Weight mismatch for edge ({new_i}→{new_j})");
            }
        }

        // Проверяем nt_types
        for old_i in 0..n {
            let new_i = perm[old_i];
            assert_eq!(new_nt[new_i], nt_types[old_i],
                "nt_types mismatch for neuron {old_i}→{new_i}");
        }

        println!("RCM semantics test: ✅ all {n_syn} edges preserved");
    }

    #[test]
    fn test_apply_permutation_to_state() {
        let perm = vec![2, 4, 0, 3, 1];
        let mut state: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        apply_permutation_to_state(&mut state[..], &perm);
        assert!((state[0] - 30.0).abs() < 1e-6, "state[0]={}", state[0]);
        assert!((state[1] - 50.0).abs() < 1e-6, "state[1]={}", state[1]);
        assert!((state[2] - 10.0).abs() < 1e-6, "state[2]={}", state[2]);
        assert!((state[3] - 40.0).abs() < 1e-6, "state[3]={}", state[3]);
        assert!((state[4] - 20.0).abs() < 1e-6, "state[4]={}", state[4]);
        println!("State permutation: ✅");
    }
}
