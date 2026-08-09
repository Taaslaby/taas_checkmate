use crate::data::ConnectomeData;

/// Построить неориентированный список смежности из CSR.
pub fn build_adjacency(connectome: &ConnectomeData) -> Vec<Vec<usize>> {
    let n = connectome.n_neurons;
    let mut adj = vec![Vec::new(); n];

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
        adj[pre].push(post);
        adj[post].push(pre);
    }
    for neighbors in adj.iter_mut() {
        neighbors.sort_unstable();
        neighbors.dedup();
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
        next_level.sort_by_key(|&v| adj[v].len());
        levels.push(next_level);
    }

    levels
}

/// Найти псевдо-периферийный узел (Gibbs-Poole-Stockmeyer).
pub fn pseudo_peripheral_node(adj: &[Vec<usize>]) -> usize {
    let n = adj.len();
    if n == 0 {
        return 0;
    }

    let s = (0..n).min_by_key(|&i| adj[i].len()).unwrap_or(0);

    let mut current_root = s;
    loop {
        let levels = bfs_levels(adj, current_root);
        if levels.is_empty() {
            break;
        }
        let last_level = levels.last().unwrap();
        let e = *last_level
            .iter()
            .min_by_key(|&&v| adj[v].len())
            .unwrap_or(&current_root);

        if e == current_root {
            break;
        }

        let levels_e = bfs_levels(adj, e);
        if levels_e.len() <= levels.len() {
            break;
        }
        current_root = e;
    }

    current_root
}

/// Cuthill-McKee ordering без reverse.
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

        let mut neighbors: Vec<usize> = adj[v].iter().filter(|&&u| !visited[u]).copied().collect();

        if neighbors.is_empty() {
            continue;
        }

        neighbors.sort_by_key(|&u| adj[u].len());

        for &u in &neighbors {
            visited[u] = true;
            order.push(u);
        }
    }

    for v in 0..n {
        if !visited[v] {
            visited[v] = true;
            order.push(v);
        }
    }

    order
}

/// Reverse Cuthill-McKee: найти псевдо-периферийный узел, CM, reverse.
pub fn reverse_cuthill_mckee(adj: &[Vec<usize>]) -> Vec<usize> {
    let n = adj.len();
    if n <= 2 {
        return (0..n).collect();
    }

    let root = pseudo_peripheral_node(adj);
    let cm = cuthill_mckee(adj, root);

    let mut perm = vec![0usize; n];
    for (i, &node) in cm.iter().enumerate() {
        perm[node] = n - 1 - i;
    }

    perm
}
