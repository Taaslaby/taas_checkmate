"""Network-science scoring of a generated connectome.

Real connectomes are: high clustering (~0.74 vs ~0.08 Erdos-Renyi), heavy-tailed
degree (var/<k> ~55), disassortative (r ~ -0.35), modular. The old models failed
exactly here — they collapsed to min-degree or produced Erdos-Renyi-like blobs.
These metrics are the "looks like a real connectome" part of the reward.
"""

from __future__ import annotations

import numpy as np


def row_ptr_to_edges(row_ptr, col_idx) -> tuple[np.ndarray, np.ndarray]:
    rows = np.repeat(np.arange(len(row_ptr) - 1), np.diff(row_ptr))
    return rows.astype(np.int64), np.asarray(col_idx, dtype=np.int64)


def degree_distribution(row_ptr, col_idx) -> tuple[np.ndarray, float]:
    """Out-degree per neuron + its coefficient of variation (heavy-tail signal)."""
    deg = np.diff(row_ptr).astype(np.float64)
    if deg.sum() == 0 or len(deg) < 3:
        return deg, 0.0
    cv = deg.std() / (deg.mean() + 1e-9)
    return deg, float(cv)


def clustering_coefficient(row_ptr, col_idx) -> float:
    """Undirected global transitivity C = 3*T / W.

    The connectome is directed; we symmetrize first (edge counts if either
    direction exists) so C is a proper bounded [0,1] clustering signal.
    """
    rows, cols = row_ptr_to_edges(row_ptr, col_idx)
    n = len(row_ptr) - 1
    if len(rows) < 4:
        return 0.0
    # undirected adjacency
    adj = [set() for _ in range(n)]
    for r, c in zip(rows, cols):
        if r != c:
            adj[r].add(int(c))
            adj[c].add(int(r))
    deg = np.array([len(a) for a in adj])
    W = (deg * (deg - 1)).sum() / 2.0  # connected triples
    if W <= 0:
        return 0.0
    T = 0
    # each triangle {u,v,x} is counted once per edge; sum_common = 3*#triangles,
    # and transitivity = 3*(#triangles)/W = sum_common / W.
    for r, c in zip(rows, cols):
        if r != c and r < c:
            T += len(adj[r] & adj[c])
    return float(T / W)


def modularity(row_ptr, col_idx, types=None, n_comm=8) -> float:
    """Newman modularity Q = 1/(2m) Σ_ij (A_ij - k_i k_j / 2m) δ(c_i, c_j).

    Community labels come from degree-quantile buckets (crude but stable and
    cheap). The full N² sum is used so the null-model term is exact.
    """
    rows, cols = row_ptr_to_edges(row_ptr, col_idx)
    n = len(row_ptr) - 1
    if n == 0:
        return 0.0
    m = float(len(rows))
    if m == 0:
        return 0.0
    deg = np.diff(row_ptr).astype(np.float64)
    indeg = np.bincount(cols, minlength=n).astype(np.float64)
    tot = deg + indeg
    # community label: n_comm quantile buckets of total degree
    order = np.argsort(tot)
    buckets = np.empty(n, dtype=np.int64)
    per = np.linspace(0, n, n_comm + 1, dtype=np.int64)
    for b in range(n_comm):
        buckets[order[per[b]:per[b + 1]]] = b
    # adjacency matrix (dense for correctness; N small)
    A = np.zeros((n, n))
    for r, c in zip(rows, cols):
        A[r, c] = 1.0
    # modularity (upper half to avoid double counting)
    q = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if buckets[i] == buckets[j]:
                q += 2.0 * (A[i, j] - deg[i] * deg[j] / (2.0 * m))
    q /= 2.0 * m
    return float(q)


def assortativity(row_ptr, col_idx) -> float:
    """Degree assortativity (Pearson on edges). Real connectomes ~ -0.35."""
    rows, cols = row_ptr_to_edges(row_ptr, col_idx)
    if len(rows) < 4:
        return 0.0
    deg = np.diff(row_ptr).astype(np.float64)
    dr, dc = deg[rows], deg[cols]
    if dr.std() == 0 or dc.std() == 0:
        return 0.0
    return float(np.corrcoef(dr, dc)[0, 1])


def ei_ratio(weights) -> float:
    """Sum of |W| for excitatory / for inhibitory."""
    w = np.asarray(weights, dtype=np.float64)
    e = np.abs(w[w > 0]).sum()
    i = np.abs(w[w < 0]).sum()
    return float(e / (i + 1e-9))


def synapse_density(row_ptr) -> float:
    n = len(row_ptr) - 1
    m = int(row_ptr[-1])
    return m / (n * (n - 1)) if n > 1 else 0.0


def graph_metrics(csr: dict) -> dict:
    """One dict of metrics for a CSR dictionary (as produced by to_csr)."""
    row_ptr = np.asarray(csr["row_ptr"])
    col_idx = np.asarray(csr["col_idx"])
    weights = np.asarray(csr["weights"])
    deg, cv = degree_distribution(row_ptr, col_idx)
    n = len(row_ptr) - 1
    m = int(row_ptr[-1])
    return {
        "n": n,
        "m": m,
        "mean_deg": float(deg.mean()) if n else 0.0,
        "deg_cv": cv,
        "clustering": clustering_coefficient(row_ptr, col_idx),
        "modularity": modularity(row_ptr, col_idx),
        "assortativity": assortativity(row_ptr, col_idx),
        "ei_ratio": ei_ratio(weights),
        "density": synapse_density(row_ptr),
        "n_synapses": m,
    }
