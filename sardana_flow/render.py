"""Render a CSR connectome to a PNG image — the "photo" the see-and-fix loop
(and a future VLM critic) can look at.

Community layout: detect communities with networkx greedy-modularity, place each
community on a circle, jitter its nodes around the community centre. Edges are
sampled for readability and coloured by E/I (excitatory blue, inhibitory red).
Input group gets a green ring, output group a red ring.
"""

from __future__ import annotations

import numpy as np

# matplotlib import is lazy — importing this module must not require it
_mpl = None


def _plt():
    global _mpl
    if _mpl is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _mpl = plt
    return _mpl


def csr_to_networkx(csr: dict):
    """Build a networkx Graph (nodes 0..n-1, weight=abs, ei=+1/-1)."""
    import networkx as nx

    row_ptr = np.asarray(csr["row_ptr"], dtype=np.int64)
    col_idx = np.asarray(csr["col_idx"], dtype=np.int64)
    w = np.asarray(csr["weights"], dtype=np.float64)
    n = len(row_ptr) - 1
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for jj in range(row_ptr[i], row_ptr[i + 1]):
            j = col_idx[jj]
            G.add_edge(i, j, weight=float(abs(w[jj])), ei=1.0 if w[jj] >= 0 else -1.0)
    return G


def _community_layout(G, seed=7):
    """Community-circle layout. Returns {node: (x, y)}."""
    import networkx as nx
    import numpy as np

    try:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    except Exception:
        comms = [set(G.nodes)]
    comms = [c for c in comms if len(c) > 1] or [set(G.nodes)]
    k = len(comms)
    node_to_comm = {}
    for ci, c in enumerate(comms):
        for nd in c:
            node_to_comm[nd] = ci
    rng = np.random.default_rng(seed)
    pos = {}
    for nd in G.nodes:
        ci = node_to_comm.get(nd, 0)
        ang = 2 * np.pi * ci / k
        cx, cy = np.cos(ang), np.sin(ang)
        r = 0.18
        pos[nd] = (cx + rng.uniform(-r, r), cy + rng.uniform(-r, r))
    return pos


def render(csr: dict, title: str = "", out_path: str = None,
           max_edges: int = 900, figsize=(11, 11), dpi=110) -> str:
    """Render a connectome CSR to a PNG. Returns the written path."""
    import networkx as nx

    plt = _plt()
    G = csr_to_networkx(csr)
    n = G.number_of_nodes()
    if n == 0:
        return None

    pos = _community_layout(G)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # node degree → size
    deg = np.array([d for _, d in G.degree()])
    sizes = 30 + 18 * (deg / (deg.max() + 1e-9))

    # edges, subsampled
    edges = list(G.edges(data=True))
    if len(edges) > max_edges:
        rng = np.random.default_rng(0)
        edges = list(rng.choice(edges, max_edges, replace=False))
    exc = [(u, v) for u, v, d in edges if d["ei"] > 0]
    inh = [(u, v) for u, v, d in edges if d["ei"] <= 0]
    if exc:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=exc, width=0.4,
                               edge_color="#1f77b4", alpha=0.35, arrows=False)
    if inh:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=inh, width=0.4,
                               edge_color="#d62728", alpha=0.35, arrows=False)

    # community colours for nodes
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("tab20")
    comm_col = {}
    for nd in G.nodes:
        # reuse community colour via layout ring (stable)
        comm_col[nd] = cmap(int(_ring_index(nd, pos)) % 20)

    node_colors = [comm_col[nd] for nd in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                           node_color=node_colors, alpha=0.85,
                           linewidths=0.3, edgecolors="black")

    # input / output probe groups
    inp = set(int(i) for i in np.asarray(csr["input_idx"]).flatten())
    out = set(int(i) for i in np.asarray(csr["output_idx"]).flatten())
    if inp:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=list(inp), node_size=90,
                               node_color="none", edgecolors="#2ca02c", linewidths=2.5)
    if out:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=list(out), node_size=90,
                               node_color="none", edgecolors="#d62728", linewidths=2.5)

    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    # legend
    ax.plot([], [], "-", color="#1f77b4", label="excitatory")
    ax.plot([], [], "-", color="#d62728", label="inhibitory")
    ax.plot([], [], "o", color="none", mec="#2ca02c", mfc="none", label="input")
    ax.plot([], [], "o", color="none", mec="#d62728", mfc="none", label="output")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.7)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        return out_path
    plt.close(fig)
    return None


def _ring_index(node, pos):
    """Which angular ring the node sits in (proxy for community)."""
    x, y = pos[node]
    import numpy as np

    a = np.arctan2(y, x) / (2 * np.pi) * 20
    return int(round(a)) % 20
