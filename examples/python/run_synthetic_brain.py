#!/usr/bin/env python3
"""
🧠 Загрузка синтетического коннектома на движок taas_checkmate.

Загружает synthetic_connections.csv.gz (693K синапсов, FlyWire формат)
и hallucinated_graph.graphml (42 нейрона, визуальная система),
строит CSR → ConnectomeData → .taas → Kuramoto симуляция.

Использование:
    cd taas_checkmate
    python3 examples/python/run_synthetic_brain.py
"""

import sys, os, time, gzip, xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJ = Path(__file__).parent.parent
os.environ.setdefault('PYTHONPATH', str(PROJ / 'target' / 'debug'))

try:
    import taas_checkmate as tcm
except ImportError:
    print("✗ Cannot import taas_checkmate. Install first:")
    print("  cd taas_checkmate && cargo build --features python")
    print("  cp target/debug/libtaas_checkmate.so .venv/lib/python*/site-packages/taas_checkmate/taas_checkmate.abi3.so")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
#  1. Загрузка synthetic_connections.csv.gz
# ═══════════════════════════════════════════════════════════

def load_flywire_csv(gz_path: str) -> dict:
    """Загружает FlyWire CSV (pre_root_id, post_root_id, neuropil, syn_count).
    
    Returns:
        dict с ключами: n_neurons, n_synapses, row_ptr, col_indices, values,
            nt_types, groups, neuropil_stats, root_id_list
    """
    print(f"\n  ── Loading: {gz_path}")
    t0 = time.time()
    
    edges = []
    neuropil_set = set()
    with gzip.open(gz_path, 'rt') as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split(',')
            pre, post, neuropil, syn = int(parts[0]), int(parts[1]), parts[2], int(parts[3])
            edges.append((pre, post, neuropil, syn))
            neuropil_set.add(neuropil)
    
    n_edges = len(edges)
    print(f"    {n_edges:,} edges, {len(neuropil_set)} neuropil regions")
    
    root_ids = sorted(set(r for e in edges for r in (e[0], e[1])))
    root_to_idx = {rid: i for i, rid in enumerate(root_ids)}
    n = len(root_ids)
    print(f"    {n:,} unique neurons")
    
    out_deg = np.zeros(n, dtype=np.int32)
    for pre, _, _, _ in edges:
        out_deg[root_to_idx[pre]] += 1
    
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(out_deg)
    n_syn = row_ptr[-1]
    col_indices = np.zeros(n_syn, dtype=np.int32)
    values = np.zeros(n_syn, dtype=np.float32)
    
    cursor = np.zeros(n, dtype=np.int32)
    for pre, post, _, syn in edges:
        pre_idx = root_to_idx[pre]
        post_idx = root_to_idx[post]
        pos = row_ptr[pre_idx] + cursor[pre_idx]
        col_indices[pos] = post_idx
        values[pos] = float(syn)
        cursor[pre_idx] += 1
    
    np_to_neurons = defaultdict(set)
    np_pre_count = defaultdict(int)
    for pre, post, neuropil, _ in edges:
        np_to_neurons[neuropil].add(root_to_idx[pre])
        np_to_neurons[neuropil].add(root_to_idx[post])
        np_pre_count[neuropil] += 1
    
    groups = [(np, sorted(ids)) for np, ids in sorted(np_to_neurons.items(), key=lambda x: -len(x[1]))]
    nt_types = np.ones(n, dtype=np.float32)
    np_stats = sorted([(np, np_pre_count[np], len(np_to_neurons[np])) for np in np_to_neurons], key=lambda x: -x[1])
    
    print(f"    CSR built: {n:,} neurons, {n_syn:,} synapses")
    print(f"    Groups: {len(groups)} neuropil groups")
    print(f"    Load time: {time.time() - t0:.2f}s")
    
    return {'n_neurons': n, 'n_synapses': n_syn, 'row_ptr': row_ptr,
            'col_indices': col_indices, 'values': values, 'nt_types': nt_types,
            'groups': groups, 'neuropil_stats': np_stats, 'root_id_list': root_ids}


def load_graphml_graphml(path: str) -> dict:
    """Загружает GraphML граф коннектома."""
    print(f"\n  ── Loading: {path}")
    t0 = time.time()
    
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    
    nodes = root.findall('.//g:node', ns)
    node_names = [n.get('id') for n in nodes]
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    n = len(node_names)
    print(f"    {n} neurons")
    
    edges_el = root.findall('.//g:edge', ns)
    out_cnt = defaultdict(int)
    edges_raw = []
    for e in edges_el:
        s, t = e.get('source'), e.get('target')
        w_el = e.find('.//g:data', ns)
        w = int(w_el.text) if w_el is not None else 1
        edges_raw.append((s, t, w))
        out_cnt[s] += 1
    
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    for name, cnt in out_cnt.items():
        row_ptr[name_to_idx[name] + 1] = cnt
    row_ptr = np.cumsum(row_ptr, dtype=np.int32)
    n_syn = row_ptr[-1]
    col_indices = np.zeros(n_syn, dtype=np.int32)
    values = np.zeros(n_syn, dtype=np.float32)
    
    cursor = np.zeros(n, dtype=np.int32)
    for s, t, w in edges_raw:
        s_idx, t_idx = name_to_idx[s], name_to_idx[t]
        pos = row_ptr[s_idx] + cursor[s_idx]
        col_indices[pos] = t_idx
        values[pos] = float(w)
        cursor[s_idx] += 1
    
    nt_types = np.ones(n, dtype=np.float32)
    type_groups = defaultdict(set)
    for name in node_names:
        prefix = ''
        for c in name:
            if c.isalpha(): prefix += c
            else: break
        type_groups[prefix].add(name_to_idx[name])
    
    groups = [(t, sorted(ids)) for t, ids in sorted(type_groups.items(), key=lambda x: -len(x[1]))]
    
    print(f"    CSR built: {n:,} neurons, {n_syn:,} synapses")
    print(f"    Groups: {[(t, len(ids)) for t, ids in groups]}")
    print(f"    Load time: {time.time() - t0:.2f}s")
    
    return {'n_neurons': n, 'n_synapses': n_syn, 'row_ptr': row_ptr,
            'col_indices': col_indices, 'values': values, 'nt_types': nt_types,
            'groups': groups, 'neuron_names': node_names}


# ═══════════════════════════════════════════════════════════
#  3. Kuramoto симуляция (через tcm.cpe.*)
# ═══════════════════════════════════════════════════════════

def run_kuramoto(data: dict, label: str, n_steps: int = 2000, dt: float = 0.05):
    """Запускает Kuramoto симуляцию на коннектоме через CPE."""
    rp = data['row_ptr']
    ci = data['col_indices']
    vl = data['values']
    n = data['n_neurons']
    
    phases = np.random.RandomState(42).rand(n).astype(np.float32) * 2 * np.pi
    omegas = np.full(n, np.pi, dtype=np.float32)
    
    print(f"\n  ═══ Kuramoto: {label} ═══")
    print(f"    {n:,} neurons, {data['n_synapses']:,} synapses, {n_steps} steps")
    
    coh_log, spike_log, ent_log = [], [], []
    t0 = time.time()
    
    # Pre-allocate buffers for CPE API
    spikes_out = np.zeros(n, dtype=np.int32)
    phases_prev = phases.copy()
    
    for step in range(n_steps):
        # Kuramoto step via CPE
        tcm.cpe.kuramoto_step(phases, omegas, rp, ci, vl, dt)
        
        if step % 200 == 0:
            # Phase coherence (просто фазы → f32)
            coh = tcm.cpe.phase_coherence(phases)
            coh_log.append((step, coh))
            
            # Detect spikes: writes to spikes_out
            tcm.cpe.detect_spikes(spikes_out, phases, phases_prev, 3.0)
            n_spikes = int(np.sum(spikes_out))
            spike_log.append((step, n_spikes, n_spikes / max(n, 1)))
            
            # Spike entropy: нужен массив firing rates
            # Считаем прямо здесь: Shannon entropy по фазе
            ent = tcm.cpe.spike_entropy(
                np.abs(np.sin(phases)).astype(np.float32), n_steps, 0.5
            )[0]  # возвращает (H, H_dev)
            ent_log.append((step, ent))
            
            phases_prev[:] = phases
    
    wall = time.time() - t0
    
    print(f"\n  ── Results ──")
    print(f"    Wall: {wall:.2f}s, Sim: {n_steps*dt*1000:.0f}ms, RT: {n_steps*dt/wall:.1f}×")
    print(f"\n    Coherence:")
    for s, c in coh_log:
        print(f"      step {s:5d}:  {c:.4f}")
    print(f"\n    Spikes:")
    for s, ns, r in spike_log:
        print(f"      step {s:5d}:  {ns:6d} spikes  ({r:.1f}/neuron)")
    print(f"\n    Entropy:")
    for s, e in ent_log:
        print(f"      step {s:5d}:  {e:.4f}")
    
    return {'phases': phases, 'coherence': coh_log, 'spikes': spike_log,
            'entropy': ent_log, 'wall_time': wall}


def save_connectome_to_taas(data: dict, path: str, label: str):
    """Создаёт ConnectomeData и сохраняет в .taas."""
    print(f"\n  ── Saving .taas: {path}")
    groups_list = [(name, indices) for name, indices in data['groups']]
    conn = tcm.ConnectomeData(data['row_ptr'], data['col_indices'],
                               data['values'], data['nt_types'], groups_list)
    print(f"    {conn.n_neurons} neurons, {conn.n_synapses} synapses")
    print(f"    Groups: {conn.group_names()[:3]}... ({len(conn.group_names())} total)")
    tcm.save_connectome(str(path), conn)
    size = os.path.getsize(str(path))
    print(f"    Saved: {size/1024:.1f} KB")
    loaded = tcm.load_connectome(str(path))
    print(f"    Verified: {loaded.n_neurons} neurons, {loaded.n_synapses} synapses")
    return conn


def main():
    print("═" * 60)
    print("  🧠 Synthetic Connectome — Engine Run")
    print("═" * 60)
    
    meii_dir = Path(os.path.expanduser("~")) / "Проекты" / "asm_engines" / "MEIIgpt" / "experiments" / "sardana_1" / "data"
    csv_gz = meii_dir / "synthetic_connections.csv.gz"
    graphml = meii_dir / "hallucinated_graph.graphml"
    
    # ── A. Synthetic ──
    print("\n" + "─" * 60)
    print("  A. Synthetic Connections CSV (FlyWire format)")
    print("─" * 60)
    data_csv = load_flywire_csv(str(csv_gz))
    
    print(f"\n  Top neuropil regions:")
    for np_name, syn_cnt, neuron_cnt in data_csv['neuropil_stats'][:10]:
        print(f"    {np_name:20s}  {syn_cnt:8,d} syn  {neuron_cnt:6,d} nrn")
    
    conn_csv = save_connectome_to_taas(data_csv, PROJ / "synth_brain.taas", "Synthetic")
    result_csv = run_kuramoto(data_csv, "Synthetic 693K", n_steps=2000, dt=0.05)
    
    # ── B. GraphML ──
    print("\n" + "─" * 60)
    print("  B. Hallucinated GraphML (visual system)")
    print("─" * 60)
    data_gml = load_graphml_graphml(str(graphml))
    conn_gml = save_connectome_to_taas(data_gml, PROJ / "hallucinated_brain.taas", "Hallucinated")
    result_gml = run_kuramoto(data_gml, "Hallucinated 42n", n_steps=2000, dt=0.05)
    
    # ── Summary ──
    print("\n" + "═" * 60)
    print("  📊 Summary")
    print("═" * 60)
    coh_c = result_csv['coherence'][-1][1] if result_csv['coherence'] else 0
    coh_g = result_gml['coherence'][-1][1] if result_gml['coherence'] else 0
    spk_c = result_csv['spikes'][-1][1] if result_csv['spikes'] else 0
    spk_g = result_gml['spikes'][-1][1] if result_gml['spikes'] else 0
    
    print(f"\n  A. Synthetic: {data_csv['n_neurons']:,} nrn, {data_csv['n_synapses']:,} syn")
    print(f"     Coh={coh_c:.4f}, Spk={spk_c}, {result_csv['wall_time']:.1f}s")
    print(f"  B. Hallucinated: {data_gml['n_neurons']} nrn, {data_gml['n_synapses']} syn")
    print(f"     Coh={coh_g:.4f}, Spk={spk_g}, {result_gml['wall_time']:.1f}s")
    print(f"\n  🔥 Total: {data_csv['n_neurons']+data_gml['n_neurons']:,} nrn, "
          f"{data_csv['n_synapses']+data_gml['n_synapses']:,} syn через движок")
    print(f"\n  Saved: synth_brain.taas, hallucinated_brain.taas")
    print("\n" + "═" * 60)
    print("  Done.")
    print("═" * 60)


if __name__ == "__main__":
    main()
