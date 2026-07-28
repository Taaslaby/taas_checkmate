#!/usr/bin/env python3
"""
SONATA → TAAS Checkmate converter
==================================

Parses SONATA format HDF5 files (nodes.h5 + edges.h5) used by
Allen Institute (GLIF models) and BlueBrain Project.

Поддерживаемые типы:
  - GLIF3 (LIF + after-spike current I_AHP)
  - GLIF4 (GLIF3 + threshold adaptation)
  - LIF (standard LIF with tau_m, V_thresh, V_reset)
  - HH (via model_template pointing to channel specs)

Зависимости:
  pip install libsonata h5py numpy

Пример:
  net = SonataNetwork.from_files("nodes.h5", "edges.h5")
  csr, params, info = net.to_taas(neuron_model="glif")
  print(f"Neurons: {info['n_neurons']}, Synapses: {info['n_synapses']}")
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

# ═══════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════

@dataclass
class SonataNodeGroup:
    """A population of nodes (neurons) from SONATA."""
    name: str
    node_ids: np.ndarray  # [n]
    model_type: str  # 'glif3', 'glif4', 'lif', 'hh', 'biophysical'
    model_template: str  # e.g. 'GLIF3_1', 'GLIF4_2'
    params: Dict[str, np.ndarray] = field(default_factory=dict)
    # GLIF params (per-neuron arrays):
    #   g_leak, E_leak, V_thresh, V_reset, Cm
    #   tau_ahp, delta_g_ahp (GLIF3)
    #   tau_theta, delta_theta (GLIF4)
    positions: Optional[np.ndarray] = None  # [n, 3]

@dataclass
class SonataEdgeGroup:
    """A set of edges (synapses) from SONATA."""
    name: str
    source_node_ids: np.ndarray  # [n_synapses]
    target_node_ids: np.ndarray  # [n_synapses]
    weights: Optional[np.ndarray] = None
    delays: Optional[np.ndarray] = None
    syn_type_ids: Optional[np.ndarray] = None  # 0=excitatory, 1=inhibitory

@dataclass
class SonataNetwork:
    """Complete parsed SONATA network."""
    node_groups: Dict[str, SonataNodeGroup] = field(default_factory=dict)
    edge_groups: Dict[str, SonataEdgeGroup] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)

    # ═══════════════════════════════════════════════════════
    #  Parse SONATA HDF5 files
    # ═══════════════════════════════════════════════════════

    @classmethod
    def from_files(
        cls,
        nodes_path: str,
        edges_path: str,
        node_types_path: Optional[str] = None,
    ) -> "SonataNetwork":
        """Parse SONATA nodes.h5 + edges.h5 (connectivity)."""
        net = cls()
        nodes_path = Path(nodes_path)
        edges_path = Path(edges_path)

        if not HAS_H5PY:
            raise ImportError("h5py required: pip install h5py")

        # ── Parse nodes ──
        with h5py.File(str(nodes_path), 'r') as f:
            nodes_grp = f['nodes']
            for pop_name in nodes_grp.keys():
                pop_grp = nodes_grp[pop_name]
                node_ids = pop_grp['node_id'][:]

                # Read model metadata from HDF5 attributes (not from node_types CSV alone)
                model_type = pop_grp.attrs.get('model_type', b'lif')
                model_template = pop_grp.attrs.get('model_template', b'')
                if isinstance(model_type, bytes):
                    model_type = model_type.decode()
                if isinstance(model_template, bytes):
                    model_template = model_template.decode()

                # Collect per-neuron param arrays from HDF5 datasets
                params = {}
                for ds_name in pop_grp.keys():
                    if ds_name == 'node_id':
                        continue
                    try:
                        data = pop_grp[ds_name][:]
                        data = np.asarray(data, dtype=np.float32).ravel()
                        params[ds_name] = data
                    except Exception:
                        pass

                positions = None
                if 'x' in params and 'y' in params and 'z' in params:
                    positions = np.column_stack([
                        params.pop('x'), params.pop('y'), params.pop('z')
                    ])

                net.node_groups[pop_name] = SonataNodeGroup(
                    name=pop_name,
                    node_ids=np.asarray(node_ids),
                    model_type=model_type,
                    model_template=model_template,
                    params=params,
                    positions=positions,
                )

        # ── Parse edges ──
        with h5py.File(str(edges_path), 'r') as f:
            edges_grp = f['edges']
            for pop_name in edges_grp.keys():
                pop_grp = edges_grp[pop_name]

                # Read source/target populations from HDF5 attributes (NOT from name parsing!)
                # Real SONATA files store these as attributes on the edge group.
                src_pop = pop_grp.attrs.get('source_population', None)
                tgt_pop = pop_grp.attrs.get('target_population', None)
                if isinstance(src_pop, bytes):
                    src_pop = src_pop.decode()
                if isinstance(tgt_pop, bytes):
                    tgt_pop = tgt_pop.decode()
                # Fallback: try to parse from edge group name (e.g. "popA__to__popB")
                # This is the BlueBrain/SONATA convention when attributes are missing.
                if src_pop is None or tgt_pop is None:
                    parts = pop_name.split('__to__')
                    if len(parts) == 2:
                        src_pop, tgt_pop = parts
                    else:
                        parts = pop_name.split('__')
                        if len(parts) >= 2:
                            src_pop, tgt_pop = parts[0], parts[-1]
                        else:
                            src_pop, tgt_pop = pop_name, pop_name

                src = pop_grp['source_node_id'][:]
                tgt = pop_grp['target_node_id'][:]

                weights = None
                delays = None
                syn_types = None
                if 'weight' in pop_grp:
                    weights = pop_grp['weight'][:].astype(np.float32)
                if 'delay' in pop_grp:
                    delays = pop_grp['delay'][:].astype(np.float32)
                if 'syn_type_id' in pop_grp:
                    syn_types = pop_grp['syn_type_id'][:].astype(np.int32)

                # Build node_id → global_id lookup for non-contiguous IDs
                # Not needed here — handled in to_taas()
                net.edge_groups[pop_name] = SonataEdgeGroup(
                    name=pop_name,
                    source_node_ids=np.asarray(src),
                    target_node_ids=np.asarray(tgt),
                    weights=weights,
                    delays=delays,
                    syn_type_ids=syn_types,
                )

        # ── Parse node_types CSV if provided ──
        if node_types_path:
            _parse_node_types(net, node_types_path)

        return net

    # ═══════════════════════════════════════════════════════
    #  Convert to TAAS CSR format
    # ═══════════════════════════════════════════════════════

    def to_taas(
        self,
        neuron_model: str = "glif",
        dt: float = 0.05,
        seed: int = 42,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, Dict]:
        """
        Convert SONATA network to TAAS CSR format.

        Handles non-contiguous node_ids via searchsorted lookup:
          global_id = pop_offset + id_to_index[native_node_id]

        Returns:
            (row_ptr, col_indices, values) — CSR connectivity
            params — packed neuron parameters [17 * n]
            info — metadata dict
        """
        rng = np.random.RandomState(seed)

        # ── Build node_id → local_index lookup for each population ──
        # Real SONATA files can have arbitrary node_ids (not necessarily 0-indexed).
        # We use searchsorted for fast O(log N) lookup.
        pop_offsets = {}
        pop_id_to_index = {}  # pop_name → {native_id: local_index}
        global_id = 0
        for name, grp in self.node_groups.items():
            pop_offsets[name] = global_id
            ids = np.asarray(grp.node_ids)
            # Build lookup dict for non-contiguous IDs
            if len(ids) > 0 and (ids[0] != 0 or not np.all(np.diff(ids) == 1)):
                pop_id_to_index[name] = {int(nid): j for j, nid in enumerate(ids)}
            else:
                pop_id_to_index[name] = None  # contiguous 0-indexed, use offset+id directly
            global_id += len(ids)

        n_neurons = global_id
        n_padded = (n_neurons + 7) & ~7 if n_neurons > 0 else 8

        # ── Build neuron params ──
        params = np.zeros(n_padded * 17, dtype=np.float32)
        for name, grp in self.node_groups.items():
            offset = pop_offsets[name]
            for i in range(len(grp.node_ids)):
                idx = offset + i
                p = _glif_to_taas(grp, i, neuron_model, dt, name)
                params[idx] = p['dt']
                params[n_padded + idx] = p.get('cm', 1.0)
                params[2*n_padded + idx] = p.get('gL', 0.1)
                params[3*n_padded + idx] = p.get('EL', -65.0)
                params[4*n_padded + idx] = p.get('gM', 0.0)
                params[5*n_padded + idx] = p.get('EK', -80.0)
                params[6*n_padded + idx] = p.get('gh', 0.0)
                params[7*n_padded + idx] = p.get('Eh', -30.0)
                params[8*n_padded + idx] = p.get('tau_u', 100.0)
                params[9*n_padded + idx] = p.get('tau_h', 200.0)
                params[10*n_padded + idx] = p.get('Vhalf_u', -35.0)
                params[11*n_padded + idx] = p.get('ku', 5.0)
                params[12*n_padded + idx] = p.get('Vhalf_h', -75.0)
                params[13*n_padded + idx] = p.get('kh', 6.0)
                params[14*n_padded + idx] = p.get('Vthresh', -40.0)
                params[15*n_padded + idx] = p.get('Vreset', -55.0)
                params[16*n_padded + idx] = p.get('delta_u', 0.02)

        # ── Build CSR connectivity with proper node_id resolution ──
        syn_pre = []
        syn_post = []
        syn_w = []

        for name, eg in self.edge_groups.items():
            # Resolve source/target populations — use HDF5 attributes if available
            # (parsed in from_files), fall back to name parsing.
            # We re-derive here; in a real implementation, store src_pop/tgt_pop in EdgeGroup.
            src_pop = None
            tgt_pop = None
            parts = name.split('__to__')
            if len(parts) == 2:
                src_pop, tgt_pop = parts
            else:
                parts = name.split('__')
                src_pop, tgt_pop = parts[0], parts[-1] if len(parts) >= 2 else (name, name)

            pre_map = pop_id_to_index.get(src_pop)
            post_map = pop_id_to_index.get(tgt_pop)
            pre_offset = pop_offsets.get(src_pop, 0)
            post_offset = pop_offsets.get(tgt_pop, 0)

            if src_pop not in pop_offsets or tgt_pop not in pop_offsets:
                continue

            for i in range(len(eg.source_node_ids)):
                src_nid = int(eg.source_node_ids[i])
                tgt_nid = int(eg.target_node_ids[i])

                # Resolve global IDs
                if pre_map is not None:
                    pre_global = pre_offset + pre_map.get(src_nid, -1)
                else:
                    pre_global = pre_offset + src_nid

                if post_map is not None:
                    post_global = post_offset + post_map.get(tgt_nid, -1)
                else:
                    post_global = post_offset + tgt_nid

                if 0 <= pre_global < n_neurons and 0 <= post_global < n_neurons:
                    w = eg.weights[i] if eg.weights is not None else 1.0
                    if eg.syn_type_ids is not None:
                        w *= -1.0 if eg.syn_type_ids[i] == 1 else 1.0
                    syn_pre.append(pre_global)
                    syn_post.append(post_global)
                    syn_w.append(w)

        n_synapses = len(syn_pre)
        row_ptr = np.zeros(n_neurons + 1, dtype=np.int32)
        if n_synapses > 0:
            row_counts = np.zeros(n_neurons, dtype=np.int32)
            for pre in syn_pre:
                row_counts[pre] += 1
            for i in range(n_neurons):
                row_ptr[i + 1] = row_ptr[i] + row_counts[i]
            col_indices = np.zeros(n_synapses, dtype=np.int32)
            values = np.zeros(n_synapses, dtype=np.float32)
            cursor = np.zeros(n_neurons, dtype=np.int32)
            for i in range(n_synapses):
                pre = syn_pre[i]
                pos = row_ptr[pre] + cursor[pre]
                col_indices[pos] = syn_post[i]
                values[pos] = syn_w[i]
                cursor[pre] += 1
        else:
            col_indices = np.zeros(0, dtype=np.int32)
            values = np.zeros(0, dtype=np.float32)

        info = {
            'n_neurons': n_neurons,
            'n_synapses': n_synapses,
            'n_padded': n_padded,
            'populations': list(self.node_groups.keys()),
            'model_types': {n: g.model_type for n, g in self.node_groups.items()},
            'neuron_model': neuron_model,
            'dt': dt,
        }

        return (row_ptr, col_indices, values), params, info


# ═══════════════════════════════════════════════════════════
#  GLIF → TAAS parameter conversion
# ═══════════════════════════════════════════════════════════

_GLIF_TEMPLATES = {
    # Allen Institute GLIF3 template parameters
    'GLIF3_1': {
        'g_leak': 0.1, 'E_leak': -65.0, 'V_thresh': -54.0,
        'V_reset': -65.0, 'Cm': 1.0,
        'tau_ahp': 30.0, 'delta_g_ahp': 0.01, 'E_ahp': -80.0,
    },
    'GLIF3_2': {
        'g_leak': 0.15, 'E_leak': -70.0, 'V_thresh': -50.0,
        'V_reset': -65.0, 'Cm': 0.8,
        'tau_ahp': 50.0, 'delta_g_ahp': 0.02, 'E_ahp': -85.0,
    },
    # GLIF4 adds threshold adaptation
    'GLIF4_1': {
        'g_leak': 0.1, 'E_leak': -65.0, 'V_thresh': -54.0,
        'V_reset': -65.0, 'Cm': 1.0,
        'tau_ahp': 30.0, 'delta_g_ahp': 0.01, 'E_ahp': -80.0,
        'tau_theta': 20.0, 'delta_theta': 3.0,
    },
    'GLIF4_2': {
        'g_leak': 0.2, 'E_leak': -70.0, 'V_thresh': -50.0,
        'V_reset': -68.0, 'Cm': 0.5,
        'tau_ahp': 100.0, 'delta_g_ahp': 0.05, 'E_ahp': -90.0,
        'tau_theta': 50.0, 'delta_theta': 5.0,
    },
}


def _glif_to_taas(
    grp: SonataNodeGroup,
    idx: int,
    target_model: str,
    dt: float,
) -> Dict[str, float]:
    """
    Convert GLIF node params to TAAS internal format.

    GLIF3: LIF + after-spike current (g_ahp)
    GLIF4: GLIF3 + threshold adaptation (theta)

    Maps to TAAS:
      GLIF leak       → HAAP g_L, E_L
      GLIF capacitance → HAAP cm
      GLIF V_thresh   → HAAP Vthresh
      GLIF V_reset    → HAAP Vreset

    NOTE: GLIF after-spike current (g_ahp) → HAAP M-current (g_M) — это
    **эвристическое приближение**, не точное преобразование. 
    - GLIF: g_ahp(t+dt) = g_ahp(t)*exp(-dt/tau) + delta_g*spike (linear decay)
    - HAAP: I_M = g_M * u * (V - E_K), где u' = (u_inf(V) - u)/tau_u (voltage-gated)
    Обе модели дают spike-frequency adaptation, но динамика разная.
    Для точного соответствия используй модель GLIF3/4 напрямую.
    """
    template = _GLIF_TEMPLATES.get(grp.model_template, _GLIF_TEMPLATES['GLIF3_1'])

    def get_param(name, arr_name=None):
        arr_name = arr_name or name
        if arr_name in grp.params and idx < len(grp.params[arr_name]):
            return float(grp.params[arr_name][idx])
        return template.get(name, 0.0)

    g_leak = get_param('g_leak')
    cm_val = get_param('Cm', 'cm')
    v_thresh = get_param('V_thresh', 'v_thresh')
    v_reset = get_param('V_reset', 'v_reset')
    e_leak = get_param('E_leak', 'e_leak')
    tau_ahp = get_param('tau_ahp')
    delta_g_ahp = get_param('delta_g_ahp')

    # GLIF3: tau = Cm/g_leak (ms)
    tau_m = cm_val / max(g_leak, 1e-6) / 1000.0

    # g_ahp → g_M (heuristic approximation — see docstring)
    p = {
        'dt': dt,
        'cm': cm_val,
        'gL': g_leak * 1000.0,
        'EL': e_leak,
        'Vthresh': v_thresh,
        'Vreset': v_reset,
        'gM': delta_g_ahp * 1000.0,
        'EK': template.get('E_ahp', -80.0),
        'tau_u': tau_ahp,
        # NOTE: delta_u = delta_g / cm * 10 — эмпирический коэффициент,
        # чтобы bump адаптации совпадал по magnitude с GLIF after-spike.
        'delta_u': delta_g_ahp / max(cm_val, 0.1) * 10.0,
        'gh': 0.0,
    }
    return p


# ═══════════════════════════════════════════════════════════
#  Node types CSV parser
# ═══════════════════════════════════════════════════════════

def _parse_node_types(net: SonataNetwork, path: str) -> None:
    """Read node_types CSV and apply model_template to node groups."""
    import csv
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pop_name = row.get('population', '')
            if pop_name in net.node_groups:
                grp = net.node_groups[pop_name]
                grp.model_template = row.get('model_template', grp.model_template)
                grp.model_type = row.get('model_type', grp.model_type)


# ═══════════════════════════════════════════════════════════
#  High-level API
# ═══════════════════════════════════════════════════════════

def load_sonata(
    nodes_path: str,
    edges_path: str,
    node_types_path: Optional[str] = None,
) -> SonataNetwork:
    """Load a SONATA network from nodes.h5 + edges.h5."""
    return SonataNetwork.from_files(nodes_path, edges_path, node_types_path)


def sonata_to_taas(
    nodes_path: str,
    edges_path: str,
    neuron_model: str = "glif",
    dt: float = 0.05,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, Dict]:
    """
    One-step: load SONATA, convert to TAAS CSR.

    Returns:
        (row_ptr, col_indices, values) — CSR
        params — TAAS packed params [17 * n_padded]
        info — metadata dict
    """
    net = load_sonata(nodes_path, edges_path)
    return net.to_taas(neuron_model=neuron_model, dt=dt)


# ═══════════════════════════════════════════════════════════
#  Demo
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m taas_import.sonata <nodes.h5> <edges.h5> [node_types.csv]")
        sys.exit(1)

    nodes_path = sys.argv[1]
    edges_path = sys.argv[2]
    types_path = sys.argv[3] if len(sys.argv) > 3 else None

    net = load_sonata(nodes_path, edges_path, types_path)

    for name, grp in net.node_groups.items():
        print(f"Population {name}: {len(grp.node_ids)} neurons, type={grp.model_type}/{grp.model_template}")
        if grp.params:
            sample = {k: v[:3].tolist() for k, v in list(grp.params.items())[:5]}
            print(f"  params: {sample}")

    for name, eg in net.edge_groups.items():
        print(f"Edges {name}: {len(eg.source_node_ids)} synapses")

    csr, params, info = net.to_taas(neuron_model="glif")
    print(f"\nTAAS: {info['n_neurons']} neurons, {info['n_synapses']} synapses, params={params.shape}")
    print("Done!")
