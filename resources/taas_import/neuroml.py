#!/usr/bin/env python3
"""
NeuroML2 → TAAS Checkmate converter
====================================

Parses NeuroML2 (.nml) + LEMS (.lems) files and converts to TAAS
CSR format (row_ptr, col_indices, values) + neuron params.

Поддерживаемые модели:
  - izhikevich2007Cell → TAAS Izhikevich
  - hhCell            → TAAS HH (fused_poly)
  - adexCell          → TAAS AdEx
  - lifCell           → TAAS LIF
  - GLIF3/4           → TAAS LIF+ahp

Зависимости:
  pip install libNeuroML

Пример:
  nml = NeuroMLNetwork.from_file("network.nml")
  csr, params, info = nml.to_taas(neuron_model="lif")
  print(f"Neurons: {info['n_neurons']}, Synapses: {info['n_synapses']}")
"""

import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ═══════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════

@dataclass
class NeuroMLCell:
    """A single cell type definition from NeuroML."""
    id: str
    model_type: str  # 'izhikevich', 'hh', 'adex', 'lif', 'glif'
    params: Dict[str, float] = field(default_factory=dict)
    # Standard NeuroML parameters mapped to internal names
    # Izhikevich: a, b, c, d, v0, thresh
    # HH: gNa, gK, gLeak, eNa, eK, eLeak, Cm
    # AdEx: Cm, gL, EL, VT, DeltaT, tauw, a, b, Vr
    # LIF: Cm, gL, EL, Vthresh, Vreset, tau_ref

@dataclass
class NeuroMLPop:
    """A population of cells."""
    id: str
    component: str  # references NeuroMLCell.id
    size: int
    # Optional: positions (NeuroML can embed x/y/z per instance)
    positions: Optional[np.ndarray] = None  # [size, 3]

@dataclass
class NeuroMLProjection:
    """A projection (connection set) between two populations."""
    id: str
    pre_pop: str
    post_pop: str
    pre_cell_ids: np.ndarray  # [n_synapses] int64
    post_cell_ids: np.ndarray  # [n_synapses] int64
    weights: Optional[np.ndarray] = None  # [n_synapses]
    delays: Optional[np.ndarray] = None  # [n_synapses]
    synapse_type: str = "excitatory"  # 'excitatory', 'inhibitory'

    @property
    def n_synapses(self) -> int:
        return len(self.pre_cell_ids)

@dataclass
class NeuroMLNetwork:
    """Complete parsed NeuroML network."""
    cells: Dict[str, NeuroMLCell] = field(default_factory=dict)
    populations: Dict[str, NeuroMLPop] = field(default_factory=dict)
    projections: List[NeuroMLProjection] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    # ═══════════════════════════════════════════════════════
    #  Parse NeuroML2 .nml file
    # ═══════════════════════════════════════════════════════

    @classmethod
    def from_file(cls, path: str) -> "NeuroMLNetwork":
        """Parse a NeuroML2 .nml file (with LEMS component defs)."""
        path = Path(path)
        ns = {
            'nml': 'http://www.neuroml.org/schema/neuroml2',
            'lems': 'https://lems-project.org/lems',
        }

        tree = ET.parse(path)
        root = tree.getroot()
        net = cls()

        # ── Parse cells (component definitions) ──
        for cell_elem in root.iter(f'{{{ns["nml"]}}}izhikevich2007Cell'):
            cid = cell_elem.get('id')
            params = {
                'a': float(cell_elem.get('a', '0.02')),
                'b': float(cell_elem.get('b', '0.2')),
                'c': float(cell_elem.get('c', '-65.0')),
                'd': float(cell_elem.get('d', '8.0')),
                'v0': float(cell_elem.get('v0', '-65.0')),
                'thresh': float(cell_elem.get('thresh', '30.0')),
            }
            net.cells[cid] = NeuroMLCell(id=cid, model_type='izhikevich', params=params)

        for cell_elem in root.iter(f'{{{ns["nml"]}}}hhCell'):
            cid = cell_elem.get('id')
            params = {
                'gna': float(cell_elem.get('gna', '0.12')),
                'gk': float(cell_elem.get('gk', '0.036')),
                'gleak': float(cell_elem.get('gl', '0.0003')),
                'ena': float(cell_elem.get('ena', '50.0')),
                'ek': float(cell_elem.get('ek', '-77.0')),
                'eleak': float(cell_elem.get('el', '-54.3')),
                'cm': float(cell_elem.get('cm', '1.0')),
            }
            net.cells[cid] = NeuroMLCell(id=cid, model_type='hh', params=params)

        for cell_elem in root.iter(f'{{{ns["nml"]}}}adexCell'):
            cid = cell_elem.get('id')
            params = {
                'cm': float(cell_elem.get('cm', '1.0')),
                'gl': float(cell_elem.get('gl', '0.01')),
                'el': float(cell_elem.get('el', '-70.0')),
                'vt': float(cell_elem.get('vt', '-50.0')),
                'delta_t': float(cell_elem.get('deltaT', '2.0')),
                'tauw': float(cell_elem.get('tauw', '100.0')),
                'a': float(cell_elem.get('a', '0.001')),
                'b': float(cell_elem.get('b', '0.005')),
                'vr': float(cell_elem.get('vr', '-70.0')),
            }
            net.cells[cid] = NeuroMLCell(id=cid, model_type='adex', params=params)

        for cell_elem in root.iter(f'{{{ns["nml"]}}}lifCell'):
            cid = cell_elem.get('id')
            params = {
                'cm': float(cell_elem.get('cm', '1.0')),
                'gl': float(cell_elem.get('gl', '0.01')),
                'el': float(cell_elem.get('el', '-65.0')),
                'vthresh': float(cell_elem.get('thresh', '-54.0')),
                'vreset': float(cell_elem.get('reset', '-65.0')),
                'tauref': float(cell_elem.get('tauRef', '2.0')),
            }
            net.cells[cid] = NeuroMLCell(id=cid, model_type='lif', params=params)

        # ── Parse populations ──
        for pop_elem in root.iter(f'{{{ns["nml"]}}}population'):
            pid = pop_elem.get('id')
            comp = pop_elem.get('component', '')
            size_elem = pop_elem.find(f'{{{ns["nml"]}}}instance')
            size = 0
            if size_elem is not None:
                size = 1
            else:
                size_attr = pop_elem.get('size', '1')
                size = int(size_attr) if size_attr else 1
            net.populations[pid] = NeuroMLPop(id=pid, component=comp, size=size)

        # ── Parse projections (connections) ──
        for proj_elem in root.iter(f'{{{ns["nml"]}}}projection'):
            proj_id = proj_elem.get('id')
            pre_pop = proj_elem.get('presynapticPopulation', '')
            post_pop = proj_elem.get('postsynapticPopulation', '')
            syn_type = proj_elem.get('synapseType', 'excitatory')

            pre_ids = []
            post_ids = []
            weights = []
            delays = []

            for conn in proj_elem.iter(f'{{{ns["nml"]}}}connection'):
                pre_path = conn.get('preCellId', '../pop[0]')
                post_path = conn.get('postCellId', '../pop[1]')
                # Parse "pop[id]" from path
                import re
                pre_match = re.search(r'\[(\d+)\]', pre_path)
                post_match = re.search(r'\[(\d+)\]', post_path)
                pre_id = int(pre_match.group(1)) if pre_match else 0
                post_id = int(post_match.group(1)) if post_match else 0
                pre_ids.append(pre_id)
                post_ids.append(post_id)

                w = conn.get('weight')
                weights.append(float(w) if w else 1.0)
                d = conn.get('delay')
                delays.append(float(d) if d else 1.0)

            if pre_ids:
                net.projections.append(NeuroMLProjection(
                    id=proj_id,
                    pre_pop=pre_pop,
                    post_pop=post_pop,
                    pre_cell_ids=np.array(pre_ids, dtype=np.int64),
                    post_cell_ids=np.array(post_ids, dtype=np.int64),
                    weights=np.array(weights, dtype=np.float32),
                    delays=np.array(delays, dtype=np.float32),
                    synapse_type=syn_type,
                ))

        return net

    # ═══════════════════════════════════════════════════════
    #  Convert to TAAS CSR format
    # ═══════════════════════════════════════════════════════

    def to_taas(
        self,
        neuron_model: str = "lif",
        dt: float = 0.05,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, Dict]:
        """
        Convert NeuroML network to TAAS CSR format.

        Returns:
            (row_ptr, col_indices, values) — CSR connectivity
            params — packed neuron parameters [17 * n_padded]
            info  — dict with n_neurons, n_synapses, cell_types, etc.

        NOTE: Izhikevich models store a,b,c,d in the 17-field array as
        EXTENDED fields at indices [17..20] (not standard HAAP fields).
        The caller must extract them via:
            izh_a = params[17*n_padded : 17*n_padded + n]
            izh_b = params[18*n_padded : 18*n_padded + n]
            # etc.
        """
        # ── Assign global neuron IDs ──
        # Map (population_id, local_id) → global_id
        pop_offsets = {}
        global_id = 0
        cell_types = []
        pop_sizes = {}
        cell_params_list = []

        for pid, pop in self.populations.items():
            pop_offsets[pid] = global_id
            pop_sizes[pid] = pop.size
            cell_type = self.cells.get(pop.component)
            for _ in range(pop.size):
                cell_types.append(pop.component)
                # Build TAAS params from NeuroML cell parameters
                params_dict = _convert_to_taas_params(cell_type, neuron_model, dt)
                cell_params_list.append(params_dict)
                global_id += 1

        n_neurons = global_id

        # ── Build CSR connectivity ──
        # First, count synapses per neuron
        row_counts = np.zeros(n_neurons, dtype=np.int32)
        syn_pre_list = []
        syn_post_list = []
        syn_weight_list = []

        for proj in self.projections:
            pre_offset = pop_offsets.get(proj.pre_pop, 0)
            post_offset = pop_offsets.get(proj.post_pop, 0)

            for i in range(proj.n_synapses):
                pre_global = int(pre_offset + proj.pre_cell_ids[i])
                post_global = int(post_offset + proj.post_cell_ids[i])
                if pre_global < n_neurons and post_global < n_neurons:
                    # Determine sign: inhibitory if synapse_type says so
                    sign = -1.0 if 'inhib' in proj.synapse_type.lower() else 1.0
                    w = proj.weights[i] if proj.weights is not None else 1.0
                    syn_pre_list.append(pre_global)
                    syn_post_list.append(post_global)
                    syn_weight_list.append(sign * w)
                    row_counts[pre_global] += 1

        n_synapses = len(syn_pre_list)

        # Build CSR
        row_ptr = np.zeros(n_neurons + 1, dtype=np.int32)
        for i in range(n_neurons):
            row_ptr[i + 1] = row_ptr[i] + row_counts[i]

        col_indices = np.zeros(n_synapses, dtype=np.int32)
        values = np.zeros(n_synapses, dtype=np.float32)
        cursor = np.zeros(n_neurons, dtype=np.int32)

        for i in range(n_synapses):
            pre = syn_pre_list[i]
            pos = row_ptr[pre] + cursor[pre]
            col_indices[pos] = syn_post_list[i]
            values[pos] = syn_weight_list[i]
            cursor[pre] += 1

        # Build packed params array [21 * n_neurons]
        # 17 standard HAAP fields + 4 Izhikevich extras (a,b,c,d)
        n = n_neurons
        n_padded = max((n + 7) & ~7, 8)
        params = np.zeros(n_padded * 21, dtype=np.float32)
        for i, p in enumerate(cell_params_list):
            if i < n:
                params[i] = p.get('dt', dt)
                params[n_padded + i] = p.get('cm', 1.0)
                params[2*n_padded + i] = p.get('gL', 0.1)
                params[3*n_padded + i] = p.get('EL', -65.0)
                params[4*n_padded + i] = p.get('gM', 0.5)
                params[5*n_padded + i] = p.get('EK', -80.0)
                params[6*n_padded + i] = p.get('gh', 0.3)
                params[7*n_padded + i] = p.get('Eh', -30.0)
                params[8*n_padded + i] = p.get('tau_u', 100.0)
                params[9*n_padded + i] = p.get('tau_h', 200.0)
                params[10*n_padded + i] = p.get('Vhalf_u', -35.0)
                params[11*n_padded + i] = p.get('ku', 5.0)
                params[12*n_padded + i] = p.get('Vhalf_h', -75.0)
                params[13*n_padded + i] = p.get('kh', 6.0)
                params[14*n_padded + i] = p.get('Vthresh', -40.0)
                params[15*n_padded + i] = p.get('Vreset', -55.0)
                params[16*n_padded + i] = p.get('delta_u', 0.02)
                # Izhikevich extras at indices [17..20]
                params[17*n_padded + i] = p.get('izh_a', 0.02)
                params[18*n_padded + i] = p.get('izh_b', 0.2)
                params[19*n_padded + i] = p.get('izh_c', -65.0)
                params[20*n_padded + i] = p.get('izh_d', 8.0)

        info = {
            'n_neurons': n_neurons,
            'n_synapses': n_synapses,
            'n_padded': n_padded,
            'params_cols': 21,
            'izh_offset': 17,  # start of Izhikevich a,b,c,d in params
            'populations': list(self.populations.keys()),
            'pop_sizes': pop_sizes,
            'cell_types': cell_types,
            'neuron_model': neuron_model,
            'dt': dt,
        }

        return (row_ptr, col_indices, values), params, info


# ═══════════════════════════════════════════════════════════
#  Parameter conversion: NeuroML cell → TAAS params
# ═══════════════════════════════════════════════════════════

_HAAP_DEFAULTS = {
    'cm': 1.0, 'gL': 0.1, 'EL': -65.0,
    'gM': 0.5, 'EK': -80.0, 'gh': 0.3, 'Eh': -30.0,
    'tau_u': 100.0, 'tau_h': 200.0,
    'Vhalf_u': -35.0, 'ku': 5.0, 'Vhalf_h': -75.0, 'kh': 6.0,
    'Vthresh': -40.0, 'Vreset': -55.0, 'delta_u': 0.02,
}


def _to_haap_like(p_in: Dict[str, float], dt: float) -> Dict[str, float]:
    """Build the 17 common HAAP-style fields from a param dict."""
    base = _HAAP_DEFAULTS.copy()
    base['dt'] = dt
    base.update(p_in)
    return base


def _convert_to_taas_params(
    cell: Optional[NeuroMLCell],
    target_model: str,
    dt: float,
) -> Dict[str, float]:
    """
    Convert NeuroML cell parameters to TAAS internal params format.

    Returns dict with 21 keys: 17 standard + 4 Izhikevich extras (izh_a,b,c,d).

    Maps:
      izhikevich2007Cell → +izh_a,b,c,d
      hhCell            → conductances converted to mS/cm²
      adexCell          → AdEx params
      lifCell           → LIF params (tau_m = Cm/gL)

    NOTE: `step_izhikevich()` expects a,b,c,d as separate arrays.
    Extract from params[17*n_padded : 21*n_padded].
    """

    if cell is None:
        return _to_haap_like({}, dt)

    p = cell.params

    if cell.model_type == 'izhikevich':
        base = _to_haap_like({
            'EL': float(p.get('v0', -65.0)),
            'Vthresh': float(p.get('thresh', 30.0)),
        }, dt)
        base['izh_a'] = float(p.get('a', 0.02))
        base['izh_b'] = float(p.get('b', 0.2))
        base['izh_c'] = float(p.get('c', -65.0))
        base['izh_d'] = float(p.get('d', 8.0))
        return base

    elif cell.model_type == 'hh':
        return _to_haap_like({
            'cm': float(p.get('cm', 1.0)),
            'gL': float(p.get('gleak', 0.0003)) * 1000.0,
            'EL': float(p.get('eleak', -54.3)),
            'gM': 0.0, 'gh': 0.0,
        }, dt)

    elif cell.model_type == 'adex':
        return _to_haap_like({
            'cm': float(p.get('cm', 1.0)),
            'gL': float(p.get('gl', 0.01)) * 1000.0,
            'EL': float(p.get('el', -70.0)),
            'gM': 0.0, 'gh': 0.0,
        }, dt)

    elif cell.model_type == 'lif':
        cm_val = float(p.get('cm', 1.0))
        gl_val = float(p.get('gl', 0.01))
        return _to_haap_like({
            'cm': cm_val,
            'gL': gl_val * 1000.0,
            'EL': float(p.get('el', -65.0)),
            'Vthresh': float(p.get('vthresh', -54.0)),
            'Vreset': float(p.get('vreset', -65.0)),
            'gM': 0.0, 'gh': 0.0,
        }, dt)

    elif cell.model_type == 'glif':
        return _to_haap_like({
            'cm': float(p.get('cm', 1.0)),
            'gL': float(p.get('gleak', 0.1)),
            'EL': float(p.get('eleak', -65.0)),
            'Vthresh': float(p.get('vthresh', -54.0)),
            'Vreset': float(p.get('vreset', -65.0)),
            'gM': float(p.get('delta_g_ahp', 0.01)) * 1000.0,
            'EK': float(p.get('e_ahp', -80.0)),
            'tau_u': float(p.get('tau_ahp', 30.0)),
            'delta_u': float(p.get('delta_g_ahp', 0.01)),
        }, dt)

    return _to_haap_like({}, dt)


# ═══════════════════════════════════════════════════════════
#  High-level API
# ═══════════════════════════════════════════════════════════

def load_neuroml(path: str) -> NeuroMLNetwork:
    """Load a NeuroML2 network from .nml file."""
    return NeuroMLNetwork.from_file(path)


def neuroml_to_taas(
    path: str,
    neuron_model: str = "lif",
    dt: float = 0.05,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, Dict]:
    """
    One-step: load NeuroML, convert to TAAS.

    Returns:
        (row_ptr, col_indices, values) — CSR
        params — TAAS packed params [17 * n_padded]
        info — metadata dict
    """
    net = load_neuroml(path)
    return net.to_taas(neuron_model=neuron_model, dt=dt)


# ═══════════════════════════════════════════════════════════
#  Demo (if run directly)
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m taas_import.neuroml <network.nml>")
        sys.exit(1)

    net = load_neuroml(sys.argv[1])
    print(f"Cells: {list(net.cells.keys())}")
    print(f"Populations: {list(net.populations.keys())}")
    for pid, pop in net.populations.items():
        print(f"  {pid}: {pop.size}x {pop.component}")
    for proj in net.projections:
        print(f"  Proj: {proj.id} ({proj.pre_pop}→{proj.post_pop}, {proj.n_synapses} synapses)")

    csr, params, info = net.to_taas(neuron_model="lif")
    print(f"\nTAAS CSR: {info['n_neurons']} neurons, {info['n_synapses']} synapses")
    print(f"Params: {params.shape}")
    print("Done!")
