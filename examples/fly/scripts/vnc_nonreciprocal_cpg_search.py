#!/usr/bin/env python3
"""
🔬 VNC NON-RECIPROCAL CPG SEARCH — Find Pugliese-style walking CPG
===================================================================
Задача: найти Pugliese 3-neuron microcircuit (2 exc + 1 inh) в той части
VNC, которая НЕ входит в 41 реципрокный GABA кластер.

Гипотеза: настоящий walking CPG (5-10 Hz) находится среди ~18K нейронов
VNC, которые не являются частью reciprocal GABA network. Там есть ACh/GLUT
нейроны с правильной directed feedback архитектурой.

Метод:
1. Загрузить MANC
2. Найти 41 GABA кластер (как в find_vnc_ring.py)
3. Non-reciprocal VNC = все нейроны НЕ в кластерах
4. Сканировать non-reciprocal VNC на directed 3-cycles (2 exc + 1 inh)
5. HH симуляция на лучших кандидатах
6. Проверить walking frequency (5-10 Hz)

MANC data: /home/taaslab/Загрузки/
"""

import os, sys, csv, time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

DLOAD = os.environ.get("FLYWIRE_DATA", "/home/taaslab/Загрузки")
NEURONS_CSV = os.path.join(DLOAD, "v1.0_manc-traced-adjacencies-v1.0_traced-neurons.csv")
CONNS_CSV  = os.path.join(DLOAD, "v1.0_manc-traced-adjacencies-v1.0_traced-connections.csv")
PROPS_FTR = os.path.join(DLOAD, "v1.0_manc-v1.0-neuron-properties.feather")

import taas_checkmate as tc


# ═══════════════════════════════════════════════════════════════════════
#  MANC LOADER (minimal, reuses full_adj)
# ═══════════════════════════════════════════════════════════════════════

class MANCLoader:
    def __init__(self):
        self.neurons = {}
        self.synapses = []
        self.nt_map = {}; self.type_map = {}; self.class_map = {}
        self.subclass_map = {}; self.modal_map = {}
        self.full_adj = defaultdict(set)
        self.full_weights = {}
        self.clusters = {}; self.cluster_membership = {}
        self.reciprocal_pairs = []
        self.n_synapses = 0

    def load(self, max_syn=None):
        t0 = time.time()
        print("📥 LOADING MANC...")
        self._load_properties()
        self._load_neurons()
        self._load_connections(max_syn)
        print(f"   ✅ {len(self.neurons)} meta, {self.n_synapses:,} syn [{time.time()-t0:.1f}s]")

    def _load_properties(self):
        import pandas as pd
        props = pd.read_feather(PROPS_FTR)
        for _, r in props.iterrows():
            bid = int(r['bodyId'])
            self.neurons.setdefault(bid, {'type':'','nt':'','class':'','subclass':'','modality':''})
            nt = str(r.get('predictedNt','')).strip().lower()
            if nt and nt not in ('nan','unknown'):
                self.nt_map[bid] = nt; self.neurons[bid]['nt'] = nt
            t = str(r.get('type','')).strip()
            if t and t!='nan':
                self.type_map[bid] = t; self.neurons[bid]['type'] = t
            c = str(r.get('class','')).strip()
            if c and c!='nan':
                self.class_map[bid] = c; self.neurons[bid]['class'] = c
            sc = str(r.get('subclass','')).strip()
            if sc and sc!='nan':
                self.subclass_map[bid] = sc; self.neurons[bid]['subclass'] = sc
            m = str(r.get('modality','')).strip()
            if m and m!='nan':
                self.modal_map[bid] = m; self.neurons[bid]['modality'] = m

    def _load_neurons(self):
        if not os.path.exists(NEURONS_CSV):
            return
        with open(NEURONS_CSV) as f:
            rdr = csv.reader(f); next(rdr)
            for row in rdr:
                bid = int(row[0]); ntype = row[1].strip()
                self.neurons.setdefault(bid, {'type':ntype,'nt':'','class':'','subclass':'','modality':''})
                self.neurons[bid]['type'] = ntype
                self.type_map[bid] = ntype

    def _load_connections(self, max_syn=None):
        t0 = time.time()
        self.synapses = []; self.full_adj.clear(); self.full_weights.clear()
        with open(CONNS_CSV) as f:
            rdr = csv.reader(f); next(rdr)
            for i, row in enumerate(rdr):
                if max_syn and i >= max_syn:
                    break
                p = int(row[0]); o = int(row[1]); w = float(row[2])
                self.synapses.append((p, o, w))
                self.full_adj[p].add(o)
                self.full_weights[(p, o)] = w
        self.n_synapses = len(self.synapses)
        print(f"      {self.n_synapses:,} synapses, full_adj={len(self.full_adj)} presynaptic [{time.time()-t0:.1f}s]")

    def find_clusters(self):
        """Find 41 reciprocal GABA clusters (identical to find_vnc_ring.py)."""
        t0 = time.time()
        print("\n🔬 FINDING RECIPROCAL GABA CLUSTERS...")
        gaba_set = set(bid for bid, nt in self.nt_map.items() if nt == 'gaba')
        print(f"   GABA neurons: {len(gaba_set):,}")

        gaba_adj = defaultdict(set); gaba_w = {}
        for p, posts in self.full_adj.items():
            if p not in gaba_set:
                continue
            for o in posts:
                if o in gaba_set:
                    gaba_adj[p].add(o)
                    gaba_w[(p,o)] = self.full_weights.get((p,o), 0)

        n_ge = sum(len(v) for v in gaba_adj.values())
        print(f"   GABA→GABA edges: {n_ge:,} [{time.time()-t0:.1f}s]")

        t1 = time.time()
        recip = []
        for pre, posts in gaba_adj.items():
            for post in posts:
                if pre >= post:
                    continue
                if pre in gaba_adj.get(post, set()):
                    recip.append((pre, gaba_w.get((pre,post),0), post, gaba_w.get((post,pre),0)))
        self.reciprocal_pairs = recip

        strong = [(a,w_ab,b,w_ba) for a,w_ab,b,w_ba in recip if abs(w_ab)>1 and abs(w_ba)>1]
        adj = defaultdict(set)
        for a,_,b,_ in strong:
            adj[a].add(b); adj[b].add(a)

        visited = set(); comps = []
        for node in adj:
            if node in visited:
                continue
            q = [node]; comp = set()
            while q:
                n = q.pop(0)
                if n in visited:
                    continue
                visited.add(n); comp.add(n)
                for nb in adj[n]:
                    if nb not in visited:
                        q.append(nb)
            if len(comp) >= 2:
                comps.append(comp)

        comps.sort(key=len, reverse=True)
        for ci, comp in enumerate(comps):
            self.clusters[ci+1] = comp
            for bid in comp:
                self.cluster_membership[bid] = ci+1

        print(f"   Reciprocal pairs: {len(recip):,} [{time.time()-t1:.1f}s]")
        print(f"   Clusters: {len(comps)}, sizes: {[len(c) for c in comps[:10]]}")
        return comps

    def build_csr(self, keep_ids):
        """Build CSR for a subset of bodyIds."""
        t0 = time.time()
        keep = set(keep_ids)
        syn = [(p,o,w) for p,o,w in self.synapses if p in keep and o in keep]
        all_ids = sorted(set(p for p,o,w in syn) | set(o for p,o,w in syn))
        b2i = {bid:i for i,bid in enumerate(all_ids)}
        N = len(all_ids); S = len(syn)
        if S == 0:
            return None, None, None, None, 0

        pre_cnt = Counter(p for p,o,w in syn)
        row_ptr = np.zeros(N+1, dtype=np.int32)
        for bid, cnt in pre_cnt.items():
            row_ptr[b2i[bid]+1] = cnt
        row_ptr = np.cumsum(row_ptr, dtype=np.int32)

        col = np.zeros(S, dtype=np.int32); val = np.ones(S, dtype=np.float32)
        nt_s = np.ones(N, dtype=np.float32)
        NT_S = {'gaba':-1.,'acetylcholine':1.,'glutamate':1.,'dopamine':-.5,'serotonin':-.5,'octopamine':.5}
        cursor = np.zeros(N, dtype=np.int32)
        for p,o,w in syn:
            pi = b2i[p]; oi = b2i[o]
            pos = row_ptr[pi] + cursor[pi]
            col[pos] = np.int32(oi); val[pos] = float(w); cursor[pi] += 1
        for bid, idx in b2i.items():
            nt_s[idx] = NT_S.get(self.nt_map.get(bid,''), 1.0)

        print(f"      CSR: {S} edges, {N} nodes [{time.time()-t0:.1f}s]")
        return row_ptr, col, val, nt_s, N


# ═══════════════════════════════════════════════════════════════════════
#  NON-RECIPROCAL CPG SEARCH
# ═══════════════════════════════════════════════════════════════════════

def find_nonreciprocal_vnc(loader):
    """Find all VNC neurons NOT in the 41 GABA clusters.

    Non-reciprocal VNC = MANC neurons that:
    - Have a class/type (are in VNC)
    - Are NOT in any of the 41 reciprocal GABA clusters
    """
    cluster_neurons = set()
    for comp in loader.clusters.values():
        cluster_neurons |= comp

    # All neurons with VNC metadata (from properties)
    all_vnc = set(loader.neurons.keys())

    # Also add neurons that appear in synapses (traced)
    traced = set(p for p,o,w in loader.synapses) | set(o for p,o,w in loader.synapses)
    all_vnc |= traced

    nonrecip = all_vnc - cluster_neurons
    return nonrecip, cluster_neurons


def scan_3neuron_microcircuits(loader, candidate_pool, max_exc=300, max_inh=300):
    """Scan a pool of neurons for Pugliese-style 3-neuron microcircuits.

    Uses full_adj + full_weights from loader.
    Only checks neurons in candidate_pool.
    """
    t0 = time.time()
    exc_candidates = [b for b in candidate_pool
                      if loader.nt_map.get(b,'') in ('acetylcholine','glutamate')]
    inh_candidates = [b for b in candidate_pool
                      if loader.nt_map.get(b,'') == 'gaba']
    other_candidates = candidate_pool - set(exc_candidates) - set(inh_candidates)

    print(f"\n   Non-reciprocal pool: {len(candidate_pool):,} neurons")
    print(f"   Excitatory (ACh/GLUT): {len(exc_candidates):,}")
    print(f"   Inhibitory (GABA): {len(inh_candidates):,}")
    print(f"   Other/Unknown: {len(other_candidates):,}")

    if len(exc_candidates) < 2 or len(inh_candidates) < 1:
        print("   ❌ Not enough exc/inh neurons for Pugliese circuit")
        return []

    # Sample for performance
    exc_s = exc_candidates[:max_exc]; inh_s = inh_candidates[:max_inh]
    print(f"   Searching {len(exc_s)} exc × {len(inh_s)} inh = "
          f"{len(exc_s)*(len(exc_s)-1)//2 * len(inh_s):,} triplets...")

    results = []; checked = set(); found = 0
    adj = loader.full_adj; weights = loader.full_weights

    for i, e1 in enumerate(exc_s):
        if i % 50 == 0 and i > 0:
            print(f"      {i}/{len(exc_s)} exc neurons, {found} circuits found [{time.time()-t0:.1f}s]")
        e1_targets = adj.get(e1, set()) & candidate_pool

        for e2 in exc_s[i+1:]:
            if e1 == e2:
                continue
            e2_targets = adj.get(e2, set()) & candidate_pool

            # Quick pre-check: e1→e2 or e2→e1 must exist
            if e2 not in e1_targets and e1 not in e2_targets:
                continue

            for inh_n in inh_s:
                key = tuple(sorted([e1, e2, inh_n]))
                if key in checked:
                    continue
                checked.add(key)            # Check all 6 directed 3-cycle topologies
            for n1, n2, n3, n4 in [(e1,e2,inh_n,e1), (e1,inh_n,e2,e1),
                                    (e2,e1,inh_n,e2), (e2,inh_n,e1,e2),
                                    (inh_n,e1,e2,inh_n), (inh_n,e2,e1,inh_n)]:
                ok = (n2 in adj.get(n1, set()) and n3 in adj.get(n2, set()) and n4 in adj.get(n3, set()))
                if not ok:
                    continue

                    # Check: NO reciprocal pairs in circuit (Pugliese)
                    recip = 0
                    for a,b in [(n1,n2),(n2,n3),(n3,n1)]:
                        if b in adj.get(a,set()) and a in adj.get(b,set()):
                            recip += 1
                    if recip > 0:
                        continue

                    w12 = weights.get((n1,n2),0)
                    w23 = weights.get((n2,n3),0)
                    w31 = weights.get((n3,n1),0)

                    results.append({
                        'n1': n1, 'n2': n2, 'n3': n3,
                        'e1': e1, 'e2': e2, 'inh': inh_n,
                        'weights': (w12, w23, w31),
                        'mean_w': (w12 + w23 + w31) / 3,
                        't1': loader.type_map.get(n1,'?'),
                        't2': loader.type_map.get(n2,'?'),
                        't3': loader.type_map.get(n3,'?'),
                        'c1': loader.class_map.get(n1,'?'),
                        'c2': loader.class_map.get(n2,'?'),
                        'c3': loader.class_map.get(n3,'?'),
                        'nt1': loader.nt_map.get(n1,'?'),
                        'nt2': loader.nt_map.get(n2,'?'),
                        'nt3': loader.nt_map.get(n3,'?'),
                        'sub1': loader.subclass_map.get(n1,''),
                        'sub2': loader.subclass_map.get(n2,''),
                        'sub3': loader.subclass_map.get(n3,''),
                        'modal1': loader.modal_map.get(n1,''),
                        'modal2': loader.modal_map.get(n2,''),
                        'modal3': loader.modal_map.get(n3,''),
                    })
                    found += 1

    results.sort(key=lambda x: -x['mean_w'])
    print(f"   Found: {len(results)} Pugliese-type circuits [{time.time()-t0:.1f}s]")
    return results


def simulate_circuit_hh(loader, circuit, n_steps=5000, dt_ms=0.5, external_I_val=10.0):
    """Run HH on a 3-neuron microcircuit and check oscillation frequency."""
    neurons = {circuit['n1'], circuit['n2'], circuit['n3']}
    if len(neurons) < 3:
        return None

    n_type = f"{circuit['t1'][:8]}→{circuit['t2'][:8]}→{circuit['t3'][:8]}"
    print(f"\n   🎬 HH SIM: {n_type} (w={circuit['mean_w']:.1f})")
    t0 = time.time()

    row_ptr, col_idx, vals, nt_s, N = loader.build_csr(list(neurons))
    if N < 3:
        return None

    conn_data = tc.ConnectomeData(row_ptr, col_idx, vals, nt_s)
    sim = tc.BrainSimulator(N, "hh", pacemaking=True)
    I_ext = np.full(N, external_I_val, dtype=np.float32)

    spike_buf = sim.step_batch(conn_data, n_steps=n_steps, dt_ms=dt_ms,
                               external_I=I_ext, scale=1.0)
    sim_t = time.time() - t0
    spikes = np.asarray(spike_buf, dtype=np.int32).reshape(n_steps, N)
    total_spikes = int(spikes.sum())
    fr = spikes.mean(axis=0) * (1000.0 / dt_ms)
    active = int((fr > 1.0).sum())
    mean_fr = fr.mean()

    # Find dominant oscillation frequency via FFT
    sigma_s = 10.0; g_r = int(4 * sigma_s)
    gx = np.arange(-g_r, g_r+1); gk = np.exp(-0.5*(gx/sigma_s)**2); gk /= gk.sum()
    dt = dt_ms / 1000.0; n_fft = 512

    osc_freqs = []
    for idx in range(N):
        if fr[idx] <= 1.0:
            continue
        st = spikes[:, idx].astype(np.float32)
        sm = np.convolve(st, gk, mode='same')
        fft = np.fft.fft(sm, n=n_fft)
        mag = np.abs(fft[:n_fft//2])
        freqs = np.fft.fftfreq(n_fft, d=dt)[:n_fft//2]
        lo = max(1, int(1.0/(freqs[1]-freqs[0]+1e-10)))
        if lo >= len(mag):
            continue
        pk = np.argmax(mag[lo:]) + lo
        if pk < len(freqs) and 1.0 < freqs[pk] < 50:
            osc_freqs.append(freqs[pk])

    is_walking = any(4 <= f <= 15 for f in osc_freqs)
    is_gamma = any(30 <= f <= 100 for f in osc_freqs)
    mean_osc = np.mean(osc_freqs) if osc_freqs else 0.0

    print(f"      {total_spikes} spk, {active}/{N} act, μ={mean_fr:.1f}Hz, "
          f"osc={osc_freqs}, walk={is_walking} [{sim_t:.1f}s]")

    return {
        'circuit': circuit, 'total_spikes': total_spikes,
        'active': active, 'mean_fr': mean_fr,
        'osc_freqs': osc_freqs, 'mean_osc': mean_osc,
        'is_walking': is_walking, 'is_gamma': is_gamma,
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("═" * 70)
    print("🔬 VNC NON-RECIPROCAL CPG SEARCH — Pugliese Circuit Hunt")
    print("═" * 70)

    # 1. Load MANC
    print("\n📥 STEP 1: LOAD MANC")
    loader = MANCLoader()
    loader.load()

    # 2. Find 41 GABA clusters
    print("\n🔬 STEP 2: FIND GABA CLUSTERS")
    comps = loader.find_clusters()

    cluster_total = sum(len(c) for c in comps)
    print(f"\n   Total neurons in GABA clusters: {cluster_total:,}")
    print(f"   Total clusters: {len(comps)}")

    # 3. Non-reciprocal VNC
    print(f"\n🎯 STEP 3: NON-RECIPROCAL VNC")
    print("─" * 70)
    nonrecip_pool, cluster_neurons = find_nonreciprocal_vnc(loader)
    print(f"   Neurons in GABA clusters: {len(cluster_neurons):,}")
    print(f"   Non-reciprocal VNC: {len(nonrecip_pool):,}")

    # 4. Scan for Pugliese circuits
    print(f"\n🔬 STEP 4: SEARCH PUGLIESE CIRCUITS")
    print("─" * 70)
    circuits = scan_3neuron_microcircuits(loader, nonrecip_pool,
                                          max_exc=300, max_inh=300)

    # 5. Results
    print(f"\n{'='*70}")
    print(f"🏆 PUGLIESE-TYPE CIRCUITS FOUND: {len(circuits)}")
    print(f"{'='*70}")

    if circuits:
        print(f"\n{'Rank':<5} {'N1 Type':<20} {'N2 Type':<20} {'N3 Type':<20} "
              f"{'N1 Class':<15} {'N2 Class':<15} {'N3 Class':<15} {'MeanW':<7} {'Freq?':<8}")
        print("-" * 130)

        for rank, mc in enumerate(circuits[:30]):
            walk_mark = ""
            print(f"#{rank+1:<3} {mc['t1'][:18]:<20} {mc['t2'][:18]:<20} {mc['t3'][:18]:<20} "
                  f"{mc['c1'][:13]:<15} {mc['c2'][:13]:<15} {mc['c3'][:13]:<15} "
                  f"{mc['mean_w']:<7.1f} {walk_mark:<8}")

    else:
        print("   ❌ No Pugliese-type circuits found in non-reciprocal VNC!")

    # 6. HH Simulation on top candidates
    if circuits:
        print(f"\n{'='*70}")
        print("🎬 STEP 5: HH SIMULATION ON TOP 10 CIRCUITS")
        print(f"{'='*70}")

        walking_circuits = []
        for i, mc in enumerate(circuits[:10]):
            print(f"\n   ─── Circuit #{i+1} ───")
            result = simulate_circuit_hh(loader, mc, n_steps=3000, dt_ms=0.5)
            if result and result['is_walking']:
                walking_circuits.append(result)
                print(f"      ✅ WALKING CPG! Freq: {result['osc_freqs']}")

        # 7. Walking CPG candidates
        if walking_circuits:
            print(f"\n{'🏆'*35}")
            print(f"🏆 WALKING CPG FOUND! {len(walking_circuits)} candidates")
            print(f"{'🏆'*35}")
            for r in walking_circuits:
                mc = r['circuit']
                print(f"   {mc['t1']} → {mc['t2']} → {mc['t3']}")
                print(f"      Freq: {r['osc_freqs']}Hz")
                print(f"      Weights: {mc['weights']}")
                print(f"      Classes: {mc['c1']}/{mc['c2']}/{mc['c3']}")
        else:
            print(f"\n   ❌ No walking CPG found (0 Hz oscillation)")

    # 8. Scientific verdict
    print(f"\n{'='*70}")
    print("🔬 SCIENTIFIC VERDICT")
    print(f"{'='*70}")

    total_vnc = len(nonrecip_pool) + len(cluster_neurons)
    print(f"\n📊 VNC COMPOSITION:")
    print(f"   VNC total:          {total_vnc:,}")
    print(f"   GABA clusters:      {len(cluster_neurons):,} neurons")
    print(f"   Non-reciprocal:     {len(nonrecip_pool):,} neurons")
    print(f"   Pugliese circuits:  {len(circuits)}")
    print(f"   Walking CPG:        {'✅' if any(r.get('is_walking') for r in locals().get('walking_circuits', [])) else '❌'}")

    if len(circuits) == 0:
        print(f"\n⚠️  НЕ НАЙДЕНО! Возможные причины:")
        print(f"   1. Pugliese CPG использует другие NT (не ACh/GLUT/GABA)")
        print(f"   2. CPG состоит из >3 нейронов")
        print(f"   3. CPG нейроны находятся в кластерных регионах")
        print(f"   4. MANC traced не включает все VNC нейроны")

    print(f"\n✅ SEARCH COMPLETE!")


if __name__ == "__main__":
    main()
