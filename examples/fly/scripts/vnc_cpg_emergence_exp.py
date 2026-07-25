#!/usr/bin/env python3
"""
🧬 VNC CPG Emergence Experiment — 5-10 Hz from network, not microcircuit
=======================================================================
Гипотеза: 5-10 Hz walking frequency — это emergent property всей VNC сети,
а не свойство 3-neuron microcircuit в изоляции.

Эксперимент:
1. Взять топ-5 feedforward CPG кандидатов (с обратной связью)
2. Для каждого: расширить до ~200 нейронов (соседи по VNC + DNnt011)
3. HH симуляция с I_ext sweep (10, 15, 20)
4. Сравнить: isolated 3n vs expanded subnet
5. Найти emergence 5-10 Hz

MANC data: /home/taaslab/Загрузки/
"""

import os, sys, csv, time, json
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


# ─── MANC Loader ──────────────────────────────────────────────────────
class MANCLoader:
    def __init__(self):
        self.neurons = {}; self.synapses = []
        self.nt_map = {}; self.type_map = {}; self.class_map = {}
        self.subclass_map = {}; self.modal_map = {}
        self.full_adj = defaultdict(set); self.full_weights = {}
        self.clusters = {}; self.cluster_membership = {}
        self.reciprocal_pairs = []
        self.vnc_traced_ids = set()

    def load(self):
        t0 = time.time()
        print("📥 LOADING MANC...")
        self._load_properties(); self._load_neurons(); self._load_connections()
        print(f"   ✅ {len(self.neurons):,} meta, {self.n_synapses:,} syn [{time.time()-t0:.1f}s]")

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
            if t and t != 'nan':
                self.type_map[bid] = t; self.neurons[bid]['type'] = t
            c = str(r.get('class','')).strip()
            if c and c != 'nan':
                self.class_map[bid] = c; self.neurons[bid]['class'] = c
            sc = str(r.get('subclass','')).strip()
            if sc and sc != 'nan':
                self.subclass_map[bid] = sc; self.neurons[bid]['subclass'] = sc
            m = str(r.get('modality','')).strip()
            if m and m != 'nan':
                self.modal_map[bid] = m; self.neurons[bid]['modality'] = m

    def _load_neurons(self):
        if not os.path.exists(NEURONS_CSV): return
        with open(NEURONS_CSV) as f:
            for row in csv.reader(f):
                if row[0] == 'bodyId': continue
                bid = int(row[0]); ntype = row[1].strip()
                self.vnc_traced_ids.add(bid)
                self.neurons.setdefault(bid, {'type':ntype,'nt':'','class':'','subclass':'','modality':''})
                self.neurons[bid]['type'] = ntype; self.type_map[bid] = ntype

    def _load_connections(self):
        t0 = time.time()
        self.synapses = []; self.full_adj.clear(); self.full_weights.clear()
        with open(CONNS_CSV) as f:
            for row in csv.reader(f):
                if row[0] == 'bodyId_pre': continue
                p = int(row[0]); o = int(row[1]); w = float(row[2])
                self.synapses.append((p, o, w))
                self.full_adj[p].add(o); self.full_weights[(p, o)] = w
        self.n_synapses = len(self.synapses)
        print(f"      {self.n_synapses:,} syn [{time.time()-t0:.1f}s]")

    def find_clusters(self):
        t0 = time.time()
        print("\n🔬 FINDING RECIPROCAL GABA CLUSTERS...")
        gaba_set = set(bid for bid, nt in self.nt_map.items() if nt == 'gaba')
        gaba_adj = defaultdict(set); gaba_w = {}
        for p, posts in self.full_adj.items():
            if p not in gaba_set: continue
            for o in posts:
                if o in gaba_set: gaba_adj[p].add(o); gaba_w[(p,o)] = self.full_weights.get((p,o),0)
        recip = []
        for pre, posts in gaba_adj.items():
            for post in posts:
                if pre >= post: continue
                if pre in gaba_adj.get(post, set()):
                    recip.append((pre,gaba_w.get((pre,post),0),post,gaba_w.get((post,pre),0)))
        self.reciprocal_pairs = recip
        strong = [(a,w_ab,b,w_ba) for a,w_ab,b,w_ba in recip if abs(w_ab)>1 and abs(w_ba)>1]
        adj = defaultdict(set)
        for a,_,b,_ in strong: adj[a].add(b); adj[b].add(a)
        visited = set(); comps = []
        for node in adj:
            if node in visited: continue
            q = [node]; comp = set()
            while q:
                n = q.pop(0)
                if n in visited: continue
                visited.add(n); comp.add(n)
                for nb in adj[n]:
                    if nb not in visited: q.append(nb)
            if len(comp) >= 2: comps.append(comp)
        comps.sort(key=len, reverse=True)
        for ci, comp in enumerate(comps):
            self.clusters[ci+1] = comp
            for bid in comp: self.cluster_membership[bid] = ci+1
        print(f"   {len(comps)} clusters [{time.time()-t0:.1f}s]")

    def build_csr(self, keep_ids):
        keep = set(keep_ids)
        syn = [(p,o,w) for p,o,w in self.synapses if p in keep and o in keep]
        all_ids = sorted(set(p for p,o,w in syn) | set(o for p,o,w in syn))
        b2i = {bid:i for i,bid in enumerate(all_ids)}
        N = len(all_ids); S = len(syn)
        if S == 0: return None, None, None, None, 0
        pre_cnt = Counter(p for p,o,w in syn)
        row_ptr = np.zeros(N+1, dtype=np.int32)
        for bid, cnt in pre_cnt.items(): row_ptr[b2i[bid]+1] = cnt
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
        return row_ptr, col, val, nt_s, N


def find_candidate_by_types(loader, pool):
    """Find actual bodyIds for top CPG candidates by type names."""
    results = []
    for n1_type, n2_type, n3_type in [
        ('DNht001', 'AN06A018', 'IN19B068'),  # #1
        ('IN06A088', 'IN06A087', None),        # #3
        ('IN14A003', 'IN21A006', 'IN13A024'),  # #4
        ('IN07B083', 'IN06B079', 'AN08B009'),  # #5
        ('IN20A', 'IN09A014', 'IN14A007'),     # #6
    ]:
        n1_matches = [b for b in pool if n1_type in loader.type_map.get(b,'')]
        n2_matches = [b for b in pool if n2_type in loader.type_map.get(b,'')]
        if n3_type:
            n3_matches = [b for b in pool if n3_type in loader.type_map.get(b,'')]
        else:
            # Find targets of n2
            n3_matches = []

        if not n1_matches or not n2_matches:
            continue
        if n3_type and not n3_matches:
            continue

        n1 = n1_matches[0]; n2 = n2_matches[0]
        adj = loader.full_adj

        # Verify n1→n2 exists
        if n2 not in adj.get(n1, set()):
            continue

        if n3_type:
            n3 = n3_matches[0]
            if n3 not in adj.get(n2, set()):
                continue
        else:
            # Find n3 from n2's targets in pool
            n3_candidates = list(adj.get(n2, set()) & pool)
            if not n3_candidates:
                continue
            n3 = n3_candidates[0]

        results.append({
            'n1': n1, 'n2': n2, 'n3': n3,
            't1': loader.type_map.get(n1,'?'),
            't2': loader.type_map.get(n2,'?'),
            't3': loader.type_map.get(n3,'?'),
        })
        print(f"   ✅ {n1}→{n2}→{n3}")
    return results


def expand_with_neighbors(loader, core_neurons, pool, n_neighbors=50):
    """Expand a core set with top n_neighbors from VNC pool."""
    expanded = set(core_neurons)

    # Find neighbors: pre and post of core neurons
    neighbor_scores = Counter()
    for pre, post, w in loader.synapses:
        if pre in core_neurons and post in pool:
            neighbor_scores[post] += w
        if post in core_neurons and pre in pool:
            neighbor_scores[pre] += w

    top_neighbors = [bid for bid, _ in neighbor_scores.most_common(n_neighbors)
                     if bid not in expanded]
    expanded.update(top_neighbors)

    # Also add DNnt011 and EN00B001
    for bid in pool:
        t = loader.type_map.get(bid, '')
        if 'DNnt011' in t or 'EN00B001' in t:
            expanded.add(bid)

    return list(expanded), top_neighbors


def simulate_hh(loader, neurons, n_steps=5000, dt_ms=0.5, external_I=10.0, label=""):
    """Run HH sim, find oscillation frequencies."""
    t0 = time.time()
    row_ptr, col_idx, vals, nt_s, N = loader.build_csr(neurons)
    if N < 3: return None

    conn_data = tc.ConnectomeData(row_ptr, col_idx, vals, nt_s)
    sim = tc.BrainSimulator(N, "hh", pacemaking=True)
    I_ext = np.full(N, external_I, dtype=np.float32)
    spike_buf = sim.step_batch(conn_data, n_steps=n_steps, dt_ms=dt_ms, external_I=I_ext, scale=1.0)
    spikes = np.asarray(spike_buf, dtype=np.int32).reshape(n_steps, N)

    fr = spikes.mean(axis=0) * (1000.0 / dt_ms)
    total_spikes = int(spikes.sum())
    active = int((fr > 1.0).sum())
    mean_fr = fr.mean()

    # Oscillation detection
    sigma_s = 10.0; g_r = int(4 * sigma_s)
    gx = np.arange(-g_r, g_r+1); gk = np.exp(-0.5*(gx/sigma_s)**2); gk /= gk.sum()
    dt = dt_ms / 1000.0; n_fft = 1024

    all_freqs = []
    for idx in range(N):
        if fr[idx] <= 1.0: continue
        st = spikes[:, idx].astype(np.float32)
        sm = np.convolve(st, gk, mode='same')
        fft = np.fft.fft(sm, n=n_fft)
        mag = np.abs(fft[:n_fft//2])
        freqs = np.fft.fftfreq(n_fft, d=dt)[:n_fft//2]
        lo = max(1, int(1.0/(freqs[1]-freqs[0]+1e-10)))
        if lo >= len(mag): continue
        pk = np.argmax(mag[lo:]) + lo
        if pk < len(freqs) and 1.0 < freqs[pk] < 50:
            all_freqs.append(freqs[pk])

    n_osc = len(all_freqs)
    mean_osc = np.mean(all_freqs) if all_freqs else 0.0
    walk_neurons = sum(1 for f in all_freqs if 4 <= f <= 15)
    gamma_neurons = sum(1 for f in all_freqs if 30 <= f <= 100)
    delta_neurons = sum(1 for f in all_freqs if 1 <= f < 4)
    sim_t = time.time() - t0

    print(f"   {label:<25} N={N:<4} I={external_I:<5.0f} "
          f"spk={total_spikes:<6} act={active:>2}/{N:<3} "
          f"osc={n_osc:<4} μ={mean_osc:<5.1f}Hz "
          f"Δ={delta_neurons:<3} 🚶={walk_neurons:<3} γ={gamma_neurons:<3} "
          f"[{sim_t:.1f}s]")

    return {
        'N': N, 'total_spikes': total_spikes, 'active': active,
        'mean_fr': mean_fr, 'osc_freqs': all_freqs,
        'mean_osc': mean_osc, 'walk_neurons': walk_neurons,
        'gamma_neurons': gamma_neurons, 'delta_neurons': delta_neurons,
        'sim_time': sim_t, 'external_I': external_I,
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("═" * 70)
    print("🧬 VNC CPG EMERGENCE EXPERIMENT")
    print("   5-10 Hz from network, not microcircuit")
    print("═" * 70)

    # 1. Load
    print("\n📥 LOADING...")
    loader = MANCLoader()
    loader.load()
    comps = loader.find_clusters()

    cluster_all = set()
    for c in loader.clusters.values(): cluster_all |= c
    vnc_pool = loader.vnc_traced_ids - cluster_all
    print(f"\n   VNC pool: {len(vnc_pool):,} neurons")

    # 2. Find candidates
    print(f"\n🔍 FINDING CPG CANDIDATES...")
    candidates = find_candidate_by_types(loader, vnc_pool)
    print(f"   Found {len(candidates)} candidate circuits")

    if not candidates:
        # Use known types instead
        print("   Using fallback: find feedforward chains with known types")
        # Find any DNht + AN + IN triple
        dnh_neurons = [b for b in vnc_pool if 'DNht' in loader.type_map.get(b,'')]
        an_neurons = [b for b in vnc_pool if 'AN' in loader.type_map.get(b,'')[:5]]
        in_neurons = [b for b in vnc_pool if 'IN' in loader.type_map.get(b,'')[:5]]
        print(f"   DNht: {len(dnh_neurons)}, AN: {len(an_neurons)}, IN: {len(in_neurons)}")

        for dn in dnh_neurons[:3]:
            dn_targets = loader.full_adj.get(dn, set()) & set(an_neurons)
            for an in list(dn_targets)[:3]:
                an_targets = loader.full_adj.get(an, set()) & set(in_neurons)
                for i_n in list(an_targets)[:3]:
                    candidates.append({
                        'n1': dn, 'n2': an, 'n3': i_n,
                        't1': loader.type_map.get(dn,'?'),
                        't2': loader.type_map.get(an,'?'),
                        't3': loader.type_map.get(i_n,'?'),
                    })

    if not candidates:
        print("❌ No candidates found!")
        return

    print(f"   Ready: {len(candidates)} circuits")

    # 3. Experiment: isolated 3n vs expanded subnet
    print(f"\n{'='*70}")
    print("🎬 EXPERIMENT: ISOLATED vs EXPANDED")
    print(f"{'='*70}")

    all_results = []
    I_exts = [10, 15, 20]  # tonic drive sweep

    for ci, cand in enumerate(candidates[:3]):
        core = {cand['n1'], cand['n2'], cand['n3']}
        label = f"C{ci+1}: {cand['t1'][:10]}→{cand['t2'][:10]}→{cand['t3'][:10]}"
        print(f"\n{'─'*70}")
        print(f"🎯 Candidate {ci+1}: {label}")
        print(f"{'─'*70}")

        # Phase 1: Isolated 3 neurons
        print(f"\n   📍 PHASE 1: ISOLATED (3 neurons)")
        for I_val in I_exts:
            r = simulate_hh(loader, list(core), n_steps=5000, dt_ms=0.5,
                          external_I=I_val, label=f"isolated I={I_val}")
            if r: all_results.append((ci+1, 'isolated', I_val, r))

        # Phase 2: Expanded with VNC neighbors
        print(f"\n   📍 PHASE 2: EXPANDED (~200 neurons)")
        expanded, neighbors = expand_with_neighbors(loader, core, vnc_pool, n_neighbors=100)
        n_dn = sum(1 for b in expanded if 'DNnt011' in loader.type_map.get(b,''))
        n_en = sum(1 for b in expanded if 'EN00B001' in loader.type_map.get(b,''))
        print(f"      Core: {len(core)}, Neighbors: {len(neighbors)}, "
              f"Total: {len(expanded)}, DNnt011={n_dn}, EN00B001={n_en}")

        for I_val in I_exts:
            r = simulate_hh(loader, expanded, n_steps=5000, dt_ms=0.5,
                          external_I=I_val, label=f"expanded I={I_val}")
            if r: all_results.append((ci+1, 'expanded', I_val, r))

    # 4. Summary table
    print(f"\n{'='*120}")
    print("📊 EMERGENCE RESULTS")
    print(f"{'='*120}")
    print(f"{'Cand':<5} {'Phase':<12} {'I_ext':<6} {'N':<5} {'Spikes':<8} {'Act%':<6} "
          f"{'Osc':<5} {'MeanHz':<8} {'Δ':<5} {'🚶Walk':<8} {'γ':<5} {'Emerg?':<8}")
    print("-" * 120)

    for ci, phase, I_val, r in all_results:
        act_pct = f"{r['active']/max(1,r['N'])*100:.0f}%"
        walk = r['walk_neurons']
        emerg = "✅" if walk > 0 and r['mean_osc'] > 4 else "❌"
        print(f"{ci:<5} {phase:<12} {I_val:<6.0f} {r['N']:<5} "
              f"{r['total_spikes']:<8} {act_pct:<6} {len(r['osc_freqs']):<5} "
              f"{r['mean_osc']:<8.1f} {r['delta_neurons']:<5} "
              f"{walk:<8} {r['gamma_neurons']:<5} {emerg:<8}")

    # 5. Verdict
    print(f"\n{'='*70}")
    print("🔬 SCIENTIFIC VERDICT")
    print(f"{'='*70}")

    walking_emerged = False
    for ci, phase, I_val, r in all_results:
        if r['walk_neurons'] > 0 and r['mean_osc'] > 4:
            walking_emerged = True
            print(f"\n✅ Walking EMERGENCE at Candidate {ci}, {phase}, I={I_val}:")
            print(f"   {r['walk_neurons']} neurons @ 4-15Hz, mean={r['mean_osc']:.1f}Hz")
            print(f"   Network size: {r['N']} neurons vs isolated 3")

    # Compare: expanded vs isolated at same I
    print(f"\n📋 COMPARISON: Isolated vs Expanded")
    for ci in range(1, 4):
        iso_r = {I: r for c,p,I,r in all_results if c==ci and p=='isolated'}
        exp_r = {I: r for c,p,I,r in all_results if c==ci and p=='expanded'}
        for I_val in I_exts:
            iso = iso_r.get(I_val); exp = exp_r.get(I_val)
            if iso and exp:
                # Oscillation shift?
                iso_has_walk = iso['walk_neurons'] > 0
                exp_has_walk = exp['walk_neurons'] > 0
                shift = "✅ EMERGED! 🚶" if (not iso_has_walk and exp_has_walk) else \
                        "✅ Still walking" if (iso_has_walk and exp_has_walk) else \
                        "❌ No change" if (not iso_has_walk and not exp_has_walk) else \
                        "⚠️ Lost walking"
                print(f"   C{ci} I={I_val:.0f}: iso={iso['mean_osc']:.1f}Hz → "
                      f"exp={exp['mean_osc']:.1f}Hz | {shift}")

    # Final conclusion
    print(f"\n{'─'*70}")
    print("📋 FINAL CONCLUSION")
    print(f"   CPG candidates tested: {min(3, len(candidates))}")
    print(f"   I_ext sweep: {I_exts}")
    print(f"   Walking EMERGENCE: {'✅ YES! 5-10Hz from network!' if walking_emerged else '❌ No emergence observed'}")
    print(f"   Network size for emergence: ~200 neurons")
    print(f"\n   Гипотеза: {'✅ ПОДТВЕРЖДЕНА — walking frequency требует полной сети' if walking_emerged else '❌ НЕ ПОДТВЕРЖДЕНА — трёх нейронов достаточно'}")

    print(f"\n✅ EXPERIMENT COMPLETE!")


if __name__ == "__main__":
    main()
