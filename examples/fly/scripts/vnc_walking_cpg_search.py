#!/usr/bin/env python3
"""
🎯 VNC WALKING CPG — Targeted Search for Pugliese Circuits
===========================================================
Стратегия: искать walking CPG среди ~17K VNC-only нейронов,
НЕ входящих в 41 GABA кластер.

Методы:
1. Relaxed feedforward motifs: E→E→I, E→I→E, I→E→E (любые 2-edge directed chains)
2. Directed 3-cycles (замыкание цепи)
3. Поиск DNg100, DNnt011, EN00B001 — известные walking типы
4. HH симуляция на топ-кандидатах → проверка 5-10 Hz

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
#  MANC LOADER
# ═══════════════════════════════════════════════════════════════════════

class MANCLoader:
    def __init__(self):
        self.neurons = {}
        self.synapses = []
        self.nt_map = {}; self.type_map = {}; self.class_map = {}
        self.subclass_map = {}; self.modal_map = {}
        self.full_adj = defaultdict(set); self.full_weights = {}
        self.clusters = {}; self.cluster_membership = {}
        self.reciprocal_pairs = []
        self.n_synapses = 0
        self.vnc_traced_ids = set()  # traced VNC neuron bodyIds

    def load(self):
        t0 = time.time()
        print("📥 LOADING MANC...")
        self._load_properties()
        self._load_neurons()
        self._load_connections()
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
        if not os.path.exists(NEURONS_CSV):
            return
        with open(NEURONS_CSV) as f:
            for row in csv.reader(f):
                if row[0] == 'bodyId':
                    continue
                bid = int(row[0]); ntype = row[1].strip()
                self.vnc_traced_ids.add(bid)
                self.neurons.setdefault(bid, {'type':ntype,'nt':'','class':'','subclass':'','modality':''})
                self.neurons[bid]['type'] = ntype
                self.type_map[bid] = ntype

    def _load_connections(self):
        t0 = time.time()
        self.synapses = []; self.full_adj.clear(); self.full_weights.clear()
        with open(CONNS_CSV) as f:
            rdr = csv.reader(f); next(rdr)
            for row in rdr:
                p = int(row[0]); o = int(row[1]); w = float(row[2])
                self.synapses.append((p, o, w))
                self.full_adj[p].add(o)
                self.full_weights[(p, o)] = w
        self.n_synapses = len(self.synapses)
        print(f"      {self.n_synapses:,} syn, full_adj={len(self.full_adj)} presynaptic [{time.time()-t0:.1f}s]")

    def find_clusters(self):
        """Find 41 reciprocal GABA clusters."""
        t0 = time.time()
        print("\n🔬 FINDING RECIPROCAL GABA CLUSTERS...")
        gaba_set = set(bid for bid, nt in self.nt_map.items() if nt == 'gaba')
        gaba_adj = defaultdict(set); gaba_w = {}
        for p, posts in self.full_adj.items():
            if p not in gaba_set:
                continue
            for o in posts:
                if o in gaba_set:
                    gaba_adj[p].add(o)
                    gaba_w[(p,o)] = self.full_weights.get((p,o), 0)

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
        print(f"   {len(comps)} clusters, {sum(len(c) for c in comps):,} neurons [{time.time()-t0:.1f}s]")
        return comps

    def build_csr(self, keep_ids):
        """Build CSR for a subset of bodyIds."""
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
        return row_ptr, col, val, nt_s, N


# ═══════════════════════════════════════════════════════════════════════
#  VNC-ONLY POOL
# ═══════════════════════════════════════════════════════════════════════

def get_vnc_nonreciprocal_pool(loader):
    """VNC-only neurons NOT in 41 GABA clusters.

    VNC neurons = traced neurons (from traced-neurons.csv, ~23K).
    Non-cluster = not in any GABA reciprocal cluster.
    """
    cluster_all = set()
    for comp in loader.clusters.values():
        cluster_all |= comp

    vnc_pool = loader.vnc_traced_ids - cluster_all
    return vnc_pool, cluster_all


# ═══════════════════════════════════════════════════════════════════════
#  RELAXED 3-NEURON MOTIF SEARCH
# ═══════════════════════════════════════════════════════════════════════

def find_feedforward_chains(loader, pool):
    """Find feedforward E→E→I and E→I→E chains in VNC pool.

    Strategy:
    1. Find ALL directed 2-edge chains: A→B→C where A,B,C are in pool
    2. Classify by NT types: E=exc(ACh/GLUT), I=inh(GABA)
    3. Keep chains where NT pattern matches Pugliese (2 exc + 1 inh in feedforward)

    Returns chains sorted by mean weight.
    """
    t0 = time.time()
    print(f"\n   Searching feedforward chains in {len(pool):,} VNC neurons...")

    # Classify neurons in pool
    exc_pool = {b for b in pool if loader.nt_map.get(b,'') in ('acetylcholine','glutamate')}
    inh_pool = {b for b in pool if loader.nt_map.get(b,'') == 'gaba'}
    other_pool = pool - exc_pool - inh_pool

    print(f"   Exc: {len(exc_pool):,}, Inh: {len(inh_pool):,}, Other: {len(other_pool):,}")

    results = []
    done_check = set()
    adj = loader.full_adj
    weights = loader.full_weights

    # --- Pattern 1: E→E→I (feedforward inhibition) ---
    print(f"   Pattern E→E→I: {len(exc_pool)} × {len(exc_pool)} × {len(inh_pool)}...")
    t1 = time.time()
    count = 0
    exc_sample = list(exc_pool)[:500]  # limit for speed
    inh_sample = list(inh_pool)[:500]

    for n1 in exc_sample:
        n1_targets = adj.get(n1, set()) & exc_pool  # n1→E
        for n2 in list(n1_targets)[:100]:  # limit targets per neuron
            n2_targets = adj.get(n2, set()) & inh_pool  # n2→I
            for n3 in n2_targets:
                key = (n1, n2, n3)
                if key in done_check:
                    continue
                done_check.add(key)
                count += 1

                # Check NO reciprocal (Pugliese constraint)
                recip = 0
                for a,b in [(n1,n2),(n2,n3),(n3,n1)]:
                    if b in adj.get(a,set()) and a in adj.get(b,set()):
                        recip += 1
                if recip > 0:
                    continue

                w12 = weights.get((n1,n2),0); w23 = weights.get((n2,n3),0)
                w31 = weights.get((n3,n1),0)  # feedback I→E?
                has_feedback = n1 in adj.get(n3, set())  # I→E1 closes loop?

                results.append({
                    'n1': n1, 'n2': n2, 'n3': n3,
                    'pattern': 'E→E→I',
                    'nts': f"{loader.nt_map.get(n1,'?')[:4]}→{loader.nt_map.get(n2,'?')[:4]}→{loader.nt_map.get(n3,'?')[:4]}",
                    'types': f"{loader.type_map.get(n1,'?')[:12]}→{loader.type_map.get(n2,'?')[:12]}→{loader.type_map.get(n3,'?')[:12]}",
                    'classes': f"{loader.class_map.get(n1,'?')[:10]}→{loader.class_map.get(n2,'?')[:10]}→{loader.class_map.get(n3,'?')[:10]}",
                    'weights': (w12, w23, w31),
                    'mean_w': (w12 + w23 + w31) / 3,
                    'has_feedback': has_feedback,
                    'is_cycle': has_feedback and n2 in adj.get(n3, set()),  # full directed 3-cycle
                    'recip': recip,
                    'c1': loader.class_map.get(n1,'?'), 'c2': loader.class_map.get(n2,'?'),
                    'c3': loader.class_map.get(n3,'?'),
                    't1': loader.type_map.get(n1,'?'), 't2': loader.type_map.get(n2,'?'),
                    't3': loader.type_map.get(n3,'?'),
                    'sub1': loader.subclass_map.get(n1,''), 'sub2': loader.subclass_map.get(n2,''),
                    'sub3': loader.subclass_map.get(n3,''),
                    'modal1': loader.modal_map.get(n1,''), 'modal2': loader.modal_map.get(n2,''),
                    'modal3': loader.modal_map.get(n3,''),
                })

    print(f"      Found {len(results)} E→E→I chains [{time.time()-t1:.1f}s]")

    # --- Pattern 2: E→I→E (feedforward disinhibition) ---
    t1 = time.time()
    count2 = 0
    for n1 in exc_sample:
        n1_inh = adj.get(n1, set()) & inh_pool  # n1→I (but n1 is E)
        for n2 in list(n1_inh)[:50]:
            n2_exc = adj.get(n2, set()) & exc_pool
            for n3 in n2_exc:
                key = (n1, n2, n3)
                if key in done_check:
                    continue
                done_check.add(key)
                count2 += 1

                recip = 0
                for a,b in [(n1,n2),(n2,n3),(n3,n1)]:
                    if b in adj.get(a,set()) and a in adj.get(b,set()):
                        recip += 1
                if recip > 0:
                    continue

                w12 = weights.get((n1,n2),0); w23 = weights.get((n2,n3),0)
                w31 = weights.get((n3,n1),0)
                has_feedback = n1 in adj.get(n3, set())

                results.append({
                    'n1': n1, 'n2': n2, 'n3': n3,
                    'pattern': 'E→I→E',
                    'nts': f"{loader.nt_map.get(n1,'?')[:4]}→{loader.nt_map.get(n2,'?')[:4]}→{loader.nt_map.get(n3,'?')[:4]}",
                    'types': f"{loader.type_map.get(n1,'?')[:12]}→{loader.type_map.get(n2,'?')[:12]}→{loader.type_map.get(n3,'?')[:12]}",
                    'classes': f"{loader.class_map.get(n1,'?')[:10]}→{loader.class_map.get(n2,'?')[:10]}→{loader.class_map.get(n3,'?')[:10]}",
                    'weights': (w12, w23, w31),
                    'mean_w': (w12 + w23 + w31) / 3,
                    'has_feedback': has_feedback,
                    'is_cycle': has_feedback and n2 in adj.get(n3, set()),
                    'recip': recip,
                    'c1': loader.class_map.get(n1,'?'), 'c2': loader.class_map.get(n2,'?'),
                    'c3': loader.class_map.get(n3,'?'),
                    't1': loader.type_map.get(n1,'?'), 't2': loader.type_map.get(n2,'?'),
                    't3': loader.type_map.get(n3,'?'),
                    'sub1': loader.subclass_map.get(n1,''), 'sub2': loader.subclass_map.get(n2,''),
                    'sub3': loader.subclass_map.get(n3,''),
                    'modal1': loader.modal_map.get(n1,''), 'modal2': loader.modal_map.get(n2,''),
                    'modal3': loader.modal_map.get(n3,''),
                })

    print(f"      Found {count2} E→I→E chains [{time.time()-t1:.1f}s]")
    print(f"   Total chains: {len(results)} [{time.time()-t0:.1f}s]")

    results.sort(key=lambda x: -x['mean_w'])
    return results


# ═══════════════════════════════════════════════════════════════════════
#  KNOWN WALKING NEURONS SEARCH
# ═══════════════════════════════════════════════════════════════════════

def find_known_walking_neurons(loader, pool):
    """Find known walking-related neuron types in VNC pool.

    Targets: DNg100 (Pugliese), DNnt011 (our discovery),
             EN00B001 (efferent), AN12B001 (ascending)
    """
    print(f"\n🔬 KNOWN WALKING NEURONS IN VNC POOL:")
    targets = {
        'DNg100': [],
        'DNnt011': [],
        'EN00B001': [],
        'AN12B001': [],
        'AN27X001': [],
        'AN05B001': [],
        'IN13A003': [],
    }

    for bid in pool:
        ntype = loader.type_map.get(bid, '')
        for key in targets:
            if key in ntype:
                targets[key].append(bid)

    for name, neurons in targets.items():
        print(f"   {name}: {len(neurons)} neurons")
        if neurons:
            for bid in neurons[:3]:
                cls = loader.class_map.get(bid, '?')
                nt = loader.nt_map.get(bid, '?')
                sub = loader.subclass_map.get(bid, '')
                info = f"      bodyId={bid} class={cls} nt={nt}"
                if sub:
                    info += f" sub={sub}"
                print(info)

    return targets


# ═══════════════════════════════════════════════════════════════════════
#  HH SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def simulate_motif_hh(loader, motif, n_steps=5000, dt_ms=0.5):
    """Run HH on 3 neurons of a motif, find oscillation frequency."""
    neurons = {motif['n1'], motif['n2'], motif['n3']}
    if len(neurons) < 3:
        return None

    t0 = time.time()
    row_ptr, col_idx, vals, nt_s, N = loader.build_csr(list(neurons))
    if N < 3:
        return None

    conn_data = tc.ConnectomeData(row_ptr, col_idx, vals, nt_s)
    sim = tc.BrainSimulator(N, "hh", pacemaking=True)
    I_ext = np.full(N, 10.0, dtype=np.float32)

    spike_buf = sim.step_batch(conn_data, n_steps=n_steps, dt_ms=dt_ms,
                               external_I=I_ext, scale=1.0)
    spikes = np.asarray(spike_buf, dtype=np.int32).reshape(n_steps, N)
    total_spikes = int(spikes.sum())
    fr = spikes.mean(axis=0) * (1000.0 / dt_ms)
    active = int((fr > 1.0).sum())

    # FFT for oscillation detection
    sigma_s = 10.0; g_r = int(4 * sigma_s)
    gx = np.arange(-g_r, g_r+1); gk = np.exp(-0.5*(gx/sigma_s)**2); gk /= gk.sum()
    dt = dt_ms / 1000.0; n_fft = 1024

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
    sim_t = time.time() - t0

    return {
        'total_spikes': total_spikes, 'active': active,
        'osc_freqs': osc_freqs, 'is_walking': is_walking,
        'sim_time': sim_t, 'neurons': neurons, 'N': N,
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("═" * 70)
    print("🎯 VNC WALKING CPG — Targeted Pugliese Search")
    print("═" * 70)

    # 1. Load
    print("\n📥 STEP 1: LOAD MANC")
    loader = MANCLoader()
    loader.load()
    print(f"   Traced VNC neurons: {len(loader.vnc_traced_ids):,}")

    # 2. Find GABA clusters
    print("\n🔬 STEP 2: FIND GABA CLUSTERS")
    comps = loader.find_clusters()

    # 3. VNC non-reciprocal pool
    print(f"\n🎯 STEP 3: VNC-ONLY NON-RECIPROCAL POOL")
    vnc_pool, cluster_set = get_vnc_nonreciprocal_pool(loader)
    print(f"   VNC traced:       {len(loader.vnc_traced_ids):,}")
    print(f"   In GABA clusters: {len(cluster_set):,}")
    print(f"   VNC non-recip:    {len(vnc_pool):,} ← Ищем здесь!")

    # 4. Known walking neurons
    print(f"\n🔬 STEP 4: KNOWN WALKING NEURONS")
    walking_neurons = find_known_walking_neurons(loader, vnc_pool)

    # 5. Feedforward chains
    print(f"\n🔗 STEP 5: FEEDFORWARD CHAINS (E→E→I + E→I→E)")
    chains = find_feedforward_chains(loader, vnc_pool)

    # 6. Results table
    print(f"\n{'='*100}")
    print(f"🏆 MOTIFS FOUND: {len(chains)}")
    print(f"{'='*100}")

    if chains:
        print(f"\n{'Rank':<5} {'Pattern':<10} {'Chain':<60} {'Weights':<15} {'Feedback':<10} {'Cycle':<8}")
        print("-" * 110)
        for i, m in enumerate(chains[:40]):
            fb = "✅" if m['has_feedback'] else "❌"
            cyc = "✅" if m['is_cycle'] else ""
            print(f"#{i+1:<3} {m['pattern']:<10} {m['types'][:58]:<60} "
                  f"{m['mean_w']:<7.1f}/{sum(m['weights']):<7.1f} {fb:<10} {cyc:<8}")

        # Show walking-relevant: has feedback
        with_feedback = [c for c in chains if c['has_feedback']]
        cycles = [c for c in chains if c['is_cycle']]
        print(f"\n   With feedback (I→E1): {len(with_feedback)}")
        if with_feedback and len(with_feedback) > 0:
            for m in with_feedback[:10]:
                print(f"      {m['pattern']} {m['types'][:50]} w={m['mean_w']:.1f}")
        print(f"   Full directed 3-cycles: {len(cycles)}")
        if cycles:
            for m in cycles[:5]:
                print(f"      {m['pattern']} {m['types'][:50]} w={m['mean_w']:.1f}")
    else:
        print("   ❌ No chains found")

    # 7. HH Simulation
    if chains:
        print(f"\n{'='*70}")
        print("🎬 STEP 6: HH SIMULATION ON TOP 20 MOTIFS")
        print(f"{'='*70}")

        top_motifs = chains[:20]  # top 20 by weight
        walking_found = []
        for i, m in enumerate(top_motifs):
            print(f"\n   ─── Motif #{i+1}: {m['pattern']} {m['types'][:40]} ───")
            result = simulate_motif_hh(loader, m, n_steps=3000, dt_ms=0.5)
            if result:
                status = "✅ WALKING!" if result['is_walking'] else f"📊 osc={result['osc_freqs']}"
                print(f"      {status} spk={result['total_spikes']} act={result['active']}/{result['N']} t={result['sim_time']:.1f}s")
                if result['is_walking']:
                    walking_found.append((m, result))
            else:
                print(f"      ❌ Sim failed")

        # 8. Walking CPG found?
        if walking_found:
            print(f"\n{'🏆'*35}")
            print(f"🏆 WALKING CPG FOUND! {len(walking_found)} candidates!")
            print(f"{'🏆'*35}")
            for m, r in walking_found:
                print(f"   {m['pattern']} {m['types'][:50]}")
                print(f"      Freq: {r['osc_freqs']} Hz  Weights: {m['weights']}")
                print(f"      Classes: {m['classes']}")

    # 9. Known walking neuron circuits
    if any(len(v) > 0 for v in walking_neurons.values()):
        print(f"\n{'='*70}")
        print("🎬 STEP 7: KNOWN NEURON CIRCUITS")
        print(f"{'='*70}")
        for name, neurons in walking_neurons.items():
            if len(neurons) == 0:
                continue
            print(f"\n   {name} ({len(neurons)}):")
            # For each known neuron, find its top targets
            for bid in neurons[:3]:
                targets = [(o, w) for p, o, w in loader.synapses if p == bid
                          and o in vnc_pool]
                targets.sort(key=lambda x: -x[1])
                print(f"      bodyId={bid} top targets:")
                for t_bid, t_w in targets[:5]:
                    t_type = loader.type_map.get(t_bid, '?')
                    t_nt = loader.nt_map.get(t_bid, '?')
                    print(f"         → {t_bid} type={t_type} nt={t_nt} w={t_w}")

    # Final verdict
    print(f"\n{'='*70}")
    print("🔬 SCIENTIFIC VERDICT")
    print(f"{'='*70}")
    print(f"\n📊 RESULTS:")
    print(f"   VNC non-reciprocal pool: {len(vnc_pool):,} neurons")
    print(f"   Excitatory: {len({b for b in vnc_pool if loader.nt_map.get(b,'') in ('acetylcholine','glutamate')}):,}")
    print(f"   Inhibitory: {len({b for b in vnc_pool if loader.nt_map.get(b,'') == 'gaba'}):,}")
    print(f"   Feedforward chains: {len(chains)}")
    print(f"   With feedback loop: {len([c for c in chains if c.get('has_feedback')])}")
    print(f"   Full directed 3-cycles: {len([c for c in chains if c.get('is_cycle')])}")
    print(f"   Walking CPG (5-10Hz): {'✅' if any(r.get('is_walking') for _, r in locals().get('walking_found',[])) else '❌'}")

    print(f"\n✅ SEARCH COMPLETE!")


if __name__ == "__main__":
    main()
