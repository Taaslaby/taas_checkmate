#!/usr/bin/env python3
"""
🧬🦗 VNC ALL CLUSTERS SCAN — Full CPG Discovery Pipeline
========================================================
Просканировать ВСЕ 41 реципрокный GABA кластер VNC на:

1. NT composition (GABA vs ACh vs GLUT)
2. 3-neuron microcircuits (2 exc + 1 inh, Pugliese-style)
3. HH симуляция на перспективных кластерах
4. Поиск walking CPG (5-10 Hz oscillations)

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
#  MANC LOADER + PRE-BUILT ADJACENCY
# ═══════════════════════════════════════════════════════════════════════

class MANCLoader:
    def __init__(self):
        self.neurons = {}
        self.synapses = []
        self.nt_map = {}; self.type_map = {}; self.class_map = {}
        self.subclass_map = {}; self.modal_map = {}
        self.bodyid_to_idx = {}; self.idx_to_bodyid = {}
        self.clusters = {}; self.cluster_membership = {}
        self.reciprocal_pairs = []

        # ⭐ Pre-built for performance
        self.full_adj = defaultdict(set)    # pre→[posts] for ALL neurons
        self.full_weights = {}              # (pre,post)→weight

    def load(self):
        t0 = time.time()
        print("📥 LOADING MANC...")
        self._load_properties()
        self._load_neurons()
        self._load_connections()
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

    def _load_connections(self):
        t0 = time.time()
        self.synapses = []
        self.full_adj.clear()
        self.full_weights.clear()
        with open(CONNS_CSV) as f:
            rdr = csv.reader(f); next(rdr)
            for row in rdr:
                p = int(row[0]); o = int(row[1]); w = float(row[2])
                self.synapses.append((p, o, w))
                self.full_adj[p].add(o)
                self.full_weights[(p, o)] = w
        self.n_synapses = len(self.synapses)
        print(f"      {self.n_synapses:,} synapses, full_adj built [{time.time()-t0:.1f}s]")

    def find_clusters(self):
        """Find GABA reciprocal pairs → connected components."""
        t0 = time.time()
        print("\n🔬 FINDING ALL RECIPROCAL GABA CLUSTERS...")
        gaba_set = set(bid for bid, nt in self.nt_map.items() if nt == 'gaba')
        print(f"   GABA neurons: {len(gaba_set):,}")

        # Use pre-built full_adj for GABA-only edges
        gaba_adj = defaultdict(set); gaba_w = {}
        for p, posts in self.full_adj.items():
            if p not in gaba_set:
                continue
            for o in posts:
                if o in gaba_set:
                    gaba_adj[p].add(o)
                    gaba_w[(p,o)] = self.full_weights.get((p,o), 0)
        n_gaba_edges = sum(len(v) for v in gaba_adj.values())
        print(f"   GABA→GABA edges: {n_gaba_edges:,} [{time.time()-t0:.1f}s]")

        # Reciprocal pairs
        t1 = time.time()
        recip = []
        for pre, posts in gaba_adj.items():
            for post in posts:
                if pre >= post:
                    continue
                if pre in gaba_adj.get(post, set()):
                    recip.append((pre, gaba_w.get((pre,post),0), post, gaba_w.get((post,pre),0)))
        self.reciprocal_pairs = recip
        print(f"   Reciprocal pairs: {len(recip):,} [{time.time()-t1:.1f}s]")

        # Strong reciprocal → connected components
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

        print(f"   Clusters: {len(comps)}")
        print(f"   Sizes: {[len(c) for c in comps[:10]]}")
        return comps

    def cluster_nt_stats(self, cluster_id):
        """NT distribution for a cluster."""
        comp = self.clusters.get(cluster_id, set())
        cnt = Counter()
        for bid in comp:
            cnt[self.nt_map.get(bid, '?')] += 1
        return cnt, len(comp)

    def build_csr(self, keep_ids):
        """Build CSR for a subset of neuron bodyIds."""
        t0 = time.time()
        keep = set(keep_ids)
        syn = [(p,o,w) for p,o,w in self.synapses if p in keep and o in keep]

        all_ids = sorted(set(p for p,o,w in syn) | set(o for p,o,w in syn))
        b2i = {bid:i for i,bid in enumerate(all_ids)}
        N = len(all_ids); S = len(syn)

        pre_cnt = Counter(p for p,o,w in syn)
        row_ptr = np.zeros(N+1, dtype=np.int32)
        for bid, cnt in pre_cnt.items():
            row_ptr[b2i[bid]+1] = cnt
        row_ptr = np.cumsum(row_ptr, dtype=np.int32)

        col = np.zeros(S, dtype=np.int32)
        val = np.ones(S, dtype=np.float32)
        nt_s = np.ones(N, dtype=np.float32)
        NT_SIGN = {'gaba':-1.,'acetylcholine':1.,'glutamate':1.,
                   'dopamine':-.5,'serotonin':-.5,'octopamine':.5}

        cursor = np.zeros(N, dtype=np.int32)
        for p,o,w in syn:
            pi = b2i[p]; oi = b2i[o]
            pos = row_ptr[pi] + cursor[pi]
            col[pos] = np.int32(oi); val[pos] = float(w)
            cursor[pi] += 1

        for bid, idx in b2i.items():
            nt_s[idx] = NT_SIGN.get(self.nt_map.get(bid,''), 1.0)

        print(f"      CSR: {S} edges, {N} nodes [{time.time()-t0:.1f}s]")
        return row_ptr, col, val, nt_s, b2i


# ═══════════════════════════════════════════════════════════════════════
#  CLUSTER SCAN FUNCTIONS (use pre-built full_adj)
# ═══════════════════════════════════════════════════════════════════════

def scan_3neuron_microcircuits(loader, cluster_id, max_exc=200, max_inh=200):
    """Scan a cluster for Pugliese-style 3-neuron microcircuits (2 exc + 1 inh).

    Uses pre-built full_adj + full_weights from loader for performance.
    """
    comp = loader.clusters.get(cluster_id, set())
    if not comp:
        return []

    exc = [b for b in comp if loader.nt_map.get(b,'') in ('acetylcholine','glutamate')]
    inh = [b for b in comp if loader.nt_map.get(b,'') == 'gaba']

    if len(exc) < 2 or len(inh) < 1:
        return []

    # Use pre-built full_adj, filter to cluster
    exc_s = exc[:max_exc]; inh_s = inh[:max_inh]
    results = []; checked = set()
    adj = loader.full_adj; weights = loader.full_weights

    for i, e1 in enumerate(exc_s):
        for e2 in exc_s[i+1:]:
            for inh_n in inh_s:
                key = tuple(sorted([e1, e2, inh_n]))
                if key in checked:
                    continue
                checked.add(key)

                # Check all 6 directed 3-cycle topologies
                for n1, n2, n3, n4 in [(e1,e2,inh_n,e1),(e1,inh_n,e2,e1),
                                        (e2,e1,inh_n,e2),(e2,inh_n,e1,e2),
                                        (inh_n,e1,e2,inh_n),(inh_n,e2,e1,inh_n)]:
                    if n2 not in adj[n1] or n3 not in adj[n2] or n4 not in adj[n3]:
                        continue
                    # NO reciprocal pairs in core CPG (Pugliese)
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
                        'cluster': cluster_id,
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
    results.sort(key=lambda x: -x['mean_w'])
    return results


def simulate_cluster_hh(loader, cluster_id, n_steps=2000, dt_ms=0.5, external_I_val=10.0):
    """Run HH simulation on a cluster, get oscillation stats."""
    comp = loader.clusters.get(cluster_id)
    if not comp or len(comp) < 3:
        return None

    print(f"\n   🎬 HH SIM Cluster {cluster_id} ({len(comp)} neurons)...")
    t0 = time.time()

    row_ptr, col_idx, vals, nt_s, b2i = loader.build_csr(list(comp))
    N = len(b2i)
    conn_data = tc.ConnectomeData(row_ptr, col_idx, vals, nt_s)

    sim = tc.BrainSimulator(N, "hh", pacemaking=True)
    rng = np.random.RandomState(cluster_id)
    I_ext = np.full(N, external_I_val, dtype=np.float32) + rng.randn(N).astype(np.float32) * 2.0

    spike_buf = sim.step_batch(conn_data, n_steps=n_steps, dt_ms=dt_ms,
                               external_I=I_ext, scale=1.0)
    sim_t = time.time() - t0
    spikes = np.asarray(spike_buf, dtype=np.int32).reshape(n_steps, N)

    fr = spikes.mean(axis=0) * (1000.0 / dt_ms)
    active = int((fr > 1.0).sum())
    total_spikes = int(spikes.sum())

    # FFT with Gaussian smoothing
    sigma_s = 10.0  # 10ms kernel
    g_r = int(4 * sigma_s)
    gx = np.arange(-g_r, g_r+1)
    gk = np.exp(-0.5*(gx/sigma_s)**2)
    gk /= gk.sum()

    dt = dt_ms / 1000.0
    n_fft = 1024

    osc = []
    for idx in range(N):
        if fr[idx] <= 5.0:
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
        if pk < len(freqs) and 0.5 < freqs[pk] < 100:
            osc.append((freqs[pk], mag[pk]))

    n_osc = len(osc)
    mean_f = float(np.mean([o[0] for o in osc])) if osc else 0.0
    walk_band = sum(1 for o in osc if 4 <= o[0] <= 15)
    gamma_band = sum(1 for o in osc if 30 <= o[0] <= 100)

    print(f"      {total_spikes:,} spk, {active}/{N} act, {n_osc} osc, "
          f"μ={mean_f:.1f}Hz walk={walk_band} γ={gamma_band} [{sim_t:.1f}s]")

    return {
        'cluster_id': cluster_id, 'n': N, 'steps': n_steps,
        'total_spikes': total_spikes, 'active': active,
        'n_osc': n_osc, 'mean_freq': mean_f,
        'walk_band': walk_band, 'gamma_band': gamma_band,
        'sim_time': sim_t,
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("═" * 70)
    print("🧬🦗 VNC ALL CLUSTERS SCAN — CPG Discovery")
    print("═" * 70)

    # 1. Load MANC
    print("\n📥 STEP 1: LOAD MANC")
    loader = MANCLoader()
    loader.load()

    # 2. Find ALL clusters
    print("\n🔬 STEP 2: FIND ALL CLUSTERS")
    comps = loader.find_clusters()
    print(f"\n📊 TOTAL: {len(comps)} clusters, "
          f"neurons={sum(len(c) for c in comps):,}, "
          f"recip_pairs={len(loader.reciprocal_pairs):,}")

    # 3. Scan ALL clusters for NT composition + 3-neuron circuits
    print(f"\n🎯 STEP 3: SCAN ALL {len(comps)} CLUSTERS")
    print("─" * 70)

    all_stats = []
    for cid in sorted(loader.clusters.keys()):
        nt_cnt, total = loader.cluster_nt_stats(cid)
        n_exc = nt_cnt.get('acetylcholine', 0) + nt_cnt.get('glutamate', 0)
        n_inh = nt_cnt.get('gaba', 0)
        n_other = total - n_exc - n_inh

        mcs = scan_3neuron_microcircuits(loader, cid, max_exc=200, max_inh=200)

        if n_exc == 0 and n_inh == total:
            ctype = "PURE GABA"
        elif n_exc > 0 and n_inh > 0:
            ctype = "MIXED"
        else:
            ctype = "OTHER"

        if len(mcs) > 0:
            ctype += " ⭐CPG!"

        all_stats.append({
            'cid': cid, 'total': total, 'exc': n_exc, 'inh': n_inh,
            'other': n_other, 'type': ctype,
            'nt_cnt': dict(nt_cnt), 'n_mc': len(mcs), 'mcs': mcs[:3],
        })

    # 4. Print table
    print(f"\n{'─'*100}")
    print(f"{'CID':<4} {'Type':<30} {'Neurons':<9} {'Exc':<6} {'Inh':<6} {'Other':<6} {'3-NC':<6} {'Composition':<50}")
    print(f"{'─'*100}")

    pugliese_clusters = []
    for s in all_stats:
        nt_str = " ".join(f"{k[:4]}={v}" for k,v in sorted(s['nt_cnt'].items(), key=lambda x:-x[1])[:5])
        mc_str = f"⭐{s['n_mc']}" if s['n_mc'] > 0 else f"{s['n_mc']}"
        bar = ""
        if s['total'] > 0:
            exc_pct = s['exc'] / s['total'] * 100
            inh_pct = s['inh'] / s['total'] * 100
            bars = 20
            e = int(exc_pct / 100 * bars)
            i = int(inh_pct / 100 * bars)
            bar = "🟦"*e + "🟥"*i + "⬜"*max(0, bars-e-i)
        print(f"{s['cid']:<4} {s['type']:<30} {s['total']:<9,} {s['exc']:<6} {s['inh']:<6} {s['other']:<6} {mc_str:<6} {bar} {nt_str[:40]}")
        if s['n_mc'] > 0:
            pugliese_clusters.append(s['cid'])

    # 5. Pugliese cluster details
    if pugliese_clusters:
        print(f"\n{'⭐'*40}")
        print(f"⭐ PUGLIESE-TYPE CLUSTERS: {pugliese_clusters}")
        print(f"{'⭐'*40}")
        for cid in pugliese_clusters:
            s = next(x for x in all_stats if x['cid'] == cid)
            print(f"\n   Cluster {cid} ({s['total']}n, {s['exc']}exc, {s['inh']}inh): {s['n_mc']} microcircuits")
            for mc in s['mcs'][:5]:
                print(f"      {mc['t1'][:18]:<20} → {mc['t2'][:18]:<20} → {mc['t3'][:18]:<20}  w={mc['mean_w']:.1f}")
            # Top neuron types
            top_t = Counter()
            for bid in loader.clusters[cid]:
                top_t[loader.type_map.get(bid,'?')[:20]] += 1
            print(f"      Types: {dict(top_t.most_common(5))}")

    # 6. HH Simulation
    print(f"\n{'='*70}")
    print("🎬 STEP 4: HH SIMULATION ON PROMISING CLUSTERS")
    print(f"{'='*70}")

    sim_candidates = sorted(pugliese_clusters)  # clusters with 3-neuron circuits
    for extra in [5, 7, 9, 10]:
        if extra not in sim_candidates:
            s = next((x for x in all_stats if x['cid'] == extra), None)
            if s and s['total'] > 1:
                sim_candidates.append(extra)

    print(f"   Candidates: {sim_candidates} ({len(sim_candidates)} clusters)")
    sim_results = []
    for cid in sim_candidates:
        s = next(x for x in all_stats if x['cid'] == cid)
        n_steps = 2000 if s['total'] < 100 else 1000
        result = simulate_cluster_hh(loader, cid, n_steps=n_steps, dt_ms=0.5, external_I_val=10.0)
        if result:
            sim_results.append(result)

    # 7. Results table
    print(f"\n{'='*100}")
    print("📊 HH SIMULATION RESULTS")
    print(f"{'='*100}")
    if sim_results:
        print(f"{'CID':<5} {'Neurons':<9} {'Spikes':<10} {'Act%':<7} {'Osc':<6} {'MeanHz':<9} {'Walk':<6} {'Gamma':<6} {'Type':<25}")
        print(f"{'─'*90}")
        for r in sim_results:
            s = next(x for x in all_stats if x['cid'] == r['cluster_id'])
            act_pct = f"{r['active']/max(1,r['n'])*100:.0f}%"
            print(f"{r['cluster_id']:<5} {r['n']:<9,} {r['total_spikes']:<10,} {act_pct:<7} "
                  f"{r['n_osc']:<6} {r['mean_freq']:<8.1f}Hz {r['walk_band']:<6} {r['gamma_band']:<6} {s['type']:<25}")
    else:
        print("   ❌ No results")

    # 8. Verdict
    print(f"\n{'='*70}")
    print("🔬 SCIENTIFIC VERDICT")
    print(f"{'='*70}")

    walking_cands = [r for r in sim_results if r['walk_band'] > r['gamma_band'] // 2]
    if walking_cands:
        print(f"\n🏆 WALKING CPG CANDIDATES (walking ≥ gamma/2):")
        for r in walking_cands:
            s = next(x for x in all_stats if x['cid'] == r['cluster_id'])
            print(f"   Cluster {r['cluster_id']}: {s['type']} — walk={r['walk_band']}, "
                  f"γ={r['gamma_band']}, μ={r['mean_freq']:.1f}Hz")

    gamma_cands = [r for r in sim_results if r['gamma_band'] > r['walk_band'] and r['n_osc'] > 0]
    if gamma_cands:
        print(f"\n🧠 GAMMA CLUSTERS (gamma > walking):")
        for r in sorted(gamma_cands, key=lambda x: -x['n_osc'])[:5]:
            s = next(x for x in all_stats if x['cid'] == r['cluster_id'])
            print(f"   Cluster {r['cluster_id']}: {s['type']} — γ={r['gamma_band']}, "
                  f"walk={r['walk_band']}, μ={r['mean_freq']:.1f}Hz")

    # Summary
    print(f"\n{'─'*70}")
    print("📋 SUMMARY")
    print(f"   Total clusters: {len(all_stats)}")
    print(f"   Pure GABA:     {sum(1 for s in all_stats if 'PURE GABA' in s['type'])}")
    print(f"   Mixed (exc+inh): {sum(1 for s in all_stats if s['type'].startswith('MIXED'))}")
    print(f"   Pugliese-type: {len(pugliese_clusters)} ({pugliese_clusters})")
    print(f"   Walking CPG:   {'✅' if walking_cands else '❌'}")
    print(f"   Gamma rings:   {len(gamma_cands)}")

    # Show the FINAL map
    print(f"\n{'🗺️'*35}")
    print("🗺️  VNC CLUSTER MAP")
    print(f"{'🗺️'*35}")
    # Classify each by size
    for s in all_stats:
        emoji = "🧬" if 'PURE GABA' in s['type'] else ("⭐" if 'CPG' in s['type'] else "🔬")
        sz = "🟢" if s['total'] < 10 else ("🟡" if s['total'] < 100 else "🟠" if s['total'] < 1000 else "🔴")
        print(f"   {emoji} {sz} Cluster {s['cid']:<3}: {s['type']:<35} {s['total']:>5,}n  "
              f"{'⭐CPG' if 'CPG' in s['type'] else ''}")

    print(f"\n✅ SCAN COMPLETE!")


if __name__ == "__main__":
    main()
