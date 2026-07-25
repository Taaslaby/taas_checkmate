#!/usr/bin/env python3
"""🧫 VNC CPG EMERGENCE + ASTROCYTE LAYER

Сравнение: isolated 3n → expanded ~200n → expanded ~200n + астроциты.

Гипотеза: астроцитарная gap-junction сеть усиливает emergence walking
frequency за счёт volume transmission модуляции I_ext.

Использует:
  - vnc_cpg_emergence_exp.py — MANCLoader, simulate_hh, expand_with_neighbors
  - vnc_astrocyte_layer.py — AstrocyteNetwork, GapJunctionConnectome
"""

import os, sys, time, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

# ── Paths ──
PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # for vnc_astrocyte_layer

from vnc_astrocyte_layer import (
    AstrocyteNetwork, GapJunctionConnectome,
    build_vnc_gap_junctions, build_vnc_top5_gap_junctions,
)

DLOAD = os.environ.get("FLYWIRE_DATA", "/home/taaslab/Загрузки")
NEURONS_CSV = os.path.join(DLOAD, "v1.0_manc-traced-adjacencies-v1.0_traced-neurons.csv")
CONNS_CSV  = os.path.join(DLOAD, "v1.0_manc-traced-adjacencies-v1.0_traced-connections.csv")
PROPS_FTR = os.path.join(DLOAD, "v1.0_manc-v1.0-neuron-properties.feather")

import taas_checkmate as tc

# Re-use MANCLoader from the original experiment
from vnc_cpg_emergence_exp import MANCLoader, find_candidate_by_types, expand_with_neighbors


def simulate_hh_baseline(loader, neurons, n_steps=5000, dt_ms=0.5, external_I=10.0, label=""):
    """Standard HH simulation (baseline, no astrocyte).
    
    Returns same dict as original simulate_hh.
    """
    t0 = time.time()
    row_ptr, col_idx, vals, nt_s, N = loader.build_csr(neurons)
    if N < 3:
        return None

    conn_data = tc.ConnectomeData(row_ptr, col_idx, vals, nt_s)
    sim = tc.BrainSimulator(N, "hh", pacemaking=True)
    I_ext = np.full(N, external_I, dtype=np.float32)
    spike_buf = sim.step_batch(conn_data, n_steps=n_steps, dt_ms=dt_ms, external_I=I_ext, scale=1.0)
    spikes = np.asarray(spike_buf, dtype=np.int32).reshape(n_steps, N)

    return _analyze_spikes(spikes, n_steps, dt_ms, N, external_I, label, time.time() - t0)


def simulate_hh_astrocyte(loader, neurons, n_steps=5000, dt_ms=0.5, external_I=10.0,
                          g_gap=0.1, mu=0.5, tau_astro_ms=2000.0, chunk_size=100,
                          gj_pairs=None, label=""):
    """HH simulation WITH astrocyte gap-junction layer.
    
    Wraps BrainSimulator in AstrocyteNetwork for volume transmission modulation.
    """
    t0 = time.time()
    row_ptr, col_idx, vals, nt_s, N = loader.build_csr(neurons)
    if N < 3:
        return None

    conn_data = tc.ConnectomeData(row_ptr, col_idx, vals, nt_s)
    
    # Create gap junction connectome
    gj = GapJunctionConnectome()
    if gj_pairs:
        for i, j, g in gj_pairs:
            gj.add(i, j, g)
    else:
        # Default: random gap junctions with 5% probability
        gj.add_random(list(range(N)), p=0.05, g_gap=g_gap)
    
    # Wrap simulator
    sim = tc.BrainSimulator(N, "hh", pacemaking=True)
    astro = AstrocyteNetwork(
        simulator=sim, n_astro=N, gap_conn=gj,
        tau_astro_ms=tau_astro_ms, g_gap=g_gap, mu=mu,
    )
    
    # Run chunked simulation with astrocyte modulation
    spike_buf_flat, I_ext_trace = astro.simulate(
        conn_data, n_steps=n_steps, dt_ms=dt_ms,
        external_I=external_I, chunk_size=chunk_size, scale=1.0,
        record_astro=True,
    )
    spikes = np.asarray(spike_buf_flat, dtype=np.int32).reshape(n_steps, N)

    result = _analyze_spikes(spikes, n_steps, dt_ms, N, external_I, label, time.time() - t0)
    result['astro_I_ext_trace'] = I_ext_trace
    result['astro_ca_final'] = astro.ca.copy()
    result['astro_gain_final'] = astro.volume_gain.copy()
    return result


def _analyze_spikes(spikes, n_steps, dt_ms, N, external_I, label, sim_t):
    """Analyze spike data: firing rates, oscillation frequencies."""
    fr = spikes.mean(axis=0) * (1000.0 / dt_ms)
    total_spikes = int(spikes.sum())
    active = int((fr > 1.0).sum())
    mean_fr = fr.mean() if N > 0 else 0.0

    # Oscillation detection
    sigma_s = 10.0; g_r = int(4 * sigma_s)
    gx = np.arange(-g_r, g_r + 1)
    gk = np.exp(-0.5 * (gx / sigma_s) ** 2)
    gk /= gk.sum()
    n_fft = 1024
    dt = dt_ms / 1000.0

    all_freqs = []
    for idx in range(N):
        if fr[idx] <= 1.0:
            continue
        st = spikes[:, idx].astype(np.float32)
        sm = np.convolve(st, gk, mode='same')
        fft = np.fft.fft(sm, n=n_fft)
        mag = np.abs(fft[:n_fft // 2])
        freqs = np.fft.fftfreq(n_fft, d=dt)[:n_fft // 2]
        lo = max(1, int(1.0 / (freqs[1] - freqs[0] + 1e-10)))
        if lo >= len(mag):
            continue
        pk = np.argmax(mag[lo:]) + lo
        if pk < len(freqs) and 1.0 < freqs[pk] < 50:
            all_freqs.append(freqs[pk])

    n_osc = len(all_freqs)
    mean_osc = np.mean(all_freqs) if all_freqs else 0.0
    walk_neurons = sum(1 for f in all_freqs if 4 <= f <= 15)
    gamma_neurons = sum(1 for f in all_freqs if 30 <= f <= 100)
    delta_neurons = sum(1 for f in all_freqs if 1 <= f < 4)

    print(f"   {label:<35} N={N:<4} I={external_I:<5.0f} "
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
    print("🧫 VNC CPG EMERGENCE + ASTROCYTE LAYER")
    print("   Сравнение: isolated → expanded → expanded + астроциты")
    print("═" * 70)

    # 1. Load MANC data
    print("\n📥 LOADING MANC...")
    loader = MANCLoader()
    loader.load()
    loader.find_clusters()

    cluster_all = set()
    for c in loader.clusters.values():
        cluster_all |= c
    vnc_pool = loader.vnc_traced_ids - cluster_all
    print(f"\n   VNC non-reciprocal pool: {len(vnc_pool):,} neurons")

    # 2. Find CPG candidates
    print(f"\n🔍 FINDING CPG CANDIDATES...")
    candidates = find_candidate_by_types(loader, vnc_pool)
    print(f"   Found {len(candidates)} candidate circuits")

    if not candidates:
        print("   ❌ No candidates found. Using fallback types.")
        dnh = [b for b in vnc_pool if 'DNht' in loader.type_map.get(b, '')]
        an = [b for b in vnc_pool if 'AN' in loader.type_map.get(b, '')[:5]]
        i_n = [b for b in vnc_pool if 'IN' in loader.type_map.get(b, '')[:5]]
        for dn in dnh[:2]:
            for a in list(loader.full_adj.get(dn, set()) & set(an))[:2]:
                for inn in list(loader.full_adj.get(a, set()) & set(i_n))[:2]:
                    candidates.append({
                        'n1': dn, 'n2': a, 'n3': inn,
                        't1': loader.type_map.get(dn, '?'),
                        't2': loader.type_map.get(a, '?'),
                        't3': loader.type_map.get(inn, '?'),
                    })

    if not candidates:
        print("❌ No candidates at all!")
        return

    n_candidates = min(3, len(candidates))
    print(f"   Testing {n_candidates} candidates")

    # 3. Experiment
    print(f"\n{'=' * 70}")
    print("🎬 EXPERIMENT: ISOLATED → EXPANDED → EXPANDED+ASTROCYTE")
    print(f"{'=' * 70}")

    all_results = []
    I_exts = [10, 15, 20]
    astro_mu_values = [0.3, 0.5, 0.8]  # volume transmission strengths
    astro_tau_values = [1000.0, 2000.0, 5000.0]  # Ca²⁺ time constants (ms)

    for ci in range(n_candidates):
        cand = candidates[ci]
        core = {cand['n1'], cand['n2'], cand['n3']}
        label = f"C{ci+1}: {str(cand.get('t1','?'))[:10]}→{str(cand.get('t2','?'))[:10]}→{str(cand.get('t3','?'))[:10]}"
        print(f"\n{'─' * 70}")
        print(f"🎯 Candidate {ci+1}: {label}")
        print(f"{'─' * 70}")

        # Phase 1: ISOLATED (3 neurons, baseline)
        print(f"\n   📍 PHASE 1: ISOLATED (3n)")
        for I_val in I_exts:
            r = simulate_hh_baseline(loader, list(core), n_steps=5000, dt_ms=0.5,
                                     external_I=I_val, label=f"isolated I={I_val}")
            if r:
                all_results.append((ci + 1, 'isolated', I_val, 0.0, 0.0, r))

        # Phase 2: EXPANDED (~200n, baseline, no astrocytes)
        print(f"\n   📍 PHASE 2: EXPANDED (~200n)")
        expanded, neighbors = expand_with_neighbors(loader, core, vnc_pool, n_neighbors=100)
        print(f"      Expanded to {len(expanded)} neurons")

        for I_val in I_exts:
            r = simulate_hh_baseline(loader, expanded, n_steps=5000, dt_ms=0.5,
                                     external_I=I_val, label=f"expanded I={I_val}")
            if r:
                all_results.append((ci + 1, 'expanded', I_val, 0.0, 0.0, r))

        # Phase 3: EXPANDED + ASTROCYTE LAYER
        print(f"\n   📍 PHASE 3: EXPANDED + ASTROCYTE LAYER")

        # Build NT-based gap junctions for the expanded subnet
        _, _, _, nt_s, _ = loader.build_csr(expanded)
        gj = build_vnc_gap_junctions(
            list(range(len(expanded))), nt_s,
            g_gap_gaba=0.05, g_gap_ach=0.15, g_gap_glut=0.10,
            p_gaba=0.03, p_ach=0.08, p_glut=0.06,
        )
        print(f"      Gap junctions: {gj.n_pairs} pairs")

        for I_val in I_exts:
            for mu in astro_mu_values:
                for tau in [2000.0]:  # фиксируем tau для управляемости
                    astro_label = f"astro μ={mu} I={I_val}"
                    r = simulate_hh_astrocyte(
                        loader, expanded, n_steps=5000, dt_ms=0.5,
                        external_I=I_val, g_gap=0.1, mu=mu, tau_astro_ms=tau,
                        chunk_size=100, label=astro_label,
                    )
                    if r:
                        all_results.append((ci + 1, f'astro_mu{mu}', I_val, mu, tau, r))

        # Phase 4: EXPANDED + DIFFERENT Ca²⁺ TIMESCALES
        print(f"\n   📍 PHASE 4: EXPANDED + ASTROCYTE (tau sweep)")
        for I_val in [15]:  # best I from baseline
            for tau in astro_tau_values:
                astro_label = f"astro τ={tau:.0f}ms I={I_val}"
                r = simulate_hh_astrocyte(
                    loader, expanded, n_steps=5000, dt_ms=0.5,
                    external_I=I_val, g_gap=0.1, mu=0.5, tau_astro_ms=tau,
                    chunk_size=100, label=astro_label,
                )
                if r:
                    all_results.append((ci + 1, f'astro_tau{tau:.0f}', I_val, 0.5, tau, r))

    # 4. Summary table
    print(f"\n\n{'=' * 130}")
    print("📊 EMERGENCE + ASTROCYTE RESULTS")
    print(f"{'=' * 130}")
    header = f"{'Cand':<5} {'Phase':<18} {'I':<5} {'N':<5} {'Spikes':<8} {'Act%':<6} {'Osc':<5} {'MeanHz':<8} {'Δ':<5} {'🚶Walk':<8} {'γ':<5} {'Emerg?':<10}"
    print(header)
    print("-" * 130)

    for ci, phase, I_val, mu, tau, r in all_results:
        act_pct = f"{r['active'] / max(1, r['N']) * 100:.0f}%"
        walk = r['walk_neurons']
        emerg = ("✅EMERGED" if walk > 0 and r['mean_osc'] > 4
                 else "✅WALK" if walk > 0
                 else "❌" if r['N'] >= 3
                 else "—")
        phase_short = phase[:18]
        print(f"{ci:<5} {phase_short:<18} {I_val:<5.0f} {r['N']:<5} "
              f"{r['total_spikes']:<8} {act_pct:<6} {len(r['osc_freqs']):<5} "
              f"{r['mean_osc']:<8.1f} {r['delta_neurons']:<5} "
              f"{walk:<8} {r['gamma_neurons']:<5} {emerg:<10}")

    # 5. Comparison: expanded vs expanded+astrocyte
    print(f"\n{'=' * 130}")
    print("📋 COMPARISON: Isolated → Expanded → Expanded+Astrocyte")
    print(f"{'=' * 130}")

    for ci in range(1, n_candidates + 1):
        print(f"\n   Candidate {ci}:")
        for I_val in I_exts:
            # Find results for this candidate + I
            iso = next((r for c, p, I, mu, tau, r in all_results
                        if c == ci and p == 'isolated' and I == I_val), None)
            exp = next((r for c, p, I, mu, tau, r in all_results
                        if c == ci and p == 'expanded' and I == I_val), None)

            if not iso or not exp:
                continue

            # Astrocyte results for this I
            astro_results = [(r, mu) for c, p, I, mu, tau, r in all_results
                             if c == ci and p.startswith('astro_mu') and I == I_val and r is not None]

            print(f"      I={I_val:.0f}: "
                  f"iso={iso['mean_osc']:.1f}Hz({iso['walk_neurons']}🚶) → "
                  f"exp={exp['mean_osc']:.1f}Hz({exp['walk_neurons']}🚶)")

            for r, mu in sorted(astro_results, key=lambda x: x[1]):
                astro_gain = r.get('astro_gain_final', np.array([1.0])).mean()
                print(f"        +astro μ={mu:.1f}: "
                      f"{r['mean_osc']:.1f}Hz({r['walk_neurons']}🚶) "
                      f"gain={astro_gain:.3f}")

    # 6. Verdict
    print(f"\n{'=' * 70}")
    print("🔬 SCIENTIFIC VERDICT")
    print(f"{'=' * 70}")

    # Find best astrocyte vs best baseline
    baseline_walk = max((r['walk_neurons'] for _, _, _, _, _, r in all_results
                        if 'astro' not in str(_)), default=0)
    astro_walk = max((r['walk_neurons'] for _, p, _, _, _, r in all_results
                      if 'astro' in str(p)), default=0)

    print(f"\n   Max walking neurons (baseline):      {baseline_walk}")
    print(f"   Max walking neurons (+astrocyte):    {astro_walk}")

    if astro_walk > baseline_walk:
        print(f"\n   ✅ АСТРОЦИТЫ УСИЛИВАЮТ WALKING!")
        print(f"      Volume transmission увеличил число walking нейронов "
              f"на {astro_walk - baseline_walk}")
    elif astro_walk == baseline_walk:
        print(f"\n   ⏸️ Астроциты не изменили число walking нейронов")
    else:
        print(f"\n   ❌ Астроциты подавили walking")

    # Best params
    best_astro = max(((mu, tau, r) for _, p, _, mu, tau, r in all_results
                      if 'astro' in str(p) and r['walk_neurons'] > 0),
                     key=lambda x: x[2]['walk_neurons'], default=None)
    if best_astro:
        mu, tau, r = best_astro
        print(f"\n   🏆 Best astrocyte params:")
        print(f"      μ (volume transmission) = {mu}")
        print(f"      τ (Ca²⁺ time constant)  = {tau:.0f} ms")
        print(f"      Walking neurons: {r['walk_neurons']}")
        print(f"      Mean oscillation: {r['mean_osc']:.1f} Hz")

    print(f"\n✅ EXPERIMENT COMPLETE!")


if __name__ == "__main__":
    main()
