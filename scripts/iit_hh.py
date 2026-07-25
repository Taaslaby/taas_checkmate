#!/usr/bin/env python3
"""🧠 IIT + HH — HH 2ch vs HH 3ch+CaV (T-type calcium pacemaking)

CaV (T-type calcium) даёт низкопороговые спайки и pacemaking,
что обогащает динамику → выше diversity → выше Phi.

Сравнение:
  - HH 2ch (NaV + KV) — стандартный HH
  - HH 3ch+CaV (NaV + KV + CaV) — с T-type calcium pacemaking
"""

import sys, os, time, math, argparse
import numpy as np
from pathlib import Path

os.environ['LD_LIBRARY_PATH'] = '/usr/lib/python3.14/site-packages/nvidia/cudnn/lib'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from scripts.iit_consciousness import IITConsciousness, CONSCIOUSNESS_THRESHOLD
from scripts.iit_vnc import build_vnc_connectome, build_random_connectome

import taas_checkmate as tcm

np.random.seed(42)

N_NEURONS = 200
N_STEPS = 400
DT_MS = 0.05


def build_hh_params(n: int, n_channels: int = 2, rng=None):
    """Build HH parameters: 2ch (NaV, KV) or 3ch (NaV, KV, CaV).
    
    g_max_arr layout: [ch0_all, ch1_all, (ch2_all)] — length n_channels * n
    E_rev_arr: broadcast (length n_channels)
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    # NaV: g_Na=120 (±40), E_Na=50
    g_Na = rng.uniform(80, 160, n).astype(np.float32)
    E_Na = 50.0
    
    # KV: g_K=36 (±12), E_K=-77
    g_K = rng.uniform(24, 60, n).astype(np.float32)
    E_K = -77.0
    
    # CaV (T-type): g_Ca=1 (±0.5), E_Ca=120
    if n_channels >= 3:
        g_Ca = rng.uniform(0.5, 2.0, n).astype(np.float32)
        E_Ca = 120.0
    
    # Leak (per-neuron)
    g_leak = rng.uniform(0.1, 0.5, n).astype(np.float32)
    E_leak = np.full(n, -65.0, dtype=np.float32)
    
    # C_m (per-neuron)
    C_m = np.full(n, 1.0, dtype=np.float32)
    
    # Pack into arrays
    if n_channels == 3:
        g_max_arr = np.concatenate([g_Na, g_K, g_Ca]).astype(np.float32)
        E_rev_arr = np.array([E_Na, E_K, E_Ca], dtype=np.float32)
    else:
        g_max_arr = np.concatenate([g_Na, g_K]).astype(np.float32)
        E_rev_arr = np.array([E_Na, E_K], dtype=np.float32)
    
    print(f"  🔬 HH: {n_channels}ch ({'NaV, KV' if n_channels == 2 else 'NaV, KV, CaV'}) "
          f"× {n}N = {n_channels * n} params")
    return g_max_arr, E_rev_arr, g_leak, E_leak, C_m


def run_iit_hh(name: str, conn, n_neurons: int, n_steps: int,
               use_hh: bool = True, dt_ms: float = DT_MS, pacemaking: bool = False):
    """Run IIT on HH (2ch or 3ch+CaV)."""
    nch = 3 if pacemaking else 2
    label = f"HH {nch}ch{' +CaV' if pacemaking else ''}" if use_hh else "LIF"
    print(f"\n  ═══ {name} ({label}) ═══")
    
    model = "hh" if use_hh else "lif"
    sim = tcm.BrainSimulator(n_neurons, model, pacemaking=pacemaking)
    iit = IITConsciousness(n_neurons, CONSCIOUSNESS_THRESHOLD)
    spike_buf = np.zeros((n_steps, n_neurons), dtype=np.float32)
    
    if use_hh:
        g_max_arr, E_rev_arr, g_leak, E_leak, C_m = build_hh_params(n_neurons, nch)
        sim.set_hh_params(g_max_arr, E_rev_arr, g_leak, E_leak, dt_ms)
        sim.set_C_m(C_m)
    
    # Warm-up
    print("  🔥 Warm-up...", end=" ", flush=True)
    for _ in range(100):
        ext_I = np.random.randn(n_neurons).astype(np.float32) * 0.5 + 2.0
        sim.step_batch(conn, 1, dt_ms, ext_I)
    print("OK")
    
    # Main loop
    t0 = time.time()
    for step in range(n_steps):
        t_sec = step * dt_ms / 1000.0
        ext_I = (np.random.randn(n_neurons) * 1.0 + 3.0).astype(np.float32)
        ext_I[:n_neurons//3] += 5.0 * math.sin(2*math.pi * 3.0 * t_sec)
        ext_I[n_neurons//3:2*n_neurons//3] += 5.0 * math.sin(2*math.pi * 8.0 * t_sec)
        
        spikes = sim.step_batch(conn, 1, dt_ms, ext_I)
        if spikes is not None and len(spikes) > 0 and hasattr(spikes, 'shape'):
            n_rec = min(spikes.shape[-1], n_neurons)
            spike_buf[step, :n_rec] = np.asarray(spikes[:n_neurons]).flatten()
        
        if step % 10 == 0 and step >= 50:
            recent = spike_buf[max(0, step-100):step+1]
            iit.compute_phi(conn, recent)
    
    stats = iit.get_consciousness_stats()
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  Φ mean:   {stats['phi_mean']:.6f} ± {stats['phi_std']:.6f}")
    print(f"  Φ final:  {stats['phi']:.6f}")
    print(f"  D:        {stats['diversity']:.4f}")
    print(f"  I:        {stats['integration']:.4f}")
    print(f"  Conscious: {stats['conscious_ratio']*100:.1f}% | {stats['trend']}")
    return iit


def main():
    p = argparse.ArgumentParser(description="🧠 IIT + HH: 2ch vs 3ch+CaV")
    p.add_argument("--neurons", type=int, default=N_NEURONS)
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--vnc", action="store_true", help="VNC connectome")
    p.add_argument("--cav", action="store_true", help="Сравнить 2ch vs 3ch+CaV")
    args = p.parse_args()
    
    print("╔══════════════════════════════════════════════════════╗")
    print("║  🧠 IIT + HH: 2ch vs 3ch+CaV (T-type calcium)     ║")
    print("╚══════════════════════════════════════════════════════╝")
    
    conn = build_vnc_connectome(args.neurons) if args.vnc else build_random_connectome(args.neurons)
    print(f"  📦 {'VNC' if args.vnc else 'Random'} коннектом: {args.neurons}N")
    
    if args.cav:
        iit_2 = run_iit_hh("HH 2ch", conn, args.neurons, args.steps, pacemaking=False)
        iit_3 = run_iit_hh("HH+CaV", conn, args.neurons, args.steps, pacemaking=True)
        
        print("\n" + "═" * 55)
        print("  📊 СРАВНЕНИЕ: HH 2ch vs HH 3ch+CaV")
        print("═" * 55)
        p2, p3 = np.mean(iit_2.phi_history), np.mean(iit_3.phi_history)
        ratio = p3 / max(p2, 1e-8)
        print(f"  HH 2ch   Φ: {p2:.6f} (D={iit_2.diversity:.4f} I={iit_2.integration:.4f})")
        print(f"  HH+CaV   Φ: {p3:.6f} (D={iit_3.diversity:.4f} I={iit_3.integration:.4f})")
        print(f"  Ratio: {ratio:.2f}x")
        if ratio > 1.2:
            print(f"  ✅ CaV ПОВЫШАЕТ Phi! T-type calcium обогащает динамику ✨")
        elif ratio > 1.0:
            print(f"  ➡ CaV немного помогает (×{ratio:.1f})")
        else:
            print(f"  ➡ CaV ≈ 2ch — разница незначительна")
    else:
        run_iit_hh("HH", conn, args.neurons, args.steps, pacemaking=False)
    
    print(f"\n  ✨ IIT+HH+CaV by @ant0n + Сардана ❤️")


if __name__ == "__main__":
    main()
