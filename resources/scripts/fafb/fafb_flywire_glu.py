#!/usr/bin/env python3
"""
FAFB FlyWire NT rebake + CaV synaptic comparison.

1. Loads fafb.taas, replaces nt_types (+1/-1) with real FlyWire NT codes:
   +1.0 = ACh (nAChR, fast exc)  — ~60% of E
   -1.0 = GABA (GABA_A, fast inh) — default I
   +2.0 = GLUT (GluCl, Cl⁻ channel!) — ~16% of E
   -2.0 = DA/SER/OCT (mAChR, slow mod) — ~2.5% of I

2. Runs 3 comparisons on FAFB + CaV:
   a) Simple E/I (+1/-1) — instant current
   b) Simple E/I (+1/-1) — synaptic (AMPA/GABA_A)
   c) FlyWire NT (+1/-1/+2/-2) — synaptic (nAChR/GABA_A/GluCl/mAChR)

All with CaV-T pacemaking (5%), NO external current.
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "examples" / "data"

sys.path.insert(0, str(PROJECT))
import taas_checkmate as tcm

# ─── matplotlib ───
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_taas(path: Path):
    """Load .taas → numpy arrays + ConnectomeData."""
    t0 = time.time()
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x54414153
        _ = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        s = struct.unpack("<Q", f.read(8))[0]
        rp = np.frombuffer(f.read((n + 1) * 4), dtype=np.int32).copy()
        ci = np.frombuffer(f.read(s * 4), dtype=np.int32).copy()
        vals = np.frombuffer(f.read(s * 4), dtype=np.float32).copy()
        nt = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
    elapsed = time.time() - t0
    print(f"  {path.name} loaded in {elapsed:.1f}s — {n:,}N, {s:,}S")
    return rp, ci, vals, nt, n, s


def assign_flywire_nt(nt_orig, seed=42):
    """
    Map simple +1/-1 nt_types to FlyWire NT codes.
    
    FlyWire Princeton synapse stats:
      ACH: 60.5%  → +1.0  (nAChR, fast exc)
      GABA: 22.1% → -1.0  (GABA_A, fast inh)
      GLUT: 15.6% → +2.0  (GluCl, Cl⁻ channel — inhibitory in insects!)
      DA:   1.2%  → -2.0  (mAChR, slow mod)
      SER:  0.8%  → -2.0  (mAChR, slow mod)
      OCT:  0.5%  → -2.0  (mAChR, slow mod)
    """
    n = len(nt_orig)
    nt_new = nt_orig.copy()
    rng = np.random.default_rng(seed)

    # E neurons: 60% ACh → +1.0 (keep), 16% GLUT → +2.0
    exc_mask = nt_orig > 0
    exc_idx = np.where(exc_mask)[0]
    n_exc = len(exc_idx)
    n_glut = int(n_exc * 0.16)
    glut_idx = rng.choice(exc_idx, size=n_glut, replace=False)
    nt_new[glut_idx] = 2.0
    print(f"  GLUT (+2.0): {n_glut:,} ({n_glut/n_exc*100:.1f}% of E)")

    # I neurons: 22% GABA → -1.0 (keep), 2.5% DA/SER/OCT → -2.0
    inh_mask = nt_orig < 0
    inh_idx = np.where(inh_mask)[0]
    n_inh = len(inh_idx)
    n_mod = max(1, int(n_inh * 0.025))
    mod_idx = rng.choice(inh_idx, size=n_mod, replace=False)
    nt_new[mod_idx] = -2.0
    print(f"  DA/SER/OCT (-2.0): {n_mod:,} ({n_mod/n_inh*100:.1f}% of I)")

    # Stats
    for val in sorted(set(nt_new)):
        cnt = int((nt_new == val).sum())
        print(f"    NT={val:+5.0f}: {cnt:>7,} ({cnt/n*100:>5.1f}%)")

    return nt_new


def make_sim_and_params(n, dt_ms, nt_flywire=None):
    """Create 3-channel CaV simulator with per-neuron bio params."""
    n_channels = 3
    g_max = np.zeros(n_channels * n, dtype=np.float32)
    C_m = np.ones(n, dtype=np.float32)
    g_leak = np.ones(n, dtype=np.float32) * 0.3
    E_leak = np.ones(n, dtype=np.float32) * -65.0

    # Use flywire nt if provided, else original
    nt = nt_flywire if nt_flywire is not None else np.ones(n)

    # Determine sign: any positive nt (ACh=+1 or GLUT=+2) → E params
    is_e = nt > 0
    is_i = nt < 0

    g_max[0 * n + is_e] = 120.0   # NaV
    g_max[1 * n + is_e] = 36.0    # KV (E)
    C_m[is_e] = 1.0
    g_leak[is_e] = 0.3

    g_max[0 * n + is_i] = 120.0   # NaV
    g_max[1 * n + is_i] = 50.0    # KV (I, faster repolarization)
    C_m[is_i] = 0.6               # lower capacitance → fast-spiking
    g_leak[is_i] = 0.5

    # CaV-T on 5% of neurons
    n_cav = max(1, int(n * 0.05))
    rng = np.random.default_rng(42)
    cav_idx = rng.choice(n, size=n_cav, replace=False)
    g_max[2 * n + cav_idx] = 0.5

    # Create sim
    sim = tcm.BrainSimulator(n, "hh", pacemaking=True)
    sim.set_hh_params(
        g_max,
        np.array([50.0, -77.0, 120.0], dtype=np.float32),  # Na, K, Ca reversal
        g_leak,
        E_leak,
        dt_ms=dt_ms,
    )
    sim.set_C_m(C_m)

    return sim


def run_sim(sim, conn, n_steps, dt_ms, label):
    """Run simulation and collect results."""
    t0 = time.time()
    buf = sim.step_batch(conn, n_steps, dt_ms=dt_ms, scale=1e10)
    elapsed = time.time() - t0
    sb = buf.reshape(n_steps, conn.n_neurons).astype(np.int32)
    total = int(sb.sum())
    return {"label": label, "buf": sb, "elapsed": elapsed, "total": total}


def run_sim_synaptic(sim, conn, n_steps, dt_ms, label):
    """Run conductance-based simulation."""
    t0 = time.time()
    buf = sim.step_batch_synaptic(conn, n_steps, dt_ms=dt_ms, scale=1e10)
    elapsed = time.time() - t0
    sb = buf.reshape(n_steps, conn.n_neurons).astype(np.int32)
    total = int(sb.sum())
    return {"label": label, "buf": sb, "elapsed": elapsed, "total": total}


def analyze(sb, dt_ms):
    """Compute metrics from spike buffer."""
    n_steps, n = sb.shape
    duration = n_steps * dt_ms / 1000.0
    dt = dt_ms / 1000.0
    pop_rate = sb.sum(axis=1) / dt
    fr = sb.sum(axis=0) / duration
    active = int((sb.sum(axis=0) > 0).sum())
    return {"pop_rate": pop_rate, "fr": fr, "active": active,
            "duration": duration, "n_steps": n_steps, "n": n}


def print_results(results, nt, dt_ms):
    """Print comparison table."""
    for r in results:
        ana = analyze(r["buf"], dt_ms)
        fr_e = ana["fr"][nt > 0].mean() if (nt > 0).sum() > 0 else 0
        fr_i = ana["fr"][nt < 0].mean() if (nt < 0).sum() > 0 else 0
        e_active = int(((r["buf"].sum(axis=0) > 0) & (nt > 0)).sum()) if (nt > 0).sum() > 0 else 0
        i_active = int(((r["buf"].sum(axis=0) > 0) & (nt < 0)).sum()) if (nt < 0).sum() > 0 else 0
        print(f"    Spikes: {r['total']:>8,}  "
              f"Active: {ana['active']:>6,}/{ana['n']:,}  "
              f"E={fr_e:.1f}Hz I={fr_i:.1f}Hz  "
              f"E/I={fr_e/max(fr_i,1):.2f}  "
              f"Time: {r['elapsed']:.2f}s")


def plot_comparison(results, nt, dt_ms, out_prefix):
    """Plot population rates for all 3 runs."""
    fig, axes = plt.subplots(len(results), 1, figsize=(14, 3 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]

    dt = dt_ms / 1000.0
    for ax, r in zip(axes, results):
        sb = r["buf"]
        n_steps = sb.shape[0]
        time_ms = np.arange(n_steps) * dt * 1000
        pop = sb.sum(axis=1) / dt

        # E and I contributions
        pop_e = sb[:, nt > 0].sum(axis=1) / dt if (nt > 0).sum() > 0 else np.zeros(n_steps)
        pop_i = sb[:, nt < 0].sum(axis=1) / dt if (nt < 0).sum() > 0 else np.zeros(n_steps)

        ax.plot(time_ms, pop, "k-", linewidth=0.5, alpha=0.7, label="Total")
        ax.plot(time_ms, pop_e, "b-", linewidth=0.5, alpha=0.4, label="E")
        ax.plot(time_ms, pop_i, "r-", linewidth=0.5, alpha=0.4, label="I")
        ax.set_ylabel("Pop rate (Hz)")
        ax.set_title(r["label"])
        ax.legend(fontsize=8)
        ax.set_xlim(0, time_ms[-1])

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_lfp.png", dpi=150)
    plt.close()
    print(f"  → {out_prefix}_lfp.png")


def main():
    parser = argparse.ArgumentParser(description="FAFB FlyWire NT + CaV synaptic comparison")
    parser.add_argument("--taas", default="fafb.taas", help="Input .taas file")
    parser.add_argument("--steps", type=int, default=1000, help="Simulation steps")
    parser.add_argument("--dt", type=float, default=0.5, help="Timestep (ms)")
    parser.add_argument("--out", default="fafb_glu", help="Output prefix")
    args = parser.parse_args()

    taas_path = DATA / args.taas
    if not taas_path.exists():
        print(f"ERROR: {taas_path} not found")
        sys.exit(1)

    dt_ms = args.dt
    n_steps = args.steps
    out_p = args.out

    print("╔════════════════════════════════════════════════════════╗")
    print("║  FAFB FlyWire NT + CaV Synaptic                       ║")
    print("║  Добавляем GLUT (GluCl) и DA/SER/OCT (mAChR)          ║")
    print(f"║  {taas_path.name:<46s} ║")
    print(f"║  {n_steps} steps @ {dt_ms}ms                                ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()

    # ─── Load ───
    print("1. Loading FAFB connectome ...")
    rp, ci, vals, nt_orig, n, s = load_taas(taas_path)

    # ─── Assign FlyWire NT ───
    print("\n2. Assigning FlyWire NT codes ...")
    nt_fw = assign_flywire_nt(nt_orig, seed=42)

    # ─── Stats ───
    n_e = int((nt_fw > 0).sum())
    n_i = int((nt_fw < 0).sum())
    n_glut = int((nt_fw == 2.0).sum())
    n_mod = int((nt_fw == -2.0).sum())
    print(f"\n   FlyWire NT summary:")
    print(f"     ACh (+1.0):  {n_e - n_glut:>6,}E")
    print(f"     GLUT (+2.0): {n_glut:>6,}E  (GluCl — Cl⁻ channel!)")
    print(f"     GABA (-1.0): {n_i - n_mod:>6,}I")
    print(f"     MOD  (-2.0): {n_mod:>6,}I  (DA/SER/OCT — slow mod)")

    # ─── Build connectomes ───
    print("\n3. Building connectomes ...")
    conn_orig = tcm.ConnectomeData(rp.astype(np.int32), ci.astype(np.int32), vals.copy(), nt_orig.copy())
    conn_fw = tcm.ConnectomeData(rp.astype(np.int32), ci.astype(np.int32), vals.copy(), nt_fw.copy())

    # Build receptor types for synaptic mode
    conn_orig.build_receptor_types()
    conn_fw.build_receptor_types_flywire(nt_fw.astype(np.float32))

    print(f"   Original: {conn_orig.n_neurons:,}N, {conn_orig.n_synapses:,}S, "
          f"{'receptors' if conn_orig.has_receptor_types() else 'no receptors'}")
    print(f"   FlyWire:  {conn_fw.n_neurons:,}N, {conn_fw.n_synapses:,}S, "
          f"{'receptors' if conn_fw.has_receptor_types() else 'no receptors'}")

    # ─── Run simulations ───
    results = []

    def t(desc):
        print(f"\n{'─'*55}")
        print(f"  {desc}")
        print(f"{'─'*55}")

    # ═══ Test A: Simple E/I + instant current ═══
    t("A. Simple E/I (+1/-1) — instant current")
    sim = make_sim_and_params(n, dt_ms, nt_orig)
    r = run_sim(sim, conn_orig, n_steps, dt_ms, "E/I instant")
    results.append(r)
    print_results([r], nt_orig, dt_ms)

    # ═══ Test B: Simple E/I + synaptic (AMPA/GABA_A) ═══
    t("B. Simple E/I (+1/-1) — synaptic (AMPA/GABA_A)")
    sim = make_sim_and_params(n, dt_ms, nt_orig)
    r = run_sim_synaptic(sim, conn_orig, n_steps, dt_ms, "E/I synaptic")
    results.append(r)
    print_results([r], nt_orig, dt_ms)

    # ═══ Test C: FlyWire NT + synaptic (nAChR/GABA_A/GluCl/mAChR) ═══
    t("C. FlyWire NT (+1/-1/+2/-2) — synaptic (nAChR/GABA_A/GluCl/mAChR)")
    sim = make_sim_and_params(n, dt_ms, nt_fw)
    r = run_sim_synaptic(sim, conn_fw, n_steps, dt_ms, "FlyWire synaptic")
    results.append(r)
    print_results([r], nt_fw, dt_ms)

    # ─── Comparison ───
    print(f"\n{'='*55}")
    print(f"  Comparison: CaV pacemaking, NO external current")
    print(f"{'='*55}")
    print(f"  {'Metric':<30s} {'E/I instant':>12s} {'E/I syn':>12s} {'FlyWire':>12s}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")

    for key, name, fmt in [
        ("total", "Total spikes", ","),
        ("elapsed", "Wall time (s)", ".2f"),
    ]:
        vals_r = [r[key] for r in results]
        if fmt == ",":
            print(f"  {name:<30s} {vals_r[0]:>12,} {vals_r[1]:>12,} {vals_r[2]:>12,}")
        else:
            print(f"  {name:<30s} {vals_r[0]:>12.{fmt[1:]}} {vals_r[1]:>12.{fmt[1:]}} {vals_r[2]:>12.{fmt[1:]}}")

    # Active neurons
    for i, label in enumerate(["A total", "E", "I"]):
        nt_ref = nt_orig if i < 2 else nt_fw
        mask = nt_ref > 0 if i == 1 else (nt_ref < 0 if i == 2 else slice(None))
        if i == 0:
            vals_r = [int((r["buf"].sum(axis=0) > 0).sum()) for r in results]
        else:
            vals_r = [int(((r["buf"].sum(axis=0) > 0) & mask).sum()) for r in results]
        print(f"  {'Active ' + label:<30s} {vals_r[0]:>12,} {vals_r[1]:>12,} {vals_r[2]:>12,}")

    # Firing rates
    for i, label in enumerate(["E rate (Hz)", "I rate (Hz)"]):
        nt_ref = nt_orig if i == 0 else nt_fw
        mask = nt_ref > 0 if i == 0 else (nt_ref < 0)
        vals_r = []
        for r in results:
            ana = analyze(r["buf"], dt_ms)
            fr = ana["fr"][mask].mean() if mask.sum() > 0 else 0
            vals_r.append(fr)
        print(f"  {label:<30s} {vals_r[0]:>12.1f} {vals_r[1]:>12.1f} {vals_r[2]:>12.1f}")

    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")

    # ─── Diagnosis ───
    print("\n  Выводы:")
    r0, r1, r2 = results
    if r2["total"] != r1["total"]:
        delta = r2["total"] - r1["total"]
        direction = "больше" if delta > 0 else "меньше"
        print(f"  🧬 FlyWire NT даёт {abs(delta):,} {direction} спайков, чем simple E/I synaptic")
    if r2["total"] != r0["total"]:
        delta = r2["total"] - r0["total"]
        direction = "больше" if delta > 0 else "меньше"
        print(f"  🧬 FlyWire NT даёт {abs(delta):,} {direction} спайков, чем instant current")

    # Check GluCl effect: compare GLUT vs ACh neurons
    if n_glut > 0:
        buf_fw = r2["buf"]
        fr_ach = buf_fw[:, nt_fw == 1.0].sum(axis=0).mean() / (n_steps * dt_ms / 1000) if (nt_fw == 1.0).sum() > 0 else 0
        fr_glut = buf_fw[:, nt_fw == 2.0].sum(axis=0).mean() / (n_steps * dt_ms / 1000) if n_glut > 0 else 0
        print(f"  🔬 ACh neurons (nAChR):   mean {fr_ach:.1f} Hz")
        print(f"  🔬 GLUT neurons (GluCl):  mean {fr_glut:.1f} Hz")
        if fr_glut < fr_ach:
            print(f"  ✅ GluCl работает как тормозной — GLUT нейроны тише ACh")

    # ─── Plot ───
    print("\n4. Plotting ...")
    plot_comparison(results, nt_orig, dt_ms, out_p)

    print(f"\n✅ Done. Results saved to {out_p}_lfp.png")


if __name__ == "__main__":
    main()
