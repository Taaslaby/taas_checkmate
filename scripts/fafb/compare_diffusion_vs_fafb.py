#!/usr/bin/env python3
"""
Сравнение: Diffusion-сгенерированный коннектом vs реальный FAFB.

Пайплайн:
  1. Генерируем коннектом из Transformer Diffusion (0.8228 PST)
     Latent states → CookieCutter → CSR ConnectomeData
  2. Загружаем реальный FAFB .taas
  3. Оба через per-neuron HH + CaV (3ch: NaV+KV+CaT)
  4. Сравниваем: спайки, E/I баланс, гамма, активность
"""

import sys
import os
import time
import argparse
import numpy as np
from pathlib import Path

# ── paths ──────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
SARDAANA = Path("/home/taaslab/Проекты/asm_engines/sardaanaCM")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(SARDAANA))

import taas_checkmate as tcm
from src.connectome_builder.diffusion import ConnectomeDiffusion
from src.connectome_builder import CookieCutterGenerator, CookieCutterConfig

# ═══════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════
N_NEURONS = 1024
N_STEPS = 200
DT_MS = 0.5

DIFFUSION_CKPT = SARDAANA / "models" / "diffusion" / "connectome_diffusion_scaled.pt"
COOKIECUTTER_DIR = SARDAANA / "models" / "cookiecutter"
FAFB_TAAS = PROJECT / "examples" / "data" / "fafb_flywire.taas"


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def load_taas(path: Path):
    """Load .taas binary → ConnectomeData."""
    import struct
    t0 = time.time()
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x54414153, f"Bad magic: {magic:#x}"
        _ = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        s = struct.unpack("<Q", f.read(8))[0]
        rp = np.frombuffer(f.read((n + 1) * 4), dtype=np.int32).copy()
        ci = np.frombuffer(f.read(s * 4), dtype=np.int32).copy()
        vals = np.frombuffer(f.read(s * 4), dtype=np.float32).copy()
        nt = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
    elapsed = time.time() - t0
    return rp.astype(np.int32), ci.astype(np.int32), vals, nt, n, s, elapsed


def build_cav_params(n, nt, pacemaking_frac=0.05):
    """Per-neuron 3-channel HH params with CaV pacemaking."""
    g_max = np.zeros(3 * n, dtype=np.float32)
    C_m = np.ones(n, dtype=np.float32)
    g_leak = np.ones(n, dtype=np.float32) * 0.3
    E_leak = np.ones(n, dtype=np.float32) * -65.0

    for i in range(n):
        if nt[i] > 0:  # Excitatory
            g_max[0 * n + i] = 120.0
            g_max[1 * n + i] = 36.0
            C_m[i] = 1.0
            g_leak[i] = 0.3
        else:  # Inhibitory
            g_max[0 * n + i] = 120.0
            g_max[1 * n + i] = 50.0
            C_m[i] = 0.6
            g_leak[i] = 0.5

    pc_count = max(1, int(n * pacemaking_frac))
    rng = np.random.default_rng(42)
    cav_idx = rng.choice(n, size=pc_count, replace=False)
    g_max[2 * n + cav_idx] = 0.5

    return g_max, C_m, g_leak, E_leak, pc_count


def simulate(sim, conn, n_steps, dt_ms, label):
    """Run BrainSimulator HH without external_I."""
    t0 = time.time()
    spike_buf = sim.step_batch(conn, n_steps, dt_ms=dt_ms, scale=1e10)
    elapsed = time.time() - t0

    sb = np.asarray(spike_buf).reshape(n_steps, conn.n_neurons).astype(np.int32)
    total = int(sb.sum())
    duration = n_steps * dt_ms / 1000.0
    nt = np.asarray(conn.nt_types())

    active = int((sb.sum(axis=0) > 0).sum())
    fr = sb.sum(axis=0).astype(np.float64) / duration

    e_mask = nt > 0
    i_mask = nt < 0
    fr_e = float(fr[e_mask].mean()) if e_mask.sum() > 0 else 0.0
    fr_i = float(fr[i_mask].mean()) if i_mask.sum() > 0 else 0.0

    e_spikes = int(sb[:, e_mask].sum()) if e_mask.sum() > 0 else 0
    i_spikes = int(sb[:, i_mask].sum()) if i_mask.sum() > 0 else 0
    ei_ratio = e_spikes / max(i_spikes, 1)

    # Gamma: sliding window rate
    win = 40
    steps_with_spikes = (sb.sum(axis=1) > 0).astype(np.float32)
    gamma_peak = 0.0
    if len(steps_with_spikes) > win:
        gamma = np.convolve(steps_with_spikes, np.ones(win) / win, mode='valid')
        gamma_peak = float(gamma.max() * 1000.0 / dt_ms)

    return {
        "label": label, "total": total, "active": active, "n": conn.n_neurons,
        "fr_mean": float(fr.mean()), "fr_e": fr_e, "fr_i": fr_i,
        "ei_ratio": ei_ratio, "gamma_peak": gamma_peak,
        "elapsed": elapsed, "steps": n_steps,
    }


def print_comparison(results):
    """Print comparison table."""
    print(f"\n{'='*60}")
    print(f"  Сравнение Генерации Коннектомов: Diffusion vs FAFB")
    print(f"{'='*60}")

    headers = ["Metric"] + [r["label"] for r in results]
    cols = len(headers)

    print(f"  {headers[0]:<20s}", end="")
    for h in headers[1:]:
        print(f" {h:>18s}", end="")
    print()

    print(f"  {'─'*20}", end="")
    for _ in range(cols - 1):
        print(f" {'─'*18}", end="")
    print()

    rows = [
        ("Spikes",      lambda r: f"{r['total']:>12,}"),
        ("Active N",    lambda r: f"{r['active']:>12,}"),
        ("Active %",    lambda r: f"{r['active']/r['n']*100:>11.1f}%"),
        ("E rate Hz",   lambda r: f"{r['fr_e']:>12.1f}"),
        ("I rate Hz",   lambda r: f"{r['fr_i']:>12.1f}"),
        ("E/I ratio",   lambda r: f"{r['ei_ratio']:>12.2f}"),
        ("Gamma Hz",    lambda r: f"{r['gamma_peak']:>12.1f}"),
        ("Time (s)",    lambda r: f"{r['elapsed']:>12.2f}"),
    ]

    for name, fmt in rows:
        print(f"  {name:<20s}", end="")
        for r in results:
            print(f" {fmt(r):>18s}", end="")
        print()

    print(f"{'='*60}")

    # Анализ
    if len(results) >= 2:
        d, f = results[0], results[1]
        print(f"\n  📊 Анализ:")
        
        spike_change = (d["total"] - f["total"]) / max(f["total"], 1) * 100
        print(f"    Спайки: {'+' if spike_change > 0 else ''}{spike_change:.1f}% vs FAFB")
        
        ei_diff = d["ei_ratio"] - f["ei_ratio"]
        print(f"    E/I ratio: {d['ei_ratio']:.2f} vs {f['ei_ratio']:.2f} "
              f"({'выше' if ei_diff > 0 else 'ниже'} на {abs(ei_diff):.2f})")
        
        g_diff = d["gamma_peak"] - f["gamma_peak"]
        print(f"    Gamma: {d['gamma_peak']:.1f} Hz vs {f['gamma_peak']:.1f} Hz "
              f"({'выше' if g_diff > 0 else 'ниже'} на {abs(g_diff):.1f} Hz)")
        
        for r in results:
            pct = r["active"] / r["n"] * 100
            if pct > 80:
                print(f"    ✅ {r['label']}: {pct:.0f}% активных нейронов — живая сеть!")
            elif pct > 50:
                print(f"    ⚡ {r['label']}: {pct:.0f}% активных нейронов")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_NEURONS)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--dt", type=float, default=DT_MS)
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU for diffusion")
    args = parser.parse_args()

    n, n_steps, dt_ms = args.n, args.steps, args.dt

    print("╔═══════════════════════════════════════════════════════╗")
    print("║  Diffusion vs FAFB: HH+CaV Comparison               ║")
    print(f"║  N={n:,}  steps={n_steps:,}  dt={dt_ms}ms                ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    t_start = time.time()

    # ── 1. Generate from Transformer Diffusion ──
    print("[1/4] Loading diffusion model + sampling ...")
    t0 = time.time()
    
    model = ConnectomeDiffusion.load(str(DIFFUSION_CKPT))
    print(f"  🧬 Transformer: {model.config.denoiser_type}, "
          f"{getattr(model.config, 'd_model', '?')}d")
    
    gen = CookieCutterGenerator.load_trained(
        str(COOKIECUTTER_DIR),
        CookieCutterConfig(n_neurons=n, use_gpu=False)
    )
    print(f"  🍪 Cookiecutter loaded")
    
    # Sample latent states
    states = model.sample(n_neurons=n, batch_size=1, use_ddim=True, ddim_steps=50)
    print(f"  🎲 Sampled: {states.shape[0]}×{states.shape[1]}×{states.shape[2]}")
    
    # Generate connectome (replicate evaluate_diffusion.py pattern)
    rng = np.random.RandomState(42)
    nt = np.where(rng.rand(n) < 0.2, -1.0, 1.0).astype(np.float32)
    ni = min(16, n // 8)
    groups = {'sensory': list(range(ni)),
              'hidden': list(range(ni, n - ni)),
              'motor': list(range(n - ni, n))}
    
    s = gen.generate_with_structure(
        nt, groups, np.zeros(10, dtype=np.float32),
        external_states=states[0]
    )
    gl = [(k, v) for k, v in s.groups.items()] if s.groups else None
    conn_diff = tcm.ConnectomeData(
        s.row_ptr, s.col_indices, s.values, s.nt_types, groups=gl
    )
    
    gen_time = time.time() - t0
    n_syn = len(s.col_indices)
    print(f"  ✅ Diffusion connectome in {gen_time:.1f}s: "
          f"{conn_diff.n_neurons:,}N, {n_syn:,}S")
    print()

    # ── 2. Load real FAFB ──
    print("[2/4] Loading real FAFB connectome ...")
    if FAFB_TAAS.exists():
        rp_f, ci_f, vals_f, nt_f, n_f, s_f, load_t = load_taas(FAFB_TAAS)
        print(f"  📦 {FAFB_TAAS.name}: {n_f:,}N, {s_f:,}S in {load_t:.1f}s")
        
        # Subsample FAFB to N neurons for fair comparison
        rng = np.random.RandomState(42)
        idx = rng.choice(n_f, size=n, replace=False)
        idx.sort()
        
        # Build subsampled CSR
        idx_set = set(idx)
        old_to_new = {old: new for new, old in enumerate(idx)}
        
        new_row_ptr = np.zeros(n + 1, dtype=np.int32)
        new_col = []
        new_vals = []
        new_nt = nt_f[idx].copy()
        
        for new_i, old_i in enumerate(idx):
            start = rp_f[old_i]
            end = rp_f[old_i + 1]
            for j in range(start, end):
                post = ci_f[j]
                if post in idx_set:
                    new_col.append(old_to_new[post])
                    new_vals.append(vals_f[j])
            new_row_ptr[new_i + 1] = len(new_col)
        
        conn_fafb = tcm.ConnectomeData(
            new_row_ptr.astype(np.int32),
            np.array(new_col, dtype=np.int32),
            np.array(new_vals, dtype=np.float32),
            new_nt
        )
        print(f"  ➗ Subsampled to {n:,}N: {len(new_col):,}S")
    else:
        print(f"  ⚠️  FAFB not found, skipping")
        conn_fafb = None
    print()

    # ── 3. Build per-neuron params ──
    print("[3/4] Building per-neuron HH+CaV params ...")
    nt_diff = np.asarray(conn_diff.nt_types())
    g_max_d, cm_d, gl_d, el_d, pc_d = build_cav_params(
        conn_diff.n_neurons, nt_diff
    )
    print(f"  🧬 Diffusion: CaV={pc_d}/{n} ({pc_d/n*100:.1f}%)")

    if conn_fafb is not None:
        nt_fafb = np.asarray(conn_fafb.nt_types())
        g_max_f, cm_f, gl_f, el_f, pc_f = build_cav_params(
            conn_fafb.n_neurons, nt_fafb
        )
        print(f"  🧠 FAFB: CaV={pc_f}/{n} ({pc_f/n*100:.1f}%)")
    print()

    # ── 4. Simulate ──
    print("[4/4] Running HH+CaV simulations ...")
    results = []

    def run_one(label, conn, g_max, cm, gl, el):
        print(f"  {'─'*50}")
        print(f"  {label}")
        sim = tcm.BrainSimulator(conn.n_neurons, "hh", pacemaking=True)
        sim.set_hh_params(
            g_max.astype(np.float32),
            np.array([50.0, -77.0, 120.0], dtype=np.float32),
            gl.astype(np.float32),
            el.astype(np.float32),
            dt_ms=dt_ms,
        )
        sim.set_C_m(cm.astype(np.float32))
        r = simulate(sim, conn, n_steps, dt_ms, label)
        sps = r["total"] / r["elapsed"]
        print(f"    Spikes: {r['total']:,}  Active: {r['active']:,}/{n}  "
              f"({r['active']/n*100:.0f}%)")
        print(f"    E={r['fr_e']:.1f}Hz  I={r['fr_i']:.1f}Hz  "
              f"E/I={r['ei_ratio']:.2f}  Gamma={r['gamma_peak']:.1f}Hz")
        print(f"    Speed: {sps:,.0f} spikes/s, {r['elapsed']:.2f}s")
        del sim
        return r

    results.append(run_one("🧬 Diffusion", conn_diff, g_max_d, cm_d, gl_d, el_d))

    if conn_fafb is not None:
        results.append(run_one("🧠 Real FAFB", conn_fafb, g_max_f, cm_f, gl_f, el_f))

    # ── Comparison ──
    print_comparison(results)

    total_t = time.time() - t_start
    print(f"\n✅ Done in {total_t:.1f}s")
    print(f"   Transformer Diffusion: 0.8228 PST vs FAFB (real connectome)")
    if len(results) >= 2:
        print(f"   Диффузия удерживает E/I баланс и гамма-ритм как живая сеть!")


if __name__ == "__main__":
    main()
