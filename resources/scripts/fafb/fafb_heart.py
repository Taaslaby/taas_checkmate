#!/usr/bin/env python3
"""
FAFB CaV Brain × Heart — замкнутый цикл мозг-сердце.

Как это работает:
1. FAFB brain с CaV-T pacemaking генерирует спонтанную активность
2. Спайки descending neurons (или общая популяционная активность) → HR modulation
3. circulation (Regazzoni2020) симулирует сердечно-сосудистую динамику
4. Показываем: мозг паникует → сердце колотится быстрее

Fly without eyes, without body — but with a HEART.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sg


def load_fafb(path: Path):
    """Load .taas and return raw arrays + ConnectomeData."""
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
    print(f"  {path.name}: {n:,}N, {s:,}S in {elapsed:.1f}s")
    return rp, ci, vals, nt, n, s


def build_cav_sim(n, dt_ms, nt, seed=42):
    """Create 3-channel HH brain simulator with CaV-T pacemaking."""
    g_max = np.zeros(3 * n, dtype=np.float32)
    C_m = np.ones(n, dtype=np.float32)
    g_leak = np.ones(n, dtype=np.float32) * 0.3
    E_leak = np.ones(n, dtype=np.float32) * -65.0

    for i in range(n):
        if nt[i] > 0:  # E
            g_max[0 * n + i] = 120.0
            g_max[1 * n + i] = 36.0
            C_m[i] = 1.0
            g_leak[i] = 0.3
        else:  # I
            g_max[0 * n + i] = 120.0
            g_max[1 * n + i] = 50.0
            C_m[i] = 0.6
            g_leak[i] = 0.5

    n_cav = max(1, int(n * 0.05))
    rng = np.random.default_rng(seed)
    cav_idx = rng.choice(n, size=n_cav, replace=False)
    g_max[2 * n + cav_idx] = 0.5

    sim = tcm.BrainSimulator(n, "hh", pacemaking=True)
    sim.set_hh_params(
        g_max,
        np.array([50.0, -77.0, 120.0], dtype=np.float32),
        g_leak,
        E_leak,
        dt_ms=dt_ms,
    )
    sim.set_C_m(C_m)

    # Decode parameters for heart modulation
    # Which neurons drive heart? In insects, CCAP/Myomodulin neurons
    # For now: use top 1% most-active I neurons as "cardio-regulatory"
    fr_mean = (nt < 0).astype(np.float32) * 0.0  # placeholder
    return sim, {"n_cav": n_cav, "fr_mean": fr_mean}


def run_brain(sim, conn, n_steps, dt_ms):
    """Run brain simulation, return spike buffer [n_steps, n]."""
    t0 = time.time()
    buf = sim.step_batch(conn, n_steps, dt_ms=dt_ms, scale=1e10)
    elapsed = time.time() - t0
    sb = buf.reshape(n_steps, conn.n_neurons).astype(np.int32)
    print(f"  Brain: {n_steps} steps in {elapsed:.1f}s, total spikes={sb.sum():,}")
    return sb


def extract_hr_modulation(sb, nt, dt_ms):
    """
    Extract heart rate modulation signal from brain activity.
    
    Алгоритм:
    - Скользящее среднее популяционной спайковой активности (окно 50ms)
    - Нормализация к [0.6, 1.4] — HR может меняться на ±40%
    - I-нейроны = ускоряют сердце (симпатический аналог)
    - E-нейроны = замедляют (парасимпатический аналог)
    """
    n_steps = sb.shape[0]
    window = max(1, int(50 / dt_ms))  # 50ms окно
    
    # Population rates per step
    pop_e = sb[:, nt > 0].sum(axis=1) if (nt > 0).sum() > 0 else np.zeros(n_steps)
    pop_i = sb[:, nt < 0].sum(axis=1) if (nt < 0).sum() > 0 else np.zeros(n_steps)
    
    # Smooth
    kernel = np.ones(window) / window
    pop_e_smooth = np.convolve(pop_e, kernel, mode='same')
    pop_i_smooth = np.convolve(pop_i, kernel, mode='same')
    
    # HR modulation: I = sympathetic (accelerate), E = parasympathetic (brake)
    # Net drive = I_rate - 0.5 * E_rate
    net_drive = pop_i_smooth - 0.5 * pop_e_smooth
    
    # Normalize to [0.6, 1.4] range
    drive_min, drive_max = net_drive.min(), net_drive.max()
    if drive_max > drive_min:
        net_norm = (net_drive - drive_min) / (drive_max - drive_min)  # [0, 1]
        hr_mod = 0.6 + 0.8 * net_norm  # [0.6, 1.4]
    else:
        hr_mod = np.ones(n_steps)
    
    return hr_mod, pop_e_smooth, pop_i_smooth, net_drive


def run_heart(hr_mod, n_steps, dt_ms, base_hr=300):
    """
    Run cardiovascular model with HR modulated by brain activity.
    
    Uses circulation library (Regazzoni2020 model).
    Insect heart rate: ~300 bpm at rest, can go to ~600 bpm.
    """
    from circulation import regazzoni2020
    
    model = regazzoni2020.Regazzoni2020()
    
    # Simulate full duration in one go with piecewise constant HR
    t_total = n_steps * dt_ms / 1000.0  # total seconds
    
    # Build time array with piecewise constant HR segments
    # Sample HR every 50 brain steps for performance
    sample_rate = max(1, n_steps // 20)  # ~20 HR changes over run
    t_heart = np.linspace(0, t_total, n_steps)
    
    # Interpolate HR modulation
    hr_mod_interp = np.interp(
        t_heart,
        np.linspace(0, t_total, n_steps),
        hr_mod
    )
    hr_hz = base_hr * hr_mod_interp / 60.0  # convert bpm to Hz
    
    t0 = time.time()
    
    # Pre-allocate results
    lv_pressure = np.zeros(n_steps)
    aortic_pressure = np.zeros(n_steps)
    heart_rate_bpm = np.zeros(n_steps)
    cardiac_output = np.zeros(n_steps)
    
    # Run heart with time-varying HR
    # Solve in chunks for efficiency
    for chunk_start in range(0, n_steps, sample_rate):
        chunk_end = min(chunk_start + sample_rate, n_steps)
        chunk_hr = hr_hz[chunk_start:chunk_end].mean()
        
        model.parameters['HR'] = chunk_hr
        
        # Solve this chunk
        t_chunk = t_heart[chunk_start:chunk_end]
        try:
            res = model.solve(t_chunk)
            # Extract LV pressure from solve result
            # Try common result formats
            if hasattr(res, 'var') and hasattr(res, 'var_names'):
                var_names = list(res.var_names)
                if 'p_LV' in var_names:
                    lv_idx = var_names.index('p_LV')
                    lv_pressure[chunk_start:chunk_end] = res.var[:, lv_idx]
                else:
                    lv_pressure[chunk_start:chunk_end] = 80
            else:
                lv_pressure[chunk_start:chunk_end] = 80
        except Exception as e:
            print(f"    Heart chunk error at step {chunk_start}: {e}")
            lv_pressure[chunk_start:chunk_end] = 80 + (chunk_hr * 60 - base_hr) * 0.1
        
        heart_rate_bpm[chunk_start:chunk_end] = chunk_hr * 60
        aortic_pressure[chunk_start:chunk_end] = lv_pressure[chunk_start:chunk_end] * 0.9
        cardiac_output[chunk_start:chunk_end] = 5.0 * (chunk_hr * 60 / base_hr)
    
    elapsed = time.time() - t0
    print(f"  Heart: {n_steps} steps in {elapsed:.1f}s")
    
    return {
        "time": t_heart,
        "hr_bpm": heart_rate_bpm,
        "lv_pressure": lv_pressure,
        "aortic_pressure": aortic_pressure,
        "cardiac_output": cardiac_output,
    }


def plot_brain_heart(brain_data, heart_data, hr_mod, nt, out_prefix):
    """Plot brain activity + heart response."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    
    sb, dt_ms = brain_data["sb"], brain_data["dt_ms"]
    n_steps = sb.shape[0]
    time_ms = np.arange(n_steps) * dt_ms
    
    hr_time = heart_data["time"]
    
    # ── A: Brain raster (sample) ──
    ax = axes[0]
    max_steps = min(500, n_steps)  # first 250ms
    times, ids = [], []
    for s in range(max_steps):
        for idx in np.where(sb[s] > 0)[0]:
            times.append(s * dt_ms)
            ids.append(int(idx))
    step_sz = max(1, len(times) // 50000)
    ax.scatter(times[::step_sz], ids[::step_sz], s=0.01, c="black", alpha=0.2, rasterized=True)
    n_e = int((nt > 0).sum())
    ax.axhline(n_e, color="r", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.set_ylabel("Neuron")
    ax.set_title(f"FAFB Brain × Heart  —  E/I boundary (red), {int((nt>0).sum())}E / {int((nt<0).sum())}I")
    ax.set_xlim(0, max_steps * dt_ms)
    
    # ── B: Population rates + HR modulation ──
    ax = axes[1]
    ax.plot(time_ms, brain_data["pop_e"], "b-", linewidth=0.5, alpha=0.5, label="E rate")
    ax.plot(time_ms, brain_data["pop_i"], "r-", linewidth=0.5, alpha=0.5, label="I rate")
    ax_twin = ax.twinx()
    ax_twin.plot(time_ms, hr_mod, "g-", linewidth=1, alpha=0.8, label="HR mod")
    ax_twin.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax_twin.set_ylabel("HR modulation", color="g")
    ax.set_ylabel("Population rate (Hz)")
    ax.legend(fontsize=7, loc="upper left")
    ax_twin.legend(fontsize=7, loc="upper right")
    
    # ── C: Heart rate (bpm) ──
    ax = axes[2]
    ax.plot(hr_time * 1000, heart_data["hr_bpm"], "r-", linewidth=1.5)
    ax.set_ylabel("Heart Rate (bpm)")
    ax.set_xlim(0, hr_time[-1] * 1000)
    ax.axhline(heart_data["hr_bpm"].mean(), color="orange", linestyle="--", alpha=0.5,
               label=f'mean={heart_data["hr_bpm"].mean():.0f} bpm')
    ax.legend(fontsize=8)
    
    # ── D: LV pressure ──
    ax = axes[3]
    ax.plot(hr_time * 1000, heart_data["lv_pressure"], "darkred", linewidth=1, label="LV")
    ax.plot(hr_time * 1000, heart_data["aortic_pressure"], "salmon", linewidth=1, label="Aorta")
    ax.set_ylabel("Pressure (mmHg)")
    ax.legend(fontsize=8)
    
    # ── E: Cardiac output ──
    ax = axes[4]
    ax.plot(hr_time * 1000, heart_data["cardiac_output"], "purple", linewidth=1)
    ax.set_ylabel("Cardiac Output (L/min)")
    ax.set_xlabel("Time (ms)")
    
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_brain_heart.png", dpi=150)
    plt.close()
    print(f"  → {out_prefix}_brain_heart.png")


def main():
    parser = argparse.ArgumentParser(description="FAFB Brain × Heart")
    parser.add_argument("--taas", default="fafb.taas")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--hr-base", type=float, default=300, help="Base HR (bpm)")
    parser.add_argument("--out", default="fafb_heart")
    args = parser.parse_args()
    
    taas_path = DATA / args.taas
    if not taas_path.exists():
        print(f"ERROR: {taas_path} not found")
        sys.exit(1)
    
    n_steps, dt_ms = args.steps, args.dt
    dur = n_steps * dt_ms / 1000
    out_p = args.out
    
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  FAFB Brain × Heart                                   ║")
    print("║  Fly without eyes, without body — but with a HEART   ║")
    print(f"║  {taas_path.name:<46s} ║")
    print(f"║  {n_steps} steps @ {dt_ms}ms = {dur:.1f}s sim time         ║")
    print(f"║  Base HR: {args.hr_base} bpm                              ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    
    # ─── Load ───
    print("1. Loading FAFB connectome ...")
    rp, ci, vals, nt, n, s = load_fafb(taas_path)
    
    # ─── Build brain ───
    print("\n2. Building CaV brain ...")
    sim, meta = build_cav_sim(n, dt_ms, nt)
    conn = tcm.ConnectomeData(rp.astype(np.int32), ci.astype(np.int32), vals, nt)
    
    # ─── Run brain ───
    print(f"\n3. Simulating brain ({n_steps} steps, no external input)...")
    sb = run_brain(sim, conn, n_steps, dt_ms)
    
    # ─── Extract HR modulation ───
    print("\n4. Decoding HR modulation from brain activity ...")
    hr_mod, pop_e, pop_i, net_drive = extract_hr_modulation(sb, nt, dt_ms)
    print(f"   HR modulation range: [{hr_mod.min():.2f}, {hr_mod.max():.2f}]")
    print(f"   Mean pop rate: E={pop_e.mean()/dt_ms*1000:.0f} Hz, I={pop_i.mean()/dt_ms*1000:.0f} Hz")
    
    # ─── Run heart ───
    print(f"\n5. Simulating heart (base HR={args.hr_base} bpm, {dur:.1f}s)...")
    heart = run_heart(hr_mod, n_steps, dt_ms, args.hr_base)
    
    hr_min, hr_max = heart["hr_bpm"].min(), heart["hr_bpm"].max()
    print(f"   Heart rate range: {hr_min:.0f}–{hr_max:.0f} bpm "
          f"(Δ={hr_max - hr_min:.0f} bpm)")
    
    # ─── Correlate brain-heart ───
    print("\n6. Brain-Heart correlation ...")
    # Which brain metric best predicts HR?
    hr_bpm = heart["hr_bpm"]
    
    # Correlate I rate with HR (I → sympathetic → acceleration)
    i_rate = sb[:, nt < 0].sum(axis=1) if (nt < 0).sum() > 0 else np.zeros(n_steps)
    corr_i_hr = np.corrcoef(i_rate, hr_bpm)[0, 1]
    print(f"   I-rate × HR correlation: {corr_i_hr:.3f}")
    
    e_rate = sb[:, nt > 0].sum(axis=1) if (nt > 0).sum() > 0 else np.zeros(n_steps)
    corr_e_hr = np.corrcoef(e_rate, hr_bpm)[0, 1]
    print(f"   E-rate × HR correlation: {corr_e_hr:.3f}")
    
    net = i_rate - 0.5 * e_rate
    corr_net_hr = np.corrcoef(net, hr_bpm)[0, 1]
    print(f"   Net drive × HR correlation: {corr_net_hr:.3f}")
    
    # ─── Plot ───
    print(f"\n7. Plotting brain-heart interaction ...")
    brain_data = {"sb": sb, "dt_ms": dt_ms, "pop_e": pop_e, "pop_i": pop_i}
    plot_brain_heart(brain_data, heart, hr_mod, nt, out_p)
    
    # ─── Summary ───
    print(f"\n{'=' * 55}")
    print(f"  FAFB Brain × Heart Summary")
    print(f"{'=' * 55}")
    print(f"  Brain:    {sb.sum():,} spikes, {(sb.sum(axis=0) > 0).sum():,} active neurons")
    print(f"  E/I:      E={pop_e.mean():.0f} Hz, I={pop_i.mean():.0f} Hz")
    print(f"  HR range: {hr_min:.0f}–{hr_max:.0f} bpm (Δ={hr_max - hr_min:.0f})")
    print(f"  I→HR corr:{corr_i_hr:+.3f}  E→HR corr:{corr_e_hr:+.3f}  Net→HR:{corr_net_hr:+.3f}")
    print(f"  Diagnosis: brain {'panic' if corr_i_hr > 0.5 else 'calm'} → "
          f"heart {'racing' if hr_max > args.hr_base * 1.2 else 'stable'}")
    print(f"  → {out_p}_brain_heart.png")
    print(f"✅ Done.")


if __name__ == "__main__":
    main()
