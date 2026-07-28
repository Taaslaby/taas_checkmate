#!/usr/bin/env python3
"""
EEG паникующего FAFB мозга.

Записывает population rate (proxy LFP) с полного FAFB с CaV-T pacemaking.
Никакого внешнего тока — только CaV-triggered спонтанная активность.

Выхлоп:
  - fafb_eeg_raster.png    — растровая диаграмма (первые 100ms)
  - fafb_eeg_lfp.png       — population rate + E/I вклады + спектр
  - fafb_eeg_spectrum.png  — спектрограмма + power spectrum

Fly without eyes, without body — just a brain panic-ing in a jar.
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

# ─── matplotlib (Agg для сервера) ───
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sg


def load_fafb(path: Path):
    """Load .taas from binary."""
    t0 = time.time()
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == 0x54414153, f"Not a .taas file (magic={magic:#x})"
        _ = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        s = struct.unpack("<Q", f.read(8))[0]
        rp = np.frombuffer(f.read((n + 1) * 4), dtype=np.int32).copy()
        ci = np.frombuffer(f.read(s * 4), dtype=np.int32).copy()
        vals = np.frombuffer(f.read(s * 4), dtype=np.float32).copy()
        nt = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
    elapsed = time.time() - t0
    print(f"  {path.name} loaded in {elapsed:.1f}s — {n:,}N, {s:,}S")
    return tcm.ConnectomeData(rp.astype(np.int32), ci.astype(np.int32), vals, nt), nt


def make_cav_params(nt, cav_pct: float, seed: int = 42):
    """
    Per-neuron HH params + CaV-T on random subset.
    Настоящие pacemaking клетки (JO, LC) в FAFB пока не замаплены —
    используем random subset, чтобы увидеть эффект.
    """
    n = len(nt)
    n_channels = 3

    g_max = np.zeros(n_channels * n, dtype=np.float32)
    C_m = np.ones(n, dtype=np.float32)
    g_leak = np.ones(n, dtype=np.float32) * 0.3
    E_leak = np.ones(n, dtype=np.float32) * -65.0

    for i in range(n):
        if nt[i] > 0:
            g_max[0 * n + i] = 120.0   # NaV
            g_max[1 * n + i] = 36.0    # KV
            C_m[i] = 1.0
            g_leak[i] = 0.3
        else:
            g_max[0 * n + i] = 120.0   # NaV
            g_max[1 * n + i] = 50.0    # KV — быстрая реполяризация
            C_m[i] = 0.6               # низкая ёмкость → fast-spiking
            g_leak[i] = 0.5

    # CaV-T на случайные нейроны
    n_cav = max(1, int(n * cav_pct / 100))
    rng = np.random.default_rng(seed)
    cav_idx = rng.choice(n, size=n_cav, replace=False)
    g_max[2 * n + cav_idx] = 0.5  # T-type Ca²⁺ conductance

    print(f"  CaV-T on {n_cav:,} / {n:,} ({cav_pct:.1f}%) neurons")
    return {"g_max": g_max, "C_m": C_m, "g_leak": g_leak, "E_leak": E_leak}, nt


def simulate(sim, conn, n_steps, dt_ms):
    """Run simulation and return spike buffer [n_steps, n_neurons]."""
    t0 = time.time()
    buf = sim.step_batch(conn, n_steps, dt_ms=dt_ms, scale=1e10)
    elapsed = time.time() - t0
    sb = buf.reshape(n_steps, conn.n_neurons).astype(np.int32)
    print(f"  Sim {n_steps} steps × {conn.n_neurons:,}N in {elapsed:.2f}s ({n_steps/elapsed:.0f} st/s)")
    return sb


def analyze(sb, nt, dt_ms):
    """Compute population metrics: rates, spectra, E/I balance."""
    n_steps, n = sb.shape
    dt = dt_ms / 1000.0  # seconds
    duration = n_steps * dt
    time_sec = np.arange(n_steps) * dt

    # Population rate (spikes/step → Hz)
    pop_rate = sb.sum(axis=1).astype(np.float32) / dt  # Hz

    # E and I population rates
    exc_mask = nt > 0
    inh_mask = nt < 0
    pop_e = sb[:, exc_mask].sum(axis=1).astype(np.float32) / dt if exc_mask.sum() > 0 else np.zeros(n_steps)
    pop_i = sb[:, inh_mask].sum(axis=1).astype(np.float32) / dt if inh_mask.sum() > 0 else np.zeros(n_steps)

    # Per-neuron firing rates
    fr = sb.sum(axis=0) / duration

    # Active neurons
    active = (sb.sum(axis=0) > 0).sum()
    active_e = (sb[:, exc_mask].sum(axis=0) > 0).sum() if exc_mask.sum() > 0 else 0
    active_i = (sb[:, inh_mask].sum(axis=0) > 0).sum() if inh_mask.sum() > 0 else 0

    # Power spectrum (Welch)
    fs = 1.0 / dt  # sampling frequency in Hz
    f, psd = sg.welch(pop_rate, fs=fs, nperseg=min(256, n_steps // 4))
    _, psd_e = sg.welch(pop_e, fs=fs, nperseg=min(256, n_steps // 4))
    _, psd_i = sg.welch(pop_i, fs=fs, nperseg=min(256, n_steps // 4))

    # Spectrogram
    nperseg = min(128, n_steps // 8)
    noverlap = nperseg // 2
    f_spec, t_spec, Sxx = sg.spectrogram(pop_rate, fs=fs, nperseg=nperseg, noverlap=noverlap)

    return {
        "time_sec": time_sec,
        "pop_rate": pop_rate,
        "pop_e": pop_e,
        "pop_i": pop_i,
        "fr": fr,
        "exc_mask": exc_mask,
        "inh_mask": inh_mask,
        "active": active,
        "active_e": active_e,
        "active_i": active_i,
        "f": f, "psd": psd,
        "f_e": f, "psd_e": psd_e,
        "f_i": f, "psd_i": psd_i,
        "f_spec": f_spec, "t_spec": t_spec, "Sxx": Sxx,
        "dt": dt, "duration": duration,
        "total_spikes": int(sb.sum()),
    }


def plot_raster(sb, nt, ana, out_path: str):
    """Raster plot — first 100ms (200 steps)."""
    n_steps, n = sb.shape
    max_steps = min(200, n_steps)
    dt = ana["dt"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort neurons by E/I, then by firing rate
    fr = ana["fr"]
    order = np.argsort(
        nt.astype(np.int32) * 1_000_000 - fr.astype(np.int32)
    )

    # Thin raster — plot every Nth spike for performance
    times, ids = [], []
    for s in range(max_steps):
        for idx in np.where(sb[s] > 0)[0]:
            times.append(s * dt * 1000)  # ms
            ids.append(int(idx))

    step_sz = max(1, len(times) // 100_000)
    ax.scatter(times[::step_sz], ids[::step_sz], s=0.02, c="black", alpha=0.3, rasterized=True)

    # E/I boundary line
    n_e = ana["exc_mask"].sum()
    ax.axhline(n_e, color="red", linestyle="--", alpha=0.5, linewidth=0.5)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index (sorted: E below, I above)")
    ax.set_title(f"FAFB Raster (first {max_steps * dt * 1000:.0f} ms)  —  red = E/I boundary")
    ax.set_xlim(0, max_steps * dt * 1000)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → {out_path}")


def plot_lfp(ana, out_path: str):
    """Population rate + E/I contributions + power spectrum."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex="col")

    time_ms = ana["time_sec"] * 1000

    # ── Panel A: Population rate ──
    ax = axes[0]
    ax.plot(time_ms, ana["pop_rate"], "k-", linewidth=0.5, alpha=0.7, label="Total")
    ax.plot(time_ms, ana["pop_e"], "b-", linewidth=0.5, alpha=0.5, label="E only")
    ax.plot(time_ms, ana["pop_i"], "r-", linewidth=0.5, alpha=0.5, label="I only")
    ax.set_ylabel("Population rate (Hz)")
    ax.legend(fontsize=8)
    ax.set_title("FAFB EEG: Population Firing Rate  (CaV pacemaking, no external input)")

    # ── Panel B: E/I ratio over time ──
    ax = axes[1]
    ei_ratio = np.where(ana["pop_i"] > 0, ana["pop_e"] / ana["pop_i"], 0)
    ax.plot(time_ms, ei_ratio, "g-", linewidth=0.7)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="E=I")
    ax.set_ylabel("E/I ratio")
    ax.set_ylim(0, min(5, ei_ratio.max() * 1.2 + 0.1))
    ax.legend(fontsize=8)

    # ── Panel C: Power spectrum ──
    ax = axes[2]
    ax.loglog(ana["f"], ana["psd"], "k-", linewidth=1.5, label="Total")
    ax.loglog(ana["f_e"], ana["psd_e"], "b-", alpha=0.7, label="E")
    ax.loglog(ana["f_i"], ana["psd_i"], "r-", alpha=0.7, label="I")

    # Find peaks
    for label, f, psd in [
        ("Total", ana["f"], ana["psd"]),
        ("E", ana["f_e"], ana["psd_e"]),
        ("I", ana["f_i"], ana["psd_i"]),
    ]:
        pk_idx = np.argmax(psd[5:]) + 5  # skip DC
        f_pk = f[pk_idx]
        ax.axvline(f_pk, color=("k" if label == "Total" else "b" if label == "E" else "r"),
                    linestyle="--", alpha=0.3)
        ax.text(f_pk * 1.1, psd[pk_idx] * 1.1, f"{label} {f_pk:.0f} Hz", fontsize=7)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (Hz²/Hz)")
    ax.set_xlim(1, 500)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → {out_path}")


def plot_spectrogram(ana, out_path: str):
    """Spectrogram of population rate."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # ── Panel A: Spectrogram ──
    ax = axes[0]
    t_spec_ms = ana["t_spec"] * 1000
    pcm = ax.pcolormesh(t_spec_ms, ana["f_spec"], 10 * np.log10(ana["Sxx"] + 1e-12),
                         shading="gouraud", cmap="inferno")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, 200)
    ax.set_title("FAFB EEG Spectrogram")
    plt.colorbar(pcm, ax=ax, label="dB")

    # ── Panel B: Mean spectrum + gamma band ──
    ax = axes[1]
    f = ana["f"]
    psd = ana["psd"]
    ax.semilogy(f, psd, "k-", linewidth=1)
    ax.axvspan(30, 80, color="g", alpha=0.1, label="Gamma (30-80 Hz)")
    ax.axvspan(4, 8, color="b", alpha=0.1, label="Theta (4-8 Hz)")
    ax.axvspan(8, 12, color="r", alpha=0.1, label="Alpha (8-12 Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (Hz²/Hz)")
    ax.set_xlim(0, 200)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="EEG of panicking FAFB brain")
    parser.add_argument("--taas", default="fafb.taas", help="Path to .taas file")
    parser.add_argument("--steps", type=int, default=2000, help="Simulation steps")
    parser.add_argument("--dt", type=float, default=0.5, help="Timestep (ms)")
    parser.add_argument("--cav", type=float, default=5.0, help="CaV-T percent of neurons")
    parser.add_argument("--out", default="fafb_eeg", help="Output prefix")
    args = parser.parse_args()

    taas_path = DATA / args.taas
    if not taas_path.exists():
        print(f"❌ {taas_path} not found")
        sys.exit(1)

    out_p = Path(args.out)
    n_steps, dt_ms = args.steps, args.dt
    duration = n_steps * dt_ms / 1000.0

    print("╔═══════════════════════════════════════════════════════╗")
    print("║  FAFB EEG — Fly Without Eyes, Without Body          ║")
    print("║  Just a brain, panicking in silicon                  ║")
    print(f"║  {taas_path.name:<44s} ║")
    print(f"║  {n_steps} steps @ {dt_ms}ms = {duration:.1f}s              ║")
    print(f"║  CaV-T: {args.cav}% of neurons                     ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()

    # ── Load ──
    print("Loading FAFB connectome ...")
    conn, nt = load_fafb(taas_path)
    n = conn.n_neurons

    # ── Bio params ──
    print("\nBuilding per-neuron 3-channel HH params ...")
    bio, nt = make_cav_params(nt, args.cav)
    n_e = int((nt > 0).sum())
    n_i = int((nt < 0).sum())
    print(f"  {n:,} neurons: {n_e:,}E + {n_i:,}I (excl. {nt[0]:.0f})")

    # ── Create sim ──
    print("\nCreating 3-channel HH simulator ...")
    sim = tcm.BrainSimulator(n, "hh", pacemaking=True)
    sim.set_hh_params(
        bio["g_max"],
        np.array([50.0, -77.0, 120.0], dtype=np.float32),  # Na, K, Ca reversal
        bio["g_leak"],
        bio["E_leak"],
        dt_ms=dt_ms,
    )
    sim.set_C_m(bio["C_m"])

    # ── Simulate ──
    print(f"\nRecording EEG ({n_steps} steps, no external input)...")
    sb = simulate(sim, conn, n_steps, dt_ms)
    del sim

    # ── Analyze ──
    print("\nAnalyzing ...")
    ana = analyze(sb, nt, dt_ms)

    total = ana["total_spikes"]
    print(f"  Total spikes: {total:,}")
    print(f"  Active neurons: {ana['active']:,}/{n:,}  "
          f"({ana['active_e']:,}E + {ana['active_i']:,}I)")
    print(f"  Mean pop rate: {ana['pop_rate'].mean():.1f} Hz  "
          f"(E: {ana['pop_e'].mean():.1f} Hz, I: {ana['pop_i'].mean():.1f} Hz)")
    print(f"  Mean E/I ratio: {ana['pop_e'].mean() / max(ana['pop_i'].mean(), 1):.2f}")

    # Peak frequency
    pk_idx = np.argmax(ana["psd"][5:]) + 5
    print(f"  Peak frequency: {ana['f'][pk_idx]:.1f} Hz")

    # ── Plot ──
    print("\nPlotting ...")
    plot_raster(sb, nt, ana, f"{out_p}_raster.png")
    plot_lfp(ana, f"{out_p}_lfp.png")
    plot_spectrogram(ana, f"{out_p}_spectrum.png")

    # ── Summary ──
    print(f"\n{'=' * 55}")
    print(f"  EEG Summary: FAFB panicking with {args.cav}% CaV-T")
    print(f"{'=' * 55}")
    print(f"  Duration:       {duration:.1f} s")
    print(f"  Total spikes:   {total:,}")
    print(f"  Active:         {ana['active']:,} / {n:,} ({ana['active']/n*100:.1f}%)")
    print(f"  E rate:         {ana['pop_e'].mean():.1f} Hz")
    print(f"  I rate:         {ana['pop_i'].mean():.1f} Hz")
    print(f"  E/I ratio:      {ana['pop_e'].mean() / max(ana['pop_i'].mean(), 1):.2f}")
    print(f"  Peak freq:      {ana['f'][pk_idx]:.1f} Hz")
    print(f"  Output:         {out_p}_raster.png, {out_p}_lfp.png, {out_p}_spectrum.png")
    print(f"\n  Diagnosis: fly without eyes, without body —")
    print(f"  just a brain, panicking in silicon.")
    print(f"✅ Done.")


if __name__ == "__main__":
    main()
