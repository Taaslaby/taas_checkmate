#!/usr/bin/env python3
"""Visualization tools for taas CLI — Neuralink brand style.

Все графики используют фирменный стиль из taas.style:
- Тёмная тема (#0a0a12) с неоновыми акцентами
- Glow-эффекты на линиях и scatter
- Водяной знак логотипа
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CLI
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
from rich.console import Console
from rich.table import Table

from .style import (
    NEON_GREEN, NEON_CYAN, NEON_PURPLE, NEON_PINK,
    NEON_ORANGE, NEON_BLUE,
    GLOW_GREEN, GLOW_CYAN, GLOW_PURPLE, GLOW_ORANGE,
    GLOW_BLUE, GLOW_PINK,
    BRAND_DARK, BRAND_DARK2, BRAND_DARK3, BRAND_WHITE,
    apply_style, add_logo, glow_line, glow_scatter, style_axis,
    get_brand_cmap,
)

console = Console()

try:
    import taas_checkmate as ne
except ImportError:
    ne = None

# Apply Neuralink style globally
apply_style()


def _show_raster(spike_buffer: np.ndarray, dt: float = 0.5,
                 title: str = "Spike Raster", window: int | None = None):
    """Neuralink-style spike raster plot."""
    steps, n = spike_buffer.shape
    if window:
        steps = min(steps, window)
        spike_buffer = spike_buffer[:steps]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Raster with glow
    spike_times, spike_neurons = np.where(spike_buffer > 0)
    glow_scatter(ax, spike_times * dt, spike_neurons,
                 color=NEON_GREEN, s=0.8, alpha=0.5, glow_alpha=0.08)

    style_axis(ax, title=title, xlabel="Time (ms)", ylabel="Neuron")
    ax.set_xlim(0, steps * dt)
    ax.set_ylim(0, n)

    # Firing rate with glow overlay
    firing_rates = spike_buffer.sum(axis=1) / n * 1000 / dt
    ax2 = ax.twinx()
    glow_line(ax2, np.arange(steps) * dt, firing_rates,
              color=NEON_ORANGE, linewidth=1.0, alpha=0.6,
              glow_alpha_range=(0.03, 0.1))
    ax2.set_ylabel("Firing Rate (Hz)", color=NEON_ORANGE)
    ax2.tick_params(colors=NEON_ORANGE)

    add_logo(fig, alpha=0.12, scale=0.1, position="lower right")
    plt.tight_layout()
    plt.show()


def plot_raster(spikes_file: str, window: int | None = None):
    """Plot spike raster from .npy file."""
    spikes = np.load(spikes_file)
    console.print(f"   Loaded spikes: {spikes.shape[0]} steps × {spikes.shape[1]} neurons")
    _show_raster(spikes, title=f"Spike Raster — {spikes_file}", window=window)


def plot_coherence(phases_file: str, n_areas: int = 8):
    """Compute and plot phase coherence map."""
    phases = np.load(phases_file)
    steps, n = phases.shape
    console.print(f"   Computing coherence sweep: {n_areas} areas × {steps} steps")

    sin_buf = np.zeros(n, dtype=np.float32)
    cos_buf = np.zeros(n, dtype=np.float32)
    output = np.zeros(n_areas, dtype=np.float32)

    coherence_map = np.zeros((n_areas, steps), dtype=np.float32)
    for t in range(steps):
        ne.phase_coherence(output, sin_buf, cos_buf, phases[t])
        if t > 0:
            coherence_map[:, t] = output / n_areas

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Coherence map with magma colormap
    cmap = get_brand_cmap("coherence")
    im = ax1.imshow(coherence_map, aspect="auto", cmap=cmap,
                    extent=[0, steps, 0, n_areas],
                    interpolation="bilinear")
    style_axis(ax1, title="Phase Coherence Map", xlabel="Time (steps)", ylabel="Area")
    plt.colorbar(im, ax=ax1, label="Coherence", shrink=0.8)

    # Average coherence barh with glow
    mean_coherence = coherence_map.mean(axis=1)
    colors = [NEON_GREEN if c > 0.5 else NEON_CYAN for c in mean_coherence]
    bars = ax2.barh(range(n_areas), mean_coherence, color=colors, alpha=0.7,
                    height=0.6, edgecolor="none")
    style_axis(ax2, title="Average Coherence per Area",
               xlabel="Mean Coherence", ylabel="Area", grid=True)
    ax2.set_xlim(0, 1.05)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, mean_coherence)):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color=BRAND_WHITE)

    add_logo(fig, alpha=0.12, scale=0.1, position="lower right")
    plt.tight_layout()
    plt.show()


# ─── T3 → EEG Visualizer ────────────────────────────────


def run_eeg_viz(n: int, steps: int, dt: float,
                save: str | None, plot: bool):
    """Generate EEG-like signal from T3 Triple Oscillator.

    Uses 3 Kuramoto populations with different frequency bands:
    - Pop 1: 2-8 Hz (theta/alpha)
    - Pop 2: 8-30 Hz (beta)
    - Pop 3: 30-80 Hz (gamma)

    EEG = weighted sum of LFP proxies from each population.
    """
    from rich.progress import Progress

    n_pad = (n + 7) // 8 * 8
    console.print(f"[bold]🧠 T3 → EEG Generator[/bold]")
    console.print(f"   {n} oscillators × 3 populations = {3*n} channels")
    console.print(f"   Frequency bands: theta/alpha (2-8Hz), beta (8-30Hz), gamma (30-80Hz)")
    console.print(f"   {steps} steps × {dt}ms = {steps * dt}ms total\n")

    tri = ne.PyTriOscillator(n_pad, freq_min=2.0, freq_max=80.0)
    tri.set_uniform_coupling(0.3, 0.2, 0.1)

    phase_out = np.zeros(n_pad, dtype=np.float32)
    amp_out = np.zeros(n_pad, dtype=np.float32)

    eeg_signal = np.zeros(steps, dtype=np.float32)
    eeg_theta = np.zeros(steps, dtype=np.float32)
    eeg_beta = np.zeros(steps, dtype=np.float32)
    eeg_gamma = np.zeros(steps, dtype=np.float32)

    rng = np.random.default_rng(42)
    w1 = rng.uniform(-1, 1, min(n, 64)).astype(np.float32)
    w2 = rng.uniform(-1, 1, min(n, 64)).astype(np.float32)
    w3 = rng.uniform(-1, 1, min(n, 64)).astype(np.float32)

    with Progress() as progress:
        task = progress.add_task("[cyan]Generating EEG...", total=steps)
        for t in range(steps):
            tri.step(dt)
            tri.copy_phases(phase_out, 1)
            sin1 = np.sin(phase_out[:min(n, 64)])
            tri.copy_amplitudes(amp_out, 1)
            r1 = amp_out[:min(n, 64)]

            tri.copy_phases(phase_out, 2)
            sin2 = np.sin(phase_out[:min(n, 64)])
            tri.copy_amplitudes(amp_out, 2)
            r2 = amp_out[:min(n, 64)]

            tri.copy_phases(phase_out, 3)
            sin3 = np.sin(phase_out[:min(n, 64)])
            tri.copy_amplitudes(amp_out, 3)
            r3 = amp_out[:min(n, 64)]

            eeg_theta[t] = float(np.sum(r1 * sin1 * w1)) / n
            eeg_beta[t] = float(np.sum(r2 * sin2 * w2)) / n
            eeg_gamma[t] = float(np.sum(r3 * sin3 * w3)) / n
            eeg_signal[t] = eeg_theta[t] + eeg_beta[t] + eeg_gamma[t]
            progress.update(task, advance=1)

    if save:
        np.save(save, eeg_signal)
        console.print(f"   [green]✓[/green] Saved EEG to {save}")

    if plot:
        _show_eeg_plot(eeg_signal, eeg_theta, eeg_beta, eeg_gamma, dt, steps, n)


def _show_eeg_plot(eeg: np.ndarray, theta: np.ndarray,
                   beta: np.ndarray, gamma: np.ndarray,
                   dt: float, steps: int, n: int):
    """Neuralink-style EEG plot with glow lines."""
    time_axis = np.arange(steps) * dt / 1000  # seconds

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # 1. Raw EEG — green glow
    glow_line(axes[0], time_axis, eeg, color=NEON_GREEN,
              linewidth=0.8, alpha=0.8, glow_alpha_range=(0.03, 0.12))
    style_axis(axes[0], title="T3-Derived EEG Signal",
               ylabel="EEG (μV)", grid=True)
    axes[0].tick_params(colors=NEON_GREEN)

    # 2. Theta/Alpha — orange fill + glow
    axes[1].fill_between(time_axis, theta, alpha=0.12, color=NEON_ORANGE)
    glow_line(axes[1], time_axis, theta, color=NEON_ORANGE,
              linewidth=0.7, alpha=0.7, glow_alpha_range=(0.03, 0.1))
    style_axis(axes[1], ylabel="θ/α (2-8 Hz)", grid=True)
    axes[1].tick_params(colors=NEON_ORANGE)

    # 3. Beta — blue fill + glow
    axes[2].fill_between(time_axis, beta, alpha=0.12, color=NEON_BLUE)
    glow_line(axes[2], time_axis, beta, color=NEON_BLUE,
              linewidth=0.7, alpha=0.7, glow_alpha_range=(0.03, 0.1))
    style_axis(axes[2], ylabel="β (8-30 Hz)", grid=True)
    axes[2].tick_params(colors=NEON_BLUE)

    # 4. Gamma — pink fill + glow
    axes[3].fill_between(time_axis, gamma, alpha=0.12, color=NEON_PINK)
    glow_line(axes[3], time_axis, gamma, color=NEON_PINK,
              linewidth=0.7, alpha=0.7, glow_alpha_range=(0.03, 0.1))
    style_axis(axes[3], ylabel="γ (30-80 Hz)", xlabel="Time (s)", grid=True)
    axes[3].tick_params(colors=NEON_PINK)

    fig.suptitle(f"T3 → EEG: {3*n} oscillators, {steps*dt:.0f}ms simulation",
                 fontsize=13, fontweight="bold", y=1.01)

    add_logo(fig, alpha=0.12, scale=0.1, position="lower right")
    plt.tight_layout()
    plt.show()

    _show_eeg_spectrum(eeg, dt)


def _show_eeg_spectrum(eeg: np.ndarray, dt: float):
    """Neuralink-style EEG power spectrum."""
    try:
        from scipy import signal as sp_signal
    except ImportError:
        console.print("[yellow]⚠ scipy not installed — skipping spectrum plot[/yellow]")
        console.print("  Install: pip install scipy")
        return

    fs = 1000.0 / dt
    freqs, psd = sp_signal.welch(eeg, fs=fs, nperseg=min(256, len(eeg) // 4))

    fig, ax = plt.subplots(figsize=(12, 4))

    # Spectrum with glow
    glow_line(ax, freqs, psd, color=NEON_GREEN, linewidth=1.0, alpha=0.8,
              glow_alpha_range=(0.03, 0.1))

    # Frequency band markers
    band_colors = [NEON_ORANGE, NEON_BLUE, NEON_PINK]
    band_labels = ["θ/α (2-8 Hz)", "β (8-30 Hz)", "γ (30-80 Hz)"]
    bands = [(2, 8), (8, 30), (30, 80)]

    for (lo, hi), color, label in zip(bands, band_colors, band_labels):
        ax.axvspan(lo, hi, alpha=0.08, color=color, label=label)
        ax.axvline(lo, color=color, alpha=0.2, linewidth=0.3, linestyle="--")
        ax.axvline(hi, color=color, alpha=0.2, linewidth=0.3, linestyle="--")

    style_axis(ax, title="EEG Power Spectrum",
               xlabel="Frequency (Hz)", ylabel="Power Spectral Density")
    ax.set_xlim(0, 100)
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper right")

    add_logo(fig, alpha=0.12, scale=0.1, position="lower right")
    plt.tight_layout()
    plt.show()


# ─── Connectome Visualizer ──────────────────────────────


def run_connectome_viz(n: int, sparsity: float, spectral_radius: float,
                       steps: int, save: str | None):
    """Neuralink-style connectome visualization."""
    try:
        from scipy.sparse import random as sparse_random
        from scipy.sparse.linalg import eigs
    except ImportError:
        console.print("[red]✗ scipy required for connectome visualizer[/red]")
        console.print("  Install: pip install scipy")
        return

    console.print(f"[bold]🕸️ Connectome Visualizer[/bold]")
    console.print(f"   {n} neurons, sparsity={sparsity}, spectral radius={spectral_radius}")
    console.print(f"   {steps} animation steps\n")

    rng = np.random.default_rng(42)

    W = sparse_random(n, n, density=sparsity,
                      data_rvs=lambda s: rng.uniform(-1, 1, s),
                      random_state=rng).tocsr()

    try:
        eig_max = abs(eigs(W, k=1, which="LM", return_eigenvectors=False))[0]
        if eig_max > 1e-10:
            W = W * (spectral_radius / eig_max)
    except Exception:
        W = W * spectral_radius

    W_coo = W.tocoo()
    edges_i = W_coo.row[:5000]
    edges_j = W_coo.col[:5000]
    edges_w = W_coo.data[:5000]

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_coords = np.cos(angles)
    y_coords = np.sin(angles)

    phases = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
    omegas = rng.uniform(0.5, 2.0, n).astype(np.float32)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BRAND_DARK)

    # Neuralink-style edges — thin, semi-transparent, cyan
    for ei, ej, ew in zip(edges_i, edges_j, edges_w):
        if ei < len(x_coords) and ej < len(y_coords):
            ax.plot([x_coords[ei], x_coords[ej]],
                    [y_coords[ei], y_coords[ej]],
                    color=NEON_CYAN, alpha=min(abs(ew) * 3, 0.25),
                    linewidth=0.2 + abs(ew) * 0.5)

    # Phase-colored nodes with glow
    scatter = ax.scatter(x_coords, y_coords, c=np.cos(phases),
                         cmap=get_brand_cmap("phase"), s=25,
                         alpha=0.85, vmin=-1, vmax=1, edgecolors="none")

    plt.colorbar(scatter, ax=ax, label="Phase", shrink=0.7,
                 ticks=[-1, 0, 1])

    ax.set_title("Neural Connectome — Phase Dynamics",
                 fontsize=14, fontweight="bold", color=BRAND_WHITE)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BRAND_DARK)

    # Logo
    add_logo(fig, alpha=0.15, scale=0.12, position="lower left")

    # Animation
    def update(frame):
        nonlocal phases
        coupling = np.zeros(n, dtype=np.float32)
        for idx in range(min(len(edges_i), 2000)):
            i, j = edges_i[idx], edges_j[idx]
            if i < n and j < n:
                coupling[i] += edges_w[idx] * np.sin(phases[j] - phases[i])
        phases += 0.1 * (omegas + coupling)
        scatter.set_array(np.cos(phases))
        ax.set_title(f"Connectome — Step {frame + 1}/{steps}",
                     fontsize=14, fontweight="bold", color=BRAND_WHITE)
        return scatter,

    ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=True)

    if save:
        ani.save(save, writer="pillow", fps=20, dpi=120)
        console.print(f"   [green]✓[/green] Saved animation to {save}")

    plt.show()
    console.print(f"   [green]✓[/green] Connectome visualization complete")
