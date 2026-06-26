#!/usr/bin/env python3
"""
Spike raster plot — визуализация спайковой активности из CSV.

Использование:
    # Из CSV файла (экспортированного spike_raster_to_csv)
    python3 examples/python/plot_raster.py /tmp/nmist_spike_raster.csv

    # Или сгенерировать из Izhikevich теста + сразу показать
    python3 examples/python/plot_raster.py --demo

Формат CSV:
    time_ms,neuron_id
    10.5,3
    11.0,7
    ...

Зависимости:
    pip install matplotlib numpy
"""

import sys
import os
import csv
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np

# Try import — allow running from project root or examples dir
HERE = Path(__file__).parent  # examples/python/
PROJ = HERE.parent.parent      # project root
if str(PROJ / 'target' / 'debug') not in sys.path:
    sys.path.insert(0, str(PROJ / 'target' / 'debug'))

# Check matplotlib availability early
try:
    import matplotlib
except ImportError:
    print("✗ matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════
#  Core plotting function
# ═══════════════════════════════════════════════════════════

def plot_spike_raster(
    csv_path: str,
    title: str = "Spike Raster",
    max_neurons: int | None = None,
    figsize: tuple = (12, 6),
    spike_color: str = "#2196F3",
    spike_alpha: float = 0.6,
    spike_size: float = 2.0,
    show_rate: bool = True,
    show_isi_hist: bool = False,
    save_path: str | None = None,
) -> None:
    """Plot spike raster from CSV file.

    Args:
        csv_path: Path to CSV file with columns 'time_ms,neuron_id'.
        title: Plot title.
        max_neurons: Only show first N neurons (None = all).
        figsize: Figure size (width, height).
        spike_color: Marker color.
        spike_alpha: Marker transparency.
        spike_size: Marker size (points).
        show_rate: Show per-neuron firing rate on right y-axis.
        show_isi_hist: Show ISI histogram as inset.
        save_path: Save figure to file (None → show interactively).
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # ── Parse CSV ────────────────────────────────────────
    times: list[float] = []
    neurons: list[int] = []
    neuron_times: dict[int, list[float]] = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        if 'time_ms' not in reader.fieldnames or 'neuron_id' not in reader.fieldnames:
            print(f"✗ Invalid CSV: expected columns 'time_ms,neuron_id', got {reader.fieldnames}")
            return

        for row in reader:
            t = float(row['time_ms'])
            n = int(row['neuron_id'])
            times.append(t)
            neurons.append(n)
            neuron_times[n].append(t)

    if not times:
        print("  No spikes to plot (empty raster).")
        return

    # Filter neurons
    all_neurons = sorted(neuron_times.keys())
    if max_neurons is not None and len(all_neurons) > max_neurons:
        keep = set(all_neurons[:max_neurons])
        times = [t for t, n in zip(times, neurons) if n in keep]
        neurons = [n for n in neurons if n in keep]
        neuron_times = {n: ts for n, ts in neuron_times.items() if n in keep}
        all_neurons = sorted(neuron_times.keys())

    t_min = min(times)
    t_max = max(times)
    duration = t_max - t_min
    n_neurons = len(all_neurons)

    if n_neurons == 0:
        print("  No neurons to plot.")
        return

    # ── Create figure ────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # ── Raster plot (main) ───────────────────────────────
    y_positions: list[float] = []
    x_positions: list[float] = []

    for i, nid in enumerate(all_neurons):
        for t in neuron_times[nid]:
            x_positions.append(t)
            y_positions.append(i)

    ax1.scatter(x_positions, y_positions,
                c=spike_color, alpha=spike_alpha, s=spike_size,
                edgecolors='none', rasterized=True)

    ax1.set_xlabel('Time (ms)', fontsize=12)
    ax1.set_ylabel('Neuron ID', fontsize=12)
    ax1.set_xlim(t_min - duration * 0.02, t_max + duration * 0.02)
    ax1.set_ylim(-0.5, n_neurons - 0.5)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(n_neurons, 20)))

    # ── Stats text ───────────────────────────────────────
    total_spikes = len(times)
    n_ids = len(set(neurons))
    mean_rate = total_spikes / max(duration / 1000.0, 0.001) / max(n_ids, 1)
    stats_text = f"Spikes: {total_spikes} | Neurons: {n_ids} | Mean rate: {mean_rate:.0f} Hz"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    # ── Firing rate bar (right side) ─────────────────────
    if show_rate and n_neurons <= 100:
        ax2 = ax1.twinx()
        duration_sec = duration / 1000.0
        rates = [len(neuron_times[nid]) / max(duration_sec, 0.001) for nid in all_neurons]
        ax2.barh(range(n_neurons), rates, height=0.8,
                 color='#FF5722', alpha=0.3, edgecolor='none')
        ax2.set_xlabel('Firing rate (Hz)', fontsize=10, color='#FF5722')
        ax2.tick_params(axis='x', labelcolor='#FF5722')
        ax2.set_ylim(-0.5, n_neurons - 0.5)

    # ── ISI histogram inset ──────────────────────────────
    if show_isi_hist and n_neurons <= 50:
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        # Compute all ISIs
        all_isi: list[float] = []
        for nid in all_neurons:
            ts = sorted(neuron_times[nid])
            for i in range(1, len(ts)):
                isi = ts[i] - ts[i - 1]
                if isi > 0:
                    all_isi.append(isi)

        if all_isi:
            ax_inset = fig.add_axes([0.78, 0.55, 0.2, 0.25])
            ax_inset.hist(all_isi, bins=30, color='#4CAF50', alpha=0.7, edgecolor='white')
            ax_inset.set_xlabel('ISI (ms)', fontsize=8)
            ax_inset.set_ylabel('Count', fontsize=8)
            ax_inset.tick_params(labelsize=7)
            mean_isi = np.mean(all_isi)
            ax_inset.axvline(mean_isi, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax_inset.text(0.95, 0.95, f'μ={mean_isi:.1f}ms',
                         transform=ax_inset.transAxes, fontsize=7,
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(facecolor='white', alpha=0.7, pad=1))

    # ── Layout & save ────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Raster saved to {save_path}")
    else:
        plt.show()

    plt.close()


# ═══════════════════════════════════════════════════════════
#  Demo: generate Izhikevich data + plot
# ═══════════════════════════════════════════════════════════

def run_demo() -> str:
    """Generate demo spike raster from Rust Izhikevich via PyO3."""
    import numpy as np

    try:
        import taas_checkmate as tcm
    except ImportError:
        print("✗ Cannot import taas_checkmate. Build first:")
        print("  cd taas_checkmate && cargo build --features python")
        sys.exit(1)

    print("  Generating Izhikevich spike raster demo...")

    n_neurons = 64
    n_steps = 200
    dt = 1.0

    # --- 4 Izhikevich patterns, 16 neurons each ---
    patterns = [
        ("RS", 0.02, 0.2, -65.0, 8.0, 10.0),   # Regular Spiking
        ("FS", 0.1,  0.2, -65.0, 2.0, 10.0),   # Fast Spiking
        ("IB", 0.02, 0.2, -55.0, 4.0, 10.0),   # Bursting
        ("CH", 0.02, 0.2, -50.0, 2.0, 10.0),   # Chattering
    ]

    n_padded = (n_neurons + 7) // 8 * 8
    v = np.full(n_padded, -65.0, dtype=np.float32)
    u = np.full(n_padded, -13.0, dtype=np.float32)
    trace = np.zeros(n_padded, dtype=np.float32)
    a_arr = np.zeros(n_padded, dtype=np.float32)
    b_arr = np.zeros(n_padded, dtype=np.float32)
    c_arr = np.zeros(n_padded, dtype=np.float32)
    d_arr = np.zeros(n_padded, dtype=np.float32)
    I_arr = np.zeros(n_padded, dtype=np.float32)

    for offset, (name, a, b, c, d, I) in enumerate(patterns):
        for j in range(16):
            idx = offset * 16 + j
            if idx < n_padded:
                a_arr[idx] = a
                b_arr[idx] = b
                c_arr[idx] = c
                d_arr[idx] = d
                I_arr[idx] = I

    spikes_out = np.zeros(n_padded, dtype=np.int32)
    trace_decay = 0.5

    # Simulate
    spike_buf = np.zeros(n_steps * n_neurons, dtype=np.int32)

    # Note: Python binding takes 11 args (refr/tau_ref handled internally via thread_local!)
    for step in range(n_steps):
        tcm.neurons.step_izhikevich(
            v, u, trace, a_arr, b_arr, I_arr,
            dt, trace_decay, spikes_out, c_arr, d_arr,
        )
        offset = step * n_neurons
        spike_buf[offset:offset + n_neurons] = spikes_out[:n_neurons]

    # Export as CSV
    try:
        csv_data = tcm.spike_raster_to_csv(spike_buf, n_neurons, dt)
    except AttributeError:
        # Fallback: manual CSV
        lines = ["time_ms,neuron_id"]
        for t in range(n_steps):
            for n in range(n_neurons):
                if spike_buf[t * n_neurons + n]:
                    lines.append(f"{t * dt:.1f},{n}")
        csv_data = "\n".join(lines)

    csv_path = os.path.join(tempfile.gettempdir(), "izhikevich_raster_demo.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_data)
    print(f"  ✓ Demo CSV saved to {csv_path}")
    return csv_path


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Spike raster plot — визуализация спайковой активности",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python3 examples/python/plot_raster.py /tmp/spikes.csv
  python3 examples/python/plot_raster.py /tmp/spikes.csv --show-rate --show-isi
  python3 examples/python/plot_raster.py /tmp/spikes.csv --save /tmp/raster.png
  python3 examples/python/plot_raster.py --demo
  python3 examples/python/plot_raster.py --demo --show-isi --save /tmp/demo_raster.png
        """,
    )
    parser.add_argument('csv', nargs='?', help='Path to spike raster CSV')
    parser.add_argument('--demo', action='store_true', help='Generate demo Izhikevich raster')
    parser.add_argument('--title', default='Spike Raster', help='Plot title')
    parser.add_argument('--max-neurons', type=int, default=None, help='Show first N neurons')
    parser.add_argument('--show-rate', action='store_true', help='Show firing rate bars')
    parser.add_argument('--show-isi', action='store_true', help='Show ISI histogram')
    parser.add_argument('--save', default=None, help='Save figure to file')
    parser.add_argument('--figsize', default='12,6', help='Figure size (width,height)')
    parser.add_argument('--color', default='#2196F3', help='Spike marker color')
    parser.add_argument('--alpha', type=float, default=0.6, help='Marker transparency')
    parser.add_argument('--size', type=float, default=2.0, help='Marker size (points)')

    args = parser.parse_args()

    # --- Validate args ---
    if args.demo:
        csv_path = run_demo()
    elif args.csv:
        csv_path = args.csv
        if not os.path.exists(csv_path):
            print(f"✗ File not found: {csv_path}")
            sys.exit(1)
    else:
        parser.print_help()
        print("\n  Need either a CSV path or --demo. Try: python3 examples/python/plot_raster.py --demo")
        sys.exit(1)

    # --- Parse figsize ---
    try:
        figsize = tuple(float(x) for x in args.figsize.split(','))
    except ValueError:
        figsize = (12, 6)

    # --- Plot ---
    plot_spike_raster(
        csv_path=csv_path,
        title=args.title,
        max_neurons=args.max_neurons,
        figsize=figsize,
        spike_color=args.color,
        spike_alpha=args.alpha,
        spike_size=args.size,
        show_rate=args.show_rate,
        show_isi_hist=args.show_isi,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
