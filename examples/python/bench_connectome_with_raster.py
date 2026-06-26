#!/usr/bin/env python3
"""
Connectome benchmark — Izhikevich + CSR propagation + spike raster CSV + plot.

Генерирует синтетический коннектом (FlyWire-scale, 138K нейронов, ~5.3M синапсов),
прогоняет Izhikevich симуляцию через bench_connectome_cpu с записью per-step спайков,
экспортирует spike raster CSV и визуализирует через plot_raster.

Использование:
    cd taas_checkmate && PYTHONPATH=target/debug python3 examples/python/bench_connectome_with_raster.py

    # Быстрый тест (маленький коннектом, 200 шагов)
    cd taas_checkmate && PYTHONPATH=target/debug python3 examples/python/bench_connectome_with_raster.py --quick

    # Только CSV, без plot
    cd taas_checkmate && PYTHONPATH=target/debug python3 examples/python/bench_connectome_with_raster.py --csv-only
"""

import sys
import os
import time
import tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent  # examples/python/
PROJ = HERE.parent.parent      # project root
if str(PROJ / 'target' / 'debug') not in sys.path:
    sys.path.insert(0, str(PROJ / 'target' / 'debug'))

try:
    import taas_checkmate as tcm
except ImportError:
    print("✗ Cannot import taas_checkmate. Build first:")
    print("  cd taas_checkmate && cargo build --features python")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════
#  CSR Connectome Generator
# ═══════════════════════════════════════════════════════════

def gen_connectome(n_neurons: int, avg_syn: int, seed: int = 42):
    """Generate synthetic connectome in CSR format.

    Returns:
        (row_ptr, col_indices, values) — numpy arrays.
    """
    rng = np.random.RandomState(seed)

    # Each neuron connects to avg_syn ± 50% random targets
    total_syn = n_neurons * avg_syn
    row_ptr = np.zeros(n_neurons + 1, dtype=np.int32)
    col_list = []
    val_list = []

    for i in range(n_neurons):
        deg = int(np.ceil(avg_syn * (0.5 + rng.rand() * 0.5)))
        deg = max(1, min(deg, n_neurons))
        row_ptr[i + 1] = row_ptr[i] + deg
        col_list.extend(rng.randint(0, n_neurons, size=deg).tolist())
        val_list.extend((0.5 + rng.rand(deg) * 2.0).tolist())  # weights in [0.5, 2.5]

    col_indices = np.array(col_list, dtype=np.int32)
    values = np.array(val_list, dtype=np.float32)
    return row_ptr, col_indices, values


def gen_small_connectome(n_neurons: int = 100, avg_syn: int = 10, seed: int = 42):
    """Small connectome for quick testing."""
    return gen_connectome(n_neurons, avg_syn, seed)


# ═══════════════════════════════════════════════════════════
#  Connectome Simulation
# ═══════════════════════════════════════════════════════════

def run_connectome(
    row_ptr: np.ndarray,
    col_indices: np.ndarray,
    values: np.ndarray,
    n_steps: int = 500,
    dt: float = 0.5,
    n_neurons: int | None = None,
    seed: int = 42,
) -> dict:
    """Run connectome simulation via Rust bench_connectome_cpu.

    Args:
        row_ptr: CSR row pointer [n_neurons + 1]
        col_indices: CSR column indices [n_synapses]
        values: CSR values [n_synapses]
        n_steps: number of simulation steps
        dt: time step (ms)
        n_neurons: number of neurons (from row_ptr if None)
        seed: RNG seed for initial conditions

    Returns:
        dict with keys:
            wall_time_sec: wall clock time
            total_spikes: total spike count
            spike_rate_hz: average firing rate per neuron
            rt_factor: realtime factor (sim/wall)
            spike_buffer: np.ndarray [n_steps × n_neurons] int32
            spike_csv: CSV string for raster plot
    """
    if n_neurons is None:
        n_neurons = len(row_ptr) - 1

    n_padded = (n_neurons + 7) // 8 * 8
    rng = np.random.RandomState(seed)

    # State arrays (padded to 8 for SIMD)
    v = np.full(n_padded, -65.0, dtype=np.float32)
    u = np.full(n_padded, -13.0, dtype=np.float32)
    trace = np.zeros(n_padded, dtype=np.float32)
    a = np.full(n_padded, 0.02, dtype=np.float32)
    b = np.full(n_padded, 0.2, dtype=np.float32)
    c = np.full(n_padded, -65.0, dtype=np.float32)
    d = np.full(n_padded, 2.0, dtype=np.float32)
    nt_types = np.ones(n_padded, dtype=np.float32)

    # External input (random walk, 0-5 range)
    external_I = np.array(rng.rand(n_padded) * 5.0, dtype=np.float32)
    I_buf = np.zeros(n_padded, dtype=np.float32)
    spikes_out = np.zeros(n_padded, dtype=np.int32)

    # Spike buffer [n_steps × n_neurons] for raster
    spike_buffer = np.zeros(n_steps * n_neurons, dtype=np.int32)

    # Run simulation
    print(f"  Simulation: {n_neurons:,} neurons × {n_steps} steps @ dt={dt}ms")
    print(f"  Synapses:   {len(values):,}")

    wall_sec, total_spikes = tcm.bench_connectome_cpu_recording(
        row_ptr, col_indices, values, n_steps, dt,
        v, u, trace, a, b, c, d, nt_types, external_I, I_buf, spikes_out,
        spike_buffer,
    )

    # Metrics
    sim_ms = n_steps * dt
    rt_factor = sim_ms / (wall_sec * 1000.0) if wall_sec > 0 else 0.0
    neurons_active = np.any(spike_buffer.reshape(n_steps, n_neurons), axis=0).sum()
    spike_rate = total_spikes / (sim_ms / 1000.0) / max(n_neurons, 1)

    # Export to CSV via Rust
    try:
        spike_csv = tcm.spike_raster_to_csv(spike_buffer, n_neurons, dt)
    except AttributeError:
        # Fallback: manual CSV
        lines = ["time_ms,neuron_id"]
        for t in range(n_steps):
            for n in range(n_neurons):
                if spike_buffer[t * n_neurons + n]:
                    lines.append(f"{t * dt:.1f},{n}")
        spike_csv = "\n".join(lines)

    return {
        'wall_time_sec': wall_sec,
        'total_spikes': total_spikes,
        'spike_rate_hz': spike_rate,
        'rt_factor': rt_factor,
        'neurons_active': neurons_active,
        'spike_buffer': spike_buffer,
        'spike_csv': spike_csv,
        'n_neurons': n_neurons,
        'n_steps': n_steps,
        'dt': dt,
    }


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Connectome benchmark — Izhikevich + CSR + raster plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--quick', action='store_true',
                        help='Small test: 100 neurons, 200 steps')
    parser.add_argument('--neurons', type=int, default=138_000,
                        help='Number of neurons (default: 138K FlyWire-scale)')
    parser.add_argument('--synapses', type=int, default=None,
                        help='Average synapses per neuron (default: 38 for 138K, 10 for small)')
    parser.add_argument('--steps', type=int, default=500,
                        help='Simulation steps (default: 500 → 250ms simulated)')
    parser.add_argument('--dt', type=float, default=0.5,
                        help='Time step in ms (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument('--save-csv', default=None,
                        help='Save spike raster CSV to file')
    parser.add_argument('--save-plot', default=None,
                        help='Save raster plot to file (PNG)')
    parser.add_argument('--show-plot', action='store_true',
                        help='Show raster plot interactively')
    parser.add_argument('--max-plot-neurons', type=int, default=200,
                        help='Max neurons in raster plot (default: 200)')

    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────
    if args.quick:
        n_neurons = 100
        avg_syn = 10
        n_steps = 200
    else:
        n_neurons = args.neurons
        avg_syn = args.synapses or (10 if n_neurons <= 1000 else 38)
        n_steps = args.steps

    dt = args.dt
    seed = args.seed

    print("═" * 60)
    print("  🧠 Connectome Simulation + Spike Raster")
    print("═" * 60)
    print()

    # ── Generate connectome ────────────────────────────
    print("─── Generating CSR Connectome ───")
    t0 = time.time()
    if args.quick:
        row_ptr, col_indices, values = gen_small_connectome(n_neurons, avg_syn, seed)
    else:
        row_ptr, col_indices, values = gen_connectome(n_neurons, avg_syn, seed)
    gen_time = time.time() - t0
    n_syn = len(values)
    print(f"  Neurons:   {n_neurons:,}")
    print(f"  Synapses:  {n_syn:,} ({n_syn / n_neurons:.0f} avg/neuron)")
    print(f"  Generated: {gen_time:.3f}s")
    print()

    # ── Run simulation ────────────────────────────────
    print("─── Running Simulation ───")
    result = run_connectome(
        row_ptr, col_indices, values,
        n_steps=n_steps, dt=dt, n_neurons=n_neurons, seed=seed,
    )

    print(f"  Wall time:  {result['wall_time_sec']:.3f}s")
    print(f"  Simulated:  {n_steps * dt:.0f}ms ({n_steps * dt / 1000:.2f}s)")
    print(f"  RT factor:  {result['rt_factor']:.1f}× realtime")
    print(f"  Spikes:     {result['total_spikes']:,}")
    print(f"  Rate:       {result['spike_rate_hz']:.0f} Hz/neuron")
    print(f"  Active:     {result['neurons_active']:,}/{n_neurons:,} neurons")
    print()

    # ── Save CSV ──────────────────────────────────────
    csv_path = args.save_csv
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write(result['spike_csv'])
        print(f"  ✓ Spike CSV saved: {csv_path}")

    # ── Plot ──────────────────────────────────────────
    if args.save_plot or args.show_plot:
        # Save CSV for plot_raster
        plot_csv = csv_path or os.path.join(tempfile.gettempdir(), "connectome_raster.csv")
        with open(plot_csv, 'w') as f:
            f.write(result['spike_csv'])

        # Import and call plot_raster
        sys.path.insert(0, str(HERE))
        from plot_raster import plot_spike_raster

        title = f"Connectome Raster: {n_neurons:,} neurons, {n_steps * dt:.0f}ms, {result['total_spikes']:,} spikes"

        plot_spike_raster(
            csv_path=plot_csv,
            title=title,
            max_neurons=args.max_plot_neurons,
            figsize=(14, 8),
            spike_color='#2196F3',
            spike_alpha=0.4,
            spike_size=1.0,
            show_rate=True,
            show_isi_hist=False,
            save_path=args.save_plot,
        )
    else:
        print("  (use --show-plot or --save-plot to visualize)")

    # ── Summary ───────────────────────────────────────
    print()
    print("─── Summary ───")
    print(f"  Connectome:  {n_neurons:,} neurons, {n_syn:,} synapses")
    print(f"  Simulated:   {n_steps} steps × {dt}ms = {n_steps * dt:.0f}ms")
    print(f"  Performance: {result['rt_factor']:.1f}× realtime")
    print(f"  Throughput:  {n_neurons * n_steps / result['wall_time_sec'] / 1e6:.1f}M neur·step/s")
    print(f"  Spikes:      {result['total_spikes']:,} ({result['spike_rate_hz']:.0f} Hz avg)")
    print(f"  Active:      {result['neurons_active']:,} / {n_neurons:,} ({100 * result['neurons_active'] / max(n_neurons, 1):.1f}%)")
    print()
    print("═" * 60)
    print("  Done.")
    print("═" * 60)


if __name__ == "__main__":
    main()
