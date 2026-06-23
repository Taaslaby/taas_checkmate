#!/usr/bin/env python3
"""taas CLI — interactive neural simulation toolkit.

Usage:
    taas sim --model izhikevich --n 1024 --steps 1000
    taas raster spikes.npy
    taas coherence phases.npy
    taas eeg --n 256 --steps 2000
    taas connectome --n 512 --sparsity 0.05
    taas bench --all
    taas info
"""

import sys
import click

from . import __version__


@click.group()
@click.version_option(__version__)
def cli():
    """taas — neural simulation toolkit (SIMD-accelerated)."""


# ─── sim ──────────────────────────────────────────────────
@cli.command()
@click.option("--model", "-m", default="izhikevich",
              type=click.Choice(["izhikevich", "lif", "lif_at", "adex", "saw", "kuramoto"]),
              help="Neuron model")
@click.option("--n", "-n", default=1024, type=int, help="Number of neurons")
@click.option("--steps", "-s", default=1000, type=int, help="Simulation steps")
@click.option("--dt", default=0.5, type=float, help="Timestep (ms)")
@click.option("--output", "-o", default=None, help="Save spikes to .npy")
@click.option("--plot/--no-plot", default=True, help="Show raster plot")
def sim(model, n, steps, dt, output, plot):
    """Run neural simulation and show raster plot."""
    from .sim import run_simulation
    run_simulation(model, n, steps, dt, output, plot)


# ─── raster ───────────────────────────────────────────────
@cli.command()
@click.argument("spikes_file", type=click.Path(exists=True))
@click.option("--window", "-w", default=None, type=int,
              help="Time window to display")
def raster(spikes_file, window):
    """Plot spike raster from .npy file."""
    from .visualize import plot_raster
    plot_raster(spikes_file, window)


# ─── coherence ────────────────────────────────────────────
@cli.command()
@click.argument("phases_file", type=click.Path(exists=True))
@click.option("--areas", "-a", default=8, type=int,
              help="Number of areas for coherence sweep")
def coherence(phases_file, areas):
    """Compute and plot phase coherence map."""
    from .visualize import plot_coherence
    plot_coherence(phases_file, areas)


# ─── eeg ──────────────────────────────────────────────────
@cli.command()
@click.option("--n", default=256, type=int, help="Number of oscillators")
@click.option("--steps", "-s", default=2000, type=int, help="Simulation steps")
@click.option("--dt", default=0.5, type=float, help="Timestep (ms)")
@click.option("--save", default=None, help="Save EEG signal to .npy")
@click.option("--plot/--no-plot", default=True, help="Show EEG + spectrum")
def eeg(n, steps, dt, save, plot):
    """Generate EEG-like signal from T3 oscillators."""
    from .visualize import run_eeg_viz
    run_eeg_viz(n, steps, dt, save, plot)


# ─── connectome ───────────────────────────────────────────
@cli.command()
@click.option("--n", default=256, type=int, help="Number of neurons")
@click.option("--sparsity", "-s", default=0.05, type=float,
              help="Connection probability")
@click.option("--spectral-radius", "-r", default=0.9, type=float,
              help="Spectral radius")
@click.option("--steps", default=100, type=int, help="Animation steps")
@click.option("--save", default=None, help="Save animation to .gif")
def connectome(n, sparsity, spectral_radius, steps, save):
    """Visualize sparse neural connectome with animated phases."""
    from .visualize import run_connectome_viz
    run_connectome_viz(n, sparsity, spectral_radius, steps, save)


# ─── bench ────────────────────────────────────────────────
@cli.command()
@click.option("--all", "all_bench", is_flag=True, help="Run all benchmarks")
@click.option("--model", default=None,
              type=click.Choice(["izhikevich", "lif", "kuramoto", "t3"]),
              help="Specific model to benchmark")
@click.option("--n", default=102400, type=int, help="Neuron count")
def bench(all_bench, model, n):
    """Benchmark Rust SIMD against NumPy and PyTorch."""
    from .sim import run_benchmark
    run_benchmark(all_bench, model, n)


# ─── info ─────────────────────────────────────────────────
@cli.command()
def info():
    """Show CPU info, SIMD support, and module health."""
    from .info import show_info
    show_info()


# ─── lsm ──────────────────────────────────────────────────
@cli.command()
@click.option("--n-reservoir", default=128, type=int)
@click.option("--steps", default=500, type=int)
def lsm(n_reservoir, steps):
    """Run Liquid State Machine demo."""
    from .sim import run_lsm_demo
    run_lsm_demo(n_reservoir, steps)


# ─── save ──────────────────────────────────────────────────
@cli.command()
@click.argument("path", type=click.Path())
@click.option("--model", "-m", default="izhikevich",
              help="Model type for simulation")
@click.option("--n", default=1024, type=int, help="Number of neurons")
@click.option("--steps", default=500, type=int, help="Simulation steps")
@click.option("--dt", default=0.5, type=float, help="Timestep")
@click.option("--description", "-d", default=None, help="Metadata description")
def save(path, model, n, steps, dt, description):
    """Run simulation and save state to .taas file."""
    from rich.console import Console
    console = Console()

    n_pad = (n + 7) // 8 * 8
    import numpy as np
    import taas_checkmate as ne

    rng = np.random.default_rng(42)

    if model == "izhikevich":
        v = np.full(n_pad, -65.0, dtype=np.float32)
        u = np.full(n_pad, -13.0, dtype=np.float32)
        tr = np.zeros(n_pad, dtype=np.float32)
        a = rng.uniform(0.02, 0.1, n_pad).astype(np.float32)
        b = rng.uniform(0.2, 0.25, n_pad).astype(np.float32)
        c = rng.uniform(-65.0, -50.0, n_pad).astype(np.float32)
        d = rng.uniform(2.0, 8.0, n_pad).astype(np.float32)
        I = rng.uniform(0, 15, n_pad).astype(np.float32)
        sp = np.zeros(n_pad, dtype=np.int32)
        spike_buffer = np.zeros((steps, n), dtype=np.int32)
        with console.status("[green]Simulating..."):
            for t in range(steps):
                ne.step_izhikevich(v, u, tr, a, b, I, dt, 0.5, sp, c, d)
                spike_buffer[t] = sp[:n]
        np.save(path.replace(".taas", "_spikes.npy"), spike_buffer)
        console.print(f"[green]✓[/green] Saved spikes to {path.replace('.taas', '_spikes.npy')}")

        # Save STDP weights as .taas
        ne.save_taas(path,
            metadata=[("model", model), ("n", str(n)), ("steps", str(steps)),
                      ("dt", str(dt)), ("firing_rate", f"{np.mean(spike_buffer):.1f}")])
        console.print(f"[green]✓[/green] Saved state to {path}")

    else:
        # T3-based save
        tri = ne.PyTriOscillator(n_pad, freq_min=2.0, freq_max=20.0)
        tri.set_uniform_coupling(0.3, 0.2, 0.1)
        with console.status("[green]Simulating T3..."):
            for _ in range(steps):
                tri.step(dt)
        dec = ne.PyNlmsDecoder(n_pad, 0.01, 1e-8)
        ne.save_taas(path, tri=tri, nlms=dec,
            metadata=[("model", model), ("n", str(n)), ("steps", str(steps)), ("dt", str(dt))])
        console.print(f"[green]✓[/green] Saved T3 state to {path}")


# ─── load ──────────────────────────────────────────────────
@cli.command()
@click.argument("path", type=click.Path(exists=True))
def load(path):
    """Load .taas file and reconstruct neural components."""
    import taas_checkmate as ne
    from rich.console import Console
    from rich.table import Table
    console = Console()

    data = ne.load_taas(path)
    meta = data.get("_meta", {})

    console.print(f"[bold cyan]📦 {path}[/bold cyan]")

    # Show reconstructed components
    component_names = {
        "tri": "TriOscillator",
        "nlms": "NlmsDecoder",
        "kalman": "KalmanDecoder",
        "sorn": "SornNetwork",
        "eprop": "EPropLIF",
        "stp": "ShortTermPlasticity",
    }

    loaded = False
    for key, name in component_names.items():
        if key in data:
            console.print(f"  [green]✓[/green] {name}: {data[key]}")
            loaded = True

    if not loaded:
        console.print("  [yellow]ℹ[/yellow] No reconstructable components found")

    # Show metadata table
    if meta:
        table = Table(title="Metadata")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        for k, v in meta.items():
            table.add_row(str(k), str(v))
        console.print(table)

    # Also show raw component info for non-reconstructable data
    raw_keys = [k for k in data if k not in component_names and k != "_meta" and not k.endswith("_error")]
    if raw_keys:
        table = Table(title="Additional Data")
        table.add_column("Key", style="yellow")
        table.add_column("Type", style="cyan")
        for k in raw_keys:
            table.add_row(k, type(data[k]).__name__)
        console.print(table)


# ─── inspect ──────────────────────────────────────────────
@cli.command()
@click.argument("path", type=click.Path(exists=True))
def inspect(path):
    """Inspect .taas file with detailed component view."""
    import taas_checkmate as ne
    output = ne.inspect_taas(path)
    print(output)


if __name__ == "__main__":
    cli()
