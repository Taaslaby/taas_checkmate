#!/usr/bin/env python3
"""Simulation runner and benchmark suite for taas CLI."""

import sys
import time
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()

try:
    import taas_checkmate as ne
except ImportError:
    console.print("[red]✗[/red] taas_checkmate module not found. Build: maturin develop --release")
    sys.exit(1)


def _pad(n: int) -> int:
    """Round up to multiple of 8 for SIMD alignment."""
    return (n + 7) // 8 * 8


def run_simulation(model: str, n: int, steps: int, dt: float,
                   output: str | None, plot: bool):
    """Run a neural simulation and optionally show raster."""
    n_pad = _pad(n)
    console.print(f"[bold]🧠 Running {model} simulation[/bold]")
    console.print(f"   {n:,} neurons × {steps:,} steps  (dt={dt}ms)")
    console.print(f"   SIMD padded to {n_pad} neurons\n")

    rng = np.random.default_rng(42)

    if model == "izhikevich":
        v = np.full(n_pad, -65.0, dtype=np.float32)
        u = np.full(n_pad, -13.0, dtype=np.float32)
        trace = np.zeros(n_pad, dtype=np.float32)
        a = rng.uniform(0.02, 0.1, n_pad).astype(np.float32)
        b = rng.uniform(0.2, 0.25, n_pad).astype(np.float32)
        c = rng.uniform(-65.0, -50.0, n_pad).astype(np.float32)
        d = rng.uniform(2.0, 8.0, n_pad).astype(np.float32)
        I = rng.uniform(0, 15, n_pad).astype(np.float32)  # noqa: E741
        spikes = np.zeros(n_pad, dtype=np.int32)
        spike_buffer = np.zeros((steps, n), dtype=np.int32)

        with Progress() as progress:
            task = progress.add_task("[cyan]Simulating...", total=steps)
            for t in range(steps):
                ne.step_izhikevich(v, u, trace, a, b, I, dt, 0.5, spikes, c, d)
                spike_buffer[t] = spikes[:n]
                progress.update(task, advance=1)

    elif model in ("lif", "lif_at"):
        v = np.full(n_pad, -65.0, dtype=np.float32)
        I = rng.uniform(0, 10, n_pad).astype(np.float32)  # noqa: E741
        spikes = np.zeros(n_pad, dtype=np.int32)
        spike_buffer = np.zeros((steps, n), dtype=np.int32)
        tau_m = np.full(n_pad, 20.0, dtype=np.float32)
        v_rest = np.full(n_pad, -65.0, dtype=np.float32)
        v_thresh = np.full(n_pad, -50.0, dtype=np.float32)
        v_reset = np.full(n_pad, -70.0, dtype=np.float32)

        if model == "lif":
            with Progress() as progress:
                task = progress.add_task("[cyan]Simulating...", total=steps)
                for t in range(steps):
                    ne.step_lif(v, I, spikes, dt, tau_m, v_rest, v_thresh, v_reset)
                    spike_buffer[t] = spikes[:n]
                    progress.update(task, advance=1)
        else:
            lif = ne.PyLifNeuron(n, dt=dt, tau_m=20.0, v_rest=-65.0,
                                  v_thresh=-50.0, v_reset=-70.0,
                                  tau_theta=50.0, beta=0.5)
            with Progress() as progress:
                task = progress.add_task("[cyan]Simulating...", total=steps)
                for t in range(steps):
                    lif.step(I, spikes)
                    spike_buffer[t] = spikes[:n]
                    progress.update(task, advance=1)

    elif model == "adex":
        v = np.full(n_pad, -65.0, dtype=np.float32)
        u = np.full(n_pad, 0.0, dtype=np.float32)
        I = rng.uniform(0, 15, n_pad).astype(np.float32)  # noqa: E741
        spikes = np.zeros(n_pad, dtype=np.int32)
        spike_buffer = np.zeros((steps, n), dtype=np.int32)
        tau_m = np.full(n_pad, 20.0, dtype=np.float32)
        tau_w = np.full(n_pad, 100.0, dtype=np.float32)
        v_rest = np.full(n_pad, -65.0, dtype=np.float32)
        v_thresh = np.full(n_pad, -50.0, dtype=np.float32)
        v_reset = np.full(n_pad, -70.0, dtype=np.float32)
        delta_t = np.full(n_pad, 2.0, dtype=np.float32)
        a = np.full(n_pad, 0.02, dtype=np.float32)  # noqa: E741
        b = np.full(n_pad, 0.02, dtype=np.float32)  # noqa: E741

        with Progress() as progress:
            task = progress.add_task("[cyan]Simulating...", total=steps)
            for t in range(steps):
                ne.step_adex(v, u, I, spikes, dt, tau_m, tau_w,
                             v_rest, v_thresh, v_reset, delta_t, a, b)
                spike_buffer[t] = spikes[:n]
                progress.update(task, advance=1)

    elif model == "saw":
        v = np.full(n_pad, -65.0, dtype=np.float32)
        u = np.full(n_pad, 0.0, dtype=np.float32)
        trace = np.zeros(n_pad, dtype=np.float32)
        spikes = np.zeros(n_pad, dtype=np.int32)
        spike_buffer = np.zeros((steps, n), dtype=np.int32)
        I = rng.uniform(0, 15, n_pad).astype(np.float32)  # noqa: E741
        a = rng.uniform(0.02, 0.1, n_pad).astype(np.float32)  # noqa: E741
        b = rng.uniform(0.2, 0.25, n_pad).astype(np.float32)  # noqa: E741

        with Progress() as progress:
            task = progress.add_task("[cyan]Simulating...", total=steps)
            for t in range(steps):
                ne.step_saw(v, u, trace, a, b, I, dt, 0.5, 0.1, spikes)
                spike_buffer[t] = spikes[:n]
                progress.update(task, advance=1)

    elif model == "kuramoto":
        phases = rng.uniform(0, 2 * np.pi, n_pad).astype(np.float32)
        omegas = rng.uniform(1.0, 5.0, n_pad).astype(np.float32)
        spike_buffer = np.zeros((steps, n), dtype=np.int32)
        sin_buf = np.zeros(n_pad, dtype=np.float32)
        cos_buf = np.zeros(n_pad, dtype=np.float32)
        coup_buf = np.zeros(n_pad, dtype=np.float32)

        row_ptr = np.arange(0, n_pad + 1, dtype=np.int32)
        col = np.arange(0, n_pad, dtype=np.int32)
        weights = rng.uniform(0.1, 1.0, n_pad).astype(np.float32)

        with Progress() as progress:
            task = progress.add_task("[cyan]Simulating...", total=steps)
            for t in range(steps):
                ne.kuramoto_step_sparse(phases, omegas, row_ptr, col,
                                        weights, sin_buf, cos_buf, coup_buf, dt)
                phase_spikes = (phases[:n] % (2 * np.pi) < 0.1).astype(np.int32)
                spike_buffer[t] = phase_spikes
                progress.update(task, advance=1)

    else:
        console.print(f"[red]Unknown model: {model}[/red]")
        return

    if output:
        np.save(output, spike_buffer)
        console.print(f"\n   [green]✓[/green] Saved spikes to {output}")

    if plot:
        from .visualize import _show_raster
        _show_raster(spike_buffer, dt, title=f"{model} — {n} neurons")


def run_benchmark(all_bench: bool, model: str | None, n: int):
    """Benchmark Rust SIMD against NumPy."""
    n_pad = _pad(n)
    rng = np.random.default_rng(42)

    models_to_bench = ["izhikevich", "lif", "kuramoto"] if all_bench or model is None else [model]

    table = Table(title=f"⚡ Benchmark — {n:,} neurons")
    table.add_column("Model", style="cyan")
    table.add_column("Backend", style="magenta")
    table.add_column("Time (ms)", justify="right", style="green")
    table.add_column("Speedup", justify="right", style="yellow")

    for m in models_to_bench:
        # Rust SIMD
        if m == "izhikevich":
            v = np.full(n_pad, -65.0, dtype=np.float32)
            u = np.full(n_pad, -13.0, dtype=np.float32)
            tr = np.zeros(n_pad, dtype=np.float32)
            a = rng.uniform(0.02, 0.1, n_pad).astype(np.float32)
            b = rng.uniform(0.2, 0.25, n_pad).astype(np.float32)
            c = rng.uniform(-65.0, -50.0, n_pad).astype(np.float32)
            d = rng.uniform(2.0, 8.0, n_pad).astype(np.float32)
            I = rng.uniform(0, 15, n_pad).astype(np.float32)  # noqa: E741
            sp = np.zeros(n_pad, dtype=np.int32)

            # Warmup
            for _ in range(10):
                ne.step_izhikevich(v, u, tr, a, b, I, 0.5, 0.5, sp, c, d)
            t0 = time.perf_counter()
            for _ in range(100):
                ne.step_izhikevich(v, u, tr, a, b, I, 0.5, 0.5, sp, c, d)
            t_rust = (time.perf_counter() - t0) * 10

            # NumPy
            v_n = np.full(n, -65.0, dtype=np.float32)
            u_n = np.full(n, -13.0, dtype=np.float32)
            I_n = I[:n]
            a_n, b_n, c_n, d_n = a[:n], b[:n], c[:n], d[:n]
            for _ in range(10):
                v_n += 0.5 * (0.04 * v_n**2 + 5 * v_n + 140 - u_n + I_n)
                u_n += 0.5 * a_n * (b_n * v_n - u_n)
                firing = v_n >= 30
                v_n[firing] = c_n[firing]
                u_n[firing] += d_n[firing]
            t0 = time.perf_counter()
            for _ in range(100):
                v_n += 0.5 * (0.04 * v_n**2 + 5 * v_n + 140 - u_n + I_n)
                u_n += 0.5 * a_n * (b_n * v_n - u_n)
                firing = v_n >= 30
                v_n[firing] = c_n[firing]
                u_n[firing] += d_n[firing]
            t_np = (time.perf_counter() - t0) * 10

            speedup = t_np / t_rust if t_rust > 0 else 0
            table.add_row(m, "Rust SIMD", f"{t_rust:.3f}", "")
            table.add_row(m, "NumPy", f"{t_np:.3f}", f"{speedup:.1f}×")

        elif m == "lif":
            v = np.full(n_pad, -65.0, dtype=np.float32)
            I_arr = rng.uniform(0, 10, n_pad).astype(np.float32)
            sp = np.zeros(n_pad, dtype=np.int32)
            tau_m = np.full(n_pad, 20.0, dtype=np.float32)
            v_rest = np.full(n_pad, -65.0, dtype=np.float32)
            v_th = np.full(n_pad, -50.0, dtype=np.float32)
            v_reset = np.full(n_pad, -70.0, dtype=np.float32)

            for _ in range(10):
                ne.step_lif(v, I_arr, sp, 0.5, tau_m, v_rest, v_th, v_reset)
            t0 = time.perf_counter()
            for _ in range(100):
                ne.step_lif(v, I_arr, sp, 0.5, tau_m, v_rest, v_th, v_reset)
            t_rust = (time.perf_counter() - t0) * 10

            v_n = np.full(n, -65.0, dtype=np.float32)
            I_n = I_arr[:n]
            for _ in range(10):
                v_n += 0.5 * (-(v_n - (-65.0)) / 20.0 + I_n / 20.0)
                firing = v_n >= -50.0
                v_n[firing] = -70.0
            t0 = time.perf_counter()
            for _ in range(100):
                v_n += 0.5 * (-(v_n - (-65.0)) / 20.0 + I_n / 20.0)
                firing = v_n >= -50.0
                v_n[firing] = -70.0
            t_np = (time.perf_counter() - t0) * 10
            speedup = t_np / t_rust if t_rust > 0 else 0
            table.add_row(m, "Rust SIMD", f"{t_rust:.3f}", "")
            table.add_row(m, "NumPy", f"{t_np:.3f}", f"{speedup:.1f}×")

        elif m == "kuramoto":
            phases = rng.uniform(0, 2 * np.pi, n_pad).astype(np.float32)
            omega = rng.uniform(1.0, 5.0, n_pad).astype(np.float32)
            sin_b = np.zeros(n_pad, dtype=np.float32)
            cos_b = np.zeros(n_pad, dtype=np.float32)
            coup = np.zeros(n_pad, dtype=np.float32)
            row_ptr = np.arange(0, n_pad + 1, dtype=np.int32)
            col = np.arange(0, n_pad, dtype=np.int32)
            w = rng.uniform(0.1, 1.0, n_pad).astype(np.float32)

            for _ in range(10):
                ne.kuramoto_step_sparse(phases, omega, row_ptr, col, w,
                                        sin_b, cos_b, coup, 0.5)
            t0 = time.perf_counter()
            for _ in range(100):
                ne.kuramoto_step_sparse(phases, omega, row_ptr, col, w,
                                        sin_b, cos_b, coup, 0.5)
            t_rust = (time.perf_counter() - t0) * 10

            # NumPy Kuramoto
            ph_n = phases[:n].copy()
            om_n = omega[:n]
            w_n = w[:n]
            for _ in range(10):
                sin_p = np.sin(ph_n)
                coup_n = np.zeros(n)
                for i in range(n):
                    coup_n[i] = w_n[i] * np.sin(ph_n[i] - ph_n[(i + 1) % n])
                ph_n += 0.5 * (om_n + coup_n)
            t0 = time.perf_counter()
            for _ in range(100):
                sin_p = np.sin(ph_n)
                coup_n = np.zeros(n)
                for i in range(n):
                    coup_n[i] = w_n[i] * np.sin(ph_n[i] - ph_n[(i + 1) % n])
                ph_n += 0.5 * (om_n + coup_n)
            t_np = (time.perf_counter() - t0) * 10
            speedup = t_np / t_rust if t_rust > 0 else 0
            table.add_row(m, "Rust SIMD", f"{t_rust:.3f}", "")
            table.add_row(m, "NumPy", f"{t_np:.3f}", f"{speedup:.1f}×")

    console.print(table)


def run_lsm_demo(n_reservoir: int, steps: int):
    """Run Liquid State Machine demo."""
    console.print(f"[bold]💧 LSM Demo — {n_reservoir} reservoir neurons[/bold]")
    console.print("   Using: T3 → Phase Encoding → LIF Reservoir → NLMS Readout")
    console.print("   Training on sine wave frequency detection...\n")

    n_pad = _pad(n_reservoir)
    rng = np.random.default_rng(42)

    # Create T3 oscillator
    tri = ne.PyTriOscillator(n_pad, freq_min=1.0, freq_max=10.0)
    tri.set_uniform_coupling(0.5, 0.3, 0.2)

    # Create reservoir
    lif = ne.PyLifNeuron(n_pad, dt=0.5, tau_m=20.0, v_rest=-65.0,
                          v_thresh=-50.0, v_reset=-70.0,
                          tau_theta=50.0, beta=0.5)

    # Create readout decoder
    decoder = ne.PyNlmsDecoder(n_pad + 1, lr=0.01, eps=1e-8)

    I_buf = np.zeros(n_pad, dtype=np.float32)
    spikes = np.zeros(n_pad, dtype=np.int32)

    features_out = np.zeros(tri.n_osc() * 6, dtype=np.float32)  # pst_linear_dim
    phase_out = np.zeros(n_pad, dtype=np.float32)

    freqs = [0.5, 1.0, 2.0, 3.0]  # Different input frequencies
    errors = []

    for epoch in range(10):
        for freq in freqs:
            for t in range(steps // len(freqs)):
                tri.step(0.5)
                tri.copy_phases(phase_out, 1)
                # Simple readout: average phase drives reservoir
                I_buf[:] = np.sin(phase_out) * freq
                lif.step(I_buf, spikes)

                features = np.zeros(n_pad + 1, dtype=np.float32)
                features[:n_pad] = spikes.astype(np.float32)
                features[n_pad] = 1.0  # bias

                target = float(freq / 5.0)  # Normalize target
                err = decoder.train(features, target)
                errors.append(abs(err))

    avg_err = float(np.mean(errors[-100:]))
    console.print(f"   [green]✓[/green] Training complete!")
    console.print(f"   Final error: {avg_err:.6f}")

    # Test
    test_freq = 1.5
    test_preds = []
    for t in range(100):
        tri.step(0.5)
        tri.copy_phases(phase_out, 1)
        I_buf[:] = np.sin(phase_out) * test_freq
        lif.step(I_buf, spikes)

        features = np.zeros(n_pad + 1, dtype=np.float32)
        features[:n_pad] = spikes.astype(np.float32)
        features[n_pad] = 1.0
        pred = decoder.predict(features)
        test_preds.append(pred)

    avg_pred = float(np.mean(test_preds))
    expected = test_freq / 5.0
    console.print(f"\n   Test {test_freq} Hz: pred={avg_pred:.4f}, expected={expected:.4f}")
    console.print(f"   [green]✓[/green] LSM generalizes!" if abs(avg_pred - expected) < 0.3
                  else "   [yellow]⚠[/yellow] LSM needs tuning")
