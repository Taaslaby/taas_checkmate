#!/usr/bin/env python3
"""System info for taas CLI."""

import platform
import struct
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

try:
    import taas_checkmate as ne
    HAVE_MODULE = True
except ImportError:
    HAVE_MODULE = False


def show_info():
    """Display system info, CPU features, and module health."""
    table = Table(title="⚡ taas-checkmate System Info")
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="green")

    # OS
    table.add_row("OS", f"{platform.system()} {platform.machine()}")
    table.add_row("Python", platform.python_version())

    # CPU
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    table.add_row("CPU", line.split(":")[1].strip())
                    break
    except Exception:
        table.add_row("CPU", "Unknown")

    # SIMD
    simd_features = _detect_simd()
    table.add_row("SIMD", simd_features if simd_features else "None detected")

    # Module
    if HAVE_MODULE:
        table.add_row("Module", "[green]✓ installed[/green]")
        table.add_row("Build", "Release (SIMD optimized)")

        # Quick benchmark (reduced iterations)
        import time
        import numpy as np
        n = 10240
        n_pad = (n + 7) // 8 * 8
        v = np.full(n_pad, -65.0, dtype=np.float32)
        u = np.full(n_pad, -13.0, dtype=np.float32)
        tr = np.zeros(n_pad, dtype=np.float32)
        a = np.full(n_pad, 0.02, dtype=np.float32)
        b = np.full(n_pad, 0.2, dtype=np.float32)
        c = np.full(n_pad, -65.0, dtype=np.float32)
        d = np.full(n_pad, 6.0, dtype=np.float32)
        I_arr = np.full(n_pad, 10.0, dtype=np.float32)
        sp = np.zeros(n_pad, dtype=np.int32)

        # Warmup
        for _ in range(5):
            ne.step_izhikevich(v, u, tr, a, b, I_arr, 0.5, 0.5, sp, c, d)

        t0 = time.perf_counter()
        n_iter = 100  # reduced from 1000
        for _ in range(n_iter):
            ne.step_izhikevich(v, u, tr, a, b, I_arr, 0.5, 0.5, sp, c, d)
        elapsed = (time.perf_counter() - t0) * 1000

        ns_per_neuron = elapsed * 1e6 / (n * n_iter)
        table.add_row("Speed", f"{ns_per_neuron:.1f} ns/neuron/step ({n:,} neurons)")
    else:
        table.add_row("Module", "[red]not installed[/red]")

    console.print(table)

    if HAVE_MODULE:
        console.print(Panel(
            "[bold]Available functions:[/bold]\n"
            "  step_izhikevich, step_lif, step_adex, step_saw\n"
            "  kuramoto_step_sparse, t3_step_sparse\n"
            "  interference, phase_coherence, band_energy\n"
            "  sigmoid_gate, thalamic_gate, safety_clamp\n"
            "  stdp_update, kalman_update, nlms_train, adam_train\n"
            "  xor_attention, rdrand_fill\n"
            "  PyTriOscillator, PyLifNeuron, PyPoissonGenerator\n"
            "  PyNlmsDecoder, PyKalmanDecoder, PyEPropLIF\n"
            "  PySornNetwork, PyTtfsEncoder, PyShortTermPlasticity\n"
            "\n[yellow]Usage: taas <command> --help[/yellow]",
            title="📦 Module API",
            border_style="green"
        ))


def _detect_simd() -> str:
    """Detect SIMD capabilities."""
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        flags = info.get("flags", [])
        simd = []
        if any("avx512" in f for f in flags):
            simd.append("AVX-512")
        if "avx2" in flags:
            simd.append("AVX2")
        if "avx" in flags:
            simd.append("AVX")
        if "sse4_2" in flags:
            simd.append("SSE4.2")
        if "neon" in flags:
            simd.append("NEON")
        return ", ".join(simd) if simd else "None detected"
    except ImportError:
        # Fallback: check /proc/cpuinfo
        try:
            with open("/proc/cpuinfo") as f:
                flags = f.read()
            simd = []
            if "avx512" in flags:
                simd.append("AVX-512")
            if "avx2" in flags:
                simd.append("AVX2")
            if "avx" in flags:
                simd.append("AVX")
            if "sse4_2" in flags:
                simd.append("SSE4.2")
            return ", ".join(simd) if simd else "Check /proc/cpuinfo"
        except Exception:
            return "Install py-cpuinfo for details"
