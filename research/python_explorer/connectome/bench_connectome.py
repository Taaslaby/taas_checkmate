#!/usr/bin/env python3
"""
Connectome benchmark: FlyWire 138K neurons on CPU (SIMD+rayon) vs GPU (CUDA).

Usage:
    python3 bench_connectome.py                  # CPU + GPU
    python3 bench_connectome.py --cpu-only       # CPU only
    python3 bench_connectome.py --gpu-only       # GPU only

Requirements:
    - taas_checkmate built with --features python,cuda
    - FlyWire NPZ at /home/taaslab/Проекты/t3-wavelet/data/connectome/flywire_138k.npz
"""

import sys, os, time, warnings, argparse
import numpy as np
warnings.filterwarnings("ignore")

N_STEPS = 500
DT = 0.5  # ms
SEED = 42

CONNECTOME_PATH = "/home/taaslab/Проекты/t3-wavelet/data/connectome/flywire_138k.npz"


def load_connectome():
    """Load FlyWire connectome, return CSR data + neuron count."""
    print("  Loading connectome...", end=" ", flush=True)
    data = np.load(CONNECTOME_PATH)
    row_ptr = data["row_ptr"].copy()
    col_indices = data["col_indices"].copy()
    weights = data["values"].copy()
    n = len(row_ptr) - 1
    print(f"✅ {n:,} neurons, {len(weights):,} synapses")
    return n, row_ptr, col_indices, weights


def setup_state(n):
    """Create neuron state arrays."""
    n_pad = (n + 7) // 8 * 8
    rng = np.random.default_rng(SEED)

    v = np.full(n_pad, -65.0, dtype=np.float32)
    u = np.full(n_pad, -13.0, dtype=np.float32)
    trace = np.zeros(n_pad, dtype=np.float32)
    a = np.full(n_pad, 0.02, dtype=np.float32)
    b = np.full(n_pad, 0.2, dtype=np.float32)
    c = np.full(n_pad, -65.0, dtype=np.float32)
    d = np.full(n_pad, 2.0, dtype=np.float32)

    I_buf = np.zeros(n_pad, dtype=np.float32)
    spikes = np.zeros(n_pad, dtype=np.int32)
    external_I = rng.uniform(0, 5, n_pad).astype(np.float32)
    nt_types = np.ones(n_pad, dtype=np.float32)

    return {
        "v": v, "u": u, "trace": trace,
        "a": a, "b": b, "c": c, "d": d,
        "I_buf": I_buf, "spikes": spikes,
        "external_I": external_I, "nt_types": nt_types,
        "n": n, "n_pad": n_pad,
    }


def pad_csr(row_ptr, col_indices, weights, n_pad):
    """Pad CSR arrays to n_pad."""
    n = len(row_ptr) - 1
    row_ptr_pad = np.zeros(n_pad + 1, dtype=np.int32)
    row_ptr_pad[:len(row_ptr)] = row_ptr
    row_ptr_pad[len(row_ptr):] = row_ptr[-1]

    col_idx_pad = np.zeros(len(col_indices) + 8, dtype=np.int32)
    col_idx_pad[:len(col_indices)] = col_indices

    weights_pad = np.zeros(len(weights) + 8, dtype=np.float32)
    weights_pad[:len(weights)] = weights

    return row_ptr_pad, col_idx_pad, weights_pad


# ═══════════════════════════════════════════════════════════
# CPU Benchmark (SIMD + rayon)
# ═══════════════════════════════════════════════════════════

def bench_cpu(n, row_ptr, col_indices, weights, s):
    """Run connectome simulation on CPU using bench_connectome_cpu()."""
    from taas_checkmate import bench_connectome_cpu

    row_ptr_pad, col_idx_pad, weights_pad = pad_csr(row_ptr, col_indices, weights, s["n_pad"])
    nt_types_pad = np.ones(s["n_pad"], dtype=np.float32)

    print(f"  Running {N_STEPS} steps (SIMD+rayon)...", end=" ", flush=True)

    t0 = time.perf_counter()
    elapsed, total_spikes = bench_connectome_cpu(
        row_ptr_pad, col_idx_pad, weights_pad,
        N_STEPS, DT,
        s["v"], s["u"], s["trace"],
        s["a"], s["b"], s["c"], s["d"],
        nt_types_pad,
        s["external_I"], s["I_buf"], s["spikes"],
    )
    t_wall = time.perf_counter() - t0

    rt_factor = (N_STEPS * DT) / elapsed  # ms simulated / ms wall

    return {
        "time": elapsed,
        "spikes": total_spikes,
        "rt_factor": rt_factor,
        "sps": total_spikes / elapsed,
        "steps_per_sec": N_STEPS / elapsed,
    }


# ═══════════════════════════════════════════════════════════
# GPU Benchmark (CUDA)
# ═══════════════════════════════════════════════════════════

def bench_gpu(n, row_ptr, col_indices, weights, s):
    """Run connectome simulation on GPU using bench_connectome_gpu()."""
    from taas_checkmate import bench_connectome_gpu

    row_ptr_pad, col_idx_pad, weights_pad = pad_csr(row_ptr, col_indices, weights, s["n_pad"])
    nt_types_pad = np.ones(s["n_pad"], dtype=np.float32)

    print(f"  Running {N_STEPS} steps (CUDA)...", end=" ", flush=True)

    t0 = time.perf_counter()
    try:
        elapsed, total_spikes = bench_connectome_gpu(
            row_ptr_pad, col_idx_pad, weights_pad,
            N_STEPS, DT,
            s["v"], s["u"],
            s["a"], s["b"], s["c"], s["d"],
            nt_types_pad,
            s["external_I"], s["spikes"],
        )
    except RuntimeError as e:
        if "CUDA not available" in str(e):
            print(f"❌ CUDA not available — skip GPU benchmark")
            return None
        raise

    t_wall = time.perf_counter() - t0
    rt_factor = (N_STEPS * DT) / elapsed

    return {
        "time": elapsed,
        "spikes": total_spikes,
        "rt_factor": rt_factor,
        "sps": total_spikes / elapsed,
        "steps_per_sec": N_STEPS / elapsed,
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connectome benchmark: CPU vs GPU")
    parser.add_argument("--cpu-only", action="store_true", help="CPU only")
    parser.add_argument("--gpu-only", action="store_true", help="GPU only")
    args = parser.parse_args()

    print("=" * 82)
    print("  CONNECTOME BENCHMARK: FlyWire 138K neurons")
    print("=" * 82)
    print()

    # Load connectome
    n, row_ptr, col_indices, weights = load_connectome()

    # Setup state (fresh copy for each backend)
    s_cpu = setup_state(n)

    # Pad CSR
    row_ptr_pad, col_idx_pad, weights_pad = pad_csr(row_ptr, col_indices, weights, s_cpu["n_pad"])

    print()
    print(f"  Simulation: {N_STEPS} steps × DT={DT}ms = {N_STEPS*DT:.0f}ms simulated")
    print(f"  Neurons: {n:,} (padded to {s_cpu['n_pad']:,})")
    print(f"  Synapses: {len(weights):,}")
    print()

    results = {}

    # ── CPU ──
    if not args.gpu_only:
        print("  ── 1. CPU (SIMD + rayon) ──")
        r = bench_cpu(n, row_ptr, col_indices, weights, s_cpu)
        if r:
            results["cpu"] = r
            print(f"  ✅ {N_STEPS} steps in {r['time']:.3f}s")
            print(f"     Spikes: {r['spikes']:,}")
            print(f"     Real-time factor: {r['rt_factor']:.1f}×")
            print(f"     Steps/s: {r['steps_per_sec']:,.0f}")
            print(f"     Spikes/s: {r['sps']:,.0f}")
        print()

    # ── GPU ──
    if not args.cpu_only:
        s_gpu = setup_state(n)  # fresh state
        print("  ── 2. GPU (CUDA) ──")
        r = bench_gpu(n, row_ptr, col_indices, weights, s_gpu)
        if r:
            results["gpu"] = r
            print(f"  ✅ {N_STEPS} steps in {r['time']:.3f}s")
            print(f"     Spikes: {r['spikes']:,}")
            print(f"     Real-time factor: {r['rt_factor']:.1f}×")
            print(f"     Steps/s: {r['steps_per_sec']:,.0f}")
            print(f"     Spikes/s: {r['sps']:,.0f}")
        print()

    # ── Summary ──
    print("=" * 82)
    print("  RESULTS")
    print("=" * 82)
    print()

    if "cpu" in results:
        r = results["cpu"]
        print(f"  CPU (SIMD+rayon):")
        print(f"    Wall time:    {r['time']:.3f}s")
        print(f"    Steps/s:      {r['steps_per_sec']:,.0f}")
        print(f"    RT factor:    {r['rt_factor']:.1f}×")
        print(f"    Spikes:       {r['spikes']:,}")

    if "gpu" in results:
        r = results["gpu"]
        print()
        print(f"  GPU (CUDA):")
        print(f"    Wall time:    {r['time']:.3f}s")
        print(f"    Steps/s:      {r['steps_per_sec']:,.0f}")
        print(f"    RT factor:    {r['rt_factor']:.1f}×")
        print(f"    Spikes:       {r['spikes']:,}")

    if "cpu" in results and "gpu" in results:
        r_cpu = results["cpu"]
        r_gpu = results["gpu"]
        ratio = r_cpu["time"] / r_gpu["time"]
        print()
        print(f"  ── Speedup: GPU is {ratio:.1f}× faster than CPU ──")
        if ratio > 1:
            print(f"     (GPU does in 1s what CPU does in {ratio:.1f}s)")
        else:
            print(f"     (CPU does in 1s what GPU does in {1/ratio:.1f}s)")

    print()
    print(f"  System:")
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    print(f"    CPU: {line.split(':')[1].strip()}")
                    break
    except:
        pass
    print(f"    GPU: NVIDIA RTX 4060 Ti (8GB)")
    print(f"    {N_STEPS} steps  |  DT={DT}ms  |  simulated={N_STEPS*DT:.0f}ms")
    print()
