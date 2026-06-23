#!/usr/bin/env python3
"""
Benchmark T2-SPIGRAD forward + backward latency at various scales.

Tests (n_osc, n_steps) combinations:
  n_osc  = [16, 32, 64, 128]
  n_steps = [10, 25, 50, 100]

Measures:
  - Forward latency (ms)
  - Backward latency (ms)
  - Total (fwd+bwd) latency
  - Throughput (samples/sec)
"""

import numpy as np
import time
from taas_checkmate import t2spigrad_forward, t2spigrad_backward, pst_linear_dim


def benchmark_config(n_osc, n_steps, n_warmup=3, n_repeat=10):
    """Benchmark one (n_osc, n_steps) config."""
    dt = 0.05
    drive_gain = 10.0
    alpha = 1.0
    freq_min = 0.5
    freq_max = 3.0
    w12, w13, w23 = 0.3, 0.15, 0.08

    # Generate random drives
    rng = np.random.default_rng(42)
    drives = rng.normal(0, 1.0, (n_steps, 3 * n_osc)).astype(np.float32)

    pst_dim = pst_linear_dim(n_osc)

    # Warmup (PyO3 init + first pass)
    for _ in range(n_warmup):
        pst, traj = t2spigrad_forward(
            n_osc, n_steps, dt, drive_gain, alpha,
            freq_min, freq_max, w12, w13, w23, drives,
        )
        gp = rng.random(pst_dim).astype(np.float32)
        _ = t2spigrad_backward(gp, traj)

    # Benchmark forward
    fwd_times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        pst, traj = t2spigrad_forward(
            n_osc, n_steps, dt, drive_gain, alpha,
            freq_min, freq_max, w12, w13, w23, drives,
        )
        fwd_times.append(time.perf_counter() - t0)

    # Benchmark backward (separate forward for each run)
    bwd_times = []
    for _ in range(n_repeat):
        pst, traj = t2spigrad_forward(
            n_osc, n_steps, dt, drive_gain, alpha,
            freq_min, freq_max, w12, w13, w23, drives,
        )
        gp = rng.random(pst_dim).astype(np.float32)
        t0 = time.perf_counter()
        _ = t2spigrad_backward(gp, traj)
        bwd_times.append(time.perf_counter() - t0)

    fwd_ms = np.median(fwd_times) * 1000
    bwd_ms = np.median(bwd_times) * 1000
    total_ms = fwd_ms + bwd_ms

    # Memory estimate: trajectory size
    # Per-step: 6 × n_osc (th1/2/3, r1/2/3) + 3 × n_osc (coupling) + 3 × n_osc (drive)
    # = 12 × n_osc × n_steps floats
    # Plus final state: 6 × n_osc
    # Total: (12 × n_steps + 6) × n_osc × 4 bytes
    traj_mb = (12 * n_steps + 6) * n_osc * 4 / (1024 * 1024)
    # pst_jacobian: 6 × (pst_dim × n_osc) floats = 6 × (3n_osc*6+12) × n_osc × 4 bytes
    jacob_mb = 6 * pst_dim * n_osc * 4 / (1024 * 1024)

    return {
        'n_osc': n_osc,
        'n_steps': n_steps,
        'pst_dim': pst_dim,
        'fwd_ms': fwd_ms,
        'bwd_ms': bwd_ms,
        'total_ms': total_ms,
        'fwd_bwd_ratio': fwd_ms / bwd_ms if bwd_ms > 0 else float('inf'),
        'thruput': 1000 / total_ms,  # samples/sec (single batch)
        'traj_mb': traj_mb,
        'jacob_mb': jacob_mb,
    }


def main():
    configs = []
    for n_osc in [16, 32, 64, 128]:
        for n_steps in [10, 25, 50, 100]:
            configs.append((n_osc, n_steps))

    print("═" * 90)
    print("  T2-SPIGRAD Performance Benchmark")
    print("═" * 90)
    print(f"  Warmup: 3  |  Repeats: 10  |  dt=0.05")
    print()

    # Header
    hdr = f"  {'n_osc':>6}  {'n_steps':>8}  {'pst_dim':>8}"
    hdr += f"  {'Fwd (ms)':>10}  {'Bwd (ms)':>10}  {'Total (ms)':>12}"
    hdr += f"  {'F/B ratio':>10}  {'samples/s':>10}"
    print(hdr)
    print("  " + "─" * 84)

    results = []
    for n_osc, n_steps in configs:
        r = benchmark_config(n_osc, n_steps)
        results.append(r)
        line = f"  {r['n_osc']:>6}  {r['n_steps']:>8}  {r['pst_dim']:>8}"
        line += f"  {r['fwd_ms']:>9.2f}  {r['bwd_ms']:>9.2f}  {r['total_ms']:>10.2f}"
        line += f"  {r['fwd_bwd_ratio']:>9.2f}x  {r['thruput']:>9.2f}"
        print(line)

    # Scaling analysis
    print("\n" + "═" * 90)
    print("  Scaling Analysis")
    print("═" * 90)
    print()
    for n_osc in [16, 32, 64, 128]:
        row = [r for r in results if r['n_osc'] == n_osc]
        # mem per step: 12 * n_osc * 4 bytes = 48 * n_osc bytes
        mem_per_step_mb = 12 * n_osc * 4 / (1024 * 1024)
        print(f"  n_osc={n_osc}:  trajectory ~{mem_per_step_mb*100:.1f} KB/step")

    print()
    print(f"  {'Scale':>20}  {'Fwd (ms)':>10}  {'Bwd (ms)':>10}  {'Total':>10}")
    print(f"  {'─'*54}")
    for n_osc in [16, 32, 64, 128]:
        r = [x for x in results if x['n_osc'] == n_osc and x['n_steps'] == 100]
        if r:
            r = r[0]
            print(f"  {'n_osc='+str(n_osc)+', n_steps=100':>20}"
                  f"  {r['fwd_ms']:>9.2f}  {r['bwd_ms']:>9.2f}  {r['total_ms']:>9.2f}")

    print()
    print("  Note: single-sample benchmark. Batch parallelization")
    print("  would improve throughput linearly with batch size.")
    print("═" * 90)


if __name__ == "__main__":
    main()
