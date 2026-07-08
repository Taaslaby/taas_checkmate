#!/usr/bin/env python3
"""Profile GPU HH batch_hh_steps timing."""
import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flywire_brain import FlyWireCSR, FlyWireHHBrain

def profile():
    print("[load] Loading FlyWire connectome...")
    csr = FlyWireCSR()
    csr.load()
    dt_ms = 0.1
    n_steps = int(25.0 / dt_ms)  # 250
    print(f"[brain] Creating FlyWireHHBrain ({csr.n_neurons:,} neurons, dt={dt_ms}ms, gpu)...")
    brain = FlyWireHHBrain(csr, dt_ms=dt_ms)

    # Warmup + connectome upload
    print("[warmup] Running 1 batch (includes connectome upload)...")
    brain.step_batch(0.025)
    print("[warmup] Done. Measuring...")

    n_warm = 20
    n_meas = 100
    times = []

    # Warmup batches
    for i in range(n_warm):
        brain.step_batch(0.025)

    # Measured batches
    for i in range(n_meas):
        t0 = time.perf_counter()
        brain.step_batch(0.025)
        times.append(time.perf_counter() - t0)
        if (i + 1) % 20 == 0:
            print(f"  batch {i+1}/{n_meas}")

    times.sort()
    n = len(times)
    mean = sum(times) / n
    median = times[n // 2]
    p90 = times[int(n * 0.9)]
    p99 = times[int(n * 0.99)]
    sim_s_per_batch = 0.025
    print()
    print(f"{'batch_hh_steps timing':-^60}")
    print(f"  batches measured:   {n}")
    print(f"  n_steps/batch:      {n_steps}")
    print(f"  sim_time/batch:     {sim_s_per_batch*1000:.1f} ms")
    print(f"  realtime/batch:     {mean*1000:.3f} ms (mean)")
    print(f"                      {median*1000:.3f} ms (median)")
    print(f"                      {p90*1000:.3f} ms (p90)")
    print(f"                      {p99*1000:.3f} ms (p99)")
    print(f"                      {times[0]*1000:.3f} ms (min)")
    print(f"                      {times[-1]*1000:.3f} ms (max)")
    print(f"  sim/realtime ratio: {sim_s_per_batch/mean:.1f}x")
    print(f"  batches/second:     {1/mean:.0f}")
    print(f"{'':-^60}")
    print(f"  Speedup vs CPU HHLUT baseline (2.3s/batch):")
    cpu_baseline = 2.3
    print(f"    GPU CSR batch:  {mean*1000:.2f} ms/batch")
    print(f"    CPU HHLUT:      {cpu_baseline:*<8.3f} ms/batch")
    print(f"    Speedup:        {cpu_baseline/mean/1000:.0f}×")
    print()

if __name__ == "__main__":
    profile()
