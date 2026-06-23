#!/usr/bin/env python3
"""
Real-world benchmark: FlyWire connectome simulation.

Simulates Izhikevich neurons with real synaptic connectivity from
the FlyWire hemibrain connectome (138,584 neurons, 5,342,446 synapses).

Runs 1000 timesteps (500ms at dt=0.5ms) with:
  1. neuron-engine (Rust + ASM via PyO3) — AVX2 kernels + CSR propagation
  2. Brian2 — academic reference

Reports:
  - Wall time for simulation
  - Total spikes
  - Simulated time per wall-second (real-time factor)
  - Spikes per second
"""

import sys, os, time, warnings, gc
import numpy as np
warnings.filterwarnings("ignore")

N_STEPS = 1000
DT = 0.5  # ms
SEED = 42

CONNECTOME_PATH = "/home/taaslab/Проекты/t3-wavelet/data/connectome/flywire_138k.npz"
PROJECT_ROOT = os.path.dirname(__file__)


def load_connectome():
    """Load FlyWire connectome, return CSR data."""
    print("  Loading connectome...", end=" ", flush=True)
    data = np.load(CONNECTOME_PATH)
    row_ptr = data["row_ptr"].copy()
    col_indices = data["col_indices"].copy()
    values = data["values"].copy()
    n = len(row_ptr) - 1
    print(f"✅ {n:,} neurons, {len(values):,} synapses")
    return n, row_ptr, col_indices, values


# ═══════════════════════════════════════════════════════════
# 1. neuron-engine (Rust + ASM)
# ═══════════════════════════════════════════════════════════

def bench_neuron_engine(n, row_ptr, col_indices, values):
    import importlib.util, shutil

    # Load .so
    for p in ['target/release/libneuron_engine.so', 'target/debug/libneuron_engine.so']:
        s = os.path.join(PROJECT_ROOT, p)
        if os.path.exists(s):
            so_path = s; break
    else:
        print(f"  ❌ libneuron_engine.so not found — build: cargo build --release")
        return None

    dest = os.path.join(PROJECT_ROOT, '_ne_real_bench.so')
    shutil.copy2(so_path, dest)
    spec = importlib.util.spec_from_file_location('neuron_engine', dest)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Setup simulation
    n_pad = (n + 7) // 8 * 8
    np.random.seed(SEED)

    # Parameters
    a = np.full(n_pad, 0.02, dtype=np.float32)
    b = np.full(n_pad, 0.2, dtype=np.float32)
    c = np.full(n_pad, -65.0, dtype=np.float32)
    d = np.full(n_pad, 8.0, dtype=np.float32)

    # State
    v = np.full(n_pad, -65.0, dtype=np.float32)
    u = np.full(n_pad, -13.0, dtype=np.float32)
    trace = np.zeros(n_pad, dtype=np.float32)
    I_buf = np.zeros(n_pad, dtype=np.float32)
    spikes = np.zeros(n_pad, dtype=np.int32)
    nt_types = np.ones(n_pad, dtype=np.float32)

    # Pad CSR arrays
    row_ptr_pad = np.zeros(n_pad + 1, dtype=np.int32)
    row_ptr_pad[:len(row_ptr)] = row_ptr
    row_ptr_pad[len(row_ptr):] = row_ptr[-1]
    col_idx_pad = np.zeros(len(col_indices) + 8, dtype=np.int32)
    col_idx_pad[:len(col_indices)] = col_indices
    weights_pad = np.zeros(len(values) + 8, dtype=np.float32)
    weights_pad[:len(values)] = values

    print(f"  Running {N_STEPS} steps (1 Rust call per step)...", end=" ", flush=True)

    external_I = np.random.uniform(0, 5, n).astype(np.float32)

    t0 = time.perf_counter()
    total_spikes = 0
    for step in range(N_STEPS):
        # One Rust call: inject current + Izhikevich + propagate spikes
        total_spikes += mod.sim_step(
            v, u, trace, a, b, c, d,
            I_buf, spikes,
            row_ptr_pad, col_idx_pad, weights_pad, nt_types,
            external_I, step, DT, 0.5, n,
        )
    elapsed = time.perf_counter() - t0
    rt_factor = (N_STEPS * DT) / elapsed  # ms simulated / ms wall

    # Cleanup
    os.unlink(dest)

    return {
        "time": elapsed,
        "spikes": total_spikes,
        "rt_factor": rt_factor,
        "sps": total_spikes / elapsed,
        "steps_per_sec": N_STEPS / elapsed,
    }


# ═══════════════════════════════════════════════════════════
# 2. Brian2
# ═══════════════════════════════════════════════════════════

def bench_brian2(n, row_ptr, col_indices, values):
    try:
        import brian2 as b2
    except ImportError:
        print("  ❌ brian2 not installed")
        return None

    np.random.seed(SEED)
    b2.defaultclock.dt = DT * b2.ms
    b2.start_scope()

    print(f"  Creating NeuronGroup ({n:,} neurons)...", end=" ", flush=True)
    eqs = '''
    dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms : 1
    du/dt = a * (b*v - u) / ms : 1
    I : 1
    a : 1
    b : 1
    c : 1
    d : 1
    '''
    G = b2.NeuronGroup(n, eqs, threshold='v >= 30', reset='v = c; u += d', method='euler')
    G.a = 0.02
    G.b = 0.2
    G.c = -65.0
    G.d = 8.0
    G.v = -65.0
    G.u = -13.0
    print("✅")

    print(f"  Creating Synapses (5.3M connections)...", end=" ", flush=True)
    S = b2.Synapses(G, G, on_pre='I += w', on_post='')
    
    # Build connections from CSR data — this is the bottleneck in Brian2
    # Connect all synapses at once to avoid O(n) Python loops
    pre_indices = []
    post_indices = []
    for i in range(n):
        start = row_ptr[i]
        end = row_ptr[i + 1]
        for j in range(start, end):
            target = col_indices[j]
            if target < n:
                pre_indices.append(i)
                post_indices.append(target)
    
    print(f"building {len(pre_indices):,} connections...", end=" ", flush=True)
    t_conn = time.perf_counter()
    S.connect(i=pre_indices, j=post_indices)
    S.w = values[:len(pre_indices)]
    conn_time = time.perf_counter() - t_conn
    print(f"✅ ({conn_time:.1f}s)")

    # Run one step to compile
    print("  Compiling (running 1 step)...", end=" ", flush=True)
    b2.run(b2.defaultclock.dt)
    print("✅")

    # Run simulation
    print(f"  Running {N_STEPS - 1} more steps...", end=" ", flush=True)
    t0 = time.perf_counter()
    spike_mon = b2.SpikeMonitor(G)
    b2.run(b2.defaultclock.dt * (N_STEPS - 1))
    elapsed = time.perf_counter() - t0

    total_spikes = len(spike_mon.i)
    rt_factor = (N_STEPS * DT) / elapsed

    return {
        "time": elapsed,
        "spikes": total_spikes,
        "rt_factor": rt_factor,
        "sps": total_spikes / elapsed if elapsed > 0 else 0,
        "steps_per_sec": N_STEPS / elapsed,
        "conn_time": conn_time,
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 82)
    print("  REAL-WORLD BENCHMARK: FlyWire Connectome Simulation")
    print("=" * 82)
    print()

    # Load connectome
    n, row_ptr, col_indices, values = load_connectome()
    print()

    # ── neuron-engine ──
    print("  ── 1. neuron-engine (Rust + ASM) ──")
    result_rust = bench_neuron_engine(n, row_ptr, col_indices, values)
    if result_rust:
        print(f"  ✅ {N_STEPS} steps in {result_rust['time']:.2f}s")
        print(f"     Spikes: {result_rust['spikes']:,}")
        print(f"     Real-time factor: {result_rust['rt_factor']:.1f}×")
    print()

    # Force GC between benchmarks
    gc.collect()

    # ── Brian2 ──
    print("  ── 2. Brian2 (academic) ──")
    result_brian = bench_brian2(n, row_ptr, col_indices, values)
    if result_brian:
        print(f"  ✅ {N_STEPS} steps in {result_brian['time']:.2f}s")
        print(f"     (connection building: {result_brian['conn_time']:.1f}s)")
        print(f"     Spikes: {result_brian['spikes']:,}")
        print(f"     Real-time factor: {result_brian['rt_factor']:.1f}×")
    print()

    # ── Summary ──
    print("=" * 82)
    print("  RESULTS")
    print("=" * 82)
    print()

    if result_rust:
        r = result_rust
        print(f"  neuron-engine:")
        print(f"    Wall time:          {r['time']:.2f}s")
        print(f"    Simulated:          {N_STEPS*DT:.0f}ms  ({N_STEPS} steps @ {DT}ms)")
        print(f"    Real-time factor:   {r['rt_factor']:.1f}×  ({r['rt_factor']:.1f}ms/sim-ms)")
        print(f"    Total spikes:       {r['spikes']:,}")
        print(f"    Spikes/s:           {r['sps']:,.0f}")
        print(f"    Steps/s:            {r['steps_per_sec']:,.0f}")

    if result_brian:
        r = result_brian
        print()
        print(f"  Brian2:")
        print(f"    Wall time:          {r['time']:.2f}s")
        print(f"    Connection build:   {r['conn_time']:.1f}s")
        print(f"    Simulated:          {N_STEPS*DT:.0f}ms  ({N_STEPS} steps @ {DT}ms)")
        print(f"    Real-time factor:   {r['rt_factor']:.1f}×")
        print(f"    Total spikes:       {r['spikes']:,}")
        print(f"    Spikes/s:           {r['sps']:,.0f}")
        print(f"    Steps/s:            {r['steps_per_sec']:,.0f}")

    if result_rust and result_brian:
        speedup = result_brian["time"] / result_rust["time"]
        print()
        print(f"  ── Speedup: Brian2 is {speedup:.0f}× slower than neuron-engine ──")
        if speedup > 1:
            print(f"     (neuron-engine does in 1s what Brian2 does in {speedup:.0f}s)")

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
    print(f"    {N_STEPS} steps  |  DT={DT}ms  |  simulated={N_STEPS*DT}ms")
    print()

    # Cleanup
    import glob
    for f in glob.glob(os.path.join(PROJECT_ROOT, '_ne_real_bench.so')):
        os.unlink(f)
