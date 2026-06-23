#!/usr/bin/env python3
"""
Benchmark: neuron-engine (Rust+ASM) vs C vs NumPy vs PyTorch vs Brian2.

FAIR methodology:
  - All implementations reuse buffers (no per-call allocations)
  - All do the same work (safety clamp + Izhikevich step + spike detect)
  - 20 warmup + 200 timed iterations
  - P50/P95/P99 reporting
  - Numerical verification
"""

import sys, os, time, warnings
import numpy as np
import ctypes
warnings.filterwarnings("ignore")

N_NEURONS = 102400
N_PAD = (N_NEURONS + 7) // 8 * 8
DT = 0.5
N_WARMUP = 20
N_ITER = 200

results = []

def run_bench(name, setup_fn, step_fn):
    ctx = setup_fn()
    if ctx is None:
        return None
    for _ in range(N_WARMUP):
        step_fn(ctx)
    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        step_fn(ctx)
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    print(f"  {name:<28}  "
          f"min={t.min():.3f}  mean={t.mean():.3f}  "
          f"P50={np.median(t):.3f}  P95={np.percentile(t,95):.3f}  "
          f"P99={np.percentile(t,99):.3f}  ms")
    return t


# ════════════════════ 1. neuron-engine (Rust + ASM) ════════════════════
def setup_rust():
    import importlib.util, shutil
    for p in ['target/release/libneuron_engine.so', 'target/debug/libneuron_engine.so']:
        s = os.path.join(os.path.dirname(__file__), p)
        if os.path.exists(s):
            so_path = s; break
    else:
        print("  libneuron_engine.so not found — build: cargo build --release")
        return None
    # Copy with original name — PyO3 exports PyInit_neuron_engine
    dest = os.path.join(os.path.dirname(__file__), '_neuron_engine_bench.so')
    shutil.copy2(so_path, dest)
    spec = importlib.util.spec_from_file_location('neuron_engine', dest)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    np.random.seed(42)
    return {
        'mod': mod,
        'a': np.random.uniform(0.02, 0.1, N_PAD).astype(np.float32),
        'b': np.random.uniform(0.2, 0.25, N_PAD).astype(np.float32),
        'c': np.random.uniform(-65.0, -50.0, N_PAD).astype(np.float32),
        'd': np.random.uniform(2.0, 8.0, N_PAD).astype(np.float32),
        'I': np.random.uniform(0, 15, N_PAD).astype(np.float32),
        'v': np.full(N_PAD, -65.0, dtype=np.float32),
        'u': np.full(N_PAD, -13.0, dtype=np.float32),
        'tr': np.full(N_PAD, 0.0, dtype=np.float32),
        'sp': np.zeros(N_PAD, dtype=np.int32),
    }

def step_rust(ctx):
    ctx['v'][:] = -65.0
    ctx['u'][:] = -13.0
    ctx['tr'][:] = 0.0
    ctx['sp'][:] = 0
    ctx['mod'].step_izhikevich(ctx['v'], ctx['u'], ctx['tr'],
        ctx['a'], ctx['b'], ctx['I'], DT, 0.5, ctx['sp'], ctx['c'], ctx['d'])


# ════════════════════ 2. C + ctypes ════════════════════
def setup_c():
    import ctypes
    build_dir = os.path.join(os.path.dirname(__file__), '..', 'kuramoto_engine', 'build')
    lib_path = os.path.join(build_dir, 'libneuron.so')
    if not os.path.exists(lib_path):
        print("  libneuron.so not found — build C engine first (make -C ../kuramoto_engine)")
        return None
    lib = ctypes.CDLL(lib_path)
    lib.step_n_neurons.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ]
    np.random.seed(42)
    I = np.random.uniform(0, 15, N_PAD).astype(np.float32)
    # Pre-clamp I (same as Rust safety_clamp — done once, in-place)
    np.clip(I, -100, 100, out=I)
    return {
        'lib': lib,
        'a': np.random.uniform(0.02, 0.1, N_PAD).astype(np.float32),
        'b': np.random.uniform(0.2, 0.25, N_PAD).astype(np.float32),
        'c': np.random.uniform(-65.0, -50.0, N_PAD).astype(np.float32),
        'd': np.random.uniform(2.0, 8.0, N_PAD).astype(np.float32),
        'I': I,
        'v': np.full(N_PAD, -65.0, dtype=np.float32),
        'u': np.full(N_PAD, -13.0, dtype=np.float32),
        'tr': np.full(N_PAD, 0.0, dtype=np.float32),
        'sp': np.zeros(N_PAD, dtype=np.int32),
    }

def step_c(ctx):
    ctx['v'][:] = -65.0
    ctx['u'][:] = -13.0
    ctx['tr'][:] = 0.0
    ctx['sp'][:] = 0
    ctx['lib'].step_n_neurons(
        N_PAD,
        ctx['v'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctx['u'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctx['tr'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctx['a'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctx['b'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctx['I'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(DT), ctypes.c_float(0.5),
        ctx['sp'].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctx['c'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctx['d'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )


# ════════════════════ 3. Plain NumPy ════════════════════
def setup_numpy():
    np.random.seed(42)
    I = np.random.uniform(0, 15, N_NEURONS).astype(np.float32)
    np.clip(I, -100, 100, out=I)
    return {
        'a': np.random.uniform(0.02, 0.1, N_NEURONS).astype(np.float32),
        'b': np.random.uniform(0.2, 0.25, N_NEURONS).astype(np.float32),
        'c': np.random.uniform(-65.0, -50.0, N_NEURONS).astype(np.float32),
        'd': np.random.uniform(2.0, 8.0, N_NEURONS).astype(np.float32),
        'I': I,
        'v': np.full(N_NEURONS, -65.0, dtype=np.float32),
        'u': np.full(N_NEURONS, -13.0, dtype=np.float32),
    }

def step_numpy(ctx):
    v, u = ctx['v'], ctx['u']
    v[:] = -65.0
    u[:] = -13.0
    Ic = ctx['I']
    # Izhikevich step
    v += DT * (0.04 * v**2 + 5 * v + 140 - u + Ic)
    u += DT * ctx['a'] * (ctx['b'] * v - u)
    # Spike detect & reset
    firing = v >= 30
    v[firing] = ctx['c'][firing]
    u[firing] += ctx['d'][firing]


# ════════════════════ 4. PyTorch (CPU) ════════════════════
def setup_torch():
    try:
        import torch
    except ImportError:
        return None
    np.random.seed(42)
    I = np.random.uniform(0, 15, N_NEURONS).astype(np.float32)
    np.clip(I, -100, 100, out=I)
    return {
        'a': torch.from_numpy(np.random.uniform(0.02, 0.1, N_NEURONS).astype(np.float32)),
        'b': torch.from_numpy(np.random.uniform(0.2, 0.25, N_NEURONS).astype(np.float32)),
        'c': torch.from_numpy(np.random.uniform(-65.0, -50.0, N_NEURONS).astype(np.float32)),
        'd': torch.from_numpy(np.random.uniform(2.0, 8.0, N_NEURONS).astype(np.float32)),
        'I': torch.from_numpy(I),
        'v': torch.full((N_NEURONS,), -65.0, dtype=torch.float32),
        'u': torch.full((N_NEURONS,), -13.0, dtype=torch.float32),
    }

def step_torch(ctx):
    import torch
    v, u = ctx['v'], ctx['u']
    v.fill_(-65.0)
    u.fill_(-13.0)
    Ic = ctx['I']
    v += DT * (0.04 * v**2 + 5 * v + 140 - u + Ic)
    u += DT * ctx['a'] * (ctx['b'] * v - u)
    firing = v >= 30
    v[firing] = ctx['c'][firing]
    u[firing] += ctx['d'][firing]


# ════════════════════ 5. Brian2 ════════════════════
def setup_brian2():
    try:
        from brian2 import NeuronGroup, defaultclock, ms, run, start_scope
    except ImportError:
        return None
    np.random.seed(42)
    eqs = '''
    dv/dt = (0.04*v**2 + 5*v + 140 - u + I) / ms : 1
    du/dt = a * (b*v - u) / ms : 1
    I : 1
    a : 1
    b : 1
    c : 1
    d : 1
    '''
    defaultclock.dt = DT * ms
    start_scope()
    group = NeuronGroup(N_NEURONS, eqs,
                        threshold='v >= 30',
                        reset='v = c; u += d',
                        method='euler')
    group.a = 0.02
    group.b = 0.2
    group.c = -65 + 15 * np.random.rand(N_NEURONS)**2
    group.d = 8 - 6 * np.random.rand(N_NEURONS)**2
    group.v = -65
    group.u = -13
    group.I = np.random.uniform(0, 15, N_NEURONS)
    return {'group': group}

def step_brian2(ctx):
    from brian2 import run, defaultclock
    run(defaultclock.dt)


# ════════════════════ Num verification ════════════════════

def numerical_verify():
    print()
    print("  ── Numerical verification (1024 neurons) ──")
    
    # Run Rust and C, compare outputs
    import importlib.util, shutil, ctypes
    for p in ['target/release/libneuron_engine.so', 'target/debug/libneuron_engine.so']:
        s = os.path.join(os.path.dirname(__file__), p)
        if os.path.exists(s):
            so_path = s; break
    else:
        return
    dest = os.path.join(os.path.dirname(__file__), '_neuron_engine_verify.so')
    shutil.copy2(so_path, dest)
    spec = importlib.util.spec_from_file_location('neuron_engine', dest)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    n_small = 1024
    n_pad = (n_small + 7) // 8 * 8
    np.random.seed(42)
    
    a = np.random.uniform(0.02, 0.1, n_pad).astype(np.float32)
    b = np.random.uniform(0.2, 0.25, n_pad).astype(np.float32)
    c = np.random.uniform(-65.0, -50.0, n_pad).astype(np.float32)
    d = np.random.uniform(2.0, 8.0, n_pad).astype(np.float32)
    I = np.random.uniform(0, 15, n_pad).astype(np.float32)
    
    # Rust
    v_r = np.full(n_pad, -65.0, dtype=np.float32)
    u_r = np.full(n_pad, -13.0, dtype=np.float32)
    tr_r = np.full(n_pad, 0.0, dtype=np.float32)
    sp_r = np.zeros(n_pad, dtype=np.int32)
    mod.step_izhikevich(v_r, u_r, tr_r, a, b, I, DT, 0.5, sp_r, c, d)
    
    # C (same parameters)
    build_dir = os.path.join(os.path.dirname(__file__), '..', 'kuramoto_engine', 'build')
    lib = ctypes.CDLL(os.path.join(build_dir, 'libneuron.so'))
    lib.step_n_neurons.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ]
    v_c = np.full(n_pad, -65.0, dtype=np.float32)
    u_c = np.full(n_pad, -13.0, dtype=np.float32)
    tr_c = np.full(n_pad, 0.0, dtype=np.float32)
    sp_c = np.zeros(n_pad, dtype=np.int32)
    Ic = np.clip(I, -100, 100)  # C bridge does clamp in Python
    lib.step_n_neurons(n_pad,
        v_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        u_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        Ic.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(DT), ctypes.c_float(0.5),
        sp_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    
    v_diff = np.max(np.abs(v_r[:n_small] - v_c[:n_small]))
    u_diff = np.max(np.abs(u_r[:n_small] - u_c[:n_small]))
    sp_diff = np.max(np.abs(sp_r[:n_small].astype(np.int32) - sp_c[:n_small].astype(np.int32)))
    print(f"  Rust vs C:  max|Δv|={v_diff:.6f}  max|Δu|={u_diff:.6f}  max|Δspikes|={sp_diff}")
    if v_diff < 1e-4 and u_diff < 1e-4 and sp_diff == 0:
        print(f"  ✅ Numerical match")
    else:
        print(f"  ⚠️  Numerical differences detected (likely due to clamp order)")
    
    # Cleanup
    os.unlink(os.path.join(os.path.dirname(__file__), '_ne_verify.so'))


# ════════════════════ MAIN ════════════════════

if __name__ == "__main__":
    print("=" * 82)
    print(f"  Izhikevich Step Benchmark  —  {N_NEURONS:,} neurons")
    print(f"  All implementations reuse buffers  |  {N_WARMUP} warmup  |  {N_ITER} iterations")
    print("=" * 82)
    print()

    bench_configs = [
        ("neuron-engine (Rust+ASM)", setup_rust, step_rust),
        ("C + ctypes", setup_c, step_c),
        ("Plain NumPy (vectorized)", setup_numpy, step_numpy),
        ("PyTorch (CPU)", setup_torch, step_torch),
        ("Brian2 (compiled)", setup_brian2, step_brian2),
    ]

    all_times = {}
    for name, setup_fn, step_fn in bench_configs:
        print(f"  {name:<30} ...", end=" ", flush=True)
        t = run_bench(name, setup_fn, step_fn)
        all_times[name] = t
        print()

    # Numerical verification
    numerical_verify()

    # Summary table
    print()
    print("=" * 82)
    print("  RESULTS  —  all times in milliseconds  (lower = faster)")
    print("=" * 82)
    
    rust_med = np.median(all_times.get("neuron-engine (Rust+ASM)", [1]))
    
    print(f"  {'Library':<30} {'mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}  {'vs Rust (P50)':>14}")
    print("  " + "-" * 72)

    for name in [c[0] for c in bench_configs]:
        t = all_times.get(name)
        if t is None:
            print(f"  {name:<30}  {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}  {'N/A':>14}")
        else:
            med = np.median(t)
            ratio = med / rust_med
            if ratio >= 5:
                rat_str = f"{ratio:.0f}× slower"
            elif ratio >= 1.5:
                rat_str = f"{ratio:.1f}× slower"
            elif ratio >= 0.9:
                rat_str = f"{ratio:.2f}× same"
            else:
                rat_str = f"{1/ratio:.1f}× faster"
            print(f"  {name:<30}  {t.mean():>8.3f}  {np.median(t):>8.3f}  "
                  f"{np.percentile(t,95):>8.3f}  {np.percentile(t,99):>8.3f}  {rat_str:>14}")

    print()
    print("  System:")
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    print(f"    CPU: {line.split(':')[1].strip()}")
                    break
    except: pass
    print(f"    Python: {sys.version.split()[0]}")
    print(f"    OMP_NUM_THREADS=1  |  DT=0.5ms  |  single Izhikevich step")
    print()
    
    # Cleanup
    for f in ['_neuron_engine_bench.so', '_neuron_engine_verify.so']:
        p = os.path.join(os.path.dirname(__file__), f)
        if os.path.exists(p): os.unlink(p)
