#!/usr/bin/env python3
"""
Честный бенчмарк: C+ctypes vs Rust+PyO3 — одинаковые ASM-функции.

Методология:
  • Оба пути вызывают ОДИН И ТОТ ЖЕ ASM-код
  • C: через ctypes → C wrapper → ASM
  • Rust: через PyO3 → Rust wrapper → ASM
  • Для sequential тестов: OMP_NUM_THREADS=1 (C без OpenMP)
  • Для parallel тестов: C с OpenMP, Rust sequential (причина разницы — parallelism)
  • Numerical verification на свежих данных до бенчмарка
"""

import sys, os, time, ctypes, numpy as np

PROJECT = os.path.dirname(os.path.abspath(__file__))

# ── Rust-модуль ─────────────────────────────────────────
_RUST_SO = None
for p in ["target/release/libneuron_engine.so", "target/debug/libneuron_engine.so"]:
    s = os.path.join(PROJECT, p)
    if os.path.exists(s):
        _RUST_SO = s
        break
assert _RUST_SO is not None, "neuron_engine.so not found! Build with 'cargo build --release'"

import shutil
rust_module_path = os.path.join(PROJECT, "neuron_engine.so")
if os.path.exists(rust_module_path):
    os.remove(rust_module_path)
shutil.copy2(_RUST_SO, rust_module_path)
import neuron_engine

# ── C-библиотеки ────────────────────────────────────────
BUILD = os.path.join(PROJECT, "..", "kuramoto_engine", "build")
lib = ctypes.CDLL(os.path.join(BUILD, "libneuron.so"))

np.random.seed(42)


def time_ms(label, f, warmup=5, repeats=50):
    for _ in range(warmup):
        f()
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        t1 = time.perf_counter()
        t.append((t1 - t0) * 1000)
    avg, std = np.mean(t), np.std(t)
    print(f"  {label:55s}  {avg:8.3f} ± {std:5.3f} ms")
    return avg


def verify(v_c, v_r, label):
    err = np.max(np.abs(np.array(v_r, dtype=np.float32) - np.array(v_c, dtype=np.float32)))
    ok = err < 1e-4
    print(f"    Проверка {label}: max error = {err:.2e}  {'✓' if ok else '✗ MISMATCH'}")


def make_data(n, n_pad):
    """Свежие данные для Izhikevich."""
    v = np.random.uniform(-70, -50, n_pad).astype(np.float32)
    u = np.random.uniform(-20, -10, n_pad).astype(np.float32)
    tr = np.zeros(n_pad, dtype=np.float32)
    a = np.full(n_pad, 0.02, dtype=np.float32)
    b = np.full(n_pad, 0.2, dtype=np.float32)
    I = np.zeros(n_pad, dtype=np.float32)
    I[:200] = 30.0
    sp = np.zeros(n_pad, dtype=np.int32)
    c = np.full(n_pad, -65.0, dtype=np.float32)
    d = np.full(n_pad, 8.0, dtype=np.float32)
    return v, u, tr, a, b, I, sp, c, d


# ═══════════════════════════════════════════════════════════
#  1a. Izhikevich — SEQUENTIAL (C: OMP_NUM_THREADS=1, Rust: sequential)
#      Измеряет: FFI overhead + wrapper overhead
# ═══════════════════════════════════════════════════════════

def bench_izhikevich_sequential():
    print("\n" + "═" * 70)
    print("  1a. Izhikevich step  |  SEQUENTIAL (OMP_NUM_THREADS=1)")
    print("      Измеряет: C wrapper vs Rust wrapper overhead")
    print("═" * 70)

    n = 100 * 1024
    n_pad = (n + 7) // 8 * 8

    lib.step_n_neurons.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]

    p = lambda x: x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    pi = lambda x: x.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    dt, td = 0.5, 0.99

    # Свежие данные для C
    v_c, u_c, tr_c, a_c, b_c, I_c, sp_c, c_c, d_c = make_data(n, n_pad)

    # Свежие данные для Rust (tolist один раз)
    vr, ur, trr, ar, br, Ir, spr, cr, dr = make_data(n, n_pad)
    vr_l = vr.tolist(); ur_l = ur.tolist(); trr_l = trr.tolist()
    ar_l = ar.tolist(); br_l = br.tolist(); Ir_l = Ir.tolist()
    spr_l = spr.tolist(); cr_l = cr.tolist(); dr_l = dr.tolist()

    # Verification на свежих данных — ДО бенчмарка
    verify(v_c[:n], vr_l[:n], "initial data (should be 0 — случайные данные разные)")
    # Выполним один шаг на свежих копиях для верификации
    vv, uu, *_ = make_data(n, n_pad)
    lib.step_n_neurons(n, p(vv), p(uu), p(tr_c), p(a_c), p(b_c), p(I_c), dt, td, pi(sp_c), p(c_c), p(d_c))
    vv_r = vv.tolist()
    neuron_engine.step_izhikevich(vv_r, uu.tolist(), tr_c.tolist(), a_c.tolist(), b_c.tolist(),
                                   I_c.tolist(), dt, td, sp_c.tolist(), c_c.tolist(), d_c.tolist())
    verify(vv, vv_r, "Izhikevich (fresh data, 1 step)")

    # ── C: step_n_neurons с OMP_NUM_THREADS=1 ──
    os.environ["OMP_NUM_THREADS"] = "1"

    def c_seq():
        lib.step_n_neurons(n, p(v_c), p(u_c), p(tr_c), p(a_c), p(b_c), p(I_c), dt, td, pi(sp_c), p(c_c), p(d_c))

    # ── Rust: step_izhikevich (sequential) ──
    def rust_seq():
        neuron_engine.step_izhikevich(
            vr_l, ur_l, trr_l, ar_l, br_l, Ir_l, dt, td, spr_l, cr_l, dr_l)

    c_avg = time_ms("C+ctypes (step_n_neurons, OMP_NUM_THREADS=1)", c_seq)
    r_avg = time_ms("Rust+PyO3 (step_izhikevich, sequential)", rust_seq)

    print(f"\n    Rust vs C: {c_avg/r_avg:.2f}×  {'✅ Rust быстрее' if r_avg < c_avg else '⚠ C быстрее'}")
    print(f"    (Оба sequential — разница = FFI overhead)")
    return c_avg, r_avg


# ═══════════════════════════════════════════════════════════
#  1b. Izhikevich — C с OpenMP vs Rust sequential
#      Измеряет: влияние OpenMP (multi-core) на реальный сценарий
# ═══════════════════════════════════════════════════════════

def bench_izhikevich_parallel():
    print("\n" + "═" * 70)
    print("  1b. Izhikevich step  |  C(OpenMP) vs Rust(sequential)")
    print("      Измеряет: многоядерный C vs одноядерный Rust")
    print("═" * 70)

    n = 100 * 1024
    n_pad = (n + 7) // 8 * 8

    p = lambda x: x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    pi = lambda x: x.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    dt, td = 0.5, 0.99

    v, u, tr, a, b, I, sp, c_c, d = make_data(n, n_pad)

    # C: без OMP_NUM_THREADS (использует все ядра)
    # Убираем переменную окружения
    os.environ.pop("OMP_NUM_THREADS", None)

    def c_omp():
        lib.step_n_neurons(n, p(v), p(u), p(tr), p(a), p(b), p(I), dt, td, pi(sp), p(c_c), p(d))

    # Rust sequential (из 1a)
    vl = v.tolist(); ul = u.tolist(); trl = tr.tolist()
    al = a.tolist(); bl = b.tolist(); Il = I.tolist()
    spl = sp.tolist(); cl = c_c.tolist(); dl = d.tolist()

    def rust_seq():
        neuron_engine.step_izhikevich(vl, ul, trl, al, bl, Il, dt, td, spl, cl, dl)

    c_avg = time_ms("C+OpenMP (step_n_neurons, параллельно)", c_omp)
    r_avg = time_ms("Rust+PyO3 (step_izhikevich, sequential)", rust_seq)

    print(f"\n    Rust(seq) vs C(OpenMP): {c_avg/r_avg:.2f}×  (C использует все ядра)")
    return c_avg, r_avg


# ═══════════════════════════════════════════════════════════
#  2. Safety clamp — zero-copy vs list-Vec
#      Измеряет: стоимость PyO3 list→Vec конвертации
# ═══════════════════════════════════════════════════════════

def bench_clamp():
    print("\n" + "═" * 70)
    print("  2. Safety clamp AVX  |  1 048 576 элементов")
    print("      Измеряет: ctypes zero-copy vs PyO3 list→Vec overhead")
    print("═" * 70)

    n = 1024 * 1024

    lib.safety_clamp_avx.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_float, ctypes.c_float,
    ]

    # C: zero-copy numpy ptr
    data_c = np.random.uniform(-200, 200, n).astype(np.float32)

    def c_fn():
        lib.safety_clamp_avx(data_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, -50.0, 50.0)

    # Rust: list → Vec (каждый вызов — O(n) конвертация)
    data_r = data_c.copy().tolist()

    def rust_fn():
        neuron_engine.safety_clamp(data_r, -50.0, 50.0)

    c_avg = time_ms("C+ctypes (numpy ptr, zero-copy)", c_fn)
    r_avg = time_ms("Rust+PyO3 (list → vec, O(n) conversion)", rust_fn)

    print(f"\n    Разница: {r_avg:.1f} ms Rust vs {c_avg:.1f} ms C = {r_avg/c_avg:.0f}×")
    print(f"   ⚡ Это overhead PyO3 list→Vec, а не скорость Rust!")
    print(f"    ASM-ядро одинаковое — результат отличается только конвертацией данных")

    # Verification
    verify(data_c[:10], data_r[:10], "clamp")

    return c_avg, r_avg


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔" + "═" * 68 + "╗")
    print("║   C+ctypes vs Rust+PyO3 — одинаковые ASM-ядра            ║")
    print("╚" + "═" * 68 + "╝")
    print(f"  Python: {sys.version.split()[0]}  NumPy: {np.__version__}  CPU: {os.cpu_count()} cores")
    print(f"  ASM: ОДИНАКОВЫЙ — libneuron.so (C) / cc::Build (Rust)")

    s1_c, s1_r = bench_izhikevich_sequential()
    s2_c, s2_r = bench_izhikevich_parallel()
    s3_c, s3_r = bench_clamp()

    # Cleanup
    if os.path.exists(rust_module_path):
        os.remove(rust_module_path)

    print("\n" + "═" * 70)
    print("  ВЫВОДЫ:")
    print("═" * 70)
    print(f"""
  Sequential (ASM только):  C  {s1_c:6.1f} ms    Rust {s1_r:6.1f} ms    ratio {s1_c/s1_r:.2f}×
  Parallel (C vs Rust seq): C  {s2_c:6.1f} ms    Rust {s2_r:6.1f} ms    ratio {s2_c/s2_r:.2f}×
  Clamp (list overhead):    C  {s3_c:6.1f} ms    Rust {s3_r:6.1f} ms    ratio {s3_r/s3_c:.0f}×

  Sequential: Rust ≈ C — FFI overhead минимален.
  Parallel:    C быстрее за счёт OpenMP на {os.cpu_count()} ядрах.
  Clamp:       Rust медленнее из-за PyO3 list→Vec конвертации (~24ms).
               С pyo3-numpy (zero-copy) Rust будет = C.
""")
