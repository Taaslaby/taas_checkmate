#!/usr/bin/env python3
"""🧠 IIT через Rust крейт — HH+CaV (3ch) на VNC-коннектоме

Использует Pearson correlation для Integration (как оригинальный Python IIT).
Tuned g_Ca и structured input для максимального Phi.

Сравнивает HH 2ch vs HH+CaV 3ch.

Запуск:
  python3 scripts/iit_rust.py --neurons 100 --steps 200 --cav
"""

import sys, os, time, math, argparse
import numpy as np
from pathlib import Path

os.environ['LD_LIBRARY_PATH'] = '/usr/lib/python3.14/site-packages/nvidia/cudnn/lib'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import taas_checkmate as tcm

np.random.seed(42)

N_NEURONS = 100
N_STEPS = 300
DT_MS = 0.05

# ═══════════════════════════════════════════════════════════
#  VNC коннектом
# ═══════════════════════════════════════════════════════════

def build_vnc(n: int, density: float = 0.12) -> tcm.ConnectomeData:
    """GABA-доминантный VNC коннектом с реципрокными парами (ring oscillator)."""
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    col_list, val_list, nt_list = [], [], []

    # Реципрокные пары (ring oscillator) — 5% нейронов
    ring_size = n // 20
    ring_set = set()
    for i in range(0, min(ring_size * 2, n), 2):
        if i + 1 < n:
            ring_set.add(i)
            ring_set.add(i + 1)

    for i in range(n):
        is_inhib = np.random.random() < 0.9
        in_ring = i in ring_set

        if in_ring:
            partner = i + 1 if i % 2 == 0 else i - 1
            if partner < n:
                col_list.append(partner)
                val_list.append(-8.0)    # сильное GABA торможение
                nt_list.append(-1.0)
            n_post = max(1, int(n * density * 0.5))
        else:
            n_post = max(1, int(n * density))

        posts = np.random.choice(n, min(n_post, n - 1), replace=False)
        posts = posts[posts != i]

        for j in posts:
            col_list.append(int(j))
            if is_inhib:
                val_list.append(float(np.random.uniform(-6.0, -0.8)))
                nt_list.append(-1.0)
            else:
                val_list.append(float(np.random.uniform(0.3, 2.0)))
                nt_list.append(1.0)

        row_ptr[i + 1] = len(col_list)

    return tcm.ConnectomeData(
        row_ptr,
        np.array(col_list, dtype=np.int32),
        np.array(val_list, dtype=np.float32),
        np.array(nt_list, dtype=np.float32),
    )


# ═══════════════════════════════════════════════════════════
#  HH параметры — с TUNED g_Ca
# ═══════════════════════════════════════════════════════════

def build_hh_params(n: int, n_channels: int = 2, rng=None):
    """Build HH per-neuron params: 2ch (NaV, KV) or 3ch (NaV, KV, CaV).

    g_Ca tuned:
      - диапазон 1.0–6.0 (было 0.5–2.0)
      - распределение скошено вправо (больше нейронов с сильным CaV)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    g_Na = rng.uniform(80, 160, n).astype(np.float32)
    g_K  = rng.uniform(24, 60, n).astype(np.float32)
    g_leak = rng.uniform(0.1, 0.5, n).astype(np.float32)
    E_leak = np.full(n, -65.0, dtype=np.float32)
    C_m    = np.full(n, 1.0, dtype=np.float32)

    if n_channels == 3:
        # g_Ca: скошенное распределение 1.0–6.0 (больше сильных CaV нейронов)
        g_Ca = np.clip(rng.exponential(scale=2.0, size=n).astype(np.float32) + 1.0, 1.0, 6.0)
        g_max_arr = np.concatenate([g_Na, g_K, g_Ca]).astype(np.float32)
        E_rev_arr = np.array([50.0, -77.0, 120.0], dtype=np.float32)
        label = "NaV, KV, CaV"
        print(f"  📊 CaV: mean={g_Ca.mean():.2f} ± {g_Ca.std():.2f}, range=[{g_Ca.min():.2f}, {g_Ca.max():.2f}]")
    else:
        g_max_arr = np.concatenate([g_Na, g_K]).astype(np.float32)
        E_rev_arr = np.array([50.0, -77.0], dtype=np.float32)
        label = "NaV, KV"

    print(f"  🔬 HH: {n_channels}ch ({label}) × {n}N")
    return g_max_arr, E_rev_arr, g_leak, E_leak, C_m


# ═══════════════════════════════════════════════════════════
#  Structured input — rich multi-frequency stimulation
# ═══════════════════════════════════════════════════════════

def build_structured_input(n_neurons: int, t_sec: float) -> np.ndarray:
    """Rich structured input: 4 frequency bands, phase-shifted."""
    ext_I = (np.random.randn(n_neurons) * 1.0 + 3.0).astype(np.float32)

    f1, f2, f3, f4 = 2.0, 5.0, 8.0, 13.0   # Hz — theta, alpha, beta, low-gamma
    ph1, ph2, ph3, ph4 = 0.0, 1.2, 2.5, 0.8  # phase shifts

    n3 = n_neurons // 3
    n4 = n_neurons // 4

    # Group 1 (0..n4): 2 Hz theta
    ext_I[:n4] += 6.0 * math.sin(2 * math.pi * f1 * t_sec + ph1)

    # Group 2 (n4..2*n4): 5 Hz alpha
    ext_I[n4:2*n4] += 6.0 * math.sin(2 * math.pi * f2 * t_sec + ph2)

    # Group 3 (2*n4..3*n4): 8 Hz beta
    if 3 * n4 <= n_neurons:
        ext_I[2*n4:3*n4] += 6.0 * math.sin(2 * math.pi * f3 * t_sec + ph3)

    # Group 4 (3*n4..): 13 Hz low-gamma
    ext_I[3*n4:] += 6.0 * math.sin(2 * math.pi * f4 * t_sec + ph4)

    # Cross-frequency coupling: amplitude modulation
    mod = 1.0 + 0.5 * math.sin(2 * math.pi * 1.5 * t_sec)  # 1.5 Hz slow modulation
    ext_I *= mod

    return ext_I.astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  IIT через Rust крейт — PEARSON integration
# ═══════════════════════════════════════════════════════════

def run_iit_rust(label: str, conn, n_neurons: int, n_steps: int,
                 pacemaking: bool = False):
    """Run IIT using Rust tcm.IITConsciousness WITH PEARSON integration.

    Ключевое отличие: Integration = Pearson correlation между половинами нейронов,
    а не CV of rates. Это даёт правильную метрику интеграции для CaV pacemaking.
    """
    nch = 3 if pacemaking else 2
    print(f"\n  ═══ {label} (HH {nch}ch{' +CaV' if pacemaking else ''}) ═══")

    # 1. Симулятор HH
    sim = tcm.BrainSimulator(n_neurons, "hh", pacemaking=pacemaking)
    g_max_arr, E_rev_arr, g_leak, E_leak, C_m = build_hh_params(n_neurons, nch)
    sim.set_hh_params(g_max_arr, E_rev_arr, g_leak, E_leak, DT_MS)
    sim.set_C_m(C_m)

    # 2. Rust IIT engine
    iit = tcm.IITConsciousness(n_neurons, threshold=0.3)
    spike_buf = np.zeros((n_steps, n_neurons), dtype=np.float32)

    # 3. Прогрев (без IIT)
    print("  🔥 Warm-up...", end=" ", flush=True)
    for _ in range(100):
        ext_I = np.random.randn(n_neurons).astype(np.float32) * 0.5 + 2.0
        sim.step_batch(conn, 1, DT_MS, ext_I)
    print("OK")

    # 4. Основной цикл
    t0 = time.time()
    vnc_values = conn.values()  # numpy array

    for step in range(n_steps):
        t_sec = step * DT_MS / 1000.0
        ext_I = build_structured_input(n_neurons, t_sec)

        spikes = sim.step_batch(conn, 1, DT_MS, ext_I)
        if spikes is not None and len(spikes) > 0:
            n_rec = min(len(spikes), n_neurons)
            spike_buf[step, :n_rec] = np.asarray(spikes[:n_neurons]).flatten()

        # Phi каждые 50 шагов — используем Pearson integration
        if step % 50 == 0 and step >= 50:
            chunk = spike_buf[:step + 1]
            chunk_flat = chunk.ravel().astype(np.float32)
            iit.compute_with_spikes(vnc_values, chunk_flat, step + 1, n_neurons)

    elapsed = time.time() - t0

    # Финальный Phi — на всех шагах
    spike_flat = spike_buf.ravel().astype(np.float32)
    phi = iit.compute_with_spikes(vnc_values, spike_flat, n_steps, n_neurons)

    print(f"  ⏱  {elapsed:.1f}s")
    print(f"  Φ = {phi:.6f}  |  C={iit.connectivity:.4f}  D={iit.diversity:.4f}  I={iit.integration:.4f}")
    print(f"  🧠 Conscious: {iit.is_conscious} ({iit.consciousness_level*100:.1f}%)  |  {iit.phi_trend()}")
    print(f"  📈 History: {len(iit.phi_history)} samples, Φ_mean={iit.mean_phi(10):.6f}")

    return iit, spike_buf


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="🧠 IIT Rust: HH+CaV на VNC-коннектоме")
    p.add_argument("--neurons", type=int, default=N_NEURONS)
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--cav", action="store_true", help="Сравнить HH 2ch vs HH+CaV 3ch")
    args = p.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║  🧠 IIT через Rust крейт — PEARSON integration     ║")
    print("║     Tuned g_Ca (1-6) + 4-band structured input     ║")
    print("╚══════════════════════════════════════════════════════╝")

    # VNC коннектом
    t0 = time.time()
    conn = build_vnc(args.neurons)
    print(f"\n  📦 VNC коннектом: {args.neurons}N, {time.time()-t0:.1f}s")

    if args.cav:
        iit_2, buf_2 = run_iit_rust("HH 2ch", conn, args.neurons, args.steps, pacemaking=False)
        iit_3, buf_3 = run_iit_rust("HH+CaV", conn, args.neurons, args.steps, pacemaking=True)

        print("\n" + "═" * 55)
        print("  📊 СРАВНЕНИЕ: HH 2ch vs HH+CaV (Pearson integration)")
        print("═" * 55)

        phi_2 = iit_2.mean_phi(10)
        phi_3 = iit_3.mean_phi(10)
        ratio = phi_3 / max(phi_2, 1e-8)

        print(f"  HH 2ch   Φ: {phi_2:.6f}  C={iit_2.connectivity:.4f} D={iit_2.diversity:.4f} I={iit_2.integration:.4f}")
        print(f"  HH+CaV   Φ: {phi_3:.6f}  C={iit_3.connectivity:.4f} D={iit_3.diversity:.4f} I={iit_3.integration:.4f}")
        print(f"  Ratio: {ratio:.2f}x")

        # Spike statistics
        fr_2 = buf_2.sum(axis=0).mean() / (args.steps * DT_MS / 1000.0)
        fr_3 = buf_3.sum(axis=0).mean() / (args.steps * DT_MS / 1000.0)
        active_2 = (buf_2.sum(axis=0) > 0).mean() * 100
        active_3 = (buf_3.sum(axis=0) > 0).mean() * 100
        print(f"\n  📊 Spike stats:")
        print(f"  HH 2ch:   {buf_2.sum():.0f} spikes, {fr_2:.1f} Hz mean rate, {active_2:.0f}% active")
        print(f"  HH+CaV:   {buf_3.sum():.0f} spikes, {fr_3:.1f} Hz mean rate, {active_3:.0f}% active")

        if ratio > 1.2:
            print(f"\n  ✅ CaV ПОВЫШАЕТ Phi ×{ratio:.2f}! Pacemaking обогащает интеграцию ✨🔥")
        elif ratio > 0.8:
            print(f"\n  ➡ CaV ≈ 2ch (×{ratio:.2f}) — разница незначительна")
        else:
            print(f"\n  ⚠ CaV даёт меньше Phi (×{ratio:.2f}) — возможно g_Ca маловат")
    else:
        iit, buf = run_iit_rust("HH", conn, args.neurons, args.steps, pacemaking=False)
        fr = buf.sum(axis=0).mean() / (args.steps * DT_MS / 1000.0)
        active = (buf.sum(axis=0) > 0).mean() * 100
        print(f"\n  📊 {buf.sum():.0f} total spikes, {fr:.1f} Hz mean rate, {active:.0f}% active")

    print(f"\n  ✨ Rust IIT engine by @ant0n + Сардана ❤️")


if __name__ == "__main__":
    main()
