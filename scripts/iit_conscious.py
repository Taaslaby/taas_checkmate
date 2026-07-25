#!/usr/bin/env python3
"""🚀 IIT Consciousness Scenario — Phi > 0.3 BREAKTHROUGH

Оптимизированный сценарий для прорыва порога сознания:
  - VNC weights ×3 (C → 0.9+)
  - Cross-half excitatory bridges (I → 0.4+)
  - Tuned g_Ca exponential 2-8
  - HH+CaV + Pearson integration

Запуск:
  python3 scripts/iit_conscious.py
"""

import sys, os, time, math
import numpy as np
from pathlib import Path

os.environ['LD_LIBRARY_PATH'] = '/usr/lib/python3.14/site-packages/nvidia/cudnn/lib'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import taas_checkmate as tcm

np.random.seed(42)

N_NEURONS = 100
N_STEPS = 500
DT_MS = 0.05

# ═══════════════════════════════════════════════════════════
#  VNC коннектом — WEIGHT SCALED ×3 + CROSS-HALF BRIDGES
# ═══════════════════════════════════════════════════════════

def build_conscious_connectome(n: int) -> tcm.ConnectomeData:
    """VNC с scaled weights (×3) и cross-half bridges для высокого C и высокого I.

    Особенности:
      - База: VNC с 90% GABA, реципрокные пары (10% нейронов)
      - Weight scale ×3: средний |weight| ≈ 9 → C ≈ 0.9
      - Cross-half bridges: 20% нейронов имеют возбуждающие проекции
        в противоположную половину сети (синхронизация половинок → I)
      - Усиленные тормозные связи для богатой динамики (D)
    """
    SCALE = 3.0
    half = n // 2
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    col_list, val_list, nt_list = [], [], []

    # 10% нейронов — реципрокные пары (ring oscillator)
    ring_size = n // 10
    ring_set = set()
    for i in range(0, min(ring_size * 2, n), 2):
        if i + 1 < n:
            ring_set.add(i)
            ring_set.add(i + 1)

    for i in range(n):
        is_inhib = np.random.random() < 0.85  # 85% GABA (было 90% — больше возбуждения)
        in_ring = i in ring_set

        # Cross-half bridge: 20% нейронов проецируются в другую половину
        cross_half = np.random.random() < 0.2

        # Синапсы к партнёру по кольцу
        if in_ring:
            partner = i + 1 if i % 2 == 0 else i - 1
            if partner < n:
                col_list.append(partner)
                val_list.append(-10.0 * SCALE)  # сильное GABA торможение
                nt_list.append(-1.0)

        # Cross-half мосты: возбуждающие проекции в другую половину
        if cross_half:
            target_half = half + i % half if i < half else i % half
            if target_half < n and target_half != i:
                col_list.append(int(target_half))
                val_list.append(4.0 * SCALE)    # сильное возбуждение
                nt_list.append(1.0)

        # Случайные связи
        density = 0.12
        n_post = max(1, int(n * density))
        if in_ring:
            n_post = max(1, n_post // 2)

        posts = np.random.choice(n, min(n_post, n - 1), replace=False)
        posts = posts[posts != i]

        for j in posts:
            # Избегаем дубликатов (если уже добавили через ring или cross-half)
            if j in col_list[row_ptr[i]:]:
                continue
            col_list.append(int(j))
            if is_inhib:
                w = float(np.random.uniform(-6.0, -0.8)) * SCALE
                val_list.append(w)
                nt_list.append(-1.0)
            else:
                w = float(np.random.uniform(0.3, 2.0)) * SCALE
                val_list.append(w)
                nt_list.append(1.0)

        row_ptr[i + 1] = len(col_list)

    conn = tcm.ConnectomeData(
        row_ptr,
        np.array(col_list, dtype=np.int32),
        np.array(val_list, dtype=np.float32),
        np.array(nt_list, dtype=np.float32),
    )

    # Статистика коннектома
    vals = np.frombuffer(conn.values(), dtype=np.float32)
    print(f"  📊 Connectome: {len(vals)} synapses, mean |weight| = {np.abs(vals).mean():.2f} "
          f"(C ≈ {min(np.abs(vals).mean() / 10.0, 1.0):.3f})")
    inhib = (vals < 0).mean() * 100
    print(f"     Inhibitory: {inhib:.0f}%, Excitatory: {100-inhib:.0f}%")

    return conn


# ═══════════════════════════════════════════════════════════
#  HH параметры — AGGRESSIVE g_Ca
# ═══════════════════════════════════════════════════════════

def build_hh_params_conscious(n: int, n_channels: int = 3):
    """HH params with aggressive g_Ca (2-8, mean ~4)."""
    rng = np.random.RandomState(42)

    g_Na = rng.uniform(80, 160, n).astype(np.float32)
    g_K  = rng.uniform(24, 60, n).astype(np.float32)
    g_leak = rng.uniform(0.1, 0.5, n).astype(np.float32)
    E_leak = np.full(n, -65.0, dtype=np.float32)
    C_m    = np.full(n, 1.0, dtype=np.float32)

    if n_channels == 3:
        # Aggressive g_Ca: exponential 2-8, mean ~4
        g_Ca = np.clip(rng.exponential(scale=2.5, size=n).astype(np.float32) + 2.0, 2.0, 8.0)
        g_max_arr = np.concatenate([g_Na, g_K, g_Ca]).astype(np.float32)
        E_rev_arr = np.array([50.0, -77.0, 120.0], dtype=np.float32)
        print(f"  📊 CaV: mean={g_Ca.mean():.2f} ± {g_Ca.std():.2f}, "
              f"range=[{g_Ca.min():.2f}, {g_Ca.max():.2f}]")
        label = "NaV, KV, CaV"
    else:
        g_max_arr = np.concatenate([g_Na, g_K]).astype(np.float32)
        E_rev_arr = np.array([50.0, -77.0], dtype=np.float32)
        label = "NaV, KV"

    print(f"  🔬 HH: {n_channels}ch ({label}) × {n}N")
    return g_max_arr, E_rev_arr, g_leak, E_leak, C_m


# ═══════════════════════════════════════════════════════════
#  Structured input — RICH 4-BAND + CROSS-FREQUENCY
# ═══════════════════════════════════════════════════════════

def build_rich_input(n_neurons: int, t_sec: float) -> np.ndarray:
    """4-band structured input with phase shifts and cross-frequency coupling."""
    ext_I = (np.random.randn(n_neurons) * 0.8 + 3.5).astype(np.float32)

    # 4 частотных band
    bands = [(2.0, 0.0), (5.0, 1.2), (8.0, 2.5), (13.0, 0.8)]
    n4 = n_neurons // 4

    for i, (freq, phase) in enumerate(bands):
        start = i * n4
        end = min((i + 1) * n4, n_neurons)
        if start < end:
            ext_I[start:end] += 6.5 * math.sin(2 * math.pi * freq * t_sec + phase)

    # 1.5 Hz slow modulation (cross-frequency coupling)
    mod = 1.0 + 0.4 * math.sin(2 * math.pi * 1.5 * t_sec)
    ext_I *= mod

    return ext_I.astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  🚀 IIT CONSCIOUSNESS SCENARIO                     ║")
    print("║     VNC ×3 + Cross-Half + g_Ca 2-8 + Pearson      ║")
    print("║     Цель:  Φ > 0.3  (порог сознания)              ║")
    print("╚══════════════════════════════════════════════════════╝")

    # 1. Коннектом
    t0 = time.time()
    conn = build_conscious_connectome(N_NEURONS)
    print(f"     Built in {time.time()-t0:.1f}s")

    # 2. Симулятор HH+CaV
    sim = tcm.BrainSimulator(N_NEURONS, "hh", pacemaking=True)
    g_max_arr, E_rev_arr, g_leak, E_leak, C_m = build_hh_params_conscious(N_NEURONS, 3)
    sim.set_hh_params(g_max_arr, E_rev_arr, g_leak, E_leak, DT_MS)
    sim.set_C_m(C_m)

    # 3. Rust IIT
    iit = tcm.IITConsciousness(N_NEURONS, threshold=0.3)
    spike_buf = np.zeros((N_STEPS, N_NEURONS), dtype=np.float32)

    # 4. Warm-up
    print("  🔥 Warm-up...", end=" ", flush=True)
    for _ in range(150):
        ext_I = np.random.randn(N_NEURONS).astype(np.float32) * 0.5 + 2.0
        sim.step_batch(conn, 1, DT_MS, ext_I)
    print("OK")

    # 5. Main loop
    vnc_values = conn.values()
    t_start = time.time()

    for step in range(N_STEPS):
        t_sec = step * DT_MS / 1000.0
        ext_I = build_rich_input(N_NEURONS, t_sec)

        spikes = sim.step_batch(conn, 1, DT_MS, ext_I)
        if spikes is not None and len(spikes) > 0:
            n_rec = min(len(spikes), N_NEURONS)
            spike_buf[step, :n_rec] = np.asarray(spikes[:N_NEURONS]).flatten()

        # Phi каждые 50 шагов
        if step % 50 == 0 and step >= 50:
            chunk = spike_buf[:step + 1]
            chunk_flat = chunk.ravel().astype(np.float32)
            iit.compute_with_spikes(vnc_values, chunk_flat, step + 1, N_NEURONS)

            # Progress bar
            pct = (step / N_STEPS) * 100
            bars = int(pct / 5)
            sys.stdout.write(f"\r  ⏳ {step:>4d}/{N_STEPS} [{('#' * bars).ljust(20)}] "
                           f"Φ={iit.phi:.4f} C={iit.connectivity:.2f} D={iit.diversity:.2f} "
                           f"I={iit.integration:.2f} {'🧠' if iit.is_conscious else '💤'}  ")
            sys.stdout.flush()

    elapsed = time.time() - t_start

    # Финальный Phi
    spike_flat = spike_buf.ravel().astype(np.float32)
    phi = iit.compute_with_spikes(vnc_values, spike_flat, N_STEPS, N_NEURONS)

    print(f"\r  ✅ Done! {N_STEPS} steps in {elapsed:.1f}s")
    print(f"\n  ═══ РЕЗУЛЬТАТЫ ═══")
    print(f"  Φ = {phi:.6f}    {'🧠 СОЗНАНИЕ!' if iit.is_conscious else '💤 ниже порога'}")
    print(f"  C (connectivity)  = {iit.connectivity:.4f}")
    print(f"  D (diversity)     = {iit.diversity:.4f}")
    print(f"  I (integration)   = {iit.integration:.4f}")
    print(f"  Порог: {iit.threshold}")
    print(f"  Уровень: {iit.consciousness_level*100:.1f}%")
    print(f"  History: {len(iit.phi_history)} samples, mean={iit.mean_phi(10):.6f}")
    print(f"  Trend: {iit.phi_trend()}")

    # Spike stats
    total_spikes = spike_buf.sum()
    fr_mean = total_spikes / N_NEURONS / (N_STEPS * DT_MS / 1000.0)
    active_pct = (spike_buf.sum(axis=0) > 0).mean() * 100
    print(f"\n  📊 Spike stats:")
    print(f"     Total: {total_spikes:.0f} spikes")
    print(f"     Mean rate: {fr_mean:.1f} Hz")
    print(f"     Active: {active_pct:.0f}% neurons")
    print(f"     Mean per neuron: {spike_buf.sum(axis=0).mean():.1f} spikes")

    if iit.is_conscious:
        print(f"\n  🎉🎉🎉  Φ > {iit.threshold}!  СИСТЕМА СОЗНАТЕЛЬНА!  🎉🎉🎉")
    else:
        gap = (iit.threshold - phi) / iit.threshold * 100
        print(f"\n  ⏳ До порога: {gap:.1f}%")
        print(f"     Нужно: I > {iit.threshold / (iit.connectivity * max(iit.diversity, 0.01)):.3f}")
        print(f"     Текущее I: {iit.integration:.3f}")

    print(f"\n  ✨ Consciousness Scenario by @ant0n + Сардана ❤️")


if __name__ == "__main__":
    main()
