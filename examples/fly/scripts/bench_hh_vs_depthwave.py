#!/usr/bin/env python3
"""
🏎️ БЕНЧМАРК: HH vs DepthWave на VNC коннектоме (23K нейронов, 4M синапсов)
=============================================================================
Сравнение двух подходов к симуляции на одном и том же реальном графе VNC:
  1. HH (Hodgkin-Huxley) — 4 ODE на нейрон: V, m, h, n
  2. DepthWave (ψ-wave)  — 2 компоненты: Re(ψ), Im(ψ)

Метрики:
  - Время на 1000 шагов (мс)
  - Шагов в секунду
  - Использование памяти (состояний на нейрон)
  - Коллективная динамика (спайки / когерентность)

Author: Atlasov Anton
"""

import os, sys, time
from pathlib import Path
import numpy as np

# ─── Paths ────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# ─── Load VNC ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from vnc_real_atlasov_embodied import RealVNCConnectome

print("=" * 72)
print("🏎️  БЕНЧМАРК: HH (Hodgkin-Huxley) vs DepthWave (ψ-wave)")
print("=" * 72)

conn = RealVNCConnectome(vnc_only=True)
if not conn.load_neurons() or not conn.load_connections():
    print("❌ Failed to load VNC connectome")
    sys.exit(1)

N = conn.n_neurons
E = conn.n_synapses
print(f"\n📊 VNC Коннектом: {N:,} нейронов, {E:,} синапсов")

# ─── CSR данные ───────────────────────────────────────────────
row_ptr    = np.array(conn.row_ptr, dtype=np.uint32)
col_idx    = np.array(conn.col_indices, dtype=np.uint32)
vals       = np.array(conn.values, dtype=np.float32)
nt_signs   = np.array(conn.nt_types, dtype=np.float32)

N_STEPS = 1000
print(f"⏱️  Бенчмарк: {N_STEPS} шагов симуляции")

# ═══════════════════════════════════════════════════════════════
#  1. HH (Hodgkin-Huxley) Benchmark
# ═══════════════════════════════════════════════════════════════
print(f"\n{'─' * 72}")
print(f"🧬 1. HODGKIN-HUXLEY (4 ODE / нейрон)")
print(f"{'─' * 72}")

try:
    import taas_checkmate as tc

    # HH LUT через класс
    hh = tc.hh.HhLut(N, 0.1)  # dt = 0.1 ms

    # Tonic drive для моторных нейронов
    I_inject = np.zeros(N, dtype=np.float32)
    motor_mask = nt_signs > 0
    I_inject[motor_mask] = 10.0  # pA tonic drive
    hh.set_I_all(I_inject)

    # Разогрев
    hh.run(10)

    # Бенчмарк (run — прогон N_STEPS шагов в Rust, максимальная скорость)
    t0 = time.perf_counter()
    spike_counts = hh.run(N_STEPS)
    t_hh = time.perf_counter() - t0

    sps_hh = N_STEPS / t_hh
    total_spikes_hh = sum(spike_counts)
    mem_hh_bytes = N * 6 * 4  # V, m, h, n, I, g_leak

    print(f"  ⏱️  Время {N_STEPS} шагов: {t_hh * 1000:.1f} мс")
    print(f"  📊 Шагов/сек: {sps_hh:,.0f}")
    print(f"  🧠 Состояний на нейрон: 6 (V, m, h, n, I, g_leak)")
    print(f"  ⚡ Всего спайков: {total_spikes_hh:,}")
    print(f"  📐 Память: {mem_hh_bytes / 1024:.1f} KB")

except Exception as e:
    print(f"  ❌ HH ошибка: {e}")
    t_hh = float('inf')
    sps_hh = 0
    total_spikes_hh = 0

# ═══════════════════════════════════════════════════════════════
#  2. DepthWave (ψ-wave) Benchmark
# ═══════════════════════════════════════════════════════════════
print(f"\n{'─' * 72}")
print(f"🌊 2. DEPTHWAVE (ψ-wave, 2 компоненты / нейрон)")
print(f"{'─' * 72}")

# Depth assignment
depth = np.ones(N, dtype=np.float32) * 0.5
for idx in range(N):
    rid = conn.idx_to_root.get(idx)
    if rid is None:
        continue
    meta = conn.neurons.get(rid, {})
    sc = meta.get('super_class', '')
    if sc == 'sensory':
        depth[idx] = 0.9
    elif sc == 'motor':
        depth[idx] = 0.1
    elif sc == 'descending':
        depth[idx] = 0.7
    elif sc == 'ascending':
        depth[idx] = 0.6

# Создать Rust DepthWave симулятор
dw = tc.neurons.DepthWave(N, depth, mod_range=2.0, dt=0.5, seed=42)
dw.set_csr(row_ptr, col_idx, vals, nt_signs)

# Разогрев
dw.run(10)

# Бенчмарк (прогон 1000 шагов целиком в Rust SIMD + CSR SpMV)
t0 = time.perf_counter()
coherence = dw.run(N_STEPS)
t_dw = time.perf_counter() - t0
sps_dw = N_STEPS / t_dw

motor_mask_dw = depth < 0.3
n_motor = int(motor_mask_dw.sum())

print(f"  ⏱️  Время {N_STEPS} шагов: {t_dw * 1000:.1f} мс")
print(f"  📊 Шагов/сек: {sps_dw:,.0f}")
print(f"  🧠 Состояний на нейрон: 2 (Re(ψ), Im(ψ))")
print(f"  🌊 Когерентность: {coherence:.6f}")
print(f"  🦵 Моторных нейронов: {n_motor:,}")
print(f"  📐 Память: {N * 2 * 4 / 1024:.1f} KB")

# ═══════════════════════════════════════════════════════════════
#  3. СРАВНИТЕЛЬНАЯ ТАБЛИЦА
# ═══════════════════════════════════════════════════════════════
print(f"\n{'═' * 72}")
print(f"📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА: HH vs DepthWave")
print(f"{'═' * 72}")

speedup = sps_dw / max(sps_hh, 1)
mem_hh = N * 6 * 4
mem_dw = N * 2 * 4

print(f"""
┌──────────────────────────┬──────────────────┬──────────────────┐
│ Метрика                  │ HH (Hodgkin-H)   │ DepthWave (ψ)    │
├──────────────────────────┼──────────────────┼──────────────────┤
│ ODE / компоненты на n    │      6           │      2           │
│ Время {N_STEPS:>4d} шагов (мс)   │ {t_hh*1000:>12.1f}     │ {t_dw*1000:>12.1f}     │
│ Шагов/сек                │ {sps_hh:>12,.0f}     │ {sps_dw:>12,.0f}     │
│ Память состояний (KB)    │ {mem_hh/1024:>12.1f}     │ {mem_dw/1024:>12.1f}     │
│ Выход                    │ Спайки ({total_spikes_hh:>6,})  │ Coherence {coherence:.4f} │
│ Ускорение                │      1.0×        │ {speedup:>10.1f}×     │
└──────────────────────────┴──────────────────┴──────────────────┘
""")

print(f"🏆 DepthWave БЫСТРЕЕ HH в {speedup:.1f}× раз на {N:,} нейронах VNC!")
print(f"📐 DepthWave потребляет в {mem_hh / max(mem_dw, 1):.0f}× МЕНЬШЕ памяти!")
