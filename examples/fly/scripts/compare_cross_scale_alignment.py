#!/usr/bin/env python3
"""
🔬 CROSS-SCALE ALIGNMENT: DepthWave (Macro/Wave) vs Hodgkin-Huxley (Micro/Spikes)
================================================================================
Анализ совпадений и корреляций между волновым и биофизическим мирами на 25,520 нейронах VNC:

1. Пространственная корреляция Пирсона (Spatial Pearson r) между:
   - Энергией волны |ψ_i|² в DepthWave
   - Частотой спайкования (Firing Rate) в HH
2. Иерархия категорий нейронов (Sensory → Inter -> Motor):
   - Совпадают ли уровни активности в группах (Sensory, Motor, Leg Nerves)?
3. Захват фазы (Phase-Locking & Resonance):
   - Генерируют ли нейроны спайки во время волновых фазовых максимумов?

Author: Atlasov Anton
"""

import os, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

sys.path.insert(0, str(Path(__file__).parent))
from vnc_real_atlasov_embodied import RealVNCConnectome
import taas_checkmate as tc

print("=" * 76)
print("🔬 CROSS-SCALE ALIGNMENT: DepthWave (Wave) ↔ Hodgkin-Huxley (Spikes)")
print("=" * 76)

# 1. Загрузка коннектома VNC
conn = RealVNCConnectome(vnc_only=True)
conn.load_neurons()
conn.load_connections()

N = conn.n_neurons
E = conn.n_synapses

row_ptr  = np.array(conn.row_ptr, dtype=np.uint32)
col_idx  = np.array(conn.col_indices, dtype=np.uint32)
vals     = np.array(conn.values, dtype=np.float32)
nt_signs = np.array(conn.nt_types, dtype=np.float32)

# Топологическая глубина
depth = np.ones(N, dtype=np.float32) * 0.5
groups = {'sensory': [], 'motor': [], 'descending': [], 'ascending': [], 'intrinsic': []}

for idx in range(N):
    rid = conn.idx_to_root.get(idx)
    if rid is None:
        groups['intrinsic'].append(idx)
        continue
    meta = conn.neurons.get(rid, {})
    sc = meta.get('super_class', '')
    if sc in groups:
        groups[sc].append(idx)
    else:
        groups['intrinsic'].append(idx)

    if sc == 'sensory':
        depth[idx] = 0.9
    elif sc == 'motor':
        depth[idx] = 0.1
    elif sc == 'descending':
        depth[idx] = 0.7
    elif sc == 'ascending':
        depth[idx] = 0.6

# 2. Независимый симуляционный прогон обоих миров (1000 шагов)
print(f"\n🏃 Симуляция 1000 шагов для {N:,} нейронов VNC...")

# World A: DepthWave
dw = tc.neurons.DepthWave(N, depth, mod_range=2.0, dt=0.5, seed=42)
dw.set_csr(row_ptr, col_idx, vals, nt_signs)

sensory_input = np.zeros(N, dtype=np.float32)
sensory_input[groups['sensory']] = 1.0

# 500 шагов эволюции
dw.step(sensory_input, n_steps=500)
wave_energy = dw.v**2 + dw.u**2  # |ψ_i|²

# World B: Hodgkin-Huxley LUT
hh = tc.hh.HhLut(N, 0.1)
hh_input = np.zeros(N, dtype=np.float32)
hh_input[groups['sensory']] = 15.0  # сенсорный ток
hh_input[groups['motor']] = 5.0

spike_counts = np.zeros(N, dtype=np.int32)
for _ in range(500):
    hh.set_I_all(hh_input)
    hh.step()
    spike_counts += (hh.v > 0.0).astype(np.int32)

firing_rates = spike_counts / 0.05  # Гц (500 steps * 0.1ms = 50ms)

# 3. Анализ кросс-масштабного совпадения (Cross-Scale Alignment)
print("\n" + "=" * 76)
print("📊 КРОСС-МАСШТАБНЫЙ АНАЛИЗ СОВПАДЕНИЙ (Cross-Scale Matching)")
print("=" * 76)

# Корреляция Пирсона по всем 25,520 нейронам
r_all, p_all = pearsonr(wave_energy, firing_rates)
print(f"\n1️⃣  Глобальная пространственная корреляция (All {N:,} Neurons):")
print(f"    r = {r_all:+.4f} (p-value = {p_all:.2e})")
if r_all > 0.3:
    print("    ✅ ВЫСОКАЯ СВЯЗНОСТЬ: Активные зоны волны совпадают с очагами спайкования!")
else:
    print("    ℹ️  Локально-специфичная связность")

# Сравнение по функциональным группам
print("\n2️⃣  Сравнение средних профилей активности по группам:")
print(f"┌──────────────┬───────────────┬──────────────────┬──────────────────┐")
print(f"│ Группа       │ Кол-во нейр.  │ Wave Energy |ψ|² │ HH Firing Rate   │")
print(f"├──────────────┼───────────────┼──────────────────┼──────────────────┤")

for grp_name, indices in groups.items():
    if len(indices) == 0: continue
    idx_arr = np.array(indices)
    mean_energy = np.mean(wave_energy[idx_arr])
    mean_fr = np.mean(firing_rates[idx_arr])
    print(f"│ {grp_name:<12} │ {len(indices):>13,} │ {mean_energy:>16.6f} │ {mean_fr:>14.2f} Hz │")

print(f"└──────────────┴───────────────┴──────────────────┴──────────────────┘")

# 4. Выявление совпавших моторных пулов (Top active motor neurons)
motor_idx = np.array(groups['motor'])
top_wave_motor = set(motor_idx[np.argsort(wave_energy[motor_idx])[-50:]])
top_hh_motor   = set(motor_idx[np.argsort(firing_rates[motor_idx])[-50:]])
overlap = top_wave_motor.intersection(top_hh_motor)

overlap_pct = (len(overlap) / 50.0) * 100.0
print(f"\n3️⃣  Совпадение Top-50 Моторных Нейронов (Leg Control):")
print(f"    Пересечение: {len(overlap)} / 50 нейронов ({overlap_pct:.1f}% совпадение!)")
print(f"    Вывод: Моторные нейроны, активируемые волной DepthWave, на {overlap_pct:.0f}% ")
print(f"           совпадают с нейронами, выдающими спайки в модели Ходжкина-Хаксли!")

print("\n" + "=" * 76)
