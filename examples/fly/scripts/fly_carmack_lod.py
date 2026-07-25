#!/usr/bin/env python3
"""
🕹️ CARMACK BSP LOD (LEVEL-OF-DETAIL) ON FLY VNC CONNECTOME
============================================================
Когнитивный движок Carmack-style BSP (по аналогии с отсечением видимых поверхностей в Doom):

  - Спайковый фронт (0-10 µm)  → LOD 0: HH LUT (64× точность)
  - Около фронта    (10-50 µm) → LOD 1: Izhikevich (8×)
  - Спокойная зона  (50-200 µm)→ LOD 2: LIF (2×)
  - Фоновый шум     (>200 µm)  → LOD 3: Tanh (1×)

Динамическое распределение вычислений по Октри (Octree O(N log N)) с вероятностным удержанием (hold counters).

Author: Atlasov Anton
"""

import os, sys, time
from pathlib import Path
import numpy as np

PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

sys.path.insert(0, str(Path(__file__).parent))
from vnc_real_atlasov_embodied import RealVNCConnectome
import taas_checkmate as tc

print("=" * 76)
print("🕹️  CARMACK BSP LOD ENGINE: 25,520 VNC Fly Neurons")
print("=" * 76)

# 1. Загрузка VNC коннекнома
conn = RealVNCConnectome(vnc_only=True)
conn.load_neurons()
conn.load_connections()

N = conn.n_neurons
print(f"📊 Загружено {N:,} нейронов VNC")

# 2. Построение 3D координат (x, y, z в µm) по нейроанатомии VNC ганглиев
# T1 (Prothoracic): y in [-300, -100]
# T2 (Mesothoracic): y in [-100, 100]
# T3 (Metathoracic): y in [100, 300]
positions = np.zeros((N, 3), dtype=np.float32)
rng = np.random.RandomState(42)

for idx in range(N):
    rid = conn.idx_to_root.get(idx)
    meta = conn.neurons.get(rid, {}) if rid else {}
    nerve = meta.get('nerve', '').lower()
    side = meta.get('soma_side', '').lower()

    # X: Left vs Right
    x_val = rng.uniform(10.0, 180.0) if 'right' in side else rng.uniform(-180.0, -10.0)

    # Y: Leg segment (Pro / Meso / Meta)
    if 'prothoracic' in nerve:
        y_val = rng.uniform(-300.0, -100.0)
    elif 'mesothoracic' in nerve:
        y_val = rng.uniform(-100.0, 100.0)
    elif 'metathoracic' in nerve:
        y_val = rng.uniform(100.0, 300.0)
    else:
        y_val = rng.uniform(-250.0, 250.0)

    # Z: Motor (dorsal > 0) vs Sensory (ventral < 0)
    sc = meta.get('super_class', '')
    z_val = rng.uniform(20.0, 120.0) if sc == 'motor' else rng.uniform(-120.0, -20.0)

    positions[idx] = [x_val, y_val, z_val]

print("✅ 3D Октри координаты VNC построены (диапазон ±300 µm)")

# 3. Инициализация Carmack BSP LOD менеджера
bsp = tc.karmac.CarmackBsp(positions.reshape(-1))
bsp.set_distances(15.0, 60.0, 200.0)   # LOD 0: 15µm, LOD 1: 60µm, LOD 2: 200µm
bsp.set_hold_steps(12, 8, 4, 0)        # Задержка понижения точности (Hold steps)
bsp.set_update_stride(10)              # ⚡ Оптимизация 1: пересчитываем Octree LOD каждые 10 шагов
bsp.set_activity_thresholds(15, 35)    # ⚡ Оптимизация 2: Activity-Based LOD (неактивные нейроны -> LIF/Tanh)

counts = bsp.get_lod_counts()
print(f"\n📊 Начальный LOD статус: HH={counts[0]}, Izh={counts[1]}, LIF={counts[2]}, Tanh={counts[3]}")
print(f"🚀 Начальное ускорение (Speedup Factor): {bsp.speedup_factor():.1f}×")

# 4. Имитация динамического прохождения волны спайков через T1 -> T2 -> T3
print("\n🎬 Имитация движения волны активности через VNC ганглии (300 шагов)...")
print("-" * 76)

t0 = time.perf_counter()

# Каждые 50 шагов фронт перемещается по Y координате
for step in range(300):
    # Положение активного фронта спайков (от T1 -250µm до T3 +250µm)
    y_wavefront = -250.0 + (step / 300.0) * 500.0
    
    # Находим нейроны в узкой полосе спайкового фронта (±15 µm)
    spiking_neurons = np.where(np.abs(positions[:, 1] - y_wavefront) < 15.0)[0]
    
    # Обновляем Carmack BSP Octree
    bsp.update_lod(spiking_neurons.astype(np.uint64))

    if (step + 1) % 50 == 0:
        c = bsp.get_lod_counts()
        spd = bsp.speedup_factor()
        print(f"  Step {step+1:3d} (Wavefront Y={y_wavefront:+5.1f}µm) | "
              f"HH(0)={c[0]:<4d} | Izh(1)={c[1]:<4d} | LIF(2)={c[2]:<5d} | Tanh(3)={c[3]:<5d} | "
              f"⚡ Speedup: {spd:.1f}×")

elapsed = time.perf_counter() - t0

# 5. Итоговый отчет
c_final = bsp.get_lod_counts()
spd_final = bsp.speedup_factor()

print("\n" + "=" * 76)
print("🏆 ИТОГИ РАБОТЫ CARMACK BSP LOD НА VNC МУХИ")
print("=" * 76)
print(f"⏱️  Время 300 обновлений Octree (N={N:,}): {elapsed*1000:.1f} мс ({300/elapsed:,.0f} шагов/сек)")
print(f"🧠 Итоговый расклад по уровню деталей:")
print(f"   • LOD 0 (HH LUT - 64× точность):      {c_final[0]:>6,} нейронов ({c_final[0]/N*100:.1f}%)")
print(f"   • LOD 1 (Izhikevich - 8× точность):   {c_final[1]:>6,} нейронов ({c_final[1]/N*100:.1f}%)")
print(f"   • LOD 2 (LIF - 2× точность):          {c_final[2]:>6,} нейронов ({c_final[2]/N*100:.1f}%)")
print(f"   • LOD 3 (Tanh - 1× точность):         {c_final[3]:>6,} нейронов ({c_final[3]/N*100:.1f}%)")
print(f"\n🚀 СРЕДНЕЕ УСКОРЕНИЕ ВЫЧИСЛЕНИЙ: {spd_final:.1f}× по сравнению с 100% HH!")
print("=" * 76)
