#!/usr/bin/env python3
"""
⚡ REAL SPIKES CARMACK LOD BENCHMARK
==================================
Тестирование Carmack BSP LOD с РЕАЛЬНЫМИ спайками VNC (HH LUT simulation):
- Спайки генерируются не искусственной волной, а НАСТОЯЩИМ Hodgkin-Huxley симулятором!
- Проверяем, сколько нейронов реально переходит в HH, Izh, LIF и Tanh при естественной спайковой активности.

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
print("⚡ REAL SPIKES CARMACK LOD BENCHMARK: 25,520 VNC Neurons")
print("=" * 76)

# 1. Загрузка коннектома VNC
conn = RealVNCConnectome(vnc_only=True)
conn.load_neurons()
conn.load_connections()

N = conn.n_neurons
print(f"📊 Загружено {N:,} нейронов VNC")

# 2. 3D координаты Октри
positions = np.zeros((N, 3), dtype=np.float32)
rng = np.random.RandomState(42)

for idx in range(N):
    rid = conn.idx_to_root.get(idx)
    meta = conn.neurons.get(rid, {}) if rid else {}
    nerve = meta.get('nerve', '').lower()
    side = meta.get('soma_side', '').lower()

    x_val = rng.uniform(10.0, 180.0) if 'right' in side else rng.uniform(-180.0, -10.0)
    if 'prothoracic' in nerve:
        y_val = rng.uniform(-300.0, -100.0)
    elif 'mesothoracic' in nerve:
        y_val = rng.uniform(-100.0, 100.0)
    elif 'metathoracic' in nerve:
        y_val = rng.uniform(100.0, 300.0)
    else:
        y_val = rng.uniform(-250.0, 250.0)

    sc = meta.get('super_class', '')
    z_val = rng.uniform(20.0, 120.0) if sc == 'motor' else rng.uniform(-120.0, -20.0)
    positions[idx] = [x_val, y_val, z_val]

# 3. Инициализация моделей
hh = tc.hh.HhLut(N, 0.1)
bsp = tc.karmac.CarmackBsp(positions.reshape(-1))

bsp.set_distances(15.0, 60.0, 200.0)
bsp.set_hold_steps(10, 6, 3, 0)
bsp.set_update_stride(5)
bsp.set_activity_thresholds(10, 25)

# Непрерывный надпороговый стимул на 1,000 сенсорных нейронов (I_inj = 30.0 pA)
I_ext = np.zeros(N, dtype=np.float32)
sensory_idx = [i for i in range(N) if conn.neurons.get(conn.idx_to_root.get(i), {}).get('super_class') == 'sensory']
I_ext[sensory_idx[:1000]] = 30.0

hh.set_I_all(I_ext)

print(f"\n🏃 Симуляция 500 шагов с Pure Activity-Based LOD (1,500 сенсоров стимуляции)...")
print("-" * 76)

t0 = time.perf_counter()

total_spikes_fired = 0
speedup_history = []

for step in range(500):
    # 1. Шаг HH спайковой симуляции
    n_spikes = hh.step()
    total_spikes_fired += n_spikes

    # 2. Извлекаем индексы нейронов, выдавших спайк на этом шаге (V > 0 mV)
    hh_v = hh.v
    spiked_indices = np.where(hh_v > 0.0)[0].astype(np.uint64)

    # 3. Dynamic Firing-Rate Spike-Triggered LOD update:
    #    - min_spikes_hh=2 in window=50 steps -> HH (LOD 0)
    #    - min_spikes=1                       -> Izhikevich (LOD 1)
    #    - inactive < 50                      -> LIF (LOD 2)
    #    - silent                             -> Tanh (LOD 3)
    bsp.update_firing_rate_lod(spiked_indices, min_spikes_hh=2, window_steps=50)

    spd = bsp.speedup_factor()
    speedup_history.append(spd)

    if (step + 1) % 100 == 0:
        c = bsp.get_lod_counts()
        print(f"  Step {step+1:3d} | HH Spikes={n_spikes:3d} | "
              f"HH(0)={c[0]:<5d} | Izh(1)={c[1]:<5d} | LIF(2)={c[2]:<5d} | Tanh(3)={c[3]:<5d} | "
              f"⚡ Speedup: {spd:.1f}×")

elapsed = time.perf_counter() - t0

c_final = bsp.get_lod_counts()
avg_speedup = np.mean(speedup_history)

print("\n" + "=" * 76)
print("📊 РЕЗУЛЬТАТЫ РЕАЛЬНОГО CARMACK LOD С HH СПАЙКАМИ")
print("=" * 76)
print(f"⏱️  Время 500 шагов: {elapsed*1000:.1f} мс ({500/elapsed:,.0f} шагов/сек)")
print(f"⚡ Всего спайков сгенерировано HH: {total_spikes_fired:,}")
print(f"\n🧠 Итоговое распределение LOD:")
print(f"   • LOD 0 (HH LUT - 64× точность):      {c_final[0]:>6,} нейронов ({c_final[0]/N*100:.1f}%)")
print(f"   • LOD 1 (Izhikevich - 8× точность):   {c_final[1]:>6,} нейронов ({c_final[1]/N*100:.1f}%)")
print(f"   • LOD 2 (LIF - 2× точность):          {c_final[2]:>6,} нейронов ({c_final[2]/N*100:.1f}%)")
print(f"   • LOD 3 (Tanh - 1× точность):         {c_final[3]:>6,} нейронов ({c_final[3]/N*100:.1f}%)")
print(f"\n🚀 СРЕДНЕЕ УСКОРЕНИЕ ВЫЧИСЛЕНИЙ: {avg_speedup:.1f}× по сравнению с 100% HH!")
print("=" * 76)
