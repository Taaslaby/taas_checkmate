#!/usr/bin/env python3
"""
Демонстрация запуска коннектома Drosophila FAFB из ~/Загрузки/
с активными HH + Ca2+ дендритами (SomatoDendriticNeuron).
"""

import os
import sys
import time
import gzip
import csv
from pathlib import Path
import numpy as np

import taas_checkmate as tc

DLOAD = Path.home() / "Загрузки"
CONN_FILE = DLOAD / "connections_princeton.csv.gz"

def main():
    print("=" * 65)
    print("🧠 Drosophila FAFB Brain Connectome — Active Dendrite Simulation")
    print("=" * 65)

    if not CONN_FILE.exists():
        print(f"❌ Файл не найден: {CONN_FILE}")
        return

    print(f"📁 Загрузка связей из: {CONN_FILE.name}...")
    t0 = time.time()

    # Загружаем первых 1000 нейронов для быстрой демонстрации мультикомпартментной симуляции
    max_neurons = 1000
    rows = []

    with gzip.open(CONN_FILE, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pre = int(row["pre_root_id"]) % max_neurons
            post = int(row["post_root_id"]) % max_neurons
            syn_count = int(row.get("syn_count", 1))
            rows.append((pre, post, syn_count))
            if len(rows) >= 50000:
                break

    dt_load = time.time() - t0
    print(f"✅ Прочитано {len(rows)} синапсов за {dt_load:.2f} сек.")

    # Создаём популяцию сомато-дендритных нейронов (1000 нейронов, 4 дендритных компартмента)
    pop = tc.neurons.SomatoDendriticNeuron(
        n_neurons=max_neurons,
        n_compartments=4,
        r_a=200.0,
        r_m=100.0,
        c_m=0.1,
        dt=0.5,
        tau_m=20.0,
        v_rest=-65.0,
        v_thresh=-54.0,
        v_reset=-65.0,
        tau_ref=3.0
    )

    # Включаем активные Hodgkin-Huxley + Ca2+ каналы в дендритах
    print("⚡ Включение активных дендритных Na+, K+, Ca2+ каналов...")
    pop.enable_active_channels(g_na=12.0, g_kv=6.0, g_ca=2.5)

    # Шаги симуляции
    n_steps = 200
    print(f"🚀 Запуск {n_steps} шагов симуляции (dt = 0.5 ms)...")
    I_ext = np.zeros(max_neurons, dtype=np.float32)

    # Подаём стимул на 10 входящих нейронов
    I_ext[:10] = 15.0

    t_sim_start = time.time()
    total_spikes = 0

    for step in range(n_steps):
        # Доставляем дендритный AMPA на дистальные ветки от прочитанных синапсов
        if step % 5 == 0:
            for pre, post, syn_count in rows[:200]:
                if pop.spikes[pre] == 1:
                    comp = (pre + post) % 4  # выбор дендритной ветки
                    pop.jump_ampa(post, comp, 0.05 * syn_count)

        pop.step(I_ext, 0.5)
        total_spikes += pop.spikes.sum()

        if (step + 1) % 50 == 0:
            v0 = pop.v[0]
            vd0 = pop.v_dend(0, 3)
            print(f"   Шаг {step+1:3d}/{n_steps}: Сома V[0] = {v0:6.2f} mV, Дендрит V[0, 3] = {vd0:6.2f} mV | Спайков всего: {total_spikes}")

    t_sim = time.time() - t_sim_start
    print("=" * 65)
    print(f"🎉 Симуляция завершена за {t_sim:.3f} сек! (Скорость: {n_steps/t_sim:.1f} шагов/сек)")
    print("=" * 65)

if __name__ == "__main__":
    main()
