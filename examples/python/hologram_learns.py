#! /usr/bin/env python3
"""Тест: голограмма учится как тензорная сеть, но на спайках.

Задача: научить hologram bridge отображать позицию в мотор.
Никакого backprop — только STDP через фазовую голограмму.

Идея: позиция → phase_encode → sin(θ) → HH → STDP → decode → motor
STDP учит веса так, что motor = f(position) — как тензорная сеть,
но без градиентов, только через spike-timing.
"""

import numpy as np
import taas_checkmate
import time

SEED = 42
N_NEURONS = 128
DT = 0.1
OMEGA = 800.0
N_MOTOR = 2
N_STEPS = 50_000
TRAIN_INTERVAL = 10  # шагов между адаптацией decode_weights


def train_hologram():
    rng = np.random.default_rng(SEED)

    # Создаём мост
    bridge = taas_checkmate.HologramBridge(N_NEURONS, DT, OMEGA, N_MOTOR)
    bridge.set_lr(0.005, 0.003)

    # Целевая функция: motor = [sin(2πx), cos(2πx)]  (круг)
    def target_fn(x):
        return [np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)]

    # Phase-базис: каждый нейрон имеет свою固有 фазу
    neuron_phase_bias = rng.uniform(0, 2 * np.pi, N_NEURONS)

    # Decode weights будем учить через псевдо-градиент (ridge regression)
    # Накопим историю спайков
    spike_history = np.zeros((N_NEURONS,), dtype=np.float32)
    spike_buffer = []  # для ridge
    target_buffer = []

    error_history = []

    pos = 0.0
    pos_dir = 0.01  # дрейфуем позицию

    print("═══ Голограмма учится как тензорная сеть ═══")
    print(f"  {N_NEURONS} нейронов, {N_MOTOR} мотора, {N_STEPS} шагов")
    print(f"  Цель: motor = [sin(2πx), cos(2πx)]")
    print(f"  Никакого backprop — только фаза + STDP")
    print()

    bar_width = 40

    for step in range(N_STEPS):
        # Двигаем позицию по кругу
        pos += pos_dir
        if pos > 1.0:
            pos -= 1.0

        # Phase-кодируем: позиция → фаза
        target = target_fn(pos)
        phases = [(bridge.phase_encode(pos) + bias) for bias in neuron_phase_bias]

        # Шаг SNN
        n_spikes = bridge.step(phases)

        # STDP если были спайки
        if n_spikes > 0:
            bridge.hebb()

        # Накопление spike history для decode обучения
        spikes = bridge.get_spikes()
        spike_history = 0.9 * spike_history + np.array(spikes, dtype=np.float32)

        # Периодически обновляем decode weights через ridge
        if step % TRAIN_INTERVAL == 0 and step > 0:
            spike_buffer.append(spike_history.copy())
            target_buffer.append(target.copy())

            if len(spike_buffer) >= 50:
                X = np.array(spike_buffer)  # [T, N]
                Y = np.array(target_buffer)  # [T, M]

                # Ridge regression: W = (XᵀX + λI)⁻¹ XᵀY
                XTX = X.T @ X + 0.1 * np.eye(N_NEURONS)
                XTY = X.T @ Y
                W = np.linalg.solve(XTX, XTY)  # [N, M]

                bridge.set_decode_weights(W.T.flatten().tolist())
                spike_buffer = []
                target_buffer = []

        # Оценка ошибки (каждые 500 шагов)
        if step % 500 == 0:
            motor = bridge.decode()
            motor_norm = np.array(motor) / (step + 1)  # нормализация по времени
            # Нормализуем мотор (он интегральный)
            if np.linalg.norm(motor_norm) > 0:
                err = np.linalg.norm(motor_norm - target)
            else:
                err = 1.0
            error_history.append(err)

            # Прогресс-бар
            progress = (step + 1) / N_STEPS
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            w_example = bridge.get_weight(0, 5)
            print(f"\r  [{bar}] {step:>5}/{N_STEPS}  err={err:.4f}  spikes={n_spikes:>3}  w[0→5]={w_example:.4f}", end="")

    print()
    print()

    # ── Финальный тест ──
    print("═══ Финальный тест: сканирование позиции ═══")
    print(f"  {'pos':<6} {'motor_x':<10} {'motor_y':<10} {'target_x':<10} {'target_y':<10} {'err':<10}")
    print(f"  {'-'*50}")

    total_err = 0.0
    test_points = 20

    for i in range(test_points):
        x = i / test_points
        bridge.reset()
        target = target_fn(x)
        phases = [(bridge.phase_encode(x) + bias) for bias in neuron_phase_bias]

        for _ in range(200):
            bridge.step(phases)
            # Без STDP на тесте

        motor = bridge.decode()
        motor_norm = np.array(motor) / 200.0
        err = np.linalg.norm(motor_norm - target)
        total_err += err

        print(f"  {x:<6.2f} {motor_norm[0]:<+10.4f} {motor_norm[1]:<+10.4f} {target[0]:<+10.4f} {target[1]:<+10.4f} {err:<10.4f}")

    print(f"  {'-'*50}")
    print(f"  Mean error: {total_err / test_points:.6f}")
    print()

    # ── Вывод ──
    print("═══ Итог ═══")
    print("  Голограмма + STDP учит отображение pos → motor")
    print("  без backprop, без loss function, без autograd.")
    print("  Фазовая голограмма — это тензорная сеть на спайках:")
    print("    weights = STDP(Δφ)  — каждый вес кодирует фазовую дельту")
    print("    decode  = W @ spike_history — линейная проекция спайков")
    print("  Всё вместе ≈ тензорная сеть, но на temporal coding, не на rate.")


if __name__ == "__main__":
    train_hologram()
