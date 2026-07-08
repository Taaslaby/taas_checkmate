#!/usr/bin/env python3
"""End-to-end SPIGRAD bridge demo: spikes → bridge → T3 → PST → motor → loss → backward.

Мост (SpigradBridge) учится проецировать случайные спайки на драйвы T3 так,
чтобы T3 генерировал PST-фичи, полезные для линейной декодировки motor-команд.

Запуск:
    cd taas_checkmate && python3 examples/python/spigrad_bridge_demo.py
"""

import numpy as np
import taas_checkmate as tcm

N_OSC = 6
N_STEPS = 40
SPIKE_DIM = 16
DT = 0.05
N_MOTOR = 2
LR_BRIDGE = 0.003
LR_DECODER = 0.01
EPOCHS = 200
GRAD_CLIP = 5.0

rng = np.random.default_rng(42)

# ─── Bridge ────────────────────────────────────────────
bridge = tcm.SpigradBridge(
    spike_dim=SPIKE_DIM, n_osc=N_OSC, dt=DT,
    drive_gain=10.0, alpha=1.0, freq_min=0.5, freq_max=3.0,
    coupling_w12=0.2, coupling_w13=0.1, coupling_w23=0.05,
    zero_init=False,
)

# ─── Decoder (Ridge init, then fine-tune via gradient) ─
pst_dim = 18 * N_OSC + 12
decode_w = rng.normal(0, 0.05, (pst_dim, N_MOTOR)).astype(np.float32)
decode_b = np.zeros(N_MOTOR, dtype=np.float32)

# ─── Спайки + цель ────────────────────────────────────
spikes = rng.uniform(0, 1, (N_STEPS, SPIKE_DIM)).astype(np.float32)
target = np.array([1.0, 0.0], dtype=np.float32)

print(f"SPIGRAD bridge end-to-end")
print(f"  Osc: {N_OSC}  Steps: {N_STEPS}  Spike dim: {SPIKE_DIM}")
print(f"  Drive dim: {bridge.drive_dim()}  PST dim: {pst_dim}")
print(f"  Target: {target}")
print()

best_loss = float('inf')
for epoch in range(EPOCHS):
    # ── Forward ──
    pst_mat, traj = bridge.forward(spikes, N_STEPS)
    pst = pst_mat[-1]  # используем последний шаг
    motor = pst @ decode_w + decode_b
    err = motor - target
    loss = float(0.5 * np.sum(err ** 2))
    if loss < best_loss:
        best_loss = loss

    # ── Backward ──
    grad_motor = err
    grad_pst = grad_motor @ decode_w.T  # (pst_dim,)

    grad_w, grad_b = bridge.backward(grad_pst, traj, spikes)

    # Clip градиентов для стабильности
    gw_clip = np.clip(grad_w, -GRAD_CLIP, GRAD_CLIP)
    gb_clip = np.clip(grad_b, -GRAD_CLIP, GRAD_CLIP)
    gm_clip = np.clip(grad_motor, -GRAD_CLIP, GRAD_CLIP)

    # Обновляем decoder (обычный SGD)
    decode_w -= LR_DECODER * np.outer(pst, gm_clip)
    decode_b -= LR_DECODER * gm_clip

    # Обновляем bridge через SPIGRAD градиент
    bridge.update(gw_clip, gb_clip, LR_BRIDGE)

    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        print(f"  epoch {epoch:>3}  loss={loss:.6f}  "
              f"motor=[{motor[0]:+.4f}, {motor[1]:+.4f}]  "
              f"|∇W|={np.linalg.norm(np.array(grad_w)):.2f}")

print(f"\nBest loss: {best_loss:.6f}")
print(f"Final motor: [{motor[0]:+.4f}, {motor[1]:+.4f}]  target=[{target[0]:.1f}, {target[1]:.1f}]")
print(f"\n✅ SPIGRAD bridge учится проецировать спайки → T3 → motor")
