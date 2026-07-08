#!/usr/bin/env python3
"""ASCII CHASE: T3 oscillators chase a target in terminal.

Запуск без графики: python3 examples/python/chase_ascii.py
Работает в любом терминале, даже по SSH.
"""

import numpy as np
import taas_checkmate as tcm
import time
import os

N_OSC = 4
N_STEPS = 40
DT = 0.05
SPIKE_DIM = 6
LR_BRIDGE = 0.003
LR_DECODER = 0.015
EPOCHS = 300
GRAD_CLIP = 5.0

pst_dim = 18 * N_OSC + 12
rng = np.random.default_rng()

bridge = tcm.SpigradBridge(
    spike_dim=SPIKE_DIM, n_osc=N_OSC, dt=DT,
    drive_gain=12.0, alpha=2.0, freq_min=0.5, freq_max=3.0,
    coupling_w12=0.3, coupling_w13=0.15, coupling_w23=0.08,
    zero_init=False,
)

decode_w = rng.normal(0, 0.1, (pst_dim, 2)).astype(np.float32)
decode_b = np.zeros(2, dtype=np.float32)
target = np.zeros(2, dtype=np.float32)
target_vel = np.zeros(2, dtype=np.float32)

COLS = 50
ROWS = 20

def clamp(v, lo, hi): return max(lo, min(hi, v))

def render(ax, ay, tx, ty, loss, dist, ep):
    cols, rows = COLS, ROWS
    grid = [[' ' for _ in range(cols)] for _ in range(rows)]

    # границы
    for x in range(cols):
        grid[0][x] = grid[rows-1][x] = '-'
    for y in range(rows):
        grid[y][0] = grid[y][cols-1] = '|'
    grid[0][0] = grid[0][cols-1] = grid[rows-1][0] = grid[rows-1][cols-1] = '+'

    def to_grid(x, y):
        gx = int((x + 2.0) / 4.0 * (cols - 2))
        gy = int((2.0 - y) / 4.0 * (rows - 2))
        return clamp(gx, 1, cols-2), clamp(gy, 1, rows-2)

    gtx, gty = to_grid(tx, ty)
    gax, gay = to_grid(ax, ay)

    if grid[gty][gtx] != ' ':
        grid[gty][gtx] = 'X'
    else:
        grid[gty][gtx] = 'T'

    if grid[gay][gax] == 'T':
        grid[gay][gax] = '!'
    elif grid[gay][gax] != ' ':
        grid[gay][gax] = 'X'
    else:
        grid[gay][gax] = 'A'

    os.system('clear')
    for row in grid:
        print(''.join(row))
    print(f' Ep {ep:>3}  Loss {loss:.4f}  Dist {dist:.3f}  Best {best_loss:.5f}')
    print(f' {"A=agent":<12} {"T=target":<12} {"!=CATCH!":<12}')


best_loss = float('inf')
t0 = time.time()

for ep in range(EPOCHS):
    # спайки
    spikes = rng.uniform(0, 1, (N_STEPS, SPIKE_DIM)).astype(np.float32)

    # цель — случайное блуждание
    target_vel += rng.normal(0, 0.06, 2).astype(np.float32)
    target_vel *= 0.92
    target += target_vel * 0.15
    target = np.clip(target, -1.8, 1.8)

    # forward
    pst_mat, traj = bridge.forward(spikes, N_STEPS)
    pos = pst_mat @ decode_w + decode_b
    err = pos - target
    loss = float(np.mean(0.5 * np.sum(err ** 2, axis=1)))
    best_loss = min(best_loss, loss)

    # backward
    per_step = err.astype(np.float32) / N_STEPS
    per_step_pst = per_step @ decode_w.T
    grad_w, grad_b = bridge.backward(
        np.zeros(pst_dim, dtype=np.float32), traj, spikes,
        per_step_grad_pst=per_step_pst)
    bridge.update(np.clip(grad_w, -GRAD_CLIP, GRAD_CLIP),
                  np.clip(grad_b, -GRAD_CLIP, GRAD_CLIP), LR_BRIDGE)
    decode_w -= LR_DECODER * (pst_mat.T @ per_step).astype(np.float32)
    decode_b -= LR_DECODER * per_step.mean(axis=0)

    ax, ay = pos[-1, 0], pos[-1, 1]
    tx, ty = target[0], target[1]
    dist = float(np.sqrt((ax - tx)**2 + (ay - ty)**2))
    render(ax, ay, tx, ty, loss, dist, ep)
    time.sleep(0.03)

elapsed = time.time() - t0
print(f'\nDone! {EPOCHS} eps in {elapsed:.0f}s, best loss={best_loss:.5f}')
if dist < 0.15:
    print('🎯 CAUGHT THE TARGET!')
elif dist < 0.3:
    print('🔥 Agent is chasing!')
else:
    print('Still learning... try more epochs')
