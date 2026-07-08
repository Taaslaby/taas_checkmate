#!/usr/bin/env python3
"""CHASE DEMO: T3 oscillators learn to chase a moving target via BPTT.

Спайки → SpigradBridge → T3 осцилляторы → PST → линейный readout → 2D позиция.
Bridge учится через BPTT так, чтобы T3 гнал агента за целью.

Запуск: python3 examples/python/chase_demo.py
"""

import numpy as np
import taas_checkmate as tcm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import time

# ─── Параметры ─────────────────────────────────────────
N_OSC = 4
N_STEPS = 50
DT = 0.05
SPIKE_DIM = 8
LR_BRIDGE = 0.003
LR_DECODER = 0.015
EPOCHS = 500
GRAD_CLIP = 5.0

pst_dim = 18 * N_OSC + 12
rng = np.random.default_rng()

bridge = tcm.SpigradBridge(
    spike_dim=SPIKE_DIM, n_osc=N_OSC, dt=DT,
    drive_gain=12.0, alpha=2.0, freq_min=0.5, freq_max=3.0,
    coupling_w12=0.3, coupling_w13=0.15, coupling_w23=0.08,
    zero_init=False,
)

# ─── Состояние ─────────────────────────────────────────
state = {
    'decode_w': rng.normal(0, 0.1, (pst_dim, 2)).astype(np.float32),
    'decode_b': np.zeros(2, dtype=np.float32),
    'target': np.zeros(2, dtype=np.float32),
    'target_vel': np.zeros(2, dtype=np.float32),
    'agent_trail': np.zeros((40, 2)),
    'target_trail': np.zeros((40, 2)),
    'loss_hist': np.zeros(EPOCHS, dtype=np.float32),
    'episode': 0,
    't0': time.time(),
}

# ─── Фигуры ────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 0.8])

ax0 = fig.add_subplot(gs[0])
ax0.set_xlim(-2.2, 2.2)
ax0.set_ylim(-2.2, 2.2)
ax0.set_aspect('equal')
ax0.set_title('T3 Chases the Target!', fontsize=13, fontweight='bold')
ax0.grid(True, alpha=0.2)
ax0.set_facecolor('#f0f5ff')

agent = Circle((0, 0), 0.15, color='#2196F3', alpha=0.9, label='Agent')
target_circle = Circle((0, 0), 0.12, color='#F44336', alpha=0.9, label='Target')
ax0.add_patch(agent)
ax0.add_patch(target_circle)
agent_trail_line, = ax0.plot([], [], '-', color='#2196F3', alpha=0.25, lw=2)
target_trail_line, = ax0.plot([], [], '-', color='#F44336', alpha=0.25, lw=2)
info = ax0.text(0.05, 0.95, '', transform=ax0.transAxes, va='top',
                fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax0.legend(loc='lower right', fontsize=9)

ax1 = fig.add_subplot(gs[1])
ax1.set_xlim(0, EPOCHS)
ax1.set_ylim(0.001, 10)
ax1.set_yscale('log')
ax1.set_title('Loss (log)', fontsize=12)
ax1.grid(True, alpha=0.3, which='both')
loss_line, = ax1.plot([], [], '-', color='#4CAF50', lw=1.5)
best_line = ax1.axhline(y=1, color='orange', ls='--', alpha=0.5, label='Best')
target_loss = ax1.axhline(y=0.01, color='red', ls=':', alpha=0.5, label='Target')
ax1.legend(fontsize=8)

# ─── Панель осцилляторов ──────────────────────────────
ax2 = fig.add_subplot(gs[2])
ax2.set_xlim(0, 1)
ax2.set_ylim(0, N_OSC + 1)
ax2.set_title('T3 Oscillators', fontsize=12)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_facecolor('#1a1a2e')

osc_bars = []
osc_labels = []
for i in range(N_OSC):
    bar = ax2.barh(i + 0.3, 0.5, height=0.4, color=plt.cm.plasma(i / N_OSC))
    osc_bars.append(bar)
    osc_labels.append(ax2.text(0.02, i + 0.3, f'θ{i}', color='white', fontsize=7, va='center'))

osc_text = ax2.text(0.5, -0.3, '', transform=ax2.transAxes,
                    ha='center', fontsize=8, color='#888')

plt.tight_layout()


def update_target():
    s = state
    s['target_vel'] += rng.normal(0, 0.06, 2).astype(np.float32)
    s['target_vel'] *= 0.92
    s['target'] += s['target_vel'] * 0.15
    s['target'] = np.clip(s['target'], -2.0, 2.0)


def animate(frame):
    s = state
    s['episode'] += 1

    # ── Спайки ──
    spikes = rng.uniform(0, 1, (N_STEPS, SPIKE_DIM)).astype(np.float32)
    update_target()

    # ── Forward ──
    pst_mat, traj = bridge.forward(spikes, N_STEPS)
    pos = pst_mat @ s['decode_w'] + s['decode_b']

    # ── Loss ──
    err = pos - s['target']
    loss = float(np.mean(0.5 * np.sum(err ** 2, axis=1)))
    s['loss_hist'][s['episode'] % EPOCHS] = loss

    # ── Backward ──
    per_step = err.astype(np.float32) / N_STEPS
    per_step_pst = per_step @ s['decode_w'].T
    grad_w, grad_b = bridge.backward(
        np.zeros(pst_dim, dtype=np.float32), traj, spikes,
        per_step_grad_pst=per_step_pst)
    bridge.update(np.clip(grad_w, -GRAD_CLIP, GRAD_CLIP),
                  np.clip(grad_b, -GRAD_CLIP, GRAD_CLIP), LR_BRIDGE)

    # ── Декодер ──
    s['decode_w'] -= LR_DECODER * (pst_mat.T @ per_step).astype(np.float32)
    s['decode_b'] -= LR_DECODER * per_step.mean(axis=0)

    # ── Хвосты ──
    s['target_trail'] = np.roll(s['target_trail'], 1, 0); s['target_trail'][0] = s['target']
    s['agent_trail'] = np.roll(s['agent_trail'], 1, 0); s['agent_trail'][0] = pos[-1]

    # ── Графика ──
    agent.center = (pos[-1, 0], pos[-1, 1])
    target_circle.center = (s['target'][0], s['target'][1])
    agent_trail_line.set_data(s['agent_trail'][:, 0], s['agent_trail'][:, 1])
    target_trail_line.set_data(s['target_trail'][:, 0], s['target_trail'][:, 1])

    ep = min(s['episode'], EPOCHS)
    loss_line.set_data(range(ep), s['loss_hist'][:ep])
    best_loss = float(min(s['loss_hist'][:ep]))
    best_line.set_ydata([best_loss])
    dist = float(np.linalg.norm(pos[-1] - s['target']))
    elapsed = time.time() - s['t0']
    info.set_text(
        f'Ep {s["episode"]}  Loss {loss:.4f}  Best {best_loss:.5f}\n'
        f'Dist {dist:.3f}  Time {elapsed:.0f}s')

    # ── Осцилляторы ──
    r1 = np.array(traj.r1_pre())[-N_OSC:] if hasattr(traj, 'r1_pre') else np.zeros(N_OSC)
    for i in range(N_OSC):
        val = float(np.clip(r1[i] / 5.0, 0, 1))
        osc_bars[i][0].set_width(val)
        osc_bars[i][0].set_color(plt.cm.plasma(val))

    if s['episode'] >= EPOCHS:
        ani.event_source.stop()
        print(f'\nDone! {EPOCHS} episodes in {elapsed:.0f}s, '
              f'best loss={best_loss:.5f}, final dist={dist:.3f}')
        if dist < 0.1:
            print('🎯 Agent learned to chase!')

    return (agent, target_circle, agent_trail_line, target_trail_line,
            loss_line, info, *[b[0] for b in osc_bars])


print('🏃 T3 Chase Demo')
print(f'  {N_OSC} oscillators, {N_STEPS} steps/ep, {EPOCHS} episodes')
print(f'  LR bridge={LR_BRIDGE} decoder={LR_DECODER}')
print('  Close window to stop.')

ani = FuncAnimation(fig, animate, interval=40, cache_frame_data=False, blit=False)
plt.show()
