#!/usr/bin/env python3
"""Load saved connectome + bridge and run chase inference with visualization.

Загрузка:
  chase_brain.taas — коннектом (LIF нейроны)
  chase_brain_weights.npz — веса bridge + decoder

Запуск: python3 examples/python/run_connectome_chase.py
"""

import numpy as np
import taas_checkmate as tcm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import time
import os

# ─── Загрузка из единого .taas ──────────────────────────
print('Loading model from chase_brain.taas...')
import json
data = tcm.load_taas('chase_brain.taas')
conn = data.get('connectome') or tcm.load_connectome('chase_brain.taas')
n_neurons = conn.n_neurons
print(f'  {n_neurons} neurons, {conn.n_synapses} synapses')

meta = data.get('_meta', {})
bridge_w = np.array(data['bridge_w'], dtype=np.float32)
bridge_b = np.array(data['bridge_b'], dtype=np.float32)
decode_w = np.array(data['decoder_w'], dtype=np.float32).reshape(-1, 2)
decode_b = np.array(data['decoder_b'], dtype=np.float32)
readout_idx = np.array(json.loads(meta.get('readout_idx', '[]')), dtype=np.int32)
n_osc = int(meta.get('n_osc', '4'))
dt = float(meta.get('dt', '0.05'))
drive_gain = float(meta.get('drive_gain', '12.0'))
alpha = float(meta.get('alpha', '2.0'))
freq_min = float(meta.get('freq_min', '0.5'))
freq_max = float(meta.get('freq_max', '3.0'))
coupling_w12 = float(meta.get('coupling_w12', '0.3'))
coupling_w13 = float(meta.get('coupling_w13', '0.15'))
coupling_w23 = float(meta.get('coupling_w23', '0.08'))
spike_dim = len(readout_idx)
pst_dim = 18 * n_osc + 12
print(f'  T3: n_osc={n_osc}, loaded bridge+decoder from .taas')

# ─── Восстанавливаем bridge ────────────────────────────
bridge = tcm.SpigradBridge(
    spike_dim=spike_dim, n_osc=n_osc, dt=dt,
    drive_gain=drive_gain, alpha=alpha,
    freq_min=freq_min, freq_max=freq_max,
    coupling_w12=coupling_w12, coupling_w13=coupling_w13,
    coupling_w23=coupling_w23, zero_init=False,
)

# ─── LIF ───────────────────────────────────────────────
sim = tcm.connectome_brain.BrainSimulator(n_neurons, 'lif')
tau_m = np.full(n_neurons, 12.0, dtype=np.float32)
v_rest = np.full(n_neurons, -65.0, dtype=np.float32)
v_thresh = np.full(n_neurons, -50.0, dtype=np.float32)
v_reset = np.full(n_neurons, -70.0, dtype=np.float32)

rng = np.random.default_rng()
target = np.zeros(2, dtype=np.float32)
target_vel = np.zeros(2, dtype=np.float32)
agent_trail = np.zeros((40, 2))
target_trail = np.zeros((40, 2))
spike_hist = np.zeros(100)

# ─── Графика ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5),
    gridspec_kw={'width_ratios': [1.2, 0.8, 0.8]})
ax_chase, ax_spikes, ax_info = axes

ax_chase.set_xlim(-2.2, 2.2); ax_chase.set_ylim(-2.2, 2.2)
ax_chase.set_aspect('equal')
ax_chase.set_title('Connectome chasing target!', fontsize=13, fontweight='bold')
ax_chase.grid(True, alpha=0.2)
ax_chase.set_facecolor('#f0f5ff')

agent = Circle((0, 0), 0.15, color='#2196F3', alpha=0.9, label='Agent')
target_c = Circle((0, 0), 0.12, color='#F44336', alpha=0.9, label='Target')
ax_chase.add_patch(agent); ax_chase.add_patch(target_c)
agent_trail_l, = ax_chase.plot([], [], '-', color='#2196F3', alpha=0.25, lw=2)
target_trail_l, = ax_chase.plot([], [], '-', color='#F44336', alpha=0.25, lw=2)
info_t = ax_chase.text(0.05, 0.95, '', transform=ax_chase.transAxes, va='top',
    fontsize=10, family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax_chase.legend(loc='lower right', fontsize=9)

ax_spikes.set_xlim(0, 100); ax_spikes.set_ylim(0, 1)
ax_spikes.set_title('Firing rate (100ep moving avg)')
ax_spikes.set_xlabel('Episode')
fr_line, = ax_spikes.plot([], [], '-', color='#9C27B0', lw=1.5)

episodes_run = 0
N_STEPS = 60
DT_MS = 2.0

def animate(frame):
    global episodes_run, target, target_vel
    
    external_I = np.zeros(n_neurons, dtype=np.float32)
    external_I[:100] = rng.uniform(10, 40, 100).astype(np.float32)
    
    spikes_flat = sim.step_batch_gpu(conn, n_steps=N_STEPS, dt_ms=DT_MS,
        external_I=external_I, tau_m=tau_m, v_rest=v_rest,
        v_thresh=v_thresh, v_reset=v_reset)
    spikes = spikes_flat.reshape(N_STEPS, n_neurons)
    spike_in = spikes[:, readout_idx].astype(np.float32)
    
    target_vel += rng.normal(0, 0.06, 2).astype(np.float32)
    target_vel *= 0.92
    target += target_vel * 0.15
    target = np.clip(target, -1.8, 1.8)
    
    pst_mat, _ = bridge.forward(spike_in, N_STEPS)
    pos = pst_mat @ decode_w + decode_b
    
    loss = float(np.mean(0.5 * np.sum((pos - target) ** 2, axis=1)))
    dist = float(np.linalg.norm(pos[-1] - target))
    fr = spikes.mean()
    
    target_trail[:] = np.roll(target_trail, 1, 0); target_trail[0] = target
    agent_trail[:] = np.roll(agent_trail, 1, 0); agent_trail[0] = pos[-1]
    
    agent.center = (pos[-1, 0], pos[-1, 1])
    target_c.center = (target[0], target[1])
    agent_trail_l.set_data(agent_trail[:, 0], agent_trail[:, 1])
    target_trail_l.set_data(target_trail[:, 0], target_trail[:, 1])
    
    episodes_run += 1
    spike_hist[:-1] = spike_hist[1:]; spike_hist[-1] = fr
    fr_line.set_data(range(100), spike_hist)
    
    info_t.set_text(f'Ep {episodes_run}  Loss {loss:.3f}  Dist {dist:.3f}\nFR {fr:.3f}')
    
    return agent, target_c, agent_trail_l, target_trail_l, fr_line, info_t

print('\nRunning inference... Close window to stop.')
ani = FuncAnimation(fig, animate, interval=30, cache_frame_data=False, blit=False)
plt.tight_layout()
plt.show()
