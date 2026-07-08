#!/usr/bin/env python3
"""CONNECTOME CHASE: спайковый мозг (LIF connectome) → Bridge → T3 → чейз.

Создаём синтетический коннектом, гоняем на GPU, спайки → SpigradBridge → T3 → PST → 2D.
Bridge учится проецировать нейронную активность в осмысленное управление T3 для преследования цели.

Запуск: python3 examples/python/connectome_chase.py
"""

"""CONNECTOME CHASE: спайковый мозг (LIF connectome) → Bridge → T3 → чейз.

Создаём синтетический коннектом, гоняем на GPU, спайки → SpigradBridge → T3 → PST → 2D.
Bridge учится проецировать нейронную активность в осмысленное управление T3 для преследования цели.

Запуск: python3 examples/python/connectome_chase.py
"""

import numpy as np
import taas_checkmate as tcm
import time
import os

NPZ = os.environ.get('NPZ', 'chase_brain_weights.npz')
TAAS = os.environ.get('TAAS', 'chase_brain.taas')

# ─── Параметры ─────────────────────────────────────────
N_NEURONS = int(os.environ.get('N_NEURONS', '1000'))
N_STEPS = int(os.environ.get('N_STEPS', '60'))
DT_MS = 2.0
DT = 0.05
N_OSC = 4
SPIKE_DIM = int(os.environ.get('SPIKE_DIM', '16'))
LR_BRIDGE = float(os.environ.get('LR_BRIDGE', '0.003'))
LR_DECODER = float(os.environ.get('LR_DECODER', '0.015'))
EPOCHS = int(os.environ.get('EPOCHS', '400'))
GRAD_CLIP = 5.0
CONN_DENSITY = float(os.environ.get('CONN_DENSITY', '0.03'))

pst_dim = 18 * N_OSC + 12
rng = np.random.default_rng()

def make_connectome(n_neurons, density, seed=42):
    rng = np.random.default_rng(seed)
    row_ptr = np.zeros(n_neurons + 1, dtype=np.int32)
    col_indices, values = [], []
    nt_types = rng.choice([-1, 1], size=n_neurons).astype(np.float32)
    for i in range(n_neurons):
        n_syn = max(1, int(rng.poisson(n_neurons * density)))
        targets = rng.choice(n_neurons, size=min(n_syn, n_neurons), replace=False)
        col_indices.extend(targets.astype(np.int32))
        w = rng.normal(1.0, 0.4, size=len(targets)).astype(np.float32) * nt_types[i]
        values.extend(w)
        row_ptr[i + 1] = row_ptr[i] + len(targets)
    return tcm.connectome_brain.ConnectomeData(
        row_ptr, np.array(col_indices, dtype=np.int32),
        np.array(values, dtype=np.float32), nt_types, [])

print(f'Creating connectome ({N_NEURONS} neurons)...')
conn = make_connectome(N_NEURONS, CONN_DENSITY)
print(f'  {conn.n_neurons} neurons, {conn.n_synapses} synapses')

v_init = np.full(N_NEURONS, -60.0, dtype=np.float32)
tau_m = np.full(N_NEURONS, 12.0, dtype=np.float32)
v_rest = np.full(N_NEURONS, -65.0, dtype=np.float32)
v_thresh = np.full(N_NEURONS, -50.0, dtype=np.float32)
v_reset = np.full(N_NEURONS, -70.0, dtype=np.float32)
external_I = np.zeros(N_NEURONS, dtype=np.float32)

sim = tcm.connectome_brain.BrainSimulator(N_NEURONS, 'lif')

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
readout_idx = np.sort(rng.choice(N_NEURONS, SPIKE_DIM, replace=False))

print(f'\nTraining bridge on connectome spikes...')
print(f'  {N_NEURONS} LIF → {SPIKE_DIM} readout → Bridge → {N_OSC} T3 → chase')
print(f'  {EPOCHS} episodes, {N_STEPS} steps/episode\n')

best_loss, t0 = float('inf'), time.time()

for ep in range(EPOCHS):
    # ── Обновляем внешний шум (сенсорный вход меняется) ──
    external_I[:50] = rng.uniform(10, 40, 50).astype(np.float32)
    external_I[50:100] = rng.uniform(0, 20, 50).astype(np.float32)

    # ── Запускаем коннектом на GPU → спайки ──
    spikes_flat = sim.step_batch_gpu(conn, n_steps=N_STEPS, dt_ms=DT_MS,
        external_I=external_I, tau_m=tau_m, v_rest=v_rest,
        v_thresh=v_thresh, v_reset=v_reset)
    spikes = spikes_flat.reshape(N_STEPS, N_NEURONS)

    # ── Выбираем readout нейроны как вход для bridge ──
    spike_in = spikes[:, readout_idx].astype(np.float32)  # (N_STEPS, SPIKE_DIM)

    # ── Цель: случайное блуждание ──
    target_vel += rng.normal(0, 0.06, 2).astype(np.float32)
    target_vel *= 0.92
    target += target_vel * 0.15
    target = np.clip(target, -1.8, 1.8)

    # ── Forward ──
    pst_mat, traj = bridge.forward(spike_in, N_STEPS)
    pos = pst_mat @ decode_w + decode_b

    # ── Loss ──
    err = pos - target
    loss = float(np.mean(0.5 * np.sum(err ** 2, axis=1)))
    best_loss = min(best_loss, loss)

    # ── Backward ──
    per_step = err.astype(np.float32) / N_STEPS
    per_step_pst = per_step @ decode_w.T
    grad_w, grad_b = bridge.backward(
        np.zeros(pst_dim, dtype=np.float32), traj, spike_in,
        per_step_grad_pst=per_step_pst)
    bridge.update(np.clip(grad_w, -GRAD_CLIP, GRAD_CLIP),
                  np.clip(grad_b, -GRAD_CLIP, GRAD_CLIP), LR_BRIDGE)

    # ── Декодер ──
    decode_w -= LR_DECODER * (pst_mat.T @ per_step).astype(np.float32)
    decode_b -= LR_DECODER * per_step.mean(axis=0)

    if ep % 50 == 0 or ep == EPOCHS - 1:
        dist = float(np.linalg.norm(pos[-1] - target))
        fr = spikes.mean()
        active = (spikes.sum(axis=0) > 0).sum()
        print(f'  ep {ep:>3}  loss={loss:.5f}  best={best_loss:.5f}  '
              f'dist={dist:.3f}  FR={fr:.3f}  active={active}')

elapsed = time.time() - t0
dist = float(np.linalg.norm(pos[-1] - target))
print(f'\nDone! {EPOCHS} eps in {elapsed:.0f}s')
print(f'Best loss: {best_loss:.5f}  Final dist: {dist:.3f}')
if dist < 0.2:
    print('🎯 Connectome + Bridge learned to chase!')
else:
    print('Try more epochs or adjust LR')

# ═══════════════════════════════════════════════════════
# 5. Сохраняем в .taas + .npz
# ═══════════════════════════════════════════════════════
print('\nSaving model...')
tcm.save_connectome('chase_brain.taas', conn)
np.savez('chase_brain_weights.npz',
    bridge_w=np.array(bridge.weights(), dtype=np.float32),
    bridge_b=np.array(bridge.bias(), dtype=np.float32),
    decode_w=decode_w,
    decode_b=decode_b,
    readout_idx=readout_idx,
    n_osc=N_OSC, dt=DT, drive_gain=12.0, alpha=2.0,
    freq_min=0.5, freq_max=3.0,
    coupling_w12=0.3, coupling_w13=0.15, coupling_w23=0.08)
print('  chase_brain.taas (connectome)')
print('  chase_brain_weights.npz (bridge + decoder)')

# One-line load test
loaded = np.load('chase_brain_weights.npz')
w_shape = loaded['bridge_w'].shape
d_shape = loaded['decode_w'].shape
print(f'  Load OK: bridge_w={w_shape}, decode_w={d_shape}')
