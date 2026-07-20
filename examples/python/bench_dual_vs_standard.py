#!/usr/bin/env python3
"""Сравнение: стандартный Izhikevich vs Dual Memory.
Исправление: root_id -> sequential index mapping."""
import sys, os, time, gzip
import numpy as np
from pathlib import Path

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ / 'target' / 'release'))
os.environ['RAYON_NUM_THREADS'] = '8'

import taas_checkmate as tcm

# ── Find connectome file ──
conn_path = None
for p in [
    Path.home() / 'Загрузки' / 'connections_princeton(5).csv.gz',
    Path.home() / 'Загрузки' / 'connections_princeton.csv.gz',
    PROJ / '..' / 'fly' / 'archive' / 'connections_princeton.csv.gz',
    Path('/home/taaslab/Проекты/asm_engines/fly/archive/connections_princeton.csv.gz'),
]:
    if p.exists():
        conn_path = p
        break
if conn_path is None:
    print("❌ Connectome file not found"); sys.exit(1)

print("=" * 60)
print("🧠 DUAL MEMORY vs STANDARD IZHIKEVICH")
print("=" * 60)

# ── Load CSV with proper 64-bit int handling ──
print(f"\n[1] Loading connectome from {conn_path.name}...")
t0 = time.time()

pre_list, post_list, w_list = [], [], []
nt_map = {}  # root_id -> nt_type

with gzip.open(str(conn_path), 'rt') as f:
    header = next(f).strip().split(',')
    # Find column indices
    pre_idx = header.index('pre_root_id')
    post_idx = header.index('post_root_id')
    w_idx = header.index('syn_count')
    nt_idx = header.index('nt_type') if 'nt_type' in header else -1
    
    for line in f:
        parts = line.strip().split(',')
        if len(parts) <= max(pre_idx, post_idx, w_idx):
            continue
        pre_list.append(int(parts[pre_idx]))
        post_list.append(int(parts[post_idx]))
        w_list.append(float(parts[w_idx]))
        if nt_idx >= 0 and len(parts) > nt_idx and parts[nt_idx]:
            nt_val = float(parts[nt_idx])
            pre_root = int(parts[pre_idx])
            nt_map[pre_root] = nt_val

# Build root_id -> index mapping
all_ids = sorted(set(pre_list) | set(post_list))
root_to_idx = {rid: i for i, rid in enumerate(all_ids)}
n_neurons = len(all_ids)
n_synapses = len(pre_list)
print(f"   {n_neurons} neurons, {n_synapses:,} synapses ({time.time()-t0:.1f}s)")

# Build CSR
row_ptr = np.zeros(n_neurons + 1, dtype=np.int32)
col_indices = np.zeros(n_synapses, dtype=np.int32)
values = np.zeros(n_synapses, dtype=np.float32)
nt_types = np.ones(n_neurons, dtype=np.float32)

# Count synapses per neuron
for pre_id in pre_list:
    idx = root_to_idx[pre_id]
    row_ptr[idx + 1] += 1
row_ptr = np.cumsum(row_ptr, dtype=np.int32)

# Fill col_indices and values
cur = np.zeros(n_neurons, dtype=np.int32)
for i in range(n_synapses):
    pre_idx = root_to_idx[pre_list[i]]
    pos = row_ptr[pre_idx] + cur[pre_idx]
    col_indices[pos] = root_to_idx[post_list[i]]
    values[pos] = w_list[i]
    cur[pre_idx] += 1

# Set nt_types from nt_map (default +1 exc)
for rid, nt_val in nt_map.items():
    if rid in root_to_idx:
        nt_types[root_to_idx[rid]] = 1.0 if nt_val > 0 else -1.0

print(f"   Build CSR: {time.time()-t0:.1f}s")

# Build ConnectomeData
print("\n[2] Building ConnectomeData...")
conn = tcm.ConnectomeData(row_ptr, col_indices, values, nt_types)
print(f"   n_neurons={conn.n_neurons}, n_synapses={conn.n_synapses:,}")

# ── Parameters ──
N_STEPS = 200
DT_MS = 0.5
TAU_M = 0.999
W_M = 3.0
BASE_I = 7.0

# ── Standard Izhikevich ──
print(f"\n[3] Standard Izhikevich ({N_STEPS} steps)...")
sim_std = tcm.BrainSimulator(conn.n_neurons, "izhikevich")
external_I = np.full(conn.n_neurons, BASE_I, dtype=np.float32)

t0 = time.time()
spikes_std = np.array(
    sim_std.step_batch(conn, N_STEPS, DT_MS, external_I)
).reshape(N_STEPS, conn.n_neurons)
wall_std = time.time() - t0
total_std = spikes_std.sum()
fr_std = total_std / (N_STEPS * DT_MS / 1000.0) / conn.n_neurons
print(f"   ⏱ {wall_std:.2f}s | Spikes: {total_std:,} | FR: {fr_std:.2f} Hz")

# ── Dual Memory (baseline=0.5, compensated, BATCH=5) ──
print(f"\n[4] Dual Memory (τ_m={TAU_M}, w_m={W_M}, BATCH=5)...")
sim_dual = tcm.BrainSimulator(conn.n_neurons, "izhikevich")
I_dual = np.full(conn.n_neurons, BASE_I - W_M * 0.5, dtype=np.float32)
m = np.full(conn.n_neurons, 0.5, dtype=np.float32)
BATCH = 5  # update slow state every 5 steps (2.5ms)

t0 = time.time()
total_dual = 0
spikes_dual = np.zeros((N_STEPS, conn.n_neurons), dtype=np.int32)
for bs in range(0, N_STEPS, BATCH):
    n_b = min(BATCH, N_STEPS - bs)
    I_mod = I_dual + W_M * m
    I_mod = np.clip(I_mod, -100.0, 100.0)
    spikes_b = np.array(sim_dual.step_batch(conn, n_b, DT_MS, I_mod)).reshape(n_b, conn.n_neurons)
    spikes_dual[bs:bs+n_b] = spikes_b
    total_dual += spikes_b.sum()
    # Update slow state from each step
    for step in range(n_b):
        m = TAU_M * m + (1.0 - TAU_M) * spikes_b[step].astype(np.float32)
        m = np.clip(m, 0.0, 1.0)
wall_dual = time.time() - t0
fr_dual = total_dual / (N_STEPS * DT_MS / 1000.0) / conn.n_neurons
change = (total_dual - total_std) / max(total_std, 1) * 100
print(f"   ⏱ {wall_dual:.2f}s | Spikes: {total_dual:,} | FR: {fr_dual:.2f} Hz | vs std: {change:+.1f}%")

# ── Results ──
print("\n" + "=" * 60)
print("📊 RESULTS")
print("=" * 60)
print(f"  Standard:    {total_std:>10,} spikes | {fr_std:.2f} Hz | {wall_std:.2f}s")
print(f"  Dual Mem:    {total_dual:>10,} spikes | {fr_dual:.2f} Hz | {wall_dual:.2f}s")
print(f"  Change:      {change:>+9.1f}%")
print(f"  Params:      τ_m={TAU_M}, w_m={W_M}, I={BASE_I}")
print(f"  Slow state:  μ={m.mean():.3f} σ={m.std():.3f}")
print(f"               >0.6: {(m>0.6).mean()*100:.1f}%  <0.4: {(m<0.4).mean()*100:.1f}%")

# Step-by-step pattern
step_std = spikes_std.sum(axis=1)
step_dual = spikes_dual.sum(axis=1)
print(f"\n  Standard step spikes:  μ={step_std.mean():.1f} max={step_std.max():,}")
print(f"  Dual step spikes:     μ={step_dual.mean():.1f} max={step_dual.max():,}")
print(f"  Silent steps (std):   {(step_std==0).sum()}/{N_STEPS}")
print(f"  Silent steps (dual):  {(step_dual==0).sum()}/{N_STEPS}")

# ── Second: baseline=0 ──
print(f"\n[5] Dual Memory (baseline=0, pure dynamics, BATCH=5)...")
sim_z = tcm.BrainSimulator(conn.n_neurons, "izhikevich")
I_z = np.full(conn.n_neurons, BASE_I, dtype=np.float32)
m_z = np.zeros(conn.n_neurons, dtype=np.float32)

t0 = time.time()
total_z = 0
for bs in range(0, N_STEPS, BATCH):
    n_b = min(BATCH, N_STEPS - bs)
    I_mod = I_z + W_M * m_z
    I_mod = np.clip(I_mod, -100.0, 100.0)
    spikes_b = np.array(sim_z.step_batch(conn, n_b, DT_MS, I_mod)).reshape(n_b, conn.n_neurons)
    total_z += spikes_b.sum()
    for step in range(n_b):
        m_z = TAU_M * m_z + (1.0 - TAU_M) * spikes_b[step].astype(np.float32)
        m_z = np.clip(m_z, 0.0, 1.0)
wall_z = time.time() - t0
fr_z = total_z / (N_STEPS * DT_MS / 1000.0) / conn.n_neurons
change_z = (total_z - total_std) / max(total_std, 1) * 100
print(f"   ⏱ {wall_z:.2f}s | Spikes: {total_z:,} | FR: {fr_z:.2f} Hz | vs std: {change_z:+.1f}%")

# Summary table
print("\n" + "━" * 60)
print(f"{'Mode':<30} {'Spikes':>10} {'FR(Hz)':>8} {'vs Std':>10} {'Time':>8}")
print(f"{'─'*30} {'─'*10} {'─'*8} {'─'*10} {'─'*8}")
print(f"{'Standard Izhikevich':<30} {total_std:>10,} {fr_std:>7.2f} {'—':>10} {wall_std:>7.2f}s")
print(f"{'Dual (bl=0.5, batch=5)':<30} {total_dual:>10,} {fr_dual:>7.2f} {change:>+9.1f}% {wall_dual:>7.2f}s")
print(f"{'Dual (bl=0.0, batch=5)':<30} {total_z:>10,} {fr_z:>7.2f} {change_z:>+9.1f}% {wall_z:>7.2f}s")
print("━" * 60)
print("✅ Done!")
