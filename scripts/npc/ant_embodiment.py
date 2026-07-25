#!/usr/bin/env python3
"""
🐜 100K Artificial Brain → MuJoCo Ant Embodiment

Полностью самодостаточный скрипт — не импортирует test_100k_connectome.
"""

import sys, os, time, json, math, argparse
import numpy as np
from pathlib import Path

os.environ['LD_LIBRARY_PATH'] = '/usr/lib/python3.14/site-packages/nvidia/cudnn/lib'

BASE = Path(__file__).resolve().parent.parent
SARDAANA = Path("/home/taaslab/Проекты/asm_engines/sardaanaCM")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(SARDAANA))

import torch
import gymnasium as gym
import taas_checkmate as tcm
from src.connectome_builder.diffusion import ConnectomeDiffusion
from src.connectome_builder import CookieCutterGenerator, CookieCutterConfig

# ═══════════════════════════════════════════════════════════
N = 100000
DT_MS = 0.5
WARMUP_STEPS = 100
BRAIN_STEPS_PER_ACTION = 10

MODEL_PATH = SARDAANA / "models" / "diffusion" / "connectome_diffusion_scaled.pt"
COOKIE_DIR = SARDAANA / "models" / "cookiecutter"
TAAS_PATH = BASE / "examples" / "data" / "generated_100k.taas"

# ── Edge predictor (copied from test_100k_connectome.py to avoid import side-effects) ──

def edge_predict_to_csr(ep, states_np, target_degs, nt_types, threshold=0.3,
                         block_size=256, n_neighbor_blocks=2, rng=None):
    """CSR-mode edge prediction без N×N матрицы."""
    n = len(states_np)
    if rng is None:
        rng = np.random.RandomState(42)

    # STDP weight pool: log-normal distribution
    n_weights_pool = 100000
    weights_pool = np.exp(rng.randn(n_weights_pool) * 0.5 - 2.0)
    weights_pool = np.clip(weights_pool, 0.001, 2.0)
    w_idx = 0

    # Sort by community (4th feature) for locality
    communities = np.floor(states_np[:, 3] * 8).clip(0, 7).astype(np.int32)
    comm_order = np.argsort(communities)
    comm_inv = np.argsort(comm_order)
    states_sorted = states_np[comm_order]
    degs_sorted = target_degs[comm_order]

    block_starts = list(range(0, n, block_size)) + [n]
    n_blocks = len(block_starts) - 1
    block_comm = np.array([communities[comm_order[min(block_starts[i], n-1)]]
                          for i in range(n_blocks)])

    # Prepare CSR output
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    col_list, val_list = [], []
    total_pairs = 0

    for bi in range(n_blocks):
        b_start, b_end = block_starts[bi], block_starts[bi + 1]
        b_neurons = b_end - b_start

        # Target blocks: same community + neighbors
        target_blocks = [bj for bj in range(n_blocks)
                         if abs(int(block_comm[bi]) - int(block_comm[bj])) <= n_neighbor_blocks]

        # Collect all (i, j) pairs for this block → batch predict
        all_i, all_j = [], []
        neuron_ranges = []
        for local_i in range(b_neurons):
            global_i = b_start + local_i
            deg = int(degs_sorted[global_i])
            deg = min(deg, n // 8)
            if deg <= 0:
                neuron_ranges.append((0, 0))
                continue

            candidates = []
            for bj in target_blocks:
                bj_start = block_starts[bj]
                bj_end = block_starts[bj + 1]
                candidates.extend(range(bj_start, bj_end))

            candidates = np.array(candidates, dtype=np.int32)
            if len(candidates) > deg * 4:
                cand_idx = np.sort(rng.choice(len(candidates),
                    size=min(deg * 4, len(candidates)), replace=False))
                candidates = candidates[cand_idx]

            s_i = states_sorted[global_i]
            for c in candidates:
                all_i.append(global_i)
                all_j.append(c)
            neuron_ranges.append((len(all_j) - len(candidates), len(all_j)))

        if not all_i:
            continue

        # Batch predict on GPU
        all_i_np = np.array(all_i, dtype=np.int32)
        all_j_np = np.array(all_j, dtype=np.int32)
        s_i_batch = states_sorted[all_i_np]
        s_j_batch = states_sorted[all_j_np]

        # EdgePredictor.forward expects [batch, 8] — concatenated pre/post states
        pairs = np.column_stack([s_i_batch, s_j_batch])  # [batch, 8]
        device = next(ep.parameters()).device

        # Predict in micro-batches via torch forward
        preds = np.zeros(len(all_i), dtype=np.float32)
        batch_size = 65536
        for start_idx in range(0, len(all_i), batch_size):
            end_idx = min(start_idx + batch_size, len(all_i))
            pair_batch = pairs[start_idx:end_idx]
            pair_tensor = torch.from_numpy(pair_batch).float().to(device)
            with torch.no_grad():
                pred_tensor = ep.forward(pair_tensor)
            preds[start_idx:end_idx] = pred_tensor.cpu().numpy().flatten()

        # Top-k per neuron
        for local_i in range(b_neurons):
            global_i = b_start + local_i
            start_j, end_j = neuron_ranges[local_i]
            deg = min(int(degs_sorted[global_i]), n // 8)
            if start_j >= end_j or deg <= 0:
                continue

            neuron_preds = preds[start_j:end_j]
            neuron_cands = all_j[start_j:end_j]
            topk = min(deg, len(neuron_cands))
            top_idx = np.argsort(-neuron_preds)[:topk]

            seen = set()
            for idx in top_idx:
                post = neuron_cands[idx]
                if post != global_i and post not in seen:
                    seen.add(post)
                    col_list.append(int(post))
                    val_list.append(float(weights_pool[w_idx % n_weights_pool]))
                    w_idx += 1

            row_ptr[global_i + 1] = len(col_list)

        total_pairs += len(all_i)
        if (bi + 1) % 10 == 0:
            print(f"    Block {bi+1}/{n_blocks} — {len(col_list):,} edges")

    # Map back to original order
    col_arr = np.array(col_list, dtype=np.int32)
    val_arr = np.array(val_list, dtype=np.float32)

    col_per_neuron = [[] for _ in range(n)]
    val_per_neuron = [[] for _ in range(n)]
    cursor = 0
    for old_i in range(n):
        new_i = comm_inv[old_i]
        start, end = cursor, row_ptr[old_i + 1]
        cursor = end
        for j in range(start, end):
            col_per_neuron[new_i].append(int(comm_inv[col_arr[j]]))
            val_per_neuron[new_i].append(float(val_arr[j]))

    row_ptr_out = np.zeros(n + 1, dtype=np.int32)
    col_out_list, val_out_list = [], []
    for i in range(n):
        unique = {}
        for c, v in zip(col_per_neuron[i], val_per_neuron[i]):
            if c != i and (c not in unique or abs(v) > abs(unique[c])):
                unique[c] = v
        for c, v in sorted(unique.items()):
            col_out_list.append(c)
            val_out_list.append(v)
        row_ptr_out[i + 1] = len(col_out_list)

    return (row_ptr_out, np.array(col_out_list, dtype=np.int32),
            np.array(val_out_list, dtype=np.float32))


# ── HH params ──

def build_hh_params(n, nt, rng):
    g_max = np.zeros(3 * n, dtype=np.float32)
    C_m = np.ones(n, dtype=np.float32)
    g_leak = np.ones(n, dtype=np.float32) * 0.3
    E_leak = np.ones(n, dtype=np.float32) * -65.0
    for i in range(n):
        if nt[i] > 0:
            g_max[0*n+i]=120.0; g_max[1*n+i]=36.0; C_m[i]=1.0
        else:
            g_max[0*n+i]=120.0; g_max[1*n+i]=50.0; C_m[i]=0.6; g_leak[i]=0.5
    pc = max(1, int(n * 0.05))
    g_max[2*n + rng.choice(n, size=pc, replace=False)] = 0.5
    return g_max, C_m, g_leak, E_leak


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ant-steps", type=int, default=100)
    parser.add_argument("--render", action="store_true", help="Show MuJoCo window")
    args = parser.parse_args()

    rng = np.random.RandomState(42)
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  🐜 100K Artificial Brain → MuJoCo Ant              ║")
    print("╚═══════════════════════════════════════════════════════╝\n")

    # ── 1. Generate / load connectome ──
    if TAAS_PATH.exists():
        print("[1/4] Loading cached connectome ...")
        import struct
        with open(TAAS_PATH, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            assert magic == 0x54414153
            _ = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            s = struct.unpack("<Q", f.read(8))[0]
            rp = np.frombuffer(f.read((n+1)*4), dtype=np.int32).copy()
            ci = np.frombuffer(f.read(s*4), dtype=np.int32).copy()
            vals = np.frombuffer(f.read(s*4), dtype=np.float32).copy()
            nt = np.frombuffer(f.read(n*4), dtype=np.float32).copy()
        print(f"  📦 {n:,}N, {s:,}S loaded\n")
    else:
        print("[1/4] Generating 100K connectome ...")
        t0 = time.time()
        model = ConnectomeDiffusion.load(str(MODEL_PATH))
        gen = CookieCutterGenerator.load_trained(
            str(COOKIE_DIR), CookieCutterConfig(n_neurons=N, use_gpu=True))
        states = model.sample(n_neurons=N, batch_size=1, use_ddim=True, ddim_steps=50)
        state_arr = states[0]

        target_degs = np.exp(rng.randn(N)*1.2+4.0).astype(np.int32).clip(1, N//8)
        nt = np.where(rng.rand(N) < 0.2, -1.0, 1.0).astype(np.float32)

        t0 = time.time()
        rp_arr, ci_arr, val_arr = edge_predict_to_csr(
            gen.edge_predictor, state_arr, target_degs, nt,
            threshold=0.3, block_size=256, n_neighbor_blocks=2, rng=rng)
        row_ptr, col_indices, values, nt_use = rp_arr, ci_arr, val_arr, nt
        print(f"  🔗 {len(col_indices):,} synapses in {time.time()-t0:.1f}s\n")

    # ── 2. Brain setup ──
    print("[2/4] BrainSimulator HH+CaV ...")
    if not TAAS_PATH.exists():
        conn = tcm.ConnectomeData(row_ptr.astype(np.int32), col_indices.astype(np.int32),
                                  values.astype(np.float32), nt_use.astype(np.float32))
    else:
        conn = tcm.ConnectomeData(rp.astype(np.int32), ci.astype(np.int32),
                                  vals.astype(np.float32), nt.astype(np.float32))

    g_max, C_m, g_leak, E_leak = build_hh_params(conn.n_neurons, np.asarray(conn.nt_types()), rng)
    sim = tcm.BrainSimulator(conn.n_neurons, "hh", pacemaking=True)
    sim.set_hh_params(g_max.astype(np.float32),
        np.array([50.0, -77.0, 120.0], dtype=np.float32),
        g_leak.astype(np.float32), E_leak.astype(np.float32), dt_ms=DT_MS)
    sim.set_C_m(C_m.astype(np.float32))
    print()

    # ── 3. Brain↔Ant mapping ──
    print("[3/4] Brain↔Ant mapping ...")
    n_sensory = 27
    n_motor = 8
    motor_start = conn.n_neurons - n_motor
    sensory_idx = list(range(n_sensory))
    motor_idx = list(range(motor_start, conn.n_neurons))
    print(f"  👁️  Sensory: {n_sensory}  🦾 Motor: {motor_start}–{conn.n_neurons-1}")

    # Warm-up: let CaV build up
    print(f"  🔥 Warming up {WARMUP_STEPS} brain steps ...")
    for _ in range(WARMUP_STEPS):
        sim.step_batch(conn, 1, dt_ms=DT_MS, scale=1e10)
    print()

    # ── 4. Ant loop ──
    print(f"[4/4] Running {args.ant_steps} ant steps ...\n")
    env = gym.make("Ant-v4", render_mode="human" if args.render else None)

    # Try reset with different APIs
    try:
        obs, info = env.reset(seed=42)
    except TypeError:
        obs = env.reset()
        info = {}

    obs = np.asarray(obs, dtype=np.float32)
    total_reward = 0.0
    action_buf = np.zeros(8, dtype=np.float32)
    brain_step = 0
    ant_step = 0

    print(f"  {'Step':>5s} {'Reward':>7s} {'Spikes':>8s} {'BrainStp':>8s}")
    print(f"  {'─'*5} {'─'*7} {'─'*8} {'─'*8}")

    while ant_step < args.ant_steps:
        # ── Brain: inject obs as external_I ──
        ext_I = np.zeros(conn.n_neurons, dtype=np.float32)
        obs_clipped = np.clip(obs / 10.0, -1.0, 1.0).astype(np.float32)
        for i in range(min(n_sensory, len(obs_clipped))):
            ext_I[sensory_idx[i]] = obs_clipped[i] * 5.0

        spike_buf = sim.step_batch(conn, 1, dt_ms=DT_MS, external_I=ext_I, scale=1e10)
        sb = np.asarray(spike_buf).flatten()
        brain_spikes = int(sb.sum())
        brain_step += 1

        # ── Decode motor → action ──
        motor_rates = sb[motor_idx].astype(np.float32)
        action = motor_rates * 2.0 - 1.0  # [0,1] → [-1,1]
        action = np.clip(action, -1.0, 1.0)
        action_buf = action_buf * 0.7 + action * 0.3

        # ── Ant step ──
        if brain_step % BRAIN_STEPS_PER_ACTION == 0:
            obs, reward, terminated, truncated, _ = env.step(action_buf)
            obs = np.asarray(obs, dtype=np.float32)
            total_reward += float(reward)
            ant_step += 1

            print(f"  {ant_step:>5d} {reward:>7.2f} {brain_spikes:>8d} {brain_step:>8d}")

            if terminated or truncated:
                print(f"\n  🏁 Ant fell over after {ant_step} steps")
                break

    env.close()

    print(f"\n{'='*55}")
    print(f"  ✅ Ant embodiment complete!")
    print(f"  {'Ant steps':<20s} {ant_step:>8d}")
    print(f"  {'Brain steps':<20s} {brain_step:>8d}")
    print(f"  {'Total reward':<20s} {total_reward:>8.1f}")
    print(f"  {'Mean reward/step':<20s} {total_reward/max(ant_step,1):>8.2f}")
    print(f"{'='*55}")
    print(f"  🐜 100K artificial brain → Ant — первый в мире!")


if __name__ == "__main__":
    main()
