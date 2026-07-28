#!/usr/bin/env python3
"""🦗 VNC Engine — сардана генерирует VNC-подобный коннектом → Kuramoto CPG → Ant

Архитектура:
  sarDNA (обучена, секси 😏)
      ↓ sample(n_neurons=5000)
  VNC-стиль state vectors
      ↓ CookieCutter (GABA-heavy)
  CSR коннектом (~5K нейронов, ~400K синапсов)
      ↓ Kuramoto CPG
  Ring oscillator (anti-phase GABA)
      ↓ DN modulation
  Ant joints (8)

Биология:
  VNC насекомого ≈ 5K нейронов, >90% GABA
  Реципрокное торможение → ring oscillator → CPG
  3 фазы = tripod gait (левый-правый-левый)
"""

import sys, os, time, math, argparse
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

# ═══ CONFIG ═══════════════════════════════════════════════
N_VNC = 5000               # VNC нейронов (реальный: 5,681)
DT_MS = 0.5                # Brain dt
CPG_DT = 0.01              # CPG Kuramoto dt (10ms)
OMEGA_BASE = 8.0           # rad/s ≈ 1.3 Hz шагания
COUPLING_SCALE = 30.0      # сила GABA-связей

MODEL_PATH = SARDAANA / "models" / "diffusion" / "connectome_diffusion_scaled.pt"
COOKIE_DIR = SARDAANA / "models" / "cookiecutter"
TAAS_PATH_VNC = BASE / "examples" / "data" / "vnc_generated.taas"


# ═══ EDGE PREDICTOR (self-contained) ═════════════════════

def edge_predict_to_csr(ep, states_np, target_degs, nt_types, threshold=0.3,
                         block_size=256, n_neighbor_blocks=2, rng=None):
    n = len(states_np)
    if rng is None:
        rng = np.random.RandomState(42)

    n_weights_pool = 100000
    weights_pool = np.exp(rng.randn(n_weights_pool) * 0.5 - 2.0)
    weights_pool = np.clip(weights_pool, 0.001, 2.0)
    w_idx = 0

    communities = np.floor(states_np[:, 3] * 8).clip(0, 7).astype(np.int32)
    comm_order = np.argsort(communities)
    comm_inv = np.argsort(comm_order)
    states_sorted = states_np[comm_order]
    degs_sorted = target_degs[comm_order]

    block_starts = list(range(0, n, block_size)) + [n]
    n_blocks = len(block_starts) - 1
    block_comm = np.array([communities[comm_order[min(block_starts[i], n-1)]]
                          for i in range(n_blocks)])

    row_ptr = np.zeros(n + 1, dtype=np.int32)
    col_list, val_list = [], []
    total_pairs = 0

    for bi in range(n_blocks):
        b_start, b_end = block_starts[bi], block_starts[bi + 1]
        b_neurons = b_end - b_start

        target_blocks = [bj for bj in range(n_blocks)
                         if abs(int(block_comm[bi]) - int(block_comm[bj])) <= n_neighbor_blocks]

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

            for c in candidates:
                all_i.append(global_i)
                all_j.append(c)
            neuron_ranges.append((len(all_j) - len(candidates), len(all_j)))

        if not all_i:
            continue

        all_i_np = np.array(all_i, dtype=np.int32)
        all_j_np = np.array(all_j, dtype=np.int32)
        s_i_batch = states_sorted[all_i_np]
        s_j_batch = states_sorted[all_j_np]

        pairs = np.column_stack([s_i_batch, s_j_batch])
        device = next(ep.parameters()).device

        preds = np.zeros(len(all_i), dtype=np.float32)
        batch_size = 65536
        for start_idx in range(0, len(all_i), batch_size):
            end_idx = min(start_idx + batch_size, len(all_i))
            pair_batch = pairs[start_idx:end_idx]
            pair_tensor = torch.from_numpy(pair_batch).float().to(device)
            with torch.no_grad():
                pred_tensor = ep.forward(pair_tensor)
            preds[start_idx:end_idx] = pred_tensor.cpu().numpy().flatten()

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


# ═══ KURAMOTO CPG на сгенерированном VNC ═══════════════

class VNCKuramotoCPG:
    """Kuramoto CPG на VNC-коннектоме от сарданы.
    
    Все связи GABA → отрицательный coupling → anti-phase oscillation.
    """
    
    def __init__(self, row_ptr, col_indices, values, nt_types,
                 omega_base=8.0, coupling_scale=30.0, dt_s=0.01):
        self.n = len(nt_types)
        self.row_ptr = row_ptr
        self.col_indices = col_indices
        self.dt_s = dt_s
        self.omega_base = omega_base
        self.coupling_scale = coupling_scale
        
        # Define neuron groups: 20% DN (descending), 60% IN (inter), 20% EN (efferent)
        self.dn_indices = np.sort(np.random.RandomState(0).choice(
            self.n, size=max(1, self.n // 5), replace=False))
        self.en_indices = np.sort(np.random.RandomState(1).choice(
            [i for i in range(self.n) if i not in self.dn_indices],
            size=max(1, self.n // 5), replace=False))
        self.in_indices = np.array([i for i in range(self.n) 
                                     if i not in self.dn_indices and i not in self.en_indices])
        
        print(f"  [VNC.CPG] Groups: DN={len(self.dn_indices)} "
              f"IN={len(self.in_indices)} EN={len(self.en_indices)}")
        
        # Kuramoto params
        self.omegas = np.random.RandomState(42).uniform(
            omega_base * 0.5, omega_base * 1.5, self.n).astype(np.float64)
        
        # Coupling: GABA → negative → anti-phase
        weights = values.astype(np.float64)
        self.coupling = -coupling_scale * np.abs(weights)
        
        # State
        self.phases = np.random.RandomState(42).uniform(0, 2*math.pi, self.n).astype(np.float64)
        self.dn_input = np.zeros(self.n, dtype=np.float64)
        
    def step(self, dt=None):
        """Kuramoto step — использует Rust SIMD kuramoto_step_sparse если доступен."""
        if dt is None:
            dt = self.dt_s
        n = self.n
        
        # Используем Rust SIMD ускорение если доступно
        if hasattr(tcm, 'kuramoto_step_sparse'):
            phases_out = np.zeros(n, dtype=np.float64)
            try:
                tcm.kuramoto_step_sparse(
                    self.phases.astype(np.float32),
                    self.omegas.astype(np.float32),
                    self.row_ptr.astype(np.int32),
                    self.col_indices.astype(np.int32),
                    self.coupling.astype(np.float32),
                    phases_out.astype(np.float32),
                    np.float32(dt),
                )
                self.phases = phases_out % (2 * math.pi)
                # Add DN input
                self.phases += dt * self.dn_input
                return
            except Exception:
                pass  # fallback to Python
        
        # Python fallback (медленно, но работает)
        phases = self.phases
        dphases = self.omegas.copy()
        
        for i in range(n):
            start = int(self.row_ptr[i])
            end = int(self.row_ptr[i + 1])
            if start >= end:
                continue
            # Vectorized per-neuron: sin(θ_j - θ_i) for all j
            js = self.col_indices[start:end]
            wjs = self.coupling[start:end]
            dphases[i] += np.sum(wjs * np.sin(phases[js] - phases[i]))
        
        dphases += self.dn_input
        self.phases = (self.phases + dt * dphases) % (2 * math.pi)
    
    def inject_dn(self, dn_rates):
        """Inject descending neuron rates as frequency modulation."""
        for idx, rate in zip(self.dn_indices, dn_rates):
            self.dn_input[idx] = rate * 2.0
    
    def coherence(self):
        """Kuramoto order parameter R ∈ [0, 1]."""
        n = self.n
        sin_sum = np.sin(self.phases).sum()
        cos_sum = np.cos(self.phases).sum()
        return math.sqrt(sin_sum**2 + cos_sum**2) / n
    
    def en_rates(self):
        """Get efferent neuron rates (smooth, for actuator mapping)."""
        # Phase velocity as rate proxy: faster phase change = more active
        return (np.sin(self.phases[self.en_indices]) + 1.0) * 0.5
    
    def phase_groups(self, n_groups=8):
        """Split VNC into n_groups phase clusters for joint mapping."""
        return np.array_split(self.phases % (2*math.pi), n_groups)
    
    def joint_angles(self, n_joints=8):
        """Map VNC phases to joint angles [-1, 1]."""
        groups = self.phase_groups(n_joints)
        angles = np.zeros(n_joints)
        for i, g in enumerate(groups):
            mean_phase = np.mean(g)
            angles[i] = math.sin(mean_phase)
        return angles


# ═══ MAIN ════════════════════════════════════════════════

def generate_vnc_connectome(rng):
    """Generate VNC-style connectome using sarDNA."""
    print("[1/3] Generating VNC connectome via sarDNA ...")
    t0 = time.time()
    
    model = ConnectomeDiffusion.load(str(MODEL_PATH))
    gen = CookieCutterGenerator.load_trained(
        str(COOKIE_DIR), CookieCutterConfig(n_neurons=N_VNC, use_gpu=True))
    
    # Sample VNC-like states
    states = model.sample(n_neurons=N_VNC, batch_size=1, use_ddim=True, ddim_steps=50)
    state_arr = states[0]
    print(f"  States: {state_arr.shape} [{time.time()-t0:.1f}s]")
    
    # VNC-like: most neurons GABA (inhibitory), some excitatory DN
    # Real VNC: 90%+ GABA, few excitatory DN
    nt = np.where(
        np.random.RandomState(42).rand(N_VNC) < 0.9,  # 90% inhibitory
        -1.0, 1.0
    ).astype(np.float32)
    n_exc = int((nt > 0).sum())
    n_inh = int((nt < 0).sum())
    print(f"  E/I: {n_exc}E / {n_inh}I ({n_exc/max(n_inh,1)*100:.1f}% excitatory)")
    
    # VNC neurons have ~80 synapses each (from MANC data)
    target_degs = np.exp(rng.randn(N_VNC) * 0.8 + 4.5).astype(np.int32).clip(5, N_VNC // 4)
    print(f"  Mean target degree: {target_degs.mean():.1f} (+- {target_degs.std():.1f})")
    
    # Edge prediction
    t1 = time.time()
    rp, ci, vals = edge_predict_to_csr(
        gen.edge_predictor, state_arr, target_degs, nt,
        threshold=0.3, block_size=256, n_neighbor_blocks=2, rng=rng)
    print(f"  Edges: {len(ci):,} in {time.time()-t1:.1f}s")
    print(f"  Total: {time.time()-t0:.1f}s\n")
    
    return rp, ci, vals, nt


def test_vnc_cpg(row_ptr, col_indices, values, nt_types, n_steps=1000):
    """Test if generated VNC oscillates."""
    print("[2/3] Testing VNC CPG oscillation ...")
    
    cpg = VNCKuramotoCPG(row_ptr, col_indices, values, nt_types,
                          omega_base=OMEGA_BASE, coupling_scale=COUPLING_SCALE,
                          dt_s=CPG_DT)
    
    # Warm-up
    for _ in range(200):
        cpg.step()
    
    # Measure
    coherences = []
    phases_log = []
    joint_log = []
    
    for step in range(n_steps):
        cpg.step()
        coherences.append(cpg.coherence())
        if step % 10 == 0:
            phases_log.append(cpg.phases.copy())
            joint_log.append(cpg.joint_angles(8))
    
    coh = np.array(coherences)
    joints = np.array(joint_log)
    
    print(f"  Kuramoto R: mean={coh.mean():.4f} max={coh.max():.4f}")
    print(f"  Phase coherence: {'✅ OSCILLATING' if coh.mean() > 0.1 else '❌ NOT oscillating'}")
    
    # Check phase separation (tripod requires ~120° between groups)
    if len(joints) > 10:
        # Auto-correlation of joint 0 to check periodicity
        j0 = joints[:, 0]
        autocorr = np.correlate(j0 - j0.mean(), j0 - j0.mean(), mode='same')
        autocorr = autocorr[len(autocorr)//2:]  # right half
        peaks = np.where(autocorr > autocorr.max() * 0.5)[0]
        if len(peaks) > 5:
            period = np.mean(np.diff(peaks[:10])) * 10 * CPG_DT
            freq_hz = 1.0 / max(period, 1e-6)
            print(f"  Oscillation freq: {freq_hz:.2f} Hz (period: {period:.3f}s)")
        else:
            print(f"  Oscillation: aperiodic (R={coh.mean():.3f})")
    
    # Phase difference between joint groups
    if len(joints) > 10:
        mean_phases = np.angle(np.mean(np.exp(1j * joints), axis=0))
        print(f"  Joint phase offsets (rad): {np.round(mean_phases, 2)}")
        phase_diff = np.diff(np.concatenate([mean_phases, [mean_phases[0]]]))
        phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))  # wrap to [-pi, pi]
        print(f"  Phase diffs (rad): {np.round(phase_diff, 2)}")
    
    return cpg


def run_ant_embodiment(row_ptr, col_indices, values, nt_types, args):
    """Run brain + VNC CPG → Ant."""
    print("[3/3] Ant embodiment with VNC CPG ...\n")
    
    # VNC CPG (напрямую из CSR массивов, без ConnectomeData)
    cpg = VNCKuramotoCPG(row_ptr, col_indices, values, nt_types,
                          omega_base=OMEGA_BASE, coupling_scale=COUPLING_SCALE,
                          dt_s=CPG_DT)
    
    # Warm-up VNC
    print("  🔥 Warming up VNC CPG ...")
    for _ in range(500):
        cpg.step()
    
    # Ant
    env = gym.make("Ant-v4", render_mode="human" if args.render else None)
    try:
        obs, info = env.reset(seed=42)
    except TypeError:
        obs = env.reset()
        info = {}
    obs = np.asarray(obs, dtype=np.float32)
    
    total_reward = 0.0
    action_buf = np.zeros(8, dtype=np.float32)
    cpg_steps_per_action = int(0.05 / CPG_DT)  # ~5 CPG steps per ant step
    last_step = 0
    
    print(f"  {'Step':>5s} {'Reward':>7s} {'R_coh':>6s} {'Act[0]':>7s}")
    print(f"  {'─'*5} {'─'*7} {'─'*6} {'─'*7}")
    
    for ant_step in range(args.ant_steps):
        # Run CPG for several steps
        for _ in range(cpg_steps_per_action):
            # Modulate VNC with obs (like DN from brain)
            obs_clipped = np.clip(obs / 10.0, -1.0, 1.0)
            dn_rates = obs_clipped[:min(len(obs_clipped), len(cpg.dn_indices))]
            cpg.inject_dn(np.pad(dn_rates, (0, max(0, len(cpg.dn_indices) - len(dn_rates)))))
            cpg.step()
        
        # Get joint angles from CPG
        joints = cpg.joint_angles(8)
        action = np.clip(joints.astype(np.float32), -1.0, 1.0)
        action_buf = action_buf * 0.7 + action * 0.3
        
        # Ant step
        obs, reward, terminated, truncated, _ = env.step(action_buf)
        obs = np.asarray(obs, dtype=np.float32)
        total_reward += float(reward)
        
        if ant_step % 10 == 0:
            coh = cpg.coherence()
            print(f"  {ant_step:>5d} {reward:>+7.2f} {coh:>6.3f} {action[0]:>+7.3f}")
        
        last_step = ant_step
        if terminated or truncated:
            print(f"  🏁 Ant fell after {ant_step} steps")
            break
    
    env.close()
    print(f"\n  Total reward: {total_reward:.1f} over {last_step+1} steps")
    print(f"  Mean VNC coherence: {cpg.coherence():.3f}")


def main():
    parser = argparse.ArgumentParser(description="🦗 VNC Engine: sarDNA → CPG → Ant")
    parser.add_argument("--ant-steps", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--test-only", action="store_true",
                        help="Only generate VNC + test CPG oscillation, skip Ant")
    parser.add_argument("--no-gen", action="store_true",
                        help="Skip VNC generation, load from cache")
    args = parser.parse_args()
    
    rng = np.random.RandomState(42)
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  🦗 VNC Engine — sarDNA → CPG → Ant                 ║")
    print("╚═══════════════════════════════════════════════════════╝\n")
    
    # Generate or load VNC connectome
    if TAAS_PATH_VNC.exists() and args.no_gen:
        print("[1/3] Loading cached VNC connectome ...")
        import struct
        with open(TAAS_PATH_VNC, "rb") as f:
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
        row_ptr, col_indices, values, nt_types = rp, ci, vals, nt
    else:
        row_ptr, col_indices, values, nt_types = generate_vnc_connectome(rng)
    
    # Test CPG
    cpg = test_vnc_cpg(row_ptr, col_indices, values, nt_types)
    
    # Ant embodiment
    if not args.test_only:
        run_ant_embodiment(row_ptr, col_indices, values, nt_types, args)
    
    print(f"\n{'='*55}")
    print(f"  🦗 VNC Engine complete!")
    print(f"  VNC neurons: {len(nt_types):,}")
    print(f"  Synapses: {len(col_indices):,}")
    print(f"  GABA ratio: {(nt_types < 0).mean()*100:.1f}%")
    print(f"  Final R: {cpg.coherence():.4f}")


if __name__ == "__main__":
    main()
