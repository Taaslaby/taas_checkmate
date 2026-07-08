#!/usr/bin/env python3
"""
🏓 Pong Chaos vs Pattern — PURE PYTHON (no taas_checkmate)
     Synthetic brain: 2244 нейрона, 1.4M синапсов
     Kuramoto + dopamine-STDP + упрощённый T2-SPIGRAD

Запуск:
    python3 examples/python/pong_synthetic_pure.py
"""

import sys, time, gzip, os
from pathlib import Path
import numpy as np

# ─── Config ────────────────────────────────────────────────────────
DT = 0.01
SPIKE_THRESH = 2.5
OMEGA_BASE = 3.0
OMEGA_SPREAD = 1.5
LR = 0.02
ALPHA = 0.3          # entropy penalty weight
BETA = 0.3           # chaos/pattern gap weight
H_TARGET = 3.0
ELIG_DECAY = 0.90
ELIG_ADD = 0.15
CEM_STD = 0.15
MAX_TICKS = 100
PADDLE_SPEED = 0.04
BALL_SPEED = 0.018
CEM_POP = 15
CEM_ELITE = 5
CEM_ITERS = 5
N_EPISODES = 30
N_KSTEPS = 10        # Kuramoto steps per game tick
N_SENSORY = 50       # input neurons per state dim
N_MOTOR = 10         # motor readout neurons

BRAIN_PATH = Path("/home/taaslab/Проекты/asm_engines/MEIIgpt/experiments/sardana_1/data")

# ─── Load synthetic brain ──────────────────────────────────────────
def load_synthetic_brain():
    """Load synthetic_trans_count.npy → CSR arrays."""
    t0 = time.time()
    adj = np.load(str(BRAIN_PATH / "synthetic_trans_count.npy"))  # (2244, 2244) int32
    
    # Self-loop stats
    diag = np.diag(adj)
    n_self = np.count_nonzero(diag)
    print(f"  Self-loops: {n_self} (mean weight={diag[diag>0].mean():.1f})")
    
    # Zero in-degree neurons = sensory candidates
    in_deg = np.sum(adj > 0, axis=0)
    zero_in = np.where(in_deg == 0)[0]
    out_deg = np.sum(adj > 0, axis=1)
    zero_out = np.where(out_deg == 0)[0]
    print(f"  Zero in-degree: {len(zero_in)} neurons (sensory candidates)")
    print(f"  Zero out-degree: {len(zero_out)} neurons (motor candidates)")
    
    n = adj.shape[0]
    rows, cols = np.nonzero(adj)
    vals = adj[rows, cols].astype(np.float32)
    n_syn = len(vals)
    print(f"  Non-zero synapses: {n_syn}")
    
    # Build CSR
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    for i in range(n):
        row_ptr[i+1] = row_ptr[i] + np.sum(adj[i] > 0)
    col_indices = cols.astype(np.int32)
    values = vals  # float32
    
    # nt_types: neurons with positive sum → E, negative/inhib → I
    row_sums = np.array([np.sum(adj[i]) for i in range(n)], dtype=np.float32)
    med = np.median(row_sums[row_sums > 0]) if np.any(row_sums > 0) else 1.0
    nt_types = np.where(row_sums > med * 0.5, 1.0, -1.0).astype(np.float32)
    e_cnt = int(np.sum(nt_types > 0))
    i_cnt = int(np.sum(nt_types < 0))
    print(f"  nt_types: E={e_cnt}, I={i_cnt} (threshold={med*0.5:.1f})")
    
    print(f"  Loaded: {n} neurons, {n_syn} synapses ({time.time()-t0:.1f}s)")
    return n, row_ptr, col_indices, values, nt_types, zero_in, zero_out

# ─── Kuramoto (pure numpy) ────────────────────────────────────────
def kuramoto_step(phases, omegas, row_ptr, col_indices, values, dt=DT):
    """One Kuramoto integration step on CSR graph."""
    n = len(phases)
    coupling = np.zeros(n, dtype=np.float32)
    
    # coupling[i] = Σ_j w_ij · sin(θ_j − θ_i)
    for i in range(n):
        start = row_ptr[i]
        end = row_ptr[i+1]
        if end <= start:
            continue
        idxs = col_indices[start:end].astype(int)
        wts = values[start:end]
        phase_diffs = phases[idxs] - phases[i]
        coupling[i] = np.sum(wts * np.sin(phase_diffs))
    
    phases += dt * (omegas + coupling)

def phase_coherence(phases):
    """Kuramoto order parameter R."""
    n = len(phases)
    if n == 0:
        return 0.0
    cos_sum = np.sum(np.cos(phases))
    sin_sum = np.sum(np.sin(phases))
    return np.sqrt(cos_sum**2 + sin_sum**2) / n

def detect_spikes(phases, prev):
    """Rising edge through threshold."""
    return ((phases > SPIKE_THRESH) & (prev <= SPIKE_THRESH)).astype(np.int32)

def spike_entropy(counts, total_steps, h_target):
    eps = 1e-10
    p = np.clip(counts / max(total_steps, 1), eps, 1.0)
    h = -np.sum(p * np.log2(p))
    return h, abs(h - h_target)

# ─── T2-SPIGRAD (упрощённый, в numpy) ─────────────────────────────
# Три популяции T3-осцилляторов: sensory, interneuron, motor
# Каждая популяция — независимые Kuramoto осцилляторы с coupling внутри и между

class T2Spigrad:
    """Упрощённый T2-SPIGRAD: 3 популяции связанных осцилляторов."""
    
    def __init__(self, n_osc=16, dt=DT):
        self.N = n_osc
        self.dt = dt
        # Frequencies
        self.omega = np.array([3.0, 4.5, 6.0], dtype=np.float32)  # sens, inter, motor
        # Coupling matrix (3×3)
        self.C = np.array([
            [0.1, 0.3, 0.2],  # sensory
            [0.3, 0.1, 0.1],  # inter
            [0.2, 0.1, 0.1],  # motor
        ], dtype=np.float32)
        # State
        self.phases = np.random.rand(3, self.N).astype(np.float32) * 2 * np.pi
        
    def forward(self, drives, n_steps, record=False):
        """
        drives: (n_steps, 3*N) — drive for each oscillator
        returns: (pst_features,) if not record, else (pst, trajectory)
        """
        n_steps = min(n_steps, len(drives))
        pst = np.zeros(3 * self.N + 3, dtype=np.float32)  # mean cos + coherence
        traj = []
        
        for t in range(n_steps):
            drv = drives[t].reshape(3, self.N)
            
            # Step each population
            for pop in range(3):
                phase = self.phases[pop]
                d = drv[pop]
                
                # Intra-pop coupling: cos(θᵢ − θⱼ) averaged over all j
                # phase[:, None]  → (N,1), phase[None, :]  → (1,N)
                cos_mat = np.cos(phase[:, None] - phase[None, :])
                intra_cpl = self.C[pop, pop] * np.mean(cos_mat, axis=1) * self.N
                
                # Inter-pop coupling: sin(ϕⱼ − θᵢ) averaged over all j in other pop
                inter_cpl = np.zeros(self.N, dtype=np.float32)
                for other in range(3):
                    if other == pop:
                        continue
                    phase_other = self.phases[other]  # (N,)
                    diff = phase_other[None, :] - phase[:, None]  # (N, N)
                    inter_cpl += self.C[pop, other] * np.mean(np.sin(diff), axis=1) * self.N
                
                # Update
                phase += self.dt * (self.omega[pop] + d + intra_cpl + inter_cpl)
                self.phases[pop] = phase % (2 * np.pi)
            
            if record:
                traj.append(self.phases.copy())
            
            # Accumulate PST features
            if t >= n_steps - 3:  # last 3 steps
                for pop in range(3):
                    pst[pop*self.N:(pop+1)*self.N] += np.cos(self.phases[pop])
                pst[3*self.N] += phase_coherence(self.phases[0])
                pst[3*self.N+1] += phase_coherence(self.phases[1])
                pst[3*self.N+2] += phase_coherence(self.phases[2])
        
        pst /= min(3, n_steps)
        
        if record:
            return pst, np.array(traj)
        return pst
    
    def backward(self, grad_pst, traj):
        """
        Simplified T2 backward: approximate gradient via trajectory.
        grad_pst: (pst_dim,) gradient from loss
        traj: (n_steps, 3, N) recorded trajectory
        returns: grad_drives (n_steps, 3*N)
        """
        n_steps = len(traj)
        grad_drives = np.zeros((n_steps, 3 * self.N), dtype=np.float32)
        
        # Simplified: gradient flows through phase differences
        for t in range(1, n_steps):
            dphase = traj[t] - traj[t-1]  # (3, N)
            # Gradient of phase w.r.t. drive = dt (linear approximation)
            grad_drives[t] = (dphase.flatten() / max(self.dt, 1e-10)) * 0.01
        
        return grad_drives
    
    def reset(self, seed=42):
        rng = np.random.RandomState(seed)
        self.phases = rng.rand(3, self.N).astype(np.float32) * 2 * np.pi

# ─── Pong ──────────────────────────────────────────────────────────
class Pong:
    def __init__(self):
        self.reset()
    def reset(self):
        self.ball_x = self.ball_y = 0.5
        self.ball_vx = BALL_SPEED
        self.ball_vy = BALL_SPEED * -0.2
        self.paddle_y = 0.5
        self.hits = self.misses = self.ticks = 0
    def state(self):
        return np.array([(self.ball_x-0.5)*2, (self.ball_y-0.5)*2,
                         self.ball_vx/BALL_SPEED, self.ball_vy/BALL_SPEED,
                         (self.paddle_y-0.5)*2])
    def step(self, action):
        self.ticks += 1
        self.paddle_y = np.clip(self.paddle_y - PADDLE_SPEED * action, 0.05, 0.95)
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        if self.ball_y <= 0 or self.ball_y >= 1:
            self.ball_vy = -self.ball_vy
            self.ball_y = np.clip(self.ball_y, 0.001, 0.999)
        if self.ball_x >= 0.92 and self.ball_vx > 0:
            if abs(self.ball_y - self.paddle_y) < 0.08:
                self.ball_vx = -self.ball_vx
                self.ball_vy += ((self.ball_y-self.paddle_y)/0.08) * BALL_SPEED * 0.3
                self.ball_x = 0.91
                self.hits += 1
        if self.ball_x > 1.1:
            self.misses += 1
            self.ball_x = 0.5
            self.ball_y = 0.5
    def done(self):
        return self.ticks >= MAX_TICKS or self.misses >= 5

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    print("=" * 65)
    print("  🔬 PONG CHAOS vs PATTERN — SPIGRAD на СИНТЕТИЧЕСКОМ МОЗГЕ")
    print("  Pure Python (numpy) — 2244 нейрона, 1.4M синапсов")
    print("=" * 65)
    
    # ── 1. Load synthetic brain ────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  1. ЗАГРУЗКА SYNTHETIC BRAIN")
    print(f"{'─'*65}")
    n, rp, ci, w, nt_types, zero_in, zero_out = load_synthetic_brain()
    n_syn = len(w)
    
    # Sensory/motor groups
    n_sensory_avail = min(len(zero_in), N_SENSORY * 5)
    sensory_idx = zero_in[:n_sensory_avail].tolist() if n_sensory_avail > 0 else list(range(N_SENSORY))
    
    # Motor: prefer zero-out neurons, fallback to last neurons
    if len(zero_out) >= N_MOTOR:
        mn_idx = zero_out[:N_MOTOR].tolist()
    elif len(zero_out) > 0:
        mn_idx = zero_out.tolist() + list(range(n - N_MOTOR + len(zero_out), n))
    else:
        mn_idx = list(range(n - N_MOTOR, n))
    n_mn = len(mn_idx)
    print(f"  Sensory group: {len(sensory_idx)} neurons")
    print(f"  Motor group:   {n_mn} neurons (IDs: {mn_idx[:5]}...{mn_idx[-3:]})")
    
    # ── 2. Init weights ────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  2. ИНИЦИАЛИЗАЦИЯ ВЕСОВ")
    print(f"{'─'*65}")
    rng = np.random.RandomState(42)
    omegas = (OMEGA_BASE + (np.arange(n)/n - 0.5)*2*OMEGA_SPREAD).astype(np.float32)
    w_in = (rng.rand(5, n).astype(np.float32) - 0.5) * 16.0   # state → input drive
    w_out = (rng.rand(2 * n_mn).astype(np.float32) - 0.5) * 0.5  # MN spike → action
    
    # T2-SPIGRAD init
    t2 = T2Spigrad(n_osc=16)
    t2_w_in = (rng.rand(5, 3*16).astype(np.float32) - 0.5) * 0.5   # state → T2 drive
    # PST dim: test forward
    test_drives = np.random.randn(10, 3*16).astype(np.float32) * 0.1
    t2.reset(42)
    test_pst = t2.forward(test_drives, 10)
    pst_dim = len(test_pst)
    t2_w_out = (rng.rand(pst_dim, 2).astype(np.float32) - 0.5) * 0.1
    print(f"  T2-SPIGRAD: 3×16=48 oscillators, PST dim={pst_dim}")
    
    # ── 3. CEM readout ─────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  3. CEM READOUT OPTIMIZATION")
    print(f"{'─'*65}")
    
    def eval_pong(w_ro):
        pong = Pong()
        while not pong.done():
            state = pong.state()
            drive = np.zeros(n, dtype=np.float32)
            for si in sensory_idx[:50]:
                if si < n:
                    drive[si] = np.dot(state, w_in[:, si]).item()
            rng_local = np.random.RandomState(pong.ticks + 42)
            phases = rng_local.rand(n).astype(np.float32) * 2 * np.pi
            pp = phases.copy()
            mc = np.zeros(n_mn)
            for _ in range(N_KSTEPS):
                kuramoto_step(phases, omegas + drive, rp, ci, w)
                sp = detect_spikes(phases, pp)
                pp = phases.copy()
                for mi, mn in enumerate(mn_idx):
                    if mn < n and sp[mn]:
                        mc[mi] += 1
            up = np.dot(mc, w_ro[:n_mn])
            dn = np.dot(mc, w_ro[n_mn:])
            pong.step(1.0 if up > dn else -1.0)
        return pong.hits - pong.misses * 0.5
    
    t1 = time.time()
    for it in range(CEM_ITERS):
        cand = []
        for _ in range(CEM_POP):
            wr = w_out + CEM_STD * rng.randn(len(w_out)).astype(np.float32)
            cand.append((eval_pong(wr), wr))
        cand.sort(key=lambda x: -x[0])
        w_out = np.mean([x[1] for x in cand[:CEM_ELITE]], axis=0)
        scores = [x[0] for x in cand]
        print(f"  iter {it}: best={cand[0][0]:.1f}  mean={np.mean(scores):.1f}±{np.std(scores):.1f}")
    print(f"  done ({time.time()-t1:.1f}s)")
    
    w_initial = w.copy()
    
    # ── 4. CHAOS/PATTERN STDP + SPIGRAD ────────────────────────────
    print(f"\n{'─'*65}")
    print("  4. CHAOS/PATTERN: dopamine-STDP + T2-SPIGRAD")
    print(f"{'─'*65}")
    header = f"  {'Ep':>3s} {'R_pat':>7s} {'R_cha':>7s} {'H_pat':>6s} {'H_cha':>6s} {'Score':>5s} {'δ':>8s} {'H/M':>5s} {'Gate':>5s}"
    print(header)
    print("  " + "-"*58)
    
    t2 = time.time()
    delta_h, rp_h, rc_h = [], [], []
    
    for ep in range(N_EPISODES):
        seed = ep * 7919 + 42
        elig = np.zeros(n_syn, dtype=np.float32)
        
        # ── PATTERN: Play Pong ──
        pong = Pong()
        rng_p = np.random.RandomState(seed)
        phases = rng_p.rand(n).astype(np.float32) * 2 * np.pi
        pp = phases.copy()
        coh_sum = 0.0
        coh_cnt = 0
        spike_counts = np.zeros(n)
        total_steps = 0
        critical_steps = 0
        recorded_states = []
        recorded_actions = []
        
        while not pong.done():
            state = pong.state()
            recorded_states.append(state)
            drive = np.zeros(n, dtype=np.float32)
            for si in sensory_idx[:50]:
                if si < n:
                    drive[si] = np.dot(state, w_in[:, si]).item()
            
            for _ in range(N_KSTEPS):
                kuramoto_step(phases, omegas + drive, rp, ci, w)
                coh_sum += phase_coherence(phases)
                coh_cnt += 1
                sp = detect_spikes(phases, pp)
                pp = phases.copy()
                spike_counts += sp.astype(np.float32)
                total_steps += 1
                
                # Eligibility trace
                for i in range(n):
                    if not sp[i]:
                        continue
                    for idx in range(rp[i], rp[i+1]):
                        if idx < n_syn:
                            elig[idx] += ELIG_ADD
                elig *= ELIG_DECAY
                
                if pong.ball_x > 0.5:
                    critical_steps += 1
            
            mc = np.array([sp[mi] for mi in mn_idx])
            up = np.dot(mc, w_out[:n_mn])
            dn = np.dot(mc, w_out[n_mn:])
            action = 1.0 if up > dn else -1.0
            recorded_actions.append(0 if action < 0 else 1)
            pong.step(action)
        
        reward = pong.hits - pong.misses * 0.5
        pattern_r = coh_sum / max(coh_cnt, 1)
        pattern_h, pat_err = spike_entropy(spike_counts, total_steps, H_TARGET)
        
        # Dopamine gate
        dopamine_gate = min(1.0, critical_steps / max(total_steps, 1) * 2.0)
        
        # ── CHAOS ──
        noise = (np.random.RandomState(seed + 555).rand(n) - 0.5) * 10.0
        phases_chaos = np.random.RandomState(seed + 999).rand(n).astype(np.float32) * 2 * np.pi
        pp_chaos = phases_chaos.copy()
        chaos_coh = 0.0
        chaos_cnt = 0
        chaos_counts = np.zeros(n)
        chaos_steps = N_KSTEPS * MAX_TICKS
        
        for _ in range(chaos_steps):
            kuramoto_step(phases_chaos, omegas + noise, rp, ci, w)
            chaos_coh += phase_coherence(phases_chaos)
            chaos_cnt += 1
            sp_c = detect_spikes(phases_chaos, pp_chaos)
            pp_chaos = phases_chaos.copy()
            chaos_counts += sp_c.astype(np.float32)
        
        chaos_r = chaos_coh / max(chaos_cnt, 1)
        chaos_h, chaos_err = spike_entropy(chaos_counts, chaos_steps, H_TARGET)
        
        # ── δ ──
        pat_util = pattern_r - ALPHA * pat_err
        chaos_util = chaos_r - ALPHA * chaos_err
        delta = reward + BETA * (pat_util - chaos_util)
        
        # ── dopamine-STDP ──
        w += LR * delta * elig * dopamine_gate
        pos_mask = w >= 0
        neg_mask = w < 0
        w = np.where(pos_mask, np.clip(w, 0.01, 3.0), np.clip(w, -3.0, -0.01))
        
        # ── T2-SPIGRAD ──
        t2_loss = 0.0
        if len(recorded_states) > 5:
            states_arr = np.array(recorded_states, dtype=np.float32)
            n_ticks = len(states_arr)
            drives_t2 = states_arr @ t2_w_in  # (n_ticks, 48)
            
            t2.reset(seed)
            pst_out, traj = t2.forward(drives_t2, n_ticks, record=True)
            
            # Readout
            action_logits = pst_out @ t2_w_out  # (pst_dim,) @ (pst_dim, 2) → (2,)
            
            actions_arr = np.array(recorded_actions, dtype=np.int32)
            n_up = np.sum(actions_arr)
            n_down = len(actions_arr) - n_up
            preferred = 1.0 if n_up > n_down else 0.0
            
            if reward > 0:
                target = np.array([0.0, 1.0]) if preferred < 0.5 else np.array([1.0, 0.0])
            elif reward < 0:
                target = np.array([1.0, 0.0]) if preferred < 0.5 else np.array([0.0, 1.0])
            else:
                target = np.array([0.5, 0.5])
            
            probs = np.exp(action_logits) / np.sum(np.exp(action_logits))
            t2_loss = float(np.mean((probs - target) ** 2))
            
            # Gradients
            d_probs = 2 * (probs - target) / len(action_logits)
            grad_logits = probs * (d_probs - np.sum(d_probs * probs))
            grad_pst = t2_w_out @ grad_logits
            
            grad_drives = t2.backward(grad_pst, traj)
            
            # dL/d_w_in
            grad_w_in = states_arr.T.astype(np.float32) @ grad_drives / n_ticks
            grad_w_out = pst_out[:, None] @ grad_logits[None, :]
            
            t2_lr = 0.005
            t2_w_in -= t2_lr * delta * grad_w_in
            t2_w_out -= t2_lr * delta * grad_w_out[:pst_dim, :]
            t2_w_in = np.clip(t2_w_in, -5.0, 5.0)
            t2_w_out = np.clip(t2_w_out, -1.0, 1.0)
        
        delta_h.append(delta)
        rp_h.append(pattern_r)
        rc_h.append(chaos_r)
        
        if ep % 5 == 0 or ep == N_EPISODES - 1:
            print(f"  {ep:3d} {pattern_r:7.4f} {chaos_r:7.4f} {pattern_h:6.3f} {chaos_h:6.3f} {reward:5.1f} {delta:8.3f} {pong.hits:2d}/{pong.misses:2d} {dopamine_gate:.2f}")
    
    print(f"  training done ({time.time()-t2:.1f}s)")
    
    # ── 5. RESULTS ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  РЕЗУЛЬТАТЫ")
    print(f"{'='*65}")
    print(f"  R_pat avg:  {np.mean(rp_h):.4f}  (first 5: {np.mean(rp_h[:5]):.4f}, last 5: {np.mean(rp_h[-5:]):.4f})")
    print(f"  R_cha avg:  {np.mean(rc_h):.4f}  (first 5: {np.mean(rc_h[:5]):.4f}, last 5: {np.mean(rc_h[-5:]):.4f})")
    print(f"  R_gap:      {np.mean(rp_h)-np.mean(rc_h):+.4f}")
    print(f"  δ first 5:  {np.mean(delta_h[:5]):.3f}")
    print(f"  δ last 5:   {np.mean(delta_h[-5:]):.3f}")
    trend = "UP 📈" if np.mean(delta_h[-5:]) > np.mean(delta_h[:5]) else "DOWN 📉"
    print(f"  δ trend:    {trend}")
    print()
    
    # ── 6. WEIGHT DELTA ────────────────────────────────────────────
    print(f"{'='*65}")
    print("  WEIGHT DELTA: BEFORE vs AFTER STDP")
    print(f"{'='*65}")
    dw = w - w_initial
    abs_dw = np.abs(dw)
    print(f"  Δ weight:    mean={np.mean(dw):+.6f}±{np.std(dw):.6f}")
    print(f"  |Δ| weight:  mean={np.mean(abs_dw):.6f}±{np.std(abs_dw):.6f}")
    print(f"  Max |Δ|:     {np.max(abs_dw):.6f}")
    print(f"  Total Δ sum: {np.sum(abs_dw):.3f} over {n_syn} synapses")
    print()
    
    print("  ── Change thresholds ──")
    for thresh in [0.001, 0.005, 0.01, 0.05, 0.1]:
        n_changed = int(np.sum(abs_dw > thresh))
        pct = n_changed / n_syn * 100
        print(f"  |Δ| > {thresh:.3f}: {n_changed:6d} / {n_syn} ({pct:5.2f}%)")
    print()
    
    ei_before = np.sum(w_initial > 0)
    ei_after = np.sum(w > 0)
    inhib_before = np.sum(w_initial < 0)
    inhib_after = np.sum(w < 0)
    print(f"  E count:    {ei_before:5d} → {ei_after:5d} (Δ={ei_after-ei_before:+d})")
    print(f"  I count:    {inhib_before:5d} → {inhib_after:5d} (Δ={inhib_after-inhib_before:+d})")
    print(f"  E/I ratio:  {ei_before/inhib_before:.2f} → {ei_after/inhib_after:.2f}")
    print()
    
    corr = np.corrcoef(w_initial, w)[0, 1]
    print(f"  Weight correlation (before vs after): {corr:.6f}")
    print()
    
    # Top-10 most changed synapses
    print("  ── Top 10 most changed synapses ──")
    top_idx = np.argsort(-abs_dw)[:10]
    for idx in top_idx:
        pre = -1
        for i in range(n):
            if rp[i] <= idx < rp[i+1]:
                pre = i
                break
        post = ci[idx]
        print(f"    Syn [{idx:5d}] {pre:4d} → {post:4d}  {w_initial[idx]:+.4f} → {w[idx]:+.4f}  |Δ|={abs_dw[idx]:.4f}")
    
    print()
    print(f"  Total time: {time.time()-t_start:.1f}s")
    print("=" * 65)
    print("  ГОТОВО 🧠🔥")
    print("=" * 65)

if __name__ == '__main__':
    main()
