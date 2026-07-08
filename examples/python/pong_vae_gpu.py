"""🏓 Pong Chaos vs Pattern — VAE + Kuramoto + dopamine-STDP (GPU)"""

import sys, time
from pathlib import Path
import numpy as np
import taas_checkmate as tcm

# ── Config ──
DT = 0.01
SPIKE_THRESHOLD = 2.5
OMEGA_BASE = 3.0
OMEGA_SPREAD = 1.5
LR = 0.03          # STDP learning rate
ALPHA = 0.3         # entropy penalty weight
BETA = 0.5          # chaos/pattern gap weight (выше = сильнее паттерн)
H_TARGET = 3.0      # target entropy
ELIG_DECAY = 0.90   # τ_trace ≈ 85ms
ELIG_ADD = 0.15     # eligibility increment per spike
CEM_STD = 0.5       # CEM noise std (exploration)
MAX_GAME_TICKS = 100
PADDLE_SPEED = 0.04
BALL_SPEED = 0.018
CEM_POP = 15
CEM_ELITE = 5
CEM_ITERS = 5
N_EPISODES = 100
EXPLORE_NOISE = 0.05 # dopamine noise when score=-1
BASELINE_DECAY = 0.9 # скользящее среднее для baseline
N_KURAMOTO_STEPS = 10
# Chaos matches max pattern episode length for fair comparison
N_CHAOS_STEPS = N_KURAMOTO_STEPS * MAX_GAME_TICKS

# ── T2-SPIGRAD Config ──
T2_N_OSC = 16          # oscillators per population (3 pops = 48 total)
T2_DRIVE_GAIN = 1.0
T2_ALPHA = 1.0
T2_FREQ_MIN = 1.0
T2_FREQ_MAX = 8.0
T2_C12 = 0.3            # coupling sensory→inter
T2_C13 = 0.2            # coupling sensory→motor (direct)
T2_C23 = 0.1            # coupling inter→motor
T2_LR = 0.005           # T2-SPIGRAD learning rate

BASE = Path(__file__).parent.parent.parent
BRAIN_PATH = BASE / "crates" / "connectome-gpt" / "models" / "brain.taas"

# ═══════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════

def spike_entropy(counts, total_steps, h_target):
    eps = 1e-10
    p = counts / max(total_steps, 1)
    p = np.clip(p, eps, 1.0)
    h = -np.sum(p * np.log2(p))
    return h, abs(h - h_target)

def compute_coherence(phases):
    """Compute phase coherence from current phases."""
    return float(tcm.cpe.phase_coherence(phases.astype(np.float32)))

def detect_spikes(phases, phases_prev):
    return ((phases > SPIKE_THRESHOLD) & (phases_prev <= SPIKE_THRESHOLD)).astype(np.int32)

def run_kuramoto_with_recording(n, omegas, rp, ci, w, n_steps, seed, drive_arr=None):
    """Run Kuramoto, return (mean_coherence, spike_counts, total_steps)."""
    rng = np.random.RandomState(seed)
    phases = rng.rand(n).astype(np.float32) * 2 * np.pi
    pp = phases.copy()
    coh_sum = 0.0
    spike_counts = np.zeros(n)
    
    for step in range(n_steps):
        omega_mod = omegas.copy()
        if drive_arr is not None:
            omega_mod += drive_arr[:n].astype(np.float32) if isinstance(drive_arr, np.ndarray) else drive_arr
        tcm.kuramoto_step_sparse(phases, omega_mod, rp, ci, w, DT)
        sp = detect_spikes(phases, pp)
        pp = phases.copy()
        coh_sum += compute_coherence(phases)
        spike_counts += sp.astype(np.float32)
    
    return coh_sum / max(n_steps, 1), spike_counts, n_steps

# ═══════════════════════════════════════════════
#  Load baked connectome
# ═══════════════════════════════════════════════

def load_brain(path):
    cd = tcm.load_connectome(str(path))
    n = cd.n_neurons
    n_syn = cd.n_synapses
    rp = np.array(cd.row_ptr(), dtype=np.int32)
    ci = np.array(cd.col_indices(), dtype=np.int32)
    v = np.array(cd.values(), dtype=np.float32)
    
    # Extract groups
    gnames = list(cd.group_names())
    si = list(cd.group('sensory')) if 'sensory' in gnames else list(range(0, max(1, n//20)))
    mi = list(cd.group('mn')) if 'mn' in gnames else list(range(max(0, n - n//50), n))
    
    print(f"  Loaded: {n} neurons, {n_syn} synapses")
    print(f"  Groups: sensory={len(si)}, mn={len(mi)}")
    return n, rp, ci, v, si, mi

# ═══════════════════════════════════════════════
#  Pong
# ═══════════════════════════════════════════════

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
        return np.array([(self.ball_x-0.5)*2, (self.ball_y-0.5)*2, self.ball_vx/BALL_SPEED, self.ball_vy/BALL_SPEED, (self.paddle_y-0.5)*2])
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
            # Reset ball only, not the whole game
            self.ball_x = 0.5
            self.ball_y = 0.5
    def done(self):
        return self.ticks >= MAX_GAME_TICKS or self.misses >= 5

# ═══════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 60)
    print("  Pong Chaos vs Pattern - VAE + Kuramoto + dopamine-STDP")
    print("=" * 60)
    
    # 1. Load brain
    print(f"\n1. Loading brain from {BRAIN_PATH}...")
    n, rp, ci, w, sensory_idx, mn_idx = load_brain(BRAIN_PATH)
    n_syn = len(w)
    n_mn = len(mn_idx)
    
    omegas = (OMEGA_BASE + (np.arange(n)/n - 0.5)*2*OMEGA_SPREAD).astype(np.float32)
    rng = np.random.RandomState(42)
    w_in = (rng.rand(5, n).astype(np.float32) - 0.5) * 16.0
    w_out = (rng.rand(2*n_mn).astype(np.float32) - 0.5) * 0.5
    
    # T2-SPIGRAD weights (separate from Kuramoto — gradient-based sensorimotor map)
    n_t2_drive = 3 * T2_N_OSC
    w_in_t2 = (rng.rand(5, n_t2_drive).astype(np.float32) - 0.5) * 0.5  # sensory → T3 drives
    # PST dim: computed from first forward pass
    _test_drives = np.random.randn(10, n_t2_drive).astype(np.float32) * 0.1
    _test_pst, _ = tcm.t2spigrad_forward(
        T2_N_OSC, 10, DT, T2_DRIVE_GAIN, T2_ALPHA,
        T2_FREQ_MIN, T2_FREQ_MAX, T2_C12, T2_C13, T2_C23, _test_drives)
    pst_dim = len(_test_pst)
    w_out_t2 = (rng.rand(pst_dim, 2).astype(np.float32) - 0.5) * 0.1  # PST → action logits
    print(f"  T2-SPIGRAD: {T2_N_OSC} osc/pop, PST dim={pst_dim}")
    
    # 2. Quick eval
    def eval_pong(w_ro):
        pong = Pong()
        while not pong.done():
            state = pong.state()
            drive = np.zeros(n, dtype=np.float32)
            for si in sensory_idx[:50]:
                if si < n: drive[si] = np.dot(state, w_in[:, si]).item()
            rng_local = np.random.RandomState(pong.ticks + 42)
            phases = rng_local.rand(n).astype(np.float32) * 2*np.pi
            pp = phases.copy()
            mc = np.zeros(n_mn)
            for _ in range(N_KURAMOTO_STEPS):
                tcm.kuramoto_step_sparse(phases, omegas+drive, rp, ci, w, DT)
                sp = detect_spikes(phases, pp)
                pp = phases.copy()
                for mi, mn in enumerate(mn_idx):
                    if mn < n and sp[mn]: mc[mi] += 1
            up = np.dot(mc, w_ro[:n_mn])
            dn = np.dot(mc, w_ro[n_mn:])
            pong.step(1.0 if up > dn else -1.0)
        return pong.hits - pong.misses * 0.5
    
    print(f"\n2. CEM readout (pop={CEM_POP}, elite={CEM_ELITE}, iters={CEM_ITERS}, std={CEM_STD})...")
    t1 = time.time()
    cem_scores = []
    for it in range(CEM_ITERS):
        cand = []
        for _ in range(CEM_POP):
            wr = w_out + CEM_STD * rng.randn(len(w_out)).astype(np.float32)
            cand.append((eval_pong(wr), wr))
        cand.sort(key=lambda x: -x[0])
        w_out = np.mean([x[1] for x in cand[:CEM_ELITE]], axis=0)
        scores_cem = [x[0] for x in cand]
        cem_scores.append(cand[0][0])
        print(f"  iter {it}: best={cand[0][0]:.1f}  mean={np.mean(scores_cem):.1f}±{np.std(scores_cem):.1f}")
    if cem_scores:
        print(f"  CEM score evolution: {' → '.join([f'{s:.1f}' for s in cem_scores])}")
    print(f"  done ({time.time()-t1:.1f}s)")
    print(f"  w_out final: mean={np.mean(w_out):.4f}, std={np.std(w_out):.4f}")
    
    # Save initial weights for delta comparison
    w_initial = w.copy()
    
    # 3. Chaos/Pattern STDP
    print(f"\n3. Hybrid: Kuramoto-STDP + T2-SPIGRAD (LR={LR}, β={BETA}, T2_LR={T2_LR}, τ≈85ms)...")
    header = f"  {'Ep':>3s} {'R_pat':>7s} {'R_cha':>7s} {'H_pat':>7s} {'H_cha':>7s} {'Score':>6s} {'δ':>8s} {'H/M':>6s} {'Gate':>5s} {'T2Loss':>7s}"
    print(header)
    print(f"  {'-'*3} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*6} {'-'*5} {'-'*7}")
    
    t2 = time.time()
    delta_h, rp_h, rc_h = [], [], []
    baseline = 0.0  # скользящее среднее награды
    
    for ep in range(N_EPISODES):
        seed = ep * 7919 + 42
        elig = np.zeros(n_syn, dtype=np.float32)
        
        # ── PATTERN: Play Pong ──
        critical_steps = 0
        recorded_states = []  # for T2-SPIGRAD
        recorded_actions = [] # action taken at each tick
        pong = Pong()
        rng_p = np.random.RandomState(seed)
        phases = rng_p.rand(n).astype(np.float32) * 2*np.pi
        pp = phases.copy()
        coh_sum = 0.0
        coh_count = 0
        spike_counts = np.zeros(n)
        total_steps = 0
        
        while not pong.done():
            state = pong.state()
            recorded_states.append(state)
            drive = np.zeros(n, dtype=np.float32)
            for si in sensory_idx[:50]:
                if si < n: drive[si] = np.dot(state, w_in[:, si]).item()
            
            for _ in range(N_KURAMOTO_STEPS):
                tcm.kuramoto_step_sparse(phases, omegas+drive, rp, ci, w, DT)
                coh_sum += compute_coherence(phases)
                coh_count += 1
                sp = detect_spikes(phases, pp)
                pp = phases.copy()
                spike_counts += sp.astype(np.float32)
                total_steps += 1
                
                # ── Eligibility trace (τ_trace ≈ 85ms) ──
                for i in range(n):
                    if not sp[i]: continue
                    for idx in range(rp[i], rp[i+1]):
                        if idx < n_syn: elig[idx] += ELIG_ADD
                elig *= ELIG_DECAY
                
                # Track critical steps (ball in paddle's half)
                if pong.ball_x > 0.5:
                    critical_steps += 1
            
            # Readout
            mc = np.array([np.sum(sp[mn_idx[m]]) for m in range(n_mn)])
            up = np.dot(mc, w_out[:n_mn])
            dn = np.dot(mc, w_out[n_mn:])
            action = 1.0 if up > dn else -1.0
            recorded_actions.append(0 if action < 0 else 1)  # 0=down, 1=up
            pong.step(action)
        
        reward = pong.hits - pong.misses * 0.5
        pattern_r = coh_sum / max(coh_count, 1)
        pattern_h, pat_err = spike_entropy(spike_counts, total_steps, H_TARGET)
        
        # ── Dopamine timing gate ──
        total_critical = max(critical_steps, 1)
        total_possible = max(total_steps, 1)
        dopamine_gate = min(1.0, total_critical / total_possible * 2.0)
        # Gate scales from 0 (no critical activity) to 1 (≥50% critical)
        
        # ── CHAOS: random noise (same number of steps) ──
        noise = (np.random.RandomState(seed+555).rand(n) - 0.5) * 10.0
        chaos_r, chaos_counts, chaos_steps = run_kuramoto_with_recording(
            n, omegas, rp, ci, w, N_CHAOS_STEPS, seed+999, noise)
        chaos_h, chaos_err = spike_entropy(chaos_counts, chaos_steps, H_TARGET)
        
        # ── Reward baseline (скользящее среднее) ──
        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * reward
        reward_centered = reward - baseline  # положительное = лучше чем обычно
        
        # ── δ = reward_centered + β · [(R_pat − α·|H_pat−H₀|) − (R_chaos − α·|H_chaos−H₀|)] ──
        pat_util = pattern_r - ALPHA * pat_err
        chaos_util = chaos_r - ALPHA * chaos_err
        delta = reward_centered + BETA * (pat_util - chaos_util)
        # Exploration noise: если δ ≈ 0, подмешиваем шум
        if abs(delta) < 0.1:
            delta += np.random.randn() * EXPLORE_NOISE
        
        # ── Apply dopamine-STDP (gated by timing) ──
        w += LR * delta * elig * dopamine_gate
        # Clip preserving sign (ЗАПРЕТ смены знака)
        orig_sign = np.sign(w_initial)
        pos_mask = orig_sign > 0
        neg_mask = orig_sign < 0
        w = np.where(pos_mask, np.clip(w, 0.01, 3.0), w)
        w = np.where(neg_mask, np.clip(w, -3.0, -0.01), w)
        
        # ── T2-SPIGRAD: gradient-based sensorimotor learning ──
        t2_loss = 0.0
        if len(recorded_states) > 5:
            # Build drives matrix from recorded states
            states_arr = np.array(recorded_states, dtype=np.float32)  # (n_ticks, 5)
            n_ticks = len(states_arr)
            drives_t2 = states_arr @ w_in_t2  # (n_ticks, 3*T2_N_OSC)
            
            # T2 forward: run T3 oscillator dynamics
            pst_out, traj = tcm.t2spigrad_forward(
                T2_N_OSC, n_ticks, DT, T2_DRIVE_GAIN, T2_ALPHA,
                T2_FREQ_MIN, T2_FREQ_MAX, T2_C12, T2_C13, T2_C23, drives_t2)
            pst_out = np.array(pst_out, dtype=np.float32)
            
            # Readout: PST → action logits
            action_logits = pst_out @ w_out_t2  # (pst_dim,) @ (pst_dim, 2) → (2,)
            
            # REINFORCE loss: maximize log_prob of taken action if reward > 0
            actions_arr = np.array(recorded_actions, dtype=np.int32)
            n_up = np.sum(actions_arr)
            n_down = len(actions_arr) - n_up
            preferred = 1.0 if n_up > n_down else 0.0 if n_down > n_up else 0.5
            
            # Target: prefer up if reward > 0, down if reward < 0
            if reward > 0:
                target = np.array([1.0, 0.0]) if preferred < 0.5 else np.array([0.0, 1.0])
            elif reward < 0:
                target = np.array([0.0, 1.0]) if preferred < 0.5 else np.array([1.0, 0.0])
            else:
                target = np.array([0.5, 0.5])
            
            # MSE loss: (logits - target)^2
            probs = np.exp(action_logits) / np.sum(np.exp(action_logits))
            t2_loss = float(np.mean((probs - target) ** 2))
            
            # Gradient of loss wrt PST
            d_probs = 2 * (probs - target) / len(action_logits)
            grad_logits = probs * (d_probs - np.sum(d_probs * probs))  # softmax jacobian
            grad_pst = w_out_t2 @ grad_logits  # (pst_dim,)
            
            # T2 backward (grad must be float32 contiguous)
            grad_pst = np.ascontiguousarray(grad_pst.astype(np.float32))
            grad_drives = tcm.t2spigrad_backward(grad_pst, traj)  # (n_ticks, 3*T2_N_OSC)
            
            # Chain: dL/d_w_in_t2 = states_arr.T @ grad_drives / n_ticks
            grad_w_in = states_arr.T.astype(np.float32) @ np.array(grad_drives, dtype=np.float32) / n_ticks
            # Chain: dL/d_w_out_t2 = pst_out[:, None] @ grad_logits[None, :]
            grad_w_out = pst_out[:, None] @ grad_logits[None, :]  # (pst_dim, 2)
            
            # Apply updates gated by δ
            w_in_t2 -= T2_LR * delta * grad_w_in
            w_out_t2 -= T2_LR * delta * grad_w_out[:pst_dim, :]
            # Clip
            w_in_t2 = np.clip(w_in_t2, -5.0, 5.0)
            w_out_t2 = np.clip(w_out_t2, -1.0, 1.0)
        
        delta_h.append(delta)
        rp_h.append(pattern_r)
        rc_h.append(chaos_r)
        
        if ep % 3 == 0 or ep == N_EPISODES - 1:
            print(f"  {ep:3d} {pattern_r:7.4f} {chaos_r:7.4f} {pattern_h:7.3f} {chaos_h:7.3f} {reward:6.1f} {delta:8.3f} {pong.hits:3d}/{pong.misses:2d} {dopamine_gate:.2f} {t2_loss:.4f}")
    
    print(f"  done ({time.time()-t2:.1f}s)")
    print()
    
    # Results
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  R_pat avg:  {np.mean(rp_h):.4f}  (first 5: {np.mean(rp_h[:5]):.4f}, last 5: {np.mean(rp_h[-5:]):.4f})")
    print(f"  R_cha avg:  {np.mean(rc_h):.4f}  (first 5: {np.mean(rc_h[:5]):.4f}, last 5: {np.mean(rc_h[-5:]):.4f})")
    print(f"  R_gap:      {np.mean(rp_h)-np.mean(rc_h):+.4f}")
    print(f"  δ first 5:  {np.mean(delta_h[:5]):.3f}")
    print(f"  δ last 5:   {np.mean(delta_h[-5:]):.3f}")
    trend = "UP 📈" if np.mean(delta_h[-5:]) > np.mean(delta_h[:5]) else "DOWN 📉"
    print(f"  δ trend:    {trend}")
    print()
    
    # ═══════════════════════════════════════════
    #  WEIGHT DELTA ANALYSIS
    # ═══════════════════════════════════════════
    print("═" * 60)
    print("  WEIGHT DELTA: BEFORE vs AFTER STDP")
    print("═" * 60)
    
    dw = w - w_initial
    abs_dw = np.abs(dw)
    
    # Basic stats
    print(f"  Δ weight:    mean={np.mean(dw):+.6f}±{np.std(dw):.6f}")
    print(f"  |Δ| weight:  mean={np.mean(abs_dw):.6f}±{np.std(abs_dw):.6f}")
    print(f"  Max |Δ|:     {np.max(abs_dw):.6f}")
    print(f"  Min |Δ|:     {np.min(abs_dw):.6f}")
    print(f"  Total Δ sum: {np.sum(abs_dw):.3f} over {n_syn} synapses")
    print()
    
    # Fraction changed at various thresholds
    print("  ── Change thresholds ──")
    for thresh in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]:
        n_changed = int(np.sum(abs_dw > thresh))
        pct = n_changed / n_syn * 100
        bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
        print(f"  |Δ| > {thresh:.3f}: {n_changed:6d} / {n_syn} ({pct:5.1f}%) {bar}")
    print()
    
    # Sign flips (E→I or I→E)
    sign_before = np.sign(w_initial)
    sign_after = np.sign(w)
    zero_mask = (sign_before == 0) | (sign_after == 0)
    flips = np.sum((sign_before != sign_after) & ~zero_mask)
    print(f"  Sign flips (E↔I): {flips} ({flips/n_syn*100:.2f}% of all synapses)")
    print()
    
    # E/I shift
    ei_before = np.sum(w_initial > 0)
    ei_after = np.sum(w > 0)
    inhib_before = np.sum(w_initial < 0)
    inhib_after = np.sum(w < 0)
    print(f"  E count:    {ei_before:5d} → {ei_after:5d} (Δ={ei_after-ei_before:+d})")
    print(f"  I count:    {inhib_before:5d} → {inhib_after:5d} (Δ={inhib_after-inhib_before:+d})")
    print(f"  E/I ratio:  {ei_before/inhib_before:.2f} → {ei_after/inhib_after:.2f}")
    print()
    
    # Weight distribution
    print("  ── Weight distribution ──")
    print(f"  Before: mean={np.mean(w_initial):.4f}±{np.std(w_initial):.4f}  "
          f"E:mean={np.mean(w_initial[w_initial>0]):.4f}  I:mean={np.mean(w_initial[w_initial<0]):.4f}")
    print(f"  After:  mean={np.mean(w):.4f}±{np.std(w):.4f}  "
          f"E:mean={np.mean(w[w>0]):.4f}  I:mean={np.mean(w[w<0]):.4f}")
    print()
    
    # Per-neuron delta (which neurons changed most?)
    print("  ── Top 10 most changed neurons (mean |Δ| per outgoing syn) ──")
    per_neuron = []
    for i in range(n):
        start = rp[i]
        end = rp[i+1]
        if end > start:
            mean_abs_dw = np.mean(abs_dw[start:end])
            total_dw = np.sum(dw[start:end])
            deg = end - start
            grp = []
            if i in sensory_idx: grp.append("sensory")
            if i in mn_idx: grp.append("MN")
            tag = f"[{','.join(grp)}]" if grp else ""
            per_neuron.append((mean_abs_dw, i, deg, total_dw, tag))
    per_neuron.sort(key=lambda x: -x[0])
    for d, idx, deg, total, tag in per_neuron[:10]:
        print(f"    Neuron {idx:4d} {tag:12s}: mean|Δ|={d:.6f}  totalΔ={total:+.6f}  deg={deg}")
    print()
    
    # Correlation
    corr = np.corrcoef(w_initial, w)[0, 1]
    print(f"  Weight correlation (before vs after): {corr:.6f}")
    print()
    
    # Top-10 most changed synapse pairs
    print("  ── Top 10 most changed synapses ──")
    top_idx = np.argsort(-abs_dw)[:10]
    for idx in top_idx:
        # Find the pre-neuron
        pre = -1
        for i in range(n):
            if rp[i] <= idx < rp[i+1]:
                pre = i
                break
        post = ci[idx]
        pre_tag = "sensory" if pre in sensory_idx else ("MN" if pre in mn_idx else "")
        post_tag = "sensory" if post in sensory_idx else ("MN" if post in mn_idx else "")
        print(f"    Syn [{idx:5d}] {pre:4d}{'('+pre_tag+')' if pre_tag else '':8s} → {post:4d}{'('+post_tag+')' if post_tag else '':8s}  "
              f"{w_initial[idx]:+.4f} → {w[idx]:+.4f}  |Δ|={abs_dw[idx]:.4f}")
    
    print()
    print(f"  Total:      {time.time()-t0:.1f}s")
    print("═" * 60)
    print("  END")
    print("═" * 60)

if __name__ == '__main__':
    main()
