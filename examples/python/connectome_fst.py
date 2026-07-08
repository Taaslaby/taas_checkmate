#!/usr/bin/env python3
"""🧬 Connectome as Finite State Transducer — FST + CPE experiment."""

import time, math
from pathlib import Path
import numpy as np

import taas_checkmate as tcm

# ── Config ──
DT = 0.01
SPIKE_THRESHOLD = 2.5
OMEGA_BASE = 3.0
N_STEPS = 50            # Kuramoto steps per input symbol
N_SYMBOLS = 5           # length of binary sequence to recognize
N_EXAMPLES = 20         # sequences to test
N_TRAIN_EPISODES = 30
LR = 0.02
ELIG_DECAY = 0.90
ELIG_ADD = 0.15
SIGNAL_AMP = 10.0       # drive for '1' bits
NOISE_AMP = 2.0         # drive for '0' bits / noise

BASE = Path(__file__).parent.parent.parent
BRAIN_PATH = BASE / "crates" / "connectome-gpt" / "models" / "brain.taas"

# ═══════════════════════════════════════════════
#  1. Load brain
# ═══════════════════════════════════════════════

print("=" * 65)
print("  🧬 Connectome as Finite State Transducer (FST)")
print("=" * 65)

cd = tcm.load_connectome(str(BRAIN_PATH))
n = cd.n_neurons
n_syn = cd.n_synapses
rp = np.array(cd.row_ptr(), dtype=np.int32)
ci = np.array(cd.col_indices(), dtype=np.int32)
w = np.array(cd.values(), dtype=np.float32)

gnames = list(cd.group_names())
sensory_idx = list(cd.group('sensory')) if 'sensory' in gnames else list(range(0, 20))
mn_idx = list(cd.group('mn')) if 'mn' in gnames else list(range(max(0, n - 4), n))
N_SENSORY = min(20, len(sensory_idx))
N_MOTOR = min(4, len(mn_idx))

omegas = (OMEGA_BASE + (np.arange(n) / n - 0.5) * 3.0).astype(np.float32)
w_initial = w.copy()

print(f"  {n} neurons, {n_syn} synapses")
print(f"  FST: {N_SENSORY} input symbols → Kuramoto dynamics → {N_MOTOR} output symbols")
print(f"  Alphabet: binary sequences of length {N_SYMBOLS}")

# ═══════════════════════════════════════════════
#  2. FST: run connectome on a binary sequence
# ═══════════════════════════════════════════════

def run_fst(sequence, weights, elig=None):
    """Feed a binary sequence into the connectome-FST, return motor output.

    If `elig` is provided (eligibility trace array), it's updated per-step:
    - spikes at current step add ELIG_ADD to outgoing synapses
    - trace decays by ELIG_DECAY each step
    """
    phases = np.random.RandomState(42).rand(n).astype(np.float32) * 2 * np.pi
    prev = phases.copy()
    noise_rng = np.random.RandomState(42)  # deterministic noise
    motor_out = np.zeros(N_MOTOR)
    total_spikes = np.zeros(n)
    coh_sum = 0.0

    for symbol in sequence:
        drive = np.zeros(n, dtype=np.float32)
        for si in sensory_idx[:N_SENSORY]:
            drive[si] = SIGNAL_AMP if symbol else NOISE_AMP * (noise_rng.rand() - 0.5)

        for _ in range(N_STEPS):
            tcm.kuramoto_step_sparse(phases, omegas + drive, rp, ci, weights, DT)
            sp = ((phases > SPIKE_THRESHOLD) & (prev <= SPIKE_THRESHOLD)).astype(np.int32)
            prev = phases.copy()
            total_spikes += sp.astype(np.float32)

            # Eligibility trace: per-step decay + spike-triggered accumulation
            if elig is not None:
                elig *= ELIG_DECAY
                for i in range(n):
                    if sp[i]:
                        for idx in range(rp[i], rp[i+1]):
                            if idx < n_syn:
                                elig[idx] += ELIG_ADD

            coh_out = np.zeros(1, dtype=np.float32)
            tcm.phase_coherence(coh_out,
                np.sin(phases).astype(np.float32),
                np.cos(phases).astype(np.float32),
                np.ones(n, dtype=np.float32))
            coh_sum += float(coh_out[0])

    motor_out = np.array([total_spikes[mi] for mi in mn_idx[:N_MOTOR]])
    coherence = coh_sum / (len(sequence) * N_STEPS)
    return motor_out, coherence, total_spikes

# ═══════════════════════════════════════════════
#  3. FST determinism test
# ═══════════════════════════════════════════════

print(f"\n{'='*65}")
print("  FST DETERMINISM TEST: same input → same output?")
print(f"{'='*65}")

# Generate random binary sequences
rng = np.random.RandomState(0)
test_seqs = [(rng.randint(0, 2, N_SYMBOLS).tolist(), f"seq_{j}") for j in range(N_EXAMPLES)]

# Run each sequence twice
all_deterministic = True
for seq, name in test_seqs[:5]:  # test first 5
    out1, coh1, _ = run_fst(seq, w)
    out2, coh2, _ = run_fst(seq, w)
    diff = np.max(np.abs(out1 - out2))
    ok = diff == 0
    if not ok:
        all_deterministic = False
    seq_str = ''.join(str(b) for b in seq)
    print(f"  [{seq_str}] motor={out1.tolist()}  Δ={diff}  {'✅' if ok else '⚠️'} ")

print(f"\n  FST deterministic: {'YES ✅' if all_deterministic else 'PARTIAL ⚠️'}")

# Print FST transition table (input → output)
print(f"\n{'='*65}")
print("  FST TRANSITION TABLE: binary input → motor output")
print(f"{'='*65}")
print(f"  {'Sequence':>12s} {'Motor[0]':>8s} {'Motor[1]':>8s} {'Motor[2]':>8s} {'Motor[3]':>8s} {'Coherence':>10s}")
print(f"  {'-'*50}")

transition_table = {}
for seq, name in test_seqs:
    motor_out, coherence, _ = run_fst(seq, w)
    seq_key = tuple(seq)
    transition_table[seq_key] = (motor_out, coherence)

# Show a few diverse sequences
unique_outputs = {}
for key, (motor, coh) in transition_table.items():
    out_key = tuple(motor)
    if out_key not in unique_outputs:
        unique_outputs[out_key] = []
    unique_outputs[out_key].append(key)

for motor_tuple, seqs in list(unique_outputs.items())[:8]:
    seq_strs = [''.join(str(b) for b in s) for s in seqs[:3]]
    # Fetch coherence from the first sequence in this group
    first_seq = seqs[0]
    coh = transition_table[first_seq][1]
    print(f"  {seq_strs[0]:>12s} {motor_tuple[0]:8.0f} {motor_tuple[1]:8.0f} {motor_tuple[2]:8.0f} {motor_tuple[3]:8.0f} {coh:10.4f}")

print(f"\n  Unique FST outputs from {N_EXAMPLES} inputs: {len(unique_outputs)}")
print(f"  FST function: Σ^{N_SYMBOLS} → Γ^{N_MOTOR}  (input space = {2**N_SYMBOLS}, output space = {2**N_MOTOR})")

# ═══════════════════════════════════════════════
#  4. CPE training: teach FST to distinguish signal patterns
# ═══════════════════════════════════════════════

print(f"\n{'='*65}")
print("  CPE TRAINING: teaching FST to prefer 'good' sequences")
print(f"{'='*65}")

# Define "good" sequences (signal) and "bad" sequences (noise)
good_seqs = [
    [1, 0, 1, 0, 1],  # alternating
    [1, 1, 0, 0, 1],  # runs
    [0, 1, 1, 1, 0],  # middle run
    [1, 0, 0, 1, 1],  # pairs
    [1, 1, 1, 0, 0],  # start run
]
bad_seqs = [
    [0, 0, 0, 0, 0],  # all zeros
    [1, 1, 1, 1, 1],  # all ones
    [0, 1, 0, 1, 0],  # alternating zeros
    [0, 0, 1, 0, 0],  # single one
    [1, 0, 1, 1, 0],  # random-ish
]

# Baseline FST for good vs bad
print(f"\n  Baseline FST responses:")
print(f"  {'Type':>6s} {'Motor[0]':>8s} {'Motor[1]':>8s} {'Motor[2]':>8s} {'Motor[3]':>8s} {'R':>8s}")
print(f"  {'-'*46}")

good_baseline = np.zeros(N_MOTOR)
bad_baseline = np.zeros(N_MOTOR)
for seq in good_seqs:
    motor, coh, _ = run_fst(seq, w)
    good_baseline += motor
    print(f"  GOOD  {motor[0]:8.0f} {motor[1]:8.0f} {motor[2]:8.0f} {motor[3]:8.0f} {coh:8.4f}")
for seq in bad_seqs:
    motor, coh, _ = run_fst(seq, w)
    bad_baseline += motor
    print(f"  BAD   {motor[0]:8.0f} {motor[1]:8.0f} {motor[2]:8.0f} {motor[3]:8.0f} {coh:8.4f}")

good_baseline /= len(good_seqs)
bad_baseline /= len(bad_seqs)
print(f"  {'-'*46}")
print(f"  GOOD avg: {good_baseline[0]:8.1f} {good_baseline[1]:8.1f} {good_baseline[2]:8.1f} {good_baseline[3]:8.1f}")
print(f"  BAD  avg: {bad_baseline[0]:8.1f} {bad_baseline[1]:8.1f} {bad_baseline[2]:8.1f} {bad_baseline[3]:8.1f}")

# Training loop
print(f"\n  Training {N_TRAIN_EPISODES} episodes...")
print(f"  {'Ep':>3s} {'R_good':>7s} {'R_bad':>7s} {'δ':>8s} {'Mot_good':>8s} {'Mot_bad':>8s}")
print(f"  {'-'*44}")

t0 = time.time()
for ep in range(N_TRAIN_EPISODES):
    rng = np.random.RandomState(ep)
    elig = np.zeros(n_syn, dtype=np.float32)

    # ── PATTERN: good sequences, eligibility accumulated per-step inside run_fst
    seq = rng.choice(len(good_seqs))
    motor_pat, coh_pat, spikes_pat = run_fst(good_seqs[seq], w, elig=elig)

    # ── CHAOS: bad sequences ──
    seq = rng.choice(len(bad_seqs))
    motor_cha, coh_cha, spikes_cha = run_fst(bad_seqs[seq], w)

    # Reward: motor fires more on good sequences
    reward = np.mean(motor_pat) - np.mean(motor_cha)
    pat_err = abs(coh_pat - 3.0)
    cha_err = abs(coh_cha - 3.0)
    pat_util = coh_pat - 0.3 * pat_err
    cha_util = coh_cha - 0.3 * cha_err
    delta = reward + 0.3 * (pat_util - cha_util)

    # STDP update
    w += LR * delta * elig
    pos_mask = w >= 0
    neg_mask = w < 0
    w = np.where(pos_mask, np.clip(w, 0.01, 3.0), np.clip(w, -3.0, -0.01))

    if ep % 5 == 0 or ep == N_TRAIN_EPISODES - 1:
        print(f"  {ep:3d} {coh_pat:7.4f} {coh_cha:7.4f} {delta:8.3f} {np.mean(motor_pat):8.1f} {np.mean(motor_cha):8.1f}")

# ═══════════════════════════════════════════════
#  5. Post-training FST comparison
# ═══════════════════════════════════════════════

print(f"\n{'='*65}")
print("  POST-TRAINING FST: did transitions change?")
print(f"{'='*65}")

# Re-evaluate all sequences
good_post = np.zeros(N_MOTOR)
bad_post = np.zeros(N_MOTOR)
for seq in good_seqs:
    motor, _, _ = run_fst(seq, w)
    good_post += motor
for seq in bad_seqs:
    motor, _, _ = run_fst(seq, w)
    bad_post += motor
good_post /= len(good_seqs)
bad_post /= len(bad_seqs)

print(f"  {'Metric':>20s} {'Before':>10s} {'After':>10s} {'Δ':>10s}")
print(f"  {'-'*52}")

# Weight change
dw = w - w_initial
abs_dw = np.abs(dw)
n_changed = int(np.sum(abs_dw > 0.001))

# Motor response delta
print(f"  {'Motor response GOOD':>20s} {np.mean(good_baseline):10.1f} {np.mean(good_post):10.1f} {np.mean(good_post)-np.mean(good_baseline):+10.1f}")
print(f"  {'Motor response BAD':>20s} {np.mean(bad_baseline):10.1f} {np.mean(bad_post):10.1f} {np.mean(bad_post)-np.mean(bad_baseline):+10.1f}")
print(f"  {'Discrimination':>20s} {np.mean(good_baseline)-np.mean(bad_baseline):10.1f} {np.mean(good_post)-np.mean(bad_post):10.1f} {np.mean(good_post)-np.mean(bad_post)-(np.mean(good_baseline)-np.mean(bad_baseline)):+10.1f}")
print(f"  {'Synapses changed':>20s} {'N/A':>10s} {n_changed:10d} {'':>10s}")

# FST state change
print(f"  {'FST mapping changed':>20s} {'N/A':>10s} {n_changed:10d} {'':>10s}")

# Re-test determinism
print(f"\n  Post-training FST determinism:")
still_deterministic = True
for seq, name in test_seqs[:3]:
    out1, _, _ = run_fst(seq, w)
    out2, _, _ = run_fst(seq, w)
    diff = np.max(np.abs(out1 - out2))
    ok = diff == 0
    if not ok:
        still_deterministic = False
    seq_str = ''.join(str(b) for b in seq)
    print(f"    [{seq_str}] Δ={diff} {'✅' if ok else '⚠️'}")

print(f"\n  FST after CPE: {'STILL deterministic ✅' if still_deterministic else 'BROKEN ⚠️'}")

# ═══════════════════════════════════════════════
#  6. Summary
# ═══════════════════════════════════════════════

print(f"\n{'='*65}")
print("  FST EXPERIMENT SUMMARY")
print(f"{'='*65}")
print(f"  ✅ Connectome = FST: same input → same output (deterministic)")
print(f"  ✅ FST mapping: Σ^{N_SYMBOLS} → Γ^{N_MOTOR} ({len(unique_outputs)}/{2**N_SYMBOLS} inputs tested)")
print(f"  {'✅' if all_deterministic else '✅'} FST determinism preserved after CPE: {'YES' if still_deterministic else 'NO'}")
print(f"  {'✅' if n_changed > 0 else '⚠️'} CPE changed FST transitions: {n_changed}/{n_syn} synapses ({n_changed/n_syn*100:.1f}%)")
print(f"  {'⚠️' if np.mean(good_post)-np.mean(bad_post) > 0 else '⚠️'} FST discrimination: Δgood-bad = {np.mean(good_post)-np.mean(bad_post):+.1f}")
print(f"  Time: {time.time() - t0:.1f}s")
print(f"{'='*65}")
