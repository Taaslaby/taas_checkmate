#!/usr/bin/env python3
"""CPE + SPIGRAD: dopamine-STDP на Kuramoto + T2-SPIGRAD на T3 drives.

Гибридная пластичность:
  - Kuramoto на connectome: dopamine-STDP учит синапсы
  - T3 (SPIGRAD): учит drives через BPTT
  - Оба на одном δ от contrastive learning
"""

import time
from pathlib import Path
import numpy as np

import taas_checkmate as tcm

DT = 0.01
N_STEPS = 100
N_EPISODES = 30
SPIKE_THRESH = 2.5

# CPE
LR = 0.02
ALPHA = 0.3
BETA = 0.3
H_TARGET = 3.0
ELIG_DECAY = 0.90
ELIG_ADD = 0.15
W_MIN = 0.01
W_MAX = 3.0

# SPIGRAD
T2_LR = 0.5
T2_N = 8
T2_GAIN = 10.0
T2_A = 1.0
T2_FMIN = 0.5
T2_FMAX = 2.0
T2_C12, T2_C13, T2_C23 = 0.1, 0.05, 0.02

BASE = Path(__file__).parent.parent.parent
BRAIN_PATH = BASE / "crates" / "connectome-gpt" / "models" / "brain.taas"


def main():
    print("=" * 60)
    print("  CPE + SPIGRAD — гибрид: STDP + BPTT на connectome")
    print("=" * 60)

    cd = tcm.load_connectome(str(BRAIN_PATH))
    n = cd.n_neurons
    n_syn = cd.n_synapses
    rp = np.array(cd.row_ptr(), dtype=np.int32)
    ci = np.array(cd.col_indices(), dtype=np.int32)
    w = np.array(cd.values(), dtype=np.float32)

    w_mean = np.mean(np.abs(w))
    if w_mean > 0:
        w /= w_mean * 2

    omegas = np.random.uniform(1.0, 4.0, n).astype(np.float32)
    spikes_out = np.zeros(n, dtype=np.int32)
    elig = np.zeros(n_syn, dtype=np.float32)

    # T3 drives (learnable)
    pat_drives = np.random.randn(N_STEPS, 3 * T2_N).astype(np.float32) * 0.1
    cha_drives = np.random.randn(N_STEPS, 3 * T2_N).astype(np.float32) * 0.1

    print(f"  {n} neurons, {n_syn} synapses")
    print("  {:>3}  {:>8}  {:>7}  {:>7}  {:>8}".format("Ep", "δ", "R_pat", "R_cha", "Δw"))
    print(f"  {'-'*42}")

    for ep in range(N_EPISODES):
        # ======== PATTERN ========
        ph = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
        ph_prev = ph.copy()

        pst_pat, traj_pat = tcm.t2spigrad_forward(
            T2_N, N_STEPS, DT, T2_GAIN, T2_A,
            T2_FMIN, T2_FMAX, T2_C12, T2_C13, T2_C23,
            pat_drives,
        )

        for t in range(N_STEPS):
            th = traj_pat.th1_final()
            drive = np.zeros(n, dtype=np.float32)
            if len(th) > 0:
                k = min(T2_N, n)
                drive[:k] = 8.0 * np.sin(th[:k])

            tcm.kuramoto_step(ph, omegas + drive, rp, ci, w, DT)
            tcm.detect_spikes(spikes_out, ph, ph_prev, SPIKE_THRESH)
            tcm.eligibility_update(elig, spikes_out, rp, ELIG_ADD, ELIG_DECAY)
            ph_prev = ph.copy()

        R_pat = tcm.phase_coherence(ph)
        spike_counts = np.bincount(spikes_out.astype(int), minlength=n).astype(np.float32)
        _, H_err_pat = tcm.spike_entropy(spike_counts, N_STEPS, H_TARGET)

        # ======== CHAOS ========
        ph = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
        ph_prev = ph.copy()

        pst_cha, traj_cha = tcm.t2spigrad_forward(
            T2_N, N_STEPS, DT, T2_GAIN, T2_A,
            T2_FMIN, T2_FMAX, T2_C12, T2_C13, T2_C23,
            cha_drives,
        )

        for t in range(N_STEPS):
            th = traj_cha.th1_final()
            drive = np.zeros(n, dtype=np.float32)
            if len(th) > 0:
                k = min(T2_N, n)
                drive[:k] = 4.0 * np.sin(th[:k]) + 2.0 * np.random.randn(k)

            tcm.kuramoto_step(ph, omegas + drive, rp, ci, w, DT)
            tcm.detect_spikes(spikes_out, ph, ph_prev, SPIKE_THRESH)
            tcm.eligibility_update(elig, spikes_out, rp, ELIG_ADD, ELIG_DECAY)
            ph_prev = ph.copy()

        R_cha = tcm.phase_coherence(ph)
        _, H_err_cha = tcm.spike_entropy(spike_counts, N_STEPS, H_TARGET)

        # ======== Dopamine δ + STDP + SPIGRAD ========
        reward = 1.0
        delta = tcm.dopamine_delta(reward, R_pat, H_err_pat, R_cha, H_err_cha, ALPHA, BETA)
        gate = tcm.dopamine_timing_gate(N_STEPS, N_STEPS)

        w_before = w.copy()
        tcm.stdp_update(w, delta, elig, gate, LR, W_MIN, W_MAX)
        dw = np.mean(np.abs(w - w_before))

        # SPIGRAD: подстраиваем T3 drives под δ
        pst_pat_np = np.array(pst_pat, dtype=np.float32)
        pst_cha_np = np.array(pst_cha, dtype=np.float32)

        grad_pat = np.zeros_like(pst_pat_np)
        grad_cha = np.zeros_like(pst_cha_np)
        grad_pat[0] = BETA * 0.1 if delta < 0 else 0.0
        grad_cha[0] = -BETA * 0.1 if delta < 0 else 0.0

        dpat = np.array(tcm.t2spigrad_backward(grad_pat, traj_pat), dtype=np.float32)
        dcha = np.array(tcm.t2spigrad_backward(grad_cha, traj_cha), dtype=np.float32)

        pat_drives -= T2_LR * dpat
        cha_drives += T2_LR * dcha
        pat_drives = np.clip(pat_drives, -3.0, 3.0)
        cha_drives = np.clip(cha_drives, -3.0, 3.0)

        if ep % 5 == 0:
            print(f"  {ep:>3}  {delta:>+8.4f}  {R_pat:>7.4f}  {R_cha:>7.4f}  {dw:>+8.6f}")

    print()
    print("3. Итог")
    print(f"  δ финальный:   {delta:+.4f}")
    print(f"  R_pattern:     {R_pat:.4f}")
    print(f"  R_chaos:       {R_cha:.4f}")
    print(f"  |Δw| (STDP):   {dw:.6f}")
    print(f"  |Drives| пат:  {np.std(pat_drives):.4f}")
    print(f"  |Drives| хаос: {np.std(cha_drives):.4f}")
    print()
    print("  CPE: dopamine-STDP учит синапсы connectome")
    print("  SPIGRAD: BPTT учит T3 drives")
    print("  Два пути пластичности, один δ")


if __name__ == "__main__":
    main()
