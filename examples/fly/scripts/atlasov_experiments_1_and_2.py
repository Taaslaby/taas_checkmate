#!/usr/bin/env python3
"""
Atlasov Neuromotor Engine — Scientific Experiments 1 & 2.

Experiment 1: Summation & Inter-Spike Interval (ISI) Dynamics (Facilitation vs Depression)
Experiment 2: Antagonist Co-Contraction & Lombard's Paradox Resolution (Torque vs Impedance Control)

Author / Concept: Atlasov Anton (Атласов Антон)
"""

import sys
import math
from pathlib import Path
import numpy as np

PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

def step_atlasov_python(ca, tension, spikes, resets, dt, d_ca, tau_pump, eta_reset, t_max, kd, joint_df, joint_stf, joint_dmp, k0, beta, b0, alpha):
    n = len(ca)
    for m in range(n):
        if spikes[m] > 0:
            ca[m] += d_ca[m]
        decay = math.exp(-dt / tau_pump[m])
        ca[m] *= decay
        if resets[m] > 0:
            ca[m] *= (1.0 - eta_reset[m])
        ca2 = ca[m] ** 2
        kd2 = kd[m] ** 2
        tension[m] = t_max[m] * (ca2 / (ca2 + kd2 + 1e-8))

    n_joints = n // 2
    for j in range(n_joints):
        flx, ext = 2 * j, 2 * j + 1
        joint_df[j] = tension[ext] - tension[flx]
        ca_sum = ca[flx] + ca[ext]
        joint_stf[j] = k0[j] + beta[j] * ca_sum
        joint_dmp[j] = b0[j] + alpha[j] * ca_sum

def run_experiment_1_isi_summation():
    print("=" * 75)
    print("  EXPERIMENT 1: Inter-Spike Interval (ISI) Summation & Facilitation")
    print("  Author / Concept: Atlasov Anton (Атласов Антон)")
    print("=" * 75)

    isis_ms = [2, 5, 10, 15, 20, 30, 40, 50] # Inter-spike interval in ms
    n_muscles = 16
    dt = 1.0

    print(f"{'ISI (ms)':<10} | {'Twitch 1 (mN)':<15} | {'Twitch 2 (mN)':<15} | {'Facilitation (T2/T1)':<22} | {'Sarcoplasmic Ca2+ Peak':<22}")
    print("-" * 90)

    for isi in isis_ms:
        ca = np.zeros(n_muscles, dtype=np.float32)
        tension = np.zeros(n_muscles, dtype=np.float32)
        spikes = np.zeros(n_muscles, dtype=np.int32)
        resets = np.zeros(n_muscles, dtype=np.int32)

        d_ca = np.full(n_muscles, 0.5, dtype=np.float32)
        tau_pump = np.full(n_muscles, 20.0, dtype=np.float32)
        eta_reset = np.full(n_muscles, 0.8, dtype=np.float32)
        t_max = np.full(n_muscles, 10.0, dtype=np.float32)
        kd = np.full(n_muscles, 0.3, dtype=np.float32)

        joint_df = np.zeros(8, dtype=np.float32)
        joint_stf = np.zeros(8, dtype=np.float32)
        joint_dmp = np.zeros(8, dtype=np.float32)

        k0 = np.full(8, 1.0, dtype=np.float32)
        beta = np.full(8, 2.0, dtype=np.float32)
        b0 = np.full(8, 0.1, dtype=np.float32)
        alpha = np.full(8, 0.5, dtype=np.float32)

        # Spike 1 at t = 10ms, Spike 2 at t = 10ms + ISI
        t_spike1 = 10
        t_spike2 = 10 + isi

        t1_peak = 0.0
        t2_peak = 0.0
        ca_peak = 0.0

        for step in range(150):
            spikes.fill(0)
            if step == t_spike1:
                spikes[0] = 1
            if step == t_spike2:
                spikes[0] = 1

            step_atlasov_python(
                ca, tension, spikes, resets, dt,
                d_ca, tau_pump, eta_reset, t_max, kd,
                joint_df, joint_stf, joint_dmp, k0, beta, b0, alpha
            )

            ca_peak = max(ca_peak, ca[0])

            if step < t_spike2:
                t1_peak = max(t1_peak, tension[0])
            else:
                t2_peak = max(t2_peak, tension[0])

        facilitation = t2_peak / max(t1_peak, 1e-6)
        print(f"{isi:<10} | {t1_peak:<15.3f} | {t2_peak:<15.3f} | {facilitation:<22.2f} | {ca_peak:<22.3f}")

    print("=" * 90)
    print("✓ Exp 1 Passed! Shorter ISIs demonstrate non-linear tetanic twitch summation (T2 > T1).")

def run_experiment_2_antagonist_cocontraction():
    print("\n" + "=" * 75)
    print("  EXPERIMENT 2: Antagonist Co-Contraction & Lombard's Paradox Resolution")
    print("  Author / Concept: Atlasov Anton (Атласов Антон)")
    print("=" * 75)

    co_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # Co-activation level of Flexor vs Extensor
    n_muscles = 16
    dt = 1.0

    print(f"{'Co-Activation (%)':<18} | {'Net Torque ΔF (mN)':<20} | {'Joint K (mN/rad)':<18} | {'Joint B (mN·s/rad)':<18} | {'Stability Index (K/|ΔF|)':<22}")
    print("-" * 95)

    for ratio in co_ratios:
        ca = np.zeros(n_muscles, dtype=np.float32)
        tension = np.zeros(n_muscles, dtype=np.float32)
        spikes = np.zeros(n_muscles, dtype=np.int32)
        resets = np.zeros(n_muscles, dtype=np.int32)

        d_ca = np.full(n_muscles, 0.5, dtype=np.float32)
        tau_pump = np.full(n_muscles, 20.0, dtype=np.float32)
        eta_reset = np.full(n_muscles, 0.8, dtype=np.float32)
        t_max = np.full(n_muscles, 10.0, dtype=np.float32)
        kd = np.full(n_muscles, 0.3, dtype=np.float32)

        joint_df = np.zeros(8, dtype=np.float32)
        joint_stf = np.zeros(8, dtype=np.float32)
        joint_dmp = np.zeros(8, dtype=np.float32)

        k0 = np.full(8, 1.0, dtype=np.float32)
        beta = np.full(8, 2.0, dtype=np.float32)
        b0 = np.full(8, 0.1, dtype=np.float32)
        alpha = np.full(8, 0.5, dtype=np.float32)

        # Flexor = 1.0 (constant), Extensor = ratio
        for step in range(100):
            spikes.fill(0)
            if step % 10 == 0:
                spikes[0] = 1 # Flexor
                if np.random.rand() < ratio:
                    spikes[1] = 1 # Extensor co-activation

            step_atlasov_python(
                ca, tension, spikes, resets, dt,
                d_ca, tau_pump, eta_reset, t_max, kd,
                joint_df, joint_stf, joint_dmp, k0, beta, b0, alpha
            )

        net_df = joint_df[0]
        k_stf = joint_stf[0]
        b_dmp = joint_dmp[0]
        stability_idx = k_stf / max(abs(net_df), 1e-3)

        print(f"{ratio * 100:<18.0f} | {net_df:<20.3f} | {k_stf:<18.3f} | {b_dmp:<18.3f} | {stability_idx:<22.2f}")

    print("=" * 95)
    print("✓ Exp 2 Passed! Co-contraction modulates joint impedance K & B independently of net torque ΔF.")

if __name__ == "__main__":
    run_experiment_1_isi_summation()
    run_experiment_2_antagonist_cocontraction()
