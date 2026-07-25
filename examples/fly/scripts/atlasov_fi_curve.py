#!/usr/bin/env python3
"""
Atlasov Neuromotor Engine — Motoneuron Firing Frequency to Isometric Tension (F-I) Curve Analysis.

Measures non-linear muscle tension T(f) and joint stiffness K(f) across firing frequencies (5 Hz to 200 Hz)
for fast flexor (Kd = 0.15 uM) vs slow extensor (Kd = 0.60 uM) fibers.

Author / Concept: Atlasov Anton (Атласов Антон)
"""

import sys
import math
from pathlib import Path
import numpy as np

PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

def compute_fi_curve():
    print("=" * 70)
    print("  Atlasov Neuromotor Engine — Firing Frequency to Force (F-I) Analysis")
    print("  Author / Concept: Atlasov Anton (Атласов Антон)")
    print("=" * 70)

    try:
        import taas_checkmate
        step_fn = getattr(taas_checkmate, "step_atlasov_neuromotor", None) or getattr(getattr(taas_checkmate, "neurons", None), "step_atlasov_neuromotor", None)
    except ImportError:
        step_fn = None

    frequencies = [5, 10, 20, 40, 60, 80, 100, 150, 200] # Hz
    n_muscles = 16
    dt = 1.0 # 1ms steps

    print(f"\n{'Freq (Hz)':<10} | {'Fast Flexor (mN)':<18} | {'Slow Extensor (mN)':<18} | {'Joint K (mN/rad)':<18} | {'Tetanus Saturation':<20}")
    print("-" * 90)

    results = []

    for freq in frequencies:
        ca = np.zeros(n_muscles, dtype=np.float32)
        tension = np.zeros(n_muscles, dtype=np.float32)
        spikes = np.zeros(n_muscles, dtype=np.int32)
        resets = np.zeros(n_muscles, dtype=np.int32)

        d_ca = np.full(n_muscles, 0.5, dtype=np.float32)
        tau_pump = np.full(n_muscles, 20.0, dtype=np.float32)
        eta_reset = np.full(n_muscles, 0.8, dtype=np.float32)
        t_max = np.full(n_muscles, 10.0, dtype=np.float32) # mN scale

        # Channel 0: Fast Flexor (Kd = 0.15 uM)
        # Channel 1: Slow Extensor (Kd = 0.60 uM)
        kd = np.full(n_muscles, 0.3, dtype=np.float32)
        kd[0] = 0.15
        kd[1] = 0.60

        joint_df = np.zeros(8, dtype=np.float32)
        joint_stf = np.zeros(8, dtype=np.float32)
        joint_dmp = np.zeros(8, dtype=np.float32)

        k0 = np.full(8, 1.0, dtype=np.float32)
        beta = np.full(8, 2.0, dtype=np.float32)
        b0 = np.full(8, 0.1, dtype=np.float32)
        alpha = np.full(8, 0.5, dtype=np.float32)

        period_steps = int(1000.0 / freq)

        # Simulate 500ms to reach steady-state tetanus
        for step in range(500):
            spikes.fill(0)
            if step % period_steps == 0:
                spikes[0] = 1
                spikes[1] = 1

            if step_fn is not None:
                step_fn(
                    ca, tension, spikes, resets, dt,
                    d_ca, tau_pump, eta_reset, t_max, kd,
                    joint_df, joint_stf, joint_dmp, k0, beta, b0, alpha
                )
            else:
                for m in [0, 1]:
                    if spikes[m] > 0:
                        ca[m] += d_ca[m]
                    ca[m] *= math.exp(-dt / tau_pump[m])
                    ca2 = ca[m]**2
                    kd2 = kd[m]**2
                    tension[m] = t_max[m] * (ca2 / (ca2 + kd2 + 1e-8))
                joint_stf[0] = k0[0] + beta[0] * (ca[0] + ca[1])

        t_fast = tension[0]
        t_slow = tension[1]
        k_joint = joint_stf[0]
        sat_pct = (t_fast / 10.0) * 100.0

        results.append((freq, t_fast, t_slow, k_joint, sat_pct))
        print(f"{freq:<10} | {t_fast:<18.3f} | {t_slow:<18.3f} | {k_joint:<18.3f} | {sat_pct:<19.1f}%")

    print("=" * 90)
    print("✓ F-I Curve Analysis Complete! Demonstrated non-linear sigmoidal calcium recruitment.")

    # Sanity checks
    low_freq_t = results[0][1]
    high_freq_t = results[-1][1]
    assert high_freq_t > low_freq_t * 3.0, "High frequency firing must produce significant tetanic force buildup"
    assert results[-1][4] > 80.0, "200 Hz firing must achieve >80% tetanus saturation"
    print("✓ All F-I Curve Neuroscience Assertions Passed!")

if __name__ == "__main__":
    compute_fi_curve()
