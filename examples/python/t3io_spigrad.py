#! /usr/bin/env python3
"""T3 IO + SPIGRAD: учим T3 drives через градиенты на PST.

Градиентный путь:
  drives → T3 (SPIGRAD forward) → PST → linear → motor → loss → SPIGRAD backward → d_drives

T3 учится генерировать правильные фазы для задачи.
HH/STDP не нужен на этом уровне — T3 сам подстраивает output.
"""

import numpy as np
import taas_checkmate


def main():
    print("═══ T3 IO + SPIGRAD ═══")
    print()

    N_OSC = 8
    N_STEPS = 50
    DT = 0.1
    N_MOTOR = 2

    rng = np.random.default_rng(42)

    # Начальные drives — случайные
    drives = rng.normal(0, 0.05, (N_STEPS, 3 * N_OSC)).astype(np.float32)

    # Цель: motor = [1, 0]
    target = np.array([1.0, 0.0], dtype=np.float32)

    # Decode: PST (pst_dim) → motor
    # PST = 3*n*6 + 12 (sin/cos + cross + coherence)
    pst_dim = 18 * N_OSC + 12
    decode_w = rng.normal(0, 0.1, (pst_dim, N_MOTOR)).astype(np.float32)
    decode_b = np.zeros(N_MOTOR, dtype=np.float32)

    print(f"  N_OSC:   {N_OSC}")
    print(f"  STEPS:   {N_STEPS}")
    print(f"  PST dim: {pst_dim}")
    print(f"  Target:  [{target[0]:.1f}, {target[1]:.1f}]")
    print()

    lr = 0.5
    best_loss = float("inf")

    for epoch in range(200):
        # Forward: drives → T3 → PST
        pst, traj = taas_checkmate.t2spigrad_forward(
            N_OSC, N_STEPS, DT,
            drive_gain=10.0,
            alpha=1.0,
            freq_min=0.5,
            freq_max=2.0,
            coupling_w12=0.1,
            coupling_w13=0.1,
            coupling_w23=0.1,
            drives=drives,
        )

        # PST → motor (linear decode)
        pst_np = np.array(pst, dtype=np.float32)
        motor = pst_np @ decode_w + decode_b
        err = motor - target
        loss = 0.5 * np.sum(err ** 2)

        if loss < best_loss:
            best_loss = loss

        # Backward: motor error → PST → drives
        grad_motor = err  # dL/dmotor
        grad_pst = grad_motor @ decode_w.T  # dL/dPST

        grad_drives = taas_checkmate.t2spigrad_backward(grad_pst, traj)

        # Update drives
        drives -= lr * np.array(grad_drives, dtype=np.float32)

        if epoch % 20 == 0:
            print(f"  epoch {epoch:>3}  loss={loss:.6f}  motor=[{motor[0]:+.4f}, {motor[1]:+.4f}]")

    print()
    print("═══ Result ═══")
    print(f"  Target:  [{target[0]:.1f}, {target[1]:.1f}]")
    print(f"  Motor:   [{motor[0]:+.4f}, {motor[1]:+.4f}]")
    print(f"  Loss:    {loss:.6f}")
    print(f"  Best:    {best_loss:.6f}")
    print()
    print(f"  PST фичи напрямую управляют motor через learnable decode")
    print(f"  SPIGRAD BPTT учит drives T3 минимизировать motor error")
    print(f"  T3 IO = градиентный путь drives → T3 → PST → motor")


if __name__ == "__main__":
    main()
