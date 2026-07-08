#! /usr/bin/env python3
"""Hologram Bridge — full pipeline demo from Python.

Bridge: position(x) → phase → sin(θ) → HH → STDP → motor

Usage:
    PYTHONPATH=target/debug python3 examples/python/hologram_bridge.py
"""

import numpy as np
import taas_checkmate  # type: ignore


def main():
    print("═══ Hologram Bridge — Python Demo ═══")
    print()

    # ── 1. Create bridge ──
    n = 64
    dt = 0.1
    bridge = taas_checkmate.HologramBridge(n, dt, omega=800.0, n_motor=2)

    # ── 2. Set decode weights ──
    # Random decoder: n_neurons × 2 motor dimensions
    rng = np.random.default_rng(42)
    decode_w = rng.normal(0, 0.1, n * 2).tolist()
    bridge.set_decode_weights(decode_w)

    print(f"  n_neurons: {bridge.n}")
    print(f"  dt:        {bridge.dt} ms")
    print(f"  motor:     {len(bridge.decode())} dims")
    print()

    # ── 3. Game loop ──
    player_pos = 0.5  # normalized [0, 1]
    npc_pos = 0.3

    # Phase offsets: each neuron gets its own phase = position * 2π + noise
    phases = [bridge.phase_encode(player_pos) + 0.1 * np.sin(i)
              for i in range(n // 2)] + \
             [bridge.phase_encode(npc_pos) + 0.1 * np.cos(i)
              for i in range(n // 2)]

    weights_before = bridge.get_weights()
    w_initial = weights_before[0][1]

    for step in range(2000):
        # Step: inject phase-encoded current
        spikes = bridge.step(phases)

        if spikes > 0:
            bridge.hebb()  # STDP for all spiking pairs

        # Every 500 steps, check motor output
        if step % 500 == 0:
            motor = bridge.decode()
            w01 = bridge.get_weight(0, 1)
            print(f"  step {step:>4}  spikes={spikes:>3}  motor=[{motor[0]:+.3f}, {motor[1]:+.3f}]  w[0→1]={w01:.4f}")

    # ── 4. Results ──
    weights_after = bridge.get_weights()
    w_final = weights_after[0][1]

    print()
    print("═══ Result ═══")
    print(f"  Δw[0→1]: {w_initial:.4f} → {w_final:.4f} ({w_final - w_initial:+.4f})")
    print(f"  Motor:   {bridge.decode()}")
    print(f"  Time:    {bridge.time:.1f} ms simulated")
    print()

    # ── 5. Phase relationship scan ──
    print("── Phase scan: varying NPC position ──")
    for npc_pos_test in [0.0, 0.25, 0.5, 0.75, 1.0]:
        bridge.reset()
        test_phases = [bridge.phase_encode(player_pos) + 0.1 * np.sin(i)
                       for i in range(n // 2)] + \
                      [bridge.phase_encode(npc_pos_test) + 0.1 * np.cos(i)
                       for i in range(n // 2)]

        for _ in range(1000):
            s = bridge.step(test_phases)
            if s > 0:
                bridge.hebb()

        w = bridge.get_weight(0, 1)
        motor = bridge.decode()
        print(f"    NPC={npc_pos_test:.2f}  Δθ={abs(bridge.phase_encode(npc_pos_test) - bridge.phase_encode(player_pos)):.4f}  w[0→1]={w:.4f}  motor=[{motor[0]:+.3f},{motor[1]:+.3f}]")


if __name__ == "__main__":
    main()
