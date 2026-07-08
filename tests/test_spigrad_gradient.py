#!/usr/bin/env python3
"""SPIGRAD bridge: gradient check — multi-seed + per-population + cross_gate.

    cd taas_checkmate && python3 tests/test_spigrad_gradient.py
"""

import numpy as np
import taas_checkmate as tcm

N_OSC = 4
N_STEPS = 20
SPIKE_DIM = 8
EPS = 1e-4
TOL = 0.30
MIN_GRAD = 1e-3

# ─── 1. Multi-seed directional check ──────────────────
print("═══ Multi-seed directional check ═══")
all_ok = True
for seed in [1, 42, 123, 999, 2024]:
    rng_seed = np.random.default_rng(seed)
    bridge = tcm.SpigradBridge(
        spike_dim=SPIKE_DIM, n_osc=N_OSC, dt=0.05,
        drive_gain=10.0, alpha=1.0, freq_min=0.5, freq_max=3.0,
        coupling_w12=0.3, coupling_w13=0.15, coupling_w23=0.08,
        zero_init=False,
    )
    spikes = rng_seed.uniform(0, 1, (N_STEPS, SPIKE_DIM)).astype(np.float32)
    pst_mat, traj = bridge.forward(spikes, N_STEPS)
    pst_dim = pst_mat.shape[1]

    grad_pst = np.zeros(pst_dim, dtype=np.float32)
    grad_pst[0:2*N_OSC] = 1.0  # только sin+cos, как в per-pop тесте
    grad_w, grad_b = bridge.backward(grad_pst, traj, spikes)
    loss_fn = lambda pp: float(np.sum(pp[0:2*N_OSC]))

    n_ok = 0
    for trial in range(10):
        dw = rng_seed.normal(0, 1, (SPIKE_DIM, bridge.drive_dim())).astype(np.float32)
        db = rng_seed.normal(0, 1, bridge.drive_dim()).astype(np.float32)
        ana = float(np.sum(grad_w * dw) + np.sum(grad_b * db))

        bridge.update(-(dw * EPS), -(db * EPS), 1.0)
        pp, _ = bridge.forward(spikes, N_STEPS)
        bridge.update(dw * (2 * EPS), db * (2 * EPS), 1.0)
        pm, _ = bridge.forward(spikes, N_STEPS)
        fd = (loss_fn(pp[-1]) - loss_fn(pm[-1])) / (2.0 * EPS)

        if abs(ana) > MIN_GRAD and abs(fd) > MIN_GRAD:
            ratio = ana / fd
            if abs(ratio - 1.0) <= TOL:
                n_ok += 1
    status = "✓" if n_ok >= 7 else "⚠"
    if n_ok < 7:
        all_ok = False
    print(f"  {status} seed={seed}: {n_ok}/10 directional derivatives match FD")

# ─── 2. Per-population losses ─────────────────────────
print(f"\n═══ Per-population loss check ═══")
rng = np.random.default_rng(42)
for pop_idx, pop_name in enumerate(["θ", "r", "vel"]):
    bridge = tcm.SpigradBridge(
        spike_dim=SPIKE_DIM, n_osc=N_OSC, dt=0.05,
        drive_gain=10.0, alpha=1.0, freq_min=0.5, freq_max=3.0,
        coupling_w12=0.3, coupling_w13=0.15, coupling_w23=0.08,
        zero_init=False,
    )
    spikes = rng.uniform(0, 1, (N_STEPS, SPIKE_DIM)).astype(np.float32)
    pst_mat, traj = bridge.forward(spikes, N_STEPS)
    pst_dim = pst_mat.shape[1]

    #     0-5:     sin(θ), cos(θ)      (6n)
    #     6n-8n:   vel                  (3n)
    #     8n-11n:  r                    (3n)
    n = N_OSC
    grad_pst = np.zeros(pst_dim, dtype=np.float32)
    if pop_name == "θ":
        grad_pst[0:2*n] = 1.0  # sin+cos
        loss_fn = lambda pp: float(np.sum(pp[0:2*n]))
    elif pop_name == "r":
        grad_pst[8*n:11*n] = 1.0
        loss_fn = lambda pp: float(np.sum(pp[8*n:11*n]))
    else:  # vel
        grad_pst[6*n:8*n] = 1.0
        loss_fn = lambda pp: float(np.sum(pp[6*n:8*n]))

    grad_w, grad_b = bridge.backward(grad_pst, traj, spikes)
    n_ok = 0
    for trial in range(10):
        dw = rng.normal(0, 1, (SPIKE_DIM, bridge.drive_dim())).astype(np.float32)
        db = rng.normal(0, 1, bridge.drive_dim()).astype(np.float32)
        ana = float(np.sum(grad_w * dw) + np.sum(grad_b * db))

        bridge.update(-(dw * EPS), -(db * EPS), 1.0)
        pp, _ = bridge.forward(spikes, N_STEPS)
        bridge.update(dw * (2 * EPS), db * (2 * EPS), 1.0)
        pm, _ = bridge.forward(spikes, N_STEPS)
        fd = (loss_fn(pp[-1]) - loss_fn(pm[-1])) / (2.0 * EPS)

        if abs(ana) > MIN_GRAD and abs(fd) > MIN_GRAD:
            if abs(ana / fd - 1.0) <= TOL:
                n_ok += 1
    status = "✓" if n_ok >= 7 else "⚠"
    if n_ok < 7:
        all_ok = False
    print(f"  {status} pop={pop_name}: {n_ok}/10 match FD")

# ─── 3. Cross-population gradient flow ─────────────────
print(f"\n═══ Cross-population gradient flow ═══")
rng = np.random.default_rng(42)
bridge = tcm.SpigradBridge(
    spike_dim=SPIKE_DIM, n_osc=N_OSC, dt=0.05,
    drive_gain=10.0, alpha=1.0, freq_min=0.5, freq_max=3.0,
    coupling_w12=0.3, coupling_w13=0.15, coupling_w23=0.08,
    zero_init=False,
)
spikes = rng.uniform(0, 1, (N_STEPS, SPIKE_DIM)).astype(np.float32)
pst_mat, traj = bridge.forward(spikes, N_STEPS)

# Loss on sin(θ3) (3rd population) → gradient must flow
# back through cross_gate to drives of population 1
# sin(θ) for pop 3 starts at index 4*N_OSC (4 populations: 0,1,2,3)
# Actually layout: sin(θ) for each osc: 0..N_OSC-1
# sin(θ3) = index 2 (0-indexed: θ0, θ1, θ2, θ3)
grad_pst = np.zeros(pst_mat.shape[1], dtype=np.float32)
grad_pst[3] = 1.0  # dL/d(sin(θ3))
cross_loss = lambda pp: float(pp[3])
loss0 = cross_loss(pst_mat[-1])

grad_w, grad_b = bridge.backward(grad_pst, traj, spikes)
n_ok = 0
for trial in range(10):
    dw = rng.normal(0, 1, (SPIKE_DIM, bridge.drive_dim())).astype(np.float32)
    db = rng.normal(0, 1, bridge.drive_dim()).astype(np.float32)
    ana = float(np.sum(grad_w * dw) + np.sum(grad_b * db))

    bridge.update(-(dw * EPS), -(db * EPS), 1.0)
    pp, _ = bridge.forward(spikes, N_STEPS)
    bridge.update(dw * (2 * EPS), db * (2 * EPS), 1.0)
    pm, _ = bridge.forward(spikes, N_STEPS)
    fd = (cross_loss(pp[-1]) - cross_loss(pm[-1])) / (2.0 * EPS)

    if abs(ana) > MIN_GRAD and abs(fd) > MIN_GRAD:
        if abs(ana / fd - 1.0) <= TOL:
            n_ok += 1
status = "✓" if n_ok >= 7 else "⚠"
if n_ok < 7:
    all_ok = False
print(f"  {status} cross-pop (sinθ3 ← spikes): {n_ok}/10 match FD")
print(f"      loss on sinθ3: {loss0:.6f}")

# ─── Result ────────────────────────────────────────────
print(f"\n═══ Result ═══")
if all_ok:
    print("  ✅ ALL PASSED")
else:
    print("  ❌ SOME FAILED")
    exit(1)
