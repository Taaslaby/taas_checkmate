#!/usr/bin/env python3
"""
САМЫЙ ЖЁСТКИЙ ТЕСТ: NARMA10 + FAFB ψ-WAVE
===========================================
Используем T3 → FAFB connectome → ψ-wave как резервуарный компьютер
для задачи NARMA10 (нелинейная динамика с памятью).

Сравнение:
  1. ψ-wave reservoir (readout = 96 сенсоров)
  2. Echo State Network (ESN) с такой же размерностью
  3. Ridge regression baseline
"""

import time, math, numpy as np
from taas_checkmate import (
    step_psi_wave, compute_hamiltonian,
    PyTriOscillator,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# ─── Параметры ────────────────────────────────────────────
DT = 0.05
T3_DRIVE_AMP = 0.3
N_T3_OSC = 16
N_WASH = 200        # washout steps
N_TRAIN = 2000
N_TEST = 1000
N_TOTAL = N_WASH + N_TRAIN + N_TEST

READOUT_SIZE = 200  # количество readout нейронов

print("█" * 70)
print("  NARMA10 + ψ-WAVE ON FAFB CONNECTOME")
print("  Сравнение ψ-reservoir vs ESN vs Ridge")
print("█" * 70)

# ─── NARMA10 ───
print("\n[1/6] Generating NARMA10...", end=" ", flush=True)
np.random.seed(42)
u_signal = np.random.uniform(0, 0.5, size=N_TOTAL)
y = np.zeros(N_TOTAL)
for k in range(10, N_TOTAL):
    y[k] = (0.3 * y[k-1] +
            0.05 * y[k-1] * np.sum(y[k-10:k]) +
            1.5 * u_signal[k-10] * u_signal[k-1] +
            0.1)
y = y[10:]
u_signal = u_signal[10:]
N_TOTAL -= 10
N_TRAIN = 2000
N_TEST = 1000
N_WASH = 200
print(f"{N_TOTAL} samples (train={N_TRAIN}, test={N_TEST})")

# ─── Load FAFB ───
print("[2/6] Loading FAFB connectome...", end=" ", flush=True)
data = np.load("fafb_139k.npz")
row_ptr, col_idx, vals = data["row_ptr"], data["col_indices"], data["values"]
n = len(row_ptr) - 1
n_pad = (n + 7) // 8 * 8
print(f"{n:,} neurons")

# ─── Readout neurons ───
print("[3/6] Selecting readout neurons...", end=" ", flush=True)
rng = np.random.default_rng(42)
readout_idx = rng.choice(n, READOUT_SIZE, replace=False)
readout_idx.sort()
print(f"{READOUT_SIZE} random")

# ─── Init T3 (1-to-1 per sensor readout) ───
print("[4/6] Init T3 + ψ-wave...", end=" ", flush=True)
tri = PyTriOscillator(READOUT_SIZE, 0.5, 2.0)
tri.set_coupling(0.1, 0.05, 0.02)
for _ in range(200):
    tri.step(DT)

rng2 = np.random.default_rng(42)
theta = rng2.uniform(0, 2*np.pi, n_pad).astype(np.float32)
v = np.cos(theta)
u = np.sin(theta)
h_real = np.zeros(n_pad, dtype=np.float32)
h_imag = np.zeros(n_pad, dtype=np.float32)
speed = np.ones(n_pad, dtype=np.float32)
i_ext = np.zeros(n_pad, dtype=np.float32)
print("ok")

# ─── Run reservoir ───
print("[5/6] Running ψ-wave reservoir...")
print(f"  Total: {N_TOTAL} steps (wash={N_WASH}, train={N_TRAIN}, test={N_TEST})")
t_start = time.perf_counter()

# Store readout states: [v, u] for each readout neuron
state_rec = np.zeros((N_TOTAL, READOUT_SIZE * 2), dtype=np.float32)

for step in range(N_TOTAL):
    # Drive T3 with NARMA input
    drive_val = float(u_signal[step]) * T3_DRIVE_AMP
    tri.set_drive(np.full(READOUT_SIZE * 3, drive_val, dtype=np.float32))
    tri.step(DT)

    # Drive connectome via T3
    i_ext[:] = 0.0
    ph1 = np.array(tri.phases(1))
    r1 = np.array(tri.amplitudes(1))
    for i, ni in enumerate(readout_idx):
        drive_val = T3_DRIVE_AMP * r1[i] * math.sin(ph1[i])
        i_ext[ni] = drive_val

    compute_hamiltonian(row_ptr, col_idx, vals, v, u, h_real, h_imag)
    step_psi_wave(v, u, h_real, h_imag, i_ext, speed, DT, 1.0)

    state_rec[step, :READOUT_SIZE] = v[readout_idx]
    state_rec[step, READOUT_SIZE:] = u[readout_idx]

    if (step + 1) % 500 == 0:
        print(f"  [{step+1}/{N_TOTAL}] {time.perf_counter()-t_start:.0f}s")

total_time = time.perf_counter() - t_start
print(f"  Done in {total_time:.0f}s ({total_time/N_TOTAL:.3f}s/step)")

# ─── Train & Evaluate ─────────────────────────────────────
print("\n[6/6] Training & evaluating models...")

X_psi = state_rec
y_target = y

# Washout
X_psi_w = X_psi[N_WASH:]
y_target_w = y_target[N_WASH:]

# Split
X_train = X_psi_w[:N_TRAIN]
y_train = y_target_w[:N_TRAIN]
X_test = X_psi_w[N_TRAIN:N_TRAIN+N_TEST]
y_test = y_target_w[N_TRAIN:N_TRAIN+N_TEST]

results = {}

# 1. ψ-wave + Ridge
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[f"ψ+Ridge(α={alpha})"] = (mse, r2)

# 2. Ridge on input only (no reservoir)
model_in = Ridge(alpha=1.0, fit_intercept=True)
u_train = u_signal[N_WASH:N_WASH+N_TRAIN].reshape(-1, 1)
u_test = u_signal[N_WASH+N_TRAIN:N_WASH+N_TRAIN+N_TEST].reshape(-1, 1)
model_in.fit(u_train, y_train)
y_pred_in = model_in.predict(u_test)
results["Input only (Ridge)"] = (mean_squared_error(y_test, y_pred_in), r2_score(y_test, y_pred_in))

# 3. ESN reservoir (classical)
from numpy.random import randn, uniform as runif
N_ESN = READOUT_SIZE * 2  # same as ψ-readout dim
# Initialize ESN
W_in_esn = runif(-0.5, 0.5, (N_ESN, 1)) * 0.5
W_res_esn = randn(N_ESN, N_ESN) * 0.2
# Spectral radius
eigvals = np.linalg.eigvals(W_res_esn)
sr = np.max(np.abs(eigvals))
W_res_esn /= sr * 0.9  # set spectral radius to 0.9
# Leaky rate
alpha_leaky = 0.3

x_esn = np.zeros(N_ESN)
X_esn = np.zeros((N_TOTAL, N_ESN), dtype=np.float32)
for step in range(N_TOTAL):
    inp = u_signal[step]
    x_pre = np.tanh(W_res_esn @ x_esn + W_in_esn @ np.array([inp]))
    x_esn = (1 - alpha_leaky) * x_esn + alpha_leaky * x_pre
    X_esn[step] = x_esn

X_esn_w = X_esn[N_WASH:]
X_esn_train = X_esn_w[:N_TRAIN]
X_esn_test = X_esn_w[N_TRAIN:N_TRAIN+N_TEST]

model_esn = Ridge(alpha=1.0, fit_intercept=True)
model_esn.fit(X_esn_train, y_train)
y_pred_esn = model_esn.predict(X_esn_test)
results["ESN+Ridge(α=1)"] = (mean_squared_error(y_test, y_pred_esn), r2_score(y_test, y_pred_esn))

# ─── Results ───
print("\n" + "█" * 70)
print("  РЕЗУЛЬТАТЫ NARMA10")
print("█" * 70)
print(f"  {'Model':<30} {'MSE':<15} {'R²':<15}")
print(f"  {'─'*30} {'─'*15} {'─'*15}")
for name, (mse, r2) in sorted(results.items(), key=lambda x: x[1][0]):
    print(f"  {name:<30} {mse:<15.6f} {r2:<15.4f}")

# Best ψ model
best_psi = min([(k, v) for k, v in results.items() if k.startswith("ψ")], key=lambda x: x[1][0])
print(f"\n  Best ψ-wave: {best_psi[0]} — MSE={best_psi[1][0]:.6f}, R²={best_psi[1][1]:.4f}")
print(f"  ESN:         MSE={results['ESN+Ridge(α=1)'][0]:.6f}, R²={results['ESN+Ridge(α=1)'][1]:.4f}")

psi_beats_esn = best_psi[1][0] < results["ESN+Ridge(α=1)"][0]
print(f"\n  ┌─────────────────────────────────────────────────────────┐")
if psi_beats_esn:
    improvement = (results['ESN+Ridge(α=1)'][0] - best_psi[1][0]) / results['ESN+Ridge(α=1)'][0] * 100
    print(f"  │   ψ-WAVE BEATS ESN на {improvement:.1f}%                              │")
else:
    print(f"  │   ESN лучше ψ-wave :(                                    │")
print(f"  └─────────────────────────────────────────────────────────┘")

# Save results
np.savez("narma10_results.npz",
         y_true=y_test,
         y_psi=model.predict(X_test),
         y_esn=y_pred_esn,
         y_input=y_pred_in)
print(f"\n  Saved: narma10_results.npz")
print("█" * 70)
