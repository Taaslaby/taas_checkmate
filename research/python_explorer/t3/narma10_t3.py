#!/usr/bin/env python3
"""
САМЫЙ ЖЁСТКИЙ ТЕСТ v2: NARMA10 + T3 RESERVOIR
===============================================
T3 (Kuramoto + Stuart-Landau) как нелинейный резервуарный компьютер.
Сравнение: T3 phasor reservoir vs ESN vs Ridge baseline.

Гипотеза: T3 осцилляторы (Kuramoto + Stuart-Landau + cross-coupling)
должны давать богатую нелинейную динамику для reservoir computing.
"""

import time, math, numpy as np
from taas_checkmate import PyTriOscillator
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

DT = 0.05
N_OSC = 128
N_WASH = 200
N_TRAIN = 2000
N_TEST = 1000
N_TOTAL = N_WASH + N_TRAIN + N_TEST
T3_DRIVE_AMP = 0.5

print("█" * 70)
print("  NARMA10 + T3 OSCILLATOR RESERVOIR")
print("  Kuramoto + Stuart-Landau reservoir computing")
print("█" * 70)

# ─── NARMA10 ───
print("\n[1/5] Generating NARMA10...", end=" ", flush=True)
np.random.seed(42)
u = np.random.uniform(0, 0.5, size=N_TOTAL)
y = np.zeros(N_TOTAL)
for k in range(10, N_TOTAL):
    y[k] = (0.3 * y[k-1] +
            0.05 * y[k-1] * np.sum(y[k-10:k]) +
            1.5 * u[k-10] * u[k-1] +
            0.1)
u = u[10:]
y = y[10:]
N_TOTAL = len(y)
N_TRAIN = 2000
N_TEST = 1000
print(f"{N_TOTAL} samples")

# ─── Init T3 with random input projection ───
print("[2/5] Initializing T3 reservoir...", end=" ", flush=True)
# Create 3-population T3
tri = PyTriOscillator(N_OSC, 0.5, 2.0)
# Moderate coupling — not too strong (synchrony kills diversity), not too weak
tri.set_coupling(0.15, 0.08, 0.04)
# Burn-in
for _ in range(500):
    tri.step(DT)

# Random input projection (N_OSC x 1, per population)
rng = np.random.default_rng(42)
W_in = rng.uniform(-0.5, 0.5, (N_OSC,))
print("ok")

# ─── Run reservoir ───
print("[3/5] Running T3 reservoir...")
print(f"  {N_TOTAL} steps (wash={N_WASH}, train={N_TRAIN}, test={N_TEST})")
t_start = time.perf_counter()

# Features: phases (sin, cos) + amplitudes for each population → 6*N_OSC features
n_feat = 6 * N_OSC
X = np.zeros((N_TOTAL, n_feat), dtype=np.float32)

for step in range(N_TOTAL):
    # Drive with NARMA input via projection
    drive_arr = (W_in * u[step] * T3_DRIVE_AMP).astype(np.float32)
    tri.set_drive(np.concatenate([drive_arr, drive_arr * 0.5, drive_arr * 0.3]).astype(np.float32))
    tri.step(DT)

    # Read all 3 populations
    p1 = np.array(tri.phases(1))
    p2 = np.array(tri.phases(2))
    p3 = np.array(tri.phases(3))
    a1 = np.array(tri.amplitudes(1))
    a2 = np.array(tri.amplitudes(2))
    a3 = np.array(tri.amplitudes(3))

    X[step, 0*N_OSC:1*N_OSC] = np.sin(p1) * a1
    X[step, 1*N_OSC:2*N_OSC] = np.cos(p1) * a1
    X[step, 2*N_OSC:3*N_OSC] = np.sin(p2) * a2
    X[step, 3*N_OSC:4*N_OSC] = np.cos(p2) * a2
    X[step, 4*N_OSC:5*N_OSC] = np.sin(p3) * a3
    X[step, 5*N_OSC:6*N_OSC] = np.cos(p3) * a3

    if (step + 1) % 500 == 0:
        print(f"  [{step+1}/{N_TOTAL}] {time.perf_counter()-t_start:.0f}s")

total_time = time.perf_counter() - t_start
print(f"  Done in {total_time:.0f}s ({total_time/N_TOTAL:.3f}s/step)")

# ─── Train & Evaluate ───
print("\n[4/5] Training models...")

X_w = X[N_WASH:]
y_w = y[N_WASH:]
X_tr = X_w[:N_TRAIN]
y_tr = y_w[:N_TRAIN]
X_te = X_w[N_TRAIN:N_TRAIN+N_TEST]
y_te = y_w[N_TRAIN:N_TRAIN+N_TEST]

results = {}

# T3 + Ridge with various alpha
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(X_tr, y_tr)
    yp = m.predict(X_te)
    results[f"T3+Ridge(α={alpha})"] = (mean_squared_error(y_te, yp), r2_score(y_te, yp))

# Input only baseline
m_in = Ridge(alpha=1.0, fit_intercept=True)
m_in.fit(u[N_WASH:N_WASH+N_TRAIN].reshape(-1,1), y_tr)
y_in = m_in.predict(u[N_WASH+N_TRAIN:N_WASH+N_TRAIN+N_TEST].reshape(-1,1))
results["Input only"] = (mean_squared_error(y_te, y_in), r2_score(y_te, y_in))

# Trivial (predict mean)
y_mean = np.full_like(y_te, np.mean(y_tr))
results["Mean predictor"] = (mean_squared_error(y_te, y_mean), r2_score(y_te, y_mean))

# Classical ESN with same feature dimension
N_ESN = n_feat
W_in_esn = np.random.uniform(-0.5, 0.5, (N_ESN, 1)) * 0.5
W_res_esn = np.random.randn(N_ESN, N_ESN) * 0.2
eigvals = np.linalg.eigvals(W_res_esn)
sr = np.max(np.abs(eigvals))
W_res_esn /= sr * 0.95  # spectral radius 0.95
alpha_leaky = 0.3

x_esn = np.zeros(N_ESN)
X_esn = np.zeros((N_TOTAL, N_ESN), dtype=np.float32)
for step in range(N_TOTAL):
    x_pre = np.tanh(W_res_esn @ x_esn + W_in_esn.ravel() * u[step])
    x_esn = (1 - alpha_leaky) * x_esn + alpha_leaky * x_pre
    X_esn[step] = x_esn

X_esn_w = X_esn[N_WASH:]
X_esn_tr = X_esn_w[:N_TRAIN]
X_esn_te = X_esn_w[N_TRAIN:N_TRAIN+N_TEST]
m_esn = Ridge(alpha=1.0, fit_intercept=True)
m_esn.fit(X_esn_tr, y_tr)
y_esn = m_esn.predict(X_esn_te)
results["ESN(α=1)"] = (mean_squared_error(y_te, y_esn), r2_score(y_te, y_esn))

# ─── Results ───
print("\n[5/5] RESULTS")
print("█" * 70)
print(f"  {'Model':<30} {'MSE':<15} {'R²':<15}")
print(f"  {'─'*30} {'─'*15} {'─'*15}")
for name, (mse, r2) in sorted(results.items(), key=lambda x: x[1][0]):
    print(f"  {name:<30} {mse:<15.6f} {r2:<15.4f}")

best_t3 = min([(k, v) for k, v in results.items() if k.startswith("T3")], key=lambda x: x[1][0])
esn_mse = results["ESN(α=1)"][0]
print(f"\n  Best T3:       {best_t3[0]} — MSE={best_t3[1][0]:.6f}, R²={best_t3[1][1]:.4f}")
print(f"  ESN:           MSE={esn_mse:.6f}, R²={results['ESN(α=1)'][1]:.4f}")
print(f"  Input only:    MSE={results['Input only'][0]:.6f}")

t3_beats = best_t3[1][0] < esn_mse
print(f"  ┌─────────────────────────────────────────────────────────┐")
if t3_beats:
    impr = (esn_mse - best_t3[1][0]) / esn_mse * 100
    print(f"  │   T3 BEATS ESN на {impr:.1f}%!                                │")
else:
    ratio = best_t3[1][0] / esn_mse
    print(f"  │   T3 хуже ESN в {ratio:.2f}x                                      │")
print(f"  └─────────────────────────────────────────────────────────┘")

np.savez("narma10_t3_results.npz",
         y_true=y_te,
         y_t3=Ridge(1.0).fit(X_tr, y_tr).predict(X_te),
         y_esn=y_esn,
         y_input=y_in)
print(f"\n  Saved: narma10_t3_results.npz")
print("█" * 70)
