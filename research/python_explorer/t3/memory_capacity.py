#!/usr/bin/env python3
"""
САМЫЙ ЖЁСТКИЙ ТЕСТ v3: ПАМЯТЬ ψ-WAVE НА FAFB
==============================================
Задача Short-Term Memory (Jaeger 2002):
может ли ψ-wave на коннектоме восстановить u[t-k] из текущего состояния?

Сравнение с ESN (tanh, random weights) одинаковой размерности.
Гипотеза: ψ-wave унитарна → memory capacity → ∞ (теоретически).
"""

import time, math, numpy as np
from taas_checkmate import (
    step_psi_wave, compute_hamiltonian,
    PyTriOscillator,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

DT = 0.1
N_TOTAL = 5000
N_WASH = 500
N_TRAIN = 3000
N_TEST = 1500
READOUT_N = 1024

print("█" * 70)
print("  SHORT-TERM MEMORY: ψ-WAVE ON FAFB")
print("  Восстановление u[t-k] из ψ-состояния")
print("█" * 70)

# ─── Input ───
print("\n[1/6] Generating input...", end=" ", flush=True)
np.random.seed(42)
u = np.random.uniform(-0.5, 0.5, N_TOTAL).astype(np.float32)
print(f"{N_TOTAL} samples")

# ─── Load FAFB ───
print("[2/6] Loading FAFB...", end=" ", flush=True)
data = np.load("fafb_139k.npz")
row_ptr, col_idx, vals = data["row_ptr"], data["col_indices"], data["values"]
n = len(row_ptr) - 1
n_pad = (n + 7) // 8 * 8
print(f"{n:,} neurons")

# ─── Readout neurons ───
print("[3/6] Selecting readout neurons...", end=" ", flush=True)
rng = np.random.default_rng(42)
readout_idx = rng.choice(n, READOUT_N, replace=False)
readout_idx.sort()
print(f"{READOUT_N} readout neurons")

# ─── Input projection ───
print("[4/6] Building input projection...", end=" ", flush=True)
n_input = 96
input_idx = rng.choice(n, n_input, replace=False)
W_in = rng.uniform(-1, 1, n_input).astype(np.float32)
print(f"{n_input} input neurons, ||W_in||={np.linalg.norm(W_in):.1f}")

# ─── Init ψ ───
print("[5/6] Running ψ-wave on FAFB...")
theta = rng.uniform(0, 2*np.pi, n_pad).astype(np.float32)
v = np.cos(theta)
u_state = np.sin(theta)  # renaming local var
h_real = np.zeros(n_pad, dtype=np.float32)
h_imag = np.zeros(n_pad, dtype=np.float32)
speed = np.ones(n_pad, dtype=np.float32)
i_ext = np.zeros(n_pad, dtype=np.float32)

t_start = time.perf_counter()
# State matrix: [v_readout, u_readout] at each time
states = np.zeros((N_TOTAL, READOUT_N * 2), dtype=np.float32)

for step in range(N_TOTAL):
    i_ext[:] = 0.0
    # Inject input into input neurons
    for j, ni in enumerate(input_idx):
        i_ext[ni] = W_in[j] * u[step]

    compute_hamiltonian(row_ptr, col_idx, vals, v, u_state, h_real, h_imag)
    step_psi_wave(v, u_state, h_real, h_imag, i_ext, speed, DT, 1.0)

    states[step, :READOUT_N] = v[readout_idx]
    states[step, READOUT_N:] = u_state[readout_idx]

    if (step + 1) % 1000 == 0:
        elapsed = time.perf_counter() - t_start
        print(f"  [{step+1}/{N_TOTAL}] {elapsed:.0f}s ({elapsed/(step+1)*1000:.1f}ms/step)")

total_time = time.perf_counter() - t_start
print(f"  Done in {total_time:.0f}s")

# ─── Memory Test ───
print("\n[6/6] Testing memory capacity...")
print(f"  k (lag) |  ψ-R²   |  ψ-MSE   | ESN-R²   | ESN-MSE  | Winner")
print(f"  ────────┼─────────┼──────────┼──────────┼──────────┼───────")

X = states[N_WASH:]
u_trim = u[N_WASH:]
X_tr = X[:N_TRAIN]
X_te = X[N_TRAIN:N_TRAIN+N_TEST]
u_tr = u_trim[:N_TRAIN]
u_te = u_trim[N_TRAIN:N_TRAIN+N_TEST]

# ESN for comparison
N_ESN = READOUT_N * 2
W_esn = np.random.randn(N_ESN, N_ESN) * 0.15
eigvals = np.linalg.eigvals(W_esn)
W_esn /= np.max(np.abs(eigvals)) * 0.95
W_in_esn = np.random.uniform(-0.5, 0.5, N_ESN)
alpha_l = 0.5

x_esn = np.zeros(N_ESN)
X_esn = np.zeros((N_TOTAL, N_ESN), dtype=np.float32)
for step in range(N_TOTAL):
    x_pre = np.tanh(W_esn @ x_esn + W_in_esn * u[step])
    x_esn = (1 - alpha_l) * x_esn + alpha_l * x_pre
    X_esn[step] = x_esn
X_esn_w = X_esn[N_WASH:]
X_esn_tr = X_esn_w[:N_TRAIN]
X_esn_te = X_esn_w[N_TRAIN:N_TRAIN+N_TEST]

# Test memory at different lags
for k in [1, 2, 3, 5, 10, 20, 50, 100]:
    y_k = u_te[:-k] if k > 0 else u_te
    X_k = X_te[:-k] if k > 0 else X_te
    y_tr_k = u_tr[:-k] if k > 0 else u_tr
    X_tr_k = X_tr[:-k] if k > 0 else X_tr
    X_esn_tr_k = X_esn_tr[:-k] if k > 0 else X_esn_tr
    X_esn_te_k = X_esn_te[:-k] if k > 0 else X_esn_te

    # ψ-wave readout
    from sklearn.linear_model import RidgeCV
    m = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], fit_intercept=True)
    m.fit(X_tr_k, y_tr_k)
    yp = m.predict(X_k)
    psi_r2 = r2_score(y_k, yp)
    psi_mse = mean_squared_error(y_k, yp)

    # ESN readout
    m_e = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], fit_intercept=True)
    m_e.fit(X_esn_tr_k, y_tr_k)
    yp_e = m_e.predict(X_esn_te_k)
    esn_r2 = r2_score(y_k, yp_e)
    esn_mse = mean_squared_error(y_k, yp_e)

    winner = "ψ" if psi_r2 > esn_r2 else "ESN" if esn_r2 > psi_r2 else "="
    print(f"  t-{k:<4} |  {psi_r2:+.3f}  | {psi_mse:.6f} |  {esn_r2:+.3f}  | {esn_mse:.6f} | {winner}")

print("\n" + "█" * 70)
print("  ВЫВОД: ψ-wave память vs ESN на FAFB коннектоме")
print("█" * 70)
