#!/usr/bin/env python3
"""
FlyWire Eigenvalue Analysis
============================

Загружает CSR матрицу FlyWire коннектома, вычисляет top-K собственных
чисел через scipy.sparse.linalg.eigs (ARPACK), и сравнивает с
измеренным ψ-wave резонансом на 122 Hz.

Гипотеза: резонансная частота ω_res = λ_n / (2π · ħ),
где λ_n — собственное число весовой матрицы W.

Запуск:
    python3 examples/python/flywire_eigenvalue_analysis.py
"""

import os
import sys
import time
import gzip
import csv
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

DATA_DIR = '/home/taaslab/Проекты/asm_engines/fly/archive'
CONN_PATH = os.path.join(DATA_DIR, 'connections_princeton.csv.gz')
OUT_PATH = '/tmp/flywire_eigenvalues.npz'

HBAR = 1.0  # matches ψ-wave config


def load_csr():
    """Load FlyWire connectome as scipy CSR matrix."""
    t0 = time.perf_counter()

    print("[1/4] Counting synapses...")
    pre_count = defaultdict(int)
    total_rows = 0
    with gzip.open(CONN_PATH, 'rt') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                pre_count[int(row[0])] += 1
                total_rows += 1

    all_ids = sorted(set(pre_count.keys()))
    root_to_idx = {rid: i for i, rid in enumerate(all_ids)}
    n = len(all_ids)
    print(f"  {n:,} neurons, {total_rows:,} rows [{time.perf_counter()-t0:.1f}s]")

    print("[2/4] Building CSR arrays...")
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    for rid, cnt in pre_count.items():
        row_ptr[root_to_idx[rid] + 1] = cnt
    row_ptr = np.cumsum(row_ptr, dtype=np.int32)

    col_indices = np.zeros(total_rows, dtype=np.int32)
    values = np.zeros(total_rows, dtype=np.float32)
    cursor = np.zeros(n, dtype=np.int32)

    with gzip.open(CONN_PATH, 'rt') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                pre = int(row[0])
                post = int(row[1])
                weight = float(row[3])
                pre_idx = root_to_idx.get(pre)
                post_idx = root_to_idx.get(post)
                if pre_idx is None or post_idx is None:
                    continue
                pos = row_ptr[pre_idx] + cursor[pre_idx]
                if pos >= total_rows:
                    continue
                col_indices[pos] = np.int32(post_idx)
                nt = row[4].strip().upper() if len(row) > 4 else ''
                if nt in ('GABA', 'GLY'):
                    values[pos] = -float(weight)
                else:
                    values[pos] = float(weight)
                cursor[pre_idx] += 1

    # Trim
    actual_n = int(np.sum(cursor))
    if actual_n < total_rows:
        new_ci = np.empty(actual_n, dtype=np.int32)
        new_vals = np.empty(actual_n, dtype=np.float32)
        new_rp = np.zeros(n + 1, dtype=np.int32)
        dst = 0
        for i in range(n):
            cnt = int(cursor[i])
            if cnt > 0:
                src_start = int(row_ptr[i])
                new_ci[dst:dst+cnt] = col_indices[src_start:src_start+cnt]
                new_vals[dst:dst+cnt] = values[src_start:src_start+cnt]
            new_rp[i] = dst
            dst += cnt
        new_rp[-1] = dst
        col_indices = new_ci
        values = new_vals
        row_ptr = new_rp
        actual_n = dst

    print(f"  CSR: {n:,} n, {actual_n:,} syn [{time.perf_counter()-t0:.1f}s]")
    return n, actual_n, row_ptr, col_indices, values


def compute_eigenvalues(n, row_ptr, col_indices, values, k=20):
    """Compute top-K eigenvalues of the weight matrix using ARPACK."""
    print(f"\n[3/4] Creating scipy CSR matrix ({n:,}×{n:,}, {len(values):,} nnz)...")
    # Convert to 64-bit indices (required by scipy)
    W = csr_matrix(
        (values.astype(np.float64), col_indices.astype(np.int32), row_ptr.astype(np.int32)),
        shape=(n, n),
    )
    print(f"  Matrix size: {W.shape}, density: {W.nnz / (n*n) * 100:.4f}%")

    # Estimate memory for eigs
    mem_nnz = W.nnz * (8 + 8) / 1024**3
    mem_vec = k * n * 8 / 1024**3
    print(f"  Estimated memory: matrix={mem_nnz:.2f} GB, "
          f"vectors={mem_vec:.2f} GB, total={mem_nnz+mem_vec:.2f} GB")

    # Compute eigenvalues with largest real part (LR)
    # The 122 Hz resonance has Re(lambda) = 2pi * 122 * hbar ~ 766
    # LR finds the most positive eigenvalues, which include the resonance
    print(f"\n  Computing top {k} eigenvalues (LR — largest real part)...")
    t0 = time.perf_counter()

    try:
        eigvals = eigs(
            W, k=k, which='LR',
            maxiter=500, tol=1e-4,
            return_eigenvectors=False,
        )
        elapsed = time.perf_counter() - t0
        print(f"  ✅ ARPACK (LR) converged in {elapsed:.1f}s")

    except Exception as e:
        print(f"  ⚠️ LR failed: {e}")
        print(f"  Trying shift-invert near 766 (resonance)...")
        t0 = time.perf_counter()
        try:
            eigvals = eigs(
                W, k=k, sigma=766.0, which='LM',
                maxiter=300, tol=1e-4,
                return_eigenvectors=False,
            )
            elapsed = time.perf_counter() - t0
            print(f"  ✅ ARPACK (shift-invert) converged in {elapsed:.1f}s")
        except Exception as e2:
            print(f"  ❌ Shift-invert also failed: {e2}")
            print(f"  Trying with k={min(k, 10)}...")
            eigvals = eigs(
                W, k=min(k, 10), which='LM',
                maxiter=300, tol=1e-3,
                return_eigenvectors=False,
            )

    # Convert to frequencies: ω = Re(λ), f = Re(λ) / (2π · ħ)
    freqs_hz = np.real(eigvals) / (2 * np.pi * HBAR)

    # Sort by magnitude of real part
    sort_idx = np.argsort(-np.abs(freqs_hz))
    freqs_hz = freqs_hz[sort_idx]
    eigvals_sorted = eigvals[sort_idx]

    return eigvals_sorted, freqs_hz


def main():
    print("═" * 60)
    print("  FlyWire Eigenvalue Analysis")
    print("═" * 60)

    n, actual_n, row_ptr, col_indices, values = load_csr()

    k = min(20, n - 2)
    eigvals, freqs_hz = compute_eigenvalues(n, row_ptr, col_indices, values, k=k)

    # ─── Results ──────────────────────────────────────
    print("\n" + "═" * 60)
    print("  EIGENVALUE RESULTS")
    print("═" * 60)
    print(f"\n  Matrix: {n:,}×{n:,}, {len(values):,} synapses")
    print(f"  ħ = {HBAR}")

    print(f"\n  {'Mode':>6s} | {'Re(λ)':>10s} | {'Im(λ)':>10s} | {'|λ|':>10s} | {'f (Hz)':>8s} | {'Match?':>8s}")
    print(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*8}")

    # Compare with ψ-wave resonance (122 Hz)
    res_freq = 122.0  # measured resonance

    for i in range(min(len(eigvals), 20)):
        lam = eigvals[i]
        f_hz = freqs_hz[i]
        re_l = np.real(lam)
        im_l = np.imag(lam)
        abs_l = np.abs(lam)

        # Check if this eigenvalue matches the resonance
        match = ''
        if abs(f_hz) > 10 and abs(f_hz) < 300:
            if abs(abs(f_hz) - res_freq) < 5:
                match = '=== RES'
            elif abs(abs(f_hz) - res_freq) < 15:
                match = '~near'

        print(f"  {i+1:>5d} | {re_l:>+10.3f} | {im_l:>+10.3f} | {abs_l:>10.3f} | {f_hz:>+8.1f} | {match:>8s}")

    # ─── Interpretation ────────────────────────────────
    print("\n" + "═" * 60)
    print("  INTERPRETATION")
    print("═" * 60)

    # Find eigenvalues near resonance
    near_res = [f for f in freqs_hz if 100 < abs(f) < 150]
    if near_res:
        closest = min(near_res, key=lambda f: abs(abs(f) - res_freq))
        delta = abs(abs(closest) - res_freq)
        print(f"\n  Resonance prediction:   {res_freq:.1f} Hz (ψ-wave measurement)")
        print(f"  Nearest eigenvalue:     {closest:+.1f} Hz (Δf = {delta:.1f} Hz)")
        if delta < 5:
            print(f"  ✅ CONFIRMED: resonance matches eigenvalue!\n"
                  f"     FlyWire connectome eigenmode = {closest:+.1f} Hz")
        else:
            print(f"  ⚠️  Partial match: Δf={delta:.1f} Hz > 5 Hz")
            print(f"     Возможные причины: non-normality (complex eigenvalues),")
            print(f"     или resonance не от одного eigenmode, а от их суперпозиции")
    else:
        print(f"\n  ⚠️  No eigenvalues found in 100-150 Hz range")
        print(f"     Resonance at {res_freq:.1f} Hz may be from:")
        print(f"     - Interaction of multiple eigenmodes")
        print(f"     - Nonlinear effect from driving")
        print(f"     - Numerical artifact in ψ-wave FFT")

    # ─── Save ──────────────────────────────────────────
    np.savez(OUT_PATH,
             eigvals=eigvals,
             freqs_hz=freqs_hz,
             resonance_hz=res_freq,
             n_neurons=n,
             n_synapses=actual_n)
    print(f"\n  Saved to: {OUT_PATH}")
    print("═" * 60)


if __name__ == '__main__':
    main()
