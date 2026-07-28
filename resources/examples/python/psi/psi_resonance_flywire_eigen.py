#!/usr/bin/env python3
"""
ψ-Wave Free Evolution — FlyWire Eigenmode Spectrum
====================================================

Запускает свободную ψ-wave эволюцию (без external drive) на
реальном FlyWire коннектоме, чтобы извлечь собственные моды
графа из FFT readout сигнала.

Сохраняет в /tmp/psi_resonance_flywire_eigen.npz

Запуск:
    PYTHONPATH=target/debug:fly:. python3 examples/python/psi_resonance_flywire_eigen.py
"""

import sys
import os
import math
import time
import gzip
import csv
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in [os.path.join(PROJ, 'target', 'release'),
           os.path.join(PROJ, 'target', 'debug')]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import taas_checkmate as tcm

DATA_DIR = '/home/taaslab/Проекты/asm_engines/fly/archive'
CONN_PATH = os.path.join(DATA_DIR, 'connections_princeton.csv.gz')


def load_connectome():
    """Load FlyWire CSR matrix (no neuron metadata needed)."""
    t0 = time.perf_counter()

    print("[1/3] Counting synapses...")
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

    print("[2/3] Building CSR matrix...")
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
                if nt == 'GABA' or nt == 'GLY':
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
    return row_ptr, col_indices, values, n, actual_n


def run_free_evolution(row_ptr, col_indices, values, n,
                       n_steps=1024, dt=1.0, hbar=1.0):
    """Run free ψ-wave evolution and capture eigenmode spectrum."""
    t0 = time.perf_counter()

    n_padded = (n + 7) // 8 * 8
    h_real = np.zeros(n_padded, dtype=np.float32)
    h_imag = np.zeros(n_padded, dtype=np.float32)
    i_ext = np.zeros(n_padded, dtype=np.float32)
    speed = np.ones(n_padded, dtype=np.float32)

    # Random initial phases
    rng = np.random.RandomState(42)
    phases = rng.uniform(0, 2 * math.pi, n_padded).astype(np.float32)
    v = np.cos(phases).astype(np.float32)
    u = np.sin(phases).astype(np.float32)

    # Sample 5000 random neurons for readout
    sample_n = min(5000, n)
    sample_idx = np.sort(rng.choice(n, sample_n, replace=False)).astype(np.int32)
    eigen_signal = np.zeros(n_steps, dtype=np.float32)
    readout_tmp = np.zeros(sample_n, dtype=np.float32)

    print(f"[3/3] Free evolution ({n_steps} steps, {sample_n} sample neurons)...")
    for step in range(n_steps):
        tcm.sparse.compute_hamiltonian(
            row_ptr, col_indices, values, v, u, h_real, h_imag,
        )
        tcm.step_psi_wave(v, u, h_real, h_imag, i_ext, speed, dt, hbar)
        readout_tmp[:] = v[sample_idx]
        eigen_signal[step] = float(np.mean(readout_tmp))

        if step % 256 == 0 and step > 0:
            print(f"  Step {step}/{n_steps} [{time.perf_counter()-t0:.1f}s]")

    elapsed = time.perf_counter() - t0

    # FFT
    signal = eigen_signal[-1024:]
    window = np.hanning(len(signal))
    fft = np.fft.rfft((signal - np.mean(signal)) * window)
    freqs_fft = np.fft.rfftfreq(len(signal), d=dt / 1000.0)
    magnitude = np.abs(fft)

    # Find top 5 eigenmode peaks
    peak_mask = np.ones(len(magnitude[1:]), dtype=bool)
    peaks_freq = []
    peaks_mag = []
    freq_res = freqs_fft[1] - freqs_fft[0] if len(freqs_fft) > 1 else 1.0

    for _ in range(5):
        if not np.any(peak_mask):
            break
        candidate = np.argmax(magnitude[1:] * peak_mask) + 1
        pk_freq = float(freqs_fft[candidate])
        pk_mag = float(magnitude[candidate])
        peaks_freq.append(pk_freq)
        peaks_mag.append(pk_mag)
        exclude_range = max(1, int(3 / freq_res))
        low = max(0, candidate - exclude_range)
        high = min(len(peak_mask), candidate + exclude_range + 1)
        peak_mask[low:high] = False

    print(f"  Done in {elapsed:.1f}s")
    print(f"  Top eigenmodes:")
    for i, (pf, pm) in enumerate(zip(peaks_freq, peaks_mag)):
        print(f"    Mode {i+1}: {pf:.1f} Hz  (|FFT|={pm:.2f})")

    return freqs_fft, magnitude, peaks_freq, peaks_mag, elapsed


def main():
    print("═" * 60)
    print("  FlyWire Eigenmode Spectrum")
    print("═" * 60)

    row_ptr, col_indices, values, n, actual_n = load_connectome()

    freqs_fft, magnitude, peaks_freq, peaks_mag, elapsed = run_free_evolution(
        row_ptr, col_indices, values, n,
        n_steps=1024, dt=1.0, hbar=1.0,
    )

    # Save
    save_path = '/tmp/psi_resonance_flywire_eigen.npz'
    np.savez(save_path,
             eigen_freqs=freqs_fft,
             eigen_mag=magnitude,
             peak_freqs=np.array(peaks_freq),
             peak_mags=np.array(peaks_mag),
             n_neurons=n,
             n_synapses=actual_n)
    print(f"\n✅ Saved to: {save_path}")
    print("═" * 60)


if __name__ == '__main__':
    main()
