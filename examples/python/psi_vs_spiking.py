#!/usr/bin/env python3
"""
ψ-Wave vs Poisson Spiking — FlyWire Resonance Comparison
=========================================================

Сравнивает отклик FlyWire коннектома на два типа входного сигнала:
1. Sine wave на resonant частоте (122 Hz)
2. Poisson spike train с той же средней частотой (122 Hz)

Гипотеза: spiking input возбуждает более широкий спектр eigenmodes,
чем sine, и даёт больший D/P ratio на DN/MN.

Запуск:
    PYTHONPATH=target/debug:fly:. python3 examples/python/psi_vs_spiking.py
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
NEURONS_PATH = os.path.join(DATA_DIR, 'neurons.csv.gz')

# Config
FREQ_HZ = 122.0      # resonant frequency
N_STEPS = 1024       # steps per run
DT = 1.0             # ms
HBAR = 1.0
INPUT_AMP = 30.0     # same as resonance experiment


def load_connectome():
    """Load minimal FlyWire connectome for ψ-wave."""
    t0 = time.perf_counter()
    print("[1/4] Loading connectome...")

    # Neurons metadata (DN/MN only)
    class_map, type_map = {}, {}
    with gzip.open(NEURONS_PATH, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid_str = row.get('Root ID', '').strip()
            if not rid_str:
                continue
            try:
                rid = int(rid_str)
            except ValueError:
                continue
            cl = row.get('Class', '').strip()
            if cl:
                class_map[rid] = cl
            pct = row.get('Primary Cell Type', '').strip()
            if pct:
                type_map[rid] = pct

    # Count synapses
    pre_count = defaultdict(int)
    with gzip.open(CONN_PATH, 'rt') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                pre_count[int(row[0])] += 1

    all_ids = sorted(set(pre_count.keys()))
    root_to_idx = {rid: i for i, rid in enumerate(all_ids)}
    n = len(all_ids)

    # Build CSR
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    for rid, cnt in pre_count.items():
        row_ptr[root_to_idx[rid] + 1] = cnt
    row_ptr = np.cumsum(row_ptr, dtype=np.int32)
    total_rows = int(row_ptr[-1])
    col_indices = np.zeros(total_rows, dtype=np.int32)
    values = np.zeros(total_rows, dtype=np.float32)
    cursor = np.zeros(n, dtype=np.int32)

    with gzip.open(CONN_PATH, 'rt') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                pre, post, weight = int(row[0]), int(row[1]), float(row[3])
                pre_idx = root_to_idx.get(pre)
                post_idx = root_to_idx.get(post)
                if pre_idx is None or post_idx is None:
                    continue
                pos = row_ptr[pre_idx] + cursor[pre_idx]
                if pos >= total_rows:
                    continue
                col_indices[pos] = np.int32(post_idx)
                nt = row[4].strip().upper() if len(row) > 4 else ''
                values[pos] = -weight if nt in ('GABA', 'GLY') else weight
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
                src = int(row_ptr[i])
                new_ci[dst:dst+cnt] = col_indices[src:src+cnt]
                new_vals[dst:dst+cnt] = values[src:src+cnt]
            new_rp[i] = dst
            dst += cnt
        new_rp[-1] = dst
        col_indices, values, row_ptr = new_ci, new_vals, new_rp

    # Find DN indices
    dn_indices = []
    dn_rids = set()
    for rid, idx in root_to_idx.items():
        cl = class_map.get(rid, '')
        if cl.startswith('descending'):
            dn_indices.append(idx)
            dn_rids.add(rid)
        pct = type_map.get(rid, '')
        if pct.startswith(('DN', 'MDN', 'APDN', 'tdDN')) and rid not in dn_rids:
            dn_indices.append(idx)

    # Find visual afferents
    VISUAL = ('Tm', 'TmY', 'Mi', 'Mi1', 'L1', 'L2', 'L3', 'L4', 'L5',
              'T4', 'T5', 'LC', 'LPTC', 'lamina', 'medulla', 'lobula')
    visual_indices = []
    for rid, idx in root_to_idx.items():
        pct = type_map.get(rid, '')
        for pat in VISUAL:
            if pct.startswith(pat) or pat in pct:
                visual_indices.append(idx)
                break

    print(f"  {n:,} neurons, {actual_n:,} syn, "
          f"{len(dn_indices)} DN, {len(visual_indices)} visual "
          f"[{time.perf_counter()-t0:.1f}s]")

    return (row_ptr, col_indices, values, n,
            np.array(dn_indices, dtype=np.int32),
            np.array(visual_indices, dtype=np.int32))


def run_psi(input_type, row_ptr, col_indices, values, n,
            dn_arr, visual_arr, sensory_all):
    """Run ψ-wave with sine or Poisson spike input at resonance frequency."""
    n_padded = (n + 7) // 8 * 8
    n_readout = len(dn_arr)
    k_readout = min(5000, n_readout)

    rng = np.random.RandomState(42)
    phases = rng.uniform(0, 2 * math.pi, n_padded).astype(np.float32)
    v = np.cos(phases).astype(np.float32)
    u = np.sin(phases).astype(np.float32)
    h_real = np.zeros(n_padded, dtype=np.float32)
    h_imag = np.zeros(n_padded, dtype=np.float32)
    i_ext = np.zeros(n_padded, dtype=np.float32)
    speed = np.ones(n_padded, dtype=np.float32)

    omega = 2.0 * math.pi * FREQ_HZ
    dt_sec = DT / 1000.0
    spike_prob = FREQ_HZ * dt_sec  # Poisson prob per step

    readout_signal = np.zeros(N_STEPS, dtype=np.float32)
    readout_tmp = np.zeros(k_readout, dtype=np.float32)

    spike_count = 0
    total_drive = 0.0  # track total energy delivered

    t0 = time.perf_counter()

    for step in range(N_STEPS):
        t_sec = step * dt_sec
        i_ext.fill(0.0)

        if input_type == 'sine':
            drive_val = INPUT_AMP * math.sin(omega * t_sec)
            i_ext[sensory_all] = drive_val
            total_drive += abs(drive_val) * len(sensory_all)
        else:  # poisson
            # Generate Poisson spikes at rate = FREQ_HZ
            if rng.random() < spike_prob:
                i_ext[sensory_all] = INPUT_AMP
                spike_count += 1
                total_drive += INPUT_AMP * len(sensory_all)

        tcm.sparse.compute_hamiltonian(
            row_ptr, col_indices, values, v, u, h_real, h_imag,
        )
        tcm.step_psi_wave(v, u, h_real, h_imag, i_ext, speed, DT, HBAR)

        readout_tmp[:] = v[dn_arr[:k_readout]]
        readout_signal[step] = float(np.mean(readout_tmp))

    elapsed = time.perf_counter() - t0

    # FFT analysis
    signal = readout_signal[-min(1024, N_STEPS):]
    window = np.hanning(len(signal))
    fft = np.fft.rfft((signal - np.mean(signal)) * window)
    freqs_fft = np.fft.rfftfreq(len(signal), d=DT / 1000.0)
    magnitude = np.abs(fft)

    # Total power
    total_power = float(np.sum(magnitude**2))

    # Power at drive frequency
    drive_bin = np.argmin(np.abs(freqs_fft - FREQ_HZ))
    drive_power = float(magnitude[drive_bin])

    # Natural peak (exclude DC)
    peak_idx = np.argmax(magnitude[1:]) + 1
    spectral_peak = float(freqs_fft[peak_idx])
    spectral_power = float(magnitude[peak_idx])

    # Fraction of power in drive bin
    drive_band = max(1, int(5.0 / (freqs_fft[1] - freqs_fft[0])))
    lo = max(1, drive_bin - drive_band)
    hi = min(len(magnitude), drive_bin + drive_band + 1)
    power_in_band = float(np.sum(magnitude[lo:hi]**2))
    band_fraction = power_in_band / max(total_power, 1e-12)

    # Spike rate
    actual_rate = spike_count / (N_STEPS * dt_sec)

    # Energy normalization factor
    energy_norm = total_drive / max(N_STEPS, 1)

    return {
        'input_type': input_type,
        'drive_power': drive_power,
        'drive_ratio': drive_power / max(spectral_power, 1e-12),
        'band_fraction': band_fraction,
        'total_power': total_power,
        'spectral_peak': spectral_peak,
        'spectral_power': spectral_power,
        'actual_rate_hz': actual_rate,
        'elapsed': elapsed,
        'total_drive': total_drive,
        'energy_norm': energy_norm,
        'drive_power_norm': drive_power / max(energy_norm, 1e-12),  # energy-normalized
    }


def main():
    print("═" * 72)
    print("  ψ-Wave vs Poisson Spiking — FlyWire Resonance at 122 Hz")
    print("═" * 72)

    row_ptr, col_indices, values, n, dn_arr, visual_arr = load_connectome()

    # Build sensory set (visual + random 3%)
    print("\n[2/4] Building sensory set...")
    readout_set = set(dn_arr.tolist())
    sensory = list(visual_arr[:min(500, len(visual_arr))])
    rng = np.random.RandomState(42)
    extra = rng.choice(n, max(1, n // 33), replace=False).tolist()
    sensory = list(set(sensory) | set(extra))
    sensory = [i for i in sensory if i not in readout_set]
    sensory_arr = np.array(sensory, dtype=np.int32)
    print(f"  Sensory neurons: {len(sensory_arr)}")

    # ─── Run comparison ────────────────────────────────
    print("\n[3/4] Running ψ-wave comparison...")

    sine_result = run_psi('sine', row_ptr, col_indices, values, n,
                          dn_arr, visual_arr, sensory_arr)
    print(f"\n  ─── Sine @ {FREQ_HZ:.1f} Hz ───")
    print(f"    drive_power={sine_result['drive_power']:.4f}")
    print(f"    drive_ratio={sine_result['drive_ratio']:.4f}")
    print(f"    band_fraction={sine_result['band_fraction']:.4f}")
    print(f"    spectral_peak={sine_result['spectral_peak']:.1f} Hz")
    print(f"    total_power={sine_result['total_power']:.1f}")
    print(f"    elapsed={sine_result['elapsed']:.1f}s")

    spike_result = run_psi('poisson', row_ptr, col_indices, values, n,
                           dn_arr, visual_arr, sensory_arr)
    print(f"\n  ─── Poisson @ {FREQ_HZ:.1f} Hz ───")
    print(f"    drive_power={spike_result['drive_power']:.4f}")
    print(f"    drive_ratio={spike_result['drive_ratio']:.4f}")
    print(f"    band_fraction={spike_result['band_fraction']:.4f}")
    print(f"    spectral_peak={spike_result['spectral_peak']:.1f} Hz")
    print(f"    actual_rate={spike_result['actual_rate_hz']:.1f} Hz")
    print(f"    total_power={spike_result['total_power']:.1f}")
    print(f"    elapsed={spike_result['elapsed']:.1f}s")

    # ─── Results ───────────────────────────────────────
    print("\n" + "═" * 72)
    print("  RESULTS — ψ vs Spiking at 122 Hz")
    print("═" * 72)

    print(f"\n  {'Metric':>20s} | {'Sine':>10s} | {'Poisson':>10s} | {'Ratio':>8s}")
    print(f"  {'-'*20} | {'-'*10} | {'-'*10} | {'-'*8}")

    metrics = [
        ('drive_power', 'drive_power'),
        ('drive_power_norm', 'drive_power_norm'),
        ('drive_ratio', 'drive_ratio'),
        ('band_fraction', 'band_fraction'),
        ('total_power', 'total_power'),
        ('total_drive', 'total_drive'),
        ('spectral_peak', 'spectral_peak'),
    ]
    for label, key in metrics:
        sv = sine_result[key]
        pv = spike_result[key]
        ratio = sv / max(pv, 1e-12) if pv != 0 else float('inf')
        fmt = '.4f' if key in ('drive_power', 'drive_power_norm', 'drive_ratio', 'band_fraction') else '.1f'
        print(f"  {label:>20s} | {sv:{fmt}} | {pv:{fmt}} | {ratio:>8.2f}x")

    # ─── Verdict ───────────────────────────────────────
    print("\n" + "═" * 72)
    print("  VERDICT")
    print("═" * 72)

    dp_ratio = sine_result['drive_power_norm'] / max(spike_result['drive_power_norm'], 1e-12)
    if dp_ratio > 1.5:
        print(f"\n  ⚡ Sine {dp_ratio:.2f}x stronger than Poisson at drive frequency")
        print(f"     ψ-wave резонирует на чистый тон лучше, чем на спайки.")
        print(f"     Poisson spikes возбуждают широкий спектр eigenmodes,")
        print(f"     энергия рассеивается по множеству частот.")
    elif dp_ratio < 0.67:
        print(f"\n  ⚡ Poisson {1/dp_ratio:.2f}x stronger than Sine at drive frequency")
        print(f"     Спайки эффективнее возбуждают резонанс, чем чистый тон.")
        print(f"     Широкополосный спектр спайков лучше матчит eigenmodes.")
    else:
        print(f"\n  ⚡ Sine vs Poisson comparable (ratio={dp_ratio:.2f}x)")
        print(f"     Оба типа входного сигнала дают схожий отклик на DN.")
        print(f"     Резонанс определяется структурой коннектома, а не формой сигнала.")

    # Check band fraction
    bf_sine = sine_result['band_fraction']
    bf_spike = spike_result['band_fraction']
    print(f"\n  Power in drive band:")
    print(f"    Sine:    {bf_sine*100:.2f}%")
    print(f"    Poisson: {bf_spike*100:.2f}%")

    if bf_sine > bf_spike * 1.5:
        print(f"  → Sine концентрирует энергию в узкой полосе вокруг drive freq")
    elif bf_spike > bf_sine * 1.5:
        print(f"  → Poisson концентрирует энергию в узкой полосе (но на nat freq)")
    else:
        print(f"  → Оба типа сигнала дают схожее распределение энергии по спектру")

    print("\n" + "═" * 72)


if __name__ == '__main__':
    main()
