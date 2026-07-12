#!/usr/bin/env python3
"""
ψ-Wave Lake — FlyWire Real Connectome
======================================

Загружает реальный FlyWire коннектом (158K нейронов, 4M синапсов),
строит CSR матрицу, находит DN (descending) и MN (motor) нейроны
по колонке Class из neurons.csv.gz, запускает ψ-wave frequency sweep.

Гипотеза: коннектом избирательно усиливает входные частоты,
соответствующие собственным модам графа → DN/MN резонируют.

Запуск:
    cd taas_checkmate
    PYTHONPATH=target/debug:fly:. python3 examples/python/psi_resonance_flywire.py
"""

import sys
import os
import math
import time
import csv
import gzip
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in [os.path.join(PROJ, 'target', 'release'),
           os.path.join(PROJ, 'target', 'debug')]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import taas_checkmate as tcm

# ═══════════════════════════════════════════════════════════
#  Data paths
# ═══════════════════════════════════════════════════════════

DATA_DIR = '/home/taaslab/Проекты/asm_engines/fly/archive'
CONN_PATH = os.path.join(DATA_DIR, 'connections_princeton.csv.gz')
NEURONS_PATH = os.path.join(DATA_DIR, 'neurons.csv.gz')


# ═══════════════════════════════════════════════════════════
#  1. Load FlyWire connectome
# ═══════════════════════════════════════════════════════════

def load_flywire():
    """Load FlyWire connectome and identify DN/MN neurons.

    Returns:
        row_ptr, col_indices, values: CSR matrix
        dn_indices: CSR indices of descending neurons
        mn_indices: CSR indices of motor neurons
        visual_indices: CSR indices of visual afferents
        n_neurons: total neuron count
        n_synapses: total synapse count
    """
    t0 = time.perf_counter()

    # ─── Load neurons metadata ────────────────────────
    print("  [LOAD] Loading neurons metadata...")
    class_map = {}        # root_id → Class
    type_map = {}         # root_id → Primary Cell Type
    flow_map = {}         # root_id → Flow
    super_class_map = {}  # root_id → Super Class

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

            fl = row.get('Flow', '').strip()
            if fl:
                flow_map[rid] = fl

            sc = row.get('Super Class', '').strip()
            if sc:
                super_class_map[rid] = sc

    print(f"    {len(class_map)} with Class, {len(type_map)} with Primary Cell Type")

    # ─── Load connections, first pass: count ──────────
    print("  [LOAD] Loading connections (first pass)...")
    pre_count = defaultdict(int)
    total_rows = 0

    with gzip.open(CONN_PATH, 'rt') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 3:
                pre = int(row[0])
                pre_count[pre] += 1
                total_rows += 1

    # Build neuron index
    all_ids = sorted(set(pre_count.keys()))
    root_to_idx = {rid: i for i, rid in enumerate(all_ids)}
    n = len(all_ids)
    print(f"    {n:,} neurons, {total_rows:,} synapses [{time.perf_counter()-t0:.1f}s]")

    # ─── Second pass: build CSR ───────────────────────
    print("  [LOAD] Building CSR matrix (second pass)...")
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    for rid, cnt in pre_count.items():
        idx = root_to_idx[rid]
        row_ptr[idx + 1] = cnt
    row_ptr = np.cumsum(row_ptr, dtype=np.int32)

    col_indices = np.zeros(total_rows, dtype=np.int32)
    values = np.zeros(total_rows, dtype=np.float32)
    nt_types = np.ones(n, dtype=np.float32)
    cursor = np.zeros(n, dtype=np.int32)

    with gzip.open(CONN_PATH, 'rt') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 3:
                pre = int(row[0])
                post = int(row[1])
                weight = float(row[3])  # syn_count column
                pre_idx = root_to_idx.get(pre)
                post_idx = root_to_idx.get(post)
                if pre_idx is None or post_idx is None:
                    continue
                pos = row_ptr[pre_idx] + cursor[pre_idx]
                if pos >= total_rows:
                    continue
                col_indices[pos] = np.int32(post_idx)
                # Use negative weight for NT type modulation
                nt = row[4].strip().upper() if len(row) > 4 else ''
                if nt == 'GABA' or nt == 'GLY':
                    values[pos] = -float(weight)
                else:
                    values[pos] = float(weight)
                cursor[pre_idx] += 1

    # Trim to actual synapses
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

    print(f"    CSR built: {n:,} n, {actual_n:,} syn [{time.perf_counter()-t0:.1f}s]")

    # ─── Find DN (descending) neurons ─────────────────
    print("  [LOAD] Finding descending neurons (DNs)...")
    dn_indices = []
    dn_root_ids = []
    for rid, idx in root_to_idx.items():
        cl = class_map.get(rid, '')
        # DN: Class starts with 'descending'
        if cl.startswith('descending') or cl == 'descending':
            dn_indices.append(idx)
            dn_root_ids.append(rid)
        # Also check Primary Cell Type starting with DN patterns
        pct = type_map.get(rid, '')
        if pct.startswith(('DN', 'MDN', 'APDN', 'tdDN')) and rid not in dn_root_ids:
            dn_indices.append(idx)
            dn_root_ids.append(rid)

    print(f"    {len(dn_indices)} DNs [{time.perf_counter()-t0:.1f}s]")

    # ─── Find MN (motor) neurons ──────────────────────
    print("  [LOAD] Finding motor neurons (MNs)...")
    mn_indices = []
    mn_root_ids = []
    for rid, idx in root_to_idx.items():
        cl = class_map.get(rid, '')
        # MN: Class == 'motor' or Primary Cell Type contains 'motor'
        if cl == 'motor' or cl == 'leg_motor':
            mn_indices.append(idx)
            mn_root_ids.append(rid)
            continue
        pct = type_map.get(rid, '')
        if 'motor' in pct.lower():
            mn_indices.append(idx)
            mn_root_ids.append(rid)

    print(f"    {len(mn_indices)} MNs [{time.perf_counter()-t0:.1f}s]")

    # ─── Find visual afferents ────────────────────────
    print("  [LOAD] Finding visual afferents...")
    VISUAL_PATTERNS = ('Tm', 'TmY', 'Mi', 'Mi1', 'L1', 'L2', 'L3', 'L4', 'L5',
                       'T4', 'T5', 'LC', 'LPTC', 'lamina', 'medulla', 'lobula')
    visual_indices = []
    for rid, idx in root_to_idx.items():
        pct = type_map.get(rid, '')
        for pat in VISUAL_PATTERNS:
            if pct.startswith(pat) or pat in pct:
                visual_indices.append(idx)
                break

    print(f"    {len(visual_indices)} visual afferents [{time.perf_counter()-t0:.1f}s]")
    print(f"\n  ✅ Total: {n:,} neurons, {actual_n:,} synapses, "
          f"{len(dn_indices)} DN, {len(mn_indices)} MN "
          f"[{time.perf_counter()-t0:.1f}s]")

    return (row_ptr, col_indices, values,
            dn_indices, mn_indices, visual_indices,
            n, actual_n,
            dn_root_ids, mn_root_ids, root_to_idx, type_map, class_map)


# ═══════════════════════════════════════════════════════════
#  2. ψ-wave frequency sweep
# ═══════════════════════════════════════════════════════════

def run_psi_flywire(row_ptr, col_indices, values,
                    readout_idx, sensory_idx,
                    freq_hz: float, n_steps: int = 800,
                    dt: float = 1.0, hbar: float = 1.0,
                    input_amp: float = 30.0):
    """Run ψ-wave on FlyWire, measure drive power at readout."""
    n = len(row_ptr) - 1
    n_readout = len(readout_idx)
    if n_readout == 0:
        return None

    n_padded = (n + 7) // 8 * 8

    # Random initial phases (seed = freq for variation)
    rng = np.random.RandomState(int(freq_hz * 100) % (2**31))
    phases = rng.uniform(0, 2 * math.pi, n_padded).astype(np.float32)
    v = np.cos(phases).astype(np.float32)
    u = np.sin(phases).astype(np.float32)

    h_real = np.zeros(n_padded, dtype=np.float32)
    h_imag = np.zeros(n_padded, dtype=np.float32)
    i_ext = np.zeros(n_padded, dtype=np.float32)
    speed = np.ones(n_padded, dtype=np.float32)

    omega = 2.0 * math.pi * freq_hz

    # Pre-compute numpy arrays for vectorized ops
    sensory_arr = np.array([i for i in sensory_idx if i < n], dtype=np.int32)
    readout_arr = np.array([i for i in readout_idx if i < n], dtype=np.int32)
    actual_readout = len(readout_arr)

    readout_signal = np.zeros(n_steps, dtype=np.float32)
    readout_v = np.zeros(min(5000, actual_readout), dtype=np.float32)
    readout_u = np.zeros(min(5000, actual_readout), dtype=np.float32)

    t_start = time.perf_counter()

    for step in range(n_steps):
        t_sec = step * dt / 1000.0
        drive_val = input_amp * math.sin(omega * t_sec)

        i_ext.fill(0.0)
        if len(sensory_arr) > 0:
            i_ext[sensory_arr] = drive_val

        tcm.sparse.compute_hamiltonian(
            row_ptr, col_indices, values, v, u, h_real, h_imag,
        )
        tcm.step_psi_wave(v, u, h_real, h_imag, i_ext, speed, dt, hbar)

        # Readout: mean ψ (sample up to 5000 readout neurons)
        k = min(5000, actual_readout)
        readout_v[:k] = v[readout_arr[:k]]
        readout_u[:k] = u[readout_arr[:k]]
        readout_signal[step] = float(np.mean(readout_v[:k]))

    elapsed = time.perf_counter() - t_start

    # FFT of readout signal
    signal = readout_signal[-min(1024, n_steps):]
    if len(signal) > 10:
        # Hann window
        window = np.hanning(len(signal))
        fft = np.fft.rfft((signal - np.mean(signal)) * window)
        freqs_fft = np.fft.rfftfreq(len(signal), d=dt / 1000.0)
        # Natural peak (exclude DC)
        peak_idx = np.argmax(np.abs(fft[1:])) + 1
        spectral_peak = float(freqs_fft[peak_idx]) if peak_idx < len(freqs_fft) else 0.0
        spectral_power = float(np.abs(fft[peak_idx]))
        # Drive power at exact drive frequency
        drive_bin = np.argmin(np.abs(freqs_fft - freq_hz))
        drive_power = float(np.abs(fft[drive_bin])) if drive_bin < len(fft) else 0.0
        drive_ratio = drive_power / max(spectral_power, 1e-12)
    else:
        spectral_peak = 0.0
        spectral_power = 0.0
        drive_power = 0.0
        drive_ratio = 0.0

    return {
        'drive_power': drive_power,
        'drive_ratio': drive_ratio,
        'spectral_peak_hz': spectral_peak,
        'spectral_power': spectral_power,
        'elapsed': elapsed,
    }


# ═══════════════════════════════════════════════════════════
#  3. Main
# ═══════════════════════════════════════════════════════════

def main():
    print("═" * 72)
    print("  ψ-Wave Lake — FlyWire Real Connectome")
    print("  Resonance on DN (descending) and MN (motor) neurons")
    print("═" * 72)

    # ─── Config ────────────────────────────────────────
    N_STEPS = 800      # 0.8 sec per freq (enough for 5-100 Hz)
    DT = 1.0           # ms
    HBAR = 1.0
    INPUT_AMP = 30.0   # strong drive for 158K neurons
    FREQS = np.linspace(5.0, 150.0, 12)  # 5-150 Hz, ~13 Hz steps

    # ─── Load FlyWire ──────────────────────────────────
    print("\n[1/3] Loading FlyWire connectome...")
    (row_ptr, col_indices, values,
     dn_indices, mn_indices, visual_indices,
     n, n_syn, dn_root_ids, mn_root_ids,
     root_to_idx, type_map, class_map) = load_flywire()

    # ─── Sweep each group ──────────────────────────────
    print("\n[2/3] Frequency sweep...")
    groups = {
        'DN': dn_indices,
        'MN': mn_indices,
    }
    # Also add combined DN+MN group
    combined = list(set(dn_indices) | set(mn_indices))
    if len(combined) > 0:
        groups['DN+MN'] = combined

    # Sensory: visual afferents + random 3% of all neurons (exclude readout)
    all_results = {}
    for group_name, readout_idx in groups.items():
        if len(readout_idx) == 0:
            print(f"\n  ⏭  {group_name}: empty, skip")
            continue

        print(f"\n  ─── {group_name} ({len(readout_idx)} neurons) ───")

        # Build sensory set (visual + random, excluding readout neurons)
        readout_set = set(readout_idx)
        sensory = list(visual_indices[:min(500, len(visual_indices))])
        extra_n = max(1, n // 33)  # ~3% of all neurons
        rng2 = np.random.RandomState(42)
        extra = rng2.choice(n, extra_n, replace=False).tolist()
        sensory = list(set(sensory) | set(extra))
        sensory = [i for i in sensory if i not in readout_set]
        print(f"    Sensory: {len(sensory)} neurons")

        group_res = []
        for i, f in enumerate(FREQS):
            t0 = time.perf_counter()
            r = run_psi_flywire(row_ptr, col_indices, values,
                                readout_idx, sensory,
                                f, N_STEPS, DT, HBAR, INPUT_AMP)
            elapsed = time.perf_counter() - t0
            if r is None:
                continue
            group_res.append(r)
            print(f"    {f:5.1f}Hz | drive_pwr={r['drive_power']:.4f} "
                  f"D/P={r['drive_ratio']:.3f} "
                  f"nat={r['spectral_peak_hz']:5.1f}Hz "
                  f"[{elapsed*1000:.0f}ms]")

        all_results[group_name] = group_res

    # ─── Results ───────────────────────────────────────
    print("\n" + "═" * 72)
    print("  RESULTS")
    print("═" * 72)

    for group_name, gr in all_results.items():
        drive_powers = np.array([r['drive_power'] for r in gr])
        drive_ratios = np.array([r['drive_ratio'] for r in gr])

        dp_peak = np.argmax(drive_powers)
        q = drive_powers.max() / max(drive_powers.mean(), 1e-12)

        print(f"\n  ─── {group_name} ───")
        print(f"  Max drive_power:  {drive_powers.max():.4f} @ {FREQS[dp_peak]:.1f} Hz")
        print(f"  Q-factor:         {q:.2f}x")
        print(f"  Mean D/P ratio:   {drive_ratios.mean():.3f}")

        # Find local peaks
        is_peak = np.zeros(len(drive_powers), dtype=bool)
        for i in range(2, len(drive_powers) - 2):
            if (drive_powers[i] > drive_powers[i-1] and
                drive_powers[i] > drive_powers[i-2] and
                drive_powers[i] > drive_powers[i+1] and
                drive_powers[i] > drive_powers[i+2] and
                drive_powers[i] > drive_powers.mean()):
                is_peak[i] = True

        n_peaks = is_peak.sum()
        if n_peaks >= 1:
            print(f"  ✅ {n_peaks} resonant mode(s):")
            for pk in np.where(is_peak)[0]:
                print(f"     {FREQS[pk]:.1f} Hz — "
                      f"drive_pwr={drive_powers[pk]:.4f}, "
                      f"D/P={drive_ratios[pk]:.3f}")

        # Bar chart
        col_w = 20
        dp_max = max(drive_powers.max(), 1e-12)
        print(f"\n  {'Hz':>5s} : {'DrvPwr':>7s} {'Bar':>{col_w+4}s}")
        print(f"  {'-'*5} : {'-'*7} {'-'*(col_w+4)}")
        for i, f in enumerate(FREQS):
            bar = '█' * int(drive_powers[i] / dp_max * col_w) + \
                  '░' * max(0, col_w - int(drive_powers[i] / dp_max * col_w))
            marker = ' ← max' if i == dp_peak else ''
            print(f"  {f:5.1f} : {drive_powers[i]:7.4f}  |{bar}|{marker}")

    # ─── Verdict ───────────────────────────────────────
    print("\n" + "═" * 72)
    print("  VERDICT")
    print("═" * 72)

    all_dp = []
    for gr in all_results.values():
        all_dp.extend([r['drive_power'] for r in gr])
    all_dp = np.array(all_dp)
    global_q = all_dp.max() / max(all_dp.mean(), 1e-12)

    print(f"\n  Connectome: {n:,} neurons, {n_syn:,} synapses")
    print(f"  DN: {len(dn_indices)}, MN: {len(mn_indices)}")
    print(f"  Global Q-factor: {global_q:.2f}x")

    if global_q > 1.3:
        print(f"\n  ✅ **РЕАЛЬНЫЙ FLYWIRE КОННЕКТОМ = РЕЗОНАНСНОЕ ОЗЕРО**")
        print(f"     FlyWire (158K нейронов, 4M синапсов) избирательно")
        print(f"     усиливает входные частоты на Q={global_q:.2f}x.")
        print(f"     DN/MN нейроны имеют измеримый резонансный отклик.")
        print(f"     Вычисление = физика волн на коннектоме.")
    elif global_q > 1.1:
        print(f"\n  📡 Слабый резонанс (Q={global_q:.2f}x)")
        print(f"     Измеримый, но слабый. Попробуйте:")
        print(f"     - Увеличить N_STEPS или input_amp")
        print(f"     - Более узкий диапазон частот")
    else:
        print(f"\n  ℹ️  Резонанс не обнаружен (Q={global_q:.2f}x)")
        print(f"     Причины: ψ-wave coupling подавляет drive,")
        print(f"     или нужна другая модель (HH/Izhikevich)")

    print("═" * 72)

    # ─── Save ──────────────────────────────────────────
    save_data = {'freqs_hz': FREQS}
    for gname, gr in all_results.items():
        save_data[f'{gname}_drive_powers'] = np.array(
            [r['drive_power'] for r in gr])
        save_data[f'{gname}_drive_ratios'] = np.array(
            [r['drive_ratio'] for r in gr])
    save_path = '/tmp/psi_resonance_flywire.npz'
    np.savez(save_path, **save_data)
    print(f"\n  Saved to: {save_path}")


if __name__ == "__main__":
    main()
