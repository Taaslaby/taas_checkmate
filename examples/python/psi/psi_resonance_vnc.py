#!/usr/bin/env python3
"""
ψ-Wave Resonance — VNC Subgraph vs Full FlyWire
=================================================

Извлекает VNC подграф из FlyWire коннектома (по Super Class ==
'ventral_nerve_cord_intrinsic'), запускает ψ-wave frequency sweep
(5-150 Hz) на VNC, и сравнивает eigenmode спектр с полным FlyWire.

Гипотеза: VNC имеет свой eigenmode spectrum, отличный от мозга.
VNC — специализированный nerve cord для локомоции, его топология
должна давать другие резонансные частоты.

Запуск:
    PYTHONPATH=target/debug:fly:. python3 examples/python/psi_resonance_vnc.py
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
FREQS = np.array([5.0, 18.2, 31.4, 44.5, 57.7, 70.9, 84.1, 97.3, 110.5, 123.6, 136.8, 150.0])
N_STEPS = 800
DT = 1.0       # ms
HBAR = 1.0
INPUT_AMP = 30.0


def load_flywire_with_vnc():
    """Load FlyWire, find VNC neurons, extract VNC subgraph + run ψ-wave."""
    t0 = time.perf_counter()

    print("[1/5] Loading neurons metadata...")
    class_map = {}
    type_map = {}
    super_class_map = {}
    flow_map = {}
    body_part_map = {}

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
            sc = row.get('Super Class', '').strip()
            if sc:
                super_class_map[rid] = sc
            fl = row.get('Flow', '').strip()
            if fl:
                flow_map[rid] = fl
            bp = row.get('Body Part', '').strip()
            if bp:
                body_part_map[rid] = bp

    print(f"  {len(class_map)} with Class, {len(super_class_map)} with Super Class")

    # ─── Load connections ──────────────────────────────
    print("[2/5] Loading connections...")
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

    # Build full CSR
    print("[3/5] Building full CSR...")
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    for rid, cnt in pre_count.items():
        row_ptr[root_to_idx[rid] + 1] = cnt
    row_ptr = np.cumsum(row_ptr, dtype=np.int32)

    col_indices = np.zeros(total_rows, dtype=np.int32)
    values = np.zeros(total_rows, dtype=np.float32)
    cursor = np.zeros(n, dtype=np.int32)
    pre_nt_type = {}

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
                w = -weight if nt in ('GABA', 'GLY') else weight
                values[pos] = float(w)
                cursor[pre_idx] += 1
                if pre_idx not in pre_nt_type:
                    pre_nt_type[pre_idx] = -1.0 if nt in ('GABA', 'GLY') else 1.0

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

    print(f"  Full CSR: {n:,} n, {len(values):,} syn [{time.perf_counter()-t0:.1f}s]")

    # ─── Find VNC neurons ─────────────────────────────
    print("[4/5] Finding VNC neurons...")
    sc_set = set(super_class_map.values())

    vnc_root_ids = set()
    dn_root_ids = set()
    for rid, sc in super_class_map.items():
        if sc == 'ventral_nerve_cord_intrinsic':
            vnc_root_ids.add(rid)
        elif sc == 'descending':
            dn_root_ids.add(rid)

    # Also check body parts
    for rid, bp in body_part_map.items():
        bp_l = bp.lower()
        if any(x in bp_l for x in ['abdomen', 'leg', 'cox', 'trochanter', 'wing', 'haltere']):
            if rid in root_to_idx:
                vnc_root_ids.add(rid)

    # Also check Flow == 'efferent' (motor output VNC)
    for rid, fl in flow_map.items():
        if fl == 'efferent' and rid in root_to_idx:
            vnc_root_ids.add(rid)

    # Convert to CSR indices
    vnc_idx_set = set()
    dn_idx_set = set()
    for rid in vnc_root_ids:
        idx = root_to_idx.get(rid)
        if idx is not None:
            vnc_idx_set.add(idx)
    for rid in dn_root_ids:
        idx = root_to_idx.get(rid)
        if idx is not None:
            dn_idx_set.add(idx)

    print(f"  VNC neurons: {len(vnc_idx_set):,}")
    print(f"  DN neurons:  {len(dn_idx_set):,}")
    print(f"  Total:       {len(vnc_idx_set | dn_idx_set):,}")

    # ─── Extract VNC subgraph ─────────────────────────
    print("  Extracting VNC subgraph...")
    vnc_both = vnc_idx_set | dn_idx_set  # include descending inputs to VNC
    vnc_list = sorted(vnc_both)
    vnc_map = {old: new for new, old in enumerate(vnc_list)}
    n_vnc = len(vnc_list)

    # Count VNC-internal edges
    vnc_edges = []
    for pre_idx in vnc_list:
        s = int(row_ptr[pre_idx])
        e = int(row_ptr[pre_idx + 1])
        for j in range(s, e):
            post_idx = int(col_indices[j])
            if post_idx in vnc_both:
                vnc_edges.append((
                    vnc_map[pre_idx],
                    vnc_map[post_idx],
                    float(values[j]),
                    pre_nt_type.get(pre_idx, 1.0),
                ))

    n_syn_vnc = len(vnc_edges)
    print(f"  VNC subgraph: {n_vnc:,} neurons, {n_syn_vnc:,} synapses")

    # Build VNC CSR
    vnc_row_ptr = np.zeros(n_vnc + 1, dtype=np.int32)
    pre_vnc_counts = defaultdict(int)
    for pre, post, w, nt in vnc_edges:
        pre_vnc_counts[pre] += 1
    for pre, cnt in pre_vnc_counts.items():
        vnc_row_ptr[pre + 1] = cnt
    vnc_row_ptr = np.cumsum(vnc_row_ptr, dtype=np.int32)

    vnc_col_indices = np.zeros(n_syn_vnc, dtype=np.int32)
    vnc_values = np.zeros(n_syn_vnc, dtype=np.float32)
    vnc_cursor = np.zeros(n_vnc, dtype=np.int32)

    for pre, post, w, nt in vnc_edges:
        pos = vnc_row_ptr[pre] + vnc_cursor[pre]
        vnc_col_indices[pos] = np.int32(post)
        # Apply NT type sign
        vnc_values[pos] = w * nt
        vnc_cursor[pre] += 1

    # Find VNC-internal DN and MN indices
    vnc_dn = []
    vnc_mn = []
    for new_idx, old_idx in enumerate(vnc_list):
        rid = all_ids[old_idx]
        cl = class_map.get(rid, '')
        sc = super_class_map.get(rid, '')
        if cl.startswith('descending') or sc == 'descending':
            vnc_dn.append(new_idx)
        pct = type_map.get(rid, '')
        if cl == 'motor' or 'motor' in pct.lower():
            vnc_mn.append(new_idx)

    print(f"  VNC DN: {len(vnc_dn)}, VNC MN: {len(vnc_mn)}")
    print(f"  Total time: {time.perf_counter()-t0:.1f}s")

    return (vnc_row_ptr, vnc_col_indices, vnc_values, n_vnc, n_syn_vnc,
            vnc_dn, vnc_mn)


def run_psi_vnc(row_ptr, col_indices, values, n,
                readout_idx, sensory_idx, freq_hz):
    """Run ψ-wave on VNC subgraph."""
    n_readout = len(readout_idx)
    if n_readout == 0:
        return None

    n_padded = (n + 7) // 8 * 8
    rng = np.random.RandomState(int(freq_hz * 100) % (2**31))
    phases = rng.uniform(0, 2 * math.pi, n_padded).astype(np.float32)
    v = np.cos(phases).astype(np.float32)
    u = np.sin(phases).astype(np.float32)
    h_real = np.zeros(n_padded, dtype=np.float32)
    h_imag = np.zeros(n_padded, dtype=np.float32)
    i_ext = np.zeros(n_padded, dtype=np.float32)
    speed = np.ones(n_padded, dtype=np.float32)

    omega = 2.0 * math.pi * freq_hz
    sensory_arr = np.array([i for i in sensory_idx if i < n], dtype=np.int32)
    readout_arr = np.array([i for i in readout_idx if i < n], dtype=np.int32)
    k_r = min(5000, len(readout_arr))

    signal = np.zeros(N_STEPS, dtype=np.float32)
    rtmp = np.zeros(k_r, dtype=np.float32)

    for step in range(N_STEPS):
        t_sec = step * DT / 1000.0
        i_ext.fill(0.0)
        if len(sensory_arr) > 0:
            i_ext[sensory_arr] = INPUT_AMP * math.sin(omega * t_sec)

        tcm.sparse.compute_hamiltonian(
            row_ptr, col_indices, values, v, u, h_real, h_imag,
        )
        tcm.step_psi_wave(v, u, h_real, h_imag, i_ext, speed, DT, HBAR)

        rtmp[:k_r] = v[readout_arr[:k_r]]
        signal[step] = float(np.mean(rtmp[:k_r]))

    # FFT
    sig = signal[-min(1024, N_STEPS):]
    window = np.hanning(len(sig))
    fft = np.fft.rfft((sig - np.mean(sig)) * window)
    freqs_fft = np.fft.rfftfreq(len(sig), d=DT / 1000.0)
    magnitude = np.abs(fft)

    peak_idx = np.argmax(magnitude[1:]) + 1
    spectral_peak = float(freqs_fft[peak_idx]) if peak_idx < len(freqs_fft) else 0.0
    spectral_power = float(magnitude[peak_idx])

    drive_bin = np.argmin(np.abs(freqs_fft - freq_hz))
    drive_power = float(magnitude[drive_bin]) if drive_bin < len(fft) else 0.0
    drive_ratio = drive_power / max(spectral_power, 1e-12)

    return {
        'drive_power': drive_power,
        'drive_ratio': drive_ratio,
        'spectral_peak_hz': spectral_peak,
    }


def main():
    print("═" * 72)
    print("  ψ-Wave Resonance — VNC Subgraph vs Full FlyWire")
    print("═" * 72)

    row_ptr, col_indices, values, n_vnc, n_syn, vnc_dn, vnc_mn = \
        load_flywire_with_vnc()

    # ─── Build sensory set ─────────────────────────────
    print("\n[5/5] Running ψ-wave frequency sweep on VNC...")

    # For VNC subgraph: use random 10% of VNC neurons as sensory
    n_sensory = max(1, n_vnc // 10)
    rng = np.random.RandomState(42)
    all_idx = np.arange(n_vnc, dtype=np.int32)
    rng.shuffle(all_idx)
    sensory = [int(i) for i in all_idx[:n_sensory] if i not in vnc_dn and i not in vnc_mn]

    groups = {}
    if vnc_dn:
        groups['VNC_DN'] = vnc_dn
    if vnc_mn:
        groups['VNC_MN'] = vnc_mn

    all_readout = list(set(vnc_dn) | set(vnc_mn))
    if all_readout:
        groups['VNC_DN+MN'] = all_readout

    results = {}
    for gname, readout_idx in groups.items():
        if len(readout_idx) == 0:
            continue
        print(f"\n  ─── {gname} ({len(readout_idx)} neurons) ───")
        gr = []
        for freq in FREQS:
            r = run_psi_vnc(row_ptr, col_indices, values, n_vnc,
                            readout_idx, sensory, freq)
            if r is None:
                continue
            gr.append(r)
            print(f"    {freq:5.1f}Hz | drive_pwr={r['drive_power']:.4f} "
                  f"D/P={r['drive_ratio']:.3f} "
                  f"nat={r['spectral_peak_hz']:5.1f}Hz")
        results[gname] = gr

    # ─── Results ───────────────────────────────────────
    print("\n" + "═" * 72)
    print("  RESULTS — VNC Subgraph")
    print("═" * 72)

    for gname, gr in results.items():
        dp = np.array([r['drive_power'] for r in gr])
        dr = np.array([r['drive_ratio'] for r in gr])
        pk = np.argmax(dp)
        q = dp.max() / max(dp.mean(), 1e-12)

        print(f"\n  ─── {gname} ───")
        print(f"  Neurons: {len(groups[gname]):,}")
        print(f"  Max drive_power: {dp.max():.4f} @ {FREQS[pk]:.1f} Hz")
        print(f"  Q-factor: {q:.2f}x")
        print(f"  Mean D/P: {dr.mean():.3f}")

        # Bar chart
        col_w = 20
        dp_max = max(dp.max(), 1e-12)
        print(f"\n  {'Hz':>5s} : {'DrvPwr':>7s}  Bar")
        for i, f in enumerate(FREQS):
            bar = '█' * int(dp[i] / dp_max * col_w) + \
                  '░' * max(0, col_w - int(dp[i] / dp_max * col_w))
            marker = ' ← max' if i == pk else ''
            print(f"  {f:5.1f} : {dp[i]:7.4f}  |{bar}|{marker}")

    # ─── Comparison with FlyWire ───────────────────────
    print("\n" + "═" * 72)
    print("  COMPARISON: VNC vs Full FlyWire")
    print("═" * 72)

    # Load FlyWire results
    fw_path = '/tmp/psi_resonance_flywire.npz'
    if os.path.exists(fw_path):
        fw = np.load(fw_path)
        fw_freqs = fw['freqs_hz']
        fw_dn = fw['DN_drive_powers']

        # Match frequencies
        print(f"\n  {'Freq':>6s} | {'VNC_DN':>8s} | {'FlyWire_DN':>10s} | {'Ratio':>6s}")
        print(f"  {'-'*6} | {'-'*8} | {'-'*10} | {'-'*6}")

        vnc_dp = np.array([r['drive_power'] for r in results.get('VNC_DN', [])])

        for i, f in enumerate(FREQS):
            fw_idx = np.argmin(np.abs(fw_freqs - f))
            if fw_idx < len(fw_dn) and i < len(vnc_dp):
                ratio = vnc_dp[i] / max(fw_dn[fw_idx], 1e-12)
                print(f"  {f:6.1f} | {vnc_dp[i]:8.4f} | {fw_dn[fw_idx]:10.4f} | {ratio:6.2f}x")
    else:
        print("\n  ⚠ FlyWire NPZ not found — can't compare")

    print("\n" + "═" * 72)
    print("  SAVING RESULTS")
    print("═" * 72)
    save_path = '/tmp/psi_resonance_vnc.npz'
    save_data = {'freqs_hz': FREQS, 'n_vnc': n_vnc, 'n_syn_vnc': n_syn}
    for gname, gr in results.items():
        save_data[f'{gname}_drive_powers'] = np.array([r['drive_power'] for r in gr])
        save_data[f'{gname}_drive_ratios'] = np.array([r['drive_ratio'] for r in gr])
    np.savez(save_path, **save_data)
    print(f"  Saved to: {save_path}")
    print("═" * 72)


if __name__ == '__main__':
    main()
