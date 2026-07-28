#!/usr/bin/env python3
"""
ψ-Wave VNC Phases — Фазы и противофазы при резонансе
=======================================================

Загружает VNC подграф из FlyWire, запускает ψ-wave на резонансной
частоте (123.6 Hz), записывает фазу каждого нейрона на каждом шаге,
и ищет антифазные отношения между leg-сегментами (tripod gait).

Гипотеза: ψ-wave eigenmode на 123.6 Hz синхронизирует нейроны
в фазе/противофазе в соответствии с топологией коннектома.
Ноги одной стороны должны быть в фазе, противоположной стороны — 
в противофазе (трёхопорная походка).

Запуск:
    PYTHONPATH=target/debug:fly:. python3 examples/python/psi_vnc_phases.py
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

FREQ_HZ = 123.6     # resonant frequency
N_STEPS = 1024       # steps for phase analysis
DT = 1.0             # ms
HBAR = 1.0
INPUT_AMP = 30.0


def load_vnc_with_bodyparts():
    """Load VNC subgraph with body part annotations."""
    t0 = time.perf_counter()

    # Load neurons
    body_part_map = {}
    class_map = {}
    type_map = {}
    super_class_map = {}
    flow_map = {}

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

    # Load connections, find VNC neurons
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

    # Build full CSR first
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    for rid, cnt in pre_count.items():
        row_ptr[root_to_idx[rid] + 1] = cnt
    row_ptr = np.cumsum(row_ptr, dtype=np.int32)
    total_rows = int(row_ptr[-1])

    col_indices = np.zeros(total_rows, dtype=np.int32)
    values = np.zeros(total_rows, dtype=np.float32)
    pre_nt_type = {}
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
                if pre_idx not in pre_nt_type:
                    pre_nt_type[pre_idx] = -1.0 if nt in ('GABA', 'GLY') else 1.0

    # ─── Find VNC neurons ─────────────────────────────
    vnc_root_ids = set()
    dn_root_ids = set()
    for rid, sc in super_class_map.items():
        if sc == 'ventral_nerve_cord_intrinsic':
            vnc_root_ids.add(rid)
        elif sc == 'descending':
            dn_root_ids.add(rid)
    for rid, fl in flow_map.items():
        if fl == 'efferent' and rid in root_to_idx:
            vnc_root_ids.add(rid)
    for rid, bp in body_part_map.items():
        bp_l = bp.lower()
        if any(x in bp_l for x in ['abdomen', 'leg', 'cox', 'trochanter', 'wing', 'haltere']):
            if rid in root_to_idx:
                vnc_root_ids.add(rid)

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

    vnc_both = vnc_idx_set | dn_idx_set
    vnc_list = sorted(vnc_both)
    vnc_map = {old: new for new, old in enumerate(vnc_list)}
    n_vnc = len(vnc_list)

    # Extract VNC subgraph edges
    vnc_edges = []
    for pre_idx in vnc_list:
        s = int(row_ptr[pre_idx])
        e = int(row_ptr[pre_idx + 1])
        for j in range(s, e):
            post_idx = int(col_indices[j])
            if post_idx in vnc_both:
                nt_sgn = pre_nt_type.get(pre_idx, 1.0)
                vnc_edges.append((
                    vnc_map[pre_idx], vnc_map[post_idx],
                    float(values[j]) * nt_sgn,
                ))

    n_syn = len(vnc_edges)

    # Build VNC CSR
    vnc_rp = np.zeros(n_vnc + 1, dtype=np.int32)
    pre_cnt = defaultdict(int)
    for pre, _, _ in vnc_edges:
        pre_cnt[pre] += 1
    for pre, cnt in pre_cnt.items():
        vnc_rp[pre + 1] = cnt
    vnc_rp = np.cumsum(vnc_rp, dtype=np.int32)

    vnc_ci = np.zeros(n_syn, dtype=np.int32)
    vnc_vals = np.zeros(n_syn, dtype=np.float32)
    vnc_cur = np.zeros(n_vnc, dtype=np.int32)
    for pre, post, w in vnc_edges:
        pos = vnc_rp[pre] + vnc_cur[pre]
        vnc_ci[pos] = np.int32(post)
        vnc_vals[pos] = float(w)
        vnc_cur[pre] += 1

    # ─── Annotate VNC neurons with body parts ─────────
    # For each VNC index, find the root_id, look up body part
    vnc_root_ids_list = [all_ids[idx] for idx in vnc_list]
    vnc_body_part = ['unknown'] * n_vnc
    vnc_class = ['unknown'] * n_vnc
    vnc_type = ['unknown'] * n_vnc

    for i, rid in enumerate(vnc_root_ids_list):
        vnc_body_part[i] = body_part_map.get(rid, 'unknown')
        vnc_class[i] = class_map.get(rid, 'unknown')
        vnc_type[i] = type_map.get(rid, 'unknown')

    # Group by body part
    body_part_groups = defaultdict(list)
    for i, bp in enumerate(vnc_body_part):
        body_part_groups[bp].append(i)

    print(f"\n  Body part groups ({len(body_part_groups)} unique):")
    for bp, indices in sorted(body_part_groups.items(), key=lambda x: -len(x[1])):
        n_show = min(3, len(indices))
        examples = [vnc_type[indices[j]] for j in range(n_show)]
        print(f"    {bp:25s}: {len(indices):6d} neurons  e.g. {examples}")

    elapsed = time.perf_counter() - t0
    print(f"\n  VNC: {n_vnc:,} neurons, {n_syn:,} synapses [{elapsed:.1f}s]")

    return (vnc_rp, vnc_ci, vnc_vals, n_vnc,
            body_part_groups, vnc_body_part, vnc_class, vnc_type)


def run_psi_with_phases(row_ptr, col_indices, values, n,
                         body_part_groups=None,
                         input_amp=30.0):
    """Run ψ-wave at resonant frequency, record phases of all neurons."""
    n_padded = (n + 7) // 8 * 8
    rng = np.random.RandomState(42)
    phases = rng.uniform(0, 2 * math.pi, n_padded).astype(np.float32)
    v = np.cos(phases).astype(np.float32)
    u = np.sin(phases).astype(np.float32)
    h_real = np.zeros(n_padded, dtype=np.float32)
    h_imag = np.zeros(n_padded, dtype=np.float32)
    i_ext = np.zeros(n_padded, dtype=np.float32)
    speed = np.ones(n_padded, dtype=np.float32)

    omega = 2.0 * math.pi * FREQ_HZ

    # Build sensory set (exclude body-part-annotated neurons)
    if body_part_groups is not None:
        all_analyzed = set()
        for indices in body_part_groups.values():
            all_analyzed.update(indices)
    else:
        all_analyzed = set()

    n_sensory = max(1, n // 10)
    all_idx = np.arange(n, dtype=np.int32)
    rng.shuffle(all_idx)
    sensory_arr = np.array([i for i in all_idx[:n_sensory] if i not in all_analyzed],
                           dtype=np.int32)
    excluded = n_sensory - len(sensory_arr)
    print(f"  Sensory: {len(sensory_arr)} (excluded {excluded} analyzed neurons)")
    if len(sensory_arr) < 10:
        sensory_arr = all_idx[~np.isin(all_idx, list(all_analyzed))][:max(10, n_sensory // 2)].astype(np.int32)
        print(f"  Sensory fallback: {len(sensory_arr)}")

    # Record phase of every neuron at each step
    # phase = atan2(u, v) → in [-π, π]
    phase_history = np.zeros((N_STEPS, n), dtype=np.float32)

    print(f"\n[RUN] ψ-wave @ {FREQ_HZ:.1f} Hz, {N_STEPS} steps, {n:,} neurons...")
    t0 = time.perf_counter()

    for step in range(N_STEPS):
        t_sec = step * DT / 1000.0
        drive_val = input_amp * math.sin(omega * t_sec)

        i_ext.fill(0.0)
        i_ext[sensory_arr] = drive_val

        tcm.sparse.compute_hamiltonian(
            row_ptr, col_indices, values, v, u, h_real, h_imag,
        )
        tcm.step_psi_wave(v, u, h_real, h_imag, i_ext, speed, DT, HBAR)

        # Record phase
        phase_history[step] = np.arctan2(u[:n], v[:n])

        if step % 128 == 0:
            # Circular mean (proper for wrapped angles)
            circ_mean = float(np.arctan2(
                np.mean(np.sin(phase_history[step])),
                np.mean(np.cos(phase_history[step])),
            ))
            phase_coherence = float(np.abs(np.mean(np.exp(1j * phase_history[step]))))
            print(f"  Step {step:4d} | circ_mean={circ_mean:+.2f} | coherence={phase_coherence:.4f}")

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")

    # FFT of mean readout signal
    mean_signal = np.mean(np.cos(phase_history), axis=1)  # mean v (cos(phase))
    sig = mean_signal[-min(1024, N_STEPS):]
    window = np.hanning(len(sig))
    fft = np.fft.rfft((sig - np.mean(sig)) * window)
    freqs_fft = np.fft.rfftfreq(len(sig), d=DT / 1000.0)
    magnitude = np.abs(fft)

    # Find actual peak
    peak_idx = np.argmax(magnitude[1:]) + 1
    actual_peak = float(freqs_fft[peak_idx]) if peak_idx < len(freqs_fft) else 0.0
    print(f"  FFT peak: {actual_peak:.1f} Hz (expected: {FREQ_HZ:.1f} Hz)")

    return phase_history, actual_peak


def analyze_phases(phase_history, body_part_groups):
    """Analyze phase relationships between body part groups."""
    print("\n" + "═" * 72)
    print("  PHASE ANALYSIS")
    print("═" * 72)

    n_steps, n_neurons = phase_history.shape
    dt = DT / 1000.0

    # Use last 50% of steps (steady state)
    half = n_steps // 2
    steady = phase_history[half:]

    # ─── Per-neuron mean phase and coherence ───────────
    mean_phase = np.arctan2(
        np.mean(np.sin(steady), axis=0),
        np.mean(np.cos(steady), axis=0),
    )
    phase_coh = np.abs(np.mean(np.exp(1j * steady), axis=0))

    print(f"\n  Per-neuron phase coherence (last {n_steps - half} steps):")
    print(f"    Mean coherence: {np.mean(phase_coh):.4f}")
    print(f"    High coherence (>0.8): {np.sum(phase_coh > 0.8):,} / {n_neurons:,} neurons")

    # ─── Group-level analysis ──────────────────────────
    print(f"\n  {'Body Part':>25s} | {'N':>6s} | {'Mean Phase':>10s} | {'Coherence':>9s} | {'Phase Pattern'}")
    print(f"  {'-'*25} | {'-'*6} | {'-'*10} | {'-'*9} | {'-'*20}")

    group_phases = {}
    for bp, indices in sorted(body_part_groups.items(), key=lambda x: -len(x[1])):
        if len(indices) < 5:
            continue
        idx_arr = np.array(indices)
        idx_arr = idx_arr[idx_arr < n_neurons]
        if len(idx_arr) < 5:
            continue

        # Mean phase of this group
        g_mean_phase = np.arctan2(
            np.mean(np.sin(steady[:, idx_arr])),
            np.mean(np.cos(steady[:, idx_arr])),
        )
        g_coh = np.abs(np.mean(np.exp(1j * steady[:, idx_arr])))

        group_phases[bp] = g_mean_phase

        # Phase pattern: early/late relative to mean
        pattern = ''
        if g_coh > 0.5:
            rel_phase = g_mean_phase
            if abs(rel_phase) < np.pi / 4:
                pattern = 'in-phase'
            elif abs(abs(rel_phase) - np.pi) < np.pi / 4:
                pattern = 'ANTIPHASE ←'
            elif rel_phase > 0:
                pattern = 'leading'
            else:
                pattern = 'lagging'

        print(f"  {bp:>25s} | {len(indices):6d} | {g_mean_phase:>+10.3f} | {g_coh:>9.4f} | {pattern}")

    # ─── Pairwise phase differences ─────────────────────
    print(f"\n  ─── Phase difference matrix (radians) ───")
    bp_list = [bp for bp in sorted(group_phases.keys())
               if bp in body_part_groups and len(body_part_groups[bp]) >= 5]
    n_bp = len(bp_list)

    # Header
    print(f"  {'':>20s}", end='')
    for bp in bp_list:
        short = bp[:10]
        print(f" {short:>10s}", end='')
    print()

    for i, bp_i in enumerate(bp_list):
        short_i = bp_i[:20]
        print(f"  {short_i:>20s}", end='')
        for j, bp_j in enumerate(bp_list):
            diff = group_phases[bp_i] - group_phases[bp_j]
            # Normalize to [-π, π]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            marker = ' ★' if abs(abs(diff) - np.pi) < 0.3 else ''
            print(f" {diff:>+8.2f}{marker}", end='')
        print()

    # ─── Find antiphase pairs ──────────────────────────
    print(f"\n  ─── Antiphase pairs (|Δφ - π| < 0.5 rad) ───")
    antiphase_count = 0
    for i, bp_i in enumerate(bp_list):
        for j in range(i + 1, len(bp_list)):
            bp_j = bp_list[j]
            diff = (group_phases[bp_i] - group_phases[bp_j])
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            if abs(abs(diff) - np.pi) < 0.5:
                print(f"  ★ {bp_i:>25s} ↔ {bp_j:25s}  Δφ={diff:+.2f} rad "
                      f"({abs(diff)*180/np.pi:.0f}°)")
                antiphase_count += 1

    if antiphase_count == 0:
        print(f"  (none found — try lower frequency?)")

    # ─── Topology check: do legs show tripod pattern? ──
    print(f"\n  ─── Leg phase topology ───")
    leg_groups = {bp: group_phases[bp] for bp in bp_list
                  if 'leg' in bp.lower() or 'cox' in bp.lower()}

    if leg_groups:
        print(f"  Found {len(leg_groups)} leg-related groups:")
        for bp, ph in sorted(leg_groups.items(), key=lambda x: x[0]):
            print(f"    {bp:30s}: phase={ph:+.2f} rad ({ph*180/np.pi:.0f}°)")

        # Check for tripod pattern
        phases_list = list(leg_groups.values())
        if len(phases_list) >= 2:
            max_diff = max(abs(a - b) for a in phases_list for b in phases_list)
            print(f"  Max phase difference: {max_diff:.2f} rad ({max_diff*180/np.pi:.0f}°)")
            if max_diff > np.pi * 0.7:
                print(f"  ✅ Legs show significant phase separation "
                      f"(potential tripod gait)")
            else:
                print(f"  ⚠ Legs mostly in-phase — not tripod pattern")
    else:
        print(f"  No leg-specific groups found in VNC subgraph")

    return group_phases


def main():
    print("═" * 72)
    print("  ψ-Wave VNC Phases — Резонанс и антифаза")
    print("═" * 72)

    # ─── Load ──────────────────────────────────────────
    (vnc_rp, vnc_ci, vnc_vals, n_vnc,
     body_part_groups, vnc_body_part, vnc_class, vnc_type) = load_vnc_with_bodyparts()

    # ─── DRIVEN RUN ───────────────────────────────────
    print("\n" + "═" * 72)
    print("  DRIVEN RUN (INPUT_AMP=30.0 @ 123.6 Hz)")
    print("═" * 72)
    phase_history, actual_peak = run_psi_with_phases(
        vnc_rp, vnc_ci, vnc_vals, n_vnc,
        body_part_groups=body_part_groups, input_amp=30.0)
    group_phases = analyze_phases(phase_history, body_part_groups)

    # ─── UNDRIVEN CONTROL ──────────────────────────────
    print("\n" + "═" * 72)
    print("  UNDRIVEN CONTROL (INPUT_AMP=0.0)")
    print("═" * 72)
    phase_history_undriven, _ = run_psi_with_phases(
        vnc_rp, vnc_ci, vnc_vals, n_vnc,
        body_part_groups=body_part_groups, input_amp=0.0)
    group_phases_undriven = analyze_phases(phase_history_undriven, body_part_groups)

    # ─── Comparison ────────────────────────────────────
    print("\n" + "═" * 72)
    print("  COMPARISON: DRIVEN vs UNDRIVEN")
    print("═" * 72)
    print(f"\n  {'Body Part':>25s} | {'Driven':>8s} | {'Undriven':>9s} | {'Δφ':>7s}")
    print(f"  {'-'*25} | {'-'*8} | {'-'*9} | {'-'*7}")
    for bp in sorted(group_phases.keys()):
        if bp in group_phases_undriven:
            dp = group_phases[bp]
            up = group_phases_undriven[bp]
            diff = (dp - up + np.pi) % (2 * np.pi) - np.pi
            print(f"  {bp:>25s} | {dp:>+8.3f} | {up:>+9.3f} | {diff:>+7.2f}")

    # ─── Save both runs ────────────────────────────────
    save_path = '/tmp/psi_vnc_phases.npz'
    np.savez(save_path,
             driven_phase_history=phase_history,
             undriven_phase_history=phase_history_undriven,
             body_part_groups={k: np.array(v, dtype=np.int32)
                              for k, v in body_part_groups.items()},
             driven_group_phases=np.array([(k, float(v))
                                            for k, v in group_phases.items()],
                                           dtype=[('name', 'U40'), ('phase', 'f4')]),
             undriven_group_phases=np.array([(k, float(v))
                                              for k, v in group_phases_undriven.items()],
                                             dtype=[('name', 'U40'), ('phase', 'f4')]),
             actual_peak_hz=actual_peak)
    print(f"\n  Saved to: {save_path}")
    print("═" * 72)


if __name__ == '__main__':
    main()
