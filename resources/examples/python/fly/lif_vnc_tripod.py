#!/usr/bin/env python3
"""
LIF VNC Tripod — LIF симуляция на VNC подграфе
================================================

Загружает VNC подграф из FlyWire, запускает LIF динамику через Rust
(lif_connectome), записывает спайки EN (efferent) нейронов по ногам,
ищет tripod gait pattern: front↔middle antiphase.

Гипотеза: LIF динамика + реципрокное торможение VNC дают
антифазные отношения между ногами (tripod), которых не было
в ψ-wave (где все ноги были в фазе).

Запуск:
    PYTHONPATH=target/release:target/debug:fly:. python3 examples/python/lif_vnc_tripod.py
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

DT_MS = 0.5           # LIF timestep (ms)
N_BATCHES = 40         # 40 batches × 25ms = 1 second
STEPS_PER_BATCH = 50   # 50 steps at 0.5ms = 25ms per batch
TOTAL_MS = N_BATCHES * STEPS_PER_BATCH * DT_MS  # 1000ms


def load_vnc_for_lif():
    """Load VNC subgraph with leg body part annotations for LIF."""
    t0 = time.perf_counter()

    # ─── Load neurons ────────────────────────────────
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
            for k, v in [('Class', class_map), ('Primary Cell Type', type_map),
                         ('Super Class', super_class_map), ('Flow', flow_map),
                         ('Body Part', body_part_map)]:
                val = row.get(k, '').strip()
                if val:
                    v[rid] = val

    # ─── Build full CSR ──────────────────────────────
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

    # ─── Find VNC neurons ────────────────────────────
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
    n_padded = (n_vnc + 7) // 8 * 8

    # ─── Extract VNC subgraph edges ──────────────────
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

    # ─── Annotate VNC neurons with body parts ────────
    vnc_root_ids_list = [all_ids[idx] for idx in vnc_list]
    vnc_body_part = ['unknown'] * n_vnc
    vnc_flow = ['unknown'] * n_vnc
    vnc_type = ['unknown'] * n_vnc

    for i, rid in enumerate(vnc_root_ids_list):
        vnc_body_part[i] = body_part_map.get(rid, 'unknown')
        vnc_flow[i] = flow_map.get(rid, 'unknown')
        vnc_type[i] = type_map.get(rid, 'unknown')

    # ─── Find EN (efferent) indices per leg ──────────
    # IMPORTANT: check specific leg FIRST, then coxa as fallback
    def classify_leg(bp_lower):
        if 'front_leg' in bp_lower:
            return 'front_leg'
        if 'middle_leg' in bp_lower:
            return 'middle_leg'
        if 'hind_leg' in bp_lower:
            return 'hind_leg'
        if 'front' in bp_lower or 'cox' in bp_lower:
            return 'front_leg'
        if 'middle' in bp_lower:
            return 'middle_leg'
        if 'hind' in bp_lower:
            return 'hind_leg'
        if 'abdomen' in bp_lower:
            return 'abdomen'
        if 'wing' in bp_lower or 'haltere' in bp_lower:
            return 'wing'
        return None

    en_by_leg = defaultdict(list)
    for i in range(n_vnc):
        bp = vnc_body_part[i].lower()
        fl = vnc_flow[i].lower()
        if fl == 'efferent':
            leg = classify_leg(bp)
            if leg:
                en_by_leg[leg].append(i)

    # Also try Primary Cell Type containing motor keywords
    for i in range(n_vnc):
        typ = vnc_type[i].lower()
        bp = vnc_body_part[i].lower()
        if any(x in typ for x in ['motor', 'mn', 'efferent', 'en']):
            leg = classify_leg(bp)
            if leg:
                if i not in en_by_leg[leg]:
                    en_by_leg[leg].append(i)

    print(f"\n  EN (efferent) by leg:")
    for leg, idxs in sorted(en_by_leg.items()):
        print(f"    {leg:15s}: {len(idxs):4d} neurons")

    elapsed = time.perf_counter() - t0
    print(f"\n  VNC: {n_vnc:,} neurons, {n_syn:,} synapses [{elapsed:.1f}s]")

    return (vnc_rp, vnc_ci, vnc_vals, n_vnc, n_padded,
            en_by_leg, vnc_body_part, vnc_type)


def run_lif_simulation(vnc_rp, vnc_ci, vnc_vals, n_vnc, n_padded,
                       en_by_leg):
    """Run LIF on VNC subgraph, record EN spikes."""
    print("\n" + "═" * 72)
    print(f"  LIF Simulation: {N_BATCHES} batches × {STEPS_PER_BATCH} steps "
          f"= {TOTAL_MS:.0f}ms")
    print("═" * 72)

    # ─── LIF params (Shiu et al. 2024) ────────────────
    tau_m = np.full(n_padded, 20.0, dtype=np.float32)
    v_rest = np.full(n_padded, -52.0, dtype=np.float32)
    v_thresh = np.full(n_padded, -45.0, dtype=np.float32)
    v_reset = np.full(n_padded, -52.0, dtype=np.float32)
    nt_types = np.ones(n_padded, dtype=np.float32)

    rng = np.random.RandomState(42)
    v = np.full(n_padded, -52.0, dtype=np.float32)
    v[:n_vnc] += rng.uniform(-1.0, 1.0, n_vnc).astype(np.float32)
    I_buf = np.zeros(n_padded, dtype=np.float32)
    spikes = np.zeros(n_padded, dtype=np.int32)
    external_I = np.zeros(n_padded, dtype=np.float32)

    # Alpha-synapse
    tau_syn = 5.0
    g_syn = np.zeros(n_padded, dtype=np.float32)
    delay_steps = 4
    delay_buf = np.zeros(n_padded * delay_steps, dtype=np.float32)
    delay_pos = np.zeros(1, dtype=np.int32)

    # ─── Spike buffer ─────────────────────────────────
    total_steps = N_BATCHES * STEPS_PER_BATCH
    spike_buf = np.zeros(total_steps * n_vnc, dtype=np.int32)

    # ─── Use Rust backend ─────────────────────────────
    use_rust = hasattr(tcm, 'lif_connectome') and \
        hasattr(tcm.lif_connectome, 'lif_sim_cpu_recording')
    if not use_rust:
        print("  ⚠ Rust LIF backend not found! Trying Python fallback.")
    else:
        print("  ✅ Rust LIF backend available")

    # ─── Poisson noise generator ──────────────────────
    poisson_rng = np.random.RandomState(0)

    print(f"\n  Running {N_BATCHES} batches...")
    t_start = time.perf_counter()

    for batch in range(N_BATCHES):
        # Poisson background noise (Shiu et al.: 100-360 Hz)
        drive_mask = rng.rand(n_vnc) < 0.3
        rate = np.where(
            drive_mask[:n_vnc],
            200.0 + 100.0 * math.sin(2 * math.pi * 5.0 * batch / N_BATCHES),
            100.0,
        )
        spike_prob = rate * DT_MS / 1000.0
        spiking = poisson_rng.rand(n_vnc) < spike_prob
        pulse_amp = 30.0
        external_I.fill(0.0)
        external_I[:n_vnc] = np.where(spiking, pulse_amp, 0.0).astype(np.float32)

        # Small tonic drive to VNC intrinsic neurons
        vnc_drive = rng.rand(n_vnc) < 0.05
        external_I[:n_vnc][vnc_drive] += 8.0

        I_buf[:n_vnc] = external_I[:n_vnc]

        if use_rust:
            tcm.lif_connectome.lif_sim_cpu_recording(
                vnc_rp, vnc_ci, vnc_vals,
                STEPS_PER_BATCH, DT_MS,
                v, tau_m, v_rest, v_thresh, v_reset,
                nt_types, external_I, I_buf, spikes,
                spike_buf[batch * STEPS_PER_BATCH * n_vnc:],
                g_syn, tau_syn,
                delay_buf, delay_steps, delay_pos,
            )

        if batch % 10 == 9 or batch == 0:
            total_spikes = int(np.sum(spike_buf[: (batch + 1) * STEPS_PER_BATCH * n_vnc]))
            elapsed = time.perf_counter() - t_start
            print(f"  Batch {batch+1}/{N_BATCHES} | "
                  f"total_spikes={total_spikes:,} | "
                  f"rate={total_spikes/n_vnc/((batch+1)*STEPS_PER_BATCH*DT_MS/1000):.1f} Hz")

    total_elapsed = time.perf_counter() - t_start
    total_spikes_final = int(np.sum(spike_buf))
    print(f"\n  Done in {total_elapsed:.1f}s | "
          f"total_spikes={total_spikes_final:,} | "
          f"mean_rate={total_spikes_final/n_vnc/(TOTAL_MS/1000):.1f} Hz")

    return spike_buf, total_steps


def analyze_tripod(spike_buf, total_steps, en_by_leg, n_vnc, dt_ms=DT_MS):
    """Analyze EN spike phases for tripod gait pattern."""
    print("\n" + "═" * 72)
    print("  TRIPOD GAIT ANALYSIS")
    print("═" * 72)

    # ─── Spike counts per leg ─────────────────────────
    print(f"\n  {'Leg':15s} | {'Spikes':>8s} | {'Rate(Hz)':>8s} | {'N_neurons':>9s}")
    print(f"  {'-'*15} | {'-'*8} | {'-'*8} | {'-'*9}")

    leg_spike_times = {}
    for leg, idxs in sorted(en_by_leg.items()):
        if len(idxs) == 0:
            continue
        idx_arr = np.array([i for i in idxs if i < n_vnc], dtype=np.int32)
        if len(idx_arr) == 0:
            continue

        leg_burst = np.zeros(total_steps, dtype=np.int32)
        this_spikes = []
        for step in range(total_steps):
            offset = step * n_vnc
            step_spikes = 0
            for idx in idx_arr:
                if offset + idx < len(spike_buf) and spike_buf[offset + idx]:
                    step_spikes += 1
                    this_spikes.append((step, idx))
            leg_burst[step] = step_spikes > 0  # any spike in this leg group

        duration_s = total_steps * dt_ms / 1000.0
        total_leg_spikes = sum(leg_burst)
        rate = total_leg_spikes / duration_s
        leg_spike_times[leg] = np.array([st for st, _ in this_spikes], dtype=np.int32)

        print(f"  {leg:15s} | {total_leg_spikes:8d} | {rate:8.1f} | {len(idx_arr):9d}")

    # ─── Cross-correlation between legs ───────────────
    print(f"\n  ─── Cross-correlation (spike timing similarity) ───")
    legs = sorted([l for l in en_by_leg.keys() if l in leg_spike_times and len(en_by_leg[l]) > 0])

    print(f"\n  {'Leg pair':>30s} | {'N_coincident':>14s} | {'Phase':>6s}")
    print(f"  {'-'*30} | {'-'*14} | {'-'*6}")

    # Bin spikes into 10ms windows for phase analysis
    bin_ms = 10.0
    n_bins = int(TOTAL_MS / bin_ms)
    half_bins = n_bins // 2

    leg_phase_vec = {}
    for leg in legs:
        times = leg_spike_times[leg]
        binned = np.zeros(n_bins, dtype=np.int32)
        for t in times:
            bin_idx = int(t * dt_ms / bin_ms)
            if bin_idx < n_bins:
                binned[bin_idx] += 1
        # FFT to find dominant frequency and phase
        fft = np.fft.rfft(binned - np.mean(binned))
        freqs = np.fft.rfftfreq(n_bins, d=bin_ms / 1000.0)
        # Find peak in 5-20 Hz range (walking frequency)
        lo = np.searchsorted(freqs, 5.0)
        hi = np.searchsorted(freqs, 20.0)
        if hi > lo and len(fft) > hi:
            peak_bin = np.argmax(np.abs(fft[lo:hi])) + lo
            phase = np.angle(fft[peak_bin])
            freq = freqs[peak_bin]
            leg_phase_vec[leg] = (freq, phase, np.abs(fft[peak_bin]))
            print(f"  {leg:30s} | freq={freq:.1f}Hz phase={phase:+.2f}")

    # ─── Phase differences ─────────────────────────────
    print(f"\n  ─── Phase differences ───")
    for i, leg_i in enumerate(legs):
        for j in range(i + 1, len(legs)):
            leg_j = legs[j]
            if leg_i in leg_phase_vec and leg_j in leg_phase_vec:
                f_i, p_i, _ = leg_phase_vec[leg_i]
                f_j, p_j, _ = leg_phase_vec[leg_j]
                diff = (p_i - p_j + np.pi) % (2 * np.pi) - np.pi
                marker = ''
                if abs(abs(diff) - np.pi) < 0.5:
                    marker = ' ★ ANTIPHASE'
                elif abs(diff) < 0.5:
                    marker = ' ● IN-PHASE'
                print(f"  {leg_i:15s} ↔ {leg_j:15s} | "
                      f"Δφ={diff:+.2f} rad ({diff*180/np.pi:.0f}°){marker}")

    # ─── Tripod check ─────────────────────────────────
    print(f"\n  ─── Tripod gait check ───")
    has_front = 'front_leg' in leg_phase_vec
    has_middle = 'middle_leg' in leg_phase_vec
    has_hind = 'hind_leg' in leg_phase_vec

    if has_front and has_middle and has_hind:
        f_f, p_f, _ = leg_phase_vec['front_leg']
        f_m, p_m, _ = leg_phase_vec['middle_leg']
        f_h, p_h, _ = leg_phase_vec['hind_leg']

        front_mid = (p_f - p_m + np.pi) % (2 * np.pi) - np.pi
        mid_hind = (p_m - p_h + np.pi) % (2 * np.pi) - np.pi
        hind_front = (p_h - p_f + np.pi) % (2 * np.pi) - np.pi

        print(f"  Front↔Middle: Δφ={front_mid:+.2f} ({front_mid*180/np.pi:.0f}°)")
        print(f"  Middle↔Hind:  Δφ={mid_hind:+.2f} ({mid_hind*180/np.pi:.0f}°)")
        print(f"  Hind↔Front:   Δφ={hind_front:+.2f} ({hind_front*180/np.pi:.0f}°)")

        # Tripod: all three pairs should be ~120° apart or antiphase
        if (abs(abs(front_mid) - np.pi) < 0.5 and
            abs(abs(mid_hind) - np.pi) < 0.5):
            print(f"\n  ✅ TRIPOD GAIT CONFIRMED!")
            print(f"     Front와 Middle в антифазе (трёхопорная походка)")
        elif (abs(abs(front_mid) - 2*np.pi/3) < 0.5 and
              abs(abs(mid_hind) - 2*np.pi/3) < 0.5):
            print(f"\n  ✅ 120° Gait pattern (three-phase)")
        else:
            print(f"\n  ⚠ No tripod pattern detected")
            print(f"     Возможно: нужно больше EN нейронов или другая частота")

    elif has_front and has_middle:
        f_f, p_f, _ = leg_phase_vec['front_leg']
        f_m, p_m, _ = leg_phase_vec['middle_leg']
        diff = (p_f - p_m + np.pi) % (2 * np.pi) - np.pi
        if abs(abs(diff) - np.pi) < 0.5:
            print(f"\n  ✅ Front↔Middle ANTIPHASE (tripod indicator)")
        else:
            print(f"\n  ⚠ Front↔Middle not antiphase (Δφ={diff:.2f})")
    else:
        print(f"\n  ⚠ Not enough leg EN groups for tripod analysis")
        print(f"     Found: {[l for l in legs if l in leg_phase_vec]}")


def main():
    print("═" * 72)
    print("  LIF VNC Tripod — LIF симуляция + анализ походки")
    print("═" * 72)

    vnc_rp, vnc_ci, vnc_vals, n_vnc, n_padded, en_by_leg, _, _ = \
        load_vnc_for_lif()

    spike_buf, total_steps = run_lif_simulation(
        vnc_rp, vnc_ci, vnc_vals, n_vnc, n_padded, en_by_leg)

    analyze_tripod(spike_buf, total_steps, en_by_leg, n_vnc)

    # ─── Save ──────────────────────────────────────────
    save_path = '/tmp/lif_vnc_tripod.npz'
    np.savez(save_path,
             spike_buf=spike_buf,
             en_by_leg={k: np.array(v, dtype=np.int32)
                        for k, v in en_by_leg.items()},
             n_vnc=n_vnc,
             total_steps=total_steps,
             dt_ms=DT_MS)
    print(f"\n  Saved to: {save_path}")
    print("═" * 72)


if __name__ == '__main__':
    main()
