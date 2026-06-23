#!/usr/bin/env python3
"""
GPU ψ-wave на FAFB: движение в глазах → активность в мозге.

Сценарий:
1. Классифицируем нейроны: сторона (L/R), тип (T4/T5/optic/central)
2. Фа́зовый градиент на оптических долях — имитация движения слева направо
3. GPU ψ-wave 10K шагов через коннектом
4. Анализируем: selectivity, propagation, coherence по регионам
"""
import numpy as np
import pandas as pd
import pickle, os, time
import taas_checkmate as tc

DATA_DIR = os.path.expanduser("~/Загрузки")
CSV_PATH   = os.path.join(DATA_DIR, "connections_princeton.csv.gz")
CLASS_PATH = os.path.join(DATA_DIR, "classification.csv.gz")
MAPPING_PATH = "/tmp/root_to_csr.pkl"

def f32(x): return np.ascontiguousarray(x, dtype=np.float32)
def i32(x): return np.ascontiguousarray(x, dtype=np.int32)

# ─── 1. Load mapping & classification ─────────────────────
print("Loading mapping...")
with open(MAPPING_PATH, "rb") as f:
    root_to_csr = pickle.load(f)
all_roots = np.array(sorted(root_to_csr.keys(), key=lambda r: root_to_csr[r]))
n_all = len(all_roots)

print(f"Loading classification ({CLASS_PATH})...")
clf = pd.read_csv(CLASS_PATH)
clf = clf[clf['root_id'].isin(all_roots)].copy()
clf['csr_idx'] = clf['root_id'].map(root_to_csr)
print(f"  {len(clf)} classified neurons")

# ─── 2. Build region masks ────────────────────────────────
masks = {}
# By side
for side in ['left', 'right', 'center']:
    idx = clf.loc[clf['side'] == side, 'csr_idx'].values.astype(np.int32)
    masks[f'side_{side}'] = idx

# By super_class
for sc in ['optic', 'central', 'sensory', 'visual_projection', 'ascending', 'descending']:
    idx = clf.loc[clf['super_class'] == sc, 'csr_idx'].values.astype(np.int32)
    masks[f'sc_{sc}'] = idx

# By sub_class — motion vision
for sub in ['t4_neuron', 't5_neuron', 't2_neuron', 't3_neuron', 't1_neuron',
            'transmedullary', 'lamina_monopolar', 'medulla_intrinsic',
            'lobula_intrinsic', 'lobula_plate_intrinsic',
            'tangential', 'Kenyon_Cell', 'CX']:
    idx = clf.loc[clf['sub_class'] == sub, 'csr_idx'].values.astype(np.int32)
    masks[f'sub_{sub}'] = idx

# T4/T5 by side
for side_name, side_val in [('L', 'left'), ('R', 'right')]:
    for sub in ['t4_neuron', 't5_neuron']:
        idx = clf.loc[(clf['sub_class'] == sub) & (clf['side'] == side_val), 'csr_idx'].values.astype(np.int32)
        masks[f'{sub}_{side_name}'] = idx

print(f"  Created {len(masks)} region masks")
for k, v in sorted(masks.items()):
    print(f"    {k}: {len(v)}")

# ─── 3. Load CSR ──────────────────────────────────────────
print("\nLoading CSR connectome...")
data = np.load('/tmp/fafb_csr_npz.npz')
row_ptr = i32(data['row_ptr'])
col_indices = i32(data['col_indices'])
values = f32(data['values'])
n = len(row_ptr) - 1
n_syn = len(values)

# ─── 4. Stimulus: drifting phase gradient on optic lobe ───
print("\nBuilding motion stimuli...")
# Stim A: leftward motion (phase gradient right→left across lobes)
# Stim B: rightward motion (phase gradient left→right)

# Find optic lobe afferent neurons (photo_receptors + lamina entry)
afferent_idx = np.unique(np.concatenate([
    masks.get('sub_photo_receptor', np.array([], dtype=np.int32)),
    masks.get('sub_lamina_monopolar', np.array([], dtype=np.int32)),
]))

# If no photo_receptor key, use all optic neurons as drive targets
optic_idx = masks.get('sc_optic', np.array([], dtype=np.int32))
if len(afferent_idx) < 100:
    afferent_idx = optic_idx[:min(5000, len(optic_idx))]

# For each hemisphere, create phase gradient along neuron index
left_optic  = np.intersect1d(optic_idx, masks.get('side_left', np.array([], dtype=np.int32)))
right_optic = np.intersect1d(optic_idx, masks.get('side_right', np.array([], dtype=np.int32)))

def make_drive_phase_grad(idx_list, direction=1.0, amp=0.15):
    """Phase gradient along sorted neuron indices.
    direction = +1: left→right, -1: right→left"""
    drive = np.zeros(n, dtype=np.float32)
    if len(idx_list) == 0:
        return drive
    order = np.argsort(idx_list)
    sorted_idx = idx_list[order]
    n_grp = len(sorted_idx)
    phase = np.linspace(0, direction * 4 * np.pi, n_grp, dtype=np.float32)
    drive[sorted_idx] = amp * np.sin(phase)
    return f32(drive)

# Rightward motion: left lobe drives with +phase, right with delayed phase
# +phase on left, -phase on right simulates rightward motion
i_ext_right = make_drive_phase_grad(left_optic, +1.0, 0.15)
i_ext_right += make_drive_phase_grad(right_optic, -1.0, 0.15)

# Leftward motion: reversed
i_ext_left = make_drive_phase_grad(left_optic, -1.0, 0.15)
i_ext_left += make_drive_phase_grad(right_optic, +1.0, 0.15)

# Blank (no motion, just uniform buzz)
i_ext_blank = f32(np.zeros(n))
if len(left_optic) > 0:
    buzz = np.random.default_rng(42).normal(0, 0.02, len(left_optic)).astype(np.float32)
    i_ext_blank[left_optic[:len(buzz)]] = buzz
if len(right_optic) > 0:
    buzz = np.random.default_rng(43).normal(0, 0.02, len(right_optic)).astype(np.float32)
    i_ext_blank[right_optic[:len(buzz)]] = buzz

# ─── 5. GPU ψ-wave ────────────────────────────────────────
DT, HBAR = 0.5, 1.0
STEPS = 10000

def run_psi(i_ext, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    v = f32(np.ones(n) * np.sqrt(1.0/n, dtype=np.float32))
    u = f32(np.zeros(n))
    t0 = time.perf_counter()
    elapsed, v_f, u_f = tc.bench_psi_wave_gpu(
        row_ptr, col_indices, values, v, u, DT, HBAR, STEPS, i_ext)
    t1 = time.perf_counter()
    v_arr = np.array(v_f, dtype=np.float32)
    u_arr = np.array(u_f, dtype=np.float32)
    nrm = np.sqrt(v_arr**2 + u_arr**2)
    print(f"  {elapsed:.3f}s ({STEPS/elapsed:.0f} step/s)")
    print(f"  |ψ| = {nrm.min():.6f} – {nrm.max():.6f}  {'✅' if abs(nrm.max()-1)<1e-5 else '⚠️'}")
    return v_arr, u_arr, elapsed

# Run all three conditions
v_right, u_right, t_right = run_psi(i_ext_right, "RIGHTWARD motion (L→R phase)")
v_left,  u_left,  t_left  = run_psi(i_ext_left,  "LEFTWARD motion (R→L phase)")
v_blank, u_blank, t_blank = run_psi(i_ext_blank, "BLANK (noise)")

# ─── 6. Analysis ──────────────────────────────────────────
print(f"\n{'='*60}")
print("  ANALYSIS: Motion selectivity across brain regions")
print(f"{'='*60}")

def region_stats(v_arr, u_arr, region_idx, name):
    if len(region_idx) == 0:
        return None
    mask = region_idx[region_idx < n]
    if len(mask) == 0:
        return None
    v_r = v_arr[mask]; u_r = u_arr[mask]
    amp = np.sqrt(v_r**2 + u_r**2)
    phase = np.arctan2(u_r, v_r)
    coherence = np.abs(np.mean(np.exp(1j * phase)))
    mean_u = np.mean(u_r)
    mean_v = np.mean(v_r)
    return {
        'region': name, 'n': len(mask),
        'coherence': coherence, 'mean_u': mean_u, 'mean_v': mean_v,
        'amp_mean': np.mean(amp), 'amp_std': np.std(amp),
    }

def compute_dsi(v_arr, u_arr, region_idx):
    """Direction selectivity index: (R - L) / (R + L) for mean u"""
    mask = region_idx[region_idx < n]
    if len(mask) < 5:
        return 0.0
    return np.mean(u_arr[mask])

all_regions = [
    ('sc_optic', 'Optic Lobe'),
    ('sc_central', 'Central Brain'),
    ('sc_visual_projection', 'Vis Projection'),
    ('sc_ascending', 'Ascending'),
    ('sc_descending', 'Descending'),
    ('sub_t4_neuron', 'T4'),
    ('sub_t5_neuron', 'T5'),
    ('sub_t4_neuron_L', 'T4 Left'),
    ('sub_t4_neuron_R', 'T4 Right'),
    ('sub_t5_neuron_L', 'T5 Left'),
    ('sub_t5_neuron_R', 'T5 Right'),
    ('sub_Kenyon_Cell', 'Kenyon Cells'),
    ('sub_CX', 'Central Complex'),
    ('sub_transmedullary', 'Transmedullary'),
    ('sub_lamina_monopolar', 'Lamina'),
    ('sub_medulla_intrinsic', 'Medulla'),
    ('sub_lobula_intrinsic', 'Lobula'),
]

print(f"\n{'Region':<25} {'n':>6} {'Direction Sel':>14} {'Coher':>7}  {'(Rightward vs Leftward)'}")
print('-' * 80)
for key, name in all_regions:
    idx = masks.get(key, np.array([], dtype=np.int32))
    if len(idx) == 0:
        continue
    dsi_r = compute_dsi(v_right, u_right, idx)
    dsi_l = compute_dsi(v_left,  u_left,  idx)
    dsi_b = compute_dsi(v_blank, u_blank, idx)
    # Coherence in rightward condition
    mask = idx[idx < n]
    v_r_s = v_right[mask]; u_r_s = u_right[mask]
    coh = np.abs(np.mean(np.exp(1j * np.arctan2(u_r_s, v_r_s)))) if len(mask) > 0 else 0.0
    # ΔDSI = rightward - leftward (positive = prefers rightward)
    dsi_diff = dsi_r - dsi_l
    print(f"{name:<25} {len(mask):>6} {dsi_diff:>+13.6f} {coh:>7.4f}  "
          f"(R={dsi_r:+.4f}, L={dsi_l:+.4f}, B={dsi_b:+.4f})")

# ─── 7. Direction Tuning Curve ────────────────────────────
print(f"\n{'='*60}")
print("  PROPAGATION: u-component across regions (rightward)")
print(f"{'='*60}")
print(f"{'Region':<25} {'n':>6} {'mean_v':>10} {'mean_u':>10} {'coherence':>10}")
for key, name in all_regions:
    idx = masks.get(key, np.array([], dtype=np.int32))
    if len(idx) < 5:
        continue
    mask = idx[idx < n]
    if len(mask) < 5:
        continue
    v_r = v_right[mask]; u_r = u_right[mask]
    phase = np.arctan2(u_r, v_r)
    coh = np.abs(np.mean(np.exp(1j * phase)))
    print(f"{name:<25} {len(mask):>6} {np.mean(v_r):>+10.6f} {np.mean(u_r):>+10.6f} {coh:>10.4f}")

# ─── 8. Hemisphere asymmetry ──────────────────────────────
print(f"\n{'='*60}")
print("  HEMISPHERE ASYMMETRY (rightward motion)")
print(f"{'='*60}")
for side_name, side_key in [('Left', 'side_left'), ('Right', 'side_right')]:
    idx = masks.get(side_key, np.array([], dtype=np.int32))
    if len(idx) < 5:
        continue
    mask = idx[idx < n]
    v_r = v_right[mask]; u_r = u_right[mask]
    coh = np.abs(np.mean(np.exp(1j * np.arctan2(u_r, v_r))))
    print(f"  {side_name:<6} hemisphere ({len(mask):>6} nrn): mean_v={np.mean(v_r):+.6f}, mean_u={np.mean(u_r):+.6f}, coh={coh:.4f}")

print(f"\n{'='*60}")
print("  DONE")
print(f"{'='*60}")
