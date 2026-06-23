#!/usr/bin/env python3
"""
GPU ψ-wave на полном FAFB + MANC с мостами.
Движение в глазах → мозг → брюшная цепочка.
"""
import numpy as np, pandas as pd, pickle, os, time
import taas_checkmate as tc

DATA_DIR = os.path.expanduser("~/Загрузки")
CLASS_PATH = os.path.join(DATA_DIR, "classification.csv.gz")

def f32(x): return np.ascontiguousarray(x, dtype=np.float32)
def i32(x): return np.ascontiguousarray(x, dtype=np.int32)

# ─── Load merged CSR ──────────────────────────────────────
print("Loading merged FAFB+MANC (bridged)...")
data = np.load('/tmp/fafb_manc_csr_bridged.npz')
row_ptr = i32(data['row_ptr'])
col_indices = i32(data['col_indices'])
values = f32(data['values'])
n_fafb = int(data['n_fafb'])
n_manc = int(data['n_manc'])
n_total = n_fafb + n_manc
n_syn = len(values)
print(f"  {n_total} neurons ({n_fafb} FAFB + {n_manc} MANC), {n_syn:,} synapses")

with open('/tmp/fafb_manc_bridge_meta.pkl', 'rb') as f:
    meta = pickle.load(f)

# ─── Region masks ─────────────────────────────────────────
clf = pd.read_csv(CLASS_PATH)
fafb_root_to_idx = {rid: i for i, rid in enumerate(np.load('/tmp/fafb_csr_npz.npz')['row_ptr'])}  # dummy, rebuild properly
# Actually just use the classification directly
import json

# Build FAFB masks properly
with open('/tmp/root_to_csr.pkl', 'rb') as f:
    root_to_csr = pickle.load(f)
clf = pd.read_csv(CLASS_PATH)
clf = clf[clf['root_id'].isin(set(root_to_csr.keys()))].copy()
clf['csr_idx'] = clf['root_id'].map(root_to_csr)

masks = {}
for side in ['left', 'right']:
    masks[f'FAFB_{side}'] = clf.loc[clf['side'] == side, 'csr_idx'].values.astype(np.int32)
for sc in ['optic', 'central', 'visual_projection', 'descending', 'ascending']:
    masks[f'FAFB_{sc}'] = clf.loc[clf['super_class'] == sc, 'csr_idx'].values.astype(np.int32)
for sub in ['t4_neuron', 't5_neuron', 'lamina_monopolar', 'medulla_intrinsic',
            'lobula_intrinsic']:
    masks[f'FAFB_{sub}'] = clf.loc[clf['sub_class'] == sub, 'csr_idx'].values.astype(np.int32)

# MANC masks from bridge metadata
for k in ['asc_idx', 'sens_asc_idx', 'motor_idx', 'intrinsic_idx']:
    masks[f'MANC_{k}'] = meta[k]

# Neuromere regions for MANC
manc_prop = pd.read_feather(f'{DATA_DIR}/v1.0_manc-v1.0-neuron-properties.feather')
with open('/tmp/fafb_manc_bridge.pkl', 'rb') as f:
    bridge = pickle.load(f)
manc_ids = bridge['manc_ids']
manc_to_idx = bridge['manc_to_idx']
manc_prop_t = manc_prop[manc_prop['bodyId'].isin(manc_ids)].copy()
manc_prop_t['csr_idx'] = manc_prop_t['bodyId'].map(manc_to_idx)
if 'somaNeuromere' in manc_prop_t.columns:
    for nm in manc_prop_t['somaNeuromere'].dropna().unique():
        idx = manc_prop_t.loc[manc_prop_t['somaNeuromere'] == nm, 'csr_idx'].values.astype(np.int32)
        masks[f'MANC_{nm}'] = idx

print(f"Region masks:")
for k, v in sorted(masks.items()):
    print(f"  {k}: {len(v)}")

# ─── Stimulus: rightward motion on optic lobes ────────────
optic_idx = masks.get('FAFB_optic', np.array([], dtype=np.int32))
left_optic = np.intersect1d(optic_idx, masks.get('FAFB_left', np.array([], dtype=np.int32)))
right_optic = np.intersect1d(optic_idx, masks.get('FAFB_right', np.array([], dtype=np.int32)))
print(f"\nOptic drive: L={len(left_optic)}, R={len(right_optic)}")

i_ext = f32(np.zeros(n_total))
amp = 0.15
for lobe, direction in [(left_optic, +1.0), (right_optic, -1.0)]:
    if len(lobe) == 0: continue
    order = np.argsort(lobe)
    sorted_idx = lobe[order]
    n_lobe = len(sorted_idx)
    phase = np.linspace(0, direction * 4 * np.pi, n_lobe, dtype=np.float32)
    i_ext[sorted_idx] = amp * np.sin(phase)

# ─── GPU ψ-wave ──────────────────────────────────────────
DT, HBAR = 0.5, 1.0
STEPS = 5000

print(f"\n{'='*60}")
print(f"  ψ-wave on merged FAFB+MANC (bridged) — {STEPS} steps")
print(f"{'='*60}")

v = f32(np.ones(n_total) * np.sqrt(1.0/n_total, dtype=np.float32))
u = f32(np.zeros(n_total))

t0 = time.perf_counter()
elapsed, v_f, u_f = tc.bench_psi_wave_gpu(
    row_ptr, col_indices, values, v, u, DT, HBAR, STEPS, i_ext)
t1 = time.perf_counter()

v_arr = np.array(v_f, dtype=np.float32)
u_arr = np.array(u_f, dtype=np.float32)
nrm = np.sqrt(v_arr**2 + u_arr**2)

print(f"  {elapsed:.3f}s ({STEPS/elapsed:.0f} step/s, {elapsed/STEPS*1e6:.0f} μs/step)")
print(f"  |ψ| = {nrm.min():.6f} – {nrm.max():.6f}  {'✅' if abs(nrm.max()-1)<1e-5 else '⚠️'}")

# ─── Analysis ─────────────────────────────────────────────
def region_stats(v_arr, u_arr, idx, name):
    """Return (mean_v, mean_u, coherence, amp_std) for region."""
    mask = idx[idx < len(v_arr)]
    if len(mask) < 5: return None
    v_r = v_arr[mask]; u_r = u_arr[mask]
    phase = np.arctan2(u_r, v_r)
    coh = np.abs(np.mean(np.exp(1j * phase)))
    amp = np.sqrt(v_r**2 + u_r**2)
    return (np.mean(v_r), np.mean(u_r), coh, np.std(amp))

print(f"\n{'─'*70}")
print(f"{'Region':<28} {'n':>6} {'mean_v':>10} {'mean_u':>10} {'coherence':>10} {'|ψ|_std':>10}")
print(f"{'─'*70}")
for key in sorted(masks):
    s = region_stats(v_arr, u_arr, masks[key], key)
    if s: print(f"{key:<28} {len(masks[key]):>6} {s[0]:>+10.6f} {s[1]:>+10.6f} {s[2]:>10.4f} {s[3]:>10.6f}")

# Brain vs VNC comparison
print(f"\n{'='*60}")
print(f"  BRAIN vs NERVE CORD (with bridges)")
print(f"{'='*60}")
for compartment, start, end, label in [
    ('FAFB brain', 0, n_fafb, 'Brain'),
    ('MANC VNC', n_fafb, n_total, 'Nerve Cord'),
]:
    v_c = v_arr[start:end]; u_c = u_arr[start:end]
    phase_c = np.arctan2(u_c, v_c)
    coh = np.abs(np.mean(np.exp(1j * phase_c)))
    print(f"  {label:<12} {end-start:>6} nrn  mean_v={np.mean(v_c):+.6f}  mean_u={np.mean(u_c):+.6f}  "
          f"coh={coh:.4f}  |ψ|_std={np.std(np.sqrt(v_c**2+u_c**2)):.6f}")

# Bridge effect: compare MANC neurons that ARE vs ARE NOT targets of descending
desc_target_set = set()
for pre in meta.get('desc_idx', np.array([], dtype=np.int32)):
    start_r = int(row_ptr[pre])
    end_r = int(row_ptr[pre+1])
    desc_target_set.update(col_indices[start_r:end_r].tolist())
# Remove FAFB targets
desc_manc_targets = np.array([t for t in desc_target_set if t >= n_fafb], dtype=np.int32)
non_target_manc = np.array([i for i in range(n_fafb, n_total) if i not in desc_target_set], dtype=np.int32)

if len(desc_manc_targets) > 5 and len(non_target_manc) > 5:
    print(f"\n{'='*60}")
    print(f"  BRIDGE EFFECT: MANC neurons with descending input vs without")
    print(f"{'='*60}")
    for idx, label in [(desc_manc_targets, 'With descending input'),
                       (non_target_manc, 'No descending input')]:
        s = region_stats(v_arr, u_arr, idx, label)
        if s:
            print(f"  {label:<25} {len(idx):>6} nrn  mean_v={s[0]:+.6f}  mean_u={s[1]:+.6f}  "
                  f"coh={s[2]:.4f}  |ψ|_std={s[3]:.6f}")

# Descending drive strength
print(f"\n{'='*60}")
print(f"  DESCENDING DRIVE: FAFB → MANC synapse counts")
print(f"{'='*60}")
n_desc = len(meta.get('desc_idx', np.array([], dtype=np.int32)))
desc_total_syn = sum(np.diff(row_ptr)[meta['desc_idx']].tolist()) if n_desc > 0 else 0
manc_total_syn = sum(np.diff(row_ptr)[n_fafb:n_total].tolist())
bridge_syn = len([p for p in range(len(row_ptr)-1)  # count bridge edges to MANC
                  if row_ptr[p+1] - row_ptr[p] > 0 and
                  any(col_indices[row_ptr[p]:row_ptr[p+1]] >= n_fafb) and
                  p < n_fafb])
n_bridge_syn = sum(max(0, row_ptr[i+1] - row_ptr[i]) for i in meta.get('desc_idx', np.array([], dtype=np.int32)))
n_asc_syn = sum(max(0, row_ptr[i+1] - row_ptr[i]) for i in meta.get('asc_idx', np.array([], dtype=np.int32)))
n_sa_syn = sum(max(0, row_ptr[i+1] - row_ptr[i]) for i in meta.get('sens_asc_idx', np.array([], dtype=np.int32)))
print(f"  FAFB descending total outgoing synapses: {desc_total_syn}")
print(f"  MANC total incoming synapses: {manc_total_syn}")
print(f"  Cross-compartment bridge synapses: ~{n_bridge_syn+n_asc_syn+n_sa_syn}")
print(f"    Desc: {n_bridge_syn}, Asc: {n_asc_syn}, SensAsc: {n_sa_syn}")

# Phase relationship
brain_phase = np.arctan2(np.mean(v_arr[:n_fafb]), np.mean(u_arr[:n_fafb]))
vnc_phase = np.arctan2(np.mean(v_arr[n_fafb:]), np.mean(u_arr[n_fafb:]))
print(f"\n{'='*60}")
print(f"  PHASE RELATIONSHIP")
print(f"{'='*60}")
print(f"  Brain phase: {brain_phase:.3f} rad")
print(f"  VNC phase:   {vnc_phase:.3f} rad")
print(f"  Phase diff:  {(vnc_phase - brain_phase + np.pi) % (2*np.pi) - np.pi:.3f} rad ({((vnc_phase - brain_phase + np.pi) % (2*np.pi) - np.pi) / np.pi:.2f}π)")

# Hemifield asymmetry across the whole system
print(f"  Hemifield asymmetry (rightward motion):")
if 'FAFB_left' in masks and 'FAFB_right' in masks:
    mu_l = region_stats(v_arr, u_arr, masks['FAFB_left'], '')[1]
    mu_r = region_stats(v_arr, u_arr, masks['FAFB_right'], '')[1]
    print(f"  Right - Left mean_u = {mu_r - mu_l:+.6f} (L={mu_l:+.4f}, R={mu_r:+.4f})")

print(f"\n  ✅ Done — brain→VNC ψ-wave propagation with synthetic bridges!")
