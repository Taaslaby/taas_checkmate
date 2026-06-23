#!/usr/bin/env python3
"""
GPU ψ-wave на полном FAFB + MANC: мозг + брюшная цепочка.
Движение в глазах → активность в мозге и нервной цепочке.
"""
import numpy as np, pandas as pd, pickle, os, time
import taas_checkmate as tc

DATA_DIR = os.path.expanduser("~/Загрузки")
CLASS_PATH = os.path.join(DATA_DIR, "classification.csv.gz")
MAPPING_PATH = "/tmp/root_to_csr.pkl"
BRIDGE_PATH = "/tmp/fafb_manc_bridge.pkl"

def f32(x): return np.ascontiguousarray(x, dtype=np.float32)
def i32(x): return np.ascontiguousarray(x, dtype=np.int32)

# ─── Load merged CSR ──────────────────────────────────────
print("Loading merged FAFB+MANC CSR...")
data = np.load('/tmp/fafb_manc_csr.npz')
row_ptr = i32(data['row_ptr'])
col_indices = i32(data['col_indices'])
values = f32(data['values'])
n_fafb = int(data['n_fafb'])
n_manc = int(data['n_manc'])
n_total = n_fafb + n_manc
n_syn = len(values)
print(f"  {n_total} neurons ({n_fafb} FAFB + {n_manc} MANC), {n_syn} synapses")

with open(BRIDGE_PATH, 'rb') as f:
    bridge = pickle.load(f)
fafb_ids = bridge['fafb_ids']
manc_ids = bridge['manc_ids']

# ─── Load classification ──────────────────────────────────
print("\nLoading classification...")
clf = pd.read_csv(CLASS_PATH)
clf = clf[clf['root_id'].isin(fafb_ids)].copy()
clf['csr_idx'] = clf['root_id'].map({rid: i for i, rid in enumerate(fafb_ids)})

# MANC neuron properties — add rough region classification
manc_prop = pd.read_feather(f'{DATA_DIR}/v1.0_manc-v1.0-neuron-properties.feather')
manc_prop = manc_prop[manc_prop['bodyId'].isin(manc_ids)].copy()
manc_prop['csr_idx'] = manc_prop['bodyId'].map({rid: i + n_fafb for i, rid in enumerate(manc_ids)})
# Classify MANC neurons by neuromere
if 'somaNeuromere' in manc_prop.columns:
    print(f"  MANC neuromeres: {manc_prop['somaNeuromere'].value_counts().to_dict()}")

# ─── Region masks ─────────────────────────────────────────
masks = {}
for side in ['left', 'right', 'center']:
    idx = clf.loc[clf['side'] == side, 'csr_idx'].values.astype(np.int32)
    masks[f'FAFB_{side}'] = idx
for sc in ['optic', 'central', 'visual_projection', 'descending', 'ascending']:
    idx = clf.loc[clf['super_class'] == sc, 'csr_idx'].values.astype(np.int32)
    masks[f'FAFB_{sc}'] = idx
for sub in ['t4_neuron', 't5_neuron', 'lamina_monopolar', 'medulla_intrinsic',
            'lobula_intrinsic', 'Kenyon_Cell', 'CX']:
    idx = clf.loc[clf['sub_class'] == sub, 'csr_idx'].values.astype(np.int32)
    masks[f'FAFB_{sub}'] = idx

# MANC masks: by neuromere
if 'somaNeuromere' in manc_prop.columns:
    for nm in manc_prop['somaNeuromere'].unique():
        idx = manc_prop.loc[manc_prop['somaNeuromere'] == nm, 'csr_idx'].values.astype(np.int32)
        masks[f'MANC_{nm}'] = idx

# Full side masks for entire merged network
left_idx = np.concatenate([masks.get('FAFB_left', np.array([], dtype=np.int32))])
right_idx = np.concatenate([masks.get('FAFB_right', np.array([], dtype=np.int32))])

print(f"\nRegion masks:")
for k, v in sorted(masks.items()):
    print(f"  {k}: {len(v)}")

# ─── Stimulus: rightward motion on optic lobes ────────────
print("\nBuilding motion stimulus...")
optic_idx = masks.get('FAFB_optic', np.array([], dtype=np.int32))
left_optic = np.intersect1d(optic_idx, masks.get('FAFB_left', np.array([], dtype=np.int32)))
right_optic = np.intersect1d(optic_idx, masks.get('FAFB_right', np.array([], dtype=np.int32)))

i_ext = f32(np.zeros(n_total))
amp = 0.15
for lobe, direction in [(left_optic, +1.0), (right_optic, -1.0)]:
    if len(lobe) == 0: continue
    order = np.argsort(lobe)
    sorted_idx = lobe[order]
    n_lobe = len(sorted_idx)
    phase = np.linspace(0, direction * 4 * np.pi, n_lobe, dtype=np.float32)
    i_ext[sorted_idx] = amp * np.sin(phase)

# ─── GPU ψ-wave on merged connectome ──────────────────────
DT, HBAR = 0.5, 1.0
STEPS = 5000

print(f"\nRunning {STEPS} ψ-wave steps on merged FAFB+MANC...")
v = f32(np.ones(n_total) * np.sqrt(1.0/n_total, dtype=np.float32))
u = f32(np.zeros(n_total))

t0 = time.perf_counter()
elapsed, v_f, u_f = tc.bench_psi_wave_gpu(
    row_ptr, col_indices, values, v, u, DT, HBAR, STEPS, i_ext)
t1 = time.perf_counter()

v_arr = np.array(v_f, dtype=np.float32)
u_arr = np.array(u_f, dtype=np.float32)
nrm = np.sqrt(v_arr**2 + u_arr**2)

print(f"\n{'='*60}")
print(f"  ψ-wave on merged FAFB+MANC ({n_total} neurons, {n_syn} synapses)")
print(f"{'='*60}")
print(f"  {elapsed:.3f}s ({STEPS/elapsed:.0f} step/s, {elapsed/STEPS*1e6:.0f} μs/step)")
print(f"  |ψ| = {nrm.min():.6f} – {nrm.max():.6f}  {'✅' if abs(nrm.max()-1)<1e-5 else '⚠️'}")

# ─── Analysis per region ──────────────────────────────────
print(f"\n{'='*60}")
print("  ψ-WAVE ACTIVATION BY REGION (rightward motion)")
print(f"{'='*60}")
print(f"{'Region':<25} {'n':>6} {'mean_v':>10} {'mean_u':>10} {'coherence':>10} {'|ψ|_std':>10}")
print('-' * 65)
for key in sorted(masks):
    idx = masks[key]
    if len(idx) < 5: continue
    mask = idx[idx < n_total]
    if len(mask) < 5: continue
    v_r = v_arr[mask]; u_r = u_arr[mask]
    phase = np.arctan2(u_r, v_r)
    coh = np.abs(np.mean(np.exp(1j * phase)))
    amp = np.sqrt(v_r**2 + u_r**2)
    print(f"{key:<25} {len(mask):>6} {np.mean(v_r):>+10.6f} {np.mean(u_r):>+10.6f} {coh:>10.4f} {np.std(amp):>10.6f}")

# ─── Brain vs VNC comparison ──────────────────────────────
print(f"\n{'='*60}")
print("  BRAIN (FAFB) vs NERVE CORD (MANC)")
print(f"{'='*60}")
for compartment, start, end, label in [
    ('FAFB brain', 0, n_fafb, 'Brain'),
    ('MANC VNC', n_fafb, n_total, 'Nerve Cord'),
]:
    v_c = v_arr[start:end]; u_c = u_arr[start:end]
    phase_c = np.arctan2(u_c, v_c)
    coh = np.abs(np.mean(np.exp(1j * phase_c)))
    print(f"  {label}: {end-start:>7} nrn, mean_v={np.mean(v_c):+.6f}, mean_u={np.mean(u_c):+.6f}, "
          f"coh={coh:.4f}, |ψ|_std={np.std(np.sqrt(v_c**2+u_c**2)):.6f}")

# ─── T4 direction selectivity ─────────────────────────────
print(f"\n{'='*60}")
print("  T4/T5 DIRECTION SELECTIVITY (side split)")
print(f"{'='*60}")
for t in ['t4_neuron', 't5_neuron']:
    for side in ['left', 'right']:
        key = f'FAFB_{t}'
        side_key = f'FAFB_{side}'
        all_t = masks.get(key, np.array([], dtype=np.int32))
        side_idx = masks.get(side_key, np.array([], dtype=np.int32))
        t_side = np.intersect1d(all_t, side_idx)
        if len(t_side) < 5: continue
        mask = t_side[t_side < n_total]
        v_r = v_arr[mask]; u_r = u_arr[mask]
        coh = np.abs(np.mean(np.exp(1j * np.arctan2(u_r, v_r))))
        print(f"  {t} {side:>5}: {len(mask):>5} nrn, mean_u={np.mean(u_r):+.6f}, coh={coh:.4f}")

print(f"\n  ✅ Done — merged ψ-wave on full brain + nerve cord!")
