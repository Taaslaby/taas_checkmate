#!/usr/bin/env python3
"""
PCI (Perturbational Complexity Index) на GPU — полный FAFB+MANC.

Метод:
1. Baseline: ψ-волна без стимула, запись ψ(t) в 21 точке (каждые 100 шагов)
2. Perturbations: сильный фазовый пинок в 6 регионов (optic L/R, T4, central, CX, descending)
3. PCI = spatial diversity × temporal complexity ответа

GPU: каждая запись = прогон bench_psi_wave_gpu от начального состояния до нужного шага.
"""
import numpy as np, pandas as pd, pickle, os, time, itertools
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
print(f"  {n_total} neurons ({n_fafb} FAFB + {n_manc} MANC), {len(values):,} synapses")

with open('/tmp/fafb_manc_bridge_meta.pkl', 'rb') as f:
    meta = pickle.load(f)

# ─── Region masks ─────────────────────────────────────────
with open('/tmp/root_to_csr.pkl', 'rb') as f:
    root_to_csr = pickle.load(f)
clf = pd.read_csv(CLASS_PATH)
clf = clf[clf['root_id'].isin(set(root_to_csr.keys()))].copy()
clf['csr_idx'] = clf['root_id'].map(root_to_csr)

masks = {
    'Left_optic':  clf.loc[(clf['super_class']=='optic') & (clf['side']=='left'), 'csr_idx'].values.astype(np.int32),
    'Right_optic': clf.loc[(clf['super_class']=='optic') & (clf['side']=='right'), 'csr_idx'].values.astype(np.int32),
    'T4':          clf.loc[clf['sub_class']=='t4_neuron', 'csr_idx'].values.astype(np.int32),
    'T5':          clf.loc[clf['sub_class']=='t5_neuron', 'csr_idx'].values.astype(np.int32),
    'Central':     clf.loc[clf['super_class']=='central', 'csr_idx'].values.astype(np.int32),
    'Descending':  clf.loc[clf['super_class']=='descending', 'csr_idx'].values.astype(np.int32),
    'VNC_motor':   meta['motor_idx'],
    'VNC_asc':     meta['asc_idx'],
}
for k, v in masks.items():
    print(f"  {k}: {len(v)} neurons")

# ─── ψ-wave parameters ───────────────────────────────────
DT, HBAR = 0.5, 1.0
TOTAL_STEPS = 2000
RECORD_EVERY = 100  # record every 100 steps → 21 time points (including 0)
RECORD_TIMES = list(range(0, TOTAL_STEPS + 1, RECORD_EVERY))
N_TIMES = len(RECORD_TIMES)
print(f"\nRecording at {N_TIMES} time points: {RECORD_TIMES}")

# ─── Helper: run ψ-wave and record at intervals ──────────
def run_and_record(step_intervals, label, i_ext_func=None):
    """Run ψ-wave in segments, recording after each segment.
    step_intervals: list of (n_steps, description)
    i_ext_func: callable(step, total_steps) → i_ext array (or None for constant)
    Returns: list of (v_arr, u_arr) at each recording point.
    """
    results = []
    v = f32(np.ones(n_total) * np.sqrt(1.0/n_total, dtype=np.float32))
    u = f32(np.zeros(n_total))
    
    # Always record initial state
    results.append((v.copy(), u.copy()))
    
    cumulative = 0
    for n_steps, desc in step_intervals:
        if n_steps == 0:
            continue
        # Build constant drive for this segment
        if i_ext_func:
            ie = f32(i_ext_func(cumulative, TOTAL_STEPS))
        else:
            ie = f32(np.zeros(n_total))
        
        elapsed, v_f, u_f = tc.bench_psi_wave_gpu(
            row_ptr, col_indices, values, v, u, DT, HBAR, n_steps, ie)
        
        v = np.array(v_f, dtype=np.float32)
        u = np.array(u_f, dtype=np.float32)
        results.append((v, u))
        cumulative += n_steps
        print(f"    {desc}: +{n_steps:>4} steps → t={cumulative:>4}, "
              f"mean_u={np.mean(u):+.4f}, {elapsed:.3f}s")
    
    return results

# ─── Baseline ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  BASELINE (no stimulus)")
print(f"{'='*60}")
t0_base = time.perf_counter()
baseline = run_and_record(
    [(s - 0, f't={s}') for s in RECORD_TIMES[1:]],
    'baseline')
t_base = time.perf_counter() - t0_base
print(f"  Baseline done: {t_base:.1f}s")

# ─── Perturbations ───────────────────────────────────────
def pulse_stimulus(mask_idx, amplitude=0.5, duration_steps=50):
    """Create a stimulus function: strong phase kick for first `duration` steps."""
    def stim_func(step_from, total):
        if step_from < duration_steps:
            ie = np.zeros(n_total, dtype=np.float32)
            ie[mask_idx] = amplitude
            return ie
        return np.zeros(n_total, dtype=np.float32)
    return stim_func

perturbations = {
    'Left_optic_pulse':   pulse_stimulus(masks['Left_optic'], 0.3, 50),
    'Right_optic_pulse':  pulse_stimulus(masks['Right_optic'], 0.3, 50),
    'T4_pulse':           pulse_stimulus(masks['T4'], 0.5, 50),
    'Central_pulse':      pulse_stimulus(masks['Central'], 0.5, 50),
    'Descending_pulse':   pulse_stimulus(masks['Descending'], 1.0, 50),
    'VNC_motor_pulse':    pulse_stimulus(masks['VNC_motor'], 1.0, 50),
    'VNC_asc_pulse':      pulse_stimulus(masks['VNC_asc'], 0.5, 50),
}

all_results = {'baseline': baseline}
for pname, pfunc in perturbations.items():
    print(f"\n{'='*60}")
    print(f"  PERTURBATION: {pname}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    all_results[pname] = run_and_record(
        [(s - 0, f't={s}') for s in RECORD_TIMES[1:]],
        pname, i_ext_func=pfunc)
    print(f"  {time.perf_counter() - t0:.1f}s")

# ─── PCI Computation ─────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PCI ANALYSIS")
print(f"{'='*60}")

def compute_pci(baseline_traj, perturbed_traj, region_masks):
    """PCI = mean(|Δψ|²) over all perturbed neurons, 
    normalized by baseline variance.
    
    Also computes:
    - Cascade size: fraction of neurons with |Δψ| > threshold
    - Spatial diversity: entropy of Δψ distribution across regions
    - Temporal complexity: variance of Δψ over time
    """
    n_times = len(baseline_traj)
    n_regions = len(region_masks)
    
    # Per-region Δψ(t) over time
    region_delta = {}
    for rname, ridx in region_masks.items():
        mask = ridx[ridx < n_total]
        if len(mask) < 3:
            continue
        deltas = []
        for t in range(n_times):
            base_v, base_u = baseline_traj[t]
            pert_v, pert_u = perturbed_traj[t]
            delta = np.sqrt((pert_v[mask] - base_v[mask])**2 + 
                          (pert_u[mask] - base_u[mask])**2)
            deltas.append(np.mean(delta))
        region_delta[rname] = np.array(deltas)
    
    # Global Δψ across all neurons at final time
    final_t = -1
    delta_all = np.sqrt((perturbed_traj[final_t][0] - baseline_traj[final_t][0])**2 +
                       (perturbed_traj[final_t][1] - baseline_traj[final_t][1])**2)
    
    # Global Δψ over time
    delta_t = []
    for t in range(n_times):
        d = np.sqrt((perturbed_traj[t][0] - baseline_traj[t][0])**2 +
                   (perturbed_traj[t][1] - baseline_traj[t][1])**2)
        delta_t.append(np.mean(d))
    delta_t = np.array(delta_t)
    
    # PCI metrics
    cascade_threshold = np.std(delta_all) * 2  # 2σ threshold
    cascade_size = np.mean(delta_all > cascade_threshold)
    
    # Temporal complexity: variance of mean Δ over time
    temporal_complexity = np.var(delta_t)
    
    # Spatial complexity: entropy of regional Δ distribution at final time
    reg_means = np.array([v[-1] for v in region_delta.values()])
    reg_dist = reg_means / (reg_means.sum() + 1e-10)
    spatial_entropy = -np.sum(reg_dist * np.log2(reg_dist + 1e-10)) / np.log2(len(reg_dist) + 1e-10)
    
    # Integrated PCI: mean |Δψ|² × cascade_size × spatial_entropy
    pci_raw = np.mean(delta_all**2) * cascade_size * spatial_entropy
    # Normalize to [0, 1]
    pci = np.clip(pci_raw / (pci_raw + 1.0), 0, 1)
    
    return {
        'pci': pci,
        'pci_raw': pci_raw,
        'cascade_size': cascade_size,
        'spatial_entropy': spatial_entropy,
        'temporal_complexity': temporal_complexity,
        'mean_delta_final': np.mean(delta_all),
        'delta_over_time': delta_t,
        'region_delta': region_delta,
    }

# Compute PCI for each perturbation
all_regions = {k: v for k, v in masks.items()}
pci_results = {}

for pname in perturbations:
    print(f"\n  {pname}:")
    pci = compute_pci(baseline, all_results[pname], all_regions)
    pci_results[pname] = pci
    print(f"    PCI             = {pci['pci']:.4f}")
    print(f"    Cascade size    = {pci['cascade_size']:.4f}")
    print(f"    Spatial entropy = {pci['spatial_entropy']:.4f}")
    print(f"    Temporal comp   = {pci['temporal_complexity']:.6f}")
    print(f"    Mean |Δψ| final = {pci['mean_delta_final']:.6f}")
    
    # Top 3 most affected regions
    reg_d = pci['region_delta']
    sorted_regs = sorted(reg_d.items(), key=lambda x: x[1][-1], reverse=True)
    print(f"    Top affected regions:")
    for reg_name, reg_deltas in sorted_regs[:5]:
        print(f"      {reg_name:<20}: Δ={reg_deltas[-1]:.6f}")

# ─── Summary table ───────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PCI SUMMARY")
print(f"{'='*60}")
print(f"{'Perturbation':<25} {'PCI':>8} {'Cascade':>8} {'Entropy':>8} {'|Δψ|_final':>10}")
print('-' * 60)
for pname in sorted(pci_results.keys()):
    p = pci_results[pname]
    print(f"{pname:<25} {p['pci']:>8.4f} {p['cascade_size']:>8.4f} {p['spatial_entropy']:>8.4f} {p['mean_delta_final']:>10.6f}")

# ─── Interpretation ──────────────────────────────────────
print(f"\n{'='*60}")
print(f"  INTERPRETATION")
print(f"{'='*60}")
best = max(pci_results, key=lambda k: pci_results[k]['pci'])
worst = min(pci_results, key=lambda k: pci_results[k]['pci'])
print(f"  Highest PCI: {best} ({pci_results[best]['pci']:.4f})")
print(f"  Lowest PCI:  {worst} ({pci_results[worst]['pci']:.4f})")
print(f"  Spread:      {max(p['pci'] for p in pci_results.values()) - min(p['pci'] for p in pci_results.values()):.4f}")

# Compare brain vs VNC perturbations
brain_pcis = [pci_results[k]['pci'] for k in perturbations if 'optic' in k or 'T4' in k or 'Central' in k or 'Descending' in k]
vnc_pcis = [pci_results[k]['pci'] for k in perturbations if 'VNC' in k]
if brain_pcis and vnc_pcis:
    print(f"\n  Brain perturbations mean PCI: {np.mean(brain_pcis):.4f} ± {np.std(brain_pcis):.4f}")
    print(f"  VNC perturbations mean PCI:   {np.mean(vnc_pcis):.4f} ± {np.std(vnc_pcis):.4f}")

print(f"\n  ✅ PCI on merged FAFB+MANC complete!")
