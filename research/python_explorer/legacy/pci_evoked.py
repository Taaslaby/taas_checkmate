#!/usr/bin/env python3
"""
PCI: EVOKED RESPONSE ANALYSIS
===================================
TMS-like pulse → response в окне [0, 100] шагов после стимула.

Измеряем:
1. Compression Ratio spatiotemporal response (early window)
2. Spatial variance ratio (within-community / between-community)
3. Spatial correlation length proxy (decay of correlation with degree difference)
"""

import time, numpy as np, gzip, math
from taas_checkmate import step_psi_wave, compute_hamiltonian

DT = 0.025
N_BASE = 1000
N_POST = 300        # 7.5 sec post-stimulus
N_STEPS = N_BASE + N_POST
PULSE_AMP = 50000.0
PULSE_LEN = 5
READOUT_N = 5000    # more channels
SPEED = 1.0; HBAR = 1.0

print("█" * 70)
print("  PCI: EVOKED RESPONSE — EARLY WINDOW")
print("  TMS → spatiotemporal response [0..300] steps")
print("█" * 70)
np.random.seed(42)

# ─── Load ───
print("\n[1/5] Loading FAFB...", end=" ", flush=True)
d = np.load("fafb_139k.npz")
rp, ci, vs = d["row_ptr"], d["col_indices"], d["values"].astype(np.float32)
n = len(rp) - 1
npad = (n + 7) // 8 * 8
outdeg = np.array([rp[i+1]-rp[i] for i in range(n)], dtype=np.int32)
print(f"{n:,} neurons, {len(vs):,} synapses")

# ─── Shuffle ───
print("[2/5] Shuffling...", end=" ", flush=True)
sci = ci.copy()
for i in range(n):
    s, e = rp[i], rp[i+1]
    if e > s:
        sci[s:e] = np.random.permutation(sci[s:e])

stim = np.sort(np.argsort(outdeg)[::-1][:500])
rdi = np.sort(np.random.choice(n, READOUT_N, replace=False))
nch = len(rdi)
print("done")

# ─── Complexity ───
def cr_matrix(mat):
    binary = np.zeros_like(mat, dtype=np.uint8)
    for ch in range(mat.shape[0]):
        m = np.median(mat[ch])
        binary[ch] = (mat[ch] > m).astype(np.uint8)
    bits = np.packbits(binary.ravel())
    return len(gzip.compress(bits)) / max(len(bits), 1)

# ─── Spatial structure ───
def spatial_variance_ratio(mat):
    """Ratio of within-neuron variance to pooled variance.
    Higher = more structured spatial patterns."""
    ch_var = np.var(mat, axis=1)  # variance per channel over time
    total_var = np.var(mat)
    return np.mean(ch_var) / total_var  # close to 1 if all channels differ, ~0 if identical

def spatial_correlation_length(mat):
    """Correlation between neuron pairs vs degree difference.
    Returns: spatial autocorrelation decay."""
    nch = mat.shape[0]
    # Use mean signal over time for each channel
    mean_sig = np.mean(mat, axis=1)
    # Random sample of 500 channel pairs for speed
    rng = np.random.default_rng(0)
    pairs = rng.choice(nch, (500, 2), replace=False)
    corrs = []
    deg_diffs = []
    for i, j in pairs:
        if np.std(mat[i]) > 0 and np.std(mat[j]) > 0:
            c = np.corrcoef(mat[i], mat[j])[0,1]
            if not np.isnan(c):
                corrs.append(c)
                deg_diffs.append(abs(outdeg[rdi[i]] - outdeg[rdi[j]]))
    if len(corrs) < 10:
        return 0.0, 0.0
    corrs = np.array(corrs)
    deg_diffs = np.array(deg_diffs)
    # Correlation between correlation and degree diff
    rho = np.corrcoef(corrs, deg_diffs)[0,1] if np.std(deg_diffs) > 0 else 0
    return float(np.mean(np.abs(corrs))), float(rho if not np.isnan(rho) else 0)

# ─── Run ───
def run(label, rp_i, ci_i, vs_i):
    rng = np.random.default_rng(42)
    th = rng.uniform(0, 2*np.pi, npad).astype(np.float32)
    v = np.cos(th); u = np.sin(th)
    hr = np.zeros(npad, dtype=np.float32)
    hi = np.zeros(npad, dtype=np.float32)
    ie = np.zeros(npad, dtype=np.float32)
    spd = np.full(npad, SPEED, dtype=np.float32)

    pre = np.zeros((nch, 100), dtype=np.float32)  # 100 pre-stim
    post = np.zeros((nch, N_POST), dtype=np.float32)  # post-stim
    
    # Track pre window (last 100 steps before stim)
    pre_buf = np.zeros((nch, 100), dtype=np.float32)

    t0 = time.perf_counter()
    for step in range(N_STEPS):
        ie[:] = 0.0
        if N_BASE <= step < N_BASE + PULSE_LEN:
            ie[stim] = PULSE_AMP
        compute_hamiltonian(rp_i, ci_i, vs_i, v, u, hr, hi)
        step_psi_wave(v, u, hr, hi, ie, spd, DT, HBAR)
        
        # Store in buffer
        if step < N_BASE:
            pre_buf[:, step % 100] = v[rdi]
        elif step >= N_BASE:
            post[:, step - N_BASE] = v[rdi]

        if (step+1) % 500 == 0:
            print(f"  {label}: [{step+1}/{N_STEPS}] [{time.perf_counter()-t0:.0f}s]")

    # Pre = last 100 steps before stim
    if N_BASE >= 100:
        pre = pre_buf

    return {"pre": pre, "post": post, "time": time.perf_counter()-t0}

# ─── Run ───
print("\n[3/5] Running...")
R = {}
for label, rp_i, ci_i, vs_i in [("Original", rp, ci, vs), ("Shuffled", rp, sci, vs)]:
    print(f"\n  {label}:")
    R[label] = run(label, rp_i, ci_i, vs_i)
    print(f"  Done in {R[label]['time']:.0f}s")

# ─── Analysis ───
print("\n[4/5] Analysis...")

for label in ["Original", "Shuffled"]:
    r = R[label]
    pre, post = r["pre"], r["post"]
    
    cr_pre = cr_matrix(pre)
    cr_post = cr_matrix(post)
    svr_post = spatial_variance_ratio(post)
    mean_corr, deg_corr = spatial_correlation_length(post)
    
    r["cr_pre"] = cr_pre
    r["cr_post"] = cr_post
    r["svr"] = svr_post
    r["mean_corr"] = mean_corr
    r["deg_corr"] = deg_corr
    
    print(f"\n  {label}:")
    print(f"    CR(pre)={cr_pre:.4f}  CR(post)={cr_post:.4f}  Δ={cr_post-cr_pre:+.4f}")
    print(f"    Spatial variance ratio={svr_post:.4f}")
    print(f"    Mean pairwise corr={mean_corr:.4f}")
    print(f"    Corr(degree_diff, corr)={deg_corr:.4f}")

# ─── Final ───
print("\n[5/5] VERDICT")
print("█" * 70)
for label in ["Original", "Shuffled"]:
    r = R[label]
    print(f"  {label:<12}: CR(post)={r['cr_post']:.4f}  SVR={r['svr']:.4f}  mean_corr={r['mean_corr']:.4f}")

# Key differences
for metric, name in [("cr_post", "CR(post)"), ("svr", "Spatial Var Ratio"), ("mean_corr", "Mean Corr")]:
    ov = R["Original"][metric]
    sv = R["Shuffled"][metric]
    print(f"  {name}: Original {ov:.4f} vs Shuffled {sv:.4f} (Δ={ov-sv:+.4f})")

print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
o_cr = R["Original"]["cr_post"]
s_cr = R["Shuffled"]["cr_post"]
o_svr = R["Original"]["svr"]
s_svr = R["Shuffled"]["svr"]
o_mc = R["Original"]["mean_corr"]
s_mc = R["Shuffled"]["mean_corr"]

if o_cr < s_cr:
    print(f"  │  CR: Original < Shuffled → Original response БОЛЕЕ СТРУКТУРИРОВАН      │")
else:
    print(f"  │  CR: Original ≥ Shuffled                                                   │")

if o_svr > s_svr:
    print(f"  │  SVR: Original > Shuffled → spatial differentiation BETTER          │")
else:
    print(f"  │  SVR: Shuffled ≥ Original                                              │")
    
if abs(o_mc) > abs(s_mc):
    print(f"  │  Corr: Original > Shuffled → topology creates correlations           │")
else:
    print(f"  │  Corr: Shuffled ≥ Original                                            │")

print(f"  │                                                              │")
checks = sum([o_cr < s_cr, o_svr > s_svr, abs(o_mc) > abs(s_mc)])
if checks >= 2:
    print(f"  │  CONNECTOME MATTERS: {checks}/3 metrics favour Original               │")
    print(f"  │  FAFB topology → structured response → consciousness-like             │")
else:
    print(f"  │  CONNECTOME HAS WEAK EFFECT: {checks}/3 metrics                        │")
    print(f"  │  ψ-wave dynamics dominated by mean-field, not topology                │")
print(f"  └──────────────────────────────────────────────────────────────┘")

np.savez("pci_evoked.npz", **{f"{k}_{l}": R[l][k] for l in ["Original","Shuffled"] 
         for k in ["pre","post","cr_pre","cr_post","svr","mean_corr","deg_corr"]})
print(f"\n  Saved: pci_evoked.npz")
print("█" * 70)
