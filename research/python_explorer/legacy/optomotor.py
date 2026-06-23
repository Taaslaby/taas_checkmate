#!/usr/bin/env python3
"""
OPTOMOTOR TEST: ψ-WAVE НА FAFB
Phase-gradient drive на левом оптическом лопасте → response
на правом. Сравнение rightward vs leftward rotation.
"""

import time, pickle, math, numpy as np, pandas as pd
from taas_checkmate import step_psi_wave, compute_hamiltonian

DT = 0.025
N_STEPS = 2000
N_BASE = 500
DRIVE_AMP = 50.0
DRIVE_FREQ = 2.0
N_DRIVE = 2000
N_READOUT = 2000
SPEED = 1.0
HBAR = 1.0

print("█" * 70)
print("  OPTOMOTOR TEST — ψ-WAVE DIRECTION SELECTIVITY")
print("█" * 70)

# ─── Load ───
print("\n[1/5] Loading FAFB CSR + mapping...", end=" ", flush=True)
d = np.load("fafb_139k.npz")
rp, ci, vs = d["row_ptr"], d["col_indices"], d["values"].astype(np.float32)
n = len(rp) - 1
npad = (n + 7) // 8 * 8
with open("/tmp/root_to_csr.pkl", "rb") as f:
    root_to_idx = pickle.load(f)
cl = pd.read_csv("/home/taaslab/Загрузки/classification.csv.gz")
cl["csr_idx"] = cl["root_id"].map(root_to_idx).astype("Int64")
cl = cl.dropna(subset=["csr_idx"])
cl["csr_idx"] = cl["csr_idx"].astype(np.int32)
print(f"138,584 mapped neurons")

# ─── Select drive (left optic) and readout (right optic) ───
print("[2/5] Selecting left/right optic neurons...", end=" ", flush=True)
left = cl[(cl["side"]=="left") & (cl["super_class"]=="optic") & (cl["flow"]=="intrinsic")]
right = cl[(cl["side"]=="right") & (cl["super_class"]=="optic") & (cl["flow"]=="intrinsic")]

drive_idx = left["csr_idx"].values[:N_DRIVE]
readout_idx = right["csr_idx"].values[:min(N_READOUT, len(right))]
n_ch = len(readout_idx)
print(f"drive={len(drive_idx)}L, readout={n_ch}R")

# ─── Helper: load coords for phase gradient ───
print("[3/5] Loading coordinates...", end=" ", flush=True)
coord = pd.read_csv("/home/taaslab/Загрузки/coordinates.csv.gz")
coord_dict = dict(zip(coord["root_id"], coord["position"]))
def parse_pos(s):
    try:
        parts = s.strip("[]").split()
        return float(parts[0]), float(parts[1]), float(parts[2])
    except: return 0,0,0

# Get positions for left drive neurons, sort by x (AP axis)
left_coords = []
for rid in left["root_id"].iloc[:N_DRIVE]:
    if rid in coord_dict:
        x, y, z = parse_pos(coord_dict[rid])
        left_coords.append((x, y, z, rid))
    else:
        left_coords.append((0, 0, 0, rid))
left_coords = np.array([(c[0], c[1], c[2]) for c in left_coords], dtype=np.float32)
# Sort by x
xs = left_coords[:, 0]
order = np.argsort(xs)
drive_idx = drive_idx[order]
print("done")

# ─── Run ───
def run(direction):
    """direction: +1 = rightward, -1 = leftward"""
    rng = np.random.default_rng(42)
    th = rng.uniform(0, 2*np.pi, npad).astype(np.float32)
    v = np.cos(th); u = np.sin(th)
    hr = np.zeros(npad, dtype=np.float32)
    hi = np.zeros(npad, dtype=np.float32)
    ie = np.zeros(npad, dtype=np.float32)
    spd = np.full(npad, SPEED, dtype=np.float32)
    resp = np.zeros((n_ch, N_STEPS), dtype=np.float32)
    
    phases = direction * 2 * np.pi * np.linspace(-1, 1, len(drive_idx))
    freq = 2 * np.pi * DRIVE_FREQ

    t0 = time.perf_counter()
    for step in range(N_STEPS):
        ie[:] = 0.0
        if step >= N_BASE:
            t = step * DT
            ie[drive_idx] = DRIVE_AMP * np.sin(freq * t + phases)
        compute_hamiltonian(rp, ci, vs, v, u, hr, hi)
        step_psi_wave(v, u, hr, hi, ie, spd, DT, HBAR)
        resp[:, step] = v[readout_idx]
        if (step+1) % 500 == 0:
            print(f"  {direction:+d}: [{step+1}/{N_STEPS}] [{time.perf_counter()-t0:.0f}s]")
    return resp, time.perf_counter()-t0

# ─── Run ───
print("\n[4/5] Running rightward (+1)...")
resp_r, tr = run(+1)
print(f"\nRunning leftward (-1)...")
resp_l, tl = run(-1)
print(f"\n  Done: R={tr:.0f}s L={tl:.0f}s")

# ─── Analysis ───
print("\n[5/5] Analysis...")
evoked_r = np.mean(resp_r[:, N_BASE:], axis=1) - np.mean(resp_r[:, :N_BASE], axis=1)
evoked_l = np.mean(resp_l[:, N_BASE:], axis=1) - np.mean(resp_l[:, :N_BASE], axis=1)

# Direction index per neuron
di = (evoked_r - evoked_l) / (np.abs(evoked_r) + np.abs(evoked_l) + 1e-10)
mean_di = np.mean(di)
print(f"  Mean direction index: {mean_di:.4f}")
print(f"  DI > 0: {np.sum(di>0)/len(di)*100:.1f}%")
print(f"  |DI| > 0.1: {np.sum(np.abs(di)>0.1)/len(di)*100:.1f}%")

# Response difference over time
te = np.arange(N_STEPS) * DT
diff_resp = np.mean(resp_r - resp_l, axis=0)
print(f"  Mean response diff (R-L): [{diff_resp[N_BASE]:+.4f} {diff_resp[N_BASE+50]:+.4f} {diff_resp[N_BASE+100]:+.4f}]")

# Spectral power at drive frequency
from numpy.fft import rfft, rfftfreq
sig = diff_resp[N_BASE:] - np.mean(diff_resp[N_BASE:])
fft = np.abs(rfft(sig * np.hanning(len(sig))))**2
fr = rfftfreq(len(sig), DT)
fd_idx = np.argmin(np.abs(fr - DRIVE_FREQ))
p_drive = fft[fd_idx]
p_total = np.sum(fft)
print(f"  Power at {DRIVE_FREQ}Hz: {p_drive/p_total*100:.1f}%")

# Coherence during stim vs baseline
def coherence(ts, base_end, stim_start):
    b = ts[:, :base_end]
    s = ts[:, stim_start:]
    c_b = np.mean(np.abs(np.mean(b, axis=1))) / np.mean(np.abs(b)) if np.std(b) > 0 else 0
    c_s = np.mean(np.abs(np.mean(s, axis=1))) / np.mean(np.abs(s)) if np.std(s) > 0 else 0
    return c_b, c_s

c_b, c_s = coherence(resp_r, N_BASE, N_BASE)
print(f"  Coherence: base={c_b:.4f} stim={c_s:.4f}")

# ─── Verdict ───
print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
if abs(mean_di) > 0.02:
    print(f"  │  DIRECTION INDEX = {mean_di:+.4f} — ЕСТЬ НАПРАВЛЕННОСТЬ!               │")
    print(f"  │  ψ-wave на FAFB детектирует направление оптического потока  │")
    print(f"  │  ← CONNECTOME PROCESSES VISUAL MOTION                       │")
else:
    print(f"  │  DI = {mean_di:+.4f} — направленность не обнаружена             │")
    print(f"  │  ψ-wave (linear phase dynamics) не детектирует направление    │")
    print(f"  │  Для optomotor нужны абертоны (нелинейный коллапс)           │")
print(f"  └──────────────────────────────────────────────────────────────┘")

np.savez("optomotor.npz", resp_r=resp_r, resp_l=resp_l, di=di,
         dr_idx=drive_idx, ro_idx=readout_idx, diff_mean=np.mean(resp_r-resp_l,axis=0))
print(f"\n  Saved: optomotor.npz")
print("█" * 70)
