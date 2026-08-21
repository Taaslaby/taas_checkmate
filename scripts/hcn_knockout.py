#!/usr/bin/env python3
"""
🧬 hcn_knockout.py — нокаут гена HCN (Ih) на реальном VNC.

Скелет один (manc_vnc.taas), физиология одна (rebound-профиль central).
Меняется ОДИН ген: HCN → g_h.

  wild-type:   g_h = 0.50  (Ih есть → post-inhibitory rebound → half-center)
  HCN knockout: g_h ≈ 0.02 (Ih нет → rebound исчез → антифаза должна рухнуть)

Если антифаза (R2) падает при нокауте — ген HCN функционально важен.
"""

import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from taas_checkmate import connectome_brain as cb
import pathlib, sys  # репо-рут для tools.taas_io при запуске из scripts/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.taas_io import load_haap, load_taas

DT = 0.05
NORM = 256.0
TAAS = "resources/data/manc_vnc.taas"
FEATHER = "D:/connectoms/datasets/v1.0_manc-v1.0-neuron-properties.feather"
HAAP = "resources/data/manc_vnc_haap_physio.haap"

KICK = 1000
TOTAL = 15000
CHUNK = 500
DRIVE = 15.0
N_KICK = 2000

COL_G_H = 6


# парсеры .taas/.haap — в tools/taas_io.py


def run(n, rp, ci, vals, nt, haap_cols, motor, central, g_h):
    conn = cb.ConnectomeData(rp, ci, vals, nt)
    runner = cb.BrainRunner(conn, model="haap", dt_ms=DT)
    for col in range(1, 17):
        runner.set_haap_param_column(col, haap_cols[col])
    # rebound-профиль central (как vnc_rhythm_test), но g_h = аргумент (ген HCN)
    overrides = {3: -55.0, 4: 0.30, 6: g_h, 9: 1.0 / 200.0, 16: 0.04}
    for col, val in overrides.items():
        tmp = haap_cols[col].copy()
        tmp[central] = val
        runner.set_haap_param_column(col, tmp)

    rng = np.random.default_rng(7)
    kick_idx = rng.choice(central, N_KICK, replace=False)
    ext_drive = np.zeros(n, dtype=np.float32)
    ext_drive[kick_idx] = DRIVE
    zero = np.zeros(n, dtype=np.float32)

    mn_spikes = []
    for start in range(0, TOTAL, CHUNK):
        ns = min(CHUNK, TOTAL - start)
        ext = ext_drive if start < KICK else zero
        s = np.asarray(runner.run(ns, ext)).reshape(ns, n)
        mn_spikes.append(s[:, motor].copy())
    return np.vstack(mn_spikes)


def analyze(mn):
    T = mn.shape[0]
    win = np.ones(20) / 20
    mn_rate = np.convolve(mn.sum(axis=1).astype(np.float32), win, mode="same")
    sig = mn_rate - mn_rate.mean()
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(T, d=DT / 1000.0)
    band = (freqs >= 0.5) & (freqs <= 50.0)
    f0 = freqs[band][np.argmax(spec[band])] if band.sum() else None
    t_axis = np.arange(T) * DT / 1000.0
    phases = []
    n_used = 0
    for i in range(mn.shape[1]):
        sp_t = t_axis[mn[:, i] > 0]
        if len(sp_t) < 3:
            continue
        n_used += 1
        z = np.exp(2j * np.pi * f0 * sp_t) if f0 else 1.0
        phases.append(np.angle(z.mean()))
    phases = np.array(phases)
    R1 = np.abs(np.exp(1j * phases).mean()) if n_used else 0.0
    R2 = np.abs(np.exp(2j * phases).mean()) if n_used else 0.0
    return f0, R1, R2, n_used


def main():
    n, rp, ci, vals0, nt, _rt = load_taas(TAAS)
    p = pd.read_feather(FEATHER)
    gaba = (p["predictedNt"].values == "gaba")
    sign = np.where(gaba, -1.0, 1.0).astype(np.float32)
    deg = (rp[1:] - rp[:-1]).astype(np.int64)
    source = np.repeat(np.arange(n), deg)
    vals = (vals0 / NORM) * sign[source]
    motor = np.where(nt == 4.0)[0]
    central = np.where(nt == 3.0)[0]
    haap = load_haap(HAAP, n)

    print(f"VNC: {n} нейронов, motor={len(motor)}, central={len(central)}, GABA={gaba.sum()}\n")

    print("=== wild-type: HCN есть (g_h=0.50) ===", flush=True)
    mn_wt = run(n, rp, ci, vals, nt, haap, motor, central, g_h=0.50)
    f0, R1, R2, used = analyze(mn_wt)
    print(f"  f0={f0:.2f} Гц  R1={R1:.3f}  R2={R2:.3f}  (MN активных {used})")

    print("\n=== HCN knockout: g_h≈0.02 (нет rebound) ===", flush=True)
    mn_ko = run(n, rp, ci, vals, nt, haap, motor, central, g_h=0.02)
    f0k, R1k, R2k, usedk = analyze(mn_ko)
    print(f"  f0={f0k:.2f} Гц  R1={R1k:.3f}  R2={R2k:.3f}  (MN активных {usedk})")

    print("\n=== итог (скелет один, меняется ген HCN) ===")
    print(f"  антифаза R2:  wild-type={R2:.3f}   HCN-KO={R2k:.3f}")
    if R2 > R2k + 0.05:
        print("  → HCN (Ih/rebound) нужен для антифазы ✓")
    else:
        print("  → разницы нет (антифаза не зависит от HCN)")


if __name__ == "__main__":
    main()
