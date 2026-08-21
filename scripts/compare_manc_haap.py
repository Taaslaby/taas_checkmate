#!/usr/bin/env python3
"""
🧬 compare_manc_haap.py — MANC (нервный шнур) : HAAP vs HH.

Один и тот же коннектом (manc_vnc.taas), одна и та же стимуляция (Poisson
по сенсорным), две физиологии:
  - HAAP (M-ток адаптация + h-ток rebound) — литературный атлас, .haap
  - HH   (NaV/KV + leak)                  — литературная физиология, .hhparams

Цель: показать, что адаптация (M-ток) снимает разгон motor-нейронов,
который даёт HH (motor лупил на ~477 Гц в compare_fafb).

Usage:
    python compare_manc_haap.py
"""

import numpy as np

from taas_checkmate import connectome_brain as cb
import pathlib, sys  # репо-рут для tools.taas_io при запуске из scripts/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.taas_io import load_haap, load_hhparams, load_taas

TAAS = "resources/data/manc_vnc.taas"
HAAP = "resources/data/manc_vnc_haap_physio.haap"
HHP = "resources/data/manc_vnc_hh_physio.hhparams"
DT = 0.05
N_STEPS = 300
POISSON_RATE = 500.0
POISSON_WEIGHT = 200.0
NORM = 256.0

# nt_types → имена (как в train_physio.CLASS_NAMES)
CLASS_NAMES = {0: "unk", 1: "optic", 2: "sensory", 3: "central", 4: "motor",
               5: "desc", 6: "asc", 7: "endocr", 8: "visProj", 9: "visCent",
               10: "sensAsc"}

# HHParamIdx: 1=G_Na, 3=G_K, 5=G_Leak, 6=E_Leak
COL_G_NA, COL_G_K, COL_G_LEAK, COL_E_LEAK = 1, 3, 5, 6


# HAAParamIdx: 0=DT 1=Inv_Cm 2=G_L 3=E_L 4=G_M 5=E_K 6=G_H 7=E_H
#              8=Inv_Tau_U 9=Inv_Tau_H 10=V_Half_U 11=K_U 12=V_Half_H
#              13=K_H 14=V_Thresh 15=V_Reset 16=Delta_U
# Парсеры .taas/.haap/.hhparams — в tools/taas_io.py


def class_rates(spikes_2d, nt):
    """firing rates (Hz) per nt_types class."""
    uniq = sorted(int(x) for x in np.unique(nt))
    rates = {}
    for c in uniq:
        idx = np.where(nt == float(c))[0]
        if len(idx) == 0:
            continue
        rates[c] = spikes_2d[:, idx].sum() / len(idx) / (N_STEPS * DT / 1000.0)
    return rates


def fmt_rates(rates):
    return "  ".join(f"{CLASS_NAMES.get(c, c)}={rates[c]:.1f}" for c in sorted(rates))


def main():
    n, rp, ci, vals0, nt, rt = load_taas(TAAS)
    vals = (vals0 / NORM).astype(np.float32)
    sensory_ids = np.where((nt == 2.0) | (nt == 10.0))[0].tolist()
    print(f"MANC: {n} нейронов | стимуляция sensory+sensAsc = {len(sensory_ids)}")

    # ─── HAAP ───
    haap = load_haap(HAAP, n)
    conn = cb.ConnectomeData(rp, ci, vals, nt)
    runner = cb.BrainRunner(conn, model="haap", dt_ms=DT)
    # DT (col 0) форсится движком из dt_ms; ставим 1..16
    for col in range(1, 17):
        runner.set_haap_param_column(col, haap[col])
    runner.clear_poisson_groups()
    runner.add_poisson_group(sensory_ids, POISSON_RATE, POISSON_WEIGHT)
    buf = runner.run(N_STEPS, np.zeros(n, dtype=np.float32))
    spikes = np.asarray(buf).reshape(N_STEPS, n)
    haap_rates = class_rates(spikes, nt)
    print(f"\n[HAAP] total={spikes.sum()} | {fmt_rates(haap_rates)}", flush=True)

    # ─── HH (current-based; MANC .taas не несёт receptor_types) ───
    hh = load_hhparams(HHP, n)
    conn2 = cb.ConnectomeData(rp, ci, vals, nt)
    runner2 = cb.BrainRunner(conn2, model="hh", dt_ms=DT)
    runner2.reset()
    runner2.set_hh_param_column(COL_G_NA, hh[0])    # g_na
    runner2.set_hh_param_column(COL_G_K, hh[1])     # g_k
    runner2.set_hh_param_column(COL_G_LEAK, hh[2])  # g_l
    runner2.set_hh_param_column(COL_E_LEAK, hh[5])  # e_l
    runner2.clear_poisson_groups()
    runner2.add_poisson_group(sensory_ids, POISSON_RATE, POISSON_WEIGHT)
    buf2 = runner2.run(N_STEPS, np.zeros(n, dtype=np.float32))
    spikes2 = np.asarray(buf2).reshape(N_STEPS, n)
    hh_rates = class_rates(spikes2, nt)
    print(f"[HH  ] total={spikes2.sum()} | {fmt_rates(hh_rates)}", flush=True)

    # ─── сводка ───
    print("\n==== СВОДКА (Гц) ====")
    print(f"{'класс':>10}  {'HAAP':>8}  {'HH':>8}")
    for c in sorted(set(haap_rates) | set(hh_rates)):
        hr = haap_rates.get(c, 0.0)
        hhr = hh_rates.get(c, 0.0)
        print(f"{CLASS_NAMES.get(c, c):>10}  {hr:8.1f}  {hhr:8.1f}")


if __name__ == "__main__":
    main()
