#!/usr/bin/env python3
"""Сравнение: литература vs обученная физиология FAFB в идентичной конфигурации."""
import sys
import numpy as np
from train_physio import BASE_PHYSIO, CLASS_NAMES, COL_G_NA, COL_G_K, COL_G_LEAK, COL_E_LEAK
from taas_checkmate import connectome_brain as cb
import pathlib, sys  # репо-рут для tools.taas_io при запуске из scripts/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.taas_io import load_hhparams, load_taas

TAAS = "resources/data/fafb_783.taas"
HHP = "resources/data/fafb_physio_trained.hhparams"
NORM = 256.0

n, rp, ci, vals0, nt, rt = load_taas(TAAS)
vals = (vals0 / NORM).astype(np.float32)
sensory_ids = np.where((nt == 2.0) | (nt == 8.0))[0].tolist()


# парсеры .taas/.hhparams — в tools/taas_io.py


def run_with(cols, label):
    conn = cb.ConnectomeData(rp, ci, vals, nt)
    conn.set_receptor_types(rt)
    runner = cb.BrainRunner(conn, model="hh_synaptic", dt_ms=0.05)
    runner.reset()
    runner.set_hh_param_column(COL_G_NA, cols[0])
    runner.set_hh_param_column(COL_G_K, cols[1])
    runner.set_hh_param_column(COL_G_LEAK, cols[2])
    runner.set_hh_param_column(COL_E_LEAK, cols[5])
    runner.clear_poisson_groups()
    runner.add_poisson_group(sensory_ids, 500.0, 200.0)
    buf = runner.run(300, np.zeros(n, dtype=np.float32))
    spikes = np.asarray(buf).reshape(300, n)
    rates = {}
    for c in sorted(int(x) for x in np.unique(nt)):
        idx = np.where(nt == float(c))[0]
        rates[c] = spikes[:, idx].sum() / len(idx) / (300 * 0.05 / 1000.0)
    print(f"{label}: total={spikes.sum()} | " +
          " ".join(f"{CLASS_NAMES[c]}={rates[c]:.0f}" for c in sorted(rates)), flush=True)
    return rates


# литература (масштабы 1, сдвиги 0)
lit = np.empty((7, n), dtype=np.float32)
for c, base in BASE_PHYSIO.items():
    m = nt == float(c)
    lit[0][m] = base["g_na"]; lit[1][m] = base["g_k"]; lit[2][m] = base["g_l"]
    lit[5][m] = base["e_l"]
lit[3] = 50.0; lit[4] = -80.0; lit[6] = 1.0  # e_na, e_k, inv_cm

trained = load_hhparams(HHP, n)

r_lit = run_with(lit, "литература ")
r_tr = run_with(trained, "обученная  ")
print()
print(f"{'класс':>9} {'лит':>7} {'обуч':>7} {'цель':>7}")
for c in sorted(r_lit):
    print(f"{CLASS_NAMES[c]:>9} {r_lit[c]:>7.0f} {r_tr[c]:>7.0f} {15.0 if c != 1 else 25.0:>7.0f}")
