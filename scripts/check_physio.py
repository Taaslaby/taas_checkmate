#!/usr/bin/env python3
"""Проверка точности обученной физиологии FAFB.

1. Обобщение: физиология училась на (rate=500, w=200, классы 2+8) —
   проверяем на ДРУГИХ стимулах (др. амплитуда/частота/классы).
2. Стабильность: вдвое длиннее симуляция — rates не дрейфуют/не убегают.
3. Отклонение от литературы: насколько двинулись параметры по классам.
"""
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
CLASS_ORDER = sorted(int(x) for x in np.unique(nt))


# парсеры .taas/.hhparams — в tools/taas_io.py


def run_rates(cols, stim_ids, rate_hz, weight, n_steps=300):
    # ВАЖНО: каждый прогон = reset → set параметры → стимул → run.
    # reset() пересоздаёт HH-популяцию и стирает параметры, поэтому set ПОСЛЕ reset.
    conn = cb.ConnectomeData(rp, ci, vals, nt)
    conn.set_receptor_types(rt)
    runner = cb.BrainRunner(conn, model="hh_synaptic", dt_ms=0.05)
    runner.reset()
    runner.set_hh_param_column(COL_G_NA, cols[0])
    runner.set_hh_param_column(COL_G_K, cols[1])
    runner.set_hh_param_column(COL_G_LEAK, cols[2])
    runner.set_hh_param_column(COL_E_LEAK, cols[5])
    runner.clear_poisson_groups()
    runner.add_poisson_group(stim_ids, rate_hz, weight)
    buf = runner.run(n_steps, np.zeros(n, dtype=np.float32))
    spikes = np.asarray(buf).reshape(n_steps, n)
    rates = {}
    for c in CLASS_ORDER:
        idx = np.where(nt == float(c))[0]
        rates[c] = spikes[:, idx].sum() / len(idx) / (n_steps * 0.05 / 1000.0)
    return rates, spikes.sum()


def fmt(rates):
    return " ".join(f"{CLASS_NAMES[c]}={rates[c]:.0f}" for c in CLASS_ORDER)


lit = np.empty((7, n), dtype=np.float32)
for c, base in BASE_PHYSIO.items():
    m = nt == float(c)
    lit[0][m] = base["g_na"]; lit[1][m] = base["g_k"]; lit[2][m] = base["g_l"]
    lit[5][m] = base["e_l"]
lit[3] = 50.0; lit[4] = -80.0; lit[6] = 1.0

trained = load_hhparams(HHP, n)

sens = np.where((nt == 2.0) | (nt == 8.0))[0].tolist()
sens_only = np.where(nt == 2.0)[0].tolist()
optic = np.where((nt == 1.0) | (nt == 2.0))[0].tolist()

print("═" * 70)
print("1. ОБОБЩЕНИЕ — обученная физиология на НЕВИДАННЫХ стимулах")
print("   (училась: rate=500, w=200, классы 2+8)")
print("═" * 70)
tests = [
    ("rate=500 w=200 cls2+8 (учебный)", sens, 500, 200),
    ("rate=500 w=150 cls2+8 (др. ампл.)", sens, 500, 150),
    ("rate=300 w=200 cls2+8 (др. частота)", sens, 300, 200),
    ("rate=500 w=200 cls2   (только сенсор)", sens_only, 500, 200),
    ("rate=500 w=250 cls1+2 (optic+сенсор)", optic, 500, 250),
]
for label, ids, r, w in tests:
    r_lit, t_lit = run_rates(lit, ids, r, w)
    r_tr, t_tr = run_rates(trained, ids, r, w)
    print(f"\n{label}")
    print(f"  литература : {fmt(r_lit)}  [total={t_lit}]")
    print(f"  обученная  : {fmt(r_tr)}  [total={t_tr}]")

print()
print("═" * 70)
print("2. СТАБИЛЬНОСТЬ — длинная симуляция (600 шагов, 30 мс)")
print("═" * 70)
r_300, t_300 = run_rates(trained, sens, 500, 200, n_steps=300)
r_600, t_600 = run_rates(trained, sens, 500, 200, n_steps=600)
print(f"  300 шагов: {fmt(r_300)}  [total={t_300}]")
print(f"  600 шагов: {fmt(r_600)}  [total={t_600}]")
drift = max(abs(r_600[c] - r_300[c]) for c in CLASS_ORDER)
print(f"  макс. дрейф rates: {drift:.1f} Гц — {'стабильно' if drift < 10 else 'ДРЕЙФ!'}")

print()
print("═" * 70)
print("3. ОТКЛОНЕНИЕ ОТ ЛИТЕРАТУРЫ — насколько двинулись параметры")
print("═" * 70)
for c in CLASS_ORDER:
    m = nt == float(c)
    g_na_scale = trained[0][m][0] / BASE_PHYSIO[c]["g_na"] if BASE_PHYSIO[c]["g_na"] else 1.0
    g_k_scale = trained[1][m][0] / BASE_PHYSIO[c]["g_k"]
    e_l_shift = trained[5][m][0] - BASE_PHYSIO[c]["e_l"]
    flag = " ← двинут" if abs(np.log(max(g_na_scale, 1e-3))) > 0.15 or abs(e_l_shift) > 8 else ""
    print(f"  {CLASS_NAMES[c]:>9}: g_na ×{g_na_scale:.2f}  g_k ×{g_k_scale:.2f}  e_l {e_l_shift:+.1f} мВ{flag}")
