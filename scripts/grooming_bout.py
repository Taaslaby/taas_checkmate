#!/usr/bin/env python3
"""
🧹 grooming_bout.py — wing grooming bout в MANC (нервный шнур).

Стимулируем wing margin bristle (краевые щетинки крыла, 725 тактильных
нейронов) и смотрим, как активность через интернейроны доходит до
мотонейронов крыла и задних/средних ног (тот самый wing grooming).

В отличие от compare_manc_haap, здесь:
  1. Торможение подписано по predictedNt (GABA/glutamate → −1, ACh → +1).
  2. Параметры — GMM-откалиброванные (motor может стрелять).
  3. Стимулируется ТОЛЬКО брисли, не все сенсорные.

Usage: python grooming_bout.py
"""

import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from taas_checkmate import connectome_brain as cb
import pathlib, sys  # репо-рут для tools.taas_io при запуске из scripts/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.taas_io import load_haap, load_taas

TAAS = "resources/data/manc_vnc.taas"
HAAP = "resources/data/manc_vnc_haap_gmm_best.haap"
FEATHER = r"D:/connectoms/datasets/v1.0_manc-v1.0-neuron-properties.feather"
DT = 0.05
N_STEPS = 2000          # 100 мс
POISSON_RATE = 300.0
POISSON_WEIGHT = 150.0
NORM = 256.0

INHIBITORY = {"gaba", "glutamate"}   # у насекомых глутамат тормозной (GluCl)


# парсеры .taas/.haap — в tools/taas_io.py


def main():
    n, rp, ci, vals0, nt_superclass, _rt = load_taas(TAAS)
    p = pd.read_feather(FEATHER)

    # 1. sign weights by predictedNt (inhibition!)
    pnt = p["predictedNt"].astype(str).fillna("unknown").values
    sign = np.where(np.isin(pnt, list(INHIBITORY)), -1.0, 1.0).astype(np.float32)
    vals = (vals0 / NORM).astype(np.float32)

    # 2. wing margin bristle (stimulated) + grooming circuit markers
    subclass = p["subclass"].astype(str).values
    klass = p["class"].astype(str).values
    target = p["target"].astype(str).values
    typ = p["type"].astype(str).values

    bristle = np.where(subclass == "wing margin bristle")[0]
    mn = np.where(klass == "motor neuron")[0]
    wing_mn = mn[np.array(["wing" in str(target[i]) for i in mn])]
    hind_mn = mn[np.array(["hind leg" in str(target[i]) for i in mn])]
    # grooming candidate interneurons (из трассировки: IN06B032, IN27X001, IN06B025, IN19A016…)
    groom_in = np.where(np.isin(typ, ["IN06B032", "IN27X001", "IN06B025", "IN19A016"]))[0]
    print(f"wing margin bristle: {len(bristle)} | grooming IN: {len(groom_in)} | "
          f"wing MN: {len(wing_mn)} | hind-leg MN: {len(hind_mn)}")

    # 3. build signed connectome + HAAP runner (synaptic decay — иначе
    #    мгновенный ток не копится и сигнал не доходит до интернейронов)
    haap = load_haap(HAAP, n)
    conn = cb.ConnectomeData(rp, ci, vals, sign)
    runner = cb.BrainRunner(conn, model="haap_synaptic", dt_ms=DT, tau_syn=10.0)
    for col in range(1, 17):
        runner.set_haap_param_column(col, haap[col])

    def run_bout(stimulate: bool):
        runner.reset()
        for col in range(1, 17):
            runner.set_haap_param_column(col, haap[col])
        runner.clear_poisson_groups()
        if stimulate:
            runner.add_poisson_group(bristle.tolist(), POISSON_RATE, POISSON_WEIGHT)
        buf = runner.run(N_STEPS, np.zeros(n, dtype=np.float32))
        return np.asarray(buf).reshape(N_STEPS, n)

    def rates(spikes, idx):
        if len(idx) == 0:
            return 0.0
        return spikes[:, idx].sum() / len(idx) / (N_STEPS * DT / 1000.0)

    print("\n=== grooming bout: стимуляция wing margin bristle ===")
    base = run_bout(stimulate=False)
    stim = run_bout(stimulate=True)

    print(f"{'популяция':<28} {'baseline':>9} {'bristle+':>9}")
    for name, idx in [("wing margin bristle", bristle),
                      ("grooming IN (IN06B032…) ", groom_in),
                      ("wing motor", wing_mn),
                      ("hind-leg motor", hind_mn)]:
        print(f"{name:<28} {rates(base, idx):8.1f} {rates(stim, idx):8.1f} Гц")

    # сколько всего спайков и в каких классах
    print(f"\ntotal spikes: baseline={base.sum()}  bristle+={stim.sum()}")
    # top active neurons under stimulation
    per = stim.sum(axis=0)
    top = np.argsort(per)[::-1][:8]
    print("top-8 активных нейронов под стимуляцией:")
    for i in top:
        print(f"  idx={i} type={typ[i]} class={klass[i]} target={str(target[i])[:40]}  "
              f"{per[i]} спайков")


if __name__ == "__main__":
    main()
