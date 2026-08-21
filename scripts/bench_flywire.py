#!/usr/bin/env python3
"""Прогон нашего движка на бенчмарке fly-brain (модель Шу): sugar GRNs 200 Hz.

Воспроизводим модель Shiu et al. LIF + alpha-синапс (приближение через наш
движок) на том же графе flywire_783.taas и сравниваем с их benchmark-results:
  Brian2 (CPU) 1s  → 16 567 spikes / 380 active / sim 2.65s (0.38x realtime)
  Brian2 (CPU) 10s → 170 940 spikes / 434 active / sim 31.06s (0.32x)
"""
import json
import os
import sys
import time

import numpy as np

from taas_checkmate import connectome_brain as cb
import pathlib, sys  # репо-рут для tools.taas_io при запуске из scripts/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.taas_io import load_taas

BASE = r"C:/Users/anton/AppData/Local/Temp/flybrain"
TAAS = os.path.join(BASE, "flywire_783.taas")
SUGAR = os.path.join(BASE, "sugar_grn.json")

# параметры модели Шу (run_pytorch.py MODEL_PARAMS / model.py default_params)
V_REST, V_TH, V_RESET = -52.0, -45.0, -52.0
TAU_MEM, TAU_REF = 20.0, 2.2
TAU_SYN, T_DELAY = 5.0, 1.8
DT_MS = 0.1
W_SYN = 0.275      # вес за синапс (мВ)
F_POI = 250.0      # масштаб Poisson
R_SUGAR = 200.0    # Гц


# парсер .taas — в tools/taas_io.py


def main():
    n, rp, ci, vals, _nt, _rt = load_taas(TAAS)
    print(f"граф: {n} нейронов, {len(ci)} синапсов")

    conn = cb.ConnectomeData(rp, ci, vals, np.ones(n, dtype=np.float32))
    runner = cb.BrainRunner(
        conn, model="lif_alpha", dt_ms=DT_MS,
        tau_ref=TAU_REF, tau_m=TAU_MEM,
        v_rest=V_REST, v_thresh=V_TH, v_reset=V_RESET,
        tau_syn=TAU_SYN, t_delay=T_DELAY,
    )

    with open(SUGAR) as f:
        sugar_idx = json.load(f)["sugar_grn"]
    print(f"сахарных GRN: {len(sugar_idx)}, стимул {R_SUGAR:.0f} Гц × {W_SYN * F_POI:.1f} мВ")

    CHUNK_S = 0.1  # 0.1s на вызов run() — буфер 1000×138K×4 = 0.55 GB
    for t_run in [0.1, 1.0, 10.0]:
        n_chunks = int(t_run / CHUNK_S)
        runner.reset()
        runner.clear_poisson_groups()
        # Сахарные GRN — без рефрактерности (rfc=0 в модели Шу)
        runner.set_alpha_refrac_ms(sugar_idx, 0.0)
        # Poisson-вход прямо в v (аналог target_var='v' у Шу): вес = w_syn * f_poi
        runner.add_poisson_group(sugar_idx, R_SUGAR, W_SYN * F_POI)
        ext = np.zeros(n, dtype=np.float32)
        chunk_steps = int(CHUNK_S * 1000.0 / DT_MS)
        t0 = time.perf_counter()
        total = 0
        active_mask = np.zeros(n, dtype=bool)
        for _ in range(n_chunks):
            buf = runner.run(chunk_steps, ext)
            spikes = np.asarray(buf).reshape(chunk_steps, n)
            total += int(spikes.sum())
            active_mask |= spikes.sum(axis=0) > 0
        sim_t = time.perf_counter() - t0
        ratio = t_run / sim_t if sim_t > 0 else float("inf")
        print(f"t={t_run:>5.1f}s: {total:>8} спайков, {active_mask.sum():>5} активных, "
              f"sim={sim_t:7.2f}s, x{ratio:.2f} реального времени")


if __name__ == "__main__":
    main()
