#!/usr/bin/env python3
"""
🔧 vnc_link_calibration.py — «разобрать VNC на запчасти и найти веса».

Изолируем звено central(IN)→motor(MN): обнуляем ВСЕ синапсы кроме
central→motor, гоняем central тоником, и бисекцией по масштабу весов
находим, при каком множителе motor выходит на целевой rate.

Это ответ на «по простому проходу найти веса»: вместо запечённого /256
меряем forward и калибруем.

Usage: python vnc_link_calibration.py --target 20
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

DT = 0.05
NORM = 256.0
FEATHER = r"D:/connectoms/datasets/v1.0_manc-v1.0-neuron-properties.feather"
TAAS = "resources/data/manc_vnc.taas"
HAAP = "resources/data/manc_vnc_haap_physio.haap"

N_STEPS = 3000      # 150 мс — достаточно для стабилизации central
CENTRAL_DRIVE = 8.0  # µA тонический ток на central


# парсеры .taas/.haap — в tools/taas_io.py


def main():
    target = 20.0
    if "--target" in sys.argv:
        target = float(sys.argv[sys.argv.index("--target") + 1])
    n, rp, ci, vals0, nt, _rt = load_taas(TAAS)
    p = pd.read_feather(FEATHER)
    gaba = (p["predictedNt"].values == "gaba")
    sign = np.where(gaba, -1.0, 1.0).astype(np.float32)
    deg = (rp[1:] - rp[:-1]).astype(np.int64)
    source = np.repeat(np.arange(n), deg)
    central = np.where(nt == 3.0)[0]
    motor = np.where(nt == 4.0)[0]

    # ── изоляция: только central→motor ──
    is_cm = (nt[source] == 3.0) & (nt[ci] == 4.0)
    print(f"central→motor синапсов: {is_cm.sum()} / {len(ci)} (остальные обнуляем)")
    vals = np.zeros_like(vals0)
    vals[is_cm] = (vals0[is_cm] / NORM) * sign[source[is_cm]]  # торможение + нормировка

    haap = load_haap(HAAP, n)

    def build_runner(scale):
        conn = cb.ConnectomeData(rp, ci, (vals * scale).astype(np.float32), nt)
        runner = cb.BrainRunner(conn, model="haap", dt_ms=DT)
        for col in range(1, 17):
            runner.set_haap_param_column(col, haap[col])
        return runner

    # тонический драйв central
    ext = np.zeros(n, dtype=np.float32)
    ext[central] = CENTRAL_DRIVE

    def motor_rate(scale):
        runner = build_runner(scale)
        buf = runner.run(N_STEPS, ext)
        s = np.asarray(buf).reshape(N_STEPS, n)
        mr = s[-1000:, motor].sum() / len(motor) / (1000 * DT / 1000.0)  # последние 50 мс
        cr = s[-1000:, central].sum() / len(central) / (1000 * DT / 1000.0)
        return mr, cr

    # сперва грубый свип
    print("\n[свип] scale → motor rate (central drive = %.0f µA):" % CENTRAL_DRIVE)
    for s in [1.0, 4.0, 16.0, 64.0, 256.0]:
        mr, cr = motor_rate(s)
        print(f"  scale={s:>5.0f}  central={cr:6.1f} Гц  motor={mr:6.1f} Гц")

    # бисекция по scale (rate монотонно растёт с scale)
    lo, hi = 1.0, 1024.0
    best = None
    print(f"\n[бисекция] target motor={target} Гц ...")
    for it in range(16):
        mid = (lo + hi) / 2.0
        mr, cr = motor_rate(mid)
        cand = (abs(mr - target), mr, cr, mid)
        if best is None or cand[0] < best[0]:
            best = cand
        if mr < target:
            lo = mid
        else:
            hi = mid
        print(f"  scale={mid:7.1f}  central={cr:6.1f}  motor={mr:6.1f} Гц")

    print(f"\n==== ИТОГ ====")
    print(f"  найденный масштаб central→motor: {best[3]:.1f}× (против запечённого 1× = /{NORM:.0f})")
    print(f"  при нём central={best[2]:.1f} Гц → motor={best[1]:.1f} Гц (target {target})")


if __name__ == "__main__":
    main()
