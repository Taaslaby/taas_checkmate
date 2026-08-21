#!/usr/bin/env python3
"""
⚡ vnc_rhythm_test.py — «голая VNC генерит трипод?»

Эксперимент: подаём на MANC VNC (102K) rebound-профиль интернейронов,
подписываем торможение по predictedNt (в .taas его не было!), даём
кратковременный пинок — и смотрим, кластеризуются ли моторные нейроны (MN)
в антифазные группы (зародыш треножного шага).

Анализ:
  - ритм: FFT популяционной частоты MN → есть ли доминирующая частота?
  - антифаза: фаза каждого MN относительно ритма; R1=|⟨e^{iφ}⟩| (синхронность),
    R2=|⟨e^{2iφ}⟩| (антифаза). R2 > R1 ⇒ две группы в противофазе.

Usage: python vnc_rhythm_test.py
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

KICK = 1000          # шагов пинка (50 мс)
TOTAL = 15000        # всего шагов (750 мс)
CHUNK = 500          # размер чанка (память)
DRIVE = 15.0         # ток пинка, µA
N_KICK = 2000        # сколько central пнуть


# парсеры .taas/.haap — в tools/taas_io.py


def main():
    n, rp, ci, vals0, nt, _rt = load_taas(TAAS)

    # ── знак торможения по predictedNt ──
    p = pd.read_feather(FEATHER)
    gaba = (p["predictedNt"].values == "gaba")
    sign = np.where(gaba, -1.0, 1.0).astype(np.float32)
    print(f"GABA (тормозные): {gaba.sum()} / {n}")

    # веса: свернуть знак в values (source-нейрон = строка CSR)
    deg = (rp[1:] - rp[:-1]).astype(np.int64)
    source = np.repeat(np.arange(n), deg)
    vals = (vals0 / NORM) * sign[source]

    central = np.where(nt == 3.0)[0]
    motor = np.where(nt == 4.0)[0]
    print(f"central={len(central)} (GABA среди них: {gaba[central].sum()}), motor={len(motor)}")

    haap = load_haap(HAAP, n)

    # ── rebound-профиль central: E_L↑, g_h↑, delta_u↑ (half-center substrate) ──
    conn = cb.ConnectomeData(rp, ci, vals, nt)
    runner = cb.BrainRunner(conn, model="haap", dt_ms=DT)
    for col in range(1, 17):
        runner.set_haap_param_column(col, haap[col])
    overrides = {3: -55.0, 4: 0.30, 6: 0.50, 9: 1.0 / 200.0, 16: 0.04}
    for col, val in overrides.items():
        tmp = haap[col].copy()
        tmp[central] = val
        runner.set_haap_param_column(col, tmp)

    # ── пинок: DRIVE на случайные central первые KICK шагов ──
    rng = np.random.default_rng(7)
    kick_idx = rng.choice(central, N_KICK, replace=False)
    ext_drive = np.zeros(n, dtype=np.float32)
    ext_drive[kick_idx] = DRIVE
    zero = np.zeros(n, dtype=np.float32)

    mn_spikes = []
    central_rate = []
    print(f"запуск: {TOTAL} шагов ({TOTAL*DT/1000:.0f} мс), пинок первые {KICK*DT/1000:.0f} мс ...", flush=True)
    for start in range(0, TOTAL, CHUNK):
        n_steps = min(CHUNK, TOTAL - start)
        ext = ext_drive if start < KICK else zero
        buf = runner.run(n_steps, ext)
        s = np.asarray(buf).reshape(n_steps, n)
        mn_spikes.append(s[:, motor].copy())
        central_rate.append(s[:, central].sum(axis=1).astype(np.float32))
        if start % 3000 == 0:
            print(f"  {start:>5}/{TOTAL}", flush=True)

    mn = np.vstack(mn_spikes)            # (TOTAL, n_mn)
    cr = np.concatenate(central_rate)    # (TOTAL,)

    # ── анализ ──
    analyze(mn, cr, motor)


def analyze(mn, cr, motor):
    T, n_mn = mn.shape
    print(f"\n==== АНАЛИЗ ({T} шагов, {n_mn} MN) ====")

    # 1. популяционная частота MN (сглаживание 1 мс = 20 шагов)
    win = np.ones(20) / 20
    mn_rate = np.convolve(mn.sum(axis=1).astype(np.float32), win, mode="same")
    cr_rate = np.convolve(cr, win, mode="same")

    # ритм: FFT популяционной MN-частоты (убираем тренд)
    sig = mn_rate - mn_rate.mean()
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(T, d=DT / 1000.0)  # Гц
    # исключаем DC и ищем пик в 0.5..50 Гц (биологический диапазон)
    band = (freqs >= 0.5) & (freqs <= 50.0)
    if band.sum() > 0:
        f0 = freqs[band][np.argmax(spec[band])]
        pk = spec[band].max()
        # «сила ритма»: пик vs медиана спектра
        base = np.median(spec[band])
        rhythm_strength = pk / max(base, 1e-9)
        print(f"  доминирующая частота MN: {f0:.2f} Гц  (сила пика {rhythm_strength:.1f}× медианы)")
    else:
        f0, rhythm_strength = None, 0.0
        print("  ритм не найден")

    # 2. фаза каждого MN относительно ритма
    if f0 is not None:
        t_axis = np.arange(T) * DT / 1000.0
        phases = []
        for i in range(n_mn):
            sp_t = t_axis[mn[:, i] > 0]
            if len(sp_t) < 3:
                continue
            z = np.exp(2j * np.pi * f0 * sp_t)
            phases.append(np.angle(z.mean()))
        phases = np.array(phases)
        n_used = len(phases)
        R1 = np.abs(np.exp(1j * phases).mean())
        R2 = np.abs(np.exp(2j * phases).mean())
        print(f"\n  MN с ≥3 спайками: {n_used}/{n_mn}")
        print(f"  R1 (синхронность)   = {R1:.3f}")
        print(f"  R2 (антифаза 180°)  = {R2:.3f}")
        if R2 > R1 + 0.05:
            print("  ⇒ ЕСТЬ антифазные группы (зародыш трипода) 🎉")
        elif R1 > 0.5:
            print("  ⇒ MN синхронны (один кластер, антифазы нет)")
        else:
            print("  ⇒ фазы размазаны — структуры нет")

        # гистограмма фаз (8 бинов)
        hist, _ = np.histogram(phases % (2 * np.pi), bins=8, range=(0, 2 * np.pi))
        print("  фазовая гистограмма:", " ".join(f"{int(h):>3}" for h in hist))
    else:
        print("  (антифаза не оценивалась — ритма нет)")


if __name__ == "__main__":
    main()
