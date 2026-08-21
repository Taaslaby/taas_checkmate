#!/usr/bin/env python3
"""
🦵 halfcenter_load_reset.py — нагрузка на ногу переключает half-center.

Гипотеза: антифаза возникает не сама по себе, а когда нога в stance-фазе
чувствует нагрузку (campaniform sensilla / хордотональный орган) — спайк
проприоцептора «щёлкает» GABA-пару и переключает фазу.

Проверка: 2 GABA-нейрона (half-center, взаимное торможение + rebound) крутят
антифазу свободно. Потом в момент, когда n0 активен (нога на земле), даём
короткий возбуждающий «load»-пинок в n1 (swing-нейрон). Смотрим: n1
спайкает сразу (фаза переключилась), и ритм перепривязывается к пинку.

Usage: python halfcenter_load_reset.py
"""

import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from taas_checkmate import connectome_brain as cb

DT = 0.05
PARAMS = {0: DT, 1: 1.0, 2: 0.10, 3: -60.0, 4: 0.4, 5: -80.0, 6: 0.8,
          7: -30.0, 8: 1.0 / 100.0, 9: 1.0 / 120.0, 10: -35.0, 11: 5.0,
          12: -75.0, 13: 6.0, 14: -40.0, 15: -55.0, 16: 0.02}
DRIVE = 2.0
W_INH = 18.0
TAU_SYN = 12.0
LOAD_PULSE = 30.0   # возбуждающий ток «нагрузки» в swing-нейрон n1
PULSE_MS = 8.0      # длительность пинка
T_PULSE = 60.0      # момент пинка (n0 активен)


def build():
    rp = np.array([0, 1, 2], dtype=np.int32)
    ci = np.array([1, 0], dtype=np.int32)
    vals = np.array([W_INH, W_INH], dtype=np.float32)
    nt = np.array([-1.0, -1.0], dtype=np.float32)
    conn = cb.ConnectomeData(rp, ci, vals, nt)
    runner = cb.BrainRunner(conn, model="haap_synaptic", dt_ms=DT, tau_syn=TAU_SYN)
    for col, val in PARAMS.items():
        runner.set_haap_param_column(col, np.full(2, val, dtype=np.float32))
    runner.set_v(np.array([-40.0, -80.0], dtype=np.float32))
    return runner


def run_free(total_ms):
    r = build()
    ext = np.full(2, DRIVE, dtype=np.float32)
    buf = np.asarray(r.run(int(total_ms / DT), ext)).reshape(-1, 2)
    return buf


def run_with_load(total_ms, t_pulse=T_PULSE):
    r = build()
    base = np.full(2, DRIVE, dtype=np.float32)
    load = np.array([DRIVE, DRIVE + LOAD_PULSE], dtype=np.float32)
    chunks = []
    # до пинка
    chunks.append(np.asarray(r.run(int(t_pulse / DT), base)).reshape(-1, 2))
    # пинок
    chunks.append(np.asarray(r.run(int(PULSE_MS / DT), load)).reshape(-1, 2))
    # после
    rem = total_ms - t_pulse - PULSE_MS
    chunks.append(np.asarray(r.run(int(rem / DT), base)).reshape(-1, 2))
    return np.vstack(chunks)


def spike_times(buf, neuron, t0=0.0):
    idx = np.where(buf[:, neuron])[0]
    return (t0 + idx * DT).tolist()


def main():
    print("=== СВОБОДНЫЙ half-center (без нагрузки) ===")
    free = run_free(250.0)
    for name, i in [("n0", 0), ("n1", 1)]:
        t = spike_times(free, i)
        print(f"  {name}: {[round(x, 1) for x in t[:16]]}")

    print(f"\n=== НАГРУЗКА в t={T_PULSE} мс (пинок в n1, +{LOAD_PULSE} ток, {PULSE_MS} мс) ===")
    load = run_with_load(250.0)
    for name, i in [("n0", 0), ("n1", 1)]:
        t = spike_times(load, i)
        print(f"  {name}: {[round(x, 1) for x in t[:16]]}")

    # фазовый сдвиг: первый спайк n1 после пинка vs в свободном режиме
    free_n1 = spike_times(free, 1)
    load_n1 = spike_times(load, 1)
    free_next = min([x for x in free_n1 if x > T_PULSE], default=None)
    load_next = min([x for x in load_n1 if x > T_PULSE], default=None)
    print(f"\n  свободный: n1 спайкнет после t={T_PULSE} в ~{free_next} мс")
    print(f"  с нагрузкой: n1 спайкнул в ~{load_next} мс")
    if free_next is not None and load_next is not None:
        print(f"  фазовый сдвиг: {free_next - load_next:.1f} мс раньше → фаза переключена нагрузкой")


if __name__ == "__main__":
    main()
