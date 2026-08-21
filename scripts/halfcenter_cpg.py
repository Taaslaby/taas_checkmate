#!/usr/bin/env python3
"""
🌀 halfcenter_cpg.py — реципрокные GABA-пары MANC → half-center CPG.

Что выяснилось (и теперь починено):
  1. В `.taas` торможение не подписано — но `predictedNt == "gaba"` в атласе
     есть, и мы строим GABA-подпись на лету.
  2. Instant-current синапс = одиночный импульс на 1 шаг (0.05 мс) — слишком
     слабый, чтобы держать half-center. Починено: у HAAP появился
     `haap_synaptic` с затухающим синаптическим током (τ_syn), и теперь
     GABA-ингибирование реально гиперполяризует цель на несколько мс.
  3. Конвенция знака: `nt_types[pre]` — нейротрансмиттер ИСТОЧНИКА
     (+1 возб., −1 торм.), а `values` положительные.

Half-center = взаимное торможение + rebound (h-ток) + адаптация (M-ток).
При нарушенной симметрии (как шум в реальных нейронах) два нейрона уходят
в ПРОТИВОФАЗУ — бурстами по очереди.

Usage: python halfcenter_cpg.py
"""

import struct
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

DT = 0.05
FEATHER = r"D:/connectoms/datasets/v1.0_manc-v1.0-neuron-properties.feather"
TAAS = "resources/data/manc_vnc.taas"
TOTAL = 40000  # 2000 мс

# Тюнинг half-center (найден свипом): rebound + адаптация + сильное GABA-ингибирование.
PARAMS = {0: DT, 1: 1.0, 2: 0.10, 3: -60.0, 4: 0.4, 5: -80.0, 6: 0.8,
          7: -30.0, 8: 1.0 / 100.0, 9: 1.0 / 120.0, 10: -35.0, 11: 5.0,
          12: -75.0, 13: 6.0, 14: -40.0, 15: -55.0, 16: 0.02}
DRIVE = 2.0
W_INH = 18.0
TAU_SYN = 12.0


def load_taas(path):
    d = open(path, "rb").read()
    n, syn = struct.unpack_from("<QQ", d, 8)
    off = 24
    rp = np.frombuffer(d, "<i4", count=n + 1, offset=off).astype(np.int32); off += 4 * (n + 1)
    ci = np.frombuffer(d, "<i4", count=syn, offset=off).astype(np.int32); off += 4 * syn
    vals = np.frombuffer(d, "<f4", count=syn, offset=off).astype(np.float32); off += 4 * syn
    nt = np.frombuffer(d, "<f4", count=n, offset=off).astype(np.float32)
    return n, rp, ci, vals, nt


def find_reciprocal_pairs(source, ci, vals, gaba):
    """Взаимотормозящие GABA-пары (a,b): a→b и b→a, оба GABA. Возвращает
    [(a, b, w_ab, w_ba), ...] — индексы и средние веса обоих направлений."""
    g2g = gaba[source] & gaba[ci]
    src, dst = source[g2g], ci[g2g]
    wgt = vals[g2g]
    N = int(gaba.sum())
    gaba_pos = np.full(len(gaba), -1, dtype=np.int64)
    gaba_idx = np.where(gaba)[0]
    gaba_pos[gaba_idx] = np.arange(N)
    s = gaba_pos[src].astype(np.int64)
    t = gaba_pos[dst].astype(np.int64)
    edges = {}
    for i in range(len(s)):
        key = (int(s[i]), int(t[i]))
        edges.setdefault(key, []).append(float(wgt[i]))
    pairs = []
    for (a, b), wl in edges.items():
        if a < b and (b, a) in edges:
            w_ab = float(np.mean(wl))
            w_ba = float(np.mean(edges[(b, a)]))
            pairs.append((int(gaba_idx[a]), int(gaba_idx[b]), w_ab, w_ba))
    return pairs


def halfcenter_run(w_inh=W_INH, drive=DRIVE, tau_syn=TAU_SYN, total=TOTAL, init=(-40.0, -80.0)):
    """2-нейронный half-center: 0⇄1 взаимное GABA-торможение (τ_syn затухание)."""
    rp = np.array([0, 1, 2], dtype=np.int32)
    ci = np.array([1, 0], dtype=np.int32)
    vals = np.array([w_inh, w_inh], dtype=np.float32)
    nt = np.array([-1.0, -1.0], dtype=np.float32)  # оба источника — GABA
    conn = cb.ConnectomeData(rp, ci, vals, nt)
    runner = cb.BrainRunner(conn, model="haap_synaptic", dt_ms=DT, tau_syn=tau_syn)
    for col, val in PARAMS.items():
        runner.set_haap_param_column(col, np.full(2, val, dtype=np.float32))
    runner.set_v(np.array(init, dtype=np.float32))  # нарушаем симметрию (как шум)
    ext = np.full(2, drive, dtype=np.float32)
    buf = np.asarray(runner.run(total, ext)).reshape(total, 2)
    return buf


def antiphase_metrics(buf, smooth_ms=40.0):
    s0 = buf[:, 0].astype(np.float32)
    s1 = buf[:, 1].astype(np.float32)
    win = np.ones(int(smooth_ms / DT)) / int(smooth_ms / DT)
    a = np.convolve(s0, win, mode="same")
    b = np.convolve(s1, win, mode="same")
    a = a - a.mean(); b = b - b.mean()
    corr = np.corrcoef(a, b)[0, 1]
    r0 = s0.sum() / len(s0) / (DT / 1000)
    r1 = s1.sum() / len(s1) / (DT / 1000)
    # доля шагов, где ровно один активен
    xor = (s0.astype(bool) != s1.astype(bool)).mean()
    # период бурстов по FFT суммы
    sig = a + b
    spec = np.abs(np.fft.rfft(sig))
    fr = np.fft.rfftfreq(len(sig), d=DT / 1000)
    band = (fr >= 0.5) & (fr <= 30.0)
    f0 = fr[band][np.argmax(spec[band])] if band.any() else 0.0
    return corr, r0, r1, f0, xor


def main():
    n, rp, ci, vals0, nt = load_taas(TAAS)
    p = pd.read_feather(FEATHER)
    gaba = (p["predictedNt"].values == "gaba")
    deg = (rp[1:] - rp[:-1]).astype(np.int64)
    source = np.repeat(np.arange(n), deg)

    pairs = find_reciprocal_pairs(source, ci, vals0, gaba)
    print(f"реципрокных GABA-пар в MANC: {len(pairs)}")
    for a, b, w_ab, w_ba in pairs[:6]:
        print(f"  {a} ⇄ {b}   w_ab={w_ab:.2f}  w_ba={w_ba:.2f}")

    print(f"\n[half-center] 2 GABA-нейрона, взаимное торможение τ_syn={TAU_SYN} мс, "
          f"rebound g_h={PARAMS[6]}, адаптация g_M={PARAMS[4]}")
    buf = halfcenter_run()
    corr, r0, r1, f0, xor = antiphase_metrics(buf)
    verdict = "АНТИФАЗА" if corr < -0.3 else ("синхрон" if corr > 0.3 else "размазано")
    print(f"  corr={corr:+.3f}  r0={r0:5.1f} r1={r1:5.1f} Гц  "
          f"f0={f0:.2f} Гц  XOR={xor:.3f}  → {verdict}")

    # спайк-тайминги первых 250 мс (показать бурстовую альтернацию)
    for name, s in [("n0", buf[:, 0]), ("n1", buf[:, 1])]:
        times = np.where(s[:5000])[0] * DT
        print(f"  {name}: {[round(t, 1) for t in times[:20]]}")


if __name__ == "__main__":
    main()
