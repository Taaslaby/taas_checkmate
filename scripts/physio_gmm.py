#!/usr/bin/env python3
"""
🍪 physio_gmm.py — «куки каттер» по литературной электрофизиологии
                  + плотный генеративный поиск + бисекция по g_M.

1. Фит GMM (diag-cov, лог-масштаб) на 17 типах атласа → непрерывное
   многообразие биологически правдоподобных HAAP-параметров.
2. Плотный поиск: --k кандидатов (смесь motor-якорь + глобально), скор
   |motor_rate − target|² + штраф за срыв сети.
3. SPSA-дожим (опционально) вокруг лучшего.
4. Биссекция по g_M (доминирующий рычаг, монотонно): точный выход на target.

Оптимизация: ConnectomeData + BrainRunner строятся ОДИН раз; между эвалами
reset() + повторная заливка колонок (без повторного клона графа 50 МБ).

Usage:
    python physio_gmm.py --target-motor 25 --k 300 --bisect
"""

import argparse
import os
import pickle
import struct
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from sklearn.mixture import GaussianMixture

from taas_checkmate import connectome_brain as cb
import pathlib, sys  # репо-рут для tools.taas_io при запуске из scripts/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.taas_io import load_haap, load_mod, load_taas

# ──────────────────────────────────────────────────────────────────
#  Порядок HAAParamIdx: 0=DT 1=Inv_Cm 2=G_L 3=E_L 4=G_M 5=E_K 6=G_H 7=E_H
#                        8=Inv_Tau_U 9=Inv_Tau_H 10=V_Half_U 11=K_U
#                        12=V_Half_H 13=K_H 14=V_Thresh 15=V_Reset 16=Delta_U
N_HAAP = 17
DT = 0.05
NORM = 256.0
COL_G_M = 4

LOG_COLS = [1, 2, 4, 6, 8, 9, 11, 13]

BOUNDS = [
    (DT, DT), (0.1, 10.0), (0.01, 2.0), (-80.0, -45.0), (0.0, 3.0),
    (-95.0, -70.0), (0.0, 2.0), (-45.0, -10.0), (0.001, 0.1), (0.001, 0.05),
    (-60.0, -20.0), (1.0, 15.0), (-90.0, -55.0), (1.0, 15.0),
    (-55.0, -30.0), (-70.0, -45.0), (0.0, 0.2),
]


# load_mod/load_taas/load_haap — в tools/taas_io.py


def atlas_to_haap_matrix(atlas):
    types = sorted(atlas.VNC_ATLAS.keys()) + [100, 101]
    M = np.zeros((len(types), N_HAAP), dtype=np.float32)
    for i, sc in enumerate(types):
        nt = atlas.get_motor_subtype_haap(sc) if sc in (100, 101) else atlas.VNC_ATLAS[sc]
        M[i, 0] = DT
        M[i, 1] = 1.0 / max(nt.cm, 1e-6)
        M[i, 2] = nt.g_L
        M[i, 3] = nt.E_L
        M[i, 4] = nt.g_M
        M[i, 5] = nt.E_K
        M[i, 6] = nt.g_h
        M[i, 7] = nt.E_h
        M[i, 8] = 1.0 / max(nt.tau_u, 1e-6)
        M[i, 9] = 1.0 / max(nt.tau_h, 1e-6)
        M[i, 10] = nt.V_half_u
        M[i, 11] = nt.k_u
        M[i, 12] = nt.V_half_h
        M[i, 13] = nt.k_h
        M[i, 14] = nt.V_thresh
        M[i, 15] = nt.V_reset
        M[i, 16] = nt.delta_u
    return types, M


def encode(M):
    X = M[:, 1:].astype(np.float64).copy()
    for j in LOG_COLS:
        X[:, j - 1] = np.log(np.maximum(X[:, j - 1], 1e-9))
    return X


def decode(x):
    out = np.zeros(N_HAAP, dtype=np.float32)
    out[0] = DT
    y = x.astype(np.float64).copy()
    for j in LOG_COLS:
        y[j - 1] = np.exp(y[j - 1])
    out[1:] = y.astype(np.float32)
    return out


def clamp_cols(cols):
    c = cols.copy()
    for i in range(N_HAAP):
        c[i] = float(np.clip(c[i], BOUNDS[i][0], BOUNDS[i][1]))
    return c


class MotorEvaluator:
    def __init__(self, rp, ci, vals, nt, haap, motor, sens, n_steps):
        self.conn = cb.ConnectomeData(rp, ci, vals, nt)
        self.runner = cb.BrainRunner(self.conn, model="haap", dt_ms=DT)
        self.runner.add_poisson_group(sens, 500.0, 200.0)
        self.haap = haap
        self.motor = motor
        self.n = nt.shape[0]
        self.n_steps = n_steps

    def eval(self, motor_cols):
        self.runner.reset()
        for col in range(1, N_HAAP):
            self.runner.set_haap_param_column(col, self.haap[col])
        for col in range(1, N_HAAP):
            tmp = self.haap[col].copy()
            tmp[self.motor] = motor_cols[col]
            self.runner.set_haap_param_column(col, tmp)
        buf = self.runner.run(self.n_steps, np.zeros(self.n, dtype=np.float32))
        s = np.asarray(buf).reshape(self.n_steps, self.n)
        mr = s[:, self.motor].sum() / len(self.motor) / (self.n_steps * DT / 1000.0)
        return mr, int(s.sum())


def objective(mr, total, target):
    err = (mr - target) / max(target, 1.0)
    blowup = max(0.0, total / 150000.0 - 1.0)
    return err * err + 0.05 * blowup * blowup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=r"C:/Users/anton/Desktop/neuron_atlas.py")
    ap.add_argument("--taas", default="resources/data/manc_vnc.taas")
    ap.add_argument("--haap", default="resources/data/manc_vnc_haap_physio.haap")
    ap.add_argument("--out", default="resources/data")
    ap.add_argument("--out-name", default="manc_vnc_haap_gmm_best.haap")
    ap.add_argument("--target-motor", type=float, default=25.0)
    ap.add_argument("--k", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.6)
    ap.add_argument("--components", type=int, default=6)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--gmm-path", default="resources/data/physio_gmm.pkl")
    ap.add_argument("--spsa-iter", type=int, default=0)
    ap.add_argument("--spsa-a0", type=float, default=0.03)
    ap.add_argument("--spsa-c0", type=float, default=0.06)
    ap.add_argument("--bisect", action="store_true")
    ap.add_argument("--bisect-iter", type=int, default=16)
    ap.add_argument("--no-search", action="store_true")
    args = ap.parse_args()

    atlas = load_mod(args.atlas, "neuron_atlas")
    types, M = atlas_to_haap_matrix(atlas)
    X = encode(M)
    n_types = len(types)

    nc = min(args.components, n_types)
    gmm = GaussianMixture(n_components=nc, covariance_type="diag",
                          random_state=42, reg_covar=1e-6, n_init=8)
    gmm.fit(X)
    with open(args.gmm_path, "wb") as f:
        pickle.dump(gmm, f)
    print(f"[fit] GMM: {nc} компонент × {X.shape[1]} фич ← {n_types} типов → {args.gmm_path}")

    resp = gmm.predict_proba(X)
    print("\n[clusters] тип → компонента:")
    for i, sc in enumerate(types):
        top = int(np.argmax(resp[i]))
        name = (atlas.get_motor_subtype_haap(sc) if sc in (100, 101) else atlas.VNC_ATLAS[sc]).name
        print(f"  {sc:>3} {name:35s} c{top} ({resp[i][top]:.2f})")

    if args.no_search:
        return

    n, rp, ci, vals0, nt, _rt = load_taas(args.taas)
    vals = (vals0 / NORM).astype(np.float32)
    haap = load_haap(args.haap, n)
    motor = np.where(nt == 4.0)[0]
    sens = np.where((nt == 2.0) | (nt == 10.0))[0].tolist()
    assert len(motor) > 0, "нет motor (nt==4)"
    ev = MotorEvaluator(rp, ci, vals, nt, haap, motor, sens, args.n_steps)
    print(f"\n[search] motor={len(motor)}, sens={len(sens)}, target={args.target_motor} Гц, k={args.k}")

    motor_anchor = encode(M[types.index(10):types.index(10) + 1])[0]
    comp_probs = gmm.predict_proba(motor_anchor[None, :])[0]
    c_dist = 0.6 * (comp_probs / comp_probs.sum()) + 0.4 * (gmm.weights_ / gmm.weights_.sum())

    rng = np.random.default_rng(42)
    # candidate = (score, motor_rate, total, cols_decoded, x_encoded)
    best = (np.inf, None, None, None, None)

    for k in range(args.k):
        c = int(rng.choice(nc, p=c_dist))
        z = rng.normal(size=X.shape[1]) * np.sqrt(gmm.covariances_[c]) * args.temperature
        x = gmm.means_[c] + z
        cols = clamp_cols(decode(x))
        mr, total = ev.eval(cols)
        sc = objective(mr, total, args.target_motor)
        if sc < best[0]:
            best = (sc, mr, total, cols, x)
        if k % 50 == 0 or k == args.k - 1:
            print(f"  [{k+1:>3}/{args.k}] motor={mr:6.1f} Hz total={total:6d} score={sc:.3f} "
                  f"best={best[1]:.1f} Hz", flush=True)

    print(f"\n[sample-best] motor={best[1]:.1f} Hz  total={best[2]}  score={best[0]:.4f}")

    # ─── 3. SPSA (опционально) ───
    spsa_best = best
    if args.spsa_iter > 0:
        theta = best[4].astype(np.float64).copy()
        lo = X.min(axis=0) - 1.5
        hi = X.max(axis=0) + 1.5
        alpha, gamma = 0.602, 0.101
        print(f"\n[spsa] {args.spsa_iter} итераций от motor={best[1]:.1f} Гц ...")
        for it in range(args.spsa_iter):
            ak = args.spsa_a0 / (it + 1) ** alpha
            ck = args.spsa_c0 / (it + 1) ** gamma
            delta = np.where(rng.random(X.shape[1]) < 0.5, -1.0, 1.0)
            xp = np.clip(theta + ck * delta, lo, hi)
            xm = np.clip(theta - ck * delta, lo, hi)
            mrp, totp = ev.eval(clamp_cols(decode(xp)))
            mrm, totm = ev.eval(clamp_cols(decode(xm)))
            lp = objective(mrp, totp, args.target_motor)
            lm = objective(mrm, totm, args.target_motor)
            grad = (lp - lm) / (2.0 * ck) * (1.0 / delta)
            theta = np.clip(theta - ak * grad, lo, hi)
            for cand in ((lp, mrp, totp, clamp_cols(decode(xp)), xp),
                         (lm, mrm, totm, clamp_cols(decode(xm)), xm)):
                if cand[0] < spsa_best[0]:
                    spsa_best = cand
            if it % 5 == 0 or it == args.spsa_iter - 1:
                print(f"  it={it+1:>2} motor={mrp:6.1f}/{mrm:6.1f} Hz best={spsa_best[1]:.1f} Hz", flush=True)

    # ─── 4. Координатный спуск по 4 доминирующим рычагам ───
    # g_M сам упирается в плато ~18 Гц — нужно ещё E_L (деполяризация),
    # delta_u (адаптация) и V_thresh (порог). Сканируем каждый рычаг сеткой.
    bisect_best = best
    if args.bisect:
        base = spsa_best if spsa_best[0] < best[0] else best
        base_cols = base[3].copy()
        LEVER_NAMES = {4: "g_M", 3: "E_L", 16: "delta_u", 14: "V_thresh"}
        levers = [
            (COL_G_M, 0.0, 0.55),   # 4  g_M
            (3, -65.0, -50.0),      # 3  E_L
            (16, 0.0, 0.04),        # 16 delta_u
            (14, -45.0, -35.0),     # 14 V_thresh
        ]
        all_evals = []
        print(f"\n[coordinate-descent] 4 рычага от base motor={base[1]:.1f} Гц ...")
        for col, lo, hi in levers:
            best_local = None
            pts = np.linspace(lo, hi, 9)
            for val in pts:
                cols = base_cols.copy()
                cols[col] = float(val)
                mr, total = ev.eval(clamp_cols(cols))
                sc = objective(mr, total, args.target_motor)
                cand = (sc, mr, total, cols, None)
                all_evals.append(cand)
                if best_local is None or sc < best_local[0]:
                    best_local = cand
            base_cols = best_local[3].copy()
            print(f"  {LEVER_NAMES[col]:>8}: best={best_local[1]:6.1f} Hz @ {best_local[3][col]:.3f}",
                  flush=True)
        bisect_best = min(all_evals, key=lambda e: e[0])

    final = min([best, spsa_best, bisect_best], key=lambda e: e[0])
    cols_best = final[3]
    print(f"\n==== ИТОГ ====")
    print(f"  motor = {final[1]:.1f} Гц (target {args.target_motor})  total={final[2]}  score={final[0]:.4f}")
    print(f"  g_M={cols_best[4]:.3f}  g_h={cols_best[6]:.3f}  delta_u={cols_best[16]:.3f}  "
          f"E_L={cols_best[3]:.0f}  V_thresh={cols_best[14]:.0f}")

    # ─── 5. Перезапечь лучшего ───
    out = np.zeros((N_HAAP, n), dtype=np.float32)
    for col in range(N_HAAP):
        out[col] = haap[col]
    for col in range(1, N_HAAP):
        out[col][motor] = cols_best[col]
    opath = os.path.join(args.out, args.out_name)
    with open(opath, "wb") as f:
        f.write(struct.pack("<IQ", 0x48414131, n))
        for col in range(N_HAAP):
            f.write(out[col].tobytes())
    print(f"\n[save] {opath}")


if __name__ == "__main__":
    main()
