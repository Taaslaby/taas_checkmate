#!/usr/bin/env python3
"""
🧬 bake_haap.py — литературная HAAP-физиология для MANC (нервный шнур мухи).

В отличие от bake_physio.py (HH: NaV/KV + leak), здесь мы НЕ конвертируем
атлас в HH, а пишем его как есть — в нативный формат HAAP-модели pillars
(M-ток адаптация + h-ток rebound). Именно адаптация (M-ток) должна снять
разгон motor-нейронов до 477 Гц, который давал HH.

Берёт типы нейронов MANC (DN/SN/MN/IN/AN/SA из v1.0_manc properties),
маппит их на суперклассы атласа (neuron_atlas.py) и пишет per-neuron
HAAP-параметры в 17 столбцов f32 (SoA), ровно в порядке HAAParamIdx:

    [0]  DT         [1]  Inv_Cm    [2]  G_L       [3]  E_L
    [4]  G_M        [5]  E_K       [6]  G_H       [7]  E_H
    [8]  Inv_Tau_U  [9]  Inv_Tau_H [10] V_Half_U  [11] K_U
    [12] V_Half_H   [13] K_H       [14] V_Thresh  [15] V_Reset
    [16] Delta_U

Формат: [magic(4)=0x48414131 | n_neurons(8) | 17·n f32 SoA]

Маппинг MANC-типа → суперкласс атласа (как в bake_physio.py):
    DN → 14, SN → 7, MN → 10, IN → 3, AN → 0, SA → 1, unknown → default

Usage:
    python bake_haap.py --data /d/connectoms/datasets \
        --atlas /c/Users/anton/Desktop/neuron_atlas.py \
        --out resources/data --out-name manc_vnc_haap_physio.haap --dt 0.05
"""

import argparse
import importlib.util
import os
import struct

import numpy as np

MAGIC = 0x48414131  # "HAA1"


# MANC type prefix → super_class атласа
MANC_ATLAS_SC = {
    "DN": 14, "DNp": 14, "DNd": 14,
    "SN": 7, "SNa": 7, "SNp": 7,
    "MN": 10, "MNa": 10, "MNp": 10,
    "IN": 3, "INa": 3, "INp": 3,
    "AN": 0,
    "SA": 1, "SApp": 1, "SAp": 1,
}


def manc_atlas_class(type_str):
    if type_str is None or (isinstance(type_str, float) and np.isnan(type_str)):
        return None
    t = str(type_str)
    for prefix, sc in MANC_ATLAS_SC.items():
        if t.startswith(prefix):
            return sc
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/d/connectoms/datasets")
    ap.add_argument("--atlas", default="/c/Users/anton/Desktop/neuron_atlas.py")
    ap.add_argument("--out", default="resources/data")
    ap.add_argument("--out-name", default="manc_vnc_haap_physio.haap")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--g-m-scale", type=float, default=1.0)
    ap.add_argument("--g-h-scale", type=float, default=1.0)
    args = ap.parse_args()

    def load_mod(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    atlas = load_mod(args.atlas, "neuron_atlas")

    # 1. типы нейронов MANC
    import pandas as pd
    prop_path = os.path.join(args.data, "v1.0_manc-v1.0-neuron-properties.feather")
    props = pd.read_feather(prop_path)
    n = len(props)
    print(f"MANC нейронов: {n}")

    super_class = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        sc = manc_atlas_class(props["type"].iloc[i])
        super_class[i] = sc if sc is not None else -1

    uniq, counts = np.unique(super_class, return_counts=True)
    names = {14: "DN", 7: "SN", 10: "MN", 3: "IN", 0: "AN", 1: "SA", -1: "default"}
    for u, c in zip(uniq, counts):
        print(f"  super_class {u:>3} ({names.get(u, '?'):>8}): {c}")

    # 2. литературная HAAP (n, 22) из атласа
    sc_for_haap = np.where(super_class < 0, -1, super_class)
    haap = atlas.build_literature_haap(
        sc_for_haap,
        g_M_scale=args.g_m_scale,
        g_h_scale=args.g_h_scale,
    )
    # колонки атласа:
    #  0=g_L 1=E_L 2=g_M 3=E_K 4=g_h 5=E_h 6=tau_u 7=tau_h 8=V_half_u
    #  9=k_u 10=V_half_h 11=k_h 12=V_thresh 13=V_reset 14=delta_u 15=cm 16=tau_ref
    g_L, E_L, g_M, E_K = haap[:, 0], haap[:, 1], haap[:, 2], haap[:, 3]
    g_h, E_h = haap[:, 4], haap[:, 5]
    tau_u, tau_h = haap[:, 6], haap[:, 7]
    V_half_u, k_u = haap[:, 8], haap[:, 9]
    V_half_h, k_h = haap[:, 10], haap[:, 11]
    V_thresh, V_reset, delta_u = haap[:, 12], haap[:, 13], haap[:, 14]
    cm = haap[:, 15]

    # 3. → 17 столбцов HAAParamIdx (SoA)
    dt = np.full(n, args.dt, dtype=np.float32)
    inv_cm = (1.0 / np.maximum(cm, 1e-6)).astype(np.float32)
    inv_tau_u = (1.0 / np.maximum(tau_u, 1e-6)).astype(np.float32)
    inv_tau_h = (1.0 / np.maximum(tau_h, 1e-6)).astype(np.float32)

    cols = [
        dt.astype(np.float32),        # 0  DT
        inv_cm,                       # 1  Inv_Cm
        g_L.astype(np.float32),       # 2  G_L
        E_L.astype(np.float32),       # 3  E_L
        g_M.astype(np.float32),       # 4  G_M
        E_K.astype(np.float32),       # 5  E_K
        g_h.astype(np.float32),       # 6  G_H
        E_h.astype(np.float32),       # 7  E_H
        inv_tau_u,                    # 8  Inv_Tau_U
        inv_tau_h,                    # 9  Inv_Tau_H
        V_half_u.astype(np.float32),  # 10 V_Half_U
        k_u.astype(np.float32),       # 11 K_U
        V_half_h.astype(np.float32),  # 12 V_Half_H
        k_h.astype(np.float32),       # 13 K_H
        V_thresh.astype(np.float32),  # 14 V_Thresh
        V_reset.astype(np.float32),   # 15 V_Reset
        delta_u.astype(np.float32),   # 16 Delta_U
    ]

    for name, arr in zip(
        ["DT", "Inv_Cm", "G_L", "E_L", "G_M", "E_K", "G_H", "E_H",
         "Inv_Tau_U", "Inv_Tau_H", "V_Half_U", "K_U", "V_Half_H", "K_H",
         "V_Thresh", "V_Reset", "Delta_U"],
        cols,
    ):
        print(f"  {name:>10}: mean={arr.mean():8.3f}  [{arr.min():8.3f} .. {arr.max():8.3f}]")

    # 4. сохранить бинарник
    out = os.path.join(args.out, args.out_name)
    with open(out, "wb") as f:
        f.write(struct.pack("<IQ", MAGIC, n))
        for c in cols:
            f.write(np.asarray(c, dtype="<f4").tobytes())
    print(f"\n[save] {out}: {n} нейронов, 17 столбцов, {os.path.getsize(out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
