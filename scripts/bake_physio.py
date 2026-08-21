#!/usr/bin/env python3
"""
🧬 bake_physio.py — литературная физиология для MANC (нервный шнур мухи).

Берёт типы нейронов MANC (DN/SN/MN/IN/AN/SA из v1.0_manc properties),
маппит их на суперклассы твоего атласа (neuron_atlas.py) и через
convert_to_hh (generate_model_params.py) генерирует per-neuron
Hodgkin-Huxley параметры с калибровкой RMP = -65 мВ.

Выход: бинарный .hhparams — 7 столбцов f32 на нейрон (SoA):
    [0] g_na  [1] g_k  [2] g_l  [3] e_na  [4] e_k  [5] e_l  [6] inv_cm

Формат: [magic(4)=0x48485031 | n_neurons(8) | 7·n f32]

Маппинг MANC-типа → суперкласс атласа:
    DN (descending)     → 14  (Descending Neuron brain→VNC, сильный M-ток)
    SN (sensory)        → 7   (Proprioceptor: FeCO, CS, hair plates)
    MN (motor)          → 10  (Motor Neuron Fast — size principle)
    IN (intrinsic)      → 3   (Premotor Interneuron)
    AN (ascending)      → 0   (Ascending Neuron)
    SA (sensory asc.)   → 1   (Sensory Neuron minor)
    unknown             → default fallback из атласа

Usage:
    python bake_physio.py --data /d/connectoms/datasets --atlas /c/Users/anton/Desktop/neuron_atlas.py --gen /c/Users/anton/Desktop/generate_model_params.py --out resources/data
"""

import argparse
import os
import struct
import sys

import numpy as np

MAGIC = 0x48485031  # "HHP1"


# MANC type prefix → super_class в атласе
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
    ap.add_argument("--gen", default="/c/Users/anton/Desktop/generate_model_params.py")
    ap.add_argument("--out", default="resources/data")
    ap.add_argument("--out-name", default="manc_vnc_hh_physio.hhparams")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1. загружаем атлас и генератор (по пути, без установки)
    import importlib.util

    def load_mod(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    atlas = load_mod(args.atlas, "neuron_atlas")
    gen = load_mod(args.gen, "gen_params")

    # 2. типы нейронов MANC
    import pandas as pd
    prop_path = os.path.join(args.data, "v1.0_manc-v1.0-neuron-properties.feather")
    props = pd.read_feather(prop_path)
    n = len(props)
    print(f"MANC нейронов: {n}")

    # NB: суперклассы здесь — из атласа (0,1,3,7,10,14), НЕ коды nt_types из .taas
    # (2,3,4,5,6,10). Статистика в bake_run.rs печатает классы по nt_types,
    # поэтому per-class цифры не совпадают 1:1 с этими группами физиологии.
    super_class = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        sc = manc_atlas_class(props["type"].iloc[i])
        super_class[i] = sc if sc is not None else -1

    # разбивка
    uniq, counts = np.unique(super_class, return_counts=True)
    names = {14: "DN", 7: "SN", 10: "MN", 3: "IN", 0: "AN", 1: "SA", -1: "default"}
    for u, c in zip(uniq, counts):
        print(f"  super_class {u:>3} ({names.get(u, '?'):>8}): {c}")

    # 3. HAAP из атласа → HH с калибровкой RMP
    #    (neuron_atlas.build_literature_haap возвращает (n, 22) колонок:
    #     g_L, E_L, g_M, E_K, g_h, E_h, tau_u, tau_h, V_half_u, k_u,
    #     V_half_h, k_h, V_thresh, V_reset, delta_u, cm, tau_ref,
    #     g_SK, ca_threshold, ca_influx, ca_rest, tau_ca)
    sc_for_haap = np.where(super_class < 0, -1, super_class)
    haap = atlas.build_literature_haap(sc_for_haap)

    # имена колонок HAAP (порядок из build_literature_haap)
    haap_keys = ["g_L", "E_L", "g_M", "E_K", "g_h", "E_h",
                 "tau_u", "tau_h", "V_half_u", "k_u",
                 "V_half_h", "k_h", "V_thresh", "V_reset",
                 "delta_u", "cm", "tau_ref",
                 "g_SK", "ca_threshold", "ca_influx", "ca_rest", "tau_ca"]
    haap_dict = {k: haap[:, i] for i, k in enumerate(haap_keys)}

    # 4. convert_to_hh (g_na/g_k масштабируются по суперклассу строковому —
    #    для MANC передаём строковые имена, которых нет в HH_SUPERCLASS_SCALES,
    #    поэтому используем дефолт 120/36 и литературные g_l/E из HAAP).
    sc_str = np.full(n, "unknown", dtype=object)
    for i, sc in enumerate(super_class):
        sc_str[i] = {14: "descending", 7: "sensory", 10: "motor",
                     3: "central", 0: "ascending", 1: "sensory_ascending"}.get(sc, "unknown")

    hh = gen.convert_to_hh(haap_dict, super_class_array=sc_str)
    g_na, g_k, g_l = hh["g_na"], hh["g_k"], hh["g_l"]
    e_na, e_k, e_l = hh["e_na"], hh["e_k"], hh["e_l"]
    cm = hh["cm"]
    inv_cm = 1.0 / np.maximum(cm, 1e-6)

    # статистика
    for name, arr in [("g_na", g_na), ("g_k", g_k), ("g_l", g_l),
                      ("e_na", e_na), ("e_k", e_k), ("e_l", e_l), ("inv_cm", inv_cm)]:
        print(f"  {name:>7}: mean={arr.mean():8.3f}  [{arr.min():8.3f} .. {arr.max():8.3f}]")

    # 5. сохранить бинарник
    cols = [g_na, g_k, g_l, e_na, e_k, e_l, inv_cm]
    out = os.path.join(args.out, args.out_name)
    with open(out, "wb") as f:
        f.write(struct.pack("<IQ", MAGIC, n))
        for c in cols:
            f.write(np.asarray(c, dtype="<f4").tobytes())
    print(f"\n[save] {out}: {n} нейронов, 7 столбцов, {os.path.getsize(out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
