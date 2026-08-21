#!/usr/bin/env python3
"""
🧠 bake_brain.py — конвертер коннектомов в .taas (формат чекмейта).

Печёт «норм мозги» из D:\\connectoms\\datasets в двоичный формат pillars-connectome:

    FAFB  (мозг мухи)     139 189 нейронов / 19.9M синапсов + super_class + нейромедиаторы
    MANC  (нервный шнур)  102 369 нейронов / 5.2M синапсов + типы нейронов (DN/SN/IN/MN)
    C. elegans             279 нейронов / 3 225 связей (hermaphrodite pharynx)

Формат .taas (см. crates/pillars-connectome/src/data.rs):
    [magic(4) | version(4) | n_neurons(8) | n_synapses(8) |
     row_ptr(4·(n+1)) | col_indices(4·s) | values(4·s) | nt_types(4·n) |
     has_receptors(1) | receptor_types(1·s)?]

Коды нейромедиаторов (для receptor_types, из data.rs):
    ACH  → nAChR (4) fast exc      GLUT → GluCl (5) fast inh (насекомые!)
    GABA → GABA_A (2) fast inh     DA/SER/OCT → mAChR (6) slow mod
    none → AMPA (0) exc fallback

Коды super_class (для nt_types, литературный атлас Антона):
    optic=1 sensory=2 central=3 motor=4 descending=5 ascending=6
    endocrine=7 visual_projection=8 visual_centrifugal=9 sensory_ascending=10
    unknown=0

Usage:
    python bake_brain.py --data /d/connectoms/datasets --out resources/data
"""

import argparse
import os
import struct

import numpy as np

import pathlib, sys  # репо-рут для tools.taas_io при запуске из scripts/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.taas_io import TAAS_MAGIC, write_taas  # crates/pillars-connectome/src/data.rs


# ────────────────────────────────────────────────────────────────────
#  Коды типов
# ────────────────────────────────────────────────────────────────────

SUPERCLASS_CODE = {
    "optic": 1, "sensory": 2, "central": 3, "motor": 4,
    "descending": 5, "ascending": 6, "endocrine": 7,
    "visual_projection": 8, "visual_centrifugal": 9,
    "sensory_ascending": 10, "unknown": 0,
}

# dominant NT нейрона → receptor code (для receptor_types по пре-нейрону)
NT_RECEPTOR = {
    "ACH": 4, "ACETYLCHOLINE": 4, "acetylcholine": 4,
    "GABA": 2, "gaba": 2,
    "GLUT": 5, "GLUTAMATE": 5, "glutamate": 5,
    "DA": 6, "SER": 6, "OCT": 6, "dopamine": 6, "serotonin": 6, "octopamine": 6,
    "None": 0, "nan": 0, "unknown": 0,
}


def _load_gt_nt(gt_csv):
    """cell_type → доминантный нейромедиатор из литературы (gt_data.csv).

    gt_data: одна строка на (cell_type, study). Доминантный NT = тот, у кого
    максимальная сумма положительных свидетельств, взвешенных по confidence.
    """
    import pandas as pd
    gt = pd.read_csv(gt_csv)
    nt_cols = ["acetylcholine", "glutamate", "gaba", "glycine", "dopamine",
               "serotonin", "octopamine", "tyramine", "histamine", "nitric_oxide"]
    gt = gt.copy()
    gt["conf"] = pd.to_numeric(gt["neurotransmitter_verified_confidence"],
                               errors="coerce").fillna(0)
    out = {}
    for ct, grp in gt.groupby("cell_type"):
        best, best_s = None, 0.0
        for nt in nt_cols:
            s = float(((grp[nt] == 1).astype(int) * grp["conf"]).sum())
            if s > best_s:
                best, best_s = nt, s
        if best is not None:
            out[str(ct)] = best
    return out


# write_taas — единая реализация в tools/taas_io.py


# ────────────────────────────────────────────────────────────────────
#  FAFB — мозг мухи (FlyWire v783)
# ────────────────────────────────────────────────────────────────────

def bake_fafb(data_dir, out_dir, magic):
    print("[1/3] FAFB (мозг мухи, FlyWire v783)…")
    npz_path = os.path.join(data_dir, "fafb_brain_connectome.npz")
    cl_path = os.path.join(data_dir, "classification.csv.gz")
    conn_path = os.path.join(data_dir, "connections_princeton.csv.gz")
    if not all(os.path.exists(p) for p in (npz_path, cl_path, conn_path)):
        print("  ⚠ не хватает файлов, пропуск")
        return None

    import pandas as pd

    # 1. CSR из готового npz (уже CSR!)
    with np.load(npz_path, allow_pickle=True) as d:
        indptr = d["indptr"].astype(np.int64)
        indices = d["indices"].astype(np.int64)
        data = d["data"]
        neuron_ids = d["neuron_ids"]
        n_neurons = int(d["n_neurons"])
    n_syn = len(indices)
    print(f"  CSR из npz: {n_neurons} нейронов, {n_syn} синапсов, "
          f"weights [{data.min():.3g} .. {data.max():.3g}] mean={data.mean():.3g}")

    # 2. super_class из classification (root_id % 1e9 == neuron_ids)
    cl = pd.read_csv(cl_path, compression="gzip",
                     usecols=["root_id", "super_class"], low_memory=False)
    cl["fid"] = cl["root_id"] % 1_000_000_000
    fid_map = dict(zip(cl["fid"], cl["super_class"]))
    nt_types = np.zeros(n_neurons, dtype=np.float32)
    matched = 0
    for i, nid in enumerate(neuron_ids):
        sc = fid_map.get(int(nid), "unknown")
        if sc != "unknown":
            matched += 1
        nt_types[i] = SUPERCLASS_CODE.get(sc, 0)
    print(f"  super_class матч: {matched}/{n_neurons} ({100*matched/n_neurons:.1f}%)")

    # 3. receptor_types: dominant NT каждого пре-нейрона из connections_princeton
    print("  читаем нейромедиаторы (connections_princeton)…")
    rec_by_neuron = np.zeros(n_neurons, dtype=np.uint8)  # default AMPA(0)
    nid_to_idx = {int(nid): i for i, nid in enumerate(neuron_ids)}
    nt_by_fid = {}
    for chunk in pd.read_csv(conn_path, compression="gzip",
                             usecols=["pre_root_id", "nt_type"],
                             chunksize=2_000_000, low_memory=False):
        chunk = chunk.dropna(subset=["nt_type"])
        if chunk.empty:
            continue
        fid = (chunk["pre_root_id"] % 1_000_000_000).astype(np.int64)
        grp = chunk.groupby(fid)["nt_type"].agg(lambda s: s.mode().iloc[0])
        for f, nt in grp.items():
            nt_by_fid[int(f)] = nt
    for fid, nt in nt_by_fid.items():
        i = nid_to_idx.get(fid)
        if i is not None:
            rec_by_neuron[i] = NT_RECEPTOR.get(nt, 0)
    print(f"  нейроны с известным NT: {np.count_nonzero(rec_by_neuron)}/{n_neurons}")

    # receptor_types per synapse: берём рецептор пре-нейрона
    receptor_types = rec_by_neuron[indices.astype(np.int64)].astype(np.uint8)

    out = os.path.join(out_dir, "fafb_783.taas")
    write_taas(out, n_neurons, indptr, indices, data, nt_types, receptor_types, magic)
    return out


# ────────────────────────────────────────────────────────────────────
#  MANC — мужской нервный шнур (VNC)
# ────────────────────────────────────────────────────────────────────

# префикс типа MANC → super_class
MANC_PREFIX_CODE = {
    "DN": 5, "DNp": 5, "DNd": 5,        # descending
    "SN": 2, "SNa": 2, "SNp": 2,        # sensory
    "MN": 4, "MNa": 4, "MNp": 4,        # motor
    "IN": 3, "INa": 3, "INp": 3,        # intrinsic
    "AN": 6,                             # ascending
    "SA": 10, "SApp": 10, "SAp": 10,    # sensory ascending
}


def manc_type_code(type_str):
    if type_str is None or (isinstance(type_str, float) and np.isnan(type_str)):
        return 0
    t = str(type_str)
    for prefix, code in MANC_PREFIX_CODE.items():
        if t.startswith(prefix):
            return code
    return 0


def bake_manc(data_dir, out_dir, magic):
    print("[2/3] MANC (мужской нервный шнур / VNC)…")
    conn_path = os.path.join(data_dir, "v1.0_manc-traced-adjacencies-v1.0_traced-connections.csv")
    prop_path = os.path.join(data_dir, "v1.0_manc-v1.0-neuron-properties.feather")
    if not all(os.path.exists(p) for p in (conn_path, prop_path)):
        print("  ⚠ не хватает файлов, пропуск")
        return None

    import pandas as pd

    # 1. нейроны из properties
    props = pd.read_feather(prop_path)
    body_ids = props["bodyId"].values
    n_neurons = len(body_ids)
    print(f"  нейронов: {n_neurons}")

    # маппинг bodyId → индекс
    bid_to_idx = {int(b): i for i, b in enumerate(body_ids)}
    type_by_idx = [manc_type_code(props["type"].iloc[i]) for i in range(n_neurons)]
    nt_types = np.array(type_by_idx, dtype=np.float32)
    print("  типы: ", {k: int(np.sum(nt_types == v)) for k, v in
                       [("DN(5)", 5), ("SN(2)", 2), ("MN(4)", 4), ("IN(3)", 3),
                        ("AN(6)", 6), ("SA(10)", 10), ("unknown(0)", 0)]})

    # 2. связи → CSR
    print("  читаем связи (5.2M)…")
    # пресорт по pre-нейрону: считаем степени
    pre_list = []
    post_list = []
    w_list = []
    for chunk in pd.read_csv(conn_path, usecols=["bodyId_pre", "bodyId_post", "weight"],
                             chunksize=1_000_000):
        m_pre = chunk["bodyId_pre"].map(bid_to_idx)
        m_post = chunk["bodyId_post"].map(bid_to_idx)
        ok = m_pre.notna() & m_post.notna()
        pre_list.append(m_pre[ok].values.astype(np.int64))
        post_list.append(m_post[ok].values.astype(np.int64))
        w_list.append(chunk["weight"][ok].values.astype(np.float32))

    pre = np.concatenate(pre_list)
    post = np.concatenate(post_list)
    weights = np.concatenate(w_list)
    n_syn = len(pre)
    print(f"  связей после фильтрации: {n_syn} (из {sum(len(x) for x in pre_list)})")

    # сортировка по pre → row_ptr
    order = np.argsort(pre, kind="stable")
    pre_sorted = pre[order]
    post_sorted = post[order]
    weights_sorted = weights[order]

    row_ptr = np.bincount(pre_sorted + 1, minlength=n_neurons + 1).astype(np.int64)
    row_ptr = np.cumsum(row_ptr)

    out = os.path.join(out_dir, "manc_vnc.taas")
    write_taas(out, n_neurons, row_ptr, post_sorted, weights_sorted, nt_types, None, magic)
    return out


def bake_manc_signed(data_dir, out_dir, magic, gt_data_path=None):
    """MANC VNC — реальные веса (synapse counts) + знак.

    Веса уже реальные (traced-connections `weight` = число синапсов). Знак
    берём с приоритетом:
      1. литература (gt_data: cell_type → NT, курировано) — самая надёжная;
      2. ML-предикт predictedNt (Eckstein) — для остальных;
      3. дефолт +1 (возб.) для нейронов без типа и без предикта — честный гэп.
    Конвенция знака (насекомые): ACh +1, GABA −1, Glut −1 (GluCl центрально),
    моторный Glut → +1 (GluR на мышце), амины (DA/Ser/Oct) → +1 медленная.
    """
    print("[MANC VNC] реальные веса + знак из predictedNt…")
    conn_path = os.path.join(data_dir, "v1.0_manc-traced-adjacencies-v1.0_traced-connections.csv")
    prop_path = os.path.join(data_dir, "v1.0_manc-v1.0-neuron-properties.feather")
    if not all(os.path.exists(p) for p in (conn_path, prop_path)):
        print("  ⚠ не хватает файлов MANC, пропуск")
        return None

    import pandas as pd
    from collections import Counter

    props = pd.read_feather(prop_path)
    body_ids = props["bodyId"].values
    n = len(body_ids)
    bid_to_idx = {int(b): i for i, b in enumerate(body_ids)}

    # знак + рецептор: литература (gt_data) > ML (predictedNt) > дефолт
    SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0,
            "histamine": -1.0, "glycine": -1.0,
            "dopamine": 1.0, "serotonin": 1.0, "octopamine": 1.0,
            "tyramine": 1.0, "nitric_oxide": 1.0, "unknown": 1.0}
    RECEPTOR = {"acetylcholine": 4, "gaba": 2, "glutamate": 5, "histamine": 0,
                "glycine": 2, "dopamine": 6, "serotonin": 6, "octopamine": 6,
                "tyramine": 6, "nitric_oxide": 6, "unknown": 0}

    pn = [str(x).lower() if x is not None and str(x) != "nan" else "unknown"
          for x in props["predictedNt"].values]
    type_arr = props["type"].values

    # 1. литература по cell_type (gt_data)
    celltype_nt = _load_gt_nt(gt_data_path) if gt_data_path and os.path.exists(gt_data_path) else {}
    final_nt = []
    gt_used = 0
    for i in range(n):
        ct = type_arr[i]
        g = celltype_nt.get(str(ct)) if (ct is not None and str(ct) != "nan") else None
        if g is not None:
            final_nt.append(g)
            gt_used += 1
        else:
            p = pn[i]
            final_nt.append(p if p in SIGN else "unknown")
    final_nt = np.array(final_nt, dtype=object)

    sign = np.array([SIGN.get(x, 1.0) for x in final_nt], dtype=np.float32)
    rec = np.array([RECEPTOR.get(x, 0) for x in final_nt], dtype=np.uint8)

    # моторные (MN*) глутаматергические → ВОЗБУЖДАЮЩИЕ: на мышце глутамат
    # действует через GluR (возб.), а не центральный GluCl (торм.).
    is_motor = np.array([str(x).startswith("MN") for x in type_arr], dtype=bool)
    motor_glut = is_motor & (final_nt == "glutamate")
    sign[motor_glut] = 1.0
    rec[motor_glut] = 0  # GluR (возб.) — нет отдельного кода в data.rs → AMPA-фоллбэк
    print(f"  моторный глутамат→возб.: {motor_glut.sum()} нейронов")
    print(f"  литературой (gt_data) перекрыто: {gt_used} нейронов")
    print("  NT (финал): ", dict(Counter(final_nt.tolist())))
    known = final_nt != "unknown"
    print(f"  знак известен: {known.sum()}/{n} ({known.sum()/n*100:.1f}%)")
    exc_known = ((sign == 1.0) & known).sum()
    inh_known = ((sign == -1.0) & known).sum()
    print(f"  среди известных: возб={exc_known} / торм={inh_known}")

    # рёбра из traced-connections (chunked)
    pre_list, post_list, w_list = [], [], []
    for chunk in pd.read_csv(conn_path, usecols=["bodyId_pre", "bodyId_post", "weight"],
                             chunksize=1_000_000):
        m_pre = chunk["bodyId_pre"].map(bid_to_idx)
        m_post = chunk["bodyId_post"].map(bid_to_idx)
        ok = m_pre.notna() & m_post.notna()
        pre_list.append(m_pre[ok].values.astype(np.int64))
        post_list.append(m_post[ok].values.astype(np.int64))
        w_list.append(chunk["weight"][ok].values.astype(np.float32))
    pre = np.concatenate(pre_list)
    post = np.concatenate(post_list)
    w = np.concatenate(w_list)
    print(f"  рёбер: {len(pre)}")

    order = np.argsort(pre, kind="stable")
    pre_s, post_s, w_s = pre[order], post[order], w[order]
    row_ptr = np.bincount(pre_s + 1, minlength=n + 1).astype(np.int64)
    row_ptr = np.cumsum(row_ptr)

    # ЗНАК уходит в nt_types (конвенция data.rs: nt_types = +1 exc / −1 inh),
    # values = магнитуда (положительное число синапсов). Так работает
    # csr_propagate_*_with_topology: I[post] += values · nt_types[pre].
    values = w_s.astype(np.float32)
    receptor_types = rec[pre_s]
    exc_edges = (sign[pre_s] > 0).sum()
    inh_edges = (sign[pre_s] < 0).sum()
    print(f"  рёбра: exc={exc_edges} ({exc_edges/len(pre_s)*100:.1f}%) / "
          f"inh={inh_edges} ({inh_edges/len(pre_s)*100:.1f}%)")

    nt_types = sign
    out = os.path.join(out_dir, "manc_vnc_signed.taas")
    write_taas(out, n, row_ptr, post_s, values, nt_types, receptor_types, magic)
    return out


# ────────────────────────────────────────────────────────────────────
#  C. elegans — нематода
# ────────────────────────────────────────────────────────────────────

def bake_celegans(data_dir, out_dir, magic):
    print("[3/3] C. elegans (hermaphrodite pharynx)…")
    graphml = os.path.join(data_dir, "c.elegans.herm_pharynx_1.graphml")
    if not os.path.exists(graphml):
        print("  ⚠ файл не найден, пропуск")
        return None

    import xml.etree.ElementTree as ET
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    tree = ET.parse(graphml)
    root = tree.getroot()

    # нейроны
    nodes = {}
    for node in root.findall(".//g:node", ns):
        nid = node.get("id")
        # n0, n1, ...
        nodes[nid] = len(nodes)
    n_neurons = len(nodes)
    print(f"  нейронов: {n_neurons}")

    # рёбра
    edges = []
    for edge in root.findall(".//g:edge", ns):
        src = nodes.get(edge.get("source"))
        dst = nodes.get(edge.get("target"))
        if src is None or dst is None:
            continue
        # вес из атрибута data, если есть
        w = 1.0
        for d in edge.findall("g:data", ns):
            if "weight" in (d.get("key") or ""):
                try:
                    w = float(d.text)
                except Exception:
                    pass
        edges.append((src, dst, w))

    edges.sort(key=lambda e: e[0])
    pre = np.array([e[0] for e in edges], dtype=np.int64)
    post = np.array([e[1] for e in edges], dtype=np.int64)
    weights = np.array([e[2] for e in edges], dtype=np.float32)

    row_ptr = np.bincount(pre + 1, minlength=n_neurons + 1).astype(np.int64)
    row_ptr = np.cumsum(row_ptr)

    nt_types = np.full(n_neurons, 2.0, dtype=np.float32)  # sensory (все возбуждающие)
    out = os.path.join(out_dir, "c_elegans_pharynx.taas")
    write_taas(out, n_neurons, row_ptr, post, weights, nt_types, None, magic)
    return out


# ────────────────────────────────────────────────────────────────────
#  FlyWire v783 — натуральные веса из proofread_connections_783.feather
# ────────────────────────────────────────────────────────────────────

def bake_flywire_proofread(data_dir, out_dir, magic):
    """FlyWire v783 из CAVE proofread-данных — натуральные веса + знак.

    В отличие от старого fafb_783.taas (веса «Excitatory × Connectivity × 0.275»),
    здесь веса и знак берутся из измерения:
      - values = sign × syn_count  — реальное число синапсов на связь;
      - nt_types = знак пре-нейрона (+1 возб / −1 торм / 0 мод.);
      - receptor_types = рецептор пре-нейрона (4=nAChR, 2=GABA_A, 5=GluCl, 6=mAChR).

    Биология: у насекомых глутамат ТОРМОЗНОЙ (GluCl, Cl⁻), ACh — возбуждающий;
    амины (DA/Ser/Oct) — медленная модуляция, не fast-передача → знак 0.
    """
    print("[FlyWire proofread] натуральные веса + знак из CAVE…")
    conn_path = os.path.join(data_dir, "proofread_connections_783.feather")
    ids_path = os.path.join(data_dir, "proofread_root_ids_783.npy")
    if not (os.path.exists(conn_path) and os.path.exists(ids_path)):
        print("  ⚠ не хватает proofread_connections_783.feather / proofread_root_ids_783.npy, пропуск")
        return None

    import pandas as pd
    from collections import Counter

    # 1. нейроны
    root_ids = np.sort(np.load(ids_path).astype(np.int64))
    n = len(root_ids)
    print(f"  нейронов: {n}")

    # 2. рёбра
    tab = pd.read_feather(conn_path)
    pre = tab["pre_pt_root_id"].to_numpy(dtype=np.int64)
    post = tab["post_pt_root_id"].to_numpy(dtype=np.int64)
    syn = tab["syn_count"].to_numpy(dtype=np.int32)
    nt = tab[["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]].to_numpy(dtype=np.float32)
    print(f"  строк: {len(pre)}, total syn_count = {syn.sum()}")

    # 3. root_id → 0..n-1
    def _map(arr):
        idx = np.minimum(np.searchsorted(root_ids, arr), n - 1)
        return idx, root_ids[idx] == arr

    pre_idx, pre_ok = _map(pre)
    post_idx, post_ok = _map(post)
    valid = pre_ok & post_ok
    pre_idx, post_idx = pre_idx[valid], post_idx[valid]
    syn, nt = syn[valid], nt[valid]
    print(f"  рёбер после маппинга: {len(pre_idx)} (несовпавших: {int((~valid).sum())})")

    # 4. per-neuron dominant NT (по исходящим связям)
    nt_sum = np.zeros((n, 6), dtype=np.float64)
    np.add.at(nt_sum, pre_idx, nt.astype(np.float64))
    dom = nt_sum.argmax(axis=1)  # 0=gaba 1=ach 2=glut 3=oct 4=ser 5=da
    nt_names = {0: "GABA", 1: "ACh", 2: "Glut", 3: "Oct", 4: "Ser", 5: "DA"}
    print("  NT по нейронам: ", {nt_names[k]: v for k, v in sorted(Counter(dom.tolist()).items())})

    # знак (Drosophila) + рецептор (data.rs)
    sign = np.where(dom == 1, 1.0, np.where((dom == 0) | (dom == 2), -1.0, 0.0)).astype(np.float32)
    rec = np.select([dom == 0, dom == 1, dom == 2, dom >= 3], [2, 4, 5, 6], default=0).astype(np.uint8)

    # 5. group by (pre,post) → sum syn_count (ключ сортирует по pre)
    key = pre_idx.astype(np.int64) * n + post_idx
    order = np.argsort(key, kind="stable")
    ks = key[order]
    ss = syn[order].astype(np.int64)
    change = np.empty(len(ks), dtype=bool)
    change[0] = True
    np.not_equal(ks[1:], ks[:-1], out=change[1:])
    starts = np.nonzero(change)[0]
    u_pre = pre_idx[order][starts]
    u_post = post_idx[order][starts]
    sums = np.add.reduceat(ss, starts)
    print(f"  уникальных рёбер: {len(sums)}")

    row_ptr = np.bincount(u_pre + 1, minlength=n + 1).astype(np.int64)
    row_ptr = np.cumsum(row_ptr)
    values = (sign[u_pre] * sums).astype(np.float32)
    receptor_types = rec[u_pre]
    print(f"  веса: |w| mean={np.abs(values).mean():.2f}  "
          f"exc={(values > 0).mean()*100:.1f}% / inh={(values < 0).mean()*100:.1f}% / "
          f"zero={(values == 0).mean()*100:.2f}%")

    nt_types = sign
    out = os.path.join(out_dir, "flywire_proofread.taas")
    write_taas(out, n, row_ptr, u_post, values, nt_types, receptor_types, magic)
    return out


# ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/d/connectoms/datasets", help="папка с данными")
    ap.add_argument("--out", default="resources/data", help="куда писать .taas")
    ap.add_argument("--brains", default="all", help="all|fafb|manc|celegans|proofread|manc_signed")
    ap.add_argument("--gt-data",
                    default="C:/Users/anton/Desktop/forks/drosophila_neurotransmitters/gt_data.csv",
                    help="gt_data.csv (литературный атлас NT по типам клеток)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    magic = TAAS_MAGIC
    print(f"TAAS magic = 0x{magic:08x}\n")

    brains = args.brains.split(",")
    results = {}
    if "all" in brains or "fafb" in brains:
        results["fafb"] = bake_fafb(args.data, args.out, magic)
    if "all" in brains or "manc" in brains:
        results["manc"] = bake_manc(args.data, args.out, magic)
    if "all" in brains or "celegans" in brains:
        results["celegans"] = bake_celegans(args.data, args.out, magic)
    if "proofread" in brains:
        results["flywire_proofread"] = bake_flywire_proofread(args.data, args.out, magic)
    if "manc_signed" in brains:
        results["manc_vnc_signed"] = bake_manc_signed(args.data, args.out, magic, args.gt_data)

    print("\n== Готово ==")
    for k, v in results.items():
        if v:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
