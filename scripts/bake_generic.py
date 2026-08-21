#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Универсальный запекатель коннектомов в MNCV-формат для connscope.

Читает любой источник, который можно свести к графу:
  * **GraphML** (networkx) — C. elegans, кора мыши, rhesus, drosophila medulla:
      узлы с именами (тип-таблица), рёбра с весами, опционально НТ/класс
      и 3D-позиции. Если позиций нет — вычисляем 3D-раскладку (spring),
      чтобы мозг было видно на атласе.
  * **FAFB npz** (полный мозг мухи, flywire): CSR (indptr/indices/data)
      + neuron_ids + fafb_brain_positions.npy — реальные 3D-координаты.

Формат .bin (little-endian, ровно как в bake_manc_viz.py и data.rs):

    magic "MNCV" u32 | version u32 = 1
    n u64 | s u64 | n_types u32 | n_subclass u32 | n_target u32
    xs f32[n] ys f32[n] zs f32[n]  (нормированы [0,1])
    has_pos u8[n] | nt u8[n] | klass u8[n] | leg u8[n] | side u8[n]
    nt_types f32[n] (+1 возб / −1 торм — знак для симуляции)
    body_id u64[n]
    type_idx u16[n] | subclass_idx u16[n] | target_idx u16[n]
    строки: (u16 len + utf8) × таблицы types/subclasses/targets
    row_ptr i32[n+1] | col i32[s] | val f32[s]

Примеры:
  python bake_generic.py --graphml "D:/connectoms/datasets/c.elegans_neural.male_1.graphml" \
      --name-col name --weight-col chemical_weight --out resources/examples/data/elegans.bin
  python bake_generic.py --fafb "D:/connectoms/datasets/fafb_brain_connectome.npz" \
      --out resources/examples/data/fafb.bin
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x4D4E4356  # "MNCV"
VERSION = 1


# ── маленькие кодировщики ─────────────────────────────────────
def nt_code(label: str) -> int:
    s = (label or "").strip().lower()
    return {
        "acetylcholine": 1, "ach": 1, "cholinergic": 1,
        "gaba": 2, "gabaergic": 2,
        "glutamate": 3, "glu": 3, "glutamatergic": 3,
        "da": 4, "dopamine": 4, "dopaminergic": 4,
        "ser": 5, "serotonin": 5, "5ht": 5, "serotonergic": 5,
        "oct": 6, "octopamine": 6, "octopaminergic": 6,
    }.get(s, 0)


def klass_code(label: str) -> int:
    s = (label or "").strip().lower()
    if s in ("motor neuron", "motor", "mn", "motorneuron"):
        return 1
    if "sensory" in s:
        return 2
    if "ascending" in s:
        return 3
    if "descending" in s:
        return 4
    if "intrinsic" in s or "interneuron" in s or "intern" in s:
        return 5
    if "glia" in s or "astro" in s:
        return 6
    if "efferent" in s:
        return 7
    return 0


def _intern(values: list[str]):
    """Интернирует строки: (idx на строку, таблица)."""
    table: list[str] = []
    idx: list[int] = []
    lookup: dict[str, int] = {}
    for v in values:
        s = v if v else "None"
        if s not in lookup:
            lookup[s] = len(table)
            table.append(s)
        idx.append(lookup[s])
    return np.asarray(idx, dtype=np.uint16), table


def write_bin(out: Path, xs, ys, zs, has_pos, nt, kl, lg, sd, nt_types,
              body_id, type_idx, type_table, sub_idx, sub_table,
              tgt_idx, tgt_table, rp, ci, val) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        n = len(xs)
        s = len(ci)
        w = f.write
        w(struct.pack("<II", MAGIC, VERSION))
        w(struct.pack("<QQ", n, s))
        w(struct.pack("<III", len(type_table), len(sub_table), len(tgt_table)))
        for arr in (xs, ys, zs):
            w(np.asarray(arr, dtype=np.float32).tobytes())
        w(np.asarray(has_pos, dtype=np.uint8).tobytes())
        w(np.asarray(nt, dtype=np.uint8).tobytes())
        w(np.asarray(kl, dtype=np.uint8).tobytes())
        w(np.asarray(lg, dtype=np.uint8).tobytes())
        w(np.asarray(sd, dtype=np.uint8).tobytes())
        w(np.asarray(nt_types, dtype=np.float32).tobytes())
        w(np.asarray(body_id, dtype=np.int64).tobytes())
        w(np.asarray(type_idx, dtype=np.uint16).tobytes())
        w(np.asarray(sub_idx, dtype=np.uint16).tobytes())
        w(np.asarray(tgt_idx, dtype=np.uint16).tobytes())
        for table in (type_table, sub_table, tgt_table):
            for st in table:
                b = st.encode("utf-8")
                w(struct.pack("<H", len(b)))
                w(b)
        w(np.asarray(rp, dtype=np.int32).tobytes())
        w(np.asarray(ci, dtype=np.int32).tobytes())
        w(np.asarray(val, dtype=np.float32).tobytes())


def report(out: Path, n, s, has_pos, nt, kl, type_table) -> None:
    print(f"записано: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  n={n} s={s} с позицией={int(has_pos.sum())} типов={len(type_table)}")
    print(f"  nt: ACh={int((nt == 1).sum())} GABA={int((nt == 2).sum())} "
          f"GLUT={int((nt == 3).sum())} unknown={int((nt == 0).sum())}")
    print(f"  klass: motor={int((kl == 1).sum())} sensory={int((kl == 2).sum())} "
          f"asc={int((kl == 3).sum())} desc={int((kl == 4).sum())} "
          f"intrinsic={int((kl == 5).sum())} glia={int((kl == 6).sum())}")


# ── GraphML ────────────────────────────────────────────────────
def bake_graphml(path: Path, out: Path, name_col: str, weight_col: str,
                 pos_cols: tuple[str, str, str] | None = None,
                 nt_col: str | None = None, klass_col: str | None = None,
                 min_weight: float = 0.0) -> None:
    import networkx as nx

    print(f"читаю GraphML: {path}")
    G = nx.read_graphml(path)
    nodes = list(G.nodes(data=True))
    n = len(nodes)
    print(f"  узлов={n}, рёбер={G.number_of_edges()}")

    # порядок узлов — как в графе; отображение id → индекс
    order = {nid: i for i, (nid, _) in enumerate(nodes)}

    def attr(nd, col):
        return nd.get(col, "")

    # ── типы: имя нейрона (или запасные колонки) ──
    types = []
    for _, nd in nodes:
        v = attr(nd, name_col)
        if not v:
            v = attr(nd, "type1") or attr(nd, "class") or attr(nd, "id")
        types.append(str(v))
    type_idx, type_table = _intern(types)

    # ── позиции: реальные x/y/z или spring-раскладка ──
    pos = np.full((n, 3), np.nan, dtype=np.float64)
    for nid, nd in nodes:
        if pos_cols and all(c in nd for c in pos_cols):
            i = order[nid]
            pos[i] = [float(nd[c]) for c in pos_cols]
    has_real = np.isfinite(pos).all(axis=1)
    if has_real.any():
        print(f"  реальных позиций: {int(has_real.sum())}/{n}")
    else:
        print("  позиций нет — считаю 3D spring-раскладку (для атласа)")
        # полный граф для layout (без весов — быстрее)
        layout = nx.spring_layout(G, dim=3, seed=42, k=1.0, iterations=80)
        for nid, _ in nodes:
            pos[order[nid]] = layout[nid]
        has_real[:] = True

    # нормализация [0,1] по bbox валидных точек
    valid = has_real
    lo = pos[valid].min(axis=0)
    hi = pos[valid].max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    norm = np.full_like(pos, 0.0)
    norm[valid] = (pos[valid] - lo) / span
    xs, ys, zs = (norm[:, k].astype(np.float32) for k in range(3))
    has_pos = has_real.astype(np.uint8)

    # ── НТ / класс (если колонки есть) ──
    nt = np.asarray([nt_code(attr(nd, nt_col)) if nt_col else 0 for _, nd in nodes], dtype=np.uint8)
    kl = np.asarray([klass_code(attr(nd, klass_col)) if klass_col else 0 for _, nd in nodes], dtype=np.uint8)
    lg = np.zeros(n, dtype=np.uint8)
    sd = np.zeros(n, dtype=np.uint8)
    nt_types = np.where(nt == 0, 1.0, np.where(np.isin(nt, [2, 3]), -1.0, 1.0)).astype(np.float32)
    body_id = np.arange(1, n + 1, dtype=np.int64)

    # ── рёбра: CSR ──
    rp = np.zeros(n + 1, dtype=np.int64)
    edges = []
    for u, v, ed in G.edges(data=True):
        wgt = ed.get(weight_col)
        if wgt is None:
            wgt = ed.get("weight", 1.0)
        w = float(wgt)
        if w < min_weight:
            continue
        edges.append((order[u], order[v], w))
    # сортировка по источнику (как CSR)
    edges.sort(key=lambda e: e[0])
    col = np.asarray([e[1] for e in edges], dtype=np.int64)
    val = np.asarray([e[2] for e in edges], dtype=np.float32)
    rp[1:] = np.cumsum([len([e for e in edges if e[0] == i]) for i in range(n)])
    s = len(edges)
    print(f"  рёбер (после фильтра w>={min_weight}): {s}")

    write_bin(out, xs, ys, zs, has_pos, nt, kl, lg, sd, nt_types,
              body_id, type_idx, type_table, np.zeros(n, np.uint16), ["None"],
              np.zeros(n, np.uint16), ["None"], rp, col, val)
    report(out, n, s, has_pos, nt, kl, type_table)


# ── FAFB (полный мозг мухи, npz CSR) ──────────────────────────
def _gzip_csv_map(path: Path, key_col: str, val_cols: list[str]):
    """root_id (полный flywire id) → кортеж значений колонок.
    Ключ — младшие 32 бита: npz хранит короткую форму id."""
    import csv
    import gzip
    out: dict[int, tuple[str, ...]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        header = next(r)
        ki = header.index(key_col)
        vis = [header.index(v) for v in val_cols]
        for row in r:
            if not row or not row[ki]:
                continue
            try:
                # flywire-id = префикс 720575940 + 9-значный номер; npz хранит
                # только номер → берём последние 9 цифр.
                rid = int(row[ki]) % 10**9
            except ValueError:
                continue
            out[rid] = tuple(row[v] for v in vis)
    return out


def _flow_klass(flow: str, klass: str) -> int:
    s = (klass or flow or "").lower()
    if "motor" in s:
        return 1
    if "sensory" in s or "afferent" in s:
        return 2
    if "ascending" in s:
        return 3
    if "descending" in s:
        return 4
    if "intrinsic" in s or "interneuron" in s or "intern" in s:
        return 5
    if "glia" in s or "astro" in s:
        return 6
    if "efferent" in s:
        return 7
    return 0


def bake_fafb(npz_path: Path, pos_path: Path, out: Path,
              types_path: Path | None = None,
              class_path: Path | None = None) -> None:
    print(f"читаю FAFB: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    rp = z["indptr"].astype(np.int64)
    col = z["indices"].astype(np.int64)
    val = z["data"].astype(np.float32)
    nids = z["neuron_ids"].astype(np.int64)
    n = int(z["n_neurons"])
    s = len(col)
    print(f"  n={n} s={s}")

    # ── реальные аннотации flywire (по root_id, младшие 32 бита) ──
    type_map: dict[int, str] = {}
    if types_path:
        for rid, (t,) in _gzip_csv_map(types_path, "root_id", ["primary_type"]).items():
            if t:
                type_map[rid] = t
        print(f"  типов клеток из аннотаций: {len(type_map)} уникальных: {len(set(type_map.values()))}")
    class_map: dict[int, tuple[str, str]] = {}
    if class_path:
        class_map = _gzip_csv_map(class_path, "root_id", ["flow", "class", "side"])
        print(f"  classification из аннотаций: {len(class_map)}")

    pos = np.load(pos_path)  # (n, 3) float64
    assert pos.shape[0] == n, f"позиции {pos.shape[0]} != n {n}"
    valid = np.isfinite(pos).all(axis=1)
    lo = pos[valid].min(axis=0)
    hi = pos[valid].max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    norm = np.zeros_like(pos)
    norm[valid] = (pos[valid] - lo) / span
    xs, ys, zs = (norm[:, k].astype(np.float32) for k in range(3))
    has_pos = valid.astype(np.uint8)
    print(f"  позиций: {int(valid.sum())}/{n} (bbox {lo}..{hi})")

    # НТ неизвестен (в аннотациях нет), знак +1 — симуляция работает.
    # Класс/сторона — из classification (flow/class/side), тип — из
    # consolidated_cell_types (реальные типы: T4b, Dm3q, …).
    nt = np.zeros(n, dtype=np.uint8)
    kl = np.zeros(n, dtype=np.uint8)
    lg = np.zeros(n, dtype=np.uint8)
    sd = np.zeros(n, dtype=np.uint8)
    nt_types = np.ones(n, dtype=np.float32)
    body_id = nids

    type_names: list[str] = []
    for i, rid in enumerate(nids.tolist()):
        key = int(rid) & 0xFFFFFFFF
        t = type_map.get(key)
        if not t:
            c = class_map.get(key)
            t = (c[1] if c else "") or "flywire"
        type_names.append(t)
        if c := class_map.get(key):
            kl[i] = _flow_klass(c[0], c[1])
            side = c[2].strip().lower()
            if side.startswith("l"):
                sd[i] = 1
            elif side.startswith("r"):
                sd[i] = 2
    type_idx, type_table = _intern(type_names)
    print(f"  разметка: klass={dict((k, int((kl == k).sum())) for k in range(8))} side L={int((sd == 1).sum())} R={int((sd == 2).sum())}")

    write_bin(out, xs, ys, zs, has_pos, nt, kl, lg, sd, nt_types,
              body_id, type_idx, type_table, np.zeros(n, np.uint16), ["None"],
              np.zeros(n, np.uint16), ["None"], rp, col, val)
    report(out, n, s, has_pos, nt, kl, type_table)


# ── CLI ────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Запекает коннектом в MNCV .bin для connscope")
    ap.add_argument("--graphml", help="GraphML-файл (networkx)")
    ap.add_argument("--name-col", default="name", help="атрибут узла с именем нейрона")
    ap.add_argument("--weight-col", default="weight", help="атрибут ребра с весом синапса")
    ap.add_argument("--pos-cols", default=None, help="x,y,z атрибуты позиций (через запятую)")
    ap.add_argument("--nt-col", default=None, help="атрибут узла с нейротрансмиттером")
    ap.add_argument("--klass-col", default=None, help="атрибут узла с классом клетки")
    ap.add_argument("--min-weight", type=float, default=0.0, help="обрезать рёбра с весом ниже")
    ap.add_argument("--fafb", help="npz FAFB (indptr/indices/data/neuron_ids)")
    ap.add_argument("--fafb-pos", default=None, help="fafb_brain_positions.npy")
    ap.add_argument("--fafb-types", default=None, help="consolidated_cell_types.csv.gz (root_id,primary_type)")
    ap.add_argument("--fafb-class", default=None, help="classification.csv.gz (root_id,flow,class,side)")
    ap.add_argument("--out", required=True, help="путь к .bin")
    args = ap.parse_args()

    if args.fafb:
        if not args.fafb_pos:
            args.fafb_pos = str(Path(args.fafb).with_suffix("")).replace("_connectome", "_positions") + ".npy"
        bake_fafb(Path(args.fafb), Path(args.fafb_pos), Path(args.out),
                  Path(args.fafb_types) if args.fafb_types else None,
                  Path(args.fafb_class) if args.fafb_class else None)
    elif args.graphml:
        pos_cols = tuple(args.pos_cols.split(",")) if args.pos_cols else None
        bake_graphml(Path(args.graphml), Path(args.out), args.name_col, args.weight_col,
                     pos_cols, args.nt_col, args.klass_col, args.min_weight)
    else:
        ap.error("нужен --graphml или --fafb")


if __name__ == "__main__":
    main()
