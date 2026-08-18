#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Запекает самодостаточный `manc_viz.bin` для Rust-приложения connscope.

Читает:
  * connkit/resources/manc_vnc_signed.taas   — CSR + знак НТ (+1 exc / -1 inh)
  * connkit/resources/manc_meta.feather      — метаданные (type/class/predictedNt/leg/side/...)
  * connkit/resources/manc_positions.npz     — 3D voxel-позиции + body_id

Пишет `resources/examples/data/manc_viz.bin` (little-endian):

    magic u32 "MNCV" (0x4D4E4356) | version u32 = 1
    n u64 | s u64 | n_types u32 | n_subclass u32 | n_target u32
    xs f32[n] | ys f32[n] | zs f32[n]        (изотропный куб [0,1]³,
                                               центр 0.5, пропорции сохранены)
    has_pos u8[n]
    nt u8[n]         (0=unknown 1=ACh 2=GABA 3=GLUT 4=DA 5=SER 6=OCT)
    klass u8[n]      (0=other 1=motor 2=sensory 3=asc 4=desc 5=intrinsic 6=glia 7=efferent)
    leg u8[n]        (0=other 1=T1 2=T2 3=T3 4=abdomen)
    side u8[n]       (0=other 1=L 2=R)
    nt_types f32[n]  (знак НТ из .taas: +1/-1 — для симуляции)
    body_id u64[n]
    type_idx u16[n] | subclass_idx u16[n] | target_idx u16[n]
    types/subclasses/targets: по (u16 len + utf8 bytes)
    row_ptr i32[n+1] | col i32[s] | val f32[s]
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

import pathlib, sys  # репо-рут для tools.taas_io при запуске из scripts/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.taas_io import load_taas

MAGIC = 0x4D4E4356  # "MNCV"
VERSION = 1

TAAS_MAGIC = 0x54414153

# ── enums ─────────────────────────────────────────────────────
def nt_code(label: str) -> int:
    s = label.strip().lower()
    return {
        "acetylcholine": 1, "ach": 1,
        "gaba": 2,
        "glutamate": 3, "glu": 3,
        "da": 4, "dopamine": 4,
        "ser": 5, "serotonin": 5, "5ht": 5,
        "oct": 6, "octopamine": 6,
    }.get(s, 0)  # unknown / None

def klass_code(label: str) -> int:
    s = label.strip().lower()
    if s == "motor neuron":
        return 1
    if "sensory" in s:
        return 2
    if s == "ascending neuron":
        return 3
    if s == "descending neuron":
        return 4
    if "intrinsic" in s or "interneuron" in s:
        return 5
    if "glia" in s:
        return 6
    if s == "efferent neuron":
        return 7
    return 0  # other / None / TBD

def leg_code(label: str) -> int:
    return {"t1": 1, "t2": 2, "t3": 3, "abdomen": 4}.get(label.strip().lower(), 0)

def side_code(label: str) -> int:
    return {"l": 1, "r": 2}.get(label.strip().lower(), 0)


def _read_feather(path):
    with pa.memory_map(str(path), "r") as src:
        return ipc.open_file(src).read_all()


def _str_col(t, key: str) -> list[str]:
    if key not in t.schema.names:
        return ["unknown"] * t.num_rows
    return [("unknown" if v is None else str(v)) for v in t.column(key).to_pylist()]


def _intern(values: list[str]):
    """Интернирует строки: (idx per row, table). None → 'None'."""
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


# парсер .taas — в tools/taas_io.py (load_taas возвращает n, rp, ci, vals, nt, receptor_types)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taas", default=r"C:\Users\anton\Desktop\Проекты\connkit\resources\manc_vnc_signed.taas")
    ap.add_argument("--meta", default=r"C:\Users\anton\Desktop\Проекты\connkit\resources\manc_meta.feather")
    ap.add_argument("--pos", default=r"C:\Users\anton\Desktop\Проекты\connkit\resources\manc_positions.npz")
    ap.add_argument("--out", default="resources/examples/data/manc_viz.bin")
    args = ap.parse_args()

    n, rp, ci, val, nt_types, _rt = load_taas(Path(args.taas))
    s = len(ci)
    print(f".taas: n={n} s={s}")

    meta = _read_feather(Path(args.meta))
    assert meta.num_rows == n, f"meta rows {meta.num_rows} != taas n {n}"
    types = _str_col(meta, "type")
    subclasses = _str_col(meta, "subclass")
    targets = _str_col(meta, "target")
    klass = _str_col(meta, "class")
    ntlab = _str_col(meta, "predictedNt")
    legs = _str_col(meta, "leg")
    sides = _str_col(meta, "side")

    z = np.load(args.pos)
    pos = z["pos"].astype(np.float32)          # (n, 3), NaN где нет
    body_id = z["body_id"].astype(np.int64)
    has_pos = z["has_pos"].astype(bool)
    assert pos.shape[0] == n, f"positions {pos.shape[0]} != n {n}"

    # Нормализуем позиции в изотропный куб [0,1]³ с центром 0.5: ОДИН масштаб
    # на все оси сохраняет пропорции мозга. Per-axis min-max (каждый столбец
    # отдельно) растягивал VNC в куб 1×1×1 — дорсальная XY-проекция выглядела
    # «сплюснутой/растянутой», хотя физически пропорции 0.51:0.63:1.
    valid = has_pos
    lo = pos[valid].min(axis=0)
    hi = pos[valid].max(axis=0)
    center = (lo + hi) / 2.0
    extent = float((hi - lo).max())
    scale = extent if extent > 0 else 1.0
    norm = np.full_like(pos, 0.0, dtype=np.float32)
    norm[valid] = ((pos[valid] - center) / scale + 0.5).astype(np.float32)
    xs = norm[:, 0].astype(np.float32)
    ys = norm[:, 1].astype(np.float32)
    zs = norm[:, 2].astype(np.float32)
    print(f"positions: {int(valid.sum())}/{n} (bbox {lo}..{hi}, uniform scale {scale:.0f})")

    nt = np.asarray([nt_code(v) for v in ntlab], dtype=np.uint8)
    kl = np.asarray([klass_code(v) for v in klass], dtype=np.uint8)
    lg = np.asarray([leg_code(v) for v in legs], dtype=np.uint8)
    sd = np.asarray([side_code(v) for v in sides], dtype=np.uint8)
    type_idx, type_table = _intern(types)
    sub_idx, sub_table = _intern(subclasses)
    tgt_idx, tgt_table = _intern(targets)

    def w(f, b): f.write(b)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        w(f, struct.pack("<II", MAGIC, VERSION))
        w(f, struct.pack("<QQ", n, s))
        w(f, struct.pack("<III", len(type_table), len(sub_table), len(tgt_table)))
        for arr in (xs, ys, zs):
            w(f, arr.tobytes())
        w(f, has_pos.astype(np.uint8).tobytes())
        w(f, nt.tobytes())
        w(f, kl.tobytes())
        w(f, lg.tobytes())
        w(f, sd.tobytes())
        w(f, nt_types.astype(np.float32).tobytes())
        w(f, body_id.astype(np.int64).tobytes())
        w(f, type_idx.tobytes())
        w(f, sub_idx.tobytes())
        w(f, tgt_idx.tobytes())
        for table in (type_table, sub_table, tgt_table):
            for st in table:
                b = st.encode("utf-8")
                w(f, struct.pack("<H", len(b)))
                w(f, b)
        w(f, rp.tobytes())
        w(f, ci.tobytes())
        w(f, val.tobytes())

    size_mb = out.stat().st_size / 1e6
    print(f"записано: {out} ({size_mb:.1f} MB)")
    print(f"  types={len(type_table)} subclasses={len(sub_table)} targets={len(tgt_table)}")
    print(f"  nt: ACh={int((nt == 1).sum())} GABA={int((nt == 2).sum())} "
          f"GLUT={int((nt == 3).sum())} unknown={int((nt == 0).sum())}")
    print(f"  klass: motor={int((kl == 1).sum())} sensory={int((kl == 2).sum())} "
          f"asc={int((kl == 3).sum())} desc={int((kl == 4).sum())} "
          f"intrinsic={int((kl == 5).sum())}")


if __name__ == "__main__":
    main()
