#!/usr/bin/env python3
"""One-off prep: build `fly_viz.bin` for the egui fly-brain visualizer.

Reads the FlyWire v783 synapse table (parquet) + per-neuron annotations
(TSV with real 3D coordinates and neurotransmitter labels) and writes a
compact self-contained binary:

    magic  b"FLYV" (4 bytes)
    u32    version = 1
    u32    n_neurons
    u32    n_synapses
    u32    n_cell_types
    f32[n*3]   normalized positions (each axis in [0,1])
    u8[n]      NT code (0=ACh 1=GABA 2=GLUT 3=DA 4=SER 5=OCT 6=unknown)
    u8[n]      side   (0=unknown 1=left 2=right 3=center)
    u64[n]     flywire root_id
    u16[n]     cell-type table index
    u16 count + (u16 len + bytes)*n_cell_types   cell-type strings
    i32[n+1]   CSR row_ptr (outgoing)
    i32[s]     CSR col_indices
    f32[s]     CSR values (synapse counts)

Usage:
    python3 resources/scripts/fafb/fly_viz_prep.py
"""
import struct
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]          # repo root
PARQUET = ROOT / "resources/neurofly/simulation_data/flywire_connectome_v783/Connectivity_783.parquet"
ANNOT = ROOT / "resources/neurofly/simulation_data/flywire_connectome_v783/flywire_annotations.tsv"
OUT = ROOT / "resources/examples/data/fly_viz.bin"

NT_CODES = {  # annotation top_nt text (lowercased, matched by substring) -> code
    "acetyl": 0, "ach": 0, "cholinerg": 0,
    "gaba": 1, "aminobutyric": 1,
    "glut": 2, "glutamaterg": 2,
    "dopamin": 3, "da": 3,
    "seroton": 4, "ser": 4, "5ht": 4,
    "octopamin": 5, "oct": 5,
}

SIDE_CODES = {"left": 1, "right": 2, "center": 3, "central": 3}


def nt_code(text: str) -> int:
    t = (text or "").lower()
    for key, code in NT_CODES.items():
        if key in t:
            return code
    return 6


def side_code(text: str) -> int:
    return SIDE_CODES.get((text or "").lower(), 0)


def main() -> int:
    print(f"parquet : {PARQUET}")
    print(f"annot   : {ANNOT}")
    if not PARQUET.exists():
        print("ERROR: parquet not found", file=sys.stderr)
        return 1

    # ─── 1. connectivity ───────────────────────────────────────
    cols = ["Presynaptic_ID", "Postsynaptic_ID", "Presynaptic_Index",
            "Postsynaptic_Index", "Connectivity"]
    print("reading parquet (15M rows) ...")
    tbl = pq.read_table(str(PARQUET), columns=cols)
    pre_id = tbl.column("Presynaptic_ID").to_numpy()
    post_id = tbl.column("Postsynaptic_ID").to_numpy()
    pre_idx = tbl.column("Presynaptic_Index").to_numpy()
    post_idx = tbl.column("Postsynaptic_Index").to_numpy()
    w = tbl.column("Connectivity").to_numpy()
    s = len(pre_id)
    print(f"  synapses: {s:,}")

    # index -> root_id (consistency check across pre/post)
    idx_to_root: dict[int, int] = {}
    for a, b in zip(pre_idx, pre_id):
        idx_to_root.setdefault(int(a), int(b))
    bad = 0
    for a, b in zip(post_idx, post_id):
        if a in idx_to_root and idx_to_root[int(a)] != int(b):
            bad += 1
        idx_to_root.setdefault(int(a), int(b))
    if bad:
        print(f"  WARNING: {bad} inconsistent index->root_id mappings")

    # inverted: flywire root_id -> compressed index
    root_to_idx: dict[int, int] = {rid: i for i, rid in idx_to_root.items()}
    n = max(idx_to_root) + 1
    print(f"  neurons in connectivity: {n:,}")

    # ─── 2. annotations ────────────────────────────────────────
    print("reading annotations (139k rows) ...")
    root_pos: dict[int, tuple[float, float, float]] = {}
    root_nt: dict[int, int] = {}
    root_side: dict[int, int] = {}
    root_ct: dict[int, str] = {}
    with open(ANNOT) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        i_root, i_x, i_y, i_z, i_nt, i_side, i_ct = (
            idx["root_id"], idx["pos_x"], idx["pos_y"], idx["pos_z"],
            idx["top_nt"], idx["side"], idx["cell_type"],
        )
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) <= max(i_root, i_x, i_y, i_z, i_nt, i_side, i_ct):
                continue
            try:
                rid = int(f[i_root])
            except ValueError:
                continue
            try:
                root_pos[rid] = (float(f[i_x]), float(f[i_y]), float(f[i_z]))
            except ValueError:
                pass
            root_nt[rid] = nt_code(f[i_nt])
            root_side[rid] = side_code(f[i_side])
            ct = f[i_ct].strip()
            if ct:
                root_ct[rid] = ct
    print(f"  annotated neurons: {len(root_pos):,}")

    # ─── 3. build arrays ───────────────────────────────────────
    # Any annotated neuron missing from connectivity gets appended.
    extra = [rid for rid in root_pos if rid not in root_to_idx]
    print(f"  extra annotated-only neurons: {len(extra):,}")
    for rid in extra:
        idx_to_root.setdefault(len(idx_to_root), rid)
    # refresh inverted map after appending
    root_to_idx = {rid: i for i, rid in idx_to_root.items()}

    n = max(idx_to_root) + 1
    xs = [0.5] * n
    ys = [0.5] * n
    zs = [0.5] * n
    nt = [6] * n
    side = [0] * n
    ct = [""] * n
    roots = [0] * n
    for i, rid in idx_to_root.items():
        roots[i] = rid
        rid = int(rid)
        if rid in root_nt:
            nt[i] = root_nt[rid]
        if rid in root_side:
            side[i] = root_side[rid]
        if rid in root_ct:
            ct[i] = root_ct[rid]
        if rid in root_pos:
            xs[i], ys[i], zs[i] = root_pos[rid]

    # normalize each axis to [0,1]
    def norm(v: list[float]) -> list[float]:
        lo, hi = min(v), max(v)
        span = (hi - lo) or 1.0
        return [(x - lo) / span for x in v]

    xs, ys, zs = norm(xs), norm(ys), norm(zs)

    # cell-type string table (dedup)
    table: dict[str, int] = {}
    ct_idx = [0] * n
    for i, name in enumerate(ct):
        if not name:
            continue
        if name not in table:
            table[name] = len(table)
        ct_idx[i] = table[name]
    table_names = sorted(table, key=table.get)

    # ─── 4. CSR (outgoing) ─────────────────────────────────────
    row_counts = [0] * (n + 1)
    for a in pre_idx:
        row_counts[int(a)] += 1
    row_ptr = [0] * (n + 1)
    cum = 0
    for i in range(n):
        row_ptr[i] = cum
        cum += row_counts[i]
    row_ptr[n] = cum
    pos = [0] * n
    col_indices = [0] * s
    values = [0.0] * s
    for a, b, wi in zip(pre_idx, post_idx, w):
        p = pos[int(a)]
        k = row_ptr[int(a)] + p
        col_indices[k] = int(b)
        values[k] = float(wi)
        pos[int(a)] = p + 1

    # ─── 5. write binary ───────────────────────────────────────
    print(f"writing {OUT} ...")
    with open(OUT, "wb") as fh:
        fh.write(b"FLYV")
        fh.write(struct.pack("<III", 1, n, s))
        fh.write(struct.pack("<I", len(table_names)))
        for v in (xs, ys, zs):
            fh.write(struct.pack(f"<{n}f", *v))
        fh.write(struct.pack(f"<{n}B", *nt))
        fh.write(struct.pack(f"<{n}B", *side))
        fh.write(struct.pack(f"<{n}Q", *roots))
        fh.write(struct.pack(f"<{n}H", *ct_idx))
        for name in table_names:
            b = name.encode()
            fh.write(struct.pack("<H", len(b)))
            fh.write(b)
        fh.write(struct.pack(f"<{n + 1}i", *row_ptr))
        fh.write(struct.pack(f"<{s}i", *col_indices))
        fh.write(struct.pack(f"<{s}f", *values))

    mb = OUT.stat().st_size / 1e6
    from collections import Counter
    dist = Counter(nt)
    print(f"  OK: {n:,} neurons, {s:,} synapses, {mb:.0f} MB")
    print(f"  NT: ACh={dist[0]:,} GABA={dist[1]:,} GLUT={dist[2]:,} "
          f"DA={dist[3]:,} SER={dist[4]:,} OCT={dist[5]:,} unknown={dist[6]:,}")
    print(f"  cell types: {len(table_names):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
