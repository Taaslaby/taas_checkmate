#!/usr/bin/env python3
"""
Influence Matrix — поиск 3-neuron loops через Neumann series.

Без scipy.sparse! Использует CSR вручную для SpMV.

Идея: r = (I - W)⁻¹ · s = Σ_{k=0}^{∞} W^k · s
Через Richardson iteration: r_{k+1} = W_norm @ r_k + s

Где W_norm = D⁻¹·A (row-normalized adjacency).

Запуск:
  cd taas_checkmate
  python3 fly/scripts/influence_matrix.py
"""

import sys
import time
import math
from pathlib import Path
from collections import defaultdict
import numpy as np

FLY_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FLY_DIR / "scripts"))


def csr_spmv(values, col_indices, row_ptr, x, n):
    """CSR Sparse Matrix-Vector product: y = A @ x."""
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = int(row_ptr[i])
        e = int(row_ptr[i + 1])
        total = 0.0
        for j in range(s, e):
            total += float(values[j]) * float(x[int(col_indices[j])])
        y[i] = total
    return y


def find_reciprocal_pairs(indices, col_indices, row_ptr, top_k=200):
    """Найти реципрокные пары (A→B и B→A) среди top_k нейронов."""
    top_set = set(indices[:top_k])
    recip = []
    for a_csr in indices[:top_k]:
        s = int(row_ptr[a_csr])
        e = int(row_ptr[a_csr + 1])
        for j in range(s, e):
            b_csr = int(col_indices[j])
            if b_csr not in top_set or b_csr <= a_csr:
                continue
            # Проверяем B → A
            s2 = int(row_ptr[b_csr])
            e2 = int(row_ptr[b_csr + 1])
            for k in range(s2, e2):
                if int(col_indices[k]) == a_csr:
                    recip.append((a_csr, b_csr))
                    break
    return recip


def find_triangles(recip_pairs, col_indices, row_ptr, top_set):
    """Найти треугольники A→B→C→A."""
    triangles = []
    for a_csr, b_csr in recip_pairs[:200]:
        s = int(row_ptr[b_csr])
        e = int(row_ptr[b_csr + 1])
        for j in range(s, e):
            c_csr = int(col_indices[j])
            if c_csr not in top_set or c_csr in (a_csr, b_csr):
                continue
            # C → A ?
            s2 = int(row_ptr[c_csr])
            e2 = int(row_ptr[c_csr + 1])
            for k in range(s2, e2):
                if int(col_indices[k]) == a_csr:
                    triangles.append((a_csr, b_csr, c_csr))
                    break
    return triangles


def compute_influence():
    from flywire_brain import FlyWireCSR

    print("═" * 65)
    print("  🧬 Influence Matrix: DNg100 → VNC (manual CSR)")
    print("═" * 65)
    print()

    t_start = time.time()

    # ─── 1. CSR ───
    print("  [1] Загрузка CSR...")
    t0 = time.time()
    csr_data = FlyWireCSR()
    csr_data.load()
    print(f"      {time.time()-t0:.1f}s: {csr_data.n_neurons:,} нейронов, {len(csr_data.values):,} синапсов")
    print()

    if not csr_data.is_v2:
        print("  ⚠ Нужен V2 датасет!")
        return

    n = csr_data.n_neurons
    row_ptr = csr_data.row_ptr
    col_indices = csr_data.col_indices
    values = csr_data.values
    n_syn = len(values)

    # ─── 2. DNg100 seed ───
    dng100_info = csr_data.named_dns.get('DNg100')
    if dng100_info is None:
        print("  ⚠ DNg100 не найден!")
        return

    dng100_rid, dng100_idx = dng100_info
    n_dng_syn = int(row_ptr[dng100_idx + 1] - row_ptr[dng100_idx])
    print(f"  [2] Seed: DNg100 (csr_idx={dng100_idx:,}, {n_dng_syn} синапсов)")
    print()

    # ─── 3. Row-normalize weights ───
    print("  [3] Row-normalization + influence iteration...")
    t0 = time.time()

    # Precompute row sums for normalization
    row_sums = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = int(row_ptr[i])
        e = int(min(row_ptr[i + 1], n_syn))  # guard against trimming issues
        if s < e:
            row_sums[i] = sum(float(values[j]) for j in range(s, e))
    row_sums[row_sums == 0] = 1.0

    # ─── 4. Richardson iteration: r_{k+1} = W_norm @ r_k + s ───
    # W_norm = D⁻¹·A, so W_norm[i,j] = W[i,j] / row_sums[i]
    # SpMV with normalization: y[i] = sum_j W[i,j] * x[j] / row_sums[i]
    s = np.zeros(n, dtype=np.float64)
    s[dng100_idx] = 1.0

    r = np.zeros(n, dtype=np.float64)
    max_iter = 200
    tol = 1e-4

    for k in range(max_iter):
        # r_new = D⁻¹·A @ r + s
        r_new = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s_row = int(row_ptr[i])
            e_row = int(min(row_ptr[i + 1], n_syn))
            total = 0.0
            for j in range(s_row, e_row):
                total += float(values[j]) * r[int(col_indices[j])]
            r_new[i] = total / row_sums[i] + s[i]

        change = float(np.max(np.abs(r_new - r)))
        r = r_new
        if k % 20 == 0:
            print(f"      iter {k:3d}: max_change={change:.2e}")
        if change < tol:
            print(f"      ✅ Converged in {k+1} iterations, change={change:.2e}")
            break
    else:
        print(f"      ⚠ Not converged after {max_iter} iterations, change={change:.2e}")

    print(f"      Iteration time: {time.time()-t0:.1f}s")

    # Last iter stats
    active = int(np.sum(r > 0.001))
    print(f"      Нейронов с influence > 0.001: {active} / {n:,}")
    print()

    # ─── 5. Top influence scores ───
    print("  [5] Анализ influence scores")
    print("═" * 65)

    top_n = min(50, n)
    top_idx = np.argsort(r)[::-1][:top_n]

    print(f"\n  ─── Top-{top_n} нейронов по influence от DNg100 ───")
    print(f"  {'#':>3} {'csr_idx':>8} {'influence':>10} {'type':>30} {'body_part':>15}")
    print(f"  {'-'*70}")
    for rank, idx in enumerate(top_idx):
        rid = next((rid for rid, csr_i in csr_data.root_to_idx.items() if csr_i == idx), None)
        typ = csr_data.type_map.get(rid, '?')[:30] if rid else '?'
        bp = csr_data.body_part_map.get(rid, '?')[:15] if rid else '?'
        print(f"  {rank+1:>3d} {idx:>8d} {r[idx]:>10.6f} {typ:>30} {bp:>15}")

    # ─── 6. By Super Class ───
    print(f"\n  ─── Распределение influence по Super Class ───")
    class_scores = defaultdict(list)
    for rid, csr_i in csr_data.root_to_idx.items():
        if r[csr_i] > 0.001:
            sc = csr_data.super_class_map.get(rid, 'unknown')
            class_scores[sc].append(r[csr_i])

    for sc, scores in sorted(class_scores.items(), key=lambda x: -len(x[1]))[:15]:
        print(f"    {sc:>30s}: {len(scores):>5d} нейронов, "
              f"mean={np.mean(scores):.4f}, max={np.max(scores):.4f}")

    # ─── 7. VNC influence ───
    print(f"\n  ─── VNC нейроны с influence > 0.01 ───")
    vnc_inf = [(csr_i, r[csr_i]) for csr_i in csr_data.vnc_indices if r[csr_i] > 0.01]
    vnc_inf.sort(key=lambda x: -x[1])
    for rank, (idx, score) in enumerate(vnc_inf[:20]):
        rid = next((rid for rid, csr_i in csr_data.root_to_idx.items() if csr_i == idx), None)
        typ = csr_data.type_map.get(rid, '?')[:25] if rid else '?'
        bp = csr_data.body_part_map.get(rid, '?')[:15] if rid else '?'
        print(f"  {rank+1:>3d} csr={idx:>6d} inf={score:.4f} {typ:>25} {bp:>15}")

    # ─── 8. Reciprocal pairs ───
    print(f"\n  [6] Поиск реципрокных пар (3-neuron loop candidates)")
    print("═" * 65)

    top_k = min(500, n)
    top_indices = np.argsort(r)[::-1][:top_k]

    t_recip = time.time()
    recip_pairs = find_reciprocal_pairs(top_indices, col_indices, row_ptr, top_k=300)
    print(f"  Реципрокных пар: {len(recip_pairs)} [{time.time()-t_recip:.1f}s]")

    # Score and sort
    recip_scored = []
    for a_csr, b_csr in recip_pairs:
        score = r[a_csr] + r[b_csr]
        rid_a = next((rid for rid, csr_i in csr_data.root_to_idx.items() if csr_i == a_csr), None)
        rid_b = next((rid for rid, csr_i in csr_data.root_to_idx.items() if csr_i == b_csr), None)
        typ_a = csr_data.type_map.get(rid_a, '?')[:25] if rid_a else '?'
        typ_b = csr_data.type_map.get(rid_b, '?')[:25] if rid_b else '?'
        bp_a = csr_data.body_part_map.get(rid_a, '?')[:15] if rid_a else '?'
        bp_b = csr_data.body_part_map.get(rid_b, '?')[:15] if rid_b else '?'
        recip_scored.append((score, a_csr, b_csr, typ_a, typ_b, bp_a, bp_b))

    recip_scored.sort(key=lambda x: -x[0])

    print(f"\n  ─── Top-20 реципрокных пар по influence ───")
    print(f"  {'score':>8} {'A_type':>25} {'A_bp':>15} | {'B_type':>25} {'B_bp':>15}")
    print(f"  {'-'*8:>8} {'-'*25:>25} {'-'*15:>15} | {'-'*25:>25} {'-'*15:>15}")
    for score, a_csr, b_csr, typ_a, typ_b, bp_a, bp_b in recip_scored[:20]:
        print(f"  {score:>8.4f} {typ_a:>25} {bp_a:>15} | {typ_b:>25} {bp_b:>15}")

    # ─── 9. Triangles ───
    top_set = set(top_indices)
    triangles = find_triangles(recip_pairs, col_indices, row_ptr, top_set)
    print(f"\n  ─── Треугольники (A→B→C→A) ───")
    print(f"  Найдено: {len(triangles)}")
    for a_csr, b_csr, c_csr in triangles[:10]:
        rid_a = next((rid for rid, csr_i in csr_data.root_to_idx.items() if csr_i == a_csr), None)
        rid_b = next((rid for rid, csr_i in csr_data.root_to_idx.items() if csr_i == b_csr), None)
        rid_c = next((rid for rid, csr_i in csr_data.root_to_idx.items() if csr_i == c_csr), None)
        def get_type(r): return csr_data.type_map.get(r, '?')[:20] if r else '?'
        print(f"    CSR {a_csr:>6d} ({get_type(rid_a)}) → {b_csr:>6d} ({get_type(rid_b)}) → "
              f"{c_csr:>6d} ({get_type(rid_c)}) → {a_csr:>6d}")

    # ─── 10. Summary ───
    print()
    print("═" * 65)
    print("  📊 Итог")
    print("═" * 65)

    mn_inf = np.mean([r[csr_i] for csr_i in csr_data.mn_indices if r[csr_i] > 0.001]) if csr_data.mn_indices else 0
    vnc_inf_mean = np.mean([r[csr_i] for csr_i in csr_data.vnc_indices if r[csr_i] > 0.001]) if csr_data.vnc_indices else 0

    print(f"  Средний influence на MNs: {mn_inf:.4f}")
    print(f"  Средний influence на VNC: {vnc_inf_mean:.4f}")
    print(f"  Реципрокных пар: {len(recip_pairs)}")
    print(f"  Треугольников: {len(triangles)}")
    print(f"  Время: {time.time()-t_start:.1f}s")

    if len(recip_pairs) > 0:
        print(f"\n  💡 Лучшие кандидаты в 3-neuron loop:")
        for score, a_csr, b_csr, typ_a, typ_b, bp_a, bp_b in recip_scored[:5]:
            print(f"    {typ_a} ↔ {typ_b} (bp={bp_a}/{bp_b}, score={score:.4f})")

    print(f"\n  ✓ Готово!")
    return r, recip_pairs, triangles


def main():
    compute_influence()


if __name__ == "__main__":
    main()
