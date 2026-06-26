#!/usr/bin/env python3
"""
Investigate CB0013 — нейрон центрального комплекса с реципрокной связью DNg100.

Находит CB0013 в CSR, показывает его connectivity, синапсы,
и ищет треугольник DNg100 → CB0013 → X → DNg100.

Запуск:
  cd taas_checkmate
  python3 fly/scripts/investigate_cb0013.py
"""

import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

FLY_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FLY_DIR / "scripts"))


def investigate():
    from flywire_brain import FlyWireCSR

    print("═" * 65)
    print("  🧬 Исследование CB0013 — 3-neuron loop кандидат")
    print("═" * 65)
    print()

    t0 = time.time()
    csr = FlyWireCSR()
    csr.load()
    print(f"  Загрузка: {time.time()-t0:.1f}s")
    print()

    n = csr.n_neurons
    row_ptr = csr.row_ptr
    col_indices = csr.col_indices
    values = csr.values

    # ─── 1. Find CB0013 ───
    print("  [1] Поиск CB0013...")
    cb0013_csr_idx = None
    cb0013_rid = None

    for rid, typ in csr.type_map.items():
        if 'CB0013' in typ or 'CB0013' in str(rid):
            idx = csr.root_to_idx.get(rid)
            if idx is not None:
                cb0013_csr_idx = idx
                cb0013_rid = rid
                print(f"      Найден: root_id={rid}, csr_idx={idx}")
                print(f"      Тип: {typ}")
                bp = csr.body_part_map.get(rid, '?')
                sc = csr.super_class_map.get(rid, '?')
                fl = csr.flow_map.get(rid, '?')
                print(f"      Body Part: {bp}")
                print(f"      Super Class: {sc}")
                print(f"      Flow: {fl}")
                break

    if cb0013_csr_idx is None:
        print("  ⚠ CB0013 не найден! Ищем вручную...")
        # Фолбек: ищем по частичному совпадению
        for rid, typ in csr.type_map.items():
            if 'CB' in typ and ('0013' in str(rid)[-4:] or 'CB0' in typ[:4]):
                idx = csr.root_to_idx.get(rid)
                if idx is not None:
                    print(f"      Кандидат: root_id={rid}, type={typ}, csr_idx={idx}")
                    if cb0013_csr_idx is None:
                        cb0013_csr_idx = idx
                        cb0013_rid = rid
                    break

    if cb0013_csr_idx is None:
        print("  ⚠ CB0013 не найден! Попробуем другой подход.")
        return
    print()

    # ─── 2. DNg100 → CB0013 синапсы ───
    dng100_info = csr.named_dns.get('DNg100')
    if dng100_info is None:
        print("  ⚠ DNg100 не найден!")
        return

    dng100_rid, dng100_idx = dng100_info
    print(f"  [2] DNg100 (idx={dng100_idx}) → CB0013 (idx={cb0013_csr_idx})")

    # Синапсы от DNg100
    s_dng = int(row_ptr[dng100_idx])
    e_dng = int(row_ptr[dng100_idx + 1])
    has_dng_to_cb = False
    for j in range(s_dng, e_dng):
        if int(col_indices[j]) == cb0013_csr_idx:
            w = float(values[j])
            print(f"      ✅ DNg100 → CB0013, weight={w:.4f}")
            has_dng_to_cb = True
            break
    if not has_dng_to_cb:
        print(f"      ❌ DNg100 → CB0013: NO DIRECT SYNAPSE")
        # Проверим есть ли в принципе CB0013 среди targets DNg100
        targets_set = set(int(col_indices[j]) for j in range(s_dng, e_dng))
        print(f"      DNg100 targets: {len(targets_set)} уникальных нейронов")
        # Поищем CB среди targets
        for j in range(s_dng, e_dng):
            tgt = int(col_indices[j])
            rid_tgt = next((r for r, i in csr.root_to_idx.items() if i == tgt), None)
            if rid_tgt:
                typ_tgt = csr.type_map.get(rid_tgt, '?')
                if 'CB' in typ_tgt:
                    print(f"      CB target: csr={tgt}, type={typ_tgt}")
                    break
    print()

    # ─── 3. CB0013 → DNg100 синапсы ───
    print(f"  [3] CB0013 → DNg100")
    s_cb = int(row_ptr[cb0013_csr_idx])
    e_cb = int(row_ptr[cb0013_csr_idx + 1])
    n_cb_targets = e_cb - s_cb
    print(f"      CB0013 targets: {n_cb_targets} синапсов")

    has_cb_to_dng = False
    for j in range(s_cb, e_cb):
        if int(col_indices[j]) == dng100_idx:
            w = float(values[j])
            print(f"      ✅ CB0013 → DNg100, weight={w:.4f}")
            has_cb_to_dng = True
            break

    if not has_cb_to_dng:
        print(f"      ❌ CB0013 → DNg100: NO DIRECT SYNAPSE")
        # Проверим target types
        target_types = defaultdict(int)
        for j in range(s_cb, min(s_cb + 200, e_cb)):
            tgt = int(col_indices[j])
            rid_tgt = next((r for r, i in csr.root_to_idx.items() if i == tgt), None)
            typ = csr.type_map.get(rid_tgt, '?')[:20] if rid_tgt else '?'
            target_types[typ] += 1
        print(f"      CB0013 target types:")
        for t, c in sorted(target_types.items(), key=lambda x: -x[1])[:10]:
            print(f"        {t}: {c}")
    print()

    # ─── 4. Треугольник DNg100 → CB0013 → X → DNg100 ───
    print("  [4] Поиск треугольника DNg100 → CB0013 → X → DNg100")

    # X — это нейрон, который получает от CB0013 И отправляет к DNg100
    triangle_candidates = []
    for j in range(s_cb, e_cb):
        x_csr = int(col_indices[j])
        if x_csr == dng100_idx:
            continue
        # Проверяем X → DNg100
        s_x = int(row_ptr[x_csr])
        e_x = int(row_ptr[x_csr + 1])
        for k in range(s_x, e_x):
            if int(col_indices[k]) == dng100_idx:
                rid_x = next((r for r, i in csr.root_to_idx.items() if i == x_csr), None)
                typ_x = csr.type_map.get(rid_x, '?')[:25] if rid_x else '?'
                bp_x = csr.body_part_map.get(rid_x, '?')[:15] if rid_x else '?'
                w_cb_to_x = float(values[j])
                w_x_to_dng = float(values[k])
                triangle_candidates.append((x_csr, typ_x, bp_x, w_cb_to_x, w_x_to_dng))
                break

    triangle_candidates.sort(key=lambda x: -x[3] * x[4])  # sort by product of weights

    print(f"      Найдено кандидатов: {len(triangle_candidates)}")
    print()
    print(f"  {'#':>3} {'X_csr':>7} {'X_type':>25} {'X_bp':>15} {'w(C→X)':>8} {'w(X→D)':>8} {'score':>8}")
    print(f"  {'-'*75}")
    for rank, (x_csr, typ_x, bp_x, w_cb_x, w_x_dng) in enumerate(triangle_candidates[:15]):
        score = w_cb_x * w_x_dng
        print(f"  {rank+1:>3d} {x_csr:>7d} {typ_x:>25s} {bp_x:>15s} {w_cb_x:>8.4f} {w_x_dng:>8.4f} {score:>8.4f}")

    # ─── 5. CB0013 connections to VNC/leg ───
    print(f"\n  [5] CB0013 → VNC / leg neurons")
    vnc_targets = []
    for j in range(s_cb, e_cb):
        tgt = int(col_indices[j])
        if tgt in csr.vnc_indices:
            vnc_targets.append((tgt, float(values[j])))
        elif tgt in csr.leg_indices:
            vnc_targets.append((tgt, float(values[j])))
    vnc_targets.sort(key=lambda x: -x[1])

    if vnc_targets:
        print(f"      VNC/leg targets: {len(vnc_targets)}")
        for tgt, w in vnc_targets[:10]:
            rid_tgt = next((r for r, i in csr.root_to_idx.items() if i == tgt), None)
            typ = csr.type_map.get(rid_tgt, '?')[:25] if rid_tgt else '?'
            bp = csr.body_part_map.get(rid_tgt, '?')[:15] if rid_tgt else '?'
            print(f"        csr={tgt:>6d} w={w:.4f} {typ:>25} {bp:>15}")
    else:
        print(f"      ❌ No direct VNC/leg targets")
    print()

    # ─── 6. Summary ───
    print("═" * 65)
    print("  📊 Итог")
    print("═" * 65)

    if has_dng_to_cb and has_cb_to_dng:
        print(f"  ✅ DNg100 ↔ CB0013: полная реципрокная пара!")
    else:
        print(f"  ⚠ Неполная реципрокная связь")

    print(f"  Треугольник: {len(triangle_candidates)} X нейронов замыкают петлю")

    # Show best triangle
    if triangle_candidates:
        best = triangle_candidates[0]
        print(f"\n  💡 Лучший треугольник:")
        print(f"      DNg100 (idx={dng100_idx})")
        print(f"        → CB0013 (idx={cb0013_csr_idx})")
        print(f"          → {best[1]} (idx={best[0]}, bp={best[2]})")
        print(f"            → DNg100")
        print(f"      Веса: {best[3]:.4f} × {best[4]:.4f} = {best[3]*best[4]:.4f}")

    print(f"\n  ✓ Готово!")
    return cb0013_csr_idx, cb0013_rid, triangle_candidates


def main():
    investigate()


if __name__ == "__main__":
    main()
