#!/usr/bin/env python3
"""
Autotune Izhikevich Parameters — подбор a,b,c,d per neuron type через GPU.

Pipeline:
  1. Загрузить FlyWire cell types → группировка по категориям
  2. Для каждой категории: sweep I_ext на GPU, измерить firing rate
  3. Подобрать bias (I_ext) для target FR каждой категории
  4. Сохранить per-type параметры в .npz
  5. Визуализация: FR curves, parameter distribution

Запуск:
  cd taas_checkmate && python3 fly/scripts/autotune_params.py

Зависимости: numpy, pandas, pyarrow, matplotlib (для визуализации)
"""

import csv
import gzip
import math
import time
import os
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

DLOAD = Path(os.environ.get("FLYWIRE_DATA", str(Path.home() / "Загрузки")))
HERE = Path(__file__).parent
OUTPUT = HERE / "autotune_results.npz"

# ═══════════════════════════════════════════════════════════════
#  Канонические Izhikevich параметры для типов нейронов
# ═══════════════════════════════════════════════════════════════
#
# Izhikevich (2003, IEEE Trans Neural Networks):
#
#   v' = 0.04v² + 5v + 140 - u + I
#   u' = a(bv - u)
#   if v ≥ 30mV: v ← c, u ← u + d
#
# Параметры определяют тип спайкового паттерна:

IZHI_PARAMS = {
    # (a, b, c, d, tau_ref) — canonical
    'RS':  (0.02, 0.2,  -65.0, 8.0,  3.0),   # Regular Spiking
    'IB':  (0.02, 0.2,  -55.0, 4.0,  3.0),   # Intrinsically Bursting
    'CH':  (0.02, 0.2,  -50.0, 2.0,  3.0),   # Chattering
    'FS':  (0.10, 0.2,  -65.0, 2.0,  2.0),   # Fast Spiking
    'LTS': (0.02, 0.25, -65.0, 2.0,  3.0),   # Low-Threshold Spiking
    'TC':  (0.02, 0.25, -65.0, 0.05, 3.0),   # Thalamo-Cortical
    'RZ':  (0.10, 0.26, -65.0, 2.0,  3.0),   # Resonator
}

# Target firing rates (Hz) для I_ext sweep
TARGET_FR_HZ = {
    'RS':  10.0,   # Regular: 5-15 Hz
    'IB':  8.0,    # Bursting: lower rate, burst packages
    'CH':  20.0,   # Chattering: 15-30 Hz
    'FS':  30.0,   # Fast: 20-50 Hz
    'LTS': 5.0,    # Low-threshold: 3-8 Hz
    'TC':  8.0,    # Thalamo-cortical: 5-10 Hz
    'RZ':  10.0,   # Resonator: ~10 Hz
}

# Маппинг FlyWire типов → Izhikevich классы
# Основан на биологии: сенсорные нейроны → RS/FS,
# DN → RS, локальные интернейроны → FS/LTS
FLYWIRE_TYPE_PATTERNS = {
    # Visual — сенсорные (RS)
    'Retinula': 'RS',
    'Lamina': 'RS',
    'Medulla': 'RS',
    'Lobula': 'RS',
    'Lobula Plate': 'RS',
    'T4': 'CH',
    'T5': 'CH',
    # Kenyon cells — обоняние (FS для быстрых ответов)
    'KC': 'FS',
    # Mushroom Body — память (RS/IB)
    'MB': 'RS',
    'APL': 'FS',
    # Central Complex — навигация (RS/IB)
    'CX': 'IB',
    'Fan': 'IB',
    'PB': 'IB',
    'FB': 'IB',
    'NO': 'RS',
    'BU': 'RS',
    # Lateral Horn — обоняние (RS)
    'LH': 'RS',
    # Descending neurons (RS/IB — моторные команды)
    'DN': 'RS',
    'MDN': 'RS',
    'APDN': 'RS',
    'tdDN': 'RS',
    # Clock neurons (LTS — медленные ритмы)
    'DN1': 'LTS',
    'DN2': 'LTS',
    'DN3': 'LTS',
    'LN': 'LTS',
    'LPN': 'LTS',
    'PDF': 'LTS',
    # Octopaminergic/Tyraminergic — модуляторы
    'OA': 'LTS',
    'TA': 'LTS',
    # GABA/general inhibitory (FS)
    'GABA': 'FS',
    # Всё остальное → RS (по умолчанию)
    '_default': 'RS',
}


def classify_neuron_type(typ: str) -> str:
    """Мэппинг FlyWire type string → Izhikevich класс.

    Сортировка по длине pattern (самые длинные первые), чтобы
    'DN1' матчился раньше 'DN', а не наоборот.
    """
    for pattern in sorted(FLYWIRE_TYPE_PATTERNS.keys(), key=len, reverse=True):
        if pattern == '_default':
            continue
        if pattern in typ or typ.startswith(pattern):
            return FLYWIRE_TYPE_PATTERNS[pattern]
    return FLYWIRE_TYPE_PATTERNS['_default']


# ═══════════════════════════════════════════════════════════════
#  GPU FR Probe — firing rate vs I_ext для заданных параметров
# ═══════════════════════════════════════════════════════════════

def probe_fr_gpu(n_neurons: int, params: tuple, I_values: np.ndarray,
                 n_steps: int = 5000) -> np.ndarray:
    """Измерить firing rate для диапазона I_ext на GPU.

    Использует bench_connectome_gpu (без записи спайк-буфера — быстрее,
    меньше памяти). Без синаптических связей — чистые Izhikevich dynamics.

    Args:
        n_neurons: размер популяции (рекомендуется ~512)
        params: (a, b, c, d)
        I_values: массив I_ext для sweep
        n_steps: шагов симуляции

    Returns:
        frs: [len(I_values)] mean firing rates (Hz)
    """
    import taas_checkmate as tcm

    a, b, c, d = params
    dt = 0.001  # 1ms
    n = n_neurons

    # Пустая CSR матрица — никаких синаптических связей
    # Нужна только чтобы Rust/GPU функция не падала
    row_ptr = np.zeros(n + 1, dtype=np.int32)  # нет синапсов
    col_indices = np.zeros(1, dtype=np.int32)
    values = np.zeros(1, dtype=np.float32)

    frs = np.zeros(len(I_values), dtype=np.float32)
    for i, I_ext in enumerate(I_values):
        v = np.full(n, -65.0, dtype=np.float32)
        u = np.full(n, -13.0, dtype=np.float32)
        a_arr = np.full(n, a, dtype=np.float32)
        b_arr = np.full(n, b, dtype=np.float32)
        c_arr = np.full(n, c, dtype=np.float32)
        d_arr = np.full(n, d, dtype=np.float32)
        nt = np.ones(n, dtype=np.float32)
        xI = np.full(n, I_ext, dtype=np.float32)
        spikes = np.zeros(n, dtype=np.int32)

        try:
            # bench_connectome_gpu без записи спайков — только total count
            t_wall, total = tcm.bench_connectome_gpu(
                row_ptr, col_indices, values,
                n_steps, dt, v, u,
                a_arr, b_arr, c_arr, d_arr,
                nt, xI, spikes,
            )
            # FR = total / n_neurons / время_симуляции
            frs[i] = total / n / (n_steps * dt)
        except Exception as e:
            print(f"  ⚠ GPU probe failed I={I_ext:.1f}: {e}")
            frs[i] = 0.0

    return frs


def find_optimal_I(frs: np.ndarray, I_values: np.ndarray,
                   target_fr: float, class_name: str) -> float:
    """Найти I_ext который даёт target FR.

    Интерполирует между измеренными точками.
    """
    if len(frs) == 0:
        return 10.0  # fallback

    # Найти ближайшее значение
    idx = np.argmin(np.abs(frs - target_fr))
    best_I = float(I_values[idx])

    # Если есть точки выше и ниже target, интерполируем
    above = np.where(frs >= target_fr)[0]
    below = np.where(frs < target_fr)[0]
    if len(above) > 0 and len(below) > 0:
        i_high = above[0]
        i_low = below[-1]
        if i_high != i_low:
            f_high = frs[i_high]
            f_low = frs[i_low]
            if f_high != f_low:
                frac = (target_fr - f_low) / (f_high - f_low)
                best_I = I_values[i_low] + frac * (I_values[i_high] - I_values[i_low])

    print(f"    {class_name:5s}: target FR={target_fr:.0f}Hz → I_ext={best_I:.1f}")
    return best_I


# ═══════════════════════════════════════════════════════════════
#  Main autotune
# ═══════════════════════════════════════════════════════════════

def main():
    print("═══════════════════════════════════════════════")
    print("  🔬 Izhikevich Parameter Autotune")
    print("═══════════════════════════════════════════════")

    # 1. Загрузить FlyWire типы
    print("\n─── 1. Loading FlyWire types ───")
    t0 = time.time()
    type_counter = Counter()
    with gzip.open(DLOAD / "consolidated_cell_types.csv.gz", 'rt') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) >= 2:
                type_counter[row[1]] += 1
    print(f"  {len(type_counter)} unique types, {sum(type_counter.values())} neurons "
          f"[{time.time()-t0:.1f}s]")

    # 2. Группировка по Izhikevich классам
    print("\n─── 2. Classification → Izhikevich classes ───")
    class_counts = Counter()
    for typ, count in type_counter.items():
        cls = classify_neuron_type(typ)
        class_counts[cls] += count
    print(f"  Izhikevich classes:")
    for cls, count in class_counts.most_common():
        print(f"    {cls:5s}: {count:7d} neurons")

    # 3. GPU FR Probe для каждого класса
    print("\n─── 3. GPU FR Probe ───")
    I_values = np.arange(0, 31, 2.0, dtype=np.float32)  # 0, 2, 4, ..., 30
    probe_n = 512  # нейронов для probe (быстро, статистически значимо)
    optimal_I = {}

    for cls, count in class_counts.most_common():
        if count < probe_n:
            print(f"  {cls}: только {count} нейронов — пропускаем")
            optimal_I[cls] = 10.0
            continue

        params = IZHI_PARAMS.get(cls, IZHI_PARAMS['RS'])
        target_fr = TARGET_FR_HZ.get(cls, 10.0)
        params_name = f"{cls} (a={params[0]:.2f}, b={params[1]:.2f})"

        print(f"\n  [{cls}] {params_name}")
        print(f"    Target FR: {target_fr:.0f} Hz, Probe n={probe_n}")

        t_start = time.time()
        frs = probe_fr_gpu(probe_n, params[:4], I_values, n_steps=3000)
        print(f"    GPU probe: {time.time()-t_start:.1f}s")

        # Find optimal I_ext
        I_opt = find_optimal_I(frs, I_values, target_fr, cls)
        optimal_I[cls] = I_opt

        # Print FR curve summary
        for j in range(0, len(I_values), 4):
            j_end = min(j + 4, len(I_values))
            vals = "  ".join(
                f"I={I_values[k]:4.0f} → {frs[k]:.2f}Hz"
                for k in range(j, j_end)
            )
            print(f"    {vals}")

    # 4. Построить per-neuron parameter map
    print("\n─── 4. Building per-neuron parameter map ───")
    n_neurons = sum(type_counter.values())
    per_neuron = {
        'a': np.zeros(n_neurons, dtype=np.float32),
        'b': np.zeros(n_neurons, dtype=np.float32),
        'c': np.zeros(n_neurons, dtype=np.float32),
        'd': np.zeros(n_neurons, dtype=np.float32),
        'I_ext': np.zeros(n_neurons, dtype=np.float32),
    }

    from flywire_brain import FlyWireCSR
    csr = FlyWireCSR()
    csr.load()

    # Map each root_id to its type
    root_to_type = {}
    with gzip.open(DLOAD / "consolidated_cell_types.csv.gz", 'rt') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) >= 2:
                root_to_type[int(row[0])] = row[1]

    for rid, idx in csr.root_to_idx.items():
        typ = root_to_type.get(rid, '')
        cls = classify_neuron_type(typ)
        params = IZHI_PARAMS.get(cls, IZHI_PARAMS['RS'])
        per_neuron['a'][idx] = params[0]
        per_neuron['b'][idx] = params[1]
        per_neuron['c'][idx] = params[2]
        per_neuron['d'][idx] = params[3]
        per_neuron['I_ext'][idx] = optimal_I.get(cls, 10.0)

    print(f"  Per-neuron params built: {n_neurons} neurons")

    # 5. Сохранить
    print(f"\n─── 5. Saving → {OUTPUT} ───")
    np.savez_compressed(OUTPUT,
                        a=per_neuron['a'],
                        b=per_neuron['b'],
                        c=per_neuron['c'],
                        d=per_neuron['d'],
                        I_ext=per_neuron['I_ext'])
    print(f"  Saved to {OUTPUT}")

    # 6. Статистика
    print("\n─── 6. Statistics ───")
    for key, arr in per_neuron.items():
        uniq = np.unique(arr)
        print(f"  {key:10s}: unique={len(uniq)}, range=[{arr.min():.3f}, {arr.max():.3f}]")

    print("\n  ✓ Autotune complete!")
    print(f"  Запусти GUI: python3 fly/scripts/vnc_walk.py --autotune")


if __name__ == "__main__":
    main()
