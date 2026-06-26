"""
Проверка Спарковки на реальном коннектоме дрозофилы (FlyWire FAFB).

Берём optic lobe нейроны, генерируем спайки от визуального стимула,
смотрим — различима ли голограмма для разных стимулов?
"""
import numpy as np
import csv, gzip, json
import os
from pathlib import Path
import sys

DLOAD = Path(os.environ.get("FLYWIRE_DATA", str(Path.home() / "Загрузки")))

# ─── Sparker ────────────────────────────────────────────────

class HologramEncoder:
    def __init__(self, positions):
        self.n = len(positions)
        angles = np.array([0.0, np.pi/3, 2*np.pi/3])
        self.proj = np.zeros((self.n, 3), dtype=np.float32)
        for k, ang in enumerate(angles):
            direction = np.array([np.cos(ang), np.sin(ang)])
            self.proj[:, k] = np.cos(positions @ direction)

    def encode(self, trace):
        return trace @ self.proj

class SpikeTrace:
    def __init__(self, n, tau_ms=50.0, dt_ms=1.0):
        self.alpha = np.exp(-dt_ms / tau_ms)
        self.trace = np.zeros(n, dtype=np.float32)

    def update(self, spikes):
        self.trace = self.alpha * self.trace + (1 - self.alpha) * spikes

class Sparker:
    def __init__(self, positions, tau_ms=50.0, dt_ms=1.0):
        self.trace = SpikeTrace(len(positions), tau_ms, dt_ms)
        self.encoder = HologramEncoder(positions)
    def feed(self, spikes):
        self.trace.update(np.asarray(spikes, dtype=np.float32))
    def hologram(self):
        return self.encoder.encode(self.trace.trace)

# ─── Загрузка данных ─────────────────────────────────────────

def load_flywire_subset(n_max=2000):
    """
    Загружаем координаты и типы optic lobe нейронов.
    Возвращает (positions_2d, type_labels, root_ids).
    """
    print("Загрузка данных FlyWire...")
    import time
    
    # 1. Карта типов нейронов
    print("  Чтение типов клеток...", end=" ", flush=True)
    t0 = time.time()
    neuron_type = {}
    with gzip.open(DLOAD / "consolidated_cell_types.csv.gz", "rt") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rid = int(row[0])
            ptype = row[1]
            neuron_type[rid] = ptype
    print(f"{len(neuron_type)} нейронов за {time.time()-t0:.1f}s")
    
    # 2. Координаты
    print("  Чтение координат...", end=" ", flush=True)
    t0 = time.time()
    coords = {}
    with gzip.open(DLOAD / "coordinates.csv.gz", "rt") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rid = int(row[0])
            pos_str = row[1].strip("[]")
            pos = np.array([float(x) for x in pos_str.split()], dtype=np.float32)
            coords[rid] = pos
    print(f"{len(coords)} нейронов за {time.time()-t0:.1f}s")
    
    # 3. Фильтруем optic lobe типы
    print("  Фильтрация optic lobe...", end=" ", flush=True)
    t0 = time.time()
    optic_types = {
        'R1-6', 'R7', 'R8', 'L1', 'L2', 'L3', 'L4', 'L5',
        'Mi1', 'Mi4', 'Mi9', 'T1', 'T2', 'T2a', 'T3', 'T4a', 'T4b', 'T4c', 'T4d',
        'Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm5', 'Tm9', 'Tm16', 'Tm20', 'TmY',
        'Y', 'Ya', 'Yb', 'YS', 'Lawf1', 'Lawf2',
        'C2', 'C3', 'Ct1', 'Li',
        'MeVC', 'MeTu', 'Lo1',
    }
    
    ids_with_type = []
    for rid, ptype in neuron_type.items():
        if ptype in optic_types and rid in coords:
            ids_with_type.append((rid, ptype, coords[rid]))
    print(f"{len(ids_with_type)} нейронов за {time.time()-t0:.1f}s")
    
    print(f"  Найдено {len(ids_with_type)} optic lobe нейронов")
    
    if len(ids_with_type) > n_max:
        # Стратифицированная выборка: сохраняем пропорции типов
        from collections import Counter
        type_counts = Counter(ptype for _, ptype, _ in ids_with_type)
        sampled = []
        per_type = {t: max(1, n_max * c // len(ids_with_type)) for t, c in type_counts.items()}
        rng = np.random.RandomState(42)
        type_seen = {t: 0 for t in type_counts}
        rng.shuffle(ids_with_type)
        for rid, ptype, pos in ids_with_type:
            if type_seen[ptype] < per_type.get(ptype, 1):
                sampled.append((rid, ptype, pos))
                type_seen[ptype] += 1
            if len(sampled) >= n_max:
                break
        ids_with_type = sampled
        print(f"  Сэмплировано до {len(ids_with_type)} (стратифицировано)")
    
    # 4. Проецируем 3D → 2D: используем главные компоненты
    print("  PCA 3D→2D...", end=" ", flush=True)
    t0 = time.time()
    pos_3d = np.array([p for _, _, p in ids_with_type])
    mean_pos = pos_3d.mean(axis=0)
    centered = pos_3d - mean_pos
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    pos_2d = centered @ eigvecs[:, order[:2]]
    
    pos_range = np.abs(pos_2d).max(axis=0)
    pos_2d = pos_2d / pos_range[np.newaxis, :]
    print(f"за {time.time()-t0:.1f}s")
    
    types = [t for _, t, _ in ids_with_type]
    rids = [rid for rid, _, _ in ids_with_type]
    
    print(f"  2D позиции: range=[{pos_2d.min():.2f}, {pos_2d.max():.2f}]")
    type_dist = {t: types.count(t) for t in sorted(set(types))}
    print(f"  Типы: {len(type_dist)}")
    for t, c in sorted(type_dist.items(), key=lambda x: -x[1])[:10]:
        print(f"    {t}: {c}")
    
    return pos_2d.astype(np.float32), types, rids


def simulate_activity(positions, types, n_frames=500, seed=42):
    """
    Симулируем активность optic lobe в ответ на визуальные стимулы.
    
    Два класса стимулов:
      class 0: стимул слева (активны нейроны с x < 0)
      class 1: стимул справа (активны нейроны с x > 0)
    
    Нейроны получают rate пропорционально exp(-dist^2 / sigma^2) от стимула.
    """
    rng = np.random.RandomState(seed)
    n = len(positions)
    
    # Карта возбудимости по типам нейронов
    type_excitability = {}
    for i, ptype in enumerate(types):
        base = 0.05
        if ptype.startswith('R'):
            base = 0.3   # фоторецепторы — высокий rate
        elif ptype in ('L1', 'L2', 'L3', 'L4', 'L5'):
            base = 0.2   # латеральные — средний
        elif ptype.startswith('Mi'):
            base = 0.15
        elif ptype.startswith('T'):
            base = 0.15
        type_excitability[i] = base
    
    half = n_frames // 2
    all_spikes = []
    all_labels = []
    
    for frame in range(n_frames):
        if frame < half:
            # class 0: стимул слева
            focus = np.array([-1.0, 0.0])
            label = 0
        else:
            # class 1: стимул справа
            focus = np.array([1.0, 0.0])
            label = 1
        
        dists = np.linalg.norm(positions - focus, axis=1)
        rates = np.array([type_excitability.get(i, 0.05) * np.exp(-dists[i]**2 / 0.5) + 0.02
                         for i in range(n)])
        spikes = (rng.rand(n) < rates).astype(np.float32)
        all_spikes.append(spikes)
        all_labels.append(label)
    
    return np.array(all_spikes), np.array(all_labels)


def simulate_activity_by_connectivity(positions, types, n_frames=500, seed=42):
    """
    Быстрая симуляция: предвычисляем топологические связи,
    распространяем активность от фоторецепторов вглубь.
    """
    rng = np.random.RandomState(seed)
    n = len(positions)
    
    # Слои
    layer = np.zeros(n, dtype=np.int32)
    for i, ptype in enumerate(types):
        if ptype.startswith('R'):
            layer[i] = 0
        elif ptype.startswith('L') or ptype in ('C2', 'C3'):
            layer[i] = 1
        elif ptype.startswith('Mi') or ptype.startswith('Tm'):
            layer[i] = 2
        elif ptype.startswith('T') or ptype.startswith('Y'):
            layer[i] = 3
        else:
            layer[i] = 2
    
    # Предвычисляем邻居 связи по слоям: distance matrix
    # (только между соседними слоями)
    conn = {}  # (from_layer, to_layer) → массив пар (i, j)
    for fl in range(3):
        tl = fl + 1
        src = np.where(layer == fl)[0]
        dst = np.where(layer == tl)[0]
        if len(src) == 0 or len(dst) == 0:
            continue
        # Матрица расстояний
        d = np.linalg.norm(positions[src][:, np.newaxis] - positions[dst][np.newaxis, :], axis=2)
        # Каждый dst получает от 3 ближайших src
        nearest = np.argsort(d, axis=0)[:3, :]
        pairs = []
        for j in range(len(dst)):
            for k in range(3):
                i = nearest[k, j]
                pairs.append((src[i], dst[j]))
        conn[(fl, tl)] = pairs
    
    half = n_frames // 2
    all_spikes = np.zeros((n_frames, n), dtype=np.float32)
    state = np.zeros(n, dtype=np.float32)
    dists_to_focus = {}
    
    for frame in range(n_frames):
        focus = np.array([-1.0, 0.0]) if frame < half else np.array([1.0, 0.0])
        
        # Layer 0: фоторецепторы по близости к стимулу
        d = np.linalg.norm(positions - focus, axis=1)
        r0 = layer == 0
        rates = np.where(r0, 0.4 * np.exp(-d**2 / 0.5) + 0.02, 0.0)
        state = (rng.rand(n) < rates).astype(np.float32)
        
        # Layers 1-3: каскад
        for fl in range(3):
            if (fl, fl+1) not in conn:
                continue
            pairs = conn[(fl, fl+1)]
            input_sum = {}
            for src_i, dst_j in pairs:
                input_sum[dst_j] = input_sum.get(dst_j, 0.0) + state[src_i]
            for dst_j, s in input_sum.items():
                rate = 0.3 * (s / 3) + 0.02
                state[dst_j] = (rng.rand() < rate).astype(np.float32)
        
        all_spikes[frame] = state.copy()
    
    labels = np.array([0]*half + [1]*(n_frames - half))
    return all_spikes, labels


def run_tests():
    # Загружаем подмножество (2000 нейронов для скорости)
    positions, types, rids = load_flywire_subset(n_max=2000)
    n = len(positions)
    print(f"\n{n} нейронов, 3 канала голограммы")
    
    # ─── Тест 1: Разные стимулы → разные голограммы? ──────
    print("\n" + "="*60)
    print("ТЕСТ 1: Левый vs Правый стимул → разные голограммы?")
    print("="*60)
    
    spikes, labels = simulate_activity(positions, types, n_frames=500)
    
    sparker = Sparker(positions, tau_ms=30.0, dt_ms=1.0)
    hs = []
    for s in spikes:
        sparker.feed(s)
        hs.append(sparker.hologram().copy())
    hs = np.array(hs)
    
    h0 = hs[labels == 0]
    h1 = hs[labels == 1]
    
    print(f"  Class 0 (left):  mean={h0.mean(axis=0)}, std={h0.std(axis=0)}")
    print(f"  Class 1 (right): mean={h1.mean(axis=0)}, std={h1.std(axis=0)}")
    
    dist = np.linalg.norm(h0.mean(axis=0) - h1.mean(axis=0))
    intra_std = (h0.std(axis=0).mean() + h1.std(axis=0).mean()) / 2
    print(f"  ||mean0 - mean1|| = {dist:.4f}")
    print(f"  avg intra-class std = {intra_std:.4f}")
    print(f"  separation ratio = {dist/(intra_std+1e-8):.2f}x")
    
    # Классификация
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X = hs
    y = labels
    scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
    print(f"  CV accuracy: {scores.mean():.1%} ± {scores.std():.1%}")
    print(f"  ➤ {'РАЗЛИЧИМЫ' if scores.mean() > 0.7 else 'СЛАБО'}")
    
    # ─── Тест 2: Пространственная чувствительность ─────────
    print("\n" + "="*60)
    print("ТЕСТ 2: 4 угла стимуляции → различимы?")
    print("="*60)
    
    focuses = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    focus_hs = {}
    for fx, fy in focuses:
        local_spikes, _ = simulate_activity_by_connectivity(positions, types, n_frames=200, seed=42)
        sparker = Sparker(positions, tau_ms=30.0, dt_ms=1.0)
        fhs = []
        for s in local_spikes:
            sparker.feed(s)
            fhs.append(sparker.hologram().copy())
        focus_hs[(fx, fy)] = np.array(fhs)
    
    means = {k: v.mean(axis=0) for k, v in focus_hs.items()}
    print("  Средние голограммы по фокусам:")
    for k, m in means.items():
        print(f"    {k}: {m}")
    
    pairs = list(means.keys())
    dists = []
    for i in range(len(pairs)):
        for j in range(i+1, len(pairs)):
            d = np.linalg.norm(means[pairs[i]] - means[pairs[j]])
            dists.append(d)
            print(f"    ||{pairs[i]} - {pairs[j]}|| = {d:.4f}")
    
    avg_dist = np.mean(dists)
    print(f"  Среднее попарное расстояние: {avg_dist:.4f}")
    print(f"  ➤ {'ЧУВСТВИТЕЛЬНА' if avg_dist > 0.02 else 'СЛАБО'}")
    
    # ─── Тест 3: Информативность голограммы ────────────────
    print("\n" + "="*60)
    print("ТЕСТ 3: Информативность — как много вариации в 3 каналах?")
    print("="*60)
    
    print(f"  Hologram range:")
    for k in range(3):
        print(f"    h[{k}]: [{hs[:,k].min():.4f}, {hs[:,k].max():.4f}], "
              f"mean={hs[:,k].mean():.4f}, std={hs[:,k].std():.4f}")
    
    # Доля уникальной информации в каждом канале (1 - R² с другими)
    corr = np.corrcoef(hs.T)
    print(f"  Correlation matrix:\n    {np.array2string(corr, precision=3)}")
    # Средняя взаимная корреляция
    off_diag = corr[np.triu_indices(3, k=1)]
    print(f"  Mean inter-channel |ρ| = {np.abs(off_diag).mean():.3f}")
    
    # ─── Тест 4: Визуализация ──────────────────────────────
    print("\n" + "="*60)
    print("ТЕСТ 4: Визуализация траекторий голограммы")
    print("="*60)
    
    # Пробуем разные tau
    for tau in [10.0, 30.0, 100.0]:
        sparker = Sparker(positions, tau_ms=tau, dt_ms=1.0)
        hs_tau = []
        for s in spikes[:100]:
            sparker.feed(s)
            hs_tau.append(sparker.hologram().copy())
        hs_tau = np.array(hs_tau)
        print(f"  tau={tau:.0f}ms:  h std = {hs_tau.std(axis=0)}")
    
    # ─── ИТОГ ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("ИТОГ")
    print("="*60)
    results = {
        "class_separation": scores.mean(),
        "spatial_sensitivity": avg_dist,
        "signal_variation": hs.std(axis=0).mean(),
        "n_neurons": n,
    }
    print(json.dumps(results, indent=2))
    
    useful = scores.mean() > 0.65 and avg_dist > 0.02
    print(f"\n  {'✅ СПАРКОВКА ПОЛЕЗНА на реальном коннектоме' if useful else '⚠️  Умеренно'}")
    print(f"  {n} нейронов → 3 числа, разделимость классов {scores.mean():.0%}")
    return results


if __name__ == "__main__":
    run_tests()
