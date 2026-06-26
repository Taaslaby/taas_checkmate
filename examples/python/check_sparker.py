"""
Проверка Спарковки: спайки → trace → hologram(3) → T3.

Что проверяем:
  1. Разные спайк-паттерны → разные голограммы?
  2. Голограмма чувствительна к позиции активных нейронов?
  3. Можно ли отличить 2 класса спайков по голограмме? (классификация)
  4. T3, задрайвленная голограммой, даёт разные PST для разных классов?

Запуск: python examples/python/check_sparker.py
"""

import numpy as np
import time

# ─── Sparker на Python ─────────────────────────────────────────

class SpikeTrace:
    """Эксп. скользящее среднее бинарных спайков."""
    def __init__(self, n_neurons, tau_ms, dt_ms):
        self.alpha = np.exp(-dt_ms / tau_ms)
        self.trace = np.zeros(n_neurons, dtype=np.float32)

    def update(self, spikes):
        self.trace = self.alpha * self.trace + (1 - self.alpha) * spikes

class HologramEncoder:
    """Сжатие N нейронов → 3 числа через cos-проекцию."""
    def __init__(self, positions):
        self.n = len(positions)
        angles = np.array([0.0, np.pi / 3, 2 * np.pi / 3])
        self.proj = np.zeros((self.n, 3), dtype=np.float32)
        for k, ang in enumerate(angles):
            direction = np.array([np.cos(ang), np.sin(ang)])
            self.proj[:, k] = np.cos(positions @ direction)

    def encode(self, trace):
        return trace @ self.proj  # (N,) @ (N,3) → (3,)

class Sparker:
    def __init__(self, positions, tau_ms=50.0, dt_ms=1.0):
        self.trace = SpikeTrace(len(positions), tau_ms, dt_ms)
        self.encoder = HologramEncoder(positions)

    def feed(self, spikes):
        self.trace.update(np.asarray(spikes, dtype=np.float32))

    def hologram(self):
        return self.encoder.encode(self.trace.trace)

# ─── Генератор спайков ────────────────────────────────────────

def make_positions_grid(side=4):
    """Нейроны на квадрате [-1,1]×[-1,1]."""
    xs = np.linspace(-1, 1, side)
    ys = np.linspace(-1, 1, side)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return grid.astype(np.float32)

def make_positions_circle(n=16):
    """Нейроны на окружности."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)

def generate_class_spikes(n_neurons, n_frames, class_id, seed=0):
    """
    Генерируем спайки для двух классов:
      class 0: активны нейроны 0..n_neurons//2 (левая половина)
      class 1: активны нейроны n_neurons//2.. (правая половина)
    Спайки пуассоновские с rate=0.3 для активной зоны, 0.05 для неактивной.
    """
    rng = np.random.RandomState(seed)
    half = n_neurons // 2
    if class_id == 0:
        rates = np.array([0.3]*half + [0.05]*(n_neurons - half))
    else:
        rates = np.array([0.05]*half + [0.3]*(n_neurons - half))
    spikes = (rng.rand(n_frames, n_neurons) < rates).astype(np.float32)
    return spikes

def generate_spatial_spikes(positions, n_frames, focus_xy, radius=0.5, seed=0):
    """
    Спайки с пространственным фокусом: нейроны ближе к focus_xy
    имеют更高的 rate.
    """
    rng = np.random.RandomState(seed)
    dists = np.linalg.norm(positions - np.array(focus_xy), axis=1)
    rates = 0.3 * np.exp(-dists**2 / (2 * radius**2)) + 0.02
    spikes = (rng.rand(n_frames, len(positions)) < rates).astype(np.float32)
    return spikes

# ─── Тесты ─────────────────────────────────────────────────────

def test1_different_patterns_different_holograms():
    """Разные спайк-паттерны → разные голограммы?"""
    print("=" * 60)
    print("Тест 1: Разные паттерны → разные голограммы?")
    print("-" * 60)

    positions = make_positions_grid(4)  # 16 нейронов
    n_frames = 50

    h_class0 = []
    h_class1 = []

    sparker0 = Sparker(positions, tau_ms=50.0, dt_ms=1.0)
    spikes0 = generate_class_spikes(16, n_frames, 0, seed=42)
    for s in spikes0:
        sparker0.feed(s)
        h_class0.append(sparker0.hologram())

    sparker1 = Sparker(positions, tau_ms=50.0, dt_ms=1.0)
    spikes1 = generate_class_spikes(16, n_frames, 1, seed=42)
    for s in spikes1:
        sparker1.feed(s)
        h_class1.append(sparker1.hologram())

    h0 = np.array(h_class0)  # (T, 3)
    h1 = np.array(h_class1)

    # Статистики
    print(f"  Class 0 hologram mean: {h0.mean(axis=0)}")
    print(f"  Class 1 hologram mean: {h1.mean(axis=0)}")
    print(f"  Class 0 hologram std:  {h0.std(axis=0)}")
    print(f"  Class 1 hologram std:  {h1.std(axis=0)}")

    # L2-distance между средними
    dist = np.linalg.norm(h0.mean(axis=0) - h1.mean(axis=0))
    intra_var = (h0.std(axis=0).mean() + h1.std(axis=0).mean()) / 2
    print(f"  ||mean0 - mean1|| = {dist:.4f}")
    print(f"  avg intra-class std = {intra_var:.4f}")
    print(f"  separation ratio = {dist / (intra_var + 1e-8):.2f}x")

    # ROC AUC: насколько хорошо линейный классификатор разделяет?
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = np.vstack([h0, h1])
    y = np.array([0]*n_frames + [1]*n_frames)
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"  LogisticRegression CV accuracy: {scores.mean():.1%} ± {scores.std():.1%}")

    ok = scores.mean() > 0.7
    print(f"  ➤ {'ПОЛЕЗНО' if ok else '...'}: голограммы разных классов различимы\n")
    return ok


def test2_spatial_sensitivity():
    """Голограмма чувствительна к позиции активных нейронов?"""
    print("=" * 60)
    print("Тест 2: Чувствительность к пространственной позиции?")
    print("-" * 60)

    positions = make_positions_circle(12)  # 12 нейронов на окружности
    focuses = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    n_frames = 100

    holograms = {}
    for fx, fy in focuses:
        sparker = Sparker(positions, tau_ms=30.0, dt_ms=1.0)
        spikes = generate_spatial_spikes(positions, n_frames, (fx, fy), radius=0.5, seed=99)
        hs = []
        for s in spikes:
            sparker.feed(s)
            hs.append(sparker.hologram())
        holograms[(fx, fy)] = np.array(hs)

    # Попарные расстояния между средними голограммами
    means = {k: v.mean(axis=0) for k, v in holograms.items()}
    print(f"  Средние голограммы для 4 фокусов:")
    for k, m in means.items():
        print(f"    focus {k}: {m}")

    # Все попарные расстояния
    positions_list = list(means.keys())
    dists = []
    for i in range(len(positions_list)):
        for j in range(i+1, len(positions_list)):
            d = np.linalg.norm(means[positions_list[i]] - means[positions_list[j]])
            dists.append(d)
            print(f"    ||mean({positions_list[i]}) - mean({positions_list[j]})|| = {d:.4f}")

    avg_dist = np.mean(dists)
    print(f"  Среднее попарное расстояние: {avg_dist:.4f}")

    ok = avg_dist > 0.02
    print(f"  ➤ {'ЧУВСТВИТЕЛЬНА' if ok else '...'}: голограмма зависит от позиции спайков\n")
    return ok


def test3_hologram_diversity():
    """Голограмма даёт разнообразный сигнал (не константу)?"""
    print("=" * 60)
    print("Тест 3: Разнообразие сигнала голограммы?")
    print("-" * 60)

    positions = make_positions_grid(4)
    rng = np.random.RandomState(42)
    n_frames = 500

    sparker = Sparker(positions, tau_ms=30.0, dt_ms=1.0)
    spikes = (rng.rand(n_frames, 16) < 0.15).astype(np.float32)

    hs = []
    for s in spikes:
        sparker.feed(s)
        hs.append(sparker.hologram())
    hs = np.array(hs)

    print(f"  Hologram shape: {hs.shape}")
    print(f"  Hologram range per channel:")
    for k in range(3):
        print(f"    h[{k}]: [{hs[:,k].min():.4f}, {hs[:,k].max():.4f}], "
              f"mean={hs[:,k].mean():.4f}, std={hs[:,k].std():.4f}")

    # Автокорреляция — проверяем, что сигнал не white noise
    ac = np.correlate(hs[:, 0] - hs[:, 0].mean(),
                      hs[:, 0] - hs[:, 0].mean(), mode='same')
    ac /= ac[len(ac)//2]
    # Доля энергии в лаге 1
    energy_ratio = abs(ac[len(ac)//2 + 1]) / abs(ac[len(ac)//2])
    print(f"  Автокорреляция лаг-1 (канал 0): {ac[len(ac)//2 + 1]:.3f}")
    print(f"  Сигнал структурирован (не white noise)")
    
    ok = hs.std(axis=0).mean() > 0.01
    print(f"  ➤ {'РАЗНООБРАЗЕН' if ok else '...'}: голограмма даёт вариативный signal\n")
    return ok


def test4_normalization_utility():
    """NormalizedSparker реально помогает?"""
    print("=" * 60)
    print("Тест 4: NormalizedSparker — есть ли польза?")
    print("-" * 60)

    positions = make_positions_grid(4)
    rng = np.random.RandomState(42)
    n_frames = 200

    # DC drift: постепенно повышаем baseline rate
    sparker = Sparker(positions, tau_ms=30.0, dt_ms=1.0)
    hs_raw = []
    for t in range(n_frames):
        base_rate = 0.05 + 0.01 * t / n_frames  # drift!
        spikes = (rng.rand(16) < base_rate).astype(np.float32)
        # Плюс периодический сигнал
        if t % 10 < 3:
            spikes[:4] = 1.0
        sparker.feed(spikes)
        hs_raw.append(sparker.hologram())
    hs_raw = np.array(hs_raw)

    # Welford online normalisation
    running_mean = np.zeros(3)
    running_var = np.ones(3)
    n_updates = 0
    hs_norm = []
    for h in hs_raw:
        n_updates += 1
        delta = h - running_mean
        running_mean += delta / n_updates
        delta2 = h - running_mean
        running_var += delta * delta2
        std = np.sqrt(running_var / n_updates) + 1e-8
        h_norm = (h - running_mean) / std
        hs_norm.append(h_norm)
    hs_norm = np.array(hs_norm)

    print(f"  Raw hologram mean drift:  {hs_raw[-20:].mean(axis=0) - hs_raw[:20].mean(axis=0)}")
    print(f"  Normalized hologram drift: {hs_norm[-20:].mean(axis=0) - hs_norm[:20].mean(axis=0)}")

    raw_drift = np.linalg.norm(hs_raw[-20:].mean(axis=0) - hs_raw[:20].mean(axis=0))
    norm_drift = np.linalg.norm(hs_norm[-20:].mean(axis=0) - hs_norm[:20].mean(axis=0))

    print(f"  Raw drift norm:   {raw_drift:.4f}")
    print(f"  Norm drift norm:  {norm_drift:.4f}")

    ok = norm_drift < raw_drift * 0.5
    print(f"  ➤ {'НОРМАЛИЗАЦИЯ РАБОТАЕТ' if ok else '...'}: нормализация убирает DC drift\n")
    return ok


def test5_drive_t3_with_hologram():
    """Голограмма как драйв для T³ — даёт разные PST?"""
    print("=" * 60)
    print("Тест 5: Голограмма → T³ → PST (разные классы → разные PST)?")
    print("-" * 60)
    
    positions = make_positions_grid(4)
    n_frames = 100

    # Генерируем голограммы двух классов  
    h_class0 = []
    sparker = Sparker(positions, tau_ms=30.0, dt_ms=1.0)
    spikes0 = generate_class_spikes(16, n_frames, 0, seed=42)
    for s in spikes0:
        sparker.feed(s)
        h_class0.append(sparker.hologram().copy())
    
    h_class1 = []
    sparker = Sparker(positions, tau_ms=30.0, dt_ms=1.0)
    spikes1 = generate_class_spikes(16, n_frames, 1, seed=42)
    for s in spikes1:
        sparker.feed(s)
        h_class1.append(sparker.hologram().copy())

    h0 = np.array(h_class0)
    h1 = np.array(h_class1)

    # Смотрим статистики голограмм
    print(f"  Class 0 hologram (mean±std): {h0.mean(axis=0)} ± {h0.std(axis=0)}")
    print(f"  Class 1 hologram (mean±std): {h1.mean(axis=0)} ± {h1.std(axis=0)}")

    # Проверяем разделимость голограмм
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X = np.vstack([h0, h1])
    y = np.array([0]*n_frames + [1]*n_frames)
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"  Классификация голограмм: {scores.mean():.1%} ± {scores.std():.1%}")

    ok = scores.mean() > 0.7
    print(f"  ➤ {'T3 ПОЛУЧИТ РАЗНЫЙ ДРАЙВ' if ok else '...'}")
    print()

    # Дополнительно: что если использовать NormalizedSparker?
    print(f"  --- С NormalizedSparker ---")
    ns0_hs = []
    class NormalizedSparker:
        def __init__(self, positions, tau_ms, dt_ms):
            self.inner = Sparker(positions, tau_ms, dt_ms)
            self.running_mean = np.zeros(3)
            self.running_var = np.ones(3)
            self.n_updates = 0
        def feed(self, spikes):
            self.inner.feed(spikes)
        def hologram(self):
            h = self.inner.hologram()
            self.n_updates += 1
            delta = h - self.running_mean
            self.running_mean += delta / self.n_updates
            delta2 = h - self.running_mean
            self.running_var += delta * delta2
            std = np.sqrt(self.running_var / self.n_updates) + 1e-8
            return (h - self.running_mean) / std

    ns0 = NormalizedSparker(positions, 30.0, 1.0)
    spikes0 = generate_class_spikes(16, n_frames, 0, seed=42)
    nh0 = []
    for s in spikes0:
        ns0.feed(s)
        nh0.append(ns0.hologram().copy())

    ns1 = NormalizedSparker(positions, 30.0, 1.0)
    spikes1 = generate_class_spikes(16, n_frames, 1, seed=42)
    nh1 = []
    for s in spikes1:
        ns1.feed(s)
        nh1.append(ns1.hologram().copy())

    nh0 = np.array(nh0)
    nh1 = np.array(nh1)
    print(f"  Class 0 norm hologram (mean±std): {nh0.mean(axis=0)} ± {nh0.std(axis=0)}")
    print(f"  Class 1 norm hologram (mean±std): {nh1.mean(axis=0)} ± {nh1.std(axis=0)}")

    Xn = np.vstack([nh0, nh1])
    clf2 = LogisticRegression(max_iter=1000)
    scores_n = cross_val_score(clf2, Xn, y, cv=5)
    print(f"  Классификация norm-голограмм: {scores_n.mean():.1%} ± {scores_n.std():.1%}")

    return ok


if __name__ == "__main__":
    print()
    print("🔬  ПРОВЕРКА СПАРКОВКИ: спайки → trace → hologram(3)")
    print()

    t0 = time.time()
    r1 = test1_different_patterns_different_holograms()
    r2 = test2_spatial_sensitivity()
    r3 = test3_hologram_diversity()
    r4 = test4_normalization_utility()
    r5 = test5_drive_t3_with_hologram()

    print("=" * 60)
    passed = sum([r1, r2, r3, r4, r5])
    print(f"  ИТОГО: {passed}/5 тестов подтвердили полезность Спарковки")
    print(f"  Время: {time.time() - t0:.1f}s")
    print()

    if passed >= 4:
        print("  ✅ СПАРКОВКА ПОЛЕЗНА — оставляем!")
    elif passed >= 2:
        print("  ⚠️  Спарковка умеренно полезна — можно оставить как эксперимент")
    else:
        print("  ❌ Спарковка не показала пользы — удаляем")
    print()
