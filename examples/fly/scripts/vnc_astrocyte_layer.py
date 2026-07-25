#!/usr/bin/env python3
"""🧫 VNC Astrocyte Gap-Junction Layer

Реализует невидимые осцилляторы VNC: астроцитарную сеть с gap junctions,
Ca²⁺ волнами и volume transmission модуляцией.

Соответствует Rust-версии ``crates/kernels/src/neuron/astrocyte.rs``.

Физиология:
  1. Gap Junction электрическая связь: I_gap = g_gap · Σ(V_j - V_i)
  2. Медленные Ca²⁺ волны: τ = 1.0 – 10.0 сек (0.1 – 2 Гц)
  3. Volume Transmission: M_i = 1.0 + μ · Ca_i → модуляция I_ext нейронов

Использование
------------
>>> from vnc_astrocyte_layer import AstrocyteNetwork, GapJunctionConnectome
>>> import taas_checkmate as tc
>>>
>>> conn = tc.ConnectomeData(rp, ci, vals, nt)
>>> sim = tc.BrainSimulator(conn.n_neurons, "hh", pacemaking=True)
>>>
>>> # Создать gap junction связи для non-reciprocal VNC нейронов
>>> gj = GapJunctionConnectome()
>>> gj.add_random(neuron_ids, p=0.05, g_gap=0.1)
>>>
>>> # Оборачиваем симулятор в астроцитарный слой
>>> astro_net = AstrocyteNetwork(sim, n_astro=conn.n_neurons,
...                              gap_conn=gj, tau_astro_ms=2000.0)
>>>
>>> # Запуск симуляции с астроцитарной модуляцией
>>> spike_buf, I_ext_trace = astro_net.simulate(
...     conn, n_steps=5000, dt_ms=0.5, external_I=10.0,
...     chunk_size=100  # обновлять астроциты каждые 100 шагов
... )
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


# ═══════════════════════════════════════════════════════════
#  GapJunctionConnectome — матрица gap junction связей
# ═══════════════════════════════════════════════════════════

class GapJunctionConnectome:
    """Матрица gap junction (электрических синапсов) между нейронами/астроцитами.

    Gap junctions — это щелевые контакты, которые пропускают ионы напрямую,
    создавая электрическую связь без химических синапсов.
    В EM-коннектоме (MANC/FAFB) они НЕ видны — это гипотетический слой.
    """

    def __init__(self):
        # Список пар (i, j, g_gap)
        self.pairs: List[Tuple[int, int, float]] = []
        self._built = False
        self._row_ptr: Optional[np.ndarray] = None
        self._col_indices: Optional[np.ndarray] = None
        self._g_vals: Optional[np.ndarray] = None

    def add(self, i: int, j: int, g_gap: float = 0.1):
        """Добавить gap junction между нейронами i и j с проводимостью g_gap."""
        if i == j:
            return  # нет самопетель
        self.pairs.append((i, j, g_gap))
        self._built = False

    def add_all_to_all(self, neuron_ids: List[int], g_gap: float = 0.01):
        """Полносвязная gap junction сеть (для малых кластеров до 50 нейронов)."""
        n = len(neuron_ids)
        for a in range(n):
            for b in range(a + 1, n):
                self.add(neuron_ids[a], neuron_ids[b], g_gap)

    def add_random(self, neuron_ids: List[int], p: float = 0.05,
                   g_gap: float = 0.1, seed: int = 42):
        """Случайные gap junction связи с вероятностью p."""
        rng = np.random.default_rng(seed)
        n = len(neuron_ids)
        for a in range(n):
            for b in range(a + 1, n):
                if rng.random() < p:
                    self.add(neuron_ids[a], neuron_ids[b], g_gap)

    def add_line(self, neuron_ids: List[int], g_gap: float = 0.1):
        """Линейная цепочка gap junction (i—i+1—i+2—...)."""
        for a in range(len(neuron_ids) - 1):
            self.add(neuron_ids[a], neuron_ids[a + 1], g_gap)

    def add_cluster_star(self, center: int, neighbors: List[int], g_gap: float = 0.2):
        """Звезда: центр соединён со всеми соседями."""
        for nid in neighbors:
            self.add(center, nid, g_gap)

    def build(self, n_total: int):
        """Построить CSR-представление для быстрого доступа."""
        if not self.pairs:
            self._row_ptr = np.array([0], dtype=np.int32)
            self._col_indices = np.array([], dtype=np.int32)
            self._g_vals = np.array([], dtype=np.float32)
            self._built = True
            return

        # Сортируем по i (строкам)
        self.pairs.sort(key=lambda x: x[0])

        col_list: List[List[int]] = [[] for _ in range(n_total)]
        g_list: List[List[float]] = [[] for _ in range(n_total)]

        for i, j, g in self.pairs:
            if i < n_total and j < n_total:
                col_list[i].append(j)
                g_list[i].append(g)
                col_list[j].append(i)
                g_list[j].append(g)  # симметрично

        row_ptr = [0]
        for i in range(n_total):
            row_ptr.append(row_ptr[-1] + len(col_list[i]))

        col_indices = []
        g_vals = []
        for i in range(n_total):
            col_indices.extend(col_list[i])
            g_vals.extend(g_list[i])

        self._row_ptr = np.array(row_ptr, dtype=np.int32)
        self._col_indices = np.array(col_indices, dtype=np.int32)
        self._g_vals = np.array(g_vals, dtype=np.float32)
        self._built = True

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def compute_gap_currents(self, v_gap: np.ndarray) -> np.ndarray:
        """Вычислить I_gap = g_gap · Σ(V_j - V_i) для каждого узла.

        Parameters
        ----------
        v_gap : np.ndarray, shape (n,)
            Потенциалы gap junction узлов.

        Returns
        -------
        i_gap : np.ndarray, shape (n,)
            Ток через gap junctions для каждого узла.
        """
        if not self._built:
            self.build(len(v_gap))

        n = len(v_gap)
        i_gap = np.zeros(n, dtype=np.float32)

        rp = self._row_ptr
        ci = self._col_indices
        gv = self._g_vals

        for i in range(n):
            start, end = rp[i], rp[i + 1]
            if start >= end:
                continue
            neighbors = ci[start:end]
            g_vals_i = gv[start:end]
            diff = v_gap[neighbors] - v_gap[i]
            i_gap[i] = np.dot(g_vals_i, diff)

        return i_gap


# ═══════════════════════════════════════════════════════════
#  AstrocyteNetwork — астроцитарная сеть для VNC симуляции
# ═══════════════════════════════════════════════════════════

class AstrocyteNetwork:
    """Астроцитарная сеть с gap junctions, Ca²⁺ волнами и volume transmission.

    Оборачивает ``BrainSimulator`` и добавляет слой астроцитарной модуляции.

    Соответствует ``crates/kernels/src/neuron/astrocyte.rs::AstrocyteState``.

    Parameters
    ----------
    n_astro : int
        Количество астроцитов (обычно = количеству нейронов).
    gap_conn : GapJunctionConnectome
        Матрица gap junction связей.
    tau_astro_ms : float
        Постоянная времени Ca²⁺ волны (мс). 1000-10000 мс = 1-10 сек.
    g_gap : float
        Проводимость gap junction (по умолчанию 0.1).
    mu : float
        Коэффициент volume transmission (0.0 = без модуляции).
    v_init : float
        Начальный потенциал gap junction (-70 mV).
    ca_init : float
        Начальная концентрация Ca²⁺ (0.1).
    """

    def __init__(
        self,
        simulator,
        n_astro: int,
        gap_conn: Optional[GapJunctionConnectome] = None,
        tau_astro_ms: float = 2000.0,
        g_gap: float = 0.1,
        mu: float = 0.5,
        v_init: float = -70.0,
        ca_init: float = 0.1,
    ):
        self.sim = simulator
        self.n = n_astro

        # Состояние астроцитов
        self.ca = np.full(n_astro, ca_init, dtype=np.float32)
        self.v_gap = np.full(n_astro, v_init, dtype=np.float32)
        self.volume_gain = np.ones(n_astro, dtype=np.float32)

        # Параметры
        self.tau_astro_ms = tau_astro_ms
        self.g_gap = g_gap
        self.mu = mu

        # Gap junction связи
        self.gap_conn = gap_conn or GapJunctionConnectome()
        if self.gap_conn.pairs:
            self.gap_conn.build(n_astro)

        # История для анализа
        self.history: Dict[str, List[np.ndarray]] = defaultdict(list)

    def reset(self):
        """Сброс состояния астроцитов."""
        self.ca.fill(0.1)
        self.v_gap.fill(-70.0)
        self.volume_gain.fill(1.0)
        self.history.clear()

    def step(self, dt_ms: float, nearby_spikes: np.ndarray):
        """Один шаг астроцитарной динамики.

        Parameters
        ----------
        dt_ms : float
            Шаг по времени (мс).
        nearby_spikes : np.ndarray, shape (n,)
            Спайковая активность нейронов на текущем шаге (0/1).
        """
        alpha = np.exp(-dt_ms / max(self.tau_astro_ms, 1.0))

        # 1. Gap junction токи
        i_gap = self.gap_conn.compute_gap_currents(self.v_gap)

        # 2. Обновление V_gap
        self.v_gap += i_gap * dt_ms * 0.1

        # 3. Ca²⁺ динамика: затухание + выброс от спайков
        spike_input = nearby_spikes.astype(np.float32) * 0.2
        self.ca = self.ca * alpha + (1.0 - alpha) * 0.1 + spike_input

        # 4. Clamp Ca²⁺
        np.clip(self.ca, 0.0, 10.0, out=self.ca)

        # 5. Volume transmission gain: M_i = 1.0 + μ * Ca_i
        self.volume_gain = 1.0 + self.mu * self.ca

        # 6. Clamp gain
        np.clip(self.volume_gain, 1.0, 2.0, out=self.volume_gain)

    def modulate_I_ext(self, I_ext: np.ndarray) -> np.ndarray:
        """Модулировать внешний ток через volume transmission gain.

        M_i = 1.0 + μ · Ca_i  →  I_ext_i *= M_i

        Parameters
        ----------
        I_ext : np.ndarray, shape (n,)
            Внешний ток.

        Returns
        -------
        I_ext_mod : np.ndarray, shape (n,)
            Модулированный внешний ток.
        """
        return I_ext * self.volume_gain

    def simulate(
        self,
        connectome,
        n_steps: int,
        dt_ms: float = 0.5,
        external_I: float = 10.0,
        chunk_size: int = 100,
        scale: float = 1.0,
        record_astro: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Запустить симуляцию с астроцитарной модуляцией.

        Разбивает ``step_batch`` на chunks по ``chunk_size`` шагов,
        между которыми обновляет астроцитарную динамику и модулирует I_ext.

        Parameters
        ----------
        connectome : ConnectomeData
            CSR граф коннектома.
        n_steps : int
            Всего шагов симуляции.
        dt_ms : float
            Шаг по времени (мс).
        external_I : float
            Базовый внешний ток (одинаковый для всех нейронов).
        chunk_size : int
            Шагов между обновлениями астроцитов.
        scale : float
            Масштаб синаптических весов.
        record_astro : bool
            Записывать траекторию астроцитов.

        Returns
        -------
        spike_buf : np.ndarray, shape (n_steps, n)
            Спайковый буфер (0/1).
        I_ext_trace : np.ndarray или None
            Траектория модулированного I_ext, если record_astro=True.
        """
        if self.sim is None:
            raise ValueError(
                'AstrocyteNetwork: simulator not set — '
                'pass BrainSimulator to constructor'
            )

        n = self.n
        n_chunks = (n_steps + chunk_size - 1) // chunk_size

        all_spikes: List[np.ndarray] = []
        I_ext_trace = [] if record_astro else None

        self.reset()

        # Базовый I_ext вектор
        base_I_ext = np.full(n, external_I, dtype=np.float32)

        for ci in range(n_chunks):
            start = ci * chunk_size
            end = min(start + chunk_size, n_steps)
            chunk_n = end - start

            # Модулируем I_ext через volume transmission (in-place, сохраняет contiguous)
            I_ext_chunk = base_I_ext.copy()
            I_ext_chunk *= self.volume_gain  # in-place → гарантирует C-contiguous float32

            if record_astro:
                I_ext_trace.append(I_ext_chunk.copy())

            # Запускаем chunk симуляции
            chunk_spikes = self.sim.step_batch(
                connectome,
                n_steps=chunk_n,
                dt_ms=dt_ms,
                external_I=I_ext_chunk,
                scale=scale,
            )

            # Решейпим из (chunk_n * n,) в (chunk_n, n)
            chunk_spikes_2d = chunk_spikes.reshape(chunk_n, n)

            # Обновляем астроциты по спайкам этого chunk'а
            for t in range(chunk_n):
                self.step(dt_ms, chunk_spikes_2d[t])

            all_spikes.append(chunk_spikes_2d)

        spike_buf = np.concatenate(all_spikes, axis=0)

        if record_astro:
            return spike_buf, np.concatenate(I_ext_trace, axis=0)
        return spike_buf, None

    def get_astro_state(self) -> Dict[str, np.ndarray]:
        """Вернуть текущее состояние астроцитов для анализа."""
        return {
            'ca': self.ca.copy(),
            'v_gap': self.v_gap.copy(),
            'volume_gain': self.volume_gain.copy(),
        }


# ═══════════════════════════════════════════════════════════
#  Convenience: построение gap junction для VNC non-reciprocal
# ═══════════════════════════════════════════════════════════

def build_vnc_gap_junctions(
    neuron_ids: List[int],
    nt_types: np.ndarray,
    g_gap_gaba: float = 0.05,
    g_gap_ach: float = 0.15,
    g_gap_glut: float = 0.10,
    p_gaba: float = 0.03,
    p_ach: float = 0.08,
    p_glut: float = 0.06,
    seed: int = 42,
) -> GapJunctionConnectome:
    """Построить gap junction связи для VNC non-reciprocal нейронов.

    Разные типы нейронов имеют разную вероятность gap junction:
    - GABA (ингибиторные): низкая вероятность, слабая связь
    - ACh (возбуждающие): высокая вероятность, сильная связь
    - GLUT (возбуждающие): средняя вероятность

    Parameters
    ----------
    neuron_ids : list[int]
        ID нейронов для gap junction связей.
    nt_types : np.ndarray, shape (n_neurons,)
        Типы нейротрансмиттеров: 1=ACh, -1=GABA, 2=GLUT.
    g_gap_gaba, g_gap_ach, g_gap_glut : float
        Проводимости gap junction.
    p_gaba, p_ach, p_glut : float
        Вероятности связи.
    seed : int
        Seed для RNG.

    Returns
    -------
    gj : GapJunctionConnectome
    """
    gj = GapJunctionConnectome()
    rng = np.random.default_rng(seed)
    n = len(neuron_ids)

    for a in range(n):
        nt_a = nt_types[neuron_ids[a]] if neuron_ids[a] < len(nt_types) else 0
        for b in range(a + 1, n):
            nt_b = nt_types[neuron_ids[b]] if neuron_ids[b] < len(nt_types) else 0

            # Определяем параметры по типу NT
            if nt_a == -1 or nt_b == -1:
                g, p = g_gap_gaba, p_gaba
            elif nt_a == 2 or nt_b == 2:
                g, p = g_gap_glut, p_glut
            else:
                g, p = g_gap_ach, p_ach

            if rng.random() < p:
                gj.add(neuron_ids[a], neuron_ids[b], g)

    return gj


def build_vnc_top5_gap_junctions(
    top5_chains: List[List[int]],
    n_total: int,
    g_within: float = 0.2,
    g_between: float = 0.05,
    g_to_cluster5: float = 0.1,
    cluster5_ids: Optional[List[int]] = None,
) -> GapJunctionConnectome:
    """Построить gap junction связи для топ-5 CPG кандидатов VNC.

    Соединяет нейроны внутри каждой цепи (сильная связь),
    между цепями (слабая связь) и с Cluster 5/GABA пулом.

    Parameters
    ----------
    top5_chains : list[list[int]]
        Списки нейронов для каждой из топ-5 цепей.
    n_total : int
        Общее количество нейронов.
    g_within : float
        Проводимость внутри цепи.
    g_between : float
        Проводимость между цепями.
    g_to_cluster5 : float
        Проводимость к Cluster 5 GABA нейронам.
    cluster5_ids : list[int] or None
        ID нейронов Cluster 5.

    Returns
    -------
    gj : GapJunctionConnectome
    """
    gj = GapJunctionConnectome()
    all_chain_ids = set()

    # Внутри каждой цепи — сильные gap junctions
    for chain in top5_chains:
        for i in range(len(chain)):
            all_chain_ids.add(chain[i])
            for j in range(i + 1, len(chain)):
                gj.add(chain[i], chain[j], g_within)

    # Между цепями — слабые gap junctions
    chain_ids_list = [set(c) for c in top5_chains]
    for ci in range(len(chain_ids_list)):
        for cj in range(ci + 1, len(chain_ids_list)):
            for i in chain_ids_list[ci]:
                for j in chain_ids_list[cj]:
                    gj.add(i, j, g_between)

    # К Cluster 5 — средняя связь
    if cluster5_ids:
        for cid in all_chain_ids:
            for gid in cluster5_ids:
                if gid < n_total:
                    gj.add(cid, gid, g_to_cluster5)

    return gj


# ═══════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════

def _test_gap_junction():
    """Тест gap junction проводимости (аналог Rust теста)."""
    gj = GapJunctionConnectome()
    gj.add(0, 1, 0.2)
    gj.add(1, 2, 0.2)
    gj.add(2, 3, 0.2)

    v = np.array([-30.0, -70.0, -70.0, -70.0], dtype=np.float32)
    i_gap = gj.compute_gap_currents(v)

    assert abs(i_gap[0] - 0.2 * (-70.0 - (-30.0))) < 1e-5, \
        f"I_gap[0] = {i_gap[0]}, expected {0.2 * (-70.0 - (-30.0))}"
    assert i_gap[1] > 0, f"I_gap[1] should be positive (depol), got {i_gap[1]}"
    print(f"  ✅ Gap junction test passed")
    print(f"     I_gap = {i_gap}")


def _test_astrocyte_step():
    """Тест шага астроцитарной динамики (аналог Rust теста)."""
    n_astro = 4
    gj = GapJunctionConnectome()
    gj.add(0, 1, 0.2)
    gj.add(1, 2, 0.2)
    gj.add(2, 3, 0.2)

    astro = AstrocyteNetwork(
        simulator=None,
        n_astro=n_astro,
        gap_conn=gj,
        tau_astro_ms=1000.0,
        g_gap=0.2,
        mu=0.5,
    )

    # Подача потенциала на первый узел
    astro.v_gap[0] = -30.0
    spikes = np.array([1, 0, 0, 0], dtype=np.int32)

    for _ in range(100):
        astro.step(1.0, spikes)

    assert astro.v_gap[1] > -70.0, \
        f"V_gap[1] should depolarize, got {astro.v_gap[1]}"
    assert astro.ca[0] > 0.1, \
        f"Ca[0] should rise, got {astro.ca[0]}"
    assert astro.volume_gain[0] > 1.0, \
        f"Gain[0] should increase, got {astro.volume_gain[0]}"

    print(f"  ✅ Astrocyte step test passed")
    print(f"     V_gap[1] = {astro.v_gap[1]:.2f} mV")
    print(f"     Ca[0] = {astro.ca[0]:.3f}")
    print(f"     Gain[0] = {astro.volume_gain[0]:.3f}")


if __name__ == '__main__':
    print("🧫 VNC Astrocyte Layer — Self-test\n")
    _test_gap_junction()
    _test_astrocyte_step()
    print("\n✅ All tests passed!")
