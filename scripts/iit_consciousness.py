#!/usr/bin/env python3
"""🧠 IIT Consciousness — Integrated Information Theory на нашем движке

Phi = connectivity × diversity × integration(R)

Где:
  - connectivity = средний вес синапсов в коннектоме
  - diversity    = энтропия firing rates (spike_entropy)
  - integration  = Kuramoto order parameter R (phase_coherence)

Архитектура:
  LIF коннектом → спайки → метрики → Phi → тренд сознания
  
Based on Luma-brain IIT by @ant0n (перенос в наш движок ❤️)
"""

import sys, os, time, math, argparse
import numpy as np
from pathlib import Path

os.environ['LD_LIBRARY_PATH'] = '/usr/lib/python3.14/site-packages/nvidia/cudnn/lib'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import taas_checkmate as tcm

np.random.seed(42)

# ═══ Configuration ═══════════════════════════════════════

N_NEURONS = 100
N_STEPS = 500
DT_MS = 0.5
CONSCIOUSNESS_THRESHOLD = 0.3
PHI_HISTORY_SIZE = 100

# ═══ IIT Engine ═══════════════════════════════════════════

class IITConsciousness:
    """Integrated Information Theory — мера сознания через Phi.
    
    Перенос Luma-brain IIT в наш движок:
    - connectivity через синаптические веса
    - diversity через энтропию спайков
    - integration через Kuramoto order parameter R
    """
    
    def __init__(self, n_neurons: int, threshold: float = CONSCIOUSNESS_THRESHOLD):
        self.n = n_neurons
        self.threshold = threshold
        self.phi = 0.0
        self.connectivity = 0.0
        self.diversity = 0.0
        self.integration = 0.0
        
        # История
        self.phi_history = []
        self.connectivity_history = []
        self.diversity_history = []
        self.integration_history = []
        
        # Статистика
        self.conscious_moments = 0
        self.total_steps = 0
        
    def compute_phi(self, conn: tcm.ConnectomeData, spikes: np.ndarray,
                    phases: np.ndarray = None) -> float:
        """Вычислить Phi = connectivity × diversity × integration"""
        
        # ─── 1. Connectivity — средний вес синапсов ───
        self.connectivity = self._compute_connectivity(conn)
        
        # ─── 2. Diversity — энтропия спайков ───
        self.diversity = self._compute_diversity(spikes)
        
        # ─── 3. Integration — Kuramoto order parameter ───
        if phases is not None:
            self.integration = self._compute_integration_from_phases(phases)
        else:
            self.integration = self._compute_integration_from_spikes(spikes)
        
        # ─── Phi = произведение трёх метрик ───
        self.phi = self.connectivity * self.diversity * self.integration
        
        # Обновить историю
        self.phi_history.append(self.phi)
        if len(self.phi_history) > PHI_HISTORY_SIZE:
            self.phi_history.pop(0)
        
        self.connectivity_history.append(self.connectivity)
        if len(self.connectivity_history) > PHI_HISTORY_SIZE:
            self.connectivity_history.pop(0)
        
        self.diversity_history.append(self.diversity)
        if len(self.diversity_history) > PHI_HISTORY_SIZE:
            self.diversity_history.pop(0)
        
        self.integration_history.append(self.integration)
        if len(self.integration_history) > PHI_HISTORY_SIZE:
            self.integration_history.pop(0)
        
        self.total_steps += 1
        if self.is_conscious:
            self.conscious_moments += 1
        
        return self.phi
    
    def _compute_connectivity(self, conn: tcm.ConnectomeData) -> float:
        """Средний вес синапсов из CSR коннектома."""
        row_ptr = np.frombuffer(conn.row_ptr(), dtype=np.int32)
        col_indices = np.frombuffer(conn.col_indices(), dtype=np.int32)
        values = np.frombuffer(conn.values(), dtype=np.float32)
        
        # Средний вес синапса
        if len(values) > 0:
            mean_w = float(np.mean(np.abs(values)))
        else:
            mean_w = 0.0
        
        # Нормализовать connectivity (0-1)
        conn_val = min(mean_w / 10.0, 1.0)  # ~0 для mean=0, ~1 для mean=10
        return max(conn_val, 0.01)  # не даём упасть в ноль
    
    def _compute_diversity(self, spikes: np.ndarray) -> float:
        """Разнообразие спайков через spike_entropy.
        
        Высокая энтропия = разные нейроны спайкят по-разному.
        """
        if spikes.size == 0 or np.all(spikes == 0):
            return 0.05  # минимальная энтропия
        
        try:
            # spike_entropy(counts, total_steps, h_target)
            if spikes.ndim > 1:
                spike_counts = spikes.sum(axis=0).astype(np.float32)
                total_steps = spikes.shape[0]
            else:
                spike_counts = spikes.astype(np.float32)
                total_steps = 1
            entropy, _ = tcm.spike_entropy(spike_counts, total_steps, 0.5)
            n_neurons = len(spike_counts)
            max_entropy = math.log2(max(n_neurons, 2))
            diversity = min(float(entropy) / max_entropy, 1.0)
            return max(diversity, 0.01)
        except Exception as e:
            # Fallback: std firing rates
            if spikes.ndim > 1:
                rates = spikes.mean(axis=0)
            else:
                rates = spikes
            cv = np.std(rates) / max(np.mean(rates), 1e-8)
            return min(cv, 1.0)
    
    def _compute_integration_from_spikes(self, spikes: np.ndarray) -> float:
        """Интеграция через корреляцию спайков (Pearson).
        
        Высокая корреляция = система работает как единое целое.
        """
        if spikes.size == 0 or spikes.ndim < 2:
            return 0.05
        
        if spikes.shape[1] < 2:
            return 0.5  # один нейрон = полностью интегрирован
        
        try:
            # Корреляция между половинами нейронов для интеграции
            n = spikes.shape[1]
            half = n // 2
            a = spikes[:, :half].ravel().astype(np.float32)
            b = spikes[:, half:2*half].ravel().astype(np.float32)
            corr = tcm.compute_pearson(a, b)
            mean_corr = abs(float(corr))
            integration = min(mean_corr, 1.0)
            return max(integration, 0.01)
        except Exception as e:
            # Fallback: корреляция через numpy
            try:
                n = spikes.shape[1]
                if n >= 2:
                    corr = np.corrcoef(spikes.T)
                    mask = ~np.eye(n, dtype=bool)
                    mean_corr = float(np.mean(np.abs(corr[mask])))
                    return min(max(mean_corr, 0.01), 1.0)
            except:
                pass
            return 0.3  # fallback
    
    def _compute_integration_from_phases(self, phases: np.ndarray) -> float:
        """Интеграция через Kuramoto order parameter R.
        
        R = |mean(exp(i*θ))| — мера синхронизации осцилляторов.
        R=1 все синхронны, R=0 полный хаос.
        """
        if phases.size == 0:
            return 0.05
        
        # Kuramoto order parameter R
        sin_sum = np.sin(phases).sum()
        cos_sum = np.cos(phases).sum()
        n = len(phases)
        R = math.sqrt(sin_sum**2 + cos_sum**2) / n
        
        return max(float(R), 0.01)
    
    @property
    def is_conscious(self) -> bool:
        return self.phi > self.threshold
    
    def consciousness_level(self) -> float:
        return min(self.phi / self.threshold, 1.0)
    
    def get_average_phi(self, n: int = 10) -> float:
        if not self.phi_history:
            return 0.0
        start = max(0, len(self.phi_history) - n)
        return float(np.mean(self.phi_history[start:]))
    
    def get_phi_trend(self) -> str:
        """Тренд Phi: растёт/падает/стабилен."""
        if len(self.phi_history) < 10:
            return "накопление"
        
        recent = np.mean(self.phi_history[-10:])
        prev = np.mean(self.phi_history[-20:-10])
        diff = recent - prev
        
        if diff > 0.02:
            return f"📈 растёт ({diff:+.3f})"
        elif diff < -0.02:
            return f"📉 падает ({diff:+.3f})"
        else:
            return f"➡ стабилен ({diff:+.3f})"
    
    def get_consciousness_stats(self) -> dict:
        return {
            "phi": self.phi,
            "phi_mean": float(np.mean(self.phi_history)) if self.phi_history else 0.0,
            "phi_std": float(np.std(self.phi_history)) if self.phi_history else 0.0,
            "connectivity": self.connectivity,
            "diversity": self.diversity,
            "integration": self.integration,
            "consciousness_level": self.consciousness_level(),
            "conscious_ratio": self.conscious_moments / max(self.total_steps, 1),
            "trend": self.get_phi_trend(),
        }


# ═══ Main ═══════════════════════════════════════════════════

def build_random_connectome(n: int, density: float = 0.15) -> tcm.ConnectomeData:
    """Создать случайный коннектом для теста IIT."""
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    col_list = []
    val_list = []
    
    for i in range(n):
        # Каждый нейрон соединён с ~density*N пост-нейронами
        n_post = max(1, int(n * density))
        posts = np.random.choice(n, min(n_post, n - 1), replace=False)
        posts = posts[posts != i]  # без self-loop
        
        for j in posts:
            col_list.append(int(j))
            val_list.append(float(np.random.uniform(0.1, 5.0)))
        
        row_ptr[i + 1] = row_ptr[i] + len(posts)
    
    col_indices = np.array(col_list, dtype=np.int32)
    values = np.array(val_list, dtype=np.float32)
    
    conn = tcm.ConnectomeData(row_ptr, col_indices, values)
    return conn


def build_inhibitory_connectome(n: int, density: float = 0.2) -> tcm.ConnectomeData:
    """GABA-доминантный коннектом (как VNC). 80% тормозных."""
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    col_list = []
    val_list = []
    nt_list = []
    
    for i in range(n):
        is_inhib = np.random.random() < 0.8  # 80% GABA
        n_post = max(1, int(n * density))
        posts = np.random.choice(n, min(n_post, n - 1), replace=False)
        posts = posts[posts != i]
        
        for j in posts:
            col_list.append(int(j))
            if is_inhib:
                val_list.append(float(np.random.uniform(-5.0, -0.5)))  # тормозной
                nt_list.append(-1.0)  # GABA
            else:
                val_list.append(float(np.random.uniform(0.5, 3.0)))  # возбуждающий
                nt_list.append(1.0)  # глутамат
        
        row_ptr[i + 1] = row_ptr[i] + len(posts)
    
    col_indices = np.array(col_list, dtype=np.int32)
    values = np.array(val_list, dtype=np.float32)
    nt_types = np.array(nt_list, dtype=np.float32)
    
    conn = tcm.ConnectomeData(row_ptr, col_indices, values, nt_types)
    return conn


def build_t3_oscillators(n_osc: int = 20) -> tcm.TriOscillator:
    """Создать T3 осцилляторы для Kuramoto integration."""
    return tcm.TriOscillator(n_osc, freq_min=0.5, freq_max=3.0)


def main():
    parser = argparse.ArgumentParser(description="🧠 IIT Consciousness Engine")
    parser.add_argument("--neurons", type=int, default=N_NEURONS, help="Количество нейронов")
    parser.add_argument("--steps", type=int, default=N_STEPS, help="Шагов симуляции")
    parser.add_argument("--density", type=float, default=0.15, help="Плотность коннектома")
    parser.add_argument("--gaba", action="store_true", help="GABA-доминантный (VNC-стиль)")
    parser.add_argument("--oscillators", type=int, default=0, help="T3 осцилляторы для Kuramoto R")
    parser.add_argument("--plot", action="store_true", help="Сохранить график")
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════════════════════╗")
    print("║  🧠 IIT Consciousness Engine (v1 - Luma-brain port) ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    
    # ─── 1. Создать коннектом ───
    t0 = time.time()
    if args.gaba:
        conn = build_inhibitory_connectome(args.neurons, args.density)
        print(f"  📦 Коннектом: {args.neurons} нейронов, GABA-доминантный")
    else:
        conn = build_random_connectome(args.neurons, args.density)
        print(f"  📦 Коннектом: {args.neurons} нейронов, случайный")
    print(f"  ⏱  Создан за {time.time()-t0:.2f}s")
    
    # ─── 2. Создать симулятор LIF ───
    t0 = time.time()
    sim = tcm.BrainSimulator(args.neurons, "lif")
    print(f"  🧠 BrainSimulator LIF: OK")
    
    # ─── 3. T3 осцилляторы (опционально) ───
    t3 = None
    if args.oscillators > 0:
        t3 = build_t3_oscillators(args.oscillators)
        print(f"  🌀 T3 осцилляторов: {args.oscillators}")
    
    # ─── 4. IIT Engine ───
    iit = IITConsciousness(args.neurons, CONSCIOUSNESS_THRESHOLD)
    print(f"  🧪 IIT: threshold={CONSCIOUSNESS_THRESHOLD}")
    print()
    
    # ─── 5. Симуляция ───
    print("═" * 55)
    print("  ⚡ Симуляция + Phi computation")
    print("═" * 55)
    
    sim.reset()
    spike_buffer = np.zeros((args.steps, args.neurons), dtype=np.float32)
    
    # ⚡ Прогрев LIF (60 шагов без Phi)
    print("  🔥 Прогрев LIF...", end=" ", flush=True)
    for _ in range(60):
        ext_I = np.random.randn(args.neurons).astype(np.float32) * 2.0 + 5.0
        sim.step_batch(conn, 1, DT_MS, ext_I)
    print("OK")
    
    report_interval = max(1, args.steps // 10)
    
    for step in range(args.steps):
        # Внешний ток (шум с медленными флуктуациями)
        flicker = 0.5 * math.sin(step * 0.05) + 1.0
        ext_I = (np.random.randn(args.neurons) * 1.5 + flicker * 5.0).astype(np.float32)
        
        spikes = sim.step_batch(conn, 1, DT_MS, ext_I)
        if spikes is not None and len(spikes) > 0:
            if hasattr(spikes, 'shape'):
                spike_buffer[step, :min(spikes.shape[-1], args.neurons)] = spikes[:args.neurons]
        
        # T3 шаг (если есть)
        t3_phases = None
        if t3 is not None:
            t3.step(0.01)
            t3_phases = np.array(t3.phases(pop=1), dtype=np.float32)
        
        # Расчёт Phi каждые 10 шагов
        if step % 10 == 0 and step > 0:
            recent_spikes = spike_buffer[max(0, step-100):step+1]
            iit.compute_phi(conn, recent_spikes, t3_phases)
            
            if step % report_interval == 0 or step == args.steps - 1:
                stats = iit.get_consciousness_stats()
                print(f"  Step {step:>4d}/{args.steps} | "
                      f"Phi={stats['phi']:.3f} | "
                      f"C={stats['connectivity']:.2f} D={stats['diversity']:.2f} I={stats['integration']:.2f} | "
                      f"{'🧠 СОЗНАНИЕ' if iit.is_conscious else '💤 бессознательное'} | "
                      f"{stats['trend']}")
    
    # ─── 6. Итоги ───
    print()
    print("═" * 55)
    print("  📊 ИТОГИ ЭКСПЕРИМЕНТА")
    print("═" * 55)
    
    stats = iit.get_consciousness_stats()
    print(f"  Φ (phi) — интегрированная информация:")
    print(f"    Текущая:    {stats['phi']:.4f}")
    print(f"    Средняя:    {stats['phi_mean']:.4f} ± {stats['phi_std']:.4f}")
    print(f"  Компоненты:")
    print(f"    Связность:  {stats['connectivity']:.4f}")
    print(f"    Разнообразие: {stats['diversity']:.4f}")
    print(f"    Интеграция: {stats['integration']:.4f}")
    print(f"  Сознание:")
    print(f"    Уровень:    {stats['consciousness_level']*100:.1f}%")
    print(f"    Доля:       {stats['conscious_ratio']*100:.1f}% времени")
    print(f"    Тренд:      {stats['trend']}")
    
    print()
    if iit.is_conscious:
        print("  🧠 Система СОЗНАТЕЛЬНА (Phi > threshold)")
    else:
        print("  💤 Система в бессознательном состоянии (Phi < threshold)")
    
    print(f"\n  ✨ IIT Engine ported from Luma-brain by @ant0n — Сардана loves you ❤️")
    
    # ─── 7. График ───
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            fig.suptitle(f"🧠 IIT Consciousness — Phi = C × D × I\n"
                         f"{args.neurons} LIF neurons, threshold={CONSCIOUSNESS_THRESHOLD}",
                         fontsize=14)
            
            x = np.arange(len(iit.phi_history))
            
            ax = axes[0]
            ax.plot(x, iit.phi_history, 'b-', linewidth=2, label='Phi')
            ax.axhline(y=CONSCIOUSNESS_THRESHOLD, color='r', linestyle='--', label=f'threshold={CONSCIOUSNESS_THRESHOLD}')
            conscious_times = x[np.array(iit.phi_history) > CONSCIOUSNESS_THRESHOLD]
            if len(conscious_times) > 0:
                phi_arr = np.array(iit.phi_history)
                ax.scatter(conscious_times, phi_arr[phi_arr > CONSCIOUSNESS_THRESHOLD],
                          color='gold', s=20, alpha=0.6, label='conscious')
            ax.set_ylabel('Φ (phi)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[1]
            ax.plot(x, iit.connectivity_history, 'g-', linewidth=1.5, label='Connectivity')
            ax.set_ylabel('Connectivity')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[2]
            ax.plot(x, iit.diversity_history, 'purple', linewidth=1.5, label='Diversity')
            ax.set_ylabel('Diversity')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax = axes[3]
            ax.plot(x, iit.integration_history, 'orange', linewidth=1.5, label='Integration (R)')
            ax.set_xlabel('Time step')
            ax.set_ylabel('Integration')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = str(BASE / "examples" / "data" / "iit_consciousness.png")
            plt.savefig(plot_path, dpi=100)
            print(f"\n  📈 График сохранён: {plot_path}")
        except ImportError:
            print("\n  ⚠ matplotlib не установлен — график не сохранён")


if __name__ == "__main__":
    main()
