#!/usr/bin/env python3
"""
💥 CONNECTOME BIG BANG — Dual Evolution of Structure + Ion Channels

ЧТО ЭТО:
  Единственный в мире эксперимент по ЭВОЛЮЦИОННОМУ ДИЗАЙНУ МОЗГА,
  где ОДНОВРЕМЕННО эволюционируют:
    Genotype A: GMM states (структура коннектома) — от trained FlyWire GMM
    Genotype B: Ion channel params (g_Na, g_K, g_leak) — от биологии
  
  Fitness = PST entropy × activity_bonus × avalanches
  
  После N поколений: сохраняем лучший коннектом как .taas

ГИПОТЕЗА ОТКРЫТИЯ:
  Эволюционировавшие коннектомы спонтанно покажут:
    📈 Power-law avalanche scaling (как в реальной коре)
    📉 1/f спектр мощности (как в реальном EEG)
    🧠 Критическое branching ratio ≈ 1.0

  Это докажет: brain-like dynamics = неизбежное следствие
  эволюции на сложность, а не уникальное свойство биологии.

ЗАПУСК:
  cd /home/taaslab/Проекты/asm_engines/taas_checkmate
  python examples/python/connectome_big_bang.py          # быстрый тест
  python examples/python/connectome_big_bang.py --full   # полная эволюция
  python examples/python/connectome_big_bang.py --plot   # с графиками
"""

import numpy as np
import sys, os, argparse, json, pickle, time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import defaultdict

# ─── Import SardaanaCM ────────────────────────────────────
SARDANA_PATH = "/home/taaslab/Проекты/asm_engines/sardaanaCM"
MODELS_PATH  = os.path.join(SARDANA_PATH, "models")
if SARDANA_PATH not in sys.path:
    sys.path.insert(0, SARDANA_PATH)

from src.connectome_builder import (
    CookieCutterGenerator, CookieCutterConfig,
    ConnectomeSample, TaasWriter, save_connectome,
)
import taas_checkmate as tcm

# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════

N_NEURONS = 128
N_STEPS   = 300
DT_MS     = 0.5
N_PULSE   = 80
PULSE_AMP = 50.0
N_OSC     = 16

POPULATION  = 20
ELITE_RATIO = 0.2
N_GEN       = 10
MUT_RATE    = 0.3
STATE_DIM   = 4

# Default ion channel bounds (HH squid axon)
G_NA_MIN, G_NA_MAX = 1.0, 200.0
G_K_MIN,  G_K_MAX  = 1.0, 80.0
G_L_MIN,  G_L_MAX  = 0.01, 3.0


# ═══════════════════════════════════════════════════════════
#  PHENOTYPE: Ion Channel Params
# ═══════════════════════════════════════════════════════════

@dataclass
class IonParams:
    g_Na: float = 120.0
    g_K: float = 36.0
    g_leak: float = 0.3
    E_Na: float = 50.0
    E_K: float = -77.0
    E_leak: float = -65.0
    
    @staticmethod
    def random(rng: np.random.RandomState) -> 'IonParams':
        return IonParams(
            g_Na = rng.uniform(20.0, 180.0),
            g_K  = rng.uniform(5.0, 60.0),
            g_leak = rng.uniform(0.05, 2.0),
        )
    
    def mutate(self, rate: float = 0.3, rng: Optional[np.random.RandomState] = None):
        if rng is None:
            rng = np.random.RandomState()
        # Log-normal mutations (multiplicative)
        self.g_Na    *= np.exp(rng.randn() * rate * 0.3)
        self.g_K     *= np.exp(rng.randn() * rate * 0.3)
        self.g_leak  *= np.exp(rng.randn() * rate * 0.3)
        # Clamp
        self.g_Na   = np.clip(self.g_Na, G_NA_MIN, G_NA_MAX)
        self.g_K    = np.clip(self.g_K, G_K_MIN, G_K_MAX)
        self.g_leak = np.clip(self.g_leak, G_L_MIN, G_L_MAX)
    
    def to_vec(self) -> np.ndarray:
        return np.array([self.g_Na, self.g_K, self.g_leak], dtype=np.float32)
    
    @classmethod
    def from_vec(cls, v: np.ndarray) -> 'IonParams':
        return cls(g_Na=float(v[0]), g_K=float(v[1]), g_leak=float(v[2]))


# ═══════════════════════════════════════════════════════════
#  FITNESS: PST Entropy + Avalanche Criticality
# ═══════════════════════════════════════════════════════════

def evaluate_fitness(conn, ion: IonParams) -> float:
    """
    Multi-objective fitness:
      60% PST entropy (T3 complexity)
      20% Avalanche criticality (power-law fit)
      20% Activity (sustained spiking)
    """
    n = conn.n_neurons
    sensory = list(range(min(32, n)))
    sens8 = sensory[:8]
    
    sim = tcm.connectome_brain.BrainSimulator(n, "hh")
    apply_hh(sim, ion)
    
    spike_bufs = []
    for step in range(N_STEPS):
        ext_I = np.zeros(n, dtype=np.float32)
        if step < N_PULSE:
            ext_I[sens8] = PULSE_AMP
        buf = sim.step_batch(conn, 1, DT_MS, ext_I, scale=5.0)
        spike_bufs.append(buf.copy())
    
    spike_flat = np.concatenate(spike_bufs).astype(np.int32)
    total = int(np.sum(spike_flat))
    
    # ── 1. PST entropy ──
    t3 = tcm.t3.TriOscillator(N_OSC, 0.5, 4.0)
    t3.set_coupling(0.2, 0.1, 0.05)
    
    pst_list = []
    for step in range(N_STEPS):
        step_spikes = spike_flat[step * n:(step + 1) * n]
        ns = int(np.sum(step_spikes))
        rate = ns / (DT_MS * 0.001 * n)
        noise = np.random.uniform(0, 0.05, N_OSC).astype(np.float32)
        drive = np.zeros(3 * N_OSC, dtype=np.float32)
        drive[:N_OSC] = np.clip(rate * 0.03 + noise, 0.0, 2.0)
        pst = t3.step(DT_MS / 1000.0, drive)
        if step > 20:
            pst_list.append(np.array(pst, dtype=np.float32))
    
    if not pst_list or total < 5:
        return 0.01  # dead brain
    
    pst_arr = np.array(pst_list)
    pst_var = np.var(pst_arr, axis=0)
    pst_var_norm = pst_var / (np.sum(pst_var) + 1e-10)
    entropy = -np.sum(pst_var_norm * np.log(pst_var_norm + 1e-10))
    max_ent = np.log(len(pst_var_norm))
    pst_score = entropy / max_ent if max_ent > 0 else 0.0
    
    # ── 2. Avalanche criticality ──
    arr = spike_flat.reshape(N_STEPS, n)
    pop = arr.sum(axis=1)  # spikes per step
    avalanche_sizes = []
    cur = 0
    for p in pop:
        if p > 0:
            cur += p
        else:
            if cur > 0:
                avalanche_sizes.append(cur)
            cur = 0
    if cur > 0:
        avalanche_sizes.append(cur)
    
    avalanche_score = 0.0
    if len(avalanche_sizes) >= 10:
        sizes = np.array(avalanche_sizes, dtype=np.float32)
        # Power-law fit via linear regression on log-log
        if len(sizes) >= 8:
            log_sizes = np.log10(sizes[sizes > 0] + 1)
            counts, bins = np.histogram(log_sizes, bins=min(8, len(set(log_sizes.astype(int)))))
            log_bins = (bins[:-1] + bins[1:]) / 2
            valid = counts > 0
            if np.sum(valid) >= 3:
                cnt = counts[valid].astype(np.float32)
                coeffs = np.polyfit(log_bins[valid], np.log(cnt + 1e-10), 1)
                slope = coeffs[0]
                # Ideal avalanche exponent α ≈ -1.5 (critical)
                avalanche_score = 1.0 - min(abs(slope + 1.5) / 2.0, 1.0)
    
    # ── 3. Activity score ──
    activity_score = min(1.0, total / (N_NEURONS * N_STEPS * 0.05))
    
    # ── Combined ──
    fitness = 0.6 * pst_score + 0.2 * avalanche_score + 0.2 * activity_score
    return float(fitness)


# ═══════════════════════════════════════════════════════════
#  EVOLUTION ENGINE
# ═══════════════════════════════════════════════════════════

class BigBang:
    """
    Dual evolution of connectome structure AND ion channel function.
    
    Each individual has:
      - states: [N × state_dim] GMM states (from trained CookieCutter)
      - ion: IonParams (g_Na, g_K, g_leak)
    
    Evolution: elite selection → crossover → mutation → next gen
    """
    
    def __init__(self, pop_size: int = POPULATION, n_neurons: int = N_NEURONS, seed: int = 42):
        self.pop_size = pop_size
        self.elite = max(2, int(pop_size * ELITE_RATIO))
        self.rng = np.random.RandomState(seed)
        self.n = n_neurons
        self.groups: Dict = {}
        
        # ── Load TRAINED CookieCutter ──
        print(f"\n  📥 Loading trained CookieCutter from {MODELS_PATH}/cookiecutter/")
        config = CookieCutterConfig(n_neurons=self.n, use_gpu=False)
        self.generator = CookieCutterGenerator.load_trained(
            os.path.join(MODELS_PATH, "cookiecutter"), config
        )
        
        # Default nt_types + groups
        self.nt_types = np.where(self.rng.rand(self.n) < 0.2, -1.0, 1.0).astype(np.float32)
        n_in = min(16, self.n // 8)
        self.groups = {
            'sensory': list(range(n_in)),
            'hidden': list(range(n_in, self.n - n_in)),
            'motor': list(range(self.n - n_in, self.n)),
        }
        self.features = np.zeros(10, dtype=np.float32)
        
        # Track evolution
        self.history = {
            'gen': [], 'best': [], 'mean': [], 'std': [],
            'best_g_na': [], 'best_g_k': [], 'best_g_leak': [],
            'avalanche': [], 'entropy': [], 'activity': [],
        }
        
        # ── Pre-compute trained GMM mean/std for mutation boundaries ──
        gmm = self.generator.gmm
        all_means = gmm.means.reshape(gmm.n_components, gmm.dim)
        self.gmm_mean = np.mean(all_means, axis=0)
        self.gmm_std  = np.std(all_means, axis=0).clip(0.01)
    
    def make_connectome(self, states: np.ndarray) -> 'tcm.connectome_brain.ConnectomeData':
        sample = self.generator.generate_with_structure(
            self.nt_types, self.groups, self.features,
            external_states=states,
        )
        groups_list = [(k, v) for k, v in sample.groups.items()] if sample.groups else None
        return tcm.connectome_brain.ConnectomeData(
            sample.row_ptr, sample.col_indices,
            sample.values, sample.nt_types,
            groups=groups_list,
        )
    
    def breed(self, elite_states: List, elite_ions: List, elite_fit: List) -> Tuple[List, List]:
        """Breed next generation: crossover states + mutate ions."""
        f = np.array(elite_fit, dtype=np.float32)
        f = f - f.min() + 0.01
        probs = f / f.sum()
        n_elite = len(elite_states)
        
        children_states, children_ions = [], []
        
        for _ in range(self.pop_size):
            # Crossover states
            p1 = elite_states[self.rng.choice(n_elite, p=probs)]
            p2 = elite_states[self.rng.choice(n_elite, p=probs)]
            mask = self.rng.rand(self.n, STATE_DIM) > 0.5
            child = np.where(mask, p1, p2).astype(np.float32)
            
            # Gaussian mutation (annealed)
            noise = self.rng.randn(self.n, STATE_DIM).astype(np.float32)
            child += noise * MUT_RATE * self.gmm_std[np.newaxis, :]
            
            children_states.append(child)
            
            # Breed ion params via crossover + mutate
            if self.rng.rand() < 0.5:
                p_i1 = self.rng.choice(n_elite, p=probs)
                children_ions.append(IonParams(
                    g_Na=elite_ions[p_i1].g_Na,
                    g_K=elite_ions[p_i1].g_K,
                    g_leak=elite_ions[p_i1].g_leak,
                ))
            else:
                p_i1 = self.rng.choice(n_elite, p=probs)
                p_i2 = self.rng.choice(n_elite, p=probs)
                mask_i = self.rng.rand(3) > 0.5
                v1 = elite_ions[p_i1].to_vec()
                v2 = elite_ions[p_i2].to_vec()
                child_v = np.where(mask_i, v1, v2)
                children_ions.append(IonParams.from_vec(child_v))
            
            # Mutate ion params
            children_ions[-1].mutate(MUT_RATE, self.rng)
        
        return children_states, children_ions
    
    def run(self, n_gen: int = N_GEN, verbose: bool = True):
        """Run dual evolution."""
        
        if verbose:
            print(f"{'='*65}")
            print(f"  💥 CONNECTOME BIG BANG")
            print(f"  Dual evolution: structure (GMM states) + function (ion channels)")
            print(f"  {self.pop_size} pop × {n_gen} gen × {self.n} neurons")
            print(f"  Trained CookieCutter: FlyWire GMM + EdgePredictor")
            print(f"  Fitness: PST entropy + avalanche criticality + activity")
            print(f"{'='*65}")
        
        # ── Gen 0: init from trained GMM ──
        if verbose:
            print(f"\n  🌱 Generation 0: sampling from trained FlyWire GMM...")
        
        all_states  = []
        all_ions    = []
        all_fitness = []
        
        for i in range(self.pop_size):
            # Sample GMM states (these already have biological structure!)
            states = self.generator.gmm.sample_n(self.n, self.rng).astype(np.float32)
            conn = self.make_connectome(states)
            ion = IonParams.random(self.rng)
            fit = evaluate_fitness(conn, ion)
            
            all_states.append(states)
            all_ions.append(ion)
            all_fitness.append(fit)
            
            if verbose and (i + 1) % 5 == 0:
                print(f"    [{i+1}/{self.pop_size}] fit={fit:.4f} "
                      f"gNa={ion.g_Na:.0f} gK={ion.g_K:.1f}")
        
        self._track(0, all_fitness, all_ions, verbose)
        
        # ── Evolve ──
        for gen in range(1, n_gen + 1):
            idx = np.argsort(all_fitness)[::-1]
            elite_idx = idx[:self.elite]
            
            elite_s = [all_states[i] for i in elite_idx]
            elite_i = [all_ions[i] for i in elite_idx]
            elite_f = [all_fitness[i] for i in elite_idx]
            
            children_s, children_i = self.breed(elite_s, elite_i, elite_f)
            
            all_states, all_ions, all_fitness = [], [], []
            for s, ion in zip(children_s, children_i):
                conn = self.make_connectome(s)
                fit = evaluate_fitness(conn, ion)
                all_states.append(s)
                all_ions.append(ion)
                all_fitness.append(fit)
            
            self._track(gen, all_fitness, all_ions, verbose)
        
        # ── Final ──
        best_idx = int(np.argmax(all_fitness))
        if verbose:
            print(f"\n{'='*65}")
            print(f"  🏆 EVOLUTION COMPLETE!")
            print(f"  Best fitness: {all_fitness[best_idx]:.4f}")
            print(f"  Gen 0 best:   {self.history['best'][0]:.4f}")
            imp = all_fitness[best_idx] - self.history['best'][0]
            pct = (all_fitness[best_idx] / max(self.history['best'][0], 0.01) - 1) * 100
            print(f"  Improvement:  {imp:+.4f} ({pct:+.1f}%)")
            print(f"  Best ion:     {all_ions[best_idx]}")
            print(f"{'='*65}")
        
        # ── Save best .taas ──
        best_states = all_states[best_idx]
        best_ion    = all_ions[best_idx]
        self._save_best(best_states, best_ion, all_fitness[best_idx])
        
        return self.history, best_states, best_ion
    
    def _track(self, gen: int, fitness: List, ions: List, verbose: bool):
        f_arr = np.array(fitness)
        best_idx = int(np.argmax(f_arr))
        
        self.history['gen'].append(gen)
        self.history['best'].append(float(f_arr.max()))
        self.history['mean'].append(float(f_arr.mean()))
        self.history['std'].append(float(f_arr.std()))
        self.history['best_g_na'].append(float(ions[best_idx].g_Na))
        self.history['best_g_k'].append(float(ions[best_idx].g_K))
        self.history['best_g_leak'].append(float(ions[best_idx].g_leak))
        
        if verbose:
            gen_str = f"Gen {gen:2d}"
            print(f"\n  {gen_str}: best={f_arr.max():.4f} mean={f_arr.mean():.4f} ±{f_arr.std():.4f}")
            print(f"         gNa={ions[best_idx].g_Na:.1f} gK={ions[best_idx].g_K:.1f} "
                  f"gL={ions[best_idx].g_leak:.3f}")
    
    def _save_best(self, states: np.ndarray, ion: IonParams, fitness: float):
        """Save best evolved connectome as .taas."""
        sample = self.generator.generate_with_structure(
            self.nt_types, self.groups, self.features,
            external_states=states,
        )
        writer = TaasWriter()
        writer.add_metadata("type", "EvolvedConnectome")
        writer.add_metadata("source", "connectome_big_bang")
        writer.add_metadata("n_neurons", str(sample.n_neurons))
        writer.add_metadata("n_synapses", str(sample.n_synapses))
        writer.add_metadata("fitness", f"{fitness:.4f}")
        writer.add_metadata("g_Na", f"{ion.g_Na:.2f}")
        writer.add_metadata("g_K", f"{ion.g_K:.2f}")
        writer.add_metadata("g_leak", f"{ion.g_leak:.4f}")
        save_connectome(sample, writer)
        
        path = "evolved_brain.taas"
        writer.save(path)
        print(f"\n  💾 Saved best evolved connectome: {path}")
        print(f"     {sample.n_neurons} neurons, {sample.n_synapses} synapses, "
              f"fitness={fitness:.4f}")


# ═══════════════════════════════════════════════════════════
#  ION PARAMS APPLY
# ═══════════════════════════════════════════════════════════

def apply_hh(sim, ion: IonParams):
    """Apply IonParams to BrainSimulator HH."""
    n = sim.n_neurons
    sim.set_hh_params(
        np.array([ion.g_Na, ion.g_K], dtype=np.float32),
        np.array([ion.E_Na, ion.E_K], dtype=np.float32),
        np.full(n, ion.g_leak, dtype=np.float32),
        np.full(n, ion.E_leak, dtype=np.float32),
        DT_MS,
    )


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="💥 Connectome Big Bang")
    parser.add_argument('--gen', type=int, default=N_GEN)
    parser.add_argument('--pop', type=int, default=POPULATION)
    parser.add_argument('--neurons', type=int, default=N_NEURONS)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    
    if args.full:
        args.gen = 15
        args.pop = 30
        args.neurons = 128
    
    t0 = time.time()
    
    bb = BigBang(pop_size=args.pop, n_neurons=args.neurons)
    history, best_states, best_ion = bb.run(n_gen=args.gen)
    
    t = time.time() - t0
    print(f"\n  ⏱️  Total time: {t:.0f}s ({t/60:.1f}min)")
    
    # ── Save history ──
    hist_path = "big_bang_history.json"
    with open(hist_path, 'w') as f:
        clean = {}
        for k, v in history.items():
            clean[k] = [float(x) if hasattr(x, '__float__') else x for x in v]
        json.dump(clean, f, indent=2)
    print(f"  📊 History saved: {hist_path}")
    
    # ── Plot ──
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            gens = history['gen']
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Fitness evolution
            ax = axes[0, 0]
            ax.errorbar(gens, history['mean'], yerr=history['std'],
                       marker='o', capsize=4, color='blue', alpha=0.5, label='Mean')
            ax.plot(gens, history['best'], marker='s', color='red', linewidth=2, label='Best')
            ax.set_xlabel('Generation'); ax.set_ylabel('Fitness')
            ax.set_title('💥 Connectome Big Bang — Fitness Evolution')
            ax.legend(); ax.grid(True, alpha=0.3)
            
            # Ion channel evolution
            ax = axes[0, 1]
            ax.plot(gens, history['best_g_na'], marker='o', color='red', label='g_Na')
            ax.plot(gens, history['best_g_k'], marker='s', color='blue', label='g_K')
            ax.axhline(y=120, color='red', linestyle='--', alpha=0.3, label='Default g_Na')
            ax.axhline(y=36, color='blue', linestyle='--', alpha=0.3, label='Default g_K')
            ax.set_xlabel('Generation'); ax.set_ylabel('Conductance (mS/cm²)')
            ax.set_title('Ion Channel Evolution')
            ax.legend(); ax.grid(True, alpha=0.3)
            
            # g_leak
            ax = axes[1, 0]
            ax.plot(gens, history['best_g_leak'], marker='D', color='green', linewidth=2)
            ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.3, label='Default g_leak')
            ax.set_xlabel('Generation'); ax.set_ylabel('g_leak (mS/cm²)')
            ax.set_title('Leak Conductance Evolution')
            ax.legend(); ax.grid(True, alpha=0.3)
            
            # Improvement %
            ax = axes[1, 1]
            base = history['best'][0]
            if base > 0:
                imp = [(b / base - 1) * 100 for b in history['best']]
                ax.plot(gens, imp, marker='D', color='purple', linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                ax.set_xlabel('Generation'); ax.set_ylabel('Improvement (%)')
                ax.set_title('Fitness Improvement Over Baseline')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('big_bang_results.png', dpi=150, bbox_inches='tight')
            print(f"  📈 Plot saved: big_bang_results.png")
            plt.show()
        except ImportError:
            print("  ⚠️  matplotlib not available")
    
    print(f"\n{'='*65}")
    print(f"  ✅ CONNECTOME BIG BANG COMPLETE!")
    print(f"  Best evolved brain saved as: evolved_brain.taas")
    print(f"  Load into epilepsy_bci.py or super_experiment.py!")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
