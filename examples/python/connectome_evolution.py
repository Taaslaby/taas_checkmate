#!/usr/bin/env python3
"""
🧬 CONNECTOME EVOLUTION — Natural Selection of Synthetic Brains

Pipeline (никем никогда не делано):
  1. SardaanaCM CookieCutter генерирует POPULATION коннектомов
  2. Каждый коннектом → TCM BrainSimulator("hh") + CSR spike propagation
  3. T3 TriOscillator измеряет PST feature ENTROPY (= fitness)
  4. Топ-10% выживают (elite selection)
  5. Breed: GMM state crossover + mutation → следующее поколение
  6. Повторить N поколений → эволюция сложности

Гипотеза открытия:
  После N поколений отбора по PST-entropy, коннектомы покажут
  ФАЗОВЫЙ ПЕРЕХОД в размерности T3-состояний.
  В какой-то момент сложность резко скакнёт — это граница
  между "мёртвым" коннектомом и "живым" мозгом.

Usage:
  cd /home/taaslab/Проекты/asm_engines/taas_checkmate
  python examples/python/connectome_evolution.py --n-gen 10 --pop-size 30
  python examples/python/connectome_evolution.py --plot
"""

import numpy as np
import sys
import os
import argparse
import json
from typing import List, Tuple, Optional, Dict

# ─── Import SardaanaCM ────────────────────────────────────
SARDANA_PATH = "/home/taaslab/Проекты/asm_engines/sardaanaCM"
if SARDANA_PATH not in sys.path:
    sys.path.insert(0, SARDANA_PATH)

from src.connectome_builder import (
    CookieCutterGenerator, CookieCutterConfig,
    ConnectomeSample,
)
import taas_checkmate as tcm

# ─── Constants ────────────────────────────────────────────
N_NEURONS = 128         # small for speed (128 neurons × 500 steps × 30 pop)
N_STEPS = 300           # HH simulation steps (150ms @ 0.5ms dt)
DT_MS = 0.5
N_PULSE = 80
PULSE_AMP = 50.0
N_OSC = 16

# Evolution
POPULATION_SIZE = 30
ELITE_RATIO = 0.15
MUTATION_RATE = 0.3
N_GENERATIONS = 8
STATE_DIM = 4  # (d, g, c, b) from GMM


# ═══════════════════════════════════════════════════════════
#  1. Bridge: SardaanaCM → TCM ConnectomeData
# ═══════════════════════════════════════════════════════════

def sample_to_connectome(sample: ConnectomeSample) -> 'tcm.connectome_brain.ConnectomeData':
    """Convert SardaanaCM ConnectomeSample to TCM ConnectomeData."""
    # Prepare groups as list of (name, indices)
    groups_list = [(k, v) for k, v in sample.groups.items()] if sample.groups else None
    
    conn = tcm.connectome_brain.ConnectomeData(
        sample.row_ptr,
        sample.col_indices,
        sample.values,
        sample.nt_types,
        groups=groups_list,
    )
    return conn


# ═══════════════════════════════════════════════════════════
#  2. T3 PST Entropy — Fitness Function
# ═══════════════════════════════════════════════════════════

def compute_pst_entropy(
    conn,
    n_steps: int = N_STEPS,
    n_osc: int = N_OSC,
    pulse_amp: float = PULSE_AMP,
    n_pulse: int = N_PULSE,
    dt_ms: float = DT_MS,
) -> float:
    """
    Run HH simulation on connectome, measure T3 PST feature entropy.

    PST features = 18n+12-dimensional state vector from TriOscillator.
    Entropy = -sum(p * log(p)) where p = normalized PST variance across dims.
    Higher entropy = richer dynamics = fitter connectome.

    Returns entropy score (higher = better).
    """
    n = conn.n_neurons
    sensory = list(range(min(32, n)))
    sens8 = sensory[:8]

    # Create HH simulator
    sim = tcm.connectome_brain.BrainSimulator(n, "hh")
    
    # Pulse then rest
    spike_bufs = []
    for step in range(n_steps):
        ext_I = np.zeros(n, dtype=np.float32)
        if step < n_pulse:
            ext_I[sens8] = pulse_amp
        buf = sim.step_batch(conn, 1, dt_ms, ext_I, scale=5.0)
        spike_bufs.append(buf.copy())
    
    spike_flat = np.concatenate(spike_bufs).astype(np.int32)
    
    # ── T3 PST features ──
    t3 = tcm.t3.TriOscillator(n_osc, 0.5, 4.0)
    t3.set_coupling(0.2, 0.1, 0.05)

    pst_features = []
    for step in range(n_steps):
        step_spikes = spike_flat[step * n:(step + 1) * n]
        n_spikes = int(np.sum(step_spikes))
        rate_hz = n_spikes / (dt_ms * 0.001 * n)
        
        noise = np.random.uniform(0, 0.05, n_osc).astype(np.float32)
        drive = np.zeros(3 * n_osc, dtype=np.float32)
        drive[:n_osc] = np.clip(rate_hz * 0.02 + noise, 0.0, 2.0)
        
        pst = t3.step(dt_ms / 1000.0, drive)
        if step > 20:  # skip warmup
            pst_features.append(np.array(pst, dtype=np.float32))
    
    if not pst_features:
        return 0.0
    
    pst_arr = np.array(pst_features)  # [n_steps × pst_dim]
    
    # Entropy: variance across time dimension → normalize → entropy
    pst_var = np.var(pst_arr, axis=0)  # variance per PST dimension
    pst_var_norm = pst_var / (np.sum(pst_var) + 1e-10)
    
    # Shannon entropy: higher = more diverse dynamics
    entropy = -np.sum(pst_var_norm * np.log(pst_var_norm + 1e-10))
    
    # Normalize by log of dimension count
    max_entropy = np.log(len(pst_var_norm))
    entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Also penalize dead connectomes (zero spikes)
    total_spikes = int(np.sum(spike_flat))
    activity_bonus = min(1.0, total_spikes / 1000.0)
    
    return float(entropy_norm * (0.5 + 0.5 * activity_bonus))


# ═══════════════════════════════════════════════════════════
#  3. Evolution Engine
# ═══════════════════════════════════════════════════════════

class ConnectomeEvolution:
    """
    Evolves connectome populations through natural selection.
    
    Genotype: GMM states [N_neurons × state_dim]
    Phenotype: CSR connectome (via CookieCutter)
    Fitness: T3 PST entropy (complexity of HH dynamics)
    """
    
    def __init__(
        self,
        pop_size: int = POPULATION_SIZE,
        elite_ratio: float = ELITE_RATIO,
        mutation_rate: float = MUTATION_RATE,
        n_neurons: int = N_NEURONS,
        seed: int = 42,
    ):
        self.pop_size = pop_size
        self.elite_count = max(2, int(pop_size * elite_ratio))
        self.mutation_rate = mutation_rate
        self.n_neurons = n_neurons
        self.rng = np.random.RandomState(seed)
        
        # Create CookieCutter generator (random GMM + EdgePredictor)
        config = CookieCutterConfig(
            n_neurons=n_neurons,
            state_dim=STATE_DIM,
            seed=seed,
            use_gpu=False,
        )
        self.generator = CookieCutterGenerator(config)
        
        # Default nt_types (80% exc, 20% inh)
        self.nt_types = np.where(
            self.rng.rand(n_neurons) < 0.2, -1.0, 1.0
        ).astype(np.float32)
        
        # Default groups (sensory + hidden + motor)
        n_in = min(16, n_neurons // 8)
        self.groups = {
            'sensory': list(range(n_in)),
            'hidden': list(range(n_in, n_neurons - n_in)),
            'motor': list(range(n_neurons - n_in, n_neurons)),
        }
        
        self.features = np.zeros(10, dtype=np.float32)
        
        # Track evolution
        self.history = {
            'gen': [],
            'best_fitness': [],
            'mean_fitness': [],
            'fitness_std': [],
            'entropy': [],
            'total_spikes': [],
        }
    
    def _generate_connectome(self, states: np.ndarray) -> 'tcm.connectome_brain.ConnectomeData':
        """Generate TCM ConnectomeData from GMM states."""
        sample = self.generator.generate_with_structure(
            self.nt_types, self.groups, self.features,
            external_states=states,
        )
        return sample_to_connectome(sample)
    
    def _evaluate_fitness(self, conn) -> float:
        """Measure PST entropy for one connectome."""
        return compute_pst_entropy(conn)
    
    def _breed(
        self,
        elite_states: List[np.ndarray],
        elite_fitness: List[float],
    ) -> List[np.ndarray]:
        """
        Breed next generation from elite parents.
        
        Crossover: random mask between two parents
        Mutation: Gaussian noise scaled by mutation_rate
        """
        n_elite = len(elite_states)
        children = []
        
        # Convert fitness to selection probabilities
        fitness_arr = np.array(elite_fitness, dtype=np.float32)
        fitness_arr = fitness_arr - fitness_arr.min() + 0.01  # non-negative
        probs = fitness_arr / fitness_arr.sum()
        
        for _ in range(self.pop_size):
            # Select two parents (weighted by fitness)
            p1_idx = self.rng.choice(n_elite, p=probs)
            p2_idx = self.rng.choice(n_elite, p=probs)
            
            p1 = elite_states[p1_idx]
            p2 = elite_states[p2_idx]
            
            # Crossover: random mask
            mask = self.rng.rand(self.n_neurons, STATE_DIM) > 0.5
            child = np.where(mask, p1, p2).astype(np.float32)
            
            # Mutation: Gaussian noise with annealing
            noise = self.rng.randn(self.n_neurons, STATE_DIM).astype(np.float32)
            child += noise * self.mutation_rate
            
            children.append(child)
        
        return children
    
    def run(
        self,
        n_generations: int = N_GENERATIONS,
        verbose: bool = True,
    ) -> Dict:
        """Run evolution for N generations."""
        
        # ── Generation 0: random GMM states ──
        if verbose:
            print(f"\n{'='*65}")
            print(f"  🧬 CONNECTOME EVOLUTION")
            print(f"  {self.pop_size} pop × {n_generations} gen × {self.n_neurons} neurons")
            print(f"  Fitness: T3 PST entropy (HH dynamics complexity)")
            print(f"  Elite: {self.elite_count}/{self.pop_size}")
            print(f"{'='*65}")
        
        # Sample initial states from GMM
        all_states = []
        all_fitness = []
        
        if verbose:
            print(f"\n📦 Generation 0: sampling {self.pop_size} random connectomes...")
        
        for i in range(self.pop_size):
            # Sample fresh GMM states for each individual
            states = self.generator.gmm.sample_n(self.n_neurons, self.rng).astype(np.float32)
            conn = self._generate_connectome(states)
            fitness = self._evaluate_fitness(conn)
            
            all_states.append(states)
            all_fitness.append(fitness)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"    [{i+1}/{self.pop_size}] mean fitness so far: {np.mean(all_fitness):.4f}")
        
        self._track_generation(0, all_fitness, all_states, verbose)
        
        # ── Evolve ──
        for gen in range(1, n_generations + 1):
            # Elite selection
            idx_sorted = np.argsort(all_fitness)[::-1]
            elite_idx = idx_sorted[:self.elite_count]
            
            elite_states = [all_states[i] for i in elite_idx]
            elite_fitness = [all_fitness[i] for i in elite_idx]
            
            # Breed
            children_states = self._breed(elite_states, elite_fitness)
            
            # Evaluate children
            all_states = []
            all_fitness = []
            
            for i, states in enumerate(children_states):
                conn = self._generate_connectome(states)
                fitness = self._evaluate_fitness(conn)
                all_states.append(states)
                all_fitness.append(fitness)
            
            self._track_generation(gen, all_fitness, all_states, verbose)
        
        # ── Final summary ──
        if verbose:
            best_idx = np.argmax(all_fitness)
            print(f"\n{'='*65}")
            print(f"  🏆 EVOLUTION COMPLETE!")
            print(f"  Best fitness: {all_fitness[best_idx]:.4f}")
            print(f"  Initial best: {self.history['best_fitness'][0]:.4f}")
            print(f"  Improvement:  {all_fitness[best_idx] - self.history['best_fitness'][0]:.4f}")
            
            if all_fitness[best_idx] > self.history['best_fitness'][0] * 1.2:
                print(f"  ✅ EVOLUTION SUCCESSFUL — complexity increased >20%!")
            elif all_fitness[best_idx] > self.history['best_fitness'][0]:
                print(f"  ⚪ Mild improvement — try more generations or larger population")
            else:
                print(f"  ❌ No improvement — fitness landscape might be flat for random GMM")
            print(f"{'='*65}")
        
        return self.history
    
    def _track_generation(self, gen: int, fitness: List[float], states: List, verbose: bool):
        """Record generation stats."""
        f_arr = np.array(fitness)
        self.history['gen'].append(gen)
        self.history['best_fitness'].append(float(f_arr.max()))
        self.history['mean_fitness'].append(float(f_arr.mean()))
        self.history['fitness_std'].append(float(f_arr.std()))
        
        if verbose:
            print(f"\n  Gen {gen:2d}: best={f_arr.max():.4f} mean={f_arr.mean():.4f} ±{f_arr.std():.4f}")


# ═══════════════════════════════════════════════════════════
#  4. Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🧬 Connectome Evolution — Natural Selection of Synthetic Brains",
    )
    parser.add_argument('--n-gen', type=int, default=N_GENERATIONS,
                        help=f'Number of generations (default: {N_GENERATIONS})')
    parser.add_argument('--pop-size', type=int, default=POPULATION_SIZE,
                        help=f'Population size (default: {POPULATION_SIZE})')
    parser.add_argument('--neurons', type=int, default=N_NEURONS,
                        help=f'Neurons per connectome (default: {N_NEURONS})')
    parser.add_argument('--mutation', type=float, default=MUTATION_RATE,
                        help=f'Mutation rate (default: {MUTATION_RATE})')
    parser.add_argument('--elite', type=float, default=ELITE_RATIO,
                        help=f'Elite ratio (default: {ELITE_RATIO})')
    parser.add_argument('--plot', action='store_true',
                        help='Plot evolution results')
    args = parser.parse_args()
    
    # Run evolution
    evolution = ConnectomeEvolution(
        pop_size=args.pop_size,
        elite_ratio=args.elite,
        mutation_rate=args.mutation,
        n_neurons=args.neurons,
    )
    
    history = evolution.run(n_generations=args.n_gen, verbose=True)
    
    # ── Save results ──
    results_path = f"connectome_evolution_results.json"
    with open(results_path, 'w') as f:
        json.dump({k: [float(v) if isinstance(v, (np.floating,)) else v 
                       for v in vals] for k, vals in history.items()}, f, indent=2)
    print(f"\n  📊 Results saved: {results_path}")
    
    # ── Plot ──
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            gens = history['gen']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1.errorbar(gens, history['mean_fitness'], yerr=history['fitness_std'],
                        marker='o', capsize=4, color='blue', alpha=0.7)
            ax1.plot(gens, history['best_fitness'], marker='s', 
                    color='red', linewidth=2, label='Best')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness (PST Entropy)')
            ax1.set_title('🧬 Connectome Evolution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Improvement ratio
            base = history['best_fitness'][0]
            if base > 0:
                improvement = [(b / base - 1) * 100 for b in history['best_fitness']]
                ax2.plot(gens, improvement, marker='D', color='green', linewidth=2)
                ax2.axhline(y=20, color='gray', linestyle='--', alpha=0.5, label='+20% threshold')
                ax2.set_xlabel('Generation')
                ax2.set_ylabel('Improvement (%)')
                ax2.set_title('Fitness Improvement Over Baseline')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = "connectome_evolution.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  📈 Plot saved: {plot_path}")
            plt.show()
        except ImportError:
            print("  ⚠️  matplotlib not available, skipping plot")
    
    return 0


if __name__ == '__main__':
    main()
