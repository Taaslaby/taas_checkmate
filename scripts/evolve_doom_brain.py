#!/usr/bin/env python3
"""
VizDoom Brain — Эволюция connectome через GPU LIF для игры в Doom.

Архитектура:
    screen (160×120 gray) → pool to 40×30 → W_in → sensory_neurons →
    GPU LIF connectome (1024) → motor_neurons → W_out → actions (3 buttons)

Сценарий: basic (corridor + monster)
Fitness: KILLCOUNT × 100 + HEALTH
"""

import os, sys, time, json, ctypes
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

BASE = Path('/home/taaslab/Проекты/asm_engines/taas_checkmate')
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'venv_doom' / 'lib' / 'python3.14' / 'site-packages'))

# ── Проверка taas_checkmate ──
try:
    import taas_checkmate as tcm
    HAS_TCM = True
except ImportError:
    HAS_TCM = False

# ── VizDoom scenarios path (cached) ──
import vizdoom as _vizdoom_mod
_DOOM_SCENARIOS = os.path.join(os.path.dirname(_vizdoom_mod.__file__), 'scenarios')

# ── GPU LIF lib ──
GPU_LIF_LIB = None

def _get_gpu_lif_lib():
    global GPU_LIF_LIB
    if GPU_LIF_LIB is not None:
        return GPU_LIF_LIB
    search = [
        BASE / 'target' / 'release' / 'libgpu_lif.so',
        BASE / 'target' / 'release' / 'libgpu_lif.dylib',
    ]
    for p in search:
        if p.exists():
            try:
                lib = ctypes.CDLL(str(p))
                GPU_LIF_LIB = lib
                print(f"  ✅ GPU LIF: {p}")
                return lib
            except Exception as e:
                print(f"  ⚠ GPU load: {e}")
    print("  ⚠ GPU lib not found")
    return None

def _setup_gpu_functions(lib):
    """Set restype/argtypes for all GPU functions."""
    lib.gpu_lif_create.restype = ctypes.c_int32
    lib.gpu_lif_create.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
    ]
    lib.gpu_lif_step.restype = ctypes.c_int32
    lib.gpu_lif_step.argtypes = [
        ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int32,
    ]
    lib.gpu_lif_read_v.restype = ctypes.c_int32
    lib.gpu_lif_read_v.argtypes = [
        ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
    ]
    lib.gpu_lif_read_spikes.restype = ctypes.c_int32
    lib.gpu_lif_read_spikes.argtypes = [
        ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
    ]


# ═══════════════════════════════════════════════════
#  1. ЗАГРУЗКА ГРАФА
# ═══════════════════════════════════════════════════

COOKIE_DIR = Path('/home/taaslab/Проекты/asm_engines/MEIIgpt/experiments/sardana_2')

def load_graph() -> dict:
    """Загрузить cookiecutter граф."""
    npz_path = COOKIE_DIR / 'data' / 'cookiecutter_graph.npz'
    if npz_path.exists():
        data = np.load(npz_path)
    else:
        n = 1024
        deg = np.random.poisson(8, n).clip(1, 50)
        row_ptr = np.zeros(n + 1, dtype=np.int32)
        cols = []
        for i in range(n):
            targets = set()
            while len(targets) < deg[i]:
                t = np.random.randint(0, n)
                if t != i: targets.add(t)
            cols.extend(sorted(targets))
            row_ptr[i+1] = len(cols)
        data = {'row_ptr': row_ptr, 'col_indices': np.array(cols, dtype=np.int32), 'out_deg': deg}
    
    row_ptr = data['row_ptr'].astype(np.int32)
    col_indices = data['col_indices'].astype(np.int32)
    out_deg = np.diff(row_ptr) if 'out_deg' not in data else data['out_deg']
    n = len(out_deg)
    e = len(col_indices)
    print(f"  Граф: {n} нейронов, {e} синапсов, avg deg={np.mean(out_deg):.1f}")
    return {'row_ptr': row_ptr, 'col_indices': col_indices, 'n_neurons': n, 'n_edges': e}


def assign_ei(row_ptr, col_indices, n, target_mean_abs=0.3) -> Tuple[np.ndarray, np.ndarray]:
    """E/I assignment + log-normal weights."""
    n_exc = int(n * 0.8)
    nt_types = np.ones(n, dtype=np.float32)
    nt_types[n_exc:] = -1.0
    np.random.shuffle(nt_types)
    
    values = np.zeros(len(col_indices), dtype=np.float32)
    for i in range(n):
        s, e = row_ptr[i], row_ptr[i+1]
        ns = e - s
        if ns == 0: continue
        sign = 1.0 if nt_types[i] > 0 else -1.0
        mag = np.exp(np.random.randn(ns) * 0.8 - 0.5).clip(0.01, 2.0)
        values[s:e] = sign * mag
    
    ma = np.abs(values).mean()
    if ma > 0: values *= target_mean_abs / ma
    print(f"  E/I: E={(nt_types>0).sum()}, I={(nt_types<0).sum()}")
    print(f"  Веса: mean={values.mean():.4f}, |mean|={np.abs(values).mean():.4f}")
    return nt_types, values


# ═══════════════════════════════════════════════════
#  2. DOOM FITNESS
# ═══════════════════════════════════════════════════

class DoomFitness:
    """VizDoom basic: pixel input → GPU LIF connectome → actions.
    
    Screen: 160×120 gray → pool to 40×30 (1200 px) → W_in → 200 sensory
    Actions: MOVE_LEFT, MOVE_RIGHT, ATTACK (3 buttons)
    Fitness = KILLCOUNT × 100 + HEALTH
    """
    
    def __init__(self, row_ptr, col_indices, nt_types,
                 pool_h=15, pool_w=20,  # pooled resolution (20×15=300)
                 n_sensory=100, n_motor=200,
                 act_dim=3,              # MOVE_LEFT, MOVE_RIGHT, ATTACK
                 n_internal=5, dt_ms=1.0,
                 seed=42):
        self.row_ptr = row_ptr
        self.col_indices = col_indices
        self.nt_types = nt_types
        self.n = len(nt_types)
        self.pool_h, self.pool_w = pool_h, pool_w
        self.pool_dim = pool_h * pool_w  # 1200
        self.n_sensory = n_sensory
        self.n_motor = n_motor
        self.act_dim = act_dim
        self.n_internal = n_internal
        self.dt_ms = dt_ms
        self._base_seed = seed
        
        print(f"  Doom: 160×120px → {pool_w}×{pool_h} pooled → {self.pool_dim} features → "
          f"{n_sensory} sensory → {n_motor} motor → {act_dim} actions")
        
        rng = np.random.default_rng(seed)
        
        # Downsample factors
        self.pool_factors = (120 // pool_h, 160 // pool_w)  # (block_h, block_w)
        
        # W_in: pooled pixels → sensory neurons (sparse 15%)
        self.w_in = np.zeros((self.pool_dim, n_sensory), dtype=np.float32)
        for i in range(self.pool_dim):
            n_conn = max(1, int(n_sensory * 0.15))
            idx = rng.choice(n_sensory, n_conn, replace=False)
            self.w_in[i, idx] = rng.normal(0, 0.1, n_conn).astype(np.float32)
        
        # W_out: motor neurons → actions (sparse 20%)
        self.w_out = np.zeros((n_motor, act_dim), dtype=np.float32)
        for a in range(act_dim):
            n_conn = max(1, int(n_motor * 0.2))
            idx = rng.choice(n_motor, n_conn, replace=False)
            self.w_out[idx, a] = rng.normal(0, 0.1, n_conn).astype(np.float32)
        
        # Select sensory/motor neurons by degree
        out_deg = np.diff(row_ptr)
        in_deg = np.zeros(self.n, dtype=np.int32)
        for i in range(self.n):
            s, e = row_ptr[i], row_ptr[i+1]
            np.add.at(in_deg, col_indices[s:e], 1)
        
        self.sensory_idx = np.where(in_deg < 3)[0][:n_sensory]
        if len(self.sensory_idx) < n_sensory:
            extra = np.setdiff1d(np.arange(self.n), self.sensory_idx)
            self.sensory_idx = np.concatenate([self.sensory_idx, extra[:n_sensory-len(self.sensory_idx)]])
        
        self.motor_idx = np.where(out_deg < 3)[0][:n_motor]
        if len(self.motor_idx) < n_motor:
            extra = np.setdiff1d(np.arange(self.n), self.motor_idx)
            self.motor_idx = np.concatenate([self.motor_idx, extra[:n_motor-len(self.motor_idx)]])
    
    def _pool_screen(self, screen: np.ndarray) -> np.ndarray:
        """Downsample 160×120 to pool_w×pool_h via block averaging."""
        h, w = screen.shape  # 120, 160
        bh, bw = self.pool_factors  # (3, 4) for 40×30
        h_pooled = h // bh
        w_pooled = w // bw
        pooled = screen[:h_pooled*bh, :w_pooled*bw].reshape(
            h_pooled, bh, w_pooled, bw
        ).mean(axis=(1, 3)).astype(np.float32)
        # Normalize [0, 255] → [0, 1]
        return pooled.flatten() / 255.0
    
    def evaluate(self, params: np.ndarray, n_seeds: int = 1) -> float:
        """Run VizDoom episode, return fitness.
        
        params = flattened [connectome_weights, w_in_flat, w_out_flat]
        Fitness = KILLCOUNT × 100 + HEALTH
        """
        try:
            from vizdoom import DoomGame, ScreenResolution, ScreenFormat, Button, GameVariable
            
            # Deconstruct params
            n_edges = len(self.col_indices)
            n_w_in = self.pool_dim * self.n_sensory
            n_w_out = self.n_motor * self.act_dim
            
            if len(params) >= n_edges + n_w_in + n_w_out:
                conn_w = params[:n_edges]
                w_in = params[n_edges:n_edges+n_w_in].reshape(self.pool_dim, self.n_sensory)
                w_out = params[n_edges+n_w_in:n_edges+n_w_in+n_w_out].reshape(self.n_motor, self.act_dim)
            elif len(params) >= n_edges:
                conn_w = params[:n_edges]
                w_in = self.w_in
                w_out = self.w_out
            else:
                return 0.001
            
            lib = _get_gpu_lif_lib()
            if lib is None:
                return 0.001
            _setup_gpu_functions(lib)
            
            # ── GPU LIF init ──
            n = self.n
            tau_m = np.full(n, 20.0, dtype=np.float32)
            v_rest = np.full(n, -70.0, dtype=np.float32)
            v_thresh = np.full(n, -52.0, dtype=np.float32)
            v_reset = np.full(n, -70.0, dtype=np.float32)
            
            total_fitness = 0.0
            
            for seed_ep in range(n_seeds):
                # ═══ GPU LIF FIRST (before VizDoom — avoids CUDA/OpenGL conflict) ═══
                ret = lib.gpu_lif_create(
                    self.row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    self.col_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    conn_w.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    self.nt_types.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int32(n), ctypes.c_int32(len(conn_w)),
                    tau_m.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    v_rest.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    v_thresh.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    v_reset.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_float(self.dt_ms),
                )
                if ret != 1:
                    continue
                
                # ── VizDoom game (second — OpenGL after CUDA) ──
                game = DoomGame()
                game.load_config(os.path.join(_DOOM_SCENARIOS, 'basic.cfg'))
                game.set_screen_resolution(ScreenResolution.RES_160X120)
                game.set_screen_format(ScreenFormat.RGB24)
                game.set_render_hud(False)
                game.set_render_crosshair(False)
                game.set_render_weapon(True)
                game.clear_available_buttons()
                game.add_available_button(Button.MOVE_LEFT)
                game.add_available_button(Button.MOVE_RIGHT)
                game.add_available_button(Button.ATTACK)
                game.set_episode_timeout(200)
                game.set_living_reward(0.0)  # no penalty — meaningful gradient from kills
                game.set_window_visible(False)
                game.set_seed(self._base_seed + seed_ep * 1000)
                game.init()
                
                total_reward = 0.0
                
                while not game.is_episode_finished():
                    # ── Get screen → pooled pixels → external_I ──
                    state = game.get_state()
                    if state is None:
                        break
                    screen = np.array(state.screen_buffer, dtype=np.float32)
                    # screen_buffer: (ch, h, w) RGB = (3, 120, 160) or gray = (1, 120, 160)
                    if screen.ndim == 3:
                        if screen.shape[0] == 3:
                            screen = screen.mean(axis=0)  # RGB → gray: (120, 160)
                        else:
                            screen = screen[0]  # already gray: (120, 160)
                    
                    pooled = self._pool_screen(screen)  # (1200,)
                    
                    # Build external_I: pooled → W_in → sensory neurons
                    external_I = np.zeros(n, dtype=np.float32)
                    for fi in range(min(len(pooled), self.pool_dim)):
                        for si, sidx in enumerate(self.sensory_idx[:len(w_in[fi])]):
                            external_I[sidx] += pooled[fi] * w_in[fi, si]
                    
                    # ── GPU LIF step ──
                    # Scale: pixels [0,1] → current [5, 35] to drive LIF above -52mV
                    ext_in = external_I * 30.0 + 5.0
                    lib.gpu_lif_step(ctypes.c_int32(1),
                        ext_in.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ctypes.c_int32(n), ctypes.c_int32(self.n_internal))
                    
                    # ── Read v → actions ──
                    v_out = np.zeros(n, dtype=np.float32)
                    lib.gpu_lif_read_v(ctypes.c_int32(1),
                        v_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ctypes.c_int32(n))
                    
                    actions = np.zeros(self.act_dim, dtype=np.float32)
                    for a in range(self.act_dim):
                        for mi, midx in enumerate(self.motor_idx[:len(w_out)]):
                            v_norm = (v_out[midx] + 70.0) / 30.0
                            actions[a] += v_norm * w_out[mi, a]
                    
                    # Continuous → discrete: sigmoid with threshold
                    # Low threshold (0.3) for exploration at start
                    doom_action = [0] * self.act_dim
                    for a in range(self.act_dim):
                        prob = 1.0 / (1.0 + np.exp(-actions[a]))
                        doom_action[a] = 1 if prob > 0.3 else 0
                    
                    reward = game.make_action(doom_action)
                    total_reward += reward
                
                game.close()
                lib.gpu_lif_destroy()
                
                # Fitness: total reward (kills × 100 - living_penalty × tics)
                episode_fitness = total_reward
                total_fitness += episode_fitness
            
            return float(max(total_fitness / max(n_seeds, 1), 0.001))
            
        except Exception as e:
            print(f"    ⚠ Doom error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return 0.001


# ═══════════════════════════════════════════════════
#  3. (μ+λ)-ES
# ═══════════════════════════════════════════════════

class GeneticEvolution:
    def __init__(self, base_weights: np.ndarray, evaluator,
                 mu=15, lambda_ratio=1.5, mut_rate=0.2,
                 mut_scale=0.4, crossover_rate=0.7, adapt_window=4):
        self.base = base_weights.copy()
        self.n_genes = len(base_weights)
        self.evaluator = evaluator
        self.mu = mu
        self.lamb = max(1, int(mu * lambda_ratio))
        self.crossover_rate = crossover_rate
        self.base_mut_rate = mut_rate
        self.base_mut_scale = mut_scale
        self.mut_rate = mut_rate
        self.mut_scale = mut_scale
        self.mut_rate_min = mut_rate * 0.3
        self.mut_rate_max = mut_rate * 3.0
        self.mut_scale_min = mut_scale * 0.3
        self.mut_scale_max = mut_scale * 3.0
        self.adapt_window = adapt_window
        self._stagnation_counter = 0
        self.rng = np.random.default_rng(42)
        self.population = []
        self.history = {'gen': [], 'best_fitness': [], 'mean_fitness': [],
                        'diversity': [], 'mut_rate': []}
        self._init_population()
    
    def _init_population(self):
        print(f"\n  Init pop (μ={self.mu}, λ={self.lamb})...", flush=True)
        for i in range(self.mu):
            noise = 0.3 + self.rng.random() * 0.5
            w = np.clip(self.base + self.rng.normal(0, noise, self.n_genes), -3.0, 3.0)
            self.population.append((w, 0.0))
        self._evaluate_list(self.population)
        self.population.sort(key=lambda x: x[1], reverse=True)
        self.population = self.population[:self.mu]
        best = max(self.population, key=lambda x: x[1])
        print(f"  Инициализация: best={best[1]:.4f}", flush=True)
    
    def _evaluate_list(self, pop):
        for i in range(len(pop)):
            w, _ = pop[i]
            pop[i] = (w, self.evaluator.evaluate(w))
            if (i + 1) % 5 == 0:
                print(f"    {i+1}/{len(pop)}", flush=True)
    
    def _mutate(self, w):
        mask = self.rng.random(self.n_genes) < self.mut_rate
        noise = self.rng.normal(0, self.mut_scale, self.n_genes)
        w = w.copy()
        w[mask] += noise[mask]
        return np.clip(w, -3.0, 3.0)
    
    def _crossover(self, w1, w2):
        mask = self.rng.random(self.n_genes) < 0.5
        return np.where(mask, w1, w2), np.where(mask, w2, w1)
    
    def _adapt_mutation(self):
        if len(self.history['best_fitness']) < self.adapt_window + 1:
            return
        recent = self.history['best_fitness'][-self.adapt_window:]
        imp = recent[-1] - recent[0]
        if imp > 1e-4:
            self.mut_rate = max(self.mut_rate_min, self.mut_rate * 0.85)
            self.mut_scale = max(self.mut_scale_min, self.mut_scale * 0.85)
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1
            if self._stagnation_counter >= self.adapt_window:
                self.mut_rate = min(self.mut_rate_max, self.mut_rate * 1.4)
                self.mut_scale = min(self.mut_scale_max, self.mut_scale * 1.4)
    
    def step(self):
        children = []
        while len(children) < self.lamb:
            rank = min(self.rng.geometric(p=0.4) - 1, self.mu - 1)
            p1 = self.population[rank][0].copy()
            if self.rng.random() < self.crossover_rate:
                rank2 = min(self.rng.geometric(p=0.4) - 1, self.mu - 1)
                c1, c2 = self._crossover(p1, self.population[rank2][0].copy())
                children.append((self._mutate(c1), 0.0))
                if len(children) < self.lamb:
                    children.append((self._mutate(c2), 0.0))
            else:
                children.append((self._mutate(p1), 0.0))
        children = children[:self.lamb]
        print(f"    Оценка λ={len(children)}...", flush=True)
        self._evaluate_list(children)
        combined = self.population + children
        combined.sort(key=lambda x: x[1], reverse=True)
        self.population = combined[:self.mu]
        self._adapt_mutation()
        
        fits = [f for _, f in self.population]
        all_w = np.array([w for w, _ in self.population[:10]])
        div = 0.0
        if len(all_w) >= 2:
            dists = []
            for i in range(min(5, len(all_w))):
                for j in range(i+1, min(5, len(all_w))):
                    dists.append(np.abs(all_w[i] - all_w[j]).mean())
            div = np.mean(dists) if dists else 0.0
        
        self.history['gen'].append(len(self.history['gen']))
        self.history['best_fitness'].append(fits[0])
        self.history['mean_fitness'].append(np.mean(fits))
        self.history['diversity'].append(div)
        self.history['mut_rate'].append(self.mut_rate)
        
        return {'best_fitness': fits[0], 'mean_fitness': np.mean(fits),
                'diversity': div, 'mut_rate': self.mut_rate,
                'mut_scale': self.mut_scale, 'lamb': len(children)}
    
    def evolve(self, n_generations=30, eval_every=5):
        print(f"\n{'='*60}")
        print(f"  (μ+λ)-ES: μ={self.mu}, λ={self.lamb}, gen={n_generations}")
        print(f"{'='*60}")
        for gen in range(n_generations):
            t0 = time.time()
            r = self.step()
            dt = time.time() - t0
            if gen % eval_every == 0:
                delta = r['best_fitness'] - self.history['best_fitness'][max(0, gen - eval_every)]
                print(f"  gen {gen:3d}: best={r['best_fitness']:.2f} "
                      f"mean={r['mean_fitness']:.2f} Δ={delta:+.2f} "
                      f"div={r['diversity']:.4f} mut={r['mut_rate']:.2f} "
                      f"λ={r['lamb']} [{dt:.1f}s]", flush=True)
        best = self.population[0]
        imp = best[1] - self.history['best_fitness'][0]
        print(f"  ✅ Best: {best[1]:.1f}, improvement: {imp:+.1f}, "
              f"gen={len(self.history['gen'])}")
        return best[0]


# ═══════════════════════════════════════════════════
#  4. MAIN
# ═══════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  DOOM BRAIN — GPU LIF Connectome Evolution")
    print("=" * 60)
    
    graph = load_graph()
    n = graph['n_neurons']
    row_ptr = graph['row_ptr']
    col_indices = graph['col_indices']
    n_edges = graph['n_edges']
    
    nt_types, base_weights = assign_ei(row_ptr, col_indices, n, 0.5)
    
    # Flattened params
    pool_dim = 15 * 20  # 300 pooled pixels
    n_sensory, n_motor = 100, 200
    act_dim = 3
    n_w_in = pool_dim * n_sensory
    n_w_out = n_motor * act_dim
    
    rng = np.random.default_rng(42)
    w_in_init = rng.normal(0, 0.1, n_w_in).astype(np.float32) * 0.3
    w_out_init = rng.normal(0, 0.1, n_w_out).astype(np.float32) * 0.3
    base_params = np.concatenate([base_weights, w_in_init, w_out_init])
    
    print(f"\nПараметры: conn={n_edges}, W_in={n_w_in}, W_out={n_w_out}, всего={len(base_params)}")
    
    print(f"\n[DoomFitness]")
    evaluator = DoomFitness(row_ptr, col_indices, nt_types,
                            pool_h=15, pool_w=20,
                            n_sensory=100, n_motor=200)
    
    print(f"\nBaseline...")
    t0 = time.time()
    baseline = evaluator.evaluate(base_params, n_seeds=1)
    print(f"  Baseline: {baseline:.4f} [{time.time()-t0:.1f}s]")
    
    print(f"\nEvolution...")
    evo = GeneticEvolution(base_params, evaluator, mu=10, lambda_ratio=1.0,
                          mut_rate=0.2, mut_scale=0.4, adapt_window=4)
    best_params = evo.evolve(n_generations=30, eval_every=5)
    
    print(f"\nDone! Fitness: {evo.history['best_fitness'][0]:.1f} → "
          f"{evo.history['best_fitness'][-1]:.1f}")
    
    if HAS_TCM:
        best_conn = best_params[:n_edges]
        conn = tcm.ConnectomeData(row_ptr.astype(np.int32),
            col_indices.astype(np.int32), best_conn.astype(np.float32),
            nt_types.astype(np.float32), [])
        tcm.save_connectome(str(BASE / 'models' / 'doom_evolved.taas'), conn,
            metadata=[("source", "doom_evolution"),
                      ("fitness", str(evo.history['best_fitness'][-1]))])
        print(f"  ✅ .taas saved")

if __name__ == '__main__':
    main()
