#!/usr/bin/env python3
"""
NPC HUNTER 2v1 + DOPAMINE-STDP LEARNING (FIXED).

3-factor plasticity (Izhikevich 2007):
  Δw = η · D · e
  e — eligibility trace (индивидуальная для каждого синапса)
  D — dopamine signal (GLOBAL нейромодулятор — одинаковый для всех)
  η — learning rate

Исправления:
  ✓ Каждый NPC — свой STDP (не shared!)
  ✓ flatten weights → один STDP.step() на всех синапсов
  ✓ Dopamine global — без per-action scaling
  ✓ Δw правильная: η · D · e

Запуск: python3 scripts/npc_hunter_learn.py
"""

import numpy as np
from typing import Optional, Tuple, List

np.random.seed(42)


# ═══════════════════════════════════════════════════════════
# 1. DopamineSTDP — 3-factor plasticity (Izhikevich 2007)
# ═══════════════════════════════════════════════════════════

class DopamineSTDP:
    """
    Аналог Rust taas_core::neuron::DopamineSTDP.
    
    Каждый синапс имеет свои:
      - trace_pre — пресинаптический след (растёт при pre-spike)
      - trace_post — постсинаптический след (растёт при post-spike)
      - eligibility — e[t] = e[t-1]·decay_e + LTP - LTD
    
    Step: обновить traces + eligibility для ВСЕХ синапсов сразу.
    apply: Δw = η · D · e для ВСЕХ синапсов сразу.
    """
    def __init__(self, n_synapses: int, dt: float = 1.0):
        self.n = n_synapses
        self.dt = dt
        self.tau_pre = 20.0
        self.tau_post = 20.0
        self.tau_e = 500.0
        self.a_ltp = 0.1
        self.a_ltd = 0.12
        self.w_min = 0.0
        self.w_max = 3.0
        
        self.trace_pre = np.zeros(n_synapses, dtype=np.float32)
        self.trace_post = np.zeros(n_synapses, dtype=np.float32)
        self.eligibility = np.zeros(n_synapses, dtype=np.float32)
    
    def step(self, pre_spikes: np.ndarray, post_spikes: np.ndarray):
        """
        Один шаг STDP для ВСЕХ синапсов.
        
        pre_spikes[i]  = 1 если пресинаптический нейрон i заспайкал
        post_spikes[i] = 1 если постсинаптический нейрон i заспайкал
        
        Для cortico-striatal:
          pre  = cortex_state (32 нейрона, tiled до 192)
          post = action (6 нейронов, repeated до 192)
        """
        assert len(pre_spikes) == self.n
        assert len(post_spikes) == self.n
        
        # Decay
        self.trace_pre *= 1.0 - self.dt / self.tau_pre
        self.trace_post *= 1.0 - self.dt / self.tau_post
        self.eligibility *= 1.0 - self.dt / self.tau_e
        
        # Spike events
        self.trace_pre[pre_spikes > 0] += 1.0
        self.trace_post[post_spikes > 0] += 1.0
        
        # Eligibility: LTP (post fires) + LTD (pre fires)
        ltp = np.where(post_spikes > 0, self.a_ltp * self.trace_pre, 0.0)
        ltd = np.where(pre_spikes > 0, self.a_ltd * self.trace_post, 0.0)
        self.eligibility += ltp - ltd
    
    def apply_dopamine(self, weights: np.ndarray, dopamine: float, lr: float = 0.01):
        """
        Δw = η · D · e  для ВСЕХ синапсов сразу.
        Dopamine — GLOBAL сигнал, одинаковый для всех синапсов.
        """
        delta = lr * dopamine * self.eligibility
        np.clip(weights + delta, self.w_min, self.w_max, out=weights)
    
    def reset(self):
        self.trace_pre.fill(0.0)
        self.trace_post.fill(0.0)
        self.eligibility.fill(0.0)


# ═══════════════════════════════════════════════════════════
# 2. Shared Components
# ═══════════════════════════════════════════════════════════

class SemanticPointer:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.basis = np.random.randn(dim, 3).astype(np.float32)
        self.basis /= np.linalg.norm(self.basis, axis=0, keepdims=True)
    def encode_pos(self, x: float, y: float) -> np.ndarray:
        r = self.basis[:, 0] * x + self.basis[:, 1] * y
        n = np.linalg.norm(r)
        return r / n if n > 1e-10 else r

class SharedWorkingMemory:
    def __init__(self, n_slots: int = 8, dim: int = 64):
        self.n_slots = n_slots; self.dim = dim
        self.buffer = np.zeros((n_slots, dim), dtype=np.float32)
        self.ages = np.zeros(n_slots, dtype=np.int32)
        self.write_ptr = 0; self.decay = 0.92; self.gate_threshold = 0.1
        self.recurrent_gain = 0.01; self.step = 0
        scale = np.sqrt(2.0 / dim)
        self.w_in = np.random.randn(dim, dim).astype(np.float32) * scale
        self.w_rec = np.random.randn(dim, dim).astype(np.float32) * scale
    def store(self, v: np.ndarray, owner_id: int = 1):
        if np.linalg.norm(v) < self.gate_threshold: return
        self.buffer[self.write_ptr] = self.w_in @ v
        self.ages[self.write_ptr] = 0; self.ages += 1; self.ages[self.write_ptr] = 0
        self.write_ptr = (self.write_ptr + 1) % self.n_slots
    def similarity_to_all(self, v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(self.buffer, axis=1, keepdims=True)
        nv = np.linalg.norm(v)
        if nv < 1e-10 or np.any(norms < 1e-10): return np.zeros(self.n_slots)
        return (self.buffer @ v) / (norms.flatten() * nv)
    def tick(self):
        self.buffer *= self.decay
        for slot in range(self.n_slots):
            self.buffer[slot] += (self.w_rec @ self.buffer[slot]) * self.recurrent_gain

class GlobalWorkspace:
    def __init__(self, n_neurons: int = 32, ws: int = 8):
        self.n = n_neurons; self.ws = ws
        gs = np.sqrt(6.0 / (n_neurons + ws))
        self.gate_w = np.random.randn(ws, n_neurons).astype(np.float32) * gs
        bs = np.sqrt(6.0 / (n_neurons + ws)) * 0.01
        self.broadcast_w = np.random.randn(n_neurons, ws).astype(np.float32) * bs
    def forward(self, state: np.ndarray) -> np.ndarray:
        gates = self.gate_w @ state
        mx = np.max(gates); gates = np.exp(gates - mx)
        gates /= np.sum(gates) + 1e-10
        return state + self.broadcast_w @ gates


# ═══════════════════════════════════════════════════════════
# 3. Plastic BasalGanglia — СВОЙ STDP у каждого!
# ═══════════════════════════════════════════════════════════

class PlasticBasalGanglia:
    """
    BasalGanglia с пластичными cortico-striatal весами.
    Каждый экземпляр имеет СВОЙ DopamineSTDP.
    Веса flatten → один STDP step на все синапсы.
    """
    
    def __init__(self, n_cortical: int, n_actions: int):
        self.n_cortical = n_cortical
        self.n_actions = n_actions
        
        # Каждому BG — свой STDP!
        self.stdp = DopamineSTDP(n_synapses=n_cortical * n_actions, dt=1.0)
        
        scale = np.sqrt(2.0 / (n_cortical + n_actions))
        self.weights = np.random.randn(n_actions, n_cortical).astype(np.float32) * scale
        
        self.state = np.zeros(n_cortical, dtype=np.float32)
        self.prev_action: Optional[int] = None
        
        self.d1_gain = 2.0; self.d2_gain = 1.5
        self.stn_excitation = 0.5; self.thalamus_gain = 1.0
        self.decay = 0.95
        
        self.last_cortex = np.zeros(n_cortical, dtype=np.float32)
    
    def step(self, cortex_input: np.ndarray) -> int:
        assert len(cortex_input) == self.n_cortical
        self.last_cortex = cortex_input.copy()
        
        self.state = self.decay * self.state + cortex_input
        drive = self.weights @ self.state
        
        drive[drive > 0] *= self.d1_gain
        drive[drive < 0] *= self.d2_gain
        drive += self.stn_excitation * np.mean(drive)
        
        mx = np.max(drive)
        exps = np.exp((drive - mx) * 2.0)
        probs = exps / (np.sum(exps) + 1e-10)
        
        if np.random.random() < 0.03:
            action = np.random.randint(self.n_actions)
        else:
            action = int(np.argmax(probs))
        
        self.prev_action = action
        return action
    
    def apply_stdp(self, dopamine: float, lr: float = 0.01):
        """
        Применить STDP к ВСЕМ весам сразу (один проход).
        
        pre_spikes:  cortex → tiled до n_cortical * n_actions
        post_spikes: action → repeated до n_cortical * n_actions
        
        synapse (a, c) = weights[a, c] — связь cortex[c] → action[a]
        trace_pre[a*n_c + c]  растёт когда cortex[c] спайкает
        trace_post[a*n_c + c] растёт когда action[A] спайкает (для всех C!)
        eligibility[a*n_c + c] комбинирует pre/post
        """
        corteX = np.abs(self.last_cortex)
        pre_spikes = (corteX > 0.05).astype(np.int32)
        
        # Post-spike: только выбранное действие
        post = np.zeros(self.n_actions, dtype=np.int32)
        if self.prev_action is not None:
            post[self.prev_action] = 1
        
        # tile/repeat до длины n_cortical * n_actions
        pre = np.tile(pre_spikes, self.n_actions)    # [A × C] — порядок: A0(C0-C31), A1(C0-C31), ...
        post_repeated = np.repeat(post, self.n_cortical)  # [A × C] — action A: все C получают post
        
        weights_flat = self.weights.ravel()
        
        # ОДИН STDP step — все синапсы сразу
        self.stdp.step(pre, post_repeated)
        self.stdp.apply_dopamine(weights_flat, dopamine, lr)
        
        self.weights[:] = weights_flat.reshape(self.n_actions, self.n_cortical)
    
    def reset(self):
        self.state.fill(0.0)
        self.prev_action = None


# ═══════════════════════════════════════════════════════════
# 4. Мир
# ═══════════════════════════════════════════════════════════

WALL, EMPTY = 1, 0
WORLD = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1],
    [1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1],
    [1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,1,1,0,0,1,1,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=np.int32)
H, W = WORLD.shape


# ═══════════════════════════════════════════════════════════
# 5. NPC Brain
# ═══════════════════════════════════════════════════════════

class NpcBrain:
    ACTIONS = ["CHASE", "FLANK", "SEARCH", "PATROL", "AMBUSH", "CUTOFF"]
    
    def __init__(self, npc_id: int, start_pos: np.ndarray,
                 shared_wm: SharedWorkingMemory, 
                 sp: SemanticPointer, gnw: GlobalWorkspace,
                 bg: PlasticBasalGanglia, role: str = "hunter"):
        self.id = npc_id
        self.pos = start_pos.copy()
        self.vel = np.zeros(2)
        self.role = role
        self.sp = sp; self.wm = shared_wm; self.gnw = gnw; self.bg = bg
        
        self.last_known = None
        self.search_target = None
        self.steps_since_seen = 0
        self.frustration = 0.0
        self.cortex = np.zeros(gnw.n, dtype=np.float32)
        self.dopamine_history: List[float] = []
        self.dist_history: List[float] = []
        
        if self.id == 1:
            self.patrol_pts = [np.array([5.,15.]), np.array([10.,12.]),
                               np.array([5.,8.]), np.array([3.,5.])]
        else:
            self.patrol_pts = [np.array([18.,15.]), np.array([14.,12.]),
                               np.array([18.,8.]), np.array([20.,5.])]
        self.patrol_idx = 0

    def line_of_sight(self, player_pos: np.ndarray) -> bool:
        dx = player_pos[0] - self.pos[0]; dy = player_pos[1] - self.pos[1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 0.1: return True
        steps = max(int(dist * 4), 1)
        for i in range(1, steps):
            t = i / steps
            x = int(round(self.pos[0] + dx*t))
            y = int(round(self.pos[1] + dy*t))
            if 0 <= x < W and 0 <= y < H and WORLD[y, x] == WALL:
                return False
        return True

    def vec_to(self, target: np.ndarray) -> Tuple[np.ndarray, float]:
        d = target - self.pos; dist = np.linalg.norm(d)
        return d / (dist + 0.1), dist

    def sense(self, player_pos: np.ndarray):
        visible = self.line_of_sight(player_pos)
        if visible:
            self.last_known = player_pos.copy()
            p_sp = self.sp.encode_pos(player_pos[0], player_pos[1])
            self.wm.store(p_sp, owner_id=self.id)
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1
        return visible

    def build_cortex(self, player_pos: np.ndarray, visible: bool,
                     partner_pos=None, partner_last_known=None) -> np.ndarray:
        cx = np.zeros(self.gnw.n, dtype=np.float32)
        d, dist = self.vec_to(player_pos)
        cx[0] = d[0]; cx[1] = d[1]; cx[2] = dist/25.0
        cx[3] = 1.0 if dist < 3.0 else 0.0
        if self.last_known is not None:
            d, dist = self.vec_to(self.last_known)
            cx[4] = d[0]; cx[5] = d[1]; cx[6] = 1.0; cx[7] = dist/25.0
        if partner_pos is not None:
            d, dist = self.vec_to(partner_pos)
            cx[8] = d[0]; cx[9] = d[1]; cx[10] = dist/25.0
        if partner_last_known is not None: cx[11] = 1.0
        cx[12] = 1.0 if visible else 0.0
        cx[13] = np.tanh(self.steps_since_seen / 50.0)
        self.frustration = np.clip(
            self.frustration + (0.008 if not visible else -0.05), 0, 1)
        cx[14] = self.frustration
        cx[15] = 1.0 if partner_last_known is not None else 0.0
        pp = self.patrol_pts[self.patrol_idx]
        d, dist = self.vec_to(pp)
        cx[16] = d[0]; cx[17] = d[1]; cx[18] = dist/25.0
        p_sp = self.sp.encode_pos(player_pos[0], player_pos[1])
        cx[20:24] = self.wm.similarity_to_all(p_sp)[:4]
        cx[28] = 0.0 if self.role == "hunter" else 1.0
        return cx

    def compute_dopamine(self, player_pos: np.ndarray, visible: bool) -> float:
        _, dist = self.vec_to(player_pos)
        self.dist_history.append(dist)
        if dist < 2.0 and visible: return 1.0
        if dist < 3.0 and visible: return 0.5
        if visible: return 0.2
        if self.steps_since_seen > 30: return -0.5
        if self.steps_since_seen > 15: return -0.2
        return 0.0

    def act(self, player_pos: np.ndarray, visible: bool,
            partner_pos=None, partner_last_known=None) -> Tuple[int, str]:
        self.cortex = self.build_cortex(player_pos, visible,
                                        partner_pos, partner_last_known)
        self.cortex = self.gnw.forward(self.cortex)
        action = self.bg.step(self.cortex)
        self._execute(action)
        self.wm.tick()
        return action, self.ACTIONS[action]

    def _execute(self, action: int):
        speed = 0.8
        if action == 0:
            if self.last_known is not None:
                d, dist = self.vec_to(self.last_known)
                self.vel = d * speed
                if dist < 0.5: self.vel = np.zeros(2)
        elif action == 1:
            if self.last_known is not None:
                d, dist = self.vec_to(self.last_known)
                angle = np.pi / 6 if self.id == 1 else -np.pi / 6
                rot = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
                self.vel = rot @ d * speed * 0.8
        elif action == 2:
            if self.search_target is None or np.linalg.norm(self.pos - self.search_target) < 1.0:
                if self.last_known is not None:
                    rx, ry = np.random.uniform(-4, 4, 2)
                    self.search_target = self.last_known + np.array([rx, ry])
                else:
                    self.search_target = np.random.uniform(3, W-3, 2)
            d, dist = self.vec_to(self.search_target)
            self.vel = d * speed * 0.6
        elif action == 3:
            target = self.patrol_pts[self.patrol_idx]
            d, dist = self.vec_to(target)
            self.vel = d * speed * 0.4
            if dist < 1.5: self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_pts)
        elif action == 4:
            if self.last_known is not None:
                d, dist = self.vec_to(self.last_known)
                self.vel = d * speed * 0.25
        elif action == 5:
            if self.last_known is not None:
                d, dist = self.vec_to(self.last_known)
                self.vel = d * speed * 0.9
        new_pos = self.pos + self.vel
        ix, iy = int(round(new_pos[0])), int(round(new_pos[1]))
        if 1 <= ix < W-1 and 1 <= iy < H-1 and WORLD[iy, ix] == EMPTY:
            self.pos = new_pos


# ═══════════════════════════════════════════════════════════
# 6. Запуск
# ═══════════════════════════════════════════════════════════

print("═" * 65)
print("  NPC HUNTER 2v1 + DOPAMINE-STDP LEARNING (FIXED)")
print("  3-factor plasticity (Izhikevich 2007)")
print("  Δw = η · D · e — каждый NPC свой STDP")
print("═" * 65)

sp = SemanticPointer(64)
shared_wm = SharedWorkingMemory(n_slots=8, dim=64)
gnw = GlobalWorkspace()

# Каждому NPC — СВОЙ STDP (не shared!)
bg1 = PlasticBasalGanglia(32, 6)
bg2 = PlasticBasalGanglia(32, 6)

npc1 = NpcBrain(1, np.array([3.0, 15.0]), shared_wm, sp, gnw, bg1, "hunter")
npc2 = NpcBrain(2, np.array([19.0, 15.0]), shared_wm, sp, gnw, bg2, "flanker")

player_pos = np.array([5.0, 3.0], dtype=np.float32)
player_path = [(5,3),(18,3),(18,15),(5,15),(5,8),(18,8),(10,10),(7,5),(18,12),(5,3)]
path_idx = 0

# Snapshot initial weights
initial_w1 = npc1.bg.weights.copy()
initial_w2 = npc2.bg.weights.copy()

print("\n  Фаза 1: Исследование (без обучения) — 100 шагов")
print("  Фаза 2: STDP обучение — 900 шагов")
print()

for step in range(1000):
    # Player move
    target = np.array(player_path[path_idx], dtype=np.float32)
    d = target - player_pos; dist = np.linalg.norm(d)
    if dist < 0.5:
        path_idx = (path_idx + 1) % len(player_path)
    else:
        player_pos += (d / (dist + 0.1)) * 0.6
    
    # Sense
    v1 = npc1.sense(player_pos)
    v2 = npc2.sense(player_pos)
    
    # Act
    a1, an1 = npc1.act(player_pos, v1, npc2.pos, npc2.last_known)
    a2, an2 = npc2.act(player_pos, v2, npc1.pos, npc1.last_known)
    
    # Dopamine + STDP (начиная со 100 шага)
    if step >= 100:
        d1 = npc1.compute_dopamine(player_pos, v1)
        d2 = npc2.compute_dopamine(player_pos, v2)
        npc1.dopamine_history.append(d1)
        npc2.dopamine_history.append(d2)
        
        dopamine = (d1 + d2) * 0.5  # средний dopamine
        lr = 0.02 * max(0.1, 1.0 - step / 1500)
        
        # Каждый NPC применяет СВОЙ STDP
        npc1.bg.apply_stdp(dopamine, lr)
        npc2.bg.apply_stdp(dopamine, lr)
    
    # Log
    if step % 100 == 99 or step == 0:
        d1 = np.linalg.norm(npc1.pos - player_pos)
        d2 = np.linalg.norm(npc2.pos - player_pos)
        md1 = np.mean(npc1.dopamine_history[-100:]) if len(npc1.dopamine_history) >= 100 else 0
        md2 = np.mean(npc2.dopamine_history[-100:]) if len(npc2.dopamine_history) >= 100 else 0
        wc1 = np.mean(np.abs(npc1.bg.weights - initial_w1))
        wc2 = np.mean(np.abs(npc2.bg.weights - initial_w2))
        v1s = "👁" if v1 else "❌"
        v2s = "👁" if v2 else "❌"
        print(f"  [{step:3d}] D1={md1:.2f} D2={md2:.2f} "
              f"NPC1:{an1:8s} d={d1:.1f} {v1s} "
              f"NPC2:{an2:8s} d={d2:.1f} {v2s} "
              f"Δw1={wc1:.4f} Δw2={wc2:.4f}")

# ─── Итоги ────────────────────────────────────────────────
print()
print("═" * 65)
print("  РЕЗУЛЬТАТЫ: DOPAMINE-STDP LEARNING (FIXED)")
print("═" * 65)
print()

wc1 = np.mean(np.abs(npc1.bg.weights - initial_w1))
wc2 = np.mean(np.abs(npc2.bg.weights - initial_w2))
print(f"  📊 ИЗМЕНЕНИЕ ВЕСОВ:")
print(f"    NPC1 (hunter):  Δw = {wc1:.4f}")
print(f"    NPC2 (flanker): Δw = {wc2:.4f}")
print()

if len(npc1.dopamine_history) > 200:
    early_dopa = np.mean(npc1.dopamine_history[100:300])
    late_dopa = np.mean(npc1.dopamine_history[-200:])
    print(f"  🧠 DOPAMINE NPC1: early={early_dopa:.3f} → late={late_dopa:.3f}")
    if late_dopa > early_dopa:
        print(f"    📈 DOPAMINE РАСТЁТ — NPC УЧИТСЯ!")
    else:
        print(f"    📉 DOPAMINE ПАДАЕТ — ...")

    ed = np.mean(npc1.dist_history[100:300])
    ld = np.mean(npc1.dist_history[-200:])
    print(f"  📏 ДИСТАНЦИЯ NPC1: early={ed:.1f} → late={ld:.1f}")
    if ld < ed:
        print(f"    📉 NPC ПРИБЛИЖАЕТСЯ К ИГРОКУ!")
else:
    print(f"  🧠 DOPAMINE NPC1: {np.mean(npc1.dopamine_history):.3f} (mean)")

print()
print(f"  🔬 ТОП-5 СИЛЬНЕЙШИХ СВЯЗЕЙ NPC1:")
flat_w1 = npc1.bg.weights.flatten()
top5 = np.argsort(-np.abs(flat_w1))[:5]
for i, idx in enumerate(top5):
    a = idx // 32; c = idx % 32
    print(f"    [{i+1}] {NpcBrain.ACTIONS[a]:8s} ← cortex[{c:2d}]  w = {flat_w1[idx]:.3f}")
print()

print(f"  🗺️  ФИНАЛЬНЫЕ ПОЗИЦИИ:")
print(f"    Player: ({player_pos[0]:.1f}, {player_pos[1]:.1f})")
print(f"    NPC1:   ({npc1.pos[0]:.1f}, {npc1.pos[1]:.1f})")
print(f"    NPC2:   ({npc2.pos[0]:.1f}, {npc2.pos[1]:.1f})")
print()
print("═" * 65)
