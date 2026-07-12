#!/usr/bin/env python3
"""
NPC Hunter Brain — HEADLESS DEMO.
Логирует ключевые моменты: встреча, потеря, поиск, патруль.
Без очистки терминала, без интерактивного ввода.
"""
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional

# ─── Все классы те же, что в npc_hunter_demo.py ─────────

class SemanticPointer:
    def __init__(self, dim: int = 64):
        self.dim = dim
        np.random.seed(42)
        self.basis = np.random.randn(dim, 3).astype(np.float32)
        self.basis /= np.linalg.norm(self.basis, axis=0, keepdims=True)
    def encode_pos(self, x: float, y: float) -> np.ndarray:
        return self.normalize(self.basis[:, 0] * x + self.basis[:, 1] * y)
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    def normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else v

class WorkingMemory:
    def __init__(self, n_slots: int = 4, dim: int = 64):
        self.n_slots = n_slots; self.dim = dim
        self.buffer = np.zeros((n_slots, dim), dtype=np.float32)
        self.ages = np.zeros(n_slots, dtype=np.int32)
        self.write_ptr = 0; self.decay = 0.9; self.gate_threshold = 0.1
        self.recurrent_gain = 0.01; self.step = 0
        np.random.seed(42)
        scale = np.sqrt(2.0 / dim)
        self.w_in = np.random.randn(dim, dim).astype(np.float32) * scale
        self.w_rec = np.random.randn(dim, dim).astype(np.float32) * scale
    def store(self, v: np.ndarray):
        assert len(v) == self.dim; self.step += 1
        norm = np.linalg.norm(v)
        if norm < self.gate_threshold: return
        projected = self.w_in @ v
        self.buffer[self.write_ptr] = projected
        self.ages[self.write_ptr] = 0; self.ages += 1; self.ages[self.write_ptr] = 0
        self.write_ptr = (self.write_ptr + 1) % self.n_slots
    def recall(self, slot: int = 0) -> np.ndarray:
        return self.buffer[slot].copy()
    def similarity_to_all(self, v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(self.buffer, axis=1, keepdims=True)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10 or np.any(norms < 1e-10): return np.zeros(self.n_slots)
        return (self.buffer @ v) / (norms.flatten() * norm_v)
    def tick(self):
        self.step += 1; self.buffer *= self.decay
        for slot in range(self.n_slots):
            self.buffer[slot] += (self.w_rec @ self.buffer[slot]) * self.recurrent_gain

class GlobalWorkspace:
    def __init__(self, n_neurons: int = 16, ws: int = 4):
        self.n = n_neurons; self.ws = ws
        np.random.seed(42)
        gs = np.sqrt(6.0 / (n_neurons + ws))
        self.gate_w = np.random.randn(ws, n_neurons).astype(np.float32) * gs
        bs = np.sqrt(6.0 / (n_neurons + ws)) * 0.01
        self.broadcast_w = np.random.randn(n_neurons, ws).astype(np.float32) * bs
    def forward(self, state: np.ndarray) -> np.ndarray:
        gates = self.gate_w @ state
        mx = np.max(gates); gates = np.exp(gates - mx)
        gates /= np.sum(gates) + 1e-10
        return state + self.broadcast_w @ gates

class BasalGanglia:
    ACTIONS = ["CHASE", "SEARCH", "PATROL", "WAIT", "AMBUSH"]
    def __init__(self, n_cortical: int = 16, n_actions: int = 5):
        self.n_cortical = n_cortical; self.n_actions = n_actions
        self.state = np.zeros(n_cortical, dtype=np.float32)
        self.prev_action: Optional[int] = None
        np.random.seed(42)
        scale = np.sqrt(2.0 / (n_cortical + n_actions))
        self.cortico_striatal = np.random.randn(n_actions, n_cortical).astype(np.float32) * scale
        self.d1_gain = 2.0; self.d2_gain = 1.5; self.stn_excitation = 0.5
        self.thalamus_gain = 1.0; self.decay = 0.95
    def step(self, cortex_input: np.ndarray) -> int:
        assert len(cortex_input) == self.n_cortical
        self.state = self.decay * self.state + cortex_input
        drive = self.cortico_striatal @ self.state
        drive[drive > 0] *= self.d1_gain; drive[drive < 0] *= self.d2_gain
        drive += self.stn_excitation * np.mean(drive)
        mx = np.max(drive)
        exps = np.exp((drive - mx) * 2.0); probs = exps / (np.sum(exps) + 1e-10)
        if np.random.random() < 0.05: action = np.random.randint(self.n_actions)
        else: action = int(np.argmax(probs))
        self.prev_action = action
        return action
    def action_name(self, idx: int) -> str:
        return self.ACTIONS[idx] if idx < len(self.ACTIONS) else f"ACT_{idx}"

# ─── Мир ─────────────────────────────────────────────────

WALL = 1; EMPTY = 0
WORLD = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,1,1,1,0,0,0,0,1,1,1,0,0,1],
    [1,0,0,1,0,1,0,0,0,0,1,0,1,0,0,1],
    [1,0,0,1,0,1,0,0,0,0,1,0,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
], dtype=np.int32)
HEIGHT, WIDTH = WORLD.shape

# ─── NPC Brain ────────────────────────────────────────────

class NpcBrain:
    def __init__(self):
        self.sp = SemanticPointer(64)
        self.wm = WorkingMemory(n_slots=4, dim=64)
        self.gnw = GlobalWorkspace()
        self.bg = BasalGanglia()
        self.npc_pos = np.array([7.0, 14.0], dtype=np.float32)
        self.npc_vel = np.array([0.0, 0.0])
        self.last_known_player = None
        self.search_target = None
        self.patrol_points = [np.array([5.,5.]), np.array([10.,5.]),
                              np.array([10.,10.]), np.array([5.,10.])]
        self.patrol_idx = 0; self.steps_since_seen = 0; self.frustration = 0.0
        self.cortex = np.zeros(16, dtype=np.float32)
        self.total_chase = 0; self.total_search = 0; self.total_patrol = 0
        self.best_chase_dist = 999.0

    def has_line_of_sight(self, player_pos: np.ndarray) -> bool:
        dx = player_pos[0] - self.npc_pos[0]; dy = player_pos[1] - self.npc_pos[1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 0.1: return True
        steps = max(int(dist * 4), 1)
        for i in range(1, steps):
            t = i / steps
            x = int(round(self.npc_pos[0] + dx * t))
            y = int(round(self.npc_pos[1] + dy * t))
            if 0 <= x < WIDTH and 0 <= y < HEIGHT and WORLD[y, x] == WALL:
                return False
        return True

    def get_vector_to(self, target: np.ndarray) -> tuple:
        d = target - self.npc_pos; dist = np.linalg.norm(d)
        return d / (dist + 0.1), dist

    def build_cortex(self, player_pos, player_visible):
        cx = np.zeros(16, dtype=np.float32)
        if player_pos is not None:
            d, dist = self.get_vector_to(player_pos)
            cx[0], cx[1], cx[2], cx[3] = d[0], d[1], dist/20.0, 1.0 if dist<5 else 0.0
        if self.last_known_player is not None:
            d, dist = self.get_vector_to(self.last_known_player)
            cx[4], cx[5], cx[6], cx[7] = d[0], d[1], 1.0, dist/20.0
        cx[8] = 1.0 if player_visible else 0.0
        cx[9] = np.tanh(self.steps_since_seen / 50.0)
        self.frustration = np.clip(self.frustration + (0.01 if not player_visible else -0.05), 0, 1)
        cx[10] = self.frustration
        d, dist = self.get_vector_to(self.patrol_points[self.patrol_idx])
        cx[11] = dist / 20.0
        if self.last_known_player is not None:
            lk_sp = self.sp.encode_pos(self.last_known_player[0], self.last_known_player[1])
            cx[12:16] = self.wm.similarity_to_all(lk_sp)
        return cx

    def step(self, player_pos: np.ndarray) -> tuple:
        player_visible = self.has_line_of_sight(player_pos)
        if player_visible:
            self.last_known_player = player_pos.copy()
            player_sp = self.sp.encode_pos(player_pos[0], player_pos[1])
            self.wm.store(player_sp)
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1
        self.cortex = self.build_cortex(player_pos, player_visible)
        self.cortex = self.gnw.forward(self.cortex)
        action = self.bg.step(self.cortex)
        self._execute_action(action, player_pos, player_visible)
        self.wm.tick()

        # Stats
        d, dist = self.get_vector_to(player_pos)
        self.best_chase_dist = min(self.best_chase_dist, dist)
        if action == 0: self.total_chase += 1
        elif action == 1: self.total_search += 1
        elif action == 2: self.total_patrol += 1

        memory_active = self.last_known_player is not None and \
            not player_visible and self.steps_since_seen < 100
        return action, self.bg.action_name(action), player_visible, memory_active, dist

    def _execute_action(self, action, player_pos, player_visible):
        speed = 0.8
        if action == 0:  # CHASE
            if self.last_known_player is not None:
                d, dist = self.get_vector_to(self.last_known_player)
                self.npc_vel = d * speed
                if dist < 0.5: self.npc_vel = np.zeros(2)
        elif action == 1:  # SEARCH
            if self.search_target is None or np.linalg.norm(self.npc_pos - self.search_target) < 1.0:
                if self.last_known_player is not None:
                    rx, ry = np.random.uniform(-3, 3, 2)
                    self.search_target = self.last_known_player + np.array([rx, ry])
                else:
                    self.search_target = np.random.uniform(2, WIDTH-2, 2)
            d, dist = self.get_vector_to(self.search_target)
            self.npc_vel = d * speed * 0.7
        elif action == 2:  # PATROL
            target = self.patrol_points[self.patrol_idx]
            d, dist = self.get_vector_to(target)
            self.npc_vel = d * speed * 0.5
            if dist < 1.0: self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_points)
        elif action == 3:  # WAIT
            self.npc_vel = np.zeros(2)
        elif action == 4:  # AMBUSH
            if self.last_known_player is not None:
                d, dist = self.get_vector_to(self.last_known_player)
                self.npc_vel = d * speed * 0.3
        new_pos = self.npc_pos + self.npc_vel
        ix, iy = int(round(new_pos[0])), int(round(new_pos[1]))
        if 1 <= ix < WIDTH-1 and 1 <= iy < HEIGHT-1 and WORLD[iy, ix] == EMPTY:
            self.npc_pos = new_pos

    def debug_info(self) -> str:
        lk = f"({self.last_known_player[0]:.1f},{self.last_known_player[1]:.1f})" if self.last_known_player is not None else "None"
        return f"NPC({self.npc_pos[0]:.1f},{self.npc_pos[1]:.1f}) vel=({self.npc_vel[0]:.2f},{self.npc_vel[1]:.2f}) last={lk} unseen={self.steps_since_seen} frust={self.frustration:.2f}"


# ═══════════════════════════════════════════════════════════
# ЗАПУСК
# ═══════════════════════════════════════════════════════════

def log(msg: str):
    print(f"  {msg}")

def render_mini(world, npc, player_pos, memory_active, player_visible):
    """Мини-карта 4×4 символа."""
    h, w = 16, 16
    px, py = int(round(player_pos[0])), int(round(player_pos[1]))
    nx, ny = int(round(npc.npc_pos[0])), int(round(npc.npc_pos[1]))
    lines = []
    for y in range(0, h, 4):
        row = ""
        for x in range(0, w, 2):
            cell = world[y, x]
            p_dist = abs(x-px) + abs(y-py)
            n_dist = abs(x-nx) + abs(y-ny)
            if p_dist <= 2 and n_dist <= 2:
                row += "⚔"
            elif p_dist <= 2:
                row += "🧑"
            elif n_dist <= 2:
                row += "🧟" if memory_active else "🔴"
            elif cell == WALL:
                row += "██"
            else:
                row += "··"
        lines.append(row)
    return lines

print("═" * 60)
print("  NPC HUNTER — HEADLESS DEMO (500 шагов)")
print("═" * 60)

npc = NpcBrain()
player_pos = np.array([3.0, 3.0], dtype=np.float32)
player_path = [(3,3),(12,3),(12,12),(3,3),(7,5),(7,10),(3,3),(12,3),(12,12),(3,12)]
path_idx = 0

key_moments = []
actions_history = []
dist_history = []

for step in range(500):
    target = np.array(player_path[path_idx], dtype=np.float32)
    d = target - player_pos; dist = np.linalg.norm(d)
    if dist < 0.5:
        path_idx = (path_idx + 1) % len(player_path)
    else:
        player_pos += (d / (dist + 0.1)) * 0.6
    
    action, action_name, player_visible, memory_active, dist_to_player = npc.step(player_pos)
    actions_history.append(action_name)
    dist_history.append(dist_to_player)
    
    # Key moments
    if step == 1 and player_visible:
        key_moments.append((step, f"🟢 НАЧАЛО: NPC видит игрока на [{player_pos[0]:.1f},{player_pos[1]:.1f}]"))
    if player_visible and not key_moments:
        key_moments.append((step, f"🟢 ВСТРЕЧА #{len(key_moments)}: NPC заметил игрока на [{player_pos[0]:.1f},{player_pos[1]:.1f}]"))
    if not player_visible and step > 0 and (len(key_moments) == 0 or key_moments[-1][0] < step - 30):
        last_entry = key_moments[-1] if key_moments else None
        if last_entry is None or "ПОТЕРЯ" not in last_entry[1]:
            key_moments.append((step, f"🔴 ПОТЕРЯ #{len(key_moments)}: NPC потерял игрока! Шаг {step}, last_known={npc.last_known_player}"))
    if not player_visible and npc.steps_since_seen == 50 and (len(key_moments) < 3 or key_moments[-1][0] < step - 20):
        key_moments.append((step, f"🧟 ПОИСК #{len(key_moments)}: 50 шагов без игрока! NPC всё ещё помнит (WM активна)"))
    if not player_visible and npc.steps_since_seen == 100 and (len(key_moments) < 4 or key_moments[-1][0] < step - 20):
        key_moments.append((step, f"😞 ЗАБЫЛ #{len(key_moments)}: 100 шагов без игрока. NPC возвращается к патрулю"))
    if player_visible and step > 100 and (len(key_moments) < 5 or key_moments[-1][0] < step - 20):
        key_moments.append((step, f"🔄 ПЕРЕХВАТ #{len(key_moments)}: NPC снова видит игрока! Активна память"))

    if step % 50 == 0:
        mk = memory_active
        pv = player_visible
        action = npc.bg.action_name(npc.bg.prev_action or 0)
        print(f"  [step {step:3d}] {action:8s} | npc({npc.npc_pos[0]:5.1f},{npc.npc_pos[1]:5.1f}) "
              f"player({player_pos[0]:5.1f},{player_pos[1]:5.1f}) "
              f"seen={pv} mem={mk} dist={dist_to_player:.1f} unseen={npc.steps_since_seen:3d}")

# ─── Итоги ────────────────────────────────────────────────
print()
print("═" * 60)
print("  РЕЗУЛЬТАТЫ ДЕМО (500 шагов)")
print("═" * 60)
print()

# Ключевые моменты
print("  📋 КЛЮЧЕВЫЕ МОМЕНТЫ:")
print()
for step, msg in key_moments:
    print(f"    [{step:3d}] {msg}")
print()

# Статистика действий
print("  📊 СТАТИСТИКА ДЕЙСТВИЙ:")
print()
total = npc.total_chase + npc.total_search + npc.total_patrol + 1
print(f"    CHASE  : {npc.total_chase:3d} ({100*npc.total_chase/total:.0f}%) — преследование")
print(f"    SEARCH : {npc.total_search:3d} ({100*npc.total_search/total:.0f}%) — поиск с памятью")
print(f"    PATROL : {npc.total_patrol:3d} ({100*npc.total_patrol/total:.0f}%) — патруль")
print(f"    WAIT   : {500-total:3d} ({(100*(500-total)/500):.0f}%) — ожидание")
print()

# Анализ памяти
visible_steps = sum(1 for a in actions_history if a == "CHASE" or actions_history.count(a) == 0)
seen_count = sum(1 for i, a in enumerate(actions_history) 
                  if a == "CHASE" and i > 0 and actions_history[i-1] != "CHASE")
memory_search_count = sum(1 for i, a in enumerate(actions_history)
                           if a == "SEARCH" and i > 0 and actions_history[i-1] != "SEARCH")
# Count chase-seen transitions
chase_after_search = sum(1 for i in range(1, len(actions_history))
                          if actions_history[i] == "CHASE" and actions_history[i-1] == "SEARCH")

print("  🧠 АНАЛИЗ ПАМЯТИ:")
print()
print(f"    Минимальная дистанция NPC→Player: {npc.best_chase_dist:.1f} клеток")
print(f"    CHASE после SEARCH (память помогла): {chase_after_search} раз")
print(f"    Максимальное время без игрока: {npc.steps_since_seen} шагов")
print(f"    Финальная фрустрация: {npc.frustration:.2f}")
print()

# Финальная позиция
print("  🗺️  ТЕКУЩАЯ СИТУАЦИЯ:")
print()
print(f"    NPC:   ({npc.npc_pos[0]:.1f}, {npc.npc_pos[1]:.1f})")
print(f"    Игрок: ({player_pos[0]:.1f}, {player_pos[1]:.1f})")
lk = f"({npc.last_known_player[0]:.1f},{npc.last_known_player[1]:.1f})" if npc.last_known_player is not None else "None"
print(f"    Last known: {lk}")
print(f"    Дистанция: {dist_history[-1]:.1f} клеток")
print(f"    Видимость: {npc.has_line_of_sight(player_pos)}")
print()
print("═" * 60)
