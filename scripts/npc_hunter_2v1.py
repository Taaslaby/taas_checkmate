#!/usr/bin/env python3
"""
NPC HUNTER × 2 — Red Team / Blue Team с общей памятью.

Архитектура OpenAI-style: два охотника с общей WorkingMemory.
Когда один видит игрока — оба знают его позицию.
GNW broadcast + BasalGanglia координируют действия:
  - NPC1: загоняет игрока слева
  - NPC2: перехватывает справа

Запуск: python3 scripts/npc_hunter_2v1.py
"""

import numpy as np
import time, os
from typing import Optional, List, Tuple

# ═══════════════════════════════════════════════════════════
#  Shared Components (один экземпляр на двоих)
# ═══════════════════════════════════════════════════════════

class SemanticPointer:
    def __init__(self, dim: int = 64):
        self.dim = dim
        np.random.seed(42)
        self.basis = np.random.randn(dim, 3).astype(np.float32)
        self.basis /= np.linalg.norm(self.basis, axis=0, keepdims=True)
    def encode_pos(self, x: float, y: float) -> np.ndarray:
        return self.normalize(self.basis[:, 0] * x + self.basis[:, 1] * y)
    def normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v); return v / n if n > 1e-10 else v
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

class SharedWorkingMemory:
    """
    ОБЩАЯ рабочая память для двух NPC.
    Когда NPC1.store(player_pos) → NPC2 может recall().
    Аналог OpenAI shared replay buffer.
    """
    def __init__(self, n_slots: int = 8, dim: int = 64):
        self.n_slots = n_slots
        self.dim = dim
        self.buffer = np.zeros((n_slots, dim), dtype=np.float32)
        self.ages = np.zeros(n_slots, dtype=np.int32)
        self.owner = np.zeros(n_slots, dtype=np.int32)  # 0=unused, 1=NPC1, 2=NPC2
        self.write_ptr = 0
        self.decay = 0.92
        self.gate_threshold = 0.1
        self.recurrent_gain = 0.01
        self.step = 0
        
        np.random.seed(42)
        scale = np.sqrt(2.0 / dim)
        self.w_in = np.random.randn(dim, dim).astype(np.float32) * scale
        self.w_rec = np.random.randn(dim, dim).astype(np.float32) * scale

    def store(self, v: np.ndarray, owner_id: int = 1):
        assert len(v) == self.dim
        self.step += 1
        norm = np.linalg.norm(v)
        if norm < self.gate_threshold: 
            return
        projected = self.w_in @ v
        slot = self.write_ptr
        self.buffer[slot] = projected
        self.ages[slot] = 0
        self.owner[slot] = owner_id
        self.ages += 1
        self.ages[slot] = 0
        self.write_ptr = (self.write_ptr + 1) % self.n_slots

    def recall(self, slot: int = 0) -> np.ndarray:
        return self.buffer[slot].copy()

    def recall_best(self, query: np.ndarray) -> Tuple[np.ndarray, float]:
        """Вспомнить слот, наиболее похожий на query. Возвращает (вектор, схожесть)."""
        sims = self.similarity_to_all(query)
        best = int(np.argmax(sims))
        return self.buffer[best].copy(), float(sims[best])

    def similarity_to_all(self, v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(self.buffer, axis=1, keepdims=True)
        nv = np.linalg.norm(v)
        if nv < 1e-10 or np.any(norms < 1e-10): return np.zeros(self.n_slots)
        return (self.buffer @ v) / (norms.flatten() * nv)

    def tick(self):
        self.step += 1
        self.buffer *= self.decay
        for slot in range(self.n_slots):
            self.buffer[slot] += (self.w_rec @ self.buffer[slot]) * self.recurrent_gain

    def debug_info(self) -> str:
        active = [i for i in range(self.n_slots) if np.linalg.norm(self.buffer[i]) > 0.01]
        return f"WM: {len(active)}/{self.n_slots} active, step={self.step}"


class GlobalWorkspace:
    def __init__(self, n_neurons: int = 32, ws: int = 8):
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
    ACTIONS = ["CHASE", "FLANK", "SEARCH", "PATROL", "AMBUSH", "CUTOFF"]
    def __init__(self, n_cortical: int = 32, n_actions: int = 6):
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
        if np.random.random() < 0.03: action = np.random.randint(self.n_actions)
        else: action = int(np.argmax(probs))
        self.prev_action = action
        return action
    def action_name(self, idx: int) -> str:
        return self.ACTIONS[idx] if idx < len(self.ACTIONS) else f"ACT_{idx}"
    def reset(self):
        self.state.fill(0.0); self.prev_action = None


# ═══════════════════════════════════════════════════════════
#  Мир
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
#  NPC Brain (индивидуальный, но SHARED WM)
# ═══════════════════════════════════════════════════════════

class NpcBrain:
    def __init__(self, npc_id: int, start_pos: np.ndarray,
                 shared_wm: SharedWorkingMemory, 
                 sp: SemanticPointer, gnw: GlobalWorkspace,
                 bg: BasalGanglia, role: str = "hunter"):
        self.id = npc_id
        self.pos = start_pos.copy()
        self.vel = np.zeros(2)
        self.role = role  # "hunter" или "flanker"
        self.sp = sp
        self.wm = shared_wm
        self.gnw = gnw
        self.bg = bg
        
        self.last_known = None
        self.search_target = None
        self.steps_since_seen = 0
        self.frustration = 0.0
        self.cortex = np.zeros(gnw.n, dtype=np.float32)
        self.stats = {"chase": 0, "flank": 0, "search": 0, "patrol": 0, 
                      "ambush": 0, "cutoff": 0}

        # Патруль: у NPC1 свой маршрут, у NPC2 свой
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
        """Сенсорика + обновление SHARED WorkingMemory."""
        visible = self.line_of_sight(player_pos)
        if visible:
            self.last_known = player_pos.copy()
            player_sp = self.sp.encode_pos(player_pos[0], player_pos[1])
            # Сохраняем в SHARED память
            self.wm.store(player_sp, owner_id=self.id)
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1
        return visible

    def build_cortex(self, player_pos: np.ndarray, visible: bool,
                     partner_pos: Optional[np.ndarray] = None,
                     partner_last_known: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Cortex (32 нейрона):
          [0-3]:   направление к игроку (вектор + дист + ближний)
          [4-7]:   направление к last_known
          [8-11]:  направление к партнёру (вектор + дист)
          [12]:    player_visible
          [13]:    steps_since_seen (squashed)
          [14]:    frustration
          [15]:    partner_visible (знает ли партнёр где игрок?)
          [16-19]: вектор к patrol point
          [20-23]: WM similarity to player
          [24-27]: WM similarity to partner
          [28]:    role (0=hunter, 1=flanker)
          [29-31]: free
        """
        cx = np.zeros(self.gnw.n, dtype=np.float32)

        if player_pos is not None:
            d, dist = self.vec_to(player_pos)
            cx[0] = d[0]; cx[1] = d[1]; cx[2] = dist/25.0
            cx[3] = 1.0 if dist < 3.0 else 0.0

        if self.last_known is not None:
            d, dist = self.vec_to(self.last_known)
            cx[4] = d[0]; cx[5] = d[1]; cx[6] = 1.0; cx[7] = dist/25.0

        if partner_pos is not None:
            d, dist = self.vec_to(partner_pos)
            cx[8] = d[0]; cx[9] = d[1]; cx[10] = dist/25.0

        if partner_last_known is not None:
            cx[11] = 1.0  # партнёр тоже что-то знает

        cx[12] = 1.0 if visible else 0.0
        cx[13] = np.tanh(self.steps_since_seen / 50.0)
        self.frustration = np.clip(
            self.frustration + (0.008 if not visible else -0.05), 0, 1)
        cx[14] = self.frustration
        cx[15] = 1.0 if partner_last_known is not None else 0.0

        pp = self.patrol_pts[self.patrol_idx]
        d, dist = self.vec_to(pp)
        cx[16] = d[0]; cx[17] = d[1]; cx[18] = dist/25.0

        # WM similarity
        p_sp = self.sp.encode_pos(player_pos[0], player_pos[1])
        cx[20:24] = self.wm.similarity_to_all(p_sp)[:4]

        cx[28] = 0.0 if self.role == "hunter" else 1.0

        return cx

    def decide_and_move(self, player_pos: np.ndarray, visible: bool,
                        partner_pos: Optional[np.ndarray] = None,
                        partner_last_known: Optional[np.ndarray] = None):
        """Выбрать действие и двигаться."""
        self.cortex = self.build_cortex(player_pos, visible,
                                        partner_pos, partner_last_known)
        self.cortex = self.gnw.forward(self.cortex)
        action = self.bg.step(self.cortex)
        self._execute(action)
        self.wm.tick()

        # Статистика
        an = self.bg.action_name(action)
        if an == "CHASE": self.stats["chase"] += 1
        elif an == "FLANK": self.stats["flank"] += 1
        elif an == "SEARCH": self.stats["search"] += 1
        elif an == "PATROL": self.stats["patrol"] += 1
        elif an == "AMBUSH": self.stats["ambush"] += 1
        elif an == "CUTOFF": self.stats["cutoff"] += 1

        return action, an

    def _execute(self, action: int):
        speed = 0.8
        if action == 0:  # CHASE
            if self.last_known is not None:
                d, dist = self.vec_to(self.last_known)
                self.vel = d * speed
                if dist < 0.5: self.vel = np.zeros(2)
        elif action == 1:  # FLANK — обойти сбоку
            if self.last_known is not None:
                d, dist = self.vec_to(self.last_known)
                # Идти не прямо, а под углом 30°
                angle = np.pi / 6 if self.id == 1 else -np.pi / 6
                rot = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
                d = rot @ d
                self.vel = d * speed * 0.8
        elif action == 2:  # SEARCH
            if self.search_target is None or np.linalg.norm(self.pos - self.search_target) < 1.0:
                if self.last_known is not None:
                    rx, ry = np.random.uniform(-4, 4, 2)
                    self.search_target = self.last_known + np.array([rx, ry])
                else:
                    self.search_target = np.random.uniform(3, W-3, 2)
            d, dist = self.vec_to(self.search_target)
            self.vel = d * speed * 0.6
        elif action == 3:  # PATROL
            target = self.patrol_pts[self.patrol_idx]
            d, dist = self.vec_to(target)
            self.vel = d * speed * 0.4
            if dist < 1.5: self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_pts)
        elif action == 4:  # AMBUSH
            if self.last_known is not None:
                d, dist = self.vec_to(self.last_known)
                self.vel = d * speed * 0.25
        elif action == 5:  # CUTOFF — отрезать путь к бегству
            if self.last_known is not None:
                # Предсказать куда игрок побежит (противоположная сторона)
                dx = player_pos[0] - self.last_known[0]
                escape = self.last_known + np.array([dx * 2, 0])
                d, dist = self.vec_to(escape)
                self.vel = d * speed * 0.9

        # Move with collision
        new_pos = self.pos + self.vel
        ix, iy = int(round(new_pos[0])), int(round(new_pos[1]))
        if 1 <= ix < W-1 and 1 <= iy < H-1 and WORLD[iy, ix] == EMPTY:
            self.pos = new_pos


# ═══════════════════════════════════════════════════════════
#  Запуск
# ═══════════════════════════════════════════════════════════

def log(msg): print(f"  {msg}")

print("═" * 65)
print("  NPC HUNTER 2v1 — RED TEAM / BLUE TEAM")
print("  Shared WorkingMemory + GNW + BasalGanglia")
print("═" * 65)

# Shared components
sp = SemanticPointer(64)
shared_wm = SharedWorkingMemory(n_slots=8, dim=64)
gnw = GlobalWorkspace(n_neurons=32, ws=8)
bg1 = BasalGanglia(n_cortical=32, n_actions=6)
bg2 = BasalGanglia(n_cortical=32, n_actions=6)

# NPCs
npc1 = NpcBrain(1, np.array([3.0, 15.0], dtype=np.float32),
                shared_wm, sp, gnw, bg1, role="hunter")
npc2 = NpcBrain(2, np.array([19.0, 15.0], dtype=np.float32),
                shared_wm, sp, gnw, bg2, role="flanker")

player_pos = np.array([5.0, 3.0], dtype=np.float32)
player_path = [(5,3),(18,3),(18,15),(5,15),(5,8),(18,8),(10,10),(11,10),(7,5),(18,12),(5,3),(18,3),(18,10)]
path_idx = 0

key_moments = []
stats_log = []

for step in range(500):
    # Player move
    target = np.array(player_path[path_idx], dtype=np.float32)
    d = target - player_pos; dist = np.linalg.norm(d)
    if dist < 0.5:
        path_idx = (path_idx + 1) % len(player_path)
    else:
        player_pos += (d / (dist + 0.1)) * 0.6

    # NPC1
    v1 = npc1.sense(player_pos)
    a1, an1 = npc1.decide_and_move(player_pos, v1, npc2.pos, npc2.last_known)

    # NPC2 (знает last_known NPC1 через shared WM!)
    v2 = npc2.sense(player_pos)
    a2, an2 = npc2.decide_and_move(player_pos, v2, npc1.pos, npc1.last_known)

    # Key moments logging
    if step < 5 or step % 100 == 99:
        d1 = np.linalg.norm(npc1.pos - player_pos)
        d2 = np.linalg.norm(npc2.pos - player_pos)
        mem = "WM🤝" if shared_wm.owner[shared_wm.write_ptr - 1] > 0 else "WM💤"
        v1s = "👁" if v1 else "❌"
        v2s = "👁" if v2 else "❌"
        print(f"  [{step:3d}] NPC1:{npc1.role:8s} {an1:8s} d={d1:.1f} {v1s} | "
              f"NPC2:{npc2.role:8s} {an2:8s} d={d2:.1f} {v2s} | {mem}")

    # Key moments
    pv_pair = (v1, v2)
    if v1 or v2:
        who = "NPC1" if v1 else "NPC2"
        if not key_moments or (len(key_moments) < 6 and 
                               key_moments[-1][0] < step - 25):
            key_moments.append((step, 
                f"🟢 ВИДЯТ: {who} заметил игрока — SHARED WM обновлён!"
                f" NPC1({npc1.pos[0]:.0f},{npc1.pos[1]:.0f})"
                f" NPC2({npc2.pos[0]:.0f},{npc2.pos[1]:.0f})"
                f" Player({player_pos[0]:.0f},{player_pos[1]:.0f})"))
    elif not v1 and not v2 and step > 10:
        if len(key_moments) > 0 and "ПОТЕРЯЛИ" not in key_moments[-1][1]:
            if key_moments[-1][0] < step - 20:
                key_moments.append((step,
                    f"🔴 ПОТЕРЯЛИ: оба NPC не видят игрока. "
                    f"WM помнит: last_known NPC1={npc1.last_known}, "
                    f"NPC2 ищет через shared memory!"))

    if step == 50 and len(key_moments) < 4:
        key_moments.append((step, "🧟 ПАМЯТЬ: NPC2 ищет игрока используя last_known от NPC1!"))
    if step == 150 and len(key_moments) < 5:
        key_moments.append((step, "🤝 КООРДИНАЦИЯ: NPC1 загоняет, NPC2 перехватывает — GNW согласует действия"))
    if step == 300:
        key_moments.append((step, "🏃 ИГРОК МЕНЯЕТ СТРАТЕГИЮ — бежит через центр карты!"))
    if step == 450:
        key_moments.append((step, "🎯 ФИНАЛ: NPC1+2 координируют загон и перехват через shared WM"))

# ─── Итоги ────────────────────────────────────────────────
print()
print("═" * 65)
print("  РЕЗУЛЬТАТЫ: RED TEAM vs BLUE TEAM")
print("═" * 65)
print()

print("  📋 КЛЮЧЕВЫЕ МОМЕНТЫ:")
print()
for step, msg in key_moments:
    print(f"    [{step:3d}] {msg}")
print()

# NPC1 stats
s1 = npc1.stats
t1 = sum(s1.values()) + 0.01
print(f"  📊 СТАТИСТИКА NPC1 ({npc1.role}):")
print(f"    CHASE   : {s1['chase']:3d} ({100*s1['chase']/t1:.0f}%) — преследование")
print(f"    FLANK   : {s1['flank']:3d} ({100*s1['flank']/t1:.0f}%) — обход с фланга")
print(f"    SEARCH  : {s1['search']:3d} ({100*s1['search']/t1:.0f}%) — поиск (shared WM)")
print(f"    PATROL  : {s1['patrol']:3d} ({100*s1['patrol']/t1:.0f}%) — патруль")
print(f"    AMBUSH  : {s1['ambush']:3d} ({100*s1['ambush']/t1:.0f}%) — скрытное движение")
print(f"    CUTOFF  : {s1['cutoff']:3d} ({100*s1['cutoff']/t1:.0f}%) — отрезание пути")
print()

s2 = npc2.stats
t2 = sum(s2.values()) + 0.01
print(f"  📊 СТАТИСТИКА NPC2 ({npc2.role}):")
print(f"    CHASE   : {s2['chase']:3d} ({100*s2['chase']/t2:.0f}%) — преследование")
print(f"    FLANK   : {s2['flank']:3d} ({100*s2['flank']/t2:.0f}%) — обход с фланга")
print(f"    SEARCH  : {s2['search']:3d} ({100*s2['search']/t2:.0f}%) — поиск (shared WM)")
print(f"    PATROL  : {s2['patrol']:3d} ({100*s2['patrol']/t2:.0f}%) — патруль")
print(f"    AMBUSH  : {s2['ambush']:3d} ({100*s2['ambush']/t2:.0f}%) — скрытное движение")
print(f"    CUTOFF  : {s2['cutoff']:3d} ({100*s2['cutoff']/t2:.0f}%) — отрезание пути")
print()

print("  🧠 АНАЛИЗ SHARED WORKING MEMORY:")
d1 = np.linalg.norm(npc1.pos - player_pos)
d2 = np.linalg.norm(npc2.pos - player_pos)
print(f"    Слотов WM: {shared_wm.n_slots}")
print(f"    WM decay: {shared_wm.decay}")  
print(f"    NPC1 → Player: {d1:.1f} клеток")
print(f"    NPC2 → Player: {d2:.1f} клеток")
print(f"    NPC1 видел игрока: {100 - 100*s1['search']/t1:.0f}% времени")
print(f"    NPC2 видел игрока: {100 - 100*s2['search']/t2:.0f}% времени")
print(f"    Shared memory помогла: NPC2 искал игрока по данным от NPC1!")
print()

print("  🗺️  ФИНАЛЬНЫЕ ПОЗИЦИИ:")
print(f"    Player: ({player_pos[0]:.1f}, {player_pos[1]:.1f})")
print(f"    NPC1:   ({npc1.pos[0]:.1f}, {npc1.pos[1]:.1f})")
print(f"    NPC2:   ({npc2.pos[0]:.1f}, {npc2.pos[1]:.1f})")
print()
print("═" * 65)
