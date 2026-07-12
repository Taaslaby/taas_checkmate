#!/usr/bin/env python3
"""
NPC Hunter Brain — демонстрация Working Memory + GNW + BasalGanglia.

Сценарий:
  1. NPC патрулирует область
  2. Видит игрока → запоминает позицию → преследует
  3. Игрок прячется за стеной → NPC помнит последнюю позицию
  4. NPC ищет, не находит → возвращается к патрулю

Архитектура (аналог karmac):
  WorkingMemory — circular buffer, decay, gate threshold
  GNW — gate(W_in @ state) → softmax → broadcast(W_out @ gates)
  BasalGanglia — cortico-striatal drive → WTA → action selection
  SemanticPointer — позиция как high-dim вектор

Запуск: python3 scripts/npc_hunter_demo.py
"""

import numpy as np
import time
import os
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# 1. Semantic Pointer — позиция как вектор
# ═══════════════════════════════════════════════════════════════

class SemanticPointer:
    """Позиция/NPC состояние как high-dim случайный вектор."""
    
    def __init__(self, dim: int = 128):
        self.dim = dim
        # Случайные базисные векторы для "позиция" кодирования
        np.random.seed(42)
        self.basis = np.random.randn(dim, 3).astype(np.float32)  # x, y, z
        self.basis /= np.linalg.norm(self.basis, axis=0, keepdims=True)
    
    def encode_pos(self, x: float, y: float) -> np.ndarray:
        """Кодировать 2D позицию в семантический указатель."""
        return self.normalize(self.basis[:, 0] * x + self.basis[:, 1] * y)
    
    def decode_pos(self, sp: np.ndarray) -> tuple:
        """Декодировать: cosine similarity к базисным осям."""
        return float(sp @ self.basis[:, 0]), float(sp @ self.basis[:, 1])
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
    
    def normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else v


# ═══════════════════════════════════════════════════════════════
# 2. Working Memory — помнит последние позиции
# ═══════════════════════════════════════════════════════════════

class WorkingMemory:
    """Аналог karmac::WorkingMemory — circular buffer с decay."""
    
    def __init__(self, n_slots: int = 4, dim: int = 128):
        self.n_slots = n_slots
        self.dim = dim
        self.buffer = np.zeros((n_slots, dim), dtype=np.float32)
        self.ages = np.zeros(n_slots, dtype=np.int32)
        self.write_ptr = 0
        self.decay = 0.9
        self.gate_threshold = 0.1
        self.recurrent_gain = 0.01
        self.step = 0
        
        # Случайные веса (аналог karmac::WorkingMemory::w_in, w_rec)
        np.random.seed(42)
        scale = np.sqrt(2.0 / dim)
        self.w_in = np.random.randn(dim, dim).astype(np.float32) * scale
        self.w_rec = np.random.randn(dim, dim).astype(np.float32) * scale
    
    def store(self, v: np.ndarray):
        """Записать вектор в память (round-robin)."""
        assert len(v) == self.dim
        self.step += 1
        
        # Gate: не записывать если сигнал слишком слабый
        norm = np.linalg.norm(v)
        if norm < self.gate_threshold:
            return
        
        # Проекция через W_in
        projected = self.w_in @ v
        self.buffer[self.write_ptr] = projected
        self.ages[self.write_ptr] = 0
        
        # Age all other slots
        self.ages += 1
        self.ages[self.write_ptr] = 0
        
        self.write_ptr = (self.write_ptr + 1) % self.n_slots
    
    def recall(self, slot: int = 0) -> np.ndarray:
        """Вспомнить слот."""
        return self.buffer[slot].copy()
    
    def recall_all(self) -> np.ndarray:
        """Все слоты как flat vector."""
        return self.buffer.flatten()
    
    def lru_slot(self) -> int:
        """Least-recently-used слот."""
        return int(np.argmax(self.ages))
    
    def similarity_to_all(self, v: np.ndarray) -> np.ndarray:
        """Cosine similarity вектора ко всем слотам."""
        norms = np.linalg.norm(self.buffer, axis=1, keepdims=True)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10 or np.any(norms < 1e-10):
            return np.zeros(self.n_slots)
        sims = (self.buffer @ v) / (norms.flatten() * norm_v)
        return sims
    
    def tick(self):
        """Один шаг рекуррентной динамики (аналог karmac WM.tick())."""
        self.step += 1
        # Decay
        self.buffer *= self.decay
        # Рекуррентное возбуждение
        for slot in range(self.n_slots):
            delta = self.w_rec @ self.buffer[slot]
            self.buffer[slot] += delta * self.recurrent_gain
    
    def clear(self):
        self.buffer.fill(0.0)
        self.ages.fill(0)
        self.write_ptr = 0


# ═══════════════════════════════════════════════════════════════
# 3. GNW — Global Neuronal Workspace (внимание)
# ═══════════════════════════════════════════════════════════════

class GlobalWorkspace:
    """Аналог karmac::gnw::GlobalWorkspace."""
    
    def __init__(self, n_neurons: int = 16, workspace_size: int = 4):
        self.n = n_neurons
        self.ws = workspace_size
        
        # Xavier init
        np.random.seed(42)
        gate_scale = np.sqrt(6.0 / (n_neurons + workspace_size))
        self.gate_w = np.random.randn(workspace_size, n_neurons).astype(np.float32) * gate_scale
        
        bc_scale = np.sqrt(6.0 / (n_neurons + workspace_size)) * 0.01
        self.broadcast_w = np.random.randn(n_neurons, workspace_size).astype(np.float32) * bc_scale
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """r += broadcast(softmax(gate(r))) — residual."""
        assert len(state) == self.n
        
        # gate: r → gates = W_gate @ r
        gates = self.gate_w @ state
        
        # softmax
        max_g = np.max(gates)
        exps = np.exp(gates - max_g)
        gates = exps / (np.sum(exps) + 1e-10)
        
        # broadcast: gates → delta = W_broadcast @ gates
        delta = self.broadcast_w @ gates
        
        # residual
        return state + delta


# ═══════════════════════════════════════════════════════════════
# 4. BasalGanglia — выбор действия
# ═══════════════════════════════════════════════════════════════

class BasalGanglia:
    """Аналог karmac::basal_ganglia::BasalGanglia."""
    
    ACTIONS = ["CHASE", "SEARCH", "PATROL", "WAIT", "AMBUSH"]
    
    def __init__(self, n_cortical: int = 16, n_actions: int = 5):
        self.n_cortical = n_cortical
        self.n_actions = n_actions
        self.state = np.zeros(n_cortical, dtype=np.float32)
        self.prev_action: Optional[int] = None
        
        # Xavier init
        np.random.seed(42)
        scale = np.sqrt(2.0 / (n_cortical + n_actions))
        self.cortico_striatal = np.random.randn(n_actions, n_cortical).astype(np.float32) * scale
        
        # Конфиг
        self.d1_gain = 2.0      # Go pathway
        self.d2_gain = 1.5      # NoGo pathway
        self.stn_excitation = 0.5
        self.thalamus_gain = 1.0
        self.decay = 0.95
    
    def step(self, cortex_input: np.ndarray) -> int:
        """Выбрать действие. Возвращает индекс действия."""
        assert len(cortex_input) == self.n_cortical
        
        # Accumulate with decay
        self.state = self.decay * self.state + cortex_input
        
        # Cortico-striatal drive
        drive = self.cortico_striatal @ self.state
        
        # D1/D2 modulation
        drive[drive > 0] *= self.d1_gain
        drive[drive < 0] *= self.d2_gain
        
        # STN broad excitation
        drive += self.stn_excitation * np.mean(drive)
        
        # Lateral inhibition (soft WTA через softmax с temperature)
        max_d = np.max(drive)
        exps = np.exp((drive - max_d) * 2.0)  # temperature = 0.5
        probs = exps / (np.sum(exps) + 1e-10)
        
        # Hysteresis: prefer previous action if ambiguous
        if self.prev_action is not None and np.max(probs) < 0.4:
            # Если неуверен — остаёмся на предыдущем действии
            pass  # softmax уже даёт распределение
        
        # Выбираем argmax с небольшим шумом для разнообразия
        if np.random.random() < 0.05:
            action = np.random.randint(self.n_actions)
        else:
            action = int(np.argmax(probs))
        
        self.prev_action = action
        return action
    
    def reset(self):
        self.state.fill(0.0)
        self.prev_action = None
    
    def action_name(self, idx: int) -> str:
        return self.ACTIONS[idx] if idx < len(self.ACTIONS) else f"ACTION_{idx}"


# ═══════════════════════════════════════════════════════════════
# 5. 2D Мир с стенами
# ═══════════════════════════════════════════════════════════════

WALL = 1
EMPTY = 0

WORLD = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
], dtype=np.int32)

HEIGHT, WIDTH = WORLD.shape


# ═══════════════════════════════════════════════════════════════
# 6. NPC Brain — полный пайплайн
# ═══════════════════════════════════════════════════════════════

class NpcBrain:
    """Мозг NPC охотника."""
    
    def __init__(self):
        self.sp = SemanticPointer(64)
        self.wm = WorkingMemory(n_slots=4, dim=64)
        self.gnw = GlobalWorkspace(n_neurons=16, workspace_size=4)
        self.bg = BasalGanglia(n_cortical=16, n_actions=5)
        
        # Состояние
        self.npc_pos = np.array([7.0, 14.0], dtype=np.float32)  # старт внизу
        self.npc_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.last_known_player = None  # последняя известная позиция
        self.search_target = None       # куда идём искать
        self.patrol_points = [
            np.array([5.0, 5.0]),
            np.array([10.0, 5.0]),
            np.array([10.0, 10.0]),
            np.array([5.0, 10.0]),
        ]
        self.patrol_idx = 0
        self.steps_since_seen = 0
        self.frustration = 0.0  # нарастает если не видит игрока
        
        # Cortex state (16 нейронов — аналог GNW n_neurons)
        self.cortex = np.zeros(16, dtype=np.float32)
    
    def has_line_of_sight(self, player_pos: np.ndarray) -> bool:
        """Проверка прямой видимости (DDA ray marching)."""
        dx = player_pos[0] - self.npc_pos[0]
        dy = player_pos[1] - self.npc_pos[1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < 0.1:
            return True
        
        steps = max(int(dist * 4), 1)
        for i in range(1, steps):
            t = i / steps
            x = int(round(self.npc_pos[0] + dx * t))
            y = int(round(self.npc_pos[1] + dy * t))
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                if WORLD[y, x] == WALL:
                    return False
        return True
    
    def get_vector_to(self, target: np.ndarray) -> np.ndarray:
        """Нормализованный вектор к цели."""
        d = target - self.npc_pos
        dist = np.linalg.norm(d)
        return d / (dist + 0.1), dist
    
    def build_cortex(self, player_pos: np.ndarray,
                     player_visible: bool) -> np.ndarray:
        """
        Построить кортикальное состояние (16 нейронов):
          [0-3]:  направление к игроку (normalized dx)
          [4-7]:  направление к last_known (normalized dx)
          [8]:    player_visible (0/1)
          [9]:    steps_since_seen (squashed)
          [10]:   frustration
          [11]:   vector to patrol point
          [12-15]: working memory similarity (4 слота)
        """
        cx = np.zeros(16, dtype=np.float32)
        
        # [0-3]: направление к игроку
        if player_pos is not None:
            d, dist = self.get_vector_to(player_pos)
            cx[0] = d[0]
            cx[1] = d[1]
            cx[2] = dist / 20.0  # normalized by max vision range
            cx[3] = 1.0 if dist < 5.0 else 0.0
        
        # [4-7]: направление к last_known
        if self.last_known_player is not None:
            d, dist = self.get_vector_to(self.last_known_player)
            cx[4] = d[0]
            cx[5] = d[1]
            cx[6] = 1.0
            cx[7] = dist / 20.0
        
        # [8]: player visible
        cx[8] = 1.0 if player_visible else 0.0
        
        # [9]: steps_since_seen (squashed)
        cx[9] = np.tanh(self.steps_since_seen / 50.0)
        
        # [10]: frustration
        self.frustration = np.clip(
            self.frustration + (0.01 if not player_visible else -0.05),
            0.0, 1.0
        )
        cx[10] = self.frustration
        
        # [11]: направление к patrol point
        patrol = self.patrol_points[self.patrol_idx]
        d, dist = self.get_vector_to(patrol)
        cx[11] = dist / 20.0
        
        # [12-15]: working memory similarity к last_known
        if self.last_known_player is not None:
            lk_sp = self.sp.encode_pos(
                self.last_known_player[0], self.last_known_player[1]
            )
            sims = self.wm.similarity_to_all(lk_sp)
            cx[12:16] = sims
        
        return cx
    
    def step(self, player_pos: np.ndarray) -> tuple:
        """
        Один шаг NPC мозга.
        
        Returns:
            (action_idx, description, player_visible, memory_active)
        """
        # 1. Сенсорика
        player_visible = self.has_line_of_sight(player_pos)
        
        if player_visible:
            # Видим игрока → запоминаем
            self.last_known_player = player_pos.copy()
            player_sp = self.sp.encode_pos(player_pos[0], player_pos[1])
            self.wm.store(player_sp)
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1
        
        # 2. Построить cortex
        self.cortex = self.build_cortex(player_pos, player_visible)
        
        # 3. GNW — attention broadcast
        self.cortex = self.gnw.forward(self.cortex)
        
        # 4. BasalGanglia — выбрать действие
        action = self.bg.step(self.cortex)
        
        # 5. Исполнить действие
        self._execute_action(action, player_pos, player_visible)
        
        # 6. Обновить WorkingMemory tick
        self.wm.tick()
        
        memory_active = self.last_known_player is not None \
            and not player_visible \
            and self.steps_since_seen < 100
        
        return (action, self.bg.action_name(action),
                player_visible, memory_active)
    
    def _execute_action(self, action: int, player_pos: np.ndarray,
                        player_visible: bool):
        """Исполнить выбранное действие."""
        speed = 0.8
        
        if action == 0:  # CHASE
            if self.last_known_player is not None:
                d, dist = self.get_vector_to(self.last_known_player)
                self.npc_vel = d * speed
                if dist < 0.5:
                    self.npc_vel = np.array([0.0, 0.0])
        
        elif action == 1:  # SEARCH
            if self.search_target is None or \
               np.linalg.norm(self.npc_pos - self.search_target) < 1.0:
                # Выбрать новую точку для поиска
                # Поиск вокруг last_known
                if self.last_known_player is not None:
                    rx = np.random.uniform(-3, 3)
                    ry = np.random.uniform(-3, 3)
                    self.search_target = self.last_known_player + \
                        np.array([rx, ry])
                else:
                    self.search_target = np.random.uniform(2, WIDTH-2, 2)
            d, dist = self.get_vector_to(self.search_target)
            self.npc_vel = d * speed * 0.7  # медленнее при поиске
        
        elif action == 2:  # PATROL
            target = self.patrol_points[self.patrol_idx]
            d, dist = self.get_vector_to(target)
            self.npc_vel = d * speed * 0.5
            if dist < 1.0:
                self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_points)
        
        elif action == 3:  # WAIT
            self.npc_vel = np.array([0.0, 0.0])
            # Раз в 10 шагов — оглянуться
            if np.random.random() < 0.1:
                self.npc_vel = np.random.randn(2) * 0.2
        
        elif action == 4:  # AMBUSH
            # Идти к последней известной позиции игрока, но медленно
            if self.last_known_player is not None:
                d, dist = self.get_vector_to(self.last_known_player)
                self.npc_vel = d * speed * 0.3  # крадучись
        
        # Применить скорость
        new_pos = self.npc_pos + self.npc_vel
        
        # Collision detection
        ix, iy = int(round(new_pos[0])), int(round(new_pos[1]))
        if 1 <= ix < WIDTH-1 and 1 <= iy < HEIGHT-1:
            if WORLD[iy, ix] == EMPTY:
                self.npc_pos = new_pos
    
    def debug_info(self) -> dict:
        """Текущее состояние для отладки."""
        return {
            "pos": self.npc_pos.copy(),
            "vel": self.npc_vel.copy(),
            "last_known": self.last_known_player.copy()
                if self.last_known_player is not None else None,
            "steps_since_seen": self.steps_since_seen,
            "frustration": self.frustration,
            "patrol_idx": self.patrol_idx,
            "action": self.bg.action_name(self.bg.prev_action or 0)
                if self.bg.prev_action is not None else "NONE",
        }


# ═══════════════════════════════════════════════════════════════
# 7. Рендер
# ═══════════════════════════════════════════════════════════════

def render_frame(world: np.ndarray, npc: NpcBrain,
                  player_pos: np.ndarray, memory_active: bool):
    """Отрисовать кадр в консоль."""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    h, w = world.shape
    display = np.full((h, w), ' ', dtype=str)
    
    for y in range(h):
        for x in range(w):
            if world[y, x] == WALL:
                display[y, x] = '█'
            else:
                display[y, x] = '·'
    
    # Player
    px, py = int(round(player_pos[0])), int(round(player_pos[1]))
    if 0 <= px < w and 0 <= py < h:
        display[py, px] = '🧑'
    
    # NPC
    nx, ny = int(round(npc.npc_pos[0])), int(round(npc.npc_pos[1]))
    if 0 <= nx < w and 0 <= ny < h:
        # Показать "память" если игрок не виден но NPC помнит
        if memory_active:
            display[ny, nx] = '🧟'
        else:
            display[ny, nx] = '🔴'
    
    # Last known position marker
    if npc.last_known_player is not None and not npc.has_line_of_sight(player_pos):
        lx, ly = int(round(npc.last_known_player[0])), \
                 int(round(npc.last_known_player[1]))
        if 0 <= lx < w and 0 <= ly < h:
            if world[ly, lx] != WALL and (lx != px or ly != py):
                display[ly, lx] = '❌'
    
    # Print
    info = npc.debug_info()
    print("╔" + "═" * (w * 2 + 2) + "╗")
    for y in range(h):
        print("║ " + " ".join(display[y]) + " ║")
    print("╚" + "═" * (w * 2 + 2) + "╝")
    print()
    print(f"  NPC:       ({info['pos'][0]:5.1f}, {info['pos'][1]:5.1f})")
    print(f"  Действие:  {info['action']}")
    print(f"  Игрок:     видим={npc.has_line_of_sight(player_pos)} "
          f"({player_pos[0]:.1f}, {player_pos[1]:.1f})")
    print(f"  Last seen: {info['last_known']}")
    print(f"  Скрылся:   {info['steps_since_seen']} шагов назад")
    print(f"  WM slots:  {[f'{s:.2f}' for s in npc.wm.similarity_to_all(npc.sp.encode_pos(player_pos[0], player_pos[1]))]}")
    print(f"  Фрустрация: {info['frustration']:.2f}")
    print()
    if memory_active:
        print("  🧟 NPC ПОМНИТ игрока и ищет!")
    elif npc.has_line_of_sight(player_pos):
        print("  🔴 NPC ВИДИТ игрока и преследует!")
    else:
        print("  NPC потерял игрока :(")
    
    time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════
# 8. Демо-запуск
# ═══════════════════════════════════════════════════════════════

def run_demo():
    """Запустить демонстрацию NPC охотника."""
    npc = NpcBrain()
    player_pos = np.array([3.0, 3.0], dtype=np.float32)
    player_vel = np.array([0.0, 0.0], dtype=np.float32)
    step = 0
    
    # Паттерн движения игрока: бегает по карте
    player_path = [
        (3, 3), (12, 3), (12, 12), (3, 3),
        (7, 5), (7, 10), (3, 3), (12, 3),
        (12, 12), (3, 12), (7, 10),
    ]
    path_idx = 0
    
    print("╔══════════════════════════════════════════════╗")
    print("║     NPC HUNTER — DEMO WORKING MEMORY        ║")
    print("║                                             ║")
    print("║  🔴  = NPC (охотник)                        ║")
    print("║  🧑  = Player (игрок)                       ║")
    print("║  ❌  = Last known (где видел)               ║")
    print("║  █  = Стена                                 ║")
    print("║                                             ║")
    print("║  NPC помнит позицию игрока 100 шагов        ║")
    print("║  после того, как игрок скрылся за стеной!   ║")
    print("╚══════════════════════════════════════════════╝")
    input("Нажми Enter для старта...")
    
    for step in range(300):
        # Двигаем игрока по пути
        target = np.array(player_path[path_idx], dtype=np.float32)
        d = target - player_pos
        dist = np.linalg.norm(d)
        if dist < 0.5:
            path_idx = (path_idx + 1) % len(player_path)
        else:
            player_pos += (d / (dist + 0.1)) * 0.6
        
        # NPC шаг
        action, action_name, player_visible, memory_active = \
            npc.step(player_pos)
        
        # Рендер (каждый 2-й шаг для читаемости)
        if step % 2 == 0 or player_visible:
            render_frame(WORLD, npc, player_pos, memory_active)
        
        # Драматические моменты
        if step == 50:
            print("\n  Игрок бежит за стену! NPC запоминает...")
            time.sleep(1)
        elif step == 100:
            print("\n  NPC всё ещё ищет — WM держит позицию!")
            time.sleep(1)
        elif step == 150:
            print("\n  Игрок выходит из укрытия...")
            time.sleep(1)
        elif step == 250:
            print("\n  Игрок опять прячется. Начинает темнеть...")
            time.sleep(0.5)
    
    print("\n  Демо завершено!")
    print(f"  Всего шагов: {step+1}")
    print(f"  Финальная фрустрация NPC: {npc.frustration:.2f}")
    print(f"  Шагов без игрока: {npc.steps_since_seen}")


if __name__ == "__main__":
    run_demo()
