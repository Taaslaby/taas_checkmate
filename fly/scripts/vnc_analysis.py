#!/usr/bin/env python3
"""
vnc_analysis.py — нейробиологический анализ MANC VNC коннектома.

4 анализа в одном скрипте (headless, без GUI):

  1. Top DN firing rates — какие DNs реально активны при ходьбе
  2. Correlation DN_i ↔ velocity — какие DNs управляют скоростью
  3. Ablation study — что будет если отключить каждый DN по одному
  4. RL on DN→VNC — REINFORCE на 270 DN modulation weights

Запуск:
  python3 fly/scripts/vnc_analysis.py --duration 10 --analyze
  python3 fly/scripts/vnc_analysis.py --ablation
  python3 fly/scripts/vnc_analysis.py --train-rl --episodes 20
  python3 fly/scripts/vnc_analysis.py --all  # всё сразу

Зависимости: pip install mujoco numpy pandas pyarrow
"""

import sys
import time
import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

FLY_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FLY_DIR / "scripts"))

raise ImportError(
    "\n\n⚠ vnc_analysis.py устарел для V2 кода.\n"
    "Импорты MANCLoader, VNCController, VNCCPG, VNCConnectomeEngine —\n"
    "эти классы удалены из vnc_walk.py (переход на V2VNCController).\n"
    "\nАнализ DN ablation, RL, корреляций требует полного переписывания\n"
    "под V2 API. Пока используйте vnc_walk.py напрямую для визуализации.\n"
    "\n"
    "from vnc_walk import (\n"
    "    MANCLoader, VNCController, VNCCPG, VNCConnectomeEngine,\n"
    "    LEGS, LEG_ROIS, SEGMENTS, TRIPOD_A, TRIPOD_B,\n"
    "    FLY_DIR, XML_PATH, DLOAD, NT_COEFF_BY_TYPE, NT_SHORT\n"
    ")\n"
)


# ═══════════════════════════════════════════════════════════════
#  Headless simulation
# ═══════════════════════════════════════════════════════════════

def _run_headless(duration, model, data, vnc, brain, quiet=False,
                  collect_dn=True, collect_velocity=True, collect_phases=True,
                  dn_mod=None):
    """Headless MuJoCo loop without GUI.

    Returns dict:
        dn_log:    [T, n_DN] — dn_act per step
        velocity:  [T] — forward velocity
        phases:    [T, 6] — leg phases
        contacts:  [T, 6] — contact states
        x_pos:     [T] — x position
        z_pos:     [T] — z position (height)
        t_steps:   [T] — timestamps
    """
    import mujoco as mu

    # Reset MuJoCo state (important for repeated calls e.g. ablation)
    mu.mj_resetData(model, data)
    data.qpos[2] = 0.01

    # Apply RL modulation if provided (per-episode, static)
    if dn_mod is not None:
        vnc.dn_mod[:] = np.clip(dn_mod.copy(), -0.5, 0.5) + 1.0

    # Settle
    for _ in range(2000):
        mu.mj_step(model, data)

    vnc.reset()

    dt = model.opt.timestep
    n_steps = int(duration / dt)

    # Pre-allocate buffers
    dn_log = np.zeros((n_steps, vnc.n_dn), dtype=np.float32) if collect_dn else None
    velocity = np.zeros(n_steps, dtype=np.float32) if collect_velocity else None
    phases = np.zeros((n_steps, 6), dtype=np.float64) if collect_phases else None
    contacts = np.zeros((n_steps, 6), dtype=bool) if collect_phases else None
    x_pos = np.zeros(n_steps, dtype=np.float64)
    z_pos = np.zeros(n_steps, dtype=np.float64)
    t_steps = np.zeros(n_steps, dtype=np.float64)

    brain_step_interval = max(1, int(0.025 / dt))

    # For velocity
    prev_x = float(data.qpos[0])
    prev_t = 0.0

    for step in range(n_steps):
        t = step * dt

        # FlyWire brain (if available)
        if brain is not None and step % brain_step_interval == 0:
            sx, sy = float(data.qpos[0]), float(data.qpos[1])
            dt_sim = t - prev_t if t > prev_t else dt
            svx = (sx - prev_x) / dt_sim if dt_sim > 0 else 0.0
            svy = 0.0
            prev_x, prev_t = sx, t
            brain.set_sensory_state(sx, sy, svx, svy)
            brain.step_batch(t)

        vnc.step(dt, data)
        mu.mj_step(model, data)

        # Collect
        x_pos[step] = float(data.qpos[0])
        z_pos[step] = float(data.qpos[2])
        t_steps[step] = t

        if collect_dn and dn_log is not None:
            dn_log[step] = vnc.dn_act.copy()

        if collect_velocity and velocity is not None:
            if step > 0:
                velocity[step] = (x_pos[step] - x_pos[step-1]) / dt

        if collect_phases and phases is not None:
            phases[step] = vnc.cpg.phases.copy()
            contacts[step] = vnc.cpg.get_contact_states(data)

    return {
        'dn_log': dn_log,
        'velocity': velocity,
        'phases': phases,
        'contacts': contacts,
        'x_pos': x_pos,
        'z_pos': z_pos,
        't_steps': t_steps,
    }


# ═══════════════════════════════════════════════════════════════
#  VNC Analyzer
# ═══════════════════════════════════════════════════════════════

class VNCAnalyzer:
    """Загрузка + headless симуляция + анализы VNC коннектома."""

    def __init__(self, use_brain=False, autotune_path=None, quiet=False, model='izhi'):
        import mujoco as mu
        self.quiet = quiet
        self.log = print if not quiet else lambda *a, **k: None

        self.log(f"  Модель: {XML_PATH}")
        self.model = mu.MjModel.from_xml_path(str(XML_PATH))
        self.data = mu.MjData(self.model)
        mu.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.01

        # MANC
        self.log("")
        self.manc = MANCLoader()
        self.manc.load()

        # FlyWire brain (опционально)
        self.brain = None
        self.model = model
        if use_brain:
            self._load_brain(autotune_path)

        # VNC
        self.log("")
        self.vnc = VNCController(self.model, self.manc, brain=self.brain)

        # DN metadata
        self.dn_list = list(self.manc.dn['bodyId'].values)
        self.dn_types = [str(self.manc.dn.iloc[i].get('type', '?'))
                         for i in range(len(self.dn_list))]
        self.n_dn = len(self.dn_list)

        # Headless settle
        self.log("\n  Settle...")
        for _ in range(2000):
            mu.mj_step(self.model, self.data)
        self.log(f"  z={self.data.qpos[2]:.4f}")

    def _load_brain(self, autotune_path=None):
        """Load FlyWire brain (Izhikevich, rate, or psi)."""
        try:
            from brain_bridge import BrainBridge
            from flywire_brain import FlyWireCSR, FlyWireBrain, FlyWireRateBrain, FlyWirePsiBrain

            bridge = BrainBridge()
            bridge.load()

            csr = FlyWireCSR()
            csr.load()

            fw_to_manc = {}
            for typ in bridge.shared_types:
                fw_roots = bridge.type_to_fw.get(typ, [])
                manc_bids = bridge.type_to_manc.get(typ, [])
                if not manc_bids:
                    continue
                for i, fw_rid in enumerate(fw_roots):
                    manc_bid = manc_bids[i % len(manc_bids)]
                    fw_to_manc[int(fw_rid)] = int(manc_bid)

            if self.model == 'rate':
                self.brain = FlyWireRateBrain(csr, fw_to_manc=fw_to_manc,
                                              manc_bodyids=list(bridge.bodyids))
                self.log(f"\n  FlyWire→VNC: {len(fw_to_manc)} DNs (Rate-based)")
            elif self.model == 'psi':
                self.brain = FlyWirePsiBrain(csr, fw_to_manc=fw_to_manc,
                                             manc_bodyids=list(bridge.bodyids))
                self.log(f"\n  FlyWire→VNC: {len(fw_to_manc)} DNs (ψ-wave)")
            else:
                self.brain = FlyWireBrain(csr, fw_to_manc=fw_to_manc,
                                          manc_bodyids=list(bridge.bodyids),
                                          autotune_path=autotune_path)
                self.log(f"\n  FlyWire→VNC: {len(fw_to_manc)} DNs (Izhikevich)")
            self.fw_to_manc = fw_to_manc
        except Exception as e:
            self.log(f"\n  ⚠ FlyWire brain не загружен: {e}")
            self.log("  → Fallback без мозга")

    # ─── 1. Top DN firing rates ─────────────────────────────

    def analyze(self, duration=15.0):
        """Анализ: топ DNs + корреляция со скоростью.

        Выводит:
          - Средние firing rates для всех DNs
          - Топ-10 активных DNs
          - Топ-10 DNs по корреляции со скоростью
        """
        self.log(f"\n{'═'*55}")
        self.log("  📊 Анализ DNs при ходьбе")
        self.log(f"  Длительность: {duration}s")
        self.log(f"{'═'*55}")

        # Headless run with logging
        data = _run_headless(duration, self.model, self.data,
                             self.vnc, self.brain,
                             quiet=self.quiet)

        dn_log = data['dn_log']
        velocity = data['velocity']

        if dn_log is None or velocity is None:
            self.log("  ⚠ Данные не собраны")
            return

        # 1a. Средние firing rates
        mean_rates = np.mean(dn_log, axis=0)

        # Топ-10 по firing rate
        top_idx = np.argsort(mean_rates)[::-1][:10]
        self.log(f"\n  ─── Top DNs по firing rate (средний за {duration}s) ───")
        self.log(f"  {'#':>3} {'bodyId':>10} {'type':>15} {'mean_rate':>10} {'velocity_corr':>14}")
        self.log(f"  {'-'*55}")
        for rank, idx in enumerate(top_idx):
            bodyid = self.dn_list[idx]
            dtype = self.dn_types[idx]

            # Correlation with velocity
            r = self._pearsonr(dn_log[:, idx], velocity)
            r_str = f"{r:+.3f}" if not np.isnan(r) else "   N/A"

            self.log(f"  {rank+1:>3} {bodyid:>10} {dtype:>15} {mean_rates[idx]:>10.4f} {r_str:>14}")

        # 1b. DN type distribution in top-20
        top20_idx = np.argsort(mean_rates)[::-1][:20]
        type_counts = defaultdict(int)
        for idx in top20_idx:
            type_counts[self.dn_types[idx]] += 1
        self.log(f"\n  ─── Типы DNs в топ-20 ───")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            self.log(f"    {t:20s}: {c:2d}")

        # 1c. Тихие DNs (rate < 0.01)
        n_silent = np.sum(mean_rates < 0.01)
        self.log(f"\n  ─── Статистика ───")
        self.log(f"    Всего DNs:        {self.n_dn}")
        self.log(f"    Активных:         {np.sum(mean_rates >= 0.01)}")
        self.log(f"    Тихих (rate<.01): {n_silent}")
        self.log(f"    Средняя скорость: {np.mean(velocity):.4f} m/s")
        self.log(f"    Макс скорость:    {np.max(velocity):.4f} m/s")

        return {
            'mean_rates': mean_rates,
            'top_idx': top_idx,
            'velocity_corr': [self._pearsonr(dn_log[:, i], velocity)
                              for i in range(self.n_dn)]
        }

    # ─── 2. Correlation DN_i ↔ velocity ───────────────────

    def correlation_analysis(self, duration=15.0):
        """Полный корреляционный анализ DN_i × velocity.

        Возвращает: список (bodyId, type, pearson_r, p_value) по убыванию |r|
        """
        self.log(f"\n{'═'*55}")
        self.log("  🔗 Корреляция DNs со скоростью")
        self.log(f"{'═'*55}")

        data = _run_headless(duration, self.model, self.data,
                             self.vnc, self.brain,
                             quiet=self.quiet)

        dn_log = data['dn_log']
        velocity = data['velocity']
        if dn_log is None or velocity is None:
            return []

        results = []
        for i in range(self.n_dn):
            r = self._pearsonr(dn_log[:, i], velocity)
            # Approximate p-value from t-distribution
            n = len(velocity)
            t_stat = r * math.sqrt((n - 2) / max(1 - r*r, 1e-10))
            p = 2.0 * (1.0 - min(abs(t_stat) / (n**0.5 + 1.0), 1.0))  # quick approx
            results.append({
                'bodyId': self.dn_list[i],
                'type': self.dn_types[i],
                'r': r,
                'p': p,
                'idx': i,
            })

        # Sort by |r|
        results.sort(key=lambda x: abs(x['r']), reverse=True)

        self.log(f"\n  ─── DNs with |r| > 0.3 ───")
        self.log(f"  {'bodyId':>10} {'type':>15} {'r':>8} {'p':>8} {'label':>15}")
        self.log(f"  {'-'*60}")
        n_significant = 0
        for r in results:
            if abs(r['r']) < 0.3:
                continue
            n_significant += 1
            label = 'ACCEL' if r['r'] > 0.3 else ('BRAKE' if r['r'] < -0.3 else '')
            p_str = f"{r['p']:.4f}" if not np.isnan(r['p']) else "N/A"
            self.log(f"  {r['bodyId']:>10} {r['type']:>15} {r['r']:>+8.3f} {p_str:>8} {label:>15}")

        self.log(f"\n  ─── Статистика ───")
        self.log(f"    |r| > 0.5: {sum(1 for r in results if abs(r['r']) > 0.5)}")
        self.log(f"    |r| > 0.3: {n_significant}")
        self.log(f"    Всего DNs: {len(results)}")

        return results

    # ─── 3. Ablation study ─────────────────────────────────

    def ablation(self, duration=5.0, baseline_speed=None):
        """Ablation study: отключаем DN по одному.

        1. Baseline: все DNs активны
        2. Для каждого DN: обнуляем его drive → измеряем скорость

        Выводит:
          - Для каждого DN: Δspeed (относительно baseline)
          - Топ-10 критичных DNs
          - Классификация: CRITICAL, REDUNDANT, INHIBITORY
        """
        self.log(f"\n{'═'*55}")
        self.log("  🔪 Ablation Study")
        self.log("  Поочередное отключение DNs")
        self.log(f'  Длительность прогона: {duration}s')
        self.log(f"{'═'*55}")

        # Baseline
        self.log(f"\n  ─── Baseline: все DNs активны ───")
        base_data = _run_headless(duration, self.model, self.data,
                                  self.vnc, self.brain,
                                  quiet=self.quiet)
        baseline_speed = np.mean(np.abs(base_data['velocity'][100:]))  # skip first 0.1s
        base_x = base_data['x_pos'][-1]
        self.log(f"    Baseline speed: {baseline_speed:.4f} m/s")
        self.log(f"    Δx={base_x:.4f}m")

        # Ablate each DN
        self.log(f"\n  ─── Ablating DNs ───")
        results = []

        # Save original dn_mod
        orig_mod = self.vnc.dn_mod.copy()

        for dn_idx in range(self.n_dn):
            # Ablate: set modulation to 0
            self.vnc.dn_mod[:] = orig_mod.copy()
            self.vnc.dn_mod[dn_idx] = 0.0

            try:
                ab_data = _run_headless(duration, self.model, self.data,
                                        self.vnc, self.brain,
                                        quiet=True)
                ab_speed = np.mean(np.abs(ab_data['velocity'][100:]))
                delta_speed = ab_speed - baseline_speed
                pct = (delta_speed / baseline_speed * 100) if baseline_speed > 0 else 0.0

                results.append({
                    'idx': dn_idx,
                    'bodyId': self.dn_list[dn_idx],
                    'type': self.dn_types[dn_idx],
                    'baseline_speed': baseline_speed,
                    'ablated_speed': ab_speed,
                    'delta': delta_speed,
                    'pct_change': pct,
                })
            except Exception as e:
                self.log(f"    ⚠ DN {dn_idx} failed: {e}")
                continue

        # Restore modulation
        self.vnc.dn_mod[:] = orig_mod

        # Sort by impact (most critical first)
        results.sort(key=lambda x: x['delta'])

        # Print
        self.log(f"\n  {'═'*55}")
        self.log(f"  📋 Результаты Ablation")
        self.log(f"  {'#':>3} {'bodyId':>10} {'type':>15} {'speed':>8} {'Δ':>8} {'Δ%':>8} {'label':>15}")
        self.log(f"  {'-'*65}")

        critical, redundant, inhibitory = 0, 0, 0
        for rank, r in enumerate(results):
            pct = r['pct_change']
            if pct < -20:
                label = '⚠ CRITICAL'
                critical += 1
            elif pct > 20:
                label = '🟢 INHIBITORY'
                inhibitory += 1
            else:
                label = '— REDUNDANT'
                redundant += 1

            self.log(f"  {rank+1:>3} {r['bodyId']:>10} {r['type']:>15} "
                     f"{r['ablated_speed']:>8.4f} {r['delta']:>+8.4f} {r['pct_change']:>+7.1f}% {label:>15}")

        self.log(f"\n  ─── Итог ───")
        self.log(f"    CRITICAL  (speed↓ >20%): {critical}")
        self.log(f"    REDUNDANT (|speedΔ| <20%): {redundant}")
        self.log(f"    INHIBITORY(speed↑ >20%): {inhibitory}")

        return results

    # ─── 4. RL on DN→VNC ──────────────────────────────────

    def train_rl(self, episodes=20, episode_duration=5.0,
                 hidden_dim=64, lr=1e-3, gamma=0.99):
        """REINFORCE with baseline on DN modulation (270 params).

        State:  [dn_rates(270), leg_phases(6), contacts(6), velocity(1)] = 283-d
        Action: modulation per DN (270-d), continuous in [-0.5, 0.5]
        Reward: vx - 0.5*|z-0.01| + 0.1*n_contacts

        Args:
            episodes: number of training episodes
            episode_duration: seconds per episode
            hidden_dim: policy network hidden size
            lr: learning rate
            gamma: discount factor
        """
        self.log(f"\n{'═'*55}")
        self.log("  🤖 RL Training: DN→VNC Modulation")
        self.log(f"  Episodes: {episodes}, duration: {episode_duration}s")
        self.log(f"  Action dim: {self.n_dn}, Hidden: {hidden_dim}")
        self.log(f"{'═'*55}")

        import torch
        import torch.nn as nn
        import torch.optim as optim

        state_dim = self.n_dn + 6 + 6 + 1  # 283

        # ─── Policy network ───
        class Policy(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim):
                super().__init__()
                self.fc1 = nn.Linear(state_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.mean = nn.Linear(hidden_dim, action_dim)
                self.log_std = nn.Parameter(torch.zeros(action_dim) - 2.0)

            def forward(self, x):
                h = torch.relu(self.fc1(x))
                h = torch.relu(self.fc2(h))
                mean = torch.tanh(self.mean(h)) * 0.5  # [-0.5, 0.5]
                std = torch.exp(self.log_std).expand_as(mean)
                return mean, std

        # ─── Baseline network ───
        class Baseline(nn.Module):
            def __init__(self, state_dim, hidden_dim):
                super().__init__()
                self.fc1 = nn.Linear(state_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.value = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                h = torch.relu(self.fc1(x))
                h = torch.relu(self.fc2(h))
                return self.value(h).squeeze(-1)

        policy = Policy(state_dim, self.n_dn, hidden_dim)
        baseline = Baseline(state_dim, hidden_dim)
        opt_p = optim.Adam(policy.parameters(), lr=lr)
        opt_b = optim.Adam(baseline.parameters(), lr=lr)

        # ─── Training loop ───
        # Per-episode modulation: each episode samples ONE action (270-d),
        # applies it as static DN modulation for the entire episode.
        episode_rewards = []
        episode_speeds = []
        best_reward = -float('inf')
        best_mod = None

        self.log(f"\n  ─── Обучение (per-episode REINFORCE) ───")
        self.log(f"  {'ep':>4} {'reward':>10} {'speed':>8} {'|action|':>8} {'best':>8}")
        self.log(f"  {'-'*45}")

        # All episodes start from same initial state (MuJoCo reset + settle)
        # State: zeros — policy learns from scratch which DNs to modulate
        state = np.zeros(state_dim, dtype=np.float32)

        for ep in range(episodes):
            # 1. Sample action from policy based on current state
            s_t = torch.from_numpy(state)
            with torch.no_grad():
                mean, std = policy(s_t)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample().numpy()

            # 3. Apply action (per-episode static modulation)
            self.vnc.dn_mod[:] = np.clip(action, -0.5, 0.5) + 1.0

            # 4. Run full episode with this modulation
            ep_data = _run_headless(episode_duration, self.model, self.data,
                                    self.vnc, self.brain,
                                    collect_dn=True, collect_velocity=True,
                                    quiet=True)
            vel = ep_data['velocity']
            zp = ep_data['z_pos']
            ct = ep_data['contacts']

            # 5. Compute episode reward
            #    Reward = forward speed - height penalty + contact stability
            speed = np.mean(np.abs(vel[100:])) if vel is not None else 0.0
            height_penalty = np.mean(np.abs(zp[100:] - 0.01)) if zp is not None else 0.0
            contact_stable = np.mean(np.sum(ct[100:], axis=1)) if ct is not None else 3.0
            ep_reward = speed - 0.5 * height_penalty + 0.05 * contact_stable

            episode_rewards.append(ep_reward)
            episode_speeds.append(speed)

            # 6. REINFORCE update (per-episode)
            if ep >= 1:
                R = torch.tensor(ep_reward, dtype=torch.float32)

                with torch.no_grad():
                    V = baseline(s_t)
                    advantage = R - V

                # Policy loss: -log_prob * advantage
                mean, std = policy(s_t)
                dist_ep = torch.distributions.Normal(mean, std)
                log_prob = dist_ep.log_prob(torch.from_numpy(action)).sum()
                policy_loss = -(log_prob * advantage)

                # Baseline loss: MSE(V, R)
                V_pred = baseline(s_t)
                baseline_loss = nn.MSELoss()(V_pred.unsqueeze(0), R.unsqueeze(0))

                opt_p.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt_p.step()

                opt_b.zero_grad()
                baseline_loss.backward()
                opt_b.step()

            # Track best
            mean_abs_action = np.mean(np.abs(action))
            if ep_reward > best_reward:
                best_reward = ep_reward
                best_mod = action.copy()
                best_str = " ★"
            else:
                best_str = ""

            if ep % 5 == 0 or ep == episodes - 1:
                self.log(f"  {ep:>4d} {ep_reward:>+10.3f} {speed:>8.4f} {mean_abs_action:>8.3f} {best_reward:>+8.3f}{best_str}")

        # ─── Результаты ───
        self.log(f"\n  ─── Результаты RL ───")
        self.log(f"    Лучший эпизод: reward={best_reward:.3f}, speed={episode_speeds[episode_rewards.index(max(episode_rewards))]:.4f}")
        if best_mod is not None:
            top_dn = np.argsort(best_mod)[::-1][:5]
            bottom_dn = np.argsort(best_mod)[:5]
            boosted_idx = [i for i in top_dn if best_mod[i] > 0.1][:5]
            suppressed_idx = [i for i in bottom_dn if best_mod[i] < -0.1][:5]
            self.log(f"    Усиленные DNs: {[(self.dn_types[i], f'{best_mod[i]:+.2f}') for i in boosted_idx]}")
            self.log(f"    Подавленные DN: {[(self.dn_types[i], f'{best_mod[i]:+.2f}') for i in suppressed_idx]}")

        return {
            'episode_rewards': episode_rewards,
            'episode_speeds': episode_speeds,
            'best_mod': best_mod,
            'best_reward': best_reward,
        }

    # ─── Utilities ─────────────────────────────────────────

    @staticmethod
    def _pearsonr(x, y):
        """Fast Pearson correlation."""
        xm = x - x.mean()
        ym = y - y.mean()
        num = (xm * ym).sum()
        den = np.sqrt((xm**2).sum() * (ym**2).sum())
        return num / den if den > 1e-10 else 0.0


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="🧠 VNC Connectome Analysis")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Длительность каждого прогона (s)")
    parser.add_argument("--analyze", action="store_true",
                        help="Анализ топ DNs + корреляция со скоростью")
    parser.add_argument("--ablation", action="store_true",
                        help="Ablation study")
    parser.add_argument("--train-rl", action="store_true",
                        help="RL обучение на DN modulation")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Эпизодов RL")
    parser.add_argument("--all", action="store_true",
                        help="Всё сразу: analyze + ablation + RL")
    parser.add_argument("--model", type=str, choices=['izhi', 'psi', 'rate'], default='izhi',
                        help="Модель нейронов: izhi, psi, или rate (∼100× быстрее)")
    parser.add_argument("--autotune", type=str, nargs='?', const='auto',
                        default=None, help="Per-type Izhikevich params")
    parser.add_argument("--brain", action="store_true",
                        help="Загрузить FlyWire brain")
    parser.add_argument("--quiet", action="store_true",
                        help="Минимум вывода")
    args = parser.parse_args()

    print("═"*55)
    print("  🧠 VNC Connectome Analysis")
    print("═"*55)

    # Autotune path
    autotune_path = None
    if args.autotune == 'auto':
        autotune_path = str(FLY_DIR / "scripts" / "autotune_results.npz")
    elif args.autotune:
        autotune_path = args.autotune

    if not XML_PATH.exists():
        print(f"  Нет файла: {XML_PATH}")
        sys.exit(1)

    analyzer = VNCAnalyzer(use_brain=args.brain,
                           autotune_path=autotune_path,
                           quiet=args.quiet, model=args.model)

    if args.all:
        print("\n" + "░"*55)
        print("  [1/3] Analysis DNs")
        print("░"*55)
        analyzer.analyze(duration=args.duration)

        print("\n" + "░"*55)
        print("  [2/3] Ablation")
        print("░"*55)
        analyzer.ablation(duration=min(5.0, args.duration))

        print("\n" + "░"*55)
        print("  [3/3] RL")
        print("░"*55)
        analyzer.train_rl(episodes=args.episodes)
    else:
        if args.analyze:
            analyzer.analyze(duration=args.duration)
        if args.ablation:
            analyzer.ablation(duration=min(5.0, args.duration))
        if args.train_rl:
            analyzer.train_rl(episodes=args.episodes)

    print(f"\n  {'✓ Готово!':>50}")


if __name__ == "__main__":
    main()
