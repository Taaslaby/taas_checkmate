#!/usr/bin/env python3
"""🐜 100K Brain (HH RANDOM) → MuJoCo Ant, обучение T2 SPIGRAD (REINFORCE)

Архитектура:
  HH brain (100K, random ER connectome, I=20 tonic) → motor spikes [8]
      ↓ (buffer over N_STEPS=40 brain steps = 20ms temporal window)
  SpikeGradBridge (spike_dim=8 → n_osc=8 → 3*8=24 drives)
      ↓ T3 forward → PST features [pst_dim]
  Linear decoder → action mean [8]
      ↓ + noise → Ant step → reward
  REINFORCE: advantage → grad_pst → Bridge.backward → update W_drive
                                          ↓
                                    Decoder: W_decode += lr * PST * (a-μ)/σ² * advantage

HH (Hodgkin-Huxley): самое биологичное ядро — ионные каналы Na⁺/K⁺.
Voltage clamp [-100, 50] встроен в ядро (починено!).
HH I=20 даёт ~50% motor firing rate — само-осцилляция!
"""

import sys, os, time, argparse
import numpy as np
from pathlib import Path

# Отключаем буферизацию вывода — иначе print() в файл висит до конца обучения
sys.stdout.reconfigure(line_buffering=True)

os.environ['LD_LIBRARY_PATH'] = '/usr/lib/python3.14/site-packages/nvidia/cudnn/lib'

BASE = Path(__file__).resolve().parent.parent
SARDAANA = Path("/home/taaslab/Проекты/asm_engines/sardaanaCM")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(SARDAANA))

import gymnasium as gym
import taas_checkmate as tcm

# ═══ CONFIG ═══════════════════════════════════════════════
N = 100000
DT_MS = 0.5

N_OSC = 8
N_STEPS = 40  # HH ISI ~10ms → 20 steps; captures 2+ spikes
SPIKE_DIM = 8
DRIVE_GAIN = 10.0
ALPHA = 1.0
FREQ_MIN = 0.5
FREQ_MAX = 3.0
W12, W13, W23 = 0.3, 0.15, 0.08

SIGMA_INIT = 0.3
SIGMA_FINAL = 0.05
LR_BRIDGE = 0.001
LR_DECODE = 0.01
LR_BASELINE = 0.01

TONIC_HH = 20.0  # HH background (I=20 → ~50% motor firing)
HH_SENSORY_SCALE = 10.0  # HH: dv = I/C_m * dt = I * 0.5
WARMUP_STEPS = 150  # HH settles fast


# ═══ RANDOM CONNECTOME ═══════════════════════════════════

def build_random_connectome(n, avg_deg=15, weight_std=0.3,
                            motor_guarantee=8, rng=None):
    """Build a purely random Erdos-Renyi connectome in CSR format.

    Каждый нейрон получает ~avg_deg случайных пост-синаптических целей.
    Motor нейроны (последние 8) гарантированно получают >= motor_guarantee
    входящих связей (иначе моторные спайки = 0 и bridge не учится).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    motor_start = n - 8
    m = int(n * avg_deg * 1.3)  # немного с запасом

    # Семплируем случайные пары (source, target)
    sources = rng.randint(0, n, size=m)
    targets = rng.randint(0, n, size=m)

    # Убираем self-loops
    mask = sources != targets
    sources = sources[mask]
    targets = targets[mask]

    # Сортируем по source для CSR
    order = np.argsort(sources, kind='stable')
    sources = sources[order]
    targets = targets[order]

    # Убираем дубликаты (source, target) — keep first
    dup_src = np.diff(sources, prepend=-1) == 0
    dup_tgt = np.diff(targets, prepend=-1) == 0
    sources = sources[~(dup_src & dup_tgt)]
    targets = targets[~(dup_src & dup_tgt)]

    # Гарантируем моторным нейронам входящие связи
    extra_src, extra_tgt = [], []
    for mi in range(motor_start, n):
        incoming = int(np.sum(targets == mi))
        if incoming < motor_guarantee:
            need = motor_guarantee - incoming
            pres = rng.randint(0, n, size=need * 2)  # запас
            pres = pres[pres != mi][:need]
            for p in pres:
                extra_src.append(p)
                extra_tgt.append(mi)

    if extra_src:
        sources = np.concatenate([sources, np.array(extra_src, dtype=np.int32)])
        targets = np.concatenate([targets, np.array(extra_tgt, dtype=np.int32)])
        # Пересортируем
        order = np.argsort(sources, kind='stable')
        sources = sources[order]
        targets = targets[order]

    # Строим row_ptr через bincount
    counts = np.bincount(sources.astype(np.int32), minlength=n)
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(counts).astype(np.int32)

    # Веса: log-normal (HH: dv = w/C_m * dt = w * 0.5. PSP ~0.01-0.5mV)
    values = np.exp(rng.randn(len(targets)) * weight_std - 1.5)
    values = np.clip(values, 0.01, 0.5).astype(np.float32)

    nt = np.where(rng.rand(n) < 0.2, -1.0, 1.0).astype(np.float32)

    print(f"  🔗 {len(targets):,} synapses (avg deg ~{len(targets)//n}) for {n:,}N")
    motor_in = [int(np.sum(targets == mi)) for mi in range(motor_start, n)]
    print(f"  🦾 Motor in-degree: {motor_in}")
    print()

    return tcm.ConnectomeData(
        row_ptr, targets.astype(np.int32), values, nt)


# ═══ MAIN ════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ant-steps", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--lr-bridge", type=float, default=LR_BRIDGE)
    parser.add_argument("--lr-decode", type=float, default=LR_DECODE)
    parser.add_argument("--avg-deg", type=int, default=15,
                        help="Average degree in random connectome")
    args = parser.parse_args()

    rng = np.random.RandomState(42)
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  🐜 100K Brain (RANDOM) → Ant, обучение T2 SPIGRAD ║")
    print("╚═══════════════════════════════════════════════════════╝\n")

    # ── 1. Build random connectome ──
    print("[1/4] Building random connectome ...")
    t0 = time.time()
    conn = build_random_connectome(N, avg_deg=args.avg_deg, rng=rng)
    print(f"  ⏱️  {time.time()-t0:.1f}s\n")

    # ── 2. Brain setup (HH — Hodgkin-Huxley, ионные каналы Na⁺/K⁺) ──
    print("[2/4] BrainSimulator HH (Hodgkin-Huxley) ...")
    sim = tcm.BrainSimulator(conn.n_neurons, "hh")
    # HH: стандартные NaV + KV каналы, I=20 → ~50% motor firing rate
    # Voltage clamp [-100, 50] mV встроен в ядро (после фикса)

    n = conn.n_neurons
    motor_start = n - SPIKE_DIM
    sensory_idx = list(range(27))
    motor_idx = list(range(motor_start, n))
    print(f"  👁️  Sensory: 27  🦾 Motor: {motor_start}–{n-1}")

    print(f"  🔥 Warming up {WARMUP_STEPS} steps (I={TONIC_HH}) ...")
    warmup_fire = []
    ext_I_warm = np.zeros(n, dtype=np.float32)
    for w in range(WARMUP_STEPS):
        ext_I_warm.fill(TONIC_HH)
        spike_buf = sim.step_batch(conn, 1, dt_ms=DT_MS, external_I=ext_I_warm)
        sb = np.asarray(spike_buf).flatten()
        warmup_fire.append(int(sb.sum()))
        if w % 10 == 0:
            print(f"    Step {w:3d}: {warmup_fire[-1]:>5d} spikes")

    avg_fire = np.mean(warmup_fire[-10:])
    tonic_training = TONIC_HH
    print(f"  🔥 Avg firing (last 10): {avg_fire:.0f} spikes/step")
    if avg_fire < 100:
        tonic_training = 30.0
        print(f"  ⚠️  Low firing ({avg_fire:.0f}). Boosting to tonic={tonic_training}")
        ext_I_warm.fill(tonic_training)
        for _ in range(20):
            sim.step_batch(conn, 1, dt_ms=DT_MS, external_I=ext_I_warm)
    else:
        print(f"  ✅ HH brain alive: {avg_fire:.0f} spikes/step")

    # Final stats — HH само-осциллирует, motor kick не нужен!
    ext_I_warm.fill(tonic_training)
    sb = np.asarray(sim.step_batch(conn, 1, dt_ms=DT_MS, external_I=ext_I_warm)).flatten()
    motor_fire = int(sb[motor_idx].sum())
    total_fire = int(sb.sum())
    print(f"  🔥 Motor spikes: {motor_fire}/{SPIKE_DIM}, Total: {total_fire:,}/{n:,} ({100*total_fire/n:.1f}%)")
    if motor_fire == 0:
        print("  ⚠️  Motor neurons still silent! Check motor_idx range.")
    else:
        print(f"  ✅ Motor neurons firing! HH self-oscillates — bridge gets input!")
    print()

    # ── 3. T2 SPIGRAD setup ──
    print("[3/4] T2 SPIGRAD setup ...")
    pst_dim = tcm.pst_linear_dim(N_OSC)
    bridge = tcm.SpikeGradBridge(
        SPIKE_DIM, N_OSC,
        dt=0.05, drive_gain=DRIVE_GAIN, alpha=ALPHA,
        freq_min=FREQ_MIN, freq_max=FREQ_MAX,
        coupling_w12=W12, coupling_w13=W13, coupling_w23=W23,
        zero_init=False,
    )
    W_decode = np.random.randn(pst_dim, SPIKE_DIM).astype(np.float32) * 0.01
    b_decode = np.zeros(SPIKE_DIM, dtype=np.float32)
    print(f"  Bridge: spike_dim={SPIKE_DIM} -> n_osc={N_OSC} (drive_dim={bridge.drive_dim()})")
    print(f"  Decoder: PST dim={pst_dim} -> {SPIKE_DIM} actions")
    total = bridge.drive_dim() * SPIKE_DIM + SPIKE_DIM + pst_dim * SPIKE_DIM + SPIKE_DIM
    print(f"  Total params: {total:,}")
    print()

    # ── 4. Training loop ──
    print(f"[4/4] Training {args.episodes} episodes ...\n")
    env = gym.make("Ant-v4", render_mode="human" if args.render else None)

    episode_rewards = []
    reward_baseline = 0.0
    global_step = 0

    for ep in range(args.episodes):
        sigma = max(SIGMA_FINAL, SIGMA_INIT * (1 - ep / max(args.episodes - 1, 1)))

        try:
            obs, info = env.reset(seed=42 + ep)
        except TypeError:
            obs = env.reset()
            info = {}
        obs = np.asarray(obs, dtype=np.float32)

        ep_reward = 1.0
        ep_actions = 0
        ep_spikes = 0
        action_buf = np.zeros(8, dtype=np.float32)

        while ep_actions < args.ant_steps:
            # Collect motor spikes over N_STEPS brain steps
            motor_buffer = np.zeros((N_STEPS, SPIKE_DIM), dtype=np.float32)

            ext_I = np.zeros(n, dtype=np.float32)
            for brain_s in range(N_STEPS):
                ext_I.fill(0.0)
                obs_clipped = np.clip(obs / 10.0, -1.0, 1.0).astype(np.float32)
                for i in range(min(27, len(obs_clipped))):
                    ext_I[sensory_idx[i]] += obs_clipped[i] * HH_SENSORY_SCALE  # HH: I/C_m * dt
                ext_I += tonic_training  # HH tonic — само-осцилляция

                spike_buf = sim.step_batch(conn, 1, dt_ms=DT_MS,
                                           external_I=ext_I)
                sb = np.asarray(spike_buf).flatten()
                motor_buffer[brain_s] = sb[motor_idx].astype(np.float32)
                ep_spikes += int(sb.sum())

            # T3 forward: motor spikes -> PST features
            pst_mat, traj = bridge.forward(motor_buffer, N_STEPS)
            pst_feat = pst_mat[-1]

            # Decode PST -> action
            action_mean = pst_feat @ W_decode + b_decode
            action = action_mean + np.random.randn(SPIKE_DIM).astype(np.float32) * sigma
            action = np.clip(action, -1.0, 1.0)
            action_buf = action_buf * 0.7 + action * 0.3

            # Ant step
            obs, reward, terminated, truncated, _ = env.step(action_buf)
            obs = np.asarray(obs, dtype=np.float32)
            ep_reward += float(reward)
            ep_actions += 1
            global_step += 1

            # REINFORCE update
            baseline_update = float(reward) - reward_baseline
            reward_baseline += LR_BASELINE * baseline_update
            advantage = baseline_update

            # Gradient ASCENT direction
            d_mu = advantage * (action - action_mean) / (sigma**2 + 1e-8)
            d_mu = np.clip(d_mu, -10.0, 10.0)

            # Decoder ASCENT
            W_decode += LR_DECODE * np.outer(pst_feat, d_mu).astype(np.float32)
            b_decode += LR_DECODE * d_mu.astype(np.float32)

            # Bridge DESCENT (bridge.update does w -= lr * grad)
            # dJ/dPST = W_decode @ d_mu_ascent  (ASCENT direction)
            # dL/dPST = -dJ/dPST = -W_decode @ d_mu  (DESCENT direction)
            grad_pst = -(W_decode.astype(np.float64) @ d_mu.astype(np.float64))
            grad_pst = grad_pst.astype(np.float32)

            grad_w, grad_b = bridge.backward(grad_pst, traj, motor_buffer)
            grad_w = np.clip(grad_w, -5.0, 5.0)
            grad_b = np.clip(grad_b, -5.0, 5.0)
            bridge.update(grad_w, grad_b, args.lr_bridge)

            if ep_actions % 10 == 0:
                mean_spikes = ep_spikes // max(ep_actions * N_STEPS, 1)
                print(f"  ep={ep+1} act={ep_actions:>3d} "
                      f"reward={reward:>+6.2f} "
                      f"avg_spike={mean_spikes:>5d} "
                      f"sigma={sigma:.2f}")

            if terminated or truncated:
                print(f"  🏁 Episode {ep+1}: ant fell after {ep_actions} steps")
                break

        episode_rewards.append(ep_reward)
        mean_spike = ep_spikes // (ep_actions * N_STEPS + 1)
        print(f"  {'-'*50}")
        print(f"  EP {ep+1}: reward={ep_reward:>+7.1f}  "
              f"steps={ep_actions:>3d}  "
              f"mean_spike={mean_spike:>5d}")
        print(f"  {'-'*50}")

    env.close()

    # Final summary
    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Episodes: {args.episodes}")
    print(f"  Total ant steps: {global_step}")
    for i, r in enumerate(episode_rewards):
        print(f"    Ep {i+1}: {r:>+8.1f}")
    if len(episode_rewards) > 1:
        trend = episode_rewards[-1] - episode_rewards[0]
        arrow = "up" if trend > 0 else "down"
        print(f"  Trend: {arrow} ({abs(trend):.1f})")
    print(f"\n  100K brain + T2 SPIGRAD + Ant - обучение пошло!")


if __name__ == "__main__":
    main()
