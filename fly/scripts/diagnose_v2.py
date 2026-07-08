#!/usr/bin/env python3
"""
Diagnose V2 — Headless диагностика активности нейронов.

Запускает LIF brain + V2VNCController без GUI,
каждые 100ms выводит:
  - Статистику MN rates по ногам и типам движения
  - CPG фазы + ext/flx активацию
  - Actuator drive (сколько actuator'ов активны, с какой силой)
  - Позицию и скорость мухи
  - Top DN rates (если доступны)

Запуск:
  cd taas_checkmate
  python3 fly/scripts/diagnose_v2.py
"""

import sys
import time
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

FLY_DIR = Path(__file__).parent.parent  # fly/
PROJ = FLY_DIR.parent  # корень проекта (taas_checkmate/)
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(FLY_DIR / "scripts"))
XML_PATH = FLY_DIR / "models" / "scene.xml"

def diagnose(duration=10.0, model='lif'):
    import mujoco as mu
    from flywire_brain import FlyWireCSR, FlyWireLIFBrain
    from fly.archive.vnc_controller import V2VNCController, LEGS, SEGMENTS

    print("═" * 65)
    print("  🧬 V2 Диагностика: активность нейронов")
    print("═" * 65)
    print()

    # ─── MuJoCo ───
    mj_model = mu.MjModel.from_xml_path(str(XML_PATH))
    data = mu.MjData(mj_model)
    mu.mj_resetData(mj_model, data)
    data.qpos[2] = 0.01

    # ─── CSR ───
    print("  [1] Загрузка CSR...")
    t0 = time.time()
    csr = FlyWireCSR()
    csr.load()
    print(f"      {time.time()-t0:.1f}s — {csr.n_neurons:,} нейронов, {len(csr.mn_indices)} MNs")
    if not csr.is_v2:
        print("  ⚠ Нет V2 датасета!")
        return
    print()

    # ─── Brain ───
    print(f"  [2] Инициализация LIF brain...")
    t0 = time.time()
    brain = FlyWireLIFBrain(csr)
    print(f"      {time.time()-t0:.1f}s — backend: {'GPU' if hasattr(brain,'use_gpu') and brain.use_gpu else 'CPU'}")
    print()

    # ─── VNC ───
    print("  [3] Инициализация V2VNCController...")
    t0 = time.time()
    vnc = V2VNCController(mj_model, csr, brain)
    print(f"      {time.time()-t0:.1f}s — {len(vnc.mn_to_ctrl)} MNs mapped")
    print()

    # ─── Settle ───
    print("  [4] Settle (0.2s)...")
    # Standby: держим ноги жёстко, чтобы муха не провалилась
    data.qpos[2] = 0.03
    vnc.standby(data)
    for _ in range(2000):
        for act_idx in vnc.actuator_map.values():
            data.ctrl[act_idx] = -0.15
        for i in vnc.adhere:
            data.ctrl[i] = 1.0
        mu.mj_step(mj_model, data)
    print(f"      z={data.qpos[2]:.4f}, contacts={data.ncon}")
    print()

    # ─── Main loop ───
    dt = mj_model.opt.timestep
    brain_step = max(1, int(0.025 / dt))  # 25ms
    max_steps = int(duration / dt)

    vnc.reset()

    # Stats accumulators
    mn_rate_log = []  # per brain update
    act_log = []  # actuator ctrl values per step
    pos_log = []

    print(f"  [5] Симуляция {duration}s...")
    print(f"      brain_step = {brain_step} MuJoCo шагов (~25ms)")
    print(f"      Модель: {model.upper()}")
    if hasattr(brain, 'named_dn_indices') and brain.named_dn_indices:
        print(f"      DN мониторинг: {', '.join(brain.named_dn_indices.keys())}")
    print()
    print(f"  {'t(s)':>6} {'x(m)':>8} {'y(m)':>8} {'z(m)':>8} {'v(m/s)':>8} | "
          f"{'leg_act_mean':>12} {'leg_act_max':>12} | "
          f"{'MN_mean':>8} {'MN_max':>8} {'MN_act':>8} | "
          f"{'actuators':>10}")
    print(f"  {'-'*6:>6} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} | "
          f"{'-'*12:>12} {'-'*12:>12} | "
          f"{'-'*8:>8} {'-'*8:>8} {'-'*8:>8} | "
          f"{'-'*10:>10}")

    prev_x = float(data.qpos[0])
    prev_y = float(data.qpos[1])

    brain_step_times = []  # для статистики в конце
    sim_start = time.time()

    for step in range(max_steps):
        t = step * dt

        # Brain update каждые 25ms
        if step % brain_step == 0:
            # Looming
            sx, sy = float(data.qpos[0]), float(data.qpos[1])
            dt_sim = brain_step * dt
            svx = (sx - prev_x) / max(dt_sim, 1e-6)
            svy = (sy - prev_y) / max(dt_sim, 1e-6)
            brain.set_sensory_state(sx, sy, svx, svy)
            # DNg100 + CB0013 injection через brain._injection_targets
            # (применяется внутри step_batch ПОСЛЕ _set_sensory_input,
            #  который обнуляет external_I — старый код через external_I терял injection!)
            if vnc._dng100_idx is not None:
                brain._injection_targets[vnc._dng100_idx] = vnc._injection_current
            if hasattr(vnc, '_cb0013_idx') and vnc._cb0013_idx is not None:
                brain._injection_targets[vnc._cb0013_idx] = vnc._injection_current_cb0013
            _t0 = time.time()
            brain.step_batch(t)
            brain_step_times.append(time.time() - _t0)
            prev_x, prev_y = sx, sy

            # ─── Каждые 100ms — детальный вывод ───
            if step % (brain_step * 4) == 0:
                mn_rates_raw = brain.get_mn_rates()
                dn_rates = brain.dn_rate

                # MN statistics
                if len(mn_rates_raw) > 0:
                    mn_mean = float(np.mean(mn_rates_raw))
                    mn_max = float(np.max(mn_rates_raw))
                    mn_active = int(np.sum(mn_rates_raw > 0.01))
                else:
                    mn_mean = mn_max = 0.0; mn_active = 0

                # Leg activation from neural drive
                leg_act = vnc._leg_act.copy()
                leg_act_mean = float(np.mean(leg_act))
                leg_act_max = float(np.max(leg_act))

                # Actuator drive
                active_acts = int(np.sum(np.abs(data.ctrl) > 0.01))
                mean_abs_ctrl = float(np.mean(np.abs(data.ctrl))) if len(data.ctrl) > 0 else 0

                # Position + velocity
                vx = svx if step > 0 else 0.0

                print(f"  {t:>6.1f} {data.qpos[0]:>8.4f} {data.qpos[1]:>8.4f} "
                      f"{data.qpos[2]:>8.4f} {vx:>8.3f} | "
                      f"{leg_act_mean:>12.4f} {leg_act_max:>12.4f} | "
                      f"{mn_mean:>8.4f} {mn_max:>8.4f} {mn_active:>8d} | "
                      f"{active_acts:>4d}@{mean_abs_ctrl:.2f}")

                # ─── Детальная статистика каждые 500ms ───
                if step % (brain_step * 20) == 0:
                    # Per-leg MN rates
                    leg_mn_rates = defaultdict(list)
                    for i, rid in enumerate(csr.mn_root_ids):
                        if i < len(mn_rates_raw) and mn_rates_raw[i] > 0.01:
                            ctrl_list = vnc.mn_to_ctrl.get(rid)
                            if ctrl_list:
                                act_idx, _, motion = ctrl_list[0]
                                leg_i = vnc._actuator_leg_idx.get(act_idx, -1)
                                if leg_i >= 0:
                                    leg_mn_rates[(leg_i, motion)].append(float(mn_rates_raw[i]))

                    print()
                    print(f"    ─── MN rates по ногам @ t={t:.1f}s ───")
                    for leg_i, (label, _) in enumerate(LEGS):
                        motions = [m for (li, m), _ in leg_mn_rates.items() if li == leg_i]
                        if not motions:
                            continue
                        parts = []
                        for motion in sorted(set(motions)):
                            vals = leg_mn_rates[(leg_i, motion)]
                            parts.append(f"{motion}={np.mean(vals):.3f}(n={len(vals)})")
                        print(f"      {label}: {', '.join(parts)}")

                    # Neural activation detail
                    print()
                    print(f"    ─── Leg activation (neural VNC) @ t={t:.1f}s ───")
                    for leg_i, (label, _) in enumerate(LEGS):
                        act = vnc._leg_act[leg_i]
                        bar = '█' * min(10, int(act * 10))
                        # Per-leg actuator control values
                        act_vals = []
                        for act_idx, leg_idx in vnc._actuator_leg_idx.items():
                            if leg_idx == leg_i and act_idx < len(data.ctrl):
                                act_vals.append(abs(float(data.ctrl[act_idx])))
                        mean_ctrl = np.mean(act_vals) if act_vals else 0.0
                        max_ctrl = np.max(act_vals) if act_vals else 0.0
                        print(f"      {label}: act={act:.3f} {bar}  ctrl_mean={mean_ctrl:.3f} max={max_ctrl:.3f}  ({len(act_vals)} actuators)")

                    # DN rates top-5
                    if len(dn_rates) > 0:
                        dn_active = np.sum(dn_rates > 0.01)
                        dn_top = np.argsort(dn_rates)[::-1][:5]
                        print()
                        print(f"    ─── DN rates (top-5/{len(dn_rates)}, active={dn_active}) @ t={t:.1f}s ───")
                        for i, dn_idx in enumerate(dn_top):
                            dn_rid = csr.dn_root_ids[dn_idx] if dn_idx < len(csr.dn_root_ids) else -1
                            dn_type = csr.type_map.get(dn_rid, '?')
                            print(f"      [{i}] #{dn_idx} {dn_type:15s} rate={dn_rates[dn_idx]:.4f}")

                    # Named DNs
                    if hasattr(brain, 'named_dn_indices') and brain.named_dn_indices:
                        print()
                        print(f"    ─── Named DNs ───")
                        n_steps = brain._last_n_steps if hasattr(brain, '_last_n_steps') else 50
                        spike_buf = brain._last_spike_buf if hasattr(brain, '_last_spike_buf') else None
                        if spike_buf is not None:
                            for name, csr_idx in brain.named_dn_indices.items():
                                spike_count = 0
                                for s in range(n_steps):
                                    offset = s * csr.n_neurons
                                    if csr_idx < csr.n_neurons and spike_buf[offset + csr_idx]:
                                        spike_count += 1
                                hz = spike_count / (n_steps * brain.dt_ms / 1000.0)
                                print(f"      {name}: {hz:.0f}Hz (spikes={spike_count}/{n_steps})")

                    print()

        # VNC step — каждые 2 MuJoCo шага
        if step % 2 == 0:
            vnc.step(dt * 2.0, data)  # dt*2 потому что 2 шага прошло
        mu.mj_step(mj_model, data)

        if t >= duration:
            break

    # ─── Final stats ───
    elapsed = time.time() - sim_start
    print()
    print("═" * 65)
    print("  📊 Итоговая статистика")
    print("═" * 65)
    print(f"  Позиция: x={data.qpos[0]:.3f}  y={data.qpos[1]:.3f}  z={data.qpos[2]:.4f}")
    print(f"  Скорость: ~{vx:.3f} m/s")
    if brain_step_times:
        bt = np.array(brain_step_times)
        print(f"\n  ⏱ Производительность:")
        print(f"    Всего brain steps: {len(bt)}")
        print(f"    Средний brain step: {np.mean(bt):.3f}s")
        print(f"    Медиана brain step: {np.median(bt):.3f}s")
        print(f"    Макс brain step: {np.max(bt):.3f}s")
        print(f"    Мин brain step: {np.min(bt):.3f}s")
        est_rt = t / elapsed if elapsed > 0 else 0
        print(f"    Симуляция: {t:.1f}s за {elapsed:.1f}s (RT={est_rt:.2f}x)")

    # Final MN rates check
    mn_rates = brain.get_mn_rates()
    if len(mn_rates) > 0:
        print(f"\n  MN rates:")
        print(f"    Всего MNs: {len(mn_rates)}")
        print(f"    Активных (>0.01): {np.sum(mn_rates > 0.01)}")
        print(f"    Средняя: {np.mean(mn_rates):.4f}")
        print(f"    Макс: {np.max(mn_rates):.4f}")
        print(f"    Ненулевых: {np.sum(mn_rates > 0)}")

        # Per-leg
        leg_mn = defaultdict(list)
        for i, rid in enumerate(csr.mn_root_ids):
            if i < len(mn_rates):
                ctrl_list = vnc.mn_to_ctrl.get(rid)
                if ctrl_list:
                    act_idx, _, motion = ctrl_list[0]
                    leg_i = vnc._actuator_leg_idx.get(act_idx, -1)
                    if leg_i >= 0:
                        leg_mn[(leg_i, motion)].append(float(mn_rates[i]))
        print(f"\n    Per-leg MN rates:")
        for leg_i, (label, _) in enumerate(LEGS):
            rates_for_leg = {}
            for m in ['flex', 'extend', 'abduct', 'twist']:
                vals = [rate for (li, m2), rates in leg_mn.items()
                        if li == leg_i and m2 == m
                        for rate in rates
                        if rate > 0.01]
                rates_for_leg[m] = float(np.mean(vals)) if len(vals) > 0 else 0.0
            parts = [f"{m}={v:.3f}" for m, v in rates_for_leg.items() if v > 0.01]
            if parts:
                print(f"      {label}: {', '.join(parts)}")

    # Neural activation final state
    print(f"\n  Leg activation (neural VNC):")
    for leg_i, (label, _) in enumerate(LEGS):
        act = vnc._leg_act[leg_i]
        bar = '█' * min(10, int(act * 10))
        act_vals = []
        for act_idx, leg_idx in vnc._actuator_leg_idx.items():
            if leg_idx == leg_i and act_idx < len(data.ctrl):
                act_vals.append(abs(float(data.ctrl[act_idx])))
        mean_ctrl = np.mean(act_vals) if act_vals else 0.0
        print(f"    {label}: act={act:.3f} {bar}  ctrl={mean_ctrl:.3f}  ({len(act_vals)} actuators)")

    print(f"\n  ✓ Диагностика завершена!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="🧬 V2 Диагностика")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--model", choices=['lif', 'izhi', 'rate', 'psi'], default='lif')
    args = parser.parse_args()

    if not XML_PATH.exists():
        print(f"  Нет файла: {XML_PATH}")
        sys.exit(1)

    diagnose(duration=args.duration, model=args.model)


if __name__ == "__main__":
    main()
