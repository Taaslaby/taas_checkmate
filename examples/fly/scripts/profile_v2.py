#!/usr/bin/env python3
"""
Profile V2 — детальная профилировка каждой компоненты симуляции.

Запускает LIF brain + VNC + MuJoCo без GUI и замеряет время каждого компонента.

Запуск:
  cd taas_checkmate
  python3 fly/scripts/profile_v2.py --duration 5

Вывод:
  - Таблица среднего времени каждого компонента на шаг/вызов
  - Суммарный breakdown (доля в %)
  - RT factor
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

FLY_DIR = Path(__file__).parent.parent
PROJ = FLY_DIR.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(FLY_DIR / "scripts"))
XML_PATH = FLY_DIR / "models" / "scene.xml"


def profile(duration=5.0):
    import mujoco as mu
    from flywire_brain import FlyWireCSR, FlyWireLIFBrain, tcm as taas_module_tcm
    from fly.archive.vnc_controller import V2VNCController

    print("═" * 65)
    print("  📊 Profile V2 — детальная профилировка")
    print("═" * 65)
    print()

    # ─── Setup ───
    mj_model = mu.MjModel.from_xml_path(str(XML_PATH))
    data = mu.MjData(mj_model)
    mu.mj_resetData(mj_model, data)
    data.qpos[2] = 0.01

    print("  [1] Загрузка CSR...")
    t0 = time.time()
    csr = FlyWireCSR()
    csr.load()
    print(f"      {time.time()-t0:.1f}s — {csr.n_neurons:,} нейронов")
    if not csr.is_v2:
        print("  ⚠ Нет V2 датасета!"); return
    print()

    print("  [2] Инициализация LIF brain...")
    t0 = time.time()
    brain = FlyWireLIFBrain(csr)
    print(f"      {time.time()-t0:.1f}s")
    print()

    print("  [3] Инициализация VNC...")
    t0 = time.time()
    vnc = V2VNCController(mj_model, csr, brain)
    print(f"      {time.time()-t0:.1f}s")
    print()

    # ─── Settle ───
    print("  [4] Settle...")
    data.qpos[2] = 0.03
    vnc.standby(data)
    for _ in range(2000):
        for act_idx in vnc.actuator_map.values():
            data.ctrl[act_idx] = -0.15
        for i in vnc.adhere:
            data.ctrl[i] = 1.0
        mu.mj_step(mj_model, data)
    print(f"      z={data.qpos[2]:.4f}")
    print()

    # ─── Main loop params ───
    dt = mj_model.opt.timestep
    brain_step = max(1, int(0.025 / dt))
    max_steps = int(duration / dt)

    vnc.reset()
    prev_x = float(data.qpos[0])
    prev_y = float(data.qpos[1])
    prev_time = 0.0

    # ─── Timers ───
    timers = defaultdict(list)
    # Each key stores list of durations in seconds

    print(f"  [5] Профилировка {duration}s...")
    print(f"      brain_step = {brain_step} MuJoCo шагов")
    print(f"      dt = {dt*1000:.2f}ms, total steps = {max_steps}")
    print()

    # Warmup steps
    for _ in range(brain_step * 2):
        mu.mj_step(mj_model, data)

    print(f"  {'Component':>30s} {'calls':>6s} {'total(s)':>10s} {'mean(ms)':>10s} {'median(ms)':>10s} {'max(ms)':>10s} {'%':>6s}")
    print(f"  {'-'*30:>30s} {'-'*6:>6s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s} {'-'*6:>6s}")
    print()

    sim_start = time.perf_counter()

    for step in range(max_steps):
        t = step * dt

        # ─── Brain update every 25ms ───
        if step % brain_step == 0:
            # Sensory
            t0 = time.perf_counter()
            sx, sy = float(data.qpos[0]), float(data.qpos[1])
            dt_sim = brain_step * dt
            svx = (sx - prev_x) / max(dt_sim, 1e-6)
            svy = (sy - prev_y) / max(dt_sim, 1e-6)
            brain.set_sensory_state(sx, sy, svx, svy)
            prev_x, prev_y = sx, sy
            timers['brain.sensory'].append(time.perf_counter() - t0)

            # Injection targets
            t0 = time.perf_counter()
            if vnc._dng100_idx is not None:
                brain._injection_targets[vnc._dng100_idx] = vnc._injection_current
            if hasattr(vnc, '_cb0013_idx') and vnc._cb0013_idx is not None:
                brain._injection_targets[vnc._cb0013_idx] = vnc._injection_current_cb0013
            timers['brain.injection'].append(time.perf_counter() - t0)

            # Full step_batch (sensory input + Rust call + DN/MN extraction + named DN print)
            t0 = time.perf_counter()
            brain.step_batch(t)
            ts = time.perf_counter() - t0
            timers['brain.step_batch'].append(ts)        # ─── Per-step components ───
        # VNC step — каждые 2 шага
        if step % 2 == 0:
            t0 = time.perf_counter()
            vnc.step(dt * 2.0, data)  # dt*2 потому что 2 шага прошло
            timers['vnc.step'].append(time.perf_counter() - t0)
        
        # MuJoCo physics
        t0 = time.perf_counter()
        mu.mj_step(mj_model, data)
        timers['mu.mj_step'].append(time.perf_counter() - t0)

        if t >= duration:
            break

    elapsed = time.perf_counter() - sim_start
    total_sim = t

    # ─── Also measure step_batch internals separately ───
    # Run a few extra brain steps with internal timers
    print("  [6] Детальная профилировка brain.step_batch...")
    internal_iters = 5
    internal_timers = defaultdict(list)
    for _ in range(internal_iters):
        sx, sy = float(data.qpos[0]), float(data.qpos[1])
        dt_sim = brain_step * dt
        svx = (sx - prev_x) / max(dt_sim, 1e-6)
        svy = (sy - prev_y) / max(dt_sim, 1e-6)
        brain.set_sensory_state(sx, sy, svx, svy)
        if vnc._dng100_idx is not None:
            brain._injection_targets[vnc._dng100_idx] = vnc._injection_current

        # Time inside step_batch
        n_steps = max(1, int(25.0 / brain.dt_ms))
        dt_brain = brain.dt_ms

        # _set_sensory_input
        t0 = time.perf_counter()
        brain._set_sensory_input(0.0)
        internal_timers['_set_sensory_input'].append(time.perf_counter() - t0)

        # injection targets + noise
        t0 = time.perf_counter()
        brain._batch_count += 1
        if brain._batch_count % 10 == 1:
            noise = np.random.RandomState(brain._batch_count).randn(brain.n).astype(np.float32) * 0.3
            brain.v[:brain.n] += noise
            np.clip(brain.v[:brain.n], -100.0, 50.0, out=brain.v[:brain.n])

        for csr_idx, current in brain._injection_targets.items():
            if csr_idx < brain.n and current > 0:
                brain.external_I[csr_idx] += current

        brain.I_buf.fill(0.0)
        brain.I_buf[:brain.n] = brain.external_I[:brain.n]
        spike_buf = np.zeros(n_steps * brain.n, dtype=np.int32)
        internal_timers['prep'].append(time.perf_counter() - t0)

        # Rust/GPU call
        t0 = time.perf_counter()
        if brain.use_gpu:
            try:
                taas_module_tcm.lif_connectome.lif_sim_gpu_recording(
                    brain.csr.row_ptr, brain.csr.col_indices, brain.csr.values,
                    n_steps, dt_brain,
                    brain.v, brain.tau_m, brain.v_rest, brain.v_thresh, brain.v_reset,
                    brain.nt_types, brain.external_I, spike_buf,
                )
            except Exception as e:
                pass
        elif brain.use_rust:
            try:
                taas_module_tcm.lif_connectome.lif_sim_cpu_recording(
                    brain.csr.row_ptr, brain.csr.col_indices, brain.csr.values,
                    n_steps, dt_brain,
                    brain.v, brain.tau_m, brain.v_rest, brain.v_thresh, brain.v_reset,
                    brain.nt_types, brain.external_I, brain.I_buf, brain.spikes,
                    spike_buf, brain.g_syn, brain.tau_syn,
                    brain.delay_buf, brain.delay_steps, brain.delay_pos,
                )
            except Exception as e:
                pass
        internal_timers['rust_call'].append(time.perf_counter() - t0)

        # DN/MN extraction via SpikeRecorder
        t0 = time.perf_counter()
        brain._extract_rates_via_recorder(spike_buf, n_steps)
        brain._last_spike_buf = spike_buf
        brain._last_n_steps = n_steps
        internal_timers['extract'].append(time.perf_counter() - t0)

    # ─── Report ───
    print()
    print("═" * 80)
    print("  📊 Результаты профилировки")
    print("═" * 80)
    print(f"  Симуляция: {total_sim:.1f}s за {elapsed:.1f}s (RT={total_sim/elapsed:.2f}×)")
    print(f"  Всего MuJoCo шагов: {len(timers['mu.mj_step'])}")
    print(f"  Всего brain шагов: {len(timers['brain.step_batch'])}")
    print()

    # Group by category
    categories = {
        '🧠 Brain': ['brain.step_batch', 'brain.sensory', 'brain.injection'],
        '🦵 VNC': ['vnc.step'],
        '⚛ MuJoCo': ['mu.mj_step'],
    }

    total_times = {}
    all_total = 0.0

    print(f"  {'Слой':>15s} {'Компонента':>25s} {'вызовов':>8s} {'сумма(s)':>10s} {'средн(ms)':>10s} {'медиана(ms)':>10s} {'макс(ms)':>10s} {'доля':>7s}")
    print(f"  {'-'*15:>15s} {'-'*25:>25s} {'-'*8:>8s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s} {'-'*7:>7s}")

    for category, keys in categories.items():
        for key in keys:
            vals = timers.get(key, [])
            if not vals:
                continue
            arr = np.array(vals)
            total = float(arr.sum())
            mean_ms = float(np.mean(arr)) * 1000
            median_ms = float(np.median(arr)) * 1000
            max_ms = float(np.max(arr)) * 1000
            n_calls = len(arr)
            total_times[key] = total
            all_total += total
            pct = total / elapsed * 100
            print(f"  {category:>15s} {key:>25s} {n_calls:>8d} {total:>10.4f} {mean_ms:>10.4f} {median_ms:>10.4f} {max_ms:>10.4f} {pct:>6.1f}%")

    print()
    print("  ─── Brain step_batch — детальный breakdown ───")
    internal_total = 0.0
    print(f"  {'Компонента':>30s} {'вызовов':>8s} {'сумма(s)':>10s} {'средн(ms)':>10s} {'макс(ms)':>10s}")
    print(f"  {'-'*30:>30s} {'-'*8:>8s} {'-'*10:>10s} {'-'*10:>10s} {'-'*10:>10s}")
    for key in ['_set_sensory_input', 'prep', 'rust_call', 'extract']:
        vals = internal_timers.get(key, [])
        if not vals:
            continue
        arr = np.array(vals)
        total = float(arr.sum())
        mean_ms = float(np.mean(arr)) * 1000
        max_ms = float(np.max(arr)) * 1000
        n_calls = len(arr)
        internal_total += total
        print(f"  {key:>30s} {n_calls:>8d} {total:>10.4f} {mean_ms:>10.4f} {max_ms:>10.4f}")

    # Summary by layer
    print()
    print("  ─── Суммарно по слоям ───")
    layer_totals = {}
    for category, keys in categories.items():
        layer_total = sum(total_times.get(k, 0.0) for k in keys)
        layer_totals[category] = layer_total
        print(f"  {category:>15s}: {layer_total:.4f}s ({layer_total/elapsed*100:.1f}%)")

    # Rest (rendering, poll_keys, etc.)
    accounted = sum(layer_totals.values())
    rest = elapsed - accounted
    if rest > 0.001:
        print(f"  {'Прочее':>15s}: {rest:.4f}s ({rest/elapsed*100:.1f}%) — loop overhead, poll_keys, etc.")

    print()
    print(f"  Итого: {elapsed:.4f}s")
    print()
    print(f"  💡 Если убрать MuJoCo: RT = {total_sim / (elapsed - layer_totals.get('⚛ MuJoCo', 0.0)):.2f}×")
    print()

    # Bottleneck analysis
    print("  ─── Анализ узких мест ───")
    sorted_layers = sorted(layer_totals.items(), key=lambda x: x[1], reverse=True)
    for i, (layer_name, layer_time) in enumerate(sorted_layers):
        pct = layer_time / elapsed * 100
        bar_len = int(pct / 2)
        bar = '█' * bar_len + '░' * (50 - bar_len)
        print(f"  {layer_name:>20s} {bar} {pct:5.1f}% ({layer_time:.4f}s)")

    print()
    print("  ✓ Профилировка завершена!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="📊 Profile V2")
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    if not XML_PATH.exists():
        print(f"  Нет файла: {XML_PATH}"); sys.exit(1)

    profile(duration=args.duration)


if __name__ == "__main__":
    main()
