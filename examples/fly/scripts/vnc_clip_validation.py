#!/usr/bin/env python3
"""
🧪 VNC CLIP VALIDATION — Is it walking or force artifact?
=========================================================
Сравниваем VNC HH → Atlasov → MuJoCo с разными clip значениями:

- clip=±0.8  (old — 206 mm/s, "flying")
- clip=±0.05 (10× меньше)
- clip=±0.005 (реалистичный для 1mg мухи — user suggested)
- clip=±0.0005 (сверхреалистичный)

Если муха ИДЁТ при clip=0.005 — CPG реален.
Если останавливается — 206mm/s был артефактом мощности.

Заимствует RealVNCConnectome, FrankenMotorMapper, AtlasovFlyMuJoCo
из vnc_real_atlasov_embodied.py
"""

import os, sys, time
from pathlib import Path
import numpy as np

PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

os.environ["MUJOCO_GL"] = "egl"

# Reuse classes from the main script
sys.path.insert(0, str(Path(__file__).parent))
from vnc_real_atlasov_embodied import (
    RealVNCConnectome, FrankenMotorMapper, AtlasovFlyMuJoCo, 
    AtlasovLegMapper, SCENE_XML
)

import taas_checkmate as tc


def run_with_clip(clip_value, label="test", n_warmup=300, n_walk=500):
    """Run VNC → HH → Atlasov → MuJoCo with a specific clip value."""
    print(f"\n{'='*70}")
    print(f"🎯 TEST: clip=±{clip_value} ({label})")
    print(f"{'='*70}")

    # 1. Load VNC
    conn = RealVNCConnectome(vnc_only=True)
    if not conn.load_neurons(): return None
    if not conn.load_connections(max_synapses=None): return None

    conn_data = conn.build_connectome_data()
    N = conn.n_neurons

    # 2. HH BrainSimulator
    sim = tc.BrainSimulator(N, "hh", pacemaking=True)

    # 3. Atlasov with MODIFIED clip
    fly = AtlasovFlyMuJoCo()
    fly.clip_value = clip_value  # store for step()

    import mujoco as mu

    # Override the step method to use our clip (closure captures fly)
    def clipped_step(motoneuron_spikes):
        n_in = min(len(motoneuron_spikes), fly.n_muscles)
        fly.spikes[:n_in] = motoneuron_spikes[:n_in]
        fly.resets[:n_in] = 0

        fly.step_fn(
            fly.ca, fly.tension, fly.spikes, fly.resets,
            float(fly.dt * 1000.0),
            fly.d_ca, fly.tau_pump, fly.eta_reset, fly.t_max, fly.kd,
            fly.joint_df, fly.joint_stf, fly.joint_dmp,
            fly.k0, fly.beta, fly.b0, fly.alpha
        )

        fly.data.qfrc_applied.fill(0.0)
        for j in range(fly.n_joints):
            dof = fly.all_dofs[j]
            q = fly.data.qpos[dof]
            qd = fly.data.qvel[dof]
            tau = fly.joint_df[j] - fly.joint_stf[j] * q - fly.joint_dmp[j] * qd
            fly.data.qfrc_applied[dof] = np.clip(tau, -fly.clip_value, fly.clip_value)

        mu.mj_step(fly.model, fly.data)
        return {
            't': fly.data.time,
            'pos': fly.data.qpos[:3].copy(),
            'df': fly.joint_df[:fly.n_joints].copy(),
            'k': fly.joint_stf[:fly.n_joints].copy(),
        }

    fly.step = clipped_step  # closure captures fly, no bound-method wrapper!

    # 4. Mapper
    mapper = FrankenMotorMapper(conn, fly)

    # 5. External I
    rng = np.random.RandomState(42)
    I_bg = rng.randn(N).astype(np.float32) * 2.0
    if len(conn.leg_motor) > 0:
        I_bg[np.array(conn.leg_motor) % N] += 30.0
    if len(conn.descending) > 0:
        I_bg[np.array(conn.descending) % N] += 20.0

    n_total = n_warmup + n_walk
    dt_ms = 0.5

    # 6. Batch simulation
    t0 = time.time()
    spike_buf = sim.step_batch(conn_data, n_steps=n_total, dt_ms=dt_ms,
                               external_I=I_bg, scale=1.0)
    t_sim = time.time() - t0
    print(f"   Sim: {n_total} steps in {t_sim:.1f}s")

    # 7. Walk phase
    start_pos = None
    displacement = 0.0
    total_motor = 0
    t_start = time.time()

    # Pre-allocate arrays for trajectory analysis
    x_positions = []
    y_positions = []
    z_positions = []

    for step in range(n_total):
        step_spikes = spike_buf[step * N:(step + 1) * N]
        if step >= n_warmup:
            motor_spikes = np.array([float(step_spikes[idx]) for idx in conn.leg_motor
                                     if idx < N], dtype=np.float32)
            motor_drive = mapper.route(motor_spikes)
            total_motor += motor_drive.sum()
            out = fly.step(motor_drive)

            x_positions.append(out['pos'][0])
            y_positions.append(out['pos'][1])
            z_positions.append(out['pos'][2])

            if start_pos is None:
                start_pos = out['pos'].copy()
            displacement = np.linalg.norm(out['pos'] - start_pos) * 1000

    total_time = time.time() - t_start
    walk_time = n_walk * dt_ms / 1000
    speed = displacement / walk_time if walk_time > 0 else 0.0

    # Gait analysis
    x_pos = np.array(x_positions)
    y_pos = np.array(y_positions)
    z_pos = np.array(z_positions)

    # Detect steps from vertical oscillation
    if len(z_pos) > 10:
        z_std = np.std(z_pos)
    else:
        z_std = 0.0

    # Classify
    if speed < 0.5:
        gait_type = "🟤 STANDING"
    elif speed < 5.0:
        gait_type = "🟡 WALKING"
    elif speed < 20.0:
        gait_type = "🟢 RUNNING"
    elif speed < 100.0:
        gait_type = "🟣 FLYING"
    else:
        gait_type = "🚀 ROCKET"

    result = {
        'clip': clip_value,
        'label': label,
        'displacement_mm': displacement,
        'speed_mm_s': speed,
        'walk_time_s': walk_time,
        'total_motor_firings': total_motor,
        'z_std': z_std,
        'gait_type': gait_type,
        'x_pos': x_pos,
        'y_pos': y_pos,
        'z_pos': z_pos,
        'sim_time': t_sim + total_time,
    }

    print(f"   ✅ {gait_type}: {displacement:.2f}mm in {walk_time:.1f}s = {speed:.2f}mm/s")
    print(f"   Motor firings: {total_motor}, Z-variance: {z_std:.4f}")
    return result


def main():
    print("═" * 70)
    print("🧪 VNC CLIP VALIDATION — Walking or Artifact?")
    print("═" * 70)

    clip_tests = [
        (0.8,   "ORIGINAL — 206 mm/s 'flying'"),
        (0.05,  "10× weaker"),
        (0.005, "REALISTIC for 1mg fly"),
        (0.0005,"Ultra-realistic"),
    ]

    results = []
    for clip, label in clip_tests:
        r = run_with_clip(clip, label)
        if r:
            results.append(r)

    # Results table
    print(f"\n{'='*100}")
    print("📊 VALIDATION RESULTS")
    print(f"{'='*100}")
    print(f"{'Clip':<10} {'Label':<30} {'Distance':<12} {'Speed':<12} {'Motor':<10} {'Z-var':<10} {'Gait':<20}")
    print(f"{'─'*10} {'─'*30} {'─'*12} {'─'*12} {'─'*10} {'─'*10} {'─'*20}")
    for r in results:
        print(f"{'±'+str(r['clip']):<10} {r['label']:<30} {r['displacement_mm']:<10.2f}mm "
              f"{r['speed_mm_s']:<10.2f} {r['total_motor_firings']:<10,} "
              f"{r['z_std']:<10.4f} {r['gait_type']:<20}")

    # Plot trajectory comparison
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(results), 1, figsize=(10, 3*len(results)))
        if len(results) == 1:
            axes = [axes]

        for i, r in enumerate(results):
            ax = axes[i]
            ax.plot(r['x_pos'], r['z_pos'], alpha=0.7, linewidth=0.8)
            ax.set_title(f"clip=±{r['clip']} — {r['gait_type']} ({r['speed_mm_s']:.1f}mm/s)")
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Z (mm)')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

        plt.tight_layout()
        out_path = Path(__file__).parent.parent / "vnc_clip_comparison.png"
        fig.savefig(str(out_path), dpi=150)
        print(f"\n💾 Trajectory plot: {out_path}")
    except Exception as e:
        print(f"\n   ⚠ Plot save error: {e}")

    # Verdict
    print(f"\n{'='*70}")
    print("🔬 SCIENTIFIC VERDICT")
    print(f"{'='*70}")

    if len(results) >= 2:
        clip_high = results[0]  # 0.8
        clip_real = results[2] if len(results) > 2 else results[1]  # 0.005

        ratio = clip_high['speed_mm_s'] / max(0.1, clip_real['speed_mm_s'])
        print(f"\n   Original clip (±0.8): {clip_high['speed_mm_s']:.2f} mm/s ({clip_high['gait_type']})")
        print(f"   Realistic clip (±0.005): {clip_real['speed_mm_s']:.2f} mm/s ({clip_real['gait_type']})")
        print(f"   Speed ratio: {ratio:.1f}×")

        # Determine if walking is real
        if clip_real['speed_mm_s'] > 1.0 and clip_real['gait_type'] in ('WALKING', 'RUNNING'):
            print(f"\n   ✅ CPG IS REAL! Муха ходит при реалистичном clip!")
            print(f"   VNC → HH → Atlasov генерирует настоящую ходьбу")
        elif clip_real['speed_mm_s'] > 0.5:
            print(f"\n   🟡 PARTIAL: Муха движется, но медленно")
            print(f"   Возможно: CPG работает, но сила/масштаб неоптимальны")
        else:
            print(f"\n   ❌ CPG ARTIFACT! Муха стоит при реалистичном clip")
            print(f"   206 mm/s был артефактом избыточной мощности")

    print(f"\n✅ VALIDATION COMPLETE!")


if __name__ == "__main__":
    import mujoco as mu  # needed for the step override
    main()
