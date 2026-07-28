#!/usr/bin/env python3
"""
Fly Wireless VNC Walk — полный коннектом (мозг + VNC) через LIF.

Ходьба управляется напрямую спайками motor neuron'ов из полного
FlyWire connectome. VNC controller заархивирован (fly/archive/).

Запуск:
  cd taas_checkmate
  python3 fly/scripts/vnc_walk.py

Управление: 1/2=freq  3/4=baseFreq  5/6=turn  ←/→=turn  0=resetTurn  R=resetAll
Зависимости: pip install mujoco numpy glfw
"""

import sys
import time
import math
import argparse
from pathlib import Path
import numpy as np
import glfw

import os


def quat_to_yaw(qw, qx, qy, qz):
    """Extract yaw (heading) from quaternion."""
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(sin_yaw, cos_yaw)

DLOAD = Path(os.environ.get("FLYWIRE_DATA", str(Path.home() / "Загрузки")))
FLY_DIR = Path(__file__).parent.parent
PROJ = FLY_DIR.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))
XML_PATH = FLY_DIR / "models" / "scene.xml"

from fly.archive.vnc_controller import V2VNCController, LEGS


# ═══════════════════════════════════════════════════════════════
#  Клавиши
# ═══════════════════════════════════════════════════════════════

_arrow_state = {'left': False, 'right': False}
_last_keys = {}
def poll_keys(window, vnc):
    keys = {
        '1': glfw.get_key(window,glfw.KEY_1)==glfw.PRESS,
        '2': glfw.get_key(window,glfw.KEY_2)==glfw.PRESS,
        '3': glfw.get_key(window,glfw.KEY_3)==glfw.PRESS,
        '4': glfw.get_key(window,glfw.KEY_4)==glfw.PRESS,
        '5': glfw.get_key(window,glfw.KEY_5)==glfw.PRESS,
        '6': glfw.get_key(window,glfw.KEY_6)==glfw.PRESS,
        '7': glfw.get_key(window,glfw.KEY_7)==glfw.PRESS,
        '8': glfw.get_key(window,glfw.KEY_8)==glfw.PRESS,
        '9': glfw.get_key(window,glfw.KEY_9)==glfw.PRESS,
        '0': glfw.get_key(window,glfw.KEY_0)==glfw.PRESS,
        'R': glfw.get_key(window,glfw.KEY_R)==glfw.PRESS,
    }
    left = glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS
    right = glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS
    if left and not _arrow_state['left']:
        vnc.turn_bias = max(-1.0, vnc.turn_bias - 0.1)
    if right and not _arrow_state['right']:
        vnc.turn_bias = min(1.0, vnc.turn_bias + 0.1)
    _arrow_state['left'] = left
    _arrow_state['right'] = right

    for k,p in keys.items():
        was = _last_keys.get(k,False)
        if p and not was:
            if k=='1': vnc.dn_freq = max(0.5, vnc.dn_freq-0.25)
            elif k=='2': vnc.dn_freq = min(10.0, vnc.dn_freq+0.25)
            elif k=='3': vnc.cpg.base_freq = max(0.5, vnc.cpg.base_freq-0.25)
            elif k=='4': vnc.cpg.base_freq = min(8.0, vnc.cpg.base_freq+0.25)
            elif k=='5': vnc.turn_bias = max(-1.0, vnc.turn_bias-0.1)
            elif k=='6': vnc.turn_bias = min(1.0, vnc.turn_bias+0.1)
            elif k=='7': vnc._cpg_freq = max(1.0, vnc._cpg_freq-1.0)
            elif k=='8': vnc._cpg_freq = min(20.0, vnc._cpg_freq+1.0)
            elif k=='9': vnc._cpg_amp = max(0.05, vnc._cpg_amp-0.05)
            elif k=='0': vnc.turn_bias = 0.0; vnc._cpg_amp = 0.5; vnc._cpg_freq = 8.0
            elif k=='R':
                vnc.dn_freq=2.0; vnc.dn_amp=1.0; vnc.turn_bias=0.0
                vnc._cpg_amp = 0.5; vnc._cpg_freq = 8.0
                vnc.cpg.reset()
        _last_keys[k] = p


# ═══════════════════════════════════════════════════════════════
#  Симуляция
# ═══════════════════════════════════════════════════════════════

def run_simulation(duration=15.0, record=False, looming=True,
                   model='lif', hbar=0.5):
    from flywire_brain import FlyWireCSR, FlyWireLIFBrain
    import mujoco as mu

    print(f"  Модель: {XML_PATH}")
    mj_model = mu.MjModel.from_xml_path(str(XML_PATH))
    data = mu.MjData(mj_model)
    mu.mj_resetData(mj_model, data)
    data.qpos[2] = 0.03

    # ─── CSR ───
    print()
    print("  ═══ Full brain+VNC connectome ═══")
    csr = FlyWireCSR()
    csr.load()
    if not csr.is_v2:
        print("  ⚠ V2 dataset required!")
        return

    # ─── Brain ───
    if model == 'lif':
        brain = FlyWireLIFBrain(csr)
    else:
        from flywire_brain import FlyWireBrain, FlyWirePsiBrain, FlyWireRateBrain
        brain_map = {'rate': FlyWireRateBrain, 'psi': FlyWirePsiBrain, 'izhi': FlyWireBrain}
        brain_cls = brain_map.get(model, FlyWireLIFBrain)
        kwargs = {'hbar': hbar} if model == 'psi' else {}
        brain = brain_cls(csr, **kwargs)

    backend_str = 'GPU CUDA' if getattr(brain, 'use_gpu', False) else 'Rust CPU'
    print(f"\n  [V2] {model.upper()} brain: {csr.n_neurons:,} нейронов, "
          f"{len(csr.dn_indices)} DN, {len(csr.mn_indices)} MN")
    print(f"  [V2] Backend: {backend_str}")

    # ─── Food tracking ───
    food_geom_id = mu.mj_name2id(mj_model, mu.mjtObj.mjOBJ_GEOM, 'food')
    tracking_food = food_geom_id >= 0
    if tracking_food:
        print(f"  [Food] 🍎 Target at initial pos")

    # ─── VNC (archived controller — для роботов) ───
    print()
    vnc = V2VNCController(mj_model, csr, brain)

    # GUI
    if not glfw.init():
        print("  glfw.init() failed"); return
    window = glfw.create_window(1200, 800, "🧠 Fly VNC Walk", None, None)
    if not window: glfw.terminate(); return
    glfw.make_context_current(window)
    glfw.swap_interval(0)  # Нет v-sync блокировки — simulation loop не зависит от FPS монитора

    scene = mu.MjvScene(mj_model, maxgeom=10000)
    cam = mu.MjvCamera()
    # Tracking camera — follows thorax
    thorax_id = mu.mj_name2id(mj_model, mu.mjtObj.mjOBJ_BODY, 'thorax')
    cam.type = mu.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = thorax_id
    cam.distance = 0.35
    cam.azimuth = 135
    cam.elevation = -25
    opt=mu.MjvOption(); pert=mu.MjvPerturb()
    ctx=mu.MjrContext(mj_model, mu.mjtFontScale.mjFONTSCALE_150)

    # Settle
    data.qpos[2] = 0.03
    vnc.standby(data)
    print("\n  Settle (0.2s)...")
    _t0_settle = time.time()
    for _ in range(2000):
        for act_idx in vnc.actuator_map.values():
            data.ctrl[act_idx] = -0.15
        for i in vnc.adhere:
            data.ctrl[i] = 1.0
        mu.mj_step(mj_model, data)
    _t_settle = time.time() - _t0_settle
    print(f"  z={data.qpos[2]:.4f} contacts={data.ncon}  ({_t_settle:.2f}s)")

    # ─── Тайминги ───
    _t_brain = 0.0
    _t_render = 0.0
    _t_vnc = 0.0
    _t_mj = 0.0
    _last_render = 0.0  # wall time of last render (for FPS cap)

    prev_x, prev_y = float(data.qpos[0]), float(data.qpos[1])
    prev_time = 0.0

    vnc.reset()
    print(f"\n  VNC ходьба ({duration:.0f}s)")
    print("  1/2=DNfreq  3/4=baseFreq  5/6=turn  7/8=CPGfreq 1-20Hz  9=CPGamp  ←/→=turn  0=reset  R=resetAll")
    print(f"  Backend: {'⚡GPU CUDA' if getattr(brain, 'use_gpu', False) else '💻Rust CPU'}")
    if looming and brain is not None and hasattr(brain, 'looming') and brain.looming_enabled:
        print(f"  [Vision looming] walls at ±{brain.looming.x_max:.2f}m, scale={brain.looming.scale:.2f}m")

    sim_t = 0.0; steps = 0; frames = 0; t0 = time.time()
    rec_frames = [] if record else None
    brain_step_interval = max(1, int(0.025 / mj_model.opt.timestep))
    # Render at most every ~33ms wall time (~30fps)
    _render_ms = 33  # ms between renders

    while not glfw.window_should_close(window):
        dt = mj_model.opt.timestep

        # Brain update каждые ~25ms
        if brain is not None and steps % brain_step_interval == 0:
            # Fly position + velocity
            sx, sy = float(data.qpos[0]), float(data.qpos[1])
            dt_sim = sim_t - prev_time if sim_t > prev_time else mj_model.opt.timestep
            svx = (sx - prev_x) / dt_sim if dt_sim > 0 else 0.0
            svy = (sy - prev_y) / dt_sim if dt_sim > 0 else 0.0
            prev_x, prev_y, prev_time = sx, sy, sim_t

            # Food position + relative bearing
            food_x = food_y = None
            if tracking_food:
                food_x = float(data.geom_xpos[food_geom_id][0])
                food_y = float(data.geom_xpos[food_geom_id][1])

            brain.set_sensory_state(sx, sy, svx, svy, food_x=food_x, food_y=food_y)

            # Override: relative bearing (set_sensory_state computes world-frame, need fly-relative)
            if tracking_food and food_x is not None:
                fdx = food_x - sx
                fdy = food_y - sy
                food_dist = math.sqrt(fdx*fdx + fdy*fdy)
                if food_dist > 0.001:
                    world_bearing = math.atan2(fdy, fdx)
                    qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
                    fly_yaw = quat_to_yaw(qw, qx, qy, qz)
                    rel_bearing = world_bearing - fly_yaw
                    while rel_bearing > math.pi: rel_bearing -= 2.0 * math.pi
                    while rel_bearing < -math.pi: rel_bearing += 2.0 * math.pi
                    brain.food_bearing = rel_bearing
                    brain.food_distance = food_dist
            # DNg100 + CB0013 injection (modulated by looming!)
            if hasattr(vnc, '_dng100_idx') and vnc._dng100_idx is not None:
                # ─── Natural stop: looming → DNg100 suppression ───
                # В реальной мухе: visual input → central complex → GABAergic → DNg100 inhibition
                # Симулируем: чем ближе стена, тем слабее DNg100 drive
                drive_mod = 1.0
                if hasattr(brain, 'looming') and brain.looming.looming > 0.05:
                    drive_mod = max(0.0, 1.0 - brain.looming.looming * 1.8)
                brain._injection_targets[vnc._dng100_idx] = vnc._injection_current * drive_mod
                brain._dng_mod = drive_mod  # for HUD
            if hasattr(vnc, '_cb0013_idx') and vnc._cb0013_idx is not None:
                brain._injection_targets[vnc._cb0013_idx] = vnc._injection_current_cb0013

            _tb = time.time()
            brain.step_batch(sim_t)
            _t_brain += time.time() - _tb

        # VNC step — каждые 2 MuJoCo шага
        if steps % 2 == 0:
            _tv = time.time()
            vnc.step(dt * 2.0, data)
            _t_vnc += time.time() - _tv

        _tm = time.time()
        mu.mj_step(mj_model, data)
        _t_mj += time.time() - _tm
        sim_t += dt; steps += 1
        poll_keys(window, vnc)

        frames += 1
        now = time.time()
        if (now - _last_render) * 1000.0 >= _render_ms:
            _last_render = now
            _tr = time.time()
            mu.mjv_updateScene(mj_model,data,opt,pert,cam,mu.mjtCatBit.mjCAT_ALL,scene)
            vp = mu.MjrRect(0,0,1200,800)
            mu.mjr_render(vp, scene, ctx)

            act_dn = len(vnc.csr.dn_indices)
            act_mn = int(vnc._leg_act.sum())
            leg_strs = []
            for leg_i, (label, lname) in enumerate(LEGS):
                act = vnc._leg_act[leg_i]
                bar = '█' * min(10, int(act * 10))
                leg_strs.append(f"{label} {bar} {act:.2f}")
            contact_line = '  '.join(leg_strs)

            looming_line = ''
            if brain is not None and hasattr(brain, 'looming') and looming:
                l = brain.looming
                if l.looming > 0.05:
                    bars = '░' * int(l.looming * 10) + '▓' * int((1-l.looming) * 10)
                    looming_line = f"\n⚠ LOOMING {l.looming:.2f}  [{bars}]  d={l.nearest_wall_dist:.3f}m {l.nearest_wall_name}"

            bal = getattr(vnc, '_display_gains', np.ones(6))
            bal_strs = [f"{label[0]}={bal[i]:.1f}" for i, (label, _) in enumerate(LEGS)]

            # Food info
            food_line = ''
            if tracking_food and hasattr(brain, 'food_bearing'):
                bearing_deg = math.degrees(brain.food_bearing)
                arrow = '↑' if abs(bearing_deg) < 30 else ('↗' if bearing_deg < 0 else '↖')
                food_line = f"\n🍎 {arrow} {bearing_deg:.0f}°  d={brain.food_distance:.3f}m"

            # DNg100 drive level
            dng_drive = vnc._injection_current * getattr(brain, '_dng_mod', 1.0)
            walk_status = '🏃' if dng_drive > 20 else ('🚶' if dng_drive > 5 else '🧘')

            gpu_tag = '⚡GPU' if getattr(brain, 'use_gpu', False) else '💻CPU'
            cpg_info = f'CPG {vnc._cpg_freq:.1f}Hz ×{vnc._cpg_amp:.2f}'
            title_line = f"{walk_status} V2 t={sim_t:.1f}s  {gpu_tag}  {cpg_info}  DNg={dng_drive:.0f}pA"
            info_line = f"DNs={act_dn}  MNs={act_mn}  x={data.qpos[0]:.2f}  y={data.qpos[1]:.2f}  z={data.qpos[2]:.3f}\n"
            info_line += f"Bal: {'  '.join(bal_strs)}\n"
            info_line += f"Legs: {contact_line}{looming_line}{food_line}"

            mu.mjr_overlay(mu.mjtFontScale.mjFONTSCALE_150,
                mu.mjtGridPos.mjGRID_TOPLEFT, vp,
                title_line, info_line, ctx)
            glfw.swap_buffers(window)
            _t_render += time.time() - _tr

        glfw.poll_events()
        if record and frames%2==0 and rec_frames is not None:
            try:
                arr=np.zeros((800,1200,3),dtype=np.uint8)
                mu.mjr_readPixels(arr,None,mu.MjrRect(0,0,1200,800),ctx)
                rec_frames.append(arr[::-1].copy())
            except Exception:
                pass
        if sim_t >= duration: break

    elapsed = time.time() - t0
    final_x = data.qpos[0]
    print(f"\n  {sim_t:.1f}s за {elapsed:.1f}s (RT={sim_t/elapsed:.2f}×)")
    print(f"  Тайминги: settle={_t_settle:.2f}s  brain={_t_brain:.2f}s  vnc={_t_vnc:.3f}s  mj={_t_mj:.3f}s  render={_t_render:.3f}s")
    print(f"  Смещение: x={final_x:.3f}  y={data.qpos[1]:.3f}")
    glfw.destroy_window(window); glfw.terminate()
    if record:
        _save_video(rec_frames, "fly_vnc_walk.mp4")
    print("  ✓ Готово!")


def _save_video(frames, path):
    if not frames:
        return
    try:
        import cv2
        h,w,_=frames[0].shape
        out=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'),30.0,(w,h))
        for f in frames: out.write(cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
        out.release()
        print(f"  Видео: {path}")
    except Exception as e:
        print(f"  Видео не сохранено: {e}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="🧠 Fly VNC Walk — neural V2 walker")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--model", type=str, choices=['izhi', 'psi', 'rate', 'lif'], default='lif',
        help="Модель нейронов: lif (LIF, default), izhi (Izhikevich), psi (ψ-wave), rate (ANN)")
    parser.add_argument("--hbar", type=float, default=0.5,
        help="ħ для ψ-wave")
    args = parser.parse_args()

    print("═"*55)
    print("  🧠 Fly VNC — Full Brain+Cord Connectome (V2)")
    print("═"*55)

    if not XML_PATH.exists():
        print(f"  Нет файла: {XML_PATH}"); sys.exit(1)
    run_simulation(duration=args.duration, record=args.record,
                   model=args.model, hbar=args.hbar)

if __name__ == "__main__":
    main()
