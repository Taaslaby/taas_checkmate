#!/usr/bin/env python3
"""Run HH brain with MuJoCo GUI."""
import sys, time, math, os, argparse
sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")))
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='hh')
parser.add_argument('--direct-mn', action='store_true')
parser.parse_args()
from flywire_brain import FlyWireCSR, FlyWireHHBrain
from fly.archive.vnc_controller import V2VNCController, LEGS
import mujoco as mu
import glfw

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "scene.xml")

csr = FlyWireCSR()
csr.load()

mj_model = mu.MjModel.from_xml_path(str(XML_PATH))
data = mu.MjData(mj_model)
mu.mj_resetData(mj_model, data)
data.qpos[2] = 0.03

brain = FlyWireHHBrain(csr)
print("  [Main] Warming up brain (500ms offline)...")
brain.warmup(500.0)
print("  [Main] Warmup done, starting body loop.")

vnc = V2VNCController(mj_model, csr, brain)

# Settle legs
for _ in range(2000):
    for act_idx in vnc.actuator_map.values():
        data.ctrl[act_idx] = -0.15
    for i in vnc.adhere:
        data.ctrl[i] = 1.0
    mu.mj_step(mj_model, data)

# GUI
glfw.init()
window = glfw.create_window(1200, 800, "HH Brain Walk", None, None)
glfw.make_context_current(window)
cam = mu.MjvCamera()
opt = mu.MjvOption()
mu.mjv_defaultCamera(cam)
mu.mjv_defaultOption(opt)
scene = mu.MjvScene(mj_model, maxgeom=10000)
context = mu.MjrContext(mj_model, mu.mjtFontScale.mjFONTSCALE_150)

cam.azimuth = 90
cam.elevation = -45
cam.distance = 1.5

t = 0.0
dt_sim = 0.025
n_steps_brain = 1
_prev_f = False
while not glfw.window_should_close(window):
    glfw.poll_events()
    # F key toggles flight
    f_pressed = glfw.get_key(window, glfw.KEY_F) == glfw.PRESS
    if f_pressed and not _prev_f:
        vnc._flight_enabled = not vnc._flight_enabled
        print(f"  Flight {'ON' if vnc._flight_enabled else 'OFF'}")
    _prev_f = f_pressed
    for _ in range(n_steps_brain):
        brain.food_x = 0.3 + 0.1 * math.cos(t * 0.5)
        brain.food_y = 0.1 * math.sin(t * 0.5)
        brain.step_batch(t)
        vnc.step(dt_sim, data)
        mu.mj_step(mj_model, data)
        t += dt_sim

    viewport = mu.MjrRect(0, 0, 1200, 800)
    mu.mjv_updateScene(mj_model, data, opt, None, cam, mu.mjtCatBit.mjCAT_ALL, scene)
    mu.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)

glfw.terminate()
