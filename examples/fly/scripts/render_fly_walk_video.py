#!/usr/bin/env python3
"""
🎥 Render MuJoCo Fly Walk Animation with Atlasov Neuromotor Engine & CPG
========================================================================
Записывает 3D кадры реального движения MuJoCo мухи под управлением
VNC CPG + Atlasov Ca²⁺ мышц (AtlasovFlyMuJoCo).

Включает динамическийрасчет крутящих моментов суставов τ = ΔF - K·q - B·q̇,
благодаря чему лапки отталкиваются от пола и муха физически шагает вперед.
"""

import os
import sys
import math
from pathlib import Path
import numpy as np

# Offscreen rendering backend for MuJoCo
os.environ["MUJOCO_GL"] = "egl"

PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# Import CPG & Atlasov Fly Engine
sys.path.insert(0, str(Path(__file__).parent))
from vnc_cpg_atlasov_embodied import VNCCPG, AtlasovFlyMuJoCo, SCENE_XML

import mujoco as mu
from PIL import Image

def main():
    print("🎥 Рендеринг физической ходьбы мухи (Atlasov Ca²⁺ + VNC CPG)...")
    
    if not SCENE_XML.exists():
        print(f"❌ Файл сцены не найден: {SCENE_XML}")
        sys.exit(1)

    # 1. Инициализация полной симуляции AtlasovFlyMuJoCo
    sim = AtlasovFlyMuJoCo(xml_path=SCENE_XML)
    cpg = VNCCPG(freq_hz=6.0)

    renderer = mu.Renderer(sim.model, height=480, width=640)
    thorax_bid = mu.mj_name2id(sim.model, mu.mjtObj.mjOBJ_BODY, 'thorax')

    # Настройка камеры 3D арены
    cam = mu.MjvCamera()
    cam.type = mu.mjtCamera.mjCAMERA_FREE
    cam.distance = 0.95     # Полный вид мухи на поле
    cam.elevation = -25.0   # 3/4 сверху
    cam.azimuth = 45.0      # Ракурс сбоку-спереди

    frames = []
    n_steps = 3000  # 1.5 сек реального времени (dt = 0.5 ms)
    render_interval = 30  # 100 кадров (~30 FPS)

    print(f"🎬 Симуляция {n_steps} физических шагов с мышцами Атласова...")

    start_pos = sim.data.qpos[:3].copy()

    for step in range(n_steps):
        # 1. CPG шаг
        cpg.step()

        # 2. Собираем углы суставов всех 6 лапок (66 суставов)
        all_angles = []
        for leg_idx in range(cpg.n_legs):
            all_angles.extend(cpg.get_joint_positions(leg_idx))

        # 3. Шаг мышц Атласова + Физика MuJoCo (Atlasov Neuromotor Engine)
        res = sim.step(all_angles)

        # 4. Запись кадра
        if step % render_interval == 0:
            if thorax_bid >= 0:
                cam.lookat = sim.data.xpos[thorax_bid].copy()
            else:
                cam.lookat = sim.data.qpos[:3].copy()

            renderer.update_scene(sim.data, camera=cam)
            rgb_frame = renderer.render()
            frames.append(Image.fromarray(rgb_frame))

    disp_cm = np.linalg.norm(sim.data.qpos[:3] - start_pos)
    disp_mm = disp_cm * 10.0
    print(f"🐾 Дистанция ходьбы мухи в видео: {disp_mm:.2f} мм ({disp_cm:.2f} см)!")

    out_file = Path("/home/taaslab/Проекты/asm_engines/taas_checkmate/examples/fly/fly_walk_cpg.webp")
    artifact_file = Path("/home/taaslab/.gemini/antigravity-ide/brain/01abdf90-8c04-48e6-8d5c-b8df411d43c4/fly_walk_cpg.webp")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Сохранение {len(frames)} кадров в {out_file.name}...")
    frames[0].save(
        out_file,
        save_all=True,
        append_images=frames[1:],
        duration=40,  # 25 FPS
        loop=0
    )
    frames[0].save(
        artifact_file,
        save_all=True,
        append_images=frames[1:],
        duration=40,  # 25 FPS
        loop=0
    )
    print(f"✅ Видео физической ходьбы мухи сохранено: {out_file}")
    print(f"✅ Копия в артефактах: {artifact_file}")

if __name__ == "__main__":
    main()
