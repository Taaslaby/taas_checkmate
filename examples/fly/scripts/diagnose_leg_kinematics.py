#!/usr/bin/env python3
"""
Deep Diagnostic Analysis of Drosophila Leg Locomotion & Vibration Causes.

Author / Concept: Atlasov Anton (Атласов Антон)
"""

import sys
import math
from pathlib import Path
import numpy as np

PROJ_ROOT = Path(__file__).parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

import mujoco as mu

SCENE_XML = Path(__file__).parent.parent / "models" / "scene.xml"

def diagnose_fly_physics():
    print("=" * 80)
    print("  DEEP PHYSICO-MECHANICAL DIAGNOSTIC ANALYSIS — FLY LOCOMOTION")
    print("  Author / Concept: Atlasov Anton (Атласов Антон)")
    print("=" * 80)

    model = mu.MjModel.from_xml_path(str(SCENE_XML))
    data = mu.MjData(model)

    print(f"\n1. Model Overview:")
    print(f"   • Total DOFs (nv): {model.nv}")
    print(f"   • Total Actuators (nu): {model.nu}")
    print(f"   • Total Geoms: {model.ngeom}")
    print(f"   • Timestep: {model.opt.timestep * 1000:.2f} ms")
    print(f"   • Gravity: {model.opt.gravity}")

    # Inspect leg joints, rest angles, and ranges
    leg_joints = [
        'coxa_T1_left', 'femur_T1_left', 'tibia_T1_left',
        'coxa_T2_left', 'femur_T2_left', 'tibia_T2_left',
        'coxa_T3_left', 'femur_T3_left', 'tibia_T3_left',
    ]

    print(f"\n2. Leg Joint Ranges & Rest Pose Analysis:")
    print(f"{'Joint Name':<20} | {'Joint ID':<10} | {'DOF Addr':<10} | {'Range (rad)':<22} | {'Initial qpos':<15}")
    print("-" * 85)

    for jname in leg_joints:
        jid = mu.mj_name2id(model, mu.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            dof = model.jnt_dofadr[jid]
            qadr = model.jnt_qposadr[jid]
            jrange = model.jnt_range[jid]
            q0 = data.qpos[qadr]
            print(f"{jname:<20} | {jid:<10} | {dof:<10} | [{jrange[0]:+5.2f}, {jrange[1]:+5.2f}]        | {q0:<15.3f}")

    # Inspect Adhesion Claw Actuators
    claw_actuators = [
        'adhere_claw_T1_left', 'adhere_claw_T1_right',
        'adhere_claw_T2_left', 'adhere_claw_T2_right',
        'adhere_claw_T3_left', 'adhere_claw_T3_right',
    ]

    print(f"\n3. Adhesion Claw Actuators Status:")
    for cname in claw_actuators:
        aid = mu.mj_name2id(model, mu.mjtObj.mjOBJ_ACTUATOR, cname)
        print(f"   • {cname:<25} (Actuator ID: {aid})")

    # Run 100 step diagnostic simulation to measure contact forces
    print(f"\n4. Diagnostic Simulation Run (100 steps)...")
    for step in range(100):
        mu.mj_step(model, data)

    print(f"   • Final Body Z Position: {data.qpos[2]*1000:.2f} mm")
    print(f"   • Total Active Contacts: {data.ncon}")

    contacts_summary = []
    for c_i in range(min(10, data.ncon)):
        con = data.contact[c_i]
        g1_name = mu.mj_id2name(model, mu.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom_{con.geom1}"
        g2_name = mu.mj_id2name(model, mu.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom_{con.geom2}"
        contacts_summary.append(f"{g1_name} <-> {g2_name} (dist={con.dist:.4f})")

    for cstr in contacts_summary[:5]:
        print(f"     - Contact: {cstr}")

if __name__ == "__main__":
    diagnose_fly_physics()
