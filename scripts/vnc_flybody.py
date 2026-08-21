#!/usr/bin/env python3
"""
🪰 vnc_flybody.py — замкнутая петля: настоящий VNC + тело flybody (MuJoCo).

Петля (пакетами по `batch` VNC-шагов ≈ 50 мс):

    DN-драйв («команда идти») ─────────────────┐
    нагрузка лапок (force_tarsus) → SN-каналы ─┤→ BrainRunner.run(batch, ext)
                                               │   (102K нейронов, 5.2M синапсов,
        │                                     ↑    real веса + знак)
        └── MN-спайки ног за пакет → lift[leg] ─┘
                                               ↓
                                          MuJoCo (flybody)
                                               ↓
                                    force_tarsus (обратная связь)

Замеры: высота тела, силы лапок, частота MN по ногам, корреляция фаз L/R.

Пакетный режим: run() в pillars пересобирает Brain на каждый вызов (~200 мс),
поэтому гоним пакетами — фидбек приходит с задержкой ~batch·dt (50 мс), что
устраивает для походки (период 100–200 мс).

Честные оговорки Phase 1:
  - привязка сенсорных нейронов к ногам в MANC почти отсутствует (2 SN) —
    нагрузка инжектится в фиксированные SN-каналы каждой ноги;
  - ножные MN берутся строго: type=MN* + exitNerve (ProLN/MesoLN/MetaLN);
  - flex-направления актуаторов измерены pose-тестом (coxa/tibia +1 = подъём).

Usage:
    python vnc_flybody.py --seconds 2.0
    python vnc_flybody.py --model haap --seconds 3.0 --dn-drive 30.0
"""

import argparse
import sys

import numpy as np

import pathlib, sys  # репо-рут для tools.taas_io при запуске из scripts/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tools.taas_io import load_taas


def load_taas_verbose(path):
    """load_taas из tools/taas_io.py + информационный print (как раньше)."""
    n, rp, ci, va, nt, rt = load_taas(path)
    print(f"  .taas: n={n} s={len(ci)} receptor_types={'да' if rt is not None else 'нет'}")
    return n, rp, ci, va, nt


# ────────────────────────────────────────────────────────────────────
#  MANC neuron groups (из neuron-properties.feather; порядок = .taas)
# ────────────────────────────────────────────────────────────────────

def load_manc_groups(props_path):
    import pandas as pd

    props = pd.read_feather(props_path)
    n = len(props)
    t = props["type"].astype(object).where(props["type"].notna(), None)

    def is_pref(x, pref):
        return x is not None and str(x).startswith(pref)

    dn = np.nonzero([is_pref(x, "DN") for x in t])[0].astype(np.int64)
    sn = np.nonzero([is_pref(x, "SN") for x in t])[0].astype(np.int64)

    # ножные MN строго: type=MN* + exitNerve (ProLN/MesoLN/MetaLN) + сторона
    def leg_of(en):
        en = str(en)
        if "ProLN" in en:
            return "T1"
        if "MesoLN" in en:
            return "T2"
        if "MetaLN" in en:
            return "T3"
        return None

    mn_groups = {}
    for i, (ty, en) in enumerate(zip(t, props["exitNerve"].values)):
        if not is_pref(ty, "MN"):
            continue
        leg = leg_of(en)
        if leg is None:
            continue
        en = str(en)
        side = "L" if en.endswith("_L") else "R" if en.endswith("_R") else None
        if side is None:
            continue
        mn_groups.setdefault(leg + side, []).append(i)

    print(f"  группы: DN={len(dn)} SN={len(sn)}")
    print(f"  MN-ноги: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(mn_groups.items())))
    return n, dn, sn, mn_groups


# ────────────────────────────────────────────────────────────────────
#  flybody (MuJoCo)
# ────────────────────────────────────────────────────────────────────

LEG_JOINTS = ["coxa_abduct", "coxa_twist", "coxa", "femur_twist",
              "femur", "tibia", "tarsus", "tarsus2"]
LEG_ORDER = ["T1L", "T1R", "T2L", "T2R", "T3L", "T3R"]

# flex-направления из pose-теста: знак ctrl, поднимающий лапку
FLEX_SIGN = {"coxa_abduct": -1.0, "coxa_twist": 0.0, "coxa": 1.0,
             "femur_twist": 0.0, "femur": 0.0, "tibia": 1.0,
             "tarsus": -1.0, "tarsus2": 0.0}
FLEX_WEIGHT = {"coxa_abduct": 0.5, "coxa": 1.0, "tibia": 1.0, "tarsus": 0.3}


def build_flybody(xml_path):
    import mujoco

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    leg_act = {}
    for leg in LEG_ORDER:
        nn = leg[1]
        side = "left" if leg[2] == "L" else "right"
        ids = []
        for jn in LEG_JOINTS:
            nm = f"{jn}_T{nn}_{side}"
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
            assert i >= 0, f"нет актуатора {nm}"
            ids.append(i)
        leg_act[leg] = ids

    force_adr = {}
    for leg in LEG_ORDER:
        nn = leg[1]
        side = "left" if leg[2] == "L" else "right"
        nm = f"force_tarsus_T{nn}_{side}"
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, nm)
        assert i >= 0, f"нет сенсора {nm}"
        force_adr[leg] = int(m.sensor_adr[i])

    return m, d, leg_act, force_adr


def leg_load(d, force_adr):
    """Норма силы (3-вектор) на каждой лапке."""
    out = {}
    for leg, adr in force_adr.items():
        v = d.sensordata[adr:adr + 3]
        out[leg] = float(np.linalg.norm(v))
    return out


def legs_to_ctrl(lift, leg_act, scale=1.0):
    """lift[leg] ∈ [0,1] → ctrl[nu]. Поднятие = flex."""
    ctrl = np.zeros(78, dtype=np.float32)
    for leg in LEG_ORDER:
        l = float(np.clip(lift.get(leg, 0.0), 0.0, 1.0)) * scale
        for jn, idx in zip(LEG_JOINTS, leg_act[leg]):
            s = FLEX_SIGN.get(jn, 0.0)
            w = FLEX_WEIGHT.get(jn, 0.0)
            if s != 0.0:
                ctrl[idx] = s * w * l
    return ctrl


# ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taas", default="resources/data/manc_vnc_signed.taas")
    ap.add_argument("--props",
                    default="D:/connectoms/datasets/v1.0_manc-v1.0-neuron-properties.feather")
    ap.add_argument("--xml",
                    default="C:/Users/anton/Desktop/forks/flybody/flybody/fruitfly/assets/floor.xml")
    ap.add_argument("--model", default="lif")
    ap.add_argument("--dt-ms", type=float, default=0.5)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=100, help="VNC-шагов на пакет (фидбек)")
    ap.add_argument("--dn-drive", type=float, default=40.0)
    ap.add_argument("--load-gain", type=float, default=2.0)
    ap.add_argument("--load-thresh", type=float, default=0.01, help="мин. сила для отклика")
    ap.add_argument("--lift-scale", type=float, default=1.0)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    import mujoco

    # 1. мозг
    print("=== 1. загрузка VNC ===")
    n, rp, ci, va, nt = load_taas_verbose(args.taas)
    n, dn, sn, mn_groups = load_manc_groups(args.props)

    groups = [("DN", dn.tolist()), ("SN", sn.tolist())] + \
             [(k, v) for k, v in sorted(mn_groups.items())]
    # SN-каналы по ногам: фиксированный срез (привязка отсутствует — честно)
    sn_arr = np.asarray(sn, dtype=np.int64)
    sn_chan = {leg: sn_arr[i * 10:(i + 1) * 10].tolist()
               for i, leg in enumerate(LEG_ORDER)}

    try:
        from taas_checkmate.connectome_brain import ConnectomeData, BrainRunner
    except ImportError:
        from taas_checkmate import ConnectomeData, BrainRunner
    conn = ConnectomeData(rp, ci, va, nt, groups)
    brain = BrainRunner(conn, args.model, args.dt_ms)
    print(f"  BrainRunner model={args.model} dt={args.dt_ms}ms")

    # 2. тело
    print("=== 2. загрузка flybody (MuJoCo) ===")
    m, d, leg_act, force_adr = build_flybody(args.xml)
    substeps = max(1, int(round((args.dt_ms / 1000.0) / m.opt.timestep)))
    print(f"  nbody={m.nbody} nu={m.nu} timestep={m.opt.timestep}s, подшагов={substeps}")

    # 3. муха устаивается
    for _ in range(3000):
        mujoco.mj_step(m, d)
    h0 = float(d.qpos[2])
    print(f"  муха встала: z={h0:.4f}")

    # 4. петля
    n_batches = max(1, int(args.seconds * 1000.0 / args.dt_ms / args.batch))
    print(f"=== 3. замкнутая петля: {n_batches} пакетов × {args.batch} шагов "
          f"(≈{args.batch * args.dt_ms:.0f} мс/пакет) ===")

    ext = np.zeros(n, dtype=np.float32)
    dn_idx = np.asarray(dn, dtype=np.int64)

    mn_spikes = {leg: 0 for leg in LEG_ORDER}
    mn_cnt = {leg: len(v) for leg, v in mn_groups.items()}
    height_log = []
    force_log = {leg: [] for leg in LEG_ORDER}
    lift_log = {leg: [] for leg in LEG_ORDER}
    contact_log = {leg: [] for leg in LEG_ORDER}

    for bi in range(n_batches):
        # внешний ток: DN-драйв + нагрузка лапок (с прошлого пакета)
        ext[:] = 0.0
        ext[dn_idx] = args.dn_drive
        for leg in LEG_ORDER:
            f = leg_load(d, force_adr)[leg]
            if f > args.load_thresh:
                ext[sn_chan[leg]] = min(f * args.load_gain, 40.0)

        sp = np.asarray(brain.run(args.batch, ext))
        sp = sp.reshape(args.batch, n)
        batch_spikes = sp.sum(axis=0)

        # MN-активность ног за пакет → lift
        lift = {}
        for leg in LEG_ORDER:
            g = mn_groups[leg]
            if not g:
                lift[leg] = 0.0
                continue
            k = int(batch_spikes[np.asarray(g, dtype=np.int64)].sum())
            mn_spikes[leg] += k
            lift[leg] = min(1.0, k / max(len(g), 1) / (args.batch * 0.02))  # >50Гц → 1
            lift_log[leg].append(lift[leg])

        ctrl = legs_to_ctrl(lift, leg_act, scale=args.lift_scale)
        for _ in range(args.batch * substeps):
            d.ctrl[:] = ctrl
            mujoco.mj_step(m, d)

        height_log.append(float(d.qpos[2]))
        loads = leg_load(d, force_adr)
        for leg in LEG_ORDER:
            force_log[leg].append(loads[leg])
            contact_log[leg].append(1.0 if loads[leg] > args.load_thresh else 0.0)

        if (bi + 1) % 10 == 0 or bi == 0:
            cont = [int(loads[l] > args.load_thresh) for l in LEG_ORDER]
            print(f"  пакет {bi+1}/{n_batches} | z={d.qpos[2]:+.4f} | "
                  f"lift={[round(lift.get(l, 0), 2) for l in LEG_ORDER]} | "
                  f"контакт={cont}")

    # 5. отчёт
    print("\n=== отчёт ===")
    hz = np.array(height_log)
    print(f"высота тела: mean={hz.mean():+.4f} min={hz.min():+.4f} max={hz.max():+.4f} "
          f"(h0={h0:.4f})")
    bio_s = args.seconds
    print("MN по ногам (спайков / Гц на MN / % времени поднята):")
    for leg in LEG_ORDER:
        cnt = mn_spikes[leg]
        hz_per = cnt / max(mn_cnt[leg], 1) / bio_s if mn_cnt[leg] else 0.0
        lifted = np.mean([l > 0.05 for l in lift_log[leg]]) * 100
        print(f"  {leg}: {cnt:6d} спайков | {len(mn_groups[leg])} MN | "
              f"{hz_per:6.1f} Гц/MN | поднята {lifted:.0f}%")
    print("корреляция lift L↔R (антифаза < −0.3):")
    for t in "123":
        a = np.array(lift_log[f"T{t}L"])
        b = np.array(lift_log[f"T{t}R"])
        if len(a) > 2 and a.std() > 1e-6 and b.std() > 1e-6:
            corr = float(np.corrcoef(a, b)[0, 1])
        else:
            corr = 0.0
        print(f"  T{t}: corr={corr:+.3f}")
    # контакт-статистика
    print("контакт лапок (% времени на земле):")
    for leg in LEG_ORDER:
        print(f"  {leg}: {np.mean(contact_log[leg]) * 100:.0f}%")


if __name__ == "__main__":
    main()
