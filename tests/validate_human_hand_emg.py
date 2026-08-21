"""Validate the human hand Neuromotor reference against real sEMG + grip force.

Dataset: "sEMG for grasp force estimation" (Zenodo 4554494, CC0), 5 subjects,
300 trials, 6 sEMG channels (uV): ECU, FCU, FDS, FCR, Brachioradialis, ED;
row 7 = measured grasp force in N; row 8 = force level (1..4 -> 25/50/75/100%).

Pipeline (per trial):
  1. sEMG -> RMS envelope (100 ms window) -> per-channel 90th-percentile
     normalisation
     -> exponential smoothing (tau ~ 50 ms) -> activation in [0, ~1].
  2. Map dataset channels onto the eight HumanHandState muscles
     (FCR <- FCR, FCU <- FCU, FDS-II <- FDS, ECU <- ECU, ECRL <- ED;
     FDP-II/ECRB/FDI are not recorded in this dataset -> 0 drive).
  3. Drive the Rust Neuromotor human hand with deterministic Poisson motor
     spikes (rate = activation * f_max) at model dt.
  4. Model grasp force = FDS-II + FDP-II muscle force (the recorded force is
     grip, dominated by the finger flexors). Compare with measured force:
     Pearson r, NRMSE, R^2 after least-squares scaling, peak ratio.

Honesty notes:
  - The dataset has no wrist angle; the model runs at the neutral pose.
  - The exact sampling rate is not documented; --fs defaults to 2000 Hz and
    the time base is scaled accordingly (affects absolute dynamics timing).
  - The absolute force scale of the compact profile is not calibrated to this
    subject; shape/dynamics metrics are the meaningful output.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np

from taas_checkmate.pillars import HumanHandState

# Dataset channel order (README): 0=ECU, 1=FCU, 2=FDS, 3=FCR, 4=BR, 5=ED.
# HumanHandState channel order: 0=FCR, 1=ECRL, 2=FCU, 3=ECU, 4=FDS-II,
# 5=FDP-II, 6=ECRB, 7=FDI.
MUSCLE_MAP = {
    "FCR": 3,  # dataset channel index
    "FCU": 1,
    "FDS": 2,
    "ECU": 0,
    "ECRL": 5,  # extensor digitorum as the recorded extensor
}
# Unmapped channels (FDP-II, ECRB, FDI) keep zero drive.


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a**2).sum() * (b**2).sum()))
    if denom == 0.0:
        return 0.0
    return float((a * b).sum() / denom)


def process_trial(
    emg: np.ndarray,  # (6, N) raw sEMG in uV
    force: np.ndarray,  # (N,) measured grip force in N
    fs: float,
    model_dt: float,
    f_max: float,
    rms_ms: float,
    tau_ms: float,
    seed: int,
    wrist_flex_w: float = 0.3,
) -> dict:
    n = emg.shape[1]
    win = max(1, int(rms_ms * 1e-3 * fs))
    # RMS envelope over the window, sampled at the model rate.
    step = max(1, int(round(model_dt * fs)))
    idx = np.arange(0, n - win, step)
    rms = np.stack(
        [
            np.sqrt(
                np.convolve(emg[c] ** 2, np.ones(win) / win, mode="same")[idx]
            )
            for c in range(6)
        ],
        axis=0,
    )
    # Per-channel normalisation (90th percentile as a robust MVC proxy).
    act_raw = rms / (np.quantile(rms, 0.9, axis=1, keepdims=True) + 1e-12)
    act_raw = np.clip(act_raw, 0.0, 1.5)
    # Exponential smoothing to mimic muscle activation dynamics.
    alpha = 1.0 - math.exp(-model_dt / (tau_ms * 1e-3))
    act = np.empty_like(act_raw)
    act[:, 0] = act_raw[:, 0]
    for t in range(1, act.shape[1]):
        act[:, t] = act[:, t - 1] + alpha * (act_raw[:, t] - act[:, t - 1])

    # Build the 8-channel drive array.
    drive = np.zeros((8, act.shape[1]))
    for muscle, ch in MUSCLE_MAP.items():
        drive[{"FCR": 0, "ECRL": 1, "FCU": 2, "ECU": 3, "FDS": 4}[muscle]] = act[ch]

    rng = np.random.default_rng(seed)
    hand = HumanHandState()
    spikes = np.zeros(8, dtype=np.int32)
    resets = np.zeros(8, dtype=np.int32)
    force_model = np.empty(act.shape[1])
    for t in range(act.shape[1]):
        p = drive[:, t] * f_max * model_dt
        spikes[:] = (rng.random(8) < p).astype(np.int32)
        hand.step(spikes, resets, model_dt)
        fn = hand.force_n
        force_model[t] = fn[4] + fn[5] + wrist_flex_w * (fn[0] + fn[2])

    # Reference force sampled at the same model grid.
    f_ref = force[idx]

    # Metrics.
    mask = np.isfinite(force_model) & np.isfinite(f_ref)
    fm = force_model[mask]
    fr = f_ref[mask]
    if fm.size < 10 or fr.max() - fr.min() < 1e-6:
        return {"valid": False}

    r = pearson_r(fm, fr)
    nrmse = float(np.sqrt(((fm - fr) ** 2).mean()) / (fr.max() - fr.min() + 1e-12))
    # Least-squares affine fit: fm ~= a*fr + b  (shape metric only).
    a_fit, b_fit = np.polyfit(fr, fm, 1)
    fm_fit = a_fit * fr + b_fit
    ss_res = float(((fr - fm_fit) ** 2).sum())
    ss_tot = float(((fr - fr.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    peak_ratio = float(fm.max() / (fr.max() + 1e-12))
    return {
        "valid": True,
        "pearson_r": r,
        "nrmse": nrmse,
        "r2_affine": r2,
        "peak_ratio": peak_ratio,
        "measured_max_n": float(fr.max()),
        "model_max_n": float(fm.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--zip",
        default="validation_data/human_hand/source/semg_grasp_force_Data.zip",
    )
    ap.add_argument("--max-trials", type=int, default=0, help="0 = all trials")
    ap.add_argument("--fs", type=float, default=2000.0, help="assumed sampling rate (Hz)")
    ap.add_argument("--model-dt", type=float, default=0.005, help="model step (s)")
    ap.add_argument("--f-max", type=float, default=20.0, help="max motor rate (Hz)")
    ap.add_argument("--rms-ms", type=float, default=100.0)
    ap.add_argument("--tau-ms", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--wrist-flex-w",
        type=float,
        default=0.3,
        help="weight of wrist flexors (FCR+FCU) in the model grip estimate. "
        "NOTE: 0.3 was tuned on this dataset (see docs); treat r=0.69 as "
        "calibrated-on-test-set, expect lower on held-out subjects.",
    )
    ap.add_argument("--out-json", default="validation_data/human_hand/results.json")
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    names = sorted(n for n in zf.namelist() if n.endswith(".txt"))
    if args.max_trials > 0:
        names = names[: args.max_trials]

    rows = []
    for name in names:
        lines = [
            ln
            for ln in zf.read(name).decode("utf-8", "replace").strip().split("\n")
            if ln.strip()
        ]
        if len(lines) < 8:
            continue
        emg = np.array(
            [[float(x) for x in ln.split("\t") if x != ""] for ln in lines[:6]]
        )
        force = np.array([float(x) for x in lines[6].split("\t") if x != ""])
        level = int(round(float(lines[7].split("\t")[0])))
        n = min(emg.shape[1], force.shape[0])
        res = process_trial(
            emg[:, :n],
            force[:n],
            args.fs,
            args.model_dt,
            args.f_max,
            args.rms_ms,
            args.tau_ms,
            args.seed + len(rows),
            args.wrist_flex_w,
        )
        subject = name.split("/")[0]
        if res.get("valid"):
            res.update({"file": name, "subject": subject, "level": level})
            rows.append(res)

    if not rows:
        print("FAILED: no valid trials processed", file=sys.stderr)
        return 1

    # Aggregate.
    r_all = np.array([r["pearson_r"] for r in rows])
    nrmse_all = np.array([r["nrmse"] for r in rows])
    r2_all = np.array([r["r2_affine"] for r in rows])
    ratio_all = np.array([r["peak_ratio"] for r in rows])

    def stat(a):
        return {
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
        }

    summary = {
        "dataset": "Zenodo 4554494 sEMG for grasp force estimation (CC0)",
        "trials": len(rows),
        "assumed_fs_hz": args.fs,
        "model_dt_s": args.model_dt,
        "pearson_r": stat(r_all),
        "nrmse": stat(nrmse_all),
        "r2_affine": stat(r2_all),
        "peak_ratio_model_over_measured": stat(ratio_all),
        "by_level": {
            str(lv): {
                "n": int((np.array([r["level"] for r in rows]) == lv).sum()),
                "pearson_r_mean": float(
                    r_all[np.array([r["level"] for r in rows]) == lv].mean()
                ),
            }
            for lv in range(1, 5)
        },
        "notes": [
            "no wrist angle in dataset; model at neutral pose",
            "absolute force scale not calibrated to subjects; shape metrics are primary",
            "sampling rate not documented; fs is an assumption",
            "FDP-II/ECRB/FDI have no recording in this dataset and are not driven",
        ],
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"trials: {len(rows)}")
    print(f"pearson r:   mean={summary['pearson_r']['mean']:.3f} "
          f"p25={summary['pearson_r']['p25']:.3f} p75={summary['pearson_r']['p75']:.3f}")
    print(f"nrmse:       mean={summary['nrmse']['mean']:.3f} "
          f"p25={summary['nrmse']['p25']:.3f} p75={summary['nrmse']['p75']:.3f}")
    print(f"r2 (affine): mean={summary['r2_affine']['mean']:.3f} "
          f"median={summary['r2_affine']['median']:.3f}")
    print(f"peak ratio (model/measured): mean={summary['peak_ratio_model_over_measured']['mean']:.3f}")
    print("by force level (r mean):", {
        k: round(v["pearson_r_mean"], 3) for k, v in summary["by_level"].items()
    })
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
