#!/usr/bin/env python3
"""Validate Atlasov against an external real-data recording.

Accepted input formats:

* CSV with columns ``time,spike,length,velocity`` and optional ``force``;
* NPZ with arrays using the same names.

The recording is deliberately not bundled here: a real-data claim requires a
measured file and an explicit calibration.  Without ``force`` this script
checks the model's physical invariants and reports that force calibration was
not evaluated.  With ``force`` it reports RMSE/NRMSE/correlation; pass
``--max-nrmse`` when a calibrated acceptance criterion is available.

Example:
    python tests/validate_atlasov_real_data.py recording.csv \
        --t-max 2e-6 --l0 3e-4 --v-max 0.5 --max-nrmse 0.25

The units are not guessed. ``length``, ``velocity`` and ``force`` must use the
same units as the explicitly supplied Atlasov parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REQUIRED = ("time", "spike", "length", "velocity")


def load_recording(path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            result = {name: np.asarray(data[name], dtype=np.float64) for name in data.files}
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or not rows[0]:
            raise ValueError("CSV is empty or has no header")
        result = {}
        for name in rows[0]:
            try:
                result[name] = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"CSV column {name!r} is not numeric") from exc
    else:
        raise ValueError("input must be .csv or .npz")

    missing = [name for name in REQUIRED if name not in result]
    if missing:
        raise ValueError(f"missing required columns/arrays: {', '.join(missing)}")
    lengths = {name: len(result[name]) for name in REQUIRED if result[name].ndim == 1}
    if len(lengths) != len(REQUIRED) or len(set(lengths.values())) != 1:
        raise ValueError(f"required arrays must be one-dimensional and equal length: {lengths}")
    if len(result["time"]) < 2:
        raise ValueError("recording must contain at least two samples")
    for name, values in result.items():
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name!r} must be a finite one-dimensional array")
    if np.any(np.diff(result["time"]) <= 0):
        raise ValueError("time must be strictly increasing")
    if not np.all(np.isin(result["spike"], (0.0, 1.0))):
        raise ValueError("spike must contain only binary 0/1 values")
    if "force" in result and len(result["force"]) != len(result["time"]):
        raise ValueError("force must have the same number of samples as time")
    return result


def simulate(data: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, object]:
    try:
        from taas_checkmate import pillars
    except ImportError as exc:
        raise RuntimeError("install the Python extension before running this validator") from exc

    positive_parameters = {
        "d_ca": args.d_ca,
        "tau_pump": args.tau_pump,
        "t_max": args.t_max,
        "kd": args.kd,
        "l0": args.l0,
        "v_max": args.v_max,
        "width": args.width,
    }
    for name, value in positive_parameters.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {value}")
    if not np.isfinite(args.eta_reset) or not 0 <= args.eta_reset <= 1:
        raise ValueError(f"eta_reset must be finite and in [0, 1], got {args.eta_reset}")

    n = len(data["time"])
    dt_samples = np.diff(data["time"])
    dt = args.dt if args.dt is not None else float(np.median(dt_samples))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and > 0")
    relative_dt_error = float(np.max(np.abs(dt_samples - dt)) / dt)

    pop = pillars.AtlasovNeuromotorPopulation(1)
    pop.dt = float(dt)
    pop.d_ca = float(args.d_ca)
    pop.ca_decay_const = float(np.exp(-dt / args.tau_pump))
    pop.eta_reset = float(args.eta_reset)
    pop.t_max = float(args.t_max)
    pop.kd = float(args.kd)
    pop.l0 = float(args.l0)
    pop.inv_vmax = float(1.0 / args.v_max)
    pop.inv_width = float(1.0 / (args.width * args.l0))
    pop.set_state_array(0, np.array([0.0], dtype=np.float32))
    pop.set_state_array(2, np.array([0.0], dtype=np.float32))

    tension = np.empty(n, dtype=np.float64)
    calcium = np.empty(n, dtype=np.float64)
    out = np.zeros(1, dtype=np.int32)
    for i in range(n):
        pop.set_state_array(3, np.array([data["length"][i]], dtype=np.float32))
        pop.set_state_array(4, np.array([data["velocity"][i]], dtype=np.float32))
        current = np.array([data["spike"][i]], dtype=np.float32)
        pop.step(current, out)
        tension[i] = float(pop.tension[0])
        calcium[i] = float(pop.ca[0])

    if not np.all(np.isfinite(tension)) or not np.all(np.isfinite(calcium)):
        raise AssertionError("model emitted NaN/Inf")
    if np.min(calcium) < -1e-7:
        raise AssertionError(f"calcium became negative: {np.min(calcium)}")
    if np.min(tension) < -1e-7:
        raise AssertionError(f"tension became negative: {np.min(tension)}")
    max_allowed = 1.5 * args.t_max + 1e-7
    if np.max(tension) > max_allowed:
        raise AssertionError(f"tension exceeded eccentric plateau: {np.max(tension)} > {max_allowed}")

    result: dict[str, object] = {
        "samples": n,
        "dt": dt,
        "max_relative_dt_jitter": relative_dt_error,
        "spikes": int(np.count_nonzero(data["spike"] > 0)),
        "calcium_min": float(np.min(calcium)),
        "calcium_max": float(np.max(calcium)),
        "tension_min": float(np.min(tension)),
        "tension_max": float(np.max(tension)),
        "force_ground_truth": "available" if "force" in data else "missing",
    }

    if "force" in data:
        measured = data["force"]
        residual = tension - measured
        scale = max(float(np.ptp(measured)), float(np.max(np.abs(measured))), 1e-12)
        rmse = float(np.sqrt(np.mean(residual * residual)))
        result["force_rmse"] = rmse
        result["force_nrmse"] = rmse / scale
        if np.std(measured) > 0 and np.std(tension) > 0:
            result["force_correlation"] = float(np.corrcoef(tension, measured)[0, 1])
        else:
            result["force_correlation"] = None
        if args.max_nrmse is not None and result["force_nrmse"] > args.max_nrmse:
            raise AssertionError(
                f"force NRMSE {result['force_nrmse']:.6g} exceeds calibrated limit {args.max_nrmse:.6g}"
            )
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recording", type=Path)
    p.add_argument("--dt", type=float, help="model timestep; default: median time difference")
    p.add_argument("--d-ca", type=float, default=0.3)
    p.add_argument("--tau-pump", type=float, default=10.0)
    p.add_argument("--eta-reset", type=float, default=0.9)
    p.add_argument("--t-max", type=float, default=2e-6)
    p.add_argument("--kd", type=float, default=0.2)
    p.add_argument("--l0", type=float, default=3e-4)
    p.add_argument("--v-max", type=float, default=0.5)
    p.add_argument("--width", type=float, default=0.5, help="length-tension width as a multiple of l0")
    p.add_argument("--max-nrmse", type=float, help="fail when measured-force NRMSE exceeds this calibrated limit")
    p.add_argument("--json", action="store_true", dest="as_json")
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        result = simulate(load_recording(args.recording), args)
    except (AssertionError, ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Atlasov validation FAILED: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("Atlasov validation PASSED")
        for key, value in result.items():
            print(f"  {key}: {value}")
        if result["force_ground_truth"] == "missing":
            print("  NOTE: no measured force supplied; absolute force accuracy was not evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
