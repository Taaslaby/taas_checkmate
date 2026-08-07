"""See-and-fix connectome generation.

The generator produces a connectome, we "look at it" (render to PNG + run it
through the Rust engine and graph metrics), the critic flags what is wrong, and
we regenerate (different seed / nudge) until the connectome is healthy or we hit
max_tries. Returns the best graph + its image + a critique report.

This is the connectome analogue of image-gen models that see their own photos
and revise bad ones: the "eyes" here are the engine dynamics + topology metrics
(and the rendered PNG is the human/VLM-facing view).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .env import ConnectomeEnv, reward_from, profile_for
from .metrics import graph_metrics


def critique(rw: dict, gm: dict, cat: str) -> dict:
    """Flag each health criterion. Returns {flag: (ok: bool, msg, severity)}."""
    p = profile_for(cat)
    dens_t = p["struct"]["density"]
    ei_t = p["struct"]["ei_ratio"]
    d = gm["mean_deg"]
    healthy_deg_lo, healthy_deg_hi = max(5.0, 20 * dens_t), 50.0
    flags = {
        "propagation": (rw["prop"] >= 0.7, f"prop={rw['prop']:.2f} (<0.7)", "major"),
        "alive": (rw["dead_frac"] <= 0.6, f"dead={rw['dead_frac']:.2f} (>0.6)", "major"),
        "ei_balance": (ei_t * 0.55 <= gm["ei_ratio"] <= ei_t * 1.7,
                       f"ei_ratio={gm['ei_ratio']:.1f} (target {ei_t})", "major"),
        "density": (healthy_deg_lo <= d <= healthy_deg_hi,
                    f"mean_deg={d:.1f} (target ~{dens_t:.2f})", "minor"),
        "diversity": (rw["diversity"] >= 0.4, f"diversity={rw['diversity']:.2f} (<0.4)", "minor"),
    }
    return flags


def generate_fixed(
    spec: str,
    cat: str,
    policy,
    env: ConnectomeEnv,
    max_tries: int = 10,
    out_dir: str = "output/connectomes",
    n_neurons: int = 128,
) -> dict:
    """See-and-fix loop. Returns dict with best csr/metrics/image/report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(ch if ch.isalnum() else "_" for ch in spec.lower())[:40].strip("_")
    if not slug:
        slug = cat

    best = None
    history = []
    p = profile_for(cat)
    ei_t = p["struct"]["ei_ratio"]
    for t in range(max_tries):
        seed = (sum(spec.encode("utf-8")) + t * 7919) & 0x7FFFFFFF
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        # try 0 = pure greedy (the model's natural best, deterministic). Later
        # tries nudge the edge top-k so each retry explores a different wiring at
        # the SAME density (the seed drives the nudge RNG), and enforce the
        # category's E/I target — the two corrections the critic can apply.
        nudge = 0.0 if t == 0 else min(2.0, 0.35 * t)
        with torch.no_grad():
            out = policy([spec], greedy=True, nudge=nudge, ei_target=ei_t)
            csrs = policy.to_csr(out)
        csr = csrs[0]
        dyn = env.simulate(csr)
        rw = reward_from(csr, dyn, cat=cat)
        gm = graph_metrics(csr)
        flags = critique(rw, gm, cat)
        n_bad = sum(1 for ok, _, _ in flags.values() if not ok)
        n_major = sum(1 for ok, _, sev in flags.values() if not ok and sev == "major")
        health = "OK" if n_bad == 0 else ("RETRY" if n_major else "minor")
        history.append({
            "try": t, "seed": seed, "nudge": nudge, "reward": rw["total"], "health": health,
            "prop": rw["prop"], "dead": rw["dead_frac"], "dv": rw["dv"],
            "ei_ratio": gm["ei_ratio"], "mean_deg": gm["mean_deg"], "n_bad": n_bad,
        })
        if best is None or rw["total"] > best["reward"]:
            best = {"csr": csr, "reward": rw["total"], "flags": flags,
                    "rw": rw, "gm": gm, "dyn": dyn}
        if health == "OK":
            break

    # render the best
    img_path = out_dir / f"{cat}_{slug}_best.png"
    title = f"{cat} | spec: {spec[:48]} | R={best['reward']:.2f}"
    from .render import render as render_fn

    render_fn(best["csr"], title=title, out_path=str(img_path))

    return {
        "cat": cat,
        "spec": spec,
        "csr": best["csr"],
        "reward": best["reward"],
        "rw": best["rw"],
        "gm": best["gm"],
        "flags": best["flags"],
        "image": str(img_path),
        "history": history,
        "tries": len(history),
    }
