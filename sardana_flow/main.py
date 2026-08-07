#!/usr/bin/env python3
"""sardana_flow CLI.

    python -m sardana_flow.main train --steps 50 --n 128
    python -m sardana_flow.main probe --steps 10
    python -m sardana_flow.main generate --spec "motion detection for a drone"

Trains the policy with the Rust engine as the reward source.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import specs as spec_mod
from .env import ConnectomeEnv
from .generator import ConnectomePolicy
from .trainer import train_step


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate(policy, env, device, n: int) -> dict:
    """Greedy validation over one spec per category — anti-drift selection.

    Returns {'R', 'behavioral', 'osc', 'persist', 'stab'} (greedy means) plus
    per_cat detail. `behavioral` is the alive-gated per-category behavioural
    credit (env's "+behavioral" reward block) — the honest measure of whether
    per-task behaviour emerged; it is what the best-checkpoint selector uses.
    """
    from .env import reward_from

    policy.eval()
    specs = [(c, spec_mod.SPECS[c][0]) for c in spec_mod.SPECS]
    with torch.no_grad():
        out = policy([s for _, s in specs], greedy=True)
        csrs = policy.to_csr(out)
    tot, oscs, perrs, stabs, behs = 0.0, [], [], [], []
    per = {}
    for i, (c, _) in enumerate(specs):
        rw = reward_from(csrs[i], env.simulate(csrs[i]), cat=c)
        tot += rw["total"]
        behs.append(rw.get("behavioral", 0.0))
        oscs.append(rw["osc"]); perrs.append(rw["persist"]); stabs.append(rw["stab"])
        per[c] = {"R": rw["total"], "behavioral": rw.get("behavioral", 0.0),
                  "osc": rw["osc"], "persist": rw["persist"],
                  "stab": rw["stab"], "alive": rw.get("alive", 0.0), "dv": rw.get("dv", 0.0)}
    policy.train()
    ncat = max(1, len(specs))
    return {"R": float(tot / ncat), "behavioral": float(np.mean(behs)),
            "osc": float(np.mean(oscs)),
            "persist": float(np.mean(perrs)), "stab": float(np.mean(stabs)), "per_cat": per}


def cmd_train(args: argparse.Namespace):
    # fully reproducible run — training previously had NO global seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _device()
    policy = ConnectomePolicy(n_neurons=args.n).to(device)
    # actor and critic learn at (possibly) different rates
    main_p = [p for k, p in policy.named_parameters() if not k.startswith("value_head")]
    crit_p = [p for k, p in policy.named_parameters() if k.startswith("value_head")]
    opt = torch.optim.Adam(
        [{"params": main_p, "lr": args.lr}, {"params": crit_p, "lr": args.critic_lr}]
    )
    if args.resume:
        sd = torch.load(args.resume, map_location=device)
        policy.load_state_dict(sd, strict=False)
        print(f"[resume] loaded {args.resume} (missing value_head keys are fine)")
    env = ConnectomeEnv(
        n_steps=args.sim_steps, drive=args.drive, gain=args.gain,
        onoff=args.onoff, inrate=args.inrate,
    )
    baseline: dict[str, float] = {}

    all_specs = spec_mod.all_specs()
    # deterministic cycling so every step sees all categories
    t0 = time.time()
    best_val = -1.0        # best greedy *total* R (reference)
    best_state = None
    best_beh = -1.0        # best per-category *behavioural* credit (the one we save)
    best_beh_state = None

    # metrics log — one csv per run, so behaviour growth is visible over hours
    mf = Path(args.metrics)
    mf.parent.mkdir(parents=True, exist_ok=True)
    mcols = ["step", "loss", "pg", "crit", "struct", "entropy", "R",
             "val_R", "val_beh", "val_osc", "val_persist", "val_stab"]
    with open(mf, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=mcols)
        w.writeheader()
        cats = spec_mod.ALL_CATEGORIES
        n_cat = len(cats)
        # every step trains on ALL categories (B = n_cat, one spec each): the
        # per-batch advantage standardisation then CONTRASTS oscillatory vs
        # memory/homeо cats within one step (osc cats hit their target → +adv,
        # persist/stab cats still failing → −adv), giving the direct pressure to
        # specialise that the rotating B=4 slice never had. k_step cycles through
        # each cat's spec list so text variety still grows over a long run.
        for step in range(1, args.steps + 1):
            k_step = (step - 1) // n_cat
            batch = []
            for k, c in enumerate(cats):
                sl = spec_mod.SPECS[c]
                batch.append((c, sl[(k_step + k) % len(sl)]))
            info = train_step(
                policy, batch, env, opt,
                entropy_coef=args.entropy, n_rollouts=args.rollouts,
                struct_coef=args.struct_coef,
                baseline=baseline, device=device,
                clip_eps=args.clip_eps, crit_coef=args.crit_coef,
            )
            # gentle LR decay — the last-100-steps entropy/struct drift (which
            # made v10 denser than target) came from training at full LR to the end
            for g in opt.param_groups:
                g["lr"] *= args.lr_decay
            row = {"step": step, "loss": info["loss"], "pg": info["pg"],
                   "crit": info["crit"], "struct": info["struct"],
                   "entropy": info["entropy"], "R": info["mean_reward"]}
            if step == 1 or step % args.log_every == 0:
                el = time.time() - t0
                bl = " ".join(f"{k[:5]}:{v:.2f}" for k, v in sorted(baseline.items()))
                print(
                    f"[step {step:4d}] loss={info['loss']:.3f} pg={info['pg']:.3f} "
                    f"crit={info['crit']:.3f} struct={info['struct']:.3f} "
                    f"ent={info['entropy']:.3f} R={info['mean_reward']:.3f} "
                    f"| base {bl} | {el:.0f}s",
                    flush=True,
                )
            # anti-drift: keep the best greedy-validation checkpoint, not the last.
            # Selection is by per-category *behavioural* credit (`behavioral` =
            # the alive-gated per-cat osc/persist/stab block) — that IS the goal
            # (motor oscillates / memory persists / homeо is stable), and it is
            # what a universal-oscillator collapse scores badly on. Total R is
            # tracked for reference only: it favours one shared regime.
            if step % args.val_every == 0:
                v = _validate(policy, env, device, args.n)
                row.update(val_R=v["R"], val_beh=v["behavioral"],
                           val_osc=v["osc"], val_persist=v["persist"], val_stab=v["stab"])
                if v["R"] > best_val:
                    best_val = v["R"]
                    best_state = {k: vv.detach().clone() for k, vv in policy.state_dict().items()}
                if v["behavioral"] > best_beh:
                    best_beh = v["behavioral"]
                    best_beh_state = {k: vv.detach().clone() for k, vv in policy.state_dict().items()}
                    print(f"  [val {step:4d}] R={v['R']:.3f} beh={v['behavioral']:.3f} "
                          f"osc={v['osc']:.3f} pers={v['persist']:.3f} stab={v['stab']:.3f}  *best-beh",
                          flush=True)
                else:
                    print(f"  [val {step:4d}] R={v['R']:.3f} beh={v['behavioral']:.3f} "
                          f"osc={v['osc']:.3f} pers={v['persist']:.3f} stab={v['stab']:.3f}  "
                          f"(best-beh {best_beh:.3f})", flush=True)
            w.writerow({k: row.get(k, "") for k in mcols})
    ckpt = Path(args.out)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if best_beh_state is not None:
        policy.load_state_dict(best_beh_state)
    elif best_state is not None:
        policy.load_state_dict(best_state)
    torch.save(policy.state_dict(), ckpt)
    print(f"saved -> {ckpt}  (best behavioural={best_beh:.3f}, best total-R={best_val:.3f})")
    print(f"metrics -> {mf}")


def cmd_generate(args: argparse.Namespace):
    device = _device()
    policy = ConnectomePolicy(n_neurons=args.n).to(device)
    if args.ckpt and Path(args.ckpt).exists():
        policy.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    policy.eval()
    env = ConnectomeEnv(n_steps=args.sim_steps, drive=args.drive, gain=args.gain)
    specs = [(args.cat or "custom", args.spec)]
    with torch.no_grad():
        # seed from the spec so generation is fully reproducible: edges are
        # greedy (deterministic), and the E/I types / cluster assignment /
        # weight jitter are seeded identically per spec.
        seed = sum(args.spec.encode("utf-8")) & 0x7FFFFFFF
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        out = policy([s for _, s in specs], greedy=args.greedy)
        csrs = policy.to_csr(out)
    from .env import reward_from
    from .metrics import graph_metrics

    for i, (cat, s) in enumerate(specs):
        gm = graph_metrics(csrs[i])
        dyn = env.simulate(csrs[i])
        rw = reward_from(csrs[i], dyn, cat=cat)
        print(f"=== {cat}: {s}  [{'(greedy)' if args.greedy else '(sampled)'}]")
        print(f"  neurons={gm['n']} synapses={gm['m']} mean_deg={gm['mean_deg']:.2f}")
        print(f"  clustering={gm['clustering']:.3f} modularity={gm['modularity']:.3f} "
              f"deg_cv={gm['deg_cv']:.2f} ei_ratio={gm['ei_ratio']:.1f}")
        print(f"  dyn: FR={dyn['mean_fr']:.1f}Hz cv={dyn['cv_fr']:.2f} "
              f"out_frac={dyn['out_frac']:.3f} dead={dyn['dead_frac']:.2f}")
        print(f"  task: osc={dyn['osc']:.2f} persist={dyn['persist']:.2f} stab={dyn['stab']:.2f} "
              f"(on_out_fr={dyn['on_out_fr']:.1f}Hz off_out_fr={dyn['off_out_fr']:.1f}Hz)")
        print(f"  reward={rw['total']:.3f} (prop={rw['prop']:.2f} fire={rw['firing']:.2f} "
              f"div={rw['diversity']:.2f} syn={rw['syn']:.2f} net={rw['network']:.2f})")


def cmd_fix(args: argparse.Namespace):
    """See-and-fix: generate → render to PNG → critique → regenerate until healthy.

    The connectome analogue of image-gen models that see their own photos and
    revise bad ones. Every try is rendered + scored by the Rust engine and graph
    metrics; unhealthy graphs trigger a re-generation with a fresh seed.
    """
    device = _device()
    policy = ConnectomePolicy(n_neurons=args.n).to(device)
    if args.ckpt and Path(args.ckpt).exists():
        policy.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    policy.eval()
    env = ConnectomeEnv(n_steps=args.sim_steps, drive=args.drive, gain=args.gain)

    from .seeandfix import generate_fixed

    res = generate_fixed(
        args.spec, args.cat, policy, env,
        max_tries=args.tries, out_dir=args.outdir, n_neurons=args.n,
    )
    print(f"=== {res['cat']}: {res['spec']}  [{res['tries']} try/tries]")
    print(f"  reward={res['reward']:.3f}  dv={res['rw']['dv']:.2f} "
          f"prop={res['rw']['prop']:.2f} dead={res['rw']['dead_frac']:.2f}")
    print(f"  mean_deg={res['gm']['mean_deg']:.1f} ei_ratio={res['gm']['ei_ratio']:.1f} "
          f"modularity={res['gm']['modularity']:.3f} clustering={res['gm']['clustering']:.3f}")
    print("  critique:")
    for k, (ok, msg, sev) in res["flags"].items():
        print(f"    [{'OK' if ok else sev:5}] {k:12} {msg}")
    print(f"  image -> {res['image']}")
    print("  history:")
    for h in res["history"]:
        print(f"    try{h['try']:2} seed={h['seed']:10d} nud={h['nudge']:.2f} R={h['reward']:.3f} "
              f"{h['health']:5} prop={h['prop']:.2f} dead={h['dead']:.2f} "
              f"ei={h['ei_ratio']:.1f} deg={h['mean_deg']:.1f} bad={h['n_bad']}")


def cmd_probe(args: argparse.Namespace):
    """Probe: does the policy produce DIFFERENT graphs for different specs?

    Reports pairwise synapse-overlap across categories — the exact metric that
    showed the old models collapse (overlap ~1.0 = all specs identical).
    """
    device = _device()
    policy = ConnectomePolicy(n_neurons=args.n).to(device)
    if args.ckpt and Path(args.ckpt).exists():
        policy.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    policy.eval()
    env = ConnectomeEnv(n_steps=args.sim_steps, drive=args.drive, gain=args.gain)

    specs = [(cat, s) for cat, ss in spec_mod.SPECS.items() for s in ss[:2]]
    with torch.no_grad():
        out = policy([s for _, s in specs])
        csrs = policy.to_csr(out)

    # Structure stats per category — the reliable separation signal. Edge overlap
    # is misleading here: at deg~25, independently-sampled graphs naturally
    # overlap ~0.2, so overlap undercounts real spec-conditioning. Degree and
    # E/I distributions are what actually differ per spec.
    cats = [c for c, _ in specs]
    from .metrics import graph_metrics
    stats = []
    for csr in csrs:
        gm = graph_metrics(csr)
        stats.append((gm["mean_deg"], gm["ei_ratio"]))
    print(f"{'category':20} {'mean_deg':>8} {'ei_ratio':>8}")
    seen = set()
    for c, (d, ei) in zip(cats, stats):
        if c in seen:
            continue
        seen.add(c)
        degs = [stats[i][0] for i, cc in enumerate(cats) if cc == c]
        eis = [stats[i][1] for i, cc in enumerate(cats) if cc == c]
        print(f"{c:20} {np.mean(degs):8.1f} {np.mean(eis):8.2f}")

    # edge-set overlap (informational)
    def eset(csr):
        row_ptr = csr["row_ptr"]
        col_idx = csr["col_idx"]
        rows = []
        for r in range(len(row_ptr) - 1):
            rows += [r] * (row_ptr[r + 1] - row_ptr[r])
        return set(zip(rows, col_idx))

    edges = [eset(c) for c in csrs]
    print(f"\n{'cat1':16} {'cat2':16} {'overlap':>8}")
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            if cats[i] == cats[j]:
                continue
            a, b = edges[i], edges[j]
            ov = len(a & b) / max(1, min(len(a), len(b)))
            if i == 0 and j < 6:
                print(f"{cats[i]:16} {cats[j]:16} {ov:8.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--steps", type=int, default=500)
    tr.add_argument("--n", type=int, default=128, help="neurons per graph")
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--batch", type=int, default=4)
    tr.add_argument("--rollouts", type=int, default=8)
    tr.add_argument("--entropy", type=float, default=0.01)
    tr.add_argument("--struct-coef", type=float, default=1.0)
    tr.add_argument("--sim-steps", type=int, default=300)
    tr.add_argument("--drive", type=float, default=60.0)
    tr.add_argument("--gain", type=float, default=50.0)
    tr.add_argument("--onoff", type=int, default=50)
    tr.add_argument("--inrate", type=float, default=0.5)
    tr.add_argument("--log-every", type=int, default=5)
    tr.add_argument("--val-every", type=int, default=50,
                    help="validate greedy (anti-drift best-checkpoint) every N steps")
    tr.add_argument("--lr-decay", type=float, default=0.9995,
                    help="per-step LR multiplier (calms end-of-training drift)")
    tr.add_argument("--seed", type=int, default=42, help="global RNG seed (reproducible run)")
    tr.add_argument("--clip-eps", type=float, default=6.0,
                    help="PPO-style advantage clamp bound (std-normalised adv)")
    tr.add_argument("--crit-coef", type=float, default=0.5,
                    help="critic MSE weight in the combined loss")
    tr.add_argument("--critic-lr", type=float, default=1e-3,
                    help="learning rate for the learned value head (V(z))")
    tr.add_argument("--metrics", default="output/train_metrics.csv",
                    help="write per-step metrics (incl. greedy val osc/persist/stab) here")
    tr.add_argument("--out", default="sardana_flow_checkpoints/policy.pt")
    tr.add_argument("--resume", default=None, metavar="CKPT",
                    help="start from this checkpoint (old checkpoints w/o value_head load with strict=False)")
    tr.set_defaults(fn=cmd_train)

    ge = sub.add_parser("generate")
    ge.add_argument("--spec", required=True)
    ge.add_argument("--cat", default="custom")
    ge.add_argument("--n", type=int, default=128)
    ge.add_argument("--sim-steps", type=int, default=300)
    ge.add_argument("--drive", type=float, default=60.0)
    ge.add_argument("--gain", type=float, default=50.0)
    ge.add_argument("--ckpt", default=None)
    ge.add_argument("--greedy", dest="greedy", action="store_true", default=True,
                    help="deterministic edges (same spec → same graph, no sampling noise)")
    ge.add_argument("--no-greedy", dest="greedy", action="store_false",
                    help="sample edges (stochastic generation)")
    ge.set_defaults(fn=cmd_generate)

    fx = sub.add_parser("fix", help="see-and-fix: generate, render, critique, regenerate")
    fx.add_argument("--spec", required=True)
    fx.add_argument("--cat", default="custom")
    fx.add_argument("--n", type=int, default=128)
    fx.add_argument("--sim-steps", type=int, default=300)
    fx.add_argument("--drive", type=float, default=60.0)
    fx.add_argument("--gain", type=float, default=50.0)
    fx.add_argument("--tries", type=int, default=10)
    fx.add_argument("--outdir", default="output/connectomes")
    fx.add_argument("--ckpt", default=None)
    fx.set_defaults(fn=cmd_fix)

    pr = sub.add_parser("probe")
    pr.add_argument("--n", type=int, default=128)
    pr.add_argument("--sim-steps", type=int, default=300)
    pr.add_argument("--drive", type=float, default=60.0)
    pr.add_argument("--gain", type=float, default=50.0)
    pr.add_argument("--ckpt", default=None)
    pr.set_defaults(fn=cmd_probe)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
