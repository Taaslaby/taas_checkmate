#!/usr/bin/env python3
"""
SPAUN Hyperparameter Sweep v2 — расширенный брутфорс.

Новое:
  - Больше параметров: sp_dim, workspace, cortical_dim, wm_slots, lr, epochs, steps, n_pixels
  - Random search (sample N из полного пространства)
  - 1D sweep вокруг оптимума
  - Анализ: какой параметр влияет больше всего

Использование:
    cd ../..
    cargo build --release --features python
    cd examples/python
    PYTHONPATH=../../target/release python3 spaun_sweep.py --mode random --n 200
    PYTHONPATH=../../target/release python3 spaun_sweep.py --mode grid    # полная сетка
    PYTHONPATH=../../target/release python3 spaun_sweep.py --mode 1d      # 1D вокруг оптимума
"""

import multiprocessing as mp
import time
import json
import os
import sys
import random
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# ── Путь к собранной библиотеке ──────────────────────────
LIB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../target/release")
)
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

# ── Путь к brain.taas ────────────────────────────────────
BRAIN_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../crates/connectome-gpt/models/brain.taas")
)

# ═══════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════

PARAM_RANGES = {
    "sp_dim":       [32, 64, 128, 256, 512],
    "workspace":    [4, 8, 16, 32, 64],
    "cortical_dim": [64, 128, 256, 512, 1024],
    "wm_slots":     [2, 4, 8, 16],
    "lr":           [0.01, 0.03, 0.05, 0.1, 0.2, 0.5],
    "n_epochs":     [5, 10, 20, 50, 100],
    "n_steps":      [0, 3, 5, 10, 20],
    "n_pixels":     [36, 64, 100, 144],
}

# Best from sweep 1
BEST_CONFIG = {
    "sp_dim": 128, "workspace": 16, "cortical_dim": 256,
    "wm_slots": 8, "lr": 0.05, "n_epochs": 20,
    "n_steps": 10, "n_pixels": 100,
}


@dataclass
class SweepConfig:
    """Single SPAUN configuration."""
    sp_dim: int = 128
    workspace: int = 16
    cortical_dim: int = 256
    wm_slots: int = 8
    lr: float = 0.05
    n_epochs: int = 20
    n_steps: int = 10
    n_pixels: int = 100

    @property
    def name(self) -> str:
        return (f"sp{self.sp_dim}_ws{self.workspace}_"
                f"ctx{self.cortical_dim}_wm{self.wm_slots}_"
                f"lr{self.lr}_ep{self.n_epochs}_"
                f"st{self.n_steps}_px{self.n_pixels}")


@dataclass
class SweepResult:
    config: SweepConfig
    loss: float
    accuracy: float
    train_ms: float
    task_ms: float
    copy_acc: float
    recall_acc: float
    count_acc: float


# ═══════════════════════════════════════════════════════════
#  Trial runner
# ═══════════════════════════════════════════════════════════

def run_trial(cfg_dict: dict) -> Optional[SweepResult]:
    cfg = SweepConfig(**cfg_dict)

    try:
        from taas_checkmate import SpaunPipeline
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return None

    try:
        pipe = SpaunPipeline(str(BRAIN_PATH))
    except Exception:
        pipe = SpaunPipeline(None)

    try:
        pipe.set_sp_dim(cfg.sp_dim)
        pipe.set_workspace_size(cfg.workspace)
        pipe.set_cortical_dim(cfg.cortical_dim)
        pipe.set_wm_slots(max(cfg.wm_slots, 2))  # min 2 for serial_recall
        pipe.set_n_outputs(10)
        pipe.set_n_pixels(cfg.n_pixels)

        # ── Train ────────────────────────────────────────
        t0 = time.perf_counter()
        loss = pipe.train(n_epochs=cfg.n_epochs, steps=cfg.n_steps, lr=cfg.lr)
        train_ms = (time.perf_counter() - t0) * 1000

        # ── Test: Copy ──────────────────────────────────
        t0 = time.perf_counter()
        copy_correct = sum(
            1 for d in range(10)
            if pipe.copy(d, steps=cfg.n_steps) == d
        )
        copy_acc = copy_correct / 10.0

        # ── Test: Serial Recall ─────────────────────────
        sequences = [[3, 7, 1], [0, 5, 9, 2], [4, 4, 4]]
        recall_correct = recall_total = 0
        for seq in sequences:
            out = pipe.serial_recall(seq, steps=cfg.n_steps)
            for o, s in zip(out, seq):
                if o == s:
                    recall_correct += 1
                recall_total += 1
        recall_acc = recall_correct / max(recall_total, 1)

        # ── Test: Counting ──────────────────────────────
        count_correct = count_total = 0
        for n in [1, 2, 3]:
            out = pipe.counting(n, steps=cfg.n_steps)
            expected = list(range(1, n + 1))
            for o, e in zip(out, expected):
                if o == e:
                    count_correct += 1
                count_total += 1
        count_acc = count_correct / max(count_total, 1)

        task_ms = (time.perf_counter() - t0) * 1000

        total_correct = copy_correct + recall_correct + count_correct
        total_tested = 10 + recall_total + count_total
        accuracy = total_correct / max(total_tested, 1)

        result = SweepResult(
            config=cfg, loss=float(loss), accuracy=float(accuracy),
            train_ms=float(train_ms), task_ms=float(task_ms),
            copy_acc=float(copy_acc), recall_acc=float(recall_acc),
            count_acc=float(count_acc),
        )

        print(f"  ✓ {cfg.name}: loss={loss:.4f} acc={accuracy:.1%} "
              f"C={copy_acc:.0%} R={recall_acc:.0%} N={count_acc:.0%} "
              f"({train_ms:.0f}ms)", flush=True)

        return result

    except Exception as e:
        print(f"  ✗ {cfg.name}: FAILED - {e}", flush=True)
        return None


# ═══════════════════════════════════════════════════════════
#  Grid builders
# ═══════════════════════════════════════════════════════════

def build_grid() -> list[SweepConfig]:
    """Full factorial grid (4×5×5×5×6×5×5×4 = 300K — too many).
    Use random search instead for full space."""
    configs = []
    for sp_dim in PARAM_RANGES["sp_dim"]:
        for ws in PARAM_RANGES["workspace"]:
            for ctx in PARAM_RANGES["cortical_dim"]:
                for wm in PARAM_RANGES["wm_slots"]:
                    for lr in PARAM_RANGES["lr"]:
                        for ep in PARAM_RANGES["n_epochs"]:
                            for st in PARAM_RANGES["n_steps"]:
                                for px in PARAM_RANGES["n_pixels"]:
                                    configs.append(SweepConfig(
                                        sp_dim=sp_dim, workspace=ws,
                                        cortical_dim=ctx, wm_slots=wm,
                                        lr=lr, n_epochs=ep,
                                        n_steps=st, n_pixels=px,
                                    ))
    return configs


def build_random(n: int) -> list[SweepConfig]:
    """Random search: sample N configs from full space."""
    configs = []
    for _ in range(n):
        configs.append(SweepConfig(
            sp_dim=random.choice(PARAM_RANGES["sp_dim"]),
            workspace=random.choice(PARAM_RANGES["workspace"]),
            cortical_dim=random.choice(PARAM_RANGES["cortical_dim"]),
            wm_slots=random.choice(PARAM_RANGES["wm_slots"]),
            lr=random.choice(PARAM_RANGES["lr"]),
            n_epochs=random.choice(PARAM_RANGES["n_epochs"]),
            n_steps=random.choice(PARAM_RANGES["n_steps"]),
            n_pixels=random.choice(PARAM_RANGES["n_pixels"]),
        ))
    return configs


def build_1d_sweep() -> list[SweepConfig]:
    """1D sweep: vary ONE parameter at a time around best config."""
    configs = [SweepConfig(**BEST_CONFIG)]  # baseline

    for param, values in PARAM_RANGES.items():
        for v in values:
            cfg = SweepConfig(**BEST_CONFIG)
            setattr(cfg, param, v)
            configs.append(cfg)

    # Remove duplicates
    seen = set()
    unique = []
    for cfg in configs:
        if cfg.name not in seen:
            seen.add(cfg.name)
            unique.append(cfg)
    return unique


# ═══════════════════════════════════════════════════════════
#  Sweep runner
# ═══════════════════════════════════════════════════════════

def run_sweep(configs: list[SweepConfig], max_workers: Optional[int] = None):
    n_workers = max_workers or max(1, mp.cpu_count() - 1)
    print(f"\n{'='*60}")
    print(f"  🧠 SPAUN Hyperparameter Sweep v2")
    print(f"  Workers: {n_workers} | Configs: {len(configs)}")
    print(f"  Model: {BRAIN_PATH}")
    print(f"{'='*60}\n")

    cfg_dicts = [asdict(c) for c in configs]

    print("  🔥 Warmup...", flush=True)
    _ = run_trial(cfg_dicts[0])
    print()

    t_start = time.perf_counter()

    with mp.Pool(n_workers) as pool:
        raw = pool.map(run_trial, cfg_dicts)

    elapsed = time.perf_counter() - t_start

    results = [r for r in raw if r is not None]
    print(f"  Successful: {len(results)}/{len(raw)} configs")

    results.sort(key=lambda r: r.accuracy, reverse=True)

    print(f"\n{'='*60}")
    print(f"  📊 Sweep Complete")
    print(f"  Time: {elapsed:.0f}s ({elapsed/max(len(results),1):.1f}s/config)")
    print(f"{'='*60}\n")

    # Top 15
    print("─── Top 15 Configs ───")
    for i, r in enumerate(results[:15]):
        print(f"  {i+1:2d}. {r.config.name:55s} "
              f"loss={r.loss:.4f} acc={r.accuracy:.1%} "
              f"[C={r.copy_acc:.0%} R={r.recall_acc:.0%} N={r.count_acc:.0%}]")
    print()

    best = results[0]
    print(f"  🏆 Best: {best.config.name}")
    for f in ["loss", "accuracy", "copy_acc", "recall_acc", "count_acc"]:
        print(f"     {f}: {getattr(best, f):.4f}" if f == "loss" else
              f"     {f}: {getattr(best, f):.1%}")
    print(f"     Train: {best.train_ms:.0f}ms")
    print()

    worst = results[-1]
    print(f"  💀 Worst: {worst.config.name} (acc={worst.accuracy:.1%}, loss={worst.loss:.4f})")
    print()

    # Parameter importance
    print("─── Parameter Importance ───")
    param_impact = {}
    for param in PARAM_RANGES.keys():
        values = {}
        for r in results:
            v = getattr(r.config, param)
            if v not in values:
                values[v] = []
            values[v].append(r.accuracy)
        if len(values) >= 2:
            means = [np.mean(vals) for vals in values.values()]
            param_impact[param] = max(means) - min(means)
            print(f"  {param:15s}: range={max(means)-min(means):.1%} "
                  f"best={max(means):.1%} worst={min(means):.1%}")

    # Sort by impact
    sorted_impact = sorted(param_impact.items(), key=lambda x: -x[1])
    print(f"\n  📈 Parameter Ranking (by impact):")
    for i, (p, imp) in enumerate(sorted_impact):
        print(f"    {i+1}. {p:15s}: {imp:.1%} spread")

    save_results(results)
    try:
        plot_results(results)
    except ImportError:
        print("  ℹ️  matplotlib not available — skipping plot")

    return results


def save_results(results: list[SweepResult], path: str = "sweep_results.json"):
    data = [{"config": asdict(r.config), "loss": r.loss,
             "accuracy": r.accuracy, "train_ms": r.train_ms,
             "task_ms": r.task_ms, "copy_acc": r.copy_acc,
             "recall_acc": r.recall_acc, "count_acc": r.count_acc}
            for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  💾 Results saved to {path} ({len(data)} configs)")


def plot_results(results: list[SweepResult]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    params = list(PARAM_RANGES.keys())
    n_cols = 4
    n_rows = (len(params) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    fig.suptitle("SPAUN Hyperparameter Sweep v2 — Accuracy vs Parameters", fontsize=16)

    for idx, param in enumerate(params):
        ax = axes.flat[idx] if n_rows > 1 else axes[idx]
        values = {}
        for r in results:
            v = getattr(r.config, param)
            values.setdefault(v, []).append(r.accuracy * 100)

        labels = sorted(values.keys())
        means = [np.mean(values[k]) for k in labels]
        stds = [np.std(values[k]) for k in labels]

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=5, color="royalblue", alpha=0.8)
        ax.set_xlabel(param)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Accuracy by {param}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in labels], fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    for i in range(len(params), n_rows * n_cols):
        fig.delaxes(axes.flat[i])

    plt.tight_layout()
    plt.savefig("sweep_results.png", dpi=150)
    print(f"  📈 Plot saved to sweep_results.png")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SPAUN Hyperparameter Sweep v2")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    parser.add_argument("--mode", choices=["grid", "random", "1d"], default="random",
                        help="Sweep mode: grid (4×... huge), random (sample N), 1d (vary 1 param at a time)")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of random configs (for random mode)")
    args = parser.parse_args()

    if args.mode == "grid":
        configs = build_grid()
        print(f"  Full grid: {len(configs)} configs")
        print(f"  ⚠️  This is huge! Consider --mode random instead.")
        if len(configs) > 500:
            print(f"  Sampling 500 random from grid instead.")
            configs = random.sample(configs, 500)
    elif args.mode == "random":
        configs = build_random(args.n)
        print(f"  Random search: {len(configs)} configs")
    elif args.mode == "1d":
        configs = build_1d_sweep()
        print(f"  1D sweep: {len(configs)} configs")

    # Remove exact duplicates
    seen = set()
    unique = []
    for c in configs:
        if c.name not in seen:
            seen.add(c.name)
            unique.append(c)
    configs = unique
    print(f"  Unique configs: {len(configs)}")

    results = run_sweep(configs, max_workers=args.workers)

    best = results[0]
    print(f"\n  🏆 Optimal Architecture:")
    for f in PARAM_RANGES.keys():
        print(f"     {f:15s}: {getattr(best.config, f)}")
    print(f"     Accuracy: {best.accuracy:.1%}")
