# sardana_flow — trained connectome model report

**Date:** 2026-08-07 · **Checkpoint:** `sardana_flow_checkpoints/policy.pt` (= v12, 300 steps,
struct-coef 1.0, behavioral rewards; verified greedy mean R = 0.581, dv = 1.60)

## What the model does

`text spec → sampled N×N connectome (edges + E/I types + weights)` via a learned
stochastic-block-model policy, trained end-to-end with the Rust `ConnectomeBrain`
engine as the reward source (REINFORCE on dynamics + differentiable structure loss).

**New in this iteration:**
- **task-behavioural reward terms** measured from engine time-series (`env.SPEC_PROFILES`):
  `osc` (intrinsic oscillation in OFF — motor/CPG), `persist` (working-memory: output stays
  active after ON — olfactory/reward), `stab` (temporal stability — homeostasis).
- **greedy inference mode** (`--greedy`, default): deterministic edges (exact top-k by `p_edge`,
  preserving expected density) + top-k E/I types (argmax degenerates to ~0 inhibition) + no
  weight jitter. Same spec → same graph, seeded from the spec text. **This kills the old
  generation lottery** (single-sample variance) — greedy is now the recommended interface.

## Final ranking (6 samples × 8 categories, N=128, current reward)

| model | sampled R | greedy R | greedy dv | greedy dead | notes |
|---|---|---|---|---|---|
| **v12** (canonical) | 0.356 | **0.581** | **1.60** | 0.56 | greedy-optimal: density pinned, strong conduction everywhere |
| v10 | **0.511** | 0.555 | 1.39 | 0.57 | best sampled; behavioral persist/stab visible |
| v9 | 0.492 | 0.581 | 0.85 | 0.67 | old champion; best balanced across both modes |

### v12 greedy, per category

| category | R | dv | dead | mean_deg | ei_ratio |
|---|---|---|---|---|---|
| motion_detection | 0.750 | 1.97 | 0.54 | 10.1 | 4.0 |
| olfactory_memory | 0.580 | 1.94 | 0.35 | 9.4 | 3.4 |
| sonar_echo | 0.644 | 1.59 | 0.56 | 10.2 | 4.2 |
| motor_gait | 0.473 | 1.77 | 0.51 | 9.8 | 3.5 |
| reward_learning | 0.503 | 1.40 | 0.65 | 9.1 | 14.3 |
| visual_recognition | 0.584 | 1.47 | 0.66 | 9.9 | 3.1 |
| auditory | 0.567 | 1.27 | 0.67 | 9.4 | 3.6 |
| homeostasis | 0.545 | 1.38 | 0.51 | 8.3 | 5.1 |

- **dv > 0 in all 8 categories** (1.27–1.97) — every generated graph conducts input drive to the
  output group. v9 had 2 categories near dv≈0; v12 conducts ~2× stronger.
- **Density pinned to target** (~9–10 edges/neuron) via struct-coef 1.0 — no end-of-training drift.
- **E/I healthy in 7/8** (3.1–5.1); `reward_learning` drifts to ei=14.3 (its persist weight 2.0
  pushed it all-excitatory — weak spot).

## Honest limitations

1. **`osc` / `persist` behaviors did NOT emerge** in any model (osc≈0, persist≈0 in greedy). The
   output rarely spikes (reward favors graded subthreshold dv), so the autocorrelation floor is ~0.
   The behavioral *terms* are live and shape dynamics (dv up, dead down, v10 sampled showed
   persist→target for motor/homeostasis) but the named behaviors didn't materialize.
2. **Trade-off between modes**: greedy-optimal models (v12) have dead sampled mode (95% dead —
   sparse p_edge fails under Bernoulli sampling); sampled-optimal models (v10) have weaker greedy.
   Greedy is the recommended interface, so v12 is canonical. For stochastic generation use
   `policy_v10.pt`.
3. End-of-training drift (entropy/struct rising, density escaping target) resisted LR decay across
   v10/v11; struct-coef 1.0 (v12) finally pinned it, at the cost of sampled-mode robustness.

## Usage

```bash
# deterministic generation (same spec → same graph) — the default & recommended
python sardana_launch.py generate --spec "High-speed visual T4/T5 motion detection" \
  --cat motion_detection --n 128 --ckpt sardana_flow_checkpoints/policy.pt

# stochastic generation (lottery) — prefer policy_v10.pt for this mode
python sardana_launch.py generate --spec "..." --cat olfactory_memory --n 128 \
  --no-greedy --ckpt sardana_flow_checkpoints/policy_v10.pt

# probe per-category differentiation
python sardana_launch.py probe --n 128 --ckpt sardana_flow_checkpoints/policy.pt

# train (anti-drift: LR decay + greedy best-checkpoint every --val-every)
python sardana_launch.py train --steps 300 --n 128 --sim-steps 200 \
  --batch 4 --rollouts 8 --entropy 0.05 --struct-coef 1.0 --lr-decay 0.999 \
  --val-every 15 --out sardana_flow_checkpoints/policy_v13.pt
```

## Checkpoints

- `policy.pt` = v12 (canonical, greedy-optimal)
- `policy_v10.pt` = best sampled-mode model (behavioral persist/stab)
- `policy_v9.pt` = best balanced across both modes
