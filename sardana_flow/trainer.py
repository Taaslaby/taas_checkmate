"""REINFORCE-with-learned-critic trainer for the connectome policy.

Update: the old per-category EMA baseline is replaced by a **learned critic**
`V(z)` (added to the policy in generator.py), giving a PPO-style advantage

    A(spec) = R(spec, A) - V(z_spec)

which is far lower-variance than an EMA-smoothed scalar, so the now-dense
behavioural reward (osc / persist / stab / alive, measured on the *membrane*)
can actually be learned over a few hours of training.

Because an episode here is a *single* connectome (a one-shot decision, no
temporal chain), GAE-over-time does not apply. The stabilisation that does
apply is: learned baseline + advantage standardisation + clamp + a small critic
MSE. This is an actor-critic (A2C-style) update, the honest subset of PPO for
"reward is a function of one generated graph".

Structure loss (density / heavy-tail / modularity / E/I on the *soft* expected
graph, generator.structure_loss) stays — it is differentiable and gives the
direct structural gradient; REINFORCE carries only the sampled dynamics.
"""

from __future__ import annotations

import torch

from .env import dynamics_reward_from, profile_for


def train_step(
    policy,
    specs: list[tuple[str, str]],
    env,
    optimizer,
    entropy_coef: float = 0.05,
    struct_coef: float = 1.0,
    n_rollouts: int = 4,
    max_grad: float = 1.0,
    baseline: dict[str, float] | None = None,
    device: torch.device | None = None,
    clip_eps: float = 6.0,
    crit_coef: float = 0.5,
) -> dict:
    """One training step. baseline is a {category: EMA} tracking dict (kept for
    display only — the *learned critic* is what anchors the advantage now).

    Combined gradient = (structure loss) + (clipped policy gradient over the
    sampled dynamics reward) − (entropy) + (critic MSE to the reward).
    """
    if baseline is None:
        baseline = {}
    device = device or next(policy.parameters()).device

    texts = [s for _, s in specs]
    cat = [c for c, _ in specs]
    B = len(texts)
    log_probs = torch.zeros(B, device=device)
    rewards = torch.zeros(B, device=device)
    ent_sum = torch.zeros(B, device=device)
    struct_sum = torch.zeros(B, device=device)
    val_sum = torch.zeros(B, device=device)

    # per-edge log-prob normalization (stable per-edge scale, N² ~ 65k for 256)
    n_edges = float(policy.n_neurons ** 2)

    # per-category structure targets (spec-dependent reward)
    struct_t = [profile_for(c)["struct"] for c in cat]
    dens_t = [s["density"] for s in struct_t]
    ei_t = [s["ei_ratio"] for s in struct_t]
    mod_t = [s["mod_within"] for s in struct_t]

    for _ in range(n_rollouts):
        out = policy(texts)
        lp = policy.log_prob_of(out) / n_edges  # [B] per-edge avg
        ent = out["ent_edge"].mean(dim=(1, 2))
        struct = policy.structure_loss(out, dens_t, ei_t, mod_t)
        val = out["value"]  # [B] critic estimate V(z) (same for all rollouts)
        csrs = policy.to_csr(out)
        for i in range(B):
            r = dynamics_reward_from(csrs[i], env.simulate(csrs[i]), cat=cat[i])["total"]
            rewards[i] += r
            log_probs[i] += lp[i]
            ent_sum[i] += ent[i]
            struct_sum[i] += struct[i]
            val_sum[i] += val[i]
    rewards /= n_rollouts
    log_probs /= n_rollouts
    ent_sum /= n_rollouts
    struct_sum /= n_rollouts
    val_sum /= n_rollouts  # = V(z_i) (rollouts share the same z → identical V)

    # per-category moving average — display/analytics only, not the advantage
    for i, c in enumerate(cat):
        prev = baseline.get(c, rewards[i].item())
        baseline[c] = 0.9 * prev + 0.1 * rewards[i].item()

    # ---- learned-critic baseline + critic MSE --------------------------------
    # critic learns to predict the reward; advantage R − V(z) then has far lower
    # variance than the EMA vector. Value head is trained by `crit_loss` alone
    # (advantage uses V(z).detach() so the actor doesn't push V through the pg).
    b = val_sum.detach()
    crit_loss = (val_sum - rewards.detach()).pow(2).mean()

    # ---- regularised policy gradient over the *sampled dynamics* reward ------
    # standardise + clamp the advantage so outlier rewards can't blow the step.
    adv = rewards - b
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    adv = adv.clamp(-clip_eps, clip_eps)

    pg = -(adv.detach() * log_probs).mean()
    ent = ent_sum.mean()
    struct = struct_sum.mean()

    loss = pg + crit_coef * crit_loss + struct_coef * struct - entropy_coef * ent

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad)
    optimizer.step()

    return {
        "loss": loss.item(),
        "pg": pg.item(),
        "crit": crit_loss.item(),
        "struct": struct.item(),
        "entropy": ent.item(),
        "mean_reward": rewards.mean().item(),
        "baselines": dict(baseline),
    }