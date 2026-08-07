"""The policy: text spec -> connectome.

Design (fixes the two failure modes found in the audit):

1.  **Learned edges, not heuristic.** Old models output per-neuron tokens and
    then wired them with a hardcoded `P(same_comm)=0.3` / `P(diff_comm)=0.003`.
    Here the policy produces *latent positions* on a sphere; edge probability
    is `sigmoid(scale * (cos_sim(e_i,e_j) - bias))`. Communities emerge from a
    soft cluster assignment onto spec-conditioned anchors — a *learnable*
    stochastic block model.

2.  **Spec actually matters.** The cluster anchors and per-neuron type/weight
    scale are functions of a text-encoder vector `z(spec)`. Two different specs
    get different wiring, because `z` rotates the anchors (the hash-rotation
    idea from the verified fix, made continuous).

Everything is differentiable where it can be; the discrete adjacency is sampled
with the Gumbel trick so we have `log_prob` for REINFORCE.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HASH_VOCAB = 4096


class CharHashEncoder(nn.Module):
    """Tiny hashed-bigram bag-of-words encoder.

    Not a pretrained transformer — we deliberately keep the text encoder small
    and train it end-to-end with the RL loop. Hashed char n-grams keep the vocab
    bounded while still letting the model see real words.
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.embed = nn.Embedding(HASH_VOCAB, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer("_vocab", torch.arange(HASH_VOCAB))

    def _hash(self, text: str) -> torch.Tensor:
        """Deterministic hash of char-bigrams -> token ids."""
        ids = []
        t = " " + text.lower() + " "
        for i in range(len(t) - 1):
            c1, c2 = t[i], t[i + 1]
            h = (ord(c1) * 31 + ord(c2)) & 0xFFFFFFFF
            ids.append(h % HASH_VOCAB)
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, texts: list[str]) -> torch.Tensor:
        zs = []
        for s in texts:
            ids = self._hash(s)
            if ids.numel() == 0:
                ids = torch.zeros(1, dtype=torch.long)
            e = self.embed(ids.to(self.embed.weight.device))
            zs.append(e.mean(0))
        z = torch.stack(zs)  # [B, d]
        return F.relu(self.proj(z))


class ConnectomePolicy(nn.Module):
    """Generates a sampled adjacency + weights + E/I types for N neurons.

    All randomness is reparameterized (Gumbel) so `log_prob` w.r.t. the sampled
    graph is computable for REINFORCE. Crucially, the sampled edge mask and the
    `log_prob` always correspond to the *same* graph (no post-hoc pruning of
    sampled edges), so the reward is unbiased.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_anchors: int = 8,
        n_neurons: int = 256,
        d_pos: int = 16,
        tau: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_anchors = n_anchors
        self.n_neurons = n_neurons
        self.tau = tau

        self.text_enc = CharHashEncoder(d_model)

        # spec vector -> anchor positions (K anchors on the unit sphere).
        # Bias toward a fixed spread-out anchor matrix so the network starts
        # *sparse and modular*, not collapsed at the centroid.
        self.anchor_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_anchors * d_model),
        )
        # spec vector -> anchor size (prior over community membership)
        self.anchor_size = nn.Linear(d_model, n_anchors)
        # per-neuron position embedding + cluster assignment head
        self.pos_embed = nn.Embedding(n_neurons, d_pos)
        self.cluster_head = nn.Linear(d_model + d_pos, n_anchors)
        self.type_head = nn.Linear(d_model + d_pos, 2)  # exc/inh logits

        # edge probability scale/bias + weight scale — all conditioned on z
        self.edge_gate = nn.Linear(d_model, 3)  # log-scale, bias, log-weight

        # learned critic V(z): expected reward given the spec — a lower-variance
        # baseline than the old per-category EMA, so the sampled behavioural
        # reward can actually be learned (PPO-style advantage R − V(z)).
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # initialization: forward anchors on the sphere, high edge bias (sparse)
        self._init_anchor_bias()

    def _init_anchor_bias(self):
        # deterministic pseudo-random orthogonal-ish anchors
        g = torch.Generator().manual_seed(7)
        A = torch.randn(self.n_anchors, self.d_model, generator=g)
        A = A / A.norm(dim=1, keepdim=True)
        with torch.no_grad():
            # anchor_mlp output at z=0 ~ A flattened; store as bias of last layer
            last = self.anchor_mlp[-1]
            last.bias.data.copy_(A.flatten() * 2.0)
            last.weight.data.mul_(0.05)
            self.edge_gate.weight.data.mul_(0.1)
            # scale=e^0=1, bias=2.0 → deg≈19, dv>0, prop≈0.3: lands the initial
            # policy in the *graded* regime (not the deg≈52 seizure regime where
            # dv<0 → prop=0 → reward is a constant floor). Same-community
            # sim≈1 → p≈0.27; cross-community sim≈0 → p≈0.12: sparse + modular,
            # explorable, not peaked at extremes.
            self.edge_gate.bias.data.copy_(torch.tensor([0.0, 2.0, 0.0]))

    def _spec_conditioning(self, z: torch.Tensor):
        B = z.shape[0]
        anchors = self.anchor_mlp(z)  # [B, K*d]
        anchors = anchors.view(B, self.n_anchors, self.d_model)
        anchors = F.normalize(anchors, dim=-1)  # unit sphere
        size_logit = self.anchor_size(z)  # [B, K]
        return anchors, size_logit

    def latent_positions(
        self, z: torch.Tensor, device: torch.device, greedy: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (e, p_assign, type_logit, size_logit).

        Cluster assignment is Gumbel-softmax with hard=True (straight-through):
        forward each neuron is in *exactly one* community; backward gradient
        flows through the soft assignment. This is what keeps communities real
        instead of averaging every neuron to the anchor centroid (the collapse).

        greedy=True: argmax instead of sampling — each neuron goes to its
        most-likely community, so generation is fully deterministic.
        """
        B = z.shape[0]
        anchors, size_logit = self._spec_conditioning(z)
        pos = self.pos_embed.weight.to(device).unsqueeze(0).expand(B, -1, -1)  # [B,N,dpos]
        z_exp = z.unsqueeze(1).expand(B, self.n_neurons, -1)  # [B,N,d]
        feat = torch.cat([z_exp, pos], dim=-1)  # [B,N,d+dpos]
        assign_logit = self.cluster_head(feat)  # [B,N,K]
        if greedy:
            p_assign = F.one_hot(assign_logit.argmax(-1), self.n_anchors).float()  # [B,N,K]
        else:
            p_assign = F.gumbel_softmax(assign_logit, tau=self.tau, hard=True)  # [B,N,K] one-hot
        e = torch.einsum("bnk,bkd->bnd", p_assign, anchors)  # [B,N,d]
        e = F.normalize(e, dim=-1)
        type_logit = self.type_head(feat)  # [B,N,2]
        return e, p_assign, type_logit, size_logit

    def _sample_edges(self, e: torch.Tensor, z: torch.Tensor, greedy: bool = False,
                      nudge: float = 0.0):
        """Bernoulli edges from latent similarity.

        greedy=True (inference): deterministic — keep the strongest p_edge,
        exactly preserving the expected edge count. This kills the generation
        lottery: the same spec always yields the same graph (the old single-sample
        variance — an occasional dead/auditory graph — disappears). Sampling
        stays the default for training.

        nudge>0 (see-and-fix retries): add zero-mean noise to p_edge BEFORE the
        top-k cut, keeping k (the expected density) pinned exactly while letting
        near-threshold wirings swap in/out. This is what makes "regenerate with a
        fresh seed" actually explore — a different wiring at the same density per
        retry — instead of re-rolling the byte-identical graph.
        """
        B, N, _ = e.shape
        gate = self.edge_gate(z)  # [B,3]
        log_scale = gate[:, 0].clamp(min=-2.0, max=4.0)
        bias = gate[:, 1]
        log_w = gate[:, 2].clamp(min=-2.0, max=3.0)

        scale = log_scale.exp().view(B, 1, 1)
        sim = torch.bmm(e, e.transpose(1, 2))  # [B,N,N] cos similarity (normalized)
        logits = scale * (sim - bias.view(B, 1, 1))  # [B,N,N]

        # remove self-edges (diagonal): sim_ii=1 so hard_ii=1 almost always
        mask = 1.0 - torch.eye(N, device=e.device)

        p_s = torch.sigmoid(logits)
        if greedy:
            # deterministic: keep exactly the top-k edges by p_edge, k = expected
            # edge count. (A `>= threshold` form overshoots badly when many edges
            # tie at the threshold — the graph came out ~3× denser than k.)
            p = p_s * mask.view(1, N, N)
            flat = p.view(B, -1)
            k = flat.sum(dim=1).round().long().clamp(0, flat.shape[1])  # [B]
            if nudge and nudge > 0:
                # explore: re-rank by noisy p, but keep the SAME k (density pinned)
                p = (p + nudge * torch.randn_like(p)) * mask.view(1, N, N)
                flat = p.view(B, -1)
            topk = flat.topk(k.max(), dim=1).indices  # [B, k_max] desc
            hard = torch.zeros_like(flat)
            for b in range(B):
                hard[b, topk[b, : k[b]]] = 1.0
            hard = hard.view(B, N, N)
            edges = hard  # no straight-through needed under no_grad
        else:
            # Exact Bernoulli sample: hard ~ Bernoulli(p) with p = sigmoid(logits).
            # (The single-Gumbel form hard=(logits+g>0) had P=1-exp(-e^logit) ≠
            # sigmoid(logits), so the sampled graph was systematically denser than
            # the differentiable p_edge the structure loss optimises — the loss
            # drove density to 0.12 while the engine saw 0.43.)
            u = torch.rand_like(logits)
            hard = (u < p_s).float()  # [B,N,N] exact Bernoulli
            # straight-through: hard forward, soft (sigmoid) backward
            soft = torch.sigmoid(logits / self.tau)
            edges = hard + (soft - soft.detach())
            edges = edges * mask.view(1, N, N)

        # REINFORCE log-prob of the *hard* sample (this is what reaches the engine)
        log_p_1 = F.logsigmoid(logits)
        log_p_0 = F.logsigmoid(-logits)
        log_prob = hard * log_p_1 + (1 - hard) * log_p_0  # [B,N,N]
        log_prob = log_prob * mask.view(1, N, N)  # diagonal contributes nothing
        # true Bernoulli edge probabilities (for entropy / diagnostics)
        p_edge = torch.sigmoid(logits) * mask.view(1, N, N)
        ent_edge = -(p_edge * F.logsigmoid(logits) + (1 - p_edge) * F.logsigmoid(-logits))
        ent_edge = ent_edge * mask.view(1, N, N)

        w_scale = log_w.exp().view(B, 1, 1)
        weights = w_scale * (sim + 0.5).clamp(min=0.2)  # [B,N,N] positive magnitude
        return edges, weights, log_prob, hard, sim, ent_edge, logits, mask, p_edge

    def forward(self, texts: list[str], greedy: bool = False,
                nudge: float = 0.0, ei_target: float | None = None) -> dict:
        """Generate a batch of graphs. greedy=True → deterministic edges.

        Training always samples (greedy=False); inference (generate/probe) can
        use greedy=True for a stable, reproducible graph per spec.

        nudge>0: explore near-greedy wirings at pinned density (see-and-fix
        retries). ei_target: override the inhibitory count to match a category's
        target ei_ratio — the "critic told me the E/I balance is off, fix it"
        correction, applied at inference with zero retraining.
        """
        device = next(self.parameters()).device
        z = self.text_enc(texts)  # [B,d]
        value: torch.Tensor = self.value_head(z).squeeze(-1)  # [B] critic baseline V(z)
        e, p_assign, type_logit, size_logit = self.latent_positions(z, device, greedy)
        edges, weights, log_prob, hard, sim, ent_edge, logits, mask, p_edge = self._sample_edges(e, z, greedy, nudge)

        # E/I types: 1=inhibitory. Gumbel-softmax for differentiable type sample;
        # in greedy mode pick the top-k most-likely-inhibitory neurons (k from the
        # learned P(inh)) — deterministic AND preserves the E/I balance. A plain
        # argmax degenerates to ~0 inhibitory when no neuron's P(inh)>0.5, which
        # kills the E/I balance (ei_ratio blew up to 1e13).
        if greedy:
            p_inh = F.softmax(type_logit, dim=-1)[..., 1]  # [B,N]
            _B = p_inh.shape[0]
            if ei_target is not None:
                # critic-driven correction: fix the inhibitory COUNT to the
                # category target; identity (who is inhibitory) still comes from
                # the learned P(inh) ranking.
                k_inh = torch.tensor(self.n_neurons / (1.0 + ei_target)).round().long().clamp(1, self.n_neurons - 1)
                k_inh = torch.full((_B,), k_inh.item(), device=type_logit.device)
            else:
                k_inh = (p_inh.mean(dim=1) * self.n_neurons).round().long()  # [B]
            types = torch.zeros(_B, self.n_neurons, device=type_logit.device)
            for b in range(_B):
                rank = p_inh[b].argsort(descending=True)
                types[b, rank[: k_inh[b]]] = 1.0
        else:
            types = F.gumbel_softmax(type_logit, tau=self.tau, hard=True)[..., 1]  # [B,N]

        # signed weights: presyn inhibitory -> negative (t[b,i] row-wise)
        t_row = types.view(*types.shape, 1).expand_as(edges)  # [B,N,N]
        signed = weights * (1.0 - 2.0 * t_row)
        # lognormal weight jitter, masked to sampled edges only
        w_jit = (torch.rand_like(edges) * 0.5 + 0.75)  # ~U[0.75,1.25]
        if greedy:
            w_jit = w_jit.zero_() + 1.0  # deterministic weights
        signed = signed * w_jit * edges.detach().clamp(0, 1)

        return {
            "edges": edges,
            "hard": hard,
            "signed": signed,
            "types": types,
            "log_prob": log_prob,
            "ent_edge": ent_edge,
            "p_edge": p_edge,          # true Bernoulli edge probs (differentiable)
            "logits": logits,          # raw edge logits (pre-Gumbel)
            "mask": mask,              # 1 - eye (diagonal exclusion)
            "z": z,
            "p_assign": p_assign,
            "size_logit": size_logit,
            "type_logit": type_logit,
            "sim": sim,
            "value": value,  # [B] critic baseline V(z) for PPO-style advantage
        }

    def log_prob_of(self, out: dict) -> torch.Tensor:
        """Sum of per-edge Bernoulli log-probs -> [B] scalar."""
        return out["log_prob"].sum(dim=(1, 2))

    def structure_loss(self, out: dict, density_target: float = 0.12,
                       ei_ratio_target: float = 4.0, mod_within_target: float = 0.6,
                       recur_e_target: float = 0.3) -> torch.Tensor:
        """Differentiable structural objectives on the *soft* expected graph.

        The graph-metric reward terms (density, degree heavy-tail, modularity)
        are computed in env.py on the *discrete* sampled graph — under REINFORCE
        each is a high-variance Monte-Carlo estimate over N² Bernoulli edges, so
        their gradient is noise. Here we compute the *same* targets on p_edge
        (true edge probabilities) and p_assign (soft cluster assignment), which
        are differentiable: the policy gets a direct, low-variance gradient to
        shape structure, and REINFORCE is left to carry only the *dynamics*
        terms (propagation/firing) that genuinely need sampling.

        Returns a [B] per-sample loss (mean over edges for density, scalars for
        the rest), to be combined with the policy-gradient loss in the trainer.
        """
        p = out["p_edge"]  # [B,N,N] true Bernoulli probs, off-diag already masked
        mask = out["mask"]  # [B? no, [N,N]] — broadcastable
        B, N, _ = p.shape
        n_pairs = N * (N - 1)
        device = p.device

        # targets may be per-sample ([B]) or scalar (broadcast)
        def T(v, default):
            if v is None:
                return torch.full((B,), default, device=device)
            t = torch.as_tensor(v, dtype=p.dtype, device=device)
            return t if t.ndim > 0 else t.expand(B)

        dt = T(density_target, 0.12)
        et = T(ei_ratio_target, 4.0)
        mt = T(mod_within_target, 0.6)

        # 1) density: expected edge count / N(N-1) -> target. Anti-collapse:
        #    penalizes both the ER-dense seizure regime and the min-degree dead one.
        exp_density = p.sum(dim=(1, 2)) / n_pairs
        L_density = (exp_density - dt).pow(2)

        # 2) degree heavy-tail: expected out-degree per node, want high CV
        d = p.sum(dim=2)  # [B,N] expected out-degree
        d_mean = d.mean(dim=1)  # [B]
        d_std = d.std(dim=1)
        cv = d_std / (d_mean + 1e-6)  # [B]
        L_degcv = (cv - 1.0).pow(2)  # want CV≈1 (heavy-ish tail), penalize uniform

        # 3) modularity: expected fraction of edge weight inside communities.
        #    q = soft cluster assignment [B,N,K]; within_ij = q_i·q_j (0..1).
        #    Target 0.6 (not 0.8): a too-sharp target collapses the softmax to
        #    deterministic one-hot clusters, killing edge entropy and with it the
        #    REINFORCE exploration on the dynamics terms.
        q = out["p_assign"]
        within = torch.einsum("bik,bjk->bij", q, q)  # [B,N,N] same-community overlap
        edge_mass = p.sum(dim=(1, 2)) + 1e-6
        within_frac = (p * within).sum(dim=(1, 2)) / edge_mass  # [B]
        L_modular = (within_frac - mt).pow(2)  # want ≥60% of edges inside comms

        # 4) E/I balance: inhibitory fraction → target (~20% = ei_ratio 4).
        #    This is the *dominant* driver of the graded propagation signal dv:
        #    at the same density, 50/50 E/I gives dv<0 (prop=0) while 80/20 gives
        #    dv>+1.0 (prop=1). Without this term the policy stays
        #    inhibition-dominated and the dynamics reward is a constant floor.
        p_inh = F.softmax(out["type_logit"], dim=-1)[..., 1]  # [B,N] P(inhibitory)
        inh_frac = p_inh.mean(dim=1)  # [B]
        target_inh = 1.0 / (1.0 + et)  # ei_ratio -> inhibitory fraction
        L_ei = (inh_frac - target_inh).pow(2)

        # 5) output-group recurrent E-connectivity — the structural prerequisite
        #    for *self-sustaining* output dynamics (CPG rhythm / working-memory
        #    hold). Over the N² edge space the policy almost never opens the
        #    specific output→output excitatory edges by chance (v13-v22: the
        #    output merely follows the ON/OFF envelope, no recurrent loop — see
        #    [[sardana-universal-oscillator-collapse]]), so give a direct
        #    differentiable target on exactly those edges. One-sided (only below
        #    the floor): the dynamics reward then shapes how much recurrence is
        #    right — too much → output seizes both phases → dv<0 kills prop.
        n_out = max(1, N // 20)  # last 5% of neurons are the output probe group
        p_out = p[:, N - n_out:, N - n_out:]  # [B, n_out, n_out] output→output
        p_e_src = (1.0 - p_inh)[:, N - n_out:, None]  # [B, n_out, 1] P(src excitatory)
        recur_e = (p_out * p_e_src).mean(dim=(1, 2))  # [B] expected recurrent E density
        rt = T(recur_e_target, 0.3)
        L_recur = (rt - recur_e).clamp(min=0).pow(2)

        loss = L_density + 0.3 * L_degcv + 0.5 * L_modular + 2.0 * L_ei + 2.0 * L_recur
        return loss  # [B]

    @torch.no_grad()
    def to_csr(self, out: dict) -> list[dict]:
        """Convert batch outputs to CSR arrays for the Rust engine.

        Returns per-sample: row_ptr, col_idx, weights, delays, input_idx,
        output_idx (first/last 5% of neurons are input/output probe groups).
        """
        B, N = out["signed"].shape[0], out["signed"].shape[1]
        res = []
        for b in range(B):
            signed = out["signed"][b].detach().cpu().numpy()
            hard = out["hard"][b].detach().cpu().numpy()
            rows, cols = np.nonzero(hard)
            w = signed[rows, cols]
            # only materialize edges that actually have nonzero weight
            m = w != 0
            rows, cols, w = rows[m], cols[m], w[m]
            order = np.argsort(rows, kind="stable")
            rows, cols, w = rows[order], cols[order], w[order]
            row_ptr = np.zeros(N + 1, dtype=np.uint32)
            np.add.at(row_ptr, rows + 1, 1)
            row_ptr = np.cumsum(row_ptr)
            n_in = max(1, N // 20)
            n_out = max(1, N // 20)
            res.append(
                {
                    "row_ptr": row_ptr.tolist(),
                    "col_idx": cols.astype(np.uint32).tolist(),
                    "weights": w.astype(np.float32).tolist(),
                    "delays": np.ones_like(cols, dtype=np.uint8).tolist(),
                    "input_idx": list(range(n_in)),
                    "output_idx": list(range(N - n_out, N)),
                }
            )
        return res
