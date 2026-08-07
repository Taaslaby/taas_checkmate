"""Rust-engine environment + reward.

The reward is what makes this *different* from every prior model:

- old models were trained to predict tokens (tables / degree lists) — so they
  learned file format, not function;
- here the generated connectome is loaded into the Rust `ConnectomeBrain` and
  actually simulated. Reward comes from *dynamics*: does signal propagate,
  are firing rates healthy, does the network do something, is it stable.

## Propagation signal — output *membrane modulation*, not spikes

The LIF engine has a hard spiking cliff: below a critical weight*rate*fan-in
product nothing fires (dead), just above it everything fires (seizure).  That
binary landscape gives REINFORCE no gradient.  But the membrane potential is
*graded*: when the input group is driven, postsynaptic membrane depolarises
even if no spike crosses threshold.  We measure the mean output-group voltage
during ON windows minus during OFF windows — `dV`.  It rises smoothly with
connectivity and *collapses* at seizure (output fires in both phases), giving
a continuous, well-shaped reward:

    prop = min(1, dV / DV_FULL)      # DV_FULL mV of modulation = full credit

## Weight gain

The generator emits normalized weights ~0.5-1.5.  The LIF needs i_syn ~ 16+ mV
sustained to fire at all (`dv = i_syn·inv_tau_m·dt`, clamp ±100).  The env maps
generator weights into that range by scaling: `w_eff = w · GAIN`.  Keeping the
gain in the env (not the generator) lets the policy learn *structure* while the
env owns the physical scale.

Component rewards (each normalised ~[0,1]):
    propagation   — input drive depolarises output group (graded, subthreshold)
    firing        — mean firing rate in a healthy band (not dead, not seizure)
    diversity     — coefficient of variation of firing rates (heterogeneous)
    syn_density   — synapse count in a healthy range (anti-collapse)
    network       — clustering, modularity, degree heavy-tail (anti-ER)
"""

from __future__ import annotations

import numpy as np

from . import metrics

# healthy firing-rate band (Hz) for LIF with these params
FR_LOW, FR_HIGH = 2.0, 60.0
# healthy synapse count per neuron (anti-collapse: old models collapsed to ~1)
SYN_PER_NEURON_LO, SYN_PER_NEURON_HI = 4, 40
# target density of the network
DENSITY_TARGET = 0.05
# output depolarisation (mV) that earns full propagation credit
DV_FULL = 0.5
# ceiling for the membrane osc metric. Verified (v24 probe chain): with the
# engine's i_syn clamp (±100) + slow stochastic input, an ALIVE output's OFF
# membrane is dominated by the ON-drive envelope, so osc caps at ~0.40-0.42
# (a genuine CPG rhythm would need the clamp removed — a Rust change). Credit
# is min(1, osc/OSC_CEIL): the envelope IS this engine's best rhythm, so it
# must not handicap the osc-weighted categories (v23: motor/sonar/motion
# earned beh≈0.40 while persist cats earned 0.69, and the policy killed
# auditory rather than pay for an unreachable 5.0-weight osc).
OSC_CEIL = 0.40

# Per-category reward & structure profiles. The reward is *spec-dependent*:
# different tasks want different connectomes (motion → dense propagation,
# olfactory → sparse modular MBON, motor → oscillating CPG, homeostasis →
# stable firing). This is what makes the policy learn to separate z(spec);
# a spec-agnostic reward has no incentive and collapses to one shared graph.
#   struct = {density, ei_ratio, mod_within} targets for the differentiable loss
#   w      = dynamics-reward weights {propagation, firing, diversity}
#   net_w  = weight on the network (structure) term in the total reward
SPEC_PROFILES: dict[str, dict] = {
    # behavioral task terms (osc / persist / stab) are rewarded per category:
    #   motor_gait   → oscillation (CPG rhythm in OFF),        osc weight 5.0
    #   *_memory     → persistence (working-memory, OFF active) persist 5.0
    #   homeostasis  → temporal stability (steady window FR),  stab 5.0
    #   auditory/echo→ oscillation (rhythmic filtering)        osc 2.0-3.0
    # The reward is split into two NORMALLY-SEPARATE credits that ADD (see
    # dynamics_reward_from): `shared` (propagation/firing/diversity/alive,
    # normalised among themselves) rewards a living output — a dead output gets
    # ~0 and can never collect behavioural credit because it is alive-gated.
    # `behavioral` (osc/persist/stab, also alive-gated, normalised among
    # themselves) is worth a full extra point. Because behavioural weights are
    # per-category DOMINANT, v12-v14's collapse — every category → the same
    # universal oscillator (the shared terms are whole-net → cat-independent,
    # so the sum-reward's optimum is one shared regime) — is broken: each
    # category must specialise into its own dynamics to collect the full +1.
    "motion_detection": dict(struct=dict(density=0.15, ei_ratio=4.0, mod_within=0.55),
                             w=dict(propagation=1.5, firing=1.2, diversity=0.5,
                                    osc=1.5, persist=0.0, stab=0.3, alive=1.0), net_w=0.8),
    "olfactory_memory": dict(struct=dict(density=0.07, ei_ratio=3.0, mod_within=0.70),
                             w=dict(propagation=0.8, firing=0.8, diversity=0.8,
                                    osc=0.2, persist=5.0, stab=0.2, alive=1.0), net_w=1.6),
    "sonar_echo":       dict(struct=dict(density=0.12, ei_ratio=4.5, mod_within=0.55),
                             w=dict(propagation=1.5, firing=1.0, diversity=0.6,
                                    osc=2.5, persist=0.0, stab=0.3, alive=1.0), net_w=1.0),
    "motor_gait":       dict(struct=dict(density=0.13, ei_ratio=4.0, mod_within=0.50),
                             w=dict(propagation=0.8, firing=1.5, diversity=1.0,
                                    osc=5.0, persist=0.0, stab=0.2, alive=1.0), net_w=0.7),
    "reward_learning":  dict(struct=dict(density=0.10, ei_ratio=3.5, mod_within=0.60),
                             w=dict(propagation=0.8, firing=1.0, diversity=1.2,
                                    osc=0.1, persist=5.0, stab=0.2, alive=1.0), net_w=1.2),
    "visual_recognition": dict(struct=dict(density=0.10, ei_ratio=3.5, mod_within=0.65),
                             w=dict(propagation=1.0, firing=0.8, diversity=0.6,
                                    osc=0.3, persist=3.0, stab=0.2, alive=1.0), net_w=1.4),
    "auditory":         dict(struct=dict(density=0.11, ei_ratio=4.0, mod_within=0.55),
                             w=dict(propagation=1.2, firing=1.0, diversity=1.0,
                                    osc=3.0, persist=0.2, stab=0.2, alive=1.0), net_w=0.9),
    "homeostasis":      dict(struct=dict(density=0.10, ei_ratio=5.0, mod_within=0.55),
                             w=dict(propagation=0.5, firing=1.5, diversity=0.6,
                                    osc=0.1, persist=0.2, stab=5.0, alive=1.0), net_w=0.8),
}


def profile_for(cat: str) -> dict:
    return SPEC_PROFILES.get(cat, dict(
        struct=dict(density=0.12, ei_ratio=4.0, mod_within=0.60),
        w=dict(propagation=2.0, firing=1.0, diversity=0.5, alive=0.8), net_w=1.0))


class ConnectomeEnv:
    """Wraps the Rust ConnectomeBrain (built lazily via the module)."""

    def __init__(
        self,
        n_steps: int = 300,
        drive: float = 60.0,
        gain: float = 50.0,
        onoff: int = 50,
        inrate: float = 0.5,
    ):
        self.n_steps = n_steps
        self.drive = drive
        self.gain = gain
        self.onoff = onoff
        self.inrate = inrate
        # import lazily so the module can be imported without the rust .pyd
        from taas_checkmate.pillars import ConnectomeBrain

        self._Brain = ConnectomeBrain

    def simulate(self, csr: dict) -> dict:
        """Run one generated connectome through the Rust engine.

        ON/OFF protocol: the input group receives stochastic drive during ON
        windows, silence during OFF windows.  Returns dynamics: output-group
        membrane modulation dV (graded propagation signal), mean firing rate,
        spike counts, stability, dead-fraction.
        """
        # scale generator weights into the LIF firing range
        gain = self.gain
        w_row = np.asarray(csr["weights"], dtype=np.float32) * gain
        col_idx = np.asarray(csr["col_idx"], dtype=np.uint32)
        row_ptr = np.asarray(csr["row_ptr"], dtype=np.uint32)
        delays = np.asarray(csr["delays"], dtype=np.uint8)
        brain = self._Brain(row_ptr.tolist(), col_idx.tolist(), w_row.tolist(), delays.tolist())
        n = brain.n
        # configure a plausible LIF: dt=0.5, tau_m=10ms
        brain.dt = 0.5
        brain.inv_tau_m = 1.0 / 10.0
        brain.v_rest = -70.0
        brain.v_thresh = -54.0
        brain.v_reset = -70.0
        brain.tau_ref = 2.0

        out_idx = np.array(csr["output_idx"], dtype=np.uint32)
        inp_idx = np.array(csr["input_idx"], dtype=np.uint32)
        n_in = len(inp_idx)
        rng = np.random.default_rng(0)

        on_v = np.zeros(n, dtype=np.float64)  # output v accumulator (ON)
        off_v = np.zeros(n, dtype=np.float64)  # output v accumulator (OFF)
        on_tot = np.zeros(n, dtype=np.float64)
        off_tot = np.zeros(n, dtype=np.float64)
        out_spk_t = np.zeros(self.n_steps, dtype=np.float64)  # per-step output spikes
        v_out_t = np.zeros(self.n_steps, dtype=np.float64)  # per-step output MEMBRANE (graded)
        on_n = 0
        off_n = 0

        stim = np.zeros(n, dtype=np.float32)
        for t in range(self.n_steps):
            if (t // self.onoff) % 2 == 0:  # ON window
                stim = np.zeros(n, dtype=np.float32)
                for k in inp_idx:
                    if rng.random() < self.inrate:
                        stim[k] = self.drive
                on_n += 1
            else:  # OFF window
                stim = np.zeros(n, dtype=np.float32)
                off_n += 1
            brain.inject(list(inp_idx), list(stim[inp_idx]))
            brain.step()
            sp = brain.spikes.astype(np.float64)
            v = brain.v.astype(np.float64)
            out_spk_t[t] = sp[out_idx].sum()
            v_out_t[t] = v[out_idx].mean()
            if (t // self.onoff) % 2 == 0:
                on_tot += sp
                on_v += v
            else:
                off_tot += sp
                off_v += v

        # graded propagation: output depolarises when input is driven
        v_out_on = on_v[out_idx].mean() / max(1, on_n)
        v_out_off = off_v[out_idx].mean() / max(1, off_n)
        dv = float(v_out_on - v_out_off)

        # spike-based stats (secondary)
        on_fr = on_tot / max(1, on_n) / 0.5 * 1000.0  # Hz (dt=0.5ms)
        n_out = len(out_idx)
        out_spikes = float(on_tot[out_idx].sum())
        out_frac = out_spikes / (max(1, on_n) * max(1, n_out))
        dead = float((on_fr < 0.5).mean())
        mean_fr = float(on_fr.mean())
        cv_fr = float(on_fr.std() / (mean_fr + 1e-9))

        # ---- task-behaviour metrics (per-window / per-step) ----
        # per-window output firing rate (Hz); dt=0.5ms → /0.5*1000 = *2000
        T = self.n_steps
        win = np.arange(T) // self.onoff
        n_win = int(win.max()) + 1
        win_is_on = (np.arange(n_win) % 2 == 0)  # per-window ON/OFF (not per-step)
        win_out_sum = np.bincount(win, weights=out_spk_t, minlength=n_win)
        win_out_len = np.bincount(win, minlength=n_win)
        win_out_fr = win_out_sum / np.maximum(1, win_out_len) / 0.5 * 1000.0

        on_out_fr = float(win_out_fr[win_is_on].mean()) if win_is_on.any() else 0.0
        off_out_fr = float(win_out_fr[~win_is_on].mean()) if (~win_is_on).any() else 0.0
        # ---- SPIKE-based task metrics (kept for diagnostics; the reward uses
        #      the membrane versions below, because the output group rarely
        #      spikes → spike-based osc/persist/stab are ~0 and give no gradient).
        # working-memory persistence: output stays active after ON drive
        persist_spk = float(off_out_fr / (on_out_fr + 1e-6))
        # temporal stability: CV of per-window output FR (low = steady firing)
        win_cv = float(win_out_fr.std() / (win_out_fr.mean() + 1e-9))
        stab_spk = float(1.0 / (1.0 + win_cv))
        # oscillation: autocorrelation of OFF-window output spikes — the
        # network's *own* rhythm (input drive is silent in OFF).
        step_is_on = ((np.arange(T) // self.onoff) % 2) == 0
        x = out_spk_t[~step_is_on].astype(np.float64)
        x = x - x.mean()
        var = float((x * x).mean())
        osc_spk = 0.0
        if var > 1e-9 and len(x) > 10:
            ac = [float((x[:-l] * x[l:]).mean() / var) for l in range(2, min(40, len(x) // 2))]
            osc_spk = max(0.0, max(ac)) if ac else 0.0
        osc_spk = float(min(1.0, osc_spk))

        # ---- MEMBRANE-based task metrics -------------------------------------
        # The output membrane V_out moves smoothly even when it never crosses
        # threshold, so these are dense, reachable targets for REINFORCE (this
        # is what makes per-category behaviour actually learnable). Baseline =
        # engine v_rest; lift = membrane elevation above it / deflection.
        V_BASE = -70.0
        m_on = float(v_out_t[step_is_on].mean()) if step_is_on.any() else V_BASE
        m_off = float(v_out_t[~step_is_on].mean()) if (~step_is_on).any() else V_BASE
        lift_on = max(0.0, m_on - V_BASE)
        lift_off = max(0.0, m_off - V_BASE)
        # persistence: how much of the driven lift the output *holds* in OFF.
        persist = float(min(2.0, lift_off / (lift_on + 1e-9)))
        # stability: CV of per-window membrane lift (low = steady homeostatic
        # activity). Dead nets give constant V≈v_rest → ~0 lift → trivial CV=0;
        # the 'alive' reward term below disambiguates that degenerate case.
        win_mem = np.array([v_out_t[a:b].mean() for a, b in
                            [(w * self.onoff, min(T, (w + 1) * self.onoff))
                             for w in range(n_win)]])
        lift_win = np.clip(win_mem - V_BASE, 0.0, None)
        win_cv_mem = float(lift_win.std() / (lift_win.mean() + 1e-9))
        stab = float(1.0 / (1.0 + win_cv_mem))
        # oscillation: how deeply the OFF-membrane autocorrelation goes NEGATIVE
        # (per OFF window, averaged — robust to ON-window boundaries). A monotone
        # envelope-following *decay* is positively autocorrelated at every shift
        # (curve 0.94→0.12; that was the "universal oscillator" of v13-v18,
        # osc≈0.86 from max-autocorr). A real rhythm MUST reverse phase: its
        # autocorr goes negative near lag ≈ period/2. So osc = depth of the most
        # negative autocorr lag. Continuous + dense (a dead network →0, an alive
        # envelope →~0.4, a genuine CPG →0.8-1.0), so REINFORCE can climb it —
        # the v19 turn-count gate (floor at turns≥4) sat exactly on the
        # reachable boundary and gave 0.00 despite real partial oscillation.
        def _osc_negac(seg):
            y = np.asarray(seg, dtype=np.float64)
            y = y - y.mean()
            v = float((y * y).mean())
            if v <= 1e-9 or len(y) <= 10:
                return 0.0
            ac = [float((y[:-l] * y[l:]).mean() / v) for l in range(1, min(40, len(y) // 2))]
            return float(np.clip(-min(ac), 0.0, 1.0))
        off_segs = [v_out_t[w * self.onoff:min(T, (w + 1) * self.onoff)]
                    for w in range(n_win) if not step_is_on[w * self.onoff]]
        oscs = [_osc_negac(sg) for sg in off_segs]
        osc = float(np.mean(oscs)) if oscs else 0.0
        osc = float(min(1.0, osc))
        return {
            "dv": dv,
            "out_frac": out_frac,
            "mean_fr": mean_fr,
            "cv_fr": cv_fr,
            "dead_frac": dead,
            "fr": on_fr,
            # task behaviour (membrane -> reward; spike versions for diagnostics)
            "osc": osc,
            "persist": persist,
            "stab": stab,
            "osc_spk": osc_spk,
            "persist_spk": persist_spk,
            "stab_spk": stab_spk,
            "on_out_fr": on_out_fr,
            "off_out_fr": off_out_fr,
            "v_out_t": v_out_t,
        }


def dynamics_reward_from(csr: dict, dyn: dict, w: dict | None = None, cat: str | None = None) -> dict:
    """Reward from *dynamics only* (propagation, firing, diversity).

    The structural terms (syn_density, network) are optimised *differentiably*
    on the soft expected graph in generator.structure_loss, so they are removed
    from the REINFORCE signal — otherwise REINFORCE would try to learn them
    through N² high-variance Bernoulli samples (which is why training stalled).
    This leaves the policy gradient to carry the genuinely-sampled dynamics.

    `cat` selects the per-category dynamics weights (spec-dependent reward).
    """
    if w is None:
        w = dict(propagation=2.0, firing=1.0, diversity=0.5)
    if cat:
        w = profile_for(cat)["w"]

    dv = dyn["dv"]
    prop = float(min(1.0, max(0.0, dv / DV_FULL)))
    prop += 0.5 * min(1.0, dyn["out_frac"] / 0.05)
    prop = float(min(1.0, prop))

    mfr = dyn["mean_fr"]
    # OUTPUT-scoped firing health: shared credit must require the output group
    # to actually fire, closing the "quiet-output" escape of v19-v20 where a
    # policy killed the output and farmed whole-net firing/diversity (satisfied
    # by the healthy rest of the net). Target ~50Hz = the alive regime's own
    # output rate (v17: on_out_fr≈53Hz) — reachable, and a dead output (0Hz)
    # scores ≈0.04. Whole-net mean_fr kept as a soft floor.
    ofr = float(dyn.get("on_out_fr", 0.0))
    firing = float(np.exp(-0.5 * ((ofr - 50.0) / 20.0) ** 2)) if 0 <= ofr < 200 else 0.0
    firing = max(firing, 0.05 * float(np.exp(-0.5 * ((mfr - 15.0) / 10.0) ** 2)))
    firing *= (1.0 - dyn["dead_frac"])

    cv = dyn["cv_fr"]
    diversity = 1.0 / (1.0 + abs(cv - 1.0))

    # ---- task-behaviour terms (per-category weights, default 0) ----
    osc = float(dyn.get("osc", 0.0))        # membrane oscillation (CPG rhythm)
    persist = float(dyn.get("persist", 0.0))  # membrane OFF/ON hold (working memory)
    stab = float(dyn.get("stab", 0.0))      # membrane per-window steadiness
    # 'alive' — the output group must actually respond (fire) at all. Behavioural
    # terms on a dead output are trivially degenerate (constant V≈v_rest → CV=0
    # → stab=1), so liveness separates "quiet & stable" from "stable & active".
    # Threshold 0.005 (v22): the alive regime's output fires ~4 spikes per ON
    # (out_frac≈0.0044, ofr≈53Hz) — genuinely alive, but at /0.02 it scored only
    # alive≈0.22, capping ALL behavioural credit at ×0.22 → behaviour was ~10% of
    # reward and REINFORCE never specialised (v21 collapse: binary 4-alive/4-dead,
    # osc∈{0.26,0.40} = two shared graphs, the four dead cats were exactly the
    # high-weight behavioural ones). /0.005 unlocks alive≈0.88 for the same honest
    # firing, making per-category behaviour a first-class (~30%) gradient.
    alive = float(min(1.0, dyn.get("out_frac", 0.0) / 0.005))
    # working-memory: output stays ~half of driven level in OFF (reverberation
    # without seizure); 0 = responds only during ON, 1 = equal response in OFF.
    # Target 0.50 (v24 recalibration, KEPT over v25): with the i_syn clamp, an
    # ALIVE output firing for its firing-credit caps persist at ~0.48 (v23 greedy
    # 0.43-0.51); the old 0.6 target was ~1σ above the reachable ceiling for
    # alive outputs, so even a perfect memory earned only persist_r≈0.71 and the
    # policy had no reason to hold (and dead outputs earn zero behavioural credit
    # via alive). v25 tried 0.60 (greedy showed alive cats at 0.57-0.73) but that
    # made the universal envelope-cheap (pers0.74→persist_r0.65) and the
    # best-beh checkpoint REGRESSED to the v13-v18 uniform oscillator; 0.50 keeps
    # the collapse penalised (persist_r≈0.28) so the policy must specialise.
    persist_r = float(np.exp(-0.5 * ((persist - 0.50) / 0.15) ** 2)) if 0.0 <= persist <= 2.0 else 0.0
    # stability credit: sharp gaussian around the homeostatic ideal (steady
    # per-window membrane). Target 0.70 (v24, KEPT over v25): the alive regime
    # (which must fire to pass the alive gate) has window-CV → stab≈0.66-0.68
    # (v23 greedy 0.66), so the old 0.85 target was structurally unreachable and
    # homeostasis (stab weight 5.0) was the LOWEST-behavioural alive category
    # (beh≈0.45) — the reward handicapped the very category it claimed to
    # specialise. v25's 0.68 let the universal envelope (stab0.80) score high;
    # 0.70 keeps the top of the reachable band as the target.
    stab_r = float(np.exp(-0.5 * ((stab - 0.70) / 0.15) ** 2)) if 0.0 <= stab <= 2.0 else 0.0
    # osc credit: min(1, osc/OSC_CEIL) — the envelope IS the engine's best rhythm
    # (see OSC_CEIL), so a driven alive output (osc≈0.40) earns full credit; a
    # flat membrane (osc→0) earns none. Replaces the raw osc in the behavioural
    # sum, which otherwise capped osc-weighted cats at beh≈0.40 (v23).
    osc_r = float(min(1.0, osc / OSC_CEIL))

    # total = shared + behavioral, each normalised separately (see SPEC_PROFILES
    # header). `shared` rewards a living output; `behavioral` adds a full extra
    # point gated by liveness — a category can only collect its per-cat
    # behaviour by keeping the output alive AND producing its own dynamics
    # (breaks the universal-oscillator collapse of v12-v14, where whole-net
    # shared terms rewarded every category identically and the sum-optimum was
    # one shared regime). The alive-gate keeps a dead net from gaming stab=1.0.
    shared = (
        w.get("propagation", 0.0) * prop
        + w.get("firing", 0.0) * firing
        + w.get("diversity", 0.0) * diversity
        + w.get("alive", 0.0) * alive
    ) / max(1e-6, sum(w.get(k, 0.0) for k in ("propagation", "firing", "diversity", "alive")))
    beh_sum = sum(w.get(k, 0.0) for k in ("osc", "persist", "stab"))
    behavioral = 0.0
    if beh_sum > 0:
        behavioral = (alive / max(1e-6, beh_sum)) * (
            w.get("osc", 0.0) * osc_r
            + w.get("persist", 0.0) * persist_r
            + w.get("stab", 0.0) * stab_r
        )
    total = shared + behavioral
    return {
        "total": float(total), "prop": float(prop), "firing": float(firing),
        "diversity": float(diversity), "osc": osc, "persist": persist, "stab": stab,
        "alive": alive, "behavioral": float(behavioral),
    }


def reward_from(csr: dict, dyn: dict, w: dict | None = None, cat: str | None = None) -> dict:
    """Composite reward from dynamics + graph metrics.

    w overrides component weights. cat selects the per-category profile (both
    dynamics weights and the network-term weight), making the reward
    spec-dependent. Returns {total, ...components, ...metrics}.
    """
    if cat:
        p = profile_for(cat)
        w = dict(propagation=p["w"]["propagation"], firing=p["w"]["firing"],
                 diversity=p["w"]["diversity"], syn_density=1.0, network=p["net_w"],
                 osc=p["w"].get("osc", 0.0), persist=p["w"].get("persist", 0.0),
                 stab=p["w"].get("stab", 0.0), alive=p["w"].get("alive", 0.0))
    elif w is None:
        w = {
            "propagation": 2.0,
            "firing": 1.0,
            "diversity": 0.5,
            "syn_density": 1.0,
            "network": 1.0,
            "osc": 0.0,
            "persist": 0.0,
            "stab": 0.0,
            "alive": 0.0,
        }

    gm = metrics.graph_metrics(csr)
    n = gm["n"]

    # 1) propagation — output *membrane* responds to input drive (graded).
    #    dV>0 = signal flows subthreshold; collapse to ~0/negative at seizure.
    dv = dyn["dv"]
    prop = float(min(1.0, max(0.0, dv / DV_FULL)))
    # small bonus for actually spiking the output on top of the graded signal
    prop += 0.5 * min(1.0, dyn["out_frac"] / 0.05)
    prop = float(min(1.0, prop))

    # 2) firing health — OUTPUT-scoped (closes the quiet-output escape of v19-v20:
    #    whole-net mean_fr was satisfied by the healthy rest of the net even when
    #    the output group was dead, letting the policy farm shared credit without
    #    a live output). Target ~50Hz = the alive regime's own output rate
    #    (v17: on_out_fr≈53Hz), so it is reachable AND decisive. Whole-net mean_fr
    #    kept as a soft floor only.
    mfr = dyn["mean_fr"]
    ofr = float(dyn.get("on_out_fr", 0.0))
    firing = float(np.exp(-0.5 * ((ofr - 50.0) / 20.0) ** 2)) if 0 <= ofr < 200 else 0.0
    firing = max(firing, 0.05 * float(np.exp(-0.5 * ((mfr - 15.0) / 10.0) ** 2)))
    # dead networks are worst
    firing *= (1.0 - dyn["dead_frac"])

    # 3) diversity — want heterogeneous rates (cv ~ 0.5-1.5), not all-same
    cv = dyn["cv_fr"]
    diversity = 1.0 / (1.0 + abs(cv - 1.0))

    # 4) synapse density anti-collapse
    spp = gm["n_synapses"] / max(1, n)
    if spp < SYN_PER_NEURON_LO:
        syn = spp / SYN_PER_NEURON_LO * 0.5  # half credit below healthy min
    elif spp > SYN_PER_NEURON_HI:
        syn = SYN_PER_NEURON_HI / spp * 0.5
    else:
        syn = 1.0

    # 5) network "looks real" — clustering, heavy-tail, modular, E/I balance.
    #    Each term is a monotone saturating credit toward a target value, so a
    #    SBM-ish graph (clust≈0.3, degcv≈1.0, mod≈0.3, ei≈4) earns near-full
    #    credit and the policy has a real gradient toward structure (old
    #    saturating-penalty form was ≈0.02 at mod=0.036 — invisible to
    #    REINFORCE). E/I balance is the dominant driver of dv (see generator):
    #    ei = E/I weight-sum ratio, want ~4 (80/20).
    net = 0.0
    net += 0.3 * min(1.0, gm["clustering"] / 0.3)
    net += 0.2 * min(1.0, gm["deg_cv"] / 1.0)
    net += 0.2 * min(1.0, gm["modularity"] / 0.3)
    net += 0.3 * min(1.0, gm["ei_ratio"] / 4.0)
    net = float(np.clip(net, 0.0, 1.0))

    # task-behaviour terms (oscillation / persistence / stability) — membrane
    # versions (dense, reachable), plus 'alive' (output actually fires).
    osc = float(dyn.get("osc", 0.0))
    persist = float(dyn.get("persist", 0.0))
    stab = float(dyn.get("stab", 0.0))
    alive = float(min(1.0, dyn.get("out_frac", 0.0) / 0.005))
    # sharp behavioural credits (see dynamics_reward_from): the shared oscillator
    # (persist≈0.30, stab≈0.62) must score ~0 so categories specialise to earn.
    # Targets recalibrated to the alive-output reachable envelope (v24): persist
    # 0.6→0.50 (alive output caps ~0.48), stab 0.85→0.70 (alive caps ~0.68);
    # osc now credit = min(1, osc/OSC_CEIL) so the osc-weighted categories can
    # actually collect their 5.0 weight (the envelope is the engine's best rhythm).
    persist_r = float(np.exp(-0.5 * ((persist - 0.50) / 0.15) ** 2)) if 0.0 <= persist <= 2.0 else 0.0
    stab_r = float(np.exp(-0.5 * ((stab - 0.70) / 0.15) ** 2)) if 0.0 <= stab <= 2.0 else 0.0
    osc_r = float(min(1.0, osc / OSC_CEIL))

    # total = shared + behavioral, each normalised separately (see the same in
    # dynamics_reward_from): structural+health credit stays alive-rewarding,
    # behavioural credit is a full extra point gated by liveness so categories
    # must specialise (alive AND their own dynamics) to collect it.
    shared = (
        w["propagation"] * prop
        + w["firing"] * firing
        + w["diversity"] * diversity
        + w["syn_density"] * syn
        + w["network"] * net
        + w.get("alive", 0.0) * alive
    ) / max(1e-6, sum(w.get(k, 0.0) for k in
                      ("propagation", "firing", "diversity", "syn_density", "network", "alive")))
    beh_sum = sum(w.get(k, 0.0) for k in ("osc", "persist", "stab"))
    behavioral = 0.0
    if beh_sum > 0:
        behavioral = (alive / max(1e-6, beh_sum)) * (
            w["osc"] * osc_r
            + w["persist"] * persist_r
            + w["stab"] * stab_r
        )
    total = shared + behavioral

    return {
        "total": float(total),
        "prop": float(prop),
        "firing": float(firing),
        "diversity": float(diversity),
        "syn": float(syn),
        "network": float(net),
        "osc": osc,
        "persist": persist,
        "stab": stab,
        "alive": alive,
        "behavioral": float(behavioral),
        "dv": float(dv),
        "mean_fr": float(mfr),
        "dead_frac": float(dyn["dead_frac"]),
        **gm,
    }


def simulate_and_reward(csr: dict, env: ConnectomeEnv, cat: str | None = None) -> dict:
    dyn = env.simulate(csr)
    return reward_from(csr, dyn, cat=cat)
