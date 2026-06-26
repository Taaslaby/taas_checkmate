#!/usr/bin/env python3
"""
RSA Fidelity — проверка, работает ли мозг осмысленно.

Адаптация подхода connectome-fidelity (michaela10c) для V2 full-brain LIF.

Метод:
  1. Предъявляем мозгу N стимулов (стена на разных расстояниях)
  2. Записываем population vectors (spike rates всех нейронов)
  3. Строим RDM (6×6 matrix of cosine distances между условиями)
  4. Сравниваем CC RDM vs Random-baseline RDM (sign-preserving GLOBAL weight shuffle)
  5. Если Spearman r < 0.7 → коннектом навязывает структуру (мозг работает)
  6. Если Spearman r > 0.85 → активность случайна

Исправления после code review:
  ✅ Brain state reset между условиями (washout + save/restore v)
  ✅ Векторизованный numpy для spike counting (не Python loop по 108K)
  ✅ DNg100 + CB0013 injection для корректного dynamic regime
  ✅ Global shuffle вместо per-neuron (sign-preserving, network-wide)

Запуск:
  cd taas_checkmate && python3 fly/scripts/rsa_fidelity.py --model lif

Зависимости:
  pip install scipy matplotlib numpy
"""

import sys
import os
import time
import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, kendalltau
import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt

FLY_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FLY_DIR / "scripts"))

# ── REPRODUCIBILITY ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── STIMULUS CONDITIONS: looming wall distances ──────────────────────────────
LOOMING_DISTANCES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]  # meters, 0.0 = no wall
N_CONDITIONS = len(LOOMING_DISTANCES)
CONDITION_LABELS = ["No wall" if d == 0.0 else f"d={d:.2f}m" for d in LOOMING_DISTANCES]

# ── DNg100/CB0013 injection (matches V2VNCController) ────────────────────────
DNg100_CURRENT = 12.5   # pA
CB0013_CURRENT = 5.0    # pA

# Number of washout batches before each condition to let brain settle
N_WASHOUT = 3


# ═══════════════════════════════════════════════════════════════════════════════
#  1. RSA Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_rdm(pop_matrix, metric="cosine"):
    """Build RDM from (n_stimuli, n_neurons) population matrix."""
    pop_matrix = np.nan_to_num(pop_matrix, nan=0.0, posinf=1e3, neginf=-1e3)
    if metric == "cosine":
        norms = np.linalg.norm(pop_matrix, axis=1, keepdims=True)
        zero_norm = (norms < 1e-10).flatten()
        if np.any(zero_norm):
            print(f"    ⚠ {zero_norm.sum()} zero-norm vectors — adding epsilon")
        pop_matrix = pop_matrix + 1e-10

    n = pop_matrix.shape[0]
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if metric == "cosine":
                    rdm[i, j] = cosine(pop_matrix[i], pop_matrix[j])
                else:
                    rdm[i, j] = np.linalg.norm(pop_matrix[i] - pop_matrix[j])
    return rdm


def rdm_similarity(rdm1, rdm2):
    """Spearman r + Kendall τ between upper triangles of two RDMs."""
    n = rdm1.shape[0]
    idx = np.triu_indices(n, k=1)
    r_s, p_s = spearmanr(rdm1[idx], rdm2[idx])
    r_k, p_k = kendalltau(rdm1[idx], rdm2[idx])
    return r_s, p_s, r_k, p_k


def permutation_test_rdm(rdm1, rdm2, n_permutations=10000, seed=SEED):
    """Stimulus-label permutation test (Nili et al. 2014)."""
    rng = np.random.default_rng(seed)
    n = rdm1.shape[0]
    idx = np.triu_indices(n, k=1)

    obs_r, _ = spearmanr(rdm1[idx], rdm2[idx])
    obs_tau, _ = kendalltau(rdm1[idx], rdm2[idx])

    null_r = np.zeros(n_permutations)
    null_tau = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        rdm2_perm = rdm2[np.ix_(perm, perm)]
        null_r[i], _ = spearmanr(rdm1[idx], rdm2_perm[idx])
        null_tau[i], _ = kendalltau(rdm1[idx], rdm2_perm[idx])

    p_r = np.mean(null_r >= obs_r)
    p_tau = np.mean(null_tau >= obs_tau)
    return obs_r, p_r, obs_tau, p_tau, null_r, null_tau


def global_sign_preserving_shuffle(values, rng=None):
    """GLOBAL sign-preserving shuffle of connectome weights.

    Shuffles absolute values ACROSS ALL SYNAPSES (network-wide), then
    reassigns preserving original signs. This is a STRONGER baseline
    than per-neuron shuffle because it breaks per-neuron weight distributions.

    This matches the 'full_shiu' approach from connectome-fidelity:
    - Preserves global E/I count
    - Preserves network architecture (connectivity pattern)
    - Preserves per-synapse sign
    - Breaks ALL weight magnitude structure (both per-neuron and network-wide)

    Args:
        values: CSR values array (n_synapses,) to shuffle
        rng: numpy random generator

    Returns:
        shuffled_values: (n_synapses,) sign-preserving global shuffle
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    signs = np.sign(values)
    abs_vals = np.abs(values)
    perm = rng.permutation(len(abs_vals))
    return signs * abs_vals[perm]


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Brain state management
# ═══════════════════════════════════════════════════════════════════════════════

def save_brain_state(brain):
    """Save LIF brain state for later restoration."""
    return {
        'v': brain.v.copy(),
        'I_buf': brain.I_buf.copy() if hasattr(brain, 'I_buf') else None,
        'spikes': brain.spikes.copy() if hasattr(brain, 'spikes') else None,
    }


def restore_brain_state(brain, state):
    """Restore LIF brain state from saved snapshot."""
    brain.v[:] = state['v']
    if state['I_buf'] is not None and hasattr(brain, 'I_buf'):
        brain.I_buf[:] = state['I_buf']
    if state['spikes'] is not None and hasattr(brain, 'spikes'):
        brain.spikes[:] = state['spikes']


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Population recording (vectorized — no Python loops over 108K neurons)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_spike_rates(brain, csr):
    """Extract per-neuron spike rates from brain's _last_spike_buf (VECTORIZED)."""
    spike_buf = getattr(brain, '_last_spike_buf', None)
    if spike_buf is None or len(spike_buf) == 0:
        return np.zeros(csr.n_neurons, dtype=np.float32)

    n_neurons = csr.n_neurons
    n_steps = brain._last_n_steps
    batch_time_s = n_steps * brain.dt_ms / 1000.0

    # VECTORIZED: sum spikes per neuron across all timesteps
    # spike_buf shape: (n_steps * n_neurons,)
    # Reshape to (n_steps, n_neurons) and sum along axis 0
    n_total = n_steps * n_neurons
    buf = np.asarray(spike_buf[:n_total], dtype=np.int32).reshape(n_steps, n_neurons)
    spikes_per_neuron = buf.sum(axis=0)  # (n_neurons,) — VECTORIZED

    # Normalize to firing rate (Hz), cap at 100Hz
    rates = spikes_per_neuron.astype(np.float32) / max(batch_time_s, 0.001)
    return np.clip(rates, 0.0, 100.0)


def inject_and_run(brain, csr, dist, n_washout=N_WASHOUT):
    """Inject DNg100/CB0013, set looming distance, run step_batch.

    Before recording, runs n_washout batches with NO stimulus to let
    the brain settle from the previous condition.

    Args:
        brain: FlyWireLIFBrain instance
        csr: FlyWireCSR instance
        dist: looming distance (0.0 = no wall)
        n_washout: number of washout batches before recording
    """
    # 1. Washout: no stimulus, just background drive
    if n_washout > 0:
        brain.set_sensory_state(0.0, 0.0)
        for _ in range(n_washout):
            brain.step_batch(0.0)

    # 2. Apply injection (same as V2VNCController does before step_batch)
    # DNg100 is found from csr.named_dns
    dng100_entry = csr.named_dns.get('DNg100')
    if dng100_entry is not None:
        _, dng100_idx = dng100_entry
        if dng100_idx is not None and dng100_idx < len(brain.external_I):
            brain.external_I[dng100_idx] += DNg100_CURRENT

    # CB0013: search type_map
    for rid, idx in csr.root_to_idx.items():
        typ = csr.type_map.get(rid, '')
        if typ == 'CB0013' and idx < len(brain.external_I):
            brain.external_I[idx] += CB0013_CURRENT
            break

    # 3. Set stimulus condition (looming wall)
    if dist > 0:
        brain.set_sensory_state(0.4 - dist, 0.0)  # fly at x=0.1, wall at x_max=0.4
    else:
        brain.set_sensory_state(0.0, 0.0)

    # 4. Run brain batch
    brain.step_batch(0.0)


def record_population_celltype(brain, csr, distances, n_repeats=2):
    """Record population vectors grouped by cell type.

    VECTORIZED spike extraction — NO Python loop over 108K neurons.

    Args:
        brain: FlyWireLIFBrain
        csr: FlyWireCSR
        distances: list of looming distances
        n_repeats: step_batch repeats per condition

    Returns:
        pop_matrix: (n_conditions, n_cell_types)
        cell_types: list of cell type names
    """
    # Build cell type → neuron indices mapping
    type_to_indices = defaultdict(list)
    for rid, idx in csr.root_to_idx.items():
        ct = csr.type_map.get(rid, 'unknown')
        ct_short = ct.split(' ')[0] if ct != 'unknown' else 'unknown'
        type_to_indices[ct_short].append(idx)

    cell_types = sorted(ct for ct, idxs in type_to_indices.items() if len(idxs) >= 5)
    print(f"  Cell types: {len(cell_types)} (≥5 neurons each)")

    n_cond = len(distances)
    n_types = len(cell_types)
    all_rates = np.zeros((n_cond, n_repeats, n_types), dtype=np.float32)

    for ci, dist in enumerate(distances):
        print(f"  Cond [{ci+1}/{n_cond}]: {CONDITION_LABELS[ci]}...", end=" ", flush=True)

        for ri in range(n_repeats):
            # Save brain state before this condition
            saved = save_brain_state(brain)

            inject_and_run(brain, csr, dist, n_washout=N_WASHOUT)
            rates = extract_spike_rates(brain, csr)

            # Average by cell type (VECTORIZED)
            for ti, ct in enumerate(cell_types):
                idxs = [i for i in type_to_indices[ct] if i < len(rates)]
                if idxs:
                    all_rates[ci, ri, ti] = float(np.mean(rates[idxs]))

            # Restore brain state for next repeat
            restore_brain_state(brain, saved)

        mean_rate = float(np.mean(all_rates[ci]))
        active = int(np.mean(np.sum(all_rates[ci] > 1.0, axis=1)))
        print(f"done. mean={mean_rate:.1f}Hz, active={active}/{n_types}")

    return all_rates.mean(axis=1), cell_types


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Main experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_rsa_experiment(model='lif', n_repeats=2):
    """Run RSA fidelity experiment: CC vs Random-baseline.

    Args:
        model: 'lif', 'izhi', 'rate', or 'psi'
        n_repeats: number of recordings per condition

    Returns:
        dict with all RDMs and statistics
    """
    print("═" * 60)
    print("  RSA FIDELITY — Full Brain Connectome")
    print("═" * 60)
    print(f"  Model:      {model}")
    print(f"  Stimuli:    {N_CONDITIONS} looming distances: {', '.join(CONDITION_LABELS)}")
    print(f"  Repeats:    {n_repeats} per condition")
    print(f"  Washout:    {N_WASHOUT} batches before each recording")
    print(f"  Injection:  DNg100 ({DNg100_CURRENT}pA) + CB0013 ({CB0013_CURRENT}pA)")
    print(f"  Shuffle:    GLOBAL sign-preserving (across all {N_CONDITIONS} synapses)")

    # ─── 4a. Load CSR + brain ───────────────────────────────────────────
    from flywire_brain import FlyWireCSR, FlyWireLIFBrain, FlyWireBrain, \
        FlyWireRateBrain, FlyWirePsiBrain

    BRAIN_CLASSES = {
        'lif': FlyWireLIFBrain,
        'izhi': FlyWireBrain,
        'rate': FlyWireRateBrain,
        'psi': FlyWirePsiBrain,
    }

    print("\n[1/4] Loading CSR...", flush=True)
    t0 = time.time()
    csr = FlyWireCSR()
    csr.load()
    print(f"  → {csr.n_neurons:,} neurons, {csr.n_synapses:,} syn "
          f"[{time.time()-t0:.1f}s]")

    BrainClass = BRAIN_CLASSES.get(model, FlyWireLIFBrain)

    print(f"\n[2/4] Creating CC {model.upper()} brain...", flush=True)
    t0 = time.time()
    brain_cc = BrainClass(csr)
    # Warmup: let brain settle before any recordings
    print("  Warming up (3 batches)...", end=" ", flush=True)
    for _ in range(3):
        brain_cc.set_sensory_state(0.0, 0.0)
        brain_cc.step_batch(0.0)
    print(f"done [{time.time()-t0:.1f}s]")

    print(f"\n[3/4] Recording CC population vectors...", flush=True)
    t0 = time.time()
    pop_cc, cell_types = record_population_celltype(
        brain_cc, csr, LOOMING_DISTANCES, n_repeats=n_repeats)
    print(f"  → Population matrix: {pop_cc.shape} [{time.time()-t0:.1f}s]")

    # ─── 4b. Random baseline ────────────────────────────────────────────
    print(f"\n[4/4] Creating RANDOM baseline (global sign-preserving shuffle)...", flush=True)
    t0 = time.time()

    # Save original, shuffle globally
    orig_values = csr.values.copy()
    csr.values = global_sign_preserving_shuffle(csr.values)

    # Create new brain with shuffled weights (fresh state)
    brain_rand = BrainClass(csr)
    print(f"  Random brain created [{time.time()-t0:.1f}s]")

    # Warmup random brain
    print("  Warming up (3 batches)...", end=" ", flush=True)
    for _ in range(3):
        brain_rand.set_sensory_state(0.0, 0.0)
        brain_rand.step_batch(0.0)
    print(f"done [{time.time()-t0:.1f}s]")

    print("  Recording RANDOM population vectors...", flush=True)
    t0 = time.time()
    pop_rand, _ = record_population_celltype(
        brain_rand, csr, LOOMING_DISTANCES, n_repeats=n_repeats)
    print(f"  → Population matrix: {pop_rand.shape} [{time.time()-t0:.1f}s]")

    # Restore original values
    csr.values = orig_values

    # ─── 4c. Build RDMs ─────────────────────────────────────────────────
    print("\n─── Building RDMs & computing statistics ───", flush=True)

    rdm_cc_cos = build_rdm(pop_cc, "cosine")
    rdm_rand_cos = build_rdm(pop_rand, "cosine")
    rdm_cc_euc = build_rdm(pop_cc, "euclidean")
    rdm_rand_euc = build_rdm(pop_rand, "euclidean")

    r_cos, p_cos, rk_cos, pk_cos = rdm_similarity(rdm_cc_cos, rdm_rand_cos)
    r_euc, p_euc, rk_euc, pk_euc = rdm_similarity(rdm_cc_euc, rdm_rand_euc)

    print(f"  Cosine RDM:    Spearman r = {r_cos:.3f}, p={p_cos:.4f}  "
          f"| Kendall τ = {rk_cos:.3f}, p={pk_cos:.4f}  [analytical]")
    print(f"  Euclidean RDM: Spearman r = {r_euc:.3f}, p={p_euc:.4f}  "
          f"| Kendall τ = {rk_euc:.3f}, p={pk_euc:.4f}  [analytical]")

    # ─── 4d. Permutation test ───────────────────────────────────────────
    print(f"\n─── PERMUTATION TEST (10,000 permutations, Nili et al. 2014) ───")

    (obs_r_cos, p_r_perm_cos, obs_tau_cos, p_tau_perm_cos,
     null_r_cos, null_tau_cos) = permutation_test_rdm(
        rdm_cc_cos, rdm_rand_cos, n_permutations=10000)

    print(f"  Cosine RDM:    Spearman r = {obs_r_cos:.3f}, p_perm = {p_r_perm_cos:.4f}  "
          f"| Kendall τ = {obs_tau_cos:.3f}, p_perm = {p_tau_perm_cos:.4f}")

    (obs_r_euc, p_r_perm_euc, obs_tau_euc, p_tau_perm_euc,
     null_r_euc, null_tau_euc) = permutation_test_rdm(
        rdm_cc_euc, rdm_rand_euc, n_permutations=10000)

    print(f"  Euclidean RDM: Spearman r = {obs_r_euc:.3f}, p_perm = {p_r_perm_euc:.4f}  "
          f"| Kendall τ = {obs_tau_euc:.3f}, p_perm = {p_tau_perm_euc:.4f}")

    # ─── 4e. Interpretation ─────────────────────────────────────────────
    print(f"\n─── INTERPRETATION ───")
    if p_r_perm_cos < 0.05:
        print(f"  ✅ CC vs Random геометрия РАЗЛИЧНА (p_perm = {p_r_perm_cos:.4f})")
        print(f"     → Коннектом навязывает структуру population activity")
    else:
        print(f"  ⚠ CC vs Random геометрия НЕ РАЗЛИЧНА (p_perm = {p_r_perm_cos:.4f})")
        print(f"     → Активность мозга может быть СЛУЧАЙНОЙ")

    if abs(r_cos) < 0.7:
        print(f"  |r| = {abs(r_cos):.3f} → сильное различие (мозг работает ✅)")
    elif abs(r_cos) < 0.85:
        print(f"  |r| = {abs(r_cos):.3f} → умеренное различие")
    else:
        print(f"  |r| = {abs(r_cos):.3f} → слабое различие (activity близка к случайной ⚠)")

    # ─── 4f. Plots ──────────────────────────────────────────────────────
    fig_dir = FLY_DIR / ".." / "figures"
    os.makedirs(str(fig_dir), exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.suptitle(f"RSA Fidelity — {model.upper()} Brain ({N_CONDITIONS} looming distances)\n"
                 f"CC vs Random (global sign-preserving weight shuffle)",
                 fontsize=10)
    for ax, rdm, title in zip(
        axes,
        [rdm_cc_cos, rdm_rand_cos, rdm_cc_euc, rdm_rand_euc],
        ["CC Cosine RDM", "Random Cosine RDM", "CC Euclidean RDM", "Random Euclidean RDM"]
    ):
        im = ax.imshow(rdm, cmap="viridis", vmin=0)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(N_CONDITIONS))
        ax.set_xticklabels(CONDITION_LABELS, fontsize=6, rotation=90)
        ax.set_yticks(range(N_CONDITIONS))
        ax.set_yticklabels(CONDITION_LABELS, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fname = f"rsa_fidelity_{model}.png"
    plt.savefig(str(fig_dir / fname), dpi=150, bbox_inches="tight")
    print(f"\n  Figure: {fname}")
    plt.close()

    # Null distribution plot
    fig_p, axes_p = plt.subplots(1, 2, figsize=(10, 3.5))
    fig_p.suptitle(f"Permutation null — {model.upper()} brain, {N_CONDITIONS} conditions", fontsize=10)
    for ax, null, obs, metric in zip(
        axes_p,
        [null_r_cos, null_tau_cos],
        [obs_r_cos, obs_tau_cos],
        ["Spearman r (cosine)", "Kendall τ (cosine)"]
    ):
        ax.hist(null, bins=50, color="steelblue", alpha=0.7, label="Null")
        ax.axvline(obs, color="red", linewidth=2, label=f"Obs={obs:.3f}")
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fname_p = f"rsa_fidelity_{model}_permtest.png"
    plt.savefig(str(fig_dir / fname_p), dpi=150, bbox_inches="tight")
    print(f"  Null dist:  {fname_p}")
    plt.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model:               {model}")
    print(f"  Stimuli:             {N_CONDITIONS} conditions")
    print(f"  Repeats/cond:        {n_repeats}")
    print(f"  Pop. dim:            {pop_cc.shape[1]} cell types")
    print(f"  Injection:           DNg100 ({DNg100_CURRENT}pA) + CB0013 ({CB0013_CURRENT}pA)")
    print(f"  Cosine RDM corr:     Spearman r = {r_cos:.3f}, "
          f"Kendall τ = {rk_cos:.3f}  [analytical]")
    print(f"                      Spearman r = {obs_r_cos:.3f}, "
          f"p_perm = {p_r_perm_cos:.4f}  [permutation]")
    print(f"  Euclidean RDM corr:  Spearman r = {r_euc:.3f}, "
          f"Kendall τ = {rk_euc:.3f}  [analytical]")
    if p_r_perm_cos < 0.05:
        print("\n  → Коннектом НАВЯЗЫВАЕТ структуру population activity ✅")
    else:
        print("\n  → Активность мозга СЛУЧАЙНА или неотличима от random ⚠")

    return {
        "pop_cc": pop_cc, "pop_rand": pop_rand,
        "rdm_cc_cosine": rdm_cc_cos, "rdm_rand_cosine": rdm_rand_cos,
        "rdm_cc_euclidean": rdm_cc_euc, "rdm_rand_euclidean": rdm_rand_euc,
        "r_cosine": r_cos, "rk_cosine": rk_cos,
        "r_euclidean": r_euc, "rk_euclidean": rk_euc,
        "perm_cosine": {
            "obs_r": obs_r_cos, "p_r": p_r_perm_cos,
            "obs_tau": obs_tau_cos, "p_tau": p_tau_perm_cos,
            "null_r": null_r_cos, "null_tau": null_tau_cos,
        },
        "cell_types": cell_types,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🧠 RSA Fidelity — проверка, работает ли мозг осмысленно")
    parser.add_argument("--model", type=str, choices=['lif', 'izhi', 'rate', 'psi'],
                        default='lif')
    parser.add_argument("--repeats", type=int, default=2,
                        help="Записей на условие (по умолчанию 2)")
    args = parser.parse_args()

    run_rsa_experiment(model=args.model, n_repeats=args.repeats)


if __name__ == "__main__":
    main()
