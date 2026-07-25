#!/usr/bin/env python3
"""
Inverse Toxin Design — T3-guided optimization of ion channel conductances.

The wild idea:
  Gradient descent over g_Na, g_K, g_leak to find the "optimal drug dose"
  that suppresses epileptic hyper-synchronization.

Pipeline:
  1. Generate epileptic EEG (3Hz SWD + HFO)
  2. BSA encode → spikes → drive PyHhLut neurons
  3. Measure T3 sync of HH output (delta order params)
  4. NLMS learns: d(sync)/d(g_Na), d(sync)/d(g_K)
  5. Gradient descent: g_Na -= lr * dLoss/dg_Na
  6. Verify: optimal g_Na/g_K reduces seizure probability

The "toxin" is literally the conductance change:
  - TTX-like:  g_Na * 0.05
  - TEA-like:  g_K * 0.10
  - BTX-like:  g_Na * 1.5 (shift E_Na too)
  - Our drug:  learned from data

Usage:
  python examples/python/inverse_toxin_design.py
  python examples/python/inverse_toxin_design.py --plot
  python examples/python/inverse_toxin_design.py --eval-only
"""

import numpy as np
import taas_checkmate as tcm
import argparse
import sys
from typing import List, Tuple
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════
#  Constants — HH channels
# ═══════════════════════════════════════════════════════════════════

# Standard HH conductances (mS/cm²)
G_NA_DEFAULT = 120.0   # NaV m³h, E_rev = +50 mV
G_K_DEFAULT = 36.0     # KV n⁴, E_rev = -77 mV
G_LEAK_DEFAULT = 0.3   # Leak, E_rev = -65 mV

E_NA = 50.0
E_K = -77.0
E_LEAK = -65.0

DT_MS = 0.01  # HH timestep
FS = 1000     # sampling rate for experiment (Hz)


# ═══════════════════════════════════════════════════════════════════
#  1. EEG → HH spike encoder
# ═══════════════════════════════════════════════════════════════════

def eeg_to_hh_current(
    eeg: np.ndarray,
    scale: float = 5.0,
) -> np.ndarray:
    """Scale EEG signal to HH-level current injection (µA/cm²)."""
    return eeg * scale / np.std(eeg[:1000])


def run_hh_batch(
    hh: tcm.hh.HhLut,
    I_inj: np.ndarray,
    n_steps: int,
    record_every: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run HH simulation with time-varying current injection.

    Args:
        hh: HhLut simulator (1 neuron)
        I_inj: current to inject per step
        n_steps: total steps
        record_every: record v every N steps

    Returns:
        v_trace: recorded voltage trace
        spike_counts: spikes per step
    """
    v_trace = []
    spike_counts = []

    for s in range(n_steps):
        if s < len(I_inj):
            hh.set_I(0, I_inj[s])
        spikes = hh.step()
        spike_counts.append(spikes)

        if s % record_every == 0:
            v = hh.v
            v_trace.append(float(v[0]))

    return np.array(v_trace, dtype=np.float32), np.array(spike_counts, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════
#  2. Toxin parameter manager
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ChannelParams:
    """Ion channel parameters that define the 'toxin dose'."""
    g_Na: float = G_NA_DEFAULT   # NaV conductance multiplier
    g_K: float = G_K_DEFAULT     # KV conductance multiplier
    g_leak: float = G_LEAK_DEFAULT
    E_Na: float = E_NA           # NaV reversal (mV)
    E_K: float = E_K             # KV reversal (mV)
    E_leak: float = E_LEAK

    @property
    def as_array(self) -> np.ndarray:
        """Flat array for NLMS optimization: [g_Na, g_K, g_leak, E_leak]."""
        return np.array([self.g_Na, self.g_K, self.g_leak, self.E_leak], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ChannelParams':
        return cls(g_Na=float(arr[0]), g_K=float(arr[1]),
                    g_leak=float(arr[2]), E_leak=float(arr[3]))

    def apply_to(self, hh: tcm.hh.HhLut):
        """Apply these parameters to an HhLut simulator."""
        n = hh.n
        hh.set_g_max_arr(np.array([self.g_Na, self.g_K], dtype=np.float32))
        hh.set_E_rev_arr(np.array([self.E_Na, self.E_K], dtype=np.float32))
        hh.set_g_leak(np.full(n, self.g_leak, dtype=np.float32))
        hh.set_E_leak(np.full(n, self.E_leak, dtype=np.float32))

    def __repr__(self) -> str:
        return (f"g_Na={self.g_Na:.1f} g_K={self.g_K:.1f} "
                f"g_leak={self.g_leak:.3f} E_leak={self.E_leak:.0f}")


# Known toxins as parameter presets
TOXIN_PRESETS = {
    'none': ChannelParams(),
    'TTX': ChannelParams(g_Na=G_NA_DEFAULT * 0.05),       # 95% NaV block
    'TEA': ChannelParams(g_K=G_K_DEFAULT * 0.10),          # 90% KV block
    'TTX+TEA': ChannelParams(g_Na=G_NA_DEFAULT * 0.05,
                              g_K=G_K_DEFAULT * 0.10),       # full channelopathy
    'BTX': ChannelParams(g_Na=G_NA_DEFAULT * 0.30,
                          E_Na=30.0),                        # NaV agonist
    'strychnine': ChannelParams(g_leak=0.01),               # GlyR block → disinhibition
    'GABAergic': ChannelParams(g_leak=1.5, E_leak=-70.0),   # Enhanced conductance
}


# ═══════════════════════════════════════════════════════════════════
#  3. T3 Sync Scanner — measures HH output synchronization
# ═══════════════════════════════════════════════════════════════════

class SyncScanner:
    """
    Measures T3 synchronization of HH neuron output.

    HH firing rate → T3 drive → delta sync order parameter
    Higher delta sync = more seizure-like (pathological).
    """

    def __init__(self, n_osc: int = 16):
        self.n_osc = n_osc
        self.t3 = tcm.t3.TriOscillator(n_osc, 0.5, 4.0)  # delta band
        self.t3.set_coupling(0.3, 0.15, 0.08)

    def measure_sync(
        self,
        spike_counts: np.ndarray,
        n_warmup: int = 50,
    ) -> Tuple[float, float, float]:
        """
        Measure T3 sync of spike train.

        Args:
            spike_counts: spikes per step from HH
            n_warmup: warmup steps before measurement

        Returns:
            (r_delta, r_theta, r_alpha): order parameters
        """
        n = self.n_osc
        r_deltas = []

        for s, n_spikes in enumerate(spike_counts):
            # Convert spikes to T3 drive
            rate = n_spikes / 0.01  # Hz (dt=0.01ms → 1 step = 0.01ms)
            drive = np.zeros(3 * n, dtype=np.float32)
            drive[:n] = np.clip(rate * 0.01, 0.0, 2.0)

            self.t3.step(DT_MS * 10, drive)  # step at ~10ms resolution

            if s >= n_warmup:
                r1, r2, r3 = self.t3.order_params()
                r_deltas.append(r1)

        mean_r = float(np.mean(r_deltas)) if r_deltas else 0.0
        max_r = float(np.max(r_deltas)) if r_deltas else 0.0

        return mean_r, max_r, float(np.std(r_deltas)) if len(r_deltas) > 1 else 0.0


# ═══════════════════════════════════════════════════════════════════
#  4. Inverse Toxin Optimizer via NLMS
# ═══════════════════════════════════════════════════════════════════

class InverseToxinDesigner:
    """
    NLMS-based optimizer that finds the optimal ion channel
    conductances to MINIMIZE T3 hyper-synchronization.

    Approach:
      1. Current params → HH sim → T3 sync measurement
      2. NLMS learns: feature = [g_Na, g_K, g_leak, E_leak] → target = 1-sync
      3. After training: NLMS weights encode the optimal direction
      4. Gradient: dLoss/dparam ≈ NLMS weights
      5. Update: params -= lr * weights

    This is literally computer-aided drug discovery:
      - "Drug" = vector of ion channel conductance changes
      - "Efficacy" = reduction in T3 delta sync
      - "Optimal dose" = params that minimize sync
    """

    def __init__(
        self,
        n_params: int = 4,
        lr: float = 0.1,
        eps: float = 1e-6,
        n_osc: int = 16,
    ):
        self.nlms = tcm.t3.NlmsDecoder(n_params, lr, eps)
        self.scanner = SyncScanner(n_osc)
        self.params_history: List[ChannelParams] = []
        self.sync_history: List[float] = []
        self.loss_history: List[float] = []
        self.best_params: ChannelParams = ChannelParams()
        self.best_sync: float = float('inf')
        self.hh_cache = None

    def evaluate(
        self,
        params: ChannelParams,
        I_inj: np.ndarray,
        n_steps: int,
    ) -> Tuple[float, float, float]:
        """
        Evaluate a parameter set on epileptic EEG.

        Args:
            params: ion channel parameters
            I_inj: current injection (from epileptic EEG)
            n_steps: simulation steps

        Returns:
            (mean_sync, max_sync, std_sync)
        """
        # Create fresh HH simulator
        hh = tcm.hh.HhLut(1, DT_MS, False)

        # Apply toxin parameters
        params.apply_to(hh)
        hh.reset()

        # Run simulation
        _, spike_counts = run_hh_batch(hh, I_inj, n_steps)

        # Measure sync
        mean_r, max_r, std_r = self.scanner.measure_sync(spike_counts)
        return mean_r, max_r, std_r

    def step(
        self,
        params: ChannelParams,
        I_inj: np.ndarray,
        n_steps: int,
    ) -> float:
        """
        One NLMS step: evaluate params, update NLMS, compute gradient.

        Returns:
            loss: sync value (lower = better)
        """
        mean_sync, max_sync, _ = self.evaluate(params, I_inj, n_steps)
        loss = mean_sync  # lower sync = better (less seizure-like)

        self.params_history.append(params)
        self.sync_history.append(mean_sync)
        self.loss_history.append(loss)

        if loss < self.best_sync:
            self.best_sync = loss
            self.best_params = ChannelParams(
                g_Na=params.g_Na, g_K=params.g_K,
                g_leak=params.g_leak, E_leak=params.E_leak,
            )

        # NLMS: feature = current params, target = 1 - sync (desirability)
        features = params.as_array
        target = 1.0 - loss  # 1 = perfect (no sync), 0 = terrible (max sync)

        self.nlms.train(features, target)
        return loss

    def compute_gradient(self, params: ChannelParams) -> np.ndarray:
        """
        Compute approximate gradient d(loss)/d(params) via NLMS weights.

        NLMS weights encode: "which params contribute to high desirability"
        i.e., which conductance values lead to LOW sync.

        Returns:
            gradient: shape (4,) - d(sync)/d(g_Na, g_K, g_leak, E_leak)
        """
        weights = np.array(self.nlms.weights(), dtype=np.float32)
        # NLMS weights: higher = more important for achieving LOW sync
        # Gradient: negative weights = reducing this param reduces sync
        # (because NLMS learned: param↑ → desirability↑ → weight↓)
        gradient = -weights / (np.linalg.norm(weights) + 1e-8)
        return gradient

    def optimize(
        self,
        I_inj: np.ndarray,
        n_steps: int,
        n_iterations: int = 50,
        lr: float = 0.05,
        bounds: Tuple[float, float] = (0.01, 200.0),
        verbose: bool = True,
    ) -> ChannelParams:
        """
        Run full optimization: gradient descent on channel parameters.

        Each iteration:
          1. Evaluate current params on HH → T3 sync
          2. NLMS learns the (param → desirability) mapping
          3. Compute gradient from NLMS weights
          4. Update params: g_Na -= lr * d(sync)/dg_Na
          5. Clamp to bounds

        Args:
            I_inj: epileptic current injection
            n_steps: HH simulation steps per eval
            n_iterations: gradient descent steps
            lr: learning rate for params
            bounds: (min, max) for all params
            verbose: print progress

        Returns:
            optimal ChannelParams
        """
        # Start from default params
        params = ChannelParams()

        if verbose:
            print(f"\n  Starting optimization from: {params}")
            print(f"  Iterations: {n_iterations}, LR: {lr}")

        for it in range(n_iterations):
            # Evaluate + NLMS step
            loss = self.step(params, I_inj, n_steps)

            # Compute gradient from NLMS
            gradient = self.compute_gradient(params)

            # Update params: gradient descent
            delta = -lr * gradient

            # Scale to param magnitudes
            scale = np.array([params.g_Na, params.g_K,
                              params.g_leak, params.E_leak], dtype=np.float32)
            scale = np.clip(scale, 1.0, None)  # don't divide by 0
            delta = delta * scale * 0.1  # 10% of current magnitude

            new_arr = params.as_array + delta
            new_arr = np.clip(new_arr, bounds[0], bounds[1])

            params = ChannelParams.from_array(new_arr)

            if verbose and (it + 1) % 10 == 0:
                print(f"    Iter {it+1:3d}/{n_iterations}: "
                      f"sync={loss:.4f} → {params} | "
                      f"∇=[{gradient[0]:+.3f}, {gradient[1]:+.3f}, "
                      f"{gradient[2]:+.3f}, {gradient[3]:+.3f}]")

        if verbose:
            print(f"\n  ✅ Optimal params found: {self.best_params}")
            print(f"     Best sync: {self.best_sync:.4f}")

        return self.best_params


# ═══════════════════════════════════════════════════════════════════
#  5. Main experiment
# ═══════════════════════════════════════════════════════════════════

def run_experiment(args):
    print("=" * 65)
    print("  💊 INVERSE TOXIN DESIGN — T3-guided channel optimization")
    print("  Gradient descent over g_Na, g_K, g_leak to suppress seizures")
    print("=" * 65)

    # ── Step 1: Generate epileptic EEG ────────────────
    print("\n📡 [1/5] Generating epileptic EEG")
    print("   3Hz spike-wave discharges + HFO pre-ictal bursts")

    # Use the generator from epilepsy_bci if available, else minimal version
    try:
        from epilepsy_bci import generate_epileptic_eeg
    except ImportError:
        # Inline minimal generator
        def generate_epileptic_eeg(duration_sec=5.0, fs=1000, seizure_times=None, seed=42):
            rng = np.random.default_rng(seed)
            n = int(duration_sec * fs)
            t = np.arange(n) / fs
            eeg = rng.normal(0, 0.3, n)
            if seizure_times is None:
                seizure_times = [(2.0, 4.0)]
            seizure_mask = np.zeros(n, dtype=bool)
            hfo_energy = np.zeros(n, dtype=np.float32)
            eeg += 2.0 * np.sin(2 * np.pi * 10 * t)
            for sz_start, sz_end in seizure_times:
                sz_s = int(sz_start * fs)
                sz_e = int(sz_end * fs)
                seizure_mask[sz_s:sz_e] = True
                sw_t = t[sz_s:sz_e] - sz_start
                spike = 8.0 * np.sin(2 * np.pi * 3.0 * sw_t)
                spike += 4.0 * np.sin(2 * np.pi * 6.0 * sw_t + 0.5)
                slow = 3.0 * np.sin(2 * np.pi * 1.5 * sw_t + np.pi)
                amp = np.clip(np.sin(np.pi * sw_t / (sz_end - sz_start)), 0.3, 1.0)
                eeg[sz_s:sz_e] += (spike + slow) * amp
            eeg = eeg / np.std(eeg) * 20.0
            return t, eeg, seizure_mask, hfo_energy

    t, eeg, seizure_mask, hfo = generate_epileptic_eeg(5.0, FS, [(2.0, 4.0)])
    n_samples = len(eeg)
    print(f"   → {n_samples} samples @ {FS} Hz (5.0s)")
    print(f"   → Seizure: 2.0-4.0s (40% of signal)")

    # ── Step 2: Convert EEG to HH current ─────────────
    print("\n⚡ [2/5] EEG → HH current injection")
    I_inj = eeg_to_hh_current(eeg, scale=8.0)
    print(f"   Current range: [{I_inj.min():.1f}, {I_inj.max():.1f}] µA/cm²")

    # ── Step 3: Evaluate toxin presets ────────────────
    print("\n🔬 [3/5] Evaluating known toxin presets")

    n_steps = min(n_samples, 5000)  # 5s * 1000Hz = 5000
    scanner = SyncScanner(16)

    results = {}
    for name, params in TOXIN_PRESETS.items():
        hh = tcm.hh.HhLut(1, DT_MS, False)
        params.apply_to(hh)
        hh.reset()
        _, spikes = run_hh_batch(hh, I_inj, n_steps)
        mean_r, max_r, std_r = scanner.measure_sync(spikes)

        results[name] = (mean_r, max_r, std_r)
        print(f"   {name:12s}: δ_sync={mean_r:.4f} (max={max_r:.4f}) — "
              f"{'SEIZURE' if mean_r > 0.3 else 'SUPPRESSED' if mean_r < 0.1 else 'moderate'}")

    # ── Step 4: Inverse Toxin Optimization ────────────
    print("\n🧪 [4/5] Inverse Toxin Optimization (NLMS gradient descent)")

    if not args.eval_only:
        designer = InverseToxinDesigner(n_params=4, lr=0.1, n_osc=16)
        optimal = designer.optimize(
            I_inj=I_inj,
            n_steps=n_steps,
            n_iterations=args.n_iter,
            lr=args.lr,
            verbose=True,
        )

        # Evaluate optimal params
        print(f"\n   📊 Evaluating optimal params...")
        hh_opt = tcm.hh.HhLut(1, DT_MS, False)
        optimal.apply_to(hh_opt)
        hh_opt.reset()
        _, spikes_opt = run_hh_batch(hh_opt, I_inj, n_steps)
        mean_opt, max_opt, _ = scanner.measure_sync(spikes_opt)
    else:
        optimal = ChannelParams()
        mean_opt = 0.0
        max_opt = 0.0

    # ── Step 5: Analysis ──────────────────────────────
    print(f"\n" + "─" * 65)
    print("📊 [5/5] ANALYSIS")
    print("─" * 65)

    # Compare all presets + optimal
    print(f"\n  {'Preset':>14} | {'Sync (mean)':>10} | {'Sync (max)':>10} | {'Status':>12}")
    print(f"  {'─'*14} | {'─'*10} | {'─'*10} | {'─'*12}")

    for name, (m, x, s) in sorted(results.items(), key=lambda kv: kv[1][0]):
        status = "✅ GOOD" if m < 0.1 else "⚠️ MEDIUM" if m < 0.3 else "🔥 SEIZURE"
        print(f"  {name:>14} | {m:10.4f} | {x:10.4f} | {status}")

    if not args.eval_only:
        status_opt = "✅ GOOD" if mean_opt < 0.1 else "⚠️ MEDIUM" if mean_opt < 0.3 else "🔥 SEIZURE"
        print(f"  {'───':>14} | {'──':>10} | {'──':>10} | {'──':>12}")
        print(f"  {'OPTIMAL':>14} | {mean_opt:10.4f} | {max_opt:10.4f} | {status_opt}")

    print(f"\n  💊 Optimal channel params: g_Na={optimal.g_Na:.1f}, "
          f"g_K={optimal.g_K:.1f}, g_leak={optimal.g_leak:.3f}")
    print(f"  🎯 Best sync achieved: {designer.best_sync:.4f}" if not args.eval_only else "")

    # Interpretation
    print(f"\n  🧬 Drug interpretation:")
    if optimal.g_Na < G_NA_DEFAULT * 0.5:
        print(f"     → TTX-like: g_Na reduced by {(1 - optimal.g_Na/G_NA_DEFAULT)*100:.0f}%")
    if optimal.g_K < G_K_DEFAULT * 0.5:
        print(f"     → TEA-like: g_K reduced by {(1 - optimal.g_K/G_K_DEFAULT)*100:.0f}%")
    if optimal.g_leak > G_LEAK_DEFAULT * 1.5:
        print(f"     → GABAergic: g_leak increased by {(optimal.g_leak/G_LEAK_DEFAULT - 1)*100:.0f}%")
    if optimal.E_leak < E_LEAK:
        print(f"     → Hyperpolarizing: E_leak shifted by {optimal.E_leak - E_LEAK:.0f} mV")
    if optimal.g_Na > G_NA_DEFAULT * 1.2:
        print(f"     → BTX-like: g_Na INCREASED by {(optimal.g_Na/G_NA_DEFAULT - 1)*100:.0f}%")

    # Drug equivalence
    print(f"\n  🎯 Closest known drug:")
    best_match = 'none'
    best_dist = float('inf')
    for name, preset in TOXIN_PRESETS.items():
        dist = (abs(optimal.g_Na - preset.g_Na) / G_NA_DEFAULT +
                abs(optimal.g_K - preset.g_K) / G_K_DEFAULT +
                abs(optimal.g_leak - preset.g_leak) / G_LEAK_DEFAULT +
                abs(optimal.E_leak - preset.E_leak) / 50.0)
        if dist < best_dist:
            best_dist = dist
            best_match = name
    print(f"     → {best_match}")

    # ── Plot ──────────────────────────────────────────
    if args.plot and not args.eval_only:
        _plot_results(t, eeg, I_inj, results, designer, optimal)

    return designer.best_sync if not args.eval_only else 0.0


def _plot_results(t, eeg, I_inj, results, designer, optimal):
    """Plot EEG, toxin presets, and optimization trajectory."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        # 1. EEG
        ax = axes[0]
        ax.plot(t, eeg, color='navy', lw=0.8)
        ax.set_ylabel('EEG (µV)')
        ax.set_title('Epileptic EEG (3Hz spike-wave)')
        ax.axvspan(2.0, 4.0, alpha=0.2, color='red', label='Seizure')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Toxin comparison bar chart
        ax = axes[1]
        names = list(results.keys())
        syncs = [results[n][0] for n in names]
        colors = ['green' if s < 0.1 else 'orange' if s < 0.3 else 'red' for s in syncs]
        bars = ax.bar(names, syncs, color=colors, alpha=0.8)
        ax.set_ylabel('T3 Delta Sync (r)')
        ax.set_title('Toxin Presets: Seizure Sync Comparison')
        ax.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Seizure threshold')
        ax.axhline(0.1, color='green', linestyle=':', alpha=0.5, label='Suppressed threshold')
        ax.legend()
        for bar, s in zip(bars, syncs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{s:.3f}', ha='center', va='bottom', fontsize=8)

        # 3. Optimization trajectory
        ax = axes[2]
        if len(designer.loss_history) > 1:
            ax.plot(designer.loss_history, color='crimson', lw=1.5)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Sync (loss)')
            ax.set_title('Inverse Toxin Optimization Trajectory')
            ax.axhline(0.3, color='red', linestyle='--', alpha=0.5)
            ax.axhline(0.1, color='green', linestyle=':', alpha=0.5)
            ax.grid(True, alpha=0.3)

        # 4. Parameter evolution
        ax = axes[3]
        if len(designer.params_history) > 1:
            g_na_trace = [p.g_Na for p in designer.params_history]
            g_k_trace = [p.g_K for p in designer.params_history]
            g_l_trace = [p.g_leak for p in designer.params_history]

            ax.plot(g_na_trace, label='g_Na', color='blue', lw=1.5)
            ax.plot(g_k_trace, label='g_K', color='green', lw=1.5)
            ax.plot(g_l_trace, label='g_leak', color='orange', lw=1.5)
            ax.axhline(G_NA_DEFAULT, color='blue', linestyle='--', alpha=0.3)
            ax.axhline(G_K_DEFAULT, color='green', linestyle='--', alpha=0.3)
            ax.axhline(G_LEAK_DEFAULT, color='orange', linestyle='--', alpha=0.3)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Conductance (mS/cm²)')
            ax.set_title('Channel Parameter Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('inverse_toxin_design.png', dpi=150, bbox_inches='tight')
        print(f"\n   📈 Plot saved: inverse_toxin_design.png")
        plt.show()
    except ImportError:
        print("   ⚠️  matplotlib not available")


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Inverse Toxin Design — T3-guided ion channel optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/python/inverse_toxin_design.py            # full optimization
  python examples/python/inverse_toxin_design.py --plot     # with plots
  python examples/python/inverse_toxin_design.py --eval-only  # evaluate presets only
  python examples/python/inverse_toxin_design.py --lr 0.1 --n-iter 100  # tune
        """,
    )

    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate known toxin presets (skip optimization)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate for params (default: 0.05)')
    parser.add_argument('--n-iter', type=int, default=50,
                        help='Optimization iterations (default: 50)')

    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
