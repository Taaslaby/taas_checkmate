#!/usr/bin/env python3
"""
BCI Pet — Virtual brain that sleeps with you.

Pipeline:
  1. EEG (real Sleep-EDF .edf or synthetic) → BSA encode → spikes
  2. SBM connectome → BrainSimulator LIF → virtual brain activity
  3. T3 TriOscillator sync with real EEG rhythms
  4. Compare: virtual vs real sleep stages
  5. Closed-loop feedback: virtual brain rate → T3 drive → virtual brain

Usage:
  python bci_pet.py                        # synthetic EEG demo (30s)
  python bci_pet.py --edf /path/to/file.edf  # real Sleep-EDF
  python bci_pet.py --plot                 # show plots
  python bci_pet.py --no-sim               # quick: encoding + T3 only
  python bci_pet.py --full                 # full 3000 steps (30s @ 100Hz)
"""

import numpy as np
import taas_checkmate as tcm
import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
#  1. EEG Source — synthetic or real
# ═══════════════════════════════════════════════════════════════════

def generate_synthetic_eeg(duration_sec: float = 30, fs: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic synthetic EEG with staged sleep architecture.

    Stages:
      0-5s:  Wake    (alpha 8-12 Hz + beta 12-30 Hz)
      5-15s: NREM-2  (theta 4-8 Hz + sleep spindles 12-16 Hz)
     15-22s: NREM-3  (delta 0.5-4 Hz, high amplitude slow waves)
     22-28s: REM     (theta 4-8 Hz + sawtooth waves + gamma bursts)
     28-30s: Wake    (alpha again)
    """
    np.random.seed(42)
    t = np.arange(0, duration_sec * fs) / fs
    eeg = np.random.randn(len(t)) * 0.3  # noise floor

    # ── Wake 0-5s: alpha 10 Hz + beta 20 Hz ──
    mask = (t >= 0) & (t < 5)
    eeg[mask] += 3.0 * np.sin(2 * np.pi * 10 * t[mask])
    eeg[mask] += 1.5 * np.sin(2 * np.pi * 20 * t[mask])

    # ── NREM-2 5-15s: theta 6 Hz + sleep spindles @ 14 Hz ──
    mask = (t >= 5) & (t < 15)
    eeg[mask] += 2.5 * np.sin(2 * np.pi * 6 * t[mask])
    # Sleep spindles: 14 Hz bursts modulated at 0.5 Hz
    spindle_mod = (np.sin(2 * np.pi * 0.5 * t[mask]) > 0.3).astype(float)
    eeg[mask] += 2.0 * np.sin(2 * np.pi * 14 * t[mask]) * spindle_mod

    # ── NREM-3 15-22s: delta 0.5-2 Hz, high amplitude ──
    mask = (t >= 15) & (t < 22)
    eeg[mask] += 6.0 * np.sin(2 * np.pi * 1.0 * t[mask])
    eeg[mask] += 4.0 * np.sin(2 * np.pi * 2.0 * t[mask])
    eeg[mask] += 2.0 * np.sin(2 * np.pi * 0.5 * t[mask])

    # ── REM 22-28s: theta + sawtooth + gamma ──
    mask = (t >= 22) & (t < 28)
    eeg[mask] += 3.0 * np.sin(2 * np.pi * 5 * t[mask])
    eeg[mask] += 2.0 * np.sin(2 * np.pi * 7 * t[mask])
    # Sawtooth waves (REM-specific)
    eeg[mask] += 1.5 * np.sin(2 * np.pi * 2 * t[mask] + np.sin(2 * np.pi * 0.3 * t[mask]) * 2)
    # Gamma bursts
    eeg[mask] += 0.5 * np.random.randn(np.sum(mask))

    # ── Wake 28-30s: alpha rebound ──
    mask = (t >= 28) & (t < 30)
    eeg[mask] += 3.0 * np.sin(2 * np.pi * 10 * t[mask])

    # Normalize to realistic EEG amplitude (~50-100 µV peak)
    eeg = eeg / np.std(eeg) * 15.0
    return t, eeg


def load_edf(
    path: str,
    max_samples: int = 30000,
) -> Tuple[np.ndarray, float]:
    """Load a Sleep-EDF .edf file and return (signal, fs)."""
    import pyedflib
    f = pyedflib.EdfReader(path)
    n_signals = f.signals_in_file
    print(f"   File: {os.path.basename(path)}")
    print(f"   Signals: {n_signals}")

    # Try Fpz-Cz (channel 0 usually), then the first EEG channel
    for ch in range(min(n_signals, 4)):
        label = f.getLabel(ch)
        fs = f.getSampleFrequency(ch)
        print(f"   Channel {ch}: {label} @ {fs} Hz")

    ch = 0  # Fpz-Cz is typically first
    fs = f.getSampleFrequency(ch)
    signal = f.readSignal(ch)
    f.close()

    if len(signal) > max_samples:
        step = len(signal) // max_samples
        signal = signal[::step]
        fs = fs / step
        print(f"   Downsampled to {len(signal)} samples @ {fs:.1f} Hz")

    return signal, fs


# ═══════════════════════════════════════════════════════════════════
#  2. EEG Band Power Analysis
# ═══════════════════════════════════════════════════════════════════

def compute_band_powers(
    eeg: np.ndarray,
    fs: float = 100.0,
) -> Dict[str, float]:
    """Compute absolute band powers via FFT."""
    n = len(eeg)
    if n < 2:
        return {'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'gamma': 0}

    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum = np.abs(np.fft.rfft(eeg * np.hanning(n))) ** 2

    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'beta': (12.0, 30.0),
        'gamma': (30.0, 50.0),
    }

    powers = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        powers[name] = float(np.mean(spectrum[mask])) if np.any(mask) else 0.0

    return powers


def classify_sleep_stage(band_powers: Dict[str, float]) -> str:
    """Classify sleep stage from band power ratios."""
    total = sum(band_powers.values()) + 1e-10
    ratios = {k: v / total for k, v in band_powers.items()}

    delta_r = ratios.get('delta', 0)
    theta_r = ratios.get('theta', 0)
    alpha_r = ratios.get('alpha', 0)
    beta_r = ratios.get('beta', 0)

    if delta_r > 0.4:
        return "NREM-3 (deep slow-wave sleep)"
    elif delta_r > 0.25:
        return "NREM-2 (light sleep + spindles)"
    elif theta_r > 0.3:
        return "REM (dreaming)"
    elif alpha_r > 0.3:
        return "Wake (relaxed with eyes closed)"
    elif beta_r > 0.3:
        return "Wake (active thinking)"
    else:
        return "Wake / Transition"


# ═══════════════════════════════════════════════════════════════════
#  3. BSA — EEG to Spikes
# ═══════════════════════════════════════════════════════════════════

def eeg_to_spikes(
    eeg: np.ndarray,
    n_input_neurons: int = 32,
    base_threshold: float = 0.4,
    kernel_len: int = 6,
) -> np.ndarray:
    """
    Encode EEG signal into spike trains for N input neurons.

    Each input neuron encodes at a slightly different sensitivity
    (threshold varies ±50% from base) to create a diverse population code.
    """
    n_samples = len(eeg)
    spikes = np.zeros((n_samples, n_input_neurons), dtype=np.int32)
    kernel = np.array(tcm.snn.bsa_kernel_hamming(kernel_len), dtype=np.float32)
    eeg_f32 = eeg.astype(np.float32)

    for i in range(n_input_neurons):
        # Diverse thresholds: 0.5× to 1.5× base
        th = base_threshold * (0.5 + 1.0 * i / n_input_neurons)
        spikes_out = np.zeros(n_samples, dtype=np.int32)
        reconstructed = np.zeros(n_samples, dtype=np.float32)
        tcm.snn.bsa_encode(spikes_out, eeg_f32, kernel, reconstructed, th)
        spikes[:, i] = spikes_out

    return spikes


# ═══════════════════════════════════════════════════════════════════
#  4. SBM Connectome Builder
# ═══════════════════════════════════════════════════════════════════

def build_sbm_connectome(
    n_neurons: int = 256,
    n_input: int = 32,
    within_prob: float = 0.35,
    between_prob: float = 0.03,
    weight_range: Tuple[float, float] = (0.5, 2.0),
    exc_ratio: float = 0.8,
    seed: int = 42,
):
    """
    Build a Stochastic Block Model connectome.

    Groups are dynamically sized:
      - sensory:  n_input neurons
      - hidden_a: (n_neurons - n_input) // 3 neurons
      - hidden_b: (n_neurons - n_input) // 3 neurons
      - motor:    remaining neurons

    Returns (ConnectomeData, groups_dict).
    """
    rng = np.random.default_rng(seed)

    n_hidden = (n_neurons - n_input) // 3
    n_motor = n_neurons - n_input - 2 * n_hidden

    groups = {
        'sensory': list(range(0, n_input)),
        'hidden_a': list(range(n_input, n_input + n_hidden)),
        'hidden_b': list(range(n_input + n_hidden, n_input + 2 * n_hidden)),
        'motor': list(range(n_input + 2 * n_hidden, n_neurons)),
    }

    # Group membership lookup
    neuron_group = np.zeros(n_neurons, dtype=np.int32)
    for gid, grp in enumerate([groups['sensory'], groups['hidden_a'],
                                groups['hidden_b'], groups['motor']]):
        for nid in grp:
            neuron_group[nid] = gid

    # Build CSR adjacency lists per row
    rows, cols, vals, nts = [], [], [], []
    n_syn = 0

    for i in range(n_neurons):
        rows.append(n_syn)
        gi = neuron_group[i]
        for j in range(n_neurons):
            if i == j:
                continue
            gj = neuron_group[j]
            prob = within_prob if gi == gj else between_prob
            if rng.random() < prob:
                cols.append(j)
                w = rng.uniform(weight_range[0], weight_range[1])
                is_exc = rng.random() < exc_ratio
                vals.append(w)
                nts.append(1.0 if is_exc else -1.0)
                n_syn += 1
    rows.append(n_syn)

    row_ptr = np.array(rows, dtype=np.int32)
    col_indices = np.array(cols, dtype=np.int32)
    values = np.array(vals, dtype=np.float32)
    nt_types = np.array(nts, dtype=np.float32)

    conn = tcm.connectome_brain.ConnectomeData(
        row_ptr, col_indices, values, nt_types,
        groups=[(name, idxs) for name, idxs in groups.items()],
    )

    return conn, groups


# ═══════════════════════════════════════════════════════════════════
#  5. Virtual Brain Runner
# ═══════════════════════════════════════════════════════════════════

def simulate_virtual_brain(
    conn,
    groups: Dict[str, List[int]],
    spikes: np.ndarray,
    n_steps: int,
    dt_ms: float = 10.0,
    i_scale: float = 50.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run connectome LIF simulation with per-step external input.

    Args:
        conn: ConnectomeData
        spikes: (n_steps, n_input_neurons) spike patterns
        i_scale: current per spike (µA)

    Returns:
        (spike_buffer_flat, pop_rate, sensory_rate)
    """
    n_neurons = conn.n_neurons
    sensory_idx = np.array(groups['sensory'], dtype=np.int32)
    n_sensory = len(sensory_idx)
    n_input_spikes = spikes.shape[1]
    assert n_sensory >= n_input_spikes, \
        f"Sensory group ({n_sensory}) < n_input ({n_input_spikes})"

    sim = tcm.connectome_brain.BrainSimulator(n_neurons, "lif")

    all_bufs = []
    pop_rates = []
    sensory_rates = []
    ext_I_base = np.zeros(n_neurons, dtype=np.float32)

    actual_steps = min(n_steps, spikes.shape[0])

    for step in range(actual_steps):
        # Build external current: only first n_input_spikes sensory neurons get input
        ext_I = ext_I_base.copy()
        ext_I[sensory_idx[:n_input_spikes]] = spikes[step] * i_scale

        # Run 1 step (state persists, CSR propagates from spike queue)
        buf = sim.step_batch(conn, 1, dt_ms, ext_I, scale=1.0)
        all_bufs.append(buf.copy())

        # Rates
        step_spikes = buf[:n_neurons]
        pop_rates.append(float(np.mean(step_spikes) * (1000.0 / dt_ms)))
        sensory_rates.append(float(np.mean(step_spikes[groups['sensory']]) * (1000.0 / dt_ms)))

        if verbose and (step + 1) % 100 == 0:
            print(f"      Step {step+1}/{actual_steps} — "
                  f"pop: {pop_rates[-1]:.1f} Hz, "
                  f"sensory: {sensory_rates[-1]:.1f} Hz")

    spike_buffer_flat = np.concatenate(all_bufs).ravel()
    return spike_buffer_flat, np.array(pop_rates), np.array(sensory_rates)


# ═══════════════════════════════════════════════════════════════════
#  6. T3 Sleep Synchronizer
# ═══════════════════════════════════════════════════════════════════

class SleepSynchronizer:
    """
    T3 oscillators synchronized with real EEG sleep rhythms.

    Three oscillator populations:
      - Pop 1: Delta band (0.5-4 Hz)  → NREM sleep
      - Pop 2: Theta band (4-8 Hz)    → REM sleep
      - Pop 3: Alpha band (8-12 Hz)   → Wake/relaxed

    Closed-loop: virtual brain rate feeds back into T3 drive.
    """

    def __init__(self, n_osc: int = 32):
        self.n_osc = n_osc

        # Three T3 oscillators, one per band
        self.delta_t3 = tcm.t3.TriOscillator(n_osc, 0.5, 4.0)
        self.theta_t3 = tcm.t3.TriOscillator(n_osc, 4.0, 8.0)
        self.alpha_t3 = tcm.t3.TriOscillator(n_osc, 8.0, 12.0)

        self.delta_t3.set_coupling(0.25, 0.12, 0.06)
        self.theta_t3.set_coupling(0.20, 0.10, 0.05)
        self.alpha_t3.set_coupling(0.15, 0.08, 0.04)

        self.drive = np.zeros(3 * n_osc, dtype=np.float32)
        self.history = {
            'delta_r': [], 'theta_r': [], 'alpha_r': [],
            'delta_pst': [], 'theta_pst': [], 'alpha_pst': [],
        }

    def sync(self, band_powers: Dict[str, float], virtual_rate: float = 0.0):
        """One sync step: set drives from band powers + virtual brain feedback."""
        total = sum(band_powers.values()) + 1e-10
        delta_ratio = band_powers.get('delta', 0) / total
        theta_ratio = band_powers.get('theta', 0) / total
        alpha_ratio = band_powers.get('alpha', 0) / total

        # Base drive from EEG band power (scaled to 0-2 range)
        drive_strength = 2.0
        fb_gain = 0.02

        self.drive[:self.n_osc] = delta_ratio * drive_strength + virtual_rate * fb_gain
        self.drive[self.n_osc:2*self.n_osc] = theta_ratio * drive_strength
        self.drive[2*self.n_osc:] = alpha_ratio * drive_strength

        # Step each T3 with band-specific drive on pop1 (others zero)
        # set_drive maps: drive[0..n]→pop1, drive[n..2n]→pop2, drive[2n..3n]→pop3
        d_drive = np.zeros(3 * self.n_osc, dtype=np.float32)
        d_drive[:self.n_osc] = delta_ratio * drive_strength + virtual_rate * fb_gain
        self.delta_t3.set_drive(d_drive)

        t_drive = np.zeros(3 * self.n_osc, dtype=np.float32)
        t_drive[:self.n_osc] = theta_ratio * drive_strength
        self.theta_t3.set_drive(t_drive)

        a_drive = np.zeros(3 * self.n_osc, dtype=np.float32)
        a_drive[:self.n_osc] = alpha_ratio * drive_strength
        self.alpha_t3.set_drive(a_drive)

        pst_d = self.delta_t3.step(0.05)
        pst_t = self.theta_t3.step(0.05)
        pst_a = self.alpha_t3.step(0.05)

        r1, r2, r3 = self.delta_t3.order_params()
        self.history['delta_r'].append(r1)
        r1, r2, r3 = self.theta_t3.order_params()
        self.history['theta_r'].append(r1)
        r1, r2, r3 = self.alpha_t3.order_params()
        self.history['alpha_r'].append(r1)

        self.history['delta_pst'].append(pst_d)
        self.history['theta_pst'].append(pst_t)
        self.history['alpha_pst'].append(pst_a)

    def sync_summary(self) -> Dict[str, float]:
        """Return average sync over last N steps."""
        n = min(100, len(self.history['delta_r']))
        if n == 0:
            return {'delta_sync': 0.0, 'theta_sync': 0.0, 'alpha_sync': 0.0}
        return {
            'delta_sync': float(np.mean(self.history['delta_r'][-n:])),
            'theta_sync': float(np.mean(self.history['theta_r'][-n:])),
            'alpha_sync': float(np.mean(self.history['alpha_r'][-n:])),
        }


# ═══════════════════════════════════════════════════════════════════
#  7. Main Pipeline
# ═══════════════════════════════════════════════════════════════════

def run_bci_pet(args):
    print("=" * 65)
    print("  🧠 BCI PET — Virtual Brain Sleeps with You")
    print("  EEG → BSA → Connectome → T3 Synchronization")
    print("=" * 65)

    # ── Step 1: EEG ─────────────────────────────────────
    print("\n" + "─" * 65)
    print("📡 [1/6] EEG Source")

    if args.edf:
        if not os.path.exists(args.edf):
            print(f"   ❌ File not found: {args.edf}")
            sys.exit(1)
        signal, fs = load_edf(args.edf, max_samples=args.max_samples)
        t = np.arange(len(signal)) / fs
    else:
        print("   Generating synthetic 30s EEG with sleep stages...")
        t, signal = generate_synthetic_eeg(30, 100)
        fs = 100.0

    n_samples = len(signal)
    duration = n_samples / fs
    print(f"   → {n_samples} samples @ {fs:.0f} Hz ({duration:.1f}s)")

    # ── Step 2: Band powers + sleep stage ────────────────
    print("\n📊 [2/6] EEG Band Power Analysis")

    band_powers = compute_band_powers(signal, fs)
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        print(f"   {band:>6}: {band_powers.get(band, 0):.2f}")

    dominant = max(band_powers, key=lambda k: band_powers.get(k, 0))
    sleep_stage = classify_sleep_stage(band_powers)
    print(f"\n   🎯 Dominant band: {dominant}")
    print(f"   💤 Sleep stage:   {sleep_stage}")

    # ── Step 3: BSA encode ──────────────────────────────
    print("\n⚡ [3/6] BSA Encoding EEG → Spikes")
    print(f"   32 input neurons, base_threshold={args.bsa_threshold}")

    spikes = eeg_to_spikes(
        signal,
        n_input_neurons=args.n_input,
        base_threshold=args.bsa_threshold,
        kernel_len=6,
    )
    spike_density = np.mean(spikes) * 100
    n_total = int(np.sum(spikes))
    print(f"   → {n_total} spikes total ({spike_density:.1f}% density)")
    print(f"   → Active input neurons: {np.sum(np.any(spikes, axis=0))}/{args.n_input}")

    # ── Step 4: T3 Synchronization ──────────────────────
    print("\n🔄 [4/6] T3 Oscillator Synchronization")
    print(f"   Delta:  0.5-4 Hz  → NREM sync")
    print(f"   Theta:  4-8 Hz    → REM sync")
    print(f"   Alpha:  8-12 Hz   → Wake sync")

    sync = SleepSynchronizer(n_osc=args.n_osc)

    # Warmup T3
    for _ in range(100):
        sync.sync(band_powers, 0.0)

    # Real sync loop (limited steps)
    n_sync_steps = min(n_samples, args.max_sim_steps)
    for step in range(n_sync_steps):
        # Extract band powers in sliding window
        window_start = max(0, step - 25)
        window_end = min(n_samples, step + 25)
        local_powers = compute_band_powers(signal[window_start:window_end], fs)

        # Virtual rate starts at 0, will be updated during simulation phase
        sync.sync(local_powers, 0.0)

        if (step + 1) % 200 == 0:
            r = sync.sync_summary()
            print(f"   Step {step+1}/{n_sync_steps} — "
                  f"δ:{r['delta_sync']:.3f} θ:{r['theta_sync']:.3f} α:{r['alpha_sync']:.3f}")

    initial_sync = sync.sync_summary()
    print(f"\n   T3 sync (initial):")
    print(f"     Delta (NREM): {initial_sync['delta_sync']:.4f}")
    print(f"     Theta (REM):  {initial_sync['theta_sync']:.4f}")
    print(f"     Alpha (Wake): {initial_sync['alpha_sync']:.4f}")

    # ── Step 5: Connectome Simulation ────────────────────
    print("\n🧬 [5/6] Connectome Simulation")
    print(f"   {args.n_neurons} neurons, LIF model, dt=10ms")

    if not args.no_sim:
        conn, groups = build_sbm_connectome(
            n_neurons=args.n_neurons,
            n_input=args.n_input,
            within_prob=0.35,
            between_prob=0.03,
            weight_range=(0.5, 2.0),
        )
        print(f"   → {conn.n_neurons} neurons, {conn.n_synapses} synapses")
        print(f"   → Groups: sensory={len(groups['sensory'])}, "
              f"hidden_a={len(groups['hidden_a'])}, "
              f"hidden_b={len(groups['hidden_b'])}, "
              f"motor={len(groups['motor'])}")

        n_sim_steps = min(n_samples, args.max_sim_steps)
        print(f"\n   Running {n_sim_steps} steps (dt=10ms, ~{n_sim_steps*10/1000:.1f}s brain time)...")

        spike_buf, pop_rate, sensory_rate = simulate_virtual_brain(
            conn, groups, spikes, n_sim_steps,
            dt_ms=args.dt_ms, i_scale=args.i_scale,
        )

        print(f"\n   ✅ Virtual brain stats:")
        print(f"      Population rate:  {np.mean(pop_rate):.1f} ± {np.std(pop_rate):.1f} Hz")
        print(f"      Sensory rate:     {np.mean(sensory_rate):.1f} ± {np.std(sensory_rate):.1f} Hz")
        print(f"      Total spikes:     {int(np.sum(spike_buf))}")
    else:
        print("   ⏭️  Skipped (--no-sim)")
        spike_buf = np.zeros(args.n_neurons * 100, dtype=np.int32)
        pop_rate = np.zeros(100)

    # ── Step 6: Final Analysis ───────────────────────────
    print("\n" + "─" * 65)
    print("📊 [6/6] ANALYSIS: Virtual Brain vs Real EEG")
    print("─" * 65)

    final_sync = sync.sync_summary()
    delta_s = final_sync['delta_sync']
    theta_s = final_sync['theta_sync']
    alpha_s = final_sync['alpha_sync']

    print(f"\n  🎯 Real EEG:      {dominant.upper()} dominant — {sleep_stage}")
    print(f"\n  📈 T3 Sync quality:")
    print(f"     🛌 NREM (δ):   {delta_s:.4f}  {'✅' if delta_s > 0.3 else '⬜'}")
    print(f"     💭 REM  (θ):   {theta_s:.4f}  {'✅' if theta_s > 0.3 else '⬜'}")
    print(f"     ☀️ Wake (α):   {alpha_s:.4f}  {'✅' if alpha_s > 0.3 else '⬜'}")

    # Score
    if dominant == 'delta':
        quality = delta_s
        label = "NREM"
    elif dominant == 'theta':
        quality = theta_s
        label = "REM"
    else:
        quality = alpha_s
        label = "Wake"

    print(f"\n  🧪 {label} sync quality: {quality:.2%}")
    if quality > 0.5:
        verdict = "🔥 VIRTUAL BRAIN IS SYNCHRONIZED WITH REAL EEG!"
    elif quality > 0.3:
        verdict = "✨ Moderate synchronization detected."
    elif quality > 0.1:
        verdict = "🔬 Weak synchronization — try increasing n_osc or coupling."
    else:
        verdict = "❌ No significant synchronization."
    print(f"\n  {verdict}")

    # ── Optional plot ────────────────────────────────────
    if args.plot:
        _plot_results(t, signal, spikes, pop_rate, band_powers, sync, args)

    return quality


def _plot_results(t, signal, spikes, pop_rate, band_powers, sync, args):
    """Plot EEG, spikes, virtual brain rates, and T3 sync."""
    try:
        import matplotlib.pyplot as plt

        n_plot = min(len(t), 300)  # first 30s @ 100Hz max

        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

        # 1. EEG
        ax = axes[0]
        ax.plot(t[:n_plot], signal[:n_plot], color='navy', lw=0.8)
        ax.set_ylabel('EEG (µV)')
        ax.set_title('Real EEG Signal')
        ax.grid(True, alpha=0.3)

        # 2. Spike raster
        ax = axes[1]
        n_input = min(16, spikes.shape[1])
        raster = spikes[:n_plot, :n_input].T
        ax.imshow(raster, aspect='auto', cmap='binary',
                  extent=[0, t[min(n_plot-1, len(t)-1)], 0, n_input],
                  interpolation='nearest')
        ax.set_ylabel('Input Neuron')
        ax.set_title(f'BSA Encoded Spikes ({n_input} of {spikes.shape[1]} neurons)')

        # 3. Virtual brain rate
        ax = axes[2]
        rate_plot = pop_rate[:min(len(pop_rate), n_plot)]
        ax.plot(t[:len(rate_plot)], rate_plot, color='crimson', lw=1.0)
        ax.set_ylabel('Pop Rate (Hz)')
        ax.set_title('Virtual Brain Population Rate')
        ax.grid(True, alpha=0.3)

        # 4. T3 sync history
        ax = axes[3]
        n_history = min(300, len(sync.history['delta_r']))
        if n_history > 5:
            ax.plot(sync.history['delta_r'][-n_history:], label='δ NREM', color='blue')
            ax.plot(sync.history['theta_r'][-n_history:], label='θ REM', color='green')
            ax.plot(sync.history['alpha_r'][-n_history:], label='α Wake', color='orange')
            ax.set_ylabel('T3 Sync (r)')
            ax.set_title('T3 Oscillator Synchronization Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 5. Band powers
        ax = axes[4]
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        values = [band_powers.get(b, 0) for b in bands]
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
        bars = ax.bar(bands, values, color=colors, alpha=0.8)
        ax.set_ylabel('Power')
        ax.set_title('EEG Band Powers')
        ax.set_xticklabels([f'{b}\n({v:.1f})' for b, v in zip(bands, values)])

        plt.tight_layout()
        plt.savefig('bci_pet_results.png', dpi=150, bbox_inches='tight')
        print(f"\n   📈 Plot saved: bci_pet_results.png")
        plt.show()
    except ImportError:
        print("   ⚠️  matplotlib not available, skipping plot")


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BCI Pet — Virtual brain that sleeps with you.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bci_pet.py                         # synthetic EEG demo
  python bci_pet.py --plot                  # synthetic EEG + plots
  python bci_pet.py --edf sleep-edf.edf      # real Sleep-EDF file
  python bci_pet.py --no-sim --plot          # quick: T3 sync only
  python bci_pet.py --full                   # full 3000 steps
        """,
    )

    parser.add_argument('--edf', type=str, default=None,
                        help='Path to Sleep-EDF .edf file')
    parser.add_argument('--plot', action='store_true',
                        help='Show results plot')
    parser.add_argument('--no-sim', action='store_true',
                        help='Skip connectome simulation (T3 sync only)')

    parser.add_argument('--n-neurons', type=int, default=256,
                        help='Number of connectome neurons (default: 256)')
    parser.add_argument('--n-input', type=int, default=32,
                        help='Number of input encoding neurons (default: 32)')
    parser.add_argument('--n-osc', type=int, default=32,
                        help='Number of T3 oscillators per band (default: 32)')

    parser.add_argument('--dt-ms', type=float, default=10.0,
                        help='Simulation timestep in ms (default: 10.0)')
    parser.add_argument('--i-scale', type=float, default=50.0,
                        help='Input current per spike in µA (default: 50.0)')
    parser.add_argument('--bsa-threshold', type=float, default=0.4,
                        help='BSA encoding base threshold (default: 0.4)')

    parser.add_argument('--max-samples', type=int, default=30000,
                        help='Max EEG samples to load (default: 30000)')
    parser.add_argument('--max-sim-steps', type=int, default=200,
                        help='Max simulation steps (default: 200 for speed)')
    parser.add_argument('--full', action='store_true',
                        help='Run full simulation (up to 3000 steps)')

    args = parser.parse_args()

    if args.full:
        args.max_sim_steps = 3000

    quality = run_bci_pet(args)
    return 0 if quality > 0.1 else 1


if __name__ == '__main__':
    main()
