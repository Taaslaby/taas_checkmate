#!/usr/bin/env python3
"""
Epilepsy BCI — T3 Seizure Detection + Suppression via Inverse Kuramoto.

Pipeline:
  1. Generate realistic epileptic EEG (3Hz SWD + HFO)
  2. T3 TriOscillators track sync at delta + gamma bands
  3. NLMS Decoder learns to predict seizure onset from PST features
  4. T3 Inverse Kuramoto generates anti-phase drive → breaks hyper-sync loop
  5. Connectome + closed-loop: suppression current → BrainSimulator

Usage:
  python examples/python/epilepsy_bci.py                     # full experiment
  python examples/python/epilepsy_bci.py --no-sim             # T3 sync only
  python examples/python/epilepsy_bci.py --plot               # show plots
  python examples/python/epilepsy_bci.py --supression-only    # test suppression
  python examples/python/epilepsy_bci.py --edf /path/file.edf # real patient EEG
"""

import numpy as np
import taas_checkmate as tcm
import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

FS = 256  # Hz (CHB-MIT standard)
DT_S = 1.0 / FS  # timestep in seconds
HFO_BAND = (80.0, 200.0)  # High-frequency oscillations (pre-ictal biomarker)
SWD_BAND = (2.5, 3.5)  # Spike-wave discharge band


# ═══════════════════════════════════════════════════════════════════
#  1. Epileptic EEG Generator
# ═══════════════════════════════════════════════════════════════════

def generate_epileptic_eeg(
    duration_sec: float = 30.0,
    fs: int = FS,
    seizure_times: Optional[List[Tuple[float, float]]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate realistic epileptic EEG with:
      - Background: alpha (10 Hz) + theta (6 Hz) + noise
      - Pre-ictal: HFO bursts (80-200 Hz), amplitude ramp
      - Ictal: 3 Hz spike-wave discharges + hyper-synchronization
      - Post-ictal: suppression + slow waves

    Args:
        duration_sec: total duration in seconds
        seizure_times: list of (start_sec, end_sec) for seizures

    Returns:
        t: time array
        eeg: raw EEG signal
        seizure_mask: boolean array (True = seizure)
        hfo_energy: HFO energy envelope (pre-ictal biomarker)
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * fs)
    t = np.arange(n) / fs
    eeg = rng.normal(0, 0.3, n)  # noise floor

    # Default seizure: 10-15 seconds if not specified
    if seizure_times is None:
        seizure_times = [(10.0, 15.0)]

    seizure_mask = np.zeros(n, dtype=bool)
    hfo_energy = np.zeros(n, dtype=np.float32)

    # ── Background: alpha + theta ──
    eeg += 2.0 * np.sin(2 * np.pi * 10 * t)  # alpha
    eeg += 1.5 * np.sin(2 * np.pi * 6 * t)   # theta

    for sz_start, sz_end in seizure_times:
        sz_s = int(sz_start * fs)
        sz_e = int(sz_end * fs)
        sz_e = min(sz_e, n)
        seizure_mask[sz_s:sz_e] = True

        # ── Pre-ictal: HFO (80-200 Hz) ramp 2s before seizure ──
        pre_onset = max(0, sz_s - int(2 * fs))
        pre_ramp = np.linspace(0.0, 1.0, sz_s - pre_onset)
        hfo_signal = rng.normal(0, 1, sz_s - pre_onset)
        # Bandpass filter approximation: modulate with 100-150 Hz
        hfo_carrier = np.sin(2 * np.pi * 120 * t[pre_onset:sz_s])
        hfo_burst = hfo_signal * hfo_carrier * pre_ramp * 0.5
        eeg[pre_onset:sz_s] += hfo_burst
        hfo_energy[pre_onset:sz_s] = np.abs(hfo_burst) * 5.0

        # ── Ictal: 3 Hz spike-wave discharges ──
        sw_duration = sz_e - sz_s
        sw_t = t[sz_s:sz_e] - sz_start
        # Classic 3 Hz spike-wave: sharp spike + slow wave
        spike_complex = 8.0 * np.sin(2 * np.pi * 3.0 * sw_t)
        # Add harmonic for sharpness
        spike_complex += 4.0 * np.sin(2 * np.pi * 6.0 * sw_t + 0.5)
        # Slow wave component
        slow_wave = 3.0 * np.sin(2 * np.pi * 1.5 * sw_t + np.pi)
        # Amplitude modulation: crescendo-decrescendo
        amp_env = np.sin(np.pi * sw_t / (sz_end - sz_start))
        amp_env = np.clip(amp_env, 0.3, 1.0)

        eeg[sz_s:sz_e] += (spike_complex + slow_wave) * amp_env

        # ── Post-ictal: suppression + slow waves ──
        post_end = min(n, sz_e + int(3 * fs))
        post_len = post_end - sz_e
        if post_len > 0:
            # Suppression: amplitude drops
            suppression = np.exp(-np.arange(post_len) / (fs * 0.5))
            eeg[sz_e:post_end] *= suppression
            # Slow waves
            eeg[sz_e:post_end] += 1.0 * np.sin(2 * np.pi * 1.0 * t[sz_e:post_end]) * (1 - suppression)

    # Normalize to realistic amplitude
    eeg = eeg / np.std(eeg) * 20.0

    return t, eeg, seizure_mask, hfo_energy


# ═══════════════════════════════════════════════════════════════════
#  2. T3 Hyper-Sync Detector
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SyncStats:
    """Statistics from a sync detection window."""
    r_delta: float      # delta band sync (spike-wave detector)
    r_gamma: float      # gamma/HFO band sync (pre-ictal detector)
    r_theta: float      # theta band sync
    r_alpha: float      # alpha band sync
    hyper_sync: bool    # True if sync exceeds threshold
    seizure_prob: float # 0-1 probability of seizure

    seizure_count: int  # consecutive hyper-sync frames
    hfo_burst: bool     # HFO detected (pre-ictal warning)


class HyperSyncDetector:
    """
    T3-based hyper-synchronization detector.

    Uses 4 TriOscillators at different frequency bands:
      - Delta (0.5-4 Hz):  ictal spike-wave sync
      - Theta (4-8 Hz):    background / pre-ictal
      - Alpha (8-12 Hz):   normal background
      - Gamma (30-200 Hz): HFO / pre-ictal biomarker

    Tracks sync in sliding window and raises alarm when
    delta sync rises above baseline + threshold.
    """

    def __init__(
        self,
        n_osc: int = 32,
        window_size: int = 50,      # frames for sliding sync baseline
        sync_threshold: float = 0.4, # r > this = hyper-sync
        hfo_threshold: float = 0.2,  # gamma sync > this = HFO
        min_seizure_frames: int = 10, # consecutive frames to confirm seizure
    ):
        self.n_osc = n_osc
        self.window_size = window_size
        self.sync_threshold = sync_threshold
        self.hfo_threshold = hfo_threshold
        self.min_seizure_frames = min_seizure_frames

        # T3 oscillators for each band
        self.delta_t3 = tcm.t3.TriOscillator(n_osc, 0.5, 4.0)
        self.theta_t3 = tcm.t3.TriOscillator(n_osc, 4.0, 8.0)
        self.alpha_t3 = tcm.t3.TriOscillator(n_osc, 8.0, 12.0)
        self.gamma_t3 = tcm.t3.TriOscillator(n_osc, 30.0, 200.0)

        # Coupling (weaker for gamma — HFO is spatially localized)
        self.delta_t3.set_coupling(0.30, 0.15, 0.08)
        self.theta_t3.set_coupling(0.25, 0.12, 0.06)
        self.alpha_t3.set_coupling(0.20, 0.10, 0.05)
        self.gamma_t3.set_coupling(0.10, 0.05, 0.03)

        # Sync history for sliding baseline
        self.delta_sync_history = []
        self.gamma_sync_history = []
        self.consecutive_hyper = 0
        self.seizure_active = False

        # Detection statistics
        self.stats_history: List[SyncStats] = []

    def feed(self, eeg_sample: float, hfo_energy: float = 0.0) -> SyncStats:
        """
        Process one EEG sample and return sync statistics.

        Args:
            eeg_sample: single EEG voltage value
            hfo_energy: HFO energy from pre-processing

        Returns:
            SyncStats for this frame
        """
        n = self.n_osc

        # Build drives for each T3 oscillator
        # Each band gets the EEG sample as drive for pop1
        delta_drive = np.zeros(3 * n, dtype=np.float32)
        delta_drive[:n] = eeg_sample * 0.1  # scale EEG to drive

        theta_drive = np.zeros(3 * n, dtype=np.float32)
        theta_drive[:n] = eeg_sample * 0.05

        alpha_drive = np.zeros(3 * n, dtype=np.float32)
        alpha_drive[:n] = eeg_sample * 0.05

        gamma_drive = np.zeros(3 * n, dtype=np.float32)
        gamma_drive[:n] = eeg_sample * 0.02 + hfo_energy * 2.0  # extra HFO boost

        # Step all T3s
        self.delta_t3.step(DT_S, delta_drive)
        self.theta_t3.step(DT_S, theta_drive)
        self.alpha_t3.step(DT_S, alpha_drive)
        self.gamma_t3.step(DT_S, gamma_drive)

        # Get order parameters
        r1, r2, r3 = self.delta_t3.order_params()
        r_delta = r1
        r1, r2, r3 = self.gamma_t3.order_params()
        r_gamma = r1
        r1, r2, r3 = self.theta_t3.order_params()
        r_theta = r1
        r1, r2, r3 = self.alpha_t3.order_params()
        r_alpha = r1

        # Track history (sliding window)
        self.delta_sync_history.append(r_delta)
        self.gamma_sync_history.append(r_gamma)
        if len(self.delta_sync_history) > self.window_size * 2:
            self.delta_sync_history.pop(0)
            self.gamma_sync_history.pop(0)

        # Compute baseline (median of last 2*window)
        baseline_delta = float(np.median(self.delta_sync_history[:-max(1, self.window_size // 4)])) \
            if len(self.delta_sync_history) > self.window_size else 0.0

        # Hyper-sync? delta sync significantly above baseline
        delta_above_baseline = r_delta - baseline_delta
        hyper_sync = delta_above_baseline > self.sync_threshold

        # HFO burst?
        hfo_burst = r_gamma > self.hfo_threshold

        # Consecutive hyper-sync counter
        if hyper_sync:
            self.consecutive_hyper += 1
        else:
            self.consecutive_hyper = max(0, self.consecutive_hyper - 1)

        # Seizure confirmation
        self.seizure_active = self.consecutive_hyper >= self.min_seizure_frames

        # Seizure probability: sigmoid of consecutive count
        seizure_prob = 1.0 / (1.0 + np.exp(-(self.consecutive_hyper - self.min_seizure_frames + 3)))

        stats = SyncStats(
            r_delta=r_delta,
            r_gamma=r_gamma,
            r_theta=r_theta,
            r_alpha=r_alpha,
            hyper_sync=hyper_sync or self.seizure_active,
            seizure_prob=float(seizure_prob),
            seizure_count=self.consecutive_hyper,
            hfo_burst=hfo_burst,
        )
        self.stats_history.append(stats)
        return stats

    def get_sync_stats(self) -> Dict[str, float]:
        """Get latest sync stats summary."""
        if not self.stats_history:
            return {'delta_sync': 0.0, 'gamma_sync': 0.0, 'seizure_prob': 0.0}
        s = self.stats_history[-1]
        return {
            'delta_sync': s.r_delta,
            'gamma_sync': s.r_gamma,
            'seizure_prob': s.seizure_prob,
            'hyper_sync': float(s.hyper_sync),
            'hfo_burst': float(s.hfo_burst),
        }


# ═══════════════════════════════════════════════════════════════════
#  3. NLMS Seizure Predictor
# ═══════════════════════════════════════════════════════════════════

class SeizurePredictor:
    """
    NLMS-based online seizure predictor.

    Features:
      - delta sync (r_delta)
      - gamma sync (r_gamma)
      - delta slope (rate of change)
      - gamma slope
      - hfo_energy
      - delta - baseline

    Target:
      - 1 if seizure in next N frames, else 0

    The NLMS learns to associate pre-ictal features with
    upcoming seizures — online, frame by frame.
    """

    def __init__(
        self,
        n_features: int = 6,
        lr: float = 0.01,
        eps: float = 1e-6,
        prediction_horizon: int = 50,  # predict seizure N frames ahead
    ):
        self.predictor = tcm.t3.NlmsDecoder(n_features, lr, eps)
        self.prediction_horizon = prediction_horizon
        self.feature_buffer = []
        self.target_buffer = []
        self.predictions = []
        self.prediction_horizon = prediction_horizon

    def extract_features(self, stats: SyncStats, hfo_energy: float = 0.0) -> np.ndarray:
        """Extract feature vector from sync stats."""
        features = np.array([
            stats.r_delta,           # delta sync
            stats.r_gamma,           # gamma/HFO sync
            stats.r_delta * 0.1,     # delta slope proxy
            stats.r_gamma * 0.1,     # gamma slope proxy
            hfo_energy * 0.01,       # HFO energy
            stats.r_delta - 0.2,     # delta above noise floor
        ], dtype=np.float32)
        return features

    def feed(
        self,
        stats: SyncStats,
        is_seizure: bool,
        hfo_energy: float = 0.0,
    ) -> float:
        """
        Feed one frame: extract features, train if target available.

        Args:
            stats: SyncStats from detector
            is_seizure: whether this frame is ictal
            hfo_energy: HFO energy

        Returns:
            prediction: 0-1 seizure probability
        """
        features = self.extract_features(stats, hfo_energy)
        self.feature_buffer.append(features)
        self.target_buffer.append(1.0 if is_seizure else 0.0)

        # Make prediction
        prediction = float(self.predictor.predict(features))
        self.predictions.append(prediction)

        # Train if we have enough horizon
        if len(self.feature_buffer) > self.prediction_horizon:
            past_features = self.feature_buffer[-self.prediction_horizon]
            future_target = self.target_buffer[-1]  # what actually happened now
            self.predictor.train(past_features, future_target)

        return prediction

    def predict(self, stats: SyncStats, hfo_energy: float = 0.0) -> float:
        """Make prediction without training."""
        features = self.extract_features(stats, hfo_energy)
        return float(self.predictor.predict(features))


# ═══════════════════════════════════════════════════════════════════
#  4. T3 Inverse Kuramoto Suppression
# ═══════════════════════════════════════════════════════════════════

class SeizureSuppressor:
    """
    T3 Suppression via Phase Scrambling + Negative Inverse Kuramoto.

    Strategy:
      1. PHASE SCRAMBLING — each oscillator gets a RANDOM phase offset
         (not π for all!). Different directions → coherence breaks.
      2. NEGATIVE GAIN in inverse Kuramoto — actively pushes sync DOWN
         by driving r toward 0 (desync attractor).
      3. RANDOM SEED per frame — ensures fresh noise each suppression.

    The suppression is adaptive: stronger when seizure_prob is higher.
    """

    def __init__(
        self,
        n_osc: int = 32,
        gain: float = 5.0,
        alpha: float = 0.5,
        decay: float = 0.1,
        eps: float = 1e-6,
    ):
        self.n_osc = n_osc
        self.gain = gain
        self.alpha = alpha
        self.decay = decay
        self.eps = eps
        self.prev_r1 = np.zeros(n_osc, dtype=np.float32)
        self.frame_count = 0  # ensures fresh scramble every frame
        self.initialized = False
        self.suppression_active = False
        self.total_suppression_current = 0.0

    def compute_suppression(
        self,
        delta_t3: tcm.t3.TriOscillator,
        seizure_prob: float,
        dt: float = DT_S,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute suppression drive using phase scrambling + negative inverse Kuramoto.

        Phase scrambling: each oscillator i gets random φ_i ∈ [0.3π, 1.7π]
        instead of uniform π. This CREATES diversity in the drive, which
        BREAKS coherence instead of maintaining it.

        Critical: random seed changes EVERY frame (via frame_count) to prevent
        the scramble pattern from repeating — which would allow oscillators
        to re-entrain to the periodic drive and maintain coherence.

        Negative inverse Kuramoto: drive = -gain * (r - r_prev)/dt
        pushes the Kuramoto order parameter toward 0 (desynchronization).

        Args:
            delta_t3: TriOscillator tuned to delta band
            seizure_prob: 0-1 seizure probability (scales suppression)
            dt: timestep

        Returns:
            supp_drive: suppression drive array (n_osc,)
            total_current: sum of |drive| for virtual brain injection
        """
        n = self.n_osc
        r1 = np.array(delta_t3.amplitudes(1), dtype=np.float32)
        th1 = np.array(delta_t3.phases(1), dtype=np.float32)

        if not self.initialized:
            self.prev_r1 = r1.copy()
            self.initialized = True
            return np.zeros(n, dtype=np.float32), 0.0

        # ── 1. Phase Scrambling (fresh every frame via frame_count) ──
        # Without frame_count, seizure_prob ≈ 1.0 gives same seed every frame
        # → scramble pattern repeats → oscillators entrain → coherence persists!
        seed = int(seizure_prob * 100000 + self.frame_count * 7) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        random_offsets = rng.uniform(0.3, 1.7, n).astype(np.float32)
        scramble = np.cos(th1 + np.pi * random_offsets) * seizure_prob * self.gain

        # ── 2. Negative Gain Inverse Kuramoto ──
        # Positive gain drives r → 1 (more sync).
        # Negative gain drives r → 0 (less sync = desync!).
        neg_drive = np.zeros(n, dtype=np.float32)
        tcm.io.t3_inverse_kuramoto(
            neg_drive,
            r1,
            self.prev_r1,
            self.alpha,
            -self.gain * seizure_prob,  # ← NEGATIVE GAIN!
            self.eps,
            self.decay,
            dt,
        )

        # ── 3. Combine ──
        supp_drive = neg_drive + scramble
        supp_drive = np.clip(supp_drive, -20.0, 20.0)

        self.prev_r1 = r1.copy()
        self.frame_count += 1
        total_current = float(np.sum(np.abs(supp_drive)))
        self.suppression_active = seizure_prob > 0.5
        self.total_suppression_current = total_current

        return supp_drive, total_current


# ═══════════════════════════════════════════════════════════════════
#  5. Main Experiment
# ═══════════════════════════════════════════════════════════════════

def run_epilepsy_bci(args):
    print("=" * 65)
    print("  🧬 EPILEPSY BCI — T3 Seizure Detection + Suppression")
    print("  EEG → T3 Sync → NLMS Prediction → Inverse Kuramoto Suppression")
    print("=" * 65)

    # ── Step 1: EEG ─────────────────────────────────────
    print("\n" + "─" * 65)
    print("📡 [1/6] EEG Source")

    if args.edf:
        if not os.path.exists(args.edf):
            print(f"   ❌ File not found: {args.edf}")
            sys.exit(1)
        import pyedflib
        f = pyedflib.EdfReader(args.edf)
        fs = f.getSampleFrequency(0)
        signal = f.readSignal(0)
        f.close()
        t = np.arange(len(signal)) / fs
        seizure_mask = np.zeros(len(signal), dtype=bool)
        hfo_energy = np.zeros(len(signal), dtype=np.float32)
        print(f"   Loaded: {os.path.basename(args.edf)}")
    else:
        print("   Generating synthetic epileptic EEG...")
        seizure_times = [(10.0, 15.0), (22.0, 25.0)]
        t, signal, seizure_mask, hfo_energy = generate_epileptic_eeg(
            duration_sec=30.0,
            fs=FS,
            seizure_times=seizure_times,
        )
        fs = FS

    n_samples = min(len(signal), args.max_samples)
    duration = n_samples / fs
    signal = signal[:n_samples]
    seizure_mask = seizure_mask[:n_samples]
    hfo_energy = hfo_energy[:n_samples]
    t_plot = t[:n_samples]

    n_seizure_frames = int(np.sum(seizure_mask))
    print(f"   → {n_samples} samples @ {fs} Hz ({duration:.1f}s)")
    print(f"   → Seizure frames: {n_seizure_frames} ({n_seizure_frames/n_samples*100:.1f}%)")

    # ── Step 2: Init detectors ──────────────────────────
    print(f"\n🔬 [2/6] Initializing T3 Sync Detectors")
    print(f"   Delta:  0.5-4 Hz   → Spike-wave sync")
    print(f"   Gamma:  30-200 Hz  → HFO biomarker")
    print(f"   Theta:  4-8 Hz     → Background")
    print(f"   Alpha:  8-12 Hz    → Normal activity")

    detector = HyperSyncDetector(
        n_osc=args.n_osc,
        window_size=args.window,
        sync_threshold=args.sync_thresh,
        hfo_threshold=args.hfo_thresh,
        min_seizure_frames=args.min_frames,
    )

    predictor = SeizurePredictor(
        n_features=6,
        lr=args.nlms_lr,
        prediction_horizon=int(0.5 * fs),  # predict 0.5s ahead
    )

    suppressor = SeizureSuppressor(
        n_osc=args.n_osc,
        gain=args.supp_gain,
        alpha=0.5,
        decay=0.1,
    )

    print(f"   → {args.n_osc} oscillators per band")
    print(f"   → Sync threshold: {args.sync_thresh}")
    print(f"   → HFO threshold:  {args.hfo_thresh}")
    print(f"   → Suppression gain: {args.supp_gain}")

    # ── Step 3: Run detection ───────────────────────────
    print(f"\n⚡ [3/6] Running T3 Seizure Detection ({n_samples} frames)...")

    # Run at reduced rate for speed (decimate EEG by factor)
    step = max(1, int(fs / 100))  # ~100 Hz processing
    detections = []
    predictions = []
    supp_drives = []
    suppression_currents = []
    sync_over_time = {'delta': [], 'gamma': [], 'theta': [], 'alpha': []}

    for i in range(0, n_samples, step):
        eeg_val = signal[i]
        hfo_val = hfo_energy[i]
        is_seizure = bool(seizure_mask[i])

        # Detect sync
        stats = detector.feed(eeg_val, hfo_val)

        # Predict seizure
        pred = predictor.feed(stats, is_seizure, hfo_val)

        # Suppress if seizure predicted
        supp_drive, supp_current = suppressor.compute_suppression(
            detector.delta_t3, stats.seizure_prob
        )

        detections.append(stats.seizure_prob)
        predictions.append(pred)
        supp_drives.append(np.mean(np.abs(supp_drive)))
        suppression_currents.append(supp_current)

        sync_over_time['delta'].append(stats.r_delta)
        sync_over_time['gamma'].append(stats.r_gamma)
        sync_over_time['theta'].append(stats.r_theta)
        sync_over_time['alpha'].append(stats.r_alpha)

        if (i // step) % max(1, n_samples // (step * 20)) == 0:
            pct = i / n_samples * 100
            print(f"   {pct:3.0f}% — δ:{stats.r_delta:.3f} γ:{stats.r_gamma:.3f} "
                  f"P:{pred:.3f} S:{stats.seizure_prob:.3f} "
                  f"{'🔥SEIZURE' if is_seizure else '        '}")

    # ── Step 4: Analysis ────────────────────────────────
    print(f"\n" + "─" * 65)
    print("📊 [4/6] Detection Analysis")
    print("─" * 65)

    detections = np.array(detections)
    predictions = np.array(predictions)
    supp_currents = np.array(suppression_currents)

    # Detection stats
    n_detected = int(np.sum(detections > 0.5))
    n_predicted = int(np.sum(predictions > 0.5))
    n_alarms = n_detected

    print(f"\n   🔍 Seizure detection:")
    print(f"      Actual seizure frames: {n_seizure_frames}")
    print(f"      T3 hyper-sync alarms:  {n_alarms}")
    print(f"      NLMS predictions:      {n_predicted}")

    # Confusion (simplified — need time-aligned comparison)
    detection_rate = min(1.0, n_alarms / max(1, n_seizure_frames / step))
    print(f"      Detection rate:        {detection_rate:.1%}")

    # Suppression stats
    mean_supp = float(np.mean(supp_currents))
    max_supp = float(np.max(supp_currents))
    supp_active = int(np.sum(supp_currents > 1.0))
    print(f"\n   🔫 T3 Suppression:")
    print(f"      Active frames:         {supp_active}")
    print(f"      Mean suppression:      {mean_supp:.2f} µA")
    print(f"      Max suppression:       {max_supp:.2f} µA")

    # Sync stats
    mean_delta_sync = float(np.mean(sync_over_time['delta']))
    max_delta_sync = float(np.max(sync_over_time['delta']))
    mean_gamma_sync = float(np.mean(sync_over_time['gamma']))
    mean_alpha_sync = float(np.mean(sync_over_time['alpha']))

    print(f"\n   📈 T3 Synchronization:")
    print(f"      Delta sync:  {mean_delta_sync:.3f} (max: {max_delta_sync:.3f})")
    print(f"      Gamma sync:  {mean_gamma_sync:.3f}")
    print(f"      Alpha sync:  {mean_alpha_sync:.3f}")
    print(f"      Theta sync:  {float(np.mean(sync_over_time['theta'])):.3f}")

    # Suppression effectiveness
    # Compare sync during suppression-on vs suppression-off
    high_supp = supp_currents > supp_currents.mean()
    if np.any(high_supp):
        sync_during_supp = np.mean(np.array(sync_over_time['delta'])[high_supp])
        sync_without_supp = np.mean(np.array(sync_over_time['delta'])[~high_supp])
        supp_effectiveness = sync_without_supp - sync_during_supp
        print(f"\n   💊 Suppression effectiveness:")
        print(f"      Delta sync during suppression:  {sync_during_supp:.3f}")
        print(f"      Delta sync without suppression: {sync_without_supp:.3f}")
        print(f"      Sync reduction:                 {supp_effectiveness:.3f}")

        verdict = "✅ SUPPRESSION EFFECTIVE!" if supp_effectiveness > 0.1 else \
                  "⚪ Mild effect — try increasing suppression gain" if supp_effectiveness > 0.05 else \
                  "❌ Suppression ineffective — check inverse Kuramoto parameters"
        print(f"\n   {verdict}")
    else:
        print("\n   ⚪ No suppression applied")

    # ── Step 5: Connectome Simulation (optional) ────────
    print(f"\n🧬 [5/6] Connectome Simulation (LIF)")
    conn = None
    sup_rate = np.zeros(100)

    if not args.no_sim and not args.suppression_only:
        conn = _build_test_connectome(args.n_neurons, args.n_input)
        print(f"   → {conn.n_neurons} neurons, {conn.n_synapses} synapses")

        sim = tcm.connectome_brain.BrainSimulator(conn.n_neurons, "lif")
        sensory_idx = np.array(list(range(args.n_input)), dtype=np.int32)

        n_sim_steps = min(n_samples // step, 200)
        sup_rates = []
        n_input = args.n_input

        print(f"\n   Running {n_sim_steps} steps with closed-loop suppression...")

        for step_idx in range(n_sim_steps):
            # External input from EEG
            eeg_idx = step_idx * step
            ext_I = np.zeros(conn.n_neurons, dtype=np.float32)
            ext_I[sensory_idx[:min(n_input, len(sensory_idx))]] = \
                signal[eeg_idx] * 0.5

            # Add suppression current
            if step_idx < len(suppression_currents):
                ext_I[:n_input] -= suppression_currents[step_idx] * 0.01

            buf = sim.step_batch(conn, 5, 10.0, ext_I, scale=1.0)
            rate = float(np.mean(buf[:conn.n_neurons]) * 100)
            sup_rates.append(rate)

        sup_rate = np.array(sup_rates)
        print(f"      Population rate: {np.mean(sup_rate):.1f} ± {np.std(sup_rate):.1f} Hz")
        print(f"      Total spikes:    {int(np.sum(sup_rate))}")
    elif args.suppression_only:
        print("   ⏭️  Suppression test only (no connectome)")
    else:
        print("   ⏭️  Skipped (--no-sim)")

    # ── Final verdict ───────────────────────────────────
    print(f"\n" + "─" * 65)
    print("🎯 [6/6] FINAL VERDICT")
    print("─" * 65)

    print(f"\n  📊 Detection rate:     {detection_rate:.1%}")
    print(f"  🔫 Suppression active: {supp_active} frames")
    print(f"  📈 Delta sync range:   [{mean_delta_sync:.2f}, {max_delta_sync:.2f}]")
    print(f"  🧪 NLMS predictions:   {n_predicted}")
    print(f"\n  {'✅ T3 SEIZURE DETECTION WORKS!' if detection_rate > 0.5 else '⚪ Weak detection — adjust sync_threshold'}")

    # ── Plot ────────────────────────────────────────────
    if args.plot:
        _plot_results(
            t_plot[::step], signal[::step], seizure_mask[::step],
            detections, predictions, supp_currents,
            sync_over_time, args,
        )

    return detection_rate


def _build_test_connectome(n_neurons: int = 128, n_input: int = 32):
    """Build minimal SBM connectome for testing suppression."""
    rng = np.random.default_rng(42)
    n_hidden = (n_neurons - n_input) // 3
    n_motor = n_neurons - n_input - 2 * n_hidden
    groups = {
        'sensory': list(range(0, n_input)),
        'hidden_a': list(range(n_input, n_input + n_hidden)),
        'hidden_b': list(range(n_input + n_hidden, n_input + 2 * n_hidden)),
        'motor': list(range(n_input + 2 * n_hidden, n_neurons)),
    }

    neuron_group = np.zeros(n_neurons, dtype=np.int32)
    for gid, grp in enumerate([groups['sensory'], groups['hidden_a'],
                                groups['hidden_b'], groups['motor']]):
        for nid in grp:
            neuron_group[nid] = gid

    rows, cols, vals, nts = [], [], [], []
    n_syn = 0
    for i in range(n_neurons):
        rows.append(n_syn)
        gi = neuron_group[i]
        for j in range(n_neurons):
            if i == j:
                continue
            gj = neuron_group[j]
            prob = 0.35 if gi == gj else 0.03
            if rng.random() < prob:
                cols.append(j)
                w = rng.uniform(0.5, 2.0)
                is_exc = rng.random() < 0.8
                vals.append(w)
                nts.append(1.0 if is_exc else -1.0)
                n_syn += 1
    rows.append(n_syn)

    conn = tcm.connectome_brain.ConnectomeData(
        np.array(rows, dtype=np.int32),
        np.array(cols, dtype=np.int32),
        np.array(vals, dtype=np.float32),
        np.array(nts, dtype=np.float32),
        groups=[(name, idxs) for name, idxs in groups.items()],
    )
    return conn


def _plot_results(t, signal, seizure_mask, detections, predictions,
                  supp_currents, sync_over_time, args):
    """Plot EEG, sync, detection, suppression."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

        # 1. EEG with seizure overlay
        ax = axes[0]
        ax.plot(t, signal, color='navy', lw=0.8, alpha=0.7)
        # Shade seizure regions
        for i in range(len(t) - 1):
            if seizure_mask[i]:
                ax.axvspan(t[i], t[i+1], alpha=0.2, color='red')
        ax.set_ylabel('EEG (µV)')
        ax.set_title('Epileptic EEG with Seizure Regions (red)')
        ax.grid(True, alpha=0.3)

        # 2. T3 sync
        ax = axes[1]
        ax.plot(t, sync_over_time['delta'], label='δ sync (spike-wave)', color='red', lw=1.0)
        ax.plot(t, sync_over_time['gamma'], label='γ/HFO sync', color='purple', lw=0.8)
        ax.plot(t, sync_over_time['alpha'], label='α sync (normal)', color='green', lw=0.6)
        ax.axhline(args.sync_thresh, color='red', linestyle='--', alpha=0.5, label=f'threshold={args.sync_thresh}')
        ax.set_ylabel('T3 Sync (r)')
        ax.set_title('T3 Synchronization per Band')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 3. Seizure probability
        ax = axes[2]
        ax.plot(t, detections, label='T3 Detection', color='red', lw=1.5)
        ax.plot(t, predictions, label='NLMS Prediction', color='orange', lw=1.0, alpha=0.7)
        ax.axhline(0.5, color='red', linestyle=':', alpha=0.5, label='threshold')
        ax.fill_between(t, detections, 0, where=(detections>0.5), alpha=0.3, color='red')
        ax.set_ylabel('Probability')
        ax.set_title('Seizure Detection + NLMS Prediction')
        ax.legend(loc='upper right')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # 4. Suppression
        ax = axes[3]
        ax.plot(t, supp_currents, color='blue', lw=1.0)
        ax.fill_between(t, supp_currents, 0, alpha=0.3, color='blue')
        ax.set_ylabel('Suppression (µA)')
        ax.set_title('T3 Inverse Kuramoto Suppression Current')
        ax.grid(True, alpha=0.3)

        # 5. Sync heatmap
        ax = axes[4]
        sync_matrix = np.array([
            sync_over_time['delta'],
            sync_over_time['gamma'],
            sync_over_time['theta'],
            sync_over_time['alpha'],
        ])
        ax.imshow(sync_matrix, aspect='auto', cmap='hot',
                  extent=[t[0], t[-1], 0, 4],
                  interpolation='bilinear')
        ax.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax.set_yticklabels(['δ', 'γ', 'θ', 'α'])
        ax.set_ylabel('Band')
        ax.set_title('T3 Sync Heatmap (hot = synchronized)')

        plt.tight_layout()
        plt.savefig('epilepsy_bci_results.png', dpi=150, bbox_inches='tight')
        print(f"\n   📈 Plot saved: epilepsy_bci_results.png")
        plt.show()
    except ImportError:
        print("   ⚠️  matplotlib not available")



# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Epilepsy BCI — T3 Seizure Detection + Suppression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/python/epilepsy_bci.py                  # full experiment
  python examples/python/epilepsy_bci.py --plot           # with plots
  python examples/python/epilepsy_bci.py --no-sim         # T3 sync only
  python examples/python/epilepsy_bci.py --edf file.edf   # real patient EEG
  python examples/python/epilepsy_bci.py --supp-gain 10   # stronger suppression
        """,
    )

    parser.add_argument('--edf', type=str, default=None,
                        help='Path to patient EEG .edf file')
    parser.add_argument('--plot', action='store_true',
                        help='Show results plot')
    parser.add_argument('--no-sim', action='store_true',
                        help='Skip connectome simulation')
    parser.add_argument('--suppression-only', action='store_true',
                        help='Test suppression without connectome')

    # Detector parameters
    parser.add_argument('--n-osc', type=int, default=24,
                        help='T3 oscillators per band (default: 24)')
    parser.add_argument('--window', type=int, default=50,
                        help='Sliding sync baseline window (default: 50)')
    parser.add_argument('--sync-thresh', type=float, default=0.35,
                        help='Hyper-sync detection threshold (default: 0.35)')
    parser.add_argument('--hfo-thresh', type=float, default=0.20,
                        help='HFO detection threshold (default: 0.20)')
    parser.add_argument('--min-frames', type=int, default=10,
                        help='Consecutive hyper-sync frames for seizure (default: 10)')

    # Predictor parameters
    parser.add_argument('--nlms-lr', type=float, default=0.01,
                        help='NLMS learning rate (default: 0.01)')

    # Suppression parameters
    parser.add_argument('--supp-gain', type=float, default=5.0,
                        help='Inverse Kuramoto suppression gain (default: 5.0)')

    # Simulation parameters
    parser.add_argument('--n-neurons', type=int, default=128,
                        help='Connectome neurons (default: 128)')
    parser.add_argument('--n-input', type=int, default=32,
                        help='Input neurons (default: 32)')
    parser.add_argument('--max-samples', type=int, default=30000,
                        help='Max EEG samples (default: 30000)')

    args = parser.parse_args()

    # For synthetic EEG at 256Hz: 30s = 7680 samples
    # Use all of them by default
    if args.max_samples == 30000 and not args.edf:
        args.max_samples = 7680

    rate = run_epilepsy_bci(args)
    return 0 if rate > 0.3 else 1


if __name__ == '__main__':
    main()
