#!/usr/bin/env python3
"""
⚡ Spike Analysis Toolbox — Metrics, Visualization, Analysis

Фишки:
- Spike train metrics: van Rossum, Victor-Purpura, ISI/CV, Pearson corr
- PSTH (Peri-Stimulus Time Histogram)
- Spike-Triggered Average (STA) — receptive field estimation
- LFP proxy from multi-electrode recordings
- Poisson spike generator (wraps neuron_engine.PyPoissonGenerator)
- Raster plot + firing rate visualization
- Mutual information estimation (binless)

Использует neuron_engine для генерации Poisson и LFP.
"""

import numpy as np
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

try:
    import neuron_engine
except ImportError:
    import sys, subprocess
    sys.path.insert(0, "target/release")
    subprocess.run(["maturin", "develop", "--release", "-q"], cwd="..", check=True)
    import neuron_engine


# ═══════════════════════════════════════════════════════════
#  1. Spike Train Metrics
# ═══════════════════════════════════════════════════════════

def van_rossum_distance(spikes_a: np.ndarray, spikes_b: np.ndarray,
                         tau: float = 10.0, dt: float = 1.0) -> float:
    """
    Van Rossum distance: фильтрует spike trains экспоненциальным ядром,
    затем считает L2 расстояние между ними.
    
    D² = 1/τ ∫ (f_a(t) - f_b(t))² dt
    
    где f(t) = Σ exp(-(t - t_i)/τ) for t_i < t
    
    Args:
        spikes_a, spikes_b: spike trains (binary arrays), same length
        tau: time constant of exponential kernel (ms)
        dt: timestep (ms)
    
    Returns:
        Van Rossum distance (normalized)
    """
    n = min(len(spikes_a), len(spikes_b))
    if n == 0:
        return 0.0
    
    # Filter with exponential kernel
    fa = np.zeros(n, dtype=np.float64)
    fb = np.zeros(n, dtype=np.float64)
    decay = np.exp(-dt / tau)
    
    for t in range(1, n):
        fa[t] = decay * fa[t-1] + spikes_a[t]
        fb[t] = decay * fb[t-1] + spikes_b[t]
    
    # L2 distance
    dist = np.sqrt(np.sum((fa - fb) ** 2) * dt / tau)
    return float(dist)


def victor_purpura_dist(spikes_a: np.ndarray, spikes_b: np.ndarray,
                         cost_per_spike: float = 1.0) -> float:
    """
    Victor-Purpura distance: минимальная стоимость превращения
    одного spike train в другой через операции:
    - Insert/delete spike: cost 1
    - Shift spike in time: cost = |Δt| * cost_per_shift
    
    Simplified version: uses collapsed spike times and DP.
    
    Args:
        spikes_a, spikes_b: binary spike trains
        cost_per_spike: cost of deleting/inserting a spike
    
    Returns:
        Victor-Purpura distance
    """
    # Extract spike times
    times_a = np.where(spikes_a > 0)[0].astype(np.float64)
    times_b = np.where(spikes_b > 0)[0].astype(np.float64)
    
    na, nb = len(times_a), len(times_b)
    if na == 0:
        return float(nb) * cost_per_spike
    if nb == 0:
        return float(na) * cost_per_spike
    
    # DP for edit distance with shift cost
    cost_shift = cost_per_spike
    dp = np.zeros((na + 1, nb + 1), dtype=np.float64)
    dp[:, 0] = np.arange(na + 1) * cost_per_spike
    dp[0, :] = np.arange(nb + 1) * cost_per_spike
    
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            shift_cost = min(abs(times_a[i-1] - times_b[j-1]), cost_per_spike)
            dp[i, j] = min(
                dp[i-1, j] + cost_per_spike,      # delete
                dp[i, j-1] + cost_per_spike,      # insert
                dp[i-1, j-1] + shift_cost,        # shift
            )
    
    return float(dp[na, nb])


def isi_cv(spike_train: np.ndarray) -> Tuple[float, float]:
    """
    Inter-Spike Interval (ISI) metrics.
    
    Returns:
        cv_isi: Coefficient of Variation of ISI (1 = Poisson, <1 = regular, >1 = bursty)
        mean_isi: mean ISI in timesteps
    """
    times = np.where(spike_train > 0)[0]
    if len(times) < 2:
        return 0.0, float('inf')
    
    isi = np.diff(times).astype(np.float64)
    if len(isi) == 0 or isi.mean() == 0:
        return 0.0, float('inf')
    
    cv = float(isi.std() / isi.mean())
    mean_isi = float(isi.mean())
    return cv, mean_isi


def firing_rate(spike_train: np.ndarray, dt: float = 1.0) -> float:
    """Firing rate in Hz."""
    return float(np.sum(spike_train > 0)) / (len(spike_train) * dt / 1000.0)


def pairwise_correlation(spikes: np.ndarray) -> np.ndarray:
    """
    Pairwise Pearson correlation matrix for multiple spike trains.
    
    Args:
        spikes: [n_neurons × n_time] matrix (int32 or float)
    
    Returns:
        corr: [n_neurons × n_neurons] correlation matrix
    """
    n_neurons = spikes.shape[0]
    corr = np.ones((n_neurons, n_neurons), dtype=np.float32)
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            ci = np.corrcoef(spikes[i].astype(np.float64),
                             spikes[j].astype(np.float64))[0, 1]
            if not np.isnan(ci):
                corr[i, j] = ci
                corr[j, i] = ci
    return corr


# ═══════════════════════════════════════════════════════════
#  2. PSTH — Peri-Stimulus Time Histogram
# ═══════════════════════════════════════════════════════════

def compute_psth(spike_trains: np.ndarray, stimulus_times: np.ndarray,
                 window_before: float = 50.0, window_after: float = 100.0,
                 bin_size: float = 5.0, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Peri-Stimulus Time Histogram.
    
    Args:
        spike_trains: [n_trials × n_time] or [n_time] matrix
        stimulus_times: trial start indices (array of int)
        window_before: ms before stimulus to include
        window_after: ms after stimulus to include
        bin_size: PSTH bin width in ms
        dt: timestep in ms
    
    Returns:
        bins: bin centers (ms relative to stimulus)
        psth: firing rate in each bin (Hz)
    """
    if spike_trains.ndim == 1:
        spike_trains = spike_trains[np.newaxis, :]
    
    n_before = int(window_before / dt)
    n_after = int(window_after / dt)
    total_window = n_before + n_after
    
    # Collect spike times relative to stimulus
    all_spikes = []
    for stim_idx in stimulus_times:
        start = int(stim_idx) - n_before
        end = int(stim_idx) + n_after
        if start < 0 or end > spike_trains.shape[1]:
            continue
        segment = spike_trains[:, start:end]
        for s in range(segment.shape[0]):
            spike_times = np.where(segment[s] > 0)[0] - n_before
            all_spikes.extend(spike_times)
    
    if len(all_spikes) == 0:
        return np.array([]), np.array([])
    
    all_spikes = np.array(all_spikes) * dt
    
    # Bin
    bin_edges = np.arange(-window_before, window_after + bin_size, bin_size)
    hist, _ = np.histogram(all_spikes, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Convert to Hz
    n_trials_used = sum(1 for s in stimulus_times
                        if int(s) - n_before >= 0 and int(s) + n_after <= spike_trains.shape[1])
    if n_trials_used > 0:
        psth = hist / (n_trials_used * spike_trains.shape[0] * bin_size / 1000.0)
    else:
        psth = np.zeros_like(hist, dtype=np.float32)
    
    return bin_centers, psth.astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  3. STA — Spike-Triggered Average
# ═══════════════════════════════════════════════════════════

def spike_triggered_average(stimulus: np.ndarray, spikes: np.ndarray,
                             window_before: int = 20) -> np.ndarray:
    """
    Spike-Triggered Average (STA) — receptive field estimation.
    
    For each spike, take the stimulus window preceding it.
    Average all windows to get the STA.
    
    Args:
        stimulus: 1D stimulus array [n_time]
        spikes: 1D spike train [n_time]
        window_before: number of timesteps before spike to include
    
    Returns:
        sta: average stimulus before spike [window_before]
    """
    spike_times = np.where(spikes > 0)[0]
    valid = spike_times[(spike_times >= window_before) & (spike_times < len(stimulus))]
    
    if len(valid) == 0:
        return np.zeros(window_before, dtype=np.float32)
    
    sta = np.zeros(window_before, dtype=np.float64)
    for t in valid:
        sta += stimulus[t - window_before:t]
    sta /= len(valid)
    
    return sta.astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  4. Mutual Information (binless)
# ═══════════════════════════════════════════════════════════

def mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """
    Mutual information I(X; Y) using histogram binning.
    
    I(X; Y) = Σ p(x,y) * log(p(x,y) / (p(x) * p(y)))
    
    Args:
        x, y: 1D arrays
        n_bins: number of bins for histogram
    
    Returns:
        MI in bits
    """
    # Use numpy histogram2d
        # 2D histogram
        hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
    except:
        x_edges = np.linspace(x.min(), x.max(), n_bins + 1)
        y_edges = np.linspace(y.min(), y.max(), n_bins + 1)
        hist_2d, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    
    p_xy = hist_2d / hist_2d.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    
    # MI = Σ p(x,y) * log2(p(x,y) / (p(x) * p(y)))
    p_xy_flat = p_xy.flatten()
    p_x_y = np.outer(p_x, p_y).flatten()
    
    valid = (p_xy_flat > 0) & (p_x_y > 0)
    mi = float(np.sum(p_xy_flat[valid] * np.log2(p_xy_flat[valid] / p_x_y[valid])))
    
    return mi


# ═══════════════════════════════════════════════════════════
#  5. LFP Proxy
# ═══════════════════════════════════════════════════════════

class LFPProxy:
    """
    LFP Proxy: multi-electrode local field potential from spikes.
    
    Каждый нейрон привязан к "электроду". LFP = smoothed spike count per electrode.
    Использует neuron_engine.compute_lfp для эффективного суммирования.
    """
    
    def __init__(self, n_neurons: int, n_electrodes: int = 8, tau_smooth: float = 20.0, dt: float = 1.0):
        """
        Args:
            n_neurons: количество нейронов
            n_electrodes: количество электродов
            tau_smooth: time constant for exponential smoothing (ms)
            dt: timestep (ms)
        """
        self.n_neurons = n_neurons
        self.n_electrodes = n_electrodes
        self.tau_smooth = tau_smooth
        self.dt = dt
        
        # Random electrode assignment
        self.electrode_map = np.random.randint(0, n_electrodes, n_neurons).astype(np.int32)
        
        # Smoothed LFP buffer
        self.lfp = np.zeros(n_electrodes, dtype=np.float32)
        self.decay = np.exp(-dt / tau_smooth) if tau_smooth > 0 else 0.0
    
    def step(self, spikes: np.ndarray) -> np.ndarray:
        """
        One step: compute LFP from spikes.
        
        Returns:
            lfp: [n_electrodes] LFP values
        """
        # Raw sum per electrode
        raw = np.zeros(self.n_electrodes, dtype=np.float32)
        neuron_engine.compute_lfp(spikes.astype(np.int32),
                                   self.electrode_map, raw, self.n_electrodes)
        
        # Exponential smoothing
        self.lfp = self.decay * self.lfp + (1.0 - self.decay) * raw
        
        return self.lfp.copy()


# ═══════════════════════════════════════════════════════════
#  6. Population Generator — параметры для разнородных популяций
# ═══════════════════════════════════════════════════════════

def generate_heterogeneous_population(
    n: int,
    model: str = "lif_at",
    seed: int = 42,
) -> dict:
    """
    Генерирует разнородные параметры для популяции нейронов.
    
    Args:
        n: количество нейронов
        model: "lif_at", "izhikevich", "adex"
        seed: random seed
    
    Returns:
        dict с numpy массивами параметров
    """
    rng = np.random.default_rng(seed)
    
    if model == "lif_at":
        return {
            "tau_m": rng.uniform(5.0, 30.0, n).astype(np.float32),
            "v_rest": rng.uniform(-70.0, -60.0, n).astype(np.float32),
            "v_thresh": rng.uniform(-55.0, -45.0, n).astype(np.float32),
            "v_reset": rng.uniform(-75.0, -65.0, n).astype(np.float32),
            "tau_theta": rng.uniform(5.0, 50.0, n).astype(np.float32),
            "beta": rng.uniform(0.1, 1.0, n).astype(np.float32),
        }
    elif model == "izhikevich":
        # Excitatory (80%) vs Inhibitory (20%)
        types = rng.choice(["exc", "inh"], n, p=[0.8, 0.2])
        a = np.where(types == "exc", rng.uniform(0.02, 0.08, n),
                     rng.uniform(0.02, 0.1, n)).astype(np.float32)
        b = np.where(types == "exc", rng.uniform(0.15, 0.25, n),
                     rng.uniform(0.2, 0.3, n)).astype(np.float32)
        c = np.where(types == "exc", rng.uniform(-65, -50, n),
                     rng.uniform(-65, -55, n)).astype(np.float32)
        d = np.where(types == "exc", rng.uniform(2, 8, n),
                     rng.uniform(2, 6, n)).astype(np.float32)
        return {"a": a, "b": b, "c": c, "d": d, "types": types}
    
    return {}


# ═══════════════════════════════════════════════════════════
#  Demo
# ═══════════════════════════════════════════════════════════

def demo_all_metrics():
    """Запуск всех метрик на синтетических данных."""
    np.random.seed(42)
    print("=" * 55)
    print("⚡ Spike Analysis Toolbox — Full Demo")
    print("=" * 55)
    
    # Generate data
    n_neurons = 10
    n_time = 500
    dt = 1.0
    spikes = np.random.random((n_neurons, n_time)) < 0.05
    spikes = spikes.astype(np.int32)
    
    stimulus = np.sin(np.linspace(0, 10 * np.pi, n_time))
    
    print(f"\n📊 Data: {n_neurons} neurons × {n_time} timesteps")
    
    # ─── Poisson Generator demo ─────────────────────
    print(f"\n{'─'*40}")
    print("🎯 1. Poisson Spike Generator")
    poisson = neuron_engine.PyPoissonGenerator(n=10, dt=1.0, seed=42)
    poisson.set_rate(0, 50.0)  # 50 Hz for neuron 0
    poisson.set_rate(1, 100.0)  # 100 Hz for neuron 1
    rates = np.full(10, 20.0, dtype=np.float32)
    poisson.set_rates(rates)
    
    poisson_out = np.zeros(10, dtype=np.int32)
    total_spikes = 0
    for t in range(100):
        poisson.step(poisson_out)
        total_spikes += int(np.sum(poisson_out))
    rate_hz = total_spikes / (100 * 10 / 1000)  # Hz
    print(f"   Generated ~{total_spikes} spikes in 100 steps")
    print(f"   Mean rate: {rate_hz:.1f} Hz (target: 20 Hz)")
    
    # ─── Van Rossum distance ───────────────────────
    print(f"\n{'─'*40}")
    print("📏 2. Van Rossum Distance")
    vr = van_rossum_distance(spikes[0], spikes[1], tau=10.0, dt=dt)
    vr_same = van_rossum_distance(spikes[0], spikes[0], tau=10.0, dt=dt)
    print(f"   D(neuron 0, neuron 1): {vr:.4f}")
    print(f"   D(neuron 0, neuron 0): {vr_same:.4f} (should be 0)")
    
    # ─── Victor-Purpura distance ────────────────────
    print(f"\n{'─'*40}")
    print("📏 3. Victor-Purpura Distance")
    vp = victor_purpura_dist(spikes[0], spikes[1], cost_per_spike=1.0)
    vp_same = victor_purpura_dist(spikes[0], spikes[0], cost_per_spike=1.0)
    print(f"   D(neuron 0, neuron 1): {vp:.4f}")
    print(f"   D(neuron 0, neuron 0): {vp_same:.4f} (should be 0)")
    
    # ─── ISI/CV ──────────────────────────────────────
    print(f"\n{'─'*40}")
    print("📏 4. ISI Coefficient of Variation")
    cv, mean_isi = isi_cv(spikes[0])
    fr = firing_rate(spikes[0], dt=dt)
    print(f"   Neuron 0: CV={cv:.4f}, Mean ISI={mean_isi:.1f} steps, FR={fr:.1f} Hz")
    
    # ─── PSTH ────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("📊 5. PSTH")
    stim_times = np.array([100, 200, 300, 400])
    bins, psth = compute_psth(spikes, stim_times,
                               window_before=20, window_after=30, bin_size=5, dt=dt)
    print(f"   PSTH: {len(psth)} bins from {bins[0]:.0f} to {bins[-1]:.0f} ms")
    print(f"   Peak rate: {psth.max():.1f} Hz")
    
    # ─── STA ─────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("🎯 6. Spike-Triggered Average")
    sta = spike_triggered_average(stimulus, spikes[0], window_before=20)
    print(f"   STA shape: {sta.shape}")
    print(f"   STA range: [{sta.min():.4f}, {sta.max():.4f}]")
    
    # ─── LFP Proxy ─────────────────────────────────
    print(f"\n{'─'*40}")
    print("🔌 7. LFP Proxy")
    lfp_proxy = LFPProxy(n_neurons=10, n_electrodes=4, tau_smooth=20.0, dt=dt)
    for t in range(50):
        lfp = lfp_proxy.step(spikes[:, t])
    print(f"   LFP shape: {lfp.shape}")
    print(f"   LFP values: {lfp}")
    
    # ─── Mutual Information ─────────────────────────
    print(f"\n{'─'*40}")
    print("📊 8. Mutual Information")
    mi = mutual_information(spikes[0].astype(float), spikes[1].astype(float))
    mi_same = mutual_information(spikes[0].astype(float), spikes[0].astype(float))
    print(f"   MI(neuron 0, neuron 1): {mi:.4f} bits")
    print(f"   MI(neuron 0, neuron 0): {mi_same:.4f} bits")
    
    # ─── STP Demo (если доступен) ────────────────
    print(f"\n{'─'*40}")
    print("🔄 9. Short-Term Plasticity Demo")
    try:
        stp = neuron_engine.PyShortTermPlasticity(n=5, dt=1.0, heterogeneous=True, seed=42)
        pre = np.array([1, 0, 0, 1, 0], dtype=np.int32)
        I_out = np.zeros(5, dtype=np.float32)
        for step in range(10):
            I_out.fill(0.0)
            pre[:] = (step % 3 == 0)  # burst every 3 steps
            stp.step(pre, I_out)
            if step < 5:
                print(f"   Step {step}: pre={pre}, I_syn={I_out}")
        print("   ✅ STP works!")
    except Exception as e:
        print(f"   ⚠️ STP not available: {e}")
    
    print(f"\n{'='*55}")
    print("✅ All metrics computed successfully!")
    print("=" * 55)
    
    return {
        "van_rossum": vr,
        "victor_purpura": vp,
        "isi_cv": cv,
        "firing_rate": fr,
        "mi": mi,
    }


if __name__ == "__main__":
    demo_all_metrics()
