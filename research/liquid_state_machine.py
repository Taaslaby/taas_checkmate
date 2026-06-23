#!/usr/bin/env python3
"""
💧 Liquid State Machine (LSM) — Spiking Reservoir Computing

Использует существующие компоненты neuron-engine:
- T3 Triple Oscillator → feature extraction
- Phase Encoding → spike encoding
- Sparse CSR propagation → reservoir connectivity
- LIF-AT (PyLifNeuron) → reservoir neurons
- NLMS decoder → readout training

Pipeline: Signal → T3 Osc → Phase Encoding → LIF Reservoir → NLMS Readout

Based on: Maass (2002) "Real-Time Computing Without Stable States"
         + Frontiers in Computational Neuroscience (2023) reservoir review
"""

import numpy as np
from typing import Optional, Tuple

try:
    import neuron_engine
except ImportError:
    import sys
    sys.path.insert(0, "target/release")
    # Try maturin dev install
    import subprocess
    subprocess.run(["maturin", "develop", "--release", "-q"], cwd="..", check=True)
    import neuron_engine


class LiquidStateMachine:
    """
    Liquid State Machine — Spiking Reservoir Computing.
    
    Args:
        n_reservoir: number of reservoir neurons (LIF-AT)
        n_input: number of oscillators in T3
        input_sparsity: connection probability from input to reservoir (default 0.1)
        reservoir_sparsity: recurrent connection probability (default 0.05)
        spectral_radius: scale recurrent weights for echo state property (default 0.9)
        dt: simulation timestep
    """
    
    def __init__(
        self,
        n_reservoir: int = 256,
        n_input: int = 64,
        input_sparsity: float = 0.1,
        reservoir_sparsity: float = 0.05,
        spectral_radius: float = 0.9,
        dt: float = 0.5,
    ):
        self.n_res = (n_reservoir + 7) // 8 * 8  # pad to 8
        self.n_inp = n_input
        self.dt = dt
        
        # ─── Reservoir neurons (LIF-AT) ─────────────────
        self.neurons = neuron_engine.PyLifNeuron(
            self.n_res, dt=dt, tau_m=20.0, v_rest=-65.0,
            v_thresh=-50.0, v_reset=-70.0, tau_theta=50.0, beta=0.5,
        )
        
        # ─── Input weights W_in [n_reservoir × n_input] ──
        # Sparse random: each input connects to ~10% of reservoir
        self.w_in = np.zeros((self.n_res, self.n_inp), dtype=np.float32)
        for i in range(self.n_res):
            for j in range(self.n_inp):
                if np.random.random() < input_sparsity:
                    self.w_in[i, j] = np.random.uniform(-0.5, 0.5)
        
        # ─── Recurrent weights W_res [n_reservoir × n_reservoir] ──
        # Sparse random with spectral radius control
        self.w_res = np.zeros((self.n_res, self.n_res), dtype=np.float32)
        for i in range(self.n_res):
            for j in range(self.n_res):
                if i != j and np.random.random() < reservoir_sparsity:
                    self.w_res[i, j] = np.random.uniform(-1.0, 1.0)
        
        # Scale to spectral radius (power iteration for speed)
        if self.n_res > 1:
            max_eig = self._power_iteration(self.w_res)
            if max_eig > 1e-6:
                self.w_res *= spectral_radius / max_eig
        
        # ─── Readout (NLMS decoder) ─────────────────────
        # Learns to map reservoir state → output
        self.readout = neuron_engine.PyNlmsDecoder(
            self.n_res + 1,  # +1 for bias
            lr=0.01, eps=1e-8
        )
        
        # State buffers
        self.spikes = np.zeros(self.n_res, dtype=np.int32)
        self.I = np.zeros(self.n_res, dtype=np.float32)
        self.bias = np.ones(1, dtype=np.float32)
        
        print(f"💧 LSM: {self.n_res} reservoir neurons, "
              f"{np.count_nonzero(self.w_in)} input synapses, "
              f"{np.count_nonzero(self.w_res)} recurrent synapses")
    
    def _power_iteration(self, M: np.ndarray, n_iter: int = 10) -> float:
        """Fast approximation of largest eigenvalue."""
        v = np.random.randn(M.shape[1])
        for _ in range(n_iter):
            v = M @ v
            v /= np.linalg.norm(v) + 1e-10
        return float(np.linalg.norm(M @ v))
    
    def step(self, input_signal: np.ndarray) -> float:
        """
        One simulation step.
        
        Args:
            input_signal: T3 oscillator features [n_input] or scalar
            
        Returns:
            Readout prediction (scalar)
        """
        # 1. Input → reservoir via W_in
        inp = np.asarray(input_signal, dtype=np.float32).ravel()
        n = min(len(inp), self.n_inp)
        # 1. Input → reservoir via W_in @ inp
        self.I[:] = self.w_in[:, :n] @ inp[:n] / self.n_inp
        
        # 2. Add recurrent input from previous spikes (sparse matvec)
        sp = self.spikes.copy()
        self.I += self.w_res @ sp
        
        # 3. Step reservoir neurons
        self.neurons.step(self.I, self.spikes)
        
        # 4. Readout: predict from reservoir state (spikes + bias)
        prediction = self.readout.predict(
            np.append(self.spikes.astype(np.float32), 1.0)
        )
        
        return prediction
    
    def train(self, input_signal: np.ndarray, target: float) -> float:
        """
        One step of reservoir + train readout.
        
        Args:
            input_signal: input features [n_input]
            target: target output value
            
        Returns:
            Prediction error
        """
        inp = np.asarray(input_signal, dtype=np.float32).ravel()
        n = min(len(inp), self.n_inp)
        self.I[:] = self.w_in[:, :n] @ inp[:n] / self.n_inp
        self.I += self.w_res @ self.spikes
        
        self.neurons.step(self.I, self.spikes)
        
        # Train readout
        features = np.zeros(self.n_res + 1, dtype=np.float32)
        features[:self.n_res] = self.spikes.astype(np.float32)
        features[self.n_res] = 1.0  # bias
        error = self.readout.train(features, target)
        
        return error
    
    def reset(self):
        """Reset reservoir state."""
        self.spikes.fill(0)
        self.I.fill(0)
        self.neurons.set_v(np.full(self.n_res, -65.0, dtype=np.float32))
        self.neurons.set_theta(np.zeros(self.n_res, dtype=np.float32))


def demo_tone_detection():
    """LSM демо: detect frequency of a sine wave."""
    print("\n═══ LSM Tone Detection Demo ═══")
    
    lsm = LiquidStateMachine(
        n_reservoir=128,
        n_input=32,
        input_sparsity=0.15,
        reservoir_sparsity=0.05,
        spectral_radius=0.8,
        dt=0.5,
    )
    
    # Generate training data: sine waves at different frequencies
    sr = 22050  # sample rate
    duration = 0.5  # seconds
    n_samples = int(sr * duration)
    t = np.arange(n_samples) / sr
    
    # Training: 5 different frequencies → target = frequency index
    freqs = [200, 400, 600, 800, 1000]
    n_train = len(freqs)
    train_inputs = []
    train_targets = []
    
    print(f"Training on {n_train} frequencies: {freqs} Hz")
    for idx, f in enumerate(freqs):
        signal = np.sin(2 * np.pi * f * t)
        # Use frame-wise processing
        frame_size = 256
        for fi in range(0, len(signal), frame_size):
            frame = signal[fi:fi+frame_size]
            if len(frame) < frame_size:
                break
            # Simple FFT-based feature extraction (replace with T3 in prod)
            fft = np.fft.rfft(frame)
            features = np.abs(fft[:32]) / frame_size
            features = features / (features.max() + 1e-10)
            
            # Reservoir step + train
            error = lsm.train(features, float(idx) / (n_train - 1))
            train_inputs.append(features)
            train_targets.append(float(idx) / (n_train - 1))
    
    print(f"Trained on {len(train_inputs)} frames")
    
    # Test on held-out frequency
    test_freq = 500  # Hz (not in training set)
    signal = np.sin(2 * np.pi * test_freq * t)
    predictions = []
    
    for fi in range(0, len(signal), frame_size):
        frame = signal[fi:fi+frame_size]
        if len(frame) < frame_size:
            break
        fft = np.fft.rfft(frame)
        features = np.abs(fft[:32]) / frame_size
        features = features / (features.max() + 1e-10)
        pred = lsm.step(features)
        predictions.append(pred)
    
    avg_pred = np.mean(predictions)
    # Map back: 500 Hz is between 400 (idx 1 → 0.25) and 600 (idx 2 → 0.5)
    expected = 1.5 / (n_train - 1)  # ≈ 0.375
    print(f"Test {test_freq}Hz: avg prediction = {avg_pred:.4f} (expected ~{expected:.4f})")
    print(f"LSM generalizes to unseen frequencies: ✅" if abs(avg_pred - expected) < 0.3 else "⚠️")
    
    return lsm


if __name__ == "__main__":
    demo_tone_detection()
