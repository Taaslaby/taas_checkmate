#!/usr/bin/env python3
"""
🧠 E-Prop SNN — Online Learning with Eligibility Traces

Реализует three-factor learning rule из:
Bellec et al. (2020) "A solution to the learning dilemma for 
recurrent networks of spiking neurons" — Nature Communications

Алгоритм:
  1. Forward pass: LIF dynamics + eligibility trace update
  2. Compute learning signal L_j = error * feedback_weights
  3. Weight update: Δw = η · L_j · e_ji

Использует neuron_engine.PyEPropLIF для эффективных вычислений.
"""

import numpy as np
from typing import Optional, Callable

try:
    import neuron_engine
except ImportError:
    import sys, subprocess
    sys.path.insert(0, "target/release")
    subprocess.run(["maturin", "develop", "--release", "-q"], cwd="..", check=True)
    import neuron_engine


class EPropSNN:
    """
    E-Prop Spiking Neural Network — online supervised learning.
    
    Architecture: Input → EPropLIF layer → Output readout
    
    Uses surrogate gradients and eligibility traces for online learning
    without Backpropagation-Through-Time (BPTT).
    
    Args:
        n_input: number of input features
        n_hidden: number of spiking neurons in hidden layer
        dt: simulation timestep
        tau_m: membrane time constant
        tau_e: eligibility trace time constant
        lr: learning rate
    """
    
    def __init__(
        self,
        n_input: int = 32,
        n_hidden: int = 64,
        dt: float = 1.0,
        tau_m: float = 10.0,
        tau_e: float = 20.0,
        lr: float = 0.01,
    ):
        self.n_in = n_input
        self.n_hid = (n_hidden + 7) // 8 * 8  # pad to 8
        self.dt = dt
        self.lr = lr
        
        # ─── Hidden layer (E-Prop LIF neurons) ─────────
        self.layer = neuron_engine.PyEPropLIF(
            self.n_hid, dt=dt, tau_m=tau_m, tau_e=tau_e,
        )
        
        # ─── Input weights W_in [n_hidden × n_input] ───
        self.w_in = np.random.uniform(
            -0.3, 0.3, (self.n_hid, self.n_in)
        ).astype(np.float32)
        
        # ─── Readout weights W_out [1 × n_hidden] ──────
        self.w_out = np.zeros(self.n_hid, dtype=np.float32)
        
        # ─── Feedback weights (random fixed) ───────────
        # Used to compute learning signal from output error
        self.w_fb = np.random.uniform(
            -0.1, 0.1, self.n_hid
        ).astype(np.float32)
        
        # State
        self.spikes = np.zeros(self.n_hid, dtype=np.int32)
        self.I = np.zeros(self.n_hid, dtype=np.float32)
        self.total_spikes = 0
        self.step_count = 0
        
        print(f"🧠 EPropSNN: {self.n_in}→{self.n_hid} hidden→1 output, "
              f"lr={lr}, τ_m={tau_m}, τ_e={tau_e}")
    
    def forward(self, x: np.ndarray) -> float:
        """
        Forward pass: input → LIF → readout.
        
        Args:
            x: input features [n_input]
            
        Returns:
            Readout prediction
        """
        # 1. Input → hidden via W_in @ inp (vectorized)
        inp = np.asarray(x, dtype=np.float32).ravel()
        self.I[:] = self.w_in[:, :min(len(inp), self.n_in)] @ inp[:min(len(inp), self.n_in)]
        
        # 2. Step E-Prop layer (forward + eligibility trace)
        self.layer.step(self.I, self.spikes, np.zeros(self.n_hid, dtype=np.int32))
        
        # 3. Readout: dot product of spikes & weights
        pred = float(self.spikes[:self.n_hid] @ self.w_out)
        self.total_spikes += int(np.sum(self.spikes))
        self.step_count += 1
        
        return pred
    
    def train(self, x: np.ndarray, target: float) -> float:
        """
        One step of E-Prop training.
        
        Args:
            x: input features [n_input]
            target: target output value
            
        Returns:
            Prediction error (target - prediction)
        """
        # 1. Forward pass
        pred = self.forward(x)
        error = target - pred
        
        # 2. Compute learning signal via feedback weights
        # L_j = error * w_fb[j] (Broadcast error through random feedback)
        learning_signals = error * self.w_fb
        
        # 3. Update readout weights (direct gradient)
        # dL/dw_out = -error * spikes
        self.w_out += self.lr * error * self.spikes[:self.n_hid].astype(np.float32)
        self.w_out = np.clip(self.w_out, -5.0, 5.0)
        
        # 4. E-Prop weight update for hidden layer (vectorized)
        # Δw_in = η · L_j · e_ji, outer product: [n_hid] × [n_in]
        elig = np.zeros(self.n_hid, dtype=np.float32)
        self.layer.copy_eligibility(elig)
        
        n_in = min(len(x), self.n_in)
        # Outer product: learning_signal[i] × elig[i] × inp[j] → Δw_in[i,j]
        self.w_in[:, :n_in] += self.lr * np.outer(learning_signals * elig, inp[:n_in])
        
        return error
    
    def reset(self):
        """Reset network state."""
        self.layer.reset()
        self.spikes.fill(0)
        self.I.fill(0)
        self.total_spikes = 0
        self.step_count = 0
    
    def get_firing_rate(self) -> float:
        """Average firing rate over all steps."""
        if self.step_count == 0:
            return 0.0
        return self.total_spikes / (self.step_count * self.n_hid)


def demo_xor_learning():
    """E-Prop демо: learn XOR function."""
    print("\n═══ E-Prop XOR Learning Demo ═══")
    
    net = EPropSNN(n_input=2, n_hidden=32, lr=0.02)
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.float32)
    
    n_epochs = 50
    errors = []
    
    for epoch in range(n_epochs):
        epoch_errors = []
        for i in range(4):
            # Multiple timesteps per input (rate coding)
            for t in range(5):
                err = net.train(X[i], y[i])
                epoch_errors.append(abs(err))
        
        avg_err = np.mean(epoch_errors)
        errors.append(avg_err)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: avg error = {avg_err:.6f}, "
                  f"firing rate = {net.get_firing_rate():.3f}")
    
    # Test
    predictions = []
    for i in range(4):
        preds = []
        for t in range(5):
            p = net.forward(X[i])
            preds.append(p)
        predictions.append(np.mean(preds))
        print(f"  {X[i]} → {predictions[-1]:.4f} (target={y[i]})")
    
    # Check if XOR learned correctly
    correct = sum(
        1 for i in range(4)
        if abs(predictions[i] - y[i]) < 0.5
    )
    print(f"XOR accuracy: {correct}/4")
    if correct >= 3:
        print("✅ E-Prop successfully learned XOR!")
    else:
        print("⚠️ E-Prop needs more training or tuning")
    
    return net


def demo_temporal_pattern():
    """E-Prop демо: learn a temporal sequence."""
    print("\n═══ E-Prop Temporal Pattern Demo ═══")
    
    net = EPropSNN(n_input=8, n_hidden=64, lr=0.01, tau_e=30.0)
    
    # Generate a simple temporal pattern: sine wave
    seq_len = 50
    t = np.arange(seq_len) / 10
    pattern = np.sin(2 * np.pi * t)  # target
    
    # Input: context window + noise
    context = 8
    errors = []
    
    for epoch in range(30):
        net.reset()
        epoch_errors = []
        for i in range(context, seq_len):
            # Input: previous context window
            inp = pattern[i-context:i]
            # Target: next value
            target = pattern[i] if i < seq_len - 1 else 0.0
            err = net.train(inp, target)
            epoch_errors.append(abs(err))
        
        avg_err = np.mean(epoch_errors)
        errors.append(avg_err)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: prediction error = {avg_err:.6f}")
    
    # Test prediction
    net.reset()
    predictions = []
    for i in range(context, seq_len):
        inp = pattern[i-context:i]
        # Only forward, no training
        pred = net.forward(inp)
        predictions.append(pred)
    
    # Quick quality check
    if len(predictions) > 5:
        pred_arr = np.array(predictions)
        target_arr = pattern[context:seq_len]
        corr = np.corrcoef(pred_arr, target_arr)[0, 1]
        print(f"  Correlation with target: {corr:.4f}")
        if corr > 0.7:
            print("✅ E-Prop learned temporal pattern!")
        else:
            print("⚠️ Temporal learning needs tuning")
    
    return net


if __name__ == "__main__":
    demo_xor_learning()
    demo_temporal_pattern()
