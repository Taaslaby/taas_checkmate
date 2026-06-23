"""
T3SpikeTrainer — training loop for SNN with T3-SPIKE surrogate gradients.

Architecture:
    spikes (binary) → W_snn @ spikes → drives
        → T3 oscillator (N steps) → PST features
        → NLMS decoder → prediction → loss
                                   ↓
    W_snn ← SGD(grad) ← dL/d(drives) ← BPTT through T3

Usage:
    trainer = T3SpikeTrainer(n_osc=16, n_steps=10, dt=0.05)
    for epoch in range(100):
        loss = trainer.train_step(spikes, target)
"""

import numpy as np
import torch
from typing import Optional, Tuple

from .encoder import T3SpikeEncoder, pst_linear_features_np, _pst_linear_dim


class T3SpikeTrainer:
    """
    Training loop for SNN → T3 → PST → decoder.

    Gradient path:
        loss → d(L)/d(PST) → BPTT(T3) → d(L)/d(drives) → d(L)/d(W_snn) → SGD
    """

    def __init__(
        self,
        n_osc: int = 16,
        n_steps: int = 10,
        dt: float = 0.05,
        n_spike_neurons: Optional[int] = None,
        lr_snn: float = 0.01,
        lr_decoder: float = 0.05,
        decoder_eps: float = 1e-8,
        freq_min: float = 0.5,
        freq_max: float = 3.0,
        drive_gain: float = 10.0,
        alpha: float = 1.0,
        coupling: tuple = (0.3, 0.15, 0.08),
        device: str = "cpu",
        seed: int = 42,
    ):
        self.n_osc = n_osc
        self.n_steps = n_steps
        self.dt = dt
        self.n_spike_neurons = n_spike_neurons or n_osc
        self.pst_dim = _pst_linear_dim(n_osc)
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.drive_gain = drive_gain
        self.alpha = alpha
        self.coupling = coupling
        self.device = torch.device(device)

        # SNN weight matrix: n_spike_neurons → 3*n_osc drives
        rng = np.random.default_rng(seed)
        W_snn = rng.normal(0, 1.0, (3 * n_osc, self.n_spike_neurons)).astype(np.float32)
        self.W_snn = torch.nn.Parameter(
            torch.tensor(W_snn / np.sqrt(self.n_spike_neurons), device=self.device)
        )

        # Adam optimizer for SNN weights
        self.optimizer = torch.optim.Adam([self.W_snn], lr=lr_snn)

        # NLMS decoder (online learning)
        self.decoder_lr = lr_decoder
        self.decoder_eps = decoder_eps
        self.decoder_weights = np.zeros(self.pst_dim, dtype=np.float64)
        self.decoder_bias = 0.0

        # T3 oscillator (reused across batch items)
        import taas_checkmate
        self._tri = None  # lazy init

    def _ensure_tri(self):
        if self._tri is None:
            import taas_checkmate as tc
            self._tri = tc.PyTriOscillator(self.n_osc, self.freq_min, self.freq_max)
            self._tri.set_coupling(*self.coupling)

    def _reset_tri(self):
        """Reset T3 to initial state by recreating it."""
        import taas_checkmate as tc
        self._tri = tc.PyTriOscillator(self.n_osc, self.freq_min, self.freq_max)
        self._tri.set_coupling(*self.coupling)

    def decode(self, pst: np.ndarray) -> float:
        """NLMS decoder prediction from PST features."""
        return float(np.dot(self.decoder_weights, pst) + self.decoder_bias)

    def train_decoder(self, pst: np.ndarray, target: float):
        """Online NLMS decoder update."""
        pred = self.decode(pst)
        error = float(target) - pred
        # NLMS: mu = lr / (||x||² + eps)
        norm_sq = float(np.dot(pst, pst))
        mu = self.decoder_lr / (norm_sq + self.decoder_eps)
        self.decoder_weights += mu * error * pst
        self.decoder_bias += mu * error
        return error

    def forward(self, spikes_np: np.ndarray) -> np.ndarray:
        """
        Forward pass without gradient tracking.

        Args:
            spikes_np: (n_spike_neurons,) binary spike pattern

        Returns:
            pst: (pst_dim,) PST features
        """
        self._ensure_tri()
        self._reset_tri()

        # drives = W_snn @ spikes (numpy)
        W = self.W_snn.detach().cpu().numpy()
        drives = W @ spikes_np  # (3*n_osc,)

        # Run T3 for n_steps
        self._tri.set_drive(drives.astype(np.float32))
        for _ in range(self.n_steps):
            pst = self._tri.step(self.dt)  # returns ndarray

        return np.array(pst, dtype=np.float64)

    def train_step(
        self,
        spikes: np.ndarray,
        target: float,
        verbose: bool = False,
    ) -> Tuple[float, float, float]:
        """
        One training step with surrogate gradient through T3.

        Args:
            spikes: (n_spike_neurons,) binary spike pattern
            target: target value for decoder

        Returns:
            (loss, decoder_error, prediction)
        """
        # ── Forward through SNN (with gradient tracking) ──
        spikes_t = torch.tensor(spikes, dtype=torch.float32, device=self.device)
        drives_t = self.W_snn @ spikes_t  # (3*n_osc,)

        # ── Forward through T3 (autograd Function) ──
        # Reshape drives to (1, n_steps, 3*n_osc) by repeating
        drives_seq = drives_t.unsqueeze(0).expand(self.n_steps, -1).unsqueeze(0)

        pst_t = T3SpikeEncoder.apply(
            drives_seq,
            None,  # tri (not used, we have numpy fallback)
            self.n_osc,
            self.n_steps,
            self.dt,
            self.drive_gain,
            self.alpha,
            self.freq_min,
            self.freq_max,
            self.coupling,
            False,
        )  # (1, pst_dim)

        pst_np = pst_t.detach().cpu().numpy().flatten()

        # ── NLMS decoder (forward + train, no gradient tracking) ──
        pred = self.decode(pst_np)
        decoder_err = self.train_decoder(pst_np, float(target))

        # ── Compute loss gradient w.r.t. PST ──
        # MSE loss = (pred - target)^2
        # d(L)/d(pst_i) = 2 * (pred - target) * d(pred)/d(pst_i) = 2 * err * w_i
        err = pred - float(target)
        grad_pst_np = 2.0 * err * self.decoder_weights  # (pst_dim,)

        # ── Backward through T3 ──
        # We need to call backward on the T3 output with the analytical gradient
        grad_pst_t = torch.tensor(grad_pst_np, dtype=torch.float32, device=self.device)
        pst_t.backward(grad_pst_t)

        # ── W_snow gradient (from drives gradient) ──
        # d(L)/d(W_snn) = d(L)/d(drives) · d(drives)/d(W_snn)
        # d(drives)/d(W_snn) = spikes^T (linear)
        # The gradient of drives w.r.t. W_snn is handled by PyTorch autograd

        # ── SGD step ──
        self.optimizer.step()
        self.optimizer.zero_grad()

        # ── MSE loss ──
        loss = err ** 2

        if verbose:
            print(f"  loss={loss:.4f}, pred={pred:.4f}, target={target:.2f}, "
                  f"decoder_err={decoder_err:.4f}, |∇W|={self.W_snn.grad.norm().item():.4f}")

        return loss, decoder_err, pred

    def train_batch(
        self,
        patterns: list,
        labels: list,
        epochs: int = 5,
        verbose: bool = True,
    ) -> list:
        """
        Train on a batch of spike patterns.

        Args:
            patterns: list of (n_neurons,) numpy arrays
            labels: list of target values
            epochs: number of passes through the data
            verbose: print progress

        Returns:
            loss_history: list of losses per step
        """
        loss_history = []
        n_samples = len(patterns)
        for epoch in range(epochs):
            epoch_loss = 0.0
            # Shuffle
            idx = np.random.permutation(n_samples)
            for i in idx:
                spikes = patterns[i]
                target = float(labels[i])
                loss, err, pred = self.train_step(spikes, target)
                epoch_loss += loss

            avg_loss = epoch_loss / n_samples
            loss_history.append(avg_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: avg_loss = {avg_loss:.6f}")

        return loss_history

    def predict(self, spikes: np.ndarray) -> float:
        """Predict from spike pattern (no training)."""
        pst = self.forward(spikes)
        return self.decode(pst)

    def accuracy(self, patterns: list, labels: list) -> float:
        """Compute classification accuracy."""
        correct = 0
        for spikes, label in zip(patterns, labels):
            pred = self.predict(spikes)
            if abs(pred - float(label)) < 0.5:
                correct += 1
        return correct / len(patterns)
