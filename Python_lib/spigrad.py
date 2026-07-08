"""SPIGRAD: T3-SPIKE GRADient optimizer.

Full-trajectory BPTT through the T3 oscillator engine using
the new per_step_grad_pst API for single-pass O(n_steps) optimization.

Usage:
    import numpy as np
    from taas_checkmate import t2spigrad_forward, t2spigrad_backward
    from spigrad import SpigradOptimizer

    opt = SpigradOptimizer(n_osc=16, ...)
    for iteration in range(100):
        pst, traj = opt.forward(drives)
        per_step_grad = opt.compute_per_step_grad(traj, targets)
        grad_drives = opt.backward(traj, per_step_grad)
        drives -= lr * grad_drives
"""

import numpy as np
from typing import Optional, Callable

try:
    from taas_checkmate import (
        t2spigrad_forward,
        t2spigrad_backward,
        t2spigrad_pst_matrix,
        pst_linear_dim,
    )
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("WARNING: taas_checkmate not installed, using pure Python fallback")
    pst_linear_dim = lambda n: 3 * n * 6 + 12


def compute_pst_features(state: np.ndarray) -> np.ndarray:
    """Compute PST features from oscillator state.
    
    Args:
        state: (6*n,) array [th1, th2, th3, r1, r2, r3]
    
    Returns:
        (pst_dim,) array of PST features
    """
    n = len(state) // 6
    th1, th2, th3 = state[0*n:1*n], state[1*n:2*n], state[2*n:3*n]
    r1, r2, r3 = state[3*n:4*n], state[4*n:5*n], state[5*n:6*n]
    
    s1, c1 = np.sin(th1), np.cos(th1)
    s2, c2 = np.sin(th2), np.cos(th2)
    s3, c3 = np.sin(th3), np.cos(th3)
    
    features = []
    # 1. sin(θ), cos(θ)
    for s, c in [(s1, c1), (s2, c2), (s3, c3)]:
        for i in range(n):
            features.append(s[i])
            features.append(c[i])
    
    # 2. dθ/dt (placeholder — not needed for per-step loss)
    for _ in range(3):
        features.extend([0.0] * n)
    
    # 3. Amplitudes r
    features.extend(r1.tolist())
    features.extend(r2.tolist())
    features.extend(r3.tolist())
    
    # 4. Squared features
    for s, c in [(s1, c1), (s2, c2), (s3, c3)]:
        for i in range(n):
            features.append(s[i] ** 2)
            features.append(c[i] ** 2)
    
    # 5. Cross order parameters
    def cross_order(sa, ca, sb, cb):
        s = np.sum(sa * cb - ca * sb)
        c = np.sum(ca * cb + sa * sb)
        return float(np.sqrt(s*s + c*c) / n)
    
    features.append(cross_order(s1, c1, s2, c2))
    features.append(cross_order(s2, c2, s3, c3))
    features.append(cross_order(s1, c1, s3, c3))
    
    # 6. Mean sin/cos + mean amplitude
    for s, c, r in [(s1, c1, r1), (s2, c2, r2), (s3, c3, r3)]:
        features.append(float(np.mean(s)))
        features.append(float(np.mean(c)))
        features.append(float(np.mean(r)))
    
    return np.array(features, dtype=np.float32)


class SpigradOptimizer:
    """SPIGRAD optimizer: full-trajectory BPTT through T3 oscillator engine.
    
    Alternating optimization:
      1. Forward pass through T3 (drives → PST features + trajectory)
      2. Compute per-step loss gradients w.r.t. PST features
      3. Single-pass backward through BPTT (gradient of drives)
      4. Update drives with gradient descent
    """
    
    def __init__(
        self,
        n_osc: int = 16,
        n_steps: int = 250,
        dt: float = 0.05,
        drive_gain: float = 10.0,
        alpha: float = 1.0,
        freq_min: float = 0.5,
        freq_max: float = 3.0,
        coupling_w12: float = 0.3,
        coupling_w13: float = 0.15,
        coupling_w23: float = 0.08,
        lr: float = 0.1,
        use_cuda: bool = False,
    ):
        self.n_osc = n_osc
        self.n_steps = n_steps
        self.pst_dim = pst_linear_dim(n_osc)
        self.lr = lr
        self.use_cuda = use_cuda
        self._forward_kwargs = dict(
            n_osc=n_osc,
            n_steps=n_steps,
            dt=dt,
            drive_gain=drive_gain,
            alpha=alpha,
            freq_min=freq_min,
            freq_max=freq_max,
            coupling_w12=coupling_w12,
            coupling_w13=coupling_w13,
            coupling_w23=coupling_w23,
        )
    
    def forward(self, drives: np.ndarray) -> tuple:
        """Forward pass through T3.
        
        Args:
            drives: (n_steps, 3*n_osc) array of drive signals
        
        Returns:
            pst: (pst_dim,) PST features at final step
            traj: T2SpigradTrajectory object for backward pass
        """
        drives_2d = np.asarray(drives, dtype=np.float32).reshape(self.n_steps, 3 * self.n_osc)
        if self.use_cuda and HAS_RUST:
            from taas_checkmate import t2spigrad_forward_cuda
            pst, traj = t2spigrad_forward_cuda(
                **self._forward_kwargs, drives=drives_2d,
            )
        else:
            pst, traj = t2spigrad_forward(
                **self._forward_kwargs, drives=drives_2d,
            )
        return pst, traj
    
    def backward(
        self,
        traj,
        grad_pst: np.ndarray,
        per_step_grad_pst: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Backward pass: BPTT through T3.
        
        Args:
            traj: T2SpigradTrajectory from forward pass
            grad_pst: (pst_dim,) gradient for final step's PST
            per_step_grad_pst: (n_steps, pst_dim) or None
                Per-step gradients for full-trajectory optimization (SPIGRAD).
                When provided, single-pass O(n_steps) BPTT.
        
        Returns:
            grad_drives: (n_steps, 3*n_osc) gradient w.r.t. drives
        """
        grad_pst = np.asarray(grad_pst, dtype=np.float32).ravel()
        if per_step_grad_pst is not None:
            per_step_arr = np.asarray(per_step_grad_pst, dtype=np.float32)
        else:
            per_step_arr = None
        
        if self.use_cuda and HAS_RUST:
            from taas_checkmate import t2spigrad_backward_cuda
            return t2spigrad_backward_cuda(
                grad_pst, traj,
                per_step_grad_pst=per_step_arr,
            )
        else:
            return t2spigrad_backward(
                grad_pst, traj,
                per_step_grad_pst=per_step_arr,
            )
    
    def _extract_step_state(self, traj, t: int) -> np.ndarray:
        """Extract oscillator state after step t from trajectory.
        
        Returns (6*n,) array: [th1, th2, th3, r1, r2, r3].
        """
        n = self.n_osc
        if t == self.n_steps - 1:
            th1 = np.array(traj.th1_final(), dtype=np.float32)
            r1 = np.array(traj.r1_final(), dtype=np.float32)
            th2 = np.array(traj.th2_final(), dtype=np.float32)
            r2 = np.array(traj.r2_final(), dtype=np.float32)
            th3 = np.array(traj.th3_final(), dtype=np.float32)
            r3 = np.array(traj.r3_final(), dtype=np.float32)
        else:
            off = (t + 1) * n
            th1 = np.array(traj.th1_pre(), dtype=np.float32)[off:off+n]
            r1 = np.array(traj.r1_pre(), dtype=np.float32)[off:off+n]
            th2 = np.array(traj.th2_pre(), dtype=np.float32)[off:off+n]
            r2 = np.array(traj.r2_pre(), dtype=np.float32)[off:off+n]
            th3 = np.array(traj.th3_pre(), dtype=np.float32)[off:off+n]
            r3 = np.array(traj.r3_pre(), dtype=np.float32)[off:off+n]
        return np.concatenate([th1, th2, th3, r1, r2, r3])
    
    def _compute_pst_matrix(self, traj) -> np.ndarray:
        """Compute PST features for all steps in trajectory.
        
        Uses Rust native t2spigrad_pst_matrix for speed (no Python loop over steps).
        Falls back to pure Python if Rust not available.
        
        Returns (n_steps, pst_dim) matrix.
        """
        if HAS_RUST:
            return np.asarray(t2spigrad_pst_matrix(traj), dtype=np.float32)
        pst_matrix = np.zeros((self.n_steps, self.pst_dim), dtype=np.float32)
        for t in range(self.n_steps):
            state = self._extract_step_state(traj, t)
            pst_matrix[t] = compute_pst_features(state)
        return pst_matrix
    
    def compute_per_step_grad(
        self,
        traj,
        decoder_weights: np.ndarray,
        decoder_bias: Optional[np.ndarray],
        targets: np.ndarray,
    ) -> np.ndarray:
        """Compute per-step gradient of MSE loss w.r.t. PST features.
        
        Uses the trajectory's stored states to compute PST at each step,
        then uses decoder weights to get dLoss/dPST_t.
        
        Args:
            traj: T2SpigradTrajectory from forward pass
            decoder_weights: (pst_dim,) or (pst_dim, 1) Ridge decoder weights
            decoder_bias: scalar bias or None
            targets: (n_steps,) target values for each step
        
        Returns:
            per_step_grad: (n_steps, pst_dim) gradient w.r.t. PST at each step
        """
        pst_dim = self.pst_dim
        n_steps = traj.n_steps()
        
        w = np.asarray(decoder_weights, dtype=np.float32).ravel()
        b = float(decoder_bias) if decoder_bias is not None else 0.0
        targets = np.asarray(targets, dtype=np.float32).ravel()
        assert len(targets) == n_steps, f"{len(targets)} != {n_steps}"
        
        per_step_grad = np.zeros((n_steps, pst_dim), dtype=np.float32)
        
        for t in range(n_steps):
            state = self._extract_step_state(traj, t)
            pst_t = compute_pst_features(state)
            pred_t = float(pst_t @ w) + b
            diff = pred_t - targets[t]
            
            # dLoss/dPST = 2 * (pred - target) * w     (MSE loss)
            per_step_grad[t] = 2.0 * diff * w
        
        return per_step_grad
    
    def optimize_step(
        self,
        drives: np.ndarray,
        traj,
        decoder_weights: np.ndarray,
        decoder_bias: Optional[np.ndarray],
        targets: np.ndarray,
        final_grad_pst: Optional[np.ndarray] = None,
    ) -> tuple:
        """One SPIGRAD optimization step: compute gradients and update drives.
        
        Args:
            drives: (n_steps, 3*n_osc) current drives
            traj: T2SpigradTrajectory from forward pass (reuse if already computed)
            decoder_weights: (pst_dim,) Ridge decoder weights
            decoder_bias: scalar bias or None
            targets: (n_steps,) target values
            final_grad_pst: (pst_dim,) optional gradient for final step only.
                If None, uses per_step_grad for all steps.
        
        Returns:
            grad_drives: (n_steps, 3*n_osc) gradient w.r.t. drives
            drives_new: (n_steps, 3*n_osc) updated drives
        """
        per_step_grad = self.compute_per_step_grad(
            traj, decoder_weights, decoder_bias, targets,
        )
        
        if final_grad_pst is None:
            final_grad_pst = per_step_grad[-1]
        
        grad_drives = self.backward(traj, final_grad_pst, per_step_grad)
        drives_new = drives - self.lr * grad_drives
        
        return grad_drives, drives_new
    
    def alternating_optimization(
        self,
        initial_drives: np.ndarray,
        targets: np.ndarray,
        n_iterations: int = 50,
        ridge_alpha: float = 1.0,
        verbose: bool = True,
    ) -> tuple:
        """Full SPIGRAD alternating optimization loop.
        
        Alternates between:
          (a) Retraining Ridge decoder on current PST features
          (b) BPTT gradient step on drives
        
        Args:
            initial_drives: (n_steps, 3*n_osc) initial drives
            targets: (n_steps,) target values
            n_iterations: number of SPIGRAD iterations
            ridge_alpha: Ridge regularization strength
            verbose: print progress
        
        Returns:
            drives: (n_steps, 3*n_osc) optimized drives
            history: list of loss values per iteration
        """
        drives = initial_drives.copy()
        history = []
        
        for it in range(n_iterations):
            # Forward pass
            pst_final, traj = self.forward(drives)
            
            # Retrain Ridge decoder on per-step PST features
            pst_matrix = self._compute_pst_matrix(traj)
            
            # Ridge regression
            X = pst_matrix
            y = targets
            I = np.eye(self.pst_dim)
            w = np.linalg.solve(X.T @ X + ridge_alpha * I, X.T @ y)
            b = 0.0
            
            # Predictions and loss
            preds = X @ w + b
            loss = float(np.mean((preds - y) ** 2))
            history.append(loss)
            
            if verbose:
                print(f"  Iter {it:3d}: MSE={loss:.4f}, pred[-1]={preds[-1]:.3f} target={targets[-1]:.3f}")
            
            # Check convergence
            if it > 2 and abs(history[-2] - history[-1]) < 1e-6:
                if verbose:
                    print(f"  Converged at iteration {it}")
                break
            
            # SPIGRAD per-step gradient
            per_step_grad = np.zeros((self.n_steps, self.pst_dim), dtype=np.float32)
            for t in range(self.n_steps):
                diff = preds[t] - targets[t]
                per_step_grad[t] = 2.0 * diff * w
            
            grad_drives = self.backward(traj, per_step_grad[-1], per_step_grad)
            
            # Gradient descent with backtracking line search
            lr = self.lr
            for _ in range(8):
                drives_new = drives - lr * grad_drives
                _, traj_new = self.forward(drives_new)
                pst_new = self._compute_pst_matrix(traj_new)
                loss_new = float(np.mean((pst_new @ w + b - targets) ** 2))
                if loss_new < loss:
                    drives = drives_new
                    loss = loss_new
                    break
                lr *= 0.5
        
        return drives, history
