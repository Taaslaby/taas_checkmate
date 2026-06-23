"""
T3SpikeEncoder — differentiable T3 oscillator dynamics for SNN surrogate gradients.

Forward pass:
    drives (batch, n_steps, 3*n_osc) → T3 oscillator (N steps) → PST features

Backward pass (analytical):
    d(loss)/d(PST) → BPTT through T3 dynamics → d(loss)/d(drives)

Architecture:
    ∇loss → ∇PST → ∇state_N → ∇state_{t+1} → ... → ∇state_1 → ∇drives
"""

import numpy as np
import torch
from typing import Optional, Tuple

# ─── T3 constants (matches Rust ScalarBackend) ─────────────────────
EPSILON = 0.1     # amplitude floor for phase drive term
PI = np.pi
TWO_PI = 2.0 * np.pi


def _wrap_phase(p: np.ndarray) -> np.ndarray:
    """Wrap phases to [-π, π]."""
    return (p + PI) % TWO_PI - PI


def _pst_linear_dim(n: int) -> int:
    """Dimension of PST linear features."""
    return 3 * n * 6 + 12


def pst_linear_features_np(
    th1: np.ndarray, r1: np.ndarray,
    th2: np.ndarray, r2: np.ndarray,
    th3: np.ndarray, r3: np.ndarray,
    prev_th1: np.ndarray, prev_th2: np.ndarray, prev_th3: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Pure NumPy PST linear features (mirrors Rust pst_linear_features).
    Returns (pst_dim,) array.
    """
    fv = np.zeros(3 * n * 6 + 12, dtype=np.float64)

    # Pre-compute sin/cos
    sin1, cos1 = np.sin(th1), np.cos(th1)
    sin2, cos2 = np.sin(th2), np.cos(th2)
    sin3, cos3 = np.sin(th3), np.cos(th3)

    pos = 0
    # 1. sin(θ), cos(θ) — 6N
    for i in range(n):
        fv[pos] = sin1[i]; fv[pos + 1] = cos1[i]; pos += 2
    for i in range(n):
        fv[pos] = sin2[i]; fv[pos + 1] = cos2[i]; pos += 2
    for i in range(n):
        fv[pos] = sin3[i]; fv[pos + 1] = cos3[i]; pos += 2

    # 2. dθ/dt — 3N (with wrap correction)
    for th, prev in [(th1, prev_th1), (th2, prev_th2), (th3, prev_th3)]:
        d = th - prev
        d = np.where(d > PI, d - TWO_PI, np.where(d < -PI, d + TWO_PI, d))
        fv[pos:pos + n] = d
        pos += n

    # 3. Amplitudes — 3N
    fv[pos:pos + n] = r1[:n]; pos += n
    fv[pos:pos + n] = r2[:n]; pos += n
    fv[pos:pos + n] = r3[:n]; pos += n

    # 4. Squared features — 6N (from pre-computed sin/cos)
    for i in range(n):
        fv[pos] = sin1[i] * sin1[i]; fv[pos + 1] = cos1[i] * cos1[i]; pos += 2
    for i in range(n):
        fv[pos] = sin2[i] * sin2[i]; fv[pos + 1] = cos2[i] * cos2[i]; pos += 2
    for i in range(n):
        fv[pos] = sin3[i] * sin3[i]; fv[pos + 1] = cos3[i] * cos3[i]; pos += 2

    # 5. Cross order parameters — 3 (from sin/cos)
    def cross_order(s1, c1, s2, c2):
        sum_s = np.sum(s1 * c2 - c1 * s2)
        sum_c = np.sum(c1 * c2 + s1 * s2)
        return np.sqrt(sum_s ** 2 + sum_c ** 2) / n

    fv[pos] = cross_order(sin1, cos1, sin2, cos2); pos += 1
    fv[pos] = cross_order(sin2, cos2, sin3, cos3); pos += 1
    fv[pos] = cross_order(sin1, cos1, sin3, cos3); pos += 1

    # 6. Mean sin/cos/r — 9
    nf = float(n)
    fv[pos] = sin1[:n].sum() / nf; fv[pos + 1] = cos1[:n].sum() / nf; fv[pos + 2] = r1[:n].sum() / nf; pos += 3
    fv[pos] = sin2[:n].sum() / nf; fv[pos + 1] = cos2[:n].sum() / nf; fv[pos + 2] = r2[:n].sum() / nf; pos += 3
    fv[pos] = sin3[:n].sum() / nf; fv[pos + 1] = cos3[:n].sum() / nf; fv[pos + 2] = r3[:n].sum() / nf; pos += 3

    return fv


def pst_jacobian(
    th1, r1, th2, r2, th3, r3,
    prev_th1, prev_th2, prev_th3,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Jacobian of PST features w.r.t. (θ₁, θ₂, θ₃, r₁, r₂, r₃).

    Returns:
        dPST/dθ₁, dPST/dθ₂, dPST/dθ₃, dPST/dr₁, dPST/dr₂, dPST/dr₃
        Each is (pst_dim, n) array.
    """
    pst_dim = _pst_linear_dim(n)
    d_th1 = np.zeros((pst_dim, n), dtype=np.float64)
    d_th2 = np.zeros((pst_dim, n), dtype=np.float64)
    d_th3 = np.zeros((pst_dim, n), dtype=np.float64)
    d_r1 = np.zeros((pst_dim, n), dtype=np.float64)
    d_r2 = np.zeros((pst_dim, n), dtype=np.float64)
    d_r3 = np.zeros((pst_dim, n), dtype=np.float64)

    # Pre-compute sin/cos
    sin1, cos1 = np.sin(th1), np.cos(th1)
    sin2, cos2 = np.sin(th2), np.cos(th2)
    sin3, cos3 = np.sin(th3), np.cos(th3)

    pos = 0

    # ── 1. sin(θ), cos(θ) — 6N ─────────────────────────────
    # d(sin(θ))/dθ = cos(θ), d(cos(θ))/dθ = -sin(θ)
    # Pop 1: sin1[0], cos1[0], sin1[1], cos1[1], ...
    for i in range(n):
        d_th1[pos, i] = cos1[i]      # d(sin1[i])/dθ1[i]
        d_th1[pos + 1, i] = -sin1[i]  # d(cos1[i])/dθ1[i]
        pos += 2
    # Pop 2
    for i in range(n):
        d_th2[pos, i] = cos2[i]
        d_th2[pos + 1, i] = -sin2[i]
        pos += 2
    # Pop 3
    for i in range(n):
        d_th3[pos, i] = cos3[i]
        d_th3[pos + 1, i] = -sin3[i]
        pos += 2

    # ── 2. dθ/dt (velocity) — 3N ──────────────────────────
    # d(velocity_i)/dθ_i = 1 (∂(θ_i - prev_θ_i)/∂θ_i = 1)
    # d(velocity_i)/dprev_θ_i = -1 (handled by BPTT, not here)
    # Each pop's velocity depends ONLY on its own phase:
    #   pop1 velocity → d_th1
    #   pop2 velocity → d_th2
    #   pop3 velocity → d_th3
    for i in range(n):
        d_th1[pos + i, i] = 1.0  # d(vel1_i)/d(θ1_i)
    pos += n
    for i in range(n):
        d_th2[pos + i, i] = 1.0  # d(vel2_i)/d(θ2_i)
    pos += n
    for i in range(n):
        d_th3[pos + i, i] = 1.0  # d(vel3_i)/d(θ3_i)
    pos += n

    # ── 3. Amplitudes r — 3N ──────────────────────────────
    # d(r_i)/d(r_j) = δ_ij
    for i in range(n):
        d_r1[pos, i] = 1.0
    pos += n
    for i in range(n):
        d_r2[pos, i] = 1.0
    pos += n
    for i in range(n):
        d_r3[pos, i] = 1.0
    pos += n

    # ── 4. Squared features sin²(θ), cos²(θ) — 6N ─────────
    # d(sin²)/dθ = 2·sin·cos, d(cos²)/dθ = -2·sin·cos
    for i in range(n):
        d_th1[pos, i] = 2.0 * sin1[i] * cos1[i]     # d(sin²)/dθ
        d_th1[pos + 1, i] = -2.0 * sin1[i] * cos1[i]  # d(cos²)/dθ
        pos += 2
    for i in range(n):
        d_th2[pos, i] = 2.0 * sin2[i] * cos2[i]
        d_th2[pos + 1, i] = -2.0 * sin2[i] * cos2[i]
        pos += 2
    for i in range(n):
        d_th3[pos, i] = 2.0 * sin3[i] * cos3[i]
        d_th3[pos + 1, i] = -2.0 * sin3[i] * cos3[i]
        pos += 2

    # ── 5. Cross order parameters R₁₂, R₂₃, R₁₃ — 3 ────────
    # R = √((Σ(s₁·c₂ - c₁·s₂))² + (Σ(c₁·c₂ + s₁·s₂))²) / N
    # dR/dθ₁ involves chain rule through s₁=sin(θ₁), c₁=cos(θ₁)
    def cross_order_and_grad(s1, c1, s2, c2, d_theta_vec, pop):
        sum_s = np.sum(s1 * c2 - c1 * s2, dtype=np.float64)
        sum_c = np.sum(c1 * c2 + s1 * s2, dtype=np.float64)
        norm = np.sqrt(sum_s ** 2 + sum_c ** 2)
        if norm < 1e-12:
            return
        inv_norm = 1.0 / norm

        # dR/dθ1[i] = (sum_s * d(sum_s)/dθ1[i] + sum_c * d(sum_c)/dθ1[i]) / (norm * N)
        # d(sum_s)/dθ1[i] = c1[i]*c2[i] - (-s1[i])*s2[i] = c1[i]*c2[i] + s1[i]*s2[i] = cos(θ1-θ2)
        # d(sum_c)/dθ1[i] = -s1[i]*c2[i] + c1[i]*s2[i] = -sin(θ1-θ2)
        # NOTE: use s1,c1,s2,c2 from params, NOT outer scope sin1,cos1,sin2,cos2!
        inv_norm_n = inv_norm / n
        d_sum_s_dth = c1 * c2 + s1 * s2  # cos(θ1-θ2)
        d_sum_c_dth = -s1 * c2 + c1 * s2  # -sin(θ1-θ2)
        d_theta_vec[pos, :] = (sum_s * d_sum_s_dth + sum_c * d_sum_c_dth) * inv_norm_n

    # R₁₂ — derivatives w.r.t. θ₁ and θ₂
    cross_order_and_grad(sin1, cos1, sin2, cos2, d_th1, 1)
    # R₁₂ derivative w.r.t. θ₂
    sum_s_12 = np.sum(sin1 * cos2 - cos1 * sin2, dtype=np.float64)
    sum_c_12 = np.sum(cos1 * cos2 + sin1 * sin2, dtype=np.float64)
    norm_12 = np.sqrt(sum_s_12 ** 2 + sum_c_12 ** 2)
    if norm_12 > 1e-12:
        inv_norm_n_12 = (1.0 / norm_12) / n
        # d(s1*c2 - c1*s2)/dθ2 = s1*(-s2) - c1*c2 = -(s1*s2 + c1*c2) = -cos(θ1-θ2)
        # d(c1*c2 + s1*s2)/dθ2 = c1*(-s2) + s1*c2 = -cos1*sin2 + sin1*cos2 = sin(θ1-θ2)
        d_sum_s_dth2 = -(cos1 * cos2 + sin1 * sin2)  # -cos(θ1-θ2)
        d_sum_c_dth2 = -cos1 * sin2 + sin1 * cos2  # sin(θ1-θ2)
        d_th2[pos, :] = (sum_s_12 * d_sum_s_dth2 + sum_c_12 * d_sum_c_dth2) * inv_norm_n_12
    pos += 1

    # R₂₃ — derivatives w.r.t. θ₂ and θ₃
    cross_order_and_grad(sin2, cos2, sin3, cos3, d_th2, 2)
    sum_s_23 = np.sum(sin2 * cos3 - cos2 * sin3, dtype=np.float64)
    sum_c_23 = np.sum(cos2 * cos3 + sin2 * sin3, dtype=np.float64)
    norm_23 = np.sqrt(sum_s_23 ** 2 + sum_c_23 ** 2)
    if norm_23 > 1e-12:
        inv_norm_n_23 = (1.0 / norm_23) / n
        d_sum_s_dth3_23 = -(cos2 * cos3 + sin2 * sin3)  # -cos(θ2-θ3)
        d_sum_c_dth3_23 = -cos2 * sin3 + sin2 * cos3  # sin(θ2-θ3)
        d_th3[pos, :] = (sum_s_23 * d_sum_s_dth3_23 + sum_c_23 * d_sum_c_dth3_23) * inv_norm_n_23
    pos += 1

    # R₁₃ — derivatives w.r.t. θ₁ and θ₃
    cross_order_and_grad(sin1, cos1, sin3, cos3, d_th1, 3)
    sum_s_13 = np.sum(sin1 * cos3 - cos1 * sin3, dtype=np.float64)
    sum_c_13 = np.sum(cos1 * cos3 + sin1 * sin3, dtype=np.float64)
    norm_13 = np.sqrt(sum_s_13 ** 2 + sum_c_13 ** 2)
    if norm_13 > 1e-12:
        inv_norm_n_13 = (1.0 / norm_13) / n
        d_sum_s_dth3_13 = -(cos1 * cos3 + sin1 * sin3)  # -cos(θ1-θ3)
        d_sum_c_dth3_13 = -cos1 * sin3 + sin1 * cos3  # sin(θ1-θ3)
        d_th3[pos, :] = (sum_s_13 * d_sum_s_dth3_13 + sum_c_13 * d_sum_c_dth3_13) * inv_norm_n_13
    pos += 1

    # ── 6. Mean sin/cos/r — 9 ──────────────────────────────
    nf = float(n)
    # Pop 1 mean sin: d(mean_sin1)/dθ1[i] = cos1[i]/n
    d_th1[pos, :] = cos1[:n] / nf
    d_th1[pos + 1, :] = -sin1[:n] / nf
    d_r1[pos + 2, :] = 1.0 / nf * np.ones(n)
    pos += 3
    # Pop 2
    d_th2[pos, :] = cos2[:n] / nf
    d_th2[pos + 1, :] = -sin2[:n] / nf
    d_r2[pos + 2, :] = 1.0 / nf * np.ones(n)
    pos += 3
    # Pop 3
    d_th3[pos, :] = cos3[:n] / nf
    d_th3[pos + 1, :] = -sin3[:n] / nf
    d_r3[pos + 2, :] = 1.0 / nf * np.ones(n)
    pos += 3

    return d_th1, d_th2, d_th3, d_r1, d_r2, d_r3


class T3SpikeEncoder(torch.autograd.Function):
    """
    Differentiable T3 oscillator step.

    Forward: drives → T3 (N steps) → PST features
    Backward: d(loss)/d(PST) → BPTT through T3 → d(loss)/d(drives)

    The backward pass computes the analytical Jacobian of PST features
    w.r.t. the drive input by backpropagating through the T3 ODE dynamics.

    T3 dynamics (per step, per oscillator):
        r += dt · ((α - r²)·r + drive · gain)
        θ += dt · (ω + coupling + drive · gain / (r + ε))
        where ε = 0.1 (amplitude floor)
    """

    @staticmethod
    def forward(ctx, drives: torch.Tensor,
                tri, n_osc: int, n_steps: int, dt: float,
                drive_gain: float = 10.0, alpha: float = 1.0,
                freq_min: float = 0.5, freq_max: float = 3.0,
                coupling: tuple = (0.3, 0.15, 0.08),
                return_full_trajectory: bool = False):
        """
        Forward pass through T3 dynamics.

        Args:
            drives: (batch, n_steps, 3*n_osc) or (n_steps, 3*n_osc) — drive inputs
            tri: PyTriOscillator instance (reused, state will be modified)
            n_osc: oscillators per population
            n_steps: number of T3 steps
            dt: simulation timestep
            drive_gain: Stuart-Landau drive gain
            alpha: Stuart-Landau nonlinearity
            freq_min, freq_max: frequency range
            coupling: (w12, w13, w23)
            return_full_trajectory: if True, return all PST features per step

        Returns:
            pst_features: (batch, pst_dim) or (pst_dim,) — final PST features
            optionally: full_trajectory
        """
        import numpy as np

        # Handle batch dimension
        was_1d = drives.ndim == 2
        if was_1d:
            drives = drives.unsqueeze(0)

        batch_size = drives.shape[0]
        n_total = n_osc * 3  # total drive dimension
        pst_dim = _pst_linear_dim(n_osc)
        device = drives.device

        # Validate drive dimensions
        assert drives.shape[-1] == n_total, \
            f"drives.shape[-1]={drives.shape[-1]} != 3*n_osc={n_total}"
        assert drives.shape[1] == n_steps, \
            f"drives.shape[1]={drives.shape[1]} != n_steps={n_steps}"

        # Record trajectory for backward
        # State shape: (batch, n_steps, 3*n_osc) for phases and amplitudes
        all_th1 = np.zeros((batch_size, n_steps, n_osc), dtype=np.float64)
        all_th2 = np.zeros((batch_size, n_steps, n_osc), dtype=np.float64)
        all_th3 = np.zeros((batch_size, n_steps, n_osc), dtype=np.float64)
        all_r1 = np.zeros((batch_size, n_steps, n_osc), dtype=np.float64)
        all_r2 = np.zeros((batch_size, n_steps, n_osc), dtype=np.float64)
        all_r3 = np.zeros((batch_size, n_steps, n_osc), dtype=np.float64)
        all_drives = np.zeros((batch_size, n_steps, n_total), dtype=np.float64)
        all_coupling1 = np.zeros((batch_size, n_steps, n_osc), dtype=np.float64)
        all_coupling2 = np.zeros((batch_size, n_steps, n_osc), dtype=np.float64)
        all_coupling3 = np.zeros((batch_size, n_steps, n_osc), dtype=np.float64)

        # Pre-compute coupling matrices (same for all batch items)
        w12 = np.full((n_osc, n_osc), coupling[0] / n_osc, dtype=np.float64)
        w13 = np.full((n_osc, n_osc), coupling[1] / n_osc, dtype=np.float64)
        w23 = np.full((n_osc, n_osc), coupling[2] / n_osc, dtype=np.float64)

        # Phase gate params
        gamma = (0.5, 1.0, 1.12)
        phase_gate_gain = 4.0

        # Natural frequencies (linspace per population)
        freq_spread = freq_max - freq_min
        omega1 = np.array([freq_min + freq_spread * i / n_osc for i in range(n_osc)], dtype=np.float64)
        omega2 = np.array([freq_min + freq_spread * (i + n_osc // 3) / n_osc for i in range(n_osc)], dtype=np.float64)
        omega3 = np.array([freq_min + freq_spread * (i + 2 * n_osc // 3) / n_osc for i in range(n_osc)], dtype=np.float64)

        pst_out = np.zeros((batch_size, pst_dim), dtype=np.float64)
        traj_out = None
        if return_full_trajectory:
            traj_out = np.zeros((batch_size, n_steps, pst_dim), dtype=np.float64)

        for b in range(batch_size):
            # Reset T3 to initial state
            # Reset by creating new state variables
            th1 = np.zeros(n_osc, dtype=np.float64)
            th2 = np.zeros(n_osc, dtype=np.float64)
            th3 = np.zeros(n_osc, dtype=np.float64)
            r1 = np.full(n_osc, 0.1, dtype=np.float64)
            r2 = np.full(n_osc, 0.1, dtype=np.float64)
            r3 = np.full(n_osc, 0.1, dtype=np.float64)
            prev_th1 = np.zeros(n_osc, dtype=np.float64)
            prev_th2 = np.zeros(n_osc, dtype=np.float64)
            prev_th3 = np.zeros(n_osc, dtype=np.float64)
            prev_r1 = np.full(n_osc, 0.1, dtype=np.float64)
            prev_r2 = np.full(n_osc, 0.1, dtype=np.float64)
            prev_r3 = np.full(n_osc, 0.1, dtype=np.float64)

            drive_b = drives[b].detach().cpu().numpy()  # (n_steps, 3*n_osc)

            for step in range(n_steps):
                # ── Save previous state ──
                prev_th1[:] = th1
                prev_th2[:] = th2
                prev_th3[:] = th3
                prev_r1[:] = r1
                prev_r2[:] = r2
                prev_r3[:] = r3

                # ── Get drive for this step ──
                d1 = drive_b[step, 0:n_osc]
                d2 = drive_b[step, n_osc:2*n_osc]
                d3 = drive_b[step, 2*n_osc:3*n_osc]
                all_drives[b, step, :] = drive_b[step]

                # ── Compute sin/cos (mark step 0 as initialised) ──
                sin1, cos1 = np.sin(th1), np.cos(th1)
                sin2, cos2 = np.sin(th2), np.cos(th2)
                sin3, cos3 = np.sin(th3), np.cos(th3)

                # ── Phase Gate ──
                def order_param(sin_vec, cos_vec):
                    r_sum_s = np.sum(sin_vec)
                    r_sum_c = np.sum(cos_vec)
                    return np.sqrt(r_sum_s ** 2 + r_sum_c ** 2) / n_osc

                def phase_gate(r, gamma_val, gain):
                    return 1.0 / (1.0 + np.exp(-gain * (r - gamma_val)))

                r1_op = order_param(sin1, cos1)
                r2_op = order_param(sin2, cos2)
                r3_op = order_param(sin3, cos3)
                g1 = phase_gate(r1_op, gamma[0], phase_gate_gain)
                g2 = phase_gate(r2_op, gamma[1], phase_gate_gain)
                g3 = phase_gate(r3_op, gamma[2], phase_gate_gain)

                # ── Cross-population coupling ──
                # θ1 from θ2 + θ3
                cpl1 = np.zeros(n_osc)
                for i in range(n_osc):
                    cpl1[i] = (
                        np.sum(w12[i] * r2 * (sin2 * cos1[i] - cos2 * sin1[i])) * g2 +
                        np.sum(w13[i] * r3 * (sin3 * cos1[i] - cos3 * sin1[i])) * g3
                    )
                all_coupling1[b, step] = cpl1

                # θ2 from θ1 + θ3
                cpl2 = np.zeros(n_osc)
                for i in range(n_osc):
                    cpl2[i] = (
                        np.sum(w12[:, i] * r1 * (sin1 * cos2[i] - cos1 * sin2[i])) * g1 +
                        np.sum(w23[i] * r3 * (sin3 * cos2[i] - cos3 * sin2[i])) * g3
                    )
                all_coupling2[b, step] = cpl2

                # θ3 from θ1 + θ2
                cpl3 = np.zeros(n_osc)
                for i in range(n_osc):
                    cpl3[i] = (
                        np.sum(w13[:, i] * r1 * (sin1 * cos3[i] - cos1 * sin3[i])) * g1 +
                        np.sum(w23[:, i] * r2 * (sin2 * cos3[i] - cos2 * sin3[i])) * g2
                    )
                all_coupling3[b, step] = cpl3

                # ── Update amplitudes (Stuart-Landau) ──
                r1 += dt * ((alpha - r1 * r1) * r1 + d1 * drive_gain)
                r2 += dt * ((alpha - r2 * r2) * r2 + d2 * drive_gain)
                r3 += dt * ((alpha - r3 * r3) * r3 + d3 * drive_gain)

                # ── Update phases (Kuramoto) ──
                th1 += dt * (omega1 + cpl1 + d1 * drive_gain / (r1 + EPSILON))
                th2 += dt * (omega2 + cpl2 + d2 * drive_gain / (r2 + EPSILON))
                th3 += dt * (omega3 + cpl3 + d3 * drive_gain / (r3 + EPSILON))

                # ── Wrap phases ──
                th1 = _wrap_phase(th1)
                th2 = _wrap_phase(th2)
                th3 = _wrap_phase(th3)

                # ── Clamp amplitudes ──
                r1 = np.clip(r1, 0.0, 10.0)
                r2 = np.clip(r2, 0.0, 10.0)
                r3 = np.clip(r3, 0.0, 10.0)

                # ── Record state ──
                all_th1[b, step] = th1
                all_th2[b, step] = th2
                all_th3[b, step] = th3
                all_r1[b, step] = r1
                all_r2[b, step] = r2
                all_r3[b, step] = r3

                # ── Compute PST features ──
                if return_full_trajectory or step == n_steps - 1:
                    fv = pst_linear_features_np(
                        th1, r1, th2, r2, th3, r3,
                        prev_th1, prev_th2, prev_th3, n_osc,
                    )
                    if return_full_trajectory:
                        traj_out[b, step] = fv
                    if step == n_steps - 1:
                        pst_out[b] = fv

        # ── Save for backward ──
        ctx.save_for_backward(
            torch.tensor(all_th1), torch.tensor(all_th2), torch.tensor(all_th3),
            torch.tensor(all_r1), torch.tensor(all_r2), torch.tensor(all_r3),
            torch.tensor(all_drives),
            torch.tensor(all_coupling1), torch.tensor(all_coupling2), torch.tensor(all_coupling3),
            torch.tensor(omega1), torch.tensor(omega2), torch.tensor(omega3),
            torch.tensor(w12), torch.tensor(w13), torch.tensor(w23),
        )
        ctx.n_osc = n_osc
        ctx.n_steps = n_steps
        ctx.dt = dt
        ctx.drive_gain = drive_gain
        ctx.alpha = alpha
        ctx.batch_size = batch_size
        ctx.was_1d = was_1d

        result = torch.tensor(pst_out, device=device)
        if was_1d:
            result = result.squeeze(0)
        if return_full_trajectory:
            traj_t = torch.tensor(traj_out, device=device)
            if was_1d:
                traj_t = traj_t.squeeze(0)
            return result, traj_t
        return result

    @staticmethod
    def backward(ctx, grad_pst):
        """
        Backward pass: BPTT through T3 dynamics.

        grad_pst: (batch, pst_dim) or (pst_dim,) — d(L)/d(PST)
        Returns: grad_drives, None, None, None, None, None, None, None, None, None, None
        """
        import torch

        # Handle batch dimension
        was_1d = grad_pst.ndim == 1
        if was_1d:
            grad_pst = grad_pst.unsqueeze(0)

        batch_size = ctx.batch_size
        n = ctx.n_osc
        n_steps = ctx.n_steps
        dt = ctx.dt
        gain = ctx.drive_gain
        alpha_val = ctx.alpha

        # Retrieve saved tensors
        (all_th1, all_th2, all_th3,
         all_r1, all_r2, all_r3,
         all_drives,
         all_cpl1, all_cpl2, all_cpl3,
         omega1, omega2, omega3,
         w12, w13, w23) = ctx.saved_tensors

        # Convert to numpy for analytical computation
        th1 = all_th1.numpy()
        th2 = all_th2.numpy()
        th3 = all_th3.numpy()
        r1 = all_r1.numpy()
        r2 = all_r2.numpy()
        r3 = all_r3.numpy()
        drives = all_drives.numpy()
        cpl1 = all_cpl1.numpy()
        cpl2 = all_cpl2.numpy()
        cpl3 = all_cpl3.numpy()
        w12_np = w12.numpy()
        w13_np = w13.numpy()
        w23_np = w23.numpy()

        grad_drives = np.zeros((batch_size, n_steps, 3 * n), dtype=np.float64)

        for b in range(batch_size):
            # ── Step 1: Compute d(L)/d(state_N) from d(L)/d(PST) ──
            # d(L)/d(s_N) = d(PST)/d(s_N)^T · d(L)/d(PST)
            # where s_N = (θ₁, θ₂, θ₃, r₁, r₂, r₃) at step N-1
            gpst = grad_pst[b].numpy()  # (pst_dim,)

            # Compute PST Jacobian at final state
            d_th1_j, d_th2_j, d_th3_j, d_r1_j, d_r2_j, d_r3_j = pst_jacobian(
                th1[b, -1], r1[b, -1], th2[b, -1], r2[b, -1], th3[b, -1], r3[b, -1],
                # prev for velocity: step -2 (or 0 if only 1 step)
                th1[b, -2] if n_steps > 1 else th1[b, -1],
                th2[b, -2] if n_steps > 1 else th2[b, -1],
                th3[b, -2] if n_steps > 1 else th3[b, -1],
                n,
            )

            # v = d(L)/d(state) — gradient w.r.t. state at step t
            v_th1 = d_th1_j.T @ gpst  # (n,)
            v_th2 = d_th2_j.T @ gpst
            v_th3 = d_th3_j.T @ gpst
            v_r1 = d_r1_j.T @ gpst
            v_r2 = d_r2_j.T @ gpst
            v_r3 = d_r3_j.T @ gpst

            # ── Step 2: BPTT from step N-1 down to 0 ──
            for step in range(n_steps - 1, -1, -1):
                # Current state variables
                th1_t = th1[b, step]
                th2_t = th2[b, step]
                th3_t = th3[b, step]
                r1_t = r1[b, step]
                r2_t = r2[b, step]
                r3_t = r3[b, step]
                d1_t = drives[b, step, 0:n]
                d2_t = drives[b, step, n:2*n]
                d3_t = drives[b, step, 2*n:3*n]
                c1_t = cpl1[b, step]
                c2_t = cpl2[b, step]
                c3_t = cpl3[b, step]

                # ── Local gradient of T3 step w.r.t. drive ──
                # r_next = r + dt · ((α - r²)·r + d · gain)
                # d(r_next)/d(d) = dt · gain
                dr1_dd1 = dt * gain
                dr2_dd2 = dt * gain
                dr3_dd3 = dt * gain

                # θ_next = θ + dt · (ω + coupling + d·gain/(r+ε))
                # d(θ_next)/d(d) = dt · gain / (r + ε)
                dth1_dd1 = dt * gain / (r1_t + EPSILON)
                dth2_dd2 = dt * gain / (r2_t + EPSILON)
                dth3_dd3 = dt * gain / (r3_t + EPSILON)

                # ── Gradient w.r.t. drive at this step ──
                # d(L)/d(d_t) = d(L)/d(θ_{t+1}) · d(θ_{t+1})/d(d_t)
                #             + d(L)/d(r_{t+1}) · d(r_{t+1})/d(d_t)
                grad_drives[b, step, 0:n] = v_th1 * dth1_dd1 + v_r1 * dr1_dd1
                grad_drives[b, step, n:2*n] = v_th2 * dth2_dd2 + v_r2 * dr2_dd2
                grad_drives[b, step, 2*n:3*n] = v_th3 * dth3_dd3 + v_r3 * dr3_dd3

                # ── Backprop through T3 step to get v for PREVIOUS step ──
                # Skip if this is the first step (no previous state to propagate to)
                # Actually, we need v for state at step t, which affects step t+1
                # The gradient w.r.t. state at step t is:
                # d(L)/d(θ_t) = d(L)/d(θ_{t+1}) · d(θ_{t+1})/d(θ_t)
                #             + d(L)/d(r_{t+1}) · d(r_{t+1})/d(θ_t)
                # But also from the velocity term in PST!
                # Actually no — the velocity term uses prev_θ from step-1, which is θ_t.
                # The gradient from velocity flows to θ_t (the prev_θ at step t+1 = θ_t).

                if step > 0:
                    # ── d(r_next)/d(r) ──
                    # r_next = r + dt · ((α - r²)·r + d·gain)
                    # d(r_next)/d(r) = 1 + dt · (α - 3·r²)
                    dr1_dr1 = 1.0 + dt * (alpha_val - 3.0 * r1_t ** 2)
                    dr2_dr2 = 1.0 + dt * (alpha_val - 3.0 * r2_t ** 2)
                    dr3_dr3 = 1.0 + dt * (alpha_val - 3.0 * r3_t ** 2)

                    # ── d(θ_next)/d(θ) via coupling ──
                    # θ_next = θ + dt · (ω + coupling(θ) + d·g/(r+ε))
                    # d(θ_next)/d(θ) = I + dt · d(coupling)/d(θ)
                    # coupling[i] = Σ_j w_ij · r_j · sin(θ_j - θ_i) · G
                    # d(coupling[i])/d(θ_i) = -Σ_j w_ij · r_j · cos(θ_j - θ_i) · G  [self damping]
                    # d(coupling[i])/d(θ_j) = w_ij · r_j · cos(θ_j - θ_i) · G  [for j≠i, cross]

                    # Phase gates at this step
                    sin1_t = np.sin(th1_t)
                    cos1_t = np.cos(th1_t)
                    sin2_t = np.sin(th2_t)
                    cos2_t = np.cos(th2_t)
                    sin3_t = np.sin(th3_t)
                    cos3_t = np.cos(th3_t)

                    def order_param_vec(s, c):
                        rs = np.sum(s)
                        rc = np.sum(c)
                        return np.sqrt(rs ** 2 + rc ** 2) / n

                    def phase_gate_val(r_val, gamma_val, g_gain):
                        return 1.0 / (1.0 + np.exp(-g_gain * (r_val - gamma_val)))

                    g1_t = phase_gate_val(order_param_vec(sin1_t, cos1_t), 0.5, 4.0)
                    g2_t = phase_gate_val(order_param_vec(sin2_t, cos2_t), 1.0, 4.0)
                    g3_t = phase_gate_val(order_param_vec(sin3_t, cos3_t), 1.12, 4.0)

                    # ── d(θ_next)/d(θ) Jacobian ──
                    # For efficiency, compute diagonal and off-diagonal separately
                    # dθ1_next[i]/dθ1[i] = 1 + dt · d(cpl1[i])/dθ1[i]
                    # dθ1_next[i]/dθ2[j] = dt · d(cpl1[i])/dθ2[j]

                    # Self-coupling derivatives (diagonal of d(coupling)/dθ)
                    # cpl1[i] = Σ_j w12[i,j]*r2[j]*sin(θ2[j]-θ1[i])*g2 + Σ_j w13[i,j]*r3[j]*sin(θ3[j]-θ1[i])*g3
                    # d(cpl1[i])/dθ1[i] = -Σ_j w12[i,j]*r2[j]*cos(θ2[j]-θ1[i])*g2 - Σ_j w13[i,j]*r3[j]*cos(θ3[j]-θ1[i])*g3
                    dcpl1_dth1 = np.zeros(n)
                    for i in range(n):
                        dcpl1_dth1[i] = -(
                            np.sum(w12_np[i] * r2_t * np.cos(th2_t - th1_t[i])) * g2_t +
                            np.sum(w13_np[i] * r3_t * np.cos(th3_t - th1_t[i])) * g3_t
                        )

                    dcpl2_dth2 = np.zeros(n)
                    for i in range(n):
                        dcpl2_dth2[i] = -(
                            np.sum(w12_np[:, i] * r1_t * np.cos(th1_t - th2_t[i])) * g1_t +
                            np.sum(w23_np[i] * r3_t * np.cos(th3_t - th2_t[i])) * g3_t
                        )

                    dcpl3_dth3 = np.zeros(n)
                    for i in range(n):
                        dcpl3_dth3[i] = -(
                            np.sum(w13_np[:, i] * r1_t * np.cos(th1_t - th3_t[i])) * g1_t +
                            np.sum(w23_np[:, i] * r2_t * np.cos(th2_t - th3_t[i])) * g2_t
                        )

                    # d(θ_next)/d(θ) = I + dt · d(coupling)/d(θ) (diagonal approx)
                    dth1_dth1_self = 1.0 + dt * dcpl1_dth1
                    dth2_dth2_self = 1.0 + dt * dcpl2_dth2
                    dth3_dth3_self = 1.0 + dt * dcpl3_dth3

                    # d(θ_next)/d(r) = -dt · d · gain / (r + ε)²
                    dth1_dr1 = -dt * d1_t * gain / (r1_t + EPSILON) ** 2
                    dth2_dr2 = -dt * d2_t * gain / (r2_t + EPSILON) ** 2
                    dth3_dr3 = -dt * d3_t * gain / (r3_t + EPSILON) ** 2

                    # ── Combine gradients ──
                    # v_th1 (for previous step) = v_th1_next · dθ1_next/dθ1
                    #                             + v_th2_next · dθ2_next/dθ1 (cross coupling)
                    #                             + [no r→θ gradient]

                    # For the diagonal approximation (ignoring cross-coupling Jacobian):
                    v_th1_prev = v_th1 * dth1_dth1_self
                    v_th2_prev = v_th2 * dth2_dth2_self
                    v_th3_prev = v_th3 * dth3_dth3_self

                    # r gradients: v_r includes d(L)/d(r) from both r_next and θ_next
                    v_r1_prev = v_r1 * dr1_dr1 + v_th1 * dth1_dr1
                    v_r2_prev = v_r2 * dr2_dr2 + v_th2 * dth2_dr2
                    v_r3_prev = v_r3 * dr3_dr3 + v_th3 * dth3_dr3

                    # ── Velocity gradient contribution ──
                    # PST features include velocity = θ_current - θ_prev (with wrap).
                    # pst_jacobian only covers d(velocity)/d(θ_current) = +1.
                    # The d(velocity)/d(θ_prev) = -1 contributes to the state at step t.
                    # For the LAST step (step = n_steps - 1), we need to add the velocity
                    # gradient from PST to θ_{N-1} AFTER the dynamics backprop.
                    #
                    # d(L)/d(θ_{N-1}) += d(L)/d(velocity_N) · (-1)
                    # velocity features are at positions 6n to 9n-1 in PST
                    if step == n_steps - 1:
                        vel_start = 6 * n
                        dL_dvel = gpst[vel_start:vel_start + 3 * n]
                        v_th1_prev -= dL_dvel[:n]    # -dL/d(vel1_i)
                        v_th2_prev -= dL_dvel[n:2*n]  # -dL/d(vel2_i)
                        v_th3_prev -= dL_dvel[2*n:3*n]  # -dL/d(vel3_i)

                    v_th1, v_th2, v_th3 = v_th1_prev, v_th2_prev, v_th3_prev
                    v_r1, v_r2, v_r3 = v_r1_prev, v_r2_prev, v_r3_prev

        # Convert to torch tensor and return
        result = torch.tensor(grad_drives, device=grad_pst.device, dtype=grad_pst.dtype)
        if was_1d:
            result = result.squeeze(0)
        return result, None, None, None, None, None, None, None, None, None, None
