#!/usr/bin/env python3
"""
КВАНТОВАЯ НЕЙРОДИНАМИКА НА МОЗГЕ МУХИ
======================================
ψ-wave на FAFB connectome (139K нейронов, 22M синапсов).

Уравнение:  ħ·dψ/dt = -i·W·ψ + external drive
            ψ ← ψ / |ψ|  (унитарная эволюция)
"""

import time, sys
import numpy as np
from taas_checkmate import step_psi_wave_full, compute_hamiltonian

DT = 0.1
HBAR = 1.0
N_STEPS = 200
REPORT_EVERY = 10

print("█" * 60)
print("██  ψ-WAVE ON FLYWIRE (FAFB) CONNECTOME")
print("█" * 60)

# ─── Load connectome ───
print("\n[1/4] Loading FAFB connectome...", end=" ", flush=True)
t0 = time.perf_counter()
data = np.load("fafb_139k.npz")
row_ptr = data["row_ptr"]       # (139256,) int32
col_indices = data["col_indices"]  # (22285323,) int32
values = data["values"]         # (22285323,) float32
n_neurons = len(row_ptr) - 1
n_synapses = len(values)
print(f"{n_neurons:,} neurons, {n_synapses:,} synapses ({time.perf_counter()-t0:.1f}s)")

# ─── Init ψ state ───
print("\n[2/4] Initializing ψ-wave state...", end=" ", flush=True)
t0 = time.perf_counter()
rng = np.random.default_rng(42)
theta = rng.uniform(0, 2 * np.pi, n_neurons).astype(np.float32)
v = np.cos(theta, dtype=np.float32)  # real part
u = np.sin(theta, dtype=np.float32)  # imag part
# Small perturbation for initial excitation
v += 0.1 * rng.standard_normal(n_neurons).astype(np.float32)
u += 0.1 * rng.standard_normal(n_neurons).astype(np.float32)
nrm = np.sqrt(v*v + u*u)
v /= nrm
u /= nrm
print(f"done ({time.perf_counter()-t0:.1f}s)")

# ─── Buffers ───
print("\n[3/4] Allocating buffers...", end=" ", flush=True)
t0 = time.perf_counter()
h_real = np.zeros(n_neurons, dtype=np.float32)
h_imag = np.zeros(n_neurons, dtype=np.float32)
print(f"done ({time.perf_counter()-t0:.1f}s)")

# ─── Run ───
print("\n[4/4] Running ψ-wave evolution...")
print()
print(f"  {'Step':>5}  {'Energy':>10}  {'Coherence':>10}  {'|ψ|_mean':>9}  {'Active':>8}  {'Time':>7}")
print(f"  {'─────':>5}  {'──────────':>10}  {'──────────':>10}  {'────────':>9}  {'──────':>8}  {'──────':>7}")

t_start = time.perf_counter()

for step in range(N_STEPS):
    # Compute Hamiltonian: H = -i·W·ψ
    # H_real = W·u, H_imag = -W·v
    compute_hamiltonian(row_ptr, col_indices, values, v, u, h_real, h_imag)

    # Full ψ-wave step: unitary evolution + normalization
    step_psi_wave_full(v, u, row_ptr, col_indices, values, DT, HBAR)

    if step % REPORT_EVERY == 0 or step == N_STEPS - 1:
        # Energy = ||Hψ||² (total energy of the wavefunction)
        energy = float(np.mean(h_real**2 + h_imag**2))

        # Coherence = |mean(ψ)| = how synchronized the phases are
        mean_v = float(np.mean(v))
        mean_u = float(np.mean(u))
        coherence = float(np.sqrt(mean_v**2 + mean_u**2))

        # Mean amplitude (should be 1.0 due to normalization)
        amp = float(np.mean(np.sqrt(v**2 + u**2)))

        # "Active" neurons (phase changing rapidly = |Hψ| > threshold)
        active = int(np.sum(h_real**2 + h_imag**2 > 1.0))

        elapsed = time.perf_counter() - t_start
        print(f"  {step:5d}  {energy:10.4f}  {coherence:10.6f}  {amp:9.6f}  {active:7d}  {elapsed:6.1f}s")

total = time.perf_counter() - t_start
print(f"\n  Total: {total:.1f}s  |  {DT*N_STEPS:.1f} time-units simulated")
print()

# ─── Final state analysis ───
print("█" * 60)
print("FINAL STATE ANALYSIS")
print("█" * 60)

# Phase histogram
phases = np.arctan2(u, v)
print(f"\n  Phase distribution:")
print(f"    Mean:     {float(np.mean(phases)):.4f} rad")
print(f"    Std:      {float(np.std(phases)):.4f} rad")
print(f"    Coherence:|⟨ψ⟩| = {coherence:.6f}  (1=sync, 0=uniform)")

# Energy distribution
energy_per_neuron = h_real**2 + h_imag**2
print(f"\n  Energy (||Hψ||²):")
print(f"    Mean:     {float(np.mean(energy_per_neuron)):.4f}")
print(f"    Max:      {float(np.max(energy_per_neuron)):.4f}")
print(f"    Top 1%:   {float(np.sort(energy_per_neuron)[-n_neurons//100]):.4f}")
print(f"    Active:   {active:,} / {n_neurons:,} ({100*active/n_neurons:.1f}%)")

# Connection energy: which synapses carry most energy?
print(f"\n  Network stats:")
print(f"    Synapses: {n_synapses:,}")
print(f"    Avg degree: {n_synapses/n_neurons:.1f} syn/neuron")

# Check if we have resonance patterns
peaks = np.sum(np.abs(np.diff(phases)) > 1.0)
print(f"    Phase boundaries: {peaks:,} ({100*peaks/n_neurons:.1f}% of pairs)")

print()
print("█" * 60)
print("██  DONE — ψ-wave evolved through FAFB connectome")
print("█" * 60)
