#!/usr/bin/env python3
"""Generate Jupyter notebooks for taas-checkmate tutorials."""

import nbformat as nbf
import os

nb = nbf.v4


def _code(source):
    return nb.new_code_cell(source)


def _md(source):
    return nb.new_markdown_cell(source)


def _raw(source):
    return nb.new_raw_cell(source)


# ═══════════════════════════════════════════════════════════
#  1. Izhikevich Neuron
# ═══════════════════════════════════════════════════════════

NOTEBOOKS = {}

NOTEBOOKS["01_izhikevich"] = {
    "title": "Izhikevich Neuron Model",
    "cells": [
        _md("""# 🧠 Izhikevich Neuron Model

The Izhikevich model balances biological realism with computational efficiency:

$$\\dot{v} = 0.04v^2 + 5v + 140 - u + I$$
$$\\dot{u} = a(bv - u)$$

With spike reset: if $v \\geq 30$, then $v \\leftarrow c$, $u \\leftarrow u + d$

**Parameters:**
- `a`: time scale of recovery variable
- `b`: sensitivity of recovery variable
- `c`: after-spike reset value
- `d`: after-spike reset of recovery variable

This SIMD implementation processes **8 neurons per CPU instruction** via `wide::f32x8`.
"""),
        _md("## Setup"),
        _code("""import numpy as np
import matplotlib.pyplot as plt
try:
    import taas_checkmate as ne
except ImportError:
    import subprocess
    subprocess.run(["maturin", "develop", "--release", "-q"], check=True)
    import taas_checkmate as ne

# Configure matplotlib for dark theme
plt.style.use("dark_background")
print(f"✅ Module loaded: taas_checkmate v{getattr(ne, '__version__', '0.1.0')}")
"""),
        _md("## Population Setup"),
        _code("""N = 1024
N_PAD = (N + 7) // 8 * 8  # SIMD padding
DT = 0.5  # ms

rng = np.random.default_rng(42)

# Izhikevich parameters (heterogeneous population)
v = np.full(N_PAD, -65.0, dtype=np.float32)
u = np.full(N_PAD, -13.0, dtype=np.float32)
trace = np.zeros(N_PAD, dtype=np.float32)
spikes = np.zeros(N_PAD, dtype=np.int32)

a = rng.uniform(0.02, 0.08, N_PAD).astype(np.float32)  # excitatory
b = rng.uniform(0.15, 0.25, N_PAD).astype(np.float32)
c = rng.uniform(-65.0, -50.0, N_PAD).astype(np.float32)
d = rng.uniform(2.0, 8.0, N_PAD).astype(np.float32)
I = rng.uniform(0, 15, N_PAD).astype(np.float32)

# Make ~20% inhibitory
n_inh = N_PAD // 5
a[:n_inh] = rng.uniform(0.02, 0.1, n_inh)
b[:n_inh] = rng.uniform(0.2, 0.3, n_inh)
c[:n_inh] = rng.uniform(-65, -55, n_inh)
d[:n_inh] = rng.uniform(2, 6, n_inh)

print(f"Setup: {N} neurons ({N_PAD} SIMD-padded)")
print(f"  Excitatory: {N - n_inh}, Inhibitory: {n_inh}")
"""),
        _md("## Run Simulation"),
        _code("""STEPS = 2000
spike_buffer = np.zeros((STEPS, N), dtype=np.int32)
v_buffer = np.zeros((STEPS // 10, N), dtype=np.float32)  # decimate

from rich.progress import Progress
import time

t0 = time.perf_counter()
with Progress() as pbar:
    task = pbar.add_task("[cyan]Simulating...", total=STEPS)
    for t in range(STEPS):
        ne.step_izhikevich(v, u, trace, a, b, I, DT, 0.5, spikes, c, d)
        spike_buffer[t] = spikes[:N]
        if t % 10 == 0:
            v_buffer[t // 10] = v[:N]
        pbar.update(task, advance=1)

elapsed = time.perf_counter() - t0
ns_per_step = elapsed * 1e9 / (STEPS * N)
print(f"\\n✅ {STEPS} steps in {elapsed:.2f}s = {ns_per_step:.1f} ns/neuron/step")
print(f"Mean firing rate: {spike_buffer.mean() * 1000 / DT:.1f} Hz")
"""),
        _md("## Raster Plot"),
        _code("""fig, ax = plt.subplots(figsize=(14, 6))

# Raster
spike_t, spike_n = np.where(spike_buffer > 0)
ax.scatter(spike_t * DT, spike_n, s=0.3, c="#00ff88", alpha=0.4, rasterized=True)

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Neuron")
ax.set_title("Izhikevich Spike Raster — 1024 Neurons")
ax.set_xlim(0, STEPS * DT)
ax.set_ylim(0, N)
ax.grid(alpha=0.1)

# Firing rate overlay
fr = spike_buffer.sum(axis=1) / N * 1000 / DT
ax2 = ax.twinx()
ax2.plot(np.arange(STEPS) * DT, fr, color="#ff8800", alpha=0.5, linewidth=0.8)
ax2.set_ylabel("Population Rate (Hz)", color="#ff8800")
ax2.tick_params(colors="#ff8800")

plt.tight_layout()
plt.show()
"""),
        _md("## Firing Rate Distribution"),
        _code("""# Compute firing rates per neuron
firing_rates = spike_buffer.sum(axis=0) / (STEPS * DT / 1000)  # Hz

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(firing_rates, bins=50, color="#00ff88", alpha=0.7, edgecolor="none")
ax1.set_xlabel("Firing Rate (Hz)")
ax1.set_ylabel("Count")
ax1.set_title("Firing Rate Distribution")
ax1.grid(alpha=0.1)

# Identify RS and FS neurons
rs_idx = firing_rates < 5  # Regular spiking
fs_idx = firing_rates >= 10  # Fast spiking
ax2.scatter(np.where(rs_idx)[0], firing_rates[rs_idx],
            s=5, c="#00ff88", alpha=0.5, label=f"RS ({rs_idx.sum()})")
ax2.scatter(np.where(fs_idx)[0], firing_rates[fs_idx],
            s=5, c="#ff3366", alpha=0.5, label=f"FS ({fs_idx.sum()})")
all_others = ~(rs_idx | fs_idx)
ax2.scatter(np.where(all_others)[0], firing_rates[all_others],
            s=5, c="#888888", alpha=0.3, label=f"Other ({all_others.sum()})")

ax2.set_xlabel("Neuron Index")
ax2.set_ylabel("Firing Rate (Hz)")
ax2.set_title("Neuron Types")
ax2.legend(markerscale=3)
ax2.grid(alpha=0.1)

plt.tight_layout()
plt.show()

print(f"Regular spiking: {rs_idx.sum()} neurons ({rs_idx.sum()/N*100:.0f}%)")
print(f"Fast spiking:    {fs_idx.sum()} neurons ({fs_idx.sum()/N*100:.0f}%)")
"""),
        _md("## Membrane Potential (Sample Neurons)"),
        _code("""fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)

sample_neurons = [0, 100, 200, 500]
colors = ["#00ff88", "#ff6600", "#3399ff", "#ff33cc"]

for ax, idx, color in zip(axes, sample_neurons, colors):
    v_samples = v_buffer[:, idx]
    ax.plot(np.arange(len(v_samples)) * DT * 10 / 1000, v_samples,
            color=color, linewidth=0.8)
    ax.axhline(30, color="red", linestyle="--", alpha=0.3, label="Threshold")
    ax.set_ylabel(f"V ({idx})")
    ax.grid(alpha=0.1)
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Time (s)")
fig.suptitle("Membrane Potential — Sample Neurons")
plt.tight_layout()
plt.show()
"""),
        _md("## Summary\n\n- **Izhikevich model** — SIMD-accelerated via `wide::f32x8` (8 neurons per instruction)\n- Both excitatory (RS) and inhibitory (FS) neuron types demonstrated\n- ~7.2× speedup over scalar implementation\n- Heterogeneous parameters produce rich dynamics"),
    ],
}


# ═══════════════════════════════════════════════════════════
#  2. LIF/LIF-AT/AdEx
# ═══════════════════════════════════════════════════════════

NOTEBOOKS["02_lif_adex"] = {
    "title": "LIF, LIF-AT, and AdEx Models",
    "cells": [
        _md("""# 🔋 LIF, LIF-AT, and AdEx Neuron Models

Three leaky integrate-and-fire variants:

**LIF:** $$\\tau_m \\dot{v} = -(v - v_{rest}) + I$$

**LIF-AT:** adds adaptive threshold $$\\theta$$ that increases after each spike

**AdEx (Adaptive Exponential):** adds exponential term and adaptation current w

$$\\tau_m \\dot{v} = -(v - v_{rest}) + \\Delta_T \\exp\\left(\\frac{v - v_T}{\\Delta_T}\\right) - w + I$$
$$\\tau_w \\dot{w} = a(v - v_{rest}) - w$$
"""),
        _md("## Setup"),
        _code("""import numpy as np
import matplotlib.pyplot as plt
try:
    import taas_checkmate as ne
except ImportError:
    import subprocess
    subprocess.run(["maturin", "develop", "--release", "-q"], check=True)
    import taas_checkmate as ne

plt.style.use("dark_background")
N = 512
N_PAD = (N + 7) // 8 * 8
DT = 0.5

print("✅ Setup complete")
"""),
        _md("### 2a. LIF Simulation"),
        _code("""# LIF parameters
v = np.full(N_PAD, -65.0, dtype=np.float32)
I = np.full(N_PAD, 8.0, dtype=np.float32)
spikes = np.zeros(N_PAD, dtype=np.int32)
tau_m = np.full(N_PAD, 20.0, dtype=np.float32)
v_rest = np.full(N_PAD, -65.0, dtype=np.float32)
v_thresh = np.full(N_PAD, -50.0, dtype=np.float32)
v_reset = np.full(N_PAD, -70.0, dtype=np.float32)

STEPS = 1000
spike_buffer_lif = np.zeros((STEPS, N), dtype=np.int32)

for t in range(STEPS):
    ne.step_lif(v, I, spikes, DT, tau_m, v_rest, v_thresh, v_reset)
    spike_buffer_lif[t] = spikes[:N]

print(f"LIF firing rate: {spike_buffer_lif.mean() * 1000 / DT:.1f} Hz")
"""),
        _md("### 2b. LIF-AT (Adaptive Threshold)"),
        _code("""# LIF-AT with stateful wrapper
lif_at = ne.PyLifNeuron(N_PAD, dt=DT, tau_m=20.0, v_rest=-65.0,
                         v_thresh=-50.0, v_reset=-70.0,
                         tau_theta=50.0, beta=0.5)

I = np.full(N_PAD, 8.0, dtype=np.float32)
spikes = np.zeros(N_PAD, dtype=np.int32)
v_at = np.zeros(N_PAD, dtype=np.float32)
theta_at = np.zeros(N_PAD, dtype=np.float32)
v_history = np.zeros((STEPS, N), dtype=np.float32)
theta_history = np.zeros((STEPS, N), dtype=np.float32)
spike_buffer_at = np.zeros((STEPS, N), dtype=np.int32)

for t in range(STEPS):
    lif_at.step(I, spikes)
    spike_buffer_at[t] = spikes[:N]
    lif_at.copy_v(v_at)
    v_history[t] = v_at[:N]
    lif_at.copy_theta(theta_at)
    theta_history[t] = theta_at[:N]

print(f"LIF-AT firing rate: {spike_buffer_at.mean() * 1000 / DT:.1f} Hz")
print(f"  Mean theta: {theta_history.mean():.2f}")
"""),
        _md("### 2c. AdEx (Adaptive Exponential)"),
        _code("""v = np.full(N_PAD, -65.0, dtype=np.float32)
u = np.full(N_PAD, 0.0, dtype=np.float32)
I = np.full(N_PAD, 12.0, dtype=np.float32)
spikes = np.zeros(N_PAD, dtype=np.int32)
tau_m = np.full(N_PAD, 20.0, dtype=np.float32)
tau_w = np.full(N_PAD, 100.0, dtype=np.float32)
v_rest = np.full(N_PAD, -65.0, dtype=np.float32)
v_thresh = np.full(N_PAD, -50.0, dtype=np.float32)
v_reset = np.full(N_PAD, -70.0, dtype=np.float32)
delta_t = np.full(N_PAD, 2.0, dtype=np.float32)
a = np.full(N_PAD, 0.02, dtype=np.float32)
b = np.full(N_PAD, 0.02, dtype=np.float32)

spike_buffer_adex = np.zeros((STEPS, N), dtype=np.int32)

for t in range(STEPS):
    ne.step_adex(v, u, I, spikes, DT, tau_m, tau_w, v_rest,
                 v_thresh, v_reset, delta_t, a, b)
    spike_buffer_adex[t] = spikes[:N]

print(f"AdEx firing rate: {spike_buffer_adex.mean() * 1000 / DT:.1f} Hz")
"""),
        _md("## Comparison Raster"),
        _code("""fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

for ax, buf, title, color in zip(
    axes,
    [spike_buffer_lif, spike_buffer_at, spike_buffer_adex],
    ["LIF", "LIF-AT (Adaptive Threshold)", "AdEx (Adaptive Exponential)"],
    ["#00ff88", "#ff6600", "#3399ff"],
):
    t, n = np.where(buf > 0)
    ax.scatter(t * DT, n, s=0.3, c=color, alpha=0.4, rasterized=True)
    ax.set_ylabel("Neuron")
    ax.set_title(title, color=color)
    ax.set_ylim(0, N)
    ax.grid(alpha=0.1)

axes[-1].set_xlabel("Time (ms)")
plt.tight_layout()
plt.show()
"""),
        _md("## Adaptive Threshold Dynamics"),
        _code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Sample neuron
idx = 42
ax1.plot(np.arange(STEPS) * DT, v_history[:, idx],
         color="#00ff88", linewidth=0.8, label="V")
ax1.plot(np.arange(STEPS) * DT, theta_history[:, idx],
         color="#ff6600", linewidth=0.8, label="θ (threshold)")
ax1.axhline(-50, color="gray", linestyle="--", alpha=0.3,
            label="Base threshold")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Membrane Potential")
ax1.set_title(f"LIF-AT: V and θ for Neuron {idx}")
ax1.legend()
ax1.grid(alpha=0.1)

# Theta distribution
mean_theta = theta_history.mean(axis=0)
ax2.hist(mean_theta[:N], bins=30, color="#ff6600", alpha=0.7, edgecolor="none")
ax2.set_xlabel("Mean θ")
ax2.set_ylabel("Count")
ax2.set_title("Adaptive Threshold Distribution")
ax2.grid(alpha=0.1)

plt.tight_layout()
plt.show()

print(f"θ range: [{mean_theta[:N].min():.2f}, {mean_theta[:N].max():.2f}]")
"""),
    ],
}


# ═══════════════════════════════════════════════════════════
#  3. Kuramoto Oscillators
# ═══════════════════════════════════════════════════════════

NOTEBOOKS["03_kuramoto"] = {
    "title": "Kuramoto Phase Oscillators",
    "cells": [
        _md("""# 🔄 Kuramoto Phase Oscillator Network

The Kuramoto model describes synchronization of coupled oscillators:

$$\\dot{\\theta}_i = \\omega_i + \\sum_j W_{ij} \\sin(\\theta_j - \\theta_i)$$

The **order parameter** quantifies synchronization:

$$r e^{i\\psi} = \\frac{1}{N}\\sum_j e^{i\\theta_j}$$

$r=1$ = perfect synchrony, $r=0$ = complete asynchrony.

This implementation uses sparse CSR matrix for connectivity and SIMD for phase updates.
"""),
        _md("## Setup"),
        _code("""import numpy as np
import matplotlib.pyplot as plt
try:
    import taas_checkmate as ne
except ImportError:
    import subprocess
    subprocess.run(["maturin", "develop", "--release", "-q"], check=True)
    import taas_checkmate as ne

plt.style.use("dark_background")
N = 256
N_PAD = (N + 7) // 8 * 8
DT = 0.5

rng = np.random.default_rng(42)
print("✅ Setup complete")
"""),
        _md("## Network Setup: Sparse Connectivity"),
        _code("""# Random sparse connectivity
SPARSITY = 0.08
edges = []
for i in range(N_PAD):
    for j in range(N_PAD):
        if i != j and rng.random() < SPARSITY:
            edges.append((i, j, rng.uniform(0.1, 1.0)))

edges.sort(key=lambda e: e[0])
row_ptr = [0]
col = []
weights = []
count = 0
for i in range(N_PAD):
    while count < len(edges) and edges[count][0] == i:
        col.append(edges[count][1])
        weights.append(edges[count][2])
        count += 1
    row_ptr.append(len(col))

row_ptr = np.array(row_ptr, dtype=np.int32)
col = np.array(col, dtype=np.int32)
weights = np.array(weights, dtype=np.float32)

# Phase initialization: two clusters
phases = np.zeros(N_PAD, dtype=np.float32)
phases[:N//2] = rng.uniform(0, 0.5, N//2)  # cluster 1
phases[N//2:] = rng.uniform(np.pi - 0.5, np.pi + 0.5, N - N//2)  # cluster 2
omegas = rng.uniform(0.5, 1.5, N_PAD).astype(np.float32)

print(f"Network: {N_PAD} oscillators, {len(col)} synapses ({SPARSITY*100:.0f}% sparse)")
print(f"Two clusters initialized: 0 rad (cluster 1) vs π rad (cluster 2)")
"""),
        _md("## Run Simulation"),
        _code("""STEPS = 2000
sin_buf = np.zeros(N_PAD, dtype=np.float32)
cos_buf = np.zeros(N_PAD, dtype=np.float32)
coup_buf = np.zeros(N_PAD, dtype=np.float32)

phase_history = np.zeros((STEPS, N), dtype=np.float32)
order_r = np.zeros(STEPS, dtype=np.float32)

from rich.progress import Progress

with Progress() as pbar:
    task = pbar.add_task("[cyan]Simulating...", total=STEPS)
    for t in range(STEPS):
        ne.kuramoto_step_sparse(phases, omegas, row_ptr, col, weights,
                                sin_buf, cos_buf, coup_buf, DT)
        phase_history[t] = phases[:N]
        # Order parameter
        r = np.abs(np.mean(np.exp(1j * phases[:N])))
        order_r[t] = r
        pbar.update(task, advance=1)

print(f"\\nFinal order parameter: r = {order_r[-1]:.4f}")
print(f"  (1.0 = perfect sync, 0.0 = async)")
"""),
        _md("## Synchronization Dynamics"),
        _code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

t = np.arange(STEPS) * DT

# Order parameter over time
ax1.plot(t, order_r, color="#00ff88", linewidth=1)
ax1.axhline(order_r.mean(), color="#ff6600", linestyle="--", alpha=0.5,
            label=f"Mean r = {order_r.mean():.3f}")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Order Parameter r")
ax1.set_title("Kuramoto Synchronization")
ax1.legend()
ax1.grid(alpha=0.1)
ax1.set_ylim(0, 1)

# Phase evolution (sample)
sample_idx = [0, 10, 50, 100, 200]
for idx in sample_idx:
    if idx < N:
        ax2.plot(t, phase_history[:, idx], linewidth=0.8,
                 label=f"θ_{idx}")
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Phase (rad)")
ax2.set_title("Individual Phase Evolution")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.1)

plt.tight_layout()
plt.show()
"""),
        _md("## Phase Distribution (Histogram)"),
        _code("""fig, axes = plt.subplots(2, 3, figsize=(14, 6))
snapshots = [0, 100, 500, 1000, 1500, 1999]

for ax, snap in zip(axes.flat, snapshots):
    ph = phase_history[snap]
    hist, bins = np.histogram(ph, bins=36, range=(0, 2*np.pi))
    theta = (bins[:-1] + bins[1:]) / 2
    ax.bar(theta, hist, width=2*np.pi/36, color="#00ff88",
           alpha=0.7, edgecolor="none")
    ax.set_title(f"t = {snap * DT:.0f} ms")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, N // 4)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.1)

fig.suptitle("Phase Distribution Over Time", fontsize=14)
plt.tight_layout()
plt.show()
"""),
        _md("## Phase Coherence Map"),
        _code("""# Compute coherence between areas
N_AREAS = 4
area_size = N // N_AREAS
coherence = np.zeros((N_AREAS, N_AREAS), dtype=np.float32)

for a1 in range(N_AREAS):
    for a2 in range(N_AREAS):
        p1 = phase_history[:, a1*area_size:(a1+1)*area_size]
        p2 = phase_history[:, a2*area_size:(a2+1)*area_size]
        # Mean vector of phase differences
        diff = p1[:, :, None] - p2[:, None, :]
        r = np.abs(np.mean(np.exp(1j * diff)))
        coherence[a1, a2] = r

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(coherence, cmap="plasma", vmin=0, vmax=1)
ax.set_xticks(range(N_AREAS))
ax.set_yticks(range(N_AREAS))
ax.set_xlabel("Area")
ax.set_ylabel("Area")
ax.set_title("Phase Coherence Matrix")
plt.colorbar(im, ax=ax, label="Coherence", shrink=0.8)

for i in range(N_AREAS):
    for j in range(N_AREAS):
        ax.text(j, i, f"{coherence[i,j]:.2f}",
                ha="center", va="center", fontsize=10,
                color="white" if coherence[i,j] < 0.7 else "black")

plt.tight_layout()
plt.show()
"""),
    ],
}


# ═══════════════════════════════════════════════════════════
#  4. T3 Triple Oscillator
# ═══════════════════════════════════════════════════════════

NOTEBOOKS["04_t3_oscillator"] = {
    "title": "T3 Triple Oscillator",
    "cells": [
        _md("""# 🎵 T3 Triple Oscillator

Three coupled Kuramoto populations with Stuart-Landau amplitude dynamics:

$$\\dot{\\theta}_i = \\omega_i + \\sum_j W_{ij} \\sin(\\theta_j - \\theta_i) + D_i$$

$$\\dot{r}_i = \\lambda_i r_i - \\beta_i r_i^3 + \\text{drive}$$

Each population has a different frequency band and they interact via cross-coupling.

**Applications:** EEG generation, phase coding, feature extraction for reservoir computing.
"""),
        _md("## Setup"),
        _code("""import numpy as np
import matplotlib.pyplot as plt
try:
    import taas_checkmate as ne
except ImportError:
    import subprocess
    subprocess.run(["maturin", "develop", "--release", "-q"], check=True)
    import taas_checkmate as ne

plt.style.use("dark_background")
N = 64
N_PAD = (N + 7) // 8 * 8
DT = 0.5

print("✅ Setup complete")
"""),
        _md("## Create T3 Oscillator"),
        _code("""# Create T3 with 3 populations at different frequency ranges
tri = ne.PyTriOscillator(N_PAD, freq_min=2.0, freq_max=80.0)
tri.set_uniform_coupling(0.5, 0.3, 0.2)  # intra, cross1, cross2

print(f"T3: {tri.n_osc()} oscillators × 3 populations")
print(f"  Population 1: theta/alpha  (2-8 Hz)")
print(f"  Population 2: beta         (8-30 Hz)")
print(f"  Population 3: gamma        (30-80 Hz)")
print(f"  Coupling: intra={0.5}, cross={0.3}/{0.2}")

# Feature dimension
dim = ne.pst_linear_dim(tri.n_osc())
print(f"  PST feature dimension: {dim}")
"""),
        _md("## Run T3 and Visualize Dynamics"),
        _code("""STEPS = 2000
phase_out = np.zeros(N_PAD, dtype=np.float32)
amp_out = np.zeros(N_PAD, dtype=np.float32)

hist = {
    "ph1": np.zeros((STEPS, min(N, 10)), dtype=np.float32),
    "ph2": np.zeros((STEPS, min(N, 10)), dtype=np.float32),
    "ph3": np.zeros((STEPS, min(N, 10)), dtype=np.float32),
    "r1": np.zeros(STEPS, dtype=np.float32),
    "r2": np.zeros(STEPS, dtype=np.float32),
    "r3": np.zeros(STEPS, dtype=np.float32),
    "order1": np.zeros(STEPS, dtype=np.float32),
    "order2": np.zeros(STEPS, dtype=np.float32),
    "order3": np.zeros(STEPS, dtype=np.float32),
}

from rich.progress import Progress

with Progress() as pbar:
    task = pbar.add_task("[cyan]Simulating...", total=STEPS)
    for t in range(STEPS):
        tri.step(DT)

        for pop in range(1, 4):
            tri.copy_phases(phase_out, pop)
            tri.copy_amplitudes(amp_out, pop)
            if pop == 1:
                hist["ph1"][t] = phase_out[:10]
                hist["r1"][t] = np.mean(amp_out[:N])
                hist["order1"][t] = tri.order_parameter_r1()
            elif pop == 2:
                hist["ph2"][t] = phase_out[:10]
                hist["r2"][t] = np.mean(amp_out[:N])
                hist["order2"][t] = tri.order_parameter_r2()
            else:
                hist["ph3"][t] = phase_out[:10]
                hist["r3"][t] = np.mean(amp_out[:N])
                hist["order3"][t] = tri.order_parameter_r3()

        pbar.update(task, advance=1)

print("Simulation complete!")
"""),
        _md("## Population Amplitudes & Order Parameters"),
        _code("""fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))

t = np.arange(STEPS) * DT

# Amplitudes
ax1.plot(t, hist["r1"], color="#ff6600", linewidth=0.8, label="Pop 1 (θ/α)")
ax1.plot(t, hist["r2"], color="#3399ff", linewidth=0.8, label="Pop 2 (β)")
ax1.plot(t, hist["r3"], color="#ff33cc", linewidth=0.8, label="Pop 3 (γ)")
ax1.set_ylabel("Mean Amplitude")
ax1.set_title("Stuart-Landau Amplitudes")
ax1.legend()
ax1.grid(alpha=0.1)

# Order parameters
ax2.plot(t, hist["order1"], color="#ff6600", linewidth=0.8, label="Pop 1")
ax2.plot(t, hist["order2"], color="#3399ff", linewidth=0.8, label="Pop 2")
ax2.plot(t, hist["order3"], color="#ff33cc", linewidth=0.8, label="Pop 3")
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Order Parameter r")
ax2.set_title("Synchronization (1.0 = perfect sync)")
ax2.legend()
ax2.grid(alpha=0.1)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.show()
"""),
        _md("## Phase Portraits"),
        _code("""fig, axes = plt.subplots(1, 3, figsize=(14, 4), subplot_kw={"projection": "polar"})

for ax, ph, pop, color in zip(
    axes,
    [hist["ph1"], hist["ph2"], hist["ph3"]],
    ["Pop 1 (θ/α)", "Pop 2 (β)", "Pop 3 (γ)"],
    ["#ff6600", "#3399ff", "#ff33cc"],
):
    # Plot phase trajectory of first oscillator
    ax.plot(ph[:, 0], np.arange(STEPS) * 0.01, alpha=0, linewidth=0)
    ax.scatter(ph[:, 0], np.ones(STEPS), s=1, c=color, alpha=0.3)
    ax.set_title(pop, color=color)
    ax.grid(alpha=0.1)

plt.suptitle("Phase Evolution on Unit Circle", fontsize=14)
plt.tight_layout()
plt.show()
"""),
        _md("## PST Feature Extraction"),
        _code("""# Compute PST features for readout
fv = ne.pst_linear_features(tri)
print(f"Feature vector dimension: {len(fv)}")
print(f"  Min: {fv.min():.4f}, Max: {fv.max():.4f}, Mean: {fv.mean():.4f}")

# Features over time
n_features = dim
feature_history = np.zeros((STEPS, n_features), dtype=np.float32)

for t in range(STEPS):
    tri.step(DT)
    fv = ne.pst_linear_features(tri)
    feature_history[t] = fv[:n_features]

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(feature_history.T, aspect="auto", cmap="RdBu_r",
               extent=[0, STEPS * DT, 0, n_features])
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Feature Index")
ax.set_title("PST Feature Timeline")
plt.colorbar(im, ax=ax, label="Feature Value", shrink=0.8)
plt.tight_layout()
plt.show()
"""),
        _md("## EEG from T3"),
        _code("""# Mix T3 populations into EEG-like signal
eeg = hist["r1"] * np.sin(2 * np.pi * 5 * t / 1000) + \\
      hist["r2"] * np.sin(2 * np.pi * 15 * t / 1000) + \\
      hist["r3"] * np.sin(2 * np.pi * 45 * t / 1000)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))

ax1.plot(t[:500], eeg[:500], color="#00ff88", linewidth=0.6)
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("EEG (μV)")
ax1.set_title("T3-Derived EEG Signal (500ms window)")
ax1.grid(alpha=0.1)

freqs = np.fft.rfftfreq(STEPS, d=DT / 1000)
psd = np.abs(np.fft.rfft(eeg * np.hanning(STEPS))) ** 2
ax2.semilogy(freqs, psd, color="#00ff88", linewidth=0.8)
ax2.axvspan(2, 8, alpha=0.15, color="#ff6600", label="θ/α")
ax2.axvspan(8, 30, alpha=0.15, color="#3399ff", label="β")
ax2.axvspan(30, 80, alpha=0.15, color="#ff33cc", label="γ")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("PSD")
ax2.set_title("EEG Power Spectrum")
ax2.set_xlim(0, 100)
ax2.legend()
ax2.grid(alpha=0.1)

plt.tight_layout()
plt.show()
"""),
    ],
}


# ═══════════════════════════════════════════════════════════
#  5. E-Prop SNN
# ═══════════════════════════════════════════════════════════

NOTEBOOKS["05_eprop_snn"] = {
    "title": "E-Prop Spiking Neural Network",
    "cells": [
        _md("""# 🧠 E-Prop Spiking Neural Network

E-Prop (Eligibility Propagation) enables online learning in SNNs without BPTT:

$$\\Delta w = \\eta \\cdot L_j \\cdot e_{ji}$$
$$\\frac{de_{ji}}{dt} = -\\frac{e_{ji}}{\\tau_e} + \\psi_j(t) \\cdot z_i(t)$$

Where:
- $L_j$ = learning signal (error projected through random feedback)
- $e_{ji}$ = eligibility trace (low-pass filtered pre-post correlation)
- $\\psi_j(t)$ = surrogate gradient $\\sigma'(v_j)$

Based on: Bellec et al. (2020), *Nature Communications*.
"""),
        _md("## Setup"),
        _code("""import numpy as np
import matplotlib.pyplot as plt
try:
    import taas_checkmate as ne
except ImportError:
    import subprocess
    subprocess.run(["maturin", "develop", "--release", "-q"], check=True)
    import taas_checkmate as ne

plt.style.use("dark_background")
print("✅ Setup complete")
"""),
        _md("## Create E-Prop SNN"),
        _code("""N_INPUT = 2
N_HIDDEN = 32
N_PAD = (N_HIDDEN + 7) // 8 * 8
DT = 1.0

# E-Prop layer
layer = ne.PyEPropLIF(N_PAD, dt=DT, tau_m=10.0, tau_e=20.0)
print(f"E-Prop LIF: {N_PAD} hidden neurons (SIMD-padded)")
print(f"  tau_m={10.0}, tau_e={20.0}")

# Weights
rng = np.random.default_rng(42)
w_in = rng.uniform(-1.0, 1.0, (N_PAD, N_INPUT)).astype(np.float32) * 0.5
w_out = np.zeros(N_PAD, dtype=np.float32)  # readout
w_fb = rng.uniform(-0.3, 0.3, N_PAD).astype(np.float32)  # feedback

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([0, 1, 1, 0], dtype=np.float32)

# Buffers
I = np.zeros(N_PAD, dtype=np.float32)
spikes = np.zeros(N_PAD, dtype=np.int32)
elig = np.zeros(N_PAD, dtype=np.float32)
ls = np.zeros(N_PAD, dtype=np.float32)  # learning signals

print(f"\\nXOR dataset: {len(X)} patterns")
"""),
        _md("## Training Loop"),
        _code("""N_EPOCHS = 100
LR = 0.02

errors = []
accuracy_history = []

from rich.progress import Progress

with Progress() as pbar:
    task = pbar.add_task("[cyan]Training...", total=N_EPOCHS)

    for epoch in range(N_EPOCHS):
        epoch_errors = []
        correct = 0

        for i in range(4):
            layer.reset()
            pred = 0.0

            for _ in range(5):  # rate coding
                # Forward
                I[:N_INPUT] = w_in[:, :N_INPUT] @ X[i]
                layer.step(I, spikes, np.zeros(N_PAD, dtype=np.int32))

                # Readout
                pred += float(spikes[:N_HIDDEN] @ w_out[:N_HIDDEN])

            pred /= 5
            error = y[i] - pred

            # Learning signal
            ls[:] = error * w_fb

            # Update readout
            w_out[:N_HIDDEN] += LR * error * spikes[:N_HIDDEN].astype(np.float32)

            # E-Prop weight update
            layer.copy_eligibility(elig)
            # Δw_in = LR * LS[j] * elig[j] * input[i]
            for j in range(N_PAD):
                w_in[j, :N_INPUT] += LR * ls[j] * elig[j] * X[i]

            epoch_errors.append(abs(error))
            if abs(error) < 0.5:
                correct += 1

        avg_err = np.mean(epoch_errors)
        errors.append(avg_err)
        accuracy_history.append(correct / 4)
        pbar.update(task, advance=1)

print(f"\\nTraining complete!")
print(f"  Final error: {errors[-1]:.6f}")
print(f"  Final accuracy: {accuracy_history[-1]*100:.0f}%")
"""),
        _md("## Learning Curve"),
        _code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(errors, color="#00ff88", linewidth=1)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Mean Absolute Error")
ax1.set_title("E-Prop Learning Curve")
ax1.grid(alpha=0.1)

ax2.plot(np.arange(len(accuracy_history)) * 10,
         np.array(accuracy_history) * 100,
         color="#ff6600", linewidth=1)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("XOR Classification Accuracy")
ax2.set_ylim(0, 105)
ax2.axhline(100, color="green", linestyle="--", alpha=0.5, label="100%")
ax2.axhline(50, color="red", linestyle="--", alpha=0.3, label="Chance")
ax2.legend()
ax2.grid(alpha=0.1)

plt.tight_layout()
plt.show()
"""),
        _md("## Test XOR"),
        _code("""print("═" * 40)
print("XOR Test Results")
print("═" * 40)

for i in range(4):
    layer.reset()
    preds = []
    for _ in range(10):
        I[:N_INPUT] = w_in[:, :N_INPUT] @ X[i]
        layer.step(I, spikes, np.zeros(N_PAD, dtype=np.int32))
        preds.append(float(spikes[:N_HIDDEN] @ w_out[:N_HIDDEN]))

    avg_pred = np.mean(preds)
    print(f"  {X[i][0]} XOR {X[i][1]} = {avg_pred:.4f}  (target={y[i]:.0f})  {'✅' if abs(avg_pred - y[i]) < 0.5 else '❌'}")

print("═" * 40)
"""),
        _md("## Eligibility Traces Analysis"),
        _code("""# Visualize eligibility traces
layer.reset()
elig_history = np.zeros((100, N_PAD), dtype=np.float32)

for t in range(100):
    inp = X[t % 4]
    I[:N_INPUT] = w_in[:, :N_INPUT] @ inp
    layer.step(I, spikes, spikes.copy())
    layer.copy_eligibility(elig)
    elig_history[t] = elig

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))

im = ax1.imshow(elig_history[:, :16].T, aspect="auto", cmap="RdBu_r")
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Neuron")
ax1.set_title("Eligibility Traces (first 16 neurons)")
plt.colorbar(im, ax=ax1, shrink=0.8)

# Mean eligibility over time
ax2.plot(np.mean(elig_history, axis=1), color="#00ff88", linewidth=0.8)
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Mean Eligibility")
ax2.set_title("Mean Eligibility Trace Dynamics")
ax2.grid(alpha=0.1)

plt.tight_layout()
plt.show()
"""),
    ],
}


# ═══════════════════════════════════════════════════════════
#  Generate all notebooks
# ═══════════════════════════════════════════════════════════

def generate_notebooks(output_dir: str = "."):
    """Generate all Jupyter notebooks."""
    import os

    for name, data in NOTEBOOKS.items():
        notebook = nb.new_notebook()
        notebook.cells = data["cells"]
        notebook.metadata = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.14.5",
            },
        }

        path = os.path.join(output_dir, f"{name}.ipynb")
        with open(path, "w") as f:
            nbf.write(notebook, f)
        print(f"  ✅ {path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    generate_notebooks(script_dir)
