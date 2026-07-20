#!/usr/bin/env python3
"""Realistic fly brain HH viz — raster + stats."""
import struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sg
import sys

print("── Loading hh_real data ──", flush=True)
neurons = np.genfromtxt('hh_real_neurons.csv', delimiter=',', skip_header=1,
                        dtype=[('i', 'i4'), ('first', 'i4'), ('spikes', 'i4'),
                               ('exc', 'i1'), ('pace', 'i1'),
                               ('g_leak', 'f4'), ('c_m', 'f4'), ('e_leak', 'f4')])
n = len(neurons)
n_pace = neurons['pace'].sum()
n_exc = neurons['exc'].sum()
n_inh = n - n_exc
total_steps = 6000
dt = 0.01
time_ms = np.arange(total_steps) * dt

# Read binary spike data
spike_raster = np.zeros((total_steps, n), dtype=np.uint8)
with open('hh_real.dat', 'rb') as f:
    for step in range(total_steps):
        cnt = struct.unpack('I', f.read(4))[0]
        for _ in range(cnt):
            idx = struct.unpack('I', f.read(4))[0]
            spike_raster[step, idx] = 1
        if step % 1000 == 0:
            print(f"  step {step}/{total_steps}", flush=True)

total = spike_raster.sum()
print(f"  Total: {total}", flush=True)

# Sort neurons by first-spike time (-1 → end)
order = np.argsort(np.where(neurons['first'] >= 0, neurons['first'], 999999))
spike_sorted = spike_raster[:, order]

# ─── FIG 1: FULL RASTER (first 50ms, 3 panels: scatter, density, heat) ───
print("── Figure 1: raster ──", flush=True)
fig, axes = plt.subplots(3, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [2, 1, 1]})

# Build sparse spike plot (sample 20K points)
times_vec = []; ids_vec = []
for step in range(total_steps):
    for idx in np.where(spike_raster[step] > 0)[0]:
        times_vec.append(step * dt)
        ids_vec.append(idx)
times_arr = np.array(times_vec)
ids_arr = np.array(ids_vec)
step_sz = max(1, len(times_arr) // 20000)

axes[0].scatter(times_arr[::step_sz], ids_arr[::step_sz],
                s=0.005, c='black', alpha=0.15, rasterized=True)
axes[0].set_xlim(0, 60)
axes[0].set_ylabel('Neuron index')
axes[0].set_title(f'FAFB HH real — {total} spikes, E={n_exc} I={n_inh}')

# Density (10-step bins)
density = spike_raster.sum(axis=1)
window = 100
kernel = np.ones(window) / window
dsmooth = np.convolve(density, kernel, mode='same')
axes[1].plot(time_ms, density, alpha=0.15, color='gray', linewidth=0.3)
axes[1].plot(time_ms, dsmooth, color='black', linewidth=1)
axes[1].set_xlim(0, 60)
axes[1].set_ylabel('Spikes/step')

# Heatmap (200×200 bins)
heat = np.zeros((200, 200))
t_chunk = total_steps // 200
n_chunk = n // 200
for ti in range(200):
    ts = ti * t_chunk
    te = (ti + 1) * t_chunk
    for ni in range(200):
        ns = ni * n_chunk
        ne = (ni + 1) * n_chunk
        tile = spike_sorted[ts:te, ns:ne]
        heat[ti, ni] = tile.sum() / max(tile.size, 1)
heat = heat.T
vmax = np.percentile(heat[heat > 0], 95) or 0.01
im = axes[2].imshow(heat, aspect='auto', origin='lower',
                     extent=[0, 60, 0, n], cmap='inferno', vmin=0, vmax=vmax)
axes[2].set_xlabel('Time (ms)')
axes[2].set_ylabel('Neuron (sorted)')
plt.colorbar(im, ax=axes[2], label='Spike density')
plt.tight_layout()
plt.savefig('fly_real_raster.png', dpi=150)
print("  → fly_real_raster.png", flush=True)

# ─── FIG 2: SAMPLE 500 NEURONS ZOOM ───
print("── Figure 2: zoom ──", flush=True)
fig, ax = plt.subplots(figsize=(16, 6))
n_show = min(500, n)
# Color by exc/inh
for step in range(total_steps):
    for idx in np.where(spike_raster[step, :n_show] > 0)[0]:
        c = 'red' if neurons['exc'][idx] else 'blue'
        ax.plot(step * dt, idx, '.', color=c, alpha=0.3, markersize=1)
ax.set_xlim(0, 60)
ax.set_ylim(-5, n_show + 5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Neuron index')
ax.set_title(f'First {n_show} neurons — red=exc, blue=inh')
plt.tight_layout()
plt.savefig('fly_real_zoom.png', dpi=150)
print("  → fly_real_zoom.png", flush=True)

# ─── FIG 3: POPULATION RATE & FFT ───
print("── Figure 3: spectrum ──", flush=True)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# Population rate
rate_hz = density / n / dt * 1000  # Hz per neuron
ax1.plot(time_ms, rate_hz, 'k-', linewidth=0.5)
ax1.set_xlim(0, 60)
ax1.set_ylabel('Population rate (Hz)')
ax1.set_xlabel('Time (ms)')
ax1.set_title('Mean firing rate')

# Spectrum
fs = 100
freqs, psd = sg.welch(density, fs=fs, nperseg=512)
ax2.semilogy(freqs[:100] * 1000, psd[:100], 'k-', linewidth=1)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('PSD')
ax2.set_xlim(0, 500)
pk = np.argmax(psd[5:100]) + 5
ax2.axvline(freqs[pk] * 1000, color='r', ls='--', alpha=0.5,
            label=f'peak={freqs[pk]*1000:.0f} Hz')
ax2.legend()
plt.tight_layout()
plt.savefig('fly_real_rate.png', dpi=150)
print("  → fly_real_rate.png", flush=True)

# ─── FIG 4: ISI HISTOGRAM ───
print("── Figure 4: ISI ──", flush=True)
all_isis = []
for ni in range(min(2000, n)):
    steps = np.where(spike_raster[:, ni] > 0)[0]
    if len(steps) > 1:
        all_isis.extend(np.diff(steps) * dt)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(all_isis, bins=80, color='black', alpha=0.6, density=True)
ax.set_xlabel('ISI (ms)')
ax.set_ylabel('Density')
ax.set_title(f'ISI distribution (n={len(all_isis)} intervals)')
if all_isis:
    mi = np.mean(all_isis); cv = np.std(all_isis) / mi
    ax.axvline(mi, color='r', ls='--', label=f'mean={mi:.1f}ms, CV={cv:.2f}')
    ax.legend()
plt.tight_layout()
plt.savefig('fly_real_isi.png', dpi=150)
print("  → fly_real_isi.png", flush=True)

# ─── FIG 5: PACEMAKER ACTIVITY ───
print("── Figure 5: pacemaker ──", flush=True)
fig, ax = plt.subplots(figsize=(14, 5))
n_show2 = min(3000, n)
pace_idx = np.where(neurons['pace'] > 0)[0]
pace_set = set(pace_idx)
exc_colors = {True: 'red', False: 'blue'}
for step in range(total_steps):
    for idx in np.where(spike_raster[step, :n_show2] > 0)[0]:
        c = 'red' if idx in pace_set else ('gray' if neurons['exc'][idx] else 'lightblue')
        ax.plot(step * dt, idx, '.', color=c, alpha=0.2 if idx not in pace_set else 0.8,
                markersize=2 if idx in pace_set else 0.5)
ax.set_xlim(0, 60)
ax.set_ylim(-5, n_show2 + 5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Neuron index')
ax.set_title(f'Pacemakers (red, n={n_pace}) drive the network')
plt.tight_layout()
plt.savefig('fly_real_pace.png', dpi=150)
print("  → fly_real_pace.png", flush=True)

print("\n── Done ──", flush=True)
