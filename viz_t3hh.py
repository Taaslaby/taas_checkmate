#!/usr/bin/env python3
"""T3 BCI × HH/FAFB closed-loop visualization."""
import struct, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sg

dt = 0.01
total_steps = 6000

print("── Loading ──")
spike_raster = np.zeros((total_steps, 139189), dtype=np.uint8)
with open('t3hh.dat', 'rb') as f:
    for s in range(total_steps):
        cnt = struct.unpack('I', f.read(4))[0]
        for _ in range(cnt): spike_raster[s, struct.unpack('I', f.read(4))[0]] = 1
        if s % 1000 == 0: print(f"  step {s}/{total_steps}")
total = spike_raster.sum()
print(f"  Total spikes: {total}")

neurons = np.genfromtxt('t3hh_neurons.csv', delimiter=',', skip_header=1,
                        dtype=[('i','i4'),('spikes','i4'),('group','i4')])
order = np.argsort(neurons['group'] * 100000 + neurons['i'])
spike_sorted = spike_raster[:, order]

t3data = np.genfromtxt('t3hh_t3.csv', delimiter=',', skip_header=1,
                        dtype=[('step','i4'),('osc','i4'),('th1','f4'),('rate','f4')])
n_t3 = 50
th1_mat = t3data['th1'].reshape(total_steps, n_t3)
rate_mat = t3data['rate'].reshape(total_steps, n_t3)
time_ms = np.arange(total_steps) * dt

# ─── FIG 1: Super-panel ───
print("── Figure 1: super-panel ──")
fig, axes = plt.subplots(4, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [3, 1, 1, 1]})

# A: Full raster
times, ids = [], []
for s in range(total_steps):
    for idx in np.where(spike_raster[s] > 0)[0]:
        times.append(s * dt); ids.append(idx)
step_sz = max(1, len(times) // 25000)
axes[0].scatter(times[::step_sz], ids[::step_sz], s=0.005, c='black', alpha=0.1, rasterized=True)
axes[0].set_xlim(0, 60); axes[0].set_ylabel('Neuron')

# B: Group rates (heatmap of T3 groups 0-49 → first 50 bins)
heat_grp = np.zeros((200, n_t3))
tbin = total_steps // 200
for ti in range(200):
    for k in range(n_t3):
        heat_grp[ti, k] = rate_mat[ti*tbin:(ti+1)*tbin, k].mean()
im = axes[1].imshow(heat_grp.T, aspect='auto', origin='lower',
                     extent=[0, 60, 0, n_t3], cmap='inferno')
axes[1].set_ylabel('T3 group'); axes[1].set_xlabel('Time (ms)')
plt.colorbar(im, ax=axes[1], label='Rate (Hz)')

# C: T3 phases
for k in range(0, n_t3, 10):
    axes[2].plot(time_ms, np.unwrap(th1_mat[:, k]), label=f'osc{k}', alpha=0.7)
axes[2].set_ylabel('Phase (rad)'); axes[2].set_xlim(0, 60); axes[2].legend(fontsize=6, ncol=2)

# D: Mean population rate
density = spike_raster.sum(axis=1)
window = 50
ds = np.convolve(density, np.ones(window)/window, mode='same')
axes[3].plot(time_ms, ds, 'k-', linewidth=0.5)
axes[3].set_xlabel('Time (ms)'); axes[3].set_ylabel('Spikes/step'); axes[3].set_xlim(0, 60)

plt.tight_layout()
plt.savefig('t3hh_super.png', dpi=150)
print("  → t3hh_super.png")

# ─── FIG 2: T3-HH correlation ───
print("── Figure 2: T3-HH spectrum ──")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
fs = 100
f, p = sg.welch(density, fs=fs, nperseg=512)
ax1.semilogy(f*1000, p, 'k-', linewidth=1)
ax1.set_xlim(0, 500); ax1.set_ylabel('PSD (spikes)')
pk = np.argmax(p[5:100]) + 5
ax1.axvline(f[pk]*1000, color='r', ls='--', alpha=0.5, label=f'{f[pk]*1000:.0f} Hz')
ax1.legend()

# T3 phase spectrum (mean of all oscillators)
mean_th1 = th1_mat.mean(axis=1)
f2, p2 = sg.welch(np.unwrap(mean_th1), fs=fs, nperseg=512)
ax2.semilogy(f2*1000, p2, 'b-', linewidth=1)
ax2.set_xlim(0, 500); ax2.set_ylabel('PSD (T3 phase)'); ax2.set_xlabel('Frequency (Hz)')
pk2 = np.argmax(p2[5:100]) + 5
ax2.axvline(f2[pk2]*1000, color='r', ls='--', alpha=0.5, label=f'{f2[pk2]*1000:.0f} Hz')
ax2.legend()
plt.tight_layout()
plt.savefig('t3hh_spectrum.png', dpi=150)
print("  → t3hh_spectrum.png")

# ─── FIG 3: T3 phase vs group rate correlation ───
print("── Figure 3: T3-HH cross-correlation ──")
fig, ax = plt.subplots(figsize=(10, 4))
# Cross-correlate each T3 phase with its group rate
xc = np.zeros((n_t3, 201))
for k in range(n_t3):
    x = np.unwrap(th1_mat[:, k])
    y = rate_mat[:, k]
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    xc[k] = np.correlate(x, y, mode='full')[len(x)-100:len(x)+101]
xc_mean = xc.mean(axis=0)
lags = np.arange(-100, 101) * dt
ax.plot(lags, xc_mean, 'k-', linewidth=1.5)
ax.axvline(0, color='r', ls='--', alpha=0.5)
ax.axhline(0, color='gray', ls=':', alpha=0.3)
ax.set_xlabel('Lag (ms)'); ax.set_ylabel('Cross-correlation (T3 phase × group rate)')
ax.set_title('Average T3-HH coupling strength')
ax.set_xlim(-50, 50)
plt.tight_layout()
plt.savefig('t3hh_coupling.png', dpi=150)
print("  → t3hh_coupling.png")

# ─── FIG 4: Data flow diagram (text) ───
print("\n── Summary ──")
print(f"  Total spikes: {total}")
print(f"  T3 oscillators: {n_t3}")
print(f"  Groups: {139189//n_t3} neur/group")
print(f"  Spike rate: {total/(total_steps*dt/1000*139189):.0f} Hz")
print(f"  T3 mean phase rate: {np.diff(np.unwrap(th1_mat.mean(axis=1))).mean()/dt:.0f} rad/s")

# Check coupling
peak_lag = lags[np.argmax(xc_mean)]
peak_val = xc_mean.max()
print(f"  T3→HH peak coupling at lag={peak_lag:.1f}ms, r={peak_val:.3f}")
