#!/usr/bin/env python3
"""Test VNC Cluster 1 oscillation — do reciprocal IN pairs oscillate in anti-phase?"""
import sys
import numpy as np
sys.path.insert(0, '.')
from fly.scripts.vnc_cpg import VNCCPGBrain

vnc = VNCCPGBrain('/tmp/vnc_cluster1.bin', base_rate=80.0)

# Top reciprocal pair
pair_a, pair_b = 47, 54

print(f'Testing oscillation: IN[{pair_a}] ↔ IN[{pair_b}]')
print(f'  Type A: {vnc.type_map.get(pair_a, "?")}  Class: {vnc.class_map.get(pair_a, "?")}')
print(f'  Type B: {vnc.type_map.get(pair_b, "?")}  Class: {vnc.class_map.get(pair_b, "?")}')

# Warm up 1s
print('\nWarming up 1s...')
for i in range(40):
    vnc.inject_dn(1.0 * np.ones(len(vnc.dn_indices)))
    vnc.step_batch(i * 0.025)

# Record 2s of spikes for the pair
print('Recording 2s of activity...')
n_batches = 80
spikes_a = np.zeros(n_batches, dtype=np.int32)
spikes_b = np.zeros(n_batches, dtype=np.int32)

for batch in range(n_batches):
    vnc.inject_dn(1.0 * np.ones(len(vnc.dn_indices)))
    vnc.step_batch(batch * 0.025)
    spike_buf = vnc._last_spike_buf
    inner = vnc._last_n_steps

    sa = sb = 0
    for step in range(inner):
        offset = step * vnc.n
        if spike_buf[offset + pair_a]: sa += 1
        if spike_buf[offset + pair_b]: sb += 1
    spikes_a[batch] = sa
    spikes_b[batch] = sb

# Compute cross-correlation
def xcorr(x, y, max_lag=20):
    n = len(x)
    result = np.zeros(2 * max_lag + 1)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            s = np.dot(x[:n+lag], y[-lag:])
        elif lag > 0:
            s = np.dot(x[lag:], y[:n-lag])
        else:
            s = np.dot(x, y)
        result[lag + max_lag] = s
    return result

xc = xcorr(spikes_a.astype(float), spikes_b.astype(float), max_lag=20)
peak_lag = np.argmax(xc) - 20

print(f'\nSpike counts:')
print(f'  IN[{pair_a}]: {spikes_a.sum()} total, mean {spikes_a.mean():.2f} per batch')
print(f'  IN[{pair_b}]: {spikes_b.sum()} total, mean {spikes_b.mean():.2f} per batch')
print(f'  Cross-correlation peak lag: {peak_lag} batches')

# Show raster
print('\nSpike raster (first 40 batches):')
print(f'  Batch  IN[{pair_a}]  IN[{pair_b}]')
for i in range(0, 40, 5):
    a_str = ''.join('█' if spikes_a[j] > 0 else '·' for j in range(i, min(i+5, 40)))
    b_str = ''.join('█' if spikes_b[j] > 0 else '·' for j in range(i, min(i+5, 40)))
    print(f'  {i:3d}-{i+4:2d}  {a_str}  {b_str}')

# Phase difference
phase_diff = np.abs((spikes_a > 0).astype(int) - (spikes_b > 0).astype(int)).sum() / n_batches
print(f'\nPhase difference index: {phase_diff:.3f} (0=same, 1=always opposite)')
if phase_diff > 0.4:
    print('  >>> ANTI-PHASE! Oscillation detected! <<<')
else:
    print('  >>> In-phase or random <<<')

# Also check more pairs
print('\n\n--- Extended: Top 10 reciprocal pairs oscillation ---')
for idx, (a, b, wa, wb, gm) in enumerate(vnc._top_recip_pairs[:10]):
    sa = spikes_a  # reuse last recording
    print(f'  Pair {idx+1}: IN[{a}] ↔ IN[{b}]  gm={gm:.0f}')
