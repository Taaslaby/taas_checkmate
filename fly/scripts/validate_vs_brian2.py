#!/usr/bin/env python3
"""
Vалидация: наш LIF engine vs Brian2 reference (Shiu et al. 2024, Nature).

Три теста:
  1. Single neuron — step current (batch calls)
  2. Two neurons — pre→post PSP
  3. Small random network (50 neurons) — Poisson-driven

Параметры — из Brian2 reference model.py:
  τ_m = 20ms, τ_syn = 5ms, τ_ref = 2.2ms, delay = 1.8ms (≈4 steps at dt=0.5ms)
  V_rest = -52mV, V_thresh = -45mV, V_reset = -52mV
  Reset: v = v_rst; g = 0 (g_syn cleared on spike)

Weight scaling:
  Наш total g_syn = weight
  Brian2 total g = w * τ_syn / dt
  To match: w_brian2 = weight * dt / τ_syn = weight * 0.1

Запуск:
  cd /home/taaslab/Проекты/asm_engines/taas_checkmate
  /home/taaslab/Проекты/asm_engines/.venv/bin/python3 fly/scripts/validate_vs_brian2.py
"""

import sys
import time
from pathlib import Path

import numpy as np

try:
    from brian2 import (
        NeuronGroup, Synapses, PoissonInput, SpikeMonitor, StateMonitor, Network,
        mV, ms, second, volt, Hz
    )
except ImportError:
    print("⚠ Brian2 не установлен. pip install brian2")
    sys.exit(1)

PROJ = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ / 'target' / 'debug'))
try:
    import taas_checkmate as tcm
except ImportError:
    print(f"⚠ taas_checkmate не найден. Собери: cd {PROJ} && maturin develop")
    sys.exit(1)

# ═══ Params (Brian2 reference) ═══
V_REST   = -52.0
V_RESET  = -52.0
V_THRESH = -45.0
TAU_M    = 20.0
TAU_SYN  = 5.0
TAU_REF  = 2.2
DT_MS    = 0.5
DELAY_MS = 1.8
DELAY_STEPS = int(round(DELAY_MS / DT_MS))  # 4
SYN_SCALE = DT_MS / TAU_SYN  # 0.1

def our_weight_to_brian2(w):
    """Convert our weight to Brian2 w (mV).
    Our total g_syn = w. Brian2 total g = w_b2 * tau/dt.
    For equal PSP: w_b2 = w * dt/tau = w * 0.1
    """
    return w * SYN_SCALE

def build_csr(edges, n_neurons):
    """Build CSR from list of (pre, post, weight) tuples."""
    row_cnt = np.zeros(n_neurons + 1, dtype=np.int32)
    for pre, _, _ in edges:
        row_cnt[pre + 1] += 1
    row_ptr = np.cumsum(row_cnt, dtype=np.int32).copy()
    n_syn = len(edges)
    col_idx = np.zeros(n_syn, dtype=np.int32)
    vals = np.zeros(n_syn, dtype=np.float32)
    pos_cnt = np.zeros(n_neurons, dtype=np.int32)
    for k, (pre, post, w) in enumerate(edges):
        p = int(row_ptr[pre]) + pos_cnt[pre]
        col_idx[p] = post
        vals[p] = w
        pos_cnt[pre] += 1
    for i in range(n_neurons):
        s, e = int(row_ptr[i]), int(row_ptr[i + 1])
        if e > s:
            order = np.argsort(col_idx[s:e])
            col_idx[s:e] = col_idx[s:e][order]
            vals[s:e] = vals[s:e][order]
    return row_ptr, col_idx, vals

def run_our_lif(row_ptr, col_idx, vals, n_neurons, n_steps, dt_ms,
                v_init, tau_m, v_rest, v_thresh, v_reset, nt_types,
                external_I=None, g_syn_init=None, with_alpha=True):
    """Run our LIF engine in batch mode. Returns spike_matrix [n_steps x n_neurons]."""
    n_pad = max(64, (n_neurons + 7) // 8 * 8)

    v = np.copy(v_init)
    if len(v) < n_pad:
        v_new = np.full(n_pad, v_rest, dtype=np.float32)
        v_new[:len(v)] = v
        v = v_new

    tau_m_arr = np.full(n_pad, tau_m, dtype=np.float32)
    v_rest_arr = np.full(n_pad, v_rest, dtype=np.float32)
    v_thresh_arr = np.full(n_pad, v_thresh, dtype=np.float32)
    v_reset_arr = np.full(n_pad, v_reset, dtype=np.float32)
    nt_types_arr = np.ones(n_pad, dtype=np.float32)
    if nt_types is not None and len(nt_types) > 0:
        nt_types_arr[:min(len(nt_types), n_pad)] = nt_types[:n_pad]

    ext_I = np.zeros(n_pad, dtype=np.float32)
    if external_I is not None:
        ext_I[:min(len(external_I), n_pad)] = external_I[:n_pad]

    I_buf = np.zeros(n_pad, dtype=np.float32)
    spikes = np.zeros(n_pad, dtype=np.int32)
    spike_buf = np.zeros(n_steps * n_neurons, dtype=np.int32)

    g_syn = np.zeros(n_pad, dtype=np.float32)
    if g_syn_init is not None and len(g_syn_init) > 0:
        g_syn[:min(len(g_syn_init), n_pad)] = g_syn_init[:n_pad]

    delay_buf = np.zeros(n_pad * DELAY_STEPS, dtype=np.float32) if with_alpha else np.zeros(0, dtype=np.float32)
    delay_pos = np.zeros(1, dtype=np.int32)

    tau_syn = TAU_SYN if with_alpha else 0.0
    delay_steps = DELAY_STEPS if with_alpha else 0

    tcm.lif_connectome.lif_sim_cpu_recording(
        row_ptr, col_idx, vals,
        n_steps, dt_ms,
        v, tau_m_arr, v_rest_arr, v_thresh_arr, v_reset_arr,
        nt_types_arr, ext_I, I_buf, spikes,
        spike_buf,
        g_syn, tau_syn,
        delay_buf, delay_steps, delay_pos,
    )

    return spike_buf.reshape((n_steps, n_neurons)), v[:n_neurons], g_syn[:n_neurons]


# ═══════════════════════════════════════════════════════════
#  TEST 1: Single neuron — step current (batch calls)
# ═══════════════════════════════════════════════════════════

def test_single_neuron():
    """Подать step current (20 mV, 50ms), сравнить spike times batch-вызовами."""
    print("\n" + "═" * 65)
    print("  TEST 1: Single neuron — step current (batch)")
    print("═" * 65)

    N = 1
    # I=0: steps 0-49, I=20: steps 50-149, I=0: steps 150-199
    segments = [
        (50, 0.0),   # 0..50,  no current
        (100, 20.0), # 50..150, with current
        (50, 0.0),   # 150..200, no current
    ]

    row_ptr = np.array([0, 0], dtype=np.int32)  # 1 neuron, no synapses
    col_idx = np.array([], dtype=np.int32)
    vals = np.array([], dtype=np.float32)

    v_init = np.full(N, V_REST, dtype=np.float32)
    nt_types = np.ones(N, dtype=np.float32)

    all_spikes = []
    v_final = v_init.copy()

    for n_steps, I_val in segments:
        ext_I = np.full(N, I_val, dtype=np.float32)
        spike_mat, v_final, _ = run_our_lif(
            row_ptr, col_idx, vals,
            N, n_steps, DT_MS,
            v_final, TAU_M, V_REST, V_THRESH, V_RESET,
            nt_types, external_I=ext_I,
        )
        # Find spikes in this segment
        for step in range(n_steps):
            if spike_mat[step, 0] > 0:
                # Global step offset
                global_step = sum(s[0] for s in segments[:segments.index((n_steps, I_val))]) + step
                all_spikes.append(global_step)

    our_spk = np.array(sorted(all_spikes)) * DT_MS
    print(f"\n  Our spikes: {our_spk}")

    # ─── Brian2 ───
    eqs = """
        dv/dt = (v_0 - v + g + I_inj) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau                       : volt (unless refractory)
        rfc                                    : second
        I_inj                                  : volt
    """
    params = {
        'v_0': V_REST * mV, 'v_rst': V_RESET * mV, 'v_th': V_THRESH * mV,
        't_mbr': TAU_M * ms, 'tau': TAU_SYN * ms, 't_rfc': TAU_REF * ms,
    }
    _neu = NeuronGroup(
        N=1, model=eqs, method='linear',
        threshold='v > v_th',
        reset='v = v_rst; g = 0 * mV',
        refractory='rfc', namespace=params,
    )
    _neu.v = params['v_0']
    _neu.g = 0 * mV
    _neu.rfc = params['t_rfc']
    _neu.I_inj = 0 * mV

    stm = StateMonitor(_neu, ('v', 'g'), record=True)
    spk = SpikeMonitor(_neu)
    net = Network(_neu, stm, spk)

    dt_sec = DT_MS * 0.001
    for n_steps, I_val in segments:
        _neu.I_inj = I_val * mV
        net.run(duration=n_steps * dt_sec * second)

    v_b2 = np.array(stm.v[0] / mV)
    b2_spk = np.array(spk.t * 1000.0)

    print(f"  Brian2 spikes: {b2_spk}")

    # Compare: spike count should be close (within 1-2 due to refractory timing)
    ratio = len(our_spk) / max(len(b2_spk), 1)
    print(f"\n  Our spike count: {len(our_spk)}")
    print(f"  Brian2 spike count: {len(b2_spk)}")
    print(f"  Ratio: {ratio:.3f}")

    if abs(len(our_spk) - len(b2_spk)) <= 2:
        print("  ✅ Spike count совпадает (≈)")
    else:
        print("  ❌ Spike count не совпадает")

    # Also compare first 3 spike times
    for i in range(min(3, len(our_spk), len(b2_spk))):
        diff = abs(our_spk[i] - b2_spk[i])
        marker = "✅" if diff < 1.0 else "⚠"
        print(f"  {marker} Spike {i+1}: ours={our_spk[i]:.1f}ms, b2={b2_spk[i]:.1f}ms, Δ={diff:.2f}ms")

    return {
        'our_spikes': our_spk, 'brian2_spikes': b2_spk,
        'our_count': len(our_spk), 'b2_count': len(b2_spk),
        'ratio': ratio,
    }


# ═══════════════════════════════════════════════════════════
#  TEST 2: Two neurons — PSP (batch call)
# ═══════════════════════════════════════════════════════════

def test_psp():
    """Pre-neuron fires at step 0 → measure PSP in post-neuron after delay."""
    print("\n" + "═" * 65)
    print("  TEST 2: Two neurons — pre→post PSP (batch)")
    print("═" * 65)

    N = 2
    n_steps = 300
    weight = 5.0
    edges = [(0, 1, weight)]
    row_ptr, col_idx, vals = build_csr(edges, N)

    # ─── Our engine ───
    v_init = np.full(N, V_REST, dtype=np.float32)
    v_init[0] = V_THRESH + 1.0  # well above threshold to guarantee fire
    nt_types = np.ones(N, dtype=np.float32)

    ext_I = np.zeros(N, dtype=np.float32)
    spike_mat, v_final, g_final = run_our_lif(
        row_ptr, col_idx, vals,
        N, n_steps, DT_MS,
        v_init, TAU_M, V_REST, V_THRESH, V_RESET,
        nt_types, external_I=ext_I,
    )

    pre_fired = np.any(spike_mat[:, 0] > 0)
    pre_spike_steps = np.where(spike_mat[:, 0] > 0)[0]
    post_v = None  # We don't have v(t) trace from batch call...

    # Re-run with per-step v recording for PSP comparison
    v_recording = np.full(N, V_REST, dtype=np.float32)
    v_recording[0] = V_THRESH + 1.0
    v_trace = np.zeros((n_steps, N), dtype=np.float32)

    # Single batch call
    n_pad = 64
    v = np.full(n_pad, V_REST, dtype=np.float32)
    v[0] = V_THRESH + 1.0
    tau_m = np.full(n_pad, TAU_M, dtype=np.float32)
    v_rest = np.full(n_pad, V_REST, dtype=np.float32)
    v_thresh = np.full(n_pad, V_THRESH, dtype=np.float32)
    v_reset = np.full(n_pad, V_RESET, dtype=np.float32)
    nt = np.ones(n_pad, dtype=np.float32)
    ext = np.zeros(n_pad, dtype=np.float32)
    I_buf = np.zeros(n_pad, dtype=np.float32)
    spk = np.zeros(n_pad, dtype=np.int32)
    spike_buf = np.zeros(n_steps * N, dtype=np.int32)
    g_syn = np.zeros(n_pad, dtype=np.float32)
    delay_buf = np.zeros(n_pad * DELAY_STEPS, dtype=np.float32)
    delay_pos = np.zeros(1, dtype=np.int32)

    for step in range(n_steps):
        tcm.lif_connectome.lif_sim_cpu_recording(
            row_ptr, col_idx, vals,
            1, DT_MS,
            v, tau_m, v_rest, v_thresh, v_reset,
            nt, ext, I_buf, spk,
            spike_buf[step * N:(step + 1) * N],
            g_syn, TAU_SYN,
            delay_buf, DELAY_STEPS, delay_pos,
        )
        v_trace[step] = v[:N]

    # ─── Brian2 ───
    w_b2 = our_weight_to_brian2(weight)
    params = {
        'v_0': V_REST * mV, 'v_rst': V_RESET * mV, 'v_th': V_THRESH * mV,
        't_mbr': TAU_M * ms, 'tau': TAU_SYN * ms, 't_rfc': TAU_REF * ms,
        't_dly': DELAY_MS * ms,
    }
    neu = NeuronGroup(
        N=2, model='''
            dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
            dg/dt = -g / tau               : volt (unless refractory)
            rfc                            : second
        ''', method='linear',
        threshold='v > v_th',
        reset='v = v_rst; g = 0 * mV',
        refractory='rfc', namespace=params,
    )
    neu.v = params['v_0']
    neu.g = 0 * mV
    neu.rfc = params['t_rfc']
    neu.v[0] = (V_THRESH + 1.0) * mV  # > threshold → fire on step 0

    syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=params['t_dly'])
    syn.connect(i=0, j=1)
    syn.w = [w_b2] * mV

    stm = StateMonitor(neu, ('v', 'g'), record=True)
    spk = SpikeMonitor(neu)
    net = Network(neu, syn, stm, spk)

    dt_sec = DT_MS * 0.001
    net.run(duration=n_steps * dt_sec * second)

    v1_b2 = np.array(stm.v[1] / mV)[:n_steps]
    g1_b2 = np.array(stm.g[1] / mV)[:n_steps]

    # ─── Compare ───
    delay_start = min(DELAY_STEPS, n_steps - 1)

    psp_ours = v_trace[delay_start:, 1] - V_REST
    psp_b2 = v1_b2[delay_start:] - V_REST

    peak_ours = np.max(psp_ours)
    peak_b2 = np.max(psp_b2)
    peak_step_ours = np.argmax(psp_ours) + delay_start
    peak_step_b2 = np.argmax(psp_b2) + delay_start

    mse = np.mean((v_trace[:, 1] - v1_b2[:n_steps]) ** 2)

    # Pre-fire check
    pre_our_spikes = np.where((spike_buf.reshape(n_steps, N))[:, 0] > 0)[0]
    b2_pre = np.where(spk.i[:] == 0)[0]
    b2_pre_times = np.array(spk.t[list(b2_pre)]) * 1000 if len(b2_pre) > 0 else np.array([])

    print(f"\n  Weight: {weight}  →  Brian2 w: {w_b2:.4f} mV")
    print(f"  Delay: {DELAY_STEPS} steps ({DELAY_MS}ms)")
    print(f"\n  Pre spike (ours):  step {pre_our_spikes[:5]}")
    print(f"  Pre spike (Brian2): {b2_pre_times[:5]}")
    print(f"\n  PSP peak (ours):   {peak_ours:.6f} mV at step {peak_step_ours}")
    print(f"  PSP peak (Brian2): {peak_b2:.6f} mV at step {peak_step_b2}")
    print(f"  Peak ratio:        {peak_ours / max(peak_b2, 1e-8):.6f}")
    print(f"  v1 MSE:            {mse:.6f} mV²")

    if abs(peak_ours - peak_b2) < 0.05 and peak_ours > 0.001:
        print("  ✅ PSP amplitude совпадает!")
    elif peak_ours < 0.001:
        print("  ❌ PSP amplitude = 0 — синапс не сработал!")
    else:
        print("  ⚡ PSP amplitude близко")

    return {
        'peak_ours': peak_ours, 'peak_b2': peak_b2,
        'peak_step_ours': peak_step_ours, 'peak_step_b2': peak_step_b2,
        'mse': mse, 'v_ours': v_trace[:, 1], 'v_brian2': v1_b2,
        'pre_fired': len(pre_our_spikes) > 0,
    }


# ═══════════════════════════════════════════════════════════
#  TEST 3: Random network with Poisson drive
# ═══════════════════════════════════════════════════════════

def test_random_network():
    """50-neuron network with brief initial kickstart (no Poisson).
    
    Оба движка получают ОДИНАКОВЫЙ начальный current pulse (50 mV, 5 steps),
    чтобы kickstart'нуть рекуррентную сеть. После pulse — только recurrent
    connections через alpha-синапс с delay.
    """
    print("\n" + "═" * 65)
    print("  TEST 3: Random network (50 neurons, initial pulse)")
    print("═" * 65)

    np.random.seed(42)
    N = 50
    n_steps = 1000  # 500ms

    # Random connectivity (5%)
    conn_prob = 0.05
    edges = []
    for i in range(N):
        for j in range(N):
            if i != j and np.random.rand() < conn_prob:
                w = np.random.uniform(1.0, 5.0)
                edges.append((i, j, w))
    print(f"  Synapses: {len(edges)}")

    row_ptr, col_idx, vals = build_csr(edges, N)

    # Background current: 18 mV constant (≈90 Hz baseline, like Poisson average)
    # Plus initial kick: 50 mV for 5 steps
    BG_CURRENT = 18.0  # mV — постоянный background drive
    KICK_AMPLITUDE = 50.0  # mV — initial kickstart
    KICK_STEPS = 5

    # ─── Our engine ───
    n_pad = max(64, (N + 7) // 8 * 8)
    v = np.full(n_pad, V_REST, dtype=np.float32)
    v[:N] += np.random.RandomState(0).uniform(-1.0, 1.0, N).astype(np.float32)
    tau_m = np.full(n_pad, TAU_M, dtype=np.float32)
    v_rest = np.full(n_pad, V_REST, dtype=np.float32)
    v_thresh = np.full(n_pad, V_THRESH, dtype=np.float32)
    v_reset = np.full(n_pad, V_RESET, dtype=np.float32)
    nt = np.ones(n_pad, dtype=np.float32)
    ext = np.zeros(n_pad, dtype=np.float32)
    I_buf = np.zeros(n_pad, dtype=np.float32)
    spk = np.zeros(n_pad, dtype=np.int32)
    spike_buf = np.zeros(n_steps * N, dtype=np.int32)
    g_syn = np.zeros(n_pad, dtype=np.float32)
    delay_buf = np.zeros(n_pad * DELAY_STEPS, dtype=np.float32)
    delay_pos = np.zeros(1, dtype=np.int32)

    t0 = time.time()
    for step in range(n_steps):
        # Background current + kickstart
        I_step = BG_CURRENT + (KICK_AMPLITUDE if step < KICK_STEPS else 0.0)
        ext[:N] = I_step
        tcm.lif_connectome.lif_sim_cpu_recording(
            row_ptr, col_idx, vals,
            1, DT_MS,
            v, tau_m, v_rest, v_thresh, v_reset,
            nt, ext, I_buf, spk,
            spike_buf[step * N:(step + 1) * N],
            g_syn, TAU_SYN,
            delay_buf, DELAY_STEPS, delay_pos,
        )
    elapsed_ours = time.time() - t0
    our_spk = spike_buf.reshape((n_steps, N))
    our_total = int(np.sum(our_spk))
    print(f"  [Our engine] {elapsed_ours:.3f}s, {our_total} spikes")

    # ─── Brian2 (identifical kickstart: I_inj = 50 mV for 5 steps) ───
    w_b2 = [our_weight_to_brian2(w) for _, _, w in edges]
    params = {
        'v_0': V_REST * mV, 'v_rst': V_RESET * mV, 'v_th': V_THRESH * mV,
        't_mbr': TAU_M * ms, 'tau': TAU_SYN * ms, 't_rfc': TAU_REF * ms,
        't_dly': DELAY_MS * ms,
    }
    neu = NeuronGroup(
        N=N, model='''
            dv/dt = (v_0 - v + g + I_inj) / t_mbr : volt (unless refractory)
            dg/dt = -g / tau                       : volt (unless refractory)
            rfc                                    : second
            I_inj                                  : volt
        ''', method='linear',
        threshold='v > v_th',
        reset='v = v_rst; g = 0 * mV',
        refractory='rfc', namespace=params,
    )
    neu.v = params['v_0']
    neu.g = 0 * mV
    neu.rfc = params['t_rfc']
    noise = np.random.RandomState(0).uniform(-1.0, 1.0, N)
    neu.v += noise * mV
    neu.I_inj = 0 * mV

    syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=params['t_dly'])
    if edges:
        syn.connect(i=[e[0] for e in edges], j=[e[1] for e in edges])
        syn.w = np.array(w_b2) * mV

    stm = StateMonitor(neu, ('v',), record=True)
    spk = SpikeMonitor(neu)
    net = Network(neu, syn, stm, spk)

    t0 = time.time()
    dt_sec = DT_MS * 0.001
    for step in range(n_steps):
        I_step = BG_CURRENT + (KICK_AMPLITUDE if step < KICK_STEPS else 0.0)
        neu.I_inj = I_step * mV
        net.run(duration=dt_sec * second)
    elapsed_b2 = time.time() - t0

    b2_spk = np.zeros((n_steps, N), dtype=np.int32)
    for idx, times in spk.spike_trains().items():
        for t in times:
            step = int(t * 1000.0 / DT_MS)
            if step < n_steps:
                b2_spk[step, idx] = 1

    b2_total = int(np.sum(b2_spk))
    print(f"  [Brian2] {elapsed_b2:.3f}s, {b2_total} spikes")
    print(f"  Speed ratio: {elapsed_b2 / max(elapsed_ours, 0.001):.1f}x Brian2/ours")

    # ─── Compare ───
    o_flat = our_spk.flatten()
    b_flat = b2_spk.flatten()
    inter = np.sum((o_flat > 0) & (b_flat > 0))
    union = np.sum((o_flat > 0) | (b_flat > 0))
    jaccard = inter / max(union, 1)

    our_rates = np.sum(our_spk, axis=0) / (n_steps * DT_MS / 1000.0)
    b2_rates = np.sum(b2_spk, axis=0) / (n_steps * DT_MS / 1000.0)

    mask = (our_rates > 0) | (b2_rates > 0)
    rate_corr = np.corrcoef(our_rates[mask], b2_rates[mask])[0, 1] if np.sum(mask) > 1 else 0.0
    sc_corr = np.corrcoef(our_rates, b2_rates)[0, 1] if N > 1 else 0.0

    print(f"\n  {'Metric':>25} {'Our':>12} {'Brian2':>12}")
    print(f"  {'-'*25:>25} {'-'*12:>12} {'-'*12:>12}")
    print(f"  {'Total spikes':>25} {our_total:>12d} {b2_total:>12d}")
    print(f"  {'Active neurons':>25} {int(np.sum(our_rates > 1)):>12d} {int(np.sum(b2_rates > 1)):>12d}")
    print(f"  {'Mean rate (Hz)':>25} {np.mean(our_rates):>12.1f} {np.mean(b2_rates):>12.1f}")
    print(f"  {'Jaccard similarity':>25} {jaccard:>12.4f}")
    print(f"  {'Rate correlation':>25} {rate_corr:>12.4f}")
    print(f"  {'Spike-count corr':>25} {sc_corr:>12.4f}")

    return {
        'jaccard': jaccard, 'rate_corr': rate_corr, 'sc_corr': sc_corr,
        'our_spikes': our_total, 'brian2_spikes': b2_total,
        'our_rates': our_rates, 'brian2_rates': b2_rates,
    }


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("═" * 65)
    print("  🧬 Vалидация LIF: наш engine vs Brian2 reference")
    print("═" * 65)
    print(f"\n  Параметры:")
    print(f"    τ_m={TAU_M}ms, τ_syn={TAU_SYN}ms, τ_ref={TAU_REF}ms")
    print(f"    V_rest={V_REST}mV, V_thresh={V_THRESH}mV, V_reset={V_RESET}mV")
    print(f"    Delay={DELAY_STEPS} steps ({DELAY_MS}ms), dt={DT_MS}ms")

    r1 = test_single_neuron()
    r2 = test_psp()
    r3 = test_random_network()

    print("\n" + "═" * 65)
    print("  📋 ИТОГ ВАЛИДАЦИИ")
    print("═" * 65)

    spike_match = abs(r1['our_count'] - r1['b2_count']) <= 2
    s1 = "✅" if spike_match else "❌"
    print(f"  {s1} TEST 1: spikes {r1['our_count']} vs {r1['b2_count']} (ratio {r1['ratio']:.2f})")

    psp_ok = r2['peak_ours'] > 0.01 and abs(r2['peak_ours'] - r2['peak_b2']) < 0.1
    s2 = "✅" if psp_ok else "❌"
    print(f"  {s2} TEST 2: PSP peak Δ={abs(r2['peak_ours']-r2['peak_b2']):.4f} mV")

    j_ok = r3['jaccard'] > 0.1
    rc_ok = abs(r3['rate_corr']) > 0.3
    s3 = "✅" if (j_ok or rc_ok) else "❌"
    print(f"  {s3} TEST 3: Jaccard={r3['jaccard']:.4f}, rate_corr={r3['rate_corr']:.4f}, sc_corr={r3['sc_corr']:.4f}")

    if spike_match and psp_ok and (j_ok or rc_ok):
        print("\n  🎉 LIF engine совместим с Brian2 reference!")
    else:
        print("\n  ⚠ Есть расхождения")


if __name__ == '__main__':
    main()
