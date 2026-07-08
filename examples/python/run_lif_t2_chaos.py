#!/usr/bin/env python3
"""
🧠 LIF + T2-SPIGRAD + Chaos/Pattern — три эксперимента на synthetic данных.

1. LIF симуляция на synth_brain.taas (52K нейронов, 693K синапсов)
2. T2-SPIGRAD на trans_count (2244×2244) — осцилляторная динамика 
3. CPE Chaos/Pattern — contrastive анализ через dopamine δ

Использование:
    cd taas_checkmate
    .venv/bin/python3 examples/python/run_lif_t2_chaos.py
"""

import sys, os, time
from pathlib import Path
import numpy as np

PROJ = Path(__file__).parent.parent
DATA_DIR = Path.home() / "Проекты" / "asm_engines" / "MEIIgpt" / "experiments" / "sardana_1" / "data"

try:
    import taas_checkmate as tcm
except ImportError:
    print("✗ Cannot import taas_checkmate")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════
#  1. LIF симуляция
# ═══════════════════════════════════════════════════════════

def run_lif_simulation():
    """LIF спайковая симуляция на connectome."""
    print("\n" + "═" * 60)
    print("  1. LIF NEURON SIMULATION")
    print("═" * 60)

    taas_path = PROJ / "synth_brain.taas"
    if not taas_path.exists():
        print("  ✗ synth_brain.taas not found. Run run_synthetic_brain.py first.")
        return

    conn = tcm.load_connectome(str(taas_path))
    n = conn.n_neurons
    n_syn = conn.n_synapses
    print(f"\n  Connectome: {n:,} neurons, {n_syn:,} synapses")

    # BrainSimulator LIF
    sim = tcm.BrainSimulator(n, "lif")
    
    n_steps = 200
    dt_ms = 0.5
    
    # External input — random walk на sensory-подобные нейроны
    rng = np.random.RandomState(42)
    ext_I = np.zeros(n, dtype=np.float32)
    # Первые 5% нейронов получают шумовой вход
    n_input = max(1, n // 20)
    ext_I[:n_input] = rng.rand(n_input).astype(np.float32) * 10.0
    
    print(f"  Input: {n_input} neurons with I=0..10")
    print(f"  Steps: {n_steps} @ {dt_ms}ms = {n_steps * dt_ms:.0f}ms")
    
    t0 = time.time()
    spike_buf = sim.step_batch(conn, n_steps, dt_ms, ext_I)
    wall = time.time() - t0
    
    # Анализ спайков
    spike_matrix = spike_buf.reshape(n_steps, n)
    total_spikes = int(np.sum(spike_matrix))
    active_neurons = int(np.sum(np.any(spike_matrix > 0, axis=0)))
    firing_rates = np.mean(spike_matrix, axis=0)  # spikes per step
    
    print(f"\n  ── Results ──")
    print(f"    Wall time:   {wall:.2f}s")
    print(f"    Total spikes: {total_spikes:,}")
    print(f"    Active neurons: {active_neurons:,} / {n:,} ({100*active_neurons/n:.1f}%)")
    print(f"    Avg firing rate: {total_spikes/n/n_steps*1000/dt_ms:.1f} Hz")
    print(f"    Top-5 firing neurons: {np.argsort(-firing_rates)[:5].tolist()}")
    print(f"    Top-5 firing rates: {np.sort(firing_rates)[-5:][::-1].tolist()}")
    
    return {'spike_buf': spike_buf, 'n_steps': n_steps, 'dt_ms': dt_ms,
            'wall': wall, 'total_spikes': total_spikes, 'active': active_neurons}


# ═══════════════════════════════════════════════════════════
#  2. T2-SPIGRAD на transition matrix
# ═══════════════════════════════════════════════════════════

def run_t2_spigrad():
    """T2-SPIGRAD на trans_count матрице как coupling weights."""
    print("\n" + "═" * 60)
    print("  2. T2-SPIGRAD ON TRANSITION MATRIX")
    print("═" * 60)

    npy_path = DATA_DIR / "synthetic_trans_count.npy"
    if not npy_path.exists():
        print("  ✗ synthetic_trans_count.npy not found")
        return

    trans = np.load(str(npy_path))
    n_states = trans.shape[0]
    n_syn = np.count_nonzero(trans)
    print(f"\n  Transition matrix: {n_states}×{n_states}, {n_syn:,} non-zero")

    # T2-SPIGRAD: интерпретируем trans как coupling между осцилляторами
    # Используем subset — top-100 состояний по out-degree
    out_deg = trans.sum(axis=1)
    top_idx = np.argsort(-out_deg)[:100]  # top 100
    print(f"  Using top-100 states by out-degree")
    
    n_osc = 100
    n_steps = 100
    dt = 0.05
    
    # Drives: случайный шум на первых 10 осцилляторах
    rng = np.random.RandomState(42)
    drives = np.zeros((n_steps, 3 * n_osc), dtype=np.float32)
    drives[:, :10] = rng.randn(n_steps, 10).astype(np.float32) * 0.5
    drives[:, n_osc:n_osc+10] = rng.randn(n_steps, 10).astype(np.float32) * 0.3
    drives[:, 2*n_osc:2*n_osc+10] = rng.randn(n_steps, 10).astype(np.float32) * 0.2
    
    print(f"  Oscillators: {n_osc} (3 banks = {3*n_osc} dims)")
    print(f"  Steps: {n_steps} @ dt={dt}")
    
    t0 = time.time()
    pst_vec, traj = tcm.t2spigrad_forward(
        n_osc, n_steps, dt,
        drive_gain=5.0,
        alpha=1.0, freq_min=0.5, freq_max=3.0,
        coupling_w12=0.2, coupling_w13=0.1, coupling_w23=0.05,
        drives=drives,
    )
    wall = time.time() - t0
    
    print(f"\n  ── Results ──")
    print(f"    Wall time:    {wall:.3f}s")
    print(f"    PST dim:      {len(pst_vec)}")
    print(f"    Trajectory:   {traj.n_osc()} osc × {traj.n_steps()} steps")
    
    # Order parameters из финальных фаз через CPE
    th1 = np.array(traj.th1_final(), dtype=np.float32)
    th2 = np.array(traj.th2_final(), dtype=np.float32)
    th3 = np.array(traj.th3_final(), dtype=np.float32)
    r1 = tcm.cpe.phase_coherence(th1)
    r2 = tcm.cpe.phase_coherence(th2)
    r3 = tcm.cpe.phase_coherence(th3)
    print(f"    Order params: r1={r1:.4f}, r2={r2:.4f}, r3={r3:.4f}")
    print(f"    Mean coherence: {(r1+r2+r3)/3:.4f}")
    
    return {'pst_vec': pst_vec, 'traj': traj, 'wall': wall, 'n_osc': n_osc}


# ═══════════════════════════════════════════════════════════
#  3. CPE Chaos/Pattern
# ═══════════════════════════════════════════════════════════

def run_cpe_chaos_pattern():
    """Chaos/Pattern discriminative analysis через CPE."""
    print("\n" + "═" * 60)
    print("  3. CPE CHAOS vs PATTERN")
    print("═" * 60)

    taas_path = PROJ / "synth_brain.taas"
    if not taas_path.exists():
        print("  ✗ synth_brain.taas not found")
        return

    conn = tcm.load_connectome(str(taas_path))
    rp = np.array(conn.row_ptr(), dtype=np.int32)
    ci = np.array(conn.col_indices(), dtype=np.int32)
    vl = np.array(conn.values(), dtype=np.float32)
    n = conn.n_neurons
    print(f"\n  Connectome: {n:,} neurons")

    # Подвыборка для скорости — первые 5000 нейронов
    sub_n = min(5000, n)
    
    # 1 проход: считаем valid синапсы (pre < sub_n И post < sub_n)
    new_rp = np.zeros(sub_n + 1, dtype=np.int32)
    for i in range(sub_n):
        start, end = int(rp[i]), int(rp[i+1])
        cnt = 0
        for j in range(start, end):
            if j < len(ci) and ci[j] < sub_n:
                cnt += 1
        new_rp[i+1] = new_rp[i] + cnt
    
    # 2 проход: заполняем compact массивы
    n_syn_new = new_rp[-1]
    new_ci = np.zeros(n_syn_new, dtype=np.int32)
    new_vl = np.zeros(n_syn_new, dtype=np.float32)
    pos = 0
    for i in range(sub_n):
        start, end = int(rp[i]), int(rp[i+1])
        for j in range(start, end):
            if j < len(ci) and ci[j] < sub_n:
                new_ci[pos] = ci[j]
                new_vl[pos] = vl[j]
                pos += 1
    
    n_sub = sub_n
    print(f"  Sub-sampled: {n_sub} neurons, {n_syn_new} synapses (kept {100*n_syn_new/max(len(vl),1):.1f}%)")
    
    rng = np.random.RandomState(42)
    dt = 0.05
    n_steps = 500
    
    # PATTERN: coherent input на первой группе
    phases_pat = rng.rand(n_sub).astype(np.float32) * 0.1  # почти синфазно
    omegas_pat = np.full(n_sub, np.pi, dtype=np.float32)
    
    # CHAOS: random phases
    phases_cha = rng.rand(n_sub).astype(np.float32) * 2 * np.pi
    omegas_cha = np.full(n_sub, np.pi, dtype=np.float32) + rng.randn(n_sub).astype(np.float32) * 2.0
    
    print(f"\n  Pattern: coherent init + ω=π")
    print(f"  Chaos:   random init + ω=π±2σ")
    print(f"  Steps:   {n_steps} @ dt={dt}")
    
    coh_pat_log, coh_cha_log = [], []
    t0 = time.time()
    
    for step in range(n_steps):
        tcm.cpe.kuramoto_step(phases_pat, omegas_pat, new_rp, new_ci, new_vl, dt)
        tcm.cpe.kuramoto_step(phases_cha, omegas_cha, new_rp, new_ci, new_vl, dt)
        
        if step % 100 == 0:
            coh_pat = tcm.cpe.phase_coherence(phases_pat)
            coh_cha = tcm.cpe.phase_coherence(phases_cha)
            coh_pat_log.append((step, coh_pat))
            coh_cha_log.append((step, coh_cha))
    
    wall = time.time() - t0
    delta = coh_pat_log[-1][1] - coh_cha_log[-1][1] if coh_pat_log and coh_cha_log else 0
    
    print(f"\n  ── Results ──")
    print(f"    Wall time: {wall:.2f}s")
    print(f"\n    Step   Pattern   Chaos     Δ")
    for (sp, cp), (sc, cc) in zip(coh_pat_log, coh_cha_log):
        print(f"    {sp:5d}  {cp:.4f}    {cc:.4f}    {cp-cc:+.4f}")
    print(f"\n    Final δ (Pattern − Chaos): {delta:+.4f}")
    
    if delta > 0.05:
        print(f"\n  ✅ Pattern > Chaos: дискриминация работает!")
    elif delta < -0.05:
        print(f"\n  ⚠️ Chaos > Pattern: инверсная дискриминация")
    else:
        print(f"\n  ➖ Δ ≈ 0: нет дискриминации (нужно больше шагов или сильнее сигнал)")
    
    return {'delta': delta, 'coh_pat': coh_pat_log, 'coh_cha': coh_cha_log, 'wall': wall}


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  🧠 LIF + T2-SPIGRAD + CHAOS/PATTERN")
    print("  📊 Synthetic Connectome Analysis")
    print("═" * 60)
    
    # 1. LIF
    result_lif = run_lif_simulation()
    
    # 2. T2-SPIGRAD
    result_t2 = run_t2_spigrad()
    
    # 3. CPE Chaos/Pattern
    result_cpe = run_cpe_chaos_pattern()
    
    # ── Summary ──
    print("\n" + "═" * 60)
    print("  📊 FINAL SUMMARY")
    print("═" * 60)
    
    if result_lif:
        print(f"\n  1. LIF: {result_lif['total_spikes']:,} spikes, "
              f"{result_lif['active']:,} active neurons, "
              f"{result_lif['wall']:.1f}s")
    if result_t2:
        th1 = np.array(result_t2['traj'].th1_final(), dtype=np.float32)
        th2 = np.array(result_t2['traj'].th2_final(), dtype=np.float32)
        th3 = np.array(result_t2['traj'].th3_final(), dtype=np.float32)
        r1 = tcm.cpe.phase_coherence(th1)
        r2 = tcm.cpe.phase_coherence(th2)
        r3 = tcm.cpe.phase_coherence(th3)
        print(f"  2. T2: osc={result_t2['n_osc']}, "
              f"order=(r1={r1:.3f}, r2={r2:.3f}, r3={r3:.3f}), "
              f"{result_t2['wall']:.2f}s")
    if result_cpe:
        print(f"  3. CPE: δ={result_cpe['delta']:+.4f} (pattern-chaos), "
              f"{result_cpe['wall']:.1f}s")
    
    print("\n" + "═" * 60)
    print("  Done.")
    print("═" * 60)


if __name__ == "__main__":
    main()
