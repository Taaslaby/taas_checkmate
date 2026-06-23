#!/usr/bin/env python3
"""
АБЕРТОНЫ: КВАНТОВАЯ МОДЕЛЬ СОЗНАНИЯ С МИКРОТРУБОЧКАМИ
=======================================================
Orch-OR (Hameroff/Penrose) на FAFB коннектоме.

ψ-wave на графе + абертоны (внутренние квантовые осцилляторы) + коллапс.
Коллапс → событие → обратная связь в ψ-wave.

Гипотеза: абертоны рождаются на частотах микротрубочек (MHz–GHz),
но коллапс — на γ (40 Hz). Связь ψ → абертоны даёт нелинейный отклик.
"""

import time, math, numpy as np
from taas_checkmate import (
    step_psi_wave, compute_hamiltonian,
)

# ─── Параметры ────────────────────────────────────────────
DT = 0.025
N_STEPS = 4000        # 100 sec
PULSE_AMP = 500.0
PULSE_STEP = 1000     # pulse at step 1000 (25 sec)
N_STIM = 500          # number of stimulated hubs

# Абертоны
OMEGA_BASE = 2.0      # базовая частота абертонов (в units 1/DT)
OMEGA_SPREAD = 0.5    # разброс частот
GAMMA = 0.1           # затухание абертонов
G_COUPLING = 0.5      # связь ψ → абертон
THRESHOLD = 0.8       # порог коллапса
COLLAPSE_CURRENT = 3.0  # обратная связь в ψ при коллапсе
RANDOM_SEED = 42

print("█" * 70)
print("  АБЕРТОНЫ: ψ-WAVE + MICROTUBULE COLLAPSE")
print("  Orch-OR на FAFB connectome")
print("█" * 70)

np.random.seed(RANDOM_SEED)

# ─── Load FAFB ───
print("\n[1/6] Loading FAFB...", end=" ", flush=True)
d = np.load("fafb_139k.npz")
rp, ci, vs = d["row_ptr"], d["col_indices"], d["values"].astype(np.float32)
n = len(rp) - 1
n_pad = (n + 7) // 8 * 8
print(f"{n:,} neurons, {len(vs):,} synapses")

# ─── Shuffled ───
print("[2/6] Shuffling...", end=" ", flush=True)
sci = ci.copy()
for i in range(n):
    s, e = rp[i], rp[i+1]
    if e > s:
        sci[s:e] = np.random.permutation(sci[s:e])
print("done")

# ─── Stim ───
outdeg = np.array([rp[i+1] - rp[i] for i in range(n)], dtype=np.int32)
stim = np.sort(np.argsort(outdeg)[::-1][:N_STIM])
rdi = np.sort(np.random.choice(n, 2000, replace=False))

# ─── Init ψ + абертоны ───
def init_state(seed=0):
    rng = np.random.default_rng(RANDOM_SEED + seed)
    th = rng.uniform(0, 2*np.pi, n_pad).astype(np.float32)
    v = np.cos(th, dtype=np.float32)
    u = np.sin(th, dtype=np.float32)
    hr = np.zeros(n_pad, dtype=np.float32)
    hi = np.zeros(n_pad, dtype=np.float32)
    spd = np.ones(n_pad, dtype=np.float32)
    ie = np.zeros(n_pad, dtype=np.float32)
    # Абертоны: комплексные амплитуды
    av = np.zeros(n_pad, dtype=np.float32)   # Re(aberton)
    au = np.zeros(n_pad, dtype=np.float32)   # Im(aberton)
    # Частоты абертонов для каждого нейрона
    om = (OMEGA_BASE + OMEGA_SPREAD * rng.uniform(-1, 1, n_pad)).astype(np.float32)
    return v, u, hr, hi, spd, ie, av, au, om

# ─── Step with abertons ───
def step_with_abertons(v, u, hr, hi, spd, ie, av, au, om,
                       rp_i, ci_i, vs_i, dt, hbar=1.0):
    """Один шаг ψ-wave + абертоны + коллапс."""
    # 1. ψ-wave step (существующий kernel)
    compute_hamiltonian(rp_i, ci_i, vs_i, v, u, hr, hi)
    step_psi_wave(v, u, hr, hi, ie, spd, dt, hbar)

    # 2. Update abertons: da/dt = (-i*ω - γ)*a + g*ψ
    # ψ = v + i*u (комплексная волна)
    # a = av + i*au (абертон)
    # da/dt = -i*ω*(av + i*au) - γ*(av + i*au) + g*(v + i*u)
    # = (-γ*av + ω*au + g*v) + i*(-γ*au - ω*av + g*u)
    
    # Векторные операции (быстро через numpy)
    dav = (-GAMMA * av + om * au + G_COUPLING * v) * dt
    dau = (-GAMMA * au - om * av + G_COUPLING * u) * dt
    av += dav
    au += dau

    # 3. Проверка коллапса: |a|² > θ²
    a2 = av*av + au*au
    collapse_mask = a2 > THRESHOLD * THRESHOLD
    n_collapse = np.sum(collapse_mask)
    if n_collapse > 0:
        # Обратная связь в ψ (ток при коллапсе)
        # Добавляем импульс к i_ext на следующий шаг
        ie[collapse_mask] += COLLAPSE_CURRENT * 5.0
        # Reset aberton to 0
        av[collapse_mask] = 0.0
        au[collapse_mask] = 0.0

    return n_collapse

# ─── Run experiment ───
def run_with_abertons(rp_i, ci_i, vs_i, label, seed=0):
    v, u, hr, hi, spd, ie, av, au, om = init_state(seed)
    nch = len(rdi)
    ts = np.zeros((nch, N_STEPS), dtype=np.float32)
    collapse_log = np.zeros(N_STEPS, dtype=np.int32)
    eeg_log = np.zeros(N_STEPS, dtype=np.float32)
    aberton_energy = np.zeros(N_STEPS, dtype=np.float32)

    t0 = time.perf_counter()
    for step in range(N_STEPS):
        # Stimulus
        ie[:] = 0.0
        if step == PULSE_STEP:
            ie[stim] = PULSE_AMP

        # Step
        nc = step_with_abertons(v, u, hr, hi, spd, ie, av, au, om,
                                rp_i, ci_i, vs_i, DT)

        collapse_log[step] = nc
        ts[:, step] = v[rdi]
        eeg_log[step] = float(np.mean(hr) + np.mean(hi)) * 0.5
        aberton_energy[step] = float(np.mean(av*av + au*au))

        if (step + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    [{step+1}/{N_STEPS}] coll={nc:5d}  EEG={eeg_log[step]:+.2f}  a_energy={aberton_energy[step]:.4f}  [{elapsed:.0f}s]")

    return {
        "ts": ts, "collapse": collapse_log, "eeg": eeg_log,
        "aberton_energy": aberton_energy,
        "time": time.perf_counter() - t0,
    }

# ─── Run ───
print("[3/6] Running ψ-wave + ABERTONS...")
results = {}
for label, rp_i, ci_i, vs_i in [
    ("Original", rp, ci, vs),
    ("Shuffled", rp, sci, vs),
]:
    print(f"\n  {label}:")
    r = run_with_abertons(rp_i, ci_i, vs_i, label)
    results[label] = r
    print(f"  Done in {r['time']:.0f}s")

# ─── Analysis ───
print("\n[4/6] Analysis...")

for label in ["Original", "Shuffled"]:
    r = results[label]
    # Collapse rate
    total_coll = np.sum(r["collapse"])
    pre_pulse = np.sum(r["collapse"][:PULSE_STEP])
    post_pulse = np.sum(r["collapse"][PULSE_STEP:])
    print(f"\n  {label}:")
    print(f"    Total collapses: {total_coll}")
    print(f"    Pre-pulse:  {pre_pulse} ({pre_pulse/PULSE_STEP:.1f}/step)")
    print(f"    Post-pulse: {post_pulse} ({(post_pulse)/(N_STEPS-PULSE_STEP):.1f}/step)")
    print(f"    Peak collapse (evoked): {np.max(r['collapse'][PULSE_STEP:PULSE_STEP+50])}")
    print(f"    Mean ab energy: {np.mean(r['aberton_energy']):.6f}")
    print(f"    Max ab energy:  {np.max(r['aberton_energy']):.6f}")

    # Spectral analysis of collapse events
    collapse_ts = r["collapse"]
    fft = np.abs(np.fft.rfft(collapse_ts - np.mean(collapse_ts)))**2
    freqs = np.fft.rfftfreq(N_STEPS, DT)
    # γ band (30-80 Hz in real units, scaled by DT)
    gamma_idx = (freqs > 2) & (freqs < 10)
    print(f"    γ power (2-10 Hz): {np.sum(fft[gamma_idx]):.1f}")

# ─── Evoked response ───
print("\n[5/6] Evoked collapse response...")
for label in ["Original", "Shuffled"]:
    r = results[label]
    # Peristimulus collapse histogram
    window = 100  # steps before/after
    ps = r["collapse"][PULSE_STEP-window:PULSE_STEP+window]
    peak = np.max(ps)
    latency = np.argmax(ps) - window
    print(f"  {label}: evoked peak={peak} at t={latency*DT*1000:.0f}ms")

# ─── Summary ───
print("\n[6/6] SUMMARY")
print("█" * 70)
for label in ["Original", "Shuffled"]:
    r = results[label]
    pre = np.mean(r["collapse"][:PULSE_STEP])
    post = np.mean(r["collapse"][PULSE_STEP:])
    print(f"  {label:<12} | collapses/step: pre={pre:.2f} post={post:.2f} "
          f"ratio={post/(pre+0.001):.2f}x | ab_energy={np.mean(r['aberton_energy']):.4f}")

# Save
np.savez("aberton_results.npz",
         collapse_orig=results["Original"]["collapse"],
         collapse_shuf=results["Shuffled"]["collapse"],
         eeg_orig=results["Original"]["eeg"],
         eeg_shuf=results["Shuffled"]["eeg"],
         ab_energy_orig=results["Original"]["aberton_energy"],
         ab_energy_shuf=results["Shuffled"]["aberton_energy"])
print(f"\n  Saved: aberton_results.npz")
print("█" * 70)
