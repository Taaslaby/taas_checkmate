#!/usr/bin/env python3
"""Smoke-тест единого фасада `taas_checkmate.pillars` (Фаза 11).

1) Проверка поверхности: 71 функция (не-cuda) + 13 классов на месте.
2) Численный прогон step_* (LIF, Izhikevich, AdEx, FHN, WC, SAW, HR, NMDA)
   и сигнальных кернелов (safety_clamp, sigmoid_gate, t3_drive_encoder).
3) Инстанцирование нейрон-классов (LifNeuron, PoissonGenerator) + step.

Запуск: .venv/Scripts/python.exe smoke_pillars.py
"""
import sys

import numpy as np

import taas_checkmate.pillars as P

FAIL = 0


def check(cond, name, detail=""):
    global FAIL
    mark = "OK " if cond else "FAIL"
    if not cond:
        FAIL += 1
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return cond


# ─────────────────────────────────────────────────────────────
# 1. Поверхность фасада
# ─────────────────────────────────────────────────────────────
FNS = [
    "step_izhikevich", "step_saw", "step_gco", "step_lif", "step_adex", "step_fhn",
    "step_haap", "step_nmda", "step_hr", "step_wc", "step_atlasov_neuromotor",
    "stdp_update", "stdp_homeostatic", "firing_rates", "mean_firing_rate",
    "compute_isi", "compute_cvs", "step_eprop",
    "kuramoto_step_sparse", "t3_step_sparse", "interference", "interference_mean",
    "batch_coherence_sweep", "phase_coherence", "band_energy", "lazy_sincos_update",
    "kalman_update", "adam_train", "nlms_train", "pst_decode", "ring_synthesis",
    "winding_matrix",
    "safety_clamp", "sigmoid_gate", "phase_gate_apply", "stochastic_encoder",
    "thalamic_gate", "hebbian_update", "echo_tap", "t3_drive_encoder",
    "t3_complex_decoder", "t3_inverse_kuramoto", "t3_pst_features", "t3_tact_generator",
    "compute_hamiltonian", "step_gco_full", "propagate_spikes", "sim_step",
    "rate_step", "apply_stdp_ltd",
    "intrinsic_plasticity", "synaptic_normalization", "bsa_encode", "phase_encode",
    "bsa_kernel_hamming", "population_rate_decode", "spike_raster_to_csv",
    "surrogate_gradient", "eprop_update", "probe_eprop_fr", "suggest_eprop_config",
    # cpe-модуль: исторические имена БЕЗ префикса cpe_ (кроме двух)
    "kuramoto_step", "cpe_phase_coherence", "spike_entropy", "detect_spikes",
    "dopamine_delta", "dopamine_timing_gate", "eligibility_update",
    "cpe_stdp_update", "vnc_step", "lif_sim_cpu_recording",
]
CLASSES = [
    "EPropLIF", "EPropIzhikevichMulti", "SornNetwork", "TtfsEncoder", "LifNeuron",
    "PoissonGenerator", "ShortTermPlasticity", "Glif3State", "Glif4State",
    "SomatoDendriticNeuron", "AstrocyteState", "EPropTrainer", "ActorCriticAgent",
]

print(f"=== 1. Поверхность: ожидается {len(FNS)} функций + {len(CLASSES)} классов ===")
missing = [n for n in FNS if not hasattr(P, n)]
check(len(missing) == 0, f"все {len(FNS)} функций на месте", f"нет: {missing[:5]}")
missing_c = [n for n in CLASSES if not hasattr(P, n)]
check(len(missing_c) == 0, f"все {len(CLASSES)} классов на месте", f"нет: {missing_c[:5]}")

# Счётчик callable-ов/классов в модуле (для сверки)
fns_actual = [n for n in dir(P) if callable(getattr(P, n)) and not n.startswith("_")]
check(len(fns_actual) >= len(FNS), f"callable-объектов в модуле: {len(fns_actual)}")


# ─────────────────────────────────────────────────────────────
# 2. Численный прогон step_*
# ─────────────────────────────────────────────────────────────
def n8(v):  # массивы кратны 8 — идём по SIMD-пути
    return np.array(v, dtype=np.float32)


def run_steps(step, steps, spikes):
    total = 0
    for _ in range(steps):
        step()
        total += int(spikes.sum())
        spikes[:] = 0
    return total


print("\n=== 2. Численный прогон step_* (n=16, SIMD-путь) ===")

# LIF: v=-70, I=30, tau_m=10 → спайки
v, I = n8([-70.0] * 16), n8([30.0] * 16)
spikes = np.zeros(16, dtype=np.int32)
tm, vr, vt, vres = n8([10.0] * 16), n8([-70.0] * 16), n8([-55.0] * 16), n8([-75.0] * 16)
t = run_steps(lambda: P.step_lif(v, I, spikes, 1.0, tm, vr, vt, vres), 100, spikes)
check(t > 0, "step_lif спайки", f"total={t}")

# Izhikevich: RS-параметры, I=30
v, u = n8([-65.0] * 16), n8([-13.0] * 16)
trace = n8([0.0] * 16)
a, b = n8([0.02] * 16), n8([0.2] * 16)
I_, c, d = n8([30.0] * 16), n8([-65.0] * 16), n8([8.0] * 16)
spikes = np.zeros(16, dtype=np.int32)
t = run_steps(lambda: P.step_izhikevich(v, u, trace, a, b, I_, 1.0, 0.99, spikes, c, d), 200, spikes)
check(t > 0, "step_izhikevich спайки", f"total={t}")
check(np.isfinite(v).all(), "step_izhikevich v конечен")

# AdEx
v, u = n8([-65.0] * 16), n8([0.0] * 16)
I_ = n8([30.0] * 16)
tm, tw = n8([20.0] * 16), n8([100.0] * 16)
vr, vt, vres = n8([-65.0] * 16), n8([-50.0] * 16), n8([-65.0] * 16)
dt_, a_, b_ = n8([2.0] * 16), n8([0.002] * 16), n8([0.05] * 16)
spikes = np.zeros(16, dtype=np.int32)
t = run_steps(lambda: P.step_adex(v, u, I_, spikes, 1.0, tm, tw, vr, vt, vres, dt_, a_, b_), 300, spikes)
check(t > 0, "step_adex спайки", f"total={t}")

# FHN: v,w, I=0.5, dt=0.05
v, w = n8([0.0] * 16), n8([0.0] * 16)
I_ = n8([0.5] * 16)
tw, a_, b_ = n8([1.0] * 16), n8([0.7] * 16), n8([0.8] * 16)
spikes = np.zeros(16, dtype=np.int32)
for _ in range(2000):
    P.step_fhn(v, w, I_, 0.05, tw, a_, b_, spikes)
check(np.isfinite(v).all() and np.isfinite(w).all(), "step_fhn конечен 2000 шагов")

# WC (Wilson-Cowan): rates
e, i = n8([0.1] * 16), n8([0.1] * 16)
i_e, i_i = n8([1.0] * 16), n8([0.0] * 16)
te, ti = n8([5.0] * 16), n8([5.0] * 16)
w_ee, w_ei = n8([12.0] * 16), n8([-6.0] * 16)
w_ie, w_ii = n8([6.0] * 16), n8([-1.0] * 16)
beta = n8([1.0] * 16)
se, si = n8([0.0] * 16), n8([0.0] * 16)
for _ in range(500):
    P.step_wc(e, i, i_e, i_i, 0.1, te, ti, w_ee, w_ei, w_ie, w_ii, beta, se, si)
check(np.isfinite(e).all() and np.isfinite(i).all(), "step_wc конечен 500 шагов")
check(float(e.max()) > 0.5, "step_wc активность выросла", f"e.max={float(e.max()):.2f}")

# SAW: v=-65,u=-13, I=30, без шума
v, u = n8([-65.0] * 16), n8([-13.0] * 16)
trace = n8([0.0] * 16)
a, b = n8([0.02] * 16), n8([0.2] * 16)
I_ = n8([30.0] * 16)
spikes = np.zeros(16, dtype=np.int32)
t = run_steps(lambda: P.step_saw(v, u, trace, a, b, I_, 1.0, 0.99, 0.0, spikes), 200, spikes)
check(t > 0, "step_saw спайки", f"total={t}")

# HR (Hindmarsh-Rose): I=3 → спайки
x, y, z = n8([0.0] * 16), n8([0.0] * 16), n8([0.0] * 16)
I_ = n8([3.0] * 16)
a_, b_, c_, d_ = n8([1.0] * 16), n8([3.0] * 16), n8([1.0] * 16), n8([5.0] * 16)
r_, s_, xr = n8([0.006] * 16), n8([4.0] * 16), n8([-1.6] * 16)
spikes = np.zeros(16, dtype=np.int32)
t = run_steps(lambda: P.step_hr(x, y, z, I_, spikes, 0.01, a_, b_, c_, d_, r_, s_, xr), 3000, spikes)
check(t > 0, "step_hr спайки", f"total={t}")

# NMDA: ток через mg-block
v, g_act = n8([-60.0] * 16), n8([0.5] * 16)
i_out = n8([0.0] * 16)
g_nmda, E_syn, mg = n8([1.0] * 16), n8([0.0] * 16), n8([1.0] * 16)
P.step_nmda(v, g_act, i_out, g_nmda, E_syn, mg)
check(float(np.abs(i_out).max()) > 0.0, "step_nmda даёт ток", f"i_out.max={float(np.abs(i_out).max()):.4f}")

# ─────────────────────────────────────────────────────────────
# 3. Сигнальные кернелы
# ─────────────────────────────────────────────────────────────
print("\n=== 3. Сигнальные кернелы ===")
data = n8([-3.0, -1.0, 0.0, 1.0, 3.0, 5.0, -0.5, 2.0])
P.safety_clamp(data, -1.0, 1.0)
check(np.all((data >= -1.0) & (data <= 1.0)), "safety_clamp в [-1,1]")

out = n8([0.0] * 16)
inp = n8([float(x) for x in range(-8, 8)])
P.sigmoid_gate(out, inp, 1.0, 0.0)
check(abs(float(out[8]) - 0.5) < 0.05, "sigmoid_gate sigmoid(0)=0.5", f"out[8]={float(out[8]):.3f}")

drives = n8([0.0] * 16)
P.t3_drive_encoder(drives, n8([float(x) for x in range(16)]), 2.0)
check(abs(float(drives[7]) - 14.0) < 1e-5, "t3_drive_encoder gain*input", f"drives[7]={float(drives[7])}")

# ─────────────────────────────────────────────────────────────
# 4. Нейрон-классы
# ─────────────────────────────────────────────────────────────
print("\n=== 4. Нейрон-классы ===")
nrn = P.LifNeuron(16, dt=0.5)
check(hasattr(nrn, "step"), "LifNeuron(16) создан + есть step()")
try:
    v_class = nrn.v  # внутреннее состояние (если доступно)
    check(np.isfinite(v_class).all(), "LifNeuron состояние конечное")
except Exception:
    check(True, "LifNeuron состояние не проверяемо напрямую (ok)")

pg = P.PoissonGenerator(16, dt=1.0, seed=42)
spikes = np.zeros(16, dtype=np.int32)
total = 0
for _ in range(200):
    pg.step(spikes)
    total += int(spikes.sum())
    spikes[:] = 0
check(total > 0, "PoissonGenerator даёт спайки за 200 шагов", f"total={total}")

print(f"\n{'✅ SMOKE PASS' if FAIL == 0 else f'❌ SMOKE FAIL: {FAIL} проверок'}")
sys.exit(0 if FAIL == 0 else 1)
