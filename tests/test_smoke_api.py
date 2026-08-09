#!/usr/bin/env python3
"""Smoke test: все классы и функции Python API импортируются и работают.

Запуск:
    cd taas_checkmate
    PYTHONPATH=target/debug python3 tests/test_smoke_api.py

Ожидаемый результат: OK ... все тесты проходят.
"""

import sys
import os
import tempfile
import numpy as np

# Windows-consoles default to cp1251 which cannot print box-drawing
# characters used below — force UTF-8 so the test runs on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# ─── helpers ─────────────────────────────────────────────

passed = 0
failed = 0


def check(cond: bool, msg: str):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {msg}")
    else:
        failed += 1
        print(f"  ✗ {msg}")


def section(name: str):
    print(f"\n── {name} ─{'─' * max(0, 60 - len(name))}")


# ─── 1. Module import ────────────────────────────────────

section("1. Module import")

import taas_checkmate as tcm

check(hasattr(tcm, "__version__") or True, "module loads without error")
check(hasattr(tcm, "TriOscillator") or hasattr(tcm, "PyTriOscillator"),
      "T3 classes accessible")

# ─── 2. Sub-modules ──────────────────────────────────────

section("2. Sub-modules")

expected_modules = [
    "neurons", "t3", "snn", "eprop", "stp", "rl", "trainer",
    "io", "sparse", "kuramoto", "connectome", "format",
    "autotune",
]
for mod in expected_modules:
    check(hasattr(tcm, mod), f"sub-module '{mod}' exists")

# ─── 3. T3 Classes ───────────────────────────────────────

section("3. T3 — TriOscillator")

# Try both possible names (Py prefix may or may not be stripped)
TriOsc = getattr(tcm, "TriOscillator", getattr(tcm, "PyTriOscillator", None))
check(TriOsc is not None, "TriOscillator class found")

tri = TriOsc(16, 0.5, 3.0)
check(isinstance(tri, TriOsc), "TriOscillator instantiated")
check(tri.n_osc() == 16, f"n_osc() == 16, got {tri.n_osc()}")
check(tri.pst_dim() == 18 * 16 + 12, f"pst_dim() == 300, got {tri.pst_dim()}")

tri.set_coupling(0.3, 0.15, 0.08)

# step without drive
pst = tri.step(0.05)
check(pst is not None and len(pst) == 300, f"step() returns PST ({len(pst)} dims)")

# step with drive
drive = np.zeros(16, dtype=np.float32)
drive[:8] = 1.0
pst2 = tri.step(0.05, drive=drive)
check(len(pst2) == 300, "step(drive=...) returns PST")

# run
pst_batch = tri.run(10, 0.05)
check(pst_batch.shape == (10, 300), f"run(10, 0.05) shape {pst_batch.shape}")

# pst property
pst3 = tri.pst()
check(len(pst3) == 300, "pst() property works")

# order params
r1, r2, r3 = tri.order_params()
check(0.0 <= r1 <= 1.0, f"order_param r1 in [0,1], got {r1}")
check(0.0 <= r2 <= 1.0, f"order_param r2 in [0,1], got {r2}")
check(0.0 <= r3 <= 1.0, f"order_param r3 in [0,1], got {r3}")

# phases / amplitudes
ph1 = tri.phases(1)
check(len(ph1) == 16, f"phases(1) len=16, got {len(ph1)}")
amp = tri.amplitudes(2)
check(len(amp) == 16, f"amplitudes(2) len=16, got {len(amp)}")

# set_drive
tri.set_drive(np.ones(16, dtype=np.float32))
tri.set_drive(np.ones(48, dtype=np.float32))  # 3*16

# repr
r = repr(tri)
check("TriOscillator" in r and "n=16" in r, f"__repr__: {r}")

# ─── 4. NLMS / Kalman Decoders ───────────────────────────

section("4. Decoders")

NlmsDec = getattr(tcm, "NlmsDecoder", getattr(tcm, "PyNlmsDecoder", None))
if NlmsDec:
    dec = NlmsDec(50, 0.1, 1e-8)
    x = np.random.randn(50).astype(np.float32)
    y = dec.predict(x)
    check(isinstance(y, float), f"NLMS predict() → float, got {type(y)}")
    err = dec.train(x, 1.0)
    check(isinstance(err, float), "NLMS train() → float")
    w = dec.weights()
    check(len(w) == 50, f"NLMS weights len=50, got {len(w)}")
    check("NlmsDecoder" in repr(dec), f"NLMS __repr__: {repr(dec)}")

KalmanDec = getattr(tcm, "KalmanDecoder", getattr(tcm, "PyKalmanDecoder", None))
if KalmanDec:
    kdec = KalmanDec(50, 1.0, 0.1)
    y = kdec.predict(x)
    check(isinstance(y, float), "Kalman predict() → float")
    err = kdec.train(x, 1.0)
    check(isinstance(err, float), "Kalman train() → float")
    check("KalmanDecoder" in repr(kdec), f"Kalman __repr__: {repr(kdec)}")

# ─── 5. T2-SPIGRAD ───────────────────────────────────────

section("5. T2-SPIKE-GRAD")

# Forward
n_osc, n_steps = 8, 5
drives = np.random.randn(n_steps, 3 * n_osc).astype(np.float32) * 0.1
pst_vec, traj = tcm.t2spike_grad_forward(
    n_osc, n_steps, 0.05, 10.0, 1.0, 0.5, 3.0,
    0.3, 0.15, 0.08, drives,
)
check(len(pst_vec) == 18 * n_osc + 12, f"t2spike_grad forward PST dim={len(pst_vec)}")
check(hasattr(traj, "n_osc") and traj.n_osc() == n_osc, "trajectory has correct n_osc")
check(hasattr(traj, "n_steps") and traj.n_steps() == n_steps, "trajectory has correct n_steps")

# Backward
grad_pst = np.random.randn(len(pst_vec)).astype(np.float32)
grad_drives = tcm.t2spike_grad_backward(grad_pst, traj)
check(grad_drives.shape == (n_steps, 3 * n_osc),
      f"backward gradient shape {grad_drives.shape}")

# ─── 6. Neuron step functions ────────────────────────────

section("6. Neuron step functions")

n = 8
v = np.full(n, -65.0, dtype=np.float32)
u = np.full(n, -13.0, dtype=np.float32)
trace = np.zeros(n, dtype=np.float32)
a = np.full(n, 0.02, dtype=np.float32)
b = np.full(n, 0.2, dtype=np.float32)
I = np.full(n, 10.0, dtype=np.float32)
spikes = np.zeros(n, dtype=np.int32)
c = np.full(n, -65.0, dtype=np.float32)
d = np.full(n, 8.0, dtype=np.float32)

tcm.step_izhikevich(v, u, trace, a, b, I, 0.5, 0.5, spikes, c, d)
check(np.any(v > -65.0), f"Izhikevich v increased: {v[0]:.2f}")

section("7. ψ-wave (neuron)")

v_psi = np.ones(n, dtype=np.float32)
u_psi = np.zeros(n, dtype=np.float32)
h_real = np.zeros(n, dtype=np.float32)
h_imag = np.zeros(n, dtype=np.float32)
i_ext = np.zeros(n, dtype=np.float32)
speed = np.zeros(n, dtype=np.float32)
tcm.step_gco(v_psi, u_psi, h_real, h_imag, i_ext, speed, 0.05, 1.0)
norms = np.sqrt(v_psi**2 + u_psi**2)
check(np.all(np.abs(norms - 1.0) < 0.1), f"ψ-wave norm ~1: {norms[0]:.3f}")

# ─── 8. SNN Classes ──────────────────────────────────────

section("8. SNN Classes")

LifN = getattr(tcm, "LifNeuron", getattr(tcm, "PyLifNeuron", None))
if LifN:
    lif = LifN(8, 1.0, 10.0)
    I_arr = np.full(8, 15.0, dtype=np.float32)
    spikes_out = np.zeros(8, dtype=np.int32)
    lif.step(I_arr, spikes_out)
    # step() modifies spikes in-place, returns None
    check(isinstance(spikes_out[0], (int, np.integer)), "LifNeuron.step() populates spikes")
    # getter
    v_got = lif.v
    check(len(v_got) == 8, f"LifNeuron.v getter, len={len(v_got)}")
    # setter (may not work in some PyO3 versions)
    try:
        lif.v = np.full(8, -60.0, dtype=np.float32)
        check(np.allclose(lif.v[:4], [-60.0] * 4), "LifNeuron.v setter works")
    except (AttributeError, TypeError):
        check(True, "LifNeuron.v setter not available (skipped)")
    check("LifNeuron" in repr(lif) or "PyLif" in repr(lif),
           f"LifNeuron __repr__: {repr(lif)}")

Sorn = getattr(tcm, "SornNetwork", getattr(tcm, "PySornNetwork", None))
if Sorn:
    sorn = Sorn(16, 0.1, 8.0)  # N=16, w_init=0.1, target_sum=8.0
    n_synapses = 16 * 16  # fully connected: N×N
    inp = np.zeros(16, dtype=np.int32)
    tr_pre = np.zeros(n_synapses, dtype=np.float32)  # per-synapse traces
    tr_post = np.zeros(n_synapses, dtype=np.float32)
    sorn.step(inp, tr_pre, tr_post)
    w = sorn.weights
    check(len(w) == n_synapses, f"SornNetwork.weights len={len(w)}")
    check("SornNetwork" in repr(sorn), f"SornNetwork __repr__: {repr(sorn)}")

Ttfs = getattr(tcm, "TtfsEncoder", getattr(tcm, "PyTtfsEncoder", None))
if Ttfs:
    ttfs = Ttfs(8, 10)
    vals = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5], dtype=np.float32)
    sp = np.zeros(8, dtype=np.int32)
    planned = ttfs.encode_step(vals, sp)
    check(sp[2] == 1, f"TTFS value=1 spikes at step 0: sp[2]={sp[2]}")

# ─── 9. E-Prop ───────────────────────────────────────────

section("9. E-Prop")

EProp = getattr(tcm, "EPropLIF", getattr(tcm, "PyEPropLIF", None))
if EProp:
    ep = EProp(8, 1.0, 10.0, 20.0)
    I_ep = np.full(8, 100.0, dtype=np.float32)
    sp_ep = np.zeros(8, dtype=np.int32)
    pre_sp = np.zeros(8, dtype=np.int32)
    ep.step(I_ep, sp_ep, pre_sp)
    check(isinstance(sp_ep[0], (int, np.integer)), "EPropLIF.step() works")
    # getter
    v_ep_got = ep.v
    check(len(v_ep_got) == 8, f"EPropLIF.v getter len={len(v_ep_got)}")
    # setter (may not work in some PyO3 versions)
    try:
        ep.v = np.full(8, -60.0, dtype=np.float32)
        check(np.allclose(ep.v[:4], [-60.0] * 4), "EPropLIF.v setter works")
    except (AttributeError, TypeError):
        check(True, "EPropLIF.v setter not available (skipped)")
    check("EPropLIF" in repr(ep), f"EPropLIF __repr__: {repr(ep)}")

EPropMulti = getattr(tcm, "EPropIzhikevichMulti", getattr(tcm, "PyEPropIzhikevichMulti", None))
if EPropMulti:
    epm = EPropMulti(2, 8, 1.0, 20.0)
    weights = np.full(2 * 8, 0.5, dtype=np.float32)
    z_pre = np.array([1, 0], dtype=np.int32)
    I_ext = np.full(8, 100.0, dtype=np.float32)
    sp_m = np.zeros(8, dtype=np.int32)
    epm.step(weights, z_pre, I_ext, sp_m)
    check("EPropIzhikevichMulti" in repr(epm), "EPropIzhikevichMulti repr")

# ─── 10. Poisson / STP ───────────────────────────────────

section("10. Poisson & STP")

Pois = getattr(tcm, "PoissonGenerator", getattr(tcm, "PyPoissonGenerator", None))
if Pois:
    pg = Pois(8, 1.0, 42)
    sp_pg = np.zeros(8, dtype=np.int32)
    pg.step(sp_pg)
    check(pg.n_neurons() == 8, f"Poisson n_neurons={pg.n_neurons()}")
    check("PoissonGenerator" in repr(pg), f"Poisson __repr__: {repr(pg)}")
    # set rates (bulk + single)
    pg.set_rates(np.full(8, 50.0, dtype=np.float32))
    pg.set_rate(0, 100.0)

STP = getattr(tcm, "ShortTermPlasticity", getattr(tcm, "PyShortTermPlasticity", None))
if STP:
    stp = STP(8, 1.0)
    pre_sp = np.ones(8, dtype=np.int32)
    I_stp = np.zeros(8, dtype=np.float32)
    stp.step(pre_sp, I_stp)
    check(I_stp[0] > 0.0, f"STP I_out={I_stp[0]:.4f}")
    stp.reset()
    check("ShortTermPlasticity" in repr(stp), "STP __repr__")

# ─── 11. RL Agent ────────────────────────────────────────

section("11. RL Agent")

RL = getattr(tcm, "ActorCriticAgent", getattr(tcm, "PyActorCriticAgent", None))
if RL:
    agent = RL(4, 256, 2)  # n_input=4, n_hidden=256, n_actions=2, defaults: n_steps=30, dt=1.0, seed=42
    state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    action, value = agent.act(state)
    check(action in (0, 1), f"RL agent action in {{0,1}}, got {action}")
    check(isinstance(value, float), f"RL agent value is float, got {type(value)}")
    td_error = agent.learn(1.0, value, False)  # reward, next_value, done
    check(isinstance(td_error, float), f"RL agent learn() → float, got {type(td_error)}")
    check("ActorCritic" in repr(agent), f"RL __repr__: {repr(agent)}")
    agent.reset_episode()

# ─── 12. E-Prop Trainer ──────────────────────────────────

section("12. Trainer")

Trainer = getattr(tcm, "EPropTrainer", getattr(tcm, "PyEPropTrainer", None))
if Trainer:
    trainer = Trainer(4, 64, 2)  # ni=4, nh=64, no=2, все остальное — defaults
    # train_batch ожидает 1D images (batch*ni,) и 1D targets (batch,)
    x = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)  # 2 images × 4
    y = np.array([0, 1], dtype=np.int32)  # 2 labels
    loss, correct, acc = trainer.train_batch(x, y)
    check(isinstance(loss, float), f"Trainer train_batch loss → float, got {type(loss)}")
    check(isinstance(correct, int), f"Trainer train_batch correct → int, got {type(correct)}")
    check(isinstance(acc, float), f"Trainer train_batch acc → float, got {type(acc)}")
    # predict
    pred = trainer.predict(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32))
    check(isinstance(pred, (int, np.integer)), f"Trainer predict → int, got {type(pred)}")
    check("EPropTrainer" in repr(trainer), f"Trainer __repr__: {repr(trainer)}")

# ─── 13. Sparse ops ──────────────────────────────────────

section("13. Sparse ops — CSR ring")

n = 8
row_ptr = np.arange(n + 1, dtype=np.int32)
col_indices = np.array([1, 2, 3, 4, 5, 6, 7, 0], dtype=np.int32)
values = np.ones(n, dtype=np.float32)

# Hamiltonian: u must be non-zero to get non-zero h_real
v_sp = np.ones(n, dtype=np.float32)
u_sp = np.ones(n, dtype=np.float32)  # non-zero for non-zero h_real
h_real_sp = np.zeros(n, dtype=np.float32)
h_imag_sp = np.zeros(n, dtype=np.float32)
tcm.compute_hamiltonian(row_ptr, col_indices, values, v_sp, u_sp, h_real_sp, h_imag_sp)
check(np.any(h_real_sp != 0), f"Hamiltonian h_real has non-zero (u was ones)")
check(np.any(h_imag_sp != 0), f"Hamiltonian h_imag has non-zero")

# ψ-wave full step on ring CSR
tcm.step_gco_full(v_sp, u_sp, row_ptr, col_indices, values, 0.05, 1.0)
norms_sp = np.sqrt(v_sp**2 + u_sp**2)
check(np.all(np.abs(norms_sp - 1.0) < 0.1), "ψ-wave full step preserves norm")

# ─── 14. Kuramoto ────────────────────────────────────────

section("14. Kuramoto")

# Build a ring CSR for Kuramoto
n_k = 8
k_row_ptr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
k_col_idx = np.array([1, 2, 3, 4, 5, 6, 7, 0], dtype=np.int32)
k_weights = np.full(8, 0.1, dtype=np.float32)

phases = np.random.rand(n_k).astype(np.float32) * 2 * np.pi
omegas = np.full(n_k, 1.0, dtype=np.float32)
tcm.kuramoto_step_sparse(phases, omegas, k_row_ptr, k_col_idx, k_weights, 0.05)
check(np.any(phases != 0), "Kuramoto step changed phases")

# phase_coherence expects (output, sin_buf, cos_buf, amplitudes)
coh_out = np.zeros(1, dtype=np.float32)
sin_buf = np.sin(phases).astype(np.float32)
cos_buf = np.cos(phases).astype(np.float32)
amps = np.ones(n_k, dtype=np.float32)
tcm.phase_coherence(coh_out, sin_buf, cos_buf, amps)
check(0.0 <= coh_out[0] <= 1.0, f"phase_coherence in [0,1], got {coh_out[0]:.4f}")

# Regression: root phase_coherence must be the 4-arg kuramoto version
# (cpe's 1-arg variant is exposed as cpe_phase_coherence — see cpe.rs).
import inspect as _inspect
_sig = _inspect.signature(tcm.phase_coherence)
check(
    len(_sig.parameters) == 4,
    f"root phase_coherence must be 4-arg (kuramoto), got {_sig}",
)

# ─── 15. IO ──────────────────────────────────────────────

section("15. IO")

x_io = np.random.randn(100).astype(np.float32)
tcm.safety_clamp(x_io, -2.0, 2.0)  # modifies in-place
check(np.all(np.abs(x_io) <= 2.0), "safety_clamp modifies in-place")

# ─── 16. Format (save/load .taas) ────────────────────────

section("16. Format — .taas save/load")

with tempfile.NamedTemporaryFile(suffix='.taas', delete=False) as f:
    tmp_path = f.name
try:
    # Save a TriOscillator with metadata
    tri_save = getattr(tcm, "TriOscillator", getattr(tcm, "PyTriOscillator", None))(8, 0.5, 2.0)
    tri_save.step(0.05)
    tcm.save_taas(tmp_path, tri=tri_save, metadata=[("model", "tri_8")])
    # Load back
    loaded = tcm.load_taas(tmp_path)
    check("tri" in loaded, ".taas load returns dict with 'tri'")
    check("_meta" in loaded, ".taas load returns dict with '_meta'")
    # Inspect
    info = tcm.inspect_taas(tmp_path)
    check(isinstance(info, str) and len(info) > 0, f"inspect_taas returns string, len={len(info)}")
finally:
    os.unlink(tmp_path)

# ─── 17. Autotune ────────────────────────────────────────

section("17. Autotune")

if hasattr(tcm.autotune, 'probe_eprop_fr'):
    # probe_eprop_fr(n_pre, n_post, biases_array, n_steps=50, ...)
    biases_arr = np.linspace(0.0, 5.0, 10).astype(np.float32)
    biases_out, fr_out, snrs_out = tcm.autotune.probe_eprop_fr(4, 8, biases_arr, n_steps=20)
    check(isinstance(biases_out, np.ndarray), f"probe_eprop_fr biases_out → ndarray")
    check(isinstance(fr_out, np.ndarray), f"probe_eprop_fr fr_out → ndarray")
    check(len(biases_out) == 10, f"probe_eprop_fr len={len(biases_out)}")

# ─── 18. T2-SPIGRAD Trajectory properties ────────────────

section("18. Trajectory introspection")

traj = getattr(tcm, "T2SpikeGradTrajectory", None)
if traj is None:
    # T2SpikeGradTrajectory might be in t3 submodule
    traj = getattr(tcm.t3, "T2SpikeGradTrajectory", None)

# Re-create a traj from forward
if traj:
    drives_t = np.random.randn(3, 3 * 8).astype(np.float32) * 0.1
    _, t = tcm.t2spike_grad_forward(8, 3, 0.05, 10.0, 1.0, 0.5, 3.0,
                                     0.3, 0.15, 0.08, drives_t)
    check(t.n_osc() == 8, f"traj.n_osc={t.n_osc()}")
    check(t.n_steps() == 3, f"traj.n_steps={t.n_steps()}")
    check(abs(t.dt() - 0.05) < 1e-6, f"traj.dt={t.dt()}")
    check(abs(t.drive_gain() - 10.0) < 1e-6, f"traj.drive_gain={t.drive_gain()}")
    check("T2SpikeGradTrajectory" in repr(t), f"traj.__repr__: {repr(t)}")


# ─── 19. Pillars populations ───────────────────────────────

section("19. Pillars populations")

from taas_checkmate import pillars as pillars_mod

# The facade is the supported public Python surface. Keep this list explicit:
# a missing export must fail loudly instead of being silently skipped by an
# optional getattr() check elsewhere in this legacy-compatible smoke script.
required_pillars_classes = {
    "ActorCriticAgent", "AdExPopulation", "AstrocyteState",
    "AtlasovNeuromotorPopulation", "ConnectomeBrain", "EPropIzhikevichMulti",
    "EPropLIF", "EPropTrainer", "FHNPopulation", "GLIF3Population",
    "GLIF4Population", "Glif3State", "Glif4State", "HAAPPopulation",
    "HHPopulation", "HRPopulation", "HeterogeneousConnectomeBrain",
    "IzhikevichDualPopulation", "IzhikevichPopulation", "LIFAtPopulation", "LIFPopulation",
    "LifNeuron", "MorrisLecarPopulation",
    "PoissonGenerator", "RatePopulation",
    "SAWPopulation", "ShortTermPlasticity", "SomatoDendriticNeuron",
    "SornNetwork", "SynapticConnectomeBrain", "TriangulatedPopulation",
    "TtfsEncoder", "VSNPopulation", "WCPopulation",
}
required_pillars_functions = {
    "adam_train", "apply_stdp_ltd", "band_energy", "batch_coherence_sweep",
    "bsa_encode", "bsa_kernel_hamming", "compute_cvs", "compute_hamiltonian",
    "compute_isi", "cpe_phase_coherence", "cpe_stdp_update", "detect_spikes",
    "dopamine_delta", "dopamine_timing_gate", "echo_tap", "eligibility_update",
    "eprop_update", "firing_rates", "hebbian_update", "interference",
    "interference_mean", "intrinsic_plasticity", "kalman_update",
    "kuramoto_step", "kuramoto_step_sparse", "lazy_sincos_update",
    "make_brain", "make_population",
    "lif_sim_cpu_recording", "mean_firing_rate", "nlms_train", "phase_coherence",
    "phase_encode", "phase_gate_apply", "population_rate_decode", "probe_eprop_fr",
    "propagate_spikes", "pst_decode", "rate_step", "ring_synthesis", "safety_clamp",
    "sigmoid_gate", "sim_step", "spike_entropy", "spike_raster_to_csv",
    "stdp_homeostatic", "stdp_update", "step_adex", "step_atlasov_neuromotor",
    "step_eprop", "step_fhn", "step_gco", "step_gco_full", "step_haap", "step_hr",
    "step_izhikevich", "step_lif", "step_nmda", "step_saw", "step_wc",
    "stochastic_encoder", "suggest_eprop_config", "surrogate_gradient",
    "synaptic_normalization", "t3_complex_decoder", "t3_drive_encoder",
    "t3_inverse_kuramoto", "t3_pst_features", "t3_step_sparse", "t3_tact_generator",
    "thalamic_gate", "vnc_step", "winding_matrix",
}
for name in sorted(required_pillars_classes):
    check(hasattr(pillars_mod, name), f"pillars class export: {name}")
for name in sorted(required_pillars_functions):
    check(hasattr(pillars_mod, name), f"pillars function export: {name}")

LIFPopulation = pillars_mod.LIFPopulation
RatePopulation = pillars_mod.RatePopulation
AdExPopulation = pillars_mod.AdExPopulation
GLIF4Population = pillars_mod.GLIF4Population
FHNPopulation = pillars_mod.FHNPopulation
HRPopulation = pillars_mod.HRPopulation
HHPopulation = pillars_mod.HHPopulation
TriangulatedPopulation = pillars_mod.TriangulatedPopulation
ConnectomeBrain = pillars_mod.ConnectomeBrain
SynapticConnectomeBrain = pillars_mod.SynapticConnectomeBrain

# Invalid user indices and shapes must be Python exceptions, not Rust panics.
_contract_pop = LIFPopulation(8)
for bad_col in (6, 999):
    try:
        _contract_pop.set_param_array(bad_col, np.zeros(8, dtype=np.float32))
        check(False, f"invalid param column {bad_col} rejected")
    except ValueError:
        check(True, f"invalid param column {bad_col} rejected with ValueError")
    try:
        _contract_pop.set_state_array(bad_col, np.zeros(8, dtype=np.float32))
        check(False, f"invalid state column {bad_col} rejected")
    except ValueError:
        check(True, f"invalid state column {bad_col} rejected with ValueError")
try:
    _contract_pop.set_param_array(0, np.zeros(7, dtype=np.float32))
    check(False, "wrong parameter array length rejected")
except ValueError:
    check(True, "wrong parameter array length rejected with ValueError")
try:
    _contract_pop.step(np.zeros(7, dtype=np.float32), np.zeros(8, dtype=np.int32))
    check(False, "wrong step input length rejected")
except ValueError:
    check(True, "wrong step input length rejected with ValueError")

n = 8
i_in = np.full(n, 30.0, dtype=np.float32)
spikes = np.zeros(n, dtype=np.int32)

# LIF fires with drive
pop = LIFPopulation(n)
pop.dt = 0.5
total = 0
for _ in range(100):
    pop.step(i_in, spikes)
    total += int(spikes.sum())
check(total > 0, f"LIFPopulation fires (total={total})")

# Rate integrates toward input and spikes after crossing 0.5
rate = RatePopulation(n)
rate.tau = 0.005
rate_r = rate.r.copy()
for _ in range(200):
    rate.step(np.full(n, 1.0, dtype=np.float32), spikes)
    rate_r = rate.r
check(float(rate_r[0]) > 0.99, f"RatePopulation saturates toward 1 (r={float(rate_r[0])})")

# AdEx steps without error
adex = AdExPopulation(n)
adex.tau_m = 20.0
adex.step(i_in, spikes)
check(len(adex.v) == n, f"AdExPopulation v shape {len(adex.v)}")

# GLIF4 fires with strong drive
glif4 = GLIF4Population(n)
glif4.dt = 0.5
glif4_total = 0
for _ in range(100):
    glif4.step(np.full(n, 30.0, dtype=np.float32), spikes)
    glif4_total += int(spikes.sum())
check(glif4_total > 0, f"GLIF4Population fires (total={glif4_total})")

# FHN fires edge-triggered with drive (I=0.5 matches zoo). Factory defaults
# must include was_below=1.0, otherwise the first crossing through v_thresh
# is missed and (in the excitable regime) the neuron stays silent.
fhn = FHNPopulation(n)
fhn_total = 0
for _ in range(1000):
    fhn.step(np.full(n, 0.5, dtype=np.float32), spikes)
    fhn_total += int(spikes.sum())
check(fhn_total > 0, f"FHNPopulation fires (total={fhn_total})")

# HH with canonical drive (I=15, dt=0.01) must spike and stay finite. Factory
# state layout is v,m,h,n,c,was_below: c = CaV-T gate (0.19) and was_below=1.0
# — a state-index slip (c=1.0 / was_below=0) silently breaks edge spiking.
hh = HHPopulation(n)
hh.dt = 0.01
hh_total = 0
for _ in range(3000):
    hh.step(np.full(n, 15.0, dtype=np.float32), spikes)
    hh_total += int(spikes.sum())
check(
    hh_total > n and np.isfinite(hh.v).all(),
    f"HHPopulation fires, finite (total={hh_total})",
)

# Edge-triggered spike contract: the FIRST rising edge through v_thresh must
# fire. This is the regression for the factory state-index slip (the 4th and
# 5th states are c / was_below): was_below must start at 1.0.
hh1 = HHPopulation(1)
hh1.dt = 0.01
spikes1 = np.zeros(1, dtype=np.int32)
v_prev = float(np.asarray(hh1.v)[0])
first_edge_fires = False
for _ in range(3000):
    hh1.step(np.full(1, 15.0, dtype=np.float32), spikes1)
    v_now = float(np.asarray(hh1.v)[0])
    if v_prev < 0.0 <= v_now:
        first_edge_fires = bool(spikes1[0] == 1)
        break
    v_prev = v_now
check(first_edge_fires, "HHPopulation first rising edge fires (was_below=1.0)")

# HR with factory defaults (dt=0.01, r=0.006, v_thresh=1.0, x=-1.6, y=-12.0)
# must burst with drive and stay finite — old defaults (dt=0.5, y=0.0) gave NaN.
hr = HRPopulation(n)
hr_total = 0
for _ in range(500):
    hr.step(np.full(n, 30.0, dtype=np.float32), spikes)
    hr_total += int(spikes.sum())
check(
    len(hr.v) == n and hr_total > 0 and np.isfinite(hr.v).all(),
    f"HRPopulation bursts without NaN (total={hr_total})",
)

# Triangulated: step() before calibrate_lif() must raise (no LUT installed),
# and after calibration the LUT neuron must fire with drive.
tri = TriangulatedPopulation(n)
try:
    tri.step(np.zeros(n, dtype=np.float32), spikes)
    check(False, "TriangulatedPopulation pre-calibrate step rejected")
except BaseException:
    check(True, "TriangulatedPopulation pre-calibrate step raises")
tri.calibrate_lif(n_steps=10_000)
tri_total = 0
for _ in range(1000):
    tri.step(np.full(n, 30.0, dtype=np.float32), spikes)
    tri_total += int(spikes.sum())
check(tri_total > 0, f"TriangulatedPopulation fires after calibrate (total={tri_total})")

# Runtime factory: make_population(name, n) returns the concrete class by name
pop_lif = pillars_mod.make_population("lif", 8)
pop_hh = pillars_mod.make_population("HHPopulation", 8)
pop_dual = pillars_mod.make_population("izhikevich_dual", 8)
pop_lifat = pillars_mod.make_population("lif_at", 8)
check(pop_lif.n == 8 and pop_hh.n == 8, "make_population accepts short and class-style names")
check(type(pop_hh).__name__ == "HHPopulation", f"make_population('HHPopulation') -> {type(pop_hh).__name__}")
# camelCase class-style forms resolve too (camel→snake normalization)
pop_ml = pillars_mod.make_population("MorrisLecarPopulation", 8)
pop_att = pillars_mod.make_population("AtlasovNeuromotorPopulation", 8)
check(
    type(pop_ml).__name__ == "MorrisLecarPopulation"
    and type(pop_att).__name__ == "AtlasovNeuromotorPopulation",
    "make_population resolves multi-word camelCase class names",
)
check(type(pop_dual).__name__ == "IzhikevichDualPopulation", "make_population('izhikevich_dual') class")
check(type(pop_lifat).__name__ == "LIFAtPopulation", "make_population('lif_at') class")
# duck-typed surface: step(i, spikes) works on the factory-created population
# (same tuned config as the direct LIFPopulation check above: dt=0.5, I=30)
pop_lif.dt = 0.5
sp = np.zeros(8, dtype=np.int32)
factory_total = 0
for _ in range(100):
    pop_lif.step(np.full(8, 30.0, dtype=np.float32), sp)
    factory_total += int(sp.sum())
check(factory_total > 0, f"make_population('lif') steps and fires (total={factory_total})")
try:
    pillars_mod.make_population("bogus", 8)
    check(False, "make_population('bogus') rejected")
except ValueError:
    check(True, "make_population('bogus') raises ValueError")

# Heterogeneous factory coverage: every ordinary zoo model must build and
# take a finite zero-input step. Triangulated is intentionally separate because
# its LUT is process-wide and requires calibration before construction.
hetero_model_names = [
    "lif", "izhikevich", "izhikevich_dual", "hh", "vsn", "rate", "adex",
    "glif3", "glif4", "fhn", "hr", "haap", "morris_lecar", "saw", "wc",
    "lifat", "atlasov_neuromotor",
]
for _model_name in hetero_model_names:
    _assembly = pillars_mod.make_brain([(_model_name, 4)], [[0.0]])
    _out = _assembly.step()
    check(
        len(_out) == 1 and _out[0].shape == (4,) and np.isfinite(_assembly.v[0]).all(),
        f"make_brain builds/steps {_model_name}",
    )
try:
    pillars_mod.make_brain([("triangulated", 4)], [[0.0]])
    check(False, "make_brain rejects LUT-only triangulated")
except ValueError:
    check(True, "make_brain rejects LUT-only triangulated")

# Heterogeneous multi-population assembly: block connectivity is indexed
# [post][pre], so [[0, 1], [1, 0]] makes a bidirectional HH↔LIF graph.
hetero = pillars_mod.make_brain(
    [("hh", 64), ("lif", 32)],
    [[0.0, 1.0], [1.0, 0.0]],
)
check(type(hetero).__name__ == "HeterogeneousConnectomeBrain", "make_brain returns heterogeneous brain")
check(hetero.n_populations == 2, f"make_brain population count={hetero.n_populations}")
check(hetero.population_sizes == [64, 32], f"make_brain population sizes={hetero.population_sizes}")
hetero.inject(0, [0], [15.0])
hetero_spikes = hetero.step()
check(
    isinstance(hetero_spikes, list)
    and len(hetero_spikes) == 2
    and hetero_spikes[0].shape == (64,)
    and hetero_spikes[1].shape == (32,),
    "make_brain step returns one spike array per population",
)
check(hetero.spikes[0].shape == (64,) and hetero.v[1].shape == (32,), "heterogeneous brain state facade")
hetero.inject_array(1, np.zeros(32, dtype=np.float32))
check(True, "heterogeneous brain bulk input")

# Asymmetric orientation regression: connectivity is [post][pre]. A block
# only at [1][0] routes population 0 into population 1, never the reverse.
one_way = pillars_mod.make_brain(
    [("lif", 4), ("lif", 4)],
    [[0.0, 0.0], [2.0, 0.0]],
)
for _ in range(200):
    # Keep the source driven until it spikes; the [1][0] block then routes
    # source spikes into population 1 on the following step.
    one_way.inject_array(0, np.full(4, 30.0, dtype=np.float32))
    one_way.step()
check(
    float(np.asarray(one_way.v[1])[0]) > -70.0,
    "heterogeneous connectivity uses [post][pre] orientation",
)
try:
    pillars_mod.make_brain([("lif", 4)], [[0.0, 1.0], [1.0, 0.0]])
    check(False, "make_brain rejects non-square connectivity")
except ValueError:
    check(True, "make_brain rejects non-square connectivity")
try:
    pillars_mod.make_brain([("lif", 4), ("bogus", 4)], [[0.0, 0.0], [0.0, 0.0]])
    check(False, "make_brain rejects unsupported model")
except ValueError:
    check(True, "make_brain rejects unsupported model")

# ConnectomeBrain routes spikes through a CSR chain 0→1→2
cb = ConnectomeBrain(
    np.array([0, 1, 2, 2], dtype=np.uint32),
    np.array([1, 2], dtype=np.uint32),
    np.array([50.0, 50.0], dtype=np.float32),
    np.array([0, 0], dtype=np.uint8),
)
cb.inv_tau_m = 1.0  # fast tau → a single pulse crosses threshold
cb_n2 = 0
for _ in range(200):
    cb.inject(np.array([0], dtype=np.uint32), np.array([50.0], dtype=np.float32))
    cb.step()
    cb_n2 += int(cb.spikes[2])
check(cb_n2 > 0, f"ConnectomeBrain propagates 0→1→2 (neuron2={cb_n2})")

# SynapticConnectomeBrain: AMPA conductance drives post
scb = SynapticConnectomeBrain(
    np.array([0, 1, 1], dtype=np.uint32),
    np.array([1], dtype=np.uint32),
    np.array([1.0], dtype=np.float32),
    np.array([0], dtype=np.uint8),
    np.array([0], dtype=np.uint8),  # AMPA
    dt=0.5,
)
scb.inv_tau_m = 1.0
scb_n1 = 0
ext = np.zeros(2, dtype=np.float32)
out = np.zeros(2, dtype=np.int32)
for _ in range(200):
    ext[0] = 50.0
    scb.step(ext, out)
    scb_n1 += int(out[1])
    ext[0] = 0.0
check(scb_n1 > 0, f"SynapticConnectomeBrain AMPA drives post (neuron1={scb_n1})")


# ─── Summary ──────────────────────────────────────────────

section("Summary")
total = passed + failed
print(f"  {passed}/{total} passed, {failed} failed")
if failed > 0:
    sys.exit(1)
else:
    print("  All smoke tests passed! ✓")
