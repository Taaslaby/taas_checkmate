TITLE Adaptive Exponential (taas_checkmate) - normalized parity form
: Brette & Gerstner 2005, J. Neurophysiol.
: NEURON 9.0 compatible.
:
: This mechanism integrates the *normalized* AdEx used by pillars:
:   v' = (v_rest - v + delta_t*exp((v - v_thresh)/delta_t) - w + i_inj) * inv_tau_m
:   w' = (a*(v - v_rest) - w) * inv_tau_w
: All voltages/currents are in mV (i_inj is the injected current already
: divided by gL), time is in ms. The spike reset (v >= v_peak -> v_reset,
: w += b, refractory) is applied by the driver (gen_adex_ref.py), because the
: reset needs to be controlled step-by-step for exact parity with pillars.

NEURON {
    SUFFIX adex_taas
    RANGE i_inj, inv_tau_m, inv_tau_w, v_rest, v_thresh, v_reset, delta_t, a, b, tau_ref, v_peak
    RANGE v_mem, w_adex
}

PARAMETER {
    inv_tau_m = 0.107 (/ms)
    inv_tau_w = 0.00694 (/ms)
    v_rest = -70.6 (mV)
    v_thresh = -50.4 (mV)
    v_reset = -70.6 (mV)
    delta_t = 2.0 (mV)
    a = 0.133 (1)
    b = 2.68 (mV)
    tau_ref = 2.0 (ms)
    v_peak = 0.0 (mV)
    i_inj = 0.0 (mV)
}

STATE {
    v_mem (mV)
    w_adex (mV)
}

INITIAL {
    v_mem = v_rest
    w_adex = 0.0
}

BREAKPOINT {
    SOLVE states METHOD derivimplicit
}

DERIVATIVE states {
    v_mem' = (v_rest - v_mem + delta_t*exp((v_mem - v_thresh)/delta_t) - w_adex + i_inj) * inv_tau_m
    w_adex' = (a*(v_mem - v_rest) - w_adex) * inv_tau_w
}
