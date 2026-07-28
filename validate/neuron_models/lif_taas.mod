TITLE LIF (taas_checkmate version)
: Leaky Integrate-and-Fire with Exponential Euler
: NEURON 9.0 compatible

NEURON {
    SUFFIX lif_taas
    NONSPECIFIC_CURRENT i
    RANGE tau_m, v_rest, v_thresh, v_reset, cm
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    tau_m = 10 (ms)
    v_rest = -65 (mV)
    v_thresh = -54 (mV)
    v_reset = -65 (mV)
    cm = 1 (uF/cm2)
}

STATE { v_mem }

ASSIGNED {
    i (mA/cm2)
    g_leak (S/cm2)
}

INITIAL {
    v_mem = v_rest
    g_leak = cm * 0.001 / tau_m
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    i = g_leak * (v_mem - v_rest)
}

DERIVATIVE states {
    v_mem' = -(v_mem - v_rest)/tau_m + i*1000/(cm*0.001)
}
