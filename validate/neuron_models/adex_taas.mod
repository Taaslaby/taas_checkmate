TITLE Adaptive Exponential (taas_checkmate version)
: Brette & Gerstner 2005, J. Neurophysiol.
: NEURON 9.0 compatible

NEURON {
    SUFFIX adex_taas
    NONSPECIFIC_CURRENT i
    RANGE gl, el, vt, delta_t, a, b, tau_w, vr, cm
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gl = 0.01 (S/cm2)
    el = -70 (mV)
    vt = -50 (mV)
    delta_t = 2 (mV)
    a = 0.001 (S/cm2)
    b = 0.005
    tau_w = 100 (ms)
    vr = -70 (mV)
    cm = 1 (uF/cm2)
}

STATE { v_adex w_adex }

ASSIGNED {
    i (mA/cm2)
}

INITIAL {
    v_adex = el
    w_adex = a * (v_adex - el)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    i = gl*(v_adex - el) - gl*delta_t*exp((v_adex - vt)/delta_t) + w_adex
}

DERIVATIVE states {
    v_adex' = (-gl*(v_adex - el) + gl*delta_t*exp((v_adex - vt)/delta_t) - w_adex + i*1000*cm) / (cm*0.001)
    w_adex' = (a*(v_adex - el) - w_adex) / tau_w
}
