TITLE Izhikevich (taas_checkmate version)
: Izhikevich 2003, IEEE Trans. Neural Networks
: NEURON 9.0 compatible

NEURON {
    SUFFIX izh_taas
    NONSPECIFIC_CURRENT i
    RANGE a, b, c, d, v_init, u_init
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
}

PARAMETER {
    a = 0.02 (/ms)
    b = 0.2 (/ms)
    c = -65 (mV)
    d = 8
    v_init = -65 (mV)
    u_init = -13
}

STATE { v_izh u_izh }

ASSIGNED {
    i (mA/cm2)
}

INITIAL {
    v_izh = v_init
    u_izh = u_init
}

BREAKPOINT {
    SOLVE states METHOD derivimplicit
    i = 0
}

DERIVATIVE states {
    v_izh' = 0.04*v_izh*v_izh + 5*v_izh + 140 - u_izh + i*1000
    u_izh' = a*(b*v_izh - u_izh)
}
