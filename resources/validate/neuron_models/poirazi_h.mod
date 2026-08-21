TITLE Poirazi 2003 CA1 Ih (HCN-like h current) - h
: Published Poirazi-Brannon-Mel (2003) CA1 `h` channel (NeuroML2 export),
: custom LEMS ComponentTypes lowered to closed form:
:   h_inf = 1 - 1/(1+exp(-(v - midpoint)/scale))
:         = 1/(1+exp((v - midpoint)/scale))          midpoint=-82, scale=8.8
:   tau_h = 2*(1/(exp((v+145)/-17.5) + exp((v+16.8)/16.5)) + 5) ms   v < -30
:   tau_h = 1 ms                                                    v >= -30
: Opens on HYPERPOLARIZATION (classic Ih sag). No Q10 (see poirazi_km.mod).

NEURON {
    SUFFIX poirazi_h
    NONSPECIFIC_CURRENT i
    RANGE gbar, eh
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gbar = 1.872e-5 (S/cm2)
    eh = -10 (mV)
}

STATE { h }

ASSIGNED { i (mA/cm2) }

INITIAL {
    h = hss(v)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    i = gbar * h * (v - eh)
}

DERIVATIVE states {
    h' = (hss(v) - h) / htau(v)
}

FUNCTION hss(v) {
    hss = 1 / (1 + exp((v + 82) / 8.8))
}

FUNCTION htau(v) {
    if (v < -30) {
        htau = 2 * (1 / (exp((v + 145) / -17.5) + exp((v + 16.8) / 16.5)) + 5)
    } else {
        htau = 1
    }
}
