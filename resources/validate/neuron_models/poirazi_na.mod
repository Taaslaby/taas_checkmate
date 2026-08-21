TITLE Poirazi 2003 CA1 Na channel (soma) - m^2 h
: Published Poirazi-Brannon-Mel (2003) CA1 pyramidal Na+ channel
: (Bartlett Mel-modified HH, soma/axon variant), NeuroML2 export:
:   m: tau = 0.05 ms, m_inf = 1/(1+exp(-(v+44)/3))          (instances 2)
:   h: tau = 1.0 ms, h_inf = 1/(1+exp((v+49)/3.5))          (instances 1)
: The slow-inactivation gate `s` (no_inact=1) is identically 1 - omitted.
: No Q10: validation runs at celsius=6.3 where pillars' global Q10 = 1.

NEURON {
    SUFFIX poirazi_na
    NONSPECIFIC_CURRENT i
    RANGE gbar, ena
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gbar = 0.007 (S/cm2)
    ena = 50 (mV)
}

STATE { m h }

ASSIGNED { i (mA/cm2) }

INITIAL {
    m = mss(v)
    h = hss(v)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    i = gbar * m * m * h * (v - ena)
}

DERIVATIVE states {
    m' = (mss(v) - m) / 0.05
    h' = (hss(v) - h) / 1.0
}

FUNCTION mss(v) {
    mss = 1 / (1 + exp(-(v + 44) / 3))
}

FUNCTION hss(v) {
    hss = 1 / (1 + exp((v + 49) / 3.5))
}
