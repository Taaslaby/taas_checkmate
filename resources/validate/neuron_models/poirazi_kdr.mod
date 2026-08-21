TITLE Poirazi 2003 CA1 delayed rectifier K channel (soma) - n^2
: Published Poirazi-Brannon-Mel (2003) CA1 pyramidal Kdr channel
: (Bartlett Mel-modified HH, soma/axon variant), NeuroML2 export:
:   n: tau = 3.5 ms, n_inf = 1/(1+exp(-(v+46.3)/3))          (instances 2)
: No Q10: validation runs at celsius=6.3 where pillars' global Q10 = 1.

NEURON {
    SUFFIX poirazi_kdr
    NONSPECIFIC_CURRENT i
    RANGE gbar, ek
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gbar = 0.0014 (S/cm2)
    ek = -77 (mV)
}

STATE { n }

ASSIGNED { i (mA/cm2) }

INITIAL {
    n = nss(v)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    i = gbar * n * n * (v - ek)
}

DERIVATIVE states {
    n' = (nss(v) - n) / 3.5
}

FUNCTION nss(v) {
    nss = 1 / (1 + exp(-(v + 46.3) / 3))
}
