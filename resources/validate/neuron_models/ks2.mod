: ks2.mod -- 2-state kinetic scheme (Na-m gate kinetics) for Markov engine parity
: State m0 (closed) <-> m1 (open), rates matching pillars na_m_alpha/na_m_beta:
:   a = 0.1 * vtrap(-(v+40), 10)
:   b = 4 * exp(-(v+65)/18)
: Used as a K-like channel (ek reversal) so opening hyperpolarizes; subthreshold.

NEURON {
    SUFFIX ks2
    USEION k READ ek WRITE ik
    RANGE gbar, m1
}

UNITS {
    (mV) = (millivolt)
    (S)  = (siemens)
}

PARAMETER {
    gbar = 0.001 (S/cm2)
}

ASSIGNED {
    ik (mA/cm2)
    ek (mV)
    a (/ms)
    b (/ms)
}

STATE {
    m0
    m1
}

INITIAL {
    rates(v)
    m0 = b / (a + b)
    m1 = a / (a + b)
}

BREAKPOINT {
    SOLVE kin METHOD sparse
    ik = gbar * m1 * (v - ek)
}

KINETIC kin {
    rates(v)
    ~ m0 <-> m1 (a, b)
}

PROCEDURE rates(v (mV)) {
    a = 0.1 * vtrap(-(v + 40), 10)
    b = 4 * exp(-(v + 65) / 18)
}

FUNCTION vtrap(x, y) {
    if (fabs(x / y) < 1e-6) {
        vtrap = y * (1 - x / y / 2)
    } else {
        vtrap = x / (exp(x / y) - 1)
    }
}
