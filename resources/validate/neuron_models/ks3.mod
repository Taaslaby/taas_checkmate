: ks3.mod -- 3-state kinetic scheme (closed <-> open <-> inactivated) for the
: Markov engine parity. Rates match pillars Na-m / Na-h gates:
:   m0 <-> m1 : a1 = 0.1*vtrap(-(v+40),10)  b1 = 4*exp(-(v+65)/18)
:   m1 <-> m2 : a2 = 0.07*exp(-(v+65)/20)   b2 = 1/(1+exp(-(v+35)/10))
: Open state is m1; used as a K-like channel (ek reversal) so it stays
: subthreshold.

NEURON {
    SUFFIX ks3
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
    a1 (/ms)
    b1 (/ms)
    a2 (/ms)
    b2 (/ms)
}

STATE {
    m0
    m1
    m2
}

INITIAL {
    rates(v)
    : exact stationary distribution of the chain (detailed balance), matching
    : pillars `markov_steady_state` at the initial potential.
    m0 = 1.0 / (1.0 + a1 / b1 + a1 * a2 / (b1 * b2))
    m1 = (a1 / b1) * m0
    m2 = (a1 * a2 / (b1 * b2)) * m0
}

BREAKPOINT {
    SOLVE kin METHOD sparse
    ik = gbar * m1 * (v - ek)
}

KINETIC kin {
    rates(v)
    ~ m0 <-> m1 (a1, b1)
    ~ m1 <-> m2 (a2, b2)
}

PROCEDURE rates(v (mV)) {
    a1 = 0.1 * vtrap(-(v + 40), 10)
    b1 = 4 * exp(-(v + 65) / 18)
    a2 = 0.07 * exp(-(v + 65) / 20)
    b2 = 1 / (1 + exp(-(v + 35) / 10))
}

FUNCTION vtrap(x, y) {
    if (fabs(x / y) < 1e-6) {
        vtrap = y * (1 - x / y / 2)
    } else {
        vtrap = x / (exp(x / y) - 1)
    }
}
