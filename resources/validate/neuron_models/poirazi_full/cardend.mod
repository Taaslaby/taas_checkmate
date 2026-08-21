TITLE Poirazi 2003 CA1 cardend (generated from NeuroML2)
: Auto-generated from the published NeuroML2 channel by
: gen_poirazi_mods.py (mini-jNeuroML). Kinetics match the pillars
: engine and the NML ComponentTypes exactly.

NEURON {
    SUFFIX cardend
    NONSPECIFIC_CURRENT i
    RANGE gbar, erev
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (mM) = (milli/liter)
    FARADAY = 96520 (coul)
    R = 8.3134 (joule/degK)
    KTOMV = .0853 (mV/degC)
}

PARAMETER {
    gbar = 0 (S/cm2)
    erev = 0 (mV)
}

STATE { m h }

ASSIGNED {
    g (mho/cm2)
    i (mA/cm2)
    minf
    mtau
    hinf
    htau
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar
    g = g * m^(3)
    g = g * h^(1)
    i = g * (v - erev)
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m)/mtau
    h' = (hinf - h)/htau
}

PROCEDURE rates(v (mV)) {
    minf = 1.0/(1+exp(-(v--48.5)/3.0))
    mtau = 50.0/(1.0)
    hinf = 1.0/(1+exp(-(v--53.0)/-1.0))
    htau = 5.0/(1.0)
}
