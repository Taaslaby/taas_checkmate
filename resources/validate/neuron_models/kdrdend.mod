TITLE Poirazi 2003 CA1 kdrdend (generated from NeuroML2)
: Auto-generated from the published NeuroML2 channel by
: gen_poirazi_mods.py (mini-jNeuroML). Kinetics match the pillars
: engine and the NML ComponentTypes exactly.

NEURON {
    SUFFIX kdrdend
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

STATE { n }

ASSIGNED {
    g (mho/cm2)
    i (mA/cm2)
    ninf
    ntau
}

INITIAL {
    rates(v)
    n = ninf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar
    g = g * n^(2)
    i = g * (v - erev)
}

DERIVATIVE states {
    rates(v)
    n' = (ninf - n)/ntau
}

PROCEDURE rates(v (mV)) {
    ninf = 1.0/(1+exp(-(v--42.0)/2.0))
    ntau = 2.2/(1.0)
}
