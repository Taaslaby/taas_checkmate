TITLE Poirazi 2003 CA1 leak_chan (generated from NeuroML2)
: Auto-generated from the published NeuroML2 channel by
: gen_poirazi_mods.py (mini-jNeuroML). Kinetics match the pillars
: engine and the NML ComponentTypes exactly.

NEURON {
    SUFFIX leak_chan
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

ASSIGNED {
    g (mho/cm2)
    i (mA/cm2)
}

INITIAL {
}

BREAKPOINT {
    g = gbar
    i = g * (v - erev)
}

PROCEDURE rates(v (mV)) {
}
