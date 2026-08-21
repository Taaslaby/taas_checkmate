TITLE Poirazi 2003 CA1 cagk (generated from NeuroML2)
: Auto-generated from the published NeuroML2 channel by
: gen_poirazi_mods.py (mini-jNeuroML). Kinetics match the pillars
: engine and the NML ComponentTypes exactly.

NEURON {
    SUFFIX cagk
    USEION ca READ cai
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
    cai = 1e-4 (mM)
}

STATE { o }

ASSIGNED {
    g (mho/cm2)
    i (mA/cm2)
    oinf
    otau
}

INITIAL {
    rates(v)
    o = oinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar
    g = g * o^(1)
    i = g * (v - erev)
}

DERIVATIVE states {
    rates(v)
    o' = (oinf - o)/otau
}

PROCEDURE rates(v (mV)) {
    LOCAL ao, bo
    ao = ((0.48/((1.+((0.18*exp(((-(((2.0*0.84)*96485.3)/((8.3144621*(308.15)))))*v)))/((cai/(1.0)))))))/(1.0))
        if ((0.011 * exp(-((2.0 * 96485.3) / (8.3144621 * (308.15))) * v)) != 0.) { bo = ((0.28/((1.+((cai/(1.0))/((0.011*exp(((-((2.0*96485.3)/((8.3144621*(308.15)))))*v))))))))/(1.0)) }
        else if ((0.011 * exp(-((2.0 * 96485.3) / (8.3144621 * (308.15))) * v)) == 0.) { bo = (0.28/(1.0)) }
        else { bo = 0 }
    oinf = ao/(ao+bo)
    otau = 1/((ao+bo)*(1.0))
}
