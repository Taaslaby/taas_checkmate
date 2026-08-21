TITLE Poirazi 2003 CA1 h (generated from NeuroML2)
: Auto-generated from the published NeuroML2 channel by
: gen_poirazi_mods.py (mini-jNeuroML). Kinetics match the pillars
: engine and the NML ComponentTypes exactly.

NEURON {
    SUFFIX h
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
    g = g * n^(1)
    i = g * (v - erev)
}

DERIVATIVE states {
    rates(v)
    n' = (ninf - n)/ntau
}

PROCEDURE rates(v (mV)) {
    LOCAL in, tn
    in = (1-(1/((1+exp((0-(((v/(1.0))-((-90.0)/(1.0)))/((8.5/(1.0))))))))))
        if ((v / 1.0) < -30) { tn = ((2*((1/((exp((((v/(1.0))+145)/((-17.5))))+exp((((v/(1.0))+16.8)/(16.5))))))+5))*1.0) }
        else { tn = (1*1.0) }
    ninf = in
    ntau = tn/(1.0)
}
