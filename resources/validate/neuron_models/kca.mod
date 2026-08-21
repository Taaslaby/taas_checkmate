TITLE Poirazi 2003 CA1 kca (generated from NeuroML2)
: Auto-generated from the published NeuroML2 channel by
: gen_poirazi_mods.py (mini-jNeuroML). Kinetics match the pillars
: engine and the NML ComponentTypes exactly.

NEURON {
    SUFFIX kca
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

STATE { m }

ASSIGNED {
    g (mho/cm2)
    i (mA/cm2)
    minf
    mtau
}

INITIAL {
    rates(v)
    m = minf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar
    g = g * m^(3)
    i = g * (v - erev)
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m)/mtau
}

PROCEDURE rates(v (mV)) {
    LOCAL im, tm
    im = ((((cai/(1.0))/(0.025))*((cai/(1.0))/(0.025)))/((1+(((cai/(1.0))/(0.025))*((cai/(1.0))/(0.025))))))
        if (((1 / (0.03*(1 + (((cai / 1.0)/0.025)*((cai / 1.0)/0.025))))) / (4.171167510947728)) < 0.5) { tm = ((0.5*(4.171167510947728))*1.0) }
        else { tm = ((1/((0.03*(1+(((cai/(1.0))/(0.025))*((cai/(1.0))/(0.025)))))))*1.0) }
    minf = im
    mtau = tm/(4.171167510947728)
}
