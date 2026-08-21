TITLE Poirazi 2003 CA1 nadend (generated from NeuroML2)
: Auto-generated from the published NeuroML2 channel by
: gen_poirazi_mods.py (mini-jNeuroML). Kinetics match the pillars
: engine and the NML ComponentTypes exactly.

NEURON {
    SUFFIX nadend_1
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

STATE { m h s }

ASSIGNED {
    g (mho/cm2)
    i (mA/cm2)
    minf
    mtau
    hinf
    htau
    sinf
    stau
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
    s = sinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar
    g = g * m^(2)
    g = g * h^(1)
    g = g * s^(1)
    i = g * (v - erev)
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m)/mtau
    h' = (hinf - h)/htau
    s' = (sinf - s)/stau
}

PROCEDURE rates(v (mV)) {
    LOCAL is, ts
    minf = 1.0/(1+exp(-(v--40.0)/3.0))
    mtau = 0.05/(1.0)
    hinf = 1.0/(1+exp(-(v--45.0)/-3.0))
    htau = 0.5/(1.0)
    is = ((1+(1.0*exp(((((v*0.001)/(0.001))+60)/(2)))))/((1+exp(((((v*0.001)/(0.001))+60)/(2))))))
        if ((((exp((1e-3*12*0.2*(((v*0.001) / 0.001)+60)*9.648e4) / (8.315*((308.15) / 1.0)))) / (3e-4 + 3e-4*(1 + (exp((1e-3*12*(((v*0.001) / 0.001)+60)*9.648e4) / (8.315*((308.15) / 1.0))))))) / (1.0)) < 3) { ts = ((3*(1.0))*1.0) }
        else { ts = ((exp((((((1e-3*12)*0.2)*(((v*0.001)/(0.001))+60))*9.648e4)/((8.315*((308.15)/(1.0))))))/((3e-4+(3e-4*(1+exp(((((1e-3*12)*(((v*0.001)/(0.001))+60))*9.648e4)/((8.315*((308.15)/(1.0)))))))))))*1.0) }
    sinf = is
    stau = ts/(1.0)
}
