TITLE Poirazi 2003 CA1 cat (generated from NeuroML2)
: Auto-generated from the published NeuroML2 channel by
: gen_poirazi_mods.py (mini-jNeuroML). Kinetics match the pillars
: engine and the NML ComponentTypes exactly.

NEURON {
    SUFFIX cat
    USEION ca READ cai, cao WRITE ica
    RANGE gbar
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
    cai = 1e-4 (mM)
    cao = 2 (mM)
}

STATE { m h }

ASSIGNED {
    g (mho/cm2)
    ica (mA/cm2)
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
    g = g * m^(2)
    g = g * h^(1)
    g = g * ((0.001/((0.001+(cai/(1.0))))))^(1)
    ica = g * ghk(v, cai, cao)
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m)/mtau
    h' = (hinf - h)/htau
}

PROCEDURE rates(v (mV)) {
    minf = (0.209/-3.8)*vtrap(-(v--27.01), -3.8)/((0.209/-3.8)*vtrap(-(v--27.01), -3.8)+0.94*exp((v--63.01)/-17.0))
    mtau = 1/(((0.209/-3.8)*vtrap(-(v--27.01), -3.8)+0.94*exp((v--63.01)/-17.0))*(1.0))
    hinf = ((1.6e-4*exp(((((v*0.001)/(0.001))+57)/((-19)))))/(((1.6e-4*exp(((((v*0.001)/(0.001))+57)/((-19)))))+(1/((1+exp(((((v*0.001)/(0.001))-15)/((-10))))))))))
    htau = (((1/(0.68))*((1.6e-4*exp(((((v*0.001)/(0.001))+57)/((-19)))))+(1/((1+exp(((((v*0.001)/(0.001))-15)/((-10)))))))))*1.0)/(1.0)
}

FUNCTION ghk(v(mV), ci(mM), co(mM)) (mV) {
    LOCAL nu, f
    f = KTF(celsius)/2
    nu = v/f
    ghk = -f*(1. - (ci/co)*exp(nu))*efun(nu)
}

FUNCTION KTF(celsius (degC)) (mV) {
    KTF = ((25./293.15)*(celsius + 273.15))
}

FUNCTION efun(z) {
    if (fabs(z) < 1e-4) {
        efun = 1 - z/2
    } else {
        efun = z/(exp(z) - 1)
    }
}

FUNCTION vtrap(x, y) {
    if (fabs(x/y) < 1e-6) {
        vtrap = y*(1 - x/y/2)
    } else {
        vtrap = x/(exp(x/y) - 1)
    }
}
