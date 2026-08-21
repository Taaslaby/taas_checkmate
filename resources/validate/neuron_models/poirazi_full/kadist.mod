TITLE Poirazi 2003 CA1 kadist (generated from NeuroML2)
: Auto-generated from the published NeuroML2 channel by
: gen_poirazi_mods.py (mini-jNeuroML). Kinetics match the pillars
: engine and the NML ComponentTypes exactly.

NEURON {
    SUFFIX kadist
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

STATE { n l }

ASSIGNED {
    g (mho/cm2)
    i (mA/cm2)
    ninf
    ntau
    linf
    ltau
}

INITIAL {
    rates(v)
    n = ninf
    l = linf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gbar
    g = g * n^(1)
    g = g * l^(1)
    i = g * (v - erev)
}

DERIVATIVE states {
    rates(v)
    n' = (ninf - n)/ntau
    l' = (linf - l)/ltau
}

PROCEDURE rates(v (mV)) {
    LOCAL in, tn
    in = (1/((1+exp(((((1e-3*((-1.8)-(1/((1+exp((((v/(1.0))+40)/(5))))))))*((v/(1.0))+1))*9.648e4)/((8.315*((308.15)/(1.0)))))))))
        if ((((exp((1e-3*(-1.8 - (1 / (1 + exp(((v / 1.0)+40)/5))))*0.39*((v / 1.0)+1)*9.648e4) / (8.315*((308.15) / 1.0)))) / (0.1*(1 + (exp((1e-3*(-1.8 - (1 / (1 + exp(((v / 1.0)+40)/5))))*((v / 1.0)+1)*9.648e4) / (8.315*((308.15) / 1.0))))))) / (3.348369522101714)) < 0.1) { tn = ((0.1*(3.348369522101714))*1.0) }
        else { tn = ((exp((((((1e-3*((-1.8)-(1/((1+exp((((v/(1.0))+40)/(5))))))))*0.39)*((v/(1.0))+1))*9.648e4)/((8.315*((308.15)/(1.0))))))/((0.1*(1+exp(((((1e-3*((-1.8)-(1/((1+exp((((v/(1.0))+40)/(5))))))))*((v/(1.0))+1))*9.648e4)/((8.315*((308.15)/(1.0))))))))))*1.0) }
    ninf = in
    ntau = tn/(3.348369522101714)
    LOCAL il, tl
    il = (1/((1+exp(((((1e-3*3)*((v/(1.0))+56))*9.648e4)/((8.315*((308.15)/(1.0)))))))))
        if (((0.26*((v / 1.0)+50)) / (3.348369522101714)) < (2 / (3.348369522101714))) { tl = (2*1.0) }
        else { tl = ((0.26*((v/(1.0))+50))*1.0) }
    linf = il
    ltau = tl/(1.0)
}
