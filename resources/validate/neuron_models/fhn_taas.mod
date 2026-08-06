TITLE FitzHugh-Nagumo (taas_checkmate version)
: FitzHugh 1961, Nagumo 1962
: NEURON 9.0 compatible

NEURON {
    SUFFIX fhn_taas
    NONSPECIFIC_CURRENT i
    RANGE a, b, tau_w
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
}

PARAMETER {
    a = 0.7
    b = 0.8
    tau_w = 12.5 (ms)
}

STATE { vm w_fhn }

ASSIGNED {
    i (mA/cm2)
}

INITIAL {
    vm = 0
    w_fhn = 0
}

BREAKPOINT {
    SOLVE states METHOD derivimplicit
    i = (vm - vm*vm*vm/3 - w_fhn + 10*i) * 0.1
}

DERIVATIVE states {
    vm' = vm - vm*vm*vm/3 - w_fhn + i*1000
    w_fhn' = (vm + a - b*w_fhn) / tau_w
}
