TITLE Hodgkin-Huxley (taas_checkmate version)
: Standard HH sodium + potassium + leak
: NEURON 9.0 compatible

NEURON {
    SUFFIX hh_taas
    NONSPECIFIC_CURRENT i
    RANGE gnabar, gkbar, gl, el
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gnabar = 0.12 (S/cm2)
    gkbar = 0.036 (S/cm2)
    gl = 0.0003 (S/cm2)
    el = -54.3 (mV)
}

STATE { m h_taas n }

ASSIGNED {
    i (mA/cm2)
    ina (mA/cm2)
    ik (mA/cm2)
    il (mA/cm2)
}

INITIAL {
    m = alpha_m(v)/(alpha_m(v) + beta_m(v))
    h_taas = alpha_h(v)/(alpha_h(v) + beta_h(v))
    n = alpha_n(v)/(alpha_n(v) + beta_n(v))
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gnabar * m*m*m * h_taas * (v - 50)
    ik = gkbar * n*n*n*n * (v + 77)
    il = gl * (v - el)
    i = ina + ik + il
}

DERIVATIVE states {
    m' = (alpha_m(v) - (alpha_m(v)+beta_m(v))*m)
    h_taas' = (alpha_h(v) - (alpha_h(v)+beta_h(v))*h_taas)
    n' = (alpha_n(v) - (alpha_n(v)+beta_n(v))*n)
}

FUNCTION alpha_m(v) {
    alpha_m = 0.1*vtrap(-(v+40), 10)
}

FUNCTION beta_m(v) {
    beta_m = 4*exp(-(v+65)/18)
}

FUNCTION alpha_h(v) {
    alpha_h = 0.07*exp(-(v+65)/20)
}

FUNCTION beta_h(v) {
    beta_h = 1/(1+exp(-(v+35)/10))
}

FUNCTION alpha_n(v) {
    alpha_n = 0.01*vtrap(-(v+55), 10)
}

FUNCTION beta_n(v) {
    beta_n = 0.125*exp(-(v+65)/80)
}

FUNCTION vtrap(x, y) {
    if (fabs(x/y) < 1e-6) {
        vtrap = y*(1 - x/y/2)
    } else {
        vtrap = x/(exp(x/y) - 1)
    }
}
