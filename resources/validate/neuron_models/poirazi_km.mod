TITLE Poirazi 2003 CA1 M-current (muscarinic K+, Mainen 1995) - n
: Published Poirazi-Brannon-Mel (2003) CA1 km channel (NeuroML2 export),
: gateHHrates with HHExpLinearRate (rate=9e-3 per_ms, midpoint=-30 mV,
: scale=9 mV) both directions:
:   alpha = (9e-3/9) * vtrap(-(v+30), 9)   [vtrap(x,y) = x/(exp(x/y)-1)]
:   beta  = (9e-3/9) * vtrap(+(v+30), 9)
: The published q10Settings (q10ExpTemp q10Factor=2.3, experimentalTemp=23C)
: is applied exactly as jNeuroML does: rates scale by q10 = 2.3^((celsius-23)/10),
: with `celsius` the NEURON global. The pillars engine applies the same factor
: per gate at the same simulation temperature.

NEURON {
    SUFFIX poirazi_km
    NONSPECIFIC_CURRENT i
    RANGE gbar, ek
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gbar = 0.06 (S/cm2)
    ek = -80 (mV)
}

STATE { n }

ASSIGNED { i (mA/cm2) }

INITIAL {
    n = alpha(v) / (alpha(v) + beta(v))
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    i = gbar * n * (v - ek)
}

DERIVATIVE states {
    n' = alpha(v) * (1 - n) - beta(v) * n
}

FUNCTION alpha(v) {
    alpha = q10() * 0.001 * vtrap(-(v + 30), 9)
}

FUNCTION beta(v) {
    beta = q10() * 0.001 * vtrap(v + 30, 9)
}

: q10 = 2.3^((celsius - 23)/10) -- the published q10ExpTemp setting.
FUNCTION q10() {
    q10 = exp(log(2.3) * (celsius - 23) / 10)
}

FUNCTION vtrap(x, y) {
    if (fabs(x / y) < 1e-6) {
        vtrap = y * (1 - x / y / 2)
    } else {
        vtrap = x / (exp(x / y) - 1)
    }
}
