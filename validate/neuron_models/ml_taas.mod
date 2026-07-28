TITLE Morris-Lecar (taas_checkmate version)
: Morris & Lecar 1981, Biophys. J.
: NEURON 9.0 compatible

NEURON {
    SUFFIX ml_taas
    NONSPECIFIC_CURRENT i
    RANGE gca, gk, gl, eca, ek, el, cm
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gca = 0.0044 (S/cm2)
    gk = 0.008 (S/cm2)
    gl = 0.002 (S/cm2)
    eca = 120 (mV)
    ek = -84 (mV)
    el = -60 (mV)
    cm = 1 (uF/cm2)
    v1 = -1.2 (mV)
    v2 = 18 (mV)
    v3 = 12 (mV)
    v4 = 17.4 (mV)
    phi_w = 0.0667 (/ms)
}

STATE { vm w_ml }

ASSIGNED {
    i (mA/cm2)
    minf
    winf
    tau_w (ms)
}

INITIAL {
    minf = 0.5*(1 + tanh((vm - v1)/v2))
    winf = 0.5*(1 + tanh((vm - v3)/v4))
    w_ml = winf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    i = gca*minf*(vm - eca) + gk*w_ml*(vm - ek) + gl*(vm - el)
}

DERIVATIVE states {
    minf = 0.5*(1 + tanh((vm - v1)/v2))
    winf = 0.5*(1 + tanh((vm - v3)/v4))
    tau_w = 1/(phi_w*cosh((vm - v3)/(2*v4)))
    vm' = (i*1000*cm - gca*minf*(vm - eca) - gk*w_ml*(vm - ek) - gl*(vm - el)) / (cm*0.001)
    w_ml' = (winf - w_ml) / tau_w
}
