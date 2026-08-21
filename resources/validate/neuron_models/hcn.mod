TITLE HAAP HCN h-current (HCN)
: Simplified HCN current matching the pillars HAAP `h` gate.
: Opens on HYPERPOLARIZATION -> rebound after inhibition.
:   h_inf = 1 / (1 + exp(+(V - vhalf) / kslope))
:   h'    = (h_inf - h) / htau
: `kslope` is the slope factor (mV, a divisor).
: (The original TNMT draft had `exp(-...)` here, which opened the gate on
:  depolarization and contradicted the model's name "hyperpolarization-
:  activated".)

NEURON {
    SUFFIX hcn
    NONSPECIFIC_CURRENT i
    RANGE gbar, vhalf, kslope, htau
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gbar = 0.0002 (S/cm2)
    vhalf = -75 (mV)
    kslope = 6 (mV)
    htau = 200 (ms)
    erev = -30 (mV)
}

STATE { h }

ASSIGNED {
    i (mA/cm2)
    hinf
}

INITIAL {
    h = 1 / (1 + exp((v - vhalf) / kslope))
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    i = gbar * h * (v - erev)
}

DERIVATIVE states {
    hinf = 1 / (1 + exp((v - vhalf) / kslope))
    h' = (hinf - h) / htau
}
