: cachan_simple.mod - ohmic Ca2+ channel, gate always open, reversal = Nernst eca.
: ica = gbar*(v - eca)  [mA/cm2], positive = outward (NEURON convention).
NEURON {
    SUFFIX cachan_s
    USEION ca READ eca WRITE ica
    RANGE gbar
}
UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (mho)
}
PARAMETER {
    gbar = 0 (S/cm2)
}
ASSIGNED {
    v (mV)
    eca (mV)
    ica (mA/cm2)
}
BREAKPOINT {
    ica = gbar * (v - eca)
}
