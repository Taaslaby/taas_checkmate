COMMENT
calcium accumulation into a volume of area*depth next to the membrane
with a decay (time constant tau) to resting level. Thin-shell (volume =
area*depth) approximation - matches NeuroML decayingPoolConcentrationModel
only for depth << radius (the LEMS model uses the exact spherical shell).
ENDCOMMENT
NEURON {
    SUFFIX cacum
    USEION ca READ ica WRITE cai
    RANGE depth, tau, cai0
}
UNITS {
    (mM) = (milli/liter)
    (mA) = (milliamp)
}
PARAMETER {
    depth = 1 (nm) : shell thickness (volume = area*depth)
    tau = 10 (ms)
    cai0 = 50e-6 (mM)
}
CONSTANT {
    FARADAY = 96485.309 (coulomb)
}
ASSIGNED {
    ica (mA/cm2)
}
STATE {
    cai (mM)
}
INITIAL {
    cai = cai0
}
BREAKPOINT {
    SOLVE integrate METHOD cnexp
}
DERIVATIVE integrate {
    : ica [mA/cm2], depth [nm] -> mM/ms:
    :   dC/dt = ica/(2*F*depth) [mol/(m3.s)] = ica*1e7/(2*F*depth) [mM/ms]
    cai' = -ica * (1e7) / (depth * FARADAY * 2) + (cai0 - cai)/tau
}
