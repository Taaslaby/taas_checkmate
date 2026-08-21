COMMENT
Poirazi 2003 CA1 calcium pool (capool) — a faithful NEURON port of the
published PoiraziCaConcentrationModel (Destexhe 1992, modified by Poirazi):

  dCon/dt = iCa / (2 * Faraday * surfaceArea * shellThickness * 18)
            - ((Con - restingConc) / decayConstant) * 7

with concentration clamped at 0. The /18 is the Golding 1999 sharp-Ca-
repolarisation factor; the *7 makes Ca removal 7x faster than the raw
decayConstant. Thin-shell volume = membrane area * shellThickness
(0.1 um = 100 nm), the model's own convention. Runs at 35 degC like the
rest of the Poirazi validation.

Flux units: ica [mA/cm2], depth [nm] -> mM/ms:
  dC/dt = -ica * 1e7 / (depth * FARADAY * 2 * 18) + (cai0 - cai) * 7 / tau
ENDCOMMENT
NEURON {
    SUFFIX capool
    USEION ca READ ica WRITE cai
    RANGE depth, tau, cai0
}
UNITS {
    (mM) = (milli/liter)
    (mA) = (milliamp)
}
PARAMETER {
    depth = 100 (nm) : shell thickness (0.1 um)
    tau = 200 (ms) : decayConstant
    cai0 = 1e-4 (mM) : restingConc
}
CONSTANT {
    FARADAY = 96520 (coulomb)
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
    if (cai < 0) { cai = 0 }
}
DERIVATIVE integrate {
    cai' = -ica * (1e7) / (depth * FARADAY * 2 * 18) + (cai0 - cai) * 7 / tau
}
