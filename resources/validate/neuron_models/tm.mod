: Tsodyks-Markram short-term plasticity (Tsodyks & Markram 1997; Tsodyks,
: Pawelzik & Markram 1998) as a synaptic point process.
:
:   delivered = weight * R * U      (pre-spike plasticityFactor, then...)
:   R <- R * (1 - U)                (depletion)
:   U <- U + U0 * (1 - U)           (facilitation, only if tau_facil > 0)
:
: Between spikes the resource recovers and the release probability decays,
: integrated with cnexp (exact exponential) to match pillars' lazy
: powi(exp(-dt/tau)) recovery. `delivered` records the weight delivered by the
: most recent spike (RANGE, so it can be sampled by a Vector).

NEURON {
    POINT_PROCESS TsodyksMarkram
    RANGE tau, e, tau_rec, tau_facil, U0
    RANGE g, delivered
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    tau = 3.0 (ms)      : synaptic decay time constant
    e = 0.0 (mV)        : reversal potential
    tau_rec = 100.0 (ms): recovery from depression
    tau_facil = 0.0 (ms): facilitation time constant (0 = depression only)
    U0 = 0.5            : baseline release probability
}

ASSIGNED {
    v (mV)
    i (nA)
    delivered (uS)
}

STATE {
    g (uS)   : synaptic conductance
    R        : available resource fraction
    U        : release probability
}

INITIAL {
    g = 0
    R = 1
    U = U0
    delivered = 0
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    i = g * (v - e)
}

DERIVATIVE states {
    g' = -g / tau
    R' = (1 - R) / tau_rec
    if (tau_facil > 0) {
        U' = (U0 - U) / tau_facil
    } else {
        U' = 0
    }
}

NET_RECEIVE(weight (uS)) {
    delivered = weight * R * U
    g = g + weight * R * U
    R = R * (1 - U)
    if (tau_facil > 0) {
        U = U + U0 * (1 - U)
    }
}
