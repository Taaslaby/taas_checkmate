# ChaosPatternEngine (CPE)

**Contrastive dopamine-STDP + T2-SPIGRAD learning on Kuramoto connectomes.**

## Core Idea

Compare network dynamics under **pattern** (meaningful input) vs **chaos** (random noise). The dopamine signal δ gates plasticity:

```
δ = reward + β · [(R_pat − α·|H_pat−H₀|) − (R_chaos − α·|H_chaos−H₀|)]
```

- **δ > 0** → network synchronises better on signal than noise → STDP strengthens
- **δ < 0** → network does worse on signal than noise → STDP weakens
- **δ ≈ 0** → plasticity frozen (plasticity-stability dilemma solved)

## Architecture

```
┌────────────────────────────────────────────────┐
│               CPE Engine                        │
│                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │Kuramoto  │    │dopamine  │    │T2-SPIGRAD│   │
│  │dynamics  │ ←──│ δ gate   │───→│gradients │   │
│  │on CSR    │    │sig≠noise │    │(optional)│   │
│  └────┬─────┘    └──────────┘    └──────────┘   │
│       │                                            │
│       ▼                                            │
│  ┌──────────┐                                     │
│  │STDP      │   w += η · δ · e · gate             │
│  │E/I safe  │   sign-preserving clip               │
│  └──────────┘                                     │
└────────────────────────────────────────────────┘
```

## Modules

| Module | Description |
|--------|-------------|
| `config` | `CpeConfig` — all hyperparameters with sensible defaults |
| `coherence` | Phase coherence R, spike entropy H, spike detection |
| `dopamine` | δ computation, timing gate |
| `stdp` | Eligibility trace, sign-preserving STDP update |
| `hybrid` | STDP + T2-SPIGRAD combined update |

## Usage

### Python

```python
import taas_checkmate as tcm

# Kuramoto step on connectome
tcm.cpe.cpe_kuramoto_step(phases, omegas, row_ptr, col_indices, values, dt)

# Phase coherence
r = tcm.cpe.cpe_phase_coherence(phases)

# Spike entropy
h, err = tcm.cpe.cpe_spike_entropy(spike_counts, total_steps, h_target)

# Dopamine δ
delta = tcm.cpe.cpe_dopamine_delta(
    reward=1.0, pattern_r=0.85, pattern_h_err=0.1,
    chaos_r=0.75, chaos_h_err=0.3,
    alpha=0.3, beta=0.3
)

# Timing gate
gate = tcm.cpe.cpe_dopamine_timing_gate(critical_steps=40, total_steps=100)

# Eligibility trace
tcm.cpe.cpe_eligibility_update(trace, spikes, row_ptr, elig_add=0.15, elig_decay=0.90)

# STDP update
tcm.cpe.cpe_stdp_update(weights, delta, elig, gate, lr=0.02, w_min=0.01, w_max=3.0)
```

### Rust

```rust
use taas_cpe::{CpeConfig, CpeEngine, kuramoto_step, phase_coherence, dopamine_delta};

let config = CpeConfig::default();
let engine = CpeEngine::new(config);

// One Kuramoto step
engine.step(&mut phases, &omegas, &row_ptr, &col_indices, &values);

// Coherence
let r = engine.coherence(&phases);

// Spikes
let spikes = engine.spikes(&phases, &prev);

// Delta
let delta = engine.delta(reward, pat_r, pat_h_err, chaos_r, chaos_h_err);

// STDP
engine.stdp_update(&mut weights, delta, &elig, gate);
```

## Parameters (tuned for Pong on 800-neuron connectome)

| Param | Value | Meaning |
|-------|-------|---------|
| `dt` | 0.01 | Kuramoto timestep |
| `lr_stdp` | 0.02 | STDP learning rate |
| `elig_decay` | 0.90 | τ_trace ≈ 85ms |
| `elig_add` | 0.15 | Per-spike trace increment |
| `alpha` | 0.3 | Entropy penalty weight |
| `beta` | 0.3 | Chaos/pattern gap weight |
| `h_target` | 3.0 | Target entropy |

## Key Results (Pong Benchmark)

- **R_pat > R_chaos**: neural dynamics are more coherent under pattern than chaos
- **δ trending UP**: plasticity works in the right direction
- **Zero sign flips**: E/I balance perfectly preserved (unlike vanilla STDP)
- **97.9% synapses changed**: significant rewiring under task pressure
- **T2-SPIGRAD compatible**: exact gradients + STDP eligibility, same δ gate
