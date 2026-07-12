# taas-core — Spiking Neural Networks + Kuramoto Dynamics

High-level Rust wrappers over `taas-kernels` SIMD primitives. Adds rayon parallelization, state management, and training infrastructure.

## Modules

| Module | Contents |
|---|---|
| `neuron` | Izhikevich, LIF, LIF-AT, AdEx, HH, FHN, Wilson-Cowan, SAW, ψ-wave with e-prop support |
| `kuramoto` | Sparse/dense Kuramoto & T3 dynamics, decoders (NLMS, Kalman, LMS, PST, ring) |
| `sparse` | CSR spike propagation, Hamiltonian, rate-based dynamics, STDP LTD |
| `koit` | XOR attention, xorshift32 RNG |

## Neuron Models

```rust
use taas_core::neuron;

// Izhikevich (single step)
neuron::step_izhikevich(&mut v, &mut u, &mut I, &mut spikes, &a, &b, &c, &d, dt);

// LIF with exponential Euler
neuron::step_lif(&mut v, &mut I, &mut spikes, &params, &mut refractory, tau_ref);

// Hodgkin-Huxley (LUT-based)
let mut cfg = neuron::standard_config(1024, 0.01);
neuron::step_hh_lut(&mut cfg);
```

## Kuramoto with Sparse CSR

```rust
use taas_core::kuramoto;

let n = 1024;
kuramoto::kuramoto_step_sparse(
    &mut phases, &omegas,
    &row_ptr, &col_indices, &weights,
    &mut sin_buf, &mut cos_buf, &mut coupling_buf, 0.01,
);
```

## Training

```rust
use taas_core::neuron::{EPropLIF, EPropTrainer};

let mut trainer = EPropTrainer::new(n_pre, n_post, lr);
trainer.train(&mut v, &mut eligibility, &pre_spikes, &I_ext, target, dt);
```

## Dependencies

- `taas-kernels` (SIMD kernels)
- `rand`, `rayon` (parallelization)
