# taas-kernels — Pure Rust SIMD Neural Kernels

Portable SIMD (`wide::f32x8`) neural computation kernels — zero ASM, zero FFI, zero `unsafe` in the public API.

## Features

| Module | Functions |
|---|---|
| **Neuron models** | LIF, LIF-AT, Izhikevich, AdEx, Hodgkin-Huxley, FHN, Wilson-Cowan, SAW, ψ-wave, STP, dopamine, e-prop |
| **Phase dynamics** | Kuramoto, Kuramoto + viscous damping, T3 (Stuart-Landau + Kuramoto) |
| **Decoders** | NLMS, LMS, Kalman filter, Adam, PST decoder, ring synthesis, complex decoder |
| **Signal** | ReLU, sigmoid, thalamic gate, phase gate, stochastic encoder, hebbian update, SGD, Adam |
| **Measurements** | Phase coherence, interference matrix, band energy, winding number, spike entropy |
| **Attention** | XOR attention (popcnt similarity), xorshift32 RNG |
| **IO** | PWM encoder, linear decoder, echo state tap |

## Usage

```rust
use taas_kernels;

// ── Izhikevich neuron step ──
let n = 16;
let mut v = vec![-65.0f32; n];
let mut u = vec![-13.0f32; n];
let mut trace = vec![0.0f32; n];
let a = vec![0.02f32; n];
let b = vec![0.2f32; n];
let I = vec![30.0f32; n];
let mut spikes = vec![0i32; n];
let c = vec![-65.0f32; n];
let d = vec![8.0f32; n];
let mut refractory = vec![0.0f32; n];

taas_kernels::izhikevich_step(
    &mut v, &mut u, &mut trace,
    &a, &b, &I, 1.0, 0.5,
    &mut spikes, &c, &d, &mut refractory, 3.0,
);

// ── Kuramoto phase update ──
let mut phases = vec![0.0f32; n];
let omegas = vec![1.0f32; n];
let coupling = vec![0.0f32; n];
taas_kernels::kuramoto_update(&mut phases, &omegas, &coupling, 0.01);

// ── T3 update (Stuart-Landau + Kuramoto) ──
let mut amps = vec![1.0f32; n];
let drives = vec![0.0f32; n];
taas_kernels::t3_update(&mut phases, &mut amps, &omegas, &coupling, &drives, 0.01, 0.0, 1.0, 10.0);

// ── NLMS decoder ──
let mut weights = vec![0.0f32; n];
let mut bias = vec![0.0f32; 1];
let states: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
taas_kernels::nlms_train(&mut weights, &mut bias, &states, 1.0, 0.01, 1e-8);
```

## SIMD

All kernels use `wide::f32x8` for 8-wide SIMD (AVX2 on x86_64, NEON on ARM, portable emulation elsewhere). When `n < 8` or `n % 8 != 0`, a scalar tail handles remaining elements transparently.

## Dependencies

- `wide` (SIMD vectors)
- `bytemuck` (zero-copy casts)

No `std` features required beyond `alloc`.
