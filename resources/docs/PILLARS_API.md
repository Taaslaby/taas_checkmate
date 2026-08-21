# Pillars API Design — Future-Proof Architecture

**Goal**: One API that scales from "student writes LIF in 5 minutes" to "production connectome with 16 compartment types, STDP, and homeostatic plasticity".

## Guiding Principles

1. **Zero-cost abstraction**: `ComputeNeuron` compiles to the same AVX2 as hand-written SIMD.
2. **One recommended path**: Three levels exist, but `#[derive(NeuronModel)]` is the default.
3. **Safe by default**: No raw indices in user code — only typed `state.v`, `params.v_thresh`.
4. **Composable bio bricks**: Ops that work on `S` values, not raw memory indices.
5. **Clear deprecation**: Old `Model::step<S>(ctx)` still works but is considered "assembly."

---

## Layer 1 — Core Engine (`pillars::core`)

**Audience**: Framework developers. Never seen by model authors.

```
core/
├── lane.rs         — SimdLane trait (SIMD ops for f32x8 / f32)
├── lane_f32x8.rs   — AVX2/NEON implementation via wide::f32x8
├── lane_f32.rs     — Scalar fallback
├── cond.rs         — Cond<S> for .then().else_()
├── context.rs      — SimdContext<S> (raw state/param arrays + ops)
├── compute.rs      — StepContext<S> + ComputeNeuron trait
├── population.rs   — NeuralPopulation (memory management + runner)
└── util.rs         — load8 / store8 helpers
```

### SimdLane trait (stable — no changes)

Primitive methods map to one hardware instruction; composite helpers stay branchless. `f32x8` (SIMD) and `f32` (scalar) both implement it.

```rust
pub trait SimdLane: Copy + Add<Output=Self> + ... {
    fn splat(v: f32) -> Self;
    fn sigmoid(v: Self, v_half: f32, k: f32) -> Self;
    fn vtrap(x: Self, y: Self) -> Self;
    fn exp(self) -> Self; fn ln(self) -> Self;
    fn sin(self) -> Self;  fn cos(self) -> Self;
    fn cmp_ge(self, other: Self) -> Self;
    fn blend(cond: Self, t: Self, f: Self) -> Self;
    fn move_mask(self) -> u32;
    // ... ~40 methods, each → one instruction
}
```

### SimdContext (internal — used by generated bridge)

Holds raw SoA pointers, processes 8 neurons at a time. **Not for user code.**

```rust
pub struct SimdContext<S: SimdLane> {
    pub state: [S; 32],      // MAX_STATES = 32, loaded from SoA
    pub params_ptr: *const f32,
    pub n: usize, pub off: usize,
    pub i_syn: S, pub i_pre: S, pub i_post: S,
    pub spike_mask: S,
}
```

The ops on `SimdContext` (leak_integrate, fire_reset, etc.) are kept for models that directly impl `Model`. They use `impl Into<usize>` for ergonomic enum access:

```rust
impl<S: SimdLane> SimdContext<S> {
    pub fn leak_integrate(&mut self, v_idx: impl Into<usize>, inv_tau: impl Into<usize>, v_rest: impl Into<usize>);
    pub fn fire_reset(&mut self, v_idx: impl Into<usize>, thresh: impl Into<usize>, reset: impl Into<usize>) -> S;
    pub fn ohmic_current(&self, g: impl Into<usize>, e_rev: impl Into<usize>, v: S, gating: S) -> S;
    // ... ~10 ops
}
```

---

## Layer 2 — Typed Model API (`ComputeNeuron`) — **THE DEFAULT**

**Audience**: Everyone writing a neuron model. Safe, typed, zero-cost.

### Pattern

```rust
#[derive(NeuronModel)]
pub struct LIFModel {
    #[state] pub v: f32,
    #[state] pub refr: f32,
    #[param] pub dt: f32,
    #[param] pub inv_tau_m: f32,
    #[param] pub v_rest: f32,
    #[param] pub v_thresh: f32,
    #[param] pub v_reset: f32,
    #[param] pub tau_ref: f32,
}

impl ComputeNeuron for LIFModel {
    type State<S: SimdLane> = LIFState<S>;
    type Params<S: SimdLane> = LIFParams<S>;

    fn step<S: SimdLane>(
        state: &mut LIFState<S>,
        params: &LIFParams<S>,
        ctx: &mut StepContext<S>,
    ) {
        // All access is typed — no raw indices anywhere
        let is_refr = state.refr.cmp_gt(S::zero());
        let v = is_refr.cond().then(params.v_reset).else_(state.v);
        let dv = (params.v_rest - v + ctx.i_syn) * params.inv_tau_m * ctx.dt();
        let spike = (v + dv).cmp_ge(params.v_thresh);
        ctx.emit_spike(spike);
        state.v = spike.cond().then(params.v_reset).else_(v + dv);
        state.refr = (state.refr - ctx.dt())
            .max(S::zero());
    }
}
```

### What `#[derive(NeuronModel)]` generates

For `struct LIFModel` with `#[state] v, refr` and `#[param] dt, inv_tau_m, ...`:

1. `LIFState<S>` struct with fields `v: S, refr: S`
2. `LIFParams<S>` struct with fields `dt: S, inv_tau_m: S, ...`
3. `impl Model for LIFModel` bridge:
   - Loads state/params from `SimdContext` into typed structs
   - Creates `StepContext` from ctx's buses
   - Calls `ComputeNeuron::step`
   - Writes state + spike back to `SimdContext`

The bridge is **identical for every model** — generated once, no special cases.

### StepContext — what goes in

```rust
pub struct StepContext<S: SimdLane> {
    pub i_syn: S,      // Post-synaptic current
    pub i_pre: S,      // Pre-spike bus (for STDP)
    pub i_post: S,     // Post-spike bus (for STDP)
    pub dt: f32,       // Scalar dt (broadcast with .dt())
    spike_mask: S,
}

impl<S: SimdLane> StepContext<S> {
    /// Canonical timestep accessor for model authors.
    pub fn dt(&self) -> S { S::splat(self.dt) }
    /// Compatibility alias for `dt()`.
    pub fn dt_simd(&self) -> S { self.dt() } // compatibility alias
    pub fn emit_spike(&mut self, cond: S);
}
```

**Key decision**: `StepContext` has NO ops (no leak_integrate, fire_reset, etc.). Ops live in `pillars::ops` as free functions that work on `S` values directly. This keeps `StepContext` minimal and the ops composable.

---

## Layer 3 — Bio Bricks (`pillars::ops`)

**Audience**: Model authors who want reusable math. Free functions, no trait bounds beyond `SimdLane`.

```rust
// pillars::ops::membrane
pub fn leak_integrate<S: SimdLane>(v: S, i_syn: S, inv_tau: S, v_rest: S, dt: S) -> S {
    v + (v_rest - v + i_syn) * inv_tau * dt
}

pub fn membrane_depolarize<S: SimdLane>(v: S, i_inj: S, i_ion: S, inv_cm: S, dt: S) -> S {
    v + (i_inj - i_ion) * inv_cm * dt
}

// pillars::ops::spike
pub fn fire_reset<S: SimdLane>(v: S, v_thresh: S, v_reset: S) -> (S, S) {
    let spike = v.cmp_ge(v_thresh);
    (spike.cond().then(v_reset).else_(v), spike)
}

// pillars::ops::channels
pub fn ohmic_current<S: SimdLane>(g: S, v: S, e_rev: S) -> S {
    g * (v - e_rev)  // NEURON convention: I = g·(V - E_rev)
}
```

**These are pure math — no memory access, no indices.** They work equally well in `ComputeNeuron::step` (with typed params) and `Model::step` (with raw `ctx.p()`).

Example: HAAP via ops:
```rust
fn step<S: SimdLane>(state: &mut HAAPState<S>, params: &HAAPParams<S>, ctx: &mut StepContext<S>) {
    use pillars::ops::*;
    let i_leak = ohmic_current(params.g_l, state.v, params.e_l);
    let i_m = ohmic_current(params.g_m * state.u, state.v, params.e_k);
    let i_h = ohmic_current(params.g_h * state.h, state.v, params.e_h);
    state.v = membrane_depolarize(state.v, ctx.i_syn, i_leak + i_m + i_h, params.inv_cm, ctx.dt());
    // ... gates, spike detection
}
```

---

## Migration Path

### Phase 1 — Stabilize (this PR)
| What | Status |
|------|--------|
| `*ParamIdx` enums for all models | ✅ Done |
| State load/save macros | ✅ Done |
| Izhikevich as `ComputeNeuron` proof | ✅ Done |

### Phase 2 — Migrate zoo (next)
| Step | Effort | Why |
|:----:|:------:|:----|
| 1. Add `leak_integrate` and `fire_reset` free functions to `ops::membrane` | 🟢 10 min | Foundation for all models |
| 2. Port FHN (simplest remaining, 5 params, 3 states) | 🟢 15 min | Proves ops + derive work |
| 3. Port HAAP (most complex, 3 gates, 17 params) | 🟡 1 hr | Proves multi-gate works |
| 4. Port remaining 10 models | 🟡 2-3 hr | Mechanical, proven pattern |
| 5. Verify all 267+ tests pass | 🟢 Auto | No regressions allowed |

### Phase 3 — Cleanup
| Change | Why |
|--------|-----|
| Mark `Model::step()` as `#[deprecated]` in docs | Users should use `ComputeNeuron` |
| Remove `run_neuron` / `run_neuron_par` | Replaced by `NeuralPopulation::step()` |
| Remove `SimdContext` ops (leak_integrate, etc.) | Replaced by free functions in `ops::` |

---

## What Stays, What Goes

### Stays (forever)
- `SimdLane` trait — too fundamental
- `NeuralPopulation` — the safe public API
- `ComputeNeuron` / `StepContext` — the model author's API
- `pillars::ops::*` — composable bio bricks

### Goes (deprecated → removed)
- `Model::step()` — replaced by derive-generated bridge
- `run_neuron_ext` / `run_neuron_par` — replaced by `NeuralPopulation`
- `SimdContext` ops (leak_integrate, fire_reset, etc.) — replaced by free functions

### Stays but restructured
- `ParamIdx` enums — stay in model files for `fill_param()` use
- `compose_*!` macros — stay for backward compat but new models don't use them

---

## Example: HH through the new API

```rust
#[derive(NeuronModel)]
pub struct HHModel {
    #[state] pub v: f32, #[state] pub m: f32,
    #[state] pub h: f32, #[state] pub n: f32,
    #[state] pub was_below: f32,

    #[param] pub dt: f32, #[param] pub g_na: f32, #[param] pub e_na: f32,
    #[param] pub g_k: f32, #[param] pub e_k: f32,
    #[param] pub g_leak: f32, #[param] pub e_leak: f32,
    #[param] pub q10: f32, #[param] pub v_thresh: f32, #[param] pub inv_cm: f32,
}

impl ComputeNeuron for HHModel {
    fn step<S: SimdLane>(
        state: &mut HHState<S>, params: &HHParams<S>, ctx: &mut StepContext<S>,
    ) {
        // Gates use SimdLane methods + GateParams structs (unchanged)
        let m_new = compute_gate(state.v, state.m, GATE_M, params.q10 * ctx.dt);
        let h_new = compute_gate(state.v, state.h, GATE_H, params.q10 * ctx.dt);
        let n_new = compute_gate(state.v, state.n, GATE_N, params.q10 * ctx.dt);

        // Ohmic currents via free functions
        let i_na = ohmic_current(params.g_na * m_new.powi(3) * h_new, state.v, params.e_na);
        let i_k = ohmic_current(params.g_k * n_new.powi(4), state.v, params.e_k);
        let i_leak = ohmic_current(params.g_leak, state.v, params.e_leak);

        state.v = membrane_depolarize(state.v, ctx.i_syn, i_na + i_k + i_leak,
                                       params.inv_cm, ctx.dt());
        state.m = m_new; state.h = h_new; state.n = n_new;

        // Edge-triggered spike
        let is_above = state.v.cmp_ge(params.v_thresh);
        ctx.emit_spike(is_above.and(state.was_below.cmp_gt(S::zero())));
        state.was_below = is_above.not().cond().then(S::one()).else_(S::zero());
    }
}
```

---

## Summary

```
                    Level 1 (DEFAULT)
                    ┌──────────────────────┐
                    │  #[derive(NeuronModel)]  │
                    │  + ComputeNeuron      │
                    │  + ops:: free fns     │
                    └──────────┬───────────┘
                               │ derive generates bridge
                    ┌──────────▼───────────┐
                    │  Model::step(ctx)    │ ← internal, not user-facing
                    │  + SimdContext ops   │
                    └──────────┬───────────┘
                               │ runner calls
                    ┌──────────▼───────────┐
                    │  NeuralPopulation    │ ← safe public API
                    │  + run_neuron        │ ← backward compat
                    └──────────────────────┘
```

**Result**: Only ONE way to write a model (`#[derive(NeuronModel)]`), ONE way to run it (`NeuralPopulation::step()`), and composable bio bricks when you need them (`ops::*`). Everything else is implementation detail.
