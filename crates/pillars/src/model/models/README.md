# Pillars Neuron Model Implementation Guidelines 🧠⚡

Follow these rules when implementing or porting neuron models (`Model` trait) to ensure **maximum SIMD execution performance** and **biological correctness**.

---

## ⚡ 1. Microarchitectural Performance Rules

### ❌ Rule 1.1: NO Vector Division in `step()`
* **Why**: Floating-point vector division (`vdivps`) stalls the CPU pipeline for **10–14 clock cycles**. Multiplication (`vmulps`) takes only **3–4 cycles** with a throughput of 2 instructions per clock cycle.
* **Bad**: `let dv = (v_rest - v + i) / tau_m * dt;`
* **Good**: Pre-invert time constants in parameters (`p[1] = 1.0 / tau_m`):
  ```rust
  let dv = (v_rest - v + i) * inv_tau_m * dt;
  ```

### ❌ Rule 1.2: NO `to_array()` or Slice Conversions
* **Why**: Calling `.to_array()` breaks vector register allocation, spilling YMM registers onto the CPU stack.
* **Good**: Use native `SimdLane` methods: `S::blend`, `cmp_ge`, `cmp_gt`, `max`, `min`, `clamp01`.

### ⚡ Rule 1.3: Eliminate Redundant Blends
* **Why**: Every `S::blend()` generates a `vblendvps` instruction.
* **Pattern**: When `v_dyn` is clamped to `v_reset` during refractory:
  Because $v_{reset} < v_{thresh}$, `v_dyn.cmp_ge(v_thresh)` automatically returns `false` during refractory!
* **Bad**: `let spike = S::blend(is_refr, S::zero(), spike_raw);`
* **Good**: `let spike = v_dyn.cmp_ge(ctx.p(3));`

---

## 🧬 2. Biological Correctness Rules

### 📌 Rule 2.1: Refractory Voltage Clamping
During the refractory period (`refr > 0`), clamp the dynamic voltage `v_dyn` directly to $v_{reset}$:
```rust
let is_refr = refr.cmp_gt(S::zero());
let v_dyn = S::blend(is_refr, ctx.p(4), v_new); // p[4] = v_reset
```

### 📌 Rule 2.2: Reversal Potentials & Current Scaling
Input current `ctx.i()` must be scaled in $mV/ms$ or pre-multiplied by membrane resistance ($I \cdot R_m$).

---

## 📋 3. Template for New Models

```rust
use crate::model::lane::SimdLane;
use crate::model::context::SimdContext;
use crate::model::Model;

/// [Model Name] Neuron Model
pub struct MyNeuronModel;

impl Model for MyNeuronModel {
    /// Params: [0]dt, [1]inv_tau_m, [2]v_rest, [3]v_thresh, [4]v_reset, [5]tau_ref
    const N_PARAMS: usize = 6;
    /// State: [0]v, [1]refr
    const N_STATE: usize = 2;
    const HAS_SPIKE: bool = true;

    #[inline(always)]
    fn step<S: SimdLane>(ctx: &mut SimdContext<S>) {
        let v = ctx.s(0);
        let refr = ctx.s(1);

        // 1. Equations (Pure Multiplication)
        let dv = (ctx.p(2) - v + ctx.i()) * ctx.p(1) * ctx.dt();
        let v_new = v + dv;

        // 2. Refractory Clamp
        let is_refr = refr.cmp_gt(S::zero());
        let v_dyn = S::blend(is_refr, ctx.p(4), v_new);

        // 3. Spike Detection
        let spike = v_dyn.cmp_ge(ctx.p(3));
        ctx.spike(spike);

        // 4. State Update
        ctx.set_s(0, S::blend(spike, ctx.p(4), v_dyn));

        let refr_dec = refr - ctx.dt();
        let refr_set = S::blend(spike, ctx.p(5), refr_dec);
        ctx.set_s(1, S::blend(refr_set.cmp_gt(S::zero()), refr_set, S::zero()));
    }
}
```
