# Workspace Coding Rules & Mathematical Verification Protocol

## 🧠 Neural Model Verification Rules (`pillars::model`)

Whenever writing, refactoring, or porting neuron models, ALWAYS perform these mandatory checks BEFORE finishing code:

1. **Pre-Inversion of Parameters (No Division in Loops)**:
   - Check every time constant ($\tau$, $\Delta_T$).
   - Never write `/ tau` or `/ delta_t` inside `step()`.
   - Store pre-inverted $1/\tau$ or $1/\Delta_T$ in parameters to use pure SIMD floating-point multiplication `* inv_tau` (`vmulps`).

2. **Early Refractory Clamping (Prevent Explosion)**:
   - If a model has a refractory period, clamp state $v = \text{blend}(is\_refr, v_{reset}, v_{raw})$ **BEFORE** evaluating exponential terms ($e^x$), quadratic terms ($v^2$), or cubic terms ($v^3$).
   - Prevents un-clamped background states from exploding during refractory.

3. **Substepping for Stiff ODEs**:
   - Quadratic/cubic models (e.g., Izhikevich $0.04 v^2 + 5 v + 140$) are numerically stiff under Forward Euler.
   - Use 2 half-steps ($0.5 \cdot dt$) for $v$ to maintain numerical stability.

4. **Horner's Method Polynomials**:
   - Rewrite polynomials like $v - \frac{v^3}{3}$ to $v \cdot (1 - \frac{v^2}{3})$ and $0.04 v^2 + 5 v$ to $v \cdot (0.04 v + 5)$.
   - Saves vector multiplications and leverages CPU FMA instructions.

5. **Pure SIMD Mask Operations**:
   - Use `.and()` for mask AND operations (`vandps`), not `S::blend(a, b, false)`.
   - Use `.not()` for mask NOT inversion (`vandnps`), avoiding redundant `cmp_gt` vector comparisons.

6. **Edge Detection for Continuous Models**:
   - For models without instant voltage resets (FHN, HH), detect rising edge spikes using `is_above.and(was_below)` to prevent continuous multi-step spike bursts.
