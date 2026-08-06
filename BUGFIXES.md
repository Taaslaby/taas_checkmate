# 🔧 Патчи для исправления найденных проблем в Pillars

## ✅ Статус: все 7 патчей применены

Проверено 2026-08-06 против рабочего дерева. Все патчи уже в коде (некоторые
реализованы лучше, чем в черновике ниже — см. адреса). `debug_assert!`/`#[cfg(debug_assertions)]`
и `const {}` блоки исчезают в release-сборке → 0% влияния на производительность.

| # | Патч | Статус | Где |
|---|------|--------|-----|
| 1 | bounds check в `SimdContext::p()`/`pf()` | ✅ | `core/context.rs` — `debug_assert!(i < MAX_PARAM_COLS)` (в `p()` и `pf()`) |
| 2 | семантика `dt()` vs `dtf()` | ✅ (без `#[deprecated]`) | `core/context.rs` — подробные doc-comments; `dtf()` НЕ депрецирован, т.к. его использует мост `ComputeNeuron` (pillars-macros) и `-D warnings` в CI упал бы |
| 3 | overlap check в `run_neuron_par` | ✅ | `core/runner.rs:62-80,440-450` — `check_no_overlap` + вызов в debug |
| 4 | HH gates feature flag `hh-precise-gates` | ✅ | `zoo/neurons/hh.rs` — cfg-выбор exp в `compute_gate_state` (233-238) и `compute_cat_gate_state` (190-194); feature объявлен в `pillars/Cargo.toml` |
| 5 | `const_assert!(N <= 16)` в `powi<N>` | ✅ | `core/lane.rs:178-186` — через `const {}` блок (стабилен, без static_assertions) |
| 6 | причина клампа `i_syn` до ±100 | ✅ | `core/population.rs:65-67` — `I_SYN_CLAMP_RANGE` с rationale + примеры |
| 7 | NaN check в `ohmic_gated` | ✅ | `ops/ohmic.rs` — finite-check через `move_mask` (покрывает NaN и inf), с тестом |

Документ ниже — исторический черновик с оригинальными формулировками.

---

## Высокий приоритет

### 1. Добавить bounds check в `SimdContext::p()` (debug mode)

**Файл:** `crates/pillars/src/core/context.rs:115`

```rust
// БЫЛО:
#[inline(always)]
pub fn p(&self, i: usize) -> S {
    unsafe { S::load_param(self.params_ptr.as_ptr(), i, self.n, self.off) }
}

// СТАЛО:
#[inline(always)]
pub fn p(&self, i: usize) -> S {
    debug_assert!(
        i < crate::zoo::Model::N_PARAMS,
        "param index {} out of bounds (max {})",
        i,
        crate::zoo::Model::N_PARAMS
    );
    unsafe { S::load_param(self.params_ptr.as_ptr(), i, self.n, self.off) }
}
```

**Примечание:** Требует добавления generic параметра `M: Model` в `SimdContext`, либо использование runtime проверки:

```rust
#[inline(always)]
pub fn p(&self, i: usize) -> S {
    #[cfg(debug_assertions)]
    {
        // Runtime проверка: предполагаем разумный максимум 32 параметра
        debug_assert!(i < 32, "param index {} exceeds reasonable limit (32)", i);
    }
    unsafe { S::load_param(self.params_ptr.as_ptr(), i, self.n, self.off) }
}
```

---

### 2. Документировать семантику `dt()` vs `dtf()`

**Файл:** `crates/pillars/src/core/context.rs:133-143`

```rust
// БЫЛО:
pub fn dt(&self) -> S {
    // 🛡️ Load dt from SoA params[0] — supports heterogeneous timesteps per lane.
    unsafe { S::load_param(self.params_ptr.as_ptr(), 0, self.n, self.off) }
}

pub fn dtf(&self) -> f32 {
    // Fallback: first neuron's dt. For heterogeneous dt use dt() which returns S.
    self.dt_val
}

// СТАЛО:
/// Load timestep from SoA params[0] — supports heterogeneous timesteps per lane.
///
/// # When to use
/// - **Always** in model implementations (supports per-neuron dt)
/// - Returns SIMD vector where each lane has its own dt value
///
/// # Example
/// ```ignore
/// let dv = (v_rest - v) * inv_tau * ctx.dt();  // ✅ Correct
/// ```
pub fn dt(&self) -> S {
    unsafe { S::load_param(self.params_ptr.as_ptr(), 0, self.n, self.off) }
}

/// Scalar timestep fallback — returns dt of the first neuron in the chunk.
///
/// # ⚠️ Warning
/// This method is **deprecated** for model implementations. Use [`dt()`](Self::dt) instead.
/// Only use `dtf()` for:
/// - Logging/debugging (single value needed)
/// - Legacy code compatibility
///
/// # Incorrect usage
/// ```ignore
/// let dv = (v_rest - v) * inv_tau * ctx.dtf();  // ❌ Wrong: ignores per-neuron dt
/// ```
///
/// # Correct usage
/// ```ignore
/// let dv = (v_rest - v) * inv_tau * ctx.dt();   // ✅ Correct
/// println!("dt[0] = {}", ctx.dtf());             // ✅ OK for logging
/// ```
#[deprecated(
    since = "0.2.0",
    note = "Use dt() for model implementations. dtf() only for logging."
)]
pub fn dtf(&self) -> f32 {
    self.dt_val
}
```

---

### 3. Добавить runtime проверку overlap в `run_neuron_par` (debug mode)

**Файл:** `crates/pillars/src/core/runner.rs` (после строки 75)

```rust
// Добавить после impl SendablePtrs:

#[cfg(debug_assertions)]
fn check_no_overlap(ranges: &[(usize, usize)]) {
    for i in 0..ranges.len() {
        for j in (i + 1)..ranges.len() {
            let (start_i, end_i) = ranges[i];
            let (start_j, end_j) = ranges[j];
            assert!(
                end_i <= start_j || end_j <= start_i,
                "Memory overlap detected: range {}=[{}, {}) overlaps with range {}=[{}, {})",
                i, start_i, end_i, j, start_j, end_j
            );
        }
    }
}

// В функции run_neuron_par (найти место вызова rayon::scope):
#[cfg(debug_assertions)]
{
    let ranges: Vec<(usize, usize)> = (0..num_threads)
        .map(|tid| {
            let start = tid * chunk_size;
            let end = (start + chunk_size).min(n);
            (start, end)
        })
        .collect();
    check_no_overlap(&ranges);
}
```

---

## Средний приоритет

### 4. Переключить HH gates на точный `exp()` или добавить feature flag

**Файл:** `crates/pillars/src/zoo/neurons/hh.rs:156`

**Вариант A: Feature flag (рекомендуется)**

```rust
// В Cargo.toml добавить:
// [features]
// hh-precise-gates = []

// В hh.rs:
#[inline(always)]
fn compute_gate_state<S: SimdLane>(v: S, gate: S, params: GateParams, dt_q10: S) -> S {
    // ... (код α/β вычислений)
    
    let sum = alpha + beta;
    let g_inf = alpha / sum;
    
    // Выбор точности exp в зависимости от feature flag
    #[cfg(feature = "hh-precise-gates")]
    let decay = (-dt_q10 * sum).exp();  // Точный exp (~1 ULP)
    
    #[cfg(not(feature = "hh-precise-gates"))]
    let decay = (-dt_q10 * sum).fast_exp();  // Быстрый exp (~1e-5)
    
    decay.mul_add(gate - g_inf, g_inf)
}

#[inline(always)]
fn compute_cat_gate_state<S: SimdLane>(v: S, c: S, dt_q10: S) -> S {
    let arg = -(v + S::splat(GATE_CAT_V_HALF)) * S::splat(GATE_CAT_INV_K);
    let m_inf = S::one() / (S::one() + arg.fast_exp());
    
    #[cfg(feature = "hh-precise-gates")]
    let decay = (-dt_q10 * S::splat(GATE_CAT_INV_TAU)).exp();
    
    #[cfg(not(feature = "hh-precise-gates"))]
    let decay = (-dt_q10 * S::splat(GATE_CAT_INV_TAU)).fast_exp();
    
    decay.mul_add(c - m_inf, m_inf)
}
```

**Вариант B: Всегда использовать точный exp (проще, но медленнее)**

```rust
// БЫЛО:
let decay = (-dt_q10 * sum).fast_exp();

// СТАЛО:
let decay = (-dt_q10 * sum).exp();  // Точный exp для биофизической точности
```

---

### 5. Добавить `const_assert!(N <= 16)` в `powi<N>`

**Файл:** `crates/pillars/src/core/lane.rs:145-155`

```rust
// БЫЛО:
fn powi<const N: usize>(self) -> Self;

// СТАЛО:
/// Integer power (compile-time constant N) — unrolled, no branches.
///
/// # Compile-time limit
/// `N` must be ≤ 16 to prevent excessive compile times and stack usage.
/// For larger exponents, use `powf()`.
///
/// # Examples
/// ```ignore
/// let x2 = x.powi::<2>();   // ✅ OK
/// let x4 = x.powi::<4>();   // ✅ OK
/// let x16 = x.powi::<16>(); // ✅ OK (max)
/// let x100 = x.powi::<100>(); // ❌ Compile error
/// ```
fn powi<const N: usize>(self) -> Self {
    const { assert!(N <= 16, "powi<N>: N must be <= 16 (use powf for larger exponents)") }
    
    // ... (существующая реализация)
}
```

**Примечание:** Для Rust < 1.79 использовать `static_assertions`:

```rust
// В Cargo.toml:
// [dependencies]
// static_assertions = "1.1"

// В lane.rs:
use static_assertions::const_assert;

fn powi<const N: usize>(self) -> Self {
    const_assert!(N <= 16);
    // ... (реализация)
}
```

---

### 6. Документировать причину клампа `i_syn` до ±100

**Файл:** `crates/pillars/src/core/population.rs` (найти определение `I_SYN_CLAMP_RANGE`)

```rust
// БЫЛО:
pub(crate) const I_SYN_CLAMP_RANGE: f32 = 100.0;

// СТАЛО:
/// Maximum absolute synaptic current (nA or pA depending on model).
///
/// # Rationale
/// Clamps input current to ±100 to prevent:
/// 1. **Numerical overflow** in exponential models (AdEx, HH)
/// 2. **Unrealistic firing rates** (>1000 Hz) that break refractory logic
/// 3. **NaN propagation** from extreme exp() arguments
///
/// # Biological context
/// - Typical cortical neuron: I_syn ∈ [-50, +50] pA
/// - Strong synaptic input: ~100 pA (saturating)
/// - Values >100 pA are non-physiological for point neuron models
///
/// # Override
/// To disable clamping (e.g., for abstract models), modify this constant
/// or use `NeuralPopulation::step_unclamped()` (if implemented).
pub(crate) const I_SYN_CLAMP_RANGE: f32 = 100.0;
```

---

## Низкий приоритет

### 7. Добавить NaN checks в `ohmic_gated` (debug mode)

**Файл:** `crates/pillars/src/ops/ohmic.rs` (создать если не существует)

```rust
/// Gated ohmic current: I = g_max · gating · (V - E_rev)
///
/// # Debug assertions
/// In debug builds, checks for NaN in gating variable to catch
/// gate integration errors early.
#[inline(always)]
pub fn ohmic_gated<S: SimdLane>(g_max: S, v: S, e_rev: f32, gating: S) -> S {
    #[cfg(debug_assertions)]
    {
        let arr = gating.to_array();
        debug_assert!(
            !arr[0].is_nan(),
            "ohmic_gated: gating is NaN (gate integration error)"
        );
    }
    
    g_max * gating * (v - S::splat(e_rev))
}
```

---

## Применение патчей

### Автоматическое применение (рекомендуется)

```bash
# 1. Создать ветку для фиксов
git checkout -b bugfix/pillars-safety-checks

# 2. Применить патчи вручную (скопировать код выше)

# 3. Запустить тесты
cargo test -p pillars --lib

# 4. Проверить производительность (release build)
cargo bench -p pillars

# 5. Коммит
git add .
git commit -m "fix(pillars): Add safety checks and documentation

- Add bounds check in SimdContext::p() (debug mode)
- Document dt() vs dtf() semantics, deprecate dtf()
- Add overlap check in run_neuron_par (debug mode)
- Add feature flag for HH precise gates
- Limit powi<N> to N <= 16
- Document I_SYN_CLAMP_RANGE rationale
- Add NaN check in ohmic_gated (debug mode)"
```

### Проверка влияния на производительность

```bash
# До применения патчей
cargo bench -p pillars --bench bench_all > before.txt

# После применения патчей
cargo bench -p pillars --bench bench_all > after.txt

# Сравнение (должно быть 0% разницы в release mode)
diff before.txt after.txt
```

**Ожидаемый результат:** Все `debug_assert!` и `#[cfg(debug_assertions)]` блоки **полностью исчезают** в release-сборке (проверено через `cargo asm`), поэтому производительность не изменится.

---

## Дополнительные рекомендации

### Добавить CI проверку

**Файл:** `.github/workflows/ci.yml`

```yaml
- name: Test with debug assertions
  run: cargo test --all-features
  
- name: Test release build (no debug overhead)
  run: cargo test --release --all-features
  
- name: Check for unsafe without justification
  run: |
    ! grep -r "unsafe" crates/pillars/src --include="*.rs" | \
      grep -v "// SAFETY:" | \
      grep -v "/// # Safety"
```

### Документация для контрибьюторов

**Файл:** `crates/pillars/CONTRIBUTING.md`

```markdown
## Safety Guidelines

1. **All `unsafe` blocks must have `// SAFETY:` comment** explaining invariants
2. **Use `debug_assert!` for runtime checks** (zero cost in release)
3. **Prefer compile-time checks** (`const {}` blocks, `static_assertions`)
4. **Test SIMD vs scalar consistency** (see `core/lane_tests.rs`)
5. **Document all `#[deprecated]` items** with migration path
```

---

## Итоговая статистика патчей

| Приоритет | Патчей | Строк кода | Влияние на производительность |
|-----------|--------|------------|-------------------------------|
| Высокий   | 3      | ~50        | 0% (debug-only)               |
| Средний   | 3      | ~80        | 0% (feature flag / docs)      |
| Низкий    | 1      | ~15        | 0% (debug-only)               |
| **Всего** | **7**  | **~145**   | **0% в release**              |

Все патчи **не влияют на производительность** release-сборки и улучшают безопасность/документацию кода.
