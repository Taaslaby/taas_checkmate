# Core — Zero-Cost SIMD Engine

**Граница: Rust-генерики → аппаратные инструкции.**

Это сердце Pillars: здесь обобщённый код превращается в AVX2/NEON-векторные
инструкции и скалярный fallback. Всё решается на этапе компиляции —
без интерпретаторов, JIT и runtime-диспетчеризации.

```text
Пользовательский код:  v.cmp_ge(v_thresh)
        ↓
SimdLane trait (генерик S: SimdLane)
        ↓
┌─────────────────┼─────────────────┐
│  S = f32x8      │  S = f32        │
│  _mm256_cmp_ps  │  ucomiss+setae  │
│  (AVX2/NEON)    │  (скаляр)       │
└─────────────────┴─────────────────┘
```

Каждый примитивный метод `SimdLane` (арифметика, сравнения, load/store)
компилируется в **ровно одну аппаратную инструкцию**. Составные хелперы
(`signum`, `round`, `hypot`, `cbrt`, `clamp01`) собираются из нескольких
инструкций без ветвлений (branchless).

---

## Карта файлов

| Файл | Роль | Безопасность |
|------|------|:------------:|
| [`arch.rs`](arch.rs) | Детекция SIMD-ширины (`LANES`, `ALIGN`, `pad_lanes`, `SimdType`) | ✅ safe |
| [`buf.rs`](buf.rs) | `AlignedVec` — выровненная куча (32 байта), auto-padding до ×8 | ⚠️ unsafe внутри |
| [`lane.rs`](lane.rs) | Трейт `SimdLane` — интерфейс к железу (примитивы = 1 инструкция) | ✅ safe |
| [`lane_f32x8.rs`](lane_f32x8.rs) | `SimdLane for f32x8` — AVX2/NEON/WASM через `wide` | 🔒 unsafe |
| [`lane_f32.rs`](lane_f32.rs) | `SimdLane for f32` — скалярный fallback (хвосты) | ✅ safe |
| [`cond.rs`](cond.rs) | `Cond<S>` — читаемый branchless `cond().then().else_()` | ✅ safe |
| [`context.rs`](context.rs) | `SimdContext<S>` — state/params/шины; `MAX_STATES = 32` | ⚠️ unsafe |
| [`compute.rs`](compute.rs) | `ComputeNeuron` + `StepContext` — типизированный API моделей | ✅ safe |
| [`population.rs`](population.rs) | `NeuralPopulation` — безопасный SoA-раннер (главный вход) | ⚠️ unsafe внутри |
| [`runner.rs`](runner.rs) | `run_neuron*` — legacy raw-pointer раннеры (≤6 состояний) + макросы load/save_state | 🔒 unsafe |
| [`util.rs`](util.rs) | `load8` / `store8` / `load8_i32` — raw SIMD load/store хелперы | 🔒 unsafe |
| [`lane_tests.rs`](lane_tests.rs) | Consistency-тесты «SIMD vs scalar» для всех операций | тесты |

---

## Поток исполнения (один шаг популяции)

```text
NeuralPopulation::<LIFModel>::step(i_in, spikes_out)   ← безопасный вход
    ├── n ≤ 8:   step_scalar()   — f32 контекст, по 1 нейрону
    └── n > 8:   step_simd()     — f32x8 контекст, по 8 нейронов
           └── хвост (n8..n):    скалярный fallback
```

`population.rs` для каждого чанка:
1. `load_state_simd!` — читает `M::N_STATE` каналов из SoA-памяти (`read_unaligned`)
2. Создаёт `SimdContext<S>` (state, params_ptr, шины i_syn/i_pre/i_post, dt)
3. Вызывает `M::step(&mut ctx)` — модель пишет физику
4. `save_state_simd!` — пишет состояние обратно, снимает `spike_mask` в `i32`-спайки

Legacy-раннеры (`run_neuron`, `run_neuron_ext`, `run_neuron_par`) — то же самое,
но с 6 сырыми указателями и опциональными шинами pre/post (для STDP).

---

## SoA-раскладка памяти

```text
capacity = n.div_ceil(8) * 8    (padding до кратности 8)

params: [dt×cap] [inv_tau×cap] [v_rest×cap] …
state:  [v×cap]  [u×cap]      [refr×cap]   …

Шаг колонки = capacity — SIMD-чанки всегда смежные.
```

Параметры и состояния хранятся **колонками** (Structure of Arrays):
`ctx.p(idx)` грузит 8 значений подряд из колонки `idx` одним векторным
загрузом; `ctx.pf(idx)` берёт скаляр для текущего офсета.

---

## Ключевые контракты

### `SimdLane` — примитивы = 1 инструкция

Трейт — единственный интерфейс, против которого пишут модели. У него две
реализации: `f32x8` (SIMD) и `f32` (скаляр). Обе обязаны вести себя
**бит-идентично** на математике (проверяется `lane_tests.rs`).

Маски: аппаратная конвенция — **NaN = true, 0.0 = false**.
⚠️ Не путайте маску (`-1.0`/NaN) с арифметическим спайком (`1.0`) — для
математики пластичности используйте `spike_to_float` (в `ops/spike.rs`).

### `MAX_STATES = 32`

Единственное место, где ограничено число состояний модели. Макросы
`load/save_state_{scalar,simd}!` итерируют до `M::N_STATE` на этапе компиляции
с гвардом `const { assert!(M::N_STATE <= MAX_STATES); }` — модель с 33
состояниями **не скомпилируется**, а не упадёт в рантайме.

Legacy `run_neuron*` принимают ровно 6 указателей по сигнатуре — для моделей
с >6 состояниями используйте `NeuralPopulation`.

### Clean Bus

Все внешние события входят только через контекст:
- `i_syn` — постсинаптический ток (клампится до ±100 в раннере)
- `i_pre` / `i_post` — шины спайков для STDP/пластичности (опционально)

---

## Безопасность — где живёт `unsafe`

| Файл | unsafe | Почему |
|------|:------:|--------|
| `lane_f32x8.rs` | 🔒 | `_mm256_*` интринсики, `read_unaligned` |
| `runner.rs` | 🔒 | raw-pointer `run_neuron*`, макросы load/save |
| `util.rs` | 🔒 | `load8`/`store8` сырые указатели |
| `buf.rs` | ⚠️ | `std::alloc::alloc` (единственное место аллокации) |
| `context.rs` | ⚠️ | `load_param` разыменование указателей |
| `population.rs` | ⚠️ | unaligned SIMD load/store из кучи |

Всё выше по слоям (`ops/`, `zoo/`, `api/`) — **safe Rust**.
Правило: если ядро скомпилировалось — оно корректно.

---

## Два способа писать модели

### 1. Типизированный (рекомендуется) — `#[derive(NeuronModel)]` + `ComputeNeuron`

```rust
impl ComputeNeuron for LIFModel {
    type State<S: SimdLane>  = LIFState<S>;
    type Params<S: SimdLane> = LIFParams<S>;

    fn step<S: SimdLane>(state: &mut Self::State<S>, params: &Self::Params<S>, ctx: &mut StepContext<S>) {
        let dv = (params.v_rest - state.v + ctx.i_syn) * params.inv_tau_m * ctx.dt_simd();
        let v_new = state.v + dv;
        let spike = v_new.cmp_ge(params.v_thresh);
        ctx.emit_spike(spike);
        state.v = spike.cond().then(params.v_reset).else_(v_new);
    }
}
```

Мост `Model::step` → `ComputeNeuron::step` генерирует `#[derive(NeuronModel)]`
(zero-cost: один инлайн, типизированные геттеры → те же SoA-адреса).

### 2. Индексный (legacy) — `SimdContext` + опкоды

```rust
ctx.set_s(0, ctx.s(0) + (ctx.p(2) - ctx.s(0) + ctx.i()) * ctx.p(1) * ctx.dt());
```

Био-опкоды (`leak_integrate`, `fire_reset`, `lif_step`, `edge_spike`, …)

---

## Изменение числа состояний

1. Поднять `MAX_STATES` в `context.rs` (если нужно >32)
2. Больше ничего! Макросы и `NeuralPopulation` уже итерируют до `M::N_STATE`
3. Legacy `run_neuron*` — не трогаем: у них фиксированная сигнатура на 6

---

## Тесты

```bash
# Consistency «SIMD vs scalar» всех операций лейна:
cargo test -p pillars --lib core::lane_tests

# Контекст (state/params/шины):
cargo test -p pillars --lib core::context::tests

# Био-опкоды (leak_integrate, fire_reset, lif_step, …):

# Популяция и раннеры:
cargo test -p pillars --lib core::population
cargo test -p pillars --lib core::runner
```
