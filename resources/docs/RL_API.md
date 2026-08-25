#  RL + Dopamine-STDP API Reference

Спайковый Actor-Critic агент с 3-factor dopamine-modulated STDP пластичностью. Основан на Izhikevich (2007) и Sutton & Barto (2018).

---

## `ActorCriticAgent` — RL Agent

```
State → PoissonEncoder → InputSpikes → Hidden Izhikevich → Output LIF → Softmax → Action
                                          ↓                    ↓
                                     STDP traces          Critic LIF → V(s)
                                          ↓
                                     Eligibility e ←── TD error δ = r + γ·V(s') - V(s)
                                          ↓
                                     Δw = η · δ · e
```

### Конструктор

```rust
pub fn new(
    n_input: usize,   // размерность состояния
    n_hidden: usize,  // скрытых Izhikevich нейронов
    n_actions: usize,  // дискретных действий
    n_steps: usize,    // внутренних шагов симуляции на шаг среды (20-50)
    dt: f32,           // шаг симуляции в ms
    seed: u32,         // seed для RNG
) -> Self
```

**Пример:**
```rust
let mut agent = ActorCriticAgent::new(4, 32, 2, 30, 1.0, 42);
agent.lr_actor = 0.02;
agent.gamma = 0.99;
```

### Методы

#### `act(&mut self, state: &[f32]) -> (usize, f32)`

Прямой проход через спайковую сеть. Возвращает `(индекс_действия, V(s))`.

**Алгоритм:**
1. `state[i] ∈ [-1,1]` → rate = 10 + clamp(state, 0, 1) × 190 Hz
2. `n_steps` итераций: Poisson → input → hidden Izhikevich → output LIF → critic LIF
3. На каждом шаге: обновление STDP traces + eligibility per synapse
4. Softmax по firing rates output нейронов → выбор действия
5. V(s) = critic firing rate / 100, clamped [0, 1]

```rust
let (action, value) = agent.act(&[0.5, -0.3, 0.1, 0.8]);
```

#### `learn(&mut self, reward: f32, next_value: f32, done: bool) -> f32`

TD-learning. Возвращает TD error δ.

**Формула:** `δ = r + γ·V(s')·(1-done) - V(s)`
**Обновление:** `∀w: Δw = η · δ · e` (dopamine-gated)

```rust
let td = agent.learn(1.0, next_value, false);
```

#### `reset_episode(&mut self)`

Сброс нейронов и eligibility для новой эпизоды.

```rust
agent.reset_episode();
```

### Публичные поля (настройка)

| Поле | Тип | Default | Описание |
|------|-----|---------|----------|
| `dt` | `f32` | `1.0` | Шаг симуляции (ms) |
| `gamma` | `f32` | `0.95` | Discount factor |
| `lr_actor` | `f32` | `0.01` | Learning rate actor |
| `lr_critic` | `f32` | `0.01` | Learning rate critic |
| `tau_trace` | `f32` | `20.0` | τ STDP trace (ms) |
| `tau_e` | `f32` | `500.0` | τ eligibility decay (ms) |
| `a_ltp` | `f32` | `0.1` | LTP amplitude |
| `a_ltd` | `f32` | `0.12` | LTD amplitude |
| `w_min` / `w_max` | `f32` | `0.0` / `2.0` | Weight bounds |
| `tau_ref` | `f32` | `3.0` | Refractory period (ms) |
| `w_ih` | `Vec<f32>` | — | Weights input→hidden `[n_input × n_hidden]` |
| `w_ho` | `Vec<f32>` | — | Weights hidden→output `[n_hidden × n_actions]` |
| `w_ic` | `Vec<f32>` | — | Weights input→critic `[n_input]` |
| `elig_ih` | `Vec<f32>` | — | Eligibility input→hidden |
| `elig_ho` | `Vec<f32>` | — | Eligibility hidden→output |
| `elig_ic` | `Vec<f32>` | — | Eligibility input→critic |

> `w_ih`, `elig_ih` и т.д. — **column-major**: `w_ih[i * n_hidden + j]` = вес от входа `i` к скрытому `j`.

### Полный цикл

```rust
use taas_checkmate::ActorCriticAgent;

let mut agent = ActorCriticAgent::new(4, 32, 2, 30, 1.0, 42);
agent.lr_actor = 0.03;
agent.gamma = 0.9;

for episode in 0..100 {
    let mut state = initial_state();
    agent.reset_episode();
    let mut total = 0.0;

    for _ in 0..max_steps {
        let (action, value) = agent.act(&state);
        let (reward, next_state, done) = env.step(action);
        let (_, next_val) = agent.act(&next_state);
        agent.learn(reward, next_val, done);
        total += reward;
        state = next_state;
        if done { break; }
    }
    println!("Episode {episode}: {total:.1}");
}
```

---

## 📦 `DopamineSTDP` — 3-factor Plasticity

Low-level STDP synapse с дофаминовым gate.

### Формулы

```
tr_pre  ← tr_pre × (1 − dt/τ_pre) + pre_spike
tr_post ← tr_post × (1 − dt/τ_post) + post_spike
e       ← e + A_ltp × tr_pre × post − A_ltd × tr_post × pre
e       ← e × (1 − dt/τ_e)
Δw      = η · D · e
```

### Конструктор

```rust
pub fn new(n: usize, dt: f32) -> Self
```

Параметры по умолчанию (Izhikevich 2007):

| Параметр | Default |
|----------|---------|
| `tau_pre` | `20.0` ms |
| `tau_post` | `20.0` ms |
| `tau_e` | `1000.0` ms |
| `a_ltp` | `0.1` |
| `a_ltd` | `0.12` |
| `w_min` | `0.0` |
| `w_max` | `1.0` |

### Методы

```rust
// Обновление traces + eligibility по pre/post спайкам
stdp.step(&pre_spikes, &post_spikes);

// Дофаминовый gate: Δw = η · D · e
stdp.apply_dopamine(&mut weights, dopamine, lr);

// Очистка eligibility (после apply_dopamine)
stdp.clear_eligibility();

// Полный сброс
stdp.reset();
```

### Пример

```rust
use taas_checkmate::neuron::DopamineSTDP;

let n = 128;
let mut stdp = DopamineSTDP::new(n, 1.0);

// Каждый шаг: STDP eligibility
stdp.step(&pre_spikes, &post_spikes);

// При вознаграждении:
stdp.apply_dopamine(&mut weights, 1.0, 0.01);
stdp.clear_eligibility();
```

---

## ⚙️ SIMD Kernels (`crate::kernels`)

### `dopamine_traces_step`

SIMD (`f32x8`) обновление traces и eligibility.

```rust
pub unsafe fn dopamine_traces_step(
    trace_pre: *mut f32,     // [n]
    trace_post: *mut f32,    // [n]
    eligibility: *mut f32,   // [n]
    pre_spikes: *const i32,  // [n] 0/1
    post_spikes: *const i32, // [n] 0/1
    dt: f32,
    tau_pre: f32,
    tau_post: f32,
    tau_e: f32,
    a_ltp: f32,
    a_ltd: f32,
    n: usize,
);
```

### `dopamine_weight_update`

SIMD weight update: `w += η·D·e`, clamped `[w_min, w_max]`

```rust
pub unsafe fn dopamine_weight_update(
    weights: *mut f32,        // [n]
    eligibility: *const f32,  // [n]
    dopamine: f32,
    lr: f32,
    w_min: f32,
    w_max: f32,
    n: usize,
);
```

---

## 📦 Вспомогательные утилиты

### `test_mdp_episode`

Бенчмарк-задача для проверки обучения: 2-state MDP.

```
State 0 → action 0 = +1, action 1 = -0.5
State 1 → action 1 = +1, action 0 = -0.5
Optimal: action = state (получать +1 каждый шаг)
```

```rust
let total = test_mdp_episode(&mut agent, max_steps);
```

### `SpikeRaster`

Разрежённый коллектор спайков для визуализации.

```rust
let mut raster = SpikeRaster::new();
raster.record(step, &spikes, n_neurons);
let (dense, n_steps) = raster.to_dense(n_neurons);
raster.reset();
```

### `PoissonGenerator`

Генератор пуассоновских спайков (rate coding).

```rust
let mut gen = PoissonGenerator::new(n, dt, seed);
gen.set_rates(&[10.0, 50.0, 100.0]);
gen.step(&mut spikes_out);
```

---

## 🔬 Научные основы

| Компонент | Источник |
|-----------|----------|
| **Dopamine-STDP** | Izhikevich (2007) — *Solving the distal reward problem through linkage of STDP and dopamine signaling* |
| **Actor-Critic** | Sutton & Barto (2018) — *Reinforcement Learning: An Introduction* (ch. 13) |
| **E-Prop** | Bellec et al. (2020) — *A solution to the learning dilemma for recurrent networks of spiking neurons* |
| **Izhikevich model** | Izhikevich (2003) — *Simple model of spiking neurons* |
| **LIF** | Gerstner & Kistler (2002) — *Spiking Neuron Models* |
| **STDP** | Bi & Poo (1998) — *Synaptic modifications in cultured hippocampal neurons* |

---

## 📊 Re-exports

Из корня крейта (`taas_checkmate::`):

```rust
use taas_checkmate::{
    ActorCriticAgent,          // rl модуль
    test_mdp_episode,
    DopamineSTDP,              // neuron модуль
    PoissonGenerator,
    SpikeRaster,
    ShortTermPlasticity,
    // и все нейронные модели
};
```
