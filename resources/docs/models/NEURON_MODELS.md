# 🧠 БИОФИЗИЧЕСКИЕ МОДЕЛИ НЕЙРОНОВ В ЯДРЕ TAAS-CHECKMATE

Документ представляет полное техническое и математическое описание всех 7 биофизических нейронных моделей, встроенных в высокопроизводительное ядро **`taas-checkmate`** (Rust + CUDA + SIMD + PyO3).

---

## 📋 Таблица поддерживаемых моделей движка

| Модель | Файл исходного кода | Порядковый дифференциал | Сложность вычислений | Область применения |
|:---|:---|:---:|:---:|:---|
| **LIF** | `crates/kernels/src/neuron/lif.rs` | 1-й порядок (ODE) | $\mathcal{O}(1)$ | Масштабные SNN (FAFB 76k) |
| **AdEx** | `crates/kernels/src/neuron/adex.rs` | 2-й порядок (ODE) | $\mathcal{O}(1)$ | Адаптация частоты, вспышечный спайкинг |
| **HH (Hodgkin-Huxley)** | `crates/kernels/src/neuron/hh.rs` | 4-й порядок (ODE) | $\mathcal{O}(4)$ | Автономный пейсмейкинг, аксонный спайкинг |
| **Izhikevich** | `crates/kernels/src/neuron/izhikevich.rs` | 2-й порядок (ODE) | $\mathcal{O}(1)$ | Реальный VNC коннектом (153k нейронов) |
| **CableDendriteState** | `crates/kernels/src/neuron/dendrite.rs` | ПФУ (PDE) кабеля | $\mathcal{O}(K)$ | Активные $Na^+/K^+/Ca^{2+}$ дендритные спайки |
| **STP (Tsodyks-Markram)** | `crates/kernels/src/neuron/stp.rs` | 2-й порядок (ODE) | $\mathcal{O}(1)$ | Кратковременная пластичность синапсов |
| **SomatoDendriticNeuron** | `crates/core/src/neuron/soma_dendrite.rs` | Мульти-компартмент | $\mathcal{O}(K+1)$ | Мульти-рецепторное торможение (GluCl, GABA) |

---

## 1. LIF (Leaky Integrate-and-Fire)

Быстрая токовая модель утечки для симуляции миллионных нейронных ансамблей.

### Математическая формула:
$$\tau_m \frac{d V(t)}{dt} = -\big(V(t) - V_{rest}\big) + R_m \cdot I_{syn}(t)$$

### Параметры по умолчанию:
- $V_{rest} = -70.0\text{ mV}$
- $V_{thresh} = -55.0\text{ mV}$
- $V_{reset} = -75.0\text{ mV}$
- $\tau_m = 20.0\text{ ms}$
- $\tau_{ref} = 2.0\text{ ms}$

---

## 2. AdEx (Adaptive Exponential Integrate-and-Fire)

Модель с экспоненциальной деполяризацией вблизи порога и адаптационной переменной $w(t)$.

### Уравнения состояния:
$$\tau_m \frac{dV}{dt} = -\big(V - V_{rest}\big) + \Delta_T \exp\left(\frac{V - V_T}{\Delta_T}\right) - R_m w + R_m I$$
$$\tau_w \frac{dw}{dt} = a \big(V - V_{rest}\big) - w$$

### При обновлении спайка:
$$\text{If } V \ge V_{peak} \implies V \leftarrow V_{reset}, \quad w \leftarrow w + b$$

---

## 3. HH (Hodgkin-Huxley)

Биофизическая модель 4-го порядка с вольт-зависимой кинетикой ионных ворот $m, h, n$.

### Уравнение мембранного потенциала:
$$C_m \frac{dV}{dt} = I_{ext} - \bar{g}_{Na} m^3 h (V - E_{Na}) - \bar{g}_K n^4 (V - E_K) - g_L (V - E_L)$$

### Кинетика воротных переменных:
$$\frac{dx}{dt} = \alpha_x(V) (1 - x) - \beta_x(V) x, \quad x \in \{m, h, n\}$$

---

## 4. CableDendriteState (Активные кабельные дендриты)

Симулирует кабельное уравнение распространения потенциала по компартментам с нелинейными ионными токами $I_{active}$.

### Пространственно-временное уравнение кабеля:
$$C_m \frac{d V_j}{dt} = \frac{V_{j-1} - 2V_j + V_{j+1}}{R_a} + g_L (E_L - V_j) + I_{active, j}$$
