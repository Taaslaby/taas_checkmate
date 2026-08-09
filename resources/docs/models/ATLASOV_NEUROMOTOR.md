# 🏋️ ЯДРО МЫШЦ И ИМПЕДАНСА ATLASOV Ca²⁺ NEUROMOTOR

Документ содержит полное описание биофизического ядра мышечного сокращения **Atlasov Ca²⁺ Neuromotor**, интегрированного в движок **`taas-checkmate`**.

---

## 🔬 Принцип работы ядра

Модель Атласова симулирует саркоплазматическую динамику $Ca^{2+}$, работу SERCA-насосов ретикулярного захвата и динамическую перестройку жесткости $K(t)$ и вязкости $B(t)$ суставов.

```text
  [Моторные спайки MN] ➔ Выброс Ca²⁺ в саркоплазму (d_ca)
         ↓
  [Актин-миозиновые мостики] ➔ Напряжение T(t) = T_max * Ca² / (Ca² + Kd²)
         ↓
  [Tug-of-War антагонистов] ➔ ΔF = T_ext - T_flex; τ = ΔF - K(t)*q - B(t)*qd
         ↓
  [SERCA Насос] ➔ Захват Ca²⁺ обратно в саркоплазматический ретикулум
```

---

## 📐 Формулы и SIMD 8-Alignment в Rust

Каноническая Rust-реализация находится в
`crates/pillars/src/zoo/neurons/atlasov_neuromotor.rs`, а stateful/API-обвязка — в
`crates/pillars/src/api/neurons/atlasov.rs`. Все массивы строго выравниваются
кратно 8 элементам (`n_muscles.is_multiple_of(8)` и `n_joints = n_muscles // 2`).

Важно: `ca` — безразмерный activation pool модели, а не автоматически
калиброванная концентрация в µM. `t_max`, `l0`, `v_max` и временные константы
становятся физическими величинами только после калибровки на конкретную
мышцу/экспериментальные записи.

### 1. Накопление Ca²⁺ и выкачивание SERCA:
В коде используется точная дискретная рекурсия (сначала spike-influx,
затем exponential pump):
$$Ca_{t+1} = (Ca_t + d_{ca} S_t)\,e^{-\Delta t/\tau_{pump}}$$
При активном reset дополнительно применяется множитель
$$(1 - \eta_{reset})$$
и результат ограничивается снизу нулём. Это не явный Euler-шаг исходного
дифференциального уравнения.

### 2. Развиваемое мышечное напряжение:
$$T_i(t) = T_{max} \cdot \frac{[Ca^{2+}]_i^2}{[Ca^{2+}]_i^2 + K_d^2}$$

### 3. Момент сустава и динамический импеданс:
В каноне пары каналов индексируются как `flex=2j`, `ext=2j+1`, а
`joint_delta_f = T_ext - T_flex`:
$$K_j(t) = K_0 + \beta \cdot \big(T_{flex, j}(t) + T_{ext, j}(t)\big)$$
$$B_j(t) = B_0 + \alpha \cdot \big(T_{flex, j}(t) + T_{ext, j}(t)\big)$$
$$\tau_{applied, j} = \big(T_{ext, j} - T_{flex, j}\big) - K_j(t) \cdot q_j - B_j(t) \cdot \dot{q}_j$$

В force–velocity кривой этой реализации `velocity >= 0` считается
concentric (shortening), а `velocity < 0` — eccentric (lengthening); это
явная convention модели и должна совпадать с направлением velocity в dataset.

---

## 🧪 Физическая валидация

Канонические инвариантные тесты находятся в Rust-модуле Atlasov и проверяют:

- точную рекурсию Ca²⁺ pump;
- ускоренное удаление Ca²⁺ при reset;
- неотрицательность и конечность Ca/tension;
- bell-shaped force–length и bounded force–velocity;
- знак agonist/antagonist output и рост joint stiffness/damping.

Для реальной записи VNC/мышцы используется data-driven validator:

```bash
python tests/validate_atlasov_real_data.py recording.csv \\
  --t-max 2e-6 --l0 3e-4 --v-max 0.5
```

CSV должен содержать `time,spike,length,velocity`; `spike` обязан быть бинарным
0/1, измеренная колонка `force` необязательна. Если `force` отсутствует,
проверяется только физическая устойчивость и форма отклика. Для calibrated
force acceptance нужно явно добавить `--max-nrmse`; без экспериментального
ground truth абсолютная точность в µN не заявляется. Параметры `dt`,
`tau_pump`, `t_max`, `kd`, `l0`, `v_max` и `width` валидируются как конечные
положительные величины.

## ⚙️ Система единиц (СГС vs СИ)

| Параметр | Описание | Значение СГС (CGS) | Значение СИ (SI) |
|:---|:---|:---:|:---:|
| **Масса мухи ($m$)** | Полная масса туловища | $0.0151\text{ g}$ | $1.51 \times 10^{-5}\text{ kg}$ |
| **Ускорение тяжести ($g$)** | Гравитационное поле | $981\text{ cm/s}^2$ | $9.81\text{ m/s}^2$ |
| **Сила тяжести ($W$)** | Вес мухи | $14.83\text{ dynes}$ | $148.3\,\mu\text{N}$ |
| **$T_{max}$ (Момент)** | Максимальный момент мышц | $0.8 \dots 2.5\text{ dyn}\cdot\text{cm}$ | $0.008 \dots 0.025\text{ N}\cdot\text{m}$ |
