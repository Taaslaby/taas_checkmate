# NEURON Gap Analysis — где мы проигрываем и почему

> Аудит 2026-08-17. Сверка pillars/taas-checkmate против NEURON (gold standard
> биофизической симуляции) + разбор схемы NeuroML2 для каналов.
> Парный документ: [NEURON_CATCHUP_PLAN.md](NEURON_CATCHUP_PLAN.md).

---

## 1. Резюме

Мы проигрываем NEURON **не в скорости** (там мы его на порядки обгоняем), а в
**биофизической выразительности**. Всё, что требует произвольных ионных каналов,
настоящей ветвящейся морфологии, жёстких адаптивных интеграторов и
реакционно-диффузионных систем — территория NEURON. Наша сила — масштаб,
реалтайм и встраиваемость.

Ключевой практический вывод: **самый дешёвый шаг с наибольшим рычагом — обобщить
ионные каналы**. Причина ниже: гейты NeuroML2 — замкнутые формулы, а не
произвольный код, поэтому «догнать NEURON по каналам» — решаемая за одну фазу
задача (см. план).

---

## 2. Где проигрываем (по убыванию важности)

### 2.1 🔴 Произвольные ионные каналы (NMODL)

| | NEURON | pillars |
|---|---|---|
| Механизмы | NMODL: любой вольт-/лиганд-зависимый канал, кинетические схемы, диффузия, буферизация, активный транспорт | каналы **захардкожены** |
| Библиотека | тысячи готовых моделей в ModelDB | нет импорта |

В `crates/pillars/src/kernels/hh_channels.rs` тип гейта — это enum из 4 вариантов:

```rust
pub enum GateType { NaM, NaH, Kn, CatC }
```

`build_channel_meta()` маппит строки `"m" | "h" | "n" | "c"` на эти варианты
(фолбэк — `NaM`). Никакого произвольного α(V)/β(V) или кинетической схемы в
рантайме не существует. Следствие: **нельзя загрузить ни одну произвольную
модель канала** (hcn.mod, km.mod, каналы из Allen/HBP/ModelDB).

### 2.2 🔴 Настоящая ветвящаяся морфология (cable equation)

| | NEURON | pillars |
|---|---|---|
| Топология | `Section` → **ветвящиеся деревья**, `nseg`, свойства как функция позиции `soma.v(0.5)` | **линейная цепочка** компартментов (ball-and-stick) |
| Импорт | Import3D: SWC / Neurolucida / Eutectic | нет |
| Дискретизация | d_lambda-правило | ручная |

`kernels/dendrite/cable.rs` (`CableDendriteState`) — цепочка: осевой `R_a` между
соседними компартментами + `R_m` к соме. Ветвления нет. Поэтому NeuroML-парсер
честно пропускает `<cell>`, `<morphology>`, `<segmentGroup>`,
`<biophysicalProperties>` — с warning, а не молча.

### 2.3 🟠 Жёсткие адаптивные интеграторы

| NEURON | pillars |
|---|---|
| implicit Euler (дефолт), Crank–Nicolson (2-й порядок), адаптивные CVODE/IDA (variable-order/step, свой интегратор на клетку), DAE для algebraic-ограничений | только фиксированный шаг: explicit Euler (SIMD), implicit Euler (скалярный хвост), exponential Euler для гейтов |

Нет адаптивности, нет Crank–Nicolson, нет DAE. Потеря точности уже
задокументирована: в `crates/pillars/tests/hh_vs_neuron.rs` конфиг `I200` помечен
*"NEURON CVode vs fixed-step edge-triggered can diverge"*. На жёстких системах
(малые τ, высокий Q10) вынуждены мельчить dt и клампить.

### 2.4 🟠 Реакция-диффузия (rxd) и стохастические каналы

- NEURON: 3D reaction-diffusion (кальциевые волны, вторичные мессенджеры),
  стохастические каналы (ChannelBuilder — дискретные переходы состояний).
- pillars: есть точечные опы `ops/physics/` (`diffusion`, `field`, `heating`,
  `metabolism`) и `ops/neurochemistry/` (`gpcr`, `receptors`, `signaling`) — но
  это **не 3D-rxd** и без стохастики каналов.

### 2.5 🟡 Электрические элементы и инструментарий

- NEURON: gap junctions, эфаптические связи, **voltage clamp / dynamic clamp**
  (Linear Circuit Builder), электротонический анализ (Impedance tools),
  автофиттинг (Multiple Run Fitter), CellBuilder/ChannelBuilder.
- pillars: gap junctions в NeuroML — `recognized-but-unsupported` (warning).
  Voltage/dynamic clamp нет. Только `electrotonic_length()`-хелпер, не полный
  импеданс-анализ. GUI — connscope (вьювер), не билдер моделей.

---

## 3. Где выигрываем

| Ось | pillars/taas | NEURON |
|---|---|---|
| Масштаб | **130K FlyWire в реалтайме**, 1M изикевичей/сек | единицы–сотни клеток |
| Железо | SIMD (AVX2/NEON) + CUDA | CPU (MPI — мультинода) |
| Встраиваемость | Rust lib + Python + WASM, без внешнего рантайма | нужен NEURON/`nrnivmodl` |
| Обучение | STDP, E-Prop, RL встроены | пластичность есть, RL нет |
| Закрытый цикл | детерминированный fixed-step → робототехника | переменный шаг усложняет realtime |

---

## 4. Исследование: схема NeuroML2 для каналов

Источники:
- [NeuroML2 → Channels (docs.neuroml.org)](https://docs.neuroml.org/Userdocs/Schemas/Channels.html)
- [NeuroML_v2.0.xsd (GitHub)](https://github.com/NeuroML/NeuroML2/blob/master/Schemas/NeuroML2/NeuroML_v2.0.xsd)
- [What is NEURON (neuronsimulator.org)](https://www.neuronsimulator.org/en/latest/guide/what_is_neuron.html)
- [NEURON (Scholarpedia)](http://www.scholarpedia.org/article/Neuron_simulation_environment)

### 4.1 Главная находка: гейты NeuroML2 — замкнутые формулы

В отличие от NMODL (произвольный код), NeuroML2-гейты HH задаются **тремя
замкнутыми формами** скоростей и **тремя** — стационарных значений. Никакого
произвольного кода, только параметры `rate/midpoint/scale`.

**Rate-формы** (`forwardRate` / `reverseRate`, `x = (v − midpoint)/scale`):

| Тип | Формула |
|---|---|
| `HHExpRate` | `r = rate · exp((v − midpoint)/scale)` |
| `HHSigmoidRate` | `r = rate / (1 + exp(−(v − midpoint)/scale))` |
| `HHExpLinearRate` | `r = rate · x / (1 − exp(−x))`; при `x=0` → `r = rate` |

**Variable-формы** (`steadyState`):

| Тип | Формула |
|---|---|
| `HHExpVariable` | `x = rate · exp((v − midpoint)/scale)` |
| `HHSigmoidVariable` | `x = rate / (1 + exp(−(v − midpoint)/scale))` |
| `HHExpLinearVariable` | `x = rate · a / (1 − exp(−a))`, `a = (v − midpoint)/scale` |

**Time course**: `fixedTimeCourse` → `t = tau`.

### 4.2 Типы гейтов и их динамика

Общая динамика всех `gateHH*`: `inf`, `tau` → `dq/dt = (inf − q)/tau`,
`fcond = q^instances`. Различаются источником `inf`/`tau`:

| Гейт | `inf` | `tau` | Дети |
|---|---|---|---|
| `gateHHrates` | `α/(α+β)` | `1/((α+β)·rateScale)` | forwardRate, reverseRate |
| `gateHHtauInf` | `steadyState` | `timeCourse/rateScale` | timeCourse, steadyState |
| `gateHHInstantaneous` | `steadyState` | `0` (q = inf) | steadyState |
| `gateHHratesTau` | `α/(α+β)` | `timeCourse/rateScale` | forwardRate, reverseRate, timeCourse |
| `gateHHratesInf` | `steadyState` | `1/((α+β)·rateScale)` | forwardRate, reverseRate, steadyState |
| `gateHHratesTauInf` | `steadyState` | `timeCourse/rateScale` | все четыре |
| `gateFractional` | — | — | subGate с `fractionalConductance` |

Q10: `q10Fixed` (`fixedQ10`) и `q10ExpTemp`
(`q10Factor^((T − experimentalTemp)/10)`).

### 4.3 Канонический XML (squid axon, Na-канал)

```xml
<ionChannelHH id="naChans" conductance="10pS" species="na">
  <gateHHrates id="m" instances="3">
    <forwardRate type="HHExpLinearRate" rate="1per_ms" midpoint="-40mV" scale="10mV"/>
    <reverseRate type="HHExpRate" rate="4per_ms" midpoint="-65mV" scale="-18mV"/>
  </gateHHrates>
  <gateHHrates id="h" instances="1">
    <forwardRate type="HHExpRate" rate="0.07per_ms" midpoint="-65mV" scale="-20mV"/>
    <reverseRate type="HHSigmoidRate" rate="1per_ms" midpoint="-35mV" scale="10mV"/>
  </gateHHrates>
</ionChannelHH>
```

### 4.4 Проверка: NeuroML ↔ наш `na_m_alpha` (бит-в-бит)

Наш код: `na_m_alpha(v) = 0.1 · vtrap(−(v+40), 10)`, `vtrap(x,y) = x/(exp(x/y)−1)`.

```
α_m = 0.1 · (v+40) / (1 − exp(−(v+40)/10))
```

NeuroML `HHExpLinearRate` (rate=1, midpoint=−40, scale=10):

```
x   = (v − (−40))/10 = (v+40)/10
r   = 1 · x / (1 − exp(−x)) = (v+40)/10 / (1 − exp(−(v+40)/10))
    = 0.1 · (v+40) / (1 − exp(−(v+40)/10))   ≡  α_m  ✅
```

`β_m` (`HHExpRate` rate=4, midpoint=−65, scale=−18):
`4·exp((v+65)/−18) = 4·exp(−(v+65)/18)` ≡ `na_m_beta` ✅.

**Вывод:** наши кальмарные rate-функции уже имеют точное NeuroML-представление.
Это идеальный якорь для round-trip валидации фазы 1–2 плана.

### 4.5 Что остаётся за рамками (отдельный, более тяжёлый уровень)

- **`gateKS`** — марковские кинетические схемы (`KSState`/`KSTransition`,
  `gateKSrates`). Нужны для многих реальных каналов (Ca-динамика, HCN в полной
  форме). Следующий горизонт после HH-гейтов.
- **Морфология** — `<cell>`/`<segment>`/`<biophysicalProperties>` — ветвящиеся
  деревья, не блокер для каналов.

---

## 5. Что это значит для NeuroML/NEURON-поддержки

Два жёстких блокера — **каналы (2.1)** и **морфология (2.2)** — ровно то, что
NeuroML-импортер сейчас скидывает в `warnings`: `ionChannel`, `ionChannelHH`,
`gateHH*`, `cell`, `morphology`, `biophysicalProperties`. Пока их нет, мы
поддерживаем только point-neuron подмножество NeuroML, а не «настоящие»
биофизические модели из ModelDB/Allen.

Рекомендуемый порядок работ — в [NEURON_CATCHUP_PLAN.md](NEURON_CATCHUP_PLAN.md).
