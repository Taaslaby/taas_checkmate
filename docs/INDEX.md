# 📚 ГЛАВНЫЙ ИНДЕКС ДОКУМЕНТАЦИИ ДВИЖКА TAAS-CHECKMATE

Настоящий справочник содержит документацию биофизического нейромоторного движка **`taas-checkmate`**.

---

## 📁 Карта документации ядра движка (`docs/`)

```text
docs/
├── models/                                # 🧠 Биофизические нейронные и мышечные модели ядра
│   ├── NEURON_MODELS.md                   # 7 Моделей нейронов: LIF, AdEx, HH, Izhikevich, CableDendrite, STP, SomatoDendritic
│   ├── ATLASOV_NEUROMOTOR.md              # Мышцы Атласова Ca²⁺, SERCA-насосы, Tug-of-War моменты, CGS/SI единицы
│   └── RECEPTORS_AND_NEUROTRANSMITTERS.md # Рецепторы: nAChR, GluCl (глутаматное торможение), GABA_A, GABA_B, NMDA
│
├── embodied/                              # 🦗 Модуль физического воплощения в MuJoCo
│   └── VNC_BIOMECHANICS.md                # Спинномозговой коннектом VNC (153k нейронов, 4M синапсов) в MuJoCo 3D
│
├── architecture/                          # ⚡ Высокопроизводительные SIMD и CUDA ядра
│   └── SIMD_CUDA_ARCHITECTURE.md          # Rust AVX-512/NEON SIMD, CSR матрицы, CUDA C++ ядра (1380 шагов/сек)
│
├── API_REFERENCE.md                       # Полный справочник Rust API
└── PYTHON_API_RU.md                       # Руководство по Python PyO3 API
```

---

### 🔗 Быстрые ссылки на модули ядра:
- 🧠 [NEURON_MODELS.md](file:///home/taaslab/Проекты/asm_engines/taas_checkmate/docs/models/NEURON_MODELS.md) — Математика и формулы всех 7 биофизических нейронных моделей.
- 🏋️ [ATLASOV_NEUROMOTOR.md](file:///home/taaslab/Проекты/asm_engines/taas_checkmate/docs/models/ATLASOV_NEUROMOTOR.md) — Кинетика кальция саркоплазмы, SERCA-насосы и развиваемый момент суставов.
- 🧪 [RECEPTORS_AND_NEUROTRANSMITTERS.md](file:///home/taaslab/Проекты/asm_engines/taas_checkmate/docs/models/RECEPTORS_AND_NEUROTRANSMITTERS.md) — Рецепторы nAChR и тормозные каналы GluCl у насекомых.
- 🦗 [VNC_BIOMECHANICS.md](file:///home/taaslab/Проекты/asm_engines/taas_checkmate/docs/embodied/VNC_BIOMECHANICS.md) — Интеграция 153k нейронов VNC коннектома с MuJoCo 3D.
- ⚡ [SIMD_CUDA_ARCHITECTURE.md](file:///home/taaslab/Проекты/asm_engines/taas_checkmate/docs/architecture/SIMD_CUDA_ARCHITECTURE.md) — Rust SIMD 8-lane alignment, CSR структуры и GPU CUDA ядра.
- 📘 [PYTHON_API_RU.md](file:///home/taaslab/Проекты/asm_engines/taas_checkmate/docs/PYTHON_API_RU.md) — Руководство по интеграции с Python через PyO3.
