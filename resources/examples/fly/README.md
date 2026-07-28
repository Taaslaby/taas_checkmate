# 🧠 FlyWire Brain Simulation

Полная симуляция мозга *Drosophila melanogaster* на реальном FlyWire коннектоме
(138K нейронов, 3.7M синапсов) с MuJoCo телом.

```
Сенсоры → Мозг (LIF/Izhikevich) → VNC → MuJoCo → Движение
```

## 🗂️ Структура

```
fly/
├── models/                  # MuJoCo 3D модель мухи
│   ├── scene.xml            #   сцена со стенами и едой
│   ├── fruitfly.xml         #   анатомическая модель
│   └── assets/              #   80+ .obj мешей
├── scripts/
│   ├── flywire_brain.py     # ⭐ Ядро: CSR + LIF/Izhikevich/Rate/ψ
│   ├── vnc_walk.py          # ⭐ GUI симуляция (ходьба)
│   ├── test_arena.py        # ⭐ 5 тестов: BABY/ALIVE/LIMBO/DEAD
│   ├── profile_v2.py        #   профилирование производительности
│   ├── diagnose_v2.py       #   диагностика коннектома
│   ├── validate_vs_brian2.py#   валидация против Brian2
│   └── autotune_params.py   #   подбор Izhikevich параметров
│   └── autotune_results/    #   готовые параметры
├── archive/
│   └── vnc_controller.py    # VNC контроллер (legacy, используется)
├── config.py                # Настройки путей
└── README.md
```

## 🚀 Быстрый старт

```bash
# LIF мозг (Shiu et al. 2024) — GUI
cd taas_checkmate
python3 fly/scripts/vnc_walk.py --model lif --duration 30

# Izhikevich мозг
python3 fly/scripts/vnc_walk.py --model izhi

# Тестовый полигон (без GUI)
python3 fly/scripts/test_arena.py --fast
python3 fly/scripts/test_arena.py --duration 10 --trials 3
```

## 🧪 Test Arena

| Тест | Что проверяет |
|---|---|
| 1. Baseline Walking | DNg100=80 → муха ходит? |
| 2. DNg100 Perturbation | DNg100=0 → останавливается? |
| 3. Looming Response | Стена близко → тормозит? |
| 4. Food Seeking | Еда сбоку → поворачивает? |
| 5. Random Baseline | Мозг ≠ random шум? |

**Вердикты:** BRAIN_ALIVE (4+/5) · BRAIN_LIMBO (2-3/5) · BRAIN_DEAD (<2/5)

## 🧠 Модели нейронов

| Модель | Описание | Скорость |
|---|---|---|
| `lif` | LIF (Shiu et al. 2024, Nature) | ~0.01× RT |
| `izhi` | Izhikevich (биологичный) | ~0.02× RT |
| `rate` | Rate-based ANN | ~100× RT |
| `psi` | ψ-wave (экспериментальная) | ~0.01× RT |

## 🎮 Управление (vnc_walk.py)

| Клавиша | Действие |
|---|---|
| `←` `→` | Поворот влево/вправо |
| `1` `2` | DN частота |
| `3` `4` | CPG частота |
| `7` `8` | CPG частота 1-20Hz |
| `9` | CPG амплитуда |
| `0` | Сброс поворота |
| `R` | Полный сброс |

## 📚 Почитать

- [Shiu et al. 2024 — Drosophila brain model](https://www.nature.com/articles/s41586-024-07621-6)
- [Lappalainen et al. 2024 — Connectome-constrained networks](https://www.nature.com/articles/s41586-024-07591-9)
- [FlyWire](https://flywire.ai/)
- [flyvis](https://github.com/TuragaLab/flyvis)
