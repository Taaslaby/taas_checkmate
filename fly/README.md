# 🪰 Fly — Drosophila Connectome → MuJoCo Body

Интеграция полного коннектома дрозофилы (FlyWire мозг + MANC VNC) с биомеханической моделью мухи в MuJoCo.

## Структура

```
fly/
├── models/                  ← MuJoCo flybody (из google-deepmind/mujoco_menagerie)
│   ├── fruitfly.xml         ← MJCF модель тела мухи
│   ├── scene.xml            ← сцена для визуализации
│   └── assets/              ← 85 .obj мешей (ноги, крылья, голова, тело)
│
├── connectome/
│   ├── brain/               ← FlyWire мозг (138K нейронов)
│   │   ├── model.py         ← LIF модель на базе FlyWire (из Drosophila_brain_model)
│   │   └── utils.py         ← утилиты загрузки коннектома
│   ├── vnc/                 ← MANC спинной мозг (102K нейронов)
│   │   └── ...
│   ├── simulation.py        ← flygym симуляция
│   ├── anatomy.py           ← анатомия мухи (нейроны → тело)
│   └── rendering.py         ← рендеринг
│
├── scripts/
│   ├── explore_manc.py      ← загрузка MANC (нейроны, синапсы, моторные цепи)
│   └── fly_sim.py           ← минимальная симуляция MuJoCo + connectome
│
└── notebooks/               ← Jupyter ноутбуки
```

## Pipeline

```
FlyWire (мозг, 138K нейронов)
       │
       ▼ CvC (cervical connective)
MANC (VNC, 102K нейронов)
       │
       ▼ 737 motor neurons
MuJoCo flybody (85 мешей, 18 DOF ног + крылья)
       │
       ▼ sensory feedback
Sensory neurons → Ascending neurons → FlyWire
```

## Источники

| Компонент | Источник | Ссылка |
|:----------|:---------|:-------|
| MuJoCo flybody | Google DeepMind + HHMI Janelia | [mujoco_menagerie/flybody](https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody) |
| FlyWire коннектом | Princeton + HHMI Janelia | [flywire.ai](https://flywire.ai/) |
| MANC VNC | HHMI Janelia | [neuprint.janelia.org](https://neuprint.janelia.org) |
| Drosophila brain model | Shiu et al. 2024 | [github.com/philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) |
| NeuroMechFly v2 | EPFL NeLy | [github.com/NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym) |

## Данные

MANC и FlyWire CSV/freezer файлы находятся в `~/Загрузки/`.

## Запуск

```bash
# Исследовать MANC моторные нейроны
cd taas_checkmate
python3 fly/scripts/explore_manc.py

# Визуализировать MuJoCo flybody
python3 -m mujoco.viewer --mjcf fly/models/fruitfly.xml
```
