# Документация taas-checkmate

Актуальная карта документации. Устаревшие документы (планы завершённых
миграций, старые аудиты) удалены — история хранится в git.

## С чего начать

- [../README.md](../../README.md) — обзор проекта, быстрый старт
- [README_SAKHA.md](README_SAKHA.md) — то же на якутском

## API

- [API_REFERENCE.md](API_REFERENCE.md) — Rust API: реэкспорты корневого крейта
- [PYTHON_API_RU.md](PYTHON_API_RU.md) — Python API (PyO3)
- [RL_API.md](RL_API.md) — обучение с подкреплением, дофамин-STDP

## Архитектура и математика

- [SIMD_CUDA_ARCHITECTURE.md](SIMD_CUDA_ARCHITECTURE.md) — SIMD/CUDA устройство движка
- [MATHEMATICAL_AUDIT.md](MATHEMATICAL_AUDIT.md) — сверка формул с научными статьями
- [T3_ARCHITECTURE.md](T3_ARCHITECTURE.md) — осцилляторная система T3
- [architecture/NAMING.md](architecture/NAMING.md) — принципы нейминга (pillars/amalgam)

## Валидация против NEURON

- [VALIDATION_NEURON.md](VALIDATION_NEURON.md) — точность всех моделей против NEURON 9 (сводный отчёт + разбор Izhikevich)

## Модели нейронов и мышц

- [models/NEURON_MODELS.md](models/NEURON_MODELS.md) — все 7 биофизических моделей
- [models/RECEPTORS_AND_NEUROTRANSMITTERS.md](models/RECEPTORS_AND_NEUROTRANSMITTERS.md) — рецепторы и медиаторы
- [models/ATLASOV_NEUROMOTOR.md](models/ATLASOV_NEUROMOTOR.md) — мышечное ядро Atlasov Ca²⁺
- [models/HUMAN_HAND_ATLASOV.md](models/HUMAN_HAND_ATLASOV.md) — применение к руке человека

## Воплощение (embodied)

- [embodied/VNC_BIOMECHANICS.md](embodied/VNC_BIOMECHANICS.md) — VNC дрозофилы в MuJoCo
- [embodied/ATLASOV_MOTOR_PHASE1.md](embodied/ATLASOV_MOTOR_PHASE1.md) — архитектура FlyWire embodied теста

## Эксперименты (корень репозитория)

- [docs/PERFORMANCE_GROW.md](../../docs/PERFORMANCE_GROW.md) — бенчмарки growth engine vs BioDynaMo
- [docs/EXPERIMENT_GENOME_TO_GAMMA.md](../../docs/EXPERIMENT_GENOME_TO_GAMMA.md) — из генома в гамма-ритм
- [docs/TASKS.md](../../docs/TASKS.md) — бэклог
