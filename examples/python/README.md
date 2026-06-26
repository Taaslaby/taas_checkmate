# 🐍 Python Examples — `taas-checkmate`

Python скрипты для визуализации, бенчмаркинга и исследования через PyO3 биндинги.
Требуют сборки `taas-checkmate` с feature `python`:

```bash
cd taas_checkmate && cargo build --features python
```

Все скрипты запускать из корня проекта:

```bash
cd taas_checkmate
PYTHONPATH=target/debug python3 examples/python/<script>.py [args]
```

---

## Примеры

| Скрипт | Описание | Запуск |
|:-------|:---------|:-------|
| `plot_raster.py` | Визуализация spike raster из CSV. Поддерживает rate bars, ISI гистограмму, demo-режим с Izhikevich через PyO3. | `python3 examples/python/plot_raster.py --demo` |
| `bench_connectome_with_raster.py` | Connectome benchmark: генерация CSR коннектома → Izhikevich + CSR propagate → spike raster CSV → plot. 138K нейронов, 500 шагов. | `python3 examples/python/bench_connectome_with_raster.py --quick` |
| `bench_nmnist.py` | N-MNIST классификация через Rust E-Prop Izhikevich. Poisson encoding, per-synapse eligibility. | `python3 examples/python/bench_nmnist.py` |
| `download_mnist.py` | Скачивание MNIST через зеркала (stdlib only, без torchvision). | `python3 examples/python/download_mnist.py` |
| `check_sparker.py` | Проверка Спарковки: спайки → trace → hologram(3) → T3. 5 тестов на разделимость голограмм. | `python3 examples/python/check_sparker.py` |
| `check_sparker_fly.py` | Проверка Спарковки на реальном FlyWire/FABF коннектоме дрозофилы. | `python3 examples/python/check_sparker_fly.py` |

## Зависимости

- **matplotlib** — для `plot_raster.py`
- **numpy** — общая зависимость
- **scikit-learn** — для `check_sparker.py`
- **torch / torchvision** — для `bench_nmnist.py` (fallback)
- **tonic** — для N-MNIST через tonic (опционально)

```bash
pip install matplotlib numpy scikit-learn
```

## Данные

MNIST IDX файлы — `examples/data/`. Скачивание:

```bash
python3 examples/python/download_mnist.py
```

## Примечание

Скрипты используют `PYTHONPATH=target/debug` для подключения PyO3 модуля.
Без этого `import taas_checkmate` не сработает.
