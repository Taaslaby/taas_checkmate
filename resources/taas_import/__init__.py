"""
taas_import — NeuroML2 & SONATA format importers for TAAS Checkmate
===================================================================

Конвертирует модели из стандартных нейронаучных форматов
в TAAS CSR + параметры нейронов для запуска на Rust SIMD.

Поддерживаемые форматы:
  - NeuroML2 (.nml + .lems)  — Open Source Brain, HBP, NeuroMorpho
  - SONATA (.h5)             — Allen Institute, BlueBrain (BBP)
  - GLIF                     — Allen Institute GLIF3/4 (через SONATA)

Быстрый старт:
    from taas_import import load_neuroml, load_sonata

    # NeuroML → TAAS
    сети = load_neuroml("network.nml")
    csr, params, info = сети.to_taas(neuron_model="lif")
    print(f"{info['n_neurons']} нейронов, {info['n_synapses']} синапсов")

    # SONATA (Allen GLIF) → TAAS
    from taas_import import load_sonata
    сети = load_sonata("nodes.h5", "edges.h5")
    csr, params, info = сети.to_taas(neuron_model="glif")
"""

from .neuroml import NeuroMLNetwork, load_neuroml
from .sonata import SonataNetwork, load_sonata

__all__ = ["NeuroMLNetwork", "load_neuroml", "load_sonata"]
