"""
t3_spike — T3-SPIKE: Tri-Torus Spiking Processing and Inference via Kuramoto Encoding.

Surrogate gradient training for SNN via differentiable T3 oscillator dynamics.

Architecture:
    spikes (binary) → W_snn → drives → T3 (N steps) → PST features → decoder → loss
                                   ↑                         ↓
                              differentiable            gradients flow back
"""
from .encoder import T3SpikeEncoder, pst_jacobian, pst_linear_features_np
from .train import T3SpikeTrainer

__all__ = ["T3SpikeEncoder", "T3SpikeTrainer", "pst_jacobian", "pst_linear_features_np"]
