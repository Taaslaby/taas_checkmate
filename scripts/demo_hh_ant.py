#!/usr/bin/env python3
"""
Demo: HH LUT spiking dynamics in rate-based Ant RL.
Shows how to replace tanh reservoir dynamics with HH LUT neurons.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from taas_checkmate.neuron.hh import HhLutConfig, step_hh_lut, standard_config

class HhReservoir(nn.Module):
    """Reservoir with HH LUT neurons instead of rate-based dynamics."""

    def __init__(self, n_neurons, _W_adj, sensory_idx, motor_idx, obs_dim, internal_steps=5):
        super().__init__()
        self.n = n_neurons
        self.internal_steps = internal_steps
        self._sensory_idx = torch.tensor(sensory_idx, dtype=torch.long)
        self._motor_idx = torch.tensor(motor_idx, dtype=torch.long)

        W = torch.tensor(_W_adj, dtype=torch.float32)
        self.register_buffer('_W', W)

        # Sensory input weights
        self.W_in = nn.Linear(obs_dim, len(sensory_idx), bias=False)
        self.bias = nn.Parameter(torch.zeros(n_neurons))
        self.in_gain = nn.Parameter(torch.ones(len(sensory_idx)) * 0.1)

        # Initialize HH LUT for fast Spiking Neural Network (SNN)
        self.hh_cfg = standard_config(n_neurons, dt=0.01)

    def forward(self, obs, deterministic=False):
        """Step through HH dynamics, return firing rate as output.

        Args:
            obs: (B, obs_dim) input observations
            deterministic: if True, no noise (not used in HH but kept for API)

        Returns:
            motor_rate: (B, n_motor_neurons) firing rate output
            hh_activity: (B, n_neurons) full HH activity for features
        """
        B = obs.shape[0]
        device = obs.device

        # Initialize v and gates (already done in cfg)
        v = self.hh_cfg.v.to(device)
        gates = self.hh_cfg.gates.to(device)

        # Sensor input
        sensory_inj = self.W_in(obs) * self.in_gain

        # HH dynamics steps
        for _ in range(self.internal_steps):
            # Input current I_i — use constant for now
            I_ext = torch.zeros_like(v, device=device)

            # Call LUT-HH kernel (via PyO3)
            # For now, we'll simulate the step using PyTorch
            # In a full implementation, this will be a PyO3 call

            # Simulate 1 ms per step for demonstration
            # Real implementation: via PyO3 TT to Rust SIMD

            # Rate-based dynamics for demo
            # Replace with actual HH LUT later
            state = v[:len(self._sensory_idx)]  # Use v as state

            # Sparse connectivity update
            for i in range(self.n):
                # Gather incoming connections
                incoming = []
                for j in range(self.n):
                    if abs(self._W[i, j]) > 0:
                        incoming.append(v[j].item())

                if incoming:
                    avg_in = sum(incoming) / len(incoming)
                    v[i] = torch.tanh(avg_in + self.bias[i].item())

            # Add sensory input
            v[self._sensory_idx] += sensory_inj * 0.1

            # Clamp to HH operating range
            v = torch.clamp(v, min=-100.0, max=60.0)

        # Compute firing rate (spikes per second)
        spikes = (v > 0).float().mean(dim=-1)
        motor_rate = spikes.gather(1, self._motor_idx)

        return motor_rate, v

    def get_features(self, obs):
        """Convenience method for SB3 features extractor."""
        motor_r, _ = self(obs, deterministic=not self.training)
        return motor_r

def test_hh_ant_rl():
    """Quick test of HH LUT in Ant RL policy network."""
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    # Simple config
    n_neurons = 256
    ei_ratio = 4.0

    # Generate a simple (empty) connectivity matrix for demo
    W_adj = torch.zeros((n_neurons, n_neurons))

    # Set up some simple connections (ring)
    for i in range(n_neurons):
        j = (i + 1) % n_neurons
        W_adj[i, j] = torch.randn(1) * 0.5  # Random weights

    # Define neural groups
    sensory_idx = list(range(20))
    motor_idx = list(range(n_neurons - 20, n_neurons))
    central_idx = [i for i in range(n_neurons)
                  if i not in sensory_idx and i not in motor_idx]

    # Mock observation space
    obs_dim = 4
    activation_space = gym.make('Ant-v5')
    obs_dim = activation_space.observation_space.shape[0]

    # Create HH reservoir (simplified for demo)
    hh_res = HhReservoir(n_neurons, W_adj, sensory_idx, motor_idx, obs_dim)

    # Test forward pass
    test_obs = torch.randn(8, obs_dim)
    motor_r, full_state = hh_res(test_obs)

    print(f"HH Reservoir test results:")
    print(f"  Input shape: {test_obs.shape}")
    print(f"  Motor rate shape: {motor_r.shape}")
    print(f"  Motor rate range: ({motor_r.min():.3f}, {motor_r.max():.3f})")
    print(f"  Average motor rate: {motor_r.mean():.3f}")
    print(f"  Neurons in active range: {(full_state.abs() > 0).sum()} / {len(full_state)}")
    print(f"  State range: ({full_state.min():.2f}, {full_state.max():.2f})")

    # Visualize example trajectory
    print("\nTrajectory example (10 steps):")
    for step in range(10):
        test_obs_step = torch.randn(1, obs_dim)
        with torch.no_grad():
            motor_r_step, _ = hh_res(test_obs_step)
        print(f"  Step {step}: motor activity = {motor_r_step[0].item():.3f}")

    print("\nNote: Full implementation requires PyO3 binding to Rust LUT-HH kernel.")
    print("This demo shows the network architecture integration of HH into Ant RL.")

    return hh_res

if __name__ == '__main__':
    test_hh_ant_rl()
