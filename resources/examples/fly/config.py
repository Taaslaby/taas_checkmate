#!/usr/bin/env python3
"""FlyConfig — единый конфиг для симуляции мухи.

Все hardcoded параметры собраны здесь вместо размазывания по
flywire_brain.py, vnc_walk.py, diagnose_v2.py.
"""

from dataclasses import dataclass, field


@dataclass
class FlyConfig:
    """Конфигурация симуляции FlyWire + VNC + MuJoCo."""

    # LIF parameters (Shiu et al. 2024, Nature 634, 124–138)
    tau_m: float = 20.0
    v_rest: float = -52.0
    v_thresh: float = -45.0
    v_reset: float = -52.0
    dt_ms: float = 0.5

    # Synaptic parameters
    tau_syn: float = 5.0
    delay_steps: int = 4

    # DNg100 / CB0013 injection
    dng100_injection: float = 80.0
    cb0013_injection: float = 5.0

    # Poisson background drive
    poisson_rate_low: float = 100.0
    poisson_rate_high: float = 200.0
    poisson_pulse_amp: float = 30.0

    # Actuator gain
    actuator_gain: float = 3.0

    # Brain update interval
    brain_step_ms: float = 25.0

    # Background I_ext bias (Izhikevich)
    i_ext_bias: float = 10.0

    # VNC motion scales
    motion_scales: dict = field(default_factory=lambda: {
        'flex': 0.5, 'extend': -0.5, 'abduct': 0.3, 'twist': 0.3,
    })

    # Balance PD gains
    balance_kp_roll: float = 3.0
    balance_kd_roll: float = 0.3
    balance_kp_pitch: float = 2.0
    balance_kd_pitch: float = 0.2

    # Named DN injection targets
    injection_targets: dict = field(default_factory=lambda: {'DNg100': 80.0})

    # Turn bias
    turn_bias: float = 0.0

    # Looming sensor
    looming_enabled: bool = True
    looming_x_min: float = -0.4
    looming_x_max: float = 0.4
    looming_y_min: float = -0.3
    looming_y_max: float = 0.3
    looming_scale: float = 0.12


# Preset configs
LIF_DEFAULT = FlyConfig()

IZHI_DEFAULT = FlyConfig(
    tau_m=10.0,
    v_rest=-65.0,
    v_thresh=-50.0,
    v_reset=-65.0,
    dng100_injection=50.0,
    poisson_rate_low=150.0,
    poisson_rate_high=300.0,
    poisson_pulse_amp=20.0,
    actuator_gain=2.0,
)

FAST_TEST = FlyConfig(
    actuator_gain=5.0,
    dng100_injection=100.0,
    poisson_rate_low=200.0,
    poisson_rate_high=400.0,
)
