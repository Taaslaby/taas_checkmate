#!/usr/bin/env python3
"""
GLIF → TAAS Neuron Model Mapping
==================================

Maps Allen Institute GLIF3/4 models to TAAS Checkmate neuron models.

GLIF3 (Generalized LIF, level 3):
  - dv/dt = -g_leak*(V - E_leak) + I_syn + g_ahp*(V - E_ahp)
  - dg_ahp/dt = -g_ahp / tau_ahp
  - On spike: V = V_reset, g_ahp += delta_g_ahp

GLIF4 (Generalized LIF, level 4):
  - Same as GLIF3 + threshold adaptation:
  - dθ/dt = -θ / tau_theta
  - V_eff_thresh = V_thresh_0 + θ
  - On spike: θ += delta_theta

TAAS mapping:
  GLIF3 → TAAS LIF (step_lif) + manual g_ahp management
  GLIF3 → TAAS HAAP (g_M ≈ g_ahp, E_K ≈ E_ahp)
  GLIF4 → TAAS HAAP + manual theta threshold adaptation

Usage:
    from taas_import.glif_map import glif_to_taas_params
    params, info = glif_to_taas_params(glif_data, target="haap")
"""

from typing import Dict, Optional
import numpy as np


def glif_to_taas_params(
    g_leak: float = 0.1,
    e_leak: float = -65.0,
    v_thresh: float = -54.0,
    v_reset: float = -65.0,
    cm: float = 1.0,
    tau_ahp: float = 30.0,
    delta_g_ahp: float = 0.01,
    e_ahp: float = -80.0,
    tau_theta: Optional[float] = None,
    delta_theta: Optional[float] = None,
    target: str = "haap",
    dt: float = 0.05,
    **kwargs,
) -> Dict[str, float]:
    """
    Convert GLIF3/4 parameters to TAAS internal format.

    Args:
        target: "haap" (uses M-current for after-spike)
                "lif"  (basic LIF, no after-spike)
                "glif" (direct GLIF implementation)
    """
    if target == "lif":
        # Simple LIF: tau_m = Cm / g_leak
        tau_m = cm / max(g_leak, 1e-6) / 1000.0  # g_leak in S/cm²
        return {
            'dt': dt, 'cm': cm,
            'gL': g_leak * 1000.0,
            'EL': e_leak,
            'Vthresh': v_thresh,
            'Vreset': v_reset,
            'tau_m': tau_m,
            'gM': 0.0, 'gh': 0.0,
        }

    elif target == "haap":
        # HAAP M-current mimics GLIF after-spike current:
        # I_M = g_M * u * (V - E_K) ≈ g_ahp * (V - E_ahp)
        # where u ≈ 1 after spike (delta_u = delta_g_ahp / cm)
        return {
            'dt': dt, 'cm': cm,
            'gL': g_leak * 1000.0,
            'EL': e_leak,
            'Vthresh': v_thresh,
            'Vreset': v_reset,
            'gM': delta_g_ahp * 1000.0,  # after-spike K+ current
            'EK': e_ahp,
            'tau_u': tau_ahp,
            'delta_u': delta_g_ahp / max(cm, 0.1) * 10.0,
            'gh': 0.0,
            'tau_h': 200.0,
        }

    elif target == "glif":
        # Direct GLIF parameters for TAAS GLIF implementation
        tau_m = cm / max(g_leak, 1e-6) / 1000.0
        result = {
            'dt': dt, 'cm': cm,
            'gL': g_leak * 1000.0,
            'EL': e_leak,
            'Vthresh': v_thresh,
            'Vreset': v_reset,
            'tau_m': tau_m,
            'tau_ahp': tau_ahp,
            'delta_g_ahp': delta_g_ahp / cm,
            'E_ahp': e_ahp,
            'gM': 0.0, 'gh': 0.0,
        }
        if tau_theta is not None:
            result['tau_theta'] = tau_theta
            result['delta_theta'] = delta_theta or 1.0
        return result

    else:
        raise ValueError(f"Unknown target: {target}")


# GLIF template library (Allen Institute, Teeter et al. 2018)
GLIF_TEMPLATES = {
    'GLIF3_1': {
        'g_leak': 0.1, 'e_leak': -65.0, 'v_thresh': -54.0,
        'v_reset': -65.0, 'cm': 1.0,
        'tau_ahp': 30.0, 'delta_g_ahp': 0.01, 'e_ahp': -80.0,
        'target': 'haap',
    },
    'GLIF3_2': {
        'g_leak': 0.15, 'e_leak': -70.0, 'v_thresh': -50.0,
        'v_reset': -65.0, 'cm': 0.8,
        'tau_ahp': 50.0, 'delta_g_ahp': 0.02, 'e_ahp': -85.0,
        'target': 'haap',
    },
    'GLIF4_1': {
        'g_leak': 0.1, 'e_leak': -65.0, 'v_thresh': -54.0,
        'v_reset': -65.0, 'cm': 1.0,
        'tau_ahp': 30.0, 'delta_g_ahp': 0.01, 'e_ahp': -80.0,
        'tau_theta': 20.0, 'delta_theta': 3.0,
        'target': 'haap',
    },
    'GLIF4_2': {
        'g_leak': 0.2, 'e_leak': -70.0, 'v_thresh': -50.0,
        'v_reset': -68.0, 'cm': 0.5,
        'tau_ahp': 100.0, 'delta_g_ahp': 0.05, 'e_ahp': -90.0,
        'tau_theta': 50.0, 'delta_theta': 5.0,
        'target': 'haap',
    },
}


def from_template(
    template_name: str = 'GLIF3_1',
    target: Optional[str] = None,
    dt: float = 0.05,
) -> Dict[str, float]:
    """Load GLIF from template and convert to TAAS params."""
    if template_name not in GLIF_TEMPLATES:
        raise KeyError(f"Unknown GLIF template: {template_name}. "
                       f"Available: {list(GLIF_TEMPLATES.keys())}")

    tpl = GLIF_TEMPLATES[template_name].copy()
    t_target = tpl.pop('target', 'haap')
    return glif_to_taas_params(target=target or t_target, dt=dt, **tpl)


# Quick self-test
if __name__ == '__main__':
    for name in GLIF_TEMPLATES:
        p = from_template(name)
        print(f"{name}: gL={p['gL']:.3f}, Vthresh={p['Vthresh']:.1f}, gM={p['gM']:.4f}")
    print("All GLIF templates mapped successfully!")
