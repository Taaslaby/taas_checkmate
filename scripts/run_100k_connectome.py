#!/usr/bin/env python3
"""
🚀 100K CONNECTOME GENERATION + HH SIM — запуск!

Импортирует проверенную edge_predict_to_csr из test_100k_connectome.py,
которая уже работает с GPU batch processing.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

os.environ['LD_LIBRARY_PATH'] = '/usr/lib/python3.14/site-packages/nvidia/cudnn/lib'

BASE = Path(__file__).resolve().parent.parent
SARDAANA = Path("/home/taaslab/Проекты/asm_engines/sardaanaCM")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(SARDAANA))

import torch
import taas_checkmate as tcm
from src.connectome_builder.diffusion import ConnectomeDiffusion
from src.connectome_builder import CookieCutterGenerator, CookieCutterConfig, \
    TaasWriter, ConnectomeSample, save_connectome
from experiments.test_100k_connectome import edge_predict_to_csr

# ═══════════════════════════════════════════════════════════
N = 100000
N_STEPS = 50
DT_MS = 0.5
DDIM_STEPS = 50

MODEL_PATH = SARDAANA / "models" / "diffusion" / "connectome_diffusion_scaled.pt"
COOKIE_DIR = SARDAANA / "models" / "cookiecutter"
OUTPUT_DIR = BASE / "examples" / "data"


def build_hh_params(n, nt, rng):
    """Per-neuron 3-channel HH + CaV pacemaking."""
    g_max = np.zeros(3 * n, dtype=np.float32)
    C_m = np.ones(n, dtype=np.float32)
    g_leak = np.ones(n, dtype=np.float32) * 0.3
    E_leak = np.ones(n, dtype=np.float32) * -65.0

    for i in range(n):
        if nt[i] > 0:
            g_max[0*n+i] = 120.0; g_max[1*n+i] = 36.0; C_m[i] = 1.0
        else:
            g_max[0*n+i] = 120.0; g_max[1*n+i] = 50.0; C_m[i] = 0.6; g_leak[i] = 0.5

    pc = max(1, int(n * 0.05))
    g_max[2*n + rng.choice(n, size=pc, replace=False)] = 0.5
    return g_max, C_m, g_leak, E_leak


def main():
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  🚀 100K CONNECTOME GENERATION + HH SIM             ║")
    print(f"║  N={N:,}  steps={N_STEPS}  dt={DT_MS}ms               ║")
    print("╚═══════════════════════════════════════════════════════╝\n")

    t_start = time.time()
    rng = np.random.RandomState(42)

    # 1. Load models
    print("[1/5] Loading models ...")
    model = ConnectomeDiffusion.load(str(MODEL_PATH))
    gen = CookieCutterGenerator.load_trained(
        str(COOKIE_DIR), CookieCutterConfig(n_neurons=N, use_gpu=True))
    print(f"  🧬 {model.config.denoiser_type}  🍪 cookiecutter loaded\n")

    # 2. Generate states
    print(f"[2/5] Generating {N:,} states ...")
    t0 = time.time()
    states = model.sample(n_neurons=N, batch_size=1, use_ddim=True, ddim_steps=DDIM_STEPS)
    print(f"  🎲 {states.shape} in {time.time()-t0:.2f}s\n")

    # 3. Edge prediction → CSR
    print(f"[3/5] Edge prediction ...")
    t0 = time.time()
    # Generate degree distribution + nt_types for the function
    n_states = states[0]
    n = len(n_states)
    target_degs = np.exp(rng.randn(n) * 1.2 + 4.0).astype(np.int32)
    target_degs = np.clip(target_degs, 1, n // 8)
    nt_pred = np.where(rng.rand(n) < 0.2, -1.0, 1.0).astype(np.float32)
    
    row_ptr, col_indices, values = edge_predict_to_csr(
        gen.edge_predictor, n_states, target_degs, nt_pred,
        threshold=0.3, block_size=256, n_neighbor_blocks=2, rng=rng
    )
    nt = nt_pred
    t_edge = time.time() - t0
    n_syn = len(col_indices)
    print(f"  🔗 {n_syn:,} synapses, mean deg={n_syn/N:.1f}, {t_edge:.1f}s\n")

    # 4. HH simulation
    print(f"[4/5] HH+CaV sim ({N_STEPS} steps) ...")
    conn = tcm.ConnectomeData(
        row_ptr.astype(np.int32), col_indices.astype(np.int32),
        values.astype(np.float32), nt.astype(np.float32))
    
    g_max, C_m, g_leak, E_leak = build_hh_params(N, nt, rng)
    sim = tcm.BrainSimulator(N, "hh", pacemaking=True)
    sim.set_hh_params(g_max.astype(np.float32),
        np.array([50.0, -77.0, 120.0], dtype=np.float32),
        g_leak.astype(np.float32), E_leak.astype(np.float32), dt_ms=DT_MS)
    sim.set_C_m(C_m.astype(np.float32))

    t0 = time.time()
    spike_buf = sim.step_batch(conn, N_STEPS, dt_ms=DT_MS, scale=1e10)
    t_sim = time.time() - t0

    sb = np.asarray(spike_buf).reshape(N_STEPS, N).astype(np.int32)
    total = int(sb.sum())
    active = int((sb.sum(axis=0) > 0).sum())
    duration = N_STEPS * DT_MS / 1000.0
    fr = sb.sum(axis=0).astype(np.float64) / duration
    e_mask, i_mask = nt > 0, nt < 0
    fr_e = float(fr[e_mask].mean()) if e_mask.sum() > 0 else 0.0
    fr_i = float(fr[i_mask].mean()) if i_mask.sum() > 0 else 0.0
    ei_ratio = sb[:, e_mask].sum() / max(sb[:, i_mask].sum(), 1)

    del sim
    print(f"  ⚡ {total:,} spikes, {active:,}/{N:,} active ({active/N*100:.1f}%)")
    print(f"  E={fr_e:.1f}Hz  I={fr_i:.1f}Hz  E/I={float(ei_ratio):.2f}")
    print(f"  {t_sim:.2f}s ({t_sim/N_STEPS*1000:.2f}ms/step)\n")

    # 5. Save
    print("[5/5] Saving .taas ...")
    out = OUTPUT_DIR / "generated_100k.taas"
    sample = ConnectomeSample(
        row_ptr=row_ptr, col_indices=col_indices, values=values,
        nt_types=nt, groups=[], cell_types=[], meta={})
    with TaasWriter(str(out)) as wr:
        save_connectome(sample, wr)
    print(f"  💾 {out}")

    total_t = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  ✅ 100K CONNECTOME — ГОТОВ!")
    print(f"{'='*55}")
    print(f"  {'Neurons':<20s} {N:>12,}")
    print(f"  {'Synapses':<20s} {n_syn:>12,}")
    print(f"  {'Mean degree':<20s} {n_syn/N:>11.1f}")
    print(f"  {'Diffusion':<20s} {'see above':>12s}")
    print(f"  {'Edge pred':<20s} {t_edge:>11.1f}s")
    print(f"  {'HH sim':<20s} {t_sim:>11.2f}s")
    print(f"  {'Total':<20s} {total_t:>11.1f}s")
    print(f"  {'Spikes':<20s} {total:>12,}")
    print(f"  {'Active':<20s} {active/N*100:>11.1f}%")
    print(f"  {'E rate':<20s} {fr_e:>11.1f}Hz")
    print(f"  {'I rate':<20s} {fr_i:>11.1f}Hz")
    print(f"  {'E/I':<20s} {float(ei_ratio):>11.2f}")
    print(f"{'='*55}")
    print(f"  🏆 100K connectome — сгенерирован, симулирован, сохранён!")


if __name__ == "__main__":
    main()
