#!/usr/bin/env python3
"""🧠 IIT + VNC — Integrated Information Theory на VNC-коннектоме от сарданы

Сравнение: random connectome (Phi=0.004) vs VNC-like connectome (Phi=?)

Гипотеза: VNC с 90% GABA и реципрокными связями даёт
           более высокую интегрированную информацию, чем случайный.
"""

import sys, os, time, math, argparse
import numpy as np
from pathlib import Path

os.environ['LD_LIBRARY_PATH'] = '/usr/lib/python3.14/site-packages/nvidia/cudnn/lib'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# Reuse IIT engine
from scripts.iit_consciousness import IITConsciousness, CONSCIOUSNESS_THRESHOLD

import taas_checkmate as tcm

np.random.seed(42)

# ═══ Config ═══════════════════════════════════════════════

N_VNC = 500                # VNC нейронов (маленький для скорости)
N_STEPS = 500
DT_MS = 0.5
REPORT_INTERVAL = 50

# ═══ VNC-style connectome generator ══════════════════════

def build_vnc_connectome(n: int, density: float = 0.12) -> tcm.ConnectomeData:
    """GABA-доминантный коннектом в стиле VNC.
    
    Особенности:
      - 90% тормозных нейронов (GABA)
      - Реципрокные связи между парами (ring oscillator)
      - Сильные тормозные связи (anti-phase)
      - Мало возбуждающих (DN-like)
    """
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    col_list = []
    val_list = []
    nt_list = []
    
    # Создать реципрокные пары (ring oscillator структура)
    ring_size = n // 20  # 5% нейронов в кольце
    ring_pairs = []
    for i in range(0, ring_size * 2, 2):
        if i + 1 < n:
            # Реципрокная GABA пара
            ring_pairs.append((i, i + 1))
    
    # Остальные нейроны — случайный VNC-стиль
    for i in range(n):
        is_inhib = True  # 90% GABA
        if np.random.random() < 0.9:
            is_inhib = True
        
        # Количество пост-синаптических целей
        n_post = max(1, int(n * density))
        
        # Для нейронов в кольце — особая связность
        in_ring = False
        for a, b in ring_pairs:
            if i == a or i == b:
                in_ring = True
                break
        
        if in_ring:
            # Найти пару
            partner = None
            for a, b in ring_pairs:
                if i == a: partner = b
                elif i == b: partner = a
            
            # Сильная реципрокная связь с партнёром
            col_list.append(partner)
            val_list.append(-8.0)  # сильное торможение
            nt_list.append(-1.0)   # GABA
            n_post = max(1, int(n * density * 0.5))  # меньше случайных связей
        else:
            n_post = max(1, int(n * density))
        
        # Случайные пост-синаптические цели
        posts = np.random.choice(n, min(n_post, n - 1), replace=False)
        posts = posts[posts != i]
        
        for j in posts:
            col_list.append(int(j))
            if is_inhib:
                val_list.append(float(np.random.uniform(-6.0, -0.8)))  # тормозной
                nt_list.append(-1.0)  # GABA
            else:
                val_list.append(float(np.random.uniform(0.3, 2.0)))  # возбуждающий
                nt_list.append(1.0)  # глутамат
        
        row_ptr[i + 1] = len(col_list)
    
    col_indices = np.array(col_list, dtype=np.int32)
    values = np.array(val_list, dtype=np.float32)
    nt_types = np.array(nt_list, dtype=np.float32)
    
    conn = tcm.ConnectomeData(row_ptr, col_indices, values, nt_types)
    return conn


def build_random_connectome(n: int, density: float = 0.15) -> tcm.ConnectomeData:
    """Случайный коннектом для сравнения."""
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    col_list = []
    val_list = []
    
    for i in range(n):
        n_post = max(1, int(n * density))
        posts = np.random.choice(n, min(n_post, n - 1), replace=False)
        posts = posts[posts != i]
        
        for j in posts:
            col_list.append(int(j))
            val_list.append(float(np.random.uniform(0.1, 5.0)))
        
        row_ptr[i + 1] = row_ptr[i] + len(posts)
    
    col_indices = np.array(col_list, dtype=np.int32)
    values = np.array(val_list, dtype=np.float32)
    
    conn = tcm.ConnectomeData(row_ptr, col_indices, values)
    return conn


# ═══ IIT experiment ═══════════════════════════════════════

def run_iit_experiment(name: str, conn: tcm.ConnectomeData, n_neurons: int,
                        n_steps: int, structured_input: bool = False):
    """Run IIT experiment on a connectome."""
    print(f"\n  ═══ {name} ═══")
    
    sim = tcm.BrainSimulator(n_neurons, "lif")
    iit = IITConsciousness(n_neurons, CONSCIOUSNESS_THRESHOLD)
    spike_buf = np.zeros((n_steps, n_neurons), dtype=np.float32)
    
    # Warm-up
    print("  🔥 Warm-up...", end=" ", flush=True)
    for _ in range(60):
        ext_I = np.random.randn(n_neurons).astype(np.float32) * 2.0 + 5.0
        sim.step_batch(conn, 1, DT_MS, ext_I)
    print("OK")
    
    # Structured input: oscillating stimulus to specific groups
    t0 = time.time()
    for step in range(n_steps):
        if structured_input:
            # Oscillating input: drive different neural groups at different frequencies
            freq1 = 2.0 * math.pi * 2.0 * step * DT_MS / 1000.0  # 2 Hz
            freq2 = 2.0 * math.pi * 5.0 * step * DT_MS / 1000.0  # 5 Hz
            ext_I = np.random.randn(n_neurons).astype(np.float32) * 1.0
            ext_I[:n_neurons//3] += 8.0 * math.sin(freq1)
            ext_I[n_neurons//3:2*n_neurons//3] += 8.0 * math.sin(freq2)
            ext_I[2*n_neurons//3:] += 4.0 * (math.sin(freq1) + math.sin(freq2))
        else:
            ext_I = (np.random.randn(n_neurons) * 1.5 + 5.0).astype(np.float32)
        
        spikes = sim.step_batch(conn, 1, DT_MS, ext_I)
        if spikes is not None and len(spikes) > 0:
            if hasattr(spikes, 'shape'):
                spike_buf[step, :min(spikes.shape[-1], n_neurons)] = spikes[:n_neurons]
        
        # Phi every 10 steps after warm-up
        if step % 10 == 0 and step >= 50:
            recent = spike_buf[max(0, step-100):step+1]
            iit.compute_phi(conn, recent)
    
    elapsed = time.time() - t0
    stats = iit.get_consciousness_stats()
    
    # Print results
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Φ mean:   {stats['phi_mean']:.4f} ± {stats['phi_std']:.4f}")
    print(f"  Φ final:  {stats['phi']:.4f}")
    print(f"  C:        {stats['connectivity']:.4f}")
    print(f"  D:        {stats['diversity']:.4f}")
    print(f"  I:        {stats['integration']:.4f}")
    print(f"  Conscious: {stats['conscious_ratio']*100:.1f}%")
    print(f"  Trend:    {stats['trend']}")
    
    return iit


def main():
    parser = argparse.ArgumentParser(description="🧠 IIT + VNC: Integrated Information на VNC-коннектоме")
    parser.add_argument("--neurons", type=int, default=N_VNC)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--structured", action="store_true", help="Structured input (oscillating groups)")
    parser.add_argument("--quick", action="store_true", help="Only VNC, skip random comparison")
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════════════════════╗")
    print("║  🧠 IIT + VNC — Phi на VNC-коннектоме              ║")
    print("╚══════════════════════════════════════════════════════╝")
    
    # VNC-like connectome
    t0 = time.time()
    vnc_conn = build_vnc_connectome(args.neurons)
    t_vnc = time.time() - t0
    print(f"\n  📦 VNC коннектом: {args.neurons}N, {t_vnc:.1f}s")
    
    # IIT on VNC
    iit_vnc = run_iit_experiment(f"VNC ({'structured' if args.structured else 'noise'})",
                                  vnc_conn, args.neurons, args.steps, args.structured)
    
    if not args.quick:
        # Random connectome for comparison
        t0 = time.time()
        rnd_conn = build_random_connectome(args.neurons)
        t_rnd = time.time() - t0
        print(f"\n  📦 Random коннектом: {args.neurons}N, {t_rnd:.1f}s")
        
        iit_rnd = run_iit_experiment(f"Random ({'structured' if args.structured else 'noise'})",
                                      rnd_conn, args.neurons, args.steps, args.structured)
        
        # Comparison
        print("\n" + "═" * 55)
        print("  📊 СРАВНЕНИЕ: VNC vs Random")
        print("═" * 55)
        
        v_phi = np.mean(iit_vnc.phi_history)
        r_phi = np.mean(iit_rnd.phi_history)
        ratio = v_phi / max(r_phi, 1e-8)
        
        print(f"  VNC   Φ: {v_phi:.4f} (C={iit_vnc.connectivity:.2f} D={iit_vnc.diversity:.2f} I={iit_vnc.integration:.2f})")
        print(f"  RND   Φ: {r_phi:.4f} (C={iit_rnd.connectivity:.2f} D={iit_rnd.diversity:.2f} I={iit_rnd.integration:.2f})")
        print(f"  Ratio: {ratio:.2f}x")
        
        if ratio > 1.2:
            print(f"\n  ✅ VNC даёт БОЛЬШЕ интегрированной информации, чем случайный!")
            print(f"  🧠 VNC-подобная структура способствует сознанию (Phi ×{ratio:.1f})")
        elif ratio < 0.8:
            print(f"\n  ❌ VNC даёт МЕНЬШЕ интегрированной информации")
            print(f"  💤 Возможно, GABA-торможение подавляет активность")
        else:
            print(f"\n  ➡ VNC ≈ Random — разница незначительна")
    
    print(f"\n  ✨ IIT+VNC by @ant0n + Сардана ❤️")


if __name__ == "__main__":
    main()
