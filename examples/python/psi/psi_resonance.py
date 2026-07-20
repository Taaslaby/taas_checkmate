#!/usr/bin/env python3
"""
ψ-Wave Lake v2 — «Мозг как резонансное озеро»

v1 показал: |ψ|² сохраняется (унитарность). Правильная метрика —
**coherence фаз** readout нейронов и **FFT** выходного сигнала.

Гипотеза:
    Коннектом = волновод. Структура графа определяет, какие
    частоты синхронизируют readout. Простой sin(f·t) на входе →
    выходной паттерн определён собственными модами графа.

Что нового:
    - Small-world топология (Watts-Strogatz) — реалистичнее кольца
    - Метрика: coherence + FFT readout сигнала
    - Измерение: на какой частоте coherence максимален → eigenmode
"""

import sys
import os
import math
import time
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in [os.path.join(PROJ, 'target', 'release'),
           os.path.join(PROJ, 'target', 'debug')]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import taas_checkmate as tcm


# ═══════════════════════════════════════════════════════════
#  1. Small-world connectome (Watts-Strogatz)
# ═══════════════════════════════════════════════════════════

def build_smallworld(n: int, k: int = 4, p_rewire: float = 0.15,
                     sensory_ratio: float = 0.1,
                     readout_ratio: float = 0.05):
    """Watts-Strogatz small-world: ring lattice + random rewiring.

    Args:
        n: number of neurons
        k: nearest neighbours per direction (so 2k connections/neuron)
        p_rewire: probability of rewiring each edge
    """
    rng = np.random.RandomState(42)

    # Each neuron connects to k nearest neighbours in each direction
    edges = []
    for i in range(n):
        for j in range(1, k + 1):
            # forward
            target_f = (i + j) % n
            if rng.random() < p_rewire and n > 2:
                # rewire: pick random target != i and != target_f
                candidates = [t for t in range(n) if t != i and t != target_f]
                target_f = int(rng.choice(candidates))
            edges.append((i, target_f, 1.0))

            # backward
            target_b = (i - j) % n
            if rng.random() < p_rewire and n > 2:
                candidates = [t for t in range(n) if t != i and t != target_b]
                target_b = int(rng.choice(candidates))
            edges.append((i, target_b, 1.0))

    nnz = len(edges)
    col_indices = np.zeros(nnz, dtype=np.int32)
    values = np.zeros(nnz, dtype=np.float32)

    # Sort edges by source for CSR
    edges.sort(key=lambda x: (x[0], x[1]))
    for i, (_, dst, w) in enumerate(edges):
        col_indices[i] = dst
        values[i] = w

    # Build row_ptr — first index of each source's outgoing edges
    row_ptr = np.zeros(n + 1, dtype=np.int32)
    prev_src = -1
    for i, (src, _, _) in enumerate(edges):
        while prev_src + 1 < src:
            prev_src += 1
            row_ptr[prev_src] = i
        if src != prev_src:
            row_ptr[src] = i
            prev_src = src
    while prev_src < n - 1:
        prev_src += 1
        row_ptr[prev_src] = nnz
    row_ptr[n] = nnz

    n_sensory = max(1, int(n * sensory_ratio))
    n_readout = max(1, int(n * readout_ratio))
    sensory_idx = list(range(n_sensory))
    readout_idx = list(range(n - n_readout, n))

    return row_ptr, col_indices, values, sensory_idx, readout_idx


def build_flywire_subgraph(n: int):
    """Load a subgraph of the FlyWire connectome (if available)."""
    # Try loading the Princeton dataset
    import glob
    files = glob.glob(os.path.expanduser('~/Загрузки/connections_princeton*.csv.gz'))
    if not files:
        raise FileNotFoundError("No FlyWire data found — use small-world instead")
    print(f"  Loading {files[0]}...")
    data = np.loadtxt(files[0], delimiter=',', skiprows=1)
    # Build subgraph of first n neurons
    raise NotImplementedError("FlyWire subgraph loading TBD")


# ═══════════════════════════════════════════════════════════
#  2. ψ-wave with coherence + FFT readout
# ═══════════════════════════════════════════════════════════

def run_psi_wave_v2(row_ptr, col_indices, values,
                    sensory_idx, readout_idx,
                    freq_hz: float, n_steps: int = 2000,
                    dt: float = 1.0, hbar: float = 1.0,
                    input_amp: float = 2.0):
    """Run ψ-wave and measure readout coherence + spectrum.

    Returns:
        readout_coherence: phase coherence of readout neurons (last 500 steps)
        spectral_peak: dominant frequency in readout signal
        readout_mean_energy: mean |ψ|² at readout
    """
    n = len(row_ptr) - 1
    n_padded = (n + 7) // 8 * 8

    # Init ψ with random phases
    rng = np.random.RandomState(42)
    phases = rng.uniform(0, 2 * math.pi, n_padded).astype(np.float32)
    v = np.cos(phases).astype(np.float32)
    u = np.sin(phases).astype(np.float32)

    h_real = np.zeros(n_padded, dtype=np.float32)
    h_imag = np.zeros(n_padded, dtype=np.float32)
    i_ext = np.zeros(n_padded, dtype=np.float32)
    speed = np.ones(n_padded, dtype=np.float32)

    omega = 2.0 * math.pi * freq_hz

    # Record readout signal for FFT
    readout_signal = np.zeros(n_steps, dtype=np.float32)

    # Window size for coherence (last 500 steps = 500ms)
    n_coh = min(500, n_steps)

    for step in range(n_steps):
        t = step * dt / 1000.0

        drive = input_amp * math.sin(omega * t)
        i_ext.fill(0.0)
        for idx in sensory_idx:
            i_ext[idx] = drive

        tcm.sparse.compute_hamiltonian(
            row_ptr, col_indices, values, v, u, h_real, h_imag,
        )
        tcm.step_psi_wave(v, u, h_real, h_imag, i_ext, speed, dt, hbar)

        # Mean ψ at readout (complex sum)
        psi_mean_real = 0.0
        psi_mean_imag = 0.0
        psi_energy = 0.0
        for idx in readout_idx:
            psi_mean_real += v[idx]
            psi_mean_imag += u[idx]
            psi_energy += v[idx]**2 + u[idx]**2
        psi_mean_real /= len(readout_idx)
        psi_mean_imag /= len(readout_idx)
        readout_signal[step] = psi_mean_real

    # 1. Coherence = |mean(exp(i·φ))| over last n_coh steps
    readout_phases = []
    for step in range(n_steps - n_coh, n_steps):
        # Compute mean phase at this step
        re = 0.0
        im = 0.0
        for idx in readout_idx:
            re += v[idx]
            im += u[idx]
        re /= len(readout_idx)
        im /= len(readout_idx)
        readout_phases.append(math.atan2(im, re))

    mean_complex = np.mean(np.exp(1j * np.array(readout_phases)))
    coherence = float(np.abs(mean_complex))

    # 2. FFT of readout signal (last 1024 steps = 1 second)
    signal = readout_signal[-min(1024, n_steps):]
    if len(signal) > 10:
        fft = np.fft.rfft(signal - np.mean(signal))
        freqs_fft = np.fft.rfftfreq(len(signal), d=dt / 1000.0)
        # Global peak (natural resonance of the graph)
        peak_idx = np.argmax(np.abs(fft[1:])) + 1
        spectral_peak = float(freqs_fft[peak_idx]) if peak_idx < len(freqs_fft) else 0.0
        spectral_power = float(np.abs(fft[peak_idx]))
        # Power AT the driving frequency
        drive_bin = np.argmin(np.abs(freqs_fft - freq_hz))
        drive_power = float(np.abs(fft[drive_bin])) if drive_bin < len(fft) else 0.0
        # Drive-to-peak ratio: how much of the output is at the drive freq vs natural
        drive_ratio = drive_power / max(spectral_power, 1e-12)
    else:
        spectral_peak = 0.0
        spectral_power = 0.0
        drive_power = 0.0
        drive_ratio = 0.0

    # 3. Mean readout energy |ψ|²
    readout_energy = 0.0
    for idx in readout_idx:
        readout_energy += v[idx]**2 + u[idx]**2
    readout_energy /= len(readout_idx)

    return {
        'coherence': coherence,
        'spectral_peak_hz': spectral_peak,
        'spectral_power': spectral_power,
        'drive_power': drive_power,
        'drive_ratio': drive_ratio,
        'readout_energy': float(readout_energy),
        'readout_signal': readout_signal,
    }


# ═══════════════════════════════════════════════════════════
#  3. Frequency sweep
# ═══════════════════════════════════════════════════════════

def sweep_v2(row_ptr, col_indices, values,
             sensory_idx, readout_idx,
             freqs_hz, n_steps=2000, dt=1.0, hbar=1.0, input_amp=2.0):
    results = []
    for i, f in enumerate(freqs_hz):
        t0 = time.perf_counter()
        r = run_psi_wave_v2(row_ptr, col_indices, values,
                            sensory_idx, readout_idx,
                            f, n_steps, dt, hbar, input_amp)
        elapsed = time.perf_counter() - t0
        results.append(r)
        if i % 5 == 0:
            print(f"  {f:5.1f}Hz | coh={r['coherence']:.3f} "
                  f"nat={r['spectral_peak_hz']:5.1f}Hz "
                  f"drive_pwr={r['drive_power']:.4f} "
                  f"D/P={r['drive_ratio']:.3f} "
                  f"E={r['readout_energy']:.3f} [{elapsed*1000:.0f}ms]")
    return results


# ═══════════════════════════════════════════════════════════
#  4. Results display
# ═══════════════════════════════════════════════════════════

def print_results_v2(results, freqs_hz):
    coherences = np.array([r['coherence'] for r in results])
    energies = np.array([r['readout_energy'] for r in results])
    spectral_peaks = np.array([r['spectral_peak_hz'] for r in results])
    drive_powers = np.array([r['drive_power'] for r in results])
    drive_ratios = np.array([r['drive_ratio'] for r in results])

    print("\n" + "═" * 72)
    print("  ψ-Wave Lake — Frequency Response")
    print("═" * 72)

    # Primary resonance metric: Drive Power (FFT power at driving frequency)
    dp_peak_idx = np.argmax(drive_powers)
    dp_peak_freq = freqs_hz[dp_peak_idx]
    print(f"\n  —— Drive Power (FFT |fft| at input frequency) ——")
    print(f"  Max drive power: {drive_powers.max():.4f} @ {dp_peak_freq:.1f} Hz")
    print(f"  Min drive power: {drive_powers.min():.4f} @ {freqs_hz[np.argmin(drive_powers)]:.1f} Hz")
    print(f"  Drive power Q:   {drive_powers.max() / max(drive_powers.mean(), 1e-12):.2f}x")

    # Find drive power peaks (local maxima)
    is_dp_peak = np.zeros(len(drive_powers), dtype=bool)
    for i in range(2, len(drive_powers) - 2):
        if (drive_powers[i] > drive_powers[i-1] and
            drive_powers[i] > drive_powers[i-2] and
            drive_powers[i] > drive_powers[i+1] and
            drive_powers[i] > drive_powers[i+2] and
            drive_powers[i] > drive_powers.mean()):
            is_dp_peak[i] = True

    n_resonances = is_dp_peak.sum()
    if n_resonances >= 1:
        print(f"\n  ✅ {n_resonances} resonant eigenmode(s) detected:")
        for i, pk in enumerate(np.where(is_dp_peak)[0]):
            print(f"     Mode {i+1}: {freqs_hz[pk]:.1f} Hz — "
                  f"drive_power={drive_powers[pk]:.4f}, "
                  f"D/P ratio={drive_ratios[pk]:.3f}")
    else:
        print(f"\n  ℹ️  No clear resonant modes detected")

    # Bar chart of drive power
    print("\n  Drive Power spectrum (power at input frequency):")
    col_width = 30
    dp_max = max(drive_powers.max(), 1e-12)
    print(f"  {'Hz':>5s} : {'DrvPwr':>7s} {'D/P':>5s} {'Bar':>{col_width+4}s}")
    print(f"  {'-'*5} : {'-'*7} {'-'*5} {'-'*(col_width+4)}")
    for i, f in enumerate(freqs_hz):
        bar_len = int(drive_powers[i] / dp_max * col_width)
        bar = '█' * bar_len + '░' * (col_width - bar_len)
        marker = ' ← max' if i == dp_peak_idx else ''
        dp_s = f"{drive_powers[i]:.4f}"
        dr_s = f"{drive_ratios[i]:.3f}"
        print(f"  {f:5.1f} : {dp_s:>7s} {dr_s:>5s}  |{bar}|{marker}")

    # Comparison: no. of resonant peaks vs natural frequency constancy
    nat_freq_std = spectral_peaks.std()
    print(f"\n  Natural frequency constancy: {nat_freq_std:.1f} Hz std")
    if nat_freq_std < 5 and n_resonances >= 1:
        print("  ✅ Natural freq stable across drives — eigenmode structure confirmed!")
    tracking = np.corrcoef(freqs_hz, spectral_peaks)[0, 1]
    print(f"\n  Input→output frequency tracking (correlation): {tracking:.3f}")
    if tracking > 0.8:
        print("  ✅ Output frequency tracks input — wave propagates cleanly!")
    elif tracking > 0.3:
        print("  📡 Partial tracking — connectome filters but doesn't distort")
    else:
        print("  🌀 No frequency tracking — connectome transforms input")

    # Summary verdict based on drive power
    print("\n" + "─" * 72)
    print("  ИТОГОВЫЙ ВЕРДИКТ ПО РЕЗОНАНСУ")
    print("─" * 72)
    q = drive_powers.max() / max(drive_powers.mean(), 1e-12)
    if n_resonances >= 2 and q > 1.3:
        print(f"\n  ✅ **МОЗГ = РЕЗОНАНСНОЕ ОЗЕРО (Q={q:.2f}x, {n_resonances} мод)")
        print("     Коннектом избирательно усиливает входные частоты,")
        print("     соответствующие собственным модам графа.")
        print("     Простой sin(f·t) → сложный выходной паттерн.")
        print("     Вычисление = физика волн на коннектоме.")
    elif n_resonances >= 1:
        print(f"\n  📡 Одна резонансная мода @ {dp_peak_freq:.1f} Hz (Q={q:.2f}x)")
        print("     Слабый, но измеримый резонанс. Нужна более")
        print("     сложная топология для множественных мод.")
    else:
        print("\n  ℹ️  Нет измеримого резонанса (Q={:.2f}x)".format(q))
        print("     Возможные причины:")
        print("     - Граф слишком мал или слишком регулярен")
        print("     - ψ-wave coupling подавляет внешний drive")
        print("     - Нужна другая модель нейрона (Izhikevich/HH)")

    print("═" * 72)


# ═══════════════════════════════════════════════════════════
#  5. Main
# ═══════════════════════════════════════════════════════════

def main():
    print("═" * 72)
    print("  ψ-Wave Lake v2 — Resonance Experiment")
    print("  «Мозг как резонансное озеро»")
    print("═" * 72)

    # ─── Config ────────────────────────────────────────
    N = 256
    K_NEIGHBOURS = 3
    P_REWIRE = 0.2  # Watts-Strogatz: 20% rewired → small-world
    N_STEPS = 2000   # 2 seconds simulation
    DT = 1.0         # ms
    HBAR = 1.0
    INPUT_AMP = 2.0

    print(f"\n  Config:")
    print(f"    Neurons:    {N}")
    print(f"    Topology:   Small-world (k={K_NEIGHBOURS}, p={P_REWIRE})")
    print(f"    Steps:      {N_STEPS} ({N_STEPS*DT/1000:.1f}s)")
    print(f"    Freqs:      2–40 Hz (19 points)")

    # ─── Build ─────────────────────────────────────────
    print("\n[1/3] Building small-world connectome...")
    row_ptr, col_indices, values, sensory_idx, readout_idx = \
        build_smallworld(N, k=K_NEIGHBOURS, p_rewire=P_REWIRE)
    print(f"  Neurons: {N}")
    print(f"  Synapses: {len(values)} ({2*K_NEIGHBOURS*N} base + rewired)")
    print(f"  Sensory:  {len(sensory_idx)} neurons")
    print(f"  Readout:  {len(readout_idx)} neurons")

    # ─── Sweep ─────────────────────────────────────────
    print("\n[2/3] Frequency sweep (coherence + FFT)...")
    freqs = np.linspace(2.0, 40.0, 19)
    t_start = time.perf_counter()
    results = sweep_v2(row_ptr, col_indices, values,
                       sensory_idx, readout_idx, freqs,
                       n_steps=N_STEPS, dt=DT, hbar=HBAR,
                       input_amp=INPUT_AMP)
    elapsed = time.perf_counter() - t_start

    # ─── Results ───────────────────────────────────────
    print(f"\n[3/3] Results ({elapsed:.1f}s total, "
          f"{elapsed/len(freqs)*1000:.0f}ms per frequency)")
    print_results_v2(results, freqs)

    # ─── Save ──────────────────────────────────────────
    data = {
        'freqs_hz': freqs,
        'coherences': np.array([r['coherence'] for r in results]),
        'energies': np.array([r['readout_energy'] for r in results]),
        'spectral_peaks': np.array([r['spectral_peak_hz'] for r in results]),
        'spectral_powers': np.array([r['spectral_power'] for r in results]),
        'drive_powers': np.array([r['drive_power'] for r in results]),
        'drive_ratios': np.array([r['drive_ratio'] for r in results]),
        'readout_signals': np.array([r['readout_signal'] for r in results]),
        'config': {
            'n': N,
            'k': K_NEIGHBOURS,
            'p_rewire': P_REWIRE,
            'n_steps': N_STEPS,
            'dt': DT,
            'hbar': HBAR,
            'input_amp': INPUT_AMP,
        },
    }
    save_path = '/tmp/psi_resonance_v2_results.npz'
    np.savez(save_path, **data)
    print(f"\n  Results saved to: {save_path}")

    # ─── Verdict ───────────────────────────────────────
    drive_powers = np.array([r['drive_power'] for r in results])
    drive_ratios = np.array([r['drive_ratio'] for r in results])
    dp_peak_idx = np.argmax(drive_powers)
    dp_peak_freq = freqs[dp_peak_idx]
    q = drive_powers.max() / max(drive_powers.mean(), 1e-12)

    n_resonances = 0
    for i in range(2, len(drive_powers) - 2):
        if (drive_powers[i] > drive_powers[i-1] and
            drive_powers[i] > drive_powers[i-2] and
            drive_powers[i] > drive_powers[i+1] and
            drive_powers[i] > drive_powers[i+2] and
            drive_powers[i] > drive_powers.mean()):
            n_resonances += 1

    print("\n" + "═" * 72)
    print("  ВЕРДИКТ")
    print("═" * 72)
    print(f"\n  Max Q:   {q:.2f}x @ {dp_peak_freq:.1f} Hz")
    print(f"  Мод:     {n_resonances}")
    print(f"  D/P max: {drive_ratios.max():.3f}")

    if n_resonances >= 2 and q > 1.3:
        print("\n  ✅ **МОЗГ = РЕЗОНАНСНОЕ ОЗЕРО**")
        print("     Коннектом избирательно усиливает входные частоты.")
        print("     Простой sin(f·t) → сложный выходной паттерн.")
        print("     Вычисление = физика волн на коннектоме.")
    elif n_resonances >= 1:
        print(f"\n  📡 Слабая резонансная мода @ {dp_peak_freq:.1f} Hz")
        print("     Нужна более сложная топология или больше N.")
    else:
        print(f"\n  ℹ️  Нет измеримого резонанса (Q={q:.2f}x)")
        print("     Причины: малый граф, ψ-wave coupling подавляет drive,")
        print("     или нужна другая модель нейрона.")
    print("═" * 72)

    print(f"\n  Saved to: {save_path}")


if __name__ == "__main__":
    main()
