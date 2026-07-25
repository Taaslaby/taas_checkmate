#!/usr/bin/env python3
"""
🚀 FULL AUTONOMOUS FAFB FLYWIRE BRAIN SIMULATION — 5000 STEPS + ATTRACTOR ANALYSIS
==================================================================================
Симуляция фрагмента/полного мозга Drosophila FAFB на супер-реалистичных
сомато-дендритных нейронах с активными ионными каналами (Na+, K+, Ca2+),
AMPA/NMDA/GABA рецепторами и фоновой самопроизвольной активностью.

Детекция бистабильности: отслеживаем переходы между режимами активности,
автокорреляцию, би-модальность распределения спайков и спектр мощности.

"Пусть живут сами!" 🧠⚡ → "Что они делают когда никто не смотрит?"
"""

import os
import sys
import time
import gzip
import csv
from pathlib import Path
import numpy as np

import taas_checkmate as tc

DLOAD = Path.home() / "Загрузки"
CONN_FILE = DLOAD / "connections_princeton.csv.gz"


def load_flywire_graph(max_synapses=200000):
    print("📁 Загрузка графа связей Drosophila FAFB из ~/Загрузки/...")
    t0 = time.time()

    if not CONN_FILE.exists():
        print(f"❌ Файл не найден: {CONN_FILE}")
        sys.exit(1)

    edges = []

    with gzip.open(CONN_FILE, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pre = int(row["pre_root_id"])
            post = int(row["post_root_id"])
            syn_count = int(row.get("syn_count", 1))
            edges.append((pre, post, syn_count))
            if len(edges) >= max_synapses:
                break

    dt = time.time() - t0
    print(f"✅ Прочитано {len(edges):,} синапсов за {dt:.2f} сек!")
    return edges


# ─── Attractor analysis functions ──────────────────────────────────────


def running_mean(x, w):
    """Running mean with window w."""
    return np.convolve(x, np.ones(w) / w, mode="valid")


def bimodality_coefficient(samples):
    """Sarle's bimodality coefficient: (g²+1)/(k+3*(n-1)²/((n-2)*(n-3))).
    > 0.555 suggests bimodal distribution.
    """
    n = len(samples)
    if n < 4:
        return 0.0
    g1 = float(np.mean((samples - np.mean(samples)) ** 3)) / (np.std(samples) ** 3 + 1e-10)
    g2 = float(np.mean((samples - np.mean(samples)) ** 4)) / (np.std(samples) ** 4 + 1e-10) - 3.0
    k = n
    bc = (g1 ** 2 + 1.0) / (g2 + 3.0 * (k - 1.0) ** 2 / ((k - 2.0) * (k - 3.0)) + 1e-10)
    return float(np.clip(bc, 0.0, 1.0))


def classify_network_state(spike_counts, bin_size=10):
    """
    Классифицирует состояние сети: Fixed Point, Limit Cycle, Chaos, Bistable.
    """
    # Бинним спайки в окна bin_size для сглаживания
    n_bins = len(spike_counts) // bin_size
    binned = np.array([spike_counts[i * bin_size:(i + 1) * bin_size].sum()
                       for i in range(n_bins)])

    cv = np.std(binned) / (np.mean(binned) + 1e-10)
    bc = bimodality_coefficient(binned)

    # Автокорреляция lag-1
    ac_lag1 = np.corrcoef(binned[:-1], binned[1:])[0, 1] if len(binned) > 2 else 0.0

    # Fano factor (variance/mean)
    fano = np.var(binned) / (np.mean(binned) + 1e-10)

    # Power spectrum: ищем доминантную частоту
    fft_vals = np.abs(np.fft.rfft(binned - np.mean(binned)))
    freqs = np.fft.rfftfreq(len(binned), d=bin_size * 0.5 / 1000)  # в Hz
    dominant_freq = freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0.0
    power_peak_ratio = np.max(fft_vals) / (np.mean(fft_vals[fft_vals > 0]) + 1e-10)

    # Разделяем на первые 40% и последние 40% для детекции дрейфа
    split = len(binned) // 2
    first_half = binned[:split]
    second_half = binned[split:]
    ttest_val = abs(np.mean(first_half) - np.mean(second_half)) / (
        np.std(first_half) + np.std(second_half) + 1e-10)

    # Classification logic
    mean_rate = np.mean(binned)
    results = {
        "CV": float(cv),
        "Fano": float(fano),
        "Bimodality": float(bc),
        "AC_lag1": float(ac_lag1),
        "Dominant_Hz": float(dominant_freq),
        "Power_peak_ratio": float(power_peak_ratio),
        "Drift_ttest": float(ttest_val),
        "Mean_spikes_per_bin": float(mean_rate),
    }

    # Fixed Point: очень низкая вариативность, низкий CV
    # Limit Cycle: сильный пик в спектре, высокая AC_lag1
    # Chaos: высокий CV, низкая AC_lag1, широкий спектр
    # Bistable: высокий bimodality, дрейф между режимами

    if cv < 0.3 and mean_rate > 0.1:
        results["state"] = "🟢 FIXED POINT (низкая вариативность)"
        results["state_code"] = "fixed_point"
    elif bc > 0.555 and cv > 0.8:
        results["state"] = "🟣 BISTABLE (би-модальное распределение)"
        results["state_code"] = "bistable"
    elif power_peak_ratio > 3.0 and ac_lag1 > 0.3:
        results["state"] = "🔵 LIMIT CYCLE (осцилляции: {:.1f} Hz)".format(dominant_freq)
        results["state_code"] = "limit_cycle"
    elif cv > 1.5 and ac_lag1 < 0.2:
        results["state"] = "🔴 CHAOS (высокий CV, низкая AC)"
        results["state_code"] = "chaos"
    elif fano > 2.0:
        results["state"] = "🟠 BURSTING (Fano > 2, пачки спайков)"
        results["state_code"] = "bursting"
    else:
        results["state"] = "⚪ STEADY (умеренная активность)"
        results["state_code"] = "steady"

    return results


def detect_phase_transitions(spike_counts, window=50, threshold=2.0):
    """
    Детектит моменты фазовых переходов: резкие изменения средней спайковой
    активности. Скользящее окно, z-score разницы средних.
    """
    if len(spike_counts) < window * 2:
        return []

    transitions = []
    for i in range(window, len(spike_counts) - window):
        before = np.mean(spike_counts[i - window:i])
        after = np.mean(spike_counts[i:i + window])
        std_before = np.std(spike_counts[i - window:i]) + 1e-10
        z = abs(after - before) / std_before
        if z > threshold:
            transitions.append((i, float(z), float(before), float(after)))

    # Merge close transitions
    if len(transitions) < 2:
        return transitions

    merged = [transitions[0]]
    for t in transitions[1:]:
        if t[0] - merged[-1][0] < window:
            merged[-1] = (merged[-1][0], max(merged[-1][1], t[1]),
                          merged[-1][2], t[3])
        else:
            merged.append(t)
    return merged


# ─── Main simulation ──────────────────────────────────────────────────


def main():
    print("\n" + "═" * 70)
    print("🚀 FAFB FLYWIRE AUTONOMOUS BRAIN — 5000 STEPS ATTRACTOR ANALYSIS")
    print("═" * 70)

    # ── 1. Load connectome ──────────────────────────────────────────
    raw_edges = load_flywire_graph(max_synapses=200000)

    unique_nodes = sorted(list(set([e[0] for e in raw_edges] + [e[1] for e in raw_edges])))
    node_to_idx = {node: i for i, node in enumerate(unique_nodes)}
    N = len(unique_nodes)
    print(f"🧠 Инициализация {N:,} супер-реалистичных нейронов...")

    edges = [(node_to_idx[pre], node_to_idx[post], syn) for pre, post, syn in raw_edges]

    # ── 2. Somato-Dendritic population ─────────────────────────────
    pop = tc.neurons.SomatoDendriticNeuron(
        n_neurons=N,
        n_compartments=4,
        r_a=180.0, r_m=120.0, c_m=0.1, dt=0.5,
        tau_m=20.0, v_rest=-65.0, v_thresh=-55.0, v_reset=-65.0, tau_ref=2.5
    )

    print("⚡ Включение активных дендритных каналов (Na+ m³h, K+ n⁴, Ca²⁺ m_ca²)...")
    pop.enable_active_channels(g_na=12.0, g_kv=6.0, g_ca=3.0)

    N = len(pop.v)
    np.random.seed(42)

    # ── 3. Simulation parameters ───────────────────────────────────
    n_steps = 5000  # 2.5 секунды физического времени!
    dt_ms = 0.5
    window = 100  # running metrics window

    # Data collection arrays
    spike_per_step = np.zeros(n_steps, dtype=np.int32)
    v_soma_mean_per_step = np.zeros(n_steps, dtype=np.float32)
    v_dend_sample_per_step = np.zeros(n_steps, dtype=np.float32)
    active_neurons_per_step = np.zeros(n_steps, dtype=np.int32)

    # ── 4. Precompute edge lookup ──────────────────────────────────
    # Для производительности: группируем рёбра по пре-нейрону
    from collections import defaultdict
    edge_map = defaultdict(list)
    for pre, post, syn in edges:
        edge_map[pre].append((post, syn))

    print("🌊 Активация фонового спонтанного шума (Poisson Background Drive)...")
    print(f"📊 Запуск {n_steps} шагов ({n_steps * dt_ms:.0f} ms физического времени)...")
    print("\n" + "─" * 70)
    print("🎬 СТАРТ СИМУЛЯЦИИ: Мозг начинает жить своей жизнью!")
    print("─" * 70)

    t_start = time.time()
    total_spikes = 0

    for step in range(n_steps):
        # ── Background Poisson drive ───────────────────────────────
        I_bg = np.random.randn(N).astype(np.float32) * 4.0 + 8.5

        # ── Periodic sensory pulse ─────────────────────────────────
        if step % 30 == 0:
            stim_nodes = np.random.choice(N, size=int(N * 0.05), replace=False)
            I_bg[stim_nodes] += 15.0

        # ── Synaptic transmission via precomputed edge_map ─────────
        spikes_prev = pop.spikes
        if step > 0 and spikes_prev.sum() > 0:
            # Быстрый проход по спайкующим нейронам через edge_map
            spiking_idx = np.where(spikes_prev == 1)[0]
            for pre in spiking_idx:
                if pre in edge_map:
                    for post, syn in edge_map[pre]:
                        comp = (pre + post) % 4
                        if (pre * 7 + post) % 5 == 0:
                            pop.jump_gaba_a(post, comp, 0.04 * syn)
                        else:
                            pop.jump_ampa(post, comp, 0.05 * syn)
                            pop.jump_nmda(post, comp, 0.02 * syn)

        # ── Somato-dendritic step ──────────────────────────────────
        pop.step(I_bg, dt_ms)

        # ── Record (читаем ПОСЛЕ step — текущий шаг) ───────────────
        current_spikes = int(pop.spikes.sum()) if step > 0 else 0
        total_spikes += current_spikes
        spike_per_step[step] = current_spikes
        v_soma_mean_per_step[step] = float(pop.v[:N].mean())
        v_dend_sample_per_step[step] = float(pop.v_dend(0, 3))
        active_neurons_per_step[step] = current_spikes

        # ── Progress + live attractor diagnostics ──────────────────
        if (step + 1) % 200 == 0 or step == 0:
            elapsed = time.time() - t_start
            fps = (step + 1) / (elapsed + 1e-6)
            bar_len = min(30, int(current_spikes / max(1, N * 0.05) * 30))
            ascii_bar = "█" * bar_len + "░" * (30 - bar_len)

            # Running attractor hint
            if step >= window:
                recent = spike_per_step[step - window:step]
                recent_mean = recent.mean()
                recent_cv = recent.std() / (recent_mean + 1e-10)
                recent_ac = np.corrcoef(recent[:-1], recent[1:])[0, 1] if window > 2 else 0.0
            else:
                recent_mean = 0.0
                recent_cv = 0.0
                recent_ac = 0.0

            print(f"⏱ Step {step+1:5d}/{n_steps} | "
                  f"V: {v_soma_mean_per_step[step]:6.2f} mV | "
                  f"Spikes: {current_spikes:4d} [{ascii_bar}] | "
                  f"μ={recent_mean:.1f} CV={recent_cv:.2f} AC={recent_ac:.2f} | "
                  f"{fps:5.0f} step/s")

    total_time = time.time() - t_start

    # ══════════════════════════════════════════════════════════════════
    #  ATTRACTOR ANALYSIS
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "═" * 70)
    print("🧠 ATTRACTOR ANALYSIS — ДИАГНОСТИКА СОСТОЯНИЯ СЕТИ")
    print("═" * 70)

    # Используем вторую половину симуляции (после разогрева)
    warmup = 1000  # пропускаем первые 1000 шагов (500ms) — разогрев
    steady_state = spike_per_step[warmup:]

    state = classify_network_state(steady_state, bin_size=10)
    transitions = detect_phase_transitions(spike_per_step, window=50, threshold=2.0)

    # Дополнительная статистика
    mean_fr = total_spikes / N / (n_steps * dt_ms / 1000)
    active_fraction = len(np.where(spike_per_step > 0)[0]) / n_steps
    last_variance = np.var(v_soma_mean_per_step[-1000:])

    print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"  • Нейронов:           {N:,}")
    print(f"  • Синапсов:           {len(edges):,}")
    print(f"  • Физическое время:   {n_steps * dt_ms:.0f} ms ({n_steps * dt_ms / 1000:.2f} sec)")
    print(f"  • Реальное время:     {total_time:.2f} sec")
    print(f"  • Скорость:           {n_steps / total_time:.1f} step/s")
    print(f"  • Всего спайков:      {total_spikes:,}")
    print(f"  • Средняя частота:    {mean_fr:.2f} Hz")
    print(f"  • V_soma дисперсия:   {last_variance:.4f}")

    print(f"\n🔬 КЛАССИФИКАЦИЯ СОСТОЯНИЯ СЕТИ (post-warmup, >{warmup} steps):")
    print(f"  {state['state']}")
    print(f"  ┌──────────────────────┬──────────┐")
    print(f"  │ CV (Coeff Variation) │ {state['CV']:>8.3f} │")
    print(f"  │ Fano Factor          │ {state['Fano']:>8.3f} │")
    print(f"  │ Bimodality Coeff     │ {state['Bimodality']:>8.3f} │")
    print(f"  │ AC lag-1             │ {state['AC_lag1']:>8.3f} │")
    print(f"  │ Dominant Frequency   │ {state['Dominant_Hz']:>8.2f} Hz │")
    print(f"  │ Power Peak Ratio     │ {state['Power_peak_ratio']:>8.2f} │")
    print(f"  └──────────────────────┴──────────┘")

    if len(transitions) > 0:
        print(f"\n⚡ ФАЗОВЫЕ ПЕРЕХОДЫ ({len(transitions)} обнаружено):")
        for t_idx, (step_at, z, before, after) in enumerate(transitions):
            t_ms = step_at * dt_ms
            print(f"  #{t_idx + 1} @ {t_ms:.1f}ms  Z={z:.1f}  μ_before={before:.1f} → μ_after={after:.1f}")
    else:
        print(f"\n⚡ ФАЗОВЫЕ ПЕРЕХОДЫ: не обнаружены (сеть стабильна)")

    print(f"\n🧬 ИНТЕРПРЕТАЦИЯ:")
    if state["state_code"] == "fixed_point":
        print(f"  🟢 Сеть пришла в фиксированную точку — нейроны спайкуют")
        print(f"     с постоянной низкой дисперсией. Классический аттрактор.")
        print(f"     FAFB коннектом самобалансируется в стабильное состояние.")
    elif state["state_code"] == "limit_cycle":
        print(f"  🔵 Сеть генерирует устойчивые осцилляции на {state['Dominant_Hz']:.1f} Hz.")
        print(f"     Это может быть CPG-подобный паттерн (генератор ритма).")
    elif state["state_code"] == "chaos":
        print(f"  🔴 Сеть в хаотическом режиме — высокая чувствительность")
        print(f"     к начальным условиям. Богатая динамика, но без устойчивого")
        print(f"     паттерна. Характерно для критических состояний.")
    elif state["state_code"] == "bistable":
        print(f"  🟣 Сеть переключается между двумя устойчивыми режимами!")
        print(f"     Это бистабильность — ключевое свойство рабочей памяти")
        print(f"     и GNW (Global Neuronal Workspace).")
        print(f"     FAFB коннектом демонстрирует метастабильность!")
    elif state["state_code"] == "bursting":
        print(f"  🟠 Сеть генерирует пачки спайков (bursts) с периодами")
        print(f"     относительного затишья. Типично для Ca²⁺-зависимых нейронов.")
    else:
        print(f"  ⚪ Умеренная активность без выраженных паттернов.")

    # Save data for post-analysis
    out_dir = Path(__file__).parent / ".." / ".." / "data"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "fafb_5000steps.npz",
        spike_per_step=spike_per_step,
        v_soma_mean=v_soma_mean_per_step,
        v_dend_sample=v_dend_sample_per_step,
        active_neurons=active_neurons_per_step,
        state_info=state,
    )
    print(f"\n💾 Данные сохранены: {out_dir / 'fafb_5000steps.npz'}")
    print("═" * 70)


if __name__ == "__main__":
    main()
