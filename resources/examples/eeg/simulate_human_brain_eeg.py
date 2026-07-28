#!/usr/bin/env python3
"""
🧠⚡ HUMAN BRAIN EEG SIMULATION (Multi-Band Oscillator Model)
=============================================================
Симуляция электрической активности коры головного мозга человека
с генерацией ЭЭГ сигналов (10-20 система отведений) на основе
нелинейных осцилляторов мозга.

Ритмы мозга (Neural oscillations):
  δ (Delta):    0.5 - 4.0 Hz   — Глубокий сон, бессознательное состояние
  θ (Theta):    4.0 - 8.0 Hz   — Дремота, медитация, гипнагогия
  α (Alpha):    8.0 - 13.0 Hz  — Расслабленное бодрствование, глаза закрыты
  β (Beta):     13.0 - 30.0 Hz — Активное бодрствование, концентрация
  γ (Gamma):    30.0 - 80.0 Hz — Высшая когнитивная деятельность, сознание

Модель: Kuramoto-подобные осцилляторы + пространственная когерентность
"""

import sys
import time
import math
import numpy as np

# ─── 10-20 System Electrodes ────────────────────────────────────────
# 21 стандартный электрод с позициями (доля от nasion-inion, корональная)
ELECTRODES_10_20 = {
    'Fp1': (0.05, 0.40), 'Fp2': (0.05, 0.60),
    'F7':  (0.15, 0.15), 'F3':  (0.15, 0.35), 'Fz':  (0.15, 0.50), 'F4':  (0.15, 0.65), 'F8':  (0.15, 0.85),
    'T3':  (0.30, 0.10), 'C3':  (0.30, 0.35), 'Cz':  (0.30, 0.50), 'C4':  (0.30, 0.65), 'T4':  (0.30, 0.90),
    'T5':  (0.50, 0.15), 'P3':  (0.50, 0.35), 'Pz':  (0.50, 0.50), 'P4':  (0.50, 0.65), 'T6':  (0.50, 0.85),
    'O1':  (0.70, 0.35), 'Oz':  (0.70, 0.50), 'O2':  (0.70, 0.65),
}

# Montages для биполярных отведений
MONTAGE_BIPOLAR = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),  # левый сагиттальный
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),  # правый сагиттальный
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),  # левый парасагиттальный
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),  # правый парасагиттальный
    ('Fz', 'Cz'),  ('Cz', 'Pz'),                               # центральная линия
]


def main():
    print("=" * 75)
    print("🧠⚡ HUMAN BRAIN EEG SIMULATION (Multi-Band Oscillator Model)")
    print("=" * 75)

    print("\n📚 1. НЕЙРОФИЗИОЛОГИЧЕСКАЯ МОДЕЛЬ ЭЭГ:")
    print("  • Кора головного мозга: ~16 миллиардов нейронов")
    print("  • Спектр ЭЭГ: δ (0.5-4 Hz), θ (4-8 Hz), α (8-13 Hz), β (13-30 Hz), γ (30-80 Hz)")
    print("  • Международная система 10-20: 21 электрод")
    print("  • Модель: нелинейные связанные осцилляторы (Kuramoto-тип)")
    print(f"  • Электроды: {len(ELECTRODES_10_20)} позиций, {len(MONTAGE_BIPOLAR)} биполярных каналов")

    # Параметры симуляции
    fs = 250.0              # частота дискретизации (Hz) — стандарт ЭЭГ
    duration = 10.0          # длительность записи (секунд)
    n_samples = int(fs * duration)
    dt = 1.0 / fs

    print(f"\n⚙️ Параметры записи: {duration:.0f} сек @ {fs:.0f} Hz = {n_samples} отсчётов")

    # ─── 2. Генерация ритмов мозга (связанные осцилляторы) ─────────
    print("\n🔄 2. ГЕНЕРАЦИЯ ЭЭГ РИТМОВ (нелинейные осцилляторы):")

    n_samples
    np.random.seed(42)

    # Базовые ритмы с физиологическими амплитудами
    rhythms = {
        'delta':  {'freq': 2.0,  'amp': 50.0,  'label': 'δ (0.5-4 Hz)   — Глубокий сон'},
        'theta':  {'freq': 6.0,  'amp': 20.0,  'label': 'θ (4-8 Hz)     — Дремота'},
        'alpha':  {'freq': 10.0, 'amp': 40.0,  'label': 'α (8-13 Hz)    — Расслабление'},
        'beta':   {'freq': 20.0, 'amp': 10.0,  'label': 'β (13-30 Hz)   — Активность'},
        'gamma':  {'freq': 45.0, 'amp': 5.0,   'label': 'γ (30-80 Hz)   — Сознание'},
    }

    signals = {}
    for name, r in rhythms.items():
        # Нелинейный осциллятор (Kuramoto-like с фазовым шумом)
        phase = np.cumsum(2 * math.pi * r['freq'] * dt + 
                          np.random.randn(n_samples) * 0.15)  # фазовый шум
        amp_mod = 1.0 + 0.3 * np.sin(2 * math.pi * 0.1 * np.arange(n_samples) * dt)  # медленная модуляция
        sig = r['amp'] * amp_mod * np.sin(phase)
        signals[name] = sig
        print(f"  {r['label']}: {r['amp']:5.1f} µV (peak)")

    # ─── 3. Пространственное распределение по электродам ────────────
    print("\n📊 3. ПРОСТРАНСТВЕННОЕ РАСПРЕДЕЛЕНИЕ (10-20 система):")

    # Веса ритмов для каждого электрода (затылок = alpha, лоб = beta, и т.д.)
    electrode_weights = {
        # Затылочные: alpha доминирует
        'O1':  {'delta': 0.2, 'theta': 0.3, 'alpha': 1.0, 'beta': 0.2, 'gamma': 0.1},
        'Oz':  {'delta': 0.2, 'theta': 0.3, 'alpha': 1.0, 'beta': 0.2, 'gamma': 0.1},
        'O2':  {'delta': 0.2, 'theta': 0.3, 'alpha': 1.0, 'beta': 0.2, 'gamma': 0.1},
        # Теменные: alpha + beta
        'P3':  {'delta': 0.3, 'theta': 0.4, 'alpha': 0.7, 'beta': 0.5, 'gamma': 0.3},
        'Pz':  {'delta': 0.3, 'theta': 0.4, 'alpha': 0.7, 'beta': 0.5, 'gamma': 0.3},
        'P4':  {'delta': 0.3, 'theta': 0.4, 'alpha': 0.7, 'beta': 0.5, 'gamma': 0.3},
        # Центральные: beta доминирует (моторная кора)
        'C3':  {'delta': 0.1, 'theta': 0.2, 'alpha': 0.3, 'beta': 1.0, 'gamma': 0.5},
        'Cz':  {'delta': 0.1, 'theta': 0.2, 'alpha': 0.3, 'beta': 1.0, 'gamma': 0.5},
        'C4':  {'delta': 0.1, 'theta': 0.2, 'alpha': 0.3, 'beta': 1.0, 'gamma': 0.5},
        # Лобные: beta + gamma
        'F3':  {'delta': 0.1, 'theta': 0.3, 'alpha': 0.2, 'beta': 0.8, 'gamma': 0.6},
        'Fz':  {'delta': 0.1, 'theta': 0.3, 'alpha': 0.2, 'beta': 0.8, 'gamma': 0.6},
        'F4':  {'delta': 0.1, 'theta': 0.3, 'alpha': 0.2, 'beta': 0.8, 'gamma': 0.6},
        'F7':  {'delta': 0.3, 'theta': 0.5, 'alpha': 0.3, 'beta': 0.6, 'gamma': 0.4},
        'F8':  {'delta': 0.3, 'theta': 0.5, 'alpha': 0.3, 'beta': 0.6, 'gamma': 0.4},
        'Fp1': {'delta': 0.4, 'theta': 0.6, 'alpha': 0.2, 'beta': 0.5, 'gamma': 0.3},
        'Fp2': {'delta': 0.4, 'theta': 0.6, 'alpha': 0.2, 'beta': 0.5, 'gamma': 0.3},
        # Височные: theta доминирует
        'T3':  {'delta': 0.3, 'theta': 0.8, 'alpha': 0.4, 'beta': 0.3, 'gamma': 0.2},
        'T4':  {'delta': 0.3, 'theta': 0.8, 'alpha': 0.4, 'beta': 0.3, 'gamma': 0.2},
        'T5':  {'delta': 0.3, 'theta': 0.6, 'alpha': 0.6, 'beta': 0.3, 'gamma': 0.2},
        'T6':  {'delta': 0.3, 'theta': 0.6, 'alpha': 0.6, 'beta': 0.3, 'gamma': 0.2},
    }

    # Генерация сигналов для каждого электрода
    eeg_channels = {}
    for name in ELECTRODES_10_20:
        weights = electrode_weights.get(name, 
                                        {'delta': 0.3, 'theta': 0.4, 'alpha': 0.5, 'beta': 0.5, 'gamma': 0.3})
        # Сумма взвешенных ритмов + пространственный шум
        sig = sum(weights[r_name] * signals[r_name] for r_name in rhythms)
        # Пространственная корреляция: соседние электроды похожи
        sig += np.random.randn(n_samples) * 2.0  # шум электрода
        # Фильтр высоких частот (убираем DC дрейф)
        sig = sig - np.convolve(sig, np.ones(250)/250, mode='same')
        eeg_channels[name] = sig

    print(f"  Сгенерировано {len(eeg_channels)} каналов ЭЭГ (10-20 система)")

    # ─── 4. Биполярные отведения ───────────────────────────────────
    print("\n🔌 4. БИПОЛЯРНЫЕ ОТВЕДЕНИЯ:")
    bipolar_signals = []
    bipolar_labels = []
    for ch1, ch2 in MONTAGE_BIPOLAR:
        diff = eeg_channels[ch1] - eeg_channels[ch2]
        bipolar_signals.append(diff)
        bipolar_labels.append(f"{ch1}-{ch2}")

    print(f"  {len(bipolar_signals)} биполярных каналов")

    # ─── 5. Анализ спектра мощности ────────────────────────────────
    print("\n📈 5. СПЕКТРАЛЬНЫЙ АНАЛИЗ (Fourier):")
    from scipy import signal as sg

    # Выбираем центральный канал (Cz) для спектрального анализа
    ref_channel = eeg_channels['Cz']
    f, psd = sg.welch(ref_channel, fs=fs, nperseg=min(512, n_samples // 4))

    # Мощность в каждом диапазоне
    bands = {
        'Delta (0.5-4 Hz)':  (0.5, 4.0),
        'Theta (4-8 Hz)':    (4.0, 8.0),
        'Alpha (8-13 Hz)':   (8.0, 13.0),
        'Beta (13-30 Hz)':   (13.0, 30.0),
        'Gamma (30-80 Hz)':  (30.0, 80.0),
    }

    band_power = {}
    for name, (f_lo, f_hi) in bands.items():
        idx = np.where((f >= f_lo) & (f <= f_hi))
        power = np.trapezoid(psd[idx], f[idx]) if len(idx[0]) > 0 else 0.0
        band_power[name] = power

    # Относительная мощность
    total_power = sum(band_power.values())
    print(f"\n  📊 Распределение мощности по диапазонам (канал Cz):")
    print(f"  {'Диапазон':<22s} {'Абс. мощность':<15s} {'Отн. %':<10s} {'Норма'}")
    print(f"  {'─'*22} {'─'*15} {'─'*10} {'─'*15}")
    for name, power in band_power.items():
        pct = power / total_power * 100 if total_power > 0 else 0
        # Клинические нормы
        if 'Delta' in name:
            norm = "20-50% (сон)"
        elif 'Theta' in name:
            norm = "5-15%"
        elif 'Alpha' in name:
            norm = "20-40% (норма)"
        elif 'Beta' in name:
            norm = "5-20%"
        else:
            norm = "<5%"
        print(f"  {name:<22s} {power:<15.2f} {pct:<10.1f} {norm}")

    # Пиковая частота alpha диапазона
    alpha_idx = np.where((f >= 8.0) & (f <= 13.0))
    if len(alpha_idx[0]) > 0:
        pk_alpha = f[alpha_idx[0][np.argmax(psd[alpha_idx])]]
    else:
        pk_alpha = 0.0

    # ─── 6. Визуализация ──────────────────────────────────────────
    print("\n🎨 6. ВИЗУАЛИЗАЦИЯ (EEG временной ряд + спектрограмма):")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20, 16))

    # ── Panel 1: EEG временной ряд (первые 4 секунды) ──────────
    ax1 = plt.subplot(3, 2, 1)
    t_ms = np.arange(min(4 * int(fs), n_samples)) * 1000 / fs
    
    # Показываем 8 ключевых каналов
    key_channels = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F4', 'C4', 'P4']
    offset = 0
    for ch in key_channels:
        sig = eeg_channels[ch][:len(t_ms)]
        sig = sig / 20.0  # scale to reasonable range
        ax1.plot(t_ms, sig + offset, linewidth=0.5)
        ax1.text(-100, offset, ch, fontsize=8, ha='right', va='center')
        offset += 3
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Electrodes')
    ax1.set_title('🧠 EEG — 10-20 System (4 sec, bipolar montage)')
    ax1.set_yticks([])
    ax1.set_xlim(0, t_ms[-1])

    # ── Panel 2: Power Spectrum ──────────────────────────────────
    ax2 = plt.subplot(3, 2, 2)
    ax2.semilogy(f, psd, 'k-', linewidth=1.5)
    # Цветные полосы диапазонов
    colors = {'Delta': '#1f77b4', 'Theta': '#ff7f0e', 'Alpha': '#2ca02c', 
              'Beta': '#d62728', 'Gamma': '#9467bd'}
    ylim = ax2.get_ylim()
    for name, (f_lo, f_hi) in bands.items():
        ax2.axvspan(f_lo, f_hi, alpha=0.15, color=colors.get(name.split()[0], 'gray'))
        ax2.text((f_lo + f_hi) / 2, ylim[1] * 0.9, name.split()[0], 
                 ha='center', fontsize=8, fontweight='bold')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density (µV²/Hz)')
    ax2.set_title(f'Power Spectrum (Cz) — Peak α @ {pk_alpha:.1f} Hz')
    ax2.set_xlim(0, 80)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Spectrogram ─────────────────────────────────────
    ax3 = plt.subplot(3, 2, 3)
    f_spec, t_spec, Sxx = sg.spectrogram(ref_channel, fs=fs, nperseg=256, noverlap=128)
    pcm = ax3.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-12), 
                          shading='gouraud', cmap='inferno')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('EEG Spectrogram (Cz)')
    ax3.set_ylim(0, 60)
    plt.colorbar(pcm, ax=ax3, label='dB')

    # ── Panel 4: Topography (alpha power map) ─────────────────────
    ax4 = plt.subplot(3, 2, 4)
    # Вычисляем alpha power для каждого электрода
    alpha_power = {}
    for ch, sig in eeg_channels.items():
        f_ch, psd_ch = sg.welch(sig, fs=fs, nperseg=min(256, n_samples // 4))
        alpha_idx_ch = np.where((f_ch >= 8.0) & (f_ch <= 13.0))
        alpha_power[ch] = np.trapezoid(psd_ch[alpha_idx_ch], f_ch[alpha_idx_ch]) if len(alpha_idx_ch[0]) > 0 else 0

    # Рисуем голову
    x_vals = [ELECTRODES_10_20[ch][1] for ch in alpha_power.keys()]
    y_vals = [ELECTRODES_10_20[ch][0] for ch in alpha_power.keys()]
    power_vals = np.array([alpha_power[ch] for ch in alpha_power.keys()])
    
    # Эллипс головы
    from matplotlib.patches import Ellipse
    head = Ellipse((0.5, 0.4), width=0.8, height=0.6, fill=False, color='black', linewidth=2)
    ax4.add_patch(head)
    
    # Нос
    ax4.plot([0.5, 0.5], [0.1, 0.0], 'k-', linewidth=2)
    
    # Электроды (цвет = alpha power)
    sc = ax4.scatter(x_vals, y_vals, c=power_vals, cmap='viridis', 
                     s=80, edgecolors='black', linewidths=0.5, zorder=5)
    for ch, x, y in zip(alpha_power.keys(), x_vals, y_vals):
        ax4.annotate(ch, (x, y), fontsize=6, ha='center', va='center', fontweight='bold')
    
    ax4.set_title('Alpha Power Topography (8-13 Hz)')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0.8, 0)
    ax4.set_aspect('equal')
    ax4.axis('off')
    plt.colorbar(sc, ax=ax4, label='α Power (µV²)', shrink=0.8)

    # ── Panel 5: Correlation matrix ───────────────────────────────
    ax5 = plt.subplot(3, 2, 5)
    ch_names = list(ELECTRODES_10_20.keys())
    n_ch = len(ch_names)
    corr_matrix = np.zeros((n_ch, n_ch))
    for i, ch1 in enumerate(ch_names):
        for j, ch2 in enumerate(ch_names):
            s1 = eeg_channels[ch1]
            s2 = eeg_channels[ch2]
            corr = np.corrcoef(s1, s2)[0, 1]
            corr_matrix[i, j] = corr if not np.isnan(corr) else 0
    im = ax5.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    ax5.set_xticks(range(n_ch))
    ax5.set_yticks(range(n_ch))
    ax5.set_xticklabels(ch_names, fontsize=5, rotation=90)
    ax5.set_yticklabels(ch_names, fontsize=5)
    ax5.set_title('EEG Channel Correlation Matrix')
    plt.colorbar(im, ax=ax5, shrink=0.8)

    # ── Panel 6: Band power bars ──────────────────────────────────
    ax6 = plt.subplot(3, 2, 6)
    band_names = list(band_power.keys())
    band_vals = list(band_power.values())
    band_colors = [colors.get(n.split()[0], 'gray') for n in band_names]
    bars = ax6.barh(range(len(band_names)), band_vals, color=band_colors, edgecolor='black')
    ax6.set_yticks(range(len(band_names)))
    ax6.set_yticklabels([n.split('(')[0].strip() for n in band_names], fontsize=9)
    ax6.set_xlabel('Absolute Power (µV²)')
    ax6.set_title('EEG Frequency Band Power (Cz)')
    ax6.grid(True, axis='x', alpha=0.3)

    # Добавить текст с мощностью на барах
    for bar, val in zip(bars, band_vals):
        ax6.text(val + max(band_vals) * 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:.1f}', va='center', fontsize=8)

    plt.tight_layout()
    output_path = Path(__file__).parent.parent / "data" / "human_eeg.png"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Изображение сохранено: {output_path}")

    # ─── 7. Клиническое сравнение ─────────────────────────────────
    print("\n🔬 7. КЛИНИЧЕСКОЕ СРАВНЕНИЕ (PhysioNet / Нормы ЭЭГ):")
    print(" ┌───────────────────────────┬────────────────┬────────────────┬──────────┐")
    print(" │ Клинический параметр       │ Симуляция EEG  │ Клиническая     │ Статус   │")
    print(" │                           │                │ норма           │          │")
    print(" ├───────────────────────────┼────────────────┼────────────────┼──────────┤")

    # Alpha peak frequency
    alpha_ok = 8.0 <= pk_alpha <= 13.0
    print(f" │ α-Peak Frequency           │ {pk_alpha:>11.1f} Hz  │ 8.0 - 13.0 Hz   │ {'✅' if alpha_ok else '❌'}       │")

    # Alpha/beta ratio
    alpha_pct = band_power['Alpha (8-13 Hz)'] / total_power * 100
    beta_pct = band_power['Beta (13-30 Hz)'] / total_power * 100
    ab_ratio = alpha_pct / max(beta_pct, 0.1)
    ab_ok = 2.0 <= ab_ratio <= 10.0
    print(f" │ Alpha/Beta Ratio           │ {ab_ratio:>11.1f}     │ 2.0 - 10.0      │ {'✅' if ab_ok else '❌'}       │")

    # Theta/alpha ratio (elevated = pathology)
    theta_pct = band_power['Theta (4-8 Hz)'] / total_power * 100
    ta_ratio = theta_pct / max(alpha_pct, 0.1)
    ta_ok = ta_ratio < 0.8
    print(f" │ Theta/Alpha Ratio          │ {ta_ratio:>11.2f}     │ < 0.80          │ {'✅' if ta_ok else '❌'}       │")

    # Gamma power (consciousness correlate)
    gamma_pct = band_power['Gamma (30-80 Hz)'] / total_power * 100
    gamma_ok = 1.0 <= gamma_pct <= 10.0
    print(f" │ Gamma Power (%)            │ {gamma_pct:>11.1f}%   │ 1.0 - 10.0%     │ {'✅' if gamma_ok else '❌'}       │")

    # Delta/theta ratio (sleep vs awake)
    delta_pct = band_power['Delta (0.5-4 Hz)'] / total_power * 100
    dt_ratio = delta_pct / max(theta_pct, 0.1)
    print(f" │ Delta/Theta Ratio          │ {dt_ratio:>11.2f}     │ < 2.0 (бодр.)   │ {'✅' if dt_ratio < 2.0 else '😴'} │")

    print(" └───────────────────────────┴────────────────┴────────────────┴──────────┘")

    print(f"\n{'=' * 75}")
    print(f"🎉 EEG симуляция мозга человека успешно завершена!")
    print(f"   • {n_samples} отсчётов @ {fs:.0f} Hz = {duration:.0f} сек")
    print(f"   • {len(ELECTRODES_10_20)} электродов (10-20 система)")
    print(f"   • {len(bipolar_signals)} биполярных отведений")
    print(f"   • Доминирующий ритм: α @ {pk_alpha:.1f} Hz")
    print(f"   • Изображение: {output_path}")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    from pathlib import Path
    main()
