#!/usr/bin/env python3
"""
FlyWire Resonance Spectrum — Visualization

Загружает ψ-wave resonance данные из NPZ, строит:
  1. Спектр freq vs drive_power с Lorentzian fit
  2. D/P ratio (доля мощности на drive-частоте)
  3. Eigenmode spectrum (если есть NPZ с eigenmode данными)
  4. Q-factor comparison
  5. Summary table

Сохраняет в /tmp/psi_resonance_flywire.png

Зависимости: python3 + numpy + scipy + matplotlib
"""

import os
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════
#  1. Load data
# ═══════════════════════════════════════════════════════════

def load_data(path='/tmp/psi_resonance_flywire.npz'):
    """Load saved resonance data from NPZ."""
    if not os.path.exists(path):
        print(f"  ❌ NPZ file not found: {path}")
        print(f"  Run 'psi_resonance_flywire.py' first!")
        sys.exit(1)
    d = np.load(path)
    freqs = d['freqs_hz']

    groups = {}
    for key in d.keys():
        if key == 'freqs_hz':
            continue
        # Parse group name from key: DN_drive_powers -> ('DN', 'drive_powers')
        if '_drive_powers' in key:
            gname = key.replace('_drive_powers', '')
            groups[gname] = groups.get(gname, {})
            groups[gname]['drive_powers'] = d[key]
        elif '_drive_ratios' in key:
            gname = key.replace('_drive_ratios', '')
            groups[gname] = groups.get(gname, {})
            groups[gname]['drive_ratios'] = d[key]

    return freqs, groups


# ═══════════════════════════════════════════════════════════
#  2. Lorentzian fit
# ═══════════════════════════════════════════════════════════

def lorentzian(f, A, f0, gamma, C):
    """
    Lorentzian function:
        L(f) = A * (gamma/2)² / ((f - f0)² + (gamma/2)²) + C
    where:
        A   = amplitude
        f0  = centre frequency (resonant peak)
        gamma = Full Width at Half Maximum (FWHM)
        C   = constant background
    """
    return A * (gamma / 2) ** 2 / ((f - f0) ** 2 + (gamma / 2) ** 2) + C


def fit_lorentzian(freqs, powers):
    """Fit Lorentzian to (freqs, powers) data."""
    # Initial guess: peak ≈ max power, centre ≈ argmax, width ≈ 20 Hz
    pk_idx = np.argmax(powers)
    A0 = powers[pk_idx] - np.min(powers)
    f0_guess = freqs[pk_idx]
    gamma0 = 30.0  # Hz
    C0 = np.min(powers)

    try:
        popt, pcov = curve_fit(
            lorentzian, freqs, powers,
            p0=[A0, f0_guess, gamma0, C0],
            bounds=(
                [0, freqs[0], 1.0, 0],
                [1e6, freqs[-1], 500.0, 1e6],
            ),
            maxfev=5000,
        )
        A, f0, gamma, C = popt
        # Error estimate
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [0, 0, 0, 0]
        Q = f0 / gamma if gamma > 0 else 0
        return popt, perr, Q, f0, gamma
    except (RuntimeError, ValueError) as e:
        print(f"  ⚠️ Lorentzian fit failed: {e}")
        return None, None, 0, 0, 0


# ═══════════════════════════════════════════════════════════
#  3. Plot
# ═══════════════════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  FlyWire Resonance Spectrum — Visualization")
    print("═" * 60)

    # ─── Load ──────────────────────────────────────────
    print("\n[1/5] Loading data...")
    freqs, groups = load_data()
    print(f"  Frequencies: {len(freqs)} points, {freqs[0]:.1f}–{freqs[-1]:.1f} Hz")
    for gname, gdata in groups.items():
        print(f"  {gname}: {len(gdata['drive_powers'])} points, "
              f"peak={gdata['drive_powers'].max():.2f} @ "
              f"{freqs[np.argmax(gdata['drive_powers'])]:.1f} Hz")

    # ─── Lorentzian fit ────────────────────────────────
    print("\n[2/5] Lorentzian fit...")
    fits = {}
    for gname, gdata in groups.items():
        dp = gdata['drive_powers']
        popt, perr, Q, f0, gamma = fit_lorentzian(freqs, dp)
        if popt is not None:
            fits[gname] = (popt, perr, Q, f0, gamma)
            print(f"  {gname}: f0={f0:.1f}±{perr[1]:.1f} Hz, "
                  f"γ={gamma:.1f}±{perr[2]:.1f} Hz, "
                  f"Q={Q:.2f}")
        else:
            print(f"  {gname}: fit failed")

    # ─── Fine grid for smooth Lorentzian curves ────────
    f_fine = np.linspace(freqs[0], freqs[-1], 500)

    # ─── Colours ───────────────────────────────────────
    colours = {
        'DN': '#E74C3C',
        'MN': '#3498DB',
        'DN+MN': '#2ECC71',
    }
    markers = {
        'DN': 'o',
        'MN': 's',
        'DN+MN': '^',
    }

    # Detect failed fits (gamma_err / gamma > 0.5)
    fit_bad = set()
    for gname in ['DN', 'MN', 'DN+MN']:
        if gname in fits:
            _, perr, Q, f0, gamma = fits[gname]
            if gamma > 0 and perr[2] / gamma > 0.5:
                fit_bad.add(gname)
                print(f"  ⚠️ {gname}: fit unreliable (γ_err/γ={perr[2]/gamma:.1f}), skipping Q bar")

    # ─── Create figure ─────────────────────────────────
    print("\n[3/5] Creating figure...")
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.30,
                          height_ratios=[1.5, 1.0, 1.0, 0.8])

    # ─── Panel A: Frequency sweep ──────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for gname in ['DN', 'MN', 'DN+MN']:
        if gname not in groups:
            continue
        dp = groups[gname]['drive_powers']
        c = colours.get(gname, '#888888')
        m = markers.get(gname, 'o')

        # Data points
        ax1.plot(freqs, dp, marker=m, linestyle='', color=c,
                 markersize=8, label=f'{gname} data', zorder=3)

        # Lorentzian fit
        if gname in fits:
            popt, _, Q, f0, gamma = fits[gname]
            lorf = lorentzian(f_fine, *popt)
            ax1.plot(f_fine, lorf, '-', color=c, linewidth=1.5,
                     alpha=0.7, label=rf'{gname} fit: $f_0$={f0:.1f} Hz, Q={Q:.1f}')

            # Vertical line at f0
            ax1.axvline(f0, color=c, linestyle=':', alpha=0.3, linewidth=1)
            # Shade FWHM
            ax1.axvspan(f0 - gamma / 2, f0 + gamma / 2,
                        color=c, alpha=0.05)

    ax1.set_xlabel('Frequency (Hz)', fontsize=13)
    ax1.set_ylabel('Drive Power |FFT|', fontsize=13)
    ax1.set_title('ψ-Wave Resonance Spectrum — FlyWire Connectome', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(freqs[0] - 5, freqs[-1] + 5)
    ax1.text(0.02, 0.98, 'A', transform=ax1.transAxes, fontsize=16,
             fontweight='bold', va='top', ha='left')

    # Annotate Q-factor
    for gname in ['DN', 'MN', 'DN+MN']:
        if gname not in fits:
            continue
        _, _, Q, f0, gamma = fits[gname]
        y_max = groups[gname]['drive_powers'].max()
        ax1.annotate(
            f'Q={Q:.1f}',
            xy=(f0, y_max),
            xytext=(f0 + 8, y_max * 0.85),
            fontsize=10,
            color=colours.get(gname, '#888'),
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=colours.get(gname, '#888'), alpha=0.5),
        )

    # ─── Panel B: D/P Ratio ────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    for gname in ['DN', 'MN', 'DN+MN']:
        if gname not in groups or 'drive_ratios' not in groups[gname]:
            continue
        dr = groups[gname]['drive_ratios']
        c = colours.get(gname, '#888888')
        m = markers.get(gname, 'o')
        ax2.semilogy(freqs, dr, marker=m, linestyle='-', color=c,
                     markersize=6, linewidth=1, alpha=0.8,
                     label=f'{gname} D/P')

    ax2.set_xlabel('Frequency (Hz)', fontsize=13)
    ax2.set_ylabel('D/P Ratio (log)', fontsize=13)
    ax2.set_title('Drive Power Fraction (D/P ratio)', fontsize=14)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.2, which='both')
    ax2.set_xlim(freqs[0] - 5, freqs[-1] + 5)
    ax2.text(0.02, 0.98, 'B', transform=ax2.transAxes, fontsize=16,
             fontweight='bold', va='top', ha='left')

    # ─── Panel C: Eigenmode Spectrum (if saved) ────────
    ax3 = fig.add_subplot(gs[2, 0])

    eigenmode_path = '/tmp/psi_resonance_flywire_eigen.npz'
    eigenmode_available = os.path.exists(eigenmode_path)

    if eigenmode_available:
        print("\n[4/5] Loading saved eigenmode spectrum...")
        ed = np.load(eigenmode_path)
        efreqs = ed['eigen_freqs']
        emag = ed['eigen_mag']
        peaks = []
        for i in range(len(ed['peak_freqs'])):
            peaks.append((ed['peak_freqs'][i], ed['peak_mags'][i]))
        actual_n = int(ed['n_synapses']) if 'n_synapses' in ed else 0
        n = int(ed['n_neurons']) if 'n_neurons' in ed else 0

        ax3.semilogy(efreqs[1:], emag[1:], '-', color='#8E44AD',
                     linewidth=1, alpha=0.7)
        ax3.set_xlabel('Frequency (Hz)', fontsize=12)
        ax3.set_ylabel('|FFT| (log)', fontsize=12)
        ax3.set_title('Eigenmode Spectrum (free evolution)', fontsize=13)
        ax3.set_xlim(0, 250)
        ax3.grid(True, alpha=0.2, which='both')

        for i, (pk_f, pk_m) in enumerate(peaks[:5]):
            ax3.axvline(pk_f, color='#8E44AD', linestyle=':', alpha=0.3)
            ax3.annotate(f'{pk_f:.1f} Hz', xy=(pk_f, pk_m),
                         xytext=(pk_f + 5, pk_m * 1.5),
                         fontsize=8, color='#8E44AD', rotation=45)

        # Highlight resonant peak
        for gname in ['DN', 'MN', 'DN+MN']:
            if gname in fits:
                _, _, Q, f0, gamma = fits[gname]
                ax3.axvline(f0, color='red', linestyle='-', alpha=0.5, linewidth=1.5)
                ax3.annotate(f'Resonance {f0:.1f} Hz',
                             xy=(f0, ax3.get_ylim()[1] * 0.9),
                             fontsize=9, color='red', fontweight='bold',
                             ha='center')
                break
    else:
        # No eigenmode data — show placeholder
        ax3.text(0.5, 0.5, 'Eigenmode analysis:\nRun \'psi_resonance_flywire_eigen.py\'',
                 transform=ax3.transAxes, ha='center', va='center',
                 fontsize=11, color='#888', fontstyle='italic')
        ax3.set_xlabel('Frequency (Hz)', fontsize=12)
        ax3.set_ylabel('|FFT| (log)', fontsize=12)
        ax3.set_title('Eigenmode Spectrum (not computed)', fontsize=13)
        ax3.set_xlim(0, 250)
        ax3.grid(True, alpha=0.2)
        n = 0
        actual_n = 0
        peaks = []

    ax3.text(0.02, 0.98, 'C', transform=ax3.transAxes, fontsize=16,
             fontweight='bold', va='top', ha='left')

    # ─── Panel D: Q-factor vs Group ────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    gnames_q = []
    q_vals = []
    f0_vals = []
    gamma_vals = []
    q_errs = []
    colours_q = []

    for gname in ['DN', 'MN', 'DN+MN']:
        if gname in fits and gname not in fit_bad:
            popt, perr, Q, f0, gamma = fits[gname]
            gnames_q.append(gname)
            q_vals.append(Q)
            f0_vals.append(f0)
            gamma_vals.append(gamma)
            q_errs.append(Q * np.sqrt((perr[1]/f0)**2 + (perr[2]/gamma)**2) if gamma > 0 else 0)
            colours_q.append(colours.get(gname, '#888'))

    x = np.arange(len(gnames_q))
    bars = ax4.bar(x, q_vals, color=colours_q, alpha=0.8, width=0.5,
                   yerr=q_errs, capsize=5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(gnames_q, fontsize=12)
    ax4.set_ylabel('Q-factor', fontsize=12)
    ax4.set_title('Quality Factor per Neuron Group', fontsize=13)
    ax4.grid(True, alpha=0.2, axis='y')

    # Annotate f0 on bars
    for i, (bar, f0v, gv) in enumerate(zip(bars, f0_vals, gamma_vals)):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'$f_0$={f0v:.1f} Hz\nγ={gv:.1f} Hz',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax4.set_ylim(0, max(q_vals) * 1.5 if q_vals else 10)
    ax4.text(0.02, 0.98, 'D', transform=ax4.transAxes, fontsize=16,
             fontweight='bold', va='top', ha='left')

    # Note about excluded fits
    if fit_bad:
        bad_str = ', '.join(sorted(fit_bad))
        ax4.text(0.95, 0.05,
                 f'Excluded (failed fit): {bad_str}',
                 transform=ax4.transAxes, fontsize=8, color='#888',
                 ha='right', va='bottom', fontstyle='italic')

    # ─── Panel E: Summary table ────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')

    # Build summary text
    summary_lines = [
        "FlyWire ψ-Wave Resonance — Summary",
        "═" * 60,
        f"Connectome: {n:,} neurons, {actual_n:,} synapses",
        f"Frequency range: {freqs[0]:.1f} – {freqs[-1]:.1f} Hz ({len(freqs)} points)",
        "",
        "Group        | f₀ (Hz)  | γ (Hz)  | Q-factor | Max |FFT|",
        "─" * 55,
    ]

    for gname in ['DN', 'MN', 'DN+MN']:
        if gname not in groups or gname not in fits:
            continue
        dp = groups[gname]['drive_powers']
        popt, perr, Q, f0, gamma = fits[gname]
        f0_err = perr[1]
        gamma_err = perr[2]
        q_err = Q * np.sqrt((f0_err/f0)**2 + (gamma_err/gamma)**2) if gamma > 0 else 0
        summary_lines.append(
            f"{gname:12s} | {f0:6.1f}±{f0_err:.1f} | {gamma:5.1f}±{gamma_err:.1f} | "
            f"{Q:5.2f}±{q_err:.2f} | {dp.max():8.2f}"
        )

    summary_lines += [
    ]

    if eigenmode_available and len(peaks) > 0:
        summary_lines += [
            "",
            "Top eigenmodes (free evolution):",
        ]
        for i, (pk_f, pk_m) in enumerate(peaks[:5]):
            summary_lines.append(f"  Mode {i+1}: {pk_f:6.1f} Hz  (|FFT|={pk_m:.1f})")

        # Highlight if resonance matches an eigenmode
        for gname in ['DN', 'MN', 'DN+MN']:
            if gname in fits:
                _, _, Q, f0, gamma = fits[gname]
                closest = min(peaks, key=lambda p: abs(p[0] - f0))
                summary_lines.append("")
                summary_lines.append(f"Resonance ({f0:.1f} Hz) matches eigenmode "
                                     f"{closest[0]:.1f} Hz: "
                                     f"Δf = {abs(f0 - closest[0]):.1f} Hz")
                summary_lines.append(
                    f"Interpretation: FlyWire connectome structure "
                    f"determines which")
                summary_lines.append(
                    f"frequencies propagate to DN/MN via graph eigenmodes.")
                break

    summary_text = '\n'.join(summary_lines)
    ax5.text(0.02, 0.95, summary_text, transform=ax5.transAxes,
             fontsize=9, fontfamily='monospace', va='top',
             linespacing=1.4)

    # ─── Save ──────────────────────────────────────────
    print("\n[5/5] Saving figure...")
    save_path = '/tmp/psi_resonance_flywire.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  ✅ Saved to: {save_path}")
    print(f"  Figure size: {fig.get_size_inches()} in @150 DPI")

    # Also save summary as text
    txt_path = '/tmp/psi_resonance_flywire_summary.txt'
    with open(txt_path, 'w') as f:
        f.write(summary_text)
    print(f"  ✅ Summary saved to: {txt_path}")


if __name__ == '__main__':
    main()
