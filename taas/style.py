"""taas Brand Identity — Neuralink-inspired visual style.

Цвета, шрифты, эффекты свечения, водяной знак логотипа.
Все визуализации используют этот модуль для единого стиля.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Цветовая палитра (Neuralink + tech-neon) ──────────

# Основные
BRAND_WHITE = "#fefefe"       # из логотипа
BRAND_DARK = "#0a0a12"         # глубокий тёмный фон
BRAND_DARK2 = "#111122"        # карточки/панели
BRAND_DARK3 = "#1a1a2e"        # границы

# Неоновые акценты
NEON_GREEN = "#00ff88"         # primary — спайки, активность
NEON_CYAN = "#00ccff"          # secondary — информация, связи
NEON_PURPLE = "#8844ff"        # tertiary — спектр, coherence
NEON_PINK = "#ff44cc"          # accent — gamma, высокая частота
NEON_ORANGE = "#ff8800"        # accent — theta, firing rate
NEON_BLUE = "#3399ff"          # accent — beta

# Нюансы для заливок
GLOW_GREEN = "#00ff8822"
GLOW_CYAN = "#00ccff22"
GLOW_PURPLE = "#8844ff22"
GLOW_ORANGE = "#ff880022"
GLOW_BLUE = "#3399ff22"
GLOW_PINK = "#ff44cc22"

# Для градиентных карт
CMAP_SPIKE = "Greens"          # для raster плотности
CMAP_PHASE = "hsv"             # для фазовых углов
CMAP_COHERENCE = "plasma"      # для coherence карт
CMAP_SPECTRUM = "magma"        # для спектрограмм

# ─── Matplotlib style params ────────────────────────────

MPL_PARAMS = {
    # Figure
    "figure.facecolor": BRAND_DARK,
    "figure.edgecolor": BRAND_DARK,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.facecolor": BRAND_DARK,
    "savefig.edgecolor": BRAND_DARK,
    "savefig.transparent": False,

    # Axes
    "axes.facecolor": BRAND_DARK,
    "axes.edgecolor": BRAND_DARK3,
    "axes.labelcolor": BRAND_WHITE,
    "axes.titlecolor": BRAND_WHITE,
    "axes.grid": True,
    "axes.grid.axis": "both",
    "axes.grid.which": "major",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.linewidth": 0.5,
    "axes.prop_cycle": mpl.cycler(color=[
        NEON_GREEN, NEON_CYAN, NEON_PURPLE, NEON_PINK,
        NEON_ORANGE, NEON_BLUE,
    ]),

    # Grid
    "grid.color": BRAND_DARK3,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.3,
    "grid.linestyle": "-",

    # Text
    "text.color": BRAND_WHITE,
    "font.family": "sans-serif",
    "font.sans-serif": ["SF Pro Display", "Inter", "system-ui", "sans-serif"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    # Ticks
    "xtick.color": "#666688",
    "ytick.color": "#666688",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,

    # Legend
    "legend.facecolor": BRAND_DARK2,
    "legend.edgecolor": BRAND_DARK3,
    "legend.fontsize": 9,
    "legend.framealpha": 0.8,
    "legend.labelcolor": BRAND_WHITE,

    # Lines
    "lines.linewidth": 1.0,
    "lines.antialiased": True,
    "patch.antialiased": True,
    "patch.facecolor": BRAND_DARK2,
    "patch.edgecolor": BRAND_DARK3,

    # Image
    "image.cmap": CMAP_SPECTRUM,
    "image.interpolation": "bilinear",
    "image.aspect": "auto",
}

# Path to logo
LOGO_PATH = Path(__file__).parent.parent / "logo.png"


# ─── Functions ──────────────────────────────────────────

def apply_style():
    """Apply Neuralink brand style to matplotlib."""
    mpl.rcParams.update(MPL_PARAMS)


def apply_ticks(ax, color="#666688"):
    """Style ticks for an axis."""
    ax.tick_params(colors=color, labelsize=9)


def add_logo(fig, alpha=0.15, scale=0.12, position="lower left"):
    """Add logo watermark to figure."""
    if not LOGO_PATH.exists():
        return

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image

    try:
        img = Image.open(LOGO_PATH)
        arr = np.array(img.convert("RGBA"))

        # Make semi-transparent
        arr = arr.astype(np.float32)
        arr[:, :, 3] *= alpha
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Create OffsetImage
        imgbox = OffsetImage(arr, zoom=scale)
        imgbox.image.axes = fig.axes[0] if fig.axes else None

        # Position
        positions = {
            "lower left": (0.02, 0.02),
            "lower right": (0.98, 0.02),
            "upper left": (0.02, 0.98),
            "upper right": (0.98, 0.98),
        }
        xy = positions.get(position, (0.98, 0.02))

        ab = AnnotationBbox(imgbox, xy, xycoords="figure fraction",
                            frameon=False, box_alignment=(1, 0) if "right" in position else (0, 0))
        fig.axes[0].add_artist(ab) if fig.axes else fig.add_artist(ab)
    except Exception:
        pass  # Silently fail if logo can't be added


def glow_line(ax, x, y, color=NEON_GREEN, linewidth=1.0, alpha=0.8,
              n_glow=3, glow_alpha_range=(0.05, 0.2), glow_width_mult=2.0):
    """Draw a line with glow effect (multiple overlaid lines with decreasing alpha/width)."""
    lines = []
    for i in range(n_glow):
        glow_alpha = glow_alpha_range[0] + (glow_alpha_range[1] - glow_alpha_range[0]) * (1 - i / n_glow)
        glow_width = linewidth * (1 + glow_width_mult * (1 - i / n_glow))
        l = ax.plot(x, y, color=color, alpha=glow_alpha,
                     linewidth=glow_width, antialiased=True)[0]
        lines.append(l)
    # Main line on top
    l = ax.plot(x, y, color=color, alpha=alpha,
                 linewidth=linewidth, antialiased=True, zorder=5)[0]
    lines.append(l)
    return lines


def glow_scatter(ax, x, y, color=NEON_GREEN, s=2, alpha=0.6, glow_alpha=0.15):
    """Scatter plot with glow effect."""
    ax.scatter(x, y, s=s * 3, c=color, alpha=glow_alpha, edgecolors="none")
    return ax.scatter(x, y, s=s, c=color, alpha=alpha, edgecolors="none", zorder=5)


def style_axis(ax, title=None, xlabel=None, ylabel=None,
               grid=True, remove_spines=True):
    """Apply brand style to an axis."""
    if title:
        ax.set_title(title, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if remove_spines:
        for sp in ["top", "right"]:
            try:
                ax.spines[sp].set_visible(False)
            except KeyError:
                pass

    if grid:
        ax.grid(True, alpha=0.15, color=BRAND_DARK3, linewidth=0.3)

    apply_ticks(ax)


def get_brand_cmap(name="spike"):
    """Get branded colormap."""
    cmaps = {
        "spike": plt.cm.Greens,
        "phase": plt.cm.hsv,
        "coherence": plt.cm.plasma,
        "spectrum": plt.cm.magma,
        "amplitude": plt.cm.viridis,
        "diverging": plt.cm.RdBu_r,
    }
    return cmaps.get(name, plt.cm.plasma)


# ─── Apply on import ────────────────────────────────────
apply_style()
