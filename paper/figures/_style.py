"""Shared style for the four paper figures (AAAI 2027, two-column).

Serif (Times New Roman, matching newtx/Times body font), >= 8pt at final
size, Okabe-Ito colorblind-safe palette (validated with the six-check
palette validator), single-column width 3.3in, vector PDF with embedded
TrueType (fonttype 42), no figure titles, tight bounding boxes.

Color roles, consistent across all four figures:
  BLUE   #0072B2  our mechanism / noise-channel results (Tier 1 = solid)
  LBLUE  #56B4E9  our mechanism at Tier 2 (lighter + hatch = tier encoding)
  VERM   #D55E00  probe-shaped baselines (projections) / loose-bound accents
  ORANGE #E69F00  CelebA vision cells (distinct dataset accent)
  GRAY   #767676  the weak certificate instrument (recessive; own panel,
                  identity never carried by this color alone)
  DARK   #333333  reference/unprotected rows, diagonals, text-adjacent marks
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#0072B2"
LBLUE = "#56B4E9"
VERM = "#D55E00"
ORANGE = "#E69F00"
GRAY = "#767676"
DARK = "#333333"

COL_W = 3.3          # AAAI single-column width in inches


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "Times"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.edgecolor": DARK,
        "axes.labelcolor": "black",
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig, path):
    fig.savefig(path, format="pdf")
    print(f"[save] {path}")
