"""Figure 3 v2 — the cost ramp WITH the 7 usable expansion cells (external
sources; the two degenerate diabetes-hospital cells are excluded because
clean lift is ~0 so the cost fraction is undefined).

Identical to fig3_cost_ramp.py (untouched) plus:
  - external cells from results/expansion_cost.json as OPEN CIRCLES in the
    vermillion accent color, legend entry "external cells";
  - the in-figure stats line updates to the combined natural correlation
    (20 existing + 7 external), recomputed here from the two JSONs.

Writes fig3_cost_ramp_v2.pdf — the original fig3_cost_ramp.pdf is NOT
overwritten. Prints every plotted point and the recomputed stats for
verification.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _style  # noqa: E402
from _style import BLUE, DARK, VERM  # noqa: E402

REPO = HERE.parent.parent
RESULTS = REPO / "results"

_style.apply()
import matplotlib.pyplot as plt  # noqa: E402

nat = json.loads((RESULTS / "continuous_cost.json").read_text())
con = json.loads((RESULTS / "cliff_or_ramp_constructed.json").read_text())
exp = json.loads((RESULTS / "expansion_cost.json").read_text())

nx = [row["predictor"] for row in nat["rows"]]
ny = [row["cost_durable"] for row in nat["rows"]]
cx = [row["predictor"] for row in con["rows"]]
cy = [row["cost_durable"] for row in con["rows"]]
ext_rows = [r for r in exp["rows"] if not r["degenerate"]]
ex = [r["predictor"] for r in ext_rows]
ey = [r["cost_durable"] for r in ext_rows]
ext_excluded = [r["cell"] for r in exp["rows"] if r["degenerate"]]

# combined natural correlation (existing 20 + external 7), recomputed live
ax_all = np.array(nx + ex)
ay_all = np.array(ny + ey)
r, _ = pearsonr(ax_all, ay_all)
rho, _ = spearmanr(ax_all, ay_all)
n = len(ax_all)

fig, ax = plt.subplots(figsize=(_style.COL_W, 2.6))

ax.axvspan(0.52, 0.60, color="#EEEEEE", zorder=0)
ax.text(0.56, 1.045, "the usable middle band", ha="center", va="bottom",
        fontsize=8, color=DARK)

ax.scatter(nx, ny, s=24, color=BLUE, zorder=3,
           label=f"original cells (n={len(nx)})")
ax.scatter(ex, ey, s=26, facecolors="none", edgecolors=VERM,
           linewidths=0.9, zorder=3, label=f"external cells (n={len(ex)})")
ax.scatter(cx, cy, s=30, facecolors="none", edgecolors=BLUE, marker="^",
           linewidths=0.9, zorder=3,
           label=f"constructed cells (n={len(cx)})")

ax.text(0.688, 0.02, f"$r={r:.3f}$, $\\rho={rho:.3f}$ (natural, $n={n}$)",
        ha="right", va="bottom", fontsize=8, color=DARK)

ax.set_xlim(0.495, 0.695)
ax.set_ylim(-0.04, 1.09)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("label-coupling predictor\n(XGB AUC, attribute from label alone)")
ax.set_ylabel("durable-removal cost\n(fraction of clean lift lost)")
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)
ax.grid(True)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.10),
          handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3)

_style.save(fig, HERE / "fig3_cost_ramp_v2.pdf")
print(f"combined natural r={r:.4f} rho={rho:.4f} n={n} "
      f"(existing {len(nx)} + external {len(ex)}; constructed {len(cx)} plotted, "
      f"not in stats)")
for row in nat["rows"]:
    print(f"  N {row['cell']:<46} ({row['predictor']:.3f}, "
          f"{row['cost_durable']:.3f})")
for row in ext_rows:
    print(f"  E {row['cell']:<46} ({row['predictor']:.4f}, "
          f"{row['cost_durable']:.4f})")
for row in con["rows"]:
    print(f"  C {row['cell']:<46} ({row['predictor']:.3f}, "
          f"{row['cost_durable']:.3f})")
print(f"external excluded (degenerate): {ext_excluded}")
