"""Figure 3 — THE COST RAMP: continuous durable-removal cost vs the
label-coupling predictor.

Sources (the spec named results/diagnostic.json; the Exp-9 continuous-cost
data actually lives in):
  - results/continuous_cost.json (Experiment 9 Goal 1): 20 natural cells;
    cost_durable = fraction of clean task lift lost at the noise level where
    the black-box battery (XGB+MLP) first reads the attribute <= 0.55 on
    BOTH representation and output (durable removal). Pearson r = 0.795,
    Spearman rho = 0.848 (stats block; the "r ~ 0.8" headline).
  - results/cliff_or_ramp_constructed.json (Experiment 9 Goal 2): 9
    constructed cells (label<->attribute coupling tuned by construction),
    same cost definition — the causal version of the ramp.

The shaded band marks the usable middle of the predictor axis
(~0.52-0.60), where the ramp rises from ~free to ~everything.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _style  # noqa: E402
from _style import BLUE, DARK  # noqa: E402

REPO = HERE.parent.parent
RESULTS = REPO / "results"

_style.apply()
import matplotlib.pyplot as plt  # noqa: E402

nat = json.loads((RESULTS / "continuous_cost.json").read_text())
con = json.loads((RESULTS / "cliff_or_ramp_constructed.json").read_text())
r = nat["stats"]["cost_durable"]["pearson_r"]
rho = nat["stats"]["cost_durable"]["spearman_r"]
n = nat["stats"]["cost_durable"]["n"]

nx = [row["predictor"] for row in nat["rows"]]
ny = [row["cost_durable"] for row in nat["rows"]]
cx = [row["predictor"] for row in con["rows"]]
cy = [row["cost_durable"] for row in con["rows"]]

fig, ax = plt.subplots(figsize=(_style.COL_W, 2.6))

ax.axvspan(0.52, 0.60, color="#EEEEEE", zorder=0)
ax.text(0.56, 1.045, "the usable middle band", ha="center", va="bottom",
        fontsize=8, color=DARK)

ax.scatter(nx, ny, s=24, color=BLUE, zorder=3,
           label=f"natural cells (n={len(nx)})")
ax.scatter(cx, cy, s=30, facecolors="none", edgecolors=BLUE, marker="^",
           linewidths=0.9, zorder=3,
           label=f"constructed cells (n={len(cx)})")

# region labels (display only; r/rho moved to the caption)
ax.text(0.5055, 0.62, "removal nearly free", ha="center", va="center",
        fontsize=8, color=DARK, rotation=90)
ax.text(0.560, 0.505, "cost climbs steeply", ha="center", va="center",
        fontsize=8, color=DARK)
ax.text(0.648, 0.42, "near-total cost", ha="center", va="center",
        fontsize=8, color=DARK)

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

_style.save(fig, HERE / "fig3_cost_ramp.pdf")
print(f"natural r={r:.4f} rho={rho:.4f} n={n}")
for row in nat["rows"]:
    print(f"  N {row['cell']:<46} ({row['predictor']:.3f}, "
          f"{row['cost_durable']:.3f})")
for row in con["rows"]:
    print(f"  C {row['cell']:<46} ({row['predictor']:.3f}, "
          f"{row['cost_durable']:.3f})")
