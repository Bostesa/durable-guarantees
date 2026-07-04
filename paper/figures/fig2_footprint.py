"""Figure 2 — THE FOOTPRINT LAW DIAGONAL: output-side leakage vs the
label-coupling predictor, with y = x.

Sources:
  - results/diagnostic.json (Experiment 8 Part C): the six tabular cells'
    (predictor, out_floor) — out_floor = best-attacker AUC on the task
    output once the representation is hidden. (The spec's
    "footprint_law*.json" does not exist; this is where the law's data
    lives.)
  - results/celeba_pipeline.json (Experiment 13): the two CelebA cells.
    Point = the certified deployment arm's floor (out_floor_best_arm);
    whisker = the min-max range of floors across the four arms (the README
    quotes this range, 0.505-0.515 low / 0.526-0.553 high).

Flagged omission: the spec asked for an arrow to the high cell's
ACHIEVED-OUTPUT coupling position (0.536 clean / 0.540 winner, README).
Those two numbers are quoted in the README but stored in NO results JSON
(verified by key search of celeba_pipeline.json), so per the no-approximation
rule the arrow is omitted; the point is annotated as the loose-upper-bound
outlier instead and the reconciliation stays in the caption.

Known scope note (README Part C): the two adult tabular cells' floors are
upper-ish estimates (rep only partly scrubbed within sigma <= 12).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _style  # noqa: E402
from _style import DARK, GRAY, ORANGE  # noqa: E402

REPO = HERE.parent.parent
RESULTS = REPO / "results"

_style.apply()
import matplotlib.pyplot as plt  # noqa: E402

diag = json.loads((RESULTS / "diagnostic.json").read_text())
tab = [(r["predictor"], r["out_floor"], r["cell"])
       for r in diag["part_b_c"]]

cel = json.loads((RESULTS / "celeba_pipeline.json").read_text())
cel_pts = []
for c in cel["cells"]:
    v = c["verdicts"]
    floors = list(v["out_floor_by_arm"].values())
    cel_pts.append(dict(pred=v["predictor"], point=v["out_floor_best_arm"],
                        lo=min(floors), hi=max(floors),
                        name=c["cfg_key"]))

# equal data ranges on both axes (the diagonal matters); figsize tuned so
# the tight-cropped PDF lands at ~3.3in column width
fig, ax = plt.subplots(figsize=(3.45, 2.95))

lims = (0.48, 0.78)
ax.plot(lims, lims, ls="--", lw=0.8, color=GRAY, zorder=1)
diag_label = ax.text(0.735, 0.748, "$y=x$", ha="right", va="bottom",
                     fontsize=8, color=DARK, rotation=45,
                     rotation_mode="anchor")

xs, ys = [p[0] for p in tab], [p[1] for p in tab]
ax.scatter(xs, ys, s=26, color=DARK, zorder=3, label="tabular cells (Exp 8)")

for p in cel_pts:
    ax.errorbar(p["pred"], p["point"],
                yerr=[[p["point"] - p["lo"]], [p["hi"] - p["point"]]],
                fmt="D", ms=5.5, color=ORANGE, ecolor=ORANGE,
                elinewidth=0.9, capsize=2, zorder=4,
                markeredgecolor=DARK, markeredgewidth=0.4)
ax.scatter([], [], marker="D", s=30, color=ORANGE, edgecolor=DARK,
           linewidth=0.4, label="CelebA cells (Exp 13)")

# annotations: the two CelebA points, the high one as the loose-bound outlier
low, high = sorted(cel_pts, key=lambda p: p["pred"])
ax.annotate("Smiling$\\to$Young", (low["pred"], low["point"]),
            xytext=(low["pred"] + 0.012, low["point"] - 0.020), fontsize=8)
ax.annotate("Attractive$\\to$Young:\nlabel predictor is a\nloose upper bound",
            (high["pred"] - 0.006, high["point"] + 0.004),
            xytext=(high["pred"] - 0.045, high["point"] + 0.048),
            ha="right", fontsize=8,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=DARK))

ax.set_xlim(*lims)
ax.set_ylim(*lims)
ax.set_xlabel("label-coupling predictor\n(XGB AUC, attribute from label alone)")
ax.set_ylabel("output-leak floor  (best-attacker AUC on outputs)")
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)
ax.grid(True)
ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)

# match the y=x label's rotation to the diagonal's on-screen angle
import numpy as np  # noqa: E402

fig.canvas.draw()
p0 = ax.transData.transform((lims[0], lims[0]))
p1 = ax.transData.transform((lims[1], lims[1]))
diag_label.set_rotation(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))

_style.save(fig, HERE / "fig2_footprint.pdf")
print("tabular (predictor, floor):")
for x, yv, c in tab:
    print(f"  {c:<46} ({x:.3f}, {yv:.3f})")
for p in cel_pts:
    print(f"  {p['name']:<46} ({p['pred']:.3f}, {p['point']:.3f}) "
          f"arm-range [{p['lo']:.3f}, {p['hi']:.3f}]")
