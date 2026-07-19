"""Figure 2 v2 — the footprint-law diagonal WITH the 7 usable expansion
cells (external sources: folktables/ACS, LSAC Law School, Dutch Census;
the two degenerate diabetes-hospital cells are excluded because the task
is not learned there, so the floor has no upper-bound reading).

Identical to fig2_footprint.py (untouched) plus:
  - external cells from results/expansion_floor.json as OPEN CIRCLES in
    the vermillion accent color, legend entry "external cells";
  - dutch/sex/occupation annotated as a loose-upper-bound point in the
    same style as the Attractive->Young annotation (whose text block moves
    up slightly to make room).

Writes fig2_footprint_v2.pdf — the original fig2_footprint.pdf is NOT
overwritten. Prints every plotted point for verification.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _style  # noqa: E402
from _style import DARK, GRAY, ORANGE, VERM  # noqa: E402

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

exp = json.loads((RESULTS / "expansion_floor.json").read_text())
ext = [(r["predictor"], r["floor"], r["cell"])
       for r in exp["rows"] if not r["degenerate"]]
ext_excluded = [r["cell"] for r in exp["rows"] if r["degenerate"]]

fig, ax = plt.subplots(figsize=(3.45, 2.95))

lims = (0.48, 0.78)
ax.plot(lims, lims, ls="--", lw=0.8, color=GRAY, zorder=1)
diag_label = ax.text(0.735, 0.748, "$y=x$", ha="right", va="bottom",
                     fontsize=8, color=DARK, rotation=45,
                     rotation_mode="anchor")

xs, ys = [p[0] for p in tab], [p[1] for p in tab]
ax.scatter(xs, ys, s=26, color=DARK, zorder=3, label="tabular cells")

for p in cel_pts:
    ax.errorbar(p["pred"], p["point"],
                yerr=[[p["point"] - p["lo"]], [p["hi"] - p["point"]]],
                fmt="D", ms=5.5, color=ORANGE, ecolor=ORANGE,
                elinewidth=0.9, capsize=2, zorder=4,
                markeredgecolor=DARK, markeredgewidth=0.4)
ax.scatter([], [], marker="D", s=30, color=ORANGE, edgecolor=DARK,
           linewidth=0.4, label="CelebA cells")

ex, ey = [p[0] for p in ext], [p[1] for p in ext]
ax.scatter(ex, ey, s=26, facecolors="none", edgecolors=VERM,
           linewidths=0.9, zorder=3, label="external cells")

# annotations: the two CelebA points; the high one moves up slightly so the
# dutch external point (0.650, 0.576) and its own label have room
low, high = sorted(cel_pts, key=lambda p: p["pred"])
ax.annotate("Smiling$\\to$Young", (low["pred"], low["point"]),
            xytext=(low["pred"] + 0.012, low["point"] - 0.020), fontsize=8)
ax.annotate("Attractive$\\to$Young:\nlabel predictor is a\nloose upper bound",
            (high["pred"] - 0.006, high["point"] + 0.004),
            xytext=(high["pred"] - 0.005, high["point"] + 0.088),
            ha="right", fontsize=8,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=DARK))

# dutch/sex/occupation: the external loose-bound point, same style
dutch = next(p for p in ext if p[2] == "dutch/sex/occupation")
ax.annotate("Dutch/sex: predictor is\na loose upper bound",
            (dutch[0], dutch[1] - 0.006),
            xytext=(dutch[0] - 0.043, dutch[1] - 0.072),
            ha="left", fontsize=8, color=VERM,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=VERM))

ax.set_xlim(*lims)
ax.set_ylim(*lims)
ax.set_xlabel("label-coupling predictor\n(XGB AUC, attribute from label alone)")
ax.set_ylabel("output-leak floor  (best-attacker AUC on outputs)")
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)
ax.grid(True)
ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2,
          labelspacing=0.3)

import numpy as np  # noqa: E402

fig.canvas.draw()
p0 = ax.transData.transform((lims[0], lims[0]))
p1 = ax.transData.transform((lims[1], lims[1]))
diag_label.set_rotation(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])))

_style.save(fig, HERE / "fig2_footprint_v2.pdf")
print("tabular (predictor, floor):")
for x, yv, c in tab:
    print(f"  {c:<46} ({x:.3f}, {yv:.3f})")
for p in cel_pts:
    print(f"  {p['name']:<46} ({p['pred']:.3f}, {p['point']:.3f}) "
          f"arm-range [{p['lo']:.3f}, {p['hi']:.3f}]")
print("external (predictor, floor):")
for x, yv, c in ext:
    print(f"  E {c:<44} ({x:.4f}, {yv:.4f})")
print(f"external excluded (degenerate): {ext_excluded}")
