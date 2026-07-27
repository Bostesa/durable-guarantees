"""Figure 4 — THE TWO-TIER CERTIFICATION: utility kept at the Tier-1 vs
Tier-2 certified operating points of the e2e noised channel, per headline
cell.

Source: results/two_tier_certification.json (Experiment 14), e2e_cells:
utility kept = tier lift_mean / clean_lift, error bar = lift_std /
clean_lift (3 training seeds — Exp 14's certification protocol).

Flagged (not silently reconciled): the spec asked for 5-seed error bars,
but 5-seed replicates (Experiment 15, results/hardening_seeds.json) exist
for only ONE of the six bars — the easy cell's Tier-1 point (sigma=8:
5-seed lift 0.0211 +/- 0.0005 = 94.6% kept, identical to the 3-seed value
plotted here). The other five points were certified at 3 seeds only, so
all bars use the 3-seed spread for consistency.

Cell difficulty labels follow the repo's canonical assignment (README):
easy = hmda/loan_decision (predictor 0.514), middle = hmda/loan_amount_band
(0.584), hard = adult/sex/income (0.603). (The task brief's parenthetical
had these permuted.)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _style  # noqa: E402
from _style import BLUE, DARK, LBLUE  # noqa: E402

REPO = HERE.parent.parent
RESULTS = REPO / "results"

_style.apply()
import matplotlib.pyplot as plt  # noqa: E402

t = json.loads((RESULTS / "two_tier_certification.json").read_text())
ORDER = ["hmda/race/loan_decision", "hmda/race/loan_amount_band",
         "adult/sex/income"]
NAMES = ["hmda\nloan_decision\n(easy,\ncoupling 0.514)",
         "hmda\nloan_amount_band\n(middle,\ncoupling 0.584)",
         "adult\nincome\n(hard,\ncoupling 0.603)"]
cells = {c["cell"]: c for c in t["e2e_cells"]}

kept, err, ops = {"tier1": [], "tier2": []}, {"tier1": [], "tier2": []}, \
    {"tier1": [], "tier2": []}
for cell in ORDER:
    c = cells[cell]
    for k in ("tier1", "tier2"):
        d = c[k]
        kept[k].append(100 * d["lift_mean"] / c["clean_lift"])
        err[k].append(100 * d["lift_std"] / c["clean_lift"])
        ops[k].append(f"$\\sigma$={d['sigma']:g}")

x = np.arange(len(ORDER))
w = 0.36
fig, ax = plt.subplots(figsize=(_style.COL_W, 2.0))

b1 = ax.bar(x - w / 2, kept["tier1"], w, yerr=err["tier1"], capsize=2,
            color=BLUE, edgecolor="none", error_kw=dict(lw=0.8),
            label="Tier 1 (black-box) surviving point")
b2 = ax.bar(x + w / 2, kept["tier2"], w, yerr=err["tier2"], capsize=2,
            color=LBLUE, edgecolor=DARK, linewidth=0.4, hatch="///",
            error_kw=dict(lw=0.8),
            label="Tier 2 (+informed LRT) surviving point")

for bars, k in ((b1, "tier1"), (b2, "tier2")):
    for rect, v, e, op in zip(bars, kept[k], err[k], ops[k]):
        ytxt = max(v, 0) + e + 3
        # one decimal on every bar so the figure matches the text (94.6/56.6)
        lab = f"{v:.1f}%"
        ax.text(rect.get_x() + rect.get_width() / 2, ytxt,
                lab, ha="center", va="bottom", fontsize=8)
        ax.text(rect.get_x() + rect.get_width() / 2, -28, op, ha="center",
                va="bottom", fontsize=8, color=DARK)

ax.axhline(0, color=DARK, lw=0.6)

# plain-language annotations (display only)
ax.text(2, -13.5, "infeasible", ha="center", va="center", fontsize=8,
        color=DARK)
ax.text(1.5, 90, "attacker who knows the\ndefense costs more", ha="center",
        va="top", fontsize=8, color=DARK)
ax.set_xticks(x, NAMES)
ax.tick_params(axis="x", length=0, pad=2)
ax.set_ylabel("utility kept at surviving point\n(% of clean e2e lift)")
ax.set_ylim(-31, 118)
ax.set_yticks([0, 25, 50, 75, 100])
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)
ax.grid(axis="y")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.06),
          handlelength=1.2, handleheight=0.9, labelspacing=0.3)

_style.save(fig, HERE / "fig4_two_tier.pdf")
for i, cell in enumerate(ORDER):
    print(f"  {cell:<30} T1 {kept['tier1'][i]:+.1f}%±{err['tier1'][i]:.1f} "
          f"({ops['tier1'][i]})  T2 {kept['tier2'][i]:+.1f}%±"
          f"{err['tier2'][i]:.1f} ({ops['tier2'][i]})")
