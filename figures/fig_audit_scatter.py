"""Audit scatter — the certificate reading against what the attacker gets.

Reads results/audit_scatter_rows.csv (built by
experiments/build_audit_scatter_rows.py). No values are embedded here.

X = the certificate AS ISSUED AT REST, which is the audit's inclusion
instrument: a configuration enters the audit iff this reads R^2 <= tau = 0.05.
Every plotted point therefore satisfies x <= 0.05 and tau sits at the right
EDGE as a boundary, not as a divider — there are no quadrants to read.

Two omissions, both because the values are not stored and neither is imputed:
  * 11 of the 67 approved configurations have no stored at-rest reading
    (10 adult/sex projections + 1 hmda/race noise config); only their
    LoRA-ATTACKED R^2 is stored, which is a different instrument.
  * the HMDA contrast projections (certificate 0.166 / 0.171, above tau, so
    they never entered the audit) have no stored Tier-1 attacker AUC in any
    results file — results/smart_erasure.json records attacked_r2, task_lift
    and stops_attack only. They are omitted rather than plotted.
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _style  # noqa: E402
from _style import DARK  # noqa: E402

REPO = HERE.parent
CSV = REPO / "results" / "audit_scatter_rows.csv"

PROJ = "#D55E00"
NOISE = "#0072B2"
BAR, TAU = 0.55, 0.05

_style.apply()
import matplotlib.pyplot as plt  # noqa: E402

rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        if r["cert_r2_at_rest"] == "":
            continue
        rows.append(dict(mech=r["mechanism"],
                         x=float(r["cert_r2_at_rest"]),
                         y=float(r["tier1_max"]),
                         cell=r["cell"], method=r["method"]))

assert rows, "no rows with a stored at-rest certificate"
assert max(r["x"] for r in rows) <= TAU + 1e-12, "a point exceeds tau"

fig, ax = plt.subplots(figsize=(3.15, 2.15))

for mech, color in (("projection", PROJ), ("noise", NOISE)):
    pts = [r for r in rows if r["mech"] == mech]
    ax.scatter([p["x"] for p in pts], [p["y"] for p in pts], s=13,
               facecolor=color, edgecolor="none", alpha=0.85, zorder=3)

# the two arms are far apart in y, so label them in place rather than with a
# legend box that would have to sit on top of one of them
ax.text(0.0035, 0.915, "targeted projection", color=PROJ, fontsize=8,
        ha="left", va="top")
ax.text(0.0035, 0.655, "noise channel", color=NOISE, fontsize=8,
        ha="left", va="bottom")

ax.axhline(BAR, color=DARK, ls="--", lw=0.7, zorder=2)
ax.text(0.0015, BAR + 0.012, "0.55 bar", ha="left", va="bottom", fontsize=8,
        color=DARK)

n_above = sum(1 for r in rows if r["y"] > BAR)
# the two mechanisms leave an empty band between them (projections >= 0.92,
# noise <= 0.69), so the annotation sits there. It counts the PLOTTED points
# only; the 59-of-67 audit total and the coverage gap belong in the caption.
ax.text(0.0485, 0.845, f"{n_above} of the {len(rows)} shown above the bar",
        ha="right", va="top", fontsize=8, color=DARK)

ax.set_xlim(0, TAU)
ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
ax.set_xticklabels(["0", ".01", ".02", ".03", ".04", r"$\tau$=.05"])
ax.set_xlabel("certificate reading, non-adaptive (linear $R^2$)")

ax.set_ylim(0.48, 1.02)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels([".50", ".60", ".70", ".80", ".90", "1.0"])
ax.set_ylabel("best Tier-1 attacker\n(AUC; 0.5 = chance)")

ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)
ax.grid(axis="y")

_style.save(fig, HERE / "fig_audit_scatter.pdf")

# ---- reported diagnostics ------------------------------------------------
import statistics  # noqa: E402

xs = [r["x"] for r in rows]
ys = [r["y"] for r in rows]
n = len(rows)
mx, my = statistics.fmean(xs), statistics.fmean(ys)
sx = statistics.pstdev(xs) or 1e-12
sy = statistics.pstdev(ys) or 1e-12
pear = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (n * sx * sy)
rank = lambda v: [sorted(v).index(t) + 1 for t in v]  # noqa: E731
rxs, rys = rank(xs), rank(ys)
mrx, mry = statistics.fmean(rxs), statistics.fmean(rys)
srx = statistics.pstdev(rxs) or 1e-12
sry = statistics.pstdev(rys) or 1e-12
spear = sum((a - mrx) * (b - mry) for a, b in zip(rxs, rys)) / (n * srx * sry)

print(f"  plotted {n} of 67 (11 have no stored at-rest reading)")
print(f"  of the plotted, {n_above} are above the 0.55 bar")
print(f"  x range {min(xs):.4f}..{max(xs):.4f}   y range {min(ys):.4f}..{max(ys):.4f}")
print(f"  spread along x: Pearson r(cert, attacker AUC) = {pear:+.4f}, "
      f"Spearman rho = {spear:+.4f}")
for mech in ("projection", "noise"):
    p = [r for r in rows if r["mech"] == mech]
    print(f"    {mech:<11} n={len(p):<3} y median {statistics.median(r['y'] for r in p):.3f} "
          f"y range {min(r['y'] for r in p):.3f}..{max(r['y'] for r in p):.3f}")
