"""Figure 1 — THE DEMOLITION: the linear certificate vs the honest probe.

Source: results/honest_reaudit.json (Experiment 5), adult/sex cell.
  - part2_smart_erasers.adult: every R^2-"stopped" eraser re-measured with
    the honest battery (certificate R^2 + XGBoost AUC per method).
  - master "E2 noise" rows: the unprotected reference (sigma=0) and the
    noise sigma=4 point (part2 lists sigma=1/2/8 only).

Cell choice (flagged, not silent): the spec suggested HMDA/race, but the
certificate-passes/honest-fails contrast only EXISTS on adult/sex — on HMDA
the projections never passed the certificate (R^2 0.17-0.65 > tau), so they
were never re-audited and have no honest-XGB reading in any results JSON.
On adult/sex the certificate reads ~0.036 (PASS, tau=0.05) for every
projection while XGBoost reads 0.68 = the unprotected level; only noise
actually reduces honest recovery. LEACE is omitted (also flagged): it was
never R^2-stopped on this representation, so Exp 5 recorded no honest-XGB
value for it. LEOPARD-MMD r=32 does not exist in the re-audit (r <= 16).

Left panel: the certificate's reading (linear R^2, tau line) — passes
everything erased. Right panel: honest XGBoost AUC (chance 0.5, bar 0.55)
— everything except noise sits at the unprotected level.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _style  # noqa: E402
from _style import BLUE, DARK, GRAY, VERM  # noqa: E402

REPO = HERE.parent.parent
RESULTS = REPO / "results"

_style.apply()
import matplotlib.pyplot as plt  # noqa: E402

d = json.loads((RESULTS / "honest_reaudit.json").read_text())
TAU = d["tau"]
BAR = d["honest_auc_bar"]

by_method = {r["method"]: r for r in d["part2_smart_erasers"]["adult"]}
master_adult_noise = {r["method"]: r for r in d["master"]
                      if r["experiment"] == "E2 noise"}

# (display label, source row, family)  — family drives the honest-bar color
# display labels kept short for column width; the caption expands
# "MMD" = LEOPARD-style class-conditional-MMD projection
ROWS = [
    ("no erasure",  master_adult_noise["noise σ=0.0"], "ref"),
    ("MMD proj r=8",   by_method["LEOPARD-MMD proj r=8"], "proj"),
    ("MMD proj r=16",  by_method["LEOPARD-MMD proj r=16"], "proj"),
    ("HSIC proj r=8",  by_method["HSIC proj r=8"], "proj"),
    ("HSIC proj r=16", by_method["HSIC proj r=16"], "proj"),
    ("HSIC proj r=32", by_method["HSIC proj r=32"], "proj"),
    ("noise σ=1",   by_method["noise sigma=1.0"], "noise"),
    ("noise σ=2",   by_method["noise sigma=2.0"], "noise"),
    ("noise σ=4",   master_adult_noise["noise σ=4.0"], "noise"),
    ("noise σ=8",   by_method["noise sigma=8.0"], "noise"),
]
FAMILY_COLOR = {"ref": DARK, "proj": VERM, "noise": BLUE}

labels = [r[0] for r in ROWS]
r2 = [r[1]["old_r2"] for r in ROWS]
xgb = [r[1]["xgb_auc"] for r in ROWS]
colors = [FAMILY_COLOR[r[2]] for r in ROWS]
y = list(range(len(ROWS)))[::-1]

# figsize is smaller than the 3.3in column because the tight crop adds the
# method labels and top legend; the cropped PDF lands at ~3.3in wide
fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(2.95, 2.16), sharey=True,
    gridspec_kw=dict(width_ratios=[1, 1.3], wspace=0.14))

axL.barh(y, r2, height=0.62, color=GRAY, edgecolor="none")
axL.axvline(TAU, color=DARK, ls="--", lw=0.7)
axL.text(TAU + 0.004, -0.62, r"$\tau=0.05$", ha="left", va="bottom",
         fontsize=8)
axL.set_yticks(y, labels)
axL.set_xlim(0, 0.12)
axL.set_xticks([0, 0.05, 0.10])
axL.set_xticklabels(["0", ".05", ".10"])
axL.set_xlabel("certificate reading\n(linear $R^2$)")
axL.tick_params(axis="y", length=0, pad=2)

axR.barh(y, [v - 0.5 for v in xgb], left=0.5, height=0.62, color=colors,
         edgecolor="none")
axR.axvline(0.55, color=DARK, ls="--", lw=0.7)
axR.text(0.555, -0.62, "0.55 bar", ha="left", va="bottom", fontsize=8)
axR.set_xlim(0.5, 0.73)
axR.set_xticks([0.5, 0.55, 0.6, 0.65, 0.7])
axR.set_xticklabels([".50", ".55", ".60", ".65", ".70"])
axR.set_xlabel("attacker recovery\n(XGBoost AUC; 0.5 = chance)")

# plain-language annotations (display only)
axL.text(0.116, 0.9, "every method\npasses the check", ha="right", va="center",
         fontsize=8, color=DARK)
axR.text(0.702, 6.5, "no removal reads 0.683", ha="center", va="center",
         fontsize=8, color=DARK, rotation=90)
axR.text(0.726, 1.6, "only noise drops\ntoward chance", ha="right",
         va="center", fontsize=8, color=DARK)

for ax in (axL, axR):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    ax.set_ylim(-0.7, len(ROWS) - 0.4)

# identity is carried by the y labels; the family legend is a reading aid
handles = [plt.Rectangle((0, 0), 1, 1, color=c)
           for c in (DARK, VERM, BLUE)]
fig.legend(handles, ["unprotected", "targeted projection", "noise channel"],
           loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.045),
           handlelength=1.0, handleheight=0.8, columnspacing=0.9,
           handletextpad=0.4)

_style.save(fig, HERE / "fig1_demolition.pdf")
for lab, a, b in zip(labels, r2, xgb):
    print(f"  {lab:<18} R2={a:.3f}  XGB={b:.3f}")
