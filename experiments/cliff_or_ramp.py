"""Experiment 9 — GOAL 2: is the cheap-vs-impossible split a CLIFF or a RAMP?

Exp 8 saw cheap durable removal below predictor ~0.55 and impossibility above —
but with only 6 cells, the middle was barely sampled. Goal 1 (continuous_cost.py)
already measures 20 NATURAL cells, 9 of them in the 0.55-0.63 zone. This script
adds CONSTRUCTED cells: within-cell families where the SAME dataset/attr/task is
subsampled to tune the label<->attribute correlation to intermediate values, so
the predictor axis is swept densely while everything else is held fixed. These
are clearly labelled constructed, not natural.

Construction: stratified subsampling to the joint  q(a,y) = (1-t)*p(a,y)
+ t*p(a)*p(y),  t in [0,1]  (t=1 -> label independent of attribute -> predictor
0.5; t=0 -> natural). The largest feasible subsample with that joint is drawn
without replacement per (a,y) stratum. The predictor is re-measured live on each
constructed cell; the removal cost is measured exactly as in Goal 1
(measure_cell: from-raw-x noised channel, XGB+MLP battery, continuous
interpolated cost).

--merge combines constructed rows with the Goal-1 natural rows, plots cost over
the full predictor range, and states CLIFF vs RAMP. Local MPS/CPU; never AWS.
macOS OpenMP guard: xgboost before torch.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401  (import before torch on purpose)

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.continuous_cost import measure_cell  # noqa: E402
from experiments.diagnostic import load_cell  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"

# (dataset, attr, task, [t values]) — t=0 is the natural cell (measured in Goal 1).
FAMILIES = [
    ("adult", "sex", "income", [0.25, 0.5, 0.75]),            # natural ~0.603 -> ~0.52
    ("adult", "marital_status", "income", [0.2, 0.4, 0.6, 0.8]),  # ~0.688 -> ~0.54
    ("hmda", "ethnicity", "loan_amount_band", [0.35, 0.7]),   # ~0.624 -> ~0.54
]


def decorrelate_subsample(X, attr, task, t, seed=0):
    """Largest subsample whose (attr, task) joint is (1-t)*p(a,y) + t*p(a)p(y)."""
    rng = np.random.default_rng(seed)
    n = len(attr)
    avals, yvals = np.unique(attr), np.unique(task)
    p_a = {a: (attr == a).mean() for a in avals}
    p_y = {y: (task == y).mean() for y in yvals}
    q, counts = {}, {}
    for a in avals:
        for y in yvals:
            counts[(a, y)] = int(((attr == a) & (task == y)).sum())
            p_ay = counts[(a, y)] / n
            q[(a, y)] = (1 - t) * p_ay + t * p_a[a] * p_y[y]
    n_max = int(min(counts[k] / q[k] for k in q if q[k] > 0))
    idx = []
    for k in q:
        take = int(round(n_max * q[k]))
        if take == 0:
            continue
        pool = np.where((attr == k[0]) & (task == k[1]))[0]
        idx.append(rng.choice(pool, size=min(take, len(pool)), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], attr[idx], task[idx]


def main():
    if "--merge" in sys.argv:
        merge()
        return
    device = get_device()
    rows = []
    for dataset, attr_name, task_name, ts in FAMILIES:
        X, attr, task, *_ = load_cell(dataset, attr_name, task_name)
        for t in ts:
            Xs, As, Ys = decorrelate_subsample(X, attr, task, t)
            tag = f"{dataset}/{attr_name}/{task_name}[t={t:g}]"
            print(f"\n[construct] {tag}: N {len(attr)} -> {len(As)}", flush=True)
            row = measure_cell(dataset, attr_name, task_name, device,
                               X=Xs, attr=As, task=Ys, tag=tag)
            row["constructed"] = True
            row["t"] = t
            rows.append(row)
    (RESULTS_DIR / "cliff_or_ramp_constructed.json").write_text(
        json.dumps({"experiment": "cliff_or_ramp", "rows": rows}, indent=2))
    print(f"\n[save] {RESULTS_DIR/'cliff_or_ramp_constructed.json'}")


def merge():
    natural = json.loads((RESULTS_DIR / "continuous_cost.json").read_text())["rows"]
    for r in natural:
        r["constructed"] = False
    constructed = json.loads(
        (RESULTS_DIR / "cliff_or_ramp_constructed.json").read_text())["rows"]
    rows = sorted(natural + constructed, key=lambda r: r["predictor"])
    good = [r for r in rows if not r["degenerate"]]

    print("\n" + "=" * 110)
    print("GOAL 2 — COST ACROSS THE FULL PREDICTOR RANGE (natural + constructed, sorted)")
    print("=" * 110)
    print(f"{'cell':<52} {'pred':>6} {'cost_rep':>9} {'cost_durable':>13}")
    print("-" * 110)
    for r in good:
        cr = f"{r['cost_rep']:.3f}" + ("c" if r["censored_rep"] else " ")
        cd = f"{r['cost_durable']:.3f}" + ("c" if r["censored_durable"] else " ")
        tagc = " CONSTRUCTED" if r["constructed"] else ""
        print(f"{r['cell']:<52} {r['predictor']:>6.3f} {cr:>9} {cd:>13}{tagc}")
    print("-" * 110)

    # Cliff-or-ramp verdict on the durable cost: is the middle populated with
    # intermediate costs, or does cost jump from cheap to ~1?
    mid = [r for r in good if 0.55 <= r["predictor"] <= 0.68]
    inter = [r for r in mid if 0.25 <= r["cost_durable"] <= 0.85]
    verdict = {
        "n_middle": len(mid),
        "n_intermediate_cost": len(inter),
        "middle_costs": sorted(round(r["cost_durable"], 3) for r in mid),
    }
    if len(inter) >= max(2, len(mid) // 4):
        v = (f"RAMP: {len(inter)}/{len(mid)} middle-zone cells (predictor 0.55-0.68) have "
             f"intermediate durable-removal cost (0.25-0.85) — cost rises smoothly, so there is "
             f"a usable middle ground where removal is real but affordable.")
    else:
        v = (f"CLIFF: only {len(inter)}/{len(mid)} middle-zone cells have intermediate durable "
             f"cost; cells jump from cheap (<0.25) to near-total (>0.85) — the middle is a true "
             f"cliff, not a ramp.")
    verdict["verdict"] = v
    print("\n>>> " + v)

    out = {"experiment": "cliff_or_ramp_merged", "verdict": verdict, "rows": rows}
    (RESULTS_DIR / "cliff_or_ramp.json").write_text(json.dumps(out, indent=2))
    _plot(good, v, RESULTS_DIR / "cliff_or_ramp.png")
    print(f"[save] {RESULTS_DIR/'cliff_or_ramp.json'} and .png")


def _plot(rows, verdict, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax, key, title in [(axes[0], "cost_rep", "hide the REPRESENTATION"),
                           (axes[1], "cost_durable", "durable removal (rep + output)")]:
        nat = [r for r in rows if not r["constructed"]]
        con = [r for r in rows if r["constructed"]]
        ax.scatter([r["predictor"] for r in nat], [r[key] for r in nat], s=80,
                   color="C0", label="natural cell", zorder=3)
        ax.scatter([r["predictor"] for r in con], [r[key] for r in con], s=80,
                   color="C3", marker="s", label="constructed (subsampled)", zorder=3)
        srt = sorted(rows, key=lambda r: r["predictor"])
        ax.plot([r["predictor"] for r in srt], [r[key] for r in srt],
                color="gray", lw=0.8, alpha=0.5, zorder=1)
        ax.axvspan(0.55, 0.68, color="orange", alpha=0.07, label="the 'middle zone'")
        ax.set_xlabel("PREDICTOR: XGB AUC(attribute | task label alone)")
        ax.set_ylabel("cost (fraction of clean lift lost at ≤0.55 crossing)")
        ax.set_title(f"Goal 2 — cliff or ramp? cost to {title}")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(verdict[:120], fontsize=9, y=1.0)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
