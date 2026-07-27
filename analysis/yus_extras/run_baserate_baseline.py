"""Base-rate-gap baseline vs the label-coupling predictor (report only).

For each of the 27 natural cells (20 originals from results/continuous_cost.json
+ 7 non-degenerate external cells from results/expansion_cost.json), compute the
trivial marginal statistic

    gap = |P(s=1|y=1) - P(s=1|y=0)|

on cells where both y and s are binary, and its exact generalization on the
multiclass cells,

    gap = max over label pairs (y, y') of  TV( P(s|y), P(s|y') )
        = max_{y,y'} 0.5 * sum_k |P(s=k|y) - P(s=k|y')|

which reduces to the binary formula when both are binary. Then correlate each
predictor with the stored durable-removal cost and compare against the
label-coupling predictor's r = 0.799 / rho = 0.828 (n = 27).

Cell loading uses the same loaders and the same max_n cap the cost runs used,
so the marginals correspond to exactly the rows those costs were measured on.
Numbers only; nothing is modified.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from experiments.diagnostic import load_cell  # noqa: E402
from experiments.expansion_pipeline import load_cell_by_tag  # noqa: E402

RESULTS = REPO / "results"


def base_rate_gap(attr, task):
    """max over label pairs of TV(P(s|y), P(s|y')); equals |P(s=1|y=1)-P(s=1|y=0)|
    when both are binary."""
    ys = np.unique(task)
    ks = int(attr.max()) + 1
    cond = []
    for y in ys:
        c = np.bincount(attr[task == y], minlength=ks).astype(np.float64)
        cond.append(c / max(c.sum(), 1.0))
    best = 0.0
    for i in range(len(cond)):
        for j in range(i + 1, len(cond)):
            best = max(best, 0.5 * float(np.abs(cond[i] - cond[j]).sum()))
    return best, len(ys), ks


def main():
    t0 = time.time()
    cc = json.loads((RESULTS / "continuous_cost.json").read_text())
    ec = json.loads((RESULTS / "expansion_cost.json").read_text())

    rows = []
    for r in cc["rows"]:
        if r.get("degenerate"):
            continue
        X, attr, task, *_ = load_cell(r["dataset"], r["attr"], r["task"])
        gap, n_y, n_s = base_rate_gap(attr, task)
        rows.append({"cell": r["cell"], "source": "original", "gap": gap,
                     "n_y": n_y, "n_s": n_s, "binary": n_y == 2 and n_s == 2,
                     "predictor": r["predictor"], "cost": r["cost_durable"]})
        print(f"  {r['cell']:<46} gap={gap:.4f} pred={r['predictor']:.4f} "
              f"cost={r['cost_durable']:+.4f} (|y|={n_y},|s|={n_s})", flush=True)

    for r in ec["rows"]:
        if r.get("degenerate"):
            print(f"  [skip degenerate] {r['cell']}", flush=True)
            continue
        _, _, _, X, attr, task, *_ = load_cell_by_tag(r["cell"])
        gap, n_y, n_s = base_rate_gap(attr, task)
        rows.append({"cell": r["cell"], "source": "external", "gap": gap,
                     "n_y": n_y, "n_s": n_s, "binary": n_y == 2 and n_s == 2,
                     "predictor": r["predictor"], "cost": r["cost_durable"]})
        print(f"  {r['cell']:<46} gap={gap:.4f} pred={r['predictor']:.4f} "
              f"cost={r['cost_durable']:+.4f} (|y|={n_y},|s|={n_s})", flush=True)

    def corr(sel, key):
        x = np.array([d[key] for d in sel], dtype=float)
        y = np.array([d["cost"] for d in sel], dtype=float)
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        return {"pearson_r": float(pr), "pearson_p": float(pp),
                "spearman_r": float(sr), "spearman_p": float(sp), "n": len(sel)}

    binary = [d for d in rows if d["binary"]]
    out = {
        "experiment": "baserate_gap_baseline",
        "n_cells": len(rows), "n_binary_cells": len(binary),
        "all27": {"gap": corr(rows, "gap"), "predictor": corr(rows, "predictor")},
        "binary_only": {"gap": corr(binary, "gap"),
                        "predictor": corr(binary, "predictor")},
        "stored_paper_predictor_27": {"pearson_r": 0.7991437704746869,
                                      "spearman_r": 0.8281658135319308, "n": 27},
        "rows": rows, "minutes": round((time.time() - t0) / 60, 1),
    }

    print("\n" + "=" * 88)
    print(f"{'set':<22}{'statistic':<14}{'pearson r':>11}{'p':>10}{'spearman rho':>14}{'p':>10}{'n':>4}")
    print("-" * 88)
    for setname, blob in (("all 27 natural", out["all27"]),
                          (f"binary-only ({len(binary)})", out["binary_only"])):
        for stat in ("predictor", "gap"):
            c = blob[stat]
            label = "label-coupling" if stat == "predictor" else "base-rate gap"
            print(f"{setname:<22}{label:<14}{c['pearson_r']:>11.4f}{c['pearson_p']:>10.2e}"
                  f"{c['spearman_r']:>14.4f}{c['spearman_p']:>10.2e}{c['n']:>4}")
    a, b = out["all27"]["predictor"], out["all27"]["gap"]
    winner = ("label-coupling predictor" if a["pearson_r"] > b["pearson_r"]
              else "base-rate gap")
    out["winner_pearson_all27"] = winner
    out["delta_pearson_all27"] = float(a["pearson_r"] - b["pearson_r"])
    out["delta_spearman_all27"] = float(a["spearman_r"] - b["spearman_r"])
    print(f"\nWINNER (Pearson, all 27): {winner}  "
          f"(delta r = {out['delta_pearson_all27']:+.4f}, "
          f"delta rho = {out['delta_spearman_all27']:+.4f})")

    (HERE / "baserate_baseline.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {HERE / 'baserate_baseline.json'} ({out['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
