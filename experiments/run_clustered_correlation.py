"""TASK 1 — does the predictor/cost correlation survive clustering on base dataset?

The 27 natural cells come from only six base datasets, so the stored
r = 0.799 (95% CI 0.60-0.90) treats non-independent observations as
independent. This computes (a) leave-one-dataset-out Pearson and Spearman,
one value per held-out dataset, and (b) a cluster-bootstrap 95% CI for
Pearson, resampling DATASETS with replacement (the standard cluster-robust
interval for a correlation).

Cells and costs are read from the same stored files the paper's r = 0.799
comes from: results/continuous_cost.json (20 originals) +
results/expansion_cost.json (7 non-degenerate external). Nothing recomputed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
HERE = Path(__file__).resolve().parent
N_BOOT = 20000
SEED = 20260728


def load_cells():
    cc = json.loads((RESULTS / "continuous_cost.json").read_text())
    ec = json.loads((RESULTS / "expansion_cost.json").read_text())
    rows = []
    for r in cc["rows"]:
        if r.get("degenerate"):
            continue
        rows.append(dict(cell=r["cell"], dataset=r["dataset"],
                         predictor=r["predictor"], cost=r["cost_durable"],
                         source="continuous_cost.json"))
    for r in ec["rows"]:
        if r.get("degenerate"):
            continue
        rows.append(dict(cell=r["cell"], dataset=r["cell"].split("/")[0],
                         predictor=r["predictor"], cost=r["cost_durable"],
                         source="expansion_cost.json"))
    return rows


def main():
    rows = load_cells()
    assert len(rows) == 27, len(rows)
    x = np.array([r["predictor"] for r in rows])
    y = np.array([r["cost"] for r in rows])
    ds = np.array([r["dataset"] for r in rows])
    groups = sorted(set(ds))

    pr_all, pp_all = pearsonr(x, y)
    sr_all, sp_all = spearmanr(x, y)
    print(f"[all 27] pearson r={pr_all:.4f} (p={pp_all:.2e})  "
          f"spearman rho={sr_all:.4f} (p={sp_all:.2e})")
    print(f"[stored paper value] r=0.799 (95% CI 0.60-0.90), rho=0.828\n")
    print("cells per dataset:",
          {g: int((ds == g).sum()) for g in groups}, "\n")

    # ---- (a) leave-one-dataset-out ----------------------------------------
    print("%-18s %5s %10s %10s %10s %10s" % ("held-out dataset", "n_out",
                                             "n_kept", "pearson", "spearman",
                                             "p(pearson)"))
    print("-" * 68)
    loo = []
    for g in groups:
        m = ds != g
        pr, pp = pearsonr(x[m], y[m])
        sr, _ = spearmanr(x[m], y[m])
        loo.append(dict(held_out=g, n_out=int((~m).sum()), n_kept=int(m.sum()),
                        pearson=float(pr), spearman=float(sr),
                        pearson_p=float(pp)))
        print("%-18s %5d %10d %10.4f %10.4f %10.2e"
              % (g, (~m).sum(), m.sum(), pr, sr, pp))
    pr_rng = (min(d["pearson"] for d in loo), max(d["pearson"] for d in loo))
    sr_rng = (min(d["spearman"] for d in loo), max(d["spearman"] for d in loo))
    print(f"\n  leave-one-dataset-out Pearson range  {pr_rng[0]:.4f} .. {pr_rng[1]:.4f}")
    print(f"  leave-one-dataset-out Spearman range {sr_rng[0]:.4f} .. {sr_rng[1]:.4f}")

    # ---- (b) cluster bootstrap on base dataset ----------------------------
    rng = np.random.default_rng(SEED)
    idx_by_g = {g: np.where(ds == g)[0] for g in groups}
    boots, n_skipped = [], 0
    for _ in range(N_BOOT):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        sel = np.concatenate([idx_by_g[groups[i]] for i in pick])
        xb, yb = x[sel], y[sel]
        if np.std(xb) < 1e-12 or np.std(yb) < 1e-12:
            n_skipped += 1
            continue
        boots.append(pearsonr(xb, yb)[0])
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\n  cluster bootstrap ({len(boots)} resamples of the {len(groups)} "
          f"datasets, {n_skipped} degenerate draws skipped)")
    print(f"  Pearson r = {pr_all:.4f}, cluster-robust 95% CI "
          f"[{lo:.4f}, {hi:.4f}]")
    print(f"  naive (unclustered) 95% CI as published: [0.60, 0.90]")
    print(f"  fraction of resamples with r <= 0: {float((boots <= 0).mean()):.4f}")

    survives = bool(lo > 0)
    print(f"\n  VERDICT: correlation {'SURVIVES' if survives else 'DOES NOT SURVIVE'} "
          f"clustering (cluster-robust lower bound {lo:.4f})")

    out = dict(
        experiment="clustered_correlation", n_cells=27, datasets=groups,
        cells_per_dataset={g: int((ds == g).sum()) for g in groups},
        all27=dict(pearson=float(pr_all), pearson_p=float(pp_all),
                   spearman=float(sr_all), spearman_p=float(sp_all)),
        stored_paper=dict(pearson=0.799, ci=[0.60, 0.90], spearman=0.828),
        leave_one_dataset_out=loo,
        loo_pearson_range=list(pr_rng), loo_spearman_range=list(sr_rng),
        cluster_bootstrap=dict(n_boot=N_BOOT, n_used=len(boots), seed=SEED,
                               ci95=[float(lo), float(hi)],
                               frac_le_zero=float((boots <= 0).mean())),
        survives_clustering=survives,
        rows=rows)
    (RESULTS / "clustered_correlation.json").write_text(
        json.dumps(out, indent=1, default=float))
    print(f"\n[save] {RESULTS / 'clustered_correlation.json'}")


if __name__ == "__main__":
    main()
