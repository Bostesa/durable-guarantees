"""DIAGNOSTIC — durable-removal cost with the REPRESENTATION SURFACE ONLY.

Recomputation from stored files; nothing is retrained. `cost_rep` and
`cost_durable` are both already stored per cell by experiments/continuous_cost.py:

  cost_rep     = fraction of clean lift lost at the first sigma where the
                 black-box suite max(XGB, MLP) reads <= 0.55 on the
                 REPRESENTATION surface alone
  cost_durable = same, but requiring BOTH the representation and the task
                 output to be at or below 0.55 (the paper's definition)

Both are linearly interpolated between swept sigma points, clipped to [0, 1],
and flagged censored (cost forced to 1.0) if no crossing occurs by sigma <= 192.

Populations:
  27 natural   = 20 non-degenerate rows of continuous_cost.json
                 + 7 non-degenerate rows of expansion_cost.json
   7 external  = the expansion subset of those 27
   9 constructed = cliff_or_ramp_constructed.json (a separate population, not
                 part of the 27)

Cluster bootstrap on base dataset, same procedure and seed as
run_clustered_correlation.py.
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
COUPLING_BAR = 0.55


def load(fname, source):
    d = json.loads((RESULTS / fname).read_text())
    out = []
    for r in d["rows"]:
        if r.get("degenerate"):
            continue
        out.append(dict(
            cell=r["cell"],
            dataset=r.get("dataset") or r["cell"].split("/")[0],
            source=source, predictor=r["predictor"],
            cost_rep=r["cost_rep"], censored_rep=bool(r["censored_rep"]),
            sigma_rep=r["sigma_rep"],
            cost_durable=r["cost_durable"],
            censored_durable=bool(r["censored_durable"]),
            sigma_durable=r["sigma_durable"],
            t=r.get("t")))
    return out


def corr_block(rows, key):
    x = np.array([r["predictor"] for r in rows], float)
    y = np.array([r[key] for r in rows], float)
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return dict(pearson=float(pr), pearson_p=float(pp),
                spearman=float(sr), spearman_p=float(sp), n=len(rows))


def cluster_ci(rows, key, rng):
    x = np.array([r["predictor"] for r in rows], float)
    y = np.array([r[key] for r in rows], float)
    ds = np.array([r["dataset"] for r in rows])
    groups = sorted(set(ds))
    idx = {g: np.where(ds == g)[0] for g in groups}
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        sel = np.concatenate([idx[groups[i]] for i in pick])
        if np.std(x[sel]) < 1e-12 or np.std(y[sel]) < 1e-12:
            continue
        boots.append(pearsonr(x[sel], y[sel])[0])
    b = np.array(boots)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return dict(ci95=[float(lo), float(hi)], n_boot=len(b),
                n_groups=len(groups), frac_le_zero=float((b <= 0).mean()))


def main():
    rng = np.random.default_rng(SEED)
    orig = load("continuous_cost.json", "original")
    ext = load("expansion_cost.json", "external")
    natural = orig + ext
    constructed = load("cliff_or_ramp_constructed.json", "constructed")
    assert len(natural) == 27, len(natural)
    assert len(ext) == 7, len(ext)
    assert len(constructed) == 9, len(constructed)

    out = {"experiment": "rep_only_vs_both_surfaces_cost",
           "definition_source": "experiments/continuous_cost.py:14-15",
           "n_natural": 27, "n_external": 7, "n_constructed": 9,
           "seed": SEED, "n_boot": N_BOOT}

    # ---- (c) per-cell table (printed first so the rest has context) -------
    print("(c) PER-CELL COST UNDER BOTH DEFINITIONS  "
          "(c = censored, cost forced to 1.0)\n")
    hdr = ("%-44s %-10s %6s %9s %7s %9s %7s %8s"
           % ("cell", "source", "pred", "cost_rep", "s_rep", "cost_both",
              "s_both", "delta"))
    print(hdr); print("-" * len(hdr))
    for r in sorted(natural, key=lambda z: z["predictor"]):
        d = r["cost_durable"] - r["cost_rep"]
        print("%-44s %-10s %6.3f %8.3f%s %7s %8.3f%s %7s %8.3f"
              % (r["cell"], r["source"], r["predictor"],
                 r["cost_rep"], "c" if r["censored_rep"] else " ",
                 "—" if r["sigma_rep"] is None else f"{r['sigma_rep']:.1f}",
                 r["cost_durable"], "c" if r["censored_durable"] else " ",
                 "—" if r["sigma_durable"] is None else f"{r['sigma_durable']:.1f}",
                 d))
    out["per_cell"] = natural

    # ---- (a) all 27 -------------------------------------------------------
    print("\n(a) ALL 27 NATURAL CELLS, predictor vs cost")
    a = {}
    for key, label in (("cost_rep", "representation only"),
                       ("cost_durable", "both surfaces")):
        c = corr_block(natural, key)
        ci = cluster_ci(natural, key, rng)
        a[key] = dict(**c, cluster=ci)
        print("  %-22s pearson r=%.4f (p=%.2e)  spearman rho=%.4f  "
              "clustered 95%% CI [%.4f, %.4f]"
              % (label, c["pearson"], c["pearson_p"], c["spearman"],
                 ci["ci95"][0], ci["ci95"][1]))
    out["a_all27"] = a

    # ---- (b) external 7 and constructed 9 --------------------------------
    print("\n(b) SUBPOPULATIONS")
    b = {}
    for rows, name in ((ext, "external_7"), (constructed, "constructed_9")):
        b[name] = {}
        print("  %s (n=%d)" % (name, len(rows)))
        for key, label in (("cost_rep", "representation only"),
                           ("cost_durable", "both surfaces")):
            c = corr_block(rows, key)
            entry = dict(**c)
            if len(set(r["dataset"] for r in rows)) > 1:
                entry["cluster"] = cluster_ci(rows, key, rng)
                ci = entry["cluster"]["ci95"]
                extra = "  clustered 95%% CI [%.4f, %.4f]" % (ci[0], ci[1])
            else:
                extra = "  (single base dataset; no clustering possible)"
            b[name][key] = entry
            print("    %-22s pearson r=%.4f (p=%.2e)  spearman rho=%.4f%s"
                  % (label, c["pearson"], c["pearson_p"], c["spearman"], extra))
    out["b_subpopulations"] = b

    # ---- (d) high-coupling cells -----------------------------------------
    hi = [r for r in natural if r["predictor"] > COUPLING_BAR]
    print("\n(d) CELLS WITH COUPLING ABOVE %.2f: %d of 27" % (COUPLING_BAR, len(hi)))
    print("  %-44s %6s %9s %10s" % ("cell", "pred", "cost_rep", "cost_both"))
    for r in sorted(hi, key=lambda z: -z["predictor"]):
        print("  %-44s %6.3f %9.3f %10.3f"
              % (r["cell"], r["predictor"], r["cost_rep"], r["cost_durable"]))
    both = np.array([r["cost_durable"] for r in hi])
    rep = np.array([r["cost_rep"] for r in hi])
    n_pinned = int((both >= 0.99).sum())
    out["d_high_coupling"] = dict(
        bar=COUPLING_BAR, n=len(hi), cells=[r["cell"] for r in hi],
        cost_both_min=float(both.min()), cost_both_max=float(both.max()),
        cost_both_mean=float(both.mean()),
        n_at_or_above_0_99=n_pinned,
        n_censored_durable=int(sum(r["censored_durable"] for r in hi)),
        cost_rep_min=float(rep.min()), cost_rep_max=float(rep.max()),
        cost_rep_mean=float(rep.mean()))
    print("  both-surfaces cost: min %.3f max %.3f mean %.3f; %d of %d at or "
          "above 0.99 (%d censored)"
          % (both.min(), both.max(), both.mean(), n_pinned, len(hi),
             sum(r["censored_durable"] for r in hi)))
    print("  rep-only cost:      min %.3f max %.3f mean %.3f"
          % (rep.min(), rep.max(), rep.mean()))

    # censoring summary (affects how much of the correlation is real variation)
    out["censoring"] = dict(
        natural_rep=int(sum(r["censored_rep"] for r in natural)),
        natural_durable=int(sum(r["censored_durable"] for r in natural)),
        external_rep=int(sum(r["censored_rep"] for r in ext)),
        external_durable=int(sum(r["censored_durable"] for r in ext)),
        constructed_rep=int(sum(r["censored_rep"] for r in constructed)),
        constructed_durable=int(sum(r["censored_durable"] for r in constructed)))
    print("\n  censored (cost forced to 1.0): natural rep %d / both %d; "
          "external rep %d / both %d; constructed rep %d / both %d"
          % tuple(out["censoring"][k] for k in
                  ("natural_rep", "natural_durable", "external_rep",
                   "external_durable", "constructed_rep", "constructed_durable")))

    (RESULTS / "rep_only_cost.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {RESULTS / 'rep_only_cost.json'}")


if __name__ == "__main__":
    main()
