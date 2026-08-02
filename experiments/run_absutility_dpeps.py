"""DIAGNOSTICS 3 + 4 — absolute utility behind every ratio, and the epsilon of
the surviving full-rank operating points.

D3 ABSOLUTE UTILITY. Every "% of clean lift" number is a ratio whose
denominator is a lift, not an accuracy. This restates each one as an absolute
lift and an absolute accuracy (majority baseline + lift), in-sample and
out-of-partition, and flags every ratio above 100% or below 0.

D4 EPSILON OF THE SURVIVING POINTS. Proposition 3's guarantee is
eps = (2C/sigma) * sqrt(2 ln(1.25/delta)) = K * C / sigma with
K = 9.6896 at delta = 1e-5 — and it requires the CLIP. The Figure-5 operating
points are the UNCLIPPED full-rank channel, whose per-release l2 sensitivity is
unbounded, so their epsilon as run is infinite. dp_fullrank.json measures the
clipped mechanism on the same cells, so for the sigmas it covers we can say
directly what a finite-epsilon mechanism retains at that same sigma.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"

NAME = {"easy": "hmda/race/loan_decision",
        "middle": "hmda/race/loan_amount_band",
        "hard": "adult/sex/income"}


def main():
    out = {}
    fp = json.loads((RESULTS / "fresh_partition_generalization.json").read_text())
    fpc = {c["cell"]: c for c in fp["cells"]}
    tt = json.loads((RESULTS / "two_tier_certification.json").read_text())
    clean_ins = {k: next(c for c in tt["e2e_cells"] if c["cell"] == v)["clean_lift"]
                 for k, v in NAME.items()}

    # ---------------- D3 ---------------------------------------------------
    print("D3  ABSOLUTE UTILITY BEHIND EVERY RATIO")
    print("    majority = task-majority baseline accuracy; lift = accuracy - majority\n")
    hdr = ("%-8s %-13s %7s %9s %9s %9s | %7s %9s %9s %9s  %s"
           % ("cell", "channel", "in%", "in_lift", "in_acc", "cleanIn",
              "out%", "out_lift", "out_acc", "cleanOut", "flag"))
    print(hdr); print("-" * len(hdr))
    rows3 = []
    for key, cellname in NAME.items():
        c = fpc[cellname]
        maj = c["task_maj_eval"]
        clean_out = c["clean_lift_eval"]
        clean_in = clean_ins[key]
        for chan, label in (("subspace", "subspace"),
                            ("fullrank_t1", "full-rank T1"),
                            ("fullrank_t2", "full-rank T2")):
            ch = c[chan]
            pct_in = ch["stored_inpartition"]["util_pct"]
            pct_out = ch["utility_kept_eval_pct"]
            lift_in = pct_in / 100.0 * clean_in
            lift_out = ch["lift_eval_mean"]
            flag = []
            if pct_in > 100 or pct_out > 100:
                flag.append(">100%")
            if pct_in < 0 or pct_out < 0:
                flag.append("negative")
            r = dict(cell=key, channel=label,
                     majority_baseline=maj,
                     clean_lift_insample=clean_in, clean_lift_outpartition=clean_out,
                     pct_insample=pct_in, abs_lift_insample=lift_in,
                     abs_acc_insample=maj + lift_in,
                     pct_outpartition=pct_out, abs_lift_outpartition=lift_out,
                     abs_acc_outpartition=maj + lift_out,
                     flag=",".join(flag) or "")
            rows3.append(r)
            print("%-8s %-13s %7.1f %9.4f %9.4f %9.4f | %7.1f %9.4f %9.4f %9.4f  %s"
                  % (key, label, pct_in, lift_in, maj + lift_in, clean_in,
                     pct_out, lift_out, maj + lift_out, clean_out,
                     r["flag"]))
    out["d3_absolute_utility"] = rows3
    flagged = [r for r in rows3 if r["flag"]]
    print(f"\n  flagged (>100% or negative): {len(flagged)} of {len(rows3)}")
    for r in flagged:
        print("    %-8s %-13s in %.1f%% (lift %+.4f, acc %.4f)  out %.1f%% "
              "(lift %+.4f, acc %.4f)  [%s]"
              % (r["cell"], r["channel"], r["pct_insample"],
                 r["abs_lift_insample"], r["abs_acc_insample"],
                 r["pct_outpartition"], r["abs_lift_outpartition"],
                 r["abs_acc_outpartition"], r["flag"]))

    # ---------------- D4 ---------------------------------------------------
    dp = json.loads((RESULTS / "dp_fullrank.json").read_text())
    K = dp["K_gaussian"]
    print("\n\nD4  EPSILON OF THE SURVIVING FULL-RANK OPERATING POINTS")
    print(f"    Prop 3: eps = K*C/sigma, K = {K:.4f} at delta = {dp['delta']}. "
          "Requires the clip.\n")
    fig5 = [("easy", "T1", 8.0, 94.6), ("middle", "T1", 12.0, 56.6),
            ("hard", "T1", 32.0, -0.5), ("easy", "T2", 20.0, 55.5),
            ("middle", "T2", 24.0, 21.7), ("hard", "T2", 64.0, -7.0)]
    dp_by = {}
    for r in dp["rows"]:
        dp_by.setdefault((r["cell"], r["sigma"]), []).append(r)

    print("%-8s %-5s %7s %9s %10s   %s"
          % ("cell", "tier", "sigma", "util%", "eps as run", "clipped mechanism at same sigma"))
    print("-" * 104)
    rows4 = []
    for cell, tier, sigma, util in fig5:
        avail = dp_by.get((cell, sigma))
        if avail:
            desc = "; ".join(f"eps={r['eps_nominal']:g} (C={r['C']:.3f}) -> "
                             f"{r['kept_pct']:+.1f}%" for r in
                             sorted(avail, key=lambda z: z["eps_nominal"]))
        else:
            desc = f"NOT IN GRID (dp_fullrank covers sigma in {sorted({r['sigma'] for r in dp['rows']})})"
        rows4.append(dict(cell=cell, tier=tier, sigma=sigma, reported_util_pct=util,
                          eps_as_run="infinite (unclipped: unbounded l2 sensitivity)",
                          clipped_points=[dict(eps=r["eps_nominal"], C=r["C"],
                                               kept_pct=r["kept_pct"],
                                               frac_clipped=r["frac_clipped_mean"],
                                               median_preclip_norm=r["median_preclip_norm_mean"])
                                          for r in (avail or [])]))
        print("%-8s %-5s %7g %9.1f %10s   %s"
              % (cell, tier, sigma, util, "infinite", desc))

    # the direct answer for the easy cell at sigma=8
    easy8 = sorted(dp_by.get(("easy", 8.0), []), key=lambda z: z["eps_nominal"])
    if easy8:
        print("\n  DIRECT ANSWER — 'what is the epsilon of sigma=8 on the easy cell':")
        print("    As run in Figure 5 the channel is UNCLIPPED, so its per-release")
        print("    l2 sensitivity is unbounded and epsilon is INFINITE. It retains 94.6%.")
        print("    Clipping it to obtain a finite epsilon at the same sigma=8 gives:")
        for r in easy8:
            print("      eps=%-4g C=%.4f  utility %+6.2f%%  (%.1f%% of rows clipped, "
                  "median pre-clip norm %.2f)"
                  % (r["eps_nominal"], r["C"], r["kept_pct"],
                     100 * r["frac_clipped_mean"], r["median_preclip_norm_mean"]))
        # what C would leave it essentially unclipped, and the eps that implies
        med = float(np.mean([r["median_preclip_norm_mean"] for r in easy8]))
        print(f"    Median pre-clip ||h|| across those runs is {med:.2f}; a clip that")
        print(f"    left the median row untouched (C = {med:.2f}) would already imply")
        print(f"    eps = K*C/sigma = {K * med / 8.0:.2f}, i.e. far outside any budget")
        print("    at which the table shows usable utility.")
        out["d4_direct_answer"] = dict(
            sigma=8.0, cell="easy", reported_util_pct=94.6,
            eps_as_run=None, eps_as_run_note="infinite (unclipped)",
            clipped=[dict(eps=r["eps_nominal"], C=r["C"], kept_pct=r["kept_pct"])
                     for r in easy8],
            median_preclip_norm=med, eps_at_median_clip=K * med / 8.0)
    out["d4_rows"] = rows4
    out["K_gaussian"] = K
    out["delta"] = dp["delta"]

    (RESULTS / "absutility_dpeps.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {RESULTS / 'absutility_dpeps.json'}")


if __name__ == "__main__":
    main()
