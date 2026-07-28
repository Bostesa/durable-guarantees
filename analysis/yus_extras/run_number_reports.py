"""TASK 5 — four number reports, every value traced to a stored file.

(a) headline cells: majority baseline, clean accuracy, clean lift (in-sample)
(b) VFAE beta sweep on the easy cell at beta = 1 and beta = 10^3
(c) Tier-2 readings for the three Adult survivors among the eight survivors
(d) post-ReLU adapter tap vs the "no erasure" certificate for Adult/sex
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "results"
HERE = Path(__file__).resolve().parent
BAR = 0.55

TT = {"easy": "hmda/race/loan_decision",
      "middle": "hmda/race/loan_amount_band",
      "hard": "adult/sex/income"}


def main():
    out = {}

    # ---- (a) headline cell accuracies ------------------------------------
    tt = json.loads((RESULTS / "two_tier_certification.json").read_text())
    fp = json.loads((HERE / "fresh_partition_generalization.json").read_text())
    fpc = {c["cell"]: c for c in fp["cells"]}
    rows_a = []
    for key, name in TT.items():
        c = next(x for x in tt["e2e_cells"] if x["cell"] == name)
        maj = fpc[name]["task_maj_eval"]
        lift = c["clean_lift"]
        rows_a.append(dict(cell=key, name=name, majority_baseline_acc=maj,
                           clean_lift_insample=lift, clean_accuracy=maj + lift,
                           one_acc_point_as_pct_of_lift=100.0 / (lift * 100),
                           source_lift="two_tier_certification.json:e2e_cells.clean_lift",
                           source_majority=("fresh_partition_generalization.json:"
                                            "cells.task_maj_eval")))
    out["a_headline_accuracies"] = rows_a
    print("(a) headline cells, in-sample")
    print("%-8s %-30s %10s %10s %10s %14s" % ("cell", "name", "majority",
                                              "cleanAcc", "cleanLift",
                                              "1pt acc = %lift"))
    for r in rows_a:
        print("%-8s %-30s %10.4f %10.4f %10.4f %13.1f%%"
              % (r["cell"], r["name"], r["majority_baseline_acc"],
                 r["clean_accuracy"], r["clean_lift_insample"],
                 r["one_acc_point_as_pct_of_lift"]))

    # ---- (b) VFAE beta sweep ---------------------------------------------
    bg = json.loads((RESULTS / "baseline_gauntlet.json").read_text())
    easy = next(c for c in bg["cells"] if c["cell"] == TT["easy"])
    vfae = next(b for b in easy["baselines"] if b["baseline"] == "VFAE")
    want = {1.0: "β=1 sampled-z", 1000.0: "β=1000 sampled-z"}
    rows_b = []
    for knob, label in want.items():
        r = next(x for x in vfae["rows"] if x["label"] == label)
        rows_b.append(dict(beta=knob, label=label, xgb=r["xgb"], mlp=r["mlp"],
                           lora=r.get("lora"), lrt=r["lrt"],
                           tier1_max=r["tier1_max"], tier2_max=r["tier2_max"],
                           source="baseline_gauntlet.json:cells[easy].baselines[VFAE].rows"))
    claimed = {1.0: 0.536, 1000.0: 0.743}
    for r in rows_b:
        r["paper_claims"] = claimed[r["beta"]]
        r["confirms"] = bool(abs(r["tier1_max"] - claimed[r["beta"]]) < 0.001)
    out["b_vfae_beta_sweep"] = rows_b
    print("\n(b) VFAE sampled-z on the easy cell (Tier-1 max)")
    for r in rows_b:
        print("  beta=%-6g tier1_max=%.4f  paper says %.3f  %s"
              % (r["beta"], r["tier1_max"], r["paper_claims"],
                 "CONFIRMED" if r["confirms"] else "MISMATCH"))

    # ---- (c) Tier-2 for the three Adult survivors -------------------------
    mi = json.loads((RESULTS / "mi_ceiling.json").read_text())
    af = mi["channels"]["adult_frozen"]
    sh = af["sigma_h"]
    lrt_by_rel = {round(r["sigma"] / sh, 4): r["gauss_lrt_auc"] for r in af["rows"]}
    tx = json.loads((HERE / "tpr_extension.json").read_text())
    cfgs = {c["config"]: c for c in tx["results"]["configs"]}
    adult_surv = [("honest_reaudit|adult/sex|E2 noise σ=4.0", 4.0),
                  ("honest_reaudit|adult/sex|E2 noise σ=8.0", 8.0),
                  ("honest_reaudit|adult/sex|E4S1 noise sigma=8.0", 8.0)]
    rows_c = []
    for cfg, srel in adult_surv:
        t1 = cfgs[cfg]["best_auc_mean"]
        lrt = lrt_by_rel[round(srel, 4)]
        t2 = max(t1, lrt)
        rows_c.append(dict(config=cfg, sigma_rel=srel, tier1_best=t1,
                           gauss_lrt=lrt, tier2_max=t2,
                           above_bar=bool(t2 > BAR),
                           source_tier1="tpr_extension.json:results.configs.best_auc_mean",
                           source_lrt="mi_ceiling.json:channels.adult_frozen.rows.gauss_lrt_auc"))
    n_above = sum(r["above_bar"] for r in rows_c)
    out["c_adult_survivors_tier2"] = dict(
        rows=rows_c, n_above_bar=n_above, n_total=len(rows_c),
        note=("rows 2 and 3 are duplicate measurements of the same sigma=8 "
              "configuration (E4S1 repeats the E2 measurement verbatim), so "
              "the three rows cover two distinct operating points"))
    print("\n(c) Tier-2 for the three Adult survivors")
    for r in rows_c:
        print("  %-46s T1=%.4f LRT=%.4f  T2=%.4f  %s"
              % (r["config"].split("|")[-1], r["tier1_best"], r["gauss_lrt"],
                 r["tier2_max"], "ABOVE 0.55" if r["above_bar"] else "below 0.55"))
    print("  -> %d of %d above the bar (two distinct sigmas: sigma_rel=4 above, "
          "sigma_rel=8 below)" % (n_above, len(rows_c)))

    # ---- (d) post-ReLU tap vs "no erasure" certificate --------------------
    fa = json.loads((RESULTS / "falsification_attack.json").read_text())
    se = json.loads((RESULTS / "smart_erasure_adult.json").read_text())
    nct = json.loads((RESULTS / "noise_channel_test.json").read_text())
    hr = json.loads((RESULTS / "honest_reaudit.json").read_text())
    post_relu = fa["final"]["attack_linear_r2"]
    no_erasure_se = next(r for r in se["rows"]
                         if r["method"] == "no erasure (frozen)")["attacked_r2"]
    no_erasure_nct = next(r for r in nct["rows"]
                          if r["sigma_rel"] == 0.0)["attacked_linear_r2_mean"]
    no_erasure_hr = next(r for r in hr["master"]
                         if r["method"] == "noise σ=0.0")["old_r2"]
    at_rest = next(r for r in nct["rows"]
                   if r["sigma_rel"] == 0.0)["baseline_linear_r2"]
    fa_baseline = fa["baseline"]["linear_r2"]
    out["d_relu_tap_vs_no_erasure"] = dict(
        post_relu_adapter_tap=post_relu,
        post_relu_source="falsification_attack.json:final.attack_linear_r2",
        no_erasure_attacked=no_erasure_se,
        no_erasure_sources_identical=dict(
            smart_erasure_adult=no_erasure_se,
            noise_channel_test=no_erasure_nct,
            honest_reaudit_master=no_erasure_hr,
            all_equal=bool(no_erasure_se == no_erasure_nct == no_erasure_hr)),
        at_rest_certificate=at_rest,
        at_rest_source="noise_channel_test.json:rows[sigma_rel=0].baseline_linear_r2",
        falsification_baseline=fa_baseline,
        difference=abs(post_relu - no_erasure_se),
        verdict=("distinct measurements that coincide to 2.1e-4: the post-ReLU "
                 "tap is a 300-step LoRA adaptation (falsification_attack.py), "
                 "the 0.1036 value is the frozen representation's attacked R^2 "
                 "(noise_channel_test.py). They are NOT a copy-through of each "
                 "other. The 0.1036 value IS a copy-through across three files."))
    print("\n(d) post-ReLU tap vs 'no erasure' certificate, Adult/sex")
    print("  post-ReLU adapter tap        = %.16f  (falsification_attack.json)" % post_relu)
    print("  'no erasure' attacked R^2    = %.16f  (three files, identical)" % no_erasure_se)
    print("     smart_erasure_adult %.16f" % no_erasure_se)
    print("     noise_channel_test  %.16f" % no_erasure_nct)
    print("     honest_reaudit      %.16f" % no_erasure_hr)
    print("  difference                   = %.2e" % abs(post_relu - no_erasure_se))
    print("  at-rest certificate (sigma=0)= %.16f" % at_rest)
    print("  falsification baseline       = %.16f" % fa_baseline)
    print("  -> distinct measurements that coincide; 0.1036 is itself a "
          "copy-through across three files")

    (HERE / "number_reports.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {HERE / 'number_reports.json'}")


if __name__ == "__main__":
    main()
