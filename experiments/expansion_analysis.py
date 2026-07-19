"""Expansion Phase 2 — analysis only. Reads the existing results JSONs
(read-only) and the new expansion_{cost,floor,reaudit}.json, prints the
comparison report, and writes results/expansion_analysis.json (a NEW file).
No figures are regenerated; nothing under paper/ is touched.

  1. Diagnostic correlation: the existing 20 continuous-cost cells vs the
     20 + new natural cells (predictor vs cost_durable and cost_rep,
     Pearson + Spearman), plus the new cells alone; per-cell pairs.
  2. Floor points against the y=x diagonal (the Fig-2 comparison), flagging
     visibly off-diagonal cells and the task-well-learned condition.
  3. Re-audit tally: what the existing 18-of-21 failing verdicts become
     with the new verdicts folded in, with per-cell detail on any verdict
     that does not follow the existing pattern.
  4. The coauthor summary table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

OFF_DIAGONAL = 0.05     # |floor - predictor| above this = visibly off-diagonal


def corr(pairs):
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp), "n": len(pairs)}


def main():
    old_cost = json.loads((RESULTS / "continuous_cost.json").read_text())
    old_reaudit = json.loads((RESULTS / "honest_reaudit.json").read_text())
    new_cost = json.loads((RESULTS / "expansion_cost.json").read_text())
    new_floor = json.loads((RESULTS / "expansion_floor.json").read_text())
    new_reaudit = json.loads((RESULTS / "expansion_reaudit.json").read_text())
    out = {"experiment": "expansion_analysis"}

    # ---------- 1. diagnostic correlation ----------
    old_rows = [r for r in old_cost["rows"] if not r["degenerate"]]
    new_rows = [r for r in new_cost["rows"] if not r["degenerate"]]
    degen_new = [r["cell"] for r in new_cost["rows"] if r["degenerate"]]

    print("=" * 100)
    print("1. DIAGNOSTIC CORRELATION — predictor vs continuous durable-removal cost")
    print("=" * 100)
    stats = {}
    for key in ("cost_durable", "cost_rep"):
        old_pairs = [(r["predictor"], r[key]) for r in old_rows]
        new_pairs = [(r["predictor"], r[key]) for r in new_rows]
        stats[key] = {"old_20": corr(old_pairs),
                      "combined": corr(old_pairs + new_pairs),
                      "new_alone": corr(new_pairs) if len(new_pairs) >= 3 else None}
        o, c, na = stats[key]["old_20"], stats[key]["combined"], stats[key]["new_alone"]
        print(f"\n{key}:")
        print(f"  old (n={o['n']}):       r={o['pearson_r']:+.3f} (p={o['pearson_p']:.1e})  "
              f"ρ={o['spearman_r']:+.3f} (p={o['spearman_p']:.1e})")
        print(f"  combined (n={c['n']}):  r={c['pearson_r']:+.3f} (p={c['pearson_p']:.1e})  "
              f"ρ={c['spearman_r']:+.3f} (p={c['spearman_p']:.1e})")
        if na:
            print(f"  new alone (n={na['n']}): r={na['pearson_r']:+.3f} "
                  f"(p={na['pearson_p']:.1e})  ρ={na['spearman_r']:+.3f} "
                  f"(p={na['spearman_p']:.1e})")
    out["correlation"] = stats
    if degen_new:
        print(f"\n  degenerate new cells excluded from correlations: {degen_new}")
    out["degenerate_new_cells"] = degen_new

    print("\n  per-cell (predictor, cost_rep, cost_durable), new cells on the ramp:")
    percell = []
    for r in sorted(new_rows, key=lambda r: r["predictor"]):
        cr = "None" if r["cost_rep"] is None else f"{r['cost_rep']:.3f}"
        cd = "None" if r["cost_durable"] is None else f"{r['cost_durable']:.3f}"
        cen = " censored" if r.get("censored_durable") else ""
        print(f"    {r['cell']:<44} ({r['predictor']:.3f}, {cr}, {cd}){cen} "
              f"clean_lift={r['clean_lift']:+.4f}")
        percell.append({k: r[k] for k in ("cell", "predictor", "cost_rep",
                                          "cost_durable", "clean_lift",
                                          "censored_durable", "degenerate")})
    out["new_cell_pairs"] = percell

    # ---------- 2. floor vs diagonal ----------
    print("\n" + "=" * 100)
    print("2. OUTPUT-LEAK FLOOR vs THE DIAGONAL (Fig-2 comparison; y = x)")
    print("=" * 100)
    print(f"{'cell':<44} {'predictor':>9} {'floor':>7} {'Δ':>7} {'well-learned':>13} "
          f"{'rep hidden':>11}")
    print("-" * 100)
    floors = []
    for r in sorted(new_floor["rows"], key=lambda r: r["predictor"]):
        delta = r["floor"] - r["predictor"]
        flag = "OFF-DIAGONAL" if abs(delta) > OFF_DIAGONAL else ""
        wl = r["task_well_learned"]
        print(f"{r['cell']:<44} {r['predictor']:>9.3f} {r['floor']:>7.3f} {delta:>+7.3f} "
              f"{str(wl):>13} {str(r['rep_hidden_within_grid']):>11}  {flag}")
        floors.append({"cell": r["cell"], "predictor": r["predictor"],
                       "floor": r["floor"], "delta": float(delta),
                       "off_diagonal": abs(delta) > OFF_DIAGONAL,
                       "task_well_learned": wl,
                       "rep_hidden_within_grid": r["rep_hidden_within_grid"],
                       "clean_out_leak": r["clean"]["out_max"]})
    print("-" * 100)
    print(f"(|Δ| > {OFF_DIAGONAL} flagged. Cells where the task is NOT well-learned "
          "(clean lift <= 0.01) do not satisfy the upper-bound condition — the "
          "label-coupling predictor is only a loose bound there.)")
    out["floor_points"] = floors

    # ---------- 3. re-audit tally ----------
    print("\n" + "=" * 100)
    print("3. RE-AUDIT TALLY — linear certificate vs the attacker suite")
    print("=" * 100)
    old_master = old_reaudit["master"]
    old_total = old_reaudit["total"]
    old_failed = old_total - old_reaudit["survived"]
    new_master = new_reaudit["master"]
    new_total = new_reaudit["total"]
    new_failed = new_total - new_reaudit["survived"]
    comb_total = old_total + new_total
    comb_failed = old_failed + new_failed
    print(f"  existing: {old_failed}/{old_total} re-audited verdicts fail honest "
          f"measurement (the published 18-of-21)")
    print(f"  new:      {new_failed}/{new_total} fail across "
          f"{len(new_reaudit['cells'])} cells")
    print(f"  combined: {comb_failed}/{comb_total}  "
          f"({100 * comb_failed / comb_total:.0f}% of verdicts)")

    # certificate-approved subset (linear pass) framing, old vs new
    old_appr = [r for r in old_master if r["old_verdict"] == "stopped"]
    old_appr_fail = [r for r in old_appr if not r["honest_stopped"]]
    new_appr = [r for r in new_master if r["linear_stopped"]]
    new_appr_fail = [r for r in new_appr if not r["honest_stopped"]]
    print(f"\n  certificate-approved subset: existing {len(old_appr_fail)}/{len(old_appr)} "
          f"false passes; new {len(new_appr_fail)}/{len(new_appr)}; combined "
          f"{len(old_appr_fail) + len(new_appr_fail)}/{len(old_appr) + len(new_appr)}")
    out["reaudit_tally"] = {
        "existing": {"failed": old_failed, "total": old_total},
        "new": {"failed": new_failed, "total": new_total},
        "combined": {"failed": comb_failed, "total": comb_total},
        "cert_approved": {
            "existing": {"false_pass": len(old_appr_fail), "approved": len(old_appr)},
            "new": {"false_pass": len(new_appr_fail), "approved": len(new_appr)},
        }}

    # deviations from the existing pattern:
    #   existing pattern per honest_reaudit: projections always pass the linear
    #   certificate yet leak; low noise (σ=1,2) passes-and-leaks; only high
    #   noise (σ=8, adult) genuinely survives; nothing that passes XGB/MLP is
    #   then caught only by the Tier-2 members.
    print("\n  per-cell deviations from the existing pattern:")
    devs = []

    def dev(cell, method, msg):
        devs.append({"cell": cell, "method": method, "note": msg})
        print(f"    {cell:<44} {method:<22} {msg}")

    for r in new_master:
        if r["method"].endswith(tuple(f"r={k}" for k in (1, 8, 16))) and \
                not r["linear_stopped"]:
            dev(r["cell"], r["method"],
                f"projection BREACHES the linear certificate (R²={r['linear_r2']:.3f}) "
                "— existing projections all passed it")
        if r["method"] == "noise σ=8" and not r["honest_stopped"]:
            dev(r["cell"], r["method"],
                f"σ=8 does NOT survive honest measurement (max XGB/MLP "
                f"{max(r['xgb_auc'], r['mlp_auc']):.3f}) — on adult it did")
        if r["honest_stopped"] and not r["tier2_stopped"]:
            dev(r["cell"], r["method"],
                f"passes XGB/MLP but Tier-2 reads above the bar "
                f"(LoRA={r['lora_auc']:.3f} LRT={r['lrt_auc']:.3f})")
        if r["linear_stopped"] and r["honest_stopped"]:
            dev(r["cell"], r["method"],
                f"GENUINE survivor (cert + suite agree stopped; "
                f"utility lift {r['task_lift']:+.4f})")
    if not devs:
        print("    none — every new verdict follows the existing pattern")
    out["pattern_deviations"] = devs

    # ---------- 4. coauthor summary table ----------
    print("\n" + "=" * 118)
    print("4. COAUTHOR SUMMARY — one line per new cell")
    print("=" * 118)
    print(f"{'cell':<44} {'source':<18} {'coupling':>8} {'floor':>7} {'cost_dur':>9} "
          f"{'re-audit (false-pass/approved, survivors)':>38}")
    print("-" * 118)
    summary = []
    by_cell_reaudit = {}
    for r in new_master:
        by_cell_reaudit.setdefault(r["cell"], []).append(r)
    floor_by_cell = {f["cell"]: f for f in floors}
    for r in sorted(new_cost["rows"], key=lambda r: r["predictor"]):
        cell = r["cell"]
        src = cell.split("/")[0]
        fl = floor_by_cell[cell]["floor"] if cell in floor_by_cell else float("nan")
        cd = "degen" if r["degenerate"] else (
            "None" if r["cost_durable"] is None else
            f"{r['cost_durable']:.3f}" + ("c" if r.get("censored_durable") else ""))
        ra = by_cell_reaudit.get(cell, [])
        appr = [x for x in ra if x["linear_stopped"]]
        fp = [x for x in appr if not x["honest_stopped"]]
        surv = [x for x in ra if x["honest_stopped"]]
        ra_s = f"{len(fp)}/{len(appr)} false-pass, {len(surv)}/{len(ra)} survive suite"
        print(f"{cell:<44} {src:<18} {r['predictor']:>8.3f} {fl:>7.3f} {cd:>9} {ra_s:>38}")
        summary.append({"cell": cell, "source": src, "coupling": r["predictor"],
                        "floor": fl, "cost_durable": r["cost_durable"],
                        "degenerate": r["degenerate"],
                        "censored_durable": r.get("censored_durable"),
                        "reaudit_false_pass": len(fp),
                        "reaudit_cert_approved": len(appr),
                        "reaudit_survivors": len(surv),
                        "reaudit_configs": len(ra)})
    print("-" * 118)
    cd_stats = stats["cost_durable"]
    print(f"\nHEADLINE: r {cd_stats['old_20']['pearson_r']:.3f} -> "
          f"{cd_stats['combined']['pearson_r']:.3f}, "
          f"ρ {cd_stats['old_20']['spearman_r']:.3f} -> "
          f"{cd_stats['combined']['spearman_r']:.3f} "
          f"(n {cd_stats['old_20']['n']} -> {cd_stats['combined']['n']}); "
          f"re-audit failures {old_failed}/{old_total} -> {comb_failed}/{comb_total}")
    out["summary_table"] = summary
    out["headline"] = {
        "r_old": cd_stats["old_20"]["pearson_r"],
        "r_combined": cd_stats["combined"]["pearson_r"],
        "rho_old": cd_stats["old_20"]["spearman_r"],
        "rho_combined": cd_stats["combined"]["spearman_r"],
        "reaudit_old": f"{old_failed}/{old_total}",
        "reaudit_combined": f"{comb_failed}/{comb_total}"}

    (RESULTS / "expansion_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\n[save] {RESULTS / 'expansion_analysis.json'}")


if __name__ == "__main__":
    main()
