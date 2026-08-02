"""DIAGNOSTICS 1 + 2 — max-over-seeds re-scoring, and worst-pair AUC.

Recomputation from stored artefacts only; nothing is retrained.

D1 MAX-OVER-SEEDS. The bar is currently applied to a MEAN over seeds. This
re-applies it to the MAX. Two seed axes exist and they are reported
separately because they are not interchangeable:
  * audit survivors (tpr_extension.json:results.auc_records) store one train
    seed x 3 probe seeds -> max over PROBE seeds.
  * Table-1 certified baselines (baseline_gauntlet.json:...cert.per_seed) and
    our own channels store 3 (or 5) TRAIN seeds, each already a probe-mean
    -> max over TRAIN seeds.

D2 WORST-PAIR AUC. For multiclass attributes the reported number is a macro
one-vs-rest average. Worst-pair AUC = max over label pairs (i, j) of the
binary AUC on the subset with true label in {i, j}, scored by
p_i / (p_i + p_j). For binary attributes the two coincide, so only the
5-class hmda/race cells can move. Computed from the stored per-class
probability matrices in analysis/tpr_scores/*.npz, which hold
every attacker x train seed x probe seed on both surfaces.
"""
from __future__ import annotations
import os

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# Generated intermediates; gitignored and regenerable.
# Override the location with DG_ARTIFACTS.
ARTIFACTS = Path(os.environ.get("DG_ARTIFACTS", REPO / "analysis"))
RESULTS = REPO / "results"
SCORES = ARTIFACTS / "tpr_scores"
BAR = 0.55

# our own configurations that the paper reports as passing, and the tier whose
# bar they are claimed to clear
OURS = [
    ("subspace_easy", "easy", "Tier 1", ["XGB", "MLP", "LoRA"]),
    ("subspace_middle", "middle", "Tier 1", ["XGB", "MLP", "LoRA"]),
    ("subspace_hard", "hard", "Tier 1", ["XGB", "MLP", "LoRA"]),
    ("fullrank_tier1_easy", "easy", "Tier 1", ["XGB", "MLP", "LoRA"]),
    ("fullrank_tier1_middle", "middle", "Tier 1", ["XGB", "MLP", "LoRA"]),
    ("fullrank_tier1_hard", "hard", "Tier 1", ["XGB", "MLP", "LoRA"]),
    ("fullrank_tier2_easy", "easy", "Tier 2", ["XGB", "MLP", "LoRA", "LRT"]),
    ("fullrank_tier2_middle", "middle", "Tier 2", ["XGB", "MLP", "LoRA", "LRT"]),
    ("fullrank_tier2_hard", "hard", "Tier 2", ["XGB", "MLP", "LoRA", "LRT"]),
]


def macro_ovr(y, p):
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))


def worst_pair(y, p):
    """max over label pairs of the binary AUC on that pair's subset."""
    k = p.shape[1]
    if k == 2:
        return float(roc_auc_score(y, p[:, 1])), (0, 1)
    best, arg = 0.0, None
    for i, j in combinations(range(k), 2):
        m = (y == i) | (y == j)
        if m.sum() < 10 or len(set(y[m])) < 2:
            continue
        den = p[m, i] + p[m, j]
        s = np.where(den > 0, p[m, j] / np.maximum(den, 1e-12), 0.5)
        a = float(roc_auc_score((y[m] == j).astype(int), s))
        a = max(a, 1.0 - a)              # orientation-free
        if a > best:
            best, arg = a, (i, j)
    return best, arg


def load_scores(slug):
    d = np.load(SCORES / f"{slug}.npz")
    keys = list(d.keys())
    seeds = sorted({(k.split("_ts")[1].split("_ps")[0],
                     k.split("_ps")[1].split("_")[0])
                    for k in keys if "_ts" in k and "_ps" in k})
    return d, seeds


def main():
    out = {"experiment": "maxseed_and_worstpair", "bar": BAR}

    # ================= D1a: audit survivors, max over PROBE seeds ==========
    tx = json.loads((RESULTS / "tpr_extension.json").read_text())
    ar = tx["results"]["auc_records"]
    surv_groups = {"hr_noise_s4": "adult/sex noise sigma_rel=4",
                   "hr_noise_s8": "adult/sex noise sigma_rel=8 (x2 rows)",
                   "exp_diabetes_hospital_race_readmit_30_days": "diabetes_hospital/race noise s8",
                   "exp_folktables_race_employment": "folktables/race/employment noise s8",
                   "exp_folktables_race_public_coverage": "folktables/race/public_coverage noise s8",
                   "exp_folktables_race_income": "folktables/race/income noise s8",
                   "exp_lawschool_sex_pass_bar": "lawschool/sex/pass_bar noise s8"}
    print("D1a  AUDIT SURVIVORS — bar re-applied to MAX over probe seeds")
    print("%-46s %8s %8s %8s  %s" % ("configuration", "mean", "max", "delta", "verdict"))
    print("-" * 92)
    rows_a = []
    for g, label in surv_groups.items():
        recs = [r for r in ar if r["group"] == g and r["attacker"] != "LRT"]
        if not recs:
            continue
        mean_by_att = {}
        for a in sorted({r["attacker"] for r in recs}):
            v = [r["auc"] for r in recs if r["attacker"] == a]
            mean_by_att[a] = (float(np.mean(v)), float(np.max(v)))
        m = max(v[0] for v in mean_by_att.values())
        mx = max(v[1] for v in mean_by_att.values())
        flips = (m <= BAR) and (mx > BAR)
        rows_a.append(dict(group=g, label=label, tier1_mean=m, tier1_max=mx,
                           passes_mean=bool(m <= BAR), passes_max=bool(mx <= BAR),
                           flips=bool(flips), per_attacker=mean_by_att))
        print("%-46s %8.4f %8.4f %+8.4f  %s" %
              (label, m, mx, mx - m,
               "FLIPS to fail" if flips else ("passes" if mx <= BAR else "already failed")))
    n_pass_mean = sum(r["passes_mean"] for r in rows_a)
    n_pass_max = sum(r["passes_max"] for r in rows_a)
    print(f"\n  distinct survivor configurations: {len(rows_a)}")
    print(f"  pass under mean: {n_pass_mean}   pass under max: {n_pass_max}")
    out["d1a_survivors"] = dict(rows=rows_a, n_distinct=len(rows_a),
                                n_pass_mean=n_pass_mean, n_pass_max=n_pass_max)

    # ================= D1b: Table-1 certified baselines, max over TRAIN ====
    bg = json.loads((RESULTS / "baseline_gauntlet.json").read_text())
    print("\nD1b  TABLE-1 CERTIFIED CELLS — bar re-applied to MAX over train seeds")
    print("%-34s %-6s %-22s %8s %8s  %s"
          % ("cell", "tier", "method", "mean", "max", "verdict"))
    print("-" * 96)
    rows_b = []
    for cell in bg["cells"]:
        for b in cell["baselines"]:
            for tier in ("tier1", "tier2"):
                t = b.get(tier) or {}
                if not t.get("certified"):
                    continue
                cert = t["cert"]
                ps = cert["per_seed"]
                keys = (["rep_xgb", "rep_mlp", "rep_lora", "out_xgb", "out_mlp", "out_lora"]
                        + (["rep_lrt", "out_lrt"] if tier == "tier2" else []))
                mean_max = max(float(np.mean(ps[k])) for k in keys if k in ps)
                max_max = max(float(np.max(ps[k])) for k in keys if k in ps)
                flips = (mean_max <= BAR) and (max_max > BAR)
                rows_b.append(dict(cell=cell["cell"], tier=tier,
                                   method=b["baseline"], label=cert.get("label"),
                                   mean_max=mean_max, max_max=max_max,
                                   passes_mean=bool(mean_max <= BAR),
                                   passes_max=bool(max_max <= BAR), flips=bool(flips)))
                print("%-34s %-6s %-22s %8.4f %8.4f  %s"
                      % (cell["cell"][:34], tier, f"{b['baseline']} {cert.get('label','')}"[:22],
                         mean_max, max_max,
                         "FLIPS to fail" if flips else ("passes" if max_max <= BAR else "already failed")))
    print(f"\n  certified cells: {len(rows_b)}   pass under mean: "
          f"{sum(r['passes_mean'] for r in rows_b)}   pass under max: "
          f"{sum(r['passes_max'] for r in rows_b)}")
    out["d1b_table1"] = dict(rows=rows_b,
                             n=len(rows_b),
                             n_pass_mean=sum(r["passes_mean"] for r in rows_b),
                             n_pass_max=sum(r["passes_max"] for r in rows_b))

    # ================= D1c + D2: our own channels ==========================
    print("\nD1c/D2  OUR OWN PASSING CONFIGURATIONS "
          "(max over train seeds; worst-pair vs macro-OVR)")
    hdr = ("%-22s %-6s %-8s %8s %8s %10s %10s  %s"
           % ("config", "tier", "surface", "macroMean", "macroMax", "worstMean",
              "worstMax", "verdict"))
    print(hdr); print("-" * len(hdr))
    rows_c = []
    for slug, cell, tier, atts in OURS:
        d, seeds = load_scores(slug)
        for surf in ("rep", "out"):
            per_ts_macro, per_ts_worst = {}, {}
            for ts, psx in seeds:
                key_y = f"y_ps{psx}"
                if key_y not in d:
                    continue
                y = d[key_y].astype(int)
                for a in atts:
                    k = f"{surf}_{a}_ts{ts}_ps{psx}_prob"
                    if k not in d:
                        continue
                    p = d[k].astype(np.float64)
                    per_ts_macro.setdefault(ts, []).append(macro_ovr(y, p))
                    per_ts_worst.setdefault(ts, []).append(worst_pair(y, p)[0])
            if not per_ts_macro:
                continue
            # per train seed: max over attackers and probe seeds
            ts_macro = {ts: max(v) for ts, v in per_ts_macro.items()}
            ts_worst = {ts: max(v) for ts, v in per_ts_worst.items()}
            mac_mean, mac_max = float(np.mean(list(ts_macro.values()))), float(max(ts_macro.values()))
            wor_mean, wor_max = float(np.mean(list(ts_worst.values()))), float(max(ts_worst.values()))
            flips_seed = (mac_mean <= BAR) and (mac_max > BAR)
            flips_pair = (mac_mean <= BAR) and (wor_mean > BAR)
            rows_c.append(dict(config=slug, cell=cell, tier=tier, surface=surf,
                               n_classes=int(d[f"y_ps{seeds[0][1]}"].max()) + 1,
                               macro_mean=mac_mean, macro_max=mac_max,
                               worst_mean=wor_mean, worst_max=wor_max,
                               passes_macro_mean=bool(mac_mean <= BAR),
                               passes_macro_max=bool(mac_max <= BAR),
                               passes_worst_mean=bool(wor_mean <= BAR),
                               flips_on_seed=bool(flips_seed),
                               flips_on_pair=bool(flips_pair)))
            v = []
            if flips_seed:
                v.append("seed-max FLIPS")
            if flips_pair:
                v.append("worst-pair FLIPS")
            print("%-22s %-6s %-8s %8.4f %8.4f %10.4f %10.4f  %s"
                  % (slug, tier, surf, mac_mean, mac_max, wor_mean, wor_max,
                     ", ".join(v) if v else ("passes" if mac_mean <= BAR else "already failed")))
    out["d1c_d2_ours"] = rows_c

    print("\n  summary of our own configs (by (config, surface)):")
    print("    pass on macro-mean : %d" % sum(r["passes_macro_mean"] for r in rows_c))
    print("    pass on macro-MAX  : %d" % sum(r["passes_macro_max"] for r in rows_c))
    print("    pass on worst-pair : %d" % sum(r["passes_worst_mean"] for r in rows_c))
    print("    flip on seed-max   : %s" % [r["config"] + "/" + r["surface"]
                                           for r in rows_c if r["flips_on_seed"]])
    print("    flip on worst-pair : %s" % [r["config"] + "/" + r["surface"]
                                           for r in rows_c if r["flips_on_pair"]])

    (RESULTS / "maxseed_worstpair.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {RESULTS / 'maxseed_worstpair.json'}")


if __name__ == "__main__":
    main()
