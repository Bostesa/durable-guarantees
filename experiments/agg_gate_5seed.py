"""Aggregate the 5-seed gate shards under BOTH conventions and score
gate_5seed_prediction.md (commit 66390e1). No training; reads shards only.

paper  = max over architectures of the mean over all (train seed, probe seed)
         values -- matches two_tier_certification.py:152-156
strict = max over every (arch, train seed, probe seed) value -- what the
         supported-pair sweep used
"""
from __future__ import annotations
import os

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"
# Generated intermediates; gitignored and regenerable.
# Override the location with DG_ARTIFACTS.
ARTIFACTS = Path(os.environ.get("DG_ARTIFACTS", REPO / "analysis"))
BAR = 0.55
CRIT = ("macro", "all", "sup")
# 3-seed readings from worstpair_supported_sweep.json (commit 5e47b8b)
SWEEP3 = {"easy": dict(rep_sup=0.5267, out_sup=0.5459, util=49.7, lift=0.0111),
          "middle": dict(rep_sup=0.5275, out_sup=0.5460, util=76.2, lift=0.3117)}


def main():
    tt = json.loads((REPO / "results" / "two_tier_certification.json").read_text())
    nm = {"easy": "hmda/race/loan_decision", "middle": "hmda/race/loan_amount_band"}
    clean = {k: next(c for c in tt["e2e_cells"] if c["cell"] == nm[k])["clean_lift"]
             for k in nm}

    shards = {}
    for f in sorted((ARTIFACTS / "gate_shards").glob("*.json")):
        d = json.loads(f.read_text())
        shards.setdefault(d["cell"], []).append(d)
    for v in shards.values():
        v.sort(key=lambda d: d["train_seed"])

    cells = []
    for cellk, sh in sorted(shards.items()):
        row = dict(cell=cellk, n_train_seeds=len(sh), sigma=sh[0]["sigma"],
                   rank=sh[0]["rank"], lam=sh[0]["lam"],
                   train_seeds=[d["train_seed"] for d in sh],
                   class_counts=sh[0]["class_counts"],
                   supported_classes=sh[0]["supported_classes"])
        for surf in ("rep", "out"):
            for c in CRIT:
                # matrix[arch] = list over (train seed x probe seed)
                per_arch = {}
                for a in sh[0]["archs"]:
                    vals = [x for d in sh for x in d[surf][a][c]]
                    per_arch[a] = vals
                paper = max(float(np.mean(v)) for v in per_arch.values())
                strict = max(float(np.max(v)) for v in per_arch.values())
                row[f"{surf}_{c}_paper"] = paper
                row[f"{surf}_{c}_strict"] = strict
                row[f"{surf}_{c}_argmax_arch"] = max(
                    per_arch, key=lambda a: np.mean(per_arch[a]))
        lifts = [d["lift"] for d in sh]
        row["lift_per_seed"] = lifts
        row["lift_mean"] = float(np.mean(lifts))
        row["lift_std"] = float(np.std(lifts))
        row["lift_min"] = float(np.min(lifts))
        row["util_pct"] = 100.0 * float(np.mean(lifts)) / clean[cellk]
        row["clean_lift"] = clean[cellk]
        row["n_seeds_negative_lift"] = int(sum(1 for x in lifts if x < 0))
        for conv in ("paper", "strict"):
            row[f"passes_{conv}"] = bool(row[f"rep_sup_{conv}"] <= BAR
                                         and row[f"out_sup_{conv}"] <= BAR)
        cells.append(row)

    # ---- scoring -----------------------------------------------------------
    g1 = {r["cell"]: r["passes_paper"] for r in cells}
    g2_fail = {r["cell"]: not r["passes_strict"] for r in cells}
    g3 = {}
    for r in cells:
        gr = r["rep_sup_strict"] - r["rep_sup_paper"]
        go = r["out_sup_strict"] - r["out_sup_paper"]
        g3[r["cell"]] = dict(rep_gap=gr, out_gap=go, out_larger=bool(go > gr))
    g4 = {}
    for r in cells:
        d = abs(r["util_pct"] - SWEEP3[r["cell"]]["util"])
        g4[r["cell"]] = dict(util_5seed=r["util_pct"],
                             util_3seed=SWEEP3[r["cell"]]["util"],
                             delta_pp=d, within_3pp=bool(d <= 3.0))
    scoring = dict(
        G1=dict(per_cell=g1, verdict="HELD" if all(g1.values()) else "FALSIFIED"),
        G2=dict(per_cell_fails=g2_fail,
                verdict="HELD" if any(g2_fail.values()) else "FALSIFIED"),
        G3=dict(per_cell=g3,
                verdict="HELD" if all(v["out_larger"] for v in g3.values())
                else "FALSIFIED"),
        G4=dict(per_cell=g4,
                verdict="HELD" if all(v["within_3pp"] for v in g4.values())
                else "FALSIFIED"))

    w = 96
    print("=" * w)
    print("5-SEED RECERTIFICATION  (gate_5seed_prediction.md @ 66390e1)")
    print("=" * w)
    for r in cells:
        print(f"\n### {r['cell']}  sigma={r['sigma']:g} rank={r['rank']} "
              f"lam={r['lam']:g}  seeds={r['train_seeds']}")
        print(f"{'':22} {'paper conv':>12} {'strict conv':>12} {'3-seed sweep':>13}")
        for surf, lbl in (("rep", "representation"), ("out", "output")):
            s3 = SWEEP3[r["cell"]][f"{surf}_sup"]
            print(f"  {lbl+' supported':<20} {r[f'{surf}_sup_paper']:>12.4f} "
                  f"{r[f'{surf}_sup_strict']:>12.4f} {s3:>13.4f}")
        print(f"  {'macro-OVR (rep)':<20} {r['rep_macro_paper']:>12.4f} "
              f"{r['rep_macro_strict']:>12.4f}")
        print(f"  verdict: paper={'PASS' if r['passes_paper'] else 'FAIL'}  "
              f"strict={'PASS' if r['passes_strict'] else 'FAIL'}")
        print(f"  utility: {r['util_pct']:+.1f}% of clean lift "
              f"(3-seed: {SWEEP3[r['cell']]['util']:+.1f}%)  "
              f"lift {r['lift_mean']:+.4f}+-{r['lift_std']:.4f} "
              f"min {r['lift_min']:+.4f}")
        if r["n_seeds_negative_lift"]:
            print(f"  !! {r['n_seeds_negative_lift']}/{r['n_train_seeds']} "
                  f"training seeds produced NEGATIVE utility")
    print("\n" + "=" * w + "\nSCORING\n" + "=" * w)
    for k in ("G1", "G2", "G3", "G4"):
        print(f"  {k}: {scoring[k]['verdict']}")
    print(f"    G1 pass under paper convention: {g1}")
    print(f"    G2 any fail under strict: {g2_fail}")
    print(f"    G3 gaps: { {c: (round(v['rep_gap'], 4), round(v['out_gap'], 4)) for c, v in g3.items()} }")
    print(f"    G4 utility delta pp: { {c: round(v['delta_pp'], 1) for c, v in g4.items()} }")

    out = dict(experiment="gate_5seed_recertification",
               prediction_file="gate_5seed_prediction.md",
               prediction_commit="66390e1", bar=BAR,
               conventions=dict(
                   paper="max over architectures of mean over (train,probe) seeds"
                         " -- two_tier_certification.py:152-156",
                   strict="max over all (arch, train seed, probe seed)"),
               comparator_3seed="worstpair_supported_sweep.json @ 5e47b8b",
               cells=cells, scoring=scoring)
    (RESULTS / "gate_5seed.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {RESULTS / 'gate_5seed.json'}")


if __name__ == "__main__":
    main()
