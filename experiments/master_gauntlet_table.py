"""The MASTER GAUNTLET TABLE with provenance (paper appendix artifact).

Merges: results/baseline_gauntlet.json   (Exp 16: reimpl LAFTR, VFAE,
                                          DANN-scrub, LEACE)
        results/laftr_official.json      (Repair 1: LAFTR on official code)
        results/obliviator_gauntlet.json (Exp 17: Obliviator official code)
        results/two_tier_certification.json (ours, via load_ours)

Provenance vocabulary (exact strings used in the paper appendix):
  official code               — the authors' released implementation, run at
                                the data boundary only (LAFTR via TF1-compat
                                shim; Obliviator vendor-imported at pinned
                                commit; LEACE = the authors' concept-erasure
                                library, as published)
  validated reimplementation  — our implementation, validated against the
                                official code head-to-head (LAFTR multiclass
                                generalization, adult cell) or against the
                                paper's own published experiment (VFAE:
                                results/vfae_calibration.json — no official
                                release exists)
  generic mechanism           — a mechanism-class baseline, not a specific
                                published system (DANN-style adversarial
                                scrubbing)

Writes results/master_gauntlet_table.txt (verbatim table) and .json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"

PROV = {
    "LAFTR (official)": "official code",
    "LAFTR": "validated reimplementation",
    "VFAE": "validated reimplementation",
    "DANN-scrub": "generic mechanism",
    "LEACE": "official code (concept-erasure lib)",
    "Obliviator": "official code",
}

DISPLAY = {
    "LAFTR (official)": "LAFTR [official TF code]",
    "LAFTR": "LAFTR [reimpl, 5-class adversary]",
    "VFAE": "VFAE",
    "DANN-scrub": "DANN-style adv. scrubbing (full)",
    "LEACE": "LEACE",
    "Obliviator": "Obliviator (NeurIPS'25)",
}


def load_ours():
    d = json.loads((RESULTS_DIR / "two_tier_certification.json").read_text())
    ours = {}
    for c in d["e2e_cells"]:
        ours[c["cell"]] = dict(
            clean_lift=c["clean_lift"],
            t1=c["tier1"] and dict(point=f"σ={c['tier1']['sigma']:g}",
                                   lift=c["tier1"]["lift_mean"]),
            t2=c["tier2"] and dict(point=f"σ={c['tier2']['sigma']:g}",
                                   lift=c["tier2"]["lift_mean"]))
    return ours


def collect():
    """-> {cell: [ (baseline_key, cell_result_dict) ... ]}, ours"""
    per_cell = {}

    def add(cell, key, res):
        per_cell.setdefault(cell, []).append((key, res))

    g = json.loads((RESULTS_DIR / "baseline_gauntlet.json").read_text())
    lo = json.loads((RESULTS_DIR / "laftr_official.json").read_text())
    for c in lo["cells"]:
        add(c["cell"], "LAFTR (official)", c)
    for c in g["cells"]:
        for b in c["baselines"]:
            add(c["cell"], b["baseline"], b)
    ob = json.loads((RESULTS_DIR / "obliviator_gauntlet.json").read_text())
    for c in ob["cells"]:
        add(c["cell"], "Obliviator", c)
    meta = {c["cell"]: dict(predictor=c["predictor"],
                            clean_lift=c["clean_lift"])
            for c in g["cells"]}
    return per_cell, meta, load_ours()


def fmt_rows(key, res, cl, lines):
    prov = PROV[key]
    name = DISPLAY[key]
    for nm, t in (("T1", res["tier1"]), ("T2", res["tier2"])):
        if t["certified"]:
            c = t["cert"]
            lines.append(
                f"  {name + ' [' + nm + ']':<38} {c['label']:>16} "
                f"{100 * c['lift_best'] / max(cl, 1e-9):>9.0f}% "
                f"{c['rep_tier1_max']:>7.3f} {c['rep_tier2_max']:>7.3f} "
                f"{c['out_tier2_max']:>7.3f} {'CERTIFIES @' + nm:>15} "
                f"{prov}")
        else:
            r = t["closest"]
            tv = r["tier1_max"] if nm == "T1" else r["tier2_max"]
            lines.append(
                f"  {name + ' [' + nm + ']':<38} "
                f"{'(' + r['label'] + ')':>16} "
                f"{100 * r['lift_best'] / max(cl, 1e-9):>9.0f}% "
                f"{r['tier1_max']:>7.3f} {r['tier2_max']:>7.3f} "
                f"{r['out_max']:>7.3f} {'fails ' + f'{tv:.3f}':>15} "
                f"{prov}")


def main():
    per_cell, meta, ours = collect()
    lines = []
    lines.append("=" * 132)
    lines.append("MASTER GAUNTLET TABLE — published methods vs ours, two-tier "
                 "battery, with provenance (utility = % of clean e2e lift "
                 "kept)")
    lines.append("=" * 132)
    hdr = (f"  {'cell / method':<38} {'op. point':>16} {'util':>10} "
           f"{'T1 rep':>7} {'T2 rep':>7} {'out T2':>7} {'verdict':>15} "
           f"provenance")
    order = ["LAFTR (official)", "LAFTR", "VFAE", "DANN-scrub", "LEACE",
             "Obliviator"]
    for cell in ("hmda/race/loan_decision", "hmda/race/loan_amount_band",
                 "adult/sex/income"):
        m = meta[cell]
        cl = m["clean_lift"]
        lines.append(f"\n{cell}  (predictor {m['predictor']:.3f}, clean e2e "
                     f"lift {cl:+.4f})")
        lines.append(hdr)
        lines.append("-" * 132)
        entries = dict(per_cell[cell])
        for key in order:
            if key in entries:
                fmt_rows(key, entries[key], cl, lines)
        o = ours[cell]
        if o["t1"]:
            lines.append(f"  {'OURS e2e noised channel [T1]':<38} "
                         f"{o['t1']['point']:>16} "
                         f"{100 * o['t1']['lift'] / max(cl, 1e-9):>9.0f}% "
                         f"{'≤.55':>7} {'—':>7} {'—':>7} "
                         f"{'CERTIFIES @T1':>15} —")
        if o["t2"]:
            lines.append(f"  {'OURS e2e noised channel [T2]':<38} "
                         f"{o['t2']['point']:>16} "
                         f"{100 * o['t2']['lift'] / max(cl, 1e-9):>9.0f}% "
                         f"{'≤.55':>7} {'≤.55':>7} {'—':>7} "
                         f"{'CERTIFIES @T2':>15} —")
    lines.append("-" * 132)
    lines.append(
        "provenance: official code = authors' release run at the data "
        "boundary only; validated reimplementation = ours, validated "
        "head-to-head against the official code (LAFTR, adult cell) or "
        "against the paper's own Adult experiment (VFAE — no official "
        "release exists; results/vfae_calibration.json); generic mechanism "
        "= a mechanism-class baseline, not a published system (DANN-style "
        "scrubbing). T1 = max(XGB, MLP, LoRA-32) on the representation; T2 "
        "adds the informed Gaussian-LRT; out T2 = same battery on the task "
        "logits. Verdict on the representation at ≤0.55, 3-seed mean.")
    txt = "\n".join(lines)
    print(txt)
    (RESULTS_DIR / "master_gauntlet_table.txt").write_text(txt + "\n")
    (RESULTS_DIR / "master_gauntlet_table.json").write_text(json.dumps(
        dict(provenance=PROV, cells={c: {k: dict(
            tier1=dict(certified=v["tier1"]["certified"]),
            tier2=dict(certified=v["tier2"]["certified"]))
            for k, v in dict(per_cell[c]).items()} for c in per_cell}),
        indent=2))
    print(f"\n[save] {RESULTS_DIR / 'master_gauntlet_table.txt'} and .json")


if __name__ == "__main__":
    main()
