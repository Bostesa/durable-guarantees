"""Dual scoring (macro-OVR and supported-pair) for every multiclass
configuration whose per-class scores are stored, with class counts visible.

No new runs. Two sources have stored probability matrices:
  * tpr_scores/*.npz      — our own channels (5 training seeds x 3 probe seeds)
  * tpr59_scores/*.npz    — audit configurations (the 59 failing ones)

Coverage is stated rather than assumed: baselines in Table 1 (LAFTR, VFAE,
LEACE, DANN-scrub, Obliviator, FARE) have NO stored per-class scores and no
cached representations, so their supported-pair numbers cannot be produced
without retraining and are reported here as NOT AVAILABLE.

Binary-attribute configurations are listed too, marked as such: for them
supported-pair equals macro-OVR by construction and no restriction applies.
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
RESULTS = REPO / "results"
# Generated intermediates; gitignored and regenerable.
# Override the location with DG_ARTIFACTS.
ARTIFACTS = Path(os.environ.get("DG_ARTIFACTS", REPO / "analysis"))
BAR = 0.55
MIN_CLASS_ROWS = 3000
NULL = {"all_pairs": 0.5556, "supported": 0.5217}
# hmda/race full-cell class counts (both HMDA cells share the attribute)
HMDA_RACE = [41134, 3796, 16968, 1344, 505]


def macro_ovr(y, p):
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))


def wp(y, p, keep=None):
    k = p.shape[1]
    if k == 2:
        a = float(roc_auc_score(y, p[:, 1]))
        return max(a, 1 - a)
    best = 0.0
    for i, j in combinations(range(k), 2):
        if keep is not None and not (keep[i] and keep[j]):
            continue
        m = (y == i) | (y == j)
        if m.sum() < 10 or len(set(y[m])) < 2:
            continue
        den = p[m, i] + p[m, j]
        s = np.where(den > 0, p[m, j] / np.maximum(den, 1e-12), 0.5)
        a = float(roc_auc_score((y[m] == j).astype(int), s))
        best = max(best, max(a, 1 - a))
    return best


def score_file(path, keep):
    """Handles both key layouts: tpr_scores uses '<surf>_<ARCH>_ts<i>_ps<j>_prob'
    (two surfaces, several training seeds); tpr59_scores uses
    '<ARCH>_ps<j>_prob' (single surface, one training seed)."""
    d = np.load(path)
    keys = [k for k in d.keys() if k.endswith("_prob")]
    out = {}
    for k in keys:
        ps = k.split("_ps")[1].split("_")[0]
        yk = f"y_ps{ps}"
        if yk not in d:
            continue
        surf = k.split("_")[0] if k.split("_")[0] in ("rep", "out") else "rep"
        y = d[yk].astype(int)
        p = d[k].astype(np.float64)
        b = out.setdefault(surf, dict(mac=[], allp=[], sup=[], n_classes=p.shape[1]))
        b["mac"].append(macro_ovr(y, p))
        b["allp"].append(wp(y, p, None))
        b["sup"].append(wp(y, p, keep) if keep else wp(y, p, None))
    return {s: dict(macro=float(max(b["mac"])), all_pairs=float(max(b["allp"])),
                    supported=float(max(b["sup"])), n_classes=int(b["n_classes"]))
            for s, b in out.items() if b["mac"]}


def main():
    keep_hmda = [c >= MIN_CLASS_ROWS for c in HMDA_RACE]
    rows = []

    print("HMDA race class counts:", HMDA_RACE,
          f"-> supported classes {[i for i, v in enumerate(keep_hmda) if v]} "
          f"(>= {MIN_CLASS_ROWS} rows); dropped "
          f"{[i for i, v in enumerate(keep_hmda) if not v]}")
    print(f"no-signal null: all-pairs {NULL['all_pairs']}, "
          f"supported {NULL['supported']}, bar {BAR}\n")

    ours = [("subspace_easy", "ours: subspace", "hmda/race", True),
            ("subspace_middle", "ours: subspace", "hmda/race", True),
            ("subspace_hard", "ours: subspace", "adult/sex", False),
            ("fullrank_tier1_easy", "ours: full-rank T1", "hmda/race", True),
            ("fullrank_tier1_middle", "ours: full-rank T1", "hmda/race", True),
            ("fullrank_tier1_hard", "ours: full-rank T1", "adult/sex", False),
            ("fullrank_tier2_easy", "ours: full-rank T2", "hmda/race", True),
            ("fullrank_tier2_middle", "ours: full-rank T2", "hmda/race", True),
            ("fullrank_tier2_hard", "ours: full-rank T2", "adult/sex", False)]

    hdr = ("%-24s %-8s %-6s %8s %10s %11s  %s"
           % ("configuration", "attr", "surf", "macro", "worst(all)",
              "worst(supp)", "supported verdict"))
    print(hdr); print("-" * len(hdr))
    for slug, label, attr, multi in ours:
        f = ARTIFACTS / "tpr_scores" / f"{slug}.npz"
        if not f.exists():
            continue
        sc = score_file(f, keep_hmda if multi else None)
        for surf in ("rep", "out"):
            if surf not in sc:
                continue
            s = sc[surf]
            v = ("n/a (binary)" if not multi
                 else ("passes" if s["supported"] <= BAR else "FAILS"))
            rows.append(dict(group="ours", config=slug, label=label, attribute=attr,
                             multiclass=multi, surface=surf, **s,
                             passes_macro=bool(s["macro"] <= BAR),
                             passes_supported=bool(s["supported"] <= BAR)))
            print("%-24s %-8s %-6s %8.4f %10.4f %11.4f  %s"
                  % (slug, attr, surf, s["macro"], s["all_pairs"], s["supported"], v))

    # audit configurations with stored scores
    print("\naudit configurations with stored per-class scores:")
    print(hdr); print("-" * len(hdr))
    for f in sorted((ARTIFACTS / "tpr59_scores").glob("*.npz")):
        stem = f.stem
        multi = stem.startswith("hmda")
        attr = "hmda/race" if multi else ("adult/sex" if stem.startswith("adult")
                                          else "binary")
        sc = score_file(f, keep_hmda if multi else None)
        for surf in sorted(sc):
            s = sc[surf]
            if s["n_classes"] == 2:
                attr, multi = attr if attr != "binary" else "binary", False
            v = ("n/a (binary)" if not multi
                 else ("passes" if s["supported"] <= BAR else "FAILS"))
            rows.append(dict(group="audit", config=stem, label="audit", attribute=attr,
                             multiclass=multi, surface=surf, **s,
                             passes_macro=bool(s["macro"] <= BAR),
                             passes_supported=bool(s["supported"] <= BAR)))
            print("%-24s %-8s %-6s %8.4f %10.4f %11.4f  %s"
                  % (stem[:24], attr, surf, s["macro"], s["all_pairs"],
                     s["supported"], v))

    multi_rows = [r for r in rows if r["multiclass"]]
    ours_multi = [r for r in multi_rows if r["group"] == "ours"]
    print(f"\nmulticlass (config, surface) pairs scored: {len(multi_rows)} "
          f"({len(ours_multi)} ours)")
    print(f"  ours: pass under macro-OVR {sum(r['passes_macro'] for r in ours_multi)}"
          f"/{len(ours_multi)}, under supported-pair "
          f"{sum(r['passes_supported'] for r in ours_multi)}/{len(ours_multi)}")
    print(f"  all:  pass under macro-OVR {sum(r['passes_macro'] for r in multi_rows)}"
          f"/{len(multi_rows)}, under supported-pair "
          f"{sum(r['passes_supported'] for r in multi_rows)}/{len(multi_rows)}")

    unavailable = ["LAFTR", "VFAE", "LEACE", "DANN-scrub", "Obliviator", "FARE"]
    print(f"\nNOT AVAILABLE without retraining (no stored per-class scores, no "
          f"cached representations): {', '.join(unavailable)}")
    print("  -> their Table-1 multiclass entries (2 HMDA cells x 2 tiers each) "
          "cannot be dual-scored from stored artefacts.")

    out = dict(experiment="multiclass_dual_report", bar=BAR,
               min_class_rows=MIN_CLASS_ROWS, hmda_race_counts=HMDA_RACE,
               supported_classes=[i for i, v in enumerate(keep_hmda) if v],
               nulls=NULL, rows=rows,
               n_multiclass=len(multi_rows), n_ours_multiclass=len(ours_multi),
               ours_pass_macro=sum(r["passes_macro"] for r in ours_multi),
               ours_pass_supported=sum(r["passes_supported"] for r in ours_multi),
               all_pass_macro=sum(r["passes_macro"] for r in multi_rows),
               all_pass_supported=sum(r["passes_supported"] for r in multi_rows),
               not_available=unavailable,
               not_available_reason=("no stored per-class probability matrices and "
                                     "no cached representations; requires retraining "
                                     "(~90 min for the two HMDA cells)"))
    (RESULTS / "multiclass_dual_report.json").write_text(
        json.dumps(out, indent=1, default=float))
    print(f"\n[save] {RESULTS / 'multiclass_dual_report.json'}")


if __name__ == "__main__":
    main()
