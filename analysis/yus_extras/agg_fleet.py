"""Aggregate fleet shards under BOTH seed conventions and BOTH pair criteria.

The point of this run: Table 1 scores every method by the mean over seeds
(two_tier_certification.py:152-156). Any comparison that scores baselines by a
max over seed combinations changes two things at once -- the pair criterion AND
the seed rule -- and a failing baseline can no longer be attributed to either.
The shards store the full per-(architecture, train seed, probe seed) matrix, so
both conventions come out of one pass and the comparison is never confounded.

Emits, for every (method, cell, surface):
  * paper convention   -- max over architectures of the mean over seeds
  * strict convention  -- max over every (arch, train seed, probe seed)
  * 5-seed and matched 3-seed views (FARE has only 3 stored releases, so the
    3-seed view is the one where every method is on identical footing)
  * macro-OVR, all-pairs worst-pair, supported-pair worst-pair

No training. Reads fleet_shards/ only.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BAR = 0.55
CRIT = ("macro", "all", "sup")
NULL = dict(all_pairs=0.5556, supported=0.5217)
TIER1 = ("XGB", "MLP", "LoRA")


def collapse(shards, surf, crit, seeds, include_lrt):
    """(paper, strict) for one (surface, criterion) over the given train seeds."""
    per_arch = defaultdict(list)
    for d in shards:
        if d["train_seed"] not in seeds:
            continue
        for a in TIER1:
            per_arch[a].extend(d[surf][a][crit])
        if include_lrt:
            per_arch["LRT"].extend(d[f"{surf}_lrt"][crit])
    if not per_arch:
        return None, None
    paper = max(float(np.mean(v)) for v in per_arch.values())
    strict = max(float(np.max(v)) for v in per_arch.values())
    return paper, strict


def main():
    shards = defaultdict(list)
    for f in sorted((HERE / "fleet_shards").glob("*.json")):
        d = json.loads(f.read_text())
        shards[(d["method"], d["cell"], d["knob"])].append(d)

    rows = []
    for (method, cell, knob), sh in sorted(shards.items()):
        sh.sort(key=lambda d: d["train_seed"])
        avail = sorted(d["train_seed"] for d in sh)
        for view, seeds in (("5seed", set(avail)), ("3seed", {0, 1, 2})):
            if not set(seeds) & set(avail):
                continue
            used = sorted(set(seeds) & set(avail))
            for tier, incl in ((1, False), (2, True)):
                r = dict(method=method, cell=cell, knob=knob, tier=tier,
                         view=view, train_seeds_used=used, n_seeds=len(used),
                         n_classes=len(sh[0]["class_counts"]),
                         class_counts=sh[0]["class_counts"],
                         supported_classes=sh[0]["supported_classes"])
                for surf in ("rep", "out"):
                    for c in CRIT:
                        p, s = collapse(sh, surf, c, set(used), incl)
                        r[f"{surf}_{c}_paper"] = p
                        r[f"{surf}_{c}_strict"] = s
                lifts = [d["lift"] for d in sh if d["train_seed"] in used]
                r["lift_mean"] = float(np.mean(lifts))
                r["lift_std"] = float(np.std(lifts))
                for conv in ("paper", "strict"):
                    r[f"passes_macro_{conv}"] = bool(
                        r[f"rep_macro_{conv}"] <= BAR and r[f"out_macro_{conv}"] <= BAR)
                    r[f"passes_supported_{conv}"] = bool(
                        r[f"rep_sup_{conv}"] <= BAR and r[f"out_sup_{conv}"] <= BAR)
                rows.append(r)

    w = 118
    for view in ("3seed", "5seed"):
        sel = [r for r in rows if r["view"] == view and r["tier"] == 1]
        if not sel:
            continue
        lbl = ("MATCHED 3-SEED VIEW (every method on identical footing)"
               if view == "3seed" else "5-SEED VIEW (FARE absent: only 3 releases stored)")
        print("\n" + "=" * w + f"\n{lbl}   tier 1, bar {BAR}\n" + "=" * w)
        print("%-12s %-7s %-6s %2s | %-19s | %-19s | %s"
              % ("method", "cell", "knob", "n", "macro (paper/strict)",
                 "supported (paper/str)", "verdict under supported"))
        print("-" * w)
        for r in sorted(sel, key=lambda r: (r["cell"], r["method"], r["knob"])):
            mp = max(r["rep_macro_paper"], r["out_macro_paper"])
            ms = max(r["rep_macro_strict"], r["out_macro_strict"])
            sp = max(r["rep_sup_paper"], r["out_sup_paper"])
            ss = max(r["rep_sup_strict"], r["out_sup_strict"])
            print("%-12s %-7s %-6g %2d | %8.4f / %8.4f | %8.4f / %8.4f | paper=%s strict=%s"
                  % (r["method"], r["cell"], r["knob"], r["n_seeds"], mp, ms, sp, ss,
                     "PASS" if sp <= BAR else "FAIL",
                     "PASS" if ss <= BAR else "FAIL"))

    # convention sensitivity: how much of any verdict change is the seed rule
    flips = []
    for r in rows:
        for c, key in (("macro", "macro"), ("supported", "sup")):
            pv = max(r[f"rep_{key}_paper"], r[f"out_{key}_paper"]) <= BAR
            sv = max(r[f"rep_{key}_strict"], r[f"out_{key}_strict"]) <= BAR
            if pv != sv:
                flips.append(dict(method=r["method"], cell=r["cell"], view=r["view"],
                                  tier=r["tier"], criterion=c,
                                  paper_passes=pv, strict_passes=sv))
    print(f"\nverdicts that change with the SEED CONVENTION alone: {len(flips)}")
    for f in flips:
        print(f"  {f['method']:<12} {f['cell']:<7} {f['view']} T{f['tier']} "
              f"{f['criterion']:<10} paper={'PASS' if f['paper_passes'] else 'FAIL'} "
              f"strict={'PASS' if f['strict_passes'] else 'FAIL'}")

    out = dict(experiment="fleet_dual_convention_dual_criterion", bar=BAR,
               conventions=dict(
                   paper="max over architectures of mean over seeds"
                         " -- two_tier_certification.py:152-156",
                   strict="max over all (arch, train seed, probe seed)"),
               nulls=NULL, rows=rows, seed_convention_flips=flips,
               not_covered=["Obliviator (separate code path)",
                            "FARE train seeds 3-4 (only 3 releases stored)"])
    (HERE / "fleet_dual.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {HERE / 'fleet_dual.json'}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
