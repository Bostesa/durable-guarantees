"""Support-restricted worst-pair rescoring of our own configurations.

The all-pairs worst-pair statistic is unusable at HMDA's class sizes: with
zero-information features it reads a null max of 0.5556, above the 0.55 bar
(worstpair_null.json). The null is driven entirely by pairs involving the two
rarest race classes (3: 336 test rows, 4: 126 test rows). Restricted to pairs
whose BOTH classes have adequate support the null max falls to 0.5217, which
leaves usable headroom under the bar.

This rescores our configurations under the support-restricted criterion, from
the stored per-class probability matrices in tpr_scores/ — no retraining.
MIN_CLASS_ROWS is the support threshold on the full-cell class count.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
SCORES = HERE / "tpr_scores"
BAR = 0.55
MIN_CLASS_ROWS = 3000          # HMDA race: keeps classes 0 (41134), 1 (3796), 2 (16968)

OURS = [("subspace_easy", "easy", "Tier 1", ["XGB", "MLP", "LoRA"]),
        ("subspace_middle", "middle", "Tier 1", ["XGB", "MLP", "LoRA"]),
        ("fullrank_tier1_easy", "easy", "Tier 1", ["XGB", "MLP", "LoRA"]),
        ("fullrank_tier1_middle", "middle", "Tier 1", ["XGB", "MLP", "LoRA"]),
        ("fullrank_tier2_easy", "easy", "Tier 2", ["XGB", "MLP", "LoRA", "LRT"]),
        ("fullrank_tier2_middle", "middle", "Tier 2", ["XGB", "MLP", "LoRA", "LRT"])]
# full-cell class counts for hmda/race (both cells share the attribute)
HMDA_RACE_COUNTS = [41134, 3796, 16968, 1344, 505]


def macro_ovr(y, p):
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))


def pair_auc(y, p, i, j):
    m = (y == i) | (y == j)
    if m.sum() < 10 or len(set(y[m])) < 2:
        return None
    den = p[m, i] + p[m, j]
    s = np.where(den > 0, p[m, j] / np.maximum(den, 1e-12), 0.5)
    a = float(roc_auc_score((y[m] == j).astype(int), s))
    return max(a, 1.0 - a)


def worst(y, p, keep):
    best, arg = 0.0, None
    for i, j in combinations(range(p.shape[1]), 2):
        if keep is not None and not (keep[i] and keep[j]):
            continue
        a = pair_auc(y, p, i, j)
        if a is not None and a > best:
            best, arg = a, (i, j)
    return best, arg


def main():
    keep = [c >= MIN_CLASS_ROWS for c in HMDA_RACE_COUNTS]
    kept = [i for i, k in enumerate(keep) if k]
    print(f"support threshold: class must have >= {MIN_CLASS_ROWS} rows")
    print(f"HMDA race counts {HMDA_RACE_COUNTS} -> kept classes {kept}, "
          f"dropped {[i for i in range(5) if i not in kept]}")
    print(f"null max (all pairs) 0.5556  vs  (supported pairs) 0.5217  "
          f"[bar {BAR}]\n")

    hdr = ("%-22s %-6s %-8s %10s %12s %12s  %s"
           % ("config", "tier", "surface", "macro", "worst(all)",
              "worst(supp)", "verdict"))
    print(hdr); print("-" * len(hdr))
    rows = []
    for slug, cell, tier, atts in OURS:
        d = np.load(SCORES / f"{slug}.npz")
        seeds = sorted({(k.split("_ts")[1].split("_ps")[0],
                         k.split("_ps")[1].split("_")[0])
                        for k in d.keys() if "_ts" in k and "_ps" in k})
        for surf in ("rep", "out"):
            mac, wall, wsup = [], [], []
            for ts, psx in seeds:
                yk = f"y_ps{psx}"
                if yk not in d:
                    continue
                y = d[yk].astype(int)
                for a in atts:
                    k = f"{surf}_{a}_ts{ts}_ps{psx}_prob"
                    if k not in d:
                        continue
                    p = d[k].astype(np.float64)
                    mac.append(macro_ovr(y, p))
                    wall.append(worst(y, p, None)[0])
                    wsup.append(worst(y, p, keep)[0])
            if not mac:
                continue
            r = dict(config=slug, cell=cell, tier=tier, surface=surf,
                     macro_max=float(max(mac)),
                     worst_all_max=float(max(wall)),
                     worst_supported_max=float(max(wsup)),
                     passes_macro=bool(max(mac) <= BAR),
                     passes_worst_all=bool(max(wall) <= BAR),
                     passes_worst_supported=bool(max(wsup) <= BAR))
            rows.append(r)
            v = ("passes" if r["passes_worst_supported"] else "FAILS")
            print("%-22s %-6s %-8s %10.4f %12.4f %12.4f  %s"
                  % (slug, tier, surf, r["macro_max"], r["worst_all_max"],
                     r["worst_supported_max"], v))

    n = len(rows)
    print(f"\n  of {n} (config, surface) pairs:")
    print(f"    pass under macro-OVR max         : {sum(r['passes_macro'] for r in rows)}")
    print(f"    pass under worst-pair, all pairs : {sum(r['passes_worst_all'] for r in rows)}")
    print(f"    pass under worst-pair, supported : {sum(r['passes_worst_supported'] for r in rows)}")
    out = dict(experiment="worstpair_support_restricted",
               min_class_rows=MIN_CLASS_ROWS, kept_classes=kept,
               hmda_race_counts=HMDA_RACE_COUNTS, bar=BAR,
               null_all_pairs=0.5556, null_supported_pairs=0.5217,
               rows=rows,
               n_pass_macro=sum(r["passes_macro"] for r in rows),
               n_pass_worst_all=sum(r["passes_worst_all"] for r in rows),
               n_pass_worst_supported=sum(r["passes_worst_supported"] for r in rows))
    (HERE / "worstpair_supported.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {HERE / 'worstpair_supported.json'}")


if __name__ == "__main__":
    main()
