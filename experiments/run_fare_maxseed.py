"""D1b completion — FARE's six certified cells under max-over-train-seeds.

baseline_gauntlet.json holds only one certified cell (VFAE easy Tier 1); the
other six of the paper's seven Table-1 passes are FARE, stored separately in
fare_gauntlet.json. Same per_seed structure, same treatment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"
BAR = 0.55


def main():
    d = json.loads((RESULTS / "fare_gauntlet.json").read_text())
    rows = []
    print("%-34s %-6s %-16s %8s %8s  %s"
          % ("cell", "tier", "operating pt", "mean", "max", "verdict"))
    print("-" * 92)
    for cell in d["cells"]:
        for tier in ("tier1", "tier2"):
            t = cell.get(tier) or {}
            if not t.get("certified"):
                continue
            cert = t["cert"]
            ps = cert["per_seed"]
            keys = ["rep_xgb", "rep_mlp", "rep_lora", "out_xgb", "out_mlp", "out_lora"]
            if tier == "tier2":
                keys += ["rep_lrt", "out_lrt"]
            keys = [k for k in keys if k in ps]
            mean_max = max(float(np.mean(ps[k])) for k in keys)
            max_max = max(float(np.max(ps[k])) for k in keys)
            flips = (mean_max <= BAR) and (max_max > BAR)
            rows.append(dict(cell=cell["cell"], tier=tier,
                             label=cert.get("label"), mean_max=mean_max,
                             max_max=max_max, passes_mean=bool(mean_max <= BAR),
                             passes_max=bool(max_max <= BAR), flips=bool(flips)))
            print("%-34s %-6s %-16s %8.4f %8.4f  %s"
                  % (cell["cell"][:34], tier, str(cert.get("label"))[:16],
                     mean_max, max_max,
                     "FLIPS to fail" if flips else
                     ("passes" if max_max <= BAR else "already failed")))
    print(f"\n  FARE certified cells: {len(rows)}   pass under mean: "
          f"{sum(r['passes_mean'] for r in rows)}   pass under max: "
          f"{sum(r['passes_max'] for r in rows)}")
    out = dict(experiment="fare_maxseed", bar=BAR, rows=rows,
               n=len(rows), n_pass_mean=sum(r["passes_mean"] for r in rows),
               n_pass_max=sum(r["passes_max"] for r in rows))
    (RESULTS / "fare_maxseed.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {RESULTS / 'fare_maxseed.json'}")


if __name__ == "__main__":
    main()
