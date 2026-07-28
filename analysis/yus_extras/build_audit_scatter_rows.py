"""TASK 2 — emit one row per approved audit configuration, from stored files only.

Enumeration is the one run_tpr_failing59.py used and asserts the same way:
  existing 21 = every row of honest_reaudit.json["master"] (all at-rest approved)
  new      46 = expansion_reaudit_paperframing.json rows with approved_at_rest
  total    67, of which 59 fail (best Tier-1 > 0.55) and 8 survive.

INSTRUMENT NOTE (the reason there are two certificate columns):
`old_r2` in honest_reaudit.json is the LoRA-ATTACKED linear R^2, not the
certificate as issued at rest. Verified: honest_reaudit old_r2 at sigma=0 is
0.10362235128290258, byte-identical to noise_channel_test.json
attacked_linear_r2_mean and to smart_erasure_adult.json attacked_r2 for
"no erasure (frozen)". The at-rest reading for that same configuration is
noise_channel_test.json baseline_linear_r2 = 0.034942350219597684. The
paperframing file states approval is on the AT-REST reading
("approved iff the weak certificate AS ISSUED AT REST reads R^2 <= tau").

So this script emits both, each tagged with its source file, and never mixes
them in one column. Nothing is imputed and nothing is recomputed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "results"
OUT = RESULTS / "audit_scatter_rows.csv"
TAU, BAR = 0.05, 0.55


def dataset_of(cell: str) -> str:
    return cell.split("/")[0]


def mechanism_of(method: str) -> str:
    return "noise" if "noise" in method.lower() else "projection"


def main():
    hr = json.loads((RESULTS / "honest_reaudit.json").read_text())
    pf = json.loads((RESULTS / "expansion_reaudit_paperframing.json").read_text())
    nct = json.loads((RESULTS / "noise_channel_test.json").read_text())

    # at-rest readings that ARE stored for the existing adult noise sweep,
    # keyed by the sigma_rel appearing in the method string
    at_rest_adult_noise = {r["sigma_rel"]: r["baseline_linear_r2"]
                           for r in nct["rows"]}

    def adult_noise_sigma(method: str):
        """'noise σ=0.25' / 'noise sigma=1.0' -> 0.25 / 1.0"""
        tail = method.replace("σ", "sigma").split("sigma=")[-1].strip()
        try:
            return float(tail)
        except ValueError:
            return None

    rows = []

    # ---- existing 21 -------------------------------------------------------
    for r in hr["master"]:
        cell = r.get("cell", "adult/sex")
        method = r["method"]
        mech = mechanism_of(method)
        tier1 = max(r["xgb_auc"], r["mlp_auc"])
        at_rest, at_rest_src = "", ""
        if cell == "adult/sex" and mech == "noise":
            s = adult_noise_sigma(method)
            if s in at_rest_adult_noise:
                at_rest = at_rest_adult_noise[s]
                at_rest_src = "noise_channel_test.json:rows.baseline_linear_r2"
        rows.append(dict(
            group="existing", cell=cell, dataset=dataset_of(cell),
            method=method, mechanism=mech,
            cert_r2_at_rest=at_rest, cert_r2_at_rest_source=at_rest_src,
            cert_r2_attacked=r["old_r2"],
            cert_r2_attacked_source="honest_reaudit.json:master.old_r2",
            tier1_max=tier1,
            tier1_source="honest_reaudit.json:master.max(xgb_auc,mlp_auc)",
            fails_bar=int(tier1 > BAR)))

    # ---- new 46 -----------------------------------------------------------
    for r in pf["rows"]:
        if not r.get("approved_at_rest"):
            continue
        cell, method = r["cell"], r["method"]
        tier1 = r["tier1_max"]
        rows.append(dict(
            group="new", cell=cell, dataset=dataset_of(cell),
            method=method, mechanism=mechanism_of(method),
            cert_r2_at_rest=r["at_rest_linear_r2"],
            cert_r2_at_rest_source=(
                "expansion_reaudit_paperframing.json:rows.at_rest_linear_r2"),
            cert_r2_attacked=r["attacked_linear_r2"],
            cert_r2_attacked_source=(
                "expansion_reaudit_paperframing.json:rows.attacked_linear_r2"),
            tier1_max=tier1,
            tier1_source="expansion_reaudit_paperframing.json:rows.tier1_max",
            fails_bar=int(r["fails_paperframing"])))

    # ---- asserts, the same way tpr_failing59 asserts -----------------------
    assert len(rows) == 67, len(rows)
    assert sum(1 for x in rows if x["group"] == "existing") == 21
    assert sum(1 for x in rows if x["group"] == "new") == 46
    assert sum(x["fails_bar"] for x in rows) == 59, sum(x["fails_bar"] for x in rows)
    assert pf["combined"] == {"approved": 67, "fail": 59, "survive": 8,
                              "total_reaudited": 102}

    have_at_rest = [x for x in rows if x["cert_r2_at_rest"] != ""]
    missing = [x for x in rows if x["cert_r2_at_rest"] == ""]
    # every stored at-rest reading must satisfy the inclusion criterion
    for x in have_at_rest:
        assert x["cert_r2_at_rest"] <= TAU + 1e-12, x
    over_tau_attacked = [x for x in rows if x["cert_r2_attacked"] > TAU]

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"[save] {OUT}  ({len(rows)} rows)")
    print(f"  existing 21 / new 46; fails_bar total = "
          f"{sum(x['fails_bar'] for x in rows)} (expect 59)")
    print(f"  at-rest certificate stored for {len(have_at_rest)}/67; "
          f"missing for {len(missing)}")
    for x in missing:
        print(f"    MISSING at-rest: {x['cell']:<12} {x['method']}")
    print(f"  attacked-instrument readings above tau=0.05: "
          f"{len(over_tau_attacked)}")
    for x in over_tau_attacked:
        print(f"    {x['cell']:<12} {x['method']:<16} "
              f"attacked={x['cert_r2_attacked']:.4f}")
    print("  mechanism split:",
          {m: sum(1 for x in rows if x["mechanism"] == m)
           for m in ("projection", "noise")})
    print("  dataset split:",
          {d: sum(1 for x in rows if x["dataset"] == d)
           for d in sorted({x["dataset"] for x in rows})})


if __name__ == "__main__":
    main()
