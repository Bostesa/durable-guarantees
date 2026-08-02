"""Isolate-then-noise vs plain full-rank noise, matched at Tier-1.

Registered prediction: docs/isolate_vs_fullrank_prediction.md (before this script).
For each headline cell, train both channels at 3 seeds and report black-box
Tier-1 max, Tier-2 Gaussian-LRT AUC, and retained utility, so the honest
tradeoff (isolate = more Tier-1 utility, no Tier-2; full-rank = less utility,
Tier-2-reachable) is a measured table. Full-rank sigma is grid-tuned per cell
to match the isolate channel's Tier-1 max within 0.01.

Reuses train_e2e_surgical (isolate), mi_ceiling.train_channel (full-rank
blunt), and the exact probe/LRT instruments. New file; nothing existing
modified. Local MPS.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from experiments.diagnostic import lift, load_cell, probe_lora, probe_sklearn  # noqa: E402
from experiments.end_to_end_surgical import train_e2e_surgical  # noqa: E402
from experiments.mi_ceiling import train_channel  # noqa: E402
from utils.battery import gaussian_lrt, iso_noise_cov, subspace_noise_cov  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

RESULTS = REPO / "results"
PS = [0, 1]
TRAIN_SEEDS = [0, 1, 2]
CELLS = {"easy": ("hmda", "race", "loan_decision", dict(rank=32, sigma=16.0, lam=10.0)),
         "middle": ("hmda", "race", "loan_amount_band", dict(rank=16, sigma=16.0, lam=100.0)),
         "hard": ("adult", "sex", "income", dict(rank=32, sigma=16.0, lam=100.0))}
# full-rank sigma grid to search for a Tier-1 match (spans the stored 8/12/32)
FR_GRID = [4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0]


def t1_max(P, attr, n_attr, device):
    x, _ = probe_sklearn("XGB", P, attr, n_attr, PS)
    m, _ = probe_sklearn("MLP", P, attr, n_attr, PS)
    l, _ = probe_lora(P, attr, n_attr, PS, device)
    return float(max(x, m, l)), (float(x), float(m), float(l))


def main():
    t0 = time.time()
    device = get_device()
    tt = json.loads((RESULTS / "two_tier_certification.json").read_text())
    ttname = {"easy": "hmda/race/loan_decision",
              "middle": "hmda/race/loan_amount_band", "hard": "adult/sex/income"}
    clean = {k: next(c for c in tt["e2e_cells"] if c["cell"] == ttname[k])["clean_lift"]
             for k in CELLS}
    print(f"[setup] device={device} clean={ {k: round(v,4) for k,v in clean.items()} }",
          flush=True)

    rows = []
    for key, (ds, at, tk, cfg) in CELLS.items():
        X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk)
        X_t = torch.from_numpy(X).to(device)
        print(f"\n[cell {key}] clean_lift={clean[key]:+.4f}", flush=True)

        # ---- isolate-then-noise (subspace-confined), 3 seeds -----------------
        iso = {"t1": [], "lrt": [], "util": []}
        for ts in TRAIN_SEEDS:
            P, L, h, Q = train_e2e_surgical(X_t, attr, task, n_attr, n_task,
                                            rank=cfg["rank"], sigma=cfg["sigma"],
                                            lam=cfg["lam"], device=device,
                                            seed=ts, return_clean=True)
            t1, _ = t1_max(P, attr, n_attr, device)
            ncov = subspace_noise_cov(Q, cfg["sigma"])
            lrt, _ = gaussian_lrt(h, P, attr, ncov, n_attr, PS)
            iso["t1"].append(t1); iso["lrt"].append(lrt)
            iso["util"].append(100 * lift(L, task, task_maj) / clean[key])
        iso_t1 = float(np.mean(iso["t1"]))
        print(f"  [isolate r={cfg['rank']} s={cfg['sigma']:g} l={cfg['lam']:g}] "
              f"T1={iso_t1:.3f} LRT={np.mean(iso['lrt']):.3f} "
              f"util={np.mean(iso['util']):.1f}%", flush=True)

        # ---- full-rank noise: tune sigma to match isolate T1 -----------------
        best = None
        for sg in FR_GRID:
            h, P, L = train_channel(X_t, attr, task, n_attr, n_task, sg, device,
                                    seed=0)
            t1, _ = t1_max(P, attr, n_attr, device)
            if best is None or abs(t1 - iso_t1) < abs(best[1] - iso_t1):
                best = (sg, t1)
        fr_sigma = best[0]
        print(f"  [full-rank match] sigma={fr_sigma:g} (T1={best[1]:.3f} vs "
              f"isolate {iso_t1:.3f})", flush=True)
        fr = {"t1": [], "lrt": [], "util": []}
        for ts in TRAIN_SEEDS:
            h, P, L = train_channel(X_t, attr, task, n_attr, n_task, fr_sigma,
                                    device, seed=ts)
            t1, _ = t1_max(P, attr, n_attr, device)
            ncov = iso_noise_cov(h.shape[1], fr_sigma)
            lrt, _ = gaussian_lrt(h, P, attr, ncov, n_attr, PS)
            fr["t1"].append(t1); fr["lrt"].append(lrt)
            fr["util"].append(100 * lift(L, task, task_maj) / clean[key])
        print(f"  [full-rank s={fr_sigma:g}] T1={np.mean(fr['t1']):.3f} "
              f"LRT={np.mean(fr['lrt']):.3f} util={np.mean(fr['util']):.1f}%",
              flush=True)

        row = {"cell": key, "coupling": {"easy": 0.514, "middle": 0.584,
                                         "hard": 0.603}[key],
               "isolate": {"config": cfg,
                           "t1_max_mean": iso_t1, "t1_max_std": float(np.std(iso["t1"])),
                           "lrt_mean": float(np.mean(iso["lrt"])),
                           "lrt_std": float(np.std(iso["lrt"])),
                           "util_mean": float(np.mean(iso["util"])),
                           "util_std": float(np.std(iso["util"]))},
               "fullrank": {"sigma": fr_sigma,
                            "t1_max_mean": float(np.mean(fr["t1"])),
                            "t1_max_std": float(np.std(fr["t1"])),
                            "lrt_mean": float(np.mean(fr["lrt"])),
                            "lrt_std": float(np.std(fr["lrt"])),
                            "util_mean": float(np.mean(fr["util"])),
                            "util_std": float(np.std(fr["util"]))},
               "t1_matched_within_0.01": bool(abs(iso_t1 - np.mean(fr["t1"])) <= 0.01)}
        rows.append(row)

    # ---- score registered claims --------------------------------------------
    c1 = all(r["isolate"]["util_mean"] > r["fullrank"]["util_mean"] for r in rows)
    c2 = all(r["isolate"]["lrt_mean"] > r["fullrank"]["lrt_mean"] for r in rows)
    c2b = all(r["isolate"]["lrt_mean"] >= 0.60 for r in rows)
    scoring = {
        "claim1_isolate_more_utility": "HELD" if c1 else "FALSIFIED",
        "claim2_isolate_leaks_tier2_more": "HELD" if c2 else "FALSIFIED",
        "claim2b_isolate_lrt_above_0.60": "HELD" if c2b else "FALSIFIED",
        "util_gap_by_cell": {r["cell"]: round(r["isolate"]["util_mean"]
                                              - r["fullrank"]["util_mean"], 1)
                             for r in rows}}
    print("\n" + "=" * 76)
    for k, v in scoring.items():
        print(f"  {k}: {v}")
    out = {"experiment": "isolate_vs_fullrank_matched_tier1",
           "prediction_file": "isolate_vs_fullrank_prediction.md",
           "train_seeds": TRAIN_SEEDS, "probe_seeds": PS,
           "clean_lifts": clean, "rows": rows, "scoring": scoring,
           "minutes": round((time.time() - t0) / 60, 1)}
    (RESULTS / "isolate_vs_fullrank.json").write_text(json.dumps(out, indent=1))
    print(f"\n[save] {RESULTS / 'isolate_vs_fullrank.json'} ({out['minutes']} min)",
          flush=True)


if __name__ == "__main__":
    main()
