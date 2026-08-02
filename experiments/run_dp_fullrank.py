"""Clipped full-rank DP channel — scoring run for the registered bet in
notes/dp_bet.md (committed e87332b BEFORE this script existed).

Mechanism (clip-in-the-loop, per dp_feasibility.md): identical architecture and
training protocol to the stored full-rank channel (mi_ceiling.train_channel:
_mlp encoder -> BatchNorm1d(64) -> linear head, Adam 1e-3, TRAIN_STEPS steps,
batch TRAIN_BS, torch.manual_seed(seed)), with the release changed to

    release(x) = clip_C(bn(E(x))) + sigma * randn(64)      (noise in ALL dims)
    clip_C(h)  = h * min(1, C / ||h||_2)

applied inside the training loop too, so the head learns through the released
channel. Per-release privacy: classical Gaussian mechanism, delta = 1e-5,
sensitivity 2C -> eps = (2C/sigma) * sqrt(2 ln(1.25/delta)) = 9.6896 * C/sigma
(conservative above eps=1, so nominal-eps points are at least as private as
labeled). Grid: sigma {8, 32} x nominal eps {0.5, 1, 3, 6} (eps=6 is ramp
context OUTSIDE the bet). Three headline cells, 5 training seeds, utility =
own-head lift over all rows (argmax acc - task majority), retained % against
the stored 5-seed clean lifts in two_tier_certification.json.

Also records, per point: fraction of rows clipped at eval (||h|| > C) and
pre-clip norm quantiles — the empirical norm measurement dp_feasibility.md
flagged as missing (full-rank channel variant).

New file; no existing experiment code path is modified. Local MPS/CPU.
"""
from __future__ import annotations


import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401  (import before torch on purpose)

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.diagnostic import lift, load_cell  # noqa: E402
from experiments.sealed_channel import (  # noqa: E402
    PUB_DIM, TRAIN_BS, TRAIN_STEPS, _mlp,
)
from utils.pcrl_io import get_device  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = REPO_ROOT / "results"

DELTA = 1e-5
K_GM = 2.0 * float(np.sqrt(2.0 * np.log(1.25 / DELTA)))   # eps = K_GM * C / sigma
SIGMAS = [8.0, 32.0]
EPS_TARGETS = [0.5, 1.0, 3.0, 6.0]        # 6.0 is outside the bet (ramp context)
TRAIN_SEEDS = [0, 1, 2, 3, 4]
CELLS = {"easy": ("hmda", "race", "loan_decision"),
         "middle": ("hmda", "race", "loan_amount_band"),
         "hard": ("adult", "sex", "income")}
TT_NAME = {"easy": "hmda/race/loan_decision",
           "middle": "hmda/race/loan_amount_band",
           "hard": "adult/sex/income"}
ANCHORS = {"easy": 55.5, "middle": 21.7, "hard": -7.0}    # Tier-2 retained %


def train_dp_channel(X_t, task_np, n_task, C, sigma, device, seed):
    """mi_ceiling.train_channel protocol with clip-in-the-loop + full-rank noise."""
    torch.manual_seed(seed)
    d_in = X_t.shape[1]
    E = _mlp(d_in, 128, PUB_DIM).to(device)
    bn = nn.BatchNorm1d(PUB_DIM).to(device)
    D = nn.Linear(PUB_DIM, n_task).to(device)
    opt = torch.optim.Adam(list(E.parameters()) + list(bn.parameters())
                           + list(D.parameters()), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    task_t = torch.from_numpy(task_np).long().to(device)
    n = X_t.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    def clip(h):
        nrm = h.norm(dim=1, keepdim=True)
        return h * torch.clamp(C / (nrm + 1e-12), max=1.0)

    E.train(); bn.train(); D.train()
    for _ in range(TRAIN_STEPS):
        idx = torch.randint(0, n, (TRAIN_BS,), generator=g).to(device)
        h = bn(E(X_t[idx]))
        release = clip(h) + sigma * torch.randn_like(h)
        loss = ce(D(release), task_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    E.eval(); bn.eval(); D.eval()
    L_all, norms = [], []
    with torch.no_grad():
        for i in range(0, n, 4096):
            h = bn(E(X_t[i:i + 4096]))
            norms.append(h.norm(dim=1).cpu().numpy())
            release = clip(h) + sigma * torch.randn_like(h)
            L_all.append(D(release).cpu().numpy())
    return np.concatenate(L_all), np.concatenate(norms)


def main():
    t0 = time.time()
    device = get_device()
    tt = json.loads((RESULTS / "two_tier_certification.json").read_text())
    clean = {k: next(c for c in tt["e2e_cells"] if c["cell"] == TT_NAME[k])
             ["clean_lift"] for k in CELLS}
    print(f"[setup] device={device} delta={DELTA} K={K_GM:.4f} sigmas={SIGMAS} "
          f"eps={EPS_TARGETS} seeds={TRAIN_SEEDS}", flush=True)
    print(f"[setup] clean lifts (stored 5-seed): "
          f"{ {k: round(v, 4) for k, v in clean.items()} } anchors={ANCHORS}",
          flush=True)

    rows = []
    for key, (ds, at, tk) in CELLS.items():
        X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk)
        X_t = torch.from_numpy(X).to(device)
        print(f"\n[cell {key}] {ds}/{at}/{tk} n={len(task)} "
              f"clean_lift={clean[key]:+.4f}", flush=True)
        for sigma in SIGMAS:
            for eps in EPS_TARGETS:
                C = eps * sigma / K_GM
                lifts, fracs, med_norms = [], [], []
                t = time.time()
                for ts in TRAIN_SEEDS:
                    L, norms = train_dp_channel(X_t, task, n_task, C, sigma,
                                                device, ts)
                    lifts.append(lift(L, task, task_maj))
                    fracs.append(float((norms > C).mean()))
                    med_norms.append(float(np.median(norms)))
                lm, ls = float(np.mean(lifts)), float(np.std(lifts))
                kept = 100.0 * lm / clean[key]
                kept_std = 100.0 * ls / clean[key]
                row = {"cell": key, "sigma": sigma, "eps_nominal": eps,
                       "C": round(C, 4), "in_bet": eps <= 3.0,
                       "lift_mean": lm, "lift_std": ls,
                       "lift_per_seed": [float(x) for x in lifts],
                       "kept_pct": kept, "kept_pct_std": kept_std,
                       "anchor_pct": ANCHORS[key],
                       "beats_anchor": bool(kept > ANCHORS[key]),
                       "frac_clipped_mean": float(np.mean(fracs)),
                       "median_preclip_norm_mean": float(np.mean(med_norms)),
                       "seconds": round(time.time() - t, 1)}
                rows.append(row)
                print(f"  [s={sigma:>4g} eps={eps:>3g} C={C:6.3f}] "
                      f"lift={lm:+.4f}±{ls:.4f} kept={kept:+6.1f}% "
                      f"(anchor {ANCHORS[key]:+.1f}%"
                      f"{' BEATS' if row['beats_anchor'] and row['in_bet'] else ''}) "
                      f"clipped={row['frac_clipped_mean']:.2f} "
                      f"med||h||={row['median_preclip_norm_mean']:.2f} "
                      f"({row['seconds']:.0f}s)", flush=True)

    in_bet = [r for r in rows if r["in_bet"]]
    primary_falsified = [r for r in in_bet if r["beats_anchor"]]
    eps1 = [r for r in in_bet if r["eps_nominal"] <= 1.0]
    secondary_falsified = [r for r in eps1 if r["kept_pct"] > 20.0]
    print("\n" + "=" * 88)
    print(f"PRIMARY  (all eps<=3 at or below anchor): "
          f"{'FALSIFIED' if primary_falsified else 'HELD'} "
          f"({len(primary_falsified)}/{len(in_bet)} points beat their anchor)")
    print(f"SECONDARY (no eps<=1 point > 20%): "
          f"{'FALSIFIED' if secondary_falsified else 'HELD'} "
          f"({len(secondary_falsified)}/{len(eps1)} points over 20%)")
    for r in primary_falsified:
        print(f"  beats: {r['cell']} s={r['sigma']:g} eps={r['eps_nominal']:g} "
              f"kept={r['kept_pct']:+.1f}% > anchor {r['anchor_pct']:+.1f}%")

    out = {"experiment": "dp_fullrank_clipped", "bet_file": "dp_bet.md",
           "bet_commit": "e87332b", "delta": DELTA, "K_gaussian": K_GM,
           "sigmas": SIGMAS, "eps_targets": EPS_TARGETS,
           "train_seeds": TRAIN_SEEDS, "clean_lifts": clean,
           "anchors_pct": ANCHORS, "rows": rows,
           "primary": "FALSIFIED" if primary_falsified else "HELD",
           "secondary": "FALSIFIED" if secondary_falsified else "HELD",
           "minutes": round((time.time() - t0) / 60, 1)}
    (RESULTS / "dp_fullrank.json").write_text(json.dumps(out, indent=1))
    print(f"\n[save] {RESULTS / 'dp_fullrank.json'} ({out['minutes']} min)",
          flush=True)


if __name__ == "__main__":
    main()
