"""Knows-Q attacker — registered in knows_q_prediction.md (commit ca420fe)
BEFORE this file existed.

The missing rung between Tier 1 (knows nothing) and Tier 2 (knows Q, sigma AND
clean class statistics): an attacker that knows only Q, projects the released
representation onto the complement of span(Q) — which recovers h_perp exactly,
since the channel confines noise to span(Q) — and trains XGB + MLP on it.

3 training seeds, probe seeds [0,1,2], standard 75/25 stratified splits, bar 0.55,
at the three stored subspace operating points. Local MPS; macOS OpenMP guard.
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
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO))

from experiments.diagnostic import load_cell, probe_lora, probe_sklearn  # noqa: E402
from experiments.end_to_end_surgical import train_e2e_surgical  # noqa: E402
from utils.battery import gaussian_lrt, subspace_noise_cov  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

BAR = 0.55
TRAIN_SEEDS = [0, 1, 2]
PROBE_SEEDS = [0, 1, 2]
CELLS = {
    "easy": ("hmda", "race", "loan_decision", dict(rank=32, sigma=16.0, lam=10.0)),
    "middle": ("hmda", "race", "loan_amount_band", dict(rank=16, sigma=16.0, lam=100.0)),
    "hard": ("adult", "sex", "income", dict(rank=32, sigma=16.0, lam=100.0)),
}
# pinned comparators (isolate_vs_fullrank.json sha256 471a4ccd)
STORED_T1 = {"easy": 0.5094, "middle": 0.5349, "hard": 0.5369}
STORED_LRT = {"easy": 0.8542, "middle": 0.8640, "hard": 0.6617}


def main():
    t0 = time.time()
    device = get_device()
    print(f"[setup] device={device} train_seeds={TRAIN_SEEDS} probe_seeds={PROBE_SEEDS} "
          f"bar={BAR}", flush=True)
    rows = []
    for key, (ds, at, tk, cfg) in CELLS.items():
        X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk)
        X_t = torch.from_numpy(X).to(device)
        r, sigma = cfg["rank"], cfg["sigma"]
        per = {k: [] for k in ("kq_xgb", "kq_mlp", "kq_lora", "rel_xgb", "rel_mlp",
                               "rel_lora", "lrt", "recon_err")}
        for ts in TRAIN_SEEDS:
            P, _, h, Q = train_e2e_surgical(X_t, attr, task, n_attr, n_task, rank=r,
                                            sigma=sigma, lam=cfg["lam"],
                                            device=device, seed=ts, return_clean=True)
            Pd = P.astype(np.float64)
            # attacker knows Q only: project the RELEASE onto complement of span(Q)
            h_perp_hat = (Pd - (Pd @ Q) @ Q.T).astype(np.float32)
            # sanity: this must equal the clean complement (noise lives in span(Q))
            h_perp_true = (h.astype(np.float64) - (h.astype(np.float64) @ Q) @ Q.T)
            per["recon_err"].append(float(np.abs(h_perp_hat - h_perp_true).max()))

            x, _ = probe_sklearn("XGB", h_perp_hat, attr, n_attr, PROBE_SEEDS)
            m, _ = probe_sklearn("MLP", h_perp_hat, attr, n_attr, PROBE_SEEDS)
            l, _ = probe_lora(h_perp_hat, attr, n_attr, PROBE_SEEDS, device)
            per["kq_xgb"].append(x); per["kq_mlp"].append(m); per["kq_lora"].append(l)

            rx, _ = probe_sklearn("XGB", P, attr, n_attr, PROBE_SEEDS)
            rm, _ = probe_sklearn("MLP", P, attr, n_attr, PROBE_SEEDS)
            rl, _ = probe_lora(P, attr, n_attr, PROBE_SEEDS, device)
            per["rel_xgb"].append(rx); per["rel_mlp"].append(rm); per["rel_lora"].append(rl)

            lrt, _ = gaussian_lrt(h, P, attr, subspace_noise_cov(Q, sigma), n_attr,
                                 PROBE_SEEDS)
            per["lrt"].append(lrt)
            print(f"  [{key} seed {ts}] knows-Q XGB={x:.4f} MLP={m:.4f} LoRA={l:.4f} | "
                  f"release XGB={rx:.4f} MLP={rm:.4f} LoRA={rl:.4f} | LRT={lrt:.4f} | "
                  f"recon_err={per['recon_err'][-1]:.2e}", flush=True)

        m = {k: float(np.mean(v)) for k, v in per.items()}
        s = {k: float(np.std(v)) for k, v in per.items()}
        row = {
            "cell": key, "dataset": ds, "attr": at, "task": tk, "config": cfg,
            "knows_q_xgb": m["kq_xgb"], "knows_q_mlp": m["kq_mlp"],
            "knows_q_lora": m["kq_lora"],
            "knows_q_max": max(m["kq_xgb"], m["kq_mlp"]),
            "knows_q_max_incl_lora": max(m["kq_xgb"], m["kq_mlp"], m["kq_lora"]),
            "release_tier1_max": max(m["rel_xgb"], m["rel_mlp"], m["rel_lora"]),
            "lrt": m["lrt"], "stored_tier1": STORED_T1[key],
            "stored_lrt": STORED_LRT[key],
            "max_recon_err": max(per["recon_err"]),
            "stds": s, "per_seed": {k: [float(x) for x in v] for k, v in per.items()},
        }
        row["breaches_bar"] = bool(row["knows_q_max"] > BAR)
        row["exceeds_release_tier1"] = bool(row["knows_q_max"] > row["release_tier1_max"])
        rows.append(row)
        print(f"  >>> {key}: knows-Q max={row['knows_q_max']:.4f} "
              f"(vs release T1 {row['release_tier1_max']:.4f}, LRT {row['lrt']:.4f}) "
              f"{'BREACHES' if row['breaches_bar'] else 'holds'} 0.55\n", flush=True)

    k1 = all(r["exceeds_release_tier1"] for r in rows)
    k2 = sum(r["breaches_bar"] for r in rows) >= 2
    order_lrt = [r["cell"] for r in sorted(rows, key=lambda r: -r["stored_lrt"])]
    order_kq = [r["cell"] for r in sorted(rows, key=lambda r: -r["knows_q_max"])]
    scoring = {
        "K1_exceeds_release_tier1_all_cells": {
            "per_cell": {r["cell"]: r["exceeds_release_tier1"] for r in rows},
            "verdict": "HELD" if k1 else "FALSIFIED"},
        "K2_breaches_bar_ge2_cells": {
            "per_cell": {r["cell"]: r["breaches_bar"] for r in rows},
            "verdict": "HELD" if k2 else "FALSIFIED"},
        "K3_ordering_follows_lrt": {
            "lrt_order": order_lrt, "knows_q_order": order_kq,
            "verdict": "HELD" if order_lrt == order_kq else "FALSIFIED"},
        "tier1_depends_on_Q_secrecy": bool(k2),
    }
    print("=" * 92 + "\nSCORING (knows_q_prediction.md @ ca420fe)\n" + "=" * 92)
    for k, v in scoring.items():
        print(f"  {k}: {v}")

    out = {"experiment": "knows_q_attacker",
           "prediction_file": "knows_q_prediction.md",
           "prediction_commit": "ca420fe", "bar": BAR,
           "train_seeds": TRAIN_SEEDS, "probe_seeds": PROBE_SEEDS,
           "cells": rows, "scoring": scoring,
           "minutes": round((time.time() - t0) / 60, 1)}
    (RESULTS / "knows_q.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {RESULTS / 'knows_q.json'} ({out['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
