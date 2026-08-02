"""Knows-Q attacker at the two supported-pair passing operating points.

Report-only diagnostic. Procedure is byte-identical to run_knows_q.py: train the
subspace channel, project the RELEASE onto the complement of span(Q) to recover
h_perp, train XGB and MLP on it directly, probe seeds [0,1,2]. Only the sigma
differs (256 easy / 1024 middle instead of the stored 16).

knows_q_max is max(XGB, MLP) of the per-training-seed means, matching
run_knows_q.py:96 exactly. LoRA is not run: it is excluded from that statistic.

Usage: run_knows_q_highsigma_shard.py <cell> <train_seed>
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
# Generated intermediates; gitignored and regenerable.
# Override the location with DG_ARTIFACTS.
ARTIFACTS = Path(os.environ.get("DG_ARTIFACTS", REPO / "analysis"))
sys.path.insert(0, str(REPO))

from experiments.diagnostic import load_cell, probe_sklearn  # noqa: E402
from experiments.end_to_end_surgical import train_e2e_surgical  # noqa: E402
from utils.battery import gaussian_lrt, subspace_noise_cov  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

PROBE_SEEDS = [0, 1, 2]
# the two supported-pair passing points (worstpair_supported_sweep.json @ 5e47b8b)
CELLS = {
    "easy": ("hmda", "race", "loan_decision", dict(rank=32, sigma=256.0, lam=10.0)),
    "middle": ("hmda", "race", "loan_amount_band",
               dict(rank=16, sigma=1024.0, lam=100.0)),
}


def main():
    cellk, ts = sys.argv[1], int(sys.argv[2])
    t0 = time.time()
    device = get_device()
    ds, at, tk, cfg = CELLS[cellk]
    X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk)
    X_t = torch.from_numpy(X).to(device)

    P, _, h, Q = train_e2e_surgical(X_t, attr, task, n_attr, n_task,
                                    rank=cfg["rank"], sigma=cfg["sigma"],
                                    lam=cfg["lam"], device=device, seed=ts,
                                    return_clean=True)
    Pd = P.astype(np.float64)
    h_perp_hat = (Pd - (Pd @ Q) @ Q.T).astype(np.float32)
    hd = h.astype(np.float64)
    h_perp_true = hd - (hd @ Q) @ Q.T
    recon = float(np.abs(h_perp_hat - h_perp_true).max())

    x, _ = probe_sklearn("XGB", h_perp_hat, attr, n_attr, PROBE_SEEDS)
    m, _ = probe_sklearn("MLP", h_perp_hat, attr, n_attr, PROBE_SEEDS)
    rx, _ = probe_sklearn("XGB", P, attr, n_attr, PROBE_SEEDS)
    rm, _ = probe_sklearn("MLP", P, attr, n_attr, PROBE_SEEDS)
    lrt, _ = gaussian_lrt(h, P, attr, subspace_noise_cov(Q, cfg["sigma"]),
                          n_attr, PROBE_SEEDS)

    res = dict(cell=cellk, train_seed=ts, sigma=cfg["sigma"], rank=cfg["rank"],
               lam=cfg["lam"], probe_seeds=PROBE_SEEDS,
               kq_xgb=float(x), kq_mlp=float(m),
               rel_xgb=float(rx), rel_mlp=float(rm), lrt=float(lrt),
               recon_err=recon, minutes=round((time.time() - t0) / 60, 2))
    sh = ARTIFACTS / "knowsq_hs_shards"
    sh.mkdir(exist_ok=True)
    (sh / f"{cellk}_ts{ts}.json").write_text(json.dumps(res, indent=1))
    print(f"[{cellk} ts{ts} sigma={cfg['sigma']:g}] knows-Q XGB={x:.4f} MLP={m:.4f} | "
          f"release XGB={rx:.4f} MLP={rm:.4f} | LRT={lrt:.4f} | "
          f"recon_err={recon:.2e} ({res['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
