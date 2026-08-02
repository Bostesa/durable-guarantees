"""One Obliviator fleet unit: (cell, training seed).

Same dual-convention, dual-criterion treatment the other baselines get, so
Obliviator's Table-1 entry is scored by the same rule as everything else.

Their loop is run verbatim via obliviator_gauntlet.run_obliviator (official
code, commit 0f2233f8). Only the STOPPING iteration's release is kept -- their
loop must run to MAX_ITER regardless, but skipping the 8-point MEASURE_ITERS
sweep and the certification search removes most of the non-loop cost.

Usage: run_obliviator_shard.py <cell> <train_seed>
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
sys.path.insert(0, str(HERE))

from experiments.diagnostic import lift, load_cell  # noqa: E402
from experiments.obliviator_gauntlet import (  # noqa: E402
    make_exposure, run_obliviator,
)
from experiments.sealed_channel import train_model  # noqa: E402
from run_fleet_shard import lrt_matrix  # noqa: E402
from run_gate_shard import ARCHS, PROBE_SEEDS, score_surface  # noqa: E402
from run_worstpair_supported_sweep import MIN_CLASS_ROWS, supported_mask  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

CELLS = {"easy": ("hmda", "race", "loan_decision"),
         "middle": ("hmda", "race", "loan_amount_band"),
         "hard": ("adult", "sex", "income")}


def main():
    cellk, ts = sys.argv[1], int(sys.argv[2])
    t0 = time.time()
    device = get_device()
    ds, at, tk = CELLS[cellk]
    X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk)
    keep, counts = supported_mask(attr, n_attr)
    X_t = torch.from_numpy(X).to(device)

    H, _, _ = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                          alpha=0.0, device=device, pub_noise=0.0, seed=ts)
    exps, stop_it, orig = run_obliviator(H, attr, task, n_attr, n_task, ts,
                                         str(device), want_iters=set())
    if not exps:
        print(f"[fail] {cellk} ts{ts}: no exposure returned", flush=True)
        return
    e = max(exps, key=lambda d: d["iter"])          # the stopping iteration
    exp = make_exposure(e, task)
    P, L = exp["P"], exp["L"]
    t_train = time.time() - t0

    res = dict(method="Obliviator", cell=cellk, knob=float(e["iter"]),
               operating_point=f"iter {int(e['iter'])}", train_seed=ts,
               their_stop_iter=int(stop_it), their_orig_probes=orig,
               their_unwanted=float(e["their_unwanted"]),
               their_utility=float(e["their_utility"]),
               class_counts=counts, min_class_rows=MIN_CLASS_ROWS,
               supported_classes=[i for i, v in enumerate(keep) if v],
               probe_seeds=PROBE_SEEDS, archs=list(ARCHS),
               rep=score_surface(P, attr, n_attr, keep, device),
               out=score_surface(L, attr, n_attr, keep, device),
               rep_lrt=lrt_matrix(P, P, attr, np.zeros((P.shape[1],) * 2),
                                  n_attr, keep),
               out_lrt=lrt_matrix(L, L, attr, np.zeros((L.shape[1],) * 2),
                                  n_attr, keep),
               lift=float(lift(L, task, task_maj)), task_maj=float(task_maj),
               refit=True, z_dim=int(P.shape[1]),
               train_minutes=round(t_train / 60, 2),
               minutes=round((time.time() - t0) / 60, 2))

    sh = ARTIFACTS / "fleet_shards"
    sh.mkdir(exist_ok=True)
    (sh / f"Obliviator_{cellk}_k{e['iter']:g}_ts{ts}.json").write_text(
        json.dumps(res, indent=1, default=float))
    print(f"[Obliviator {cellk} ts{ts}] stop_iter={stop_it} z_dim={res['z_dim']} "
          f"loop={res['train_minutes']}min total={res['minutes']}min "
          f"rep_sup={max(max(v['sup']) for v in res['rep'].values()):.4f} "
          f"out_sup={max(max(v['sup']) for v in res['out'].values()):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
