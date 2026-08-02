"""One work unit of the 5-seed recertification: (cell, training seed).

Registered in gate_5seed_prediction.md (commit 66390e1) BEFORE this file.

Usage:  run_gate_shard.py <cell> <train_seed>

Writes gate_shards/<cell>_ts<seed>.json holding the FULL per-(surface,
criterion, architecture, probe seed) AUC matrix plus lift -- nothing is
collapsed to a max here, so both the paper convention (max over architectures
of the mean over seeds) and the strict convention (max over everything) are
computable downstream from the same compute. Scoring is ~4% of the cost of a
unit (measured: FARE scored 3 seeds x 2 surfaces in 0.9 min), so storing the
matrix is free and re-scoring later never needs a retrain.

The trained representation is cached to gate_reps/ so any future rescoring --
different bar, different pair criterion, different aggregation -- costs only
the scoring pass.
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
from run_worstpair_sweep import (  # noqa: E402
    CELLS, build, lora_probs, macro_ovr, probe_probs, worst_pair,
)
from run_worstpair_supported_sweep import (  # noqa: E402
    MIN_CLASS_ROWS, supported_mask, worst_pair_supported,
)
from utils.pcrl_io import get_device  # noqa: E402

PROBE_SEEDS = [0, 1, 2]          # matches CERT_SEEDS in two_tier_certification
ARCHS = ("XGB", "MLP", "LoRA")
# the two operating points the supported-pair sweep found, unchanged
POINTS = {
    "easy": dict(kind="subspace", sigma=256.0, rank=32, lam=10.0, tier=1),
    "middle": dict(kind="subspace", sigma=1024.0, rank=16, lam=100.0, tier=1),
}


def score_surface(H, attr, n_attr, keep, device):
    """Per-(arch, probe seed) AUCs on one surface. No aggregation."""
    out = {a: {c: [] for c in ("macro", "all", "sup")} for a in ARCHS}
    for s in PROBE_SEEDS:
        for arch in ARCHS:
            if arch == "LoRA":
                y, p = lora_probs(np.asarray(H, np.float32), attr, n_attr, s, device)
            else:
                y, p = probe_probs(arch, np.asarray(H, np.float64), attr, n_attr, s)
            out[arch]["macro"].append(float(macro_ovr(y, p)))
            out[arch]["all"].append(float(worst_pair(y, p)[0]))
            out[arch]["sup"].append(float(worst_pair_supported(y, p, keep)[0]))
    return out


def main():
    cellk, ts = sys.argv[1], int(sys.argv[2])
    t0 = time.time()
    device = get_device()
    cfg = POINTS[cellk]
    ds, at, tk = CELLS[cellk]
    X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk)
    keep, counts = supported_mask(attr, n_attr)
    X_t = torch.from_numpy(X).to(device)

    P, L, h, ncov = build(cfg["kind"], cfg, cfg["sigma"], X_t, attr, task,
                          n_attr, n_task, device, ts)
    t_train = time.time() - t0

    reps = ARTIFACTS / "gate_reps"
    reps.mkdir(exist_ok=True)
    np.savez_compressed(reps / f"{cellk}_ts{ts}.npz",
                        P=np.asarray(P, np.float32), L=np.asarray(L, np.float32),
                        h=np.asarray(h, np.float32))

    res = dict(cell=cellk, train_seed=ts, sigma=cfg["sigma"], rank=cfg["rank"],
               lam=cfg["lam"], tier=cfg["tier"], class_counts=counts,
               min_class_rows=MIN_CLASS_ROWS,
               supported_classes=[i for i, v in enumerate(keep) if v],
               probe_seeds=PROBE_SEEDS, archs=list(ARCHS),
               rep=score_surface(P, attr, n_attr, keep, device),
               out=score_surface(L, attr, n_attr, keep, device),
               lift=float(lift(L, task, task_maj)), task_maj=float(task_maj),
               train_minutes=round(t_train / 60, 2),
               minutes=round((time.time() - t0) / 60, 2))

    sh = ARTIFACTS / "gate_shards"
    sh.mkdir(exist_ok=True)
    (sh / f"{cellk}_ts{ts}.json").write_text(json.dumps(res, indent=1))
    print(f"[{cellk} ts{ts}] train={res['train_minutes']}min "
          f"total={res['minutes']}min lift={res['lift']:+.4f} "
          f"rep_sup_max={max(max(v['sup']) for v in res['rep'].values()):.4f} "
          f"out_sup_max={max(max(v['sup']) for v in res['out'].values()):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
