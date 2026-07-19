"""Phase 0 of the folktables extension: label-coupling scan over the 6
candidate ACS cells (3 tasks x 2 attributes) on CA 2018 1-Year.

Same construction as diagnostic.py's PREDICTOR (and _scout_predictor.py):
XGB AUC recovering the attribute from the 1-dim task label alone, 75/25
stratified held-out splits, 3 seeds. No models trained. The point is cell
SELECTION: flag which cells land below / inside / above the 0.52-0.60
predictor band so 4-6 cells spanning the range can be picked for the
pipeline (Phase 1).

Writes results/folktables_coupling_scan.json (new file; touches nothing
else). macOS OpenMP guard: xgboost before torch.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: F401,E402  (import before torch on purpose)

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.hardening_test import _probe  # noqa: E402
from utils.folktables_io import (  # noqa: E402
    ATTR_NAMES, PROVENANCE, TASK_NAMES, load_folktables_cell,
)

RESULTS_PATH = REPO_ROOT / "results" / "folktables_coupling_scan.json"
BAND = (0.52, 0.60)
SEEDS = (0, 1, 2)


def label_predictor(attr, task, n_attr, seeds=SEEDS):
    """XGB AUC recovering attr from the 1-dim task label alone (3 seeds)."""
    from sklearn.model_selection import train_test_split

    aucs = []
    X = task.reshape(-1, 1).astype(np.float64)
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, attr, test_size=0.25,
                                              random_state=s, stratify=attr)
        _, auc = _probe("XGB", Xtr, ytr, Xte, yte, s, n_attr)
        aucs.append(auc)
    return float(np.mean(aucs)), float(np.std(aucs))


def band_flag(auc: float) -> str:
    if auc < BAND[0]:
        return "below"
    if auc <= BAND[1]:
        return "inside"
    return "above"


def main():
    print(f"[setup] folktables {PROVENANCE['state']} {PROVENANCE['survey_year']} "
          f"{PROVENANCE['horizon']} | predictor = XGB AUC(attr | task label alone), "
          f"{len(SEEDS)} seeds, 75/25 stratified | band {BAND[0]}-{BAND[1]}")
    rows = []
    print(f"\n{'cell':<34} {'n':>8} {'x_dim':>6} {'attr_maj':>9} {'task_maj':>9} "
          f"{'AUC(attr|label)':>16} {'band':>7}")
    print("-" * 96)
    for task_name in TASK_NAMES:
        for attr_name in ATTR_NAMES:
            t0 = time.time()
            X, attr, task, n_attr, n_task, attr_maj, task_maj = \
                load_folktables_cell(task_name, attr_name)
            auc, std = label_predictor(attr, task, n_attr)
            dt = time.time() - t0
            cell = f"folktables/{attr_name}/{task_name}"
            flag = band_flag(auc)
            rows.append({"cell": cell, "task": task_name, "attr": attr_name,
                         "n": int(len(attr)), "x_dim": int(X.shape[1]),
                         "n_attr": n_attr, "n_task": n_task,
                         "attr_majority": attr_maj, "task_majority": task_maj,
                         "predictor_auc": auc, "predictor_std": std,
                         "band": flag, "seconds": round(dt, 1)})
            print(f"{cell:<34} {len(attr):>8} {X.shape[1]:>6} {attr_maj:>9.3f} "
                  f"{task_maj:>9.3f} {auc:>10.4f} ±{std:.3f} {flag:>7}   [{dt:.1f}s]")
    print("-" * 96)

    print("\nSorted by predictor (label->attr XGB AUC):")
    for r in sorted(rows, key=lambda r: r["predictor_auc"]):
        print(f"  {r['predictor_auc']:.4f} ±{r['predictor_std']:.3f}  "
              f"{r['band']:<7} {r['cell']}")
    tally = {b: sum(r["band"] == b for r in rows) for b in ("below", "inside", "above")}
    print(f"\nBand tally vs {BAND[0]}-{BAND[1]}: "
          f"{tally['below']} below, {tally['inside']} inside, {tally['above']} above.")

    out = {"experiment": "folktables_coupling_scan", "phase": 0,
           "predictor": "XGB AUC(attr | task label alone), 75/25 stratified, 3 seeds",
           "band": list(BAND), "seeds": list(SEEDS),
           "provenance": PROVENANCE, "cells": rows, "band_tally": tally}
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {RESULTS_PATH}")


if __name__ == "__main__":
    main()
