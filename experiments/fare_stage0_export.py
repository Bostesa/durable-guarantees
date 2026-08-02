"""FARE gauntlet stage 0 — export the three headline cells to npz for the FARE
venv (py3.9, patched sklearn) which cannot import this repo's stack.

Writes analysis/fare_cells/{easy,middle,hard}.npz with X float32,
attr int64, task int64 — the same load_cell output the other from-raw-x
baselines consume."""
from __future__ import annotations


import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.diagnostic import load_cell  # noqa: E402

CELLS = [("easy", "hmda", "race", "loan_decision"),
         ("middle", "hmda", "race", "loan_amount_band"),
         ("hard", "adult", "sex", "income")]

# Generated intermediates; gitignored and regenerable.
# Override the location with DG_ARTIFACTS.
ARTIFACTS = Path(os.environ.get("DG_ARTIFACTS", REPO_ROOT / "analysis"))
out_dir = ARTIFACTS / "fare_cells"
out_dir.mkdir(parents=True, exist_ok=True)
for key, ds, attr_name, task_name in CELLS:
    X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, attr_name, task_name)
    np.savez_compressed(out_dir / f"{key}.npz", X=X.astype(np.float32),
                        attr=attr.astype(np.int64), task=task.astype(np.int64))
    print(f"{key}: {ds}/{attr_name}/{task_name} N={len(attr)} d={X.shape[1]} "
          f"n_attr={n_attr} n_task={n_task}")
