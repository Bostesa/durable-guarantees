"""Export the three gauntlet cells in the OFFICIAL LAFTR npz format.

Data-loading-boundary adaptation only: our cell arrays are written into the
schema data/adult/adult.npz uses (x_train/x_test, y (n,2+) one-hot,
attr (n,1) float, train_inds/valid_inds). The LAFTR code is untouched.

Choices (favorable/neutral to the baseline, mirroring the gauntlet protocol):
- train = ALL rows, since every other gauntlet baseline trains on the full
  cell. valid = a seed-0 10% subset. Their patience=1000 means no early
  stopping, so valid is used for logging only.
- x_test = ALL rows in original order, so the tester's saved Z aligns
  row-for-row with our attr/task arrays for the probe battery.

Usage: .venv/bin/python utils/export_cells_for_laftr.py [out_dir]
       (out_dir defaults to ../laftr_cells. Each cell lands in
        <out_dir>/<dataset>_<attr>_<task>/adult/adult.npz + meta.json,
        pointed at via `-o dirs.data_dir=...` in the official runs)
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: F401  (before torch, macOS OpenMP guard)

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.diagnostic import load_cell  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT.parent / "laftr_cells"

CELLS = [
    ("hmda", "race", "loan_decision"),
    ("hmda", "race", "loan_amount_band"),
    ("adult", "sex", "income"),
]

for dataset, attr_name, task_name in CELLS:
    X, attr, task, n_attr, n_task, attr_maj, task_maj = load_cell(
        dataset, attr_name, task_name)
    n = X.shape[0]
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    valid_inds = np.sort(perm[: n // 10])
    train_inds = np.sort(perm[n // 10:])
    y_oh = np.eye(n_task)[task].astype(np.float64)
    cell_dir = OUT / f"{dataset}_{attr_name}_{task_name}" / "adult"
    cell_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cell_dir / "adult.npz",
        x_train=X.astype(np.float64),
        x_test=X.astype(np.float64),
        y_train=y_oh,
        y_test=y_oh,
        attr_train=attr.reshape(-1, 1).astype(np.float64),
        attr_test=attr.reshape(-1, 1).astype(np.float64),
        train_inds=train_inds,
        valid_inds=valid_inds,
    )
    meta = dict(cell=f"{dataset}/{attr_name}/{task_name}", n=n,
                x_dim=int(X.shape[1]), n_attr=int(n_attr), n_task=int(n_task),
                attr_maj=attr_maj, task_maj=task_maj)
    (cell_dir.parent / "meta.json").write_text(json.dumps(meta, indent=2))
    print(meta, flush=True)
print("done")
