"""Scouting: measure the PREDICTOR (XGB AUC of attribute recovered from the task
LABEL alone) across candidate cells, so I can pick >=4 cells that SPAN the
predictor axis for the diagnostic. Cheap: one 1-feature XGB per cell, 3 seeds.
No models trained. macOS OpenMP guard: xgboost before torch."""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: F401,E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.hardening_test import _probe  # noqa: E402
from utils.pcrl_io import build_train_loader, extract_attr, extract_task_label  # noqa: E402

CANDIDATES = [
    ("hmda", "race", "loan_decision"),
    ("hmda", "ethnicity", "loan_decision"),
    ("hmda", "sex", "loan_decision"),
    ("hmda", "sex", "loan_amount_band"),
    ("hmda", "race", "loan_amount_band"),
    ("adult", "sex", "income"),
    ("adult", "sex", "occupation_group"),
    ("adult", "race", "income"),
    ("adult", "sex", "education_level"),
    ("adult", "marital_status", "income"),
    ("diabetes", "gender", "primary_diagnosis_category"),
    ("diabetes", "gender", "readmission_outcome"),
    ("diabetes", "race", "primary_diagnosis_category"),
    ("diabetes", "gender", "medication_change_outcome"),
]


def label_predictor(attr, task, n_attr, seeds=(0, 1, 2)):
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


def main():
    cache = {}
    print(f"{'cell':<44} {'attr_cls':>8} {'attr_maj':>9} {'task_maj':>9} "
          f"{'XGB_AUC(attr|label)':>20}")
    print("-" * 96)
    rows = []
    for dataset, attr_name, task_name in CANDIDATES:
        if dataset not in cache:
            _, _, loader = build_train_loader(dataset)
            cache[dataset] = list(loader)
        loader = cache[dataset]
        attr = extract_attr(loader, attr_name)
        task = extract_task_label(loader, task_name)
        n_attr = int(attr.max()) + 1
        attr_maj = float(np.bincount(attr).max() / len(attr))
        task_maj = float(np.bincount(task).max() / len(task))
        auc, std = label_predictor(attr, task, n_attr)
        cell = f"{dataset}/{attr_name}/{task_name}"
        rows.append((cell, n_attr, attr_maj, task_maj, auc, std))
        print(f"{cell:<44} {n_attr:>8} {attr_maj:>9.3f} {task_maj:>9.3f} "
              f"{auc:>13.4f} ±{std:.3f}")
    print("\nSorted by predictor (label->attr XGB AUC):")
    for cell, n_attr, _, _, auc, std in sorted(rows, key=lambda r: r[4]):
        print(f"  {auc:.4f}  {cell}")


if __name__ == "__main__":
    main()
