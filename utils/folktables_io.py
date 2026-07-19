"""Folktables (ACS PUMS) natural cells — CA, 2018 1-Year person survey.

Extension loader for the diagnostic: three official folktables prediction
tasks (ACSIncome, ACSEmployment, ACSPublicCoverage; Ding et al., NeurIPS
2021) x two protected attributes (SEX, RAC1P), giving 6 candidate
(task, attribute) cells on natural census data.

Conventions mirror the existing tabular cells (utils/pcrl_io.py + the PCRL
tabular datasets), so downstream code can treat a folktables cell exactly
like an adult/hmda/diabetes cell:

  * TASKS: the official folktables task objects, each with its OWN standard
    population filter and target transform (e.g. ACSIncome keeps employed
    adults with PINCP > 100 and predicts PINCP > 50k). We use the official
    per-task populations — not a unified frame — so the label-coupling
    numbers are comparable to the fairness literature that uses these tasks.
  * STATE/YEAR: fixed single state CA (largest 1-Year sample), 2018 1-Year.
  * ATTRIBUTES (binary, matching PCRL's pcrl/data/folktables.py and the
    FFB convention of Han et al. ICLR 2024):
      sex  = 1 iff SEX == 1   (ACS code 1 = Male)
      race = 1 iff RAC1P == 1 (ACS code 1 = "White alone"; 0 = non-White)
  * FEATURES: standardized numerics (AGEP, WKHP, PINCP) + one-hot
    categoricals with a NaN -> -1 bucket; SCHL is bucketed to the same
    4-level education group and OCCP/POBP to the same coarse buckets as
    pcrl/data/folktables.py. SEX and RAC1P (binarized) stay in the feature
    set — the same "attribute is also an input" convention as the existing
    tabular cells. ST is dropped (constant: single state).
  * SPLIT: deterministic stratified 80/20 train/holdout split, seed 42,
    stratified on (task label, sex, race) jointly so both attributes of a
    task share one train partition — the analogue of the PCRL datasets'
    seed-42 dataset-level train split. Experiments run on the train
    partition; probes make their own 75/25 stratified held-out splits
    downstream, exactly like the existing cells.

Raw PUMS source: reuses the (read-only) CSV already cached under
$PCRL_ROOT/data/folktables when present; otherwise downloads into this
repo's data_cache/folktables. Processed per-task arrays are cached to
data_cache/folktables/*.npz so re-runs skip the 255 MB CSV parse.

Requires the optional `folktables` package (see requirements.txt).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data_cache" / "folktables"

STATE = "CA"
YEAR = "2018"
HORIZON = "1-Year"
SPLIT_SEED = 42
TRAIN_FRAC = 0.80

SEX_MALE = 1     # ACS SEX == 1 is Male
WHITE_RAC1P = 1  # ACS RAC1P == 1 is "White alone"

TASK_NAMES = ("income", "employment", "public_coverage")
ATTR_NAMES = ("sex", "race")

# Columns treated as standardized numerics; everything else in a task's
# feature list is one-hot encoded (after the bucketings below).
NUMERIC_COLS = ("AGEP", "WKHP", "PINCP")
DROP_COLS = ("ST",)  # constant under a single-state pull

PROVENANCE = {
    "source": "folktables ACS PUMS person survey",
    "state": STATE,
    "survey_year": YEAR,
    "horizon": HORIZON,
    "tasks": {
        "income": "ACSIncome (official filter; label = PINCP > 50000)",
        "employment": "ACSEmployment (official filter; label = ESR == 1)",
        "public_coverage": "ACSPublicCoverage (official filter; label = PUBCOV == 1)",
    },
    "attributes": {
        "sex": "1 iff SEX == 1 (Male)",
        "race": "1 iff RAC1P == 1 (White alone) — binarized White vs non-White",
    },
    "features": "standardized numerics (AGEP/WKHP/PINCP where present) + one-hot "
                "categoricals with NaN->-1 bucket; SCHL 4-level bucket, OCCP 2-digit "
                "bucket, POBP {state,territory,foreign} bucket (same bucketings as "
                "pcrl/data/folktables.py); binarized SEX/RAC1P kept as features; ST dropped",
    "split": f"stratified {int(TRAIN_FRAC*100)}/{int((1-TRAIN_FRAC)*100)} train/holdout, "
             f"seed {SPLIT_SEED}, stratified on (task label, sex, race); experiments use "
             "the train partition; probes use 75/25 stratified held-out splits downstream",
}


# ── Bucketings (same as pcrl/data/folktables.py) ────────────────────────
def _bucket_schl(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    out = pd.Series(0, index=s.index, dtype=np.int64)  # NaN and <HS -> 0
    out[v.between(16, 17)] = 1    # HS-grad / GED
    out[v.between(18, 20)] = 2    # Some college / associate
    out[v.between(21, 24)] = 3    # Bachelors+
    return out


def _bucket_occp(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    return (v // 100).clip(lower=0, upper=99).fillna(-1).astype(np.int64)


def _bucket_pobp(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    out = pd.Series(-1, index=s.index, dtype=np.int64)
    out[v.between(1, 56)] = 0     # US state
    out[v.between(60, 78)] = 1    # US territory
    out[v >= 100] = 2             # foreign
    return out


# ── Raw data ────────────────────────────────────────────────────────────
def _acs_root() -> tuple[Path, bool]:
    """(root_dir for ACSDataSource, download flag). Prefer the read-only CSV
    already cached under $PCRL_ROOT/data/folktables; else our data_cache."""
    pcrl_root = os.environ.get("PCRL_ROOT")
    if pcrl_root:
        cand = Path(pcrl_root) / "data" / "folktables"
        if (cand / YEAR / HORIZON).exists():
            return cand, False
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR, True


def _load_raw() -> pd.DataFrame:
    from folktables import ACSDataSource

    root, download = _acs_root()
    src = ACSDataSource(survey_year=YEAR, horizon=HORIZON, survey="person",
                        root_dir=str(root))
    return src.get_data(states=[STATE], download=download)


def _task_obj(task_name: str):
    import folktables

    return {"income": folktables.ACSIncome,
            "employment": folktables.ACSEmployment,
            "public_coverage": folktables.ACSPublicCoverage}[task_name]


# ── Encoding ────────────────────────────────────────────────────────────
def _encode(feat: pd.DataFrame, train_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """One-hot categoricals + train-standardized numerics -> float32 X."""
    cols: list[np.ndarray] = []
    names: list[str] = []
    for col in feat.columns:
        if col in DROP_COLS:
            continue
        if col == "SCHL":
            series = _bucket_schl(feat[col])
        elif col == "OCCP":
            series = _bucket_occp(feat[col])
        elif col == "POBP":
            series = _bucket_pobp(feat[col])
        elif col == "SEX":
            series = (pd.to_numeric(feat[col], errors="coerce") == SEX_MALE).astype(np.int64)
        elif col == "RAC1P":
            series = (pd.to_numeric(feat[col], errors="coerce") == WHITE_RAC1P).astype(np.int64)
        elif col in NUMERIC_COLS:
            v = pd.to_numeric(feat[col], errors="coerce").fillna(0).to_numpy(np.float32)
            mean = float(v[train_mask].mean())
            std = float(v[train_mask].std()) + 1e-8
            cols.append(((v - mean) / std).reshape(-1, 1))
            names.append(col)
            continue
        else:
            series = pd.to_numeric(feat[col], errors="coerce").fillna(-1).astype(np.int64)
        if col in ("SEX", "RAC1P"):
            # already binary 0/1 — keep as a single column, like the PCRL loader
            cols.append(series.to_numpy(np.float32).reshape(-1, 1))
            names.append(f"{col}_bin")
            continue
        for val in sorted(series.unique()):
            cols.append((series == val).to_numpy(np.float32).reshape(-1, 1))
            names.append(f"{col}={val}")
    return np.concatenate(cols, axis=1).astype(np.float32), names


def _stratified_train_mask(label: np.ndarray, sex: np.ndarray, race: np.ndarray) -> np.ndarray:
    """Deterministic stratified 80/20 mask on (label, sex, race), seed 42."""
    rng = np.random.RandomState(SPLIT_SEED)
    mask = np.zeros(len(label), dtype=bool)
    strata = label * 4 + sex * 2 + race
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        rng.shuffle(idx)
        mask[idx[: int(round(TRAIN_FRAC * len(idx)))]] = True
    return mask


# ── Public API ──────────────────────────────────────────────────────────
def build_task(task_name: str, use_cache: bool = True) -> dict[str, np.ndarray]:
    """Build (or load cached) train-partition arrays for one folktables task.

    Returns dict with X (float32, standardized/one-hot), sex, race, label
    (int64), n_total (task population before the 80/20 split), x_names.
    """
    if task_name not in TASK_NAMES:
        raise ValueError(f"unknown folktables task {task_name!r}; use one of {TASK_NAMES}")
    cache = CACHE_DIR / f"{task_name}_{STATE}_{YEAR}_train.npz"
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}

    t0 = time.time()
    raw = _load_raw()
    task = _task_obj(task_name)
    feat, lab, _ = task.df_to_pandas(raw)
    label = lab.iloc[:, 0].astype(bool).to_numpy(np.int64)
    sex = (pd.to_numeric(feat["SEX"], errors="coerce") == SEX_MALE).to_numpy(np.int64)
    race = (pd.to_numeric(feat["RAC1P"], errors="coerce") == WHITE_RAC1P).to_numpy(np.int64)

    train_mask = _stratified_train_mask(label, sex, race)
    X, x_names = _encode(feat, train_mask)

    out = {"X": X[train_mask], "sex": sex[train_mask], "race": race[train_mask],
           "label": label[train_mask], "n_total": np.int64(len(label)),
           "x_names": np.array(x_names)}
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)
        cache.with_suffix(".provenance.json").write_text(
            json.dumps({**PROVENANCE, "task": task_name, "n_total": int(len(label)),
                        "n_train": int(train_mask.sum()), "x_dim": int(X.shape[1]),
                        "build_seconds": round(time.time() - t0, 1)}, indent=2))
    return out


def load_folktables_cell(task_name: str, attr_name: str, max_n: int | None = None):
    """Mirror of diagnostic.load_cell for a folktables cell.

    Returns (X, attr, task, n_attr, n_task, attr_maj, task_maj) on the train
    partition. max_n: optional deterministic stratified subsample (seed 42,
    stratified on (label, attr)) to keep per-cell runtime comparable to the
    existing tabular cells.
    """
    if attr_name not in ATTR_NAMES:
        raise ValueError(f"unknown attribute {attr_name!r}; use one of {ATTR_NAMES}")
    d = build_task(task_name)
    X, attr, task = d["X"], d[attr_name].astype(np.int64), d["label"].astype(np.int64)
    if max_n is not None and max_n < len(task):
        rng = np.random.RandomState(SPLIT_SEED)
        keep = np.zeros(len(task), dtype=bool)
        strata = task * 2 + attr
        frac = max_n / len(task)
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            rng.shuffle(idx)
            keep[idx[: int(round(frac * len(idx)))]] = True
        X, attr, task = X[keep], attr[keep], task[keep]
    n_attr = int(attr.max()) + 1
    n_task = int(task.max()) + 1
    attr_maj = float(np.bincount(attr).max() / len(attr))
    task_maj = float(np.bincount(task).max() / len(task))
    return X, attr, task, n_attr, n_task, attr_maj, task_maj
