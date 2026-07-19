"""Expansion natural cells — three additional public tabular sources beyond
folktables (see utils/folktables_io.py for the fourth):

  * lawschool          LSAC bar-passage (Wightman 1998), cleaned CSV from the
                       Le Quy et al. (2022) fairness-dataset survey mirror
                       (tailequy/fairness_dataset). Label pass_bar — the file's
                       single bar-passage label; per the survey's dataset table
                       this is FIRST-TIME bar passage (recorded in provenance).
                       Attrs: sex (male=1), race binarized White=1/Non-White=0
                       (the file ships race already binarized to those two
                       levels).
  * dutch              Dutch Census 2001, cleaned CSV from the same survey
                       mirror. Label occupation (1 = high-level/prestigious
                       occupation, 0 = low). Attr: sex (male=1). All feature
                       columns are integer-coded categoricals -> one-hot.
  * diabetes_hospital  Diabetes 130-US Hospitals via
                       fairlearn.datasets.fetch_diabetes_hospital (fairlearn's
                       cleaned OpenML build; weight/payer_code already absent).
                       Label readmit_30_days (30-day readmission). Cleaning:
                       drop medical_specialty (~49% Missing), drop the
                       readmitted/readmit_binary leakage columns, drop rows
                       with race Unknown or gender Unknown/Invalid.
                       Attrs: gender (Male=1), race binarized Caucasian=1/
                       non-Caucasian=0.

Conventions identical to utils/folktables_io.py (and thereby to the existing
tabular cells): standardized numerics (train-fit stats), one-hot categoricals,
binarized protected attribute kept in the feature set, deterministic
stratified 80/20 train/holdout split (seed 42, stratified on label x attrs;
both attributes of a source share one train partition). Experiments run on
the train partition; probes make their own 75/25 stratified held-out splits
downstream. Processed arrays cached to data_cache/expansion/*.npz.

Requires the optional `fairlearn` package for diabetes_hospital
(see requirements.txt); lawschool/dutch need only the cached CSVs (downloaded
from the recorded URLs if absent).
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data_cache" / "expansion"

SPLIT_SEED = 42
TRAIN_FRAC = 0.80

LAW_URL = ("https://raw.githubusercontent.com/tailequy/fairness_dataset/main/"
           "experiments/data/law_school_clean.csv")
DUTCH_URL = ("https://raw.githubusercontent.com/tailequy/fairness_dataset/main/"
             "experiments/data/dutch.csv")

SOURCES = {
    "lawschool": {
        "tasks": ("pass_bar",),
        "attrs": ("sex", "race"),
        "provenance": {
            "source": "LSAC National Longitudinal Bar Passage Study (Wightman 1998)",
            "url": LAW_URL,
            "label": "pass_bar — the cleaned file's single bar-passage label; per the "
                     "Le Quy et al. (2022) survey (arXiv:2110.00530) dataset table this "
                     "is first-time bar passage (not eventual passage)",
            "attributes": {"sex": "1 iff male (file column 'male')",
                           "race": "1 iff White, 0 Non-White (file ships race already "
                                   "binarized to these two levels)"},
        },
    },
    "dutch": {
        "tasks": ("occupation",),
        "attrs": ("sex",),
        "provenance": {
            "source": "Dutch Census 2001 (Dutch Central Bureau of Statistics)",
            "url": DUTCH_URL,
            "label": "occupation — 1 = high-level (prestigious) occupation, 0 = low, "
                     "as distributed in the cleaned survey CSV",
            "attributes": {"sex": "1 iff male (file column 'sex' in {male,female})"},
        },
    },
    "diabetes_hospital": {
        "tasks": ("readmit_30_days",),
        "attrs": ("gender", "race"),
        "provenance": {
            "source": "Diabetes 130-US Hospitals 1999-2008 (Strack et al. 2014) via "
                      "fairlearn.datasets.fetch_diabetes_hospital (OpenML build)",
            "url": "fairlearn.datasets.fetch_diabetes_hospital",
            "label": "readmit_30_days — readmitted within 30 days (fairlearn target)",
            "attributes": {"gender": "1 iff Male (rows with Unknown/Invalid dropped)",
                           "race": "1 iff Caucasian, 0 non-Caucasian (rows with race "
                                   "Unknown dropped)"},
            "cleaning": "dropped medical_specialty (~49% Missing) and the "
                        "readmitted/readmit_binary target-leakage columns; "
                        "weight/payer_code already absent from the fairlearn build",
        },
    },
}

SPLIT_PROVENANCE = (
    f"stratified {int(TRAIN_FRAC*100)}/{int((1-TRAIN_FRAC)*100)} train/holdout, "
    f"seed {SPLIT_SEED}, stratified on (label x attributes); experiments use the "
    "train partition; probes use 75/25 stratified held-out splits downstream"
)


# ── Shared helpers (same conventions as folktables_io) ──────────────────
def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[expansion_io] downloading {url}")
    urllib.request.urlretrieve(url, dest)


def _encode(df: pd.DataFrame, numeric_cols: list[str], train_mask: np.ndarray,
            binary_cols: list[str] = ()) -> tuple[np.ndarray, list[str]]:
    """Standardized numerics (train-fit) + passthrough binaries + one-hot rest."""
    cols, names = [], []
    for col in df.columns:
        if col in numeric_cols:
            v = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(np.float32)
            mean = float(v[train_mask].mean())
            std = float(v[train_mask].std()) + 1e-8
            cols.append(((v - mean) / std).reshape(-1, 1))
            names.append(col)
        elif col in binary_cols:
            cols.append(df[col].to_numpy(np.float32).reshape(-1, 1))
            names.append(f"{col}_bin")
        else:
            s = df[col].astype(str)
            for val in sorted(s.unique()):
                cols.append((s == val).to_numpy(np.float32).reshape(-1, 1))
                names.append(f"{col}={val}")
    return np.concatenate(cols, axis=1).astype(np.float32), names


def _stratified_train_mask(strata: np.ndarray) -> np.ndarray:
    rng = np.random.RandomState(SPLIT_SEED)
    mask = np.zeros(len(strata), dtype=bool)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        rng.shuffle(idx)
        mask[idx[: int(round(TRAIN_FRAC * len(idx)))]] = True
    return mask


# ── Per-source builders: return (feature_df, numeric_cols, binary_cols,
#    attrs dict, labels dict) on the CLEANED full frame ─────────────────
def _build_lawschool():
    path = CACHE_DIR / "law_school_clean.csv"
    if not path.exists():
        _download(LAW_URL, path)
    df = pd.read_csv(path)
    attrs = {"sex": df["male"].astype(int).to_numpy(np.int64),
             "race": (df["race"] == "White").astype(np.int64).to_numpy()}
    labels = {"pass_bar": df["pass_bar"].astype(int).to_numpy(np.int64)}
    feat = df.drop(columns=["pass_bar", "male", "race"]).copy()
    feat["sex"] = attrs["sex"]
    feat["race"] = attrs["race"]
    numeric = ["decile1b", "decile3", "lsat", "ugpa", "zfygpa", "zgpa", "fam_inc", "tier"]
    binary = ["sex", "race", "fulltime"]
    feat["fulltime"] = (feat["fulltime"] == 1.0).astype(int)
    return feat, numeric, binary, attrs, labels


def _build_dutch():
    path = CACHE_DIR / "dutch.csv"
    if not path.exists():
        _download(DUTCH_URL, path)
    df = pd.read_csv(path)
    attrs = {"sex": (df["sex"] == "male").astype(np.int64).to_numpy()}
    labels = {"occupation": df["occupation"].astype(int).to_numpy(np.int64)}
    feat = df.drop(columns=["occupation", "sex"]).copy()
    feat["sex"] = attrs["sex"]
    # every remaining column is an integer-coded categorical -> one-hot
    return feat, [], ["sex"], attrs, labels


def _build_diabetes_hospital():
    from fairlearn.datasets import fetch_diabetes_hospital

    b = fetch_diabetes_hospital(as_frame=True, cache=True)
    df = b.data.copy()
    df["readmit_30_days"] = b.target.astype(int)
    keep = (df["race"] != "Unknown") & (df["gender"].isin(["Male", "Female"]))
    df = df.loc[keep].reset_index(drop=True)
    attrs = {"gender": (df["gender"] == "Male").astype(np.int64).to_numpy(),
             "race": (df["race"] == "Caucasian").astype(np.int64).to_numpy()}
    labels = {"readmit_30_days": df["readmit_30_days"].to_numpy(np.int64)}
    feat = df.drop(columns=["readmit_30_days", "readmitted", "readmit_binary",
                            "medical_specialty", "gender", "race"]).copy()
    feat["gender"] = attrs["gender"]
    feat["race"] = attrs["race"]
    numeric = ["time_in_hospital", "num_lab_procedures", "num_procedures",
               "num_medications", "number_diagnoses"]
    binary = ["gender", "race"]
    return feat, numeric, binary, attrs, labels


_BUILDERS = {"lawschool": _build_lawschool, "dutch": _build_dutch,
             "diabetes_hospital": _build_diabetes_hospital}


# ── Public API (mirrors folktables_io) ──────────────────────────────────
def build_source(source: str, use_cache: bool = True) -> dict[str, np.ndarray]:
    """Build (or load cached) train-partition arrays for one source.

    Returns dict with X, per-attr arrays, per-label arrays, n_total, x_names.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; use one of {sorted(SOURCES)}")
    cache = CACHE_DIR / f"{source}_train.npz"
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}

    t0 = time.time()
    feat, numeric, binary, attrs, labels = _BUILDERS[source]()
    label = labels[SOURCES[source]["tasks"][0]]
    strata = label.copy()
    for a in SOURCES[source]["attrs"]:
        strata = strata * 2 + attrs[a]
    train_mask = _stratified_train_mask(strata)
    X, x_names = _encode(feat, numeric, train_mask, binary)

    out = {"X": X[train_mask], "n_total": np.int64(len(label)),
           "x_names": np.array(x_names)}
    for a, v in attrs.items():
        out[f"attr_{a}"] = v[train_mask]
    for t, v in labels.items():
        out[f"label_{t}"] = v[train_mask]
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)
        cache.with_suffix(".provenance.json").write_text(json.dumps(
            {**SOURCES[source]["provenance"], "split": SPLIT_PROVENANCE,
             "n_total": int(len(label)), "n_train": int(train_mask.sum()),
             "x_dim": int(X.shape[1]),
             "build_seconds": round(time.time() - t0, 1)}, indent=2))
    return out


def load_expansion_cell(source: str, attr_name: str, task_name: str,
                        max_n: int | None = None):
    """Mirror of diagnostic.load_cell for an expansion cell (folktables cells
    live in utils/folktables_io.load_folktables_cell).

    Returns (X, attr, task, n_attr, n_task, attr_maj, task_maj) on the train
    partition. max_n: deterministic stratified subsample (seed 42, stratified
    on (label, attr)) for runtime parity with the existing tabular cells.
    """
    d = build_source(source)
    X = d["X"]
    attr = d[f"attr_{attr_name}"].astype(np.int64)
    task = d[f"label_{task_name}"].astype(np.int64)
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


def cell_provenance(source: str) -> dict:
    """Provenance block for one source (for the expansion results JSONs)."""
    if source == "folktables":
        from utils.folktables_io import PROVENANCE

        return dict(PROVENANCE)
    return {**SOURCES[source]["provenance"], "split": SPLIT_PROVENANCE}
