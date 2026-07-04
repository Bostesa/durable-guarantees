"""Repair 1 — LAFTR row re-run on the OFFICIAL code (VectorInstitute/laftr).

PROVENANCE: OFFICIAL CODE. https://github.com/VectorInstitute/laftr at commit
a166ba3c95fd48afcb4657208be6f6d01b91aedc (2018-09-07) — the release for
Madras et al. 2018. The release is TensorFlow-1.9/Python-3.6-era; it runs
UNMODIFIED under TF 2.21's tf.compat.v1 via an environment shim
(laftr_shim.py next to the clone: sys.modules['tensorflow']=tf.compat.v1,
collections.Iterator/Mapping -> collections.abc, tf.contrib.layers.
xavier_initializer -> glorot_uniform_initializer — the identical
initializer; their source tree is untouched). Shim sanity anchor: on THEIR
data/adult/adult.npz at their full-sweep protocol, the shimmed code lands in
the paper's Adult region (accuracy 0.844-0.848 vs published ~0.84; DI
0.05-0.08 at the final epoch; the joint adversary reads A at the majority
rate — it fools itself, as our Exp-16 reimplementation also found).

Released protocol, their defaults throughout: WeightedDemParWassGan (their
conf/laftr/config.json default — the group-normalized L1 adversary),
enc/cla/rec/aud hidden [8], zdim 8, use_attr=true (their Adult config
concatenates the protected attribute onto x!), batch 64, 1000 epochs,
patience 1000, Adam 1e-3, final-epoch model evaluated (their laftr.py).

Data-loading-boundary adaptation only (utils/export_cells_for_laftr.py, see
the exported laftr_cells/*/meta.json): our three gauntlet cells in their npz
schema; train = ALL rows (every gauntlet baseline trains on the full cell;
valid = seed-0 10%, logging-only at patience 1000); x_test = ALL rows in
original order so the saved tester tensors align row-for-row with our
attr/task for the battery (their tester trims to batch multiples; we
truncate our labels to match).

SCOPE FACT (5-class attribute): the official release is BINARY-A/BINARY-Y by
construction (2x2 AY weight grids; PR/NR/TP/FP metrics on 0/1 arrays). On
the two HMDA race cells it does NOT crash — mean(A)+1-mean(A)=1 passes their
assert — it silently treats race in {0..4} as a scalar "probability" target
for a 2-group objective, and (for loan_amount_band) trains its head on the
"band==1" indicator column. We run it as released anyway (their code, our
cells, method untouched) and let the battery judge the representation; the
multiclass-generalized reimplementation row (experiments/baseline_gauntlet.py,
macro-per-group CE adversary) remains the charitable version, and the two are
validated head-to-head on the adult cell where both are well-defined.

Battery/certification protocol mirrors experiments/baseline_gauntlet.py
exactly: knob sweep gamma in {0.5,1,2,4,8} at model.seed 0, Tier candidates
(<=0.55) certified best-utility-first over model.seed {0,1,2} (this script
shells out to the official pipeline for the extra seeds), full battery on
representation AND output (their Y_hat sigmoid -> 2-col logits; for the
5-class band task their binary head cannot express the task, so utility
additionally comes from the retrained-LR head through Z — lift_best as
everywhere). Sigma_noise=0 for the Tier-2 LRT (deterministic encoder).

Launches any missing official runs itself (run_official checks done.txt), so
it is self-contained given: a clone of the official repo, a python venv with
modern tensorflow (LAFTR_PY), the shim (utils/laftr_shim.py), and the exported
cells (utils/export_cells_for_laftr.py). Env overrides: LAFTR_ROOT,
LAFTR_RUNS, LAFTR_CELLS, LAFTR_PY, LAFTR_SHIM.
Local MPS/CPU; never AWS. macOS OpenMP guard: xgboost before torch.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401  (import before torch on purpose)

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.baseline_gauntlet import (  # noqa: E402
    BAR, CERT_SEEDS, MAX_CERT_CANDIDATES, TRAIN_SEEDS, _exposure,
    load_ours, measure_sweep,
)
from experiments.diagnostic import label_predictor, lift, load_cell  # noqa: E402
from experiments.targeted_noise import utility_through  # noqa: E402
from experiments.two_tier_certification import fmt_point, full_point  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

# clone https://github.com/VectorInstitute/laftr (commit below) as a sibling of
# this repo (or set LAFTR_ROOT); LAFTR_PY = a python3.12 venv with modern
# tensorflow; LAFTR_SHIM = utils/laftr_shim.py; LAFTR_CELLS = the exported
# cells (utils/export_cells_for_laftr.py)
LAFTR_ROOT = Path(os.environ.get("LAFTR_ROOT", REPO_ROOT.parent / "laftr"))
LAFTR_RUNS = Path(os.environ.get("LAFTR_RUNS", LAFTR_ROOT / "experiments"))
LAFTR_CELLS = Path(os.environ.get("LAFTR_CELLS",
                                  REPO_ROOT.parent / "laftr_cells"))
LAFTR_PY = os.environ.get("LAFTR_PY", "python3")
LAFTR_SHIM = os.environ.get("LAFTR_SHIM",
                            str(REPO_ROOT / "utils" / "laftr_shim.py"))

RESULTS_DIR = REPO_ROOT / "results"
OFFICIAL_COMMIT = "a166ba3c95fd48afcb4657208be6f6d01b91aedc"
GAMMAS = [0.5, 1.0, 2.0, 4.0, 8.0]
BATCH = 64                       # their batch size (tensor trimming)

CELLS = [
    dict(name="hmda/race/loan_decision", dataset="hmda", attr="race",
         task="loan_decision", tag="hmda_race_loan_decision", xdim=78),
    dict(name="hmda/race/loan_amount_band", dataset="hmda", attr="race",
         task="loan_amount_band", tag="hmda_race_loan_amount_band", xdim=78),
    dict(name="adult/sex/income", dataset="adult", attr="sex", task="income",
         tag="adult_sex_income", xdim=105),
]

CALIBRATION = dict(
    note="official code under the shim on THEIR data/adult/adult.npz, their "
         "full protocol (WeightedDemParWassGan, 1000 epochs); paper Adult "
         "region is accuracy ~0.84",
    runs={"gamma=0.1": dict(acc=0.8436, DP=0.1224, DI=0.0493, adv_acc=0.4515),
          "gamma=1.0": dict(acc=0.8476, DP=0.1813, DI=0.0842, adv_acc=0.6737),
          "gamma=4.0": dict(acc=0.8459, DP=0.1738, DI=0.0699, adv_acc=0.6737)},
    attr_majority=0.674,
)


def run_official(tag, xdim, gamma, seed):
    """Launch one official run (blocking) if its outputs don't exist yet."""
    name = f"our_{tag}_g{gamma:g}_s{seed}"
    exp_dir = LAFTR_RUNS / name
    if (exp_dir / "done.txt").exists():
        return exp_dir
    print(f"    [official run] {name} (1000 epochs)...", flush=True)
    cmd = [LAFTR_PY, LAFTR_SHIM, "src/laftr.py", "conf/laftr/config.json",
           "-o", f"exp_name={name},train.n_epochs=1000,"
                 f"model.fair_coeff={gamma:g},model.seed={seed},"
                 f"model.xdim={xdim},dirs.data_dir={LAFTR_CELLS / tag}",
           "--data", "adult", "--dirs", "local"]
    r = subprocess.run(cmd, cwd=LAFTR_ROOT, capture_output=True, text=True)
    if not (exp_dir / "done.txt").exists():
        raise RuntimeError(f"official run {name} failed:\n{r.stdout[-2000:]}"
                           f"\n{r.stderr[-2000:]}")
    return exp_dir


def load_exposure(exp_dir, attr, task, gamma):
    """Z + Y_hat from the official tester (test phase = all rows, trimmed to
    batch multiples, original order)."""
    Z = np.load(exp_dir / "npz" / "Z.npz")["X"].astype(np.float32)
    p = np.load(exp_dir / "npz" / "Y_hat.npz")["X"].astype(np.float64)
    n = Z.shape[0]
    assert n == (len(attr) // BATCH) * BATCH, (n, len(attr))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    L = np.concatenate([np.log(1 - p), np.log(p)], axis=1).astype(np.float32)
    exp = _exposure(f"γ={gamma:g}", gamma, Z, L)
    return exp, attr[:n], task[:n]


def certify_official(cell, gamma, ctx_full, device):
    """3-model-seed certification at gamma: official runs at model.seed
    {0,1,2}, full battery on representation AND output."""
    attr_f, task_f = ctx_full["attr"], ctx_full["task"]
    n_attr, task_maj = ctx_full["n_attr"], ctx_full["task_maj"]
    keys = ("xgb", "mlp", "lora", "lrt")
    per = {f"rep_{k}": [] for k in keys}
    per.update({f"out_{k}": [] for k in keys})
    per["lift_own"], per["lift_lr"] = [], []
    for ts in TRAIN_SEEDS:
        exp_dir = run_official(cell["tag"], cell["xdim"], gamma, ts)
        exp, attr, task = load_exposure(exp_dir, attr_f, task_f, gamma)
        d = exp["P"].shape[1]
        rp = full_point(exp["P"], exp["P"], attr, n_attr,
                        np.zeros((d, d)), device, CERT_SEEDS)
        d_out = exp["L"].shape[1]
        op = full_point(exp["L"], exp["L"], attr, n_attr,
                        np.zeros((d_out, d_out)), device, CERT_SEEDS)
        for k in keys:
            per[f"rep_{k}"].append(rp[k])
            per[f"out_{k}"].append(op[k])
        per["lift_own"].append(lift(exp["L"], task, task_maj))
        per["lift_lr"].append(utility_through(exp["P"], task,
                                              lambda X, rng: X,
                                              CERT_SEEDS, task_maj))
        print(f"      [model.seed {ts}] rep {fmt_point(rp)} | "
              f"out T2={op['tier2_max']:.3f} | lift own="
              f"{per['lift_own'][-1]:+.4f} LR={per['lift_lr'][-1]:+.4f}",
              flush=True)
    c = {f"{k}_mean": float(np.mean(v)) for k, v in per.items()}
    c.update({f"{k}_std": float(np.std(v)) for k, v in per.items()})
    c["per_seed"] = per
    c["rep_tier1_max"] = float(max(c[f"rep_{k}_mean"] for k in ("xgb", "mlp", "lora")))
    c["rep_tier2_max"] = float(max(c["rep_tier1_max"], c["rep_lrt_mean"]))
    c["out_tier1_max"] = float(max(c[f"out_{k}_mean"] for k in ("xgb", "mlp", "lora")))
    c["out_tier2_max"] = float(max(c["out_tier1_max"], c["out_lrt_mean"]))
    c["lift_best"] = float(max(c["lift_own_mean"], c["lift_lr_mean"]))
    c["label"], c["knob"] = f"γ={gamma:g}", gamma
    return c


def main():
    device = get_device()
    ours = load_ours()
    print(f"[setup] official LAFTR commit {OFFICIAL_COMMIT[:12]} at "
          f"{LAFTR_ROOT}; runs in {LAFTR_RUNS}", flush=True)
    results = []
    for cell in CELLS:
        t0 = time.time()
        print("\n" + "#" * 100 + f"\n# CELL: {cell['name']} (official LAFTR)"
              + "\n" + "#" * 100, flush=True)
        X, attr, task, n_attr, n_task, attr_maj, task_maj = load_cell(
            cell["dataset"], cell["attr"], cell["task"])
        pred, _ = label_predictor(attr, task, n_attr)
        rows = []
        n_trim = (len(attr) // BATCH) * BATCH
        ctx = dict(name=cell["name"], attr=attr[:n_trim], task=task[:n_trim],
                   n_attr=n_attr, n_task=n_task, task_maj=task_maj)
        ctx_full = dict(attr=attr, task=task, n_attr=n_attr,
                        task_maj=task_maj)
        for g in GAMMAS:
            exp_dir = run_official(cell["tag"], cell["xdim"], g, 0)
            exp, a_t, t_t = load_exposure(exp_dir, attr, task, g)
            row = measure_sweep(exp, ctx, device)
            rows.append(row)

        cert_cache = {}

        def tier_pick(tkey):
            cands = sorted([r for r in rows if r[f"{tkey}_max"] <= BAR],
                           key=lambda r: -r["lift_best"])
            tried = []
            for r in cands[:MAX_CERT_CANDIDATES]:
                if r["label"] not in cert_cache:
                    print(f"    [certify official-LAFTR {r['label']}]",
                          flush=True)
                    cert_cache[r["label"]] = certify_official(
                        cell, r["knob"], ctx_full, device)
                c = cert_cache[r["label"]]
                tried.append(r["label"])
                if c[f"rep_{tkey}_max"] <= BAR:
                    return dict(certified=True, cert=c, tried=tried)
            best = min(rows, key=lambda r: r[f"{tkey}_max"])
            return dict(certified=False, cert=None, tried=tried, closest=best)

        t1, t2 = tier_pick("tier1"), tier_pick("tier2")
        for nm, t in (("TIER 1", t1), ("TIER 2", t2)):
            if t["certified"]:
                c = t["cert"]
                print(f"  >>> official LAFTR {nm}: CERTIFIES at {c['label']} "
                      f"— lift {c['lift_best']:+.4f}", flush=True)
            else:
                b = t["closest"]
                tv = b["tier1_max"] if nm == "TIER 1" else b["tier2_max"]
                print(f"  >>> official LAFTR {nm}: does NOT certify — "
                      f"closest {b['label']} max={tv:.3f} at lift "
                      f"{b['lift_best']:+.4f}", flush=True)

        def _clean(o):
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()
                        if k not in ("P", "L", "h_clean", "ncov")}
            if isinstance(o, list):
                return [_clean(v) for v in o]
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            return o

        results.append(_clean(dict(
            cell=cell["name"], predictor=pred,
            clean_lift=ours[cell["name"]]["clean_lift"],
            baseline="LAFTR (official)", rows=rows, tier1=t1, tier2=t2,
            certs=cert_cache, minutes=round((time.time() - t0) / 60, 1),
            scope_note=("binary-A/binary-Y release run as released; 5-class "
                        "race treated by their code as a scalar target"
                        if n_attr > 2 else "natively supported (binary/binary)"))))

    out = {
        "experiment": "laftr_official", "bar": BAR, "gammas": GAMMAS,
        "provenance": {
            "kind": "official code",
            "repo": "https://github.com/VectorInstitute/laftr",
            "commit": OFFICIAL_COMMIT,
            "environment": "TF-1.9-era source unmodified under TF 2.21 "
                           "tf.compat.v1 via laftr_shim.py (module alias, "
                           "collections.abc alias, xavier->glorot_uniform "
                           "— identical initializer)",
            "protocol": "WeightedDemParWassGan, their defaults: hidden [8], "
                        "zdim 8, use_attr=true, batch 64, 1000 epochs, "
                        "final-epoch model; data boundary only (train=all "
                        "rows, x_test=all rows for tensor alignment)",
            "calibration_on_their_data": CALIBRATION,
        },
        "cells": results,
    }
    (RESULTS_DIR / "laftr_official.json").write_text(json.dumps(out, indent=2))
    print(f"\n[save] {RESULTS_DIR / 'laftr_official.json'}", flush=True)


if __name__ == "__main__":
    main()
