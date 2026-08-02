"""Fair PCA (amazon-science/fair-pca, Apache-2.0) under the two-tier gauntlet.

Registered prediction: docs/fairpca_prediction.md, committed at 8b7d873 BEFORE this
file existed. Conventions identical to the FARE / baseline-gauntlet runs:
3 training seeds, sweep probe seeds [0,1], certification probe seeds [0,1,2],
bar 0.55 on the 3-seed mean, both surfaces (representation and output),
utility normalized by the cell's clean e2e lift from
results/two_tier_certification.json. Deterministic release => Tier-2 is the
informed QDA (Sigma_noise = 0), the LEACE convention.

Their code is called unmodified from src/fair_pca/fair_PCA.py; nothing is
reimplemented. Local CPU/MPS; never AWS. macOS OpenMP guard: xgboost first.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
# Official fair-PCA checkout; defaults to ../fair-pca beside this repository.
FAIRPCA_ROOT = Path(os.environ.get("FAIRPCA_ROOT", REPO.parent / "fair-pca")).resolve()
if not FAIRPCA_ROOT.exists():
    raise FileNotFoundError(
        f"fair-PCA checkout not found at {FAIRPCA_ROOT}. Set $FAIRPCA_ROOT to "
        "the official clone.")
sys.path.insert(0, str(FAIRPCA_ROOT))

from experiments.diagnostic import (  # noqa: E402
    lift, load_cell, probe_sklearn,
)
from experiments.sealed_channel import train_model  # noqa: E402
from experiments.targeted_noise import utility_through  # noqa: E402
from experiments.two_tier_certification import full_point, fmt_point  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

RESULTS = REPO / "results"
BAR = 0.55
SWEEP_SEEDS = [0, 1]
CERT_SEEDS = [0, 1, 2]
TRAIN_SEEDS = [0, 1, 2]
FAIRPCA_COMMIT = "18fac69ed9237b567eb7aad94774d501a13a04bf"

CELLS = {
    "easy": ("hmda", "race", "loan_decision"),
    "middle": ("hmda", "race", "loan_amount_band"),
    "hard": ("adult", "sex", "income"),
}
TARGET_DIMS = [2, 4, 8, 16, 32]
TRADEOFFS = [0.0, 0.25, 0.5]


# --------------------------------------------------------------------------- #
#  Gate                                                                        #
# --------------------------------------------------------------------------- #
def gate():
    print("=" * 92 + "\nGATE — fair-pca @ " + FAIRPCA_COMMIT[:7] + "\n" + "=" * 92,
          flush=True)
    out = {"commit": FAIRPCA_COMMIT, "license": None, "shipped_example": None,
           "invariant_mean_equalization": None, "sec36_improves_cov": None}

    lic = (FAIRPCA_ROOT / "LICENSE").read_text()[:400]
    out["license"] = "Apache-2.0" if "Apache License" in lic else "UNKNOWN"
    print(f"  [license] {out['license']}", flush=True)

    # 1. their shipped pure-python example, unmodified
    # Their README: "Set the project root directory as your working directory."
    # Python puts the SCRIPT's dir on sys.path (not cwd), so the documented
    # invocation needs the root on PYTHONPATH. Their code is not touched.
    env = dict(os.environ, PYTHONPATH=str(FAIRPCA_ROOT), MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, "illustration_Figure1/illustration_Figure1.py"],
                       cwd=str(FAIRPCA_ROOT), capture_output=True, text=True,
                       timeout=900, env=env)
    ok = r.returncode == 0 and "Reconstruction error" in r.stdout
    out["shipped_example"] = {"returncode": r.returncode, "ok": bool(ok),
                              "stdout": r.stdout[-1200:],
                              "stderr": r.stderr[-600:] if r.returncode else ""}
    print(f"  [shipped example] rc={r.returncode} ok={ok}")
    for ln in r.stdout.strip().splitlines():
        print(f"    | {ln}")

    # 2+3. seeded invariant checks through their public API
    from src.fair_pca.fair_PCA import (
        apply_fair_PCA_to_dataset, apply_fair_PCA_equalize_covariance_to_dataset,
    )
    rng = np.random.default_rng(0)
    n, d, td = 500, 10, 2
    d1 = rng.normal(size=(n, d))
    d2 = rng.normal(size=(n, d))
    trafo = np.eye(d)
    trafo[:3, :3] = np.array([[1, 0, 10], [1, 10, 10], [0, 10, 0]])
    d2 = d2 @ trafo
    data = np.vstack([d1, d2]).astype(np.float64)
    prot = np.ones(2 * n)
    prot[:n] = 0

    pipe = apply_fair_PCA_to_dataset((data, prot, prot), td, fit_classifier=False,
                                     standardize=False)
    low = pipe.just_transform(data)[:, :td]
    m0, m1 = low[:n].mean(0), low[n:].mean(0)
    scale = float(np.abs(low).mean()) + 1e-12
    mean_dev = float(np.max(np.abs(m0 - m1)) / scale)
    out["invariant_mean_equalization"] = {"rel_max_mean_gap": mean_dev,
                                          "tol": 1e-6, "passed": mean_dev <= 1e-6}
    print(f"  [invariant] projected group-mean gap (relative) = {mean_dev:.3e} "
          f"{'PASS' if mean_dev <= 1e-6 else 'FAIL'}", flush=True)
    cov_sec2 = float(np.linalg.norm(np.cov(low[:n].T) - np.cov(low[n:].T)))

    pipe2 = apply_fair_PCA_equalize_covariance_to_dataset(
        (data, prot, prot), td, fit_classifier=False, standardize=False,
        nr_eigenvecs_cov_constraint=int(0.5 * d))
    low2 = pipe2.just_transform(data)[:, :td]
    cov_sec36 = float(np.linalg.norm(np.cov(low2[:n].T) - np.cov(low2[n:].T)))
    out["sec36_improves_cov"] = {"sec2_cov_dev": cov_sec2, "sec36_cov_dev": cov_sec36,
                                 "passed": bool(cov_sec36 < cov_sec2)}
    print(f"  [invariant] cov deviation sec2={cov_sec2:.4f} sec3.6={cov_sec36:.4f} "
          f"{'PASS' if cov_sec36 < cov_sec2 else 'FAIL'}", flush=True)

    out["passed"] = bool(ok and out["invariant_mean_equalization"]["passed"]
                         and out["sec36_improves_cov"]["passed"])
    print(f"  >>> GATE {'PASS' if out['passed'] else 'FAIL'}", flush=True)
    return out


# --------------------------------------------------------------------------- #
#  Applicability                                                               #
# --------------------------------------------------------------------------- #
def applicability(n_attr_by_cell):
    """Their code REQUIRES prot in {0,1}. Verify by calling it, do not assume."""
    from src.fair_pca.fair_PCA import apply_fair_PCA_to_dataset
    out = {}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8))
    y = (rng.random(300) > 0.5).astype(int)
    multi = rng.integers(0, 5, size=300)
    try:
        apply_fair_PCA_to_dataset((X, y, multi), 2, fit_classifier=False)
        out["multiclass_s_accepted"] = True
        out["error"] = None
    except Exception as e:
        out["multiclass_s_accepted"] = False
        out["error"] = f"{type(e).__name__}: {e}"
    out["cells_in_scope"] = [k for k, v in n_attr_by_cell.items() if v == 2]
    out["cells_out_of_scope"] = {k: v for k, v in n_attr_by_cell.items() if v != 2}
    print(f"\n[applicability] multiclass s accepted: {out['multiclass_s_accepted']} "
          f"({out['error']})")
    print(f"[applicability] in scope: {out['cells_in_scope']}; out of scope "
          f"(not binarized): {out['cells_out_of_scope']}", flush=True)
    return out


# --------------------------------------------------------------------------- #
#  Exposures                                                                   #
# --------------------------------------------------------------------------- #
def _drop_dead_cols(Z):
    keep = Z.std(0) > 1e-12
    return Z[:, keep] if keep.any() else Z[:, :1]


def make_exposure(Xsrc, attr, task, target_dim, tradeoff, seed):
    """Fit their fair PCA on Xsrc, release the pipeline's transform output, and
    retrain an LR head on the release (LEACE convention: utility + logits)."""
    from sklearn.linear_model import LogisticRegression
    from src.fair_pca.fair_PCA import apply_fair_PCA_to_dataset
    Xd = np.asarray(Xsrc, dtype=np.float64)
    pipe = apply_fair_PCA_to_dataset((Xd, task.astype(int), attr.astype(int)),
                                     int(target_dim), fit_classifier=False,
                                     tradeoff_param=float(tradeoff),
                                     standardize=True)
    Z = _drop_dead_cols(np.asarray(pipe.just_transform(Xd), dtype=np.float64))
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    lr.fit(Z, task)
    L = np.log(np.clip(lr.predict_proba(Z), 1e-12, None)).astype(np.float32)
    return Z.astype(np.float32), L


def measure(Z, L, attr, task, n_attr, task_maj, device, seeds, with_lora=True):
    d, dl = Z.shape[1], L.shape[1]
    rp = full_point(Z, Z, attr, n_attr, np.zeros((d, d)), device, seeds,
                    with_lora=with_lora)
    op = full_point(L, L, attr, n_attr, np.zeros((dl, dl)), device, seeds,
                    with_lora=with_lora)
    return rp, op, lift(L, task, task_maj), utility_through(
        Z, task, lambda X, rng: X, seeds, task_maj)


# --------------------------------------------------------------------------- #
def run_cell(key, device, clean_lift):
    ds, at, tk = CELLS[key]
    X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk)
    print("\n" + "#" * 92 + f"\n# {key} {ds}/{at}/{tk}  n_attr={n_attr} "
          f"d_raw={X.shape[1]} clean_lift={clean_lift:+.4f}\n" + "#" * 92, flush=True)
    X_t = torch.from_numpy(X).to(device)

    # clean trained P (post-hoc surface base), one per training seed
    P_by_seed = {}
    for ts in TRAIN_SEEDS:
        P, _, _ = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                              alpha=0.0, device=device, pub_noise=0.0, seed=ts)
        P_by_seed[ts] = P
    surfaces = {"rawX": lambda ts: X, "posthocP": lambda ts: P_by_seed[ts]}

    rows = []
    for sname, getter in surfaces.items():
        base = getter(0)
        dims = [t for t in TARGET_DIMS if t < base.shape[1]]
        print(f"\n  --- surface {sname} (d={base.shape[1]}), target dims {dims} ---",
              flush=True)
        for td in dims:
            for tr in TRADEOFFS:
                t0 = time.time()
                Z, L = make_exposure(base, attr, task, td, tr, 0)
                rp, op, l_own, l_lr = measure(Z, L, attr, task, n_attr, task_maj,
                                              device, SWEEP_SEEDS, with_lora=False)
                row = {"surface": sname, "target_dim": td, "tradeoff": tr,
                       "z_dim": int(Z.shape[1]), "sweep_rep": rp, "sweep_out": op,
                       "lift_own": l_own, "lift_lr": l_lr,
                       "lift_best": float(max(l_own, l_lr)),
                       "kept_pct": 100.0 * float(max(l_own, l_lr)) / clean_lift,
                       "seconds": round(time.time() - t0, 1)}
                rows.append(row)
                print(f"    dim={td:<3} tradeoff={tr:<5g} z={Z.shape[1]:<3} "
                      f"rep {fmt_point(rp)} | out T2={op['tier2_max']:.3f} | "
                      f"kept={row['kept_pct']:.0f}%", flush=True)

    # certify: any sweep point at or under the bar gets the 3-seed full battery;
    # if none, certify the single least-leaky point so the row is never empty.
    cands = [r for r in rows if r["sweep_rep"]["tier1_max"] <= BAR]
    chosen = (sorted(cands, key=lambda r: -r["lift_best"])[:2] if cands
              else [min(rows, key=lambda r: r["sweep_rep"]["tier1_max"])])
    certs = []
    for c in chosen:
        print(f"\n  [certify] {c['surface']} dim={c['target_dim']} "
              f"tradeoff={c['tradeoff']:g} over {len(TRAIN_SEEDS)} training seeds",
              flush=True)
        per = {k: [] for k in ("rep_xgb", "rep_mlp", "rep_lora", "rep_lrt",
                               "out_xgb", "out_mlp", "out_lora", "out_lrt",
                               "lift_own", "lift_lr")}
        for ts in TRAIN_SEEDS:
            base = surfaces[c["surface"]](ts)
            Z, L = make_exposure(base, attr, task, c["target_dim"], c["tradeoff"], ts)
            rp, op, l_own, l_lr = measure(Z, L, attr, task, n_attr, task_maj,
                                          device, CERT_SEEDS)
            for k in ("xgb", "mlp", "lora", "lrt"):
                per[f"rep_{k}"].append(rp[k]); per[f"out_{k}"].append(op[k])
            per["lift_own"].append(l_own); per["lift_lr"].append(l_lr)
            print(f"      [seed {ts}] rep {fmt_point(rp)} | out T2={op['tier2_max']:.3f}",
                  flush=True)
        m = {k: float(np.mean(v)) for k, v in per.items()}
        s = {k: float(np.std(v)) for k, v in per.items()}
        cert = {"surface": c["surface"], "target_dim": c["target_dim"],
                "tradeoff": c["tradeoff"], "means": m, "stds": s,
                "per_seed": {k: [float(x) for x in v] for k, v in per.items()},
                "rep_tier1_max": max(m["rep_xgb"], m["rep_mlp"], m["rep_lora"]),
                "out_tier1_max": max(m["out_xgb"], m["out_mlp"], m["out_lora"]),
                "lift_best": float(max(m["lift_own"], m["lift_lr"]))}
        cert["rep_tier2_max"] = max(cert["rep_tier1_max"], m["rep_lrt"])
        cert["out_tier2_max"] = max(cert["out_tier1_max"], m["out_lrt"])
        cert["kept_pct"] = 100.0 * cert["lift_best"] / clean_lift
        cert["tier1_pass"] = bool(cert["rep_tier1_max"] <= BAR
                                  and cert["out_tier1_max"] <= BAR)
        cert["tier2_pass"] = bool(cert["rep_tier2_max"] <= BAR
                                  and cert["out_tier2_max"] <= BAR)
        certs.append(cert)
        print(f"    >>> rep T1={cert['rep_tier1_max']:.3f} T2={cert['rep_tier2_max']:.3f} "
              f"| out T1={cert['out_tier1_max']:.3f} T2={cert['out_tier2_max']:.3f} "
              f"| kept={cert['kept_pct']:.0f}% | T1 {'PASS' if cert['tier1_pass'] else 'FAIL'}",
              flush=True)

    best_t1 = min(r["sweep_rep"]["tier1_max"] for r in rows)
    return {"cell": key, "dataset": ds, "attr": at, "task": tk, "n_attr": n_attr,
            "clean_lift": clean_lift, "sweep": rows, "certified": certs,
            "min_sweep_rep_tier1": float(best_t1),
            "max_sweep_rep_tier1": float(max(r["sweep_rep"]["tier1_max"] for r in rows))}


def score(cells):
    if not cells:
        return {"P1": "NOT RUN", "P1b": "NOT RUN"}
    p1 = {}
    p1b = {}
    for c in cells:
        readings = [r["sweep_rep"]["tier1_max"] for r in c["sweep"]]
        p1[c["cell"]] = {"min_tier1": min(readings), "max_tier1": max(readings),
                         "ge_090": bool(min(readings) >= 0.90),
                         "any_pass_bar": bool(min(readings) <= BAR)}
        by_dim = {}
        for r in c["sweep"]:
            by_dim.setdefault(r["target_dim"], []).append(r["sweep_rep"]["tier1_max"])
        p1b[c["cell"]] = {"min_per_dim": {str(k): min(v) for k, v in sorted(by_dim.items())},
                          "all_dims_ge_080": bool(all(min(v) >= 0.80
                                                      for v in by_dim.values()))}
    return {
        "P1_fails_tier1_near_unprotected": {
            "per_cell": p1,
            "verdict": "HELD" if all(v["ge_090"] for v in p1.values()) else "FALSIFIED"},
        "P1b_no_monotone_rescue_ge_080": {
            "per_cell": p1b,
            "verdict": "HELD" if all(v["all_dims_ge_080"] for v in p1b.values())
                       else "FALSIFIED"},
    }


def main():
    t0 = time.time()
    device = get_device()
    tt = json.loads((RESULTS / "two_tier_certification.json").read_text())
    ttname = {"easy": "hmda/race/loan_decision",
              "middle": "hmda/race/loan_amount_band", "hard": "adult/sex/income"}
    clean = {k: next(c for c in tt["e2e_cells"] if c["cell"] == ttname[k])["clean_lift"]
             for k in CELLS}
    print(f"[setup] device={device} bar={BAR} train_seeds={TRAIN_SEEDS} "
          f"sweep_seeds={SWEEP_SEEDS} cert_seeds={CERT_SEEDS}", flush=True)

    g = gate()
    n_attr_by_cell = {}
    for k, (ds, at, tk) in CELLS.items():
        _, _, _, n_attr, _, _, _ = load_cell(ds, at, tk)
        n_attr_by_cell[k] = n_attr
    app = applicability(n_attr_by_cell)

    cells = []
    if not g["passed"]:
        print("\n*** GATE FAILED — stopping, no cell run, nothing borrowed.",
              flush=True)
    else:
        for k in app["cells_in_scope"]:
            cells.append(run_cell(k, device, clean[k]))

    scoring = score(cells)
    print("\n" + "=" * 92 + "\nSCORING (docs/fairpca_prediction.md @ 8b7d873)\n" + "=" * 92)
    for k, v in scoring.items():
        print(f"  {k}: {v if isinstance(v, str) else v['verdict']}")
        if not isinstance(v, str):
            print(f"    {v['per_cell']}")

    out = {"experiment": "fairpca_gauntlet",
           "prediction_file": "fairpca_prediction.md",
           "prediction_commit": "8b7d873", "fairpca_commit": FAIRPCA_COMMIT,
           "license": g["license"], "bar": BAR, "train_seeds": TRAIN_SEEDS,
           "sweep_seeds": SWEEP_SEEDS, "cert_seeds": CERT_SEEDS,
           "target_dims": TARGET_DIMS, "tradeoffs": TRADEOFFS,
           "clean_lifts": clean, "gate": g, "applicability": app,
           "cells": cells, "scoring": scoring,
           "minutes": round((time.time() - t0) / 60, 1)}
    (RESULTS / "fairpca_gauntlet.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {RESULTS / 'fairpca_gauntlet.json'} ({out['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
