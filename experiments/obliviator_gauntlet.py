"""Experiment 17 — OBLIVIATOR (NeurIPS 2025) through the two-tier gauntlet.

Obliviator (Akbari et al., "Obliviator Reveals the Cost of Nonlinear
Guardedness in Concept Erasure", NeurIPS 2025) is the field's strongest
HSIC-family post-hoc eraser: iterative RFF-kernelized HSIC minimization — a
learned encoder morphs the feature space against HSIC(w,s) while HSIC-
preserving x (and, in supervised mode, y), followed each iteration by a
kernel-PCA step (EVP) in the null space of Cov(s,·), repeated until THEIR
MLP probe reads the unwanted attribute at <= 0.6 accuracy (or 15 iterations).
It is exactly the "HSIC done seriously" upgrade of the controlled HSIC
projections Experiment 4 tested — which makes it the decisive baseline for
the Exp-4/5 finding that probe-shaped erasure only fools its own probe.

PROVENANCE: OFFICIAL CODE. https://github.com/ramin-akbari/Obliviator at
commit 0f2233f8cdc30981a994af64321bc86917723f97 (2026-02-25), imported
in-place (the repo ships no LICENSE, so it is not vendored here; set
OBLIVIATOR_ROOT to the clone). We import their obliviator.Supervised /
SupervisedData, their configs.defaults.BaseSup eraser configuration, their
evaluation.probing.MLPCrossEntropy probes with their DeepClassifier /
ClassifierOptim defaults, and replicate the main.py experiment loop verbatim
(constants ENCODER_EPOCH=5, MAX_EPOCHS=200, MAX_ITER=15, TARGET_ACC=0.6,
tolerances tol_dim=1e-4, tol_evp=1e-5). Their method code is untouched.

Faithful-minimal adaptation, favorable to the baseline, data boundary only:
  - x = the project-standard clean trained representation P (the same input
    LEACE got — Obliviator is a post-hoc eraser of learned representations;
    its paper erases foundation-model embeddings, P is the tabular analogue).
  - SUPERVISED mode (their strongest: the eraser sees the task label and
    preserves HSIC(w,y) — maximum utility retention).
  - The eraser fits on ALL rows (every gauntlet baseline trains on the full
    cell); their stopping-rule probes get a seed-0 75/25 split as in their
    experiment pipeline.
  - s and y one-hot, their convention.

Protocol (mirrors experiments/baseline_gauntlet.py): the knob is Obliviator's
own axis — the erasure iteration. Sweep at train seed 0: our battery
(XGB+MLP+LRT; LoRA on Tier-1 candidates) measures the exposed representation
z_t at iterations {0 (post dim-reduction), 1, 2, 3, 5, 7, 10, 15} and at
their stopping point; the output channel is a retrained-LR head's logits on
z_t (the LEACE pattern — the method ships no task head). Tier candidates
(<= 0.55) are certified best-utility-first over 3 P-training seeds (fresh
clean P + fresh full Obliviator run per seed, taken at the same iteration
index, or at that seed's own earlier stop if their rule fires first — the
method's own deployment output). Sigma_noise = 0 for the Tier-2 LRT: the
exposure is a deterministic transform of P, so the informed tier gets the
exposed representation itself as side information (same convention as LEACE /
LAFTR / DANN-scrub rows).

Utility normalized by the cell's clean e2e lift (the master-table
denominator) at report time; this script stores raw lifts.

Local MPS/CPU; never AWS. macOS OpenMP guard: xgboost before torch.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401  (import before torch on purpose)

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# clone https://github.com/ramin-akbari/Obliviator (commit below) as a sibling
# of this repo, or point OBLIVIATOR_ROOT at it
OBLIV_ROOT = Path(os.environ.get("OBLIVIATOR_ROOT",
                                 REPO_ROOT.parent / "Obliviator"))
sys.path.insert(0, str(OBLIV_ROOT / "src"))
sys.path.insert(0, str(OBLIV_ROOT))

from configs.defaults import (  # noqa: E402  (theirs)
    BaseSup, ClassifierOptim, DeepClassifier,
)
from evaluation.probing import (  # noqa: E402  (theirs)
    MLPCrossEntropy, ProbConfig, ProbData,
)
from obliviator.schemas import SupervisedData  # noqa: E402  (theirs)
from obliviator.supervised import Supervised  # noqa: E402  (theirs)

from experiments.baseline_gauntlet import (  # noqa: E402
    BAR, CERT_SEEDS, MAX_CERT_CANDIDATES, SWEEP_SEEDS, TRAIN_SEEDS,
    _exposure, measure_sweep,
)
from experiments.diagnostic import label_predictor, lift, load_cell  # noqa: E402
from experiments.sealed_channel import train_model  # noqa: E402
from experiments.targeted_noise import utility_through  # noqa: E402
from experiments.two_tier_certification import fmt_point, full_point  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
PARTIAL = RESULTS_DIR / "obliviator_gauntlet_partial.json"

OFFICIAL_COMMIT = "0f2233f8cdc30981a994af64321bc86917723f97"
# their main.py constants, replicated verbatim
ENCODER_EPOCH = 5
MAX_EPOCHS = 200
MAX_ITER = 15
TARGET_ACC = 0.6
TOL_DIM = 1e-4
TOL_EVP = 1e-5
PROBE_ORIG_EPOCHS = 25          # main.py: update_accuracy(25) on original x

MEASURE_ITERS = {0, 1, 2, 3, 5, 7, 10, 15}

CELLS = [
    dict(name="hmda/race/loan_decision", dataset="hmda", attr="race",
         task="loan_decision"),
    dict(name="hmda/race/loan_amount_band", dataset="hmda", attr="race",
         task="loan_amount_band"),
    dict(name="adult/sex/income", dataset="adult", attr="sex", task="income"),
]


def run_obliviator(H, attr, task, n_attr, n_task, seed, probe_device,
                   want_iters=None):
    """Their main.py loop, verbatim, on clean rep H. Returns the list of
    exposures [(iter_idx, z (np), their unwanted/utility max_acc)] plus the
    stop iteration. want_iters=None keeps every iteration's z; otherwise only
    those (plus the stopping point)."""
    torch.manual_seed(seed)
    n = H.shape[0]
    x = torch.from_numpy(H.astype(np.float32)).clone()
    soh = F.one_hot(torch.from_numpy(attr).long(), n_attr).float()
    yoh = F.one_hot(torch.from_numpy(task).long(), n_task).float()
    data = SupervisedData(x=x, s=soh, x_test=x[:512].clone(), y=yoh)
    eraser = Supervised(data, BaseSup())

    rng = np.random.RandomState(0)                 # their probes' split
    perm = rng.permutation(n)
    te, tr = perm[: n // 4], perm[n // 4:]

    def mk_probe(labels, name):
        pd = ProbData(x=H[tr].astype(np.float32), y=labels[tr],
                      x_test=H[te].astype(np.float32), y_test=labels[te])
        return MLPCrossEntropy(pd, ProbConfig(
            device=probe_device, mlp_config=DeepClassifier(),
            optim_config=ClassifierOptim(), name=name))

    unw, utl = mk_probe(attr, "Unwanted"), mk_probe(task, "Utility")

    def update_cls(z):
        zt = z if torch.is_tensor(z) else torch.from_numpy(z)
        unw.update_input(x=zt[tr], x_test=zt[te])
        utl.update_input(x=zt[tr], x_test=zt[te])

    def update_accuracy(epochs):
        unw.train(epochs)
        utl.train(epochs)
        return unw.max_acc

    update_accuracy(PROBE_ORIG_EPOCHS)             # original accuracy
    orig = dict(unwanted=unw.max_acc, utility=utl.max_acc)

    exposures = []

    def record(it, z):
        if want_iters is None or it in want_iters:
            exposures.append(dict(iter=it, z=np.asarray(z, dtype=np.float32),
                                  their_unwanted=float(unw.max_acc),
                                  their_utility=float(utl.max_acc)))

    z, _ = eraser.null_dim_reduction(TOL_DIM)
    update_cls(z)
    update_accuracy(MAX_EPOCHS)
    record(0, z.cpu().numpy())

    z, _ = eraser.init_erasure(epochs=ENCODER_EPOCH, tol=TOL_EVP)
    update_cls(z)
    unwanted_acc = update_accuracy(MAX_EPOCHS)
    record(1, z.cpu().numpy())
    it = 1
    while unwanted_acc > TARGET_ACC and it < MAX_ITER:
        it += 1
        z, _ = eraser.erasure_step(z=z, epochs=ENCODER_EPOCH, tol=TOL_EVP,
                                   update_x=it == 2)
        update_cls(z)
        unwanted_acc = update_accuracy(MAX_EPOCHS)
        record(it, z.cpu().numpy())
    if not exposures or exposures[-1]["iter"] != it:   # always keep the stop
        exposures.append(dict(iter=it, z=np.asarray(z.cpu().numpy(),
                                                    dtype=np.float32),
                              their_unwanted=float(unw.max_acc),
                              their_utility=float(utl.max_acc)))
    return exposures, it, orig


def lr_logits(z, task, seed=0):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                            random_state=seed)
    lr.fit(z.astype(np.float64), task)
    return np.log(np.clip(lr.predict_proba(z.astype(np.float64)),
                          1e-12, None)).astype(np.float32)


def make_exposure(e, task, name_prefix="iter"):
    z = e["z"]
    L = lr_logits(z, task)
    exp = _exposure(f"{name_prefix} {e['iter']}", float(e["iter"]), z, L)
    exp["their_unwanted"], exp["their_utility"] = (e["their_unwanted"],
                                                   e["their_utility"])
    return exp


def certify_obliv(iter_idx, ctx, device, probe_device):
    """3 P-training-seed certification at erasure iteration iter_idx (or a
    seed's own earlier stopping point): full battery on rep AND LR-head
    output logits."""
    attr, task = ctx["attr"], ctx["task"]
    n_attr, n_task, task_maj = ctx["n_attr"], ctx["n_task"], ctx["task_maj"]
    keys = ("xgb", "mlp", "lora", "lrt")
    per = {f"rep_{k}": [] for k in keys}
    per.update({f"out_{k}": [] for k in keys})
    per["lift_own"], per["lift_lr"] = [], []
    per["actual_iter"], per["their_unwanted"] = [], []
    for ts in TRAIN_SEEDS:
        H, _, _ = train_model(ctx["X_t"], attr, task, n_attr, n_task,
                              use_seal=False, alpha=0.0, device=device,
                              pub_noise=0.0, seed=ts)
        exps, stop_it, _ = run_obliviator(
            H, attr, task, n_attr, n_task, ts, probe_device,
            want_iters={int(iter_idx)})
        e = next((x for x in exps if x["iter"] == int(iter_idx)),
                 exps[-1])   # seed stopped earlier -> its own final output
        z, L = e["z"], lr_logits(e["z"], task)
        d = z.shape[1]
        rp = full_point(z, z, attr, n_attr, np.zeros((d, d)), device,
                        CERT_SEEDS)
        d_out = L.shape[1]
        op = full_point(L, L, attr, n_attr, np.zeros((d_out, d_out)), device,
                        CERT_SEEDS)
        for k in keys:
            per[f"rep_{k}"].append(rp[k])
            per[f"out_{k}"].append(op[k])
        per["lift_own"].append(lift(L, task, task_maj))
        per["lift_lr"].append(utility_through(z, task, lambda X, rng: X,
                                              CERT_SEEDS, task_maj))
        per["actual_iter"].append(e["iter"])
        per["their_unwanted"].append(e["their_unwanted"])
        print(f"      [seed {ts}] iter={e['iter']} rep {fmt_point(rp)} | "
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
    c["label"], c["knob"] = f"iter {int(iter_idx)}", float(iter_idx)
    return c


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()
                if k not in ("P", "L", "h_clean", "ncov", "z")}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


def main():
    device = get_device()
    probe_device = str(device)
    from experiments.baseline_gauntlet import load_ours
    ours = load_ours()
    partial = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    print(f"[setup] official Obliviator commit {OFFICIAL_COMMIT[:12]} at "
          f"{OBLIV_ROOT}\n[setup] device={device} probes on {probe_device} "
          f"bar<={BAR} their constants: enc_epochs={ENCODER_EPOCH} "
          f"probe_epochs={MAX_EPOCHS} max_iter={MAX_ITER} "
          f"target={TARGET_ACC}", flush=True)
    results = []
    for cell in CELLS:
        t0 = time.time()
        key = f"{cell['name']}|Obliviator"
        print("\n" + "#" * 100 + f"\n# CELL: {cell['name']}\n" + "#" * 100,
              flush=True)
        if key in partial:
            print(f"  [resume] {key} from checkpoint", flush=True)
            results.append(partial[key])
            continue
        X, attr, task, n_attr, n_task, attr_maj, task_maj = load_cell(
            cell["dataset"], cell["attr"], cell["task"])
        pred, _ = label_predictor(attr, task, n_attr)
        ctx = dict(name=cell["name"], X_t=torch.from_numpy(X).to(device),
                   attr=attr, task=task, n_attr=n_attr, n_task=n_task,
                   task_maj=task_maj)
        print(f"[setup] N={X.shape[0]} attr={n_attr}cls task={n_task}cls "
              f"predictor={pred:.4f}", flush=True)

        # ---- sweep at train seed 0: their full protocol, our battery ----
        H, _, _ = train_model(ctx["X_t"], attr, task, n_attr, n_task,
                              use_seal=False, alpha=0.0, device=device,
                              pub_noise=0.0, seed=0)
        exps, stop_it, orig = run_obliviator(H, attr, task, n_attr, n_task, 0,
                                             probe_device,
                                             want_iters=MEASURE_ITERS)
        print(f"  [obliviator] their stop at iteration {stop_it}; their "
              f"probes on original P: {orig}", flush=True)
        rows = []
        for e in exps:
            exp = make_exposure(e, task)
            row = measure_sweep(exp, ctx, device)
            row["their_unwanted"] = e["their_unwanted"]
            row["their_utility"] = e["their_utility"]
            row["z_dim"] = int(e["z"].shape[1])
            rows.append(row)

        # ---- tier pick + certification (baseline_gauntlet tier logic) ----
        cert_cache = {}

        def tier_pick(tkey):
            cands = sorted([r for r in rows if r[f"{tkey}_max"] <= BAR],
                           key=lambda r: -r["lift_best"])
            tried = []
            for r in cands[:MAX_CERT_CANDIDATES]:
                if r["label"] not in cert_cache:
                    print(f"    [certify Obliviator {r['label']} over "
                          f"{TRAIN_SEEDS}]", flush=True)
                    cert_cache[r["label"]] = certify_obliv(
                        r["knob"], ctx, device, probe_device)
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
                print(f"  >>> Obliviator {nm}: CERTIFIES at {c['label']} — "
                      f"lift {c['lift_best']:+.4f}, rep T1="
                      f"{c['rep_tier1_max']:.3f} T2={c['rep_tier2_max']:.3f}",
                      flush=True)
            else:
                b = t["closest"]
                tv = b["tier1_max"] if nm == "TIER 1" else b["tier2_max"]
                print(f"  >>> Obliviator {nm}: does NOT certify — closest "
                      f"{b['label']} max={tv:.3f} at lift "
                      f"{b['lift_best']:+.4f}", flush=True)

        r = dict(cell=cell["name"], predictor=pred,
                 clean_lift=ours[cell["name"]]["clean_lift"],
                 baseline="Obliviator", their_stop_iter=stop_it,
                 their_orig_probes=orig, rows=rows,
                 tier1=_clean(t1), tier2=_clean(t2), certs=cert_cache,
                 minutes=round((time.time() - t0) / 60, 1))
        r = _clean(r)
        partial[key] = r
        PARTIAL.write_text(json.dumps(partial))
        results.append(r)

    out = {
        "experiment": "obliviator_gauntlet", "bar": BAR,
        "provenance": {
            "kind": "official code",
            "repo": "https://github.com/ramin-akbari/Obliviator",
            "commit": OFFICIAL_COMMIT,
            "mode": "supervised (their strongest: task label available to "
                    "the eraser)",
            "config": "configs.defaults.BaseSup + their main.py loop "
                      "constants, verbatim",
            "adaptation": "data boundary only: x = clean trained rep P (as "
                          "LEACE), s/y one-hot, eraser fit on all rows, "
                          "their stopping probes on a seed-0 75/25 split",
        },
        "exp4_characterization": {
            "claim": "Obliviator is the 'HSIC done seriously' upgrade of the "
                     "Exp-4 controlled HSIC projections (iterative, "
                     "RFF-kernelized, encoder-morphing, null-space EVP) — "
                     "the decisive test of whether the Exp-4/5 finding "
                     "(probe-shaped erasure only fools its own probe) was an "
                     "artifact of our simple implementation",
            "exp4_reference": "results/smart_erasure.json (HSIC proj r<=32: "
                              "attacked R^2 plateaus ~0.17 on HMDA, XGB "
                              "reads projections at unprotected ~0.68 AUC "
                              "on Adult — results/honest_reaudit.json)",
        },
        "cells": _clean(results),
    }
    (RESULTS_DIR / "obliviator_gauntlet.json").write_text(
        json.dumps(out, indent=2))
    PARTIAL.unlink(missing_ok=True)
    print(f"\n[save] {RESULTS_DIR / 'obliviator_gauntlet.json'}", flush=True)


if __name__ == "__main__":
    main()
