"""TASK 4 — recompute the Prop-2 witness under the appendix-faithful construction.

THE MISMATCH (file/line, verified):
  experiments/mi_ceiling.py:158-163
      def worst_case_auc(s2, sigma):
          return float(norm.cdf(np.sqrt(2.0 * np.sum(s2)) / sigma))
  experiments/mi_ceiling.py:269
      s2 = h.var(0).astype(np.float64)
  => the stored ceiling is a two-point +-mu construction whose separation is
     set by the MARGINAL variance of h summed over dimensions. Attribute
     labels never enter it. Docstring at mi_ceiling.py:41-46 states the
     construction as "h = +-mu with ||mu||^2 = sum_i s_i^2".
  The supplement instead describes a CLASS-CONDITIONAL construction:
     h | y = mu_y + eps*sqrt(lambda_y) v_y + W_y
  with (lambda_y, v_y) the top eigenpair of Sigma_y and W_y carrying the
  remaining spectrum. These are different witnesses against different
  certificate families. This script rebuilds the second one.

RECOMPUTE: per operating point, retrain the channel, take the clean pre-noise
h, form per-class (mu_y, Sigma_y), build the two-sheet witness that matches
BOTH moments exactly, add the channel noise, and score the exact Bayes-optimal
attacker (each class is a 2-component Gaussian mixture, so the class
log-likelihood is available in closed form). Macro-OVR AUC for multiclass s,
matching the battery convention.

Also reports the corrected side condition: recovery is governed by the minimum
distance between OPPOSITE-CLASS sheet centers relative to sigma, not by
tr(Sigma_y).
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
from scipy.linalg import solve_triangular  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from experiments.diagnostic import load_cell  # noqa: E402
from experiments.mi_ceiling import train_channel  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

N_SAMPLE = 60000
SEED = 20260728
BAR = 0.55

# the six operating points exactly as stored in
# results/mi_ceiling.json["operating_point_extension"]["points"].
# `channel` is the channel whose numbers the stored value actually used —
# note the two "middle" rows were computed from the loan_decision channel.
POINTS = [
    dict(label="easy (HMDA/decision)", tier="T1", sigma=8.0,
         channel=("hmda", "race", "loan_decision"), stored=0.993192991411881),
    dict(label="middle (HMDA/amount)", tier="T1", sigma=12.0,
         channel=("hmda", "race", "loan_decision"), stored=0.9598947468180754),
    dict(label="easy (HMDA/decision)", tier="T2", sigma=20.0,
         channel=("hmda", "race", "loan_decision"), stored=0.8493127655711068),
    dict(label="middle (HMDA/amount)", tier="T2", sigma=24.0,
         channel=("hmda", "race", "loan_decision"), stored=0.8035629014523376),
    dict(label="hard (Adult/income)", tier="T1", sigma=32.0,
         channel=("adult", "sex", "income"), stored=0.7172147626471087),
    dict(label="hard (Adult/income)", tier="T2", sigma=64.0,
         channel=("adult", "sex", "income"), stored=0.5925164803062863),
]


def class_stats(h, attr, n_attr):
    out = []
    for k in range(n_attr):
        Hk = h[attr == k].astype(np.float64)
        out.append(dict(k=k, n=len(Hk), mu=Hk.mean(0), Sigma=np.cov(Hk.T),
                        prior=len(Hk) / len(h)))
    return out


def witness_components(st):
    """Two sheet centers and the shared within-sheet covariance for class k.

    h|y = mu_y +- sqrt(lambda_y) v_y + W_y,  Cov(W_y) = Sigma_y - lambda_y v v^T
    Mean is mu_y; covariance is lambda_y v v^T + (Sigma_y - lambda_y v v^T)
    = Sigma_y exactly.
    """
    Sig = st["Sigma"]
    w, V = np.linalg.eigh(Sig)
    lam, v = float(w[-1]), V[:, -1]
    resid = Sig - lam * np.outer(v, v)
    # numerical PSD repair of the zeroed top eigendirection only
    ww, VV = np.linalg.eigh(resid)
    ww = np.clip(ww, 0.0, None)
    resid = (VV * ww) @ VV.T
    centers = [st["mu"] + np.sqrt(lam) * v, st["mu"] - np.sqrt(lam) * v]
    return centers, resid, lam, v


def logpdf_mix(X, centers, cov, rng_ridge=1e-8):
    """log of the 0.5/0.5 two-component Gaussian mixture, stable via Cholesky."""
    d = X.shape[1]
    C = cov + rng_ridge * np.eye(d)
    L = np.linalg.cholesky(C)
    logdet = 2.0 * np.log(np.diag(L)).sum()
    lps = []
    for m in centers:
        z = solve_triangular(L, (X - m).T, lower=True)
        lps.append(-0.5 * ((z ** 2).sum(0) + logdet + d * np.log(2 * np.pi)))
    a = np.stack(lps, 1)
    mx = a.max(1, keepdims=True)
    return (mx[:, 0] + np.log(np.exp(a - mx).sum(1))) - np.log(2.0)


def bayes_auc(stats, sigma, n_attr, rng):
    """Exact Bayes-optimal macro-OVR AUC on the witness, after channel noise."""
    comps = []
    for st in stats:
        centers, resid, lam, v = witness_components(st)
        comps.append(dict(centers=centers, cov=resid + sigma ** 2 * np.eye(len(st["mu"])),
                          prior=st["prior"], lam=lam, v=v, mu=st["mu"]))
    # sample the witness (prior-weighted), then score every class log-density
    Xs, ys = [], []
    for k, c in enumerate(comps):
        nk = max(int(round(N_SAMPLE * c["prior"])), 50)
        pick = rng.integers(0, 2, size=nk)
        M = np.stack([c["centers"][p] for p in pick])
        Ls = np.linalg.cholesky(c["cov"] + 1e-8 * np.eye(M.shape[1]))
        Xs.append(M + rng.standard_normal((nk, M.shape[1])) @ Ls.T)
        ys.append(np.full(nk, k))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    logp = np.stack([logpdf_mix(X, c["centers"], c["cov"]) + np.log(c["prior"])
                     for c in comps], 1)
    post = logp - logp.max(1, keepdims=True)
    post = np.exp(post); post /= post.sum(1, keepdims=True)
    if n_attr == 2:
        auc = float(roc_auc_score(y, post[:, 1]))
    else:
        auc = float(roc_auc_score(y, post, multi_class="ovr", average="macro"))
    # corrected side condition: min opposite-class sheet-center distance / sigma
    dmin = np.inf
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            for a in comps[i]["centers"]:
                for b in comps[j]["centers"]:
                    dmin = min(dmin, float(np.linalg.norm(a - b)))
    return auc, dmin


def main():
    t0 = time.time()
    device = get_device()
    rng = np.random.default_rng(SEED)
    print(f"[setup] device={device} n_sample={N_SAMPLE} seed={SEED}\n", flush=True)

    cache, rows = {}, []
    for p in POINTS:
        ds, at, tk = p["channel"]
        key = (ds, at, tk, p["sigma"])
        if key not in cache:
            X, attr, task, n_attr, n_task, _, _ = load_cell(ds, at, tk)
            X_t = torch.from_numpy(X).to(device)
            h, P, L = train_channel(X_t, attr, task, n_attr, n_task, p["sigma"],
                                    device, seed=0)
            cache[key] = (h, attr, n_attr)
        h, attr, n_attr = cache[key]
        stats = class_stats(h, attr, n_attr)
        total_var = float(h.var(0).astype(np.float64).sum())
        old = float(norm.cdf(np.sqrt(2.0 * total_var) / p["sigma"]))
        new, dmin = bayes_auc(stats, p["sigma"], n_attr, rng)
        row = dict(
            label=p["label"], tier=p["tier"], sigma=p["sigma"],
            channel=f"{ds}/{at}/{tk}", n_attr=n_attr,
            stored_ceiling=p["stored"], recomputed_marginal_ceiling=old,
            appendix_faithful_auc=new,
            total_var=total_var,
            tr_Sigma_y_mean=float(np.mean([np.trace(s["Sigma"]) for s in stats])),
            min_opposite_sheet_distance=dmin,
            dmin_over_sigma=dmin / p["sigma"],
            old_exceeds_bar=bool(p["stored"] > BAR),
            new_exceeds_bar=bool(new > BAR))
        rows.append(row)
        print(f"  {p['label']:<22} {p['tier']} s={p['sigma']:<5g} "
              f"stored={p['stored']:.4f} marginal-recompute={old:.4f} "
              f"appendix-faithful={new:.4f} "
              f"{'>0.55' if new > BAR else '<=0.55'}  "
              f"dmin/sigma={dmin / p['sigma']:.3f}", flush=True)

    n_old = sum(r["old_exceeds_bar"] for r in rows)
    n_new = sum(r["new_exceeds_bar"] for r in rows)
    print(f"\n  exceeds 0.55: stored {n_old}/6 -> appendix-faithful {n_new}/6")

    out = dict(
        experiment="prop2_witness_recompute",
        mismatch=dict(
            stored_formula="Phi(sqrt(2*sum_i Var(h_i))/sigma)",
            stored_code="experiments/mi_ceiling.py:158-163 (worst_case_auc)",
            stored_input="experiments/mi_ceiling.py:269 (s2 = h.var(0)) - marginal, "
                         "attribute labels never used",
            stored_docstring="experiments/mi_ceiling.py:41-46",
            supplement_construction="h|y = mu_y + eps*sqrt(lambda_y) v_y + W_y, "
                                    "(lambda_y, v_y) top eigenpair of Sigma_y",
            consequence="different witnesses against different certificate "
                        "families; the marginal one does not preserve Sigma_y",
            second_mismatch="the two rows labelled 'middle (HMDA/amount)' were "
                            "computed from the hmda/race/loan_decision channel "
                            "(mi_ceiling.json channels contain no "
                            "loan_amount_band channel)"),
        corrected_side_condition=(
            "recovery approaches the noise-free maximum when the MINIMUM "
            "opposite-class sheet-center distance is large relative to sigma, "
            "i.e. min_{y!=y', s,s' in {+-1}} ||mu_y + s*sqrt(lam_y) v_y - mu_y' "
            "- s'*sqrt(lam_y') v_y'|| >> sigma. tr(Sigma_y) >> sigma^2 does NOT "
            "imply this: aligned top eigenvectors with small mean separation "
            "give large trace and chance-level recovery."),
        n_sample=N_SAMPLE, seed=SEED, bar=BAR, rows=rows,
        n_exceed_bar_stored=n_old, n_exceed_bar_recomputed=n_new,
        minutes=round((time.time() - t0) / 60, 1))
    (HERE / "prop2_witness_recompute.json").write_text(
        json.dumps(out, indent=1, default=float))
    print(f"\n[save] {HERE / 'prop2_witness_recompute.json'} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
