"""The standing attacker battery, in two certification tiers.

Experiment 11's side finding. A Gaussian-LRT attacker that knows the channel,
fitting class Gaussians on the CLEAN (pre-noise) representation and integrating
the known channel noise analytically, sits ABOVE the trained battery at
operating sigma by 0.02 to 0.05 depending on the cell. On the HMDA e2e cell at
sigma=8 it reads 0.574 where the battery certified 0.530.
Per the project's own thesis (a guarantee is only as strong as the strongest
attacker used to certify it), that attacker joins the battery here as a standard
member, and the battery splits into two explicit threat-model tiers:

  TIER 1 (black box)  XGBoost + deep MLP + rank-32 ReLU LoRA. The attacker sees
                      only the deployed (noised) representation and outputs.
                      This is the standing battery of Exp 5 through 10.
  TIER 2 (informed)   Tier 1 + the Gaussian-LRT. The attacker additionally has
                      clean-representation side information and knows the channel
                      (noise level sigma, and the destroyed subspace Q for
                      surgical channels). This models the vendor themselves, an
                      auditor with pipeline access, or an attacker who can average
                      repeated queries of the same row through the fresh-noise
                      channel.

The LRT is channel-aware through its noise covariance:
  * blunt isotropic channel:  Sigma_noise = sigma^2 I
  * subspace/surgical channel (noise confined to span(Q), Q d x r orthonormal):
    Sigma_noise = sigma^2 Q Q^T   (zero noise in the complement, so the LRT
    sees the surviving directions at full strength. Surgical channels have no
    distribution-free ceiling, Exp 11)
Per class k it fits N(mu_k, Sigma_k + Sigma_noise) on the clean train split and
scores the EXPOSED (noised) held-out rows by class posterior, giving the same
kind of held-out AUC as every other battery member. Never R^2.
"""

from __future__ import annotations

import numpy as np


def iso_noise_cov(d, sigma):
    """Known noise covariance of the blunt isotropic channel."""
    return (sigma ** 2) * np.eye(d)


def subspace_noise_cov(Q, sigma):
    """Known noise covariance of a subspace (surgical) channel: sigma^2 QQ^T."""
    Q = np.asarray(Q, dtype=np.float64)
    return (sigma ** 2) * (Q @ Q.T)


def _split_idx(y, seed):
    """The battery's held-out split (diagnostic._split), returned as indices so
    the clean and exposed matrices can be split identically."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, te, _, _ = train_test_split(idx, y, test_size=0.25, random_state=seed,
                                    stratify=y)
    return tr, te


def _lrt_scores(H_clean_tr, y_tr, X_te, noise_cov, n_classes):
    """Fit N(mu_k, Sigma_k + Sigma_noise) per class on clean train rows, then
    return class-posterior scores for the (noised) test rows."""
    from scipy.linalg import solve_triangular
    d = H_clean_tr.shape[1]
    ridge = 1e-4 * np.eye(d)
    logps = []
    for k in range(n_classes):
        Hk = H_clean_tr[y_tr == k]
        mu = Hk.mean(0)
        C = np.cov(Hk.T) + noise_cov + ridge
        Lc = np.linalg.cholesky(C)
        logdet = 2.0 * float(np.log(np.diag(Lc)).sum())
        z = solve_triangular(Lc, (X_te - mu).T, lower=True)
        logp = -0.5 * ((z ** 2).sum(0) + logdet + d * np.log(2 * np.pi))
        logps.append(logp + np.log(len(Hk) / len(H_clean_tr)))
    L = np.stack(logps, 1)
    L -= L.max(1, keepdims=True)
    return np.exp(L) / np.exp(L).sum(1, keepdims=True)


def _auc(y_te, post, n_classes):
    from sklearn.metrics import roc_auc_score
    if n_classes == 2:
        return float(roc_auc_score(y_te, post[:, 1]))
    return float(roc_auc_score(y_te, post, multi_class="ovr", average="macro"))


def gaussian_lrt(H_clean, P_exposed, y, noise_cov, n_classes, seeds):
    """The Tier-2 Gaussian-LRT battery member on a FIXED exposed matrix.

    H_clean is the pre-noise representation, the side information. P_exposed is
    the deployed noised representation, row-aligned with H_clean, the same matrix
    the Tier-1 members attack. Returns (mean, std) held-out AUC over seeds.
    """
    H_clean = np.asarray(H_clean, dtype=np.float64)
    P_exposed = np.asarray(P_exposed, dtype=np.float64)
    noise_cov = np.asarray(noise_cov, dtype=np.float64)
    aucs = []
    for s in seeds:
        tr, te = _split_idx(y, s)
        post = _lrt_scores(H_clean[tr], y[tr], P_exposed[te], noise_cov, n_classes)
        aucs.append(_auc(y[te], post, n_classes))
    return float(np.mean(aucs)), float(np.std(aucs))


def gaussian_lrt_channel(H_clean, y, channel, noise_cov, n_classes, seeds):
    """The Tier-2 member through a POST-HOC channel: per seed, one fresh channel
    draw over all rows (probe_through convention), fit on clean train rows,
    score the noised held-out rows."""
    H_clean = np.asarray(H_clean, dtype=np.float64)
    noise_cov = np.asarray(noise_cov, dtype=np.float64)
    aucs = []
    for s in seeds:
        Hn = channel(H_clean, np.random.default_rng(s))
        tr, te = _split_idx(y, s)
        post = _lrt_scores(H_clean[tr], y[tr], Hn[te], noise_cov, n_classes)
        aucs.append(_auc(y[te], post, n_classes))
    return float(np.mean(aucs)), float(np.std(aucs))


TIER1_ARCHS = ("XGB", "MLP", "LoRA32")


def two_tier_battery(H_clean, P_exposed, y, n_classes, device, seeds, noise_cov):
    """The full standing battery on a fixed exposed matrix, reported per tier.

    Returns {"XGB": {...}, "MLP": {...}, "LoRA32": {...}, "LRT": {...},
             "tier1_max": float, "tier2_max": float}.
    tier1_max is the max over the black box members. tier2_max additionally
    includes the informed Gaussian-LRT.
    """
    from experiments.diagnostic import battery as _t1_battery
    out = _t1_battery(P_exposed, y, n_classes, device, seeds, archs=TIER1_ARCHS)
    m, s = gaussian_lrt(H_clean, P_exposed, y, noise_cov, n_classes, seeds)
    out["LRT"] = {"auc": m, "auc_std": s}
    out["tier1_max"] = float(max(out[a]["auc"] for a in TIER1_ARCHS))
    out["tier2_max"] = float(max(out["tier1_max"], m))
    return out
