"""Experiment 6 — can ANISOTROPIC (targeted) noise buy honest durability cheaper
than blunt isotropic noise on Adult/sex?

Experiment 5 settled that only isotropic noise gives durability that survives an
honest XGBoost/MLP probe, and it is expensive (Adult/sex needs σ≥4, keeping ~57% of
income lift; comfortable only at σ=8, ~20%). Noise works because it DESTROYS
information (a data-processing bound), not because it hides it from one probe — which
is why the targeted *projections* failed (they only removed a linear/kernel subspace
the R² certificate could see, leaking to XGB at the unprotected ~0.68 AUC).

Question: can we destroy the sex information MORE SURGICALLY — pour noise only into the
directions/dimensions that carry sex, sparing task-only directions — and so pay less
income utility for the same XGBoost-proof protection? Two targeted variants, both
compared head-to-head against isotropic noise at matched protection:

  1. SUBSPACE noise: add Gaussian noise only inside the learned rank-r most-
     sex-carrying subspace (the same HSIC subspace `fit_projection` finds — but we
     DESTROY it with noise instead of deleting it). Sweep r and σ.
  2. IMPORTANCE noise: add noise only to the top-k repr dimensions XGBoost itself uses
     to predict sex (axis-aligned targeting matched to the honest attacker). Sweep k, σ.

Honest measurement only: XGBoost + deep MLP sex recovery (acc + AUC), never R². A
config achieves honest durability iff BOTH probes are ≤ 0.55 AUC (chance 0.5). For
each method we report the MAX income lift among its honest configs — the cheapest
honest protection it can buy. If targeted noise leaks to XGB like the projections did,
that means even directional information-destruction is not enough and blunt isotropic
noise is the floor. macOS OpenMP guard: xgboost before torch. Local MPS/CPU; never AWS.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401  (import before torch on purpose)

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.hardening_test import _probe  # noqa: E402  (PCRL-auditor probe)
from experiments.smart_erasure import fit_projection  # noqa: E402  (most-sex-carrying subspace)
from utils.pcrl_io import (  # noqa: E402
    INCOME_PREDICTION_PURPOSE_IDX, build_adult_train_loader, extract_attr,
    extract_frozen_reps, extract_task_label, get_device, load_adult_encoder,
)

RESULTS_DIR = REPO_ROOT / "results"
SEEDS = [0, 1, 2]
HONEST_AUC_BAR = 0.55           # max(XGB,MLP) AUC must be ≤ this (chance 0.5)

ISO_SIGMAS = [2.0, 3.0, 4.0, 6.0, 8.0]
SUB_RANKS = [8, 16, 32]
SUB_SIGMAS = [4.0, 8.0, 16.0]
IMP_KS = [16, 32, 48]
IMP_SIGMAS = [4.0, 8.0, 16.0]


# --------------------------------------------------------------------------- #
#  Noise channels (each returns a callable H,rng -> noised H)                  #
# --------------------------------------------------------------------------- #
def iso_channel(sigma_abs):
    def f(H, rng):
        return H + rng.normal(0.0, sigma_abs, size=H.shape)
    return f


def subspace_channel(Q, sigma_abs):
    """Gaussian noise of std sigma_abs confined to span(Q) (Q: d×r orthonormal)."""
    r = Q.shape[1]

    def f(H, rng):
        Z = rng.normal(0.0, 1.0, size=(H.shape[0], r))
        return H + sigma_abs * (Z @ Q.T)
    return f


def importance_channel(mask, sigma_abs):
    """Gaussian noise of std sigma_abs only on the dims where mask==1."""
    def f(H, rng):
        return H + sigma_abs * (rng.normal(0.0, 1.0, size=H.shape) * mask)
    return f


# --------------------------------------------------------------------------- #
#  Honest measurement through a channel                                        #
# --------------------------------------------------------------------------- #
def probe_through(arch, H, y, n_classes, channel, seeds):
    from sklearn.model_selection import train_test_split
    accs, aucs = [], []
    for seed in seeds:
        X = channel(H.astype(np.float64), np.random.default_rng(seed))
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y)
        acc, auc = _probe(arch, Xtr, ytr, Xte, yte, seed, n_classes)
        accs.append(acc)
        aucs.append(auc)
    return float(np.mean(accs)), float(np.mean(aucs))


def utility_through(H, income, channel, seeds, majority):
    """Income lift over majority of a linear (LR) head through the same channel."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    lifts = []
    for seed in seeds:
        X = channel(H.astype(np.float64), np.random.default_rng(seed))
        Xtr, Xte, ytr, yte = train_test_split(
            X, income, test_size=0.25, random_state=seed, stratify=income)
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
        clf.fit(Xtr, ytr)
        lifts.append(clf.score(Xte, yte) - majority)
    return float(np.mean(lifts))


def eval_config(method, label, channel, H, sex, n_sex, income, income_majority, params):
    xa, xauc = probe_through("XGB", H, sex, n_sex, channel, SEEDS)
    ma, mauc = probe_through("MLP", H, sex, n_sex, channel, SEEDS)
    lift = utility_through(H, income, channel, SEEDS, income_majority)
    honest = bool(max(xauc, mauc) <= HONEST_AUC_BAR)
    row = {"method": method, "label": label, **params, "xgb_acc": xa, "xgb_auc": xauc,
           "mlp_acc": ma, "mlp_auc": mauc, "income_lift": lift, "honest": honest}
    print(f"  {label:<26} XGB AUC={xauc:.4f} MLP AUC={mauc:.4f}  income_lift={lift:+.4f}  "
          f"{'HONEST(≤0.55)' if honest else 'leaks'}")
    return row


# --------------------------------------------------------------------------- #
def main():
    device = get_device()
    print(f"[setup] device={device}  (OpenMP guard set; xgboost before torch)")
    encoder, _ = load_adult_encoder()
    purposes, train_ds, loader = build_adult_train_loader()
    H = extract_frozen_reps(encoder, loader, INCOME_PREDICTION_PURPOSE_IDX, device)
    sex = extract_attr(loader, "sex")
    income = extract_task_label(loader, "income")
    n_sex = int(sex.max()) + 1
    sigma_h = float(H.std(0).mean())
    income_majority = float(np.bincount(income).max() / len(income))
    H_t = torch.from_numpy(H).to(device)
    clean_lift = utility_through(H, income, lambda X, rng: X, SEEDS, income_majority)
    print(f"[setup] N={H.shape[0]} d={H.shape[1]} sigma_h={sigma_h:.4f}  sex={n_sex}cls  "
          f"income majority={income_majority:.4f}  clean income lift={clean_lift:+.4f}")

    rows = []

    # 1) Isotropic baseline ----------------------------------------------------
    print("\n--- ISOTROPIC noise (the Experiment-5 floor) ---")
    for sr in ISO_SIGMAS:
        rows.append(eval_config("isotropic", f"iso σ={sr:g}", iso_channel(sr * sigma_h),
                                H, sex, n_sex, income, income_majority, {"sigma_rel": sr}))

    # 2) Subspace (HSIC most-sex-carrying subspace) noise ----------------------
    print("\n--- SUBSPACE noise (Gaussian inside the rank-r most-sex-carrying subspace) ---")
    Qcache = {}
    for r in SUB_RANKS:
        Qcache[r] = fit_projection(H_t, sex, r, "hsic", device, seed=0).cpu().numpy().astype(np.float64)
    for r in SUB_RANKS:
        for sr in SUB_SIGMAS:
            rows.append(eval_config(
                "subspace", f"subspace r={r} σ={sr:g}", subspace_channel(Qcache[r], sr * sigma_h),
                H, sex, n_sex, income, income_majority, {"rank": r, "sigma_rel": sr}))

    # 3) Importance (top-k XGB-sex dims) noise ---------------------------------
    print("\n--- IMPORTANCE noise (Gaussian on the top-k dims XGB uses to predict sex) ---")
    from xgboost import XGBClassifier
    ranker = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                           objective="binary:logistic", tree_method="hist",
                           eval_metric="logloss", random_state=0, n_jobs=2, verbosity=0)
    ranker.fit(H.astype(np.float64), sex)
    importance = ranker.feature_importances_
    order = np.argsort(importance)[::-1]            # most sex-important dims first
    print(f"  [rank] top-8 sex-important repr dims: {order[:8].tolist()}")
    for k in IMP_KS:
        mask = np.zeros(H.shape[1], dtype=np.float64)
        mask[order[:k]] = 1.0
        for sr in IMP_SIGMAS:
            rows.append(eval_config(
                "importance", f"importance k={k} σ={sr:g}", importance_channel(mask, sr * sigma_h),
                H, sex, n_sex, income, income_majority, {"k": k, "sigma_rel": sr}))

    _head_to_head(rows, clean_lift, income_majority)
    verdict = _verdict(rows, clean_lift)

    out = {"experiment": "targeted_noise", "cell": "adult/income/sex", "honest_auc_bar": HONEST_AUC_BAR,
           "seeds": SEEDS, "sigma_h": sigma_h, "income_majority": income_majority,
           "clean_income_lift": clean_lift, "xgb_sex_importance_order": order.tolist(),
           "rows": rows, "verdict": verdict}
    (RESULTS_DIR / "targeted_noise.json").write_text(json.dumps(out, indent=2))
    _plot(rows, clean_lift, RESULTS_DIR / "targeted_noise.png")
    print(f"\n[save] {RESULTS_DIR/'targeted_noise.json'} and .png")


def _best_honest(rows, method):
    cands = [r for r in rows if r["method"] == method and r["honest"]]
    return max(cands, key=lambda r: r["income_lift"]) if cands else None


def _head_to_head(rows, clean_lift, income_majority):
    print("\n" + "=" * 84)
    print("HEAD-TO-HEAD — cheapest config reaching honest durability (XGB & MLP ≤ 0.55 AUC)")
    print("=" * 84)
    print(f"{'method':<14} {'best honest config':<22} {'XGB AUC':>8} {'MLP AUC':>8} "
          f"{'income lift':>12} {'% of clean':>11}")
    print("-" * 84)
    for method, name in [("isotropic", "isotropic"), ("subspace", "subspace (HSIC)"),
                         ("importance", "importance (XGB)")]:
        b = _best_honest(rows, method)
        if b is None:
            best_leak = min((r for r in rows if r["method"] == method),
                            key=lambda r: max(r["xgb_auc"], r["mlp_auc"]))
            print(f"{name:<14} {'— none reached ≤0.55 —':<22} "
                  f"{best_leak['xgb_auc']:>8.4f} {best_leak['mlp_auc']:>8.4f} "
                  f"{'(leaks)':>12} {'—':>11}")
        else:
            pct = b["income_lift"] / clean_lift * 100 if clean_lift > 0 else 0.0
            print(f"{name:<14} {b['label']:<22} {b['xgb_auc']:>8.4f} {b['mlp_auc']:>8.4f} "
                  f"{b['income_lift']:>+12.4f} {pct:>10.0f}%")
    print("-" * 84)
    print(f"clean income lift={clean_lift:+.4f} (majority={income_majority:.4f}); "
          f"higher lift at honest durability = cheaper protection")


def _verdict(rows, clean_lift):
    iso = _best_honest(rows, "isotropic")
    sub = _best_honest(rows, "subspace")
    imp = _best_honest(rows, "importance")
    print("\n================== EXPERIMENT 6 VERDICT ==================")
    targeted_winners = [(n, b) for n, b in [("subspace", sub), ("importance", imp)] if b is not None]
    iso_lift = iso["income_lift"] if iso else None
    beats = [(n, b) for n, b in targeted_winners if iso_lift is not None and b["income_lift"] > iso_lift + 0.003]
    if not targeted_winners:
        v = (f"BLUNT NOISE IS THE FLOOR. Neither targeted variant (subspace-in-HSIC nor "
             f"XGB-importance) reaches honest durability (XGB & MLP ≤ {HONEST_AUC_BAR}) at any "
             f"swept (rank/k, σ) — they LEAK to XGBoost exactly as the projection methods did, "
             f"because the tree-exploitable sex signal is spread across more directions than the "
             f"targeted noise covers. Only isotropic noise destroys enough information; "
             f"directional information-destruction is not enough. "
             f"Isotropic honest floor: lift {iso_lift:+.4f} "
             f"({iso_lift/clean_lift*100:.0f}% of clean) at {iso['label']}.")
    elif beats:
        n, b = max(beats, key=lambda t: t[1]["income_lift"])
        v = (f"GENUINE IMPROVEMENT over blunt noise. Targeted '{n}' noise reaches honest "
             f"durability ({b['label']}: XGB {b['xgb_auc']:.3f}, MLP {b['mlp_auc']:.3f} ≤ "
             f"{HONEST_AUC_BAR}) while keeping income lift {b['income_lift']:+.4f} "
             f"({b['income_lift']/clean_lift*100:.0f}% of clean) — vs isotropic's "
             f"{iso_lift:+.4f} ({iso_lift/clean_lift*100:.0f}%) at matched protection. "
             f"Surgical information-destruction pays less utility than blunt noise for the same "
             f"XGBoost-proof protection.")
    else:
        v = (f"NO CHEAPER THAN BLUNT NOISE. A targeted variant does reach honest durability, but "
             f"at no better income lift than isotropic (targeted best "
             f"{max(b['income_lift'] for _, b in targeted_winners):+.4f} vs isotropic "
             f"{iso_lift:+.4f}). To kill the tree-recoverable sex signal it must noise so much of "
             f"the space that it converges to isotropic; the surgical advantage does not "
             f"materialize. Blunt noise remains the practical floor.")
    print(v)
    return v


def _plot(rows, clean_lift, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    style = {"isotropic": ("o", "C0"), "subspace": ("s", "C1"), "importance": ("^", "C3")}
    for method, (mk, c) in style.items():
        pts = [(r["xgb_auc"], r["income_lift"], r["label"]) for r in rows if r["method"] == method]
        xs, ys, _ = zip(*pts)
        ax.scatter(xs, ys, marker=mk, color=c, s=70, label=method, zorder=4)
    # mark honest region and the best honest point per method
    ax.axvspan(0.45, HONEST_AUC_BAR, color="green", alpha=0.06)
    ax.axvline(HONEST_AUC_BAR, color="green", ls="--", lw=1.2, label=f"attacker bar XGB AUC={HONEST_AUC_BAR}")
    ax.axvline(0.5, color="gray", ls=":", lw=1, label="chance AUC=0.5")
    ax.axhline(clean_lift, color="gray", ls=":", lw=1, label=f"clean income lift={clean_lift:+.3f}")
    for method, (mk, c) in style.items():
        b = _best_honest(rows, method)
        if b:
            ax.annotate(b["label"], (b["xgb_auc"], b["income_lift"]), fontsize=7,
                        color=c, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("XGBoost sex AUC (attacker)  — left of green = attribute removed")
    ax.set_ylabel("income lift over majority  — higher = utility kept")
    ax.set_title("Targeted (anisotropic) vs isotropic noise on Adult/sex\n"
                 "(want: inside green band AND high — cheaper durable removal)")
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
