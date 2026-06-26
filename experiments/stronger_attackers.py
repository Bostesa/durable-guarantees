"""TEST A: is the noise channel's "STOPPED" verdict attacker-specific?

Experiment 2 stopped the rank-8 ReLU LoRA attacker at sigma=2 on Adult/sex. This
script asks whether a STRONGER attacker breaks through the SAME channel at the
SAME sigma=2.0. At fixed sigma (0.0 for contrast, 2.0 for the test) we run a
battery of attackers trying to recover "sex" from the noisy frozen repr:

  * rank-8 ReLU LoRA   (the Experiment-2 control)
  * rank-32 ReLU LoRA  (higher capacity)
  * a deep MLP probe   (PCRL's auditor: 2 hidden layers 256-256, ReLU)
  * an XGBoost probe    (PCRL's auditor)

LoRA attackers report attacked R² (the certificate) plus sex accuracy/AUC; the
MLP/XGB probes report accuracy/AUC (R² of a fixed adapted repr does not apply).
The MLP/XGB use PCRL's auditor configuration verbatim (copied from
scripts/crosspurp/run_eval_multi.py::attack_acc, which is also imported and used
as a cross-check that our accuracy matches the auditor's).

macOS note: PyTorch and XGBoost each bundle an OpenMP runtime; loading both
segfaults. We import xgboost FIRST and set the OpenMP guard env vars before any
torch import. Runs locally on MPS/CPU; never AWS.
"""

from __future__ import annotations

import os

# OpenMP guard MUST be set before torch/xgboost load (macOS duplicate-runtime crash).
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

from experiments.noise_channel_test import (  # noqa: E402
    BATCH_SIZE, SEEDS, TAU, run_attack, task_accuracy,
)
from utils.pcrl_io import (  # noqa: E402
    INCOME_PREDICTION_PURPOSE_IDX, build_adult_train_loader, extract_attr,
    extract_frozen_reps, extract_task_label, get_device, load_adult_encoder,
)
from scripts.crosspurp.run_eval_multi import attack_acc  # noqa: E402  (PCRL auditor, reused)

RESULTS_DIR = REPO_ROOT / "results"
SIGMAS_REL = [0.0, 2.0]
ESCALATION_SIGMAS_REL = [2.0, 4.0, 8.0, 16.0]   # to find where the residual AUC reaches ~chance


def _probe(arch, Xtr, ytr, Xte, yte, seed) -> tuple[float, float]:
    """Fit one PCRL-auditor classifier; return (accuracy, AUC) on held-out.

    Configs copied verbatim from run_eval_multi.py::attack_acc (binary case).
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    if arch == "MLP":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(256, 256), activation="relu",
                            alpha=1e-4, solver="adam", learning_rate_init=1e-3,
                            max_iter=100, early_stopping=True, validation_fraction=0.1,
                            n_iter_no_change=8, random_state=seed, batch_size=256)
    elif arch == "XGB":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                            objective="binary:logistic", tree_method="hist",
                            eval_metric="logloss", random_state=seed, n_jobs=2,
                            verbosity=0, use_label_encoder=False)
    elif arch == "LR":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    else:
        raise ValueError(arch)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    return float(accuracy_score(yte, clf.predict(Xte))), float(roc_auc_score(yte, prob))


def probe_battery(arch, H, y, sigma_abs, seeds) -> dict:
    """Average a probe's held-out sex accuracy/AUC over seeds (fresh channel noise)."""
    from sklearn.model_selection import train_test_split
    accs, aucs = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        X = H.astype(np.float64)
        if sigma_abs > 0.0:
            X = X + rng.normal(0.0, sigma_abs, size=X.shape)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y)
        acc, auc = _probe(arch, Xtr, ytr, Xte, yte, seed)
        accs.append(acc)
        aucs.append(auc)
    return {"acc": float(np.mean(accs)), "auc": float(np.mean(aucs)),
            "acc_std": float(np.std(accs)), "auc_std": float(np.std(aucs))}


def lora_battery(H_t, y_t, y_np, device, sigma_abs, rank, alpha, seeds) -> dict:
    """Average the ReLU LoRA attacker's R²/acc/AUC over seeds."""
    runs = [run_attack(H_t, y_t, y_np, device, s, sigma_abs, rank=rank, alpha=alpha)
            for s in seeds]
    return {
        "r2": float(np.mean([r["final_linear_r2"] for r in runs])),
        "r2_std": float(np.std([r["final_linear_r2"] for r in runs])),
        "acc": float(np.mean([r["acc"] for r in runs])),
        "auc": float(np.mean([r["auc"] for r in runs])),
    }


def main() -> None:
    device = get_device()
    print(f"[setup] device={device}  (OpenMP guard set; xgboost imported before torch)")
    encoder, lora_target = load_adult_encoder()
    purposes, train_ds, loader = build_adult_train_loader(batch_size=BATCH_SIZE)
    H = extract_frozen_reps(encoder, loader, INCOME_PREDICTION_PURPOSE_IDX, device)
    y = extract_attr(loader, "sex")
    income = extract_task_label(loader, "income")
    sigma_h = float(H.std(0).mean())
    sex_majority = float(np.bincount(y).max() / len(y))
    print(f"[setup] N={H.shape[0]} sigma_h={sigma_h:.4f} sex majority acc={sex_majority:.4f}")

    H_t = torch.from_numpy(H).to(device)
    y_t = torch.from_numpy(y).long().to(device)

    # Cross-check that our probe == PCRL's auditor (genuine reuse).
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(H.astype(np.float64), y, test_size=0.25,
                                          random_state=0, stratify=y)
    pcrl_acc = attack_acc(Xtr, ytr, Xte, yte, "XGB", 0)
    ours_acc, _ = _probe("XGB", Xtr, ytr, Xte, yte, 0)
    print(f"[reuse-check] XGB acc: PCRL attack_acc={pcrl_acc:.4f}  ours={ours_acc:.4f}  "
          f"(|Δ|={abs(pcrl_acc - ours_acc):.4f})")

    ATTACKERS = ["rank8 ReLU LoRA", "rank32 ReLU LoRA", "MLP probe (256-256)", "XGB probe"]
    table = {}   # (attacker, sigma_rel) -> metrics
    for sig_rel in SIGMAS_REL:
        sa = sig_rel * sigma_h
        print(f"\n--- battery at sigma_rel={sig_rel} (abs={sa:.3f}) ---")
        table[("rank8 ReLU LoRA", sig_rel)] = lora_battery(H_t, y_t, y, device, sa, 8, 16.0, SEEDS)
        table[("rank32 ReLU LoRA", sig_rel)] = lora_battery(H_t, y_t, y, device, sa, 32, 64.0, SEEDS)
        table[("MLP probe (256-256)", sig_rel)] = probe_battery("MLP", H, y, sa, SEEDS)
        table[("XGB probe", sig_rel)] = probe_battery("XGB", H, y, sa, SEEDS)
        for a in ATTACKERS:
            m = table[(a, sig_rel)]
            r2 = f"{m['r2']:.4f}" if "r2" in m else "  n/a "
            print(f"  {a:<22} R²={r2}  sex_acc={m['acc']:.4f}  sex_AUC={m['auc']:.4f}")

    _print_table(table, ATTACKERS, sex_majority)

    # Verdict. The question is whether a STRONGER attacker beats the rank-8
    # CONTROL at sigma=2 — not whether absolute AUC clears a fixed line. We also
    # report the (attacker-independent) residual AUC leak the R²-certificate
    # misses, and how much noise is needed to close it to ~chance.
    sig2 = 2.0
    control = table[("rank8 ReLU LoRA", sig2)]
    control_auc = control["auc"]
    stronger = ["rank32 ReLU LoRA", "MLP probe (256-256)", "XGB probe"]
    auc2 = {a: table[(a, sig2)]["auc"] for a in ATTACKERS}
    strongest = max(stronger, key=lambda a: auc2[a])
    capacity_helps = auc2[strongest] > control_auc + 0.03  # does extra capacity break through?
    lora_r2 = {a: table[(a, sig2)]["r2"] for a in ATTACKERS if "r2" in table[(a, sig2)]}

    print("\n================== TEST A VERDICT ==================")
    print(f"At sigma=2: all LoRA attacked R² = {min(lora_r2.values()):.4f}-{max(lora_r2.values()):.4f} "
          f"(all < τ={TAU}); sex accuracy ≈ majority {sex_majority:.3f} for every attacker; "
          f"sex AUC: control(rank8)={control_auc:.3f}, "
          f"rank32={auc2['rank32 ReLU LoRA']:.3f}, MLP={auc2['MLP probe (256-256)']:.3f}, "
          f"XGB={auc2['XGB probe']:.3f}.")
    if not capacity_helps:
        print(f"VERDICT: ATTACKER-ROBUST at sigma=2. Extra capacity does NOT break through — "
              f"rank-32 LoRA, deep MLP, and XGB recover sex no better than the rank-8 control "
              f"(max stronger AUC {auc2[strongest]:.3f} vs control {control_auc:.3f}). "
              f"'STOPPED' is not attacker-specific.")
    else:
        print(f"VERDICT: NOT attacker-robust — {strongest} (AUC {auc2[strongest]:.3f}) beats the "
              f"rank-8 control (AUC {control_auc:.3f}) by >0.03 at sigma=2.")

    # Honest caveat: the residual AUC (~0.55-0.58) is a few SE above chance and is
    # NOT captured by the R²-certificate. Report how much noise closes it to ~chance.
    print(f"\nCaveat: at sigma=2 a faint residual rank-signal remains (AUC≈{control_auc:.3f}) that "
          f"the R²-certificate scores as stopped and that no attacker exceeds. Closing it to ~chance:")
    escalation = []
    arch_map = {"rank32 ReLU LoRA": None, "MLP probe (256-256)": "MLP", "XGB probe": "XGB"}
    arch = arch_map[strongest]
    for sr in ESCALATION_SIGMAS_REL:
        sa = sr * sigma_h
        if arch:
            m = probe_battery(arch, H, y, sa, SEEDS)
        else:
            m = lora_battery(H_t, y_t, y, device, sa, 32, 64.0, SEEDS)
        tacc = float(np.mean([task_accuracy(H, income, sa, s) for s in SEEDS]))
        escalation.append({"sigma_rel": sr, "auc": m["auc"], "task_acc": tacc})
        print(f"  sigma_rel={sr}: {strongest} sex_AUC={m['auc']:.4f} | income_acc={tacc:.4f}")
        if m["auc"] <= 0.53:
            print(f"  -> residual closed (AUC≈chance) at sigma_rel={sr}; "
                  f"income acc there = {tacc:.4f}")
            break

    out = {"experiment": "stronger_attackers", "sigma_rel_tested": SIGMAS_REL,
           "sigma_h": sigma_h, "sex_majority_acc": sex_majority, "tau": TAU,
           "table": {f"{a}|sigma{sr}": table[(a, sr)] for a in ATTACKERS for sr in SIGMAS_REL},
           "control_auc_sigma2": control_auc, "strongest_stronger_attacker": strongest,
           "capacity_helps_at_sigma2": bool(capacity_helps),
           "residual_closing_escalation": escalation}
    (RESULTS_DIR / "stronger_attackers.json").write_text(json.dumps(out, indent=2))
    print(f"\n[save] {RESULTS_DIR / 'stronger_attackers.json'}")


def _print_table(table, attackers, sex_majority) -> None:
    print("\n=========== TEST A: stronger attackers vs noise channel ===========")
    print(f"{'attacker':<22} | {'σ=0: R²':>8} {'acc':>6} {'AUC':>6} | "
          f"{'σ=2: R²':>8} {'acc':>6} {'AUC':>6}")
    print("-" * 74)
    for a in attackers:
        m0, m2 = table[(a, 0.0)], table[(a, 2.0)]
        r20 = f"{m0['r2']:.3f}" if "r2" in m0 else " n/a "
        r22 = f"{m2['r2']:.3f}" if "r2" in m2 else " n/a "
        print(f"{a:<22} | {r20:>8} {m0['acc']:>6.3f} {m0['auc']:>6.3f} | "
              f"{r22:>8} {m2['acc']:>6.3f} {m2['auc']:>6.3f}")
    print("-" * 74)
    print(f"τ = {TAU} (R²)   sex majority-class acc = {sex_majority:.4f}   chance AUC = 0.5")


if __name__ == "__main__":
    main()
