"""Experiment 2: does a NOISE-CHANNEL erasure module survive the ReLU attack?

Experiment 1 showed LEACE-style *linear* erasure fails against a nonlinear
(ReLU) downstream probe: a fresh rank-8 LoRA + linear head recovers "sex" from
the frozen PCRL representation, with linear R^2 climbing 0.035 -> ~0.104 (past
tau=0.05). Linear erasure is defeated because the linear certificate is blind to
the nonlinearly-encoded signal the attacker surfaces.

A Gaussian NOISE CHANNEL is the one erasure approach with a provable
information-theoretic guarantee: adding h_noisy = h + N(0, sigma^2 I) caps the
mutual information I(h_noisy; A) regardless of how the downstream attacker is
built (data-processing inequality). This experiment measures whether that
guarantee actually stops the SAME ReLU attacker, and at what cost to task
utility (income accuracy).

Design
------
STAGE 1 — clean baseline (sigma=0), seeds 0/1/2: rerun the Experiment-1 ReLU
attack on the plain frozen representation; confirm the breach reproduces.

STAGE 2 — noise channel: the noise is part of the FROZEN module and is resampled
on every forward pass, so the attacker faces the *channel*, not a fixed
perturbation it could invert. Sweep sigma (relative to the repr's per-dim std
~0.37) over {0, 0.25, 0.5, 1.0, 2.0}. For each sigma:
  * run the full ReLU attack (3 seeds), record the attacked-R^2 curve;
  * measure the certificate-at-rest (raw linear R^2 of h_noisy vs sex);
  * measure task utility: a simple linear head (logistic regression) on h_noisy
    predicting income, test accuracy.

The same attacker (FreshLoRAAttacker) and certificate instruments
(measure_certificate -> PCRL's linear_r2 + dominant-axis R^2) from Experiment 1
are reused verbatim. Runs locally on MPS/CPU; never AWS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse Experiment 1's attacker and certificate instrument verbatim.
from experiments.falsification_attack import (  # noqa: E402
    TAU,
    FreshLoRAAttacker,
    measure_certificate,
)
from utils.pcrl_io import (  # noqa: E402
    ADULT_LORA_ALPHA,
    ADULT_LORA_RANK,
    CHECKPOINT,
    INCOME_PREDICTION_PURPOSE_IDX,
    build_adult_train_loader,
    extract_attr,
    extract_task_label,
    extract_frozen_reps,
    get_device,
    load_adult_encoder,
)

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = [0, 1, 2]
SIGMA_REL_SWEEP = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]   # multiples of the repr per-dim std
N_STEPS = 300
EVAL_EVERY = 10
BATCH_SIZE = 512
LR = 1e-2
SENSITIVE_ATTR = "sex"
TASK = "income"
EVAL_DRAWS = 5          # noise draws averaged per curve eval point
FINAL_DRAWS = 20        # noise draws for the stable end-of-training estimate


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def _noisy(H_t: torch.Tensor, sigma_abs: float) -> torch.Tensor:
    """h_noisy = h + N(0, sigma_abs^2 I), fresh draw (the frozen channel)."""
    if sigma_abs <= 0.0:
        return H_t
    return H_t + sigma_abs * torch.randn_like(H_t)


@torch.no_grad()
def _measure_avg(rep_fn, H_t, y, sigma_abs, draws) -> dict:
    """Average a certificate measurement over `draws` fresh noise channels.

    rep_fn maps a (noisy) representation tensor to the representation to probe
    (identity for the raw certificate-at-rest; attacker.adapt for the attack).
    """
    n = 1 if sigma_abs <= 0.0 else draws
    lin, dom = [], []
    for _ in range(n):
        H_probe = rep_fn(_noisy(H_t, sigma_abs)).cpu().numpy().astype(np.float32)
        cert = measure_certificate(H_probe, y)
        lin.append(cert["linear_r2"])
        dom.append(cert["dominant_axis_r2"])
    return {
        "linear_r2": float(np.mean(lin)),
        "linear_r2_std": float(np.std(lin)),
        "dominant_axis_r2": float(np.mean(dom)),
    }


def run_attack(H_t, y_t, y_np, device, seed, sigma_abs,
               rank=ADULT_LORA_RANK, alpha=ADULT_LORA_ALPHA, n_steps=N_STEPS) -> dict:
    """Train the ReLU LoRA attacker against the (frozen, resampled) noise channel."""
    _seed_all(seed)
    n = H_t.shape[0]
    n_classes = int(y_np.max()) + 1
    attacker = FreshLoRAAttacker(repr_dim=H_t.shape[1], n_classes=n_classes,
                                 rank=rank, alpha=alpha).to(device)
    opt = torch.optim.Adam(attacker.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    curve = []

    def eval_point(step: int) -> None:
        attacker.eval()
        m = _measure_avg(lambda h: attacker.adapt(h), H_t, y_np, sigma_abs, EVAL_DRAWS)
        attacker.train()
        curve.append({"step": step, **m})

    eval_point(0)
    attacker.train()
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, n, (BATCH_SIZE,), device=device)
        hb = _noisy(H_t[idx], sigma_abs)          # fresh channel each forward
        logits = attacker(hb)
        loss = loss_fn(logits, y_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            eval_point(step)

    # Stable end-of-training estimate over more draws.
    attacker.eval()
    final = _measure_avg(lambda h: attacker.adapt(h), H_t, y_np, sigma_abs, FINAL_DRAWS)
    acc, auc = _attacker_acc_auc(attacker, H_t, y_np, sigma_abs)
    return {"seed": seed, "curve": curve,
            "final_linear_r2": final["linear_r2"],
            "final_linear_r2_std": final["linear_r2_std"],
            "final_dominant_axis_r2": final["dominant_axis_r2"],
            "acc": acc, "auc": auc}


@torch.no_grad()
def _attacker_acc_auc(attacker, H_t, y_np, sigma_abs) -> tuple[float, float]:
    """Sex-recovery accuracy + AUC of the trained LoRA attacker's head, through a
    fresh noise draw (held-out-style: independent channel noise per sample)."""
    from sklearn.metrics import roc_auc_score
    attacker.eval()
    logits = attacker(_noisy(H_t, sigma_abs)).cpu().numpy()
    pred = logits.argmax(1)
    acc = float((pred == y_np).mean())
    n_classes = logits.shape[1]
    try:
        if n_classes == 2:
            prob = _softmax(logits)[:, 1]
            auc = float(roc_auc_score(y_np, prob))
        else:
            auc = float(roc_auc_score(y_np, _softmax(logits),
                                      multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return acc, auc


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def task_accuracy(H_np, income_np, sigma_abs, seed) -> float:
    """Simple LINEAR head (logistic regression) on h_noisy -> income test acc.

    Noise is drawn per-sample, so test rows carry independent channel noise the
    model never saw — a held-out evaluation through the same channel.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    X = H_np.astype(np.float64)
    if sigma_abs > 0.0:
        X = X + rng.normal(0.0, sigma_abs, size=X.shape)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, income_np, test_size=0.25, random_state=seed, stratify=income_np
    )
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    clf.fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))


def run_noise_sweep(H, y_np, task_np, device, *, cell_name, results_stem,
                    attr_name, task_name, sigma_rel_sweep=SIGMA_REL_SWEEP,
                    seeds=SEEDS, extra_meta=None) -> list:
    """Full noise-channel sweep + ReLU attack for ONE (dataset, purpose, attr, task)
    cell. Reused by Adult (here) and by the generalization experiment (HMDA/Diabetes).

    Returns the rows list; writes results/<results_stem>.json and .png.
    """
    sigma_h = float(H.std(0).mean())
    task_majority = float(np.bincount(task_np).max() / len(task_np))
    attr_majority = float(np.bincount(y_np).max() / len(y_np))
    print(f"\n##### CELL: {cell_name} #####")
    print(f"[setup] N={H.shape[0]} D={H.shape[1]} repr per-dim std (sigma_h)={sigma_h:.4f}")
    print(f"[setup] attr='{attr_name}' ({int(y_np.max())+1} classes, majority {attr_majority:.3f}) | "
          f"task='{task_name}' (majority {task_majority:.3f})")

    H_t = torch.from_numpy(H).to(device)
    y_t = torch.from_numpy(y_np).long().to(device)

    rows = []
    for sig_rel in sigma_rel_sweep:
        sigma_abs = sig_rel * sigma_h
        attacks = [run_attack(H_t, y_t, y_np, device, s, sigma_abs) for s in seeds]
        baseline = _measure_avg(lambda h: h, H_t, y_np, sigma_abs, FINAL_DRAWS)
        task_accs = [task_accuracy(H, task_np, sigma_abs, s) for s in seeds]
        finals = [a["final_linear_r2"] for a in attacks]
        finals_dom = [a["final_dominant_axis_r2"] for a in attacks]
        row = {
            "sigma_rel": sig_rel, "sigma_abs": sigma_abs,
            "task_acc_mean": float(np.mean(task_accs)), "task_acc_std": float(np.std(task_accs)),
            "baseline_linear_r2": baseline["linear_r2"],
            "baseline_dominant_axis_r2": baseline["dominant_axis_r2"],
            "attacked_linear_r2_mean": float(np.mean(finals)),
            "attacked_linear_r2_std": float(np.std(finals)),
            "attacked_dominant_axis_r2_mean": float(np.mean(finals_dom)),
            "attacked_acc_mean": float(np.mean([a["acc"] for a in attacks])),
            "attacked_auc_mean": float(np.mean([a["auc"] for a in attacks])),
            "breaches_tau": bool(np.mean(finals) > TAU),
            "per_seed_attacks": attacks,
        }
        rows.append(row)
        print(f"  sigma_rel={sig_rel:<4} (abs={sigma_abs:.3f}) | "
              f"task_acc={row['task_acc_mean']:.4f} | "
              f"cert@rest linR^2={row['baseline_linear_r2']:.4f} | "
              f"attacked linR^2={row['attacked_linear_r2_mean']:.4f}"
              f"+/-{row['attacked_linear_r2_std']:.4f} | "
              f"{'BREACH' if row['breaches_tau'] else 'STOPPED'}")

    _print_table(rows, task_majority, attr_name, task_name, cell_name)

    out = {
        "experiment": "noise_channel_test", "cell": cell_name,
        "sensitive_attr": attr_name, "task": task_name, "tau": TAU,
        "device": str(device), "seeds": seeds, "sigma_rel_sweep": sigma_rel_sweep,
        "repr_per_dim_std": sigma_h, "task_majority_acc": task_majority,
        "attr_majority_acc": attr_majority,
        "config": {"n_steps": N_STEPS, "eval_every": EVAL_EVERY, "batch_size": BATCH_SIZE,
                   "lr": LR, "eval_draws": EVAL_DRAWS, "final_draws": FINAL_DRAWS},
        "extra_meta": extra_meta or {}, "rows": rows,
    }
    json_path = RESULTS_DIR / f"{results_stem}.json"
    json_path.write_text(json.dumps(out, indent=2))
    _plot(rows, task_majority, attr_name, task_name, cell_name, json_path.with_suffix(".png"))
    print(f"[save] {json_path}  and  {json_path.with_suffix('.png')}")
    return rows


def main() -> None:
    """Adult / income_prediction / sex — the original Experiment 2 cell."""
    device = get_device()
    print(f"[setup] device={device}  checkpoint={CHECKPOINT}")
    encoder, lora_target = load_adult_encoder(CHECKPOINT)
    purposes, train_ds, train_loader = build_adult_train_loader(batch_size=BATCH_SIZE)
    H = extract_frozen_reps(encoder, train_loader, INCOME_PREDICTION_PURPOSE_IDX, device)
    y = extract_attr(train_loader, SENSITIVE_ATTR)
    income = extract_task_label(train_loader, TASK)
    run_noise_sweep(H, y, income, device,
                    cell_name="adult / income_prediction / sex",
                    results_stem="noise_channel_test",
                    attr_name=SENSITIVE_ATTR, task_name=TASK)


def _print_table(rows, task_majority, attr_name, task_name, cell_name) -> None:
    print(f"\n========= DURABILITY vs UTILITY: {cell_name} =========")
    print(f"{'sigma':>6} {'task_acc':>9} {'cert@rest':>10} "
          f"{'attacked linR^2 (mean+/-std)':>30} {'breach?':>8}")
    print("-" * 70)
    for r in rows:
        att = f"{r['attacked_linear_r2_mean']:.4f}+/-{r['attacked_linear_r2_std']:.4f}"
        print(f"{r['sigma_rel']:>6} {r['task_acc_mean']:>9.4f} "
              f"{r['baseline_linear_r2']:>10.4f} {att:>30} "
              f"{('BREACH' if r['breaches_tau'] else 'STOPPED'):>8}")
    print("-" * 70)
    print(f"tau = {TAU}   task '{task_name}' majority acc = {task_majority:.4f}")


def _plot(rows, task_majority, attr_name, task_name, cell_name, png_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = [r["sigma_rel"] for r in rows]
    att = [r["attacked_linear_r2_mean"] for r in rows]
    att_sd = [r["attacked_linear_r2_std"] for r in rows]
    base = [r["baseline_linear_r2"] for r in rows]
    acc = [r["task_acc_mean"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    ax1.errorbar(sig, att, yerr=att_sd, marker="o", color="C3", lw=2, capsize=3,
                 label=f"attacked R² (ReLU LoRA, mean±std/{len(rows[0]['per_seed_attacks'])} seeds)")
    ax1.plot(sig, base, marker="^", color="C0", lw=1.5, ls="-",
             label="certificate at rest (raw linear R²)")
    ax1.axhline(TAU, color="black", ls="--", lw=1.2, label=f"τ = {TAU} threshold")
    ax1.fill_between(sig, TAU, att, where=[v > TAU for v in att],
                     color="C3", alpha=0.08)
    ax1.set_xlabel("noise σ (relative to repr per-dim std)")
    ax1.set_ylabel(f"R²(representation, {attr_name})")
    ax1.set_ylim(0, max(max(att), max(base), TAU) * 1.2)

    ax2 = ax1.twinx()
    ax2.plot(sig, acc, marker="s", color="C2", lw=2, label=f"task accuracy ({task_name})")
    ax2.axhline(task_majority, color="C2", ls=":", lw=1,
                label=f"{task_name} majority baseline = {task_majority:.3f}")
    ax2.set_ylabel(f"task accuracy ({task_name})", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    ax2.set_ylim(min(task_majority - 0.02, min(acc) - 0.02), max(acc) + 0.02)

    ax1.set_title(f"Noise-channel erasure: durability vs utility\n{cell_name}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=7.5, ncol=1)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
