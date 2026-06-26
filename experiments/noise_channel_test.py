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


def run_attack(H_t, y_t, y_np, device, seed, sigma_abs) -> dict:
    """Train the ReLU attacker against the (frozen, resampled) noise channel."""
    _seed_all(seed)
    n = H_t.shape[0]
    attacker = FreshLoRAAttacker(rank=ADULT_LORA_RANK, alpha=ADULT_LORA_ALPHA).to(device)
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
    for step in range(1, N_STEPS + 1):
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
    return {"seed": seed, "curve": curve,
            "final_linear_r2": final["linear_r2"],
            "final_linear_r2_std": final["linear_r2_std"],
            "final_dominant_axis_r2": final["dominant_axis_r2"]}


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


def main() -> None:
    device = get_device()
    print(f"[setup] device={device}  checkpoint={CHECKPOINT}")

    encoder, lora_target = load_adult_encoder(CHECKPOINT)
    purposes, train_ds, train_loader = build_adult_train_loader(batch_size=BATCH_SIZE)
    H = extract_frozen_reps(encoder, train_loader, INCOME_PREDICTION_PURPOSE_IDX, device)
    y = extract_attr(train_loader, SENSITIVE_ATTR)
    income = extract_task_label(train_loader, TASK)

    sigma_h = float(H.std(0).mean())   # per-dim std of the representation (~0.37)
    print(f"[setup] N={H.shape[0]} D={H.shape[1]} lora_target={lora_target!r} "
          f"repr per-dim std (sigma_h)={sigma_h:.4f}")
    income_majority = float(np.bincount(income).max() / len(income))
    print(f"[setup] income majority-class acc = {income_majority:.4f}")

    H_t = torch.from_numpy(H).to(device)
    y_t = torch.from_numpy(y).long().to(device)

    # ---- STAGE 1: clean baseline (sigma=0), 3 seeds ------------------------
    print("\n=== STAGE 1: clean baseline (sigma=0) ===")
    raw0 = _measure_avg(lambda h: h, H_t, y, 0.0, 1)
    print(f"  certificate at rest (raw linear R^2)         = {raw0['linear_r2']:.5f}")
    stage1 = [run_attack(H_t, y_t, y, device, s, 0.0) for s in SEEDS]
    finals0 = [r["final_linear_r2"] for r in stage1]
    print(f"  attacked final linear R^2 per seed           = "
          f"{[round(v, 4) for v in finals0]}")
    print(f"  attacked final linear R^2 (mean +/- std)     = "
          f"{np.mean(finals0):.4f} +/- {np.std(finals0):.4f}  "
          f"({'BREACH' if np.mean(finals0) > TAU else 'no breach'} vs tau={TAU})")

    # ---- STAGE 2: noise-channel sweep --------------------------------------
    print("\n=== STAGE 2: noise-channel sweep ===")
    rows = []
    for sig_rel in SIGMA_REL_SWEEP:
        sigma_abs = sig_rel * sigma_h
        # Reuse Stage-1 results for sigma=0 (identical computation).
        if sig_rel == 0.0:
            attacks = stage1
        else:
            attacks = [run_attack(H_t, y_t, y, device, s, sigma_abs) for s in SEEDS]
        baseline = _measure_avg(lambda h: h, H_t, y, sigma_abs, FINAL_DRAWS)
        task_accs = [task_accuracy(H, income, sigma_abs, s) for s in SEEDS]
        finals = [a["final_linear_r2"] for a in attacks]
        finals_dom = [a["final_dominant_axis_r2"] for a in attacks]
        row = {
            "sigma_rel": sig_rel,
            "sigma_abs": sigma_abs,
            "task_acc_mean": float(np.mean(task_accs)),
            "task_acc_std": float(np.std(task_accs)),
            "baseline_linear_r2": baseline["linear_r2"],
            "attacked_linear_r2_mean": float(np.mean(finals)),
            "attacked_linear_r2_std": float(np.std(finals)),
            "attacked_dominant_axis_r2_mean": float(np.mean(finals_dom)),
            "breaches_tau": bool(np.mean(finals) > TAU),
            "per_seed_attacks": attacks,
        }
        rows.append(row)
        print(f"  sigma_rel={sig_rel:<4} (abs={sigma_abs:.3f}) | "
              f"task_acc={row['task_acc_mean']:.4f} | "
              f"baseline R^2={row['baseline_linear_r2']:.4f} | "
              f"attacked R^2={row['attacked_linear_r2_mean']:.4f}"
              f"+/-{row['attacked_linear_r2_std']:.4f} | "
              f"{'BREACH' if row['breaches_tau'] else 'STOPPED'}")

    _print_table(rows, income_majority)

    out = {
        "experiment": "noise_channel_test",
        "checkpoint": str(CHECKPOINT),
        "purpose": purposes[INCOME_PREDICTION_PURPOSE_IDX].name,
        "sensitive_attr": SENSITIVE_ATTR,
        "task": TASK,
        "tau": TAU,
        "device": str(device),
        "seeds": SEEDS,
        "sigma_rel_sweep": SIGMA_REL_SWEEP,
        "repr_per_dim_std": sigma_h,
        "income_majority_acc": income_majority,
        "config": {"n_steps": N_STEPS, "eval_every": EVAL_EVERY, "batch_size": BATCH_SIZE,
                   "lr": LR, "rank": ADULT_LORA_RANK, "alpha": ADULT_LORA_ALPHA,
                   "eval_draws": EVAL_DRAWS, "final_draws": FINAL_DRAWS},
        "rows": rows,
    }
    json_path = RESULTS_DIR / "noise_channel_test.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\n[save] curves -> {json_path}")
    _plot(rows, income_majority, json_path.with_suffix(".png"))
    print(f"[save] plot   -> {json_path.with_suffix('.png')}")


def _print_table(rows, income_majority) -> None:
    print("\n================ DURABILITY vs UTILITY ================")
    print(f"{'sigma':>6} {'task_acc':>9} {'baseR^2':>8} {'attackedR^2 (mean+/-std)':>26} "
          f"{'breach?':>8}")
    print("-" * 64)
    for r in rows:
        att = f"{r['attacked_linear_r2_mean']:.4f}+/-{r['attacked_linear_r2_std']:.4f}"
        print(f"{r['sigma_rel']:>6} {r['task_acc_mean']:>9.4f} "
              f"{r['baseline_linear_r2']:>8.4f} {att:>26} "
              f"{('BREACH' if r['breaches_tau'] else 'STOPPED'):>8}")
    print("-" * 64)
    print(f"tau = {TAU}   income majority-class acc = {income_majority:.4f}")


def _plot(rows, income_majority, png_path) -> None:
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
                 label="attacked R² (ReLU LoRA, mean±std/3 seeds)")
    ax1.plot(sig, base, marker="^", color="C0", lw=1.5, ls="-",
             label="certificate at rest (raw linear R²)")
    ax1.axhline(TAU, color="black", ls="--", lw=1.2, label=f"τ = {TAU} threshold")
    ax1.fill_between(sig, TAU, att, where=[v > TAU for v in att],
                     color="C3", alpha=0.08)
    ax1.set_xlabel("noise σ (relative to repr per-dim std ≈ 0.37)")
    ax1.set_ylabel("R²(representation, sex)")
    ax1.set_ylim(0, max(max(att), max(base)) * 1.2)

    ax2 = ax1.twinx()
    ax2.plot(sig, acc, marker="s", color="C2", lw=2, label="task accuracy (income)")
    ax2.axhline(income_majority, color="C2", ls=":", lw=1,
                label=f"income majority baseline = {income_majority:.3f}")
    ax2.set_ylabel("task accuracy (income)", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    ax2.set_ylim(min(income_majority - 0.02, min(acc) - 0.02), max(acc) + 0.02)

    ax1.set_title("Noise-channel erasure: durability (attacked R²) vs utility\n"
                  "(frozen PCRL repr + Gaussian channel; same ReLU attacker)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=7.5, ncol=1)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
