"""Experiment 9 — GOAL 3: does SURGICAL information destruction beat blunt noise
beyond Adult/sex, under the FULL attacker battery?

Exp 6 found one live architectural hope: noise confined to the attribute-carrying
HSIC subspace ("surgical") kept 86% of Adult income lift vs isotropic's 57% at
matched honest protection (XGB+MLP <= 0.55). But that was one cell, and the
battery had no LoRA. Here the same head-to-head runs on cells across the
predictor spectrum, and every winning config must survive XGBoost + deep MLP +
rank-32 LoRA (NEVER R^2):

  * anchor:  adult/sex on the FROZEN PCRL income repr — the exact Exp-6 setting,
             now with the LoRA attacker added;
  * easy:    hmda/race/loan_decision      (predictor ~0.514) on the clean trained
             channel P (from-raw-x, sigma=0 — the Exp-8 architecture);
  * middle:  hmda/race/loan_amount_band   (~0.584) on clean trained P;
  * hard:    adult/sex/income             (~0.603) on clean trained P.

Protocol per cell (Exp-6 protocol, generalised): fix the base representation H;
apply post-hoc noise channels (fresh noise per row) — isotropic vs subspace
(Gaussian confined to the rank-r most-attribute-carrying HSIC subspace of H,
fit with smart_erasure.fit_projection); measure attribute recovery through the
channel with XGB + MLP, and task utility as the lift of a logistic head retrained
through the same channel. Configs with max(XGB, MLP) <= 0.55 are candidates; the
best candidate per method (highest lift) must then also hold the rank-32 LoRA
attacker <= 0.55, else it is discarded and the next-best candidate is tried.
Report utility kept, surgical vs blunt, at full-battery-honest protection.
Local MPS/CPU; never AWS. macOS OpenMP guard: xgboost before torch.
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

from experiments.diagnostic import label_predictor, load_cell, probe_lora  # noqa: E402
from experiments.sealed_channel import train_model  # noqa: E402
from experiments.smart_erasure import fit_projection  # noqa: E402
from experiments.targeted_noise import (  # noqa: E402
    iso_channel, probe_through, subspace_channel, utility_through,
)
from utils.pcrl_io import (  # noqa: E402
    INCOME_PREDICTION_PURPOSE_IDX, build_adult_train_loader, extract_attr,
    extract_frozen_reps, extract_task_label, get_device, load_adult_encoder,
)

RESULTS_DIR = REPO_ROOT / "results"
BAR = 0.55
SEEDS = [0, 1]                       # probe/utility seeds inside the grid
ISO_SIGMAS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
SUB_RANKS = [8, 16, 32]
SUB_SIGMAS = [8.0, 16.0, 32.0, 64.0]

CELLS = [
    # (key, dataset, attr, task, base). base: "frozen" = frozen PCRL repr
    # (Exp-6 anchor, adult income purpose); "trainedP" = clean from-raw-x channel.
    ("anchor(frozen) adult/sex/income", "adult", "sex", "income", "frozen"),
    ("easy hmda/race/loan_decision", "hmda", "race", "loan_decision", "trainedP"),
    ("middle hmda/race/loan_amount_band", "hmda", "race", "loan_amount_band", "trainedP"),
    ("hard adult/sex/income", "adult", "sex", "income", "trainedP"),
]


def get_base_rep(dataset, attr_name, task_name, base, device):
    """The fixed representation H the channels act on, plus labels/majorities."""
    if base == "frozen":
        encoder, _ = load_adult_encoder()
        _, _, loader = build_adult_train_loader()
        loader = list(loader)
        H = extract_frozen_reps(encoder, loader, INCOME_PREDICTION_PURPOSE_IDX, device)
        attr = extract_attr(loader, attr_name)
        task = extract_task_label(loader, task_name)
        n_attr, n_task = int(attr.max()) + 1, int(task.max()) + 1
        task_maj = float(np.bincount(task).max() / len(task))
    else:
        X, attr, task, n_attr, n_task, _, task_maj = load_cell(dataset, attr_name, task_name)
        X_t = torch.from_numpy(X).to(device)
        H, _, _ = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                              alpha=0.0, device=device, pub_noise=0.0, seed=0)
    return H, attr, task, n_attr, n_task, task_maj


def eval_config(method, label, channel, H, attr, n_attr, task, task_maj, params):
    xa, xauc = probe_through("XGB", H, attr, n_attr, channel, SEEDS)
    ma, mauc = probe_through("MLP", H, attr, n_attr, channel, SEEDS)
    tl = utility_through(H, task, channel, SEEDS, task_maj)
    row = {"method": method, "label": label, **params, "xgb_auc": xauc, "mlp_auc": mauc,
           "lift": tl, "candidate": bool(max(xauc, mauc) <= BAR)}
    print(f"  {label:<24} XGB={xauc:.4f} MLP={mauc:.4f} lift={tl:+.4f} "
          f"{'candidate(≤0.55)' if row['candidate'] else 'leaks'}", flush=True)
    return row


def lora_check(H, attr, n_attr, channel, device):
    """Rank-32 LoRA attack through the channel (one fresh noise draw per seed)."""
    aucs = []
    for s in SEEDS:
        Hn = channel(H.astype(np.float64), np.random.default_rng(100 + s)).astype(np.float32)
        m, _ = probe_lora(Hn, attr, n_attr, [s], device)
        aucs.append(m)
    return float(np.mean(aucs))


def run_cell(key, dataset, attr_name, task_name, base, device):
    print("\n" + "#" * 92 + f"\n# {key}  (base={base})\n" + "#" * 92, flush=True)
    H, attr, task, n_attr, n_task, task_maj = get_base_rep(
        dataset, attr_name, task_name, base, device)
    pred, _ = label_predictor(attr, task, n_attr)
    sigma_h = float(H.std(0).mean())
    ident = lambda X, rng: X
    clean_lift = utility_through(H, task, ident, SEEDS, task_maj)
    _, clean_xgb = probe_through("XGB", H, attr, n_attr, ident, [0])
    print(f"[setup] N={H.shape[0]} d={H.shape[1]} sigma_h={sigma_h:.3f} predictor={pred:.4f} "
          f"clean_lift={clean_lift:+.4f} clean XGB attr AUC={clean_xgb:.3f}", flush=True)

    rows = []
    print("\n--- isotropic (blunt) ---", flush=True)
    for sr in ISO_SIGMAS:
        rows.append(eval_config("isotropic", f"iso σ={sr:g}", iso_channel(sr * sigma_h),
                                H, attr, n_attr, task, task_maj, {"sigma_rel": sr}))
        if rows[-1]["candidate"] and max(rows[-1]["xgb_auc"], rows[-1]["mlp_auc"]) <= BAR - 0.01:
            break
    print("\n--- subspace (surgical, HSIC most-attribute-carrying subspace) ---", flush=True)
    H_t = torch.from_numpy(H.astype(np.float32)).to(device)
    for r in SUB_RANKS:
        Q = fit_projection(H_t, attr, r, "hsic", device, seed=0).cpu().numpy().astype(np.float64)
        for sr in SUB_SIGMAS:
            rows.append(eval_config("subspace", f"sub r={r} σ={sr:g}",
                                    subspace_channel(Q, sr * sigma_h),
                                    H, attr, n_attr, task, task_maj,
                                    {"rank": r, "sigma_rel": sr}))

    # Best full-battery-honest config per method: best-lift candidate that also
    # holds the rank-32 LoRA ≤ bar; otherwise fall through to the next candidate.
    best = {}
    for method in ("isotropic", "subspace"):
        cands = sorted([r for r in rows if r["method"] == method and r["candidate"]],
                       key=lambda r: -r["lift"])
        chosen = None
        for c in cands:
            if c["method"] == "isotropic":
                ch = iso_channel(c["sigma_rel"] * sigma_h)
            else:
                Q = fit_projection(H_t, attr, c["rank"], "hsic", device, seed=0)
                ch = subspace_channel(Q.cpu().numpy().astype(np.float64),
                                      c["sigma_rel"] * sigma_h)
            la = lora_check(H, attr, n_attr, ch, device)
            c["lora_auc"] = la
            print(f"  [LoRA check] {method} {c['label']}: LoRA32 AUC={la:.4f} "
                  f"{'holds' if la <= BAR else 'BREACHES — discarded'}", flush=True)
            if la <= BAR:
                chosen = c
                break
        best[method] = chosen

    out = {"cell": key, "dataset": dataset, "attr": attr_name, "task": task_name,
           "base": base, "predictor": pred, "sigma_h": sigma_h,
           "clean_lift": clean_lift, "clean_xgb_auc": clean_xgb,
           "rows": rows, "best": best}
    iso, sub = best["isotropic"], best["subspace"]
    fmt = lambda b: ("none reaches full-battery ≤0.55" if b is None else
                     f"{b['label']}: lift {b['lift']:+.4f} "
                     f"({100 * b['lift'] / max(clean_lift, 1e-9):.0f}% of clean, "
                     f"XGB {b['xgb_auc']:.3f} MLP {b['mlp_auc']:.3f} LoRA {b['lora_auc']:.3f})")
    print(f"\n>>> {key}: clean lift {clean_lift:+.4f}\n"
          f"    blunt    → {fmt(iso)}\n    surgical → {fmt(sub)}", flush=True)
    return out


def main():
    device = get_device()
    print(f"[setup] device={device} bar≤{BAR} iso σ={ISO_SIGMAS} "
          f"sub r={SUB_RANKS} σ={SUB_SIGMAS}", flush=True)
    cells = [run_cell(*c, device) for c in CELLS]

    print("\n" + "=" * 100)
    print("GOAL 3 — SURGICAL vs BLUNT at full-battery honest protection (XGB+MLP+LoRA32 ≤ 0.55)")
    print("=" * 100)
    print(f"{'cell':<38} {'pred':>6} {'clean':>8} {'blunt kept':>11} {'surgical kept':>14}")
    print("-" * 100)
    for c in cells:
        pk = lambda b: "—" if c["best"][b] is None else \
            f"{100 * c['best'][b]['lift'] / max(c['clean_lift'], 1e-9):.0f}%"
        print(f"{c['cell']:<38} {c['predictor']:>6.3f} {c['clean_lift']:>+8.4f} "
              f"{pk('isotropic'):>11} {pk('subspace'):>14}")
    print("-" * 100)

    (RESULTS_DIR / "surgical_vs_blunt.json").write_text(json.dumps(
        {"experiment": "surgical_vs_blunt", "bar": BAR, "seeds": SEEDS,
         "cells": cells}, indent=2))
    _plot(cells, RESULTS_DIR / "surgical_vs_blunt.png")
    print(f"\n[save] {RESULTS_DIR/'surgical_vs_blunt.json'} and .png")


def _plot(cells, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(cells), figsize=(5.2 * len(cells), 5.6), squeeze=False)
    for ax, c in zip(axes[0], cells):
        for method, mk, col in [("isotropic", "o", "C0"), ("subspace", "s", "C1")]:
            pts = [r for r in c["rows"] if r["method"] == method]
            ax.scatter([max(r["xgb_auc"], r["mlp_auc"]) for r in pts],
                       [r["lift"] for r in pts], marker=mk, color=col, s=55,
                       label=method, zorder=3)
            b = c["best"][method]
            if b:
                ax.scatter([max(b["xgb_auc"], b["mlp_auc"])], [b["lift"]], marker=mk,
                           s=170, facecolors="none", edgecolors="red", zorder=4)
        ax.axvline(BAR, color="green", ls="--", lw=1.1)
        ax.axhline(c["clean_lift"], color="gray", ls=":", lw=1)
        ax.set_title(f"{c['cell']}\npred={c['predictor']:.3f} clean={c['clean_lift']:+.3f}",
                     fontsize=9)
        ax.set_xlabel("max(XGB, MLP) attribute AUC")
        ax.set_ylabel("task lift through channel")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.suptitle("Goal 3 — surgical (HSIC-subspace) vs blunt (isotropic) noise; "
                 "red ring = best full-battery-honest config", fontsize=10)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
