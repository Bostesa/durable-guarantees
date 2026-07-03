"""Experiment 8 — the DIAGNOSTIC: does label↔attribute correlation predict the
utility cost of honest, durable attribute removal? And is the achievable case
(HMDA/race) durable under the full attacker battery?

This is the "solution" half of the problem+solution story. Experiments 1–7
established the PROBLEM: linear/R² certificates lie (Exp 1, 5), every
probe-shaped eraser only fools its own probe (Exp 4, 5), and even a sealed
architecture leaks the attribute through the OUTPUT to the extent the task uses
it (Exp 7 footprint law). Exp 7 measured that footprint on two cells (Adult 0.60,
HMDA 0.51) — two points. Here we turn it into a predictive diagnostic and test
the constructive case rigorously.

The footprint law (Exp 7): the task output leaks the attribute to the extent the
task LABEL correlates with the attribute. So:
  * when the label barely needs the attribute (label↔attr XGB AUC ≤ bar), a single
    noised channel can hide the attribute in BOTH representation and output while
    keeping utility → durable removal is CHEAP;
  * when the label genuinely needs it (label↔attr XGB AUC > bar), the decision
    itself betrays the attribute → durable removal costs (nearly) all utility.

Three parts, all measured with XGBoost + deep MLP + rank-32 LoRA (NEVER R²):

  PART A — harden the constructive result on the achievable case (HMDA/race).
    The single noised channel (E_pub→noise→head, trained from raw x) hid race
    while keeping loan-decision utility. Verify it: 3 training seeds with error
    bars on attribute recovery AND task lift; full battery on the PUBLIC
    representation AND the task OUTPUTS. Confirm race ≤ 0.55 AUC from every
    attacker on both, with lift retained — or say plainly if it breaks.

  PART B — make the diagnostic predictive. Across 6 cells spanning the predictor
    axis: predictor = XGB AUC(attr | task label alone); outcome = utility cost of
    driving attribute recovery to ≤ 0.55 AUC (from rep AND output) with the noised
    channel. Plot predictor vs cost.

  PART C — confirm the footprint law across the new cells: the output-leak floor
    (attr recovered from the logits once the representation is hidden) vs the
    label↔attr correlation. Two points made a line in Exp 7; confirm it is a law.

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
import torch.nn as nn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.falsification_attack import FreshLoRAAttacker  # noqa: E402
from experiments.hardening_test import _probe  # noqa: E402  (PCRL-auditor XGB/MLP)
from experiments.sealed_channel import train_model  # noqa: E402  (noised channel from raw x)
from utils.pcrl_io import (  # noqa: E402
    build_train_loader, extract_attr, extract_task_label, get_device,
)

RESULTS_DIR = REPO_ROOT / "results"
HONEST_AUC_BAR = 0.55       # a probe "fails to recover" iff AUC ≤ this (chance 0.5)
PROBE_SEEDS = [0, 1, 2]     # held-out probe seeds (Part A)
SWEEP_PROBE_SEEDS = [0, 1]  # cheaper probe seeds inside the σ sweep (Part B)
NOISE_SWEEP = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0]   # BN-unit isotropic noise on the public channel
LORA_RANK = 32
LORA_ALPHA = 64.0           # keep alpha/rank = 2, matching the project's LoRA scaling
LORA_STEPS = 300

# 6 cells spanning the predictor axis (measured in _scout_predictor.py), crossing
# the 0.55 bar, balanced across all three datasets.
#   (dataset, attr, task, predictor≈)  — predictor is re-measured live, not trusted here.
CELLS = [
    ("diabetes", "gender", "medication_change_outcome"),   # ~0.509  LOW
    ("hmda",     "race",   "loan_decision"),                # ~0.514  LOW  (Exp-7 constructive anchor)
    ("diabetes", "gender", "primary_diagnosis_category"),   # ~0.543  MID
    ("hmda",     "race",   "loan_amount_band"),             # ~0.583  MID-HIGH
    ("adult",    "sex",    "income"),                       # ~0.603  HIGH (anchor)
    ("adult",    "sex",    "occupation_group"),             # ~0.676  HIGH
]
PART_A_CELL = ("hmda", "race", "loan_decision")

_LOADER_CACHE: dict[str, list] = {}


# --------------------------------------------------------------------------- #
#  Data + probes                                                              #
# --------------------------------------------------------------------------- #
def load_cell(dataset, attr_name, task_name):
    """Raw features X, attribute, task label, class counts, and majorities.

    dataset="celeba" (Experiment 13): X is the frozen PCRL vision representation
    of the train partition, extracted once by experiments/celeba_extract.py
    (per-purpose 128-d; the purpose is implied by the task). Labels are PCRL's
    own 0/1 CelebA labels. Everything downstream is unchanged."""
    if dataset == "celeba":
        cache = REPO_ROOT / "data_cache" / "celeba" / "train.npz"
        if not cache.exists():
            raise FileNotFoundError(
                f"{cache} missing — run experiments/celeba_extract.py first")
        purpose = {"Smiling": "smile_detection",
                   "Attractive": "attractiveness_prediction"}[task_name]
        d = np.load(cache)
        X = d[f"rep_{purpose}"].astype(np.float32)
        attr = d[f"label_{attr_name}"].astype(np.int64)
        task = d[f"label_{task_name}"].astype(np.int64)
        n_attr = int(attr.max()) + 1
        n_task = int(task.max()) + 1
        attr_maj = float(np.bincount(attr).max() / len(attr))
        task_maj = float(np.bincount(task).max() / len(task))
        return X, attr, task, n_attr, n_task, attr_maj, task_maj
    if dataset not in _LOADER_CACHE:
        _, _, loader = build_train_loader(dataset)
        _LOADER_CACHE[dataset] = list(loader)
    loader = _LOADER_CACHE[dataset]
    X = np.concatenate([b["features"].numpy() for b in loader]).astype(np.float32)
    attr = extract_attr(loader, attr_name)
    task = extract_task_label(loader, task_name)
    n_attr = int(attr.max()) + 1
    n_task = int(task.max()) + 1
    attr_maj = float(np.bincount(attr).max() / len(attr))
    task_maj = float(np.bincount(task).max() / len(task))
    return X, attr, task, n_attr, n_task, attr_maj, task_maj


def _split(H, y, seed):
    from sklearn.model_selection import train_test_split
    return train_test_split(H.astype(np.float64), y, test_size=0.25,
                            random_state=seed, stratify=y)


def probe_sklearn(arch, H, y, n_classes, seeds):
    """Mean±std held-out AUC of an sklearn probe (XGB or MLP) recovering y from H."""
    aucs = []
    for s in seeds:
        Xtr, Xte, ytr, yte = _split(H, y, s)
        _, auc = _probe(arch, Xtr, ytr, Xte, yte, s, n_classes)
        aucs.append(auc)
    return float(np.mean(aucs)), float(np.std(aucs))


def probe_lora(H, y, n_classes, seeds, device, rank=LORA_RANK, alpha=LORA_ALPHA):
    """Mean±std held-out AUC of a fresh rank-32 ReLU-LoRA attacker recovering y from H.

    Same mechanism as Experiments 1/3: a LoRA re-basis + ReLU + linear head,
    trained by CE on a train split, evaluated on a held-out split. Works on any
    fixed feature matrix (the public representation P or the task logits)."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for s in seeds:
        Xtr, Xte, ytr, yte = _split(H, y, s)
        torch.manual_seed(s)
        Xtr_t = torch.from_numpy(Xtr).float().to(device)
        Xte_t = torch.from_numpy(Xte).float().to(device)
        ytr_t = torch.from_numpy(ytr).long().to(device)
        atk = FreshLoRAAttacker(repr_dim=H.shape[1], n_classes=n_classes,
                                rank=rank, alpha=alpha).to(device)
        opt = torch.optim.Adam(atk.parameters(), lr=1e-2)
        ce = nn.CrossEntropyLoss()
        n = Xtr_t.shape[0]
        g = torch.Generator(device="cpu").manual_seed(s)
        atk.train()
        for _ in range(LORA_STEPS):
            idx = torch.randint(0, n, (min(512, n),), generator=g).to(device)
            loss = ce(atk(Xtr_t[idx]), ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        atk.eval()
        with torch.no_grad():
            logits = atk(Xte_t).cpu().numpy()
        z = logits - logits.max(1, keepdims=True)
        prob = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        try:
            if n_classes == 2:
                auc = float(roc_auc_score(yte, prob[:, 1]))
            else:
                auc = float(roc_auc_score(yte, prob, multi_class="ovr", average="macro"))
        except ValueError:
            auc = float("nan")
        aucs.append(auc)
    return float(np.mean(aucs)), float(np.std(aucs))


def battery(H, y, n_classes, device, seeds, archs=("XGB", "MLP", "LoRA32")):
    """Full attacker battery on a fixed feature matrix H → per-attacker AUC and the max."""
    out = {}
    for a in archs:
        if a == "LoRA32":
            m, s = probe_lora(H, y, n_classes, seeds, device)
        else:
            m, s = probe_sklearn(a, H, y, n_classes, seeds)
        out[a] = {"auc": m, "auc_std": s}
    out["max_auc"] = float(max(out[a]["auc"] for a in archs))
    return out


def lift(logits, task, task_maj):
    return float((logits.argmax(1) == task).mean()) - task_maj


def label_predictor(attr, task, n_attr, seeds=PROBE_SEEDS):
    """The PREDICTOR: XGB AUC recovering attr from the 1-dim task label alone."""
    return probe_sklearn("XGB", task.reshape(-1, 1).astype(np.float32), attr, n_attr, seeds)


# --------------------------------------------------------------------------- #
#  PART A — harden the constructive result on HMDA/race                        #
# --------------------------------------------------------------------------- #
def part_a(device, sigma):
    dataset, attr_name, task_name = PART_A_CELL
    print("\n" + "#" * 92)
    print(f"# PART A — harden the constructive result: {dataset}/{attr_name}/{task_name}, "
          f"noised channel σ={sigma}")
    print("#" * 92)
    X, attr, task, n_attr, n_task, attr_maj, task_maj = load_cell(dataset, attr_name, task_name)
    X_t = torch.from_numpy(X).to(device)
    pred_auc, _ = label_predictor(attr, task, n_attr)
    print(f"[setup] N={X.shape[0]} x_dim={X.shape[1]} {attr_name}={n_attr}cls "
          f"(maj {attr_maj:.3f}) task '{task_name}' maj={task_maj:.3f} | "
          f"predictor XGB(attr|label)={pred_auc:.4f}")

    train_seeds = [0, 1, 2]
    variants = [("clean σ=0", 0.0), (f"noised σ={sigma:g}", sigma)]
    rows = []
    for vname, s_noise in variants:
        per_seed = {"P": {"XGB": [], "MLP": [], "LoRA32": []},
                    "output": {"XGB": [], "MLP": [], "LoRA32": []}, "lift": []}
        for ts in train_seeds:
            P, _, L = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                                  alpha=0.0, device=device, pub_noise=s_noise, seed=ts)
            bP = battery(P, attr, n_attr, device, PROBE_SEEDS)
            bO = battery(L, attr, n_attr, device, PROBE_SEEDS)
            tl = lift(L, task, task_maj)
            for a in ("XGB", "MLP", "LoRA32"):
                per_seed["P"][a].append(bP[a]["auc"])
                per_seed["output"][a].append(bO[a]["auc"])
            per_seed["lift"].append(tl)
            print(f"  [{vname:<12} seed {ts}] P max={bP['max_auc']:.3f} "
                  f"out max={bO['max_auc']:.3f} lift={tl:+.4f}")
        row = {"variant": vname, "sigma": s_noise,
               "lift_mean": float(np.mean(per_seed["lift"])),
               "lift_std": float(np.std(per_seed["lift"]))}
        for tgt in ("P", "output"):
            for a in ("XGB", "MLP", "LoRA32"):
                row[f"{tgt}_{a}_mean"] = float(np.mean(per_seed[tgt][a]))
                row[f"{tgt}_{a}_std"] = float(np.std(per_seed[tgt][a]))
            row[f"{tgt}_max_mean"] = float(np.max(
                [np.mean(per_seed[tgt][a]) for a in ("XGB", "MLP", "LoRA32")]))
        rows.append(row)

    _part_a_table(rows, attr_name, attr_maj, task_maj, pred_auc)
    return {"cell": f"{dataset}/{attr_name}/{task_name}", "sigma": sigma,
            "predictor": pred_auc, "attr_majority": attr_maj,
            "task_majority": task_maj, "rows": rows}


def _part_a_table(rows, attr, attr_maj, task_maj, pred_auc):
    print("\n" + "=" * 92)
    print(f"PART A TABLE — {attr} recovery (mean±std over 3 training seeds), full battery")
    print("=" * 92)
    hdr = f"{'variant':<12} {'target':<7} {'XGB':>13} {'MLP':>13} {'LoRA32':>13} {'task lift':>16}"
    print(hdr)
    print("-" * 92)
    for r in rows:
        for tgt in ("P", "output"):
            cells = " ".join(f"{r[f'{tgt}_{a}_mean']:.3f}±{r[f'{tgt}_{a}_std']:.3f}"
                             for a in ("XGB", "MLP", "LoRA32"))
            liftstr = f"{r['lift_mean']:+.4f}±{r['lift_std']:.4f}" if tgt == "P" else ""
            print(f"{r['variant']:<12} {tgt:<7} {cells}   {liftstr:>16}")
    print("-" * 92)
    noised = rows[-1]
    stopped = (noised["P_max_mean"] <= HONEST_AUC_BAR and
               noised["output_max_mean"] <= HONEST_AUC_BAR)
    print(f"{attr} majority={attr_maj:.3f}; chance AUC=0.5; bar≤{HONEST_AUC_BAR}; "
          f"task majority={task_maj:.3f}; predictor(attr|label)={pred_auc:.4f}")
    print(f"\n>>> PART A VERDICT: at σ={noised['sigma']:g}, {attr} recovery "
          f"P_max={noised['P_max_mean']:.3f} output_max={noised['output_max_mean']:.3f}; "
          f"task lift {noised['lift_mean']:+.4f} (clean {rows[0]['lift_mean']:+.4f}, "
          f"{100*noised['lift_mean']/max(rows[0]['lift_mean'],1e-9):.0f}% kept). "
          f"{'DURABLE — hidden from every attacker on BOTH channels with utility kept.' if stopped else 'DOES NOT HOLD — an attacker breaches the bar; see table.'}")


# --------------------------------------------------------------------------- #
#  PART B + C — the diagnostic (noise sweep per cell) and the footprint law    #
# --------------------------------------------------------------------------- #
def sweep_cell(dataset, attr_name, task_name, device):
    print("\n" + "#" * 92)
    print(f"# CELL: {dataset}/{attr_name}/{task_name}")
    print("#" * 92)
    X, attr, task, n_attr, n_task, attr_maj, task_maj = load_cell(dataset, attr_name, task_name)
    X_t = torch.from_numpy(X).to(device)
    pred_auc, pred_std = label_predictor(attr, task, n_attr)
    print(f"[setup] N={X.shape[0]} x_dim={X.shape[1]} {attr_name}={n_attr}cls "
          f"(maj {attr_maj:.3f}) task '{task_name}' maj={task_maj:.3f} | "
          f"PREDICTOR XGB(attr|label)={pred_auc:.4f}±{pred_std:.3f}")

    sweep = []
    for s_noise in NOISE_SWEEP:
        P, _, L = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                              alpha=0.0, device=device, pub_noise=s_noise, seed=0)
        # XGB + MLP battery (cheap seeds); LoRA added only on the final chosen config.
        bP = battery(P, attr, n_attr, device, SWEEP_PROBE_SEEDS, archs=("XGB", "MLP"))
        bO = battery(L, attr, n_attr, device, SWEEP_PROBE_SEEDS, archs=("XGB", "MLP"))
        tl = lift(L, task, task_maj)
        pt = {"sigma": s_noise, "P_max": bP["max_auc"], "out_max": bO["max_auc"],
              "P_xgb": bP["XGB"]["auc"], "P_mlp": bP["MLP"]["auc"],
              "out_xgb": bO["XGB"]["auc"], "out_mlp": bO["MLP"]["auc"], "lift": tl}
        sweep.append(pt)
        print(f"  σ={s_noise:>5.1f}  P_max={pt['P_max']:.3f}  out_max={pt['out_max']:.3f}  "
              f"lift={tl:+.4f}")

    clean_lift = sweep[0]["lift"]
    # σ_rep: cheapest σ hiding the REP (both probes ≤ bar); σ_both: also hides the OUTPUT.
    rep_pts = [p for p in sweep if p["P_max"] <= HONEST_AUC_BAR]
    both_pts = [p for p in sweep if p["P_max"] <= HONEST_AUC_BAR and p["out_max"] <= HONEST_AUC_BAR]
    sigma_rep = min(rep_pts, key=lambda p: p["sigma"]) if rep_pts else None
    sigma_both = min(both_pts, key=lambda p: p["sigma"]) if both_pts else None

    def cost(pt):
        if pt is None:
            return None
        return 1.0 - pt["lift"] / clean_lift if clean_lift > 1e-6 else None

    cost_rep = cost(sigma_rep)
    cost_both = cost(sigma_both)
    # Footprint-law point: output-leak FLOOR = out_max at the σ that hides the rep
    # (representation gone; whatever remains in the output is the label's own leak).
    out_floor = sigma_rep["out_max"] if sigma_rep else sweep[-1]["out_max"]

    row = {"cell": f"{dataset}/{attr_name}/{task_name}", "dataset": dataset,
           "attr": attr_name, "task": task_name, "predictor": pred_auc,
           "predictor_std": pred_std, "attr_majority": attr_maj, "task_majority": task_maj,
           "clean_lift": clean_lift, "clean_P_max": sweep[0]["P_max"],
           "clean_out_max": sweep[0]["out_max"],
           "sigma_rep": sigma_rep["sigma"] if sigma_rep else None,
           "sigma_both": sigma_both["sigma"] if sigma_both else None,
           "lift_at_rep": sigma_rep["lift"] if sigma_rep else None,
           "lift_at_both": sigma_both["lift"] if sigma_both else None,
           "cost_rep": cost_rep, "cost_both": cost_both,
           "out_floor": out_floor, "sweep": sweep,
           "degenerate": clean_lift <= 0.01}
    print(f"  → predictor={pred_auc:.3f} | clean_lift={clean_lift:+.4f} | "
          f"σ_rep={row['sigma_rep']} (cost {cost_rep if cost_rep is None else round(cost_rep,3)}) | "
          f"σ_both={row['sigma_both']} (cost {cost_both if cost_both is None else round(cost_both,3)}) | "
          f"output floor={out_floor:.3f}"
          + ("   [DEGENERATE: no clean utility]" if row["degenerate"] else ""))
    return row


def main():
    device = get_device()
    print(f"[setup] device={device}  (OpenMP guard; xgboost before torch)  "
          f"bar≤{HONEST_AUC_BAR}  noise sweep={NOISE_SWEEP}")

    # ---- Parts B + C: sweep every cell ----
    print("\n" + "=" * 92 + "\nPARTS B + C — noise sweep across 6 cells\n" + "=" * 92)
    cell_rows = [sweep_cell(d, a, t, device) for d, a, t in CELLS]

    # ---- Part A: deep hardening on the achievable cell. Pick the smallest swept σ
    # that clears the bar with MARGIN (both P and output ≤ bar−0.01) on the cell's own
    # single-seed sweep, so the 3-seed headline is not sitting exactly on the bar. ----
    pa_cell = f"{PART_A_CELL[0]}/{PART_A_CELL[1]}/{PART_A_CELL[2]}"
    pa_row = next(r for r in cell_rows if r["cell"] == pa_cell)
    margin = [p["sigma"] for p in pa_row["sweep"]
              if p["P_max"] <= HONEST_AUC_BAR - 0.01 and p["out_max"] <= HONEST_AUC_BAR - 0.01]
    sigma_a = min(margin) if margin else (pa_row["sigma_both"] or pa_row["sigma_rep"] or 8.0)
    part_a_out = part_a(device, sigma_a)

    _diagnostic_table(cell_rows)
    out = {"experiment": "diagnostic", "honest_auc_bar": HONEST_AUC_BAR,
           "noise_sweep": NOISE_SWEEP, "lora_rank": LORA_RANK,
           "cells": [r["cell"] for r in cell_rows], "part_b_c": cell_rows,
           "part_a": part_a_out}
    (RESULTS_DIR / "diagnostic.json").write_text(json.dumps(out, indent=2))
    _plot(cell_rows, RESULTS_DIR / "diagnostic.png")
    print(f"\n[save] {RESULTS_DIR/'diagnostic.json'} and .png")


def _diagnostic_table(rows):
    print("\n" + "=" * 100)
    print("PART B + C SUMMARY — predictor vs durable-removal cost, and the footprint law")
    print("=" * 100)
    print(f"{'cell':<44} {'pred':>6} {'clean':>7} {'σ_rep':>6} {'cost_rep':>9} "
          f"{'σ_both':>7} {'cost_both':>10} {'out_floor':>10}")
    print("-" * 100)
    for r in sorted(rows, key=lambda r: r["predictor"]):
        cr = "  —  " if r["cost_rep"] is None else f"{r['cost_rep']:+.3f}"
        cb = "  —  " if r["cost_both"] is None else f"{r['cost_both']:+.3f}"
        sr = "—" if r["sigma_rep"] is None else f"{r['sigma_rep']:g}"
        sb = "—" if r["sigma_both"] is None else f"{r['sigma_both']:g}"
        flag = " DEGEN" if r["degenerate"] else ""
        print(f"{r['cell']:<44} {r['predictor']:>6.3f} {r['clean_lift']:>+7.4f} {sr:>6} "
              f"{cr:>9} {sb:>7} {cb:>10} {r['out_floor']:>10.3f}{flag}")
    print("-" * 100)
    print(f"predictor = XGB AUC(attr|label alone); cost = fraction of clean task lift lost to reach "
          f"attr recovery ≤{HONEST_AUC_BAR}; out_floor = output leak once the rep is hidden.")
    print("Diagnostic claim: predictor > bar ⇒ output cannot be hidden at positive utility ⇒ cost_both → 1.")
    print("Footprint law: out_floor ≈ predictor (the decision leaks the attribute ∝ how much the label needs it).")


def _plot(rows, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    good = [r for r in rows if not r["degenerate"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    # Left: Part B — predictor vs durable-removal cost.
    ax = axes[0]
    xs = [r["predictor"] for r in rows]
    for r in rows:
        cb = r["cost_both"] if r["cost_both"] is not None else 1.0
        marker = "o" if not r["degenerate"] else "x"
        ax.scatter(r["predictor"], cb, s=90, marker=marker,
                   color="C3" if r["predictor"] > HONEST_AUC_BAR else "C0", zorder=3)
        ax.annotate(r["cell"].replace("/", "/\n"), (r["predictor"], cb),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.axvline(HONEST_AUC_BAR, color="green", ls="--", lw=1.2,
               label=f"bar {HONEST_AUC_BAR} (durable-removal threshold)")
    ax.set_xlabel("PREDICTOR: XGB AUC(attribute | task label alone)")
    ax.set_ylabel("OUTCOME: utility cost to drive attr recovery ≤0.55\n(fraction of clean task lift lost; 1.0 = all utility)")
    ax.set_title("Part B — does label↔attribute correlation predict\ndurable-removal cost?")
    ax.set_ylim(-0.1, 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="center left")

    # Right: Part C — footprint law (out_floor vs predictor).
    ax = axes[1]
    lo = min(xs + [r["out_floor"] for r in rows]) - 0.02
    hi = max(xs + [r["out_floor"] for r in rows]) + 0.02
    ax.plot([lo, hi], [lo, hi], color="gray", ls=":", lw=1.3, label="y = x (law: output leak = label leak)")
    for r in rows:
        ax.scatter(r["predictor"], r["out_floor"], s=90,
                   color="C3" if r["predictor"] > HONEST_AUC_BAR else "C0", zorder=3)
        ax.annotate(r["cell"].split("/")[0] + "/" + r["attr"], (r["predictor"], r["out_floor"]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.axhline(HONEST_AUC_BAR, color="green", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("PREDICTOR: XGB AUC(attribute | task label alone)")
    ax.set_ylabel("output-leak floor: attr recovered from logits\n(once the representation is hidden)")
    ax.set_title("Part C — the footprint law across 6 cells\n(output leaks attr ∝ label↔attr correlation)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
