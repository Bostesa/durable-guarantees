"""Experiment 13 — the full CelebA pipeline on the two scan-picked vision cells.

Experiment 12 picked the two CelebA cells by measurement (label-only coupling):
  * Smiling    -> Young   predictor 0.522  — gauge prediction: durable removal CHEAP
  * Attractive -> Young   predictor 0.734  — prediction: cost ~1, output floor ~0.73
Both are existing PCRL purposes (smile_detection, attractiveness_prediction; both
list Young as a disallowed attribute). X is the frozen PCRL vision representation
(128-d per purpose) extracted by experiments/celeba_extract.py through PCRL's
trained celeba_v2 CNN encoder — the non-tabular modality check of the whole story.

Pipeline per cell (the standard protocol, unchanged from Exp 8-10):
  * clean baseline (trained clean channel P, own-head lift, attr recovery from P
    and from the OUTPUT logits);
  * the 2x2 {blunt, surgical} x {post-hoc, e2e} of Experiment 10, i.e. blunt noise
    sweep, post-hoc surgical (HSIC-subspace) noise, and the e2e shape-then-destroy
    channel — every winner vetted by the full Tier-1 battery (XGB + MLP + LoRA-32
    <= 0.55), e2e winners retrained over 3 seeds with rep AND output batteries;
  * TIER-2 add-on: the channel-aware Gaussian-LRT (utils/battery.py) on every
    winner, with a bounded sigma-escalation search for a Tier-2 operating point
    (blunt channels only — surgical channels have no noise in the complement, so
    the informed adversary is expected to read it; Exp-11/14);
  * the OUTPUT footprint measurement (attr from the task outputs) vs the predictor.

Public predictions under test (from the tabular ramp, stated before this run):
  1. Smiling/Young removal is CHEAP (predictor 0.522, just below the 0.55 wall).
  2. Attractive/Young costs ~1 with an output-leak floor near 0.73.
Any deviation of CelebA from the tabular patterns (footprint law, surgical-beats-
blunt, e2e-beats-post-hoc) is reported plainly — scope information, not a failure.

Local MPS/CPU; never AWS. macOS OpenMP guard: xgboost before torch. Never R^2.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import experiments.end_to_end_surgical as e2e  # noqa: E402
from experiments.diagnostic import (  # noqa: E402
    label_predictor, lift, load_cell, probe_sklearn,
)
from experiments.mi_ceiling import train_channel  # noqa: E402
from experiments.sealed_channel import train_model  # noqa: E402
from experiments.two_tier_certification import (  # noqa: E402
    fmt_point, full_point, posthoc_point,
)
from experiments.end_to_end_surgical import train_e2e_surgical  # noqa: E402
from utils.battery import (  # noqa: E402
    gaussian_lrt, gaussian_lrt_channel, iso_noise_cov, subspace_noise_cov,
)
from utils.pcrl_io import get_device  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
BAR = 0.55
SWEEP_SEEDS = [0, 1]
MAX_T2_ESCALATIONS = 3

CELLS = [
    dict(key="celeba Smiling->Young (low)", dataset="celeba", attr="Young",
         task="Smiling", iso_sigmas=[8.0, 16.0, 32.0],
         sub_grid=[(16, 8.0), (16, 16.0), (32, 16.0), (8, 16.0)]),
    dict(key="celeba Attractive->Young (high)", dataset="celeba", attr="Young",
         task="Attractive", iso_sigmas=[8.0, 16.0, 32.0],
         sub_grid=[(16, 8.0), (16, 16.0), (32, 16.0), (8, 16.0)]),
]
PRED_EXPECTED = {"celeba Smiling->Young (low)": 0.522,
                 "celeba Attractive->Young (high)": 0.734}


# --------------------------------------------------------------------------- #
#  Tier-2 add-on: channel-aware LRT on every arm winner (+ escalation)          #
# --------------------------------------------------------------------------- #
def tier2_addon(cell_cfg, c, device):
    """LRT on each arm's winner; bounded Tier-2 search on the blunt channels."""
    print(f"\n--- TIER-2 add-on: {c['cell']} ---", flush=True)
    X, attr, task, n_attr, n_task, _, task_maj = load_cell(
        cell_cfg["dataset"], cell_cfg["attr"], cell_cfg["task"])
    X_t = torch.from_numpy(X).to(device)
    # same clean trained P as run_cell (deterministic, seed 0)
    H, _, L0 = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                           alpha=0.0, device=device, pub_noise=0.0, seed=0)
    H_t = torch.from_numpy(H.astype(np.float32)).to(device)
    sigma_h = c["sigma_h"]
    Qcache = {}
    add = {}

    # clean OUTPUT leak (footprint ceiling context)
    xg0, _ = probe_sklearn("XGB", L0, attr, n_attr, SWEEP_SEEDS)
    add["clean_out_xgb"] = xg0
    print(f"  clean output XGB attr AUC = {xg0:.3f} (clean rep XGB = "
          f"{c['clean_xgb_auc']:.3f})", flush=True)

    # ---- A: post-hoc blunt winner + Tier-2 escalation ----
    bA = c["arms"]["A_posthoc_blunt"]["best"]
    if bA is not None:
        rows = []
        sr = bA["sigma_rel"]
        for _ in range(MAX_T2_ESCALATIONS + 1):
            pt = posthoc_point(f"iso s={sr:g}", "isotropic", None, sr, H, H_t,
                               attr, n_attr, task, task_maj, sigma_h, device, Qcache)
            rows.append(pt)
            if pt["tier2_max"] <= BAR:
                break
            sr *= 2
        add["A_rows"] = rows
        add["A_tier2"] = rows[-1] if rows[-1]["tier2_max"] <= BAR else None

    # ---- B: post-hoc surgical winner (+ sigma escalation, expected to fail) ----
    bB = c["arms"]["B_posthoc_surgical"]["best"]
    if bB is not None:
        rows = []
        r, sr = bB["rank"], bB["sigma_rel"]
        for _ in range(MAX_T2_ESCALATIONS + 1):
            pt = posthoc_point(f"sub r={r} s={sr:g}", "surgical", r, sr, H, H_t,
                               attr, n_attr, task, task_maj, sigma_h, device, Qcache)
            rows.append(pt)
            if pt["tier2_max"] <= BAR:
                break
            sr *= 2
        add["B_rows"] = rows
        add["B_tier2"] = rows[-1] if rows[-1]["tier2_max"] <= BAR else None

    # ---- C: e2e blunt winner + Tier-2 escalation ----
    bC = c["arms"]["C_e2e_blunt"]["best"]
    if bC is not None:
        rows = []
        sg = bC["sigma"]
        for _ in range(MAX_T2_ESCALATIONS + 1):
            h, P, L = train_channel(X_t, attr, task, n_attr, n_task, sg, device, seed=0)
            pt = full_point(h, P, attr, n_attr, iso_noise_cov(h.shape[1], sg),
                            device, SWEEP_SEEDS)
            pt["sigma"] = sg
            pt["lift"] = lift(L, task, task_maj)
            rows.append(pt)
            print(f"  e2e-iso σ={sg:g} {fmt_point(pt)} lift={pt['lift']:+.4f}",
                  flush=True)
            if pt["tier2_max"] <= BAR:
                break
            sg *= 2
        add["C_rows"] = rows
        add["C_tier2"] = rows[-1] if rows[-1]["tier2_max"] <= BAR else None

    # ---- D: e2e surgical winner + sigma escalation ----
    bD = c["arms"]["D_e2e_surgical"]["best"]
    if bD is not None:
        rows = []
        sg = bD["sigma"]
        for _ in range(MAX_T2_ESCALATIONS + 1):
            P, L, h, Q = train_e2e_surgical(
                X_t, attr, task, n_attr, n_task, rank=bD["rank"], sigma=sg,
                lam=bD["lam"], device=device, seed=0, return_clean=True)
            pt = full_point(h, P, attr, n_attr, subspace_noise_cov(Q, sg), device,
                            SWEEP_SEEDS)
            pt["sigma"] = sg
            pt["lift"] = lift(L, task, task_maj)
            rows.append(pt)
            print(f"  e2e-sur r={bD['rank']} σ={sg:g} {fmt_point(pt)} "
                  f"lift={pt['lift']:+.4f}", flush=True)
            if pt["tier2_max"] <= BAR:
                break
            sg *= 2
        add["D_rows"] = rows
        add["D_tier2"] = rows[-1] if rows[-1]["tier2_max"] <= BAR else None

    for arm in ("A", "B", "C", "D"):
        rows = add.get(f"{arm}_rows")
        if rows is None:
            continue
        first, t2 = rows[0], add.get(f"{arm}_tier2")
        print(f"  >>> arm {arm}: winner LRT={first['lrt']:.3f} → Tier-2 "
              f"{'HOLDS at winner' if first['tier2_max'] <= BAR else 'breaches'}"
              + ("" if first["tier2_max"] <= BAR else
                 (f"; Tier-2 point at σ-esc {t2.get('sigma', t2.get('sigma_rel')):g} "
                  f"(lift {t2['lift']:+.4f})" if t2 else
                  "; NO Tier-2 point within escalation budget")), flush=True)
    return add


# --------------------------------------------------------------------------- #
#  Prediction verdicts + footprint                                             #
# --------------------------------------------------------------------------- #
def verdicts(c, add, predictor):
    """Durable-removal cost + output floor at the best Tier-1 winner."""
    out = {}
    # best utility kept across arms at Tier-1 (the protocol the predictions used)
    kept = {}
    for name, arm in c["arms"].items():
        b, fin = arm["best"], arm.get("final")
        if b is None:
            continue
        l = fin["lift_mean"] if fin else b["lift"]
        kept[name] = 100.0 * l / max(arm["clean_lift"], 1e-9)
    out["kept_pct_by_arm"] = kept
    out["best_arm"] = max(kept, key=kept.get) if kept else None
    out["best_kept_pct"] = kept.get(out["best_arm"]) if kept else None
    out["rep_cost"] = (None if out["best_kept_pct"] is None
                       else max(0.0, 1.0 - out["best_kept_pct"] / 100.0))
    # output floor at the winners (rep hidden -> what the logits still leak)
    floors = {}
    for name, arm in c["arms"].items():
        b, fin = arm["best"], arm.get("final")
        om = (fin["out_max_mean"] if fin else (b or {}).get("output", {}).get("out_max"))
        if om is not None:
            floors[name] = om
    out["out_floor_by_arm"] = floors
    out["out_floor"] = max(floors.values()) if floors else None   # law: floor of a USEFUL model
    out["out_floor_best_arm"] = floors.get(out["best_arm"])
    out["predictor"] = predictor
    return out


def main():
    device = get_device()
    print(f"[setup] device={device} bar≤{BAR} — Experiment 13, CelebA pipeline "
          f"(X = frozen PCRL vision reps, 128-d)", flush=True)
    results = []
    for cfg in CELLS:
        t0 = time.time()
        X, attr, task, n_attr, n_task, attr_maj, task_maj = load_cell(
            cfg["dataset"], cfg["attr"], cfg["task"])
        pred, pred_std = label_predictor(attr, task, n_attr)
        print(f"\n[{cfg['key']}] N={len(attr)} predictor={pred:.4f}±{pred_std:.3f} "
              f"(scan said {PRED_EXPECTED[cfg['key']]}) attr_maj={attr_maj:.3f} "
              f"task_maj={task_maj:.3f}", flush=True)
        c = e2e.run_cell(cfg, device)             # the full Exp-10 2x2, Tier-1 vetted
        add = tier2_addon(cfg, c, device)
        v = verdicts(c, add, pred)
        results.append({"cfg_key": cfg["key"], "predictor": pred,
                        "predictor_std": pred_std, "cell": c, "tier2": add,
                        "verdicts": v, "minutes": round((time.time() - t0) / 60, 1)})
        print(f"[{cfg['key']}] done in {results[-1]['minutes']} min", flush=True)

    # ---------------- final summary ----------------
    print("\n" + "=" * 104)
    print("EXPERIMENT 13 — CelebA: gauge predictions vs measurement")
    print("=" * 104)
    for r in results:
        v = r["verdicts"]
        print(f"\n{r['cfg_key']}  predictor={r['predictor']:.3f}")
        for name, k in v["kept_pct_by_arm"].items():
            fl = v["out_floor_by_arm"].get(name)
            print(f"  {name:<22} utility kept {k:6.1f}%   out_max "
                  f"{'—' if fl is None else f'{fl:.3f}'}")
        print(f"  → best Tier-1 arm: {v['best_arm']} ({v['best_kept_pct']:.1f}% kept, "
              f"rep-channel cost {v['rep_cost']:.2f}); output floor at winners: "
              f"{v['out_floor']}")

    (RESULTS_DIR / "celeba_pipeline.json").write_text(json.dumps(
        {"experiment": "celeba_pipeline", "bar": BAR, "cells": results}, indent=2,
        default=float))
    _plot(results, RESULTS_DIR / "celeba_pipeline.png")
    print(f"\n[save] {RESULTS_DIR / 'celeba_pipeline.json'} and .png", flush=True)


def _plot(results, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["A_posthoc_blunt", "B_posthoc_surgical", "C_e2e_blunt", "D_e2e_surgical"]
    short = ["post-hoc\nblunt", "post-hoc\nsurgical", "e2e\nblunt", "e2e\nsurgical"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, r in zip(axes, results):
        v = r["verdicts"]
        kept = [v["kept_pct_by_arm"].get(n, 0.0) for n in names]
        ax.bar(np.arange(4), kept, color=["C0", "C1", "C2", "C3"])
        for xi, k in enumerate(kept):
            ax.text(xi, k + 1.5, f"{k:.0f}%", ha="center", fontsize=8)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(short, fontsize=8)
        ax.set_ylim(0, 115)
        ax.set_ylabel("utility kept at Tier-1 ≤0.55 (%)")
        ax.set_title(f"{r['cfg_key']}\npredictor={r['predictor']:.3f}, "
                     f"output floor={v['out_floor']}", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Experiment 13 — CelebA vision cells: the 2×2 at Tier-1, "
                 "with output footprints", fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
