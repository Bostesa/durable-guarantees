"""Expansion Phase 1 — the full pipeline on the cells auto-selected by
expansion_coupling_scan.py (results/expansion_coupling_scan.json).

Per cell, three measurements, all with the EXISTING instruments (imported,
never reimplemented, never modified):

  COST    the continuous-cost sweep of continuous_cost.measure_cell — clean
          model = sealed_channel.train_model (the existing tabular
          architecture/protocol), adaptive sigma grid, durable-removal cost
          at the 0.55 bar (max(XGB,MLP) crossing, linear interpolation).
  FLOOR   the output-leak floor: best-attacker AUC (XGB + MLP) on the task
          OUTPUTS at the first swept sigma that hides the representation
          (P_max <= 0.55) — the Fig-2 quantity — re-measured with 3 probe
          seeds (75/25 stratified), plus the clean model's output leak.
  REAUDIT the standard removal configurations on the clean model's frozen
          representation P — noise sigma_rel in {1,2,8} (x per-dim std) and
          LEOPARD-MMD / HSIC projections at ranks {1,8,16} — each measured
          with BOTH the linear certificate (noise_channel_test.run_attack:
          LoRA-attacked linear R^2, verdict stopped iff R^2 <= tau=0.05) and
          the two-tier attacker suite (Tier 1 XGB+MLP+LoRA-32; Tier 2
          + channel-aware Gaussian-LRT). honest_stopped keeps the existing
          re-audit convention: max(XGB, MLP) <= 0.55.

Cells larger than MAX_N are subsampled (deterministic, stratified on
(label, attr), seed 42) for runtime parity with the existing tabular cells
(~36k-80k rows); n_used is recorded in provenance.

Sharded per cell (--cell <source/attr/task>) so cells run as parallel
processes; --merge assembles the three NEW results files:
  results/expansion_floor.json, expansion_cost.json, expansion_reaudit.json.
Existing results JSONs are never touched. Local MPS/CPU; never AWS.
macOS OpenMP guard: xgboost before torch.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: F401,E402  (import before torch on purpose)

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.continuous_cost import BAR, SIGMA_GRID, measure_cell  # noqa: E402
from experiments.diagnostic import (  # noqa: E402
    PROBE_SEEDS, battery, probe_lora, probe_sklearn,
)
from experiments.end_to_end_surgical import lora_channel  # noqa: E402
from experiments.noise_channel_test import TAU, run_attack, task_accuracy  # noqa: E402
from experiments.sealed_channel import train_model  # noqa: E402
from experiments.smart_erasure import apply_projection, fit_projection  # noqa: E402
from experiments.targeted_noise import iso_channel, probe_through  # noqa: E402
from utils.battery import gaussian_lrt, gaussian_lrt_channel, iso_noise_cov  # noqa: E402
from utils.expansion_io import cell_provenance, load_expansion_cell  # noqa: E402
from utils.folktables_io import load_folktables_cell  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
SHARD_DIR = RESULTS_DIR / "expansion_shards"
SCAN_PATH = RESULTS_DIR / "expansion_coupling_scan.json"

MAX_N = 60_000                    # runtime-parity cap (existing cells ~36k-80k)
REAUDIT_SEEDS = [0, 1, 2]         # cert + suite seeds (honest_reaudit convention)
NOISE_SIGMA_RELS = [1.0, 2.0, 8.0]   # the E2/E4-Stage-1 standard noise configs
PROJ_RANKS = [1, 8, 16]              # standard projection configs per family
HONEST_AUC_BAR = 0.55


def load_cell_by_tag(tag: str):
    source, attr_name, task_name = tag.split("/")
    if source == "folktables":
        loaded = load_folktables_cell(task_name, attr_name, max_n=MAX_N)
    else:
        loaded = load_expansion_cell(source, attr_name, task_name, max_n=MAX_N)
    return (source, attr_name, task_name) + loaded


def provenance_for(source, n_used, runtime_s):
    return {**cell_provenance(source),
            "n_used": int(n_used), "max_n_cap": MAX_N,
            "subsample": "deterministic stratified on (label, attr), seed 42, "
                         "applied only if train partition > max_n_cap",
            "train_seed": 0,
            "probe_seeds": {"cost_sweep": [0, 1], "floor": PROBE_SEEDS,
                            "reaudit": REAUDIT_SEEDS},
            "probe_split": "75/25 stratified held-out (diagnostic._split)",
            "runtime_seconds": round(runtime_s, 1)}


# --------------------------------------------------------------------------- #
#  FLOOR — 3-seed output-leak floor at the rep-hiding sigma                    #
# --------------------------------------------------------------------------- #
def measure_floor(cost_row, X_t, attr, task, n_attr, n_task, task_maj, device):
    sweep = cost_row["sweep"]
    hid = [p for p in sweep if p["P_max"] <= BAR]
    sigma_floor = hid[0]["sigma"] if hid else sweep[-1]["sigma"]
    rep_hidden = bool(hid)

    out = {}
    for label, sigma in (("clean", 0.0), ("at_sigma_rep", sigma_floor)):
        P, _, L = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                              alpha=0.0, device=device, pub_noise=sigma, seed=0)
        bO = battery(L, attr, n_attr, device, PROBE_SEEDS, archs=("XGB", "MLP"))
        bP = battery(P, attr, n_attr, device, PROBE_SEEDS, archs=("XGB", "MLP"))
        acc = float((L.argmax(1) == task).mean())
        out[label] = {"sigma": sigma, "out_xgb": bO["XGB"]["auc"],
                      "out_xgb_std": bO["XGB"]["auc_std"],
                      "out_mlp": bO["MLP"]["auc"], "out_mlp_std": bO["MLP"]["auc_std"],
                      "out_max": bO["max_auc"], "P_max": bP["max_auc"],
                      "task_acc": acc, "task_lift": acc - task_maj}
        if sigma == 0.0:
            clean_P = P  # reused by the re-audit
        print(f"  [floor {label}] σ={sigma:g} out_max={out[label]['out_max']:.3f} "
              f"P_max={out[label]['P_max']:.3f} lift={out[label]['task_lift']:+.4f}",
              flush=True)
        if sigma_floor == 0.0:   # rep already hidden clean — one model suffices
            out["at_sigma_rep"] = out["clean"]
            break

    floor = {"sigma_rep_grid": sigma_floor, "rep_hidden_within_grid": rep_hidden,
             "floor": out["at_sigma_rep"]["out_max"],
             "floor_seeds": PROBE_SEEDS, "clean": out["clean"],
             "at_sigma_rep": out["at_sigma_rep"],
             "task_well_learned": bool(out["clean"]["task_lift"] > 0.01)}
    return floor, clean_P


# --------------------------------------------------------------------------- #
#  RE-AUDIT — standard removal configs, linear certificate vs two-tier suite  #
# --------------------------------------------------------------------------- #
def _cert_r2(H, attr, device, sigma_abs=0.0):
    """The linear certificate: LoRA-attacked linear R^2 (run_attack), 3 seeds."""
    H_t = torch.from_numpy(H.astype(np.float32)).to(device)
    attr_t = torch.from_numpy(attr).long().to(device)
    r2 = [run_attack(H_t, attr_t, attr, device, s, sigma_abs)["final_linear_r2"]
          for s in REAUDIT_SEEDS]
    return float(np.mean(r2)), float(np.std(r2))


def _reaudit_config(label, method, P, attr, task, n_attr, task_maj, device,
                    sigma_abs=None, H_er=None):
    """One removal config: certificate + two-tier suite + utility."""
    t0 = time.time()
    d = P.shape[1]
    if method == "noise":
        r2, r2_std = _cert_r2(P, attr, device, sigma_abs=sigma_abs)
        ch = iso_channel(sigma_abs)
        _, xgb = probe_through("XGB", P, attr, n_attr, ch, REAUDIT_SEEDS)
        _, mlp = probe_through("MLP", P, attr, n_attr, ch, REAUDIT_SEEDS)
        lora = lora_channel(P, attr, n_attr, ch, device, REAUDIT_SEEDS)
        lrt, _ = gaussian_lrt_channel(P, attr, ch, iso_noise_cov(d, sigma_abs),
                                      n_attr, REAUDIT_SEEDS)
        lifts = [task_accuracy(P, task, sigma_abs, s) - task_maj for s in REAUDIT_SEEDS]
    else:  # deterministic projection
        r2, r2_std = _cert_r2(H_er, attr, device)
        xgb, _ = probe_sklearn("XGB", H_er, attr, n_attr, REAUDIT_SEEDS)
        mlp, _ = probe_sklearn("MLP", H_er, attr, n_attr, REAUDIT_SEEDS)
        lora, _ = probe_lora(H_er, attr, n_attr, REAUDIT_SEEDS, device)
        lrt, _ = gaussian_lrt(P, H_er, attr, np.zeros((d, d)), n_attr, REAUDIT_SEEDS)
        lifts = [task_accuracy(H_er, task, 0.0, s) - task_maj for s in REAUDIT_SEEDS]

    tier1 = float(max(xgb, mlp, lora))
    tier2 = float(max(tier1, lrt))
    row = {"method": label, "linear_r2": r2, "linear_r2_std": r2_std,
           "linear_stopped": bool(r2 <= TAU),
           "xgb_auc": float(xgb), "mlp_auc": float(mlp), "lora_auc": float(lora),
           "lrt_auc": float(lrt), "tier1_max": tier1, "tier2_max": tier2,
           "honest_stopped": bool(max(xgb, mlp) <= HONEST_AUC_BAR),
           "tier1_stopped": bool(tier1 <= HONEST_AUC_BAR),
           "tier2_stopped": bool(tier2 <= HONEST_AUC_BAR),
           "task_lift": float(np.mean(lifts)),
           "seconds": round(time.time() - t0, 1)}
    row["false_linear_pass"] = row["linear_stopped"] and not row["honest_stopped"]
    print(f"  [reaudit {label:<22}] R²={r2:.4f} ({'STOP' if row['linear_stopped'] else 'breach'}) "
          f"| XGB={xgb:.3f} MLP={mlp:.3f} LoRA={lora:.3f} LRT={lrt:.3f} "
          f"T1={tier1:.3f} T2={tier2:.3f} lift={row['task_lift']:+.4f} "
          f"[{row['seconds']:.0f}s]", flush=True)
    return row


def measure_reaudit(P, attr, task, n_attr, task_maj, device):
    sigma_h = float(P.std(0).mean())
    P_t = torch.from_numpy(P.astype(np.float32)).to(device)
    rows = []
    for sr in NOISE_SIGMA_RELS:
        rows.append(_reaudit_config(f"noise σ={sr:g}", "noise", P, attr, task,
                                    n_attr, task_maj, device,
                                    sigma_abs=sr * sigma_h))
    for fam, kind in (("LEOPARD-MMD proj", "mmd"), ("HSIC proj", "hsic")):
        for rank in PROJ_RANKS:
            Q = fit_projection(P_t, attr, rank, kind, device, seed=0)
            H_er = apply_projection(P, Q)
            rows.append(_reaudit_config(f"{fam} r={rank}", "proj", P, attr, task,
                                        n_attr, task_maj, device, H_er=H_er))
    return {"sigma_h": sigma_h, "noise_sigma_rels": NOISE_SIGMA_RELS,
            "proj_ranks": PROJ_RANKS, "rows": rows}


# --------------------------------------------------------------------------- #
def run_cell(tag: str):
    device = get_device()
    print(f"\n{'#' * 92}\n# EXPANSION CELL: {tag}  (device={device}, max_n={MAX_N})\n"
          f"{'#' * 92}", flush=True)
    t_start = time.time()
    source, attr_name, task_name, X, attr, task, n_attr, n_task, attr_maj, task_maj = \
        load_cell_by_tag(tag)
    t_load = time.time() - t_start
    X_t = torch.from_numpy(X).to(device)

    t0 = time.time()
    cost_row = measure_cell(source, attr_name, task_name, device,
                            X=X, attr=attr, task=task, tag=tag)
    t_cost = time.time() - t0

    t0 = time.time()
    floor, clean_P = measure_floor(cost_row, X_t, attr, task, n_attr, n_task,
                                   task_maj, device)
    t_floor = time.time() - t0

    t0 = time.time()
    reaudit = measure_reaudit(clean_P, attr, task, n_attr, task_maj, device)
    t_reaudit = time.time() - t0

    total = time.time() - t_start
    print(f"[runtime {tag}] load={t_load:.0f}s cost={t_cost:.0f}s floor={t_floor:.0f}s "
          f"reaudit={t_reaudit:.0f}s total={total / 60:.1f}min", flush=True)

    shard = {"cell": tag, "source": source, "attr": attr_name, "task": task_name,
             "n_used": int(len(attr)), "cost": cost_row, "floor": floor,
             "reaudit": reaudit,
             "runtimes_seconds": {"load": round(t_load, 1), "cost": round(t_cost, 1),
                                  "floor": round(t_floor, 1),
                                  "reaudit": round(t_reaudit, 1),
                                  "total": round(total, 1)},
             "provenance": provenance_for(source, len(attr), total)}
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SHARD_DIR / (tag.replace("/", "__") + ".json")
    out_path.write_text(json.dumps(shard, indent=2))
    print(f"[save] {out_path}", flush=True)


# --------------------------------------------------------------------------- #
def merge():
    selected = json.loads(SCAN_PATH.read_text())["selected"]
    shards = []
    for tag in selected:
        p = SHARD_DIR / (tag.replace("/", "__") + ".json")
        if not p.exists():
            print(f"[merge] MISSING shard for {tag} — skipped")
            continue
        shards.append(json.loads(p.read_text()))
    shards.sort(key=lambda s: s["cost"]["predictor"])

    # ---- expansion_cost.json ----
    rows = [{**s["cost"], "provenance": s["provenance"]} for s in shards]
    good = [r for r in rows if not r["degenerate"]]
    from scipy.stats import pearsonr, spearmanr
    stats = {}
    for key in ("cost_rep", "cost_durable"):
        x = np.array([r["predictor"] for r in good])
        y = np.array([r[key] for r in good])
        if len(good) >= 3:
            pr, pp = pearsonr(x, y)
            sr, sp = spearmanr(x, y)
            stats[key] = {"pearson_r": float(pr), "pearson_p": float(pp),
                          "spearman_r": float(sr), "spearman_p": float(sp),
                          "n": len(good)}
    (RESULTS_DIR / "expansion_cost.json").write_text(json.dumps(
        {"experiment": "expansion_cost", "bar": BAR, "sigma_grid": SIGMA_GRID,
         "max_n_cap": MAX_N, "stats_new_cells_only": stats, "rows": rows}, indent=2))

    # ---- expansion_floor.json ----
    frows = [{"cell": s["cell"], "predictor": s["cost"]["predictor"],
              "predictor_std": s["cost"]["predictor_std"], **s["floor"],
              "clean_lift": s["cost"]["clean_lift"],
              "degenerate": s["cost"]["degenerate"],
              "provenance": s["provenance"]} for s in shards]
    (RESULTS_DIR / "expansion_floor.json").write_text(json.dumps(
        {"experiment": "expansion_floor", "bar": BAR,
         "floor_definition": "best-attacker (XGB/MLP, 3 probe seeds, 75/25 "
                             "stratified) AUC on the task outputs at the first "
                             "swept sigma with P_max <= bar",
         "rows": frows}, indent=2))

    # ---- expansion_reaudit.json ----
    master, survived = [], 0
    for s in shards:
        for r in s["reaudit"]["rows"]:
            master.append({"cell": s["cell"], **r})
            survived += int(r["honest_stopped"])
    (RESULTS_DIR / "expansion_reaudit.json").write_text(json.dumps(
        {"experiment": "expansion_reaudit", "tau": TAU,
         "honest_auc_bar": HONEST_AUC_BAR,
         "protocol": "standard removal configs (noise sigma_rel 1/2/8, "
                     "LEOPARD-MMD + HSIC projections r 1/8/16) on the clean "
                     "model's frozen P; linear certificate = LoRA-attacked "
                     "linear R^2 (run_attack, 3 seeds), stopped iff <= tau; "
                     "suite = Tier1 XGB+MLP+LoRA32, Tier2 + Gaussian-LRT; "
                     "honest_stopped = max(XGB,MLP) <= bar (existing re-audit "
                     "convention)",
         "survived": survived, "total": len(master),
         "cells": [{"cell": s["cell"], "sigma_h": s["reaudit"]["sigma_h"],
                    "provenance": s["provenance"]} for s in shards],
         "master": master}, indent=2))

    print(f"[merge] {len(shards)} shards -> expansion_cost.json "
          f"(stats {json.dumps(stats.get('cost_durable', {}))}), "
          f"expansion_floor.json, expansion_reaudit.json "
          f"(survived {survived}/{len(master)})")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge()
    elif "--cell" in sys.argv:
        run_cell(sys.argv[sys.argv.index("--cell") + 1])
    else:
        raise SystemExit("usage: expansion_pipeline.py --cell <source/attr/task> | --merge")
