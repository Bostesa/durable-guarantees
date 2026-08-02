"""Isolate-noise-coarsen hybrid — registered in hybrid_coarsen_prediction.md
(commit 364870e) BEFORE this file existed.

Per cell (one fixed config, no tuning): retrain the e2e subspace operating
point at 3 seeds, GATE against the stored isolate_vs_fullrank readings, then
replace the noiseless complement with a task-aware tree coarsening (sklearn
DecisionTreeClassifier on h_perp -> y, defense-train rows only, per-leaf
training medians) and re-run the full two-tier battery plus the registered
defense-aware Tier-2 attackers on the hybrid release. sklearn only; no FARE
code. Local MPS; never AWS. macOS OpenMP guard: xgboost before torch.
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
import torch.nn as nn  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO))

from experiments.diagnostic import (  # noqa: E402
    battery, lift, load_cell, probe_lora, probe_sklearn,
)
from experiments.hardening_test import _probe  # noqa: E402
from experiments.end_to_end_surgical import train_e2e_surgical  # noqa: E402
from utils.battery import (  # noqa: E402
    _auc, _lrt_scores, _split_idx, gaussian_lrt, subspace_noise_cov,
)
from utils.pcrl_io import get_device  # noqa: E402

BAR = 0.55
TRAIN_SEEDS = [0, 1, 2]
PROBE_SEEDS = [0, 1, 2]          # hybrid battery (registered)
GATE_SEEDS = [0, 1]              # the stored instrument (isolate_vs_fullrank)
GATE_TOL = 0.01
MAX_LEAF_NODES = 32
MIN_SAMPLES_LEAF = 100
HEAD_STEPS = 1500                # sealed_channel TRAIN_STEPS (same budget)
HEAD_BS = 1024                   # sealed_channel TRAIN_BS

CELLS = {
    "easy": ("hmda", "race", "loan_decision", dict(rank=32, sigma=16.0, lam=10.0)),
    "middle": ("hmda", "race", "loan_amount_band", dict(rank=16, sigma=16.0, lam=100.0)),
    "hard": ("adult", "sex", "income", dict(rank=32, sigma=16.0, lam=100.0)),
}
# stored isolate_vs_fullrank.json 3-seed means (sha256 471a4ccd, pinned in the
# prediction file) — the reproduction gate targets
STORED = {"easy": {"t1": 0.5094, "lrt": 0.8542},
          "middle": {"t1": 0.5349, "lrt": 0.8640},
          "hard": {"t1": 0.5369, "lrt": 0.6617}}
# comparators for the results table (same pinned files)
COMPARATORS = {"subspace_util": {"easy": 101.6, "middle": 100.4, "hard": 84.3},
               "subspace_lrt": {"easy": 0.8542, "middle": 0.8640, "hard": 0.6617},
               "fullrank_t2_util": {"easy": 55.5, "middle": 21.7, "hard": -7.0},
               "fare_util": {"easy": 0.0, "middle": 28.3, "hard": 39.0}}


def t1_members(H, y, n_classes, device, seeds):
    x, _ = probe_sklearn("XGB", H, y, n_classes, seeds)
    m, _ = probe_sklearn("MLP", H, y, n_classes, seeds)
    l, _ = probe_lora(H, y, n_classes, seeds, device)
    return {"xgb": float(x), "mlp": float(m), "lora": float(l)}


def leaf_posteriors(leaf_all, attr, tr_idx, n_attr):
    """P(attr | leaf) from probe-train counts, Laplace +1, for ALL rows."""
    leaves = np.unique(leaf_all)
    lut = {}
    for lf in leaves:
        rows = tr_idx[leaf_all[tr_idx] == lf]
        cnt = np.bincount(attr[rows], minlength=n_attr).astype(np.float64) + 1.0
        lut[lf] = cnt / cnt.sum()
    return np.stack([lut[lf] for lf in leaf_all])


def train_head(release, task, n_task, tr_idx, device, seed):
    """Same head architecture and budget as the e2e own head (Linear + Adam
    1e-3, 1500 steps, batch 1024), trained on defense-train release rows."""
    torch.manual_seed(seed)
    R_t = torch.from_numpy(release[tr_idx]).float().to(device)
    y_t = torch.from_numpy(task[tr_idx]).long().to(device)
    head = nn.Linear(release.shape[1], n_task).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = len(tr_idx)
    for _ in range(HEAD_STEPS):
        idx = torch.randint(0, n, (min(HEAD_BS, n),), generator=g).to(device)
        loss = ce(head(R_t[idx]), y_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        L = np.concatenate(
            [head(torch.from_numpy(release[i:i + 8192]).float().to(device))
             .cpu().numpy() for i in range(0, len(release), 8192)])
    return L.astype(np.float32)


def run_cell(key, device, clean_lift_stored):
    ds, at, tk, cfg = CELLS[key]
    r, sigma = cfg["rank"], cfg["sigma"]
    print("\n" + "#" * 92 + f"\n# {key} {ds}/{at}/{tk} r={r} s={sigma:g} "
          f"l={cfg['lam']:g}\n" + "#" * 92, flush=True)
    X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk)
    X_t = torch.from_numpy(X).to(device)

    per = {k: [] for k in
           ("gate_t1", "gate_lrt",
            "rep_xgb", "rep_mlp", "rep_lora", "out_xgb", "out_mlp", "out_lora",
            "t2_lrt_sub", "t2_leaf", "t2_combined", "t2_joint_xgb",
            "comp_leaf", "comp_xgb", "comp_mlp", "comp_lora",
            "util_in", "util_held", "n_leaves")}
    for ts in TRAIN_SEEDS:
        t0 = time.time()
        P, _, h, Q = train_e2e_surgical(
            X_t, attr, task, n_attr, n_task, rank=r, sigma=sigma,
            lam=cfg["lam"], device=device, seed=ts, return_clean=True)

        # ---- reproduction gate readings (stored instrument: probe seeds [0,1])
        g1 = t1_members(P, attr, n_attr, device, GATE_SEEDS)
        gl, _ = gaussian_lrt(h, P, attr, subspace_noise_cov(Q, sigma), n_attr,
                             GATE_SEEDS)
        per["gate_t1"].append(max(g1.values()))
        per["gate_lrt"].append(gl)
        print(f"  [seed {ts}] gate: T1={max(g1.values()):.4f} LRT={gl:.4f}",
              flush=True)

        # ---- hybrid release ------------------------------------------------
        h64 = h.astype(np.float64)
        u_clean = h64 @ Q                                  # N x r
        h_perp = h64 - u_clean @ Q.T                       # N x 64
        rng = np.random.default_rng(2000 + ts)
        u_noised = u_clean + sigma * rng.standard_normal(u_clean.shape)

        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier
        idx = np.arange(len(task))
        dtr, dte = train_test_split(idx, test_size=0.25, random_state=1000 + ts,
                                    stratify=task)
        tree = DecisionTreeClassifier(max_leaf_nodes=MAX_LEAF_NODES,
                                      min_samples_leaf=MIN_SAMPLES_LEAF,
                                      random_state=ts)
        tree.fit(h_perp[dtr], task[dtr])
        leaf_all = tree.apply(h_perp)
        med = {lf: np.median(h_perp[dtr][leaf_all[dtr] == lf], axis=0)
               for lf in np.unique(leaf_all[dtr])}
        m_leaf = np.stack([med[lf] for lf in leaf_all])
        release = np.concatenate([u_noised, m_leaf], 1).astype(np.float32)
        per["n_leaves"].append(len(med))

        # ---- own head + utility -------------------------------------------
        L = train_head(release, task, n_task, dtr, device, ts)
        li = float((L[dtr].argmax(1) == task[dtr]).mean()) - task_maj
        lh = float((L[dte].argmax(1) == task[dte]).mean()) - task_maj
        per["util_in"].append(100.0 * li / clean_lift_stored)
        per["util_held"].append(100.0 * lh / clean_lift_stored)

        # ---- Tier-1 on release and y-hat ----------------------------------
        b_rep = t1_members(release, attr, n_attr, device, PROBE_SEEDS)
        b_out = t1_members(L, attr, n_attr, device, PROBE_SEEDS)
        for k, v in b_rep.items():
            per[f"rep_{k}"].append(v)
        for k, v in b_out.items():
            per[f"out_{k}"].append(v)

        # ---- Tier-2 defense-aware attackers -------------------------------
        ncov = (sigma ** 2) * np.eye(r)
        a_lrt, a_leaf, a_comb, a_joint = [], [], [], []
        for s in PROBE_SEEDS:
            tr_i, te_i = _split_idx(attr, s)
            post_a = _lrt_scores(u_clean[tr_i], attr[tr_i], u_noised, ncov,
                                 n_attr)                      # all rows
            post_b = leaf_posteriors(leaf_all, attr, tr_i, n_attr)
            a_lrt.append(_auc(attr[te_i], post_a[te_i], n_attr))
            a_leaf.append(_auc(attr[te_i], post_b[te_i], n_attr))
            logc = np.log(post_a + 1e-12) + np.log(post_b + 1e-12)
            logc -= logc.max(1, keepdims=True)
            post_c = np.exp(logc) / np.exp(logc).sum(1, keepdims=True)
            a_comb.append(_auc(attr[te_i], post_c[te_i], n_attr))
            feats = np.concatenate([post_a, post_b, release], 1)
            _, auc_d = _probe("XGB", feats[tr_i], attr[tr_i], feats[te_i],
                              attr[te_i], s, n_attr)
            a_joint.append(auc_d)
        per["t2_lrt_sub"].append(float(np.mean(a_lrt)))
        per["t2_leaf"].append(float(np.mean(a_leaf)))
        per["t2_combined"].append(float(np.mean(a_comb)))
        per["t2_joint_xgb"].append(float(np.mean(a_joint)))

        # ---- complement-side informed reading (P2) ------------------------
        b_comp = t1_members(m_leaf.astype(np.float32), attr, n_attr, device,
                            PROBE_SEEDS)
        per["comp_leaf"].append(float(np.mean(a_leaf)))
        for k, v in b_comp.items():
            per[f"comp_{k}"].append(v)

        print(f"  [seed {ts}] leaves={len(med)} rep_max="
              f"{max(b_rep.values()):.4f} out_max={max(b_out.values()):.4f} | "
              f"T2 lrt={np.mean(a_lrt):.4f} "
              f"leaf={np.mean(a_leaf):.4f} comb={np.mean(a_comb):.4f} "
              f"joint={np.mean(a_joint):.4f} | comp_max="
              f"{max(max(b_comp.values()), np.mean(a_leaf)):.4f} | "
              f"util in={per['util_in'][-1]:.1f}% held={per['util_held'][-1]:.1f}% "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    mean = {k: float(np.mean(v)) for k, v in per.items()}
    std = {k: float(np.std(v)) for k, v in per.items()}
    gate_ok = (abs(mean["gate_t1"] - STORED[key]["t1"]) <= GATE_TOL
               and abs(mean["gate_lrt"] - STORED[key]["lrt"]) <= GATE_TOL)
    row = {
        "cell": key, "dataset": ds, "attr": at, "task": tk, "config": cfg,
        "n_attr": n_attr, "clean_lift_stored": clean_lift_stored,
        "gate": {"t1_mean": mean["gate_t1"], "lrt_mean": mean["gate_lrt"],
                 "stored": STORED[key], "tol": GATE_TOL, "passed": bool(gate_ok)},
        "tier1_rep_max": max(mean["rep_xgb"], mean["rep_mlp"], mean["rep_lora"]),
        "tier1_out_max": max(mean["out_xgb"], mean["out_mlp"], mean["out_lora"]),
        "tier2": {"lrt_subspace": mean["t2_lrt_sub"], "leaf_posterior": mean["t2_leaf"],
                  "combined_llr": mean["t2_combined"], "joint_xgb": mean["t2_joint_xgb"]},
        "tier2_max": max(mean["t2_lrt_sub"], mean["t2_leaf"],
                         mean["t2_combined"], mean["t2_joint_xgb"]),
        "complement_informed_max": max(mean["comp_leaf"], mean["comp_xgb"],
                                       mean["comp_mlp"], mean["comp_lora"]),
        "util_in_mean": mean["util_in"], "util_in_std": std["util_in"],
        "util_held_mean": mean["util_held"], "util_held_std": std["util_held"],
        "n_leaves_mean": mean["n_leaves"],
        "per_seed": {k: [float(x) for x in v] for k, v in per.items()},
        "means": mean, "stds": std,
    }
    print(f"\n>>> {key}: gate {'PASS' if gate_ok else 'FAIL'} "
          f"(T1 {mean['gate_t1']:.4f} vs {STORED[key]['t1']}, "
          f"LRT {mean['gate_lrt']:.4f} vs {STORED[key]['lrt']}) | "
          f"T1 rep/out {row['tier1_rep_max']:.4f}/{row['tier1_out_max']:.4f} | "
          f"T2 max {row['tier2_max']:.4f} | comp {row['complement_informed_max']:.4f} | "
          f"util {mean['util_in']:.1f}%/{mean['util_held']:.1f}%", flush=True)
    return row


def score(rows):
    by = {r["cell"]: r for r in rows}
    valid = {k: r for k, r in by.items() if r["gate"]["passed"]}
    p1_cells = {k: (r["tier1_rep_max"] <= BAR and r["tier1_out_max"] <= BAR)
                for k, r in valid.items()}
    p2_cells = {k: r["complement_informed_max"] <= 0.60 for k, r in valid.items()}
    p3_cells = {k: r["tier2_max"] <= BAR for k, r in valid.items()}
    p4_need = {"easy": 70.0, "middle": 70.0, "hard": 50.0}
    p4_cells = {k: r["util_in_mean"] >= p4_need[k] for k, r in valid.items()}
    p1 = len(valid) == 3 and all(p1_cells.values())
    p2 = sum(p2_cells.values()) >= 2
    p3 = sum(p3_cells.values()) >= 2
    p4_on_p3 = all(p4_cells[k] for k, ok in p3_cells.items() if ok)
    p4 = all(p4_cells.values())
    success = (len(valid) == 3 and p1 and p3 and p4_on_p3)
    return {
        "gate_failures": [k for k in by if not by[k]["gate"]["passed"]],
        "P1_tier1_all_cells": {"per_cell": p1_cells,
                               "verdict": "HELD" if p1 else "FALSIFIED"},
        "P2_complement_le_060_ge2": {"per_cell": p2_cells,
                                     "verdict": "HELD" if p2 else "FALSIFIED"},
        "P3_tier2_pass_ge2": {"per_cell": p3_cells,
                              "verdict": "HELD" if p3 else "FALSIFIED"},
        "P4_utility": {"per_cell": p4_cells, "thresholds": p4_need,
                       "verdict": "HELD" if p4 else "FALSIFIED",
                       "holds_on_p3_passing_cells": bool(p4_on_p3)},
        "success_criterion_met": bool(success),
    }


def main():
    t0 = time.time()
    device = get_device()
    tt = json.loads((REPO / "results" / "two_tier_certification.json").read_text())
    ttname = {"easy": "hmda/race/loan_decision",
              "middle": "hmda/race/loan_amount_band", "hard": "adult/sex/income"}
    clean = {k: next(c for c in tt["e2e_cells"] if c["cell"] == ttname[k])["clean_lift"]
             for k in CELLS}
    print(f"[setup] device={device} train_seeds={TRAIN_SEEDS} "
          f"probe_seeds={PROBE_SEEDS} gate_seeds={GATE_SEEDS} bar={BAR} "
          f"clean={ {k: round(v, 5) for k, v in clean.items()} }", flush=True)

    rows = [run_cell(k, device, clean[k]) for k in CELLS]
    scoring = score(rows)

    print("\n" + "=" * 92)
    print("SCORING (registered predictions, hybrid_coarsen_prediction.md @ 364870e)")
    print("=" * 92)
    for k, v in scoring.items():
        print(f"  {k}: {v}")

    out = {"experiment": "hybrid_coarsen",
           "prediction_file": "hybrid_coarsen_prediction.md",
           "prediction_commit": "364870e",
           "bar": BAR, "train_seeds": TRAIN_SEEDS, "probe_seeds": PROBE_SEEDS,
           "gate_seeds": GATE_SEEDS,
           "clean_lifts": clean, "comparators": COMPARATORS,
           "cells": rows, "scoring": scoring,
           "minutes": round((time.time() - t0) / 60, 1)}
    (RESULTS / "hybrid_coarsen.json").write_text(json.dumps(out, indent=1))
    print(f"\n[save] {RESULTS / 'hybrid_coarsen.json'} ({out['minutes']} min)",
          flush=True)


if __name__ == "__main__":
    main()
