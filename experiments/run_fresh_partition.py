"""Fresh-partition generalization check (blind-reviewer R2 W1).

The paper's attacker suite is fit and scored on held-out rows of the SAME data
partition the representation model trained on. Objection: an encoder can
memorize/embed the attribute for training rows, so the audit should be checked
on rows the representation model never saw.

Protocol (pinned in docs/fresh_partition_prediction.md, committed 592c551 BEFORE this
script ran):
  * 50/50 disjoint row split per dataset, split seed 20260724, stratified on
    (attr, task) jointly, before any training.
  * Encoder + head (+ subspace Q) trained on the REPRESENTATION partition only,
    at the registered operating points, no re-tuning:
      - e2e full-rank: tier-1 sigma = 8/12/32, tier-2 sigma = 20/24/64
      - e2e subspace:  easy r32 s16 lam10, middle r16 s16 lam100,
                       hard r32 s16 lam100
      - clean sigma=0 baseline per cell (fresh-partition clean lift)
  * 3 training seeds [0,1,2] per point.
  * Full suite (XGB + MLP + LoRA32 + channel-aware Gaussian-LRT) fit and scored
    entirely within the EVALUATION partition (standard 75/25 internal probe
    split, probe seeds [0,1,2]). Bar 0.55 on the 3-seed mean.
  * Utility = own-head lift on the evaluation partition / clean lift on the
    same partition. In-sample (rep-partition) lift recorded alongside.

Training loops are line-for-line replicas of mi_ceiling.train_channel and
end_to_end_surgical.train_e2e_surgical, except the batch sampler draws from the
representation partition only and the eval-mode forward additionally runs over
the evaluation partition. Local MPS/CPU. macOS OpenMP guard: xgboost before
torch.
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
import torch.nn.functional as F  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT))

from experiments.diagnostic import (  # noqa: E402
    lift, load_cell, probe_lora, probe_sklearn,
)
from experiments.sealed_channel import PUB_DIM, TRAIN_BS, TRAIN_STEPS, _mlp  # noqa: E402
from experiments.smart_erasure import hsic_loss  # noqa: E402
from utils.battery import (  # noqa: E402
    gaussian_lrt, iso_noise_cov, subspace_noise_cov,
)
from utils.pcrl_io import get_device  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
BAR = 0.55
SPLIT_SEED = 20260724
TRAIN_SEEDS = [0, 1, 2]
PROBE_SEEDS = [0, 1, 2]
SMOKE = os.environ.get("SMOKE", "") == "1"
STEPS = 60 if SMOKE else TRAIN_STEPS

CELLS = [
    dict(key="easy", name="hmda/race/loan_decision", dataset="hmda", attr="race",
         task="loan_decision", fullrank_t1=8.0, fullrank_t2=20.0,
         subspace=dict(rank=32, sigma=16.0, lam=10.0)),
    dict(key="middle", name="hmda/race/loan_amount_band", dataset="hmda",
         attr="race", task="loan_amount_band", fullrank_t1=12.0, fullrank_t2=24.0,
         subspace=dict(rank=16, sigma=16.0, lam=100.0)),
    dict(key="hard", name="adult/sex/income", dataset="adult", attr="sex",
         task="income", fullrank_t1=32.0, fullrank_t2=64.0,
         subspace=dict(rank=32, sigma=16.0, lam=100.0)),
]


def split_partitions(attr, task, n_task):
    """50/50 disjoint row split, stratified on (attr, task) jointly."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(attr))
    strat = attr * n_task + task
    rep_idx, ev_idx = train_test_split(idx, test_size=0.5,
                                       random_state=SPLIT_SEED, stratify=strat)
    return np.sort(rep_idx), np.sort(ev_idx)


# --------------------------------------------------------------------------- #
#  Training replicas (train on rep partition, forward both partitions)         #
# --------------------------------------------------------------------------- #
def train_fullrank_split(X_rep_t, task_rep, n_attr, n_task, sigma, X_ev_t,
                         device, seed):
    """mi_ceiling.train_channel, batches drawn from the rep partition only.
    Returns (h_ev, P_ev, L_ev, L_rep): clean/noised reps + logits on the eval
    partition, and logits on the rep partition (in-sample utility)."""
    torch.manual_seed(seed)
    d_in = X_rep_t.shape[1]
    E_pub = _mlp(d_in, 128, PUB_DIM).to(device)
    bn = nn.BatchNorm1d(PUB_DIM).to(device) if sigma > 0 else None
    D = nn.Linear(PUB_DIM, n_task).to(device)
    Cd = _mlp(PUB_DIM, 64, n_attr).to(device)   # unused; mirrors train_model
    params = list(E_pub.parameters()) + list(D.parameters()) + list(Cd.parameters())
    if bn is not None:
        params += list(bn.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    ce = nn.CrossEntropyLoss()
    task_t = torch.from_numpy(task_rep).long().to(device)
    n = X_rep_t.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    def public(xb):
        P = E_pub(xb)
        if bn is not None:
            P = bn(P) + sigma * torch.randn_like(P)
        return P

    E_pub.train(); D.train(); Cd.train()
    if bn is not None:
        bn.train()
    for _ in range(STEPS):
        idx = torch.randint(0, n, (TRAIN_BS,), generator=g).to(device)
        loss = ce(D(public(X_rep_t[idx])), task_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    E_pub.eval(); D.eval()
    if bn is not None:
        bn.eval()

    def forward_all(X_t, want_h):
        H_all, P_all, L_all = [], [], []
        with torch.no_grad():
            for i in range(0, X_t.shape[0], 4096):
                xb = X_t[i:i + 4096]
                hb = bn(E_pub(xb)) if bn is not None else E_pub(xb)
                P = hb + sigma * torch.randn_like(hb) if bn is not None else hb
                if want_h:
                    H_all.append(hb.cpu().numpy())
                P_all.append(P.cpu().numpy())
                L_all.append(D(P).cpu().numpy())
        h = np.concatenate(H_all).astype(np.float32) if want_h else None
        return h, np.concatenate(P_all).astype(np.float32), \
            np.concatenate(L_all).astype(np.float32)

    h_ev, P_ev, L_ev = forward_all(X_ev_t, want_h=True)
    _, _, L_rep = forward_all(X_rep_t, want_h=False)
    return h_ev, P_ev, L_ev, L_rep


def train_subspace_split(X_rep_t, attr_rep, task_rep, n_attr, n_task, *, rank,
                         sigma, lam, X_ev_t, device, seed):
    """end_to_end_surgical.train_e2e_surgical, batches from the rep partition
    only. Returns (P_ev, L_ev, h_ev, Q, L_rep)."""
    torch.manual_seed(seed)
    d_in = X_rep_t.shape[1]
    E = _mlp(d_in, 128, PUB_DIM).to(device)
    bn = nn.BatchNorm1d(PUB_DIM).to(device)
    M = nn.Parameter(torch.randn(PUB_DIM, rank, device=device) * 0.1)
    D = nn.Linear(PUB_DIM, n_task).to(device)
    opt = torch.optim.Adam(
        list(E.parameters()) + list(bn.parameters()) + [M] + list(D.parameters()),
        lr=1e-3)
    ce = nn.CrossEntropyLoss()
    attr_t = torch.from_numpy(attr_rep).long().to(device)
    task_t = torch.from_numpy(task_rep).long().to(device)
    n = X_rep_t.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    E.train(); bn.train()
    for _ in range(STEPS):
        idx = torch.randint(0, n, (TRAIN_BS,), generator=g).to(device)
        xb = X_rep_t[idx]
        h = bn(E(xb))
        Q, _ = torch.linalg.qr(M)
        z = torch.randn(h.shape[0], rank, device=device)
        h_noised = h + sigma * (z @ Q.T)
        loss = ce(D(h_noised), task_t[idx])
        if lam > 0:
            h_perp = h - (h @ Q) @ Q.T
            yoh = F.one_hot(attr_t[idx], n_attr).float()
            loss = loss + lam * hsic_loss(h_perp, yoh)
        opt.zero_grad()
        loss.backward()
        opt.step()

    E.eval(); bn.eval()
    with torch.no_grad():
        Q, _ = torch.linalg.qr(M)

        def forward_all(X_t, want_h):
            H_all, P_all, L_all = [], [], []
            for i in range(0, X_t.shape[0], 4096):
                h = bn(E(X_t[i:i + 4096]))
                z = torch.randn(h.shape[0], rank, device=device)
                hn = h + sigma * (z @ Q.T)
                if want_h:
                    H_all.append(h.cpu().numpy())
                P_all.append(hn.cpu().numpy())
                L_all.append(D(hn).cpu().numpy())
            h_np = np.concatenate(H_all).astype(np.float32) if want_h else None
            return h_np, np.concatenate(P_all).astype(np.float32), \
                np.concatenate(L_all).astype(np.float32)

        h_ev, P_ev, L_ev = forward_all(X_ev_t, want_h=True)
        _, _, L_rep = forward_all(X_rep_t, want_h=False)
    return P_ev, L_ev, h_ev, Q.cpu().numpy().astype(np.float64), L_rep


# --------------------------------------------------------------------------- #
#  Measurement on the evaluation partition                                     #
# --------------------------------------------------------------------------- #
def eval_point(h_ev, P_ev, attr_ev, n_attr, noise_cov, device):
    pt = {}
    pt["xgb"], _ = probe_sklearn("XGB", P_ev, attr_ev, n_attr, PROBE_SEEDS)
    pt["mlp"], _ = probe_sklearn("MLP", P_ev, attr_ev, n_attr, PROBE_SEEDS)
    pt["lora"], _ = probe_lora(P_ev, attr_ev, n_attr, PROBE_SEEDS, device)
    pt["lrt"], _ = gaussian_lrt(h_ev, P_ev, attr_ev, noise_cov, n_attr, PROBE_SEEDS)
    pt["tier1_max"] = float(max(pt["xgb"], pt["mlp"], pt["lora"]))
    pt["tier2_max"] = float(max(pt["tier1_max"], pt["lrt"]))
    return pt


def seed_mean(points):
    keys = points[0].keys()
    out = {f"{k}_mean": float(np.mean([p[k] for p in points])) for k in keys}
    out.update({f"{k}_std": float(np.std([p[k] for p in points])) for k in keys})
    out["per_seed"] = {k: [p[k] for p in points] for k in keys}
    # verdict on 3-seed means, matching the paper's convention
    out["tier1_pass"] = bool(out["tier1_max_mean"] <= BAR)
    out["tier2_pass"] = bool(out["tier2_max_mean"] <= BAR)
    return out


def fmt(pt):
    return (f"XGB={pt['xgb']:.3f} MLP={pt['mlp']:.3f} LoRA={pt['lora']:.3f} "
            f"LRT={pt['lrt']:.3f} | T1={pt['tier1_max']:.3f} "
            f"T2={pt['tier2_max']:.3f}")


def run_cell(cell, device):
    print("\n" + "#" * 96 + f"\n# FRESH PARTITION: {cell['name']}\n" + "#" * 96,
          flush=True)
    t0 = time.time()
    X, attr, task, n_attr, n_task, _, _ = load_cell(
        cell["dataset"], cell["attr"], cell["task"])
    rep_idx, ev_idx = split_partitions(attr, task, n_task)
    X_rep_t = torch.from_numpy(X[rep_idx]).to(device)
    X_ev_t = torch.from_numpy(X[ev_idx]).to(device)
    attr_rep, task_rep = attr[rep_idx], task[rep_idx]
    attr_ev, task_ev = attr[ev_idx], task[ev_idx]
    task_maj_ev = float(np.bincount(task_ev).max() / len(task_ev))
    task_maj_rep = float(np.bincount(task_rep).max() / len(task_rep))
    print(f"[split] N={len(attr)} rep={len(rep_idx)} eval={len(ev_idx)} "
          f"seed={SPLIT_SEED} task_maj eval={task_maj_ev:.3f}", flush=True)

    out = {"cell": cell["name"], "n_rep": int(len(rep_idx)),
           "n_eval": int(len(ev_idx)), "task_maj_eval": task_maj_ev}

    # clean baseline (sigma=0): fresh-partition clean lift over the same seeds
    clean_ev, clean_rep = [], []
    for ts in TRAIN_SEEDS:
        _, _, L_ev, L_rep = train_fullrank_split(
            X_rep_t, task_rep, n_attr, n_task, 0.0, X_ev_t, device, ts)
        clean_ev.append(lift(L_ev, task_ev, task_maj_ev))
        clean_rep.append(lift(L_rep, task_rep, task_maj_rep))
    out["clean_lift_eval"] = float(np.mean(clean_ev))
    out["clean_lift_eval_std"] = float(np.std(clean_ev))
    out["clean_lift_rep_insample"] = float(np.mean(clean_rep))
    print(f"[clean] eval lift={out['clean_lift_eval']:+.4f}"
          f"±{out['clean_lift_eval_std']:.4f} "
          f"(in-sample {out['clean_lift_rep_insample']:+.4f})", flush=True)
    cl = max(out["clean_lift_eval"], 1e-9)

    # full-rank points
    for tier_key in ("fullrank_t1", "fullrank_t2"):
        sigma = cell[tier_key]
        pts, lifts_ev, lifts_rep = [], [], []
        for ts in TRAIN_SEEDS:
            h_ev, P_ev, L_ev, L_rep = train_fullrank_split(
                X_rep_t, task_rep, n_attr, n_task, sigma, X_ev_t, device, ts)
            pt = eval_point(h_ev, P_ev, attr_ev, n_attr,
                            iso_noise_cov(PUB_DIM, sigma), device)
            pts.append(pt)
            lifts_ev.append(lift(L_ev, task_ev, task_maj_ev))
            lifts_rep.append(lift(L_rep, task_rep, task_maj_rep))
            print(f"  [{tier_key} σ={sigma:g} seed {ts}] {fmt(pt)} "
                  f"lift_ev={lifts_ev[-1]:+.4f}", flush=True)
        m = seed_mean(pts)
        m["sigma"] = sigma
        m["lift_eval_mean"] = float(np.mean(lifts_ev))
        m["lift_eval_std"] = float(np.std(lifts_ev))
        m["lift_rep_insample_mean"] = float(np.mean(lifts_rep))
        m["utility_kept_eval_pct"] = float(100.0 * np.mean(lifts_ev) / cl)
        out[tier_key] = m
        print(f"  >>> {tier_key} σ={sigma:g}: T1={m['tier1_max_mean']:.3f} "
              f"({'pass' if m['tier1_pass'] else 'BREACH'}) "
              f"LRT={m['lrt_mean']:.3f} T2={m['tier2_max_mean']:.3f} "
              f"({'pass' if m['tier2_pass'] else 'BREACH'}) "
              f"util={m['utility_kept_eval_pct']:.1f}%", flush=True)

    # subspace point
    sub = cell["subspace"]
    pts, lifts_ev, lifts_rep = [], [], []
    for ts in TRAIN_SEEDS:
        P_ev, L_ev, h_ev, Q, L_rep = train_subspace_split(
            X_rep_t, attr_rep, task_rep, n_attr, n_task, rank=sub["rank"],
            sigma=sub["sigma"], lam=sub["lam"], X_ev_t=X_ev_t, device=device,
            seed=ts)
        pt = eval_point(h_ev, P_ev, attr_ev, n_attr,
                        subspace_noise_cov(Q, sub["sigma"]), device)
        pts.append(pt)
        lifts_ev.append(lift(L_ev, task_ev, task_maj_ev))
        lifts_rep.append(lift(L_rep, task_rep, task_maj_rep))
        print(f"  [subspace r={sub['rank']} σ={sub['sigma']:g} λ={sub['lam']:g} "
              f"seed {ts}] {fmt(pt)} lift_ev={lifts_ev[-1]:+.4f}", flush=True)
    m = seed_mean(pts)
    m.update(sub)
    m["lift_eval_mean"] = float(np.mean(lifts_ev))
    m["lift_eval_std"] = float(np.std(lifts_ev))
    m["lift_rep_insample_mean"] = float(np.mean(lifts_rep))
    m["utility_kept_eval_pct"] = float(100.0 * np.mean(lifts_ev) / cl)
    out["subspace"] = m
    print(f"  >>> subspace: T1={m['tier1_max_mean']:.3f} "
          f"({'pass' if m['tier1_pass'] else 'BREACH'}) "
          f"LRT={m['lrt_mean']:.3f} ({'fails T2 as before' if m['lrt_mean'] > BAR else 'T2 pass'}) "
          f"util={m['utility_kept_eval_pct']:.1f}%", flush=True)

    out["minutes"] = round((time.time() - t0) / 60, 1)
    return out


# --------------------------------------------------------------------------- #
#  Stored in-partition comparators                                             #
# --------------------------------------------------------------------------- #
def stored_comparators():
    tt = json.loads((RESULTS_DIR / "two_tier_certification.json").read_text())
    hs = json.loads((RESULTS_DIR / "hardening_seeds.json").read_text())
    ivf_p = OUT_DIR / "isolate_vs_fullrank.json"
    ivf = json.loads(ivf_p.read_text()) if ivf_p.exists() else None
    comp = {}
    for i, key in enumerate(("easy", "middle", "hard")):
        c = tt["e2e_cells"][i]
        comp[key] = {
            "fullrank_t1": {"tier1_max": c["tier1"]["tier1_max_mean"],
                            "lrt": c["tier1"]["lrt_mean"],
                            "util_pct": 100.0 * c["tier1"]["lift_mean"] / c["clean_lift"]},
            "fullrank_t2": {"tier1_max": c["tier2"]["tier1_max_mean"],
                            "lrt": c["tier2"]["lrt_mean"],
                            "util_pct": 100.0 * c["tier2"]["lift_mean"] / c["clean_lift"]},
        }
    sub_stored = {
        "easy": hs["exp10_5seed"][0]["D"],
        "middle": hs["exp10_5seed"][1]["D"],
        "hard": hs["hard_cell_sigma16_alternative"],
    }
    for key, blk in sub_stored.items():
        comp[key]["subspace"] = {"tier1_max": blk["rep_max_mean"],
                                 "util_pct": blk.get("kept_pct")}
    if ivf:
        for row in ivf["rows"]:
            comp[row["cell"]]["subspace"]["lrt"] = row["isolate"]["lrt_mean"]
    return comp


def main():
    device = get_device()
    print(f"[setup] device={device} split_seed={SPLIT_SEED} bar≤{BAR} "
          f"train_seeds={TRAIN_SEEDS} probe_seeds={PROBE_SEEDS} steps={STEPS}"
          + ("  *** SMOKE ***" if SMOKE else ""), flush=True)
    cells = [run_cell(c, device) for c in (CELLS[:1] if SMOKE else CELLS)]

    comp = stored_comparators()
    flips = []
    for c, cfg in zip(cells, CELLS):
        key = cfg["key"]
        for ch in ("fullrank_t1", "fullrank_t2", "subspace"):
            fresh = c[ch]
            stored = comp[key][ch]
            fresh["stored_inpartition"] = stored
            fresh["delta_tier1_max"] = float(
                fresh["tier1_max_mean"] - stored["tier1_max"])
            if "lrt" in stored:
                fresh["delta_lrt"] = float(fresh["lrt_mean"] - stored["lrt"])
            stored_t1_pass = stored["tier1_max"] <= BAR
            if fresh["tier1_pass"] != stored_t1_pass:
                flips.append(f"{key}/{ch}: tier1 {stored_t1_pass}→{fresh['tier1_pass']}")
            if ch == "fullrank_t2":
                stored_t2_pass = max(stored["tier1_max"], stored["lrt"]) <= BAR
                if fresh["tier2_pass"] != stored_t2_pass:
                    flips.append(f"{key}/{ch}: tier2 {stored_t2_pass}→{fresh['tier2_pass']}")

    out = {"experiment": "fresh_partition_generalization", "bar": BAR,
           "split_seed": SPLIT_SEED, "split": "50/50 stratified (attr,task)",
           "train_seeds": TRAIN_SEEDS, "probe_seeds": PROBE_SEEDS,
           "steps": STEPS, "smoke": SMOKE,
           "prediction_commit": "592c551", "cells": cells, "flips": flips}
    path = OUT_DIR / ("fresh_partition_generalization_SMOKE.json" if SMOKE
                      else "fresh_partition_generalization.json")
    path.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 96)
    print(f"FLIPS vs in-partition verdicts: {flips if flips else 'NONE'}")
    print(f"[save] {path}", flush=True)


if __name__ == "__main__":
    main()
