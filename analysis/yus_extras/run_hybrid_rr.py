"""Hybrid + leaf randomized response — registered in hybrid_rr_prediction.md
(commit 7b8e956) BEFORE this file existed.

Identical to run_hybrid_coarsen.py (sha256 8939c7db) except at release: each
row's leaf assignment is passed through randomized response over the K=32
leaves (keep w.p. 1-p, else uniform over all K), and the released complement
block is the median of the EMITTED leaf. The head retrains through the flip
noise. Tier-2 is fully informed and flip-AWARE: its leaf posterior marginalizes
over the known randomized-response kernel.

Base model, tree and medians depend only on (cell, seed), not on p, so they are
built once per (cell, seed) and reused across the sweep; p=0 is included as the
reproduction gate against the coarsen run. sklearn only; no FARE code.
Local MPS; never AWS. macOS OpenMP guard: xgboost before torch.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from experiments.diagnostic import load_cell, probe_lora, probe_sklearn  # noqa: E402
from experiments.hardening_test import _probe  # noqa: E402
from experiments.end_to_end_surgical import train_e2e_surgical  # noqa: E402
from utils.battery import (  # noqa: E402
    _auc, _lrt_scores, _split_idx, gaussian_lrt, subspace_noise_cov,
)
from utils.pcrl_io import get_device  # noqa: E402

BAR = 0.55
TRAIN_SEEDS = [0, 1, 2]
PROBE_SEEDS = [0, 1, 2]
GATE_SEEDS = [0, 1]
GATE_TOL = 0.01
MAX_LEAF_NODES = 32
MIN_SAMPLES_LEAF = 100
HEAD_STEPS = 1500
HEAD_BS = 1024
P_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]     # p=0 is the gate point

CELLS = {
    "easy": ("hmda", "race", "loan_decision", dict(rank=32, sigma=16.0, lam=10.0)),
    "middle": ("hmda", "race", "loan_amount_band", dict(rank=16, sigma=16.0, lam=100.0)),
    "hard": ("adult", "sex", "income", dict(rank=32, sigma=16.0, lam=100.0)),
}
# hybrid_coarsen.json (sha256 7778a0fd) p=0 readings — the gate targets
GATE = {"easy": {"t2": 0.5843, "util": 107.6},
        "middle": {"t2": 0.6026, "util": 94.4},
        "hard": {"t2": 0.5892, "util": 83.4}}
COMPARATORS = {"subspace_util": {"easy": 101.6, "middle": 100.4, "hard": 84.3},
               "subspace_lrt": {"easy": 0.8542, "middle": 0.8640, "hard": 0.6617},
               "fullrank_t2_util": {"easy": 55.5, "middle": 21.7, "hard": -7.0},
               "fare_util": {"easy": 0.0, "middle": 28.3, "hard": 39.0},
               "coarsen_p0_t2": {"easy": 0.5843, "middle": 0.6026, "hard": 0.5892}}


def eps_leaf(p, K):
    """Per-release epsilon on LEAF MEMBERSHIP (not on the attribute)."""
    if p <= 0:
        return float("inf")
    keep = 1.0 - p + p / K
    return float(math.log(keep / (p / K)))


def t1_members(H, y, n_classes, device, seeds):
    x, _ = probe_sklearn("XGB", H, y, n_classes, seeds)
    m, _ = probe_sklearn("MLP", H, y, n_classes, seeds)
    l, _ = probe_lora(H, y, n_classes, seeds, device)
    return {"xgb": float(x), "mlp": float(m), "lora": float(l)}


def flip(leaf_idx, p, K, rng):
    """Randomized response over K leaves: keep w.p. 1-p, else uniform over K."""
    if p <= 0:
        return leaf_idx.copy()
    out = leaf_idx.copy()
    hit = rng.random(len(leaf_idx)) < p
    out[hit] = rng.integers(0, K, size=int(hit.sum()))
    return out


def flip_aware_posterior(emitted, attr, tr_idx, true_leaf, n_attr, K, p):
    """P(s | emitted leaf) marginalizing over the KNOWN flip kernel.

    The attacker has the tree, so on probe-train rows it knows the TRUE leaf and
    s, giving the joint P(leaf, s); at test time it sees only the emitted leaf.
      P(s | l') propto sum_l P(l'|l) P(l, s)
                = (1-p) * joint[l', s] + (p/K) * colsum[s]
    """
    joint = np.zeros((K, n_attr), dtype=np.float64)
    for k in range(K):
        rows = tr_idx[true_leaf[tr_idx] == k]
        if len(rows):
            joint[k] = np.bincount(attr[rows], minlength=n_attr)
    joint += 1.0                                    # Laplace, as in the coarsen run
    joint /= joint.sum()
    colsum = joint.sum(0)
    post = (1.0 - p) * joint + (p / K) * colsum[None, :]
    post /= post.sum(1, keepdims=True)
    return post[emitted]


def train_head_rr(u_noised, M, true_leaf, task, n_task, tr_idx, p, K, device,
                  seed, flip_seed):
    """Same head architecture/budget as the coarsen run, trained THROUGH the
    flip noise: fresh emitted leaves per batch (at p=0 this is exactly the
    coarsen run's fixed release)."""
    torch.manual_seed(seed)
    u_t = torch.from_numpy(u_noised).float().to(device)
    M_t = torch.from_numpy(M).float().to(device)
    y_t = torch.from_numpy(task).long().to(device)
    leaf_t = torch.from_numpy(true_leaf).long().to(device)
    tr_t = torch.from_numpy(tr_idx).long().to(device)
    head = nn.Linear(u_noised.shape[1] + M.shape[1], n_task).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    g = torch.Generator(device="cpu").manual_seed(seed)
    frng = np.random.default_rng(flip_seed)
    n = len(tr_idx)
    for _ in range(HEAD_STEPS):
        sel = torch.randint(0, n, (min(HEAD_BS, n),), generator=g).to(device)
        idx = tr_t[sel]
        lf = leaf_t[idx].cpu().numpy()
        em = torch.from_numpy(flip(lf, p, K, frng)).long().to(device)
        xb = torch.cat([u_t[idx], M_t[em]], 1)
        loss = ce(head(xb), y_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    return head


def head_logits(head, release, device):
    with torch.no_grad():
        return np.concatenate(
            [head(torch.from_numpy(release[i:i + 8192]).float().to(device))
             .cpu().numpy() for i in range(0, len(release), 8192)]).astype(np.float32)


def build_base(key, device, ts):
    """Base model + tree + medians for one (cell, seed) — independent of p."""
    ds, at, tk, cfg = CELLS[key]
    X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk)
    X_t = torch.from_numpy(X).to(device)
    r, sigma = cfg["rank"], cfg["sigma"]
    P, _, h, Q = train_e2e_surgical(X_t, attr, task, n_attr, n_task, rank=r,
                                    sigma=sigma, lam=cfg["lam"], device=device,
                                    seed=ts, return_clean=True)
    g1 = t1_members(P, attr, n_attr, device, GATE_SEEDS)
    glrt, _ = gaussian_lrt(h, P, attr, subspace_noise_cov(Q, sigma), n_attr,
                           GATE_SEEDS)

    h64 = h.astype(np.float64)
    u_clean = h64 @ Q
    h_perp = h64 - u_clean @ Q.T
    rng = np.random.default_rng(2000 + ts)          # same stream as the coarsen run
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
    raw = tree.apply(h_perp)
    leaves = np.unique(raw[dtr])
    remap = {lf: i for i, lf in enumerate(leaves)}
    true_leaf = np.array([remap.get(l, 0) for l in raw], dtype=np.int64)
    K = len(leaves)
    M = np.stack([np.median(h_perp[dtr][raw[dtr] == lf], axis=0) for lf in leaves])
    return dict(attr=attr, task=task, n_attr=n_attr, n_task=n_task,
                task_maj=task_maj, u_noised=u_noised, M=M, true_leaf=true_leaf,
                K=K, dtr=dtr, dte=dte, u_clean=u_clean, sigma=sigma, r=r,
                gate_t1=max(g1.values()), gate_lrt=glrt)


def eval_point(b, p, device, ts, clean_lift):
    """One (seed, p) point: release, head, Tier-1, flip-aware Tier-2, utility."""
    attr, task, n_attr = b["attr"], b["task"], b["n_attr"]
    K, M, true_leaf = b["K"], b["M"], b["true_leaf"]
    frng = np.random.default_rng(5000 + ts)
    emitted = flip(true_leaf, p, K, frng)           # the released realization
    release = np.concatenate([b["u_noised"], M[emitted]], 1).astype(np.float32)

    head = train_head_rr(b["u_noised"], M, true_leaf, task, b["n_task"],
                         b["dtr"], p, K, device, ts, flip_seed=7000 + ts)
    L = head_logits(head, release, device)
    li = float((L[b["dtr"]].argmax(1) == task[b["dtr"]]).mean()) - b["task_maj"]
    lh = float((L[b["dte"]].argmax(1) == task[b["dte"]]).mean()) - b["task_maj"]

    b_rep = t1_members(release, attr, n_attr, device, PROBE_SEEDS)
    b_out = t1_members(L, attr, n_attr, device, PROBE_SEEDS)

    ncov = (b["sigma"] ** 2) * np.eye(b["r"])
    a_lrt, a_leaf, a_comb, a_joint = [], [], [], []
    for s in PROBE_SEEDS:
        tr_i, te_i = _split_idx(attr, s)
        post_a = _lrt_scores(b["u_clean"][tr_i], attr[tr_i], b["u_noised"],
                             ncov, n_attr)
        post_b = flip_aware_posterior(emitted, attr, tr_i, true_leaf, n_attr, K, p)
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

    t2 = {"lrt_subspace": float(np.mean(a_lrt)), "leaf_posterior": float(np.mean(a_leaf)),
          "combined_llr": float(np.mean(a_comb)), "joint_xgb": float(np.mean(a_joint))}
    return {
        "rep": b_rep, "out": b_out, "t2": t2,
        "tier1_rep_max": max(b_rep.values()), "tier1_out_max": max(b_out.values()),
        "tier2_max": max(t2.values()),
        "util_in": 100.0 * li / clean_lift, "util_held": 100.0 * lh / clean_lift,
    }


def run_cell(key, device, clean_lift):
    ds, at, tk, cfg = CELLS[key]
    print("\n" + "#" * 96 + f"\n# {key} {ds}/{at}/{tk} {cfg}\n" + "#" * 96, flush=True)
    bases, gate_t1, gate_lrt = {}, [], []
    for ts in TRAIN_SEEDS:
        t0 = time.time()
        bases[ts] = build_base(key, device, ts)
        gate_t1.append(bases[ts]["gate_t1"]); gate_lrt.append(bases[ts]["gate_lrt"])
        print(f"  [base seed {ts}] K={bases[ts]['K']} gate T1={gate_t1[-1]:.4f} "
              f"LRT={gate_lrt[-1]:.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)
    K = bases[TRAIN_SEEDS[0]]["K"]

    rows = []
    for p in P_GRID:
        t0 = time.time()
        pts = [eval_point(bases[ts], p, device, ts, clean_lift) for ts in TRAIN_SEEDS]
        m = lambda k: float(np.mean([x[k] for x in pts]))     # noqa: E731
        sd = lambda k: float(np.std([x[k] for x in pts]))     # noqa: E731
        row = {
            "p": p, "eps_leaf": eps_leaf(p, K), "K": K,
            "tier1_rep_max": m("tier1_rep_max"), "tier1_out_max": m("tier1_out_max"),
            "tier2_max": m("tier2_max"),
            "t2_members": {k: float(np.mean([x["t2"][k] for x in pts]))
                           for k in pts[0]["t2"]},
            "rep_members": {k: float(np.mean([x["rep"][k] for x in pts]))
                            for k in pts[0]["rep"]},
            "out_members": {k: float(np.mean([x["out"][k] for x in pts]))
                            for k in pts[0]["out"]},
            "util_in": m("util_in"), "util_in_std": sd("util_in"),
            "util_held": m("util_held"), "util_held_std": sd("util_held"),
            "tier1_pass": bool(m("tier1_rep_max") <= BAR and m("tier1_out_max") <= BAR),
            "tier2_pass": bool(m("tier2_max") <= BAR
                               and m("tier1_rep_max") <= BAR
                               and m("tier1_out_max") <= BAR),
            "minutes": round((time.time() - t0) / 60, 1),
        }
        rows.append(row)
        print(f"  p={p:<4g} eps_leaf={row['eps_leaf']:.2f} | T1 rep/out "
              f"{row['tier1_rep_max']:.3f}/{row['tier1_out_max']:.3f} | T2 "
              f"{row['tier2_max']:.3f} (leaf {row['t2_members']['leaf_posterior']:.3f} "
              f"lrt {row['t2_members']['lrt_subspace']:.3f}) | util "
              f"{row['util_in']:.1f}%/{row['util_held']:.1f}% | "
              f"T1 {'PASS' if row['tier1_pass'] else 'fail'} "
              f"T2 {'PASS' if row['tier2_pass'] else 'fail'} ({row['minutes']} min)",
              flush=True)

    p0 = next(r for r in rows if r["p"] == 0.0)
    g = GATE[key]
    gate_ok = (abs(p0["tier2_max"] - g["t2"]) <= GATE_TOL
               and abs(p0["util_in"] - g["util"]) <= GATE_TOL * 100)
    print(f"  >>> GATE {'PASS' if gate_ok else 'FAIL'}: p=0 T2={p0['tier2_max']:.4f} "
          f"(target {g['t2']}), util={p0['util_in']:.1f}% (target {g['util']})",
          flush=True)
    return {"cell": key, "dataset": ds, "attr": at, "task": tk, "config": cfg,
            "K": K, "clean_lift": clean_lift,
            "gate": {"t1_mean": float(np.mean(gate_t1)),
                     "lrt_mean": float(np.mean(gate_lrt)),
                     "p0_tier2": p0["tier2_max"], "p0_util_in": p0["util_in"],
                     "targets": g, "passed": bool(gate_ok)},
            "sweep": rows}


def score(cells):
    valid = [c for c in cells if c["gate"]["passed"]]
    t2_by_cell = {}
    for c in valid:
        ok = [r for r in c["sweep"] if r["p"] > 0 and r["tier2_pass"]]
        t2_by_cell[c["cell"]] = min([r["p"] for r in ok], default=None)
    p1_cells = [k for k, v in t2_by_cell.items() if v is not None]
    p1 = len(p1_cells) >= 2

    p2_detail, p3_detail = {}, {}
    for c in valid:
        pmin = t2_by_cell[c["cell"]]
        if pmin is None:
            continue
        r = next(x for x in c["sweep"] if x["p"] == pmin)
        p2_detail[c["cell"]] = {"p": pmin, "util_in": r["util_in"],
                                "ge_50": bool(r["util_in"] >= 50.0)}
        p3_detail[c["cell"]] = {"p": pmin, "tier1_rep": r["tier1_rep_max"],
                                "tier1_out": r["tier1_out_max"],
                                "tier1_pass": r["tier1_pass"]}
    p2 = len(p2_detail) >= 2 and sum(v["ge_50"] for v in p2_detail.values()) >= 2
    # P3: Tier-1 passes at EVERY point where Tier-2 passes
    p3_all = True
    for c in valid:
        for r in c["sweep"]:
            if r["tier2_pass"] and not r["tier1_pass"]:
                p3_all = False
    joint = [k for k in p1_cells
             if p2_detail.get(k, {}).get("ge_50") and p3_detail.get(k, {}).get("tier1_pass")]
    return {
        "gate_failures": [c["cell"] for c in cells if not c["gate"]["passed"]],
        "smallest_passing_p": t2_by_cell,
        "P1_exists_p_tier2_ge2_cells": {"cells": p1_cells,
                                        "verdict": "HELD" if p1 else "FALSIFIED"},
        "P2_utility_ge50_at_smallest_p": {"per_cell": p2_detail,
                                          "verdict": "HELD" if p2 else "FALSIFIED"},
        "P3_tier1_passes_where_tier2_passes": {"per_cell": p3_detail,
                                               "verdict": "HELD" if p3_all else "FALSIFIED"},
        "cells_satisfying_all_three": joint,
        "success_criterion_met": bool(len(joint) >= 2 and not [
            c["cell"] for c in cells if not c["gate"]["passed"]]),
    }


def main():
    t0 = time.time()
    device = get_device()
    tt = json.loads((REPO / "results" / "two_tier_certification.json").read_text())
    ttname = {"easy": "hmda/race/loan_decision",
              "middle": "hmda/race/loan_amount_band", "hard": "adult/sex/income"}
    clean = {k: next(c for c in tt["e2e_cells"] if c["cell"] == ttname[k])["clean_lift"]
             for k in CELLS}
    print(f"[setup] device={device} p_grid={P_GRID} train_seeds={TRAIN_SEEDS} "
          f"probe_seeds={PROBE_SEEDS} bar={BAR}", flush=True)

    cells = [run_cell(k, device, clean[k]) for k in CELLS]
    scoring = score(cells)
    print("\n" + "=" * 96 + "\nSCORING (hybrid_rr_prediction.md @ 7b8e956)\n" + "=" * 96)
    for k, v in scoring.items():
        print(f"  {k}: {v}")

    out = {"experiment": "hybrid_rr", "prediction_file": "hybrid_rr_prediction.md",
           "prediction_commit": "7b8e956", "bar": BAR, "p_grid": P_GRID,
           "train_seeds": TRAIN_SEEDS, "probe_seeds": PROBE_SEEDS,
           "gate_seeds": GATE_SEEDS, "clean_lifts": clean,
           "comparators": COMPARATORS,
           "eps_leaf_note": ("eps_leaf is a per-release guarantee on LEAF "
                             "MEMBERSHIP, not on the attribute; it does not "
                             "compose across repeated releases."),
           "cells": cells, "scoring": scoring,
           "minutes": round((time.time() - t0) / 60, 1)}
    (HERE / "hybrid_rr.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {HERE / 'hybrid_rr.json'} ({out['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
