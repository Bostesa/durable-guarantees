"""FARE gauntlet stage A — runs in the FARE venv (/Users/nathansamson/fare/.venv,
py3.9, patched fair-Gini sklearn). Fits the FARE tree over the registered grid
on each exported cell and writes leaf assignments + leaf medians + the FARE
dp_ub certificate per (config, training seed). No attacker measurements here —
those happen in stage B inside the project venv.

Protocol pinned in fare_prediction.md (committed ab76813 before any run):
  grid max_k in {5,10,25,50,100} x alpha in {0.5,0.75,0.9,0.975,0.999},
  min_samples_leaf=100, cat_pos=[] (features are preprocessed floats),
  60/20/20 train/val/test split with numpy seed 42+ts (FARE's own 60:20:20
  convention; tree + medians fit on train, certificate on train/val/test),
  tree random_state 43+ts, ts in {0,1,2}.

Usage: cd /Users/nathansamson/fare/code && ../.venv/bin/python \
  /Users/nathansamson/durable-guarantees/analysis/yus_extras/fare_stage_a.py [cell ...]
(must run from fare/code so `from src.tree...` imports resolve).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/nathansamson/fare/code")
from sklearn.tree import DecisionTreeClassifier  # noqa: E402  (patched build)
from src.tree.alphabeta_adversary import AlphaBetaAdversary  # noqa: E402

HERE = Path("/Users/nathansamson/durable-guarantees/analysis/yus_extras")
CELL_DIR = HERE / "fare_cells"
MAX_KS = [5, 10, 25, 50, 100]
ALPHAS = [0.5, 0.75, 0.9, 0.975, 0.999]
MIN_NI = 100
TRAIN_SEEDS = [0, 1, 2]
ERR_BUDGET = 0.05


def split_60_20_20(n, ts):
    rng = np.random.RandomState(42 + ts)
    perm = rng.permutation(n)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def leaf_medians(X_tr, leaves_tr):
    ids = np.unique(leaves_tr)
    med = np.zeros((len(ids), X_tr.shape[1]), dtype=np.float32)
    for i, cid in enumerate(ids):
        med[i] = np.median(X_tr[leaves_tr == cid], axis=0)
    return ids, med


def dp_ub_cert(k, z, s, tr, va, te):
    """FARE's certificate: AlphaBetaAdversary on the leaf-median embeddings.
    Binary s -> single call; multiclass s -> all-pairs with Bonferroni-split
    budget, exactly as src/tree/main.py's multi branch.

    Their bound code asserts every leaf cell appears in every split of every
    class pair; with sparse minority classes this fails for some (pair, seed).
    We do NOT alter their certificate — if any pair aborts, no valid upper
    bound exists at this config and we return (None, n_failed_pairs);
    verdicts never depend on dp_ub."""
    def one_pair(emb, budget):
        emb = {kk: (v.reshape(-1, 1) if kk.startswith("c_") else v)
               for kk, v in emb.items()}   # adversary asserts s.shape[1] == 1
        adv = AlphaBetaAdversary(k, budget, eps_glob=0.005, eps_ab=0.005,
                                 method="cp", verbose=False)
        ret = adv.ub_demographic_parity(emb)
        return float(ret[1] if len(ret) > 1 else ret[0])

    n_s = int(s.max()) + 1
    if n_s == 2:
        emb = {"z_train": z[tr], "z_val": z[va], "z_test": z[te],
               "c_train": s[tr], "c_val": s[va], "c_test": s[te]}
        try:
            return one_pair(emb, ERR_BUDGET), 0
        except (AssertionError, IndexError):
            return None, 1
    budget = ERR_BUDGET / (n_s * (n_s - 1) / 2)
    total, failed = 0.0, 0
    for i in range(n_s):
        for j in range(i + 1, n_s):
            emb = {}
            for name, idx in (("train", tr), ("val", va), ("test", te)):
                m = (s[idx] == i) | (s[idx] == j)
                c = s[idx][m].copy()
                c[c == i] = -1
                c[c == j] = 1
                c[c == -1] = 0
                emb[f"c_{name}"] = c
                emb[f"z_{name}"] = z[idx][m]
            try:
                total = max(total, one_pair(emb, budget))
            except (AssertionError, IndexError):
                failed += 1
    return (total if failed == 0 else None), failed


def run_one(key, ts, k_max, alpha):
    """One config in one process (the patched sklearn build corrupts memory
    across sequential fits in a single process — verified: config 2 of a loop
    segfaults, the same config in a fresh process is clean). Writes a per-config
    npz + json row; C-level printf noise routed to /dev/null."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)   # the compiled tree printfs write to fd 1 directly
    d = np.load(CELL_DIR / f"{key}.npz")
    X, attr, task = d["X"], d["attr"], d["task"]
    tr, va, te = split_60_20_20(len(attr), ts)
    tag = f"k{k_max}_a{alpha:g}_ts{ts}"
    t0 = time.time()
    T = DecisionTreeClassifier(criterion="fair_gini_dp", max_leaf_nodes=k_max,
                               random_state=43 + ts, min_samples_leaf=MIN_NI)
    T.fit(X[tr], task[tr].reshape(-1, 1), attr[tr].reshape(-1, 1),
          cat_pos=np.asarray([], dtype=np.int32), alpha=alpha)
    leaves_all = T.apply(X).astype(np.int32)
    ids, med = leaf_medians(X[tr], leaves_all[tr])
    lut = {cid: i for i, cid in enumerate(ids)}
    cell_idx = np.array([lut[c] for c in leaves_all], dtype=np.int32)
    z = med[cell_idx]
    acc_tree = float((T.predict(X[te]) == task[te]).mean())
    ub, failed_pairs = dp_ub_cert(len(ids), z, attr, tr, va, te)
    np.savez_compressed(SHARD_DIR / f"{key}_{tag}.npz",
                        cells=cell_idx, medians=med)
    (SHARD_DIR / f"{key}_{tag}.json").write_text(json.dumps(dict(
        tag=tag, ts=ts, max_k=k_max, alpha=alpha, nb_cells=int(len(ids)),
        dp_ub=ub, cert_failed_pairs=failed_pairs, tree_test_acc=acc_tree,
        seconds=round(time.time() - t0, 1))))


SHARD_DIR = CELL_DIR / "shards"


def drive(keys):
    import subprocess
    SHARD_DIR.mkdir(exist_ok=True)
    for key in keys:
        d = np.load(CELL_DIR / f"{key}.npz")
        print(f"\n### {key}: N={len(d['attr'])} d={d['X'].shape[1]} "
              f"n_attr={d['attr'].max()+1} n_task={d['task'].max()+1}",
              flush=True)
        out_npz, manifest = {}, []
        for ts in TRAIN_SEEDS:
            for k_max in MAX_KS:
                for alpha in ALPHAS:
                    tag = f"k{k_max}_a{alpha:g}_ts{ts}"
                    shard = SHARD_DIR / f"{key}_{tag}.json"
                    if not shard.exists():
                        r = subprocess.run(
                            [sys.executable, __file__, "--one", key, str(ts),
                             str(k_max), str(alpha)],
                            cwd="/Users/nathansamson/fare/code",
                            capture_output=True, text=True)
                        if r.returncode != 0:
                            tail = (r.stderr or "")[-300:]
                            manifest.append(dict(tag=tag, ts=ts, max_k=k_max,
                                                 alpha=alpha,
                                                 error=f"rc={r.returncode} {tail}"))
                            print(f"  {tag:<22} ERROR rc={r.returncode} {tail}",
                                  flush=True)
                            continue
                    row = json.loads(shard.read_text())
                    manifest.append(row)
                    sh = np.load(SHARD_DIR / f"{key}_{tag}.npz")
                    out_npz[f"{tag}_cells"] = sh["cells"]
                    out_npz[f"{tag}_medians"] = sh["medians"]
                    ub_s = ("none" if row["dp_ub"] is None
                            else f"{row['dp_ub']:.4f}")
                    print(f"  {tag:<22} cells={row['nb_cells']:>3} "
                          f"dp_ub={ub_s} "
                          f"tree_acc={row['tree_test_acc']:.4f} "
                          f"({row['seconds']:.0f}s)", flush=True)
        np.savez_compressed(CELL_DIR / f"{key}_fare_embeddings.npz", **out_npz)
        (CELL_DIR / f"{key}_fare_manifest.json").write_text(
            json.dumps(dict(cell=key, min_ni=MIN_NI, err_budget=ERR_BUDGET,
                            split="60/20/20 seed 42+ts", rows=manifest),
                       indent=2))
        print(f"[save] {key}_fare_embeddings.npz + manifest", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        run_one(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                float(sys.argv[5]))
    else:
        drive(sys.argv[1:] or ["easy", "middle", "hard"])
