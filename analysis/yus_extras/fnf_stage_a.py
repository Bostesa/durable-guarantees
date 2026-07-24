"""FNF gauntlet stage A — train FNF's official continuous-path encoders on our
hard cell and export the release for stage B.

Runs in the FNF venv (/Users/nathansamson/fnf/.venv, py3.11 + numpy 1.23.5 +
pandas 1.5.3), which cannot import this repo's stack; stage B scores the
battery in the project venv. Same two-stage structure as the FARE run.

Their code is called unmodified: real_nvp_encoder.FlowEncoder,
generative.gmm.train_gmm, train_fnf.train_flow. Only the module-level
`device = 'cuda'` global is retargeted (no CUDA on this machine), exactly what
their own --device flag exists to do.

Preprocessing mirrors their continuous scripts (crime_flow_multi.py): per
feature min-max into [alpha/2, 1-alpha/2] using TRAIN statistics, q = 0, and
the flow consumes logit(x) exactly as their train/eval paths do.

Hyperparameters are their shipped crime settings (scripts/run_crime_gammas.py):
batch 128, n_epochs 60, adv_epochs 60, kl 0..50, lr 1e-2, wd 1e-4, n_blocks 4,
gmm prior with comps (4, 2), p_val 0.01, p_test 0.2, gammas [0,.02,.1,.2,.9].

Exports per (gamma, seed): release z for ALL rows (encoder selected by the
row's own sensitive attribute — FNF requires s at inference), their reported
statistical distance Delta and CSV metrics, and log-densities under both
flows so stage B can run FNF's own optimal adversary (their Lemma 5.1).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

FNF_ROOT = Path("/Users/nathansamson/fnf")
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "fnf_cells"
OUT_DIR.mkdir(exist_ok=True)

os.chdir(FNF_ROOT)                       # their relative paths need repo root
sys.path.insert(0, str(FNF_ROOT))
sys.path.insert(0, "/Users/nathansamson/durable-guarantees")

import utils.fnf_shim as fnf_shim        # noqa: E402
fnf_shim.install("cpu")

import train_fnf                          # noqa: E402
from generative.gmm import train_gmm      # noqa: E402
from train_fnf import get_log_pz, train_flow  # noqa: E402

fnf_shim.retarget_device(("train_fnf",), "cpu")

DEVICE = "cpu"
ALPHA = 0.05
GAMMAS = [0.0, 0.02, 0.1, 0.2, 0.9]
SEEDS = [100, 101, 102]                  # their seed base, 3 training seeds
CRIME = dict(batch_size=128, n_epochs=60, adv_epochs=60, kl_start=0, kl_end=50,
             log_epochs=10, lr=1e-2, weight_decay=1e-4, n_blocks=4,
             gmm_comps1=4, gmm_comps2=2, p_val=0.01, p_test=0.2)


def make_args(gamma, seed, n_flows=1):
    return argparse.Namespace(
        alpha=ALPHA, gamma=gamma, seed=seed, n_flows=n_flows, prior="gmm",
        fair_criterion="stat_parity", scalarization="convex", device=DEVICE,
        with_test=True, train_dec=True, load_prior=False, load_enc=False,
        no_early_stop=False, save_enc=False, save_encoding=False,
        out_file=None, verbose=False, protected_att=None, **CRIME)


def preprocess(X, tr_idx):
    """Their continuous preprocessing: per-feature min-max into
    [alpha/2, 1-alpha/2] using TRAIN statistics (crime_flow_multi.py)."""
    lo = X[tr_idx].min(0)
    hi = X[tr_idx].max(0)
    rng = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    Z = 0.5 + (1 - ALPHA) * ((X - lo) / rng - 0.5)
    return np.clip(Z, ALPHA / 2, 1 - ALPHA / 2).astype(np.float32)


def loaders_for(Xs, ys, prot, idx, bs, shuffle):
    x = torch.from_numpy(Xs[idx]).float().to(DEVICE)
    y = torch.from_numpy(ys[idx]).long().to(DEVICE)
    p = prot[idx]
    l1 = torch.utils.data.DataLoader(TensorDataset(x[p == 1], y[p == 1]),
                                     batch_size=bs, shuffle=shuffle)
    l2 = torch.utils.data.DataLoader(TensorDataset(x[p == 0], y[p == 0]),
                                     batch_size=bs, shuffle=shuffle)
    return l1, l2


def run_one(Xs, attr, task, splits, gamma, seed, out_prefix):
    tr, va, te = splits
    args = make_args(gamma, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = Xs.shape[1]
    q = torch.zeros(1, d).float().to(DEVICE)      # crime convention: quants = 0

    train_loaders = loaders_for(Xs, task, attr, tr, args.batch_size, True)
    valid_loaders = loaders_for(Xs, task, attr, va, 8, True)
    test_loaders = loaders_for(Xs, task, attr, te, 8, True)

    xt = torch.from_numpy(Xs).float().to(DEVICE)
    tt = torch.from_numpy(task).long().to(DEVICE)
    tr1 = xt[tr][attr[tr] == 1]; tr2 = xt[tr][attr[tr] == 0]
    y1 = tt[tr][attr[tr] == 1]; y2 = tt[tr][attr[tr] == 0]
    va1 = xt[va][attr[va] == 1]; va2 = xt[va][attr[va] == 0]
    vy1 = tt[va][attr[va] == 1]; vy2 = tt[va][attr[va] == 0]

    prior1, prior2 = train_gmm(args, q, (tr1, tr2, y1, y2), (va1, va2, vy1, vy2),
                               DEVICE)

    csv_path = FNF_ROOT / f"logs/adult_cell_g{gamma}_s{seed}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=[
            "gamma", "stat_dist", "valid_unbal_acc", "valid_bal_acc",
            "test_unbal_acc", "test_bal_acc", "adv_valid_acc", "adv_test_acc",
            "test_dem_par", "test_eq_0", "test_eq_1"])
        w.writeheader()
    args.out_file = str(csv_path)

    flows = train_flow(args, d, q, prior1, prior2, [50, 50], [100],
                       train_loaders, valid_loaders, test_loaders)

    # --- encode ALL rows; FNF picks the encoder by the row's own attribute ---
    with torch.no_grad():
        x_tf = torch.clamp(xt + q * torch.rand(xt.shape).to(DEVICE),
                           ALPHA / 2, 1 - ALPHA / 2).logit()
        z = torch.zeros(x_tf.shape[0], d, device=DEVICE)
        a1 = torch.from_numpy(attr == 1).to(DEVICE)
        a0 = torch.from_numpy(attr == 0).to(DEVICE)
        if a1.any():
            z[a1] = flows[0][0].inverse(x_tf[a1])[0]
        if a0.any():
            z[a0] = flows[1][0].inverse(x_tf[a0])[0]
        # FNF's own optimal adversary (Lemma 5.1): densities under both flows
        lp1 = get_log_pz(z, prior1[0], flows[0]).cpu().numpy()   # group attr==1
        lp0 = get_log_pz(z, prior2[0], flows[1]).cpu().numpy()   # group attr==0

    row = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            row = {k: (float(v) if v not in ("", None) else None)
                   for k, v in r.items()}

    np.savez_compressed(
        OUT_DIR / f"{out_prefix}_g{gamma}_s{seed}.npz",
        z=z.cpu().numpy().astype(np.float32),
        logp_a1=lp1.astype(np.float64), logp_a0=lp0.astype(np.float64),
        attr=attr.astype(np.int64), task=task.astype(np.int64),
        train_idx=tr.astype(np.int64), test_idx=te.astype(np.int64))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="hard")
    ap.add_argument("--tag", default="hard")
    ap.add_argument("--pca", type=int, default=0,
                    help="if >0, PCA-reduce features to this many dims first "
                         "(best-shot variant for the density estimate)")
    a = ap.parse_args()

    npz = np.load(HERE / "fare_cells" / f"{a.cell}.npz")
    X, attr, task = npz["X"], npz["attr"], npz["task"]
    assert set(np.unique(attr)) <= {0, 1}, "FNF requires binary s"

    rs = np.random.RandomState(0)
    idx = rs.permutation(len(attr))
    n_te = int(CRIME["p_test"] * len(idx))
    n_va = int(CRIME["p_val"] * len(idx))
    te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]

    if a.pca > 0:
        from sklearn.decomposition import PCA
        p = PCA(n_components=a.pca, random_state=0).fit(X[tr])
        X = p.transform(X).astype(np.float32)
        print(f"[pca] reduced to {X.shape[1]} dims, "
              f"evr={p.explained_variance_ratio_.sum():.3f}", flush=True)

    Xs = preprocess(X, tr)
    print(f"[setup] cell={a.cell} N={len(attr)} d={Xs.shape[1]} "
          f"train/valid/test={len(tr)}/{len(va)}/{len(te)} "
          f"p(a=1)={float((attr == 1).mean()):.3f}", flush=True)

    results = []
    for seed in SEEDS:
        for gamma in GAMMAS:
            t0 = time.time()
            try:
                row = run_one(Xs, attr, task, (tr, va, te), gamma, seed, a.tag)
                row.update(gamma=gamma, seed=seed, ok=True,
                           minutes=round((time.time() - t0) / 60, 2))
            except Exception as e:                       # applicability failure
                row = dict(gamma=gamma, seed=seed, ok=False,
                           error=f"{type(e).__name__}: {e}",
                           minutes=round((time.time() - t0) / 60, 2))
                print(f"  !! gamma={gamma} seed={seed} FAILED: {row['error']}",
                      flush=True)
            results.append(row)
            print(f"  [g={gamma} s={seed}] stat_dist={row.get('stat_dist')} "
                  f"test_bal_acc={row.get('test_bal_acc')} "
                  f"adv_test_acc={row.get('adv_test_acc')} "
                  f"({row['minutes']} min)", flush=True)
            (OUT_DIR / f"{a.tag}_stage_a.json").write_text(
                json.dumps({"cell": a.cell, "pca": a.pca, "gammas": GAMMAS,
                            "seeds": SEEDS, "crime_hparams": CRIME,
                            "rows": results}, indent=1))
    print(f"[save] {OUT_DIR / (a.tag + '_stage_a.json')}", flush=True)


if __name__ == "__main__":
    main()
