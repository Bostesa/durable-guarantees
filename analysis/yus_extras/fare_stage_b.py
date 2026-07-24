"""FARE gauntlet stage B — runs in the project venv. Scores FARE's leaf-median
embeddings (produced by fare_stage_a.py in the FARE venv) under the identical
two-tier protocol used for the other published baselines in
experiments/baseline_gauntlet.py:

  * exposure: P = z (leaf-median embedding, all rows), L = log-probs of a
    logistic head retrained on z (LEACE-row convention), ncov = 0 (deterministic
    channel; the Tier-2 LRT degenerates to an informed QDA).
  * sweep at training seed 0 (probe seeds [0,1]; LoRA only when XGB/MLP <= bar),
    then per tier certify at most 2 best-utility candidates over training seeds
    [0,1,2] with probe seeds [0,1,2] — rep-gated verdicts at bar 0.55, same as
    run_baseline/tier_pick.
  * FARE's own dp_ub certificate is carried through from stage A and correlated
    with the attacker readings (registered claim 4).

Usage: PCRL_ROOT=... .venv/bin/python analysis/yus_extras/fare_stage_b.py
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.diagnostic import (  # noqa: E402
    lift, load_cell, probe_lora, probe_sklearn,
)
from experiments.targeted_noise import utility_through  # noqa: E402
from experiments.two_tier_certification import fmt_point, full_point  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

HERE = Path(__file__).resolve().parent
CELL_DIR = HERE / "fare_cells"
RESULTS_DIR = REPO_ROOT / "results"
BAR = 0.55
SWEEP_SEEDS = [0, 1]
CERT_SEEDS = [0, 1, 2]
TRAIN_SEEDS = [0, 1, 2]
MAX_CERT_CANDIDATES = 2

CELLS = [("easy", "hmda", "race", "loan_decision"),
         ("middle", "hmda", "race", "loan_amount_band"),
         ("hard", "adult", "sex", "income")]


def load_embeddings(key):
    npz = np.load(CELL_DIR / f"{key}_fare_embeddings.npz")
    man = json.loads((CELL_DIR / f"{key}_fare_manifest.json").read_text())
    return npz, {r["tag"]: r for r in man["rows"] if "error" not in r}


def exposure(npz, tag, task, seed):
    """P = leaf-median z, L = retrained logistic head log-probs (LEACE row)."""
    from sklearn.linear_model import LogisticRegression
    cells = npz[f"{tag}_cells"]
    med = npz[f"{tag}_medians"]
    P = med[cells].astype(np.float32)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                            random_state=seed)
    lr.fit(P.astype(np.float64), task)
    L = np.log(np.clip(lr.predict_proba(P.astype(np.float64)), 1e-12, None)
               ).astype(np.float32)
    return P, L


def measure(P, L, ctx, device, probe_seeds, with_lora_gate=True):
    attr, task = ctx["attr"], ctx["task"]
    n_attr, task_maj = ctx["n_attr"], ctx["task_maj"]
    d = P.shape[1]
    pt = full_point(P, P, attr, n_attr, np.zeros((d, d)), device, probe_seeds,
                    with_lora=not with_lora_gate)
    if with_lora_gate and max(pt["xgb"], pt["mlp"]) <= BAR:
        pt["lora"], _ = probe_lora(P, attr, n_attr, probe_seeds, device)
        pt["tier1_max"] = float(max(pt["xgb"], pt["mlp"], pt["lora"]))
        pt["tier2_max"] = float(max(pt["tier1_max"], pt["lrt"]))
    out_x, _ = probe_sklearn("XGB", L, attr, n_attr, probe_seeds)
    out_m, _ = probe_sklearn("MLP", L, attr, n_attr, probe_seeds)
    pt["out_xgb"], pt["out_mlp"] = out_x, out_m
    pt["out_max"] = float(max(out_x, out_m))
    pt["lift_own"] = lift(L, task, task_maj)
    pt["lift_lr"] = utility_through(P, task, lambda X, rng: X, probe_seeds,
                                    task_maj)
    pt["lift_best"] = float(max(pt["lift_own"], pt["lift_lr"]))
    return pt


def certify(npz, cfg_tags, ctx, device):
    """3-training-seed certification (full battery incl. LoRA on rep and
    output), mirroring baseline_gauntlet.certify."""
    keys = ("xgb", "mlp", "lora", "lrt")
    per = {f"rep_{k}": [] for k in keys}
    per.update({f"out_{k}": [] for k in keys})
    per["lift_own"], per["lift_lr"] = [], []
    for ts in TRAIN_SEEDS:
        P, L = exposure(npz, cfg_tags[ts], ctx["task"], seed=ts)
        d = P.shape[1]
        rp = full_point(P, P, ctx["attr"], ctx["n_attr"], np.zeros((d, d)),
                        device, CERT_SEEDS)
        d_out = L.shape[1]
        op = full_point(L, L, ctx["attr"], ctx["n_attr"],
                        np.zeros((d_out, d_out)), device, CERT_SEEDS)
        for k in keys:
            per[f"rep_{k}"].append(rp[k])
            per[f"out_{k}"].append(op[k])
        per["lift_own"].append(lift(L, ctx["task"], ctx["task_maj"]))
        per["lift_lr"].append(utility_through(P, ctx["task"],
                                              lambda X, rng: X, CERT_SEEDS,
                                              ctx["task_maj"]))
        print(f"      [ts {ts}] rep {fmt_point(rp)} | out T2={op['tier2_max']:.3f}"
              f" | lift own={per['lift_own'][-1]:+.4f} "
              f"LR={per['lift_lr'][-1]:+.4f}", flush=True)
    c = {f"{k}_mean": float(np.mean(v)) for k, v in per.items()}
    c.update({f"{k}_std": float(np.std(v)) for k, v in per.items()})
    c["per_seed"] = per
    c["rep_tier1_max"] = float(max(c[f"rep_{k}_mean"] for k in ("xgb", "mlp", "lora")))
    c["rep_tier2_max"] = float(max(c["rep_tier1_max"], c["rep_lrt_mean"]))
    c["out_tier1_max"] = float(max(c[f"out_{k}_mean"] for k in ("xgb", "mlp", "lora")))
    c["out_tier2_max"] = float(max(c["out_tier1_max"], c["out_lrt_mean"]))
    c["lift_best"] = float(max(c["lift_own_mean"], c["lift_lr_mean"]))
    return c


def spearman(a, b):
    from scipy.stats import spearmanr
    r, p = spearmanr(a, b)
    return float(r), float(p)


def run_cell(key, ds, attr_name, task_name, clean_lift, device):
    print("\n" + "#" * 96 + f"\n# FARE GAUNTLET: {key} {ds}/{attr_name}/{task_name}"
          f" (clean_lift {clean_lift:+.4f})\n" + "#" * 96, flush=True)
    t0 = time.time()
    X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, attr_name, task_name)
    ctx = dict(attr=attr, task=task, n_attr=n_attr, task_maj=task_maj)
    npz, man = load_embeddings(key)

    # ---- sweep at ts=0 ----
    rows = []
    for tag, m in sorted(man.items()):
        if m["ts"] != 0:
            continue
        P, L = exposure(npz, tag, task, seed=0)
        pt = measure(P, L, ctx, device, SWEEP_SEEDS)
        pt.update(tag=tag, max_k=m["max_k"], alpha=m["alpha"],
                  nb_cells=m["nb_cells"], dp_ub=m["dp_ub"])
        rows.append(pt)
        print(f"  k={m['max_k']:>3} α={m['alpha']:<5g} cells={m['nb_cells']:>3} "
              f"dp_ub={m['dp_ub']:.3f} {fmt_point(pt)} out={pt['out_max']:.3f} "
              f"lift={pt['lift_best']:+.4f}", flush=True)

    # ---- claim-4 correlation on the ts=0 grid ----
    rho, pval = spearman([r["dp_ub"] for r in rows],
                         [r["tier1_max"] for r in rows])

    # ---- tier_pick, mirroring run_baseline ----
    cert_cache = {}

    def tags_all_ts(row):
        return {ts: f"k{row['max_k']}_a{row['alpha']:g}_ts{ts}"
                for ts in TRAIN_SEEDS}

    def tier_pick(tkey):
        cands = sorted([r for r in rows if r[f"{tkey}_max"] <= BAR],
                       key=lambda r: -r["lift_best"])
        tried = []
        for r in cands[:MAX_CERT_CANDIDATES]:
            if r["tag"] not in cert_cache:
                print(f"    [certify {r['tag']} over ts={TRAIN_SEEDS}]", flush=True)
                cert_cache[r["tag"]] = certify(npz, tags_all_ts(r), ctx, device)
                cert_cache[r["tag"]].update(tag=r["tag"], max_k=r["max_k"],
                                            alpha=r["alpha"], dp_ub=r["dp_ub"])
            c = cert_cache[r["tag"]]
            tried.append(r["tag"])
            if c[f"rep_{tkey}_max"] <= BAR:
                return dict(certified=True, cert=c, tried=tried)
        best = min(rows, key=lambda r: r[f"{tkey}_max"]) if rows else None
        return dict(certified=False, cert=None, tried=tried,
                    closest={k: v for k, v in best.items()} if best else None)

    t1, t2 = tier_pick("tier1"), tier_pick("tier2")
    for nm, t in (("TIER 1", t1), ("TIER 2", t2)):
        if t["certified"]:
            c = t["cert"]
            print(f"  >>> FARE {nm}: CERTIFIES at {c['tag']} — lift "
                  f"{c['lift_best']:+.4f} ({100*c['lift_best']/clean_lift:.0f}%), "
                  f"rep T1={c['rep_tier1_max']:.3f} T2={c['rep_tier2_max']:.3f}, "
                  f"out T2={c['out_tier2_max']:.3f}, dp_ub={c['dp_ub']:.3f}",
                  flush=True)
        else:
            b = t["closest"]
            print(f"  >>> FARE {nm}: does NOT certify — closest {b['tag']} "
                  f"{tkey_of(nm)}={b['tier1_max' if nm == 'TIER 1' else 'tier2_max']:.3f} "
                  f"at lift {b['lift_best']:+.4f}", flush=True)
    return dict(cell=f"{ds}/{attr_name}/{task_name}", key=key,
                clean_lift=clean_lift, rows=rows,
                spearman_dpub_tier1=dict(rho=rho, p=pval),
                tier1=t1, tier2=t2, certs=cert_cache,
                minutes=round((time.time() - t0) / 60, 1))


def tkey_of(nm):
    return "tier1_max" if nm == "TIER 1" else "tier2_max"


def main():
    device = get_device()
    bg = json.loads((RESULTS_DIR / "baseline_gauntlet.json").read_text())
    clean = {c["cell"]: c["clean_lift"] for c in bg["cells"]}
    print(f"[setup] device={device} bar≤{BAR} sweep_seeds={SWEEP_SEEDS} "
          f"cert_seeds={CERT_SEEDS} train_seeds={TRAIN_SEEDS}", flush=True)
    out = {"experiment": "fare_gauntlet", "bar": BAR,
           "prediction_commit": "ab76813",
           "fare_commit": "89cb1b66ed268c16659cbf7428c43e60da2df641",
           "grid": "max_k {5,10,25,50,100} x alpha {0.5,0.75,0.9,0.975,0.999}, "
                   "min_ni=100"}
    out["cells"] = [run_cell(key, ds, a, t, clean[f"{ds}/{a}/{t}"], device)
                    for key, ds, a, t in CELLS]

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "per_seed"}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o

    (HERE / "fare_gauntlet.json").write_text(json.dumps(out, indent=2))
    print(f"\n[save] {HERE / 'fare_gauntlet.json'}", flush=True)


if __name__ == "__main__":
    main()
