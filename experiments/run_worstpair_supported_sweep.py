"""Supported-pair sigma sweep — registered in
worstpair_supported_sweep_prediction.md (commit c084ad3) BEFORE this file.

Identical to run_worstpair_sweep.py except the scoring is worst-pair restricted
to SUPPORTED pairs (both classes >= MIN_CLASS_ROWS rows), whose no-signal null
is 0.5217 rather than the unusable all-pairs 0.5556, and the sigma ceiling is
raised to 8192 because middle/subspace was previously cut off while still
holding +73.1% utility. All-pairs and macro-OVR are recorded alongside.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from experiments.diagnostic import lift, load_cell  # noqa: E402
from run_worstpair_sweep import (  # noqa: E402
    BAR, CELLS, CHANNELS, GATE_TOL, PROBE_SEEDS, TRAIN_SEEDS,
    build, lora_probs, macro_ovr, probe_probs, worst_pair,
)
from utils.battery import _lrt_scores, _split_idx  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

MIN_CLASS_ROWS = 3000
MAX_SIGMA = 8192.0
NULL_SUPPORTED = 0.5217
NULL_ALL_PAIRS = 0.5556


def supported_mask(attr, n_classes):
    c = np.bincount(attr, minlength=n_classes)
    return [bool(v >= MIN_CLASS_ROWS) for v in c], c.tolist()


def worst_pair_supported(y, p, keep):
    k = p.shape[1]
    if k == 2:
        a = float(roc_auc_score(y, p[:, 1]))
        return max(a, 1 - a), (0, 1)
    best, arg = 0.0, None
    for i, j in combinations(range(k), 2):
        if not (keep[i] and keep[j]):
            continue
        m = (y == i) | (y == j)
        if m.sum() < 10 or len(set(y[m])) < 2:
            continue
        den = p[m, i] + p[m, j]
        s = np.where(den > 0, p[m, j] / np.maximum(den, 1e-12), 0.5)
        a = float(roc_auc_score((y[m] == j).astype(int), s))
        a = max(a, 1.0 - a)
        if a > best:
            best, arg = a, (i, j)
    return best, arg


def surface_scores(H, attr, n_attr, device, tier, keep, h_clean=None, ncov=None):
    sup, allp, mac = [], [], []

    def add(y, p):
        sup.append(worst_pair_supported(y, p, keep)[0])
        allp.append(worst_pair(y, p)[0])
        mac.append(macro_ovr(y, p))

    for s in PROBE_SEEDS:
        for arch in ("XGB", "MLP"):
            y, p = probe_probs(arch, H.astype(np.float64), attr, n_attr, s)
            add(y, p)
        y, p = lora_probs(H.astype(np.float32), attr, n_attr, s, device)
        add(y, p)
        if tier == 2 and h_clean is not None:
            tr, te = _split_idx(attr, s)
            post = _lrt_scores(np.asarray(h_clean, np.float64)[tr], attr[tr],
                               np.asarray(H, np.float64)[te], ncov, n_attr)
            add(attr[te], post)
    return max(sup), max(allp), max(mac)


def main():
    t0 = time.time()
    device = get_device()
    tt = json.loads((REPO / "results" / "two_tier_certification.json").read_text())
    name = {"easy": "hmda/race/loan_decision", "middle": "hmda/race/loan_amount_band"}
    clean = {k: next(c for c in tt["e2e_cells"] if c["cell"] == name[k])["clean_lift"]
             for k in CELLS}
    print(f"[setup] device={device} bar={BAR} MIN_CLASS_ROWS={MIN_CLASS_ROWS} "
          f"MAX_SIGMA={MAX_SIGMA:g}")
    print(f"        nulls: all-pairs {NULL_ALL_PAIRS}, supported {NULL_SUPPORTED}\n",
          flush=True)

    # S4: supported-pair label-coupling predictor (label-only)
    pred = {}
    for k, (ds, at, tk) in CELLS.items():
        X, attr, task, n_attr, _, _, _ = load_cell(ds, at, tk)
        keep, counts = supported_mask(attr, n_attr)
        sup, allp = [], []
        for s in PROBE_SEEDS:
            y, p = probe_probs("XGB", task.reshape(-1, 1).astype(np.float64),
                               attr, n_attr, s)
            sup.append(worst_pair_supported(y, p, keep)[0])
            allp.append(worst_pair(y, p)[0])
        pred[k] = dict(supported=float(max(sup)), all_pairs=float(max(allp)),
                       class_counts=counts, kept=[i for i, v in enumerate(keep) if v],
                       exceeds_bar=bool(max(sup) > BAR))
        print(f"[predictor {k}] supported-pair={max(sup):.4f} "
              f"all-pairs={max(allp):.4f} counts={counts} -> "
              f"{'ABOVE' if max(sup) > BAR else 'below'} the bar", flush=True)

    results = []
    for (cellk, chan), cfg in CHANNELS.items():
        ds, at, tk = CELLS[cellk]
        X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk)
        X_t = torch.from_numpy(X).to(device)
        keep, counts = supported_mask(attr, n_attr)
        tier = cfg.get("tier", 1)
        print(f"\n{'#' * 92}\n# {cellk} / {chan} (tier {tier}, start sigma "
              f"{cfg['sigma']:g}, supported classes "
              f"{[i for i, v in enumerate(keep) if v]})\n{'#' * 92}", flush=True)
        sweep, sigma, passing = [], cfg["sigma"], None
        while sigma <= MAX_SIGMA:
            ts = time.time()
            rs, ra, rm, os_, oa, om, lifts = [], [], [], [], [], [], []
            for seed in TRAIN_SEEDS:
                P, L, h, ncov = build(cfg["kind"], cfg, sigma, X_t, attr, task,
                                      n_attr, n_task, device, seed)
                a, b, c = surface_scores(P, attr, n_attr, device, tier, keep, h, ncov)
                rs.append(a); ra.append(b); rm.append(c)
                d = L.shape[1]
                a, b, c = surface_scores(L, attr, n_attr, device, tier, keep, L,
                                         np.zeros((d, d)))
                os_.append(a); oa.append(b); om.append(c)
                lifts.append(lift(L, task, task_maj))
            lm = float(np.mean(lifts))
            row = dict(sigma=sigma,
                       rep_supported=float(max(rs)), rep_allpairs=float(max(ra)),
                       rep_macro=float(max(rm)),
                       out_supported=float(max(os_)), out_allpairs=float(max(oa)),
                       out_macro=float(max(om)),
                       lift_mean=lm, util_pct=100.0 * lm / clean[cellk],
                       abs_lift=lm, abs_acc=task_maj + lm,
                       passes=bool(max(rs) <= BAR and max(os_) <= BAR),
                       minutes=round((time.time() - ts) / 60, 1))
            sweep.append(row)
            print(f"  sigma={sigma:<7g} rep sup={row['rep_supported']:.4f} "
                  f"(all {row['rep_allpairs']:.4f}, macro {row['rep_macro']:.4f}) | "
                  f"out sup={row['out_supported']:.4f} "
                  f"(all {row['out_allpairs']:.4f}) | util={row['util_pct']:+.1f}% "
                  f"lift={lm:+.4f} acc={task_maj + lm:.4f}  "
                  f"{'PASS' if row['passes'] else 'fail'} ({row['minutes']} min)",
                  flush=True)
            if row["passes"]:
                passing = row
                break
            if lm < 0:
                print("  -> utility negative; stopping sweep", flush=True)
                break
            sigma *= 2
        results.append(dict(cell=cellk, channel=chan, tier=tier,
                            stored_sigma=cfg["sigma"], class_counts=counts,
                            supported_classes=[i for i, v in enumerate(keep) if v],
                            clean_lift_insample=clean[cellk], sweep=sweep,
                            passing=passing, passing_exists=bool(passing)))

    S1 = {("easy", "subspace"): True, ("middle", "subspace"): True,
          ("easy", "fullrank_t1"): False, ("middle", "fullrank_t1"): False,
          ("easy", "fullrank_t2"): False, ("middle", "fullrank_t2"): False}
    s1 = {}
    for r in results:
        k = (r["cell"], r["channel"])
        s1["/".join(k)] = dict(predicted=S1[k], actual=r["passing_exists"],
                               correct=bool(S1[k] == r["passing_exists"]))
    s2 = {}
    for r in results:
        if r["passing_exists"]:
            u = r["passing"]["util_pct"]
            lo, hi = (60, 1e9) if (r["cell"] == "middle" and r["channel"] == "subspace") \
                else (20, 60)
            s2["/".join((r["cell"], r["channel"]))] = dict(
                util_pct=u, predicted_range=[lo, None if hi > 1e8 else hi],
                in_range=bool(lo <= u <= hi))
    binding = {}
    for r in results:
        last = r["sweep"][-1]
        binding["/".join((r["cell"], r["channel"]))] = (
            "output" if last["out_supported"] > last["rep_supported"] else "representation")
    scoring = dict(
        S1=dict(per_combo=s1, n_correct=sum(v["correct"] for v in s1.values()),
                n_total=len(s1)),
        S2=dict(per_combo=s2),
        S3=dict(binding_surface=binding,
                output_binds_count=sum(1 for v in binding.values() if v == "output")),
        S4=dict(predictor=pred))
    print("\n" + "=" * 92 + "\nSCORING (worstpair_supported_sweep_prediction.md "
          "@ c084ad3)\n" + "=" * 92)
    print(f"  S1 correct: {scoring['S1']['n_correct']}/{scoring['S1']['n_total']}")
    for k, v in s1.items():
        print(f"    {k:<24} predicted={v['predicted']!s:<5} actual={v['actual']!s:<5} "
              f"{'correct' if v['correct'] else 'WRONG'}")
    print(f"  S2: {s2}")
    print(f"  S3 binding surface: {binding}")
    print(f"  S4 predictor: { {k: round(v['supported'], 4) for k, v in pred.items()} }")

    out = dict(experiment="worstpair_supported_sigma_sweep",
               prediction_file="worstpair_supported_sweep_prediction.md",
               prediction_commit="c084ad3", bar=BAR,
               min_class_rows=MIN_CLASS_ROWS, max_sigma=MAX_SIGMA,
               null_supported=NULL_SUPPORTED, null_all_pairs=NULL_ALL_PAIRS,
               train_seeds=TRAIN_SEEDS, probe_seeds=PROBE_SEEDS,
               clean_lifts=clean, label_coupling_predictor=pred,
               results=results, scoring=scoring,
               minutes=round((time.time() - t0) / 60, 1))
    (RESULTS / "worstpair_supported_sweep.json").write_text(
        json.dumps(out, indent=1, default=float))
    print(f"\n[save] {RESULTS / 'worstpair_supported_sweep.json'} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
