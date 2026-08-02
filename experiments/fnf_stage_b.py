"""FNF gauntlet stage B — score the two-tier battery on the releases exported
by fnf_stage_a.py, in the project venv.

Conventions identical to the FARE run: 3 training seeds, sweep probe seeds
[0,1], certification probe seeds [0,1,2], bar 0.55 on the 3-seed mean, both
surfaces (release z and the retrained-head output), utility normalized by the
cell's clean e2e lift from results/two_tier_certification.json. FNF's release
is deterministic and continuous, so Tier-2 = informed QDA (Sigma_noise = 0),
PLUS FNF's own optimal adversary (their Lemma 5.1: compare exact log-densities
under both trained flows) — the strongest attacker their own theory admits.

Every reading is recorded in BOTH metrics: AUC (our standing bar, comparable
to the rest of Table 2) and BALANCED ACCURACY (the metric their (1+Delta)/2
bound is stated in), so certificate coherence is apples-to-apples.
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# Generated intermediates; gitignored and regenerable.
# Override the location with DG_ARTIFACTS.
ARTIFACTS = Path(os.environ.get("DG_ARTIFACTS", REPO / "analysis"))
sys.path.insert(0, str(REPO))

from experiments.diagnostic import lift, probe_lora, probe_sklearn  # noqa: E402
from experiments.hardening_test import _probe  # noqa: E402
from experiments.two_tier_certification import fmt_point, full_point  # noqa: E402
from utils.battery import _split_idx  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

CELL_DIR = ARTIFACTS / "fnf_cells"
RESULTS = REPO / "results"
BAR = 0.55
SWEEP_SEEDS = [0, 1]
CERT_SEEDS = [0, 1, 2]
FNF_COMMIT = "f57401262dc12b258353b26b308ec02f618433d6"


def balanced_acc(y_true, y_pred):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


def mu_star(logp_a1, logp_a0, attr, seeds):
    """FNF's optimal adversary (Lemma 5.1): predict the group whose encoded
    density at z is higher. Evaluated on the same held-out probe splits."""
    score = logp_a1 - logp_a0
    aucs, baccs = [], []
    from sklearn.metrics import roc_auc_score
    for s in seeds:
        _, te = _split_idx(attr, s)
        aucs.append(float(roc_auc_score(attr[te], score[te])))
        baccs.append(balanced_acc(attr[te], (score[te] > 0).astype(int)))
    return float(np.mean(aucs)), float(np.mean(baccs))


def xgb_bacc(Z, attr, n_attr, seeds):
    """Balanced accuracy of the strongest black-box probe, same splits."""
    out = []
    for s in seeds:
        tr, te = _split_idx(attr, s)
        Zd = np.asarray(Z, dtype=np.float64)
        clf_acc, _ = _probe("XGB", Zd[tr], attr[tr], Zd[te], attr[te], s, n_attr)
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                            objective="binary:logistic", tree_method="hist",
                            eval_metric="logloss", random_state=s, n_jobs=2,
                            verbosity=0)
        clf.fit(Zd[tr], attr[tr])
        out.append(balanced_acc(attr[te], clf.predict(Zd[te])))
    return float(np.mean(out))


def head_logits(Z, task, seed):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    Zd = np.asarray(Z, dtype=np.float64)
    lr.fit(Zd, task)
    return np.log(np.clip(lr.predict_proba(Zd), 1e-12, None)).astype(np.float32)


def score_one(tag, gamma, seed, device, seeds, with_lora, clean_lift, task_maj):
    f = CELL_DIR / f"{tag}_g{gamma}_s{seed}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    Z, attr, task = d["z"], d["attr"], d["task"]
    n_attr = int(attr.max()) + 1
    keep = np.isfinite(Z).all(0) & (Z.std(0) > 1e-12)
    Z = Z[:, keep] if keep.any() else Z[:, :1]
    L = head_logits(Z, task, seed)

    rp = full_point(Z, Z, attr, n_attr, np.zeros((Z.shape[1],) * 2), device,
                    seeds, with_lora=with_lora)
    op = full_point(L, L, attr, n_attr, np.zeros((L.shape[1],) * 2), device,
                    seeds, with_lora=with_lora)
    mu_auc, mu_bacc = mu_star(d["logp_a1"], d["logp_a0"], attr, seeds)
    rep_bacc = xgb_bacc(Z, attr, n_attr, seeds)
    l = lift(L, task, task_maj)
    return {
        "gamma": gamma, "seed": seed, "z_dim": int(Z.shape[1]),
        "rep": rp, "out": op, "mu_star_auc": mu_auc, "mu_star_bacc": mu_bacc,
        "rep_xgb_bacc": rep_bacc,
        "tier1_max": rp["tier1_max"],
        "tier2_max": float(max(rp["tier2_max"], mu_auc)),
        "best_bacc": float(max(rep_bacc, mu_bacc)),
        "lift": float(l), "kept_pct": 100.0 * float(l) / clean_lift,
    }


def main():
    t0 = time.time()
    device = get_device()
    tag = sys.argv[1] if len(sys.argv) > 1 else "hard"
    stage_a = json.loads((CELL_DIR / f"{tag}_stage_a.json").read_text())
    tt = json.loads((RESULTS / "two_tier_certification.json").read_text())
    clean_lift = next(c for c in tt["e2e_cells"]
                      if c["cell"] == "adult/sex/income")["clean_lift"]
    npz0 = np.load(ARTIFACTS / "fare_cells" / "hard.npz")
    task_maj = float(np.bincount(npz0["task"]).max() / len(npz0["task"]))
    gammas = stage_a["gammas"]
    seeds = stage_a["seeds"]
    delta = {(r["gamma"], r["seed"]): r.get("stat_dist") for r in stage_a["rows"]}
    their = {(r["gamma"], r["seed"]): r for r in stage_a["rows"]}

    print(f"[setup] device={device} tag={tag} clean_lift={clean_lift:+.4f} "
          f"task_maj={task_maj:.4f} bar={BAR}", flush=True)

    # ---- sweep at the first training seed -----------------------------------
    print("\n--- sweep (train seed %d, probe seeds %s) ---" % (seeds[0], SWEEP_SEEDS),
          flush=True)
    sweep = []
    for g in gammas:
        r = score_one(tag, g, seeds[0], device, SWEEP_SEEDS, False, clean_lift,
                      task_maj)
        if r is None:
            print(f"  gamma={g}: MISSING (stage A failed)", flush=True)
            continue
        dl = delta.get((g, seeds[0]))
        r["delta"] = dl
        r["bound"] = (1 + dl) / 2 if dl is not None else None
        sweep.append(r)
        print(f"  gamma={g:<5g} rep {fmt_point(r['rep'])} | mu*={r['mu_star_auc']:.3f} "
              f"| T2={r['tier2_max']:.3f} | bacc={r['best_bacc']:.3f} vs bound "
              f"{r['bound'] if r['bound'] is None else round(r['bound'], 3)} "
              f"| kept={r['kept_pct']:.0f}%", flush=True)

    # ---- certify every gamma over all training seeds (cheap enough) ---------
    print(f"\n--- certification (train seeds {seeds}, probe seeds {CERT_SEEDS}) ---",
          flush=True)
    certs = []
    for g in gammas:
        rows = [score_one(tag, g, s, device, CERT_SEEDS, True, clean_lift, task_maj)
                for s in seeds]
        rows = [r for r in rows if r is not None]
        if not rows:
            continue
        m = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
        sd = lambda k: float(np.std([r[k] for r in rows]))  # noqa: E731
        ds = [delta[(g, s)] for s in seeds if delta.get((g, s)) is not None]
        bnd = (1 + float(np.mean(ds))) / 2 if ds else None
        c = {
            "gamma": g, "n_seeds": len(rows),
            "rep_tier1_max": float(np.mean([r["rep"]["tier1_max"] for r in rows])),
            "rep_tier2_max": m("tier2_max"),
            "out_tier1_max": float(np.mean([r["out"]["tier1_max"] for r in rows])),
            "out_tier2_max": float(np.mean([r["out"]["tier2_max"] for r in rows])),
            "mu_star_auc": m("mu_star_auc"), "mu_star_bacc": m("mu_star_bacc"),
            "rep_xgb_bacc": m("rep_xgb_bacc"), "best_bacc": m("best_bacc"),
            "best_bacc_std": sd("best_bacc"),
            "kept_pct": m("kept_pct"), "kept_pct_std": sd("kept_pct"),
            "delta_mean": float(np.mean(ds)) if ds else None, "bound": bnd,
            "their_test_bal_acc": their[(g, seeds[0])].get("test_bal_acc"),
            "their_adv_test_acc": their[(g, seeds[0])].get("adv_test_acc"),
            "per_seed": rows,
        }
        c["tier1_pass"] = bool(c["rep_tier1_max"] <= BAR and c["out_tier1_max"] <= BAR)
        c["tier2_pass"] = bool(c["rep_tier2_max"] <= BAR and c["out_tier2_max"] <= BAR)
        c["exceeds_bound"] = bool(bnd is not None and c["best_bacc"] > bnd)
        certs.append(c)
        print(f"  gamma={g:<5g} rep T1={c['rep_tier1_max']:.3f} T2={c['rep_tier2_max']:.3f} "
              f"| out T1={c['out_tier1_max']:.3f} T2={c['out_tier2_max']:.3f} "
              f"| bacc={c['best_bacc']:.3f} vs bound {bnd if bnd is None else round(bnd, 3)}"
              f"{'  <-- EXCEEDS' if c['exceeds_bound'] else ''} "
              f"| kept={c['kept_pct']:.0f}% | T1 {'PASS' if c['tier1_pass'] else 'FAIL'} "
              f"T2 {'PASS' if c['tier2_pass'] else 'FAIL'}", flush=True)

    # ---- score the registered predictions -----------------------------------
    t1_pass = [c for c in certs if c["tier1_pass"]]
    util_gt60 = [c for c in certs if c["kept_pct"] > 60]
    util_gt40 = [c for c in certs if c["kept_pct"] > 40]
    p1 = {"n_gammas_passing_tier1": len(t1_pass),
          "gammas_passing": [c["gamma"] for c in t1_pass],
          "in_1_to_2_range": 1 <= len(t1_pass) <= 2,
          "passing_all_utility_le_40": all(c["kept_pct"] <= 40 for c in t1_pass),
          "all_high_utility_fail_tier1": all(not c["tier1_pass"] for c in util_gt60)}
    p1["verdict"] = ("HELD" if (p1["in_1_to_2_range"]
                                and p1["passing_all_utility_le_40"]
                                and p1["all_high_utility_fail_tier1"])
                     else "FALSIFIED")
    p2_viol = [c["gamma"] for c in util_gt40 if c["tier2_pass"]]
    p2 = {"gammas_with_util_gt40": [c["gamma"] for c in util_gt40],
          "violations_pass_tier2_with_util_gt40": p2_viol,
          "verdict": "HELD" if not p2_viol else "FALSIFIED"}
    exceed = [c["gamma"] for c in certs if c["exceeds_bound"]]
    p3 = {"gammas_exceeding_bound": exceed,
          "margins": {str(c["gamma"]): round(c["best_bacc"] - c["bound"], 4)
                      for c in certs if c["bound"] is not None},
          "verdict": "HELD" if exceed else "FALSIFIED"}
    scoring = {"P1_tier1_range": p1, "P2_tier2_fails_at_useful_points": p2,
               "P3_measured_exceeds_own_bound": p3}

    print("\n" + "=" * 92 + "\nSCORING (fnf_prediction.md @ 8b7d873)\n" + "=" * 92)
    for k, v in scoring.items():
        print(f"  {k}: {v['verdict']}\n    {v}")

    out = {"experiment": "fnf_gauntlet", "prediction_file": "fnf_prediction.md",
           "prediction_commit": "8b7d873", "fnf_commit": FNF_COMMIT,
           "license": "Apache-2.0", "bar": BAR, "tag": tag,
           "sweep_seeds": SWEEP_SEEDS, "cert_seeds": CERT_SEEDS,
           "train_seeds": seeds, "gammas": gammas, "clean_lift": clean_lift,
           "stage_a": stage_a, "sweep": sweep, "certified": certs,
           "scoring": scoring, "minutes": round((time.time() - t0) / 60, 1)}
    p = RESULTS / f"fnf_gauntlet_{tag}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {p} ({out['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
