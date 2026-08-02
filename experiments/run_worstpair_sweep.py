"""Worst-pair sigma sweep on the two HMDA/race cells.

Registered in worstpair_sweep_prediction.md (commit 50d7537) BEFORE this file
existed.

Scoring rule for this run: WORST-PAIR AUC — max over label pairs (i, j) of the
binary AUC on the subset with true label in {i, j}, scored by p_j/(p_i+p_j) and
taken orientation-free — computed on BOTH the representation and the output
surface, and taken as the MAX over seeds (3 training x 3 probe), not the mean.
Macro-OVR is reported alongside at the same seeds for comparison.

Sweep: sigma doubles from each channel's stored operating point, with k and
lambda held fixed, until worst-pair <= 0.55 on both surfaces or utility goes
negative. Reproduction gate at the stored operating point (first sweep point)
within 0.01 before the verdict for that cell is admitted.

Local MPS; macOS OpenMP guard: xgboost before torch.
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

from experiments.diagnostic import _split, lift, load_cell, probe_lora  # noqa: E402
from experiments.end_to_end_surgical import train_e2e_surgical  # noqa: E402
from experiments.hardening_test import _probe  # noqa: E402
from experiments.mi_ceiling import train_channel  # noqa: E402
from utils.battery import _lrt_scores, _split_idx, iso_noise_cov, subspace_noise_cov  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

BAR = 0.55
TRAIN_SEEDS = [0, 1, 2]
PROBE_SEEDS = [0, 1, 2]
MAX_SIGMA = 2048.0
GATE_TOL = 0.01

CELLS = {"easy": ("hmda", "race", "loan_decision"),
         "middle": ("hmda", "race", "loan_amount_band")}
# stored operating points and the stored reading the gate must reproduce
CHANNELS = {
    ("easy", "subspace"): dict(kind="subspace", sigma=16.0, rank=32, lam=10.0,
                               gate_key="tier1_macro", gate_val=0.5094),
    ("middle", "subspace"): dict(kind="subspace", sigma=16.0, rank=16, lam=100.0,
                                 gate_key="tier1_macro", gate_val=0.5349),
    ("easy", "fullrank_t1"): dict(kind="fullrank", sigma=8.0, tier=1,
                                  gate_key="tier1_macro", gate_val=0.5351),
    ("middle", "fullrank_t1"): dict(kind="fullrank", sigma=12.0, tier=1,
                                    gate_key="tier1_macro", gate_val=0.5408),
    ("easy", "fullrank_t2"): dict(kind="fullrank", sigma=20.0, tier=2,
                                  gate_key="tier2_macro", gate_val=0.5377),
    ("middle", "fullrank_t2"): dict(kind="fullrank", sigma=24.0, tier=2,
                                    gate_key="tier2_macro", gate_val=0.5496),
}


# --------------------------------------------------------------------------- #
def macro_ovr(y, p):
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))


def worst_pair(y, p):
    k = p.shape[1]
    if k == 2:
        a = float(roc_auc_score(y, p[:, 1]))
        return max(a, 1 - a), (0, 1)
    best, arg = 0.0, None
    for i, j in combinations(range(k), 2):
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


def probe_probs(arch, H, y, n_classes, seed):
    """Held-out per-class probabilities from one probe at one split seed."""
    Xtr, Xte, ytr, yte = _split(H, y, seed)
    if arch in ("XGB", "MLP"):
        from sklearn.neural_network import MLPClassifier
        from xgboost import XGBClassifier
        if arch == "MLP":
            clf = MLPClassifier(hidden_layer_sizes=(256, 256), activation="relu",
                                alpha=1e-4, solver="adam", learning_rate_init=1e-3,
                                max_iter=100, early_stopping=True,
                                validation_fraction=0.1, n_iter_no_change=8,
                                random_state=seed, batch_size=256)
        else:
            obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
            clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                                objective=obj, tree_method="hist",
                                eval_metric="logloss", random_state=seed,
                                n_jobs=2, verbosity=0)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)
        return yte, (p if p.shape[1] == n_classes else
                     np.column_stack([1 - p.ravel(), p.ravel()]))
    raise ValueError(arch)


def lora_probs(H, y, n_classes, seed, device):
    """rank-32 ReLU LoRA probe, per-class probabilities on the held-out split."""
    import torch.nn as nn
    from experiments.falsification_attack import FreshLoRAAttacker
    Xtr, Xte, ytr, yte = _split(H, y, seed)
    torch.manual_seed(seed)
    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    Xte_t = torch.from_numpy(Xte).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    atk = FreshLoRAAttacker(repr_dim=H.shape[1], n_classes=n_classes,
                            rank=32, alpha=64.0).to(device)
    opt = torch.optim.Adam(atk.parameters(), lr=1e-2)
    ce = nn.CrossEntropyLoss()
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = Xtr_t.shape[0]
    atk.train()
    for _ in range(300):
        idx = torch.randint(0, n, (min(512, n),), generator=g).to(device)
        loss = ce(atk(Xtr_t[idx]), ytr_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    atk.eval()
    with torch.no_grad():
        lg = atk(Xte_t).cpu().numpy()
    z = lg - lg.max(1, keepdims=True)
    return yte, np.exp(z) / np.exp(z).sum(1, keepdims=True)


def surface_scores(H, attr, n_attr, device, tier, h_clean=None, ncov=None):
    """max-over-seeds worst-pair and macro-OVR over the tier's attacker suite."""
    wp, mo = [], []
    for s in PROBE_SEEDS:
        for arch in ("XGB", "MLP"):
            yte, p = probe_probs(arch, H.astype(np.float64), attr, n_attr, s)
            wp.append(worst_pair(yte, p)[0]); mo.append(macro_ovr(yte, p))
        yte, p = lora_probs(H.astype(np.float32), attr, n_attr, s, device)
        wp.append(worst_pair(yte, p)[0]); mo.append(macro_ovr(yte, p))
        if tier == 2 and h_clean is not None:
            tr, te = _split_idx(attr, s)
            post = _lrt_scores(np.asarray(h_clean, np.float64)[tr], attr[tr],
                               np.asarray(H, np.float64)[te], ncov, n_attr)
            wp.append(worst_pair(attr[te], post)[0])
            mo.append(macro_ovr(attr[te], post))
    return max(wp), max(mo)


def build(kind, cfg, sigma, X_t, attr, task, n_attr, n_task, device, seed):
    if kind == "subspace":
        P, L, h, Q = train_e2e_surgical(X_t, attr, task, n_attr, n_task,
                                        rank=cfg["rank"], sigma=sigma,
                                        lam=cfg["lam"], device=device, seed=seed,
                                        return_clean=True)
        return P, L, h, subspace_noise_cov(Q, sigma)
    h, P, L = train_channel(X_t, attr, task, n_attr, n_task, sigma, device,
                            seed=seed)
    return P, L, h, iso_noise_cov(P.shape[1], sigma)


def main():
    t0 = time.time()
    device = get_device()
    tt = json.loads((REPO / "results" / "two_tier_certification.json").read_text())
    name = {"easy": "hmda/race/loan_decision", "middle": "hmda/race/loan_amount_band"}
    clean = {k: next(c for c in tt["e2e_cells"] if c["cell"] == name[k])["clean_lift"]
             for k in CELLS}
    fp = json.loads((RESULTS / "fresh_partition_generalization.json").read_text())
    fpc = {c["cell"]: c for c in fp["cells"]}
    print(f"[setup] device={device} bar={BAR} train_seeds={TRAIN_SEEDS} "
          f"probe_seeds={PROBE_SEEDS} (max over both)\n", flush=True)

    # worst-pair label-coupling predictor (label-only; P3's decisive quantity)
    pred = {}
    for k, (ds, at, tk) in CELLS.items():
        X, attr, task, n_attr, _, _, _ = load_cell(ds, at, tk)
        wp, mo = [], []
        for s in PROBE_SEEDS:
            yte, p = probe_probs("XGB", task.reshape(-1, 1).astype(np.float64),
                                 attr, n_attr, s)
            wp.append(worst_pair(yte, p)[0]); mo.append(macro_ovr(yte, p))
        pred[k] = dict(worst_pair=float(max(wp)), macro_ovr=float(np.mean(mo)),
                       exceeds_bar=bool(max(wp) > BAR))
        print(f"[label-coupling predictor] {k}: worst-pair={max(wp):.4f} "
              f"macro={np.mean(mo):.4f} -> "
              f"{'ABOVE' if max(wp) > BAR else 'below'} the bar", flush=True)

    results = []
    for (cellk, chan), cfg in CHANNELS.items():
        ds, at, tk = CELLS[cellk]
        X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk)
        X_t = torch.from_numpy(X).to(device)
        tier = cfg.get("tier", 1)
        maj = fpc[name[cellk]]["task_maj_eval"]
        print(f"\n{'#' * 92}\n# {cellk} / {chan}  (tier {tier}, start sigma "
              f"{cfg['sigma']:g})\n{'#' * 92}", flush=True)
        sweep, sigma, passing = [], cfg["sigma"], None
        while sigma <= MAX_SIGMA:
            ts0 = time.time()
            rep_wp, rep_mo, out_wp, out_mo, lifts = [], [], [], [], []
            for seed in TRAIN_SEEDS:
                P, L, h, ncov = build(cfg["kind"], cfg, sigma, X_t, attr, task,
                                      n_attr, n_task, device, seed)
                a, b = surface_scores(P, attr, n_attr, device, tier, h, ncov)
                rep_wp.append(a); rep_mo.append(b)
                d = L.shape[1]
                a, b = surface_scores(L, attr, n_attr, device, tier, L,
                                      np.zeros((d, d)))
                out_wp.append(a); out_mo.append(b)
                lifts.append(lift(L, task, task_maj))
            lm = float(np.mean(lifts))
            row = dict(sigma=sigma,
                       rep_worstpair=float(max(rep_wp)), rep_macro=float(max(rep_mo)),
                       out_worstpair=float(max(out_wp)), out_macro=float(max(out_mo)),
                       lift_mean=lm, util_pct=100.0 * lm / clean[cellk],
                       abs_lift=lm, abs_acc=task_maj + lm,
                       passes=bool(max(rep_wp) <= BAR and max(out_wp) <= BAR),
                       minutes=round((time.time() - ts0) / 60, 1))
            sweep.append(row)
            print(f"  sigma={sigma:<7g} rep wp={row['rep_worstpair']:.4f} "
                  f"(macro {row['rep_macro']:.4f})  out wp={row['out_worstpair']:.4f} "
                  f"(macro {row['out_macro']:.4f})  util={row['util_pct']:+.1f}% "
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

        gate_row = sweep[0]
        gate_meas = gate_row["rep_macro"] if cfg["gate_key"] == "tier1_macro" \
            else max(gate_row["rep_macro"], gate_row["out_macro"])
        gate_ok = abs(gate_meas - cfg["gate_val"]) <= GATE_TOL
        print(f"  >>> gate at stored sigma {cfg['sigma']:g}: measured "
              f"{gate_meas:.4f} vs stored {cfg['gate_val']:.4f} -> "
              f"{'PASS' if gate_ok else 'FAIL'}", flush=True)
        results.append(dict(cell=cellk, channel=chan, tier=tier,
                            stored_sigma=cfg["sigma"], config=cfg,
                            clean_lift_insample=clean[cellk],
                            majority=maj,
                            gate=dict(measured=gate_meas, stored=cfg["gate_val"],
                                      tol=GATE_TOL, passed=bool(gate_ok)),
                            sweep=sweep, passing=passing,
                            passing_exists=bool(passing is not None)))

    # ---- scoring ---------------------------------------------------------
    P1_pred = {("easy", "subspace"): True, ("easy", "fullrank_t1"): True,
               ("easy", "fullrank_t2"): True, ("middle", "subspace"): False,
               ("middle", "fullrank_t1"): False, ("middle", "fullrank_t2"): False}
    p1 = {}
    for r in results:
        k = (r["cell"], r["channel"])
        p1["/".join(k)] = dict(predicted=P1_pred[k], actual=r["passing_exists"],
                               correct=bool(P1_pred[k] == r["passing_exists"]))
    p1_correct = sum(v["correct"] for v in p1.values())
    p2 = {}
    for r in results:
        if r["passing_exists"]:
            u = r["passing"]["util_pct"]
            lo, hi = (0, 25) if r["channel"] == "subspace" else (0, 20)
            p2["/".join((r["cell"], r["channel"]))] = dict(
                util_pct=u, predicted_range=[lo, hi],
                in_range=bool(lo <= u <= hi))
    scoring = dict(
        P1=dict(per_combo=p1, n_correct=p1_correct, n_total=len(p1),
                verdict="HELD" if p1_correct == len(p1) else "PARTIAL"),
        P2=dict(per_combo=p2,
                verdict=("HELD" if p2 and all(v["in_range"] for v in p2.values())
                         else ("FALSIFIED" if p2 else "NOT APPLICABLE"))),
        P3=dict(predictor_worstpair=pred,
                both_cells_above_bar=bool(all(v["exceeds_bar"] for v in pred.values())),
                verdict=("HELD" if all(v["exceeds_bar"] for v in pred.values())
                         else "FALSIFIED")),
        gate_failures=[f"{r['cell']}/{r['channel']}" for r in results
                       if not r["gate"]["passed"]])
    print("\n" + "=" * 92 + "\nSCORING (worstpair_sweep_prediction.md @ 50d7537)\n"
          + "=" * 92)
    for k, v in scoring.items():
        print(f"  {k}: {v if isinstance(v, list) else v.get('verdict', v)}")

    out = dict(experiment="worstpair_sigma_sweep",
               prediction_file="worstpair_sweep_prediction.md",
               prediction_commit="50d7537", bar=BAR,
               train_seeds=TRAIN_SEEDS, probe_seeds=PROBE_SEEDS,
               scoring_rule="worst-pair AUC, max over all seeds, both surfaces",
               clean_lifts=clean, label_coupling_predictor=pred,
               results=results, scoring=scoring,
               minutes=round((time.time() - t0) / 60, 1))
    (RESULTS / "worstpair_sweep.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {RESULTS / 'worstpair_sweep.json'} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
