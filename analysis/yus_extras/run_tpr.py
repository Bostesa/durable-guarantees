"""Score-capture re-runs for TPR@low-FPR (worst-case leakage) — Job 2.

New file; NO existing experiment script, figure, or result file is modified.
Everything under analysis/yus_extras/.

Reproduces the stored pipeline exactly and captures per-example attacker scores
(held-out class posteriors) + labels on both surfaces (representation and
logits), per (cell, point, train seed, probe seed, attacker):

  A  Unprotected baseline    sealed_channel.train_model(sigma=0), 3 headline cells
  B  Subspace Tier-1 points  train_e2e_surgical: easy r=32 s=16 lam=10,
                             middle r=16 s=16 lam=100, hard r=32 s=16 lam=100
                             (the sigma=16 alternative)
  C  Full-rank points        mi_ceiling.train_channel: T1 sigma=8/12/32,
                             T2 sigma=20/24/64 (+ channel-aware Gaussian LRT on rep)
  D  Adult floor points      train_model at the first-hidden sigmas from Job 1's
                             longer grid: income sigma=32, occupation_group sigma=16
  E  Averaging attack        middle + hard subspace points, N in {1,4,8,16},
                             nested draws, reusing the section-B trained models
                             (AUC per N; protocol of experiments/averaging_attack.py)

Five training seeds [0..4] per point throughout. Probe protocol matched to each
point's stored pipeline: sections A/B/D/E probe seeds [0,1]
(end_to_end_surgical.PROBE_SEEDS / continuous_cost SWEEP_PROBE_SEEDS); section C
probe seeds [0,1,2] (two_tier CERT_SEEDS). 75/25 attribute-stratified splits via
diagnostic._split; attacker configs copied verbatim from
hardening_test._probe / diagnostic.probe_lora / utils.battery._lrt_scores; a
single fresh noise draw per input (the training functions' own convention).

NOTE on the 60k/seed-42 subsample rule: that cap belongs to the EXPANSION
pipeline (expansion_pipeline.MAX_N). The stored numbers for every point above
were measured on the FULL train partitions (adult 24,145 / hmda 63,747), so this
script uses the full partitions too — applying the cap to HMDA would break the
Job-3 reproduction gate against the stored JSONs.

Outputs: analysis/yus_extras/tpr_scores/<point>.npz (compressed scores+labels),
analysis/yus_extras/tpr_capture.json (AUCs, TPR@FPR, reproduction gate, config).
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.diagnostic import (  # noqa: E402
    LORA_ALPHA, LORA_RANK, LORA_STEPS, _split, lift, load_cell,
)
from experiments.end_to_end_surgical import train_e2e_surgical  # noqa: E402
from experiments.falsification_attack import FreshLoRAAttacker  # noqa: E402
from experiments.mi_ceiling import train_channel  # noqa: E402
from experiments.sealed_channel import train_model  # noqa: E402
from utils.battery import _lrt_scores, _split_idx, iso_noise_cov  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

HERE = Path(__file__).resolve().parent
SCORE_DIR = HERE / "tpr_scores"
SCORE_DIR.mkdir(exist_ok=True)
RESULTS = REPO_ROOT / "results"

BAR = 0.55
TRAIN_SEEDS = [0, 1, 2, 3, 4]
PS_2 = [0, 1]        # sections A/B/D/E (e2e PROBE_SEEDS / SWEEP_PROBE_SEEDS)
PS_3 = [0, 1, 2]     # section C (two_tier CERT_SEEDS)
FPRS = [0.01, 0.05]
NS = [1, 4, 8, 16]
DRAW_SEED = 20260723  # same constant as experiments/averaging_attack.py
TOL = 0.01            # Job-3 reproduction gate

CELLS = {"easy": ("hmda", "race", "loan_decision"),
         "middle": ("hmda", "race", "loan_amount_band"),
         "hard": ("adult", "sex", "income")}
SUBSPACE = {"easy": dict(rank=32, sigma=16.0, lam=10.0),
            "middle": dict(rank=16, sigma=16.0, lam=100.0),
            "hard": dict(rank=32, sigma=16.0, lam=100.0)}   # sigma=16 alternative
FULLRANK = {"easy": dict(tier1=8.0, tier2=20.0),
            "middle": dict(tier1=12.0, tier2=24.0),
            "hard": dict(tier1=32.0, tier2=64.0)}
FLOOR_POINTS = [("adult", "sex", "income", 32.0),
                ("adult", "sex", "occupation_group", 16.0)]


# --------------------------------------------------------------------------- #
#  Scoring attackers — configs copied verbatim from the stored pipeline        #
# --------------------------------------------------------------------------- #
def _clf(arch, seed, n_classes):
    """hardening_test._probe's classifier configs, verbatim."""
    if arch == "MLP":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(256, 256), activation="relu",
                             alpha=1e-4, solver="adam", learning_rate_init=1e-3,
                             max_iter=100, early_stopping=True,
                             validation_fraction=0.1, n_iter_no_change=8,
                             random_state=seed, batch_size=256)
    from xgboost import XGBClassifier
    obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
    return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                         objective=obj, tree_method="hist", eval_metric="logloss",
                         random_state=seed, n_jobs=2, verbosity=0)


def _auc(yte, prob, n_classes):
    from sklearn.metrics import roc_auc_score
    if n_classes == 2:
        return float(roc_auc_score(yte, prob[:, 1]))
    return float(roc_auc_score(yte, prob, multi_class="ovr", average="macro"))


def score_sklearn(arch, H, y, n_classes, seed):
    """One probe seed of diagnostic.probe_sklearn, returning per-example probs."""
    Xtr, Xte, ytr, yte = _split(H, y, seed)
    clf = _clf(arch, seed, n_classes)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)
    return prob.astype(np.float32), yte, _auc(yte, prob, n_classes)


def score_lora(H, y, n_classes, seed, device):
    """One probe seed of diagnostic.probe_lora, verbatim loop, returning probs."""
    Xtr, Xte, ytr, yte = _split(H, y, seed)
    torch.manual_seed(seed)
    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    Xte_t = torch.from_numpy(Xte).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    atk = FreshLoRAAttacker(repr_dim=H.shape[1], n_classes=n_classes,
                            rank=LORA_RANK, alpha=LORA_ALPHA).to(device)
    opt = torch.optim.Adam(atk.parameters(), lr=1e-2)
    ce = nn.CrossEntropyLoss()
    n = Xtr_t.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    atk.train()
    for _ in range(LORA_STEPS):
        idx = torch.randint(0, n, (min(512, n),), generator=g).to(device)
        loss = ce(atk(Xtr_t[idx]), ytr_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    atk.eval()
    with torch.no_grad():
        logits = atk(Xte_t).cpu().numpy()
    z = logits - logits.max(1, keepdims=True)
    prob = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return prob.astype(np.float32), yte, _auc(yte, prob, n_classes)


def score_lrt(h_clean, P_exposed, y, noise_cov, n_classes, seed):
    """One probe seed of utils.battery.gaussian_lrt, returning posteriors."""
    tr, te = _split_idx(y, seed)
    post = _lrt_scores(h_clean[tr].astype(np.float64), y[tr],
                       P_exposed[te].astype(np.float64), noise_cov, n_classes)
    return post.astype(np.float32), y[te], _auc(y[te], post, n_classes)


SCORERS = {"XGB": lambda H, y, nc, s, dev: score_sklearn("XGB", H, y, nc, s),
           "MLP": lambda H, y, nc, s, dev: score_sklearn("MLP", H, y, nc, s),
           "LoRA": score_lora}


# --------------------------------------------------------------------------- #
#  Capture plumbing                                                            #
# --------------------------------------------------------------------------- #
records = []            # flat AUC records
npz_store = {}          # point -> {key: array}


def capture(point, cell, surface, attacker, ts, ps, prob, yte, auc):
    d = npz_store.setdefault(point, {})
    d[f"{surface}_{attacker}_ts{ts}_ps{ps}_prob"] = prob
    ykey = f"y_ps{ps}"
    if ykey not in d:
        d[ykey] = yte.astype(np.int16)
    records.append({"point": point, "cell": cell, "surface": surface,
                    "attacker": attacker, "train_seed": ts, "probe_seed": ps,
                    "auc": auc})


def battery_capture(point, cell, surface, H, y, n_classes, seeds, device,
                    attackers=("XGB", "MLP", "LoRA"), ts=0):
    for ps in seeds:
        for a in attackers:
            prob, yte, auc = SCORERS[a](H, y, n_classes, ps, device)
            capture(point, cell, surface, a, ts, ps, prob, yte, auc)


def probe_mean(point, surface, attacker, ts):
    v = [r["auc"] for r in records
         if r["point"] == point and r["surface"] == surface
         and r["attacker"] == attacker and r["train_seed"] == ts]
    return float(np.mean(v)) if v else None


# --------------------------------------------------------------------------- #
#  TPR at low FPR                                                              #
# --------------------------------------------------------------------------- #
def tpr_at_fpr(y_bin, score, fpr_target):
    """Max TPR at FPR <= target (conservative step-function convention)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_bin, score)
    ok = fpr <= fpr_target + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def tpr_block(prob, yte, n_classes):
    """Binary: direct. Multiclass: per-class OvR + worst-class + macro."""
    out = {}
    if n_classes == 2:
        for f in FPRS:
            out[f"tpr@{f}"] = tpr_at_fpr((yte == 1).astype(int), prob[:, 1], f)
        out["n_pos_eval"] = int((yte == 1).sum())
    else:
        per = {}
        for c in range(n_classes):
            yb = (yte == c).astype(int)
            per[str(c)] = {"n_pos_eval": int(yb.sum()),
                           **{f"tpr@{f}": tpr_at_fpr(yb, prob[:, c], f)
                              for f in FPRS}}
        out["per_class"] = per
        for f in FPRS:
            vals = [per[str(c)][f"tpr@{f}"] for c in range(n_classes)]
            out[f"worst_class_tpr@{f}"] = float(max(vals))
            out[f"macro_tpr@{f}"] = float(np.mean(vals))
    return out


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    device = get_device()
    print(f"[setup] device={device} train_seeds={TRAIN_SEEDS} tol={TOL}", flush=True)

    data = {}
    for key, (ds, at, tk) in CELLS.items():
        X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk)
        data[key] = dict(X_t=torch.from_numpy(X).to(device), attr=attr, task=task,
                         n_attr=n_attr, n_task=n_task, task_maj=task_maj)
        print(f"[load] {key} {ds}/{at}/{tk} n={len(attr)} n_attr={n_attr} "
              f"n_task={n_task}", flush=True)
    # occupation_group shares the adult loader (cached)
    Xo, attr_o, task_o, na_o, nt_o, _, tm_o = load_cell("adult", "sex",
                                                        "occupation_group")
    Xo_t = torch.from_numpy(Xo).to(device)

    avg_rows = []   # section E AUC rows

    for key, (ds, at, tk) in CELLS.items():
        d = data[key]

        # ---- A: unprotected baseline (sigma=0) --------------------------------
        point = f"unprot_{key}"
        print(f"\n[A {point}] train_model sigma=0, 5 seeds", flush=True)
        for ts in TRAIN_SEEDS:
            t = time.time()
            P, _, L = train_model(d["X_t"], d["attr"], d["task"], d["n_attr"],
                                  d["n_task"], use_seal=False, alpha=0.0,
                                  device=device, pub_noise=0.0, seed=ts)
            battery_capture(point, key, "rep", P, d["attr"], d["n_attr"], PS_2,
                            device, ts=ts)
            battery_capture(point, key, "out", L, d["attr"], d["n_attr"], PS_2,
                            device, ts=ts)
            print(f"  [ts {ts}] rep XGB={probe_mean(point,'rep','XGB',ts):.4f} "
                  f"MLP={probe_mean(point,'rep','MLP',ts):.4f} "
                  f"out XGB={probe_mean(point,'out','XGB',ts):.4f} "
                  f"({time.time()-t:.0f}s)", flush=True)

        # ---- B: subspace Tier-1 surviving point (+ E averaging inline) --------
        cfg = SUBSPACE[key]
        point = f"subspace_{key}"
        print(f"\n[B {point}] e2e surgical r={cfg['rank']} s={cfg['sigma']:g} "
              f"lam={cfg['lam']:g}, 5 seeds", flush=True)
        for ts in TRAIN_SEEDS:
            t = time.time()
            P, L, h, Q = train_e2e_surgical(
                d["X_t"], d["attr"], d["task"], d["n_attr"], d["n_task"],
                rank=cfg["rank"], sigma=cfg["sigma"], lam=cfg["lam"],
                device=device, seed=ts, return_clean=True)
            battery_capture(point, key, "rep", P, d["attr"], d["n_attr"], PS_2,
                            device, ts=ts)
            battery_capture(point, key, "out", L, d["attr"], d["n_attr"], PS_2,
                            device, ts=ts)
            print(f"  [ts {ts}] rep XGB={probe_mean(point,'rep','XGB',ts):.4f} "
                  f"MLP={probe_mean(point,'rep','MLP',ts):.4f} "
                  f"LoRA={probe_mean(point,'rep','LoRA',ts):.4f} "
                  f"({time.time()-t:.0f}s)", flush=True)

            if key in ("middle", "hard"):        # ---- E: averaging, this model
                g = torch.Generator(device="cpu").manual_seed(DRAW_SEED + ts)
                h64, Q64 = h.astype(np.float64), Q.astype(np.float64)
                running = P.astype(np.float64).copy()
                n_in_sum = 1
                for N in NS:
                    while n_in_sum < N:
                        z = torch.randn(h.shape[0], cfg["rank"],
                                        generator=g).numpy().astype(np.float64)
                        running += h64 + cfg["sigma"] * (z @ Q64.T)
                        n_in_sum += 1
                    A = P if N == 1 else (running / N).astype(np.float32)
                    aucs = {}
                    for a in ("XGB", "MLP", "LoRA"):
                        vals = [SCORERS[a](A, d["attr"], d["n_attr"], ps,
                                           device)[2] for ps in PS_2]
                        aucs[a.lower()] = float(np.mean(vals))
                    row = {"cell": key, "train_seed": ts, "N": N,
                           "sigma_eff": float(cfg["sigma"] / np.sqrt(N)), **aucs,
                           "tier1_max": float(max(aucs.values()))}
                    avg_rows.append(row)
                    print(f"    [avg N={N:>2}] s_eff={row['sigma_eff']:5.2f} "
                          f"T1={row['tier1_max']:.4f} "
                          f"{'BREACHES' if row['tier1_max'] > BAR else 'holds'}",
                          flush=True)

        # ---- C: full-rank Tier-1 / Tier-2 points ------------------------------
        for tier, sg in FULLRANK[key].items():
            point = f"fullrank_{tier}_{key}"
            print(f"\n[C {point}] e2e blunt sigma={sg:g}, 5 seeds", flush=True)
            cov = iso_noise_cov(64, sg)
            for ts in TRAIN_SEEDS:
                t = time.time()
                h, P, L = train_channel(d["X_t"], d["attr"], d["task"],
                                        d["n_attr"], d["n_task"], sg, device,
                                        seed=ts)
                battery_capture(point, key, "rep", P, d["attr"], d["n_attr"],
                                PS_3, device, ts=ts)
                for ps in PS_3:
                    prob, yte, auc = score_lrt(h, P, d["attr"], cov,
                                               d["n_attr"], ps)
                    capture(point, key, "rep", "LRT", ts, ps, prob, yte, auc)
                battery_capture(point, key, "out", L, d["attr"], d["n_attr"],
                                PS_3, device, ts=ts)
                print(f"  [ts {ts}] rep XGB={probe_mean(point,'rep','XGB',ts):.4f} "
                      f"LoRA={probe_mean(point,'rep','LoRA',ts):.4f} "
                      f"LRT={probe_mean(point,'rep','LRT',ts):.4f} "
                      f"({time.time()-t:.0f}s)", flush=True)

    # ---- D: Adult floor points (Job 1's longer-grid first-hidden sigmas) ------
    for ds, at, tk, sg in FLOOR_POINTS:
        point = f"floor_{tk}_s{sg:g}"
        if tk == "income":
            dd = data["hard"]
            X_t, attr, task, n_attr, n_task = (dd["X_t"], dd["attr"], dd["task"],
                                               dd["n_attr"], dd["n_task"])
        else:
            X_t, attr, task, n_attr, n_task = Xo_t, attr_o, task_o, na_o, nt_o
        print(f"\n[D {point}] train_model sigma={sg:g}, 5 seeds", flush=True)
        for ts in TRAIN_SEEDS:
            t = time.time()
            P, _, L = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                                  alpha=0.0, device=device, pub_noise=sg, seed=ts)
            battery_capture(point, f"adult/{at}/{tk}", "rep", P, attr, n_attr,
                            PS_2, device, ts=ts)
            battery_capture(point, f"adult/{at}/{tk}", "out", L, attr, n_attr,
                            PS_2, device, ts=ts)
            print(f"  [ts {ts}] rep XGB={probe_mean(point,'rep','XGB',ts):.4f} "
                  f"out XGB={probe_mean(point,'out','XGB',ts):.4f} "
                  f"({time.time()-t:.0f}s)", flush=True)

    # ---- save scores ----------------------------------------------------------
    for point, arrs in npz_store.items():
        np.savez_compressed(SCORE_DIR / f"{point}.npz", **arrs)
    print(f"\n[save] {len(npz_store)} npz files -> {SCORE_DIR}", flush=True)

    # ---- TPR at low FPR from the captured scores ------------------------------
    tpr = {}
    for point, arrs in npz_store.items():
        surfaces = sorted({r["surface"] for r in records if r["point"] == point})
        tpr[point] = {}
        for surf in surfaces:
            atks = sorted({r["attacker"] for r in records
                           if r["point"] == point and r["surface"] == surf})
            best = max(atks, key=lambda a: np.mean(
                [r["auc"] for r in records if r["point"] == point
                 and r["surface"] == surf and r["attacker"] == a]))
            per_seed = {}
            for r in [r for r in records if r["point"] == point
                      and r["surface"] == surf and r["attacker"] == best]:
                ts, ps = r["train_seed"], r["probe_seed"]
                prob = arrs[f"{surf}_{best}_ts{ts}_ps{ps}_prob"]
                yte = arrs[f"y_ps{ps}"]
                nc = prob.shape[1]
                per_seed[f"ts{ts}_ps{ps}"] = {"auc": r["auc"],
                                              **tpr_block(prob, yte, nc)}
            tpr[point][surf] = {"best_attacker": best,
                                "best_auc_mean": float(np.mean(
                                    [r["auc"] for r in records
                                     if r["point"] == point
                                     and r["surface"] == surf
                                     and r["attacker"] == best])),
                                "per_seed": per_seed}
    print("[tpr] computed", flush=True)

    # ---- Job-3 reproduction gate ---------------------------------------------
    repro = []

    def gate(point, surface, attacker, ts, stored, label, src):
        got = probe_mean(point, surface, attacker, ts)
        if got is None or stored is None:
            return
        delta = got - stored
        repro.append({"point": point, "metric": label, "train_seed": ts,
                      "stored": stored, "rerun": got, "delta": delta,
                      "source": src, "pass": bool(abs(delta) <= TOL)})

    hs = json.loads((RESULTS / "hardening_seeds.json").read_text())
    tt = json.loads((RESULTS / "two_tier_certification.json").read_text())
    cc = json.loads((RESULTS / "continuous_cost.json").read_text())
    hkeys = {"easy": "easy hmda/race/loan_decision",
             "middle": "middle hmda/race/loan_amount_band",
             "hard": "hard adult/sex/income"}

    for key in CELLS:
        # subspace vs hardening (easy/middle: exp10 D; hard: sigma16 alternative)
        blk = (hs["hard_cell_sigma16_alternative"] if key == "hard" else
               next(e for e in hs["exp10_5seed"]
                    if e["cell"] == hkeys[key])["D"])
        src = ("hardening_seeds.json hard_cell_sigma16_alternative" if key == "hard"
               else "hardening_seeds.json exp10_5seed.D")
        for ts in TRAIN_SEEDS:
            for surf, pre in (("rep", "rep"), ("out", "out")):
                for a, name in (("XGB", "xgb"), ("MLP", "mlp"), ("LoRA", "lora")):
                    gate(f"subspace_{key}", surf, a, ts,
                         blk["per_seed"][f"{pre}_{name}"][ts],
                         f"{surf}_{name}", src)
        # full-rank vs two_tier e2e_cells
        cell_tt = next(c for c in tt["e2e_cells"]
                       if c["cell"] == hkeys[key].split(" ", 1)[1]
                       or c["cell"] == hkeys[key])
        for tier in ("tier1", "tier2"):
            for ts in TRAIN_SEEDS:
                for a, name in (("XGB", "xgb"), ("MLP", "mlp"),
                                ("LoRA", "lora"), ("LRT", "lrt")):
                    gate(f"fullrank_{tier}_{key}", "rep", a, ts,
                         cell_tt[tier]["per_seed"][name][ts], f"rep_{name}",
                         f"two_tier_certification.json e2e_cells.{tier}")
        # unprotected vs continuous_cost sweep sigma=0 (train seed 0 only)
        cellname = "/".join(CELLS[key])
        row = next(r for r in cc["rows"] if r["cell"] == cellname)
        s0 = next(p for p in row["sweep"] if p["sigma"] == 0.0)
        for a, pk, ok_ in (("XGB", "P_xgb", "out_xgb"), ("MLP", "P_mlp", "out_mlp")):
            gate(f"unprot_{key}", "rep", a, 0, s0[pk], f"rep_{a.lower()}",
                 "continuous_cost.json sweep sigma=0")
            gate(f"unprot_{key}", "out", a, 0, s0[ok_], f"out_{a.lower()}",
                 "continuous_cost.json sweep sigma=0")
    # Adult floor points vs continuous_cost sweep entries (train seed 0)
    for ds, at, tk, sg in FLOOR_POINTS:
        row = next(r for r in cc["rows"] if r["cell"] == f"adult/{at}/{tk}")
        pt = next(p for p in row["sweep"] if p["sigma"] == sg)
        for a, pk, ok_ in (("XGB", "P_xgb", "out_xgb"), ("MLP", "P_mlp", "out_mlp")):
            gate(f"floor_{tk}_s{sg:g}", "rep", a, 0, pt[pk], f"rep_{a.lower()}",
                 f"continuous_cost.json sweep sigma={sg:g}")
            gate(f"floor_{tk}_s{sg:g}", "out", a, 0, pt[ok_], f"out_{a.lower()}",
                 f"continuous_cost.json sweep sigma={sg:g}")

    n_fail = sum(1 for r in repro if not r["pass"])
    print(f"[gate] {len(repro)} comparisons, {n_fail} over tol {TOL}", flush=True)
    for r in repro:
        if not r["pass"]:
            print(f"  FAIL {r['point']} {r['metric']} ts{r['train_seed']}: "
                  f"stored={r['stored']:.4f} rerun={r['rerun']:.4f} "
                  f"delta={r['delta']:+.4f}", flush=True)

    out = {"experiment": "tpr_at_low_fpr_capture", "bar": BAR,
           "train_seeds": TRAIN_SEEDS, "probe_seeds": {"A_B_D_E": PS_2, "C": PS_3},
           "fprs": FPRS, "tol": TOL, "draw_seed": DRAW_SEED,
           "tpr_convention": "max TPR at FPR <= target (step function); "
                             "multiclass = per-class OvR",
           "points": {"subspace": SUBSPACE, "fullrank": FULLRANK,
                      "floor": [list(f) for f in FLOOR_POINTS]},
           "auc_records": records, "averaging": avg_rows, "tpr": tpr,
           "reproduction": repro, "n_gate_fail": n_fail,
           "minutes": round((time.time() - t0) / 60, 1)}
    (HERE / "tpr_capture.json").write_text(json.dumps(out, indent=1))
    print(f"\n[save] {HERE / 'tpr_capture.json'} ({out['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
