"""Worst-case TPR for the 59 AUC-failing audit configurations — option (b).

Registered prediction: docs/tpr_failing59_prediction.md (commit b4deca6, BEFORE this
script existed). Completes the dual verdict: with tpr_extension.json's 8
survivors, all 67 approved audit configurations are measured under both
criteria.

Populations (enumerated from the stored verdict files, never hand-listed):
  P1  existing adult/sex on the frozen PCRL rep (honest_reaudit.json master):
      E2 noise sigma_rel {0, 0.25, 0.5, 1, 2} + E4S1 duplicates {1, 2}
      (measured once per sigma, emitted per master row — the original E4S1
      rows repeat the E2 measurements verbatim), and the ten E4S1 projections
      (LEOPARD-MMD r=1/2/4/8/16, HSIC r=1/4/8/16/32; fit_projection seed 0,
      apply_projection, static probes).
  P2  existing hmda/race noise sigma=8 on the frozen PCRL hmda rep
      (purpose 0; 5-class attribute -> worst-class OvR).
  P3  expansion: the 41 approved-but-failing configs of
      expansion_reaudit_paperframing.json (noise via the probe_through
      convention, projections via fit_projection seed 0 + static probes),
      clean rep retrained once per cell (train_model seed 0).

Suite per config = its originally scored approval suite: XGB + MLP
(max(XGB, MLP) is the paperframing failure criterion). Probe seeds [0, 1, 2]
everywhere (honest_reaudit attacker_seeds == expansion REAUDIT_SEEDS).
Per-config reproduction gate at 0.01 against the stored xgb/mlp AUCs; a
gate failure invalidates that config's reading (registered scoring rule).

Outputs: results/tpr_failing59.json (+ tpr59_scores/*.npz,
gitignored). Checkpoints after every group so an overnight crash resumes.
"""
from __future__ import annotations

import os

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
# Generated intermediates; gitignored and regenerable.
# Override the location with DG_ARTIFACTS.
ARTIFACTS = Path(os.environ.get("DG_ARTIFACTS", REPO_ROOT / "analysis"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

# run_tpr_extension performs the xgboost-before-torch import guard on import
from run_tpr_extension import (  # noqa: E402
    FPRS, probe_capture, tpr_at_fpr, tpr_block,
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

RESULTS = REPO_ROOT / "results"
SCORE_DIR = ARTIFACTS / "tpr59_scores"
SCORE_DIR.mkdir(exist_ok=True)
OUT = RESULTS / "tpr_failing59.json"
CKPT = ARTIFACTS / "tpr_failing59_ckpt.json"
SEEDS = [0, 1, 2]
TOL = 0.01
SURVIVOR_EXISTING = {("E2 noise", "adult/sex", "noise σ=4.0"),
                     ("E2 noise", "adult/sex", "noise σ=8.0"),
                     ("E4S1 noise", "adult/sex", "noise sigma=8.0")}


def masterkey(r):
    return (r["experiment"], r.get("cell", "adult/sex"), r["method"])


def load_ckpt():
    if CKPT.exists():
        d = json.loads(CKPT.read_text())
        return d["configs"], set(d["done_groups"])
    return [], set()


def save_ckpt(configs, done):
    CKPT.write_text(json.dumps({"configs": configs,
                                "done_groups": sorted(done)}))


def measure_fixed(H_er, y, n_classes, slug):
    """Static probes (fixed matrix), XGB+MLP, seeds [0,1,2], with capture."""
    from sklearn.model_selection import train_test_split
    arrs, aucs = {}, {"XGB": [], "MLP": []}
    for s in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(
            H_er.astype(np.float64), y, test_size=0.25, random_state=s,
            stratify=y)
        for arch in ("XGB", "MLP"):
            prob, auc = probe_capture(arch, Xtr, ytr, Xte, yte, s, n_classes)
            arrs[f"{arch}_ps{s}_prob"] = prob
            arrs[f"y_ps{s}"] = yte.astype(np.int16)
            aucs[arch].append(auc)
    np.savez_compressed(SCORE_DIR / f"{slug}.npz", **arrs)
    return arrs, {a: float(np.mean(v)) for a, v in aucs.items()}


def measure_noised(H, y, n_classes, sigma_abs, slug):
    """probe_battery/probe_through convention: fresh noise per seed rng(s)."""
    from sklearn.model_selection import train_test_split
    arrs, aucs = {}, {"XGB": [], "MLP": []}
    for s in SEEDS:
        X = H.astype(np.float64)
        if sigma_abs > 0.0:
            X = X + np.random.default_rng(s).normal(0.0, sigma_abs,
                                                    size=X.shape)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=s, stratify=y)
        for arch in ("XGB", "MLP"):
            prob, auc = probe_capture(arch, Xtr, ytr, Xte, yte, s, n_classes)
            arrs[f"{arch}_ps{s}_prob"] = prob
            arrs[f"y_ps{s}"] = yte.astype(np.int16)
            aucs[arch].append(auc)
    np.savez_compressed(SCORE_DIR / f"{slug}.npz", **arrs)
    return arrs, {a: float(np.mean(v)) for a, v in aucs.items()}


def config_entry(config, kind, arrs, mean_aucs, stored, src, n_classes):
    """Best attacker per the config's suite, TPR per probe seed, gates."""
    best = max(mean_aucs, key=mean_aucs.get)
    t1s, t01s, blocks = [], [], {}
    for s in SEEDS:
        prob, yte = arrs[f"{best}_ps{s}_prob"], arrs[f"y_ps{s}"]
        blk = tpr_block(prob, yte, n_classes)
        blocks[f"ps{s}"] = blk
        if n_classes == 2:
            t01s.append(blk["tpr@0.001"]); t1s.append(blk["tpr@0.01"])
        else:
            t01s.append(blk["worst_class_tpr@0.001"])
            t1s.append(blk["worst_class_tpr@0.01"])
    gates = []
    for arch, key in (("XGB", "xgb_auc"), ("MLP", "mlp_auc")):
        d = mean_aucs[arch] - stored[key]
        gates.append({"attacker": arch, "stored": stored[key],
                      "rerun": mean_aucs[arch], "delta": d,
                      "pass": bool(abs(d) <= TOL), "source": src})
    return {"config": config, "kind": kind, "n_classes": n_classes,
            "suite_auc_means": mean_aucs, "best_attacker": best,
            "best_auc_mean": mean_aucs[best],
            "tpr@0.01_mean": float(np.mean(t1s)),
            "tpr@0.01_worst": float(max(t1s)),
            "tpr@0.001_mean": float(np.mean(t01s)),
            "tpr@0.001_worst": float(max(t01s)),
            "per_seed": blocks, "gates": gates,
            "gate_ok": bool(all(g["pass"] for g in gates))}


def main():
    t0 = time.time()
    from utils.pcrl_io import get_device
    device = get_device()
    configs, done = load_ckpt()
    print(f"[setup] device={device} resume: {len(configs)} configs, "
          f"{len(done)} groups done", flush=True)

    hr = json.loads((RESULTS / "honest_reaudit.json").read_text())
    master = hr["master"]
    failing_existing = [r for r in master
                        if masterkey(r) not in SURVIVOR_EXISTING]
    assert len(failing_existing) == 18, len(failing_existing)

    # ---- P1: adult/sex on the frozen PCRL rep --------------------------------
    if "P1" not in done:
        from experiments.smart_erasure import apply_projection, fit_projection
        from utils.pcrl_io import (INCOME_PREDICTION_PURPOSE_IDX,
                                   build_adult_train_loader, extract_attr,
                                   extract_frozen_reps, load_adult_encoder)
        encoder, _ = load_adult_encoder()
        _, _, loader = build_adult_train_loader()
        H = extract_frozen_reps(encoder, loader,
                                INCOME_PREDICTION_PURPOSE_IDX, device)
        sex = extract_attr(loader, "sex")
        n_sex = int(sex.max()) + 1
        sigma_h = float(H.std(0).mean())
        H_t = torch.from_numpy(H).to(device)
        print(f"[P1] adult rep n={len(sex)} sigma_h={sigma_h:.4f}", flush=True)

        noise_rows = [r for r in failing_existing if "noise" in r["experiment"]
                      and r["cell"] == "adult/sex"]
        by_sigma = {}
        for r in noise_rows:
            sr = float(r["method"].split("=")[1])
            by_sigma.setdefault(sr, []).append(r)
        for sr, rows in sorted(by_sigma.items()):
            slug = f"adult_noise_s{sr:g}"
            arrs, ma = measure_noised(H, sex, n_sex, sr * sigma_h, slug)
            for r in rows:
                e = config_entry(f"{r['experiment']}|adult/sex|{r['method']}",
                                 "noise", arrs, ma, r,
                                 "honest_reaudit.json master", n_sex)
                configs.append(e)
                print(f"  [{e['config']:<44}] best={e['best_attacker']} "
                      f"auc={e['best_auc_mean']:.4f} "
                      f"tpr1%={100*e['tpr@0.01_mean']:.1f}% "
                      f"gate={'OK' if e['gate_ok'] else 'FAIL'}", flush=True)

        for r in [r for r in failing_existing if r["experiment"] == "E4S1 proj"]:
            fam, rank = r["method"].rsplit(" r=", 1)
            kind = "mmd" if "MMD" in fam else "hsic"
            Q = fit_projection(H_t, sex, int(rank), kind, device, seed=0)
            H_er = apply_projection(H, Q)
            slug = f"adult_{kind}_r{rank}"
            arrs, ma = measure_fixed(H_er, sex, n_sex, slug)
            e = config_entry(f"E4S1 proj|adult/sex|{r['method']}", "proj",
                             arrs, ma, r, "honest_reaudit.json master", n_sex)
            configs.append(e)
            print(f"  [{e['config']:<44}] best={e['best_attacker']} "
                  f"auc={e['best_auc_mean']:.4f} "
                  f"tpr1%={100*e['tpr@0.01_mean']:.1f}% "
                  f"gate={'OK' if e['gate_ok'] else 'FAIL'}", flush=True)
        done.add("P1"); save_ckpt(configs, done)

    # ---- P2: hmda/race noise sigma=8 ----------------------------------------
    if "P2" not in done:
        from utils.pcrl_io import (build_train_loader, extract_attr,
                                   extract_frozen_reps, load_encoder)
        purposes_h, train_ds_h, loader_h = build_train_loader("hmda")
        enc_h, _ = load_encoder("hmda", train_ds_h.info.num_features,
                                len(purposes_h))
        Hh = extract_frozen_reps(enc_h, loader_h, 0, device)
        race = extract_attr(loader_h, "race")
        n_race = int(race.max()) + 1
        sigma_hh = float(Hh.std(0).mean())
        r = next(r for r in failing_existing if r["cell"] == "hmda/race")
        arrs, ma = measure_noised(Hh, race, n_race, 8.0 * sigma_hh,
                                  "hmda_noise_s8")
        e = config_entry("E4S1 noise|hmda/race|noise sigma=8.0", "noise",
                         arrs, ma, r, "honest_reaudit.json master", n_race)
        configs.append(e)
        print(f"  [{e['config']:<44}] best={e['best_attacker']} "
              f"auc={e['best_auc_mean']:.4f} worst-class "
              f"tpr1%={100*e['tpr@0.01_mean']:.1f}% "
              f"gate={'OK' if e['gate_ok'] else 'FAIL'}", flush=True)
        done.add("P2"); save_ckpt(configs, done)

    # ---- P3: expansion 41 ----------------------------------------------------
    pf = json.loads((RESULTS / "expansion_reaudit_paperframing.json").read_text())
    fails = [r for r in pf["rows"]
             if r["approved_at_rest"] and r["fails_paperframing"]]
    assert len(fails) == 41, len(fails)
    ra = json.loads((RESULTS / "expansion_reaudit.json").read_text())
    ra_rows = {(r["cell"], r["method"]): r for r in ra["master"]}
    from collections import defaultdict
    by_cell = defaultdict(list)
    for r in fails:
        by_cell[r["cell"]].append(r["method"])

    from experiments.expansion_pipeline import load_cell_by_tag
    from experiments.sealed_channel import train_model
    from experiments.smart_erasure import apply_projection, fit_projection
    for cell, methods in sorted(by_cell.items()):
        gname = "P3_" + cell.replace("/", "_")
        if gname in done:
            continue
        tc = time.time()
        _, _, _, X, attr, task, n_attr, n_task, _, _ = load_cell_by_tag(cell)
        X_t = torch.from_numpy(X).to(device)
        P, _, _ = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                              alpha=0.0, device=device, pub_noise=0.0, seed=0)
        sigma_h = float(P.std(0).mean())
        P_t = torch.from_numpy(P.astype(np.float32)).to(device)
        print(f"[P3 {cell}] rep trained ({time.time()-tc:.0f}s), "
              f"{len(methods)} configs", flush=True)
        for m in sorted(methods):
            stored = ra_rows[(cell, m)]
            slug = ("exp59_" + cell.replace("/", "_") + "_"
                    + m.replace(" ", "").replace("σ=", "s").replace("=", ""))
            if m.startswith("noise"):
                sr = float(m.split("=")[1])
                arrs, ma = measure_noised(P, attr, n_attr, sr * sigma_h, slug)
                kind = "noise"
            else:
                fam, rank = m.rsplit(" r=", 1)
                pkind = "mmd" if "MMD" in fam else "hsic"
                Q = fit_projection(P_t, attr, int(rank), pkind, device, seed=0)
                H_er = apply_projection(P, Q)
                arrs, ma = measure_fixed(H_er, attr, n_attr, slug)
                kind = "proj"
            e = config_entry(f"expansion|{cell}|{m}", kind, arrs, ma, stored,
                             "expansion_reaudit.json master", n_attr)
            configs.append(e)
            print(f"  [{e['config']:<56}] best={e['best_attacker']} "
                  f"auc={e['best_auc_mean']:.4f} "
                  f"tpr1%={100*e['tpr@0.01_mean']:.1f}% "
                  f"gate={'OK' if e['gate_ok'] else 'FAIL'}", flush=True)
        done.add(gname); save_ckpt(configs, done)

    # ---- score the registered claims ----------------------------------------
    assert len(configs) == 59, len(configs)
    valid = [c for c in configs if c["gate_ok"]]
    t1 = [c["tpr@0.01_mean"] for c in valid]
    projs = [c for c in valid if c["kind"] == "proj"]
    c1 = min(t1) > 0.05
    c2 = 0.50 <= float(np.mean(t1)) <= 0.75
    c3 = any(a["best_auc_mean"] > 0.60 and b["best_auc_mean"] < 0.60
             and a["tpr@0.01_mean"] < b["tpr@0.01_mean"]
             for a in valid for b in valid)
    c4 = all(c["tpr@0.01_mean"] > 0.20 for c in projs)
    n_gate_fail = len(configs) - len(valid)
    scoring = {
        "claim1_all_above_5pct": {"verdict": "HELD" if c1 else "FALSIFIED",
                                  "min_tpr@0.01": float(min(t1))},
        "claim2_mean_in_50_75": {"verdict": "HELD" if c2 else "FALSIFIED",
                                 "mean_tpr@0.01": float(np.mean(t1))},
        "claim3_ordering_exception": {"verdict": "HELD" if c3 else "FALSIFIED"},
        "claim4_projections_above_20pct": {
            "verdict": "HELD" if c4 else "FALSIFIED",
            "min_proj_tpr@0.01": float(min(c["tpr@0.01_mean"] for c in projs))},
        "n_gate_invalidated": n_gate_fail}
    print("\n" + "=" * 80)
    for k, v in scoring.items():
        print(f"  {k}: {v}")
    out = {"experiment": "tpr_failing59", "prediction_file":
           "tpr_failing59_prediction.md", "prediction_commit": "b4deca6",
           "fprs": FPRS, "seeds": SEEDS, "tol": TOL,
           "n_configs": len(configs), "n_gate_fail": n_gate_fail,
           "configs": configs, "prediction_scoring": scoring,
           "minutes": round((time.time() - t0) / 60, 1)}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\n[save] {OUT} ({out['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
