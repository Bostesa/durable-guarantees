"""TPR@low-FPR extension — score capture for (a) the 8 surviving audit
configurations and (b) the 36 published-method gauntlet combinations at their
scored operating points. Registered predictions: tpr_extension.json header,
committed at a69a3fd BEFORE any run.

Same pattern as run_tpr.py: reproduce each stored measurement exactly (same
attacker configs, splits, seeds, fresh-noise conventions), capture the
held-out per-example class posteriors before the AUC reduction, gate every
re-run AUC against the stored JSON value (tol 0.01), then compute TPR at
FPR {0.001, 0.01} from the captured scores. FPR budgets differ from
run_tpr.py's {0.01, 0.05} per the task; the step-function convention is
unchanged. Representation surface only (the verdict surface).

Sections (CLI: --section a|b1|b2|b3|merge):
  a   the 8 survivors — honest_reaudit adult/sex noise sigma=4/8 (XGB+MLP,
      seeds [0,1,2], noise-before-split via default_rng(seed)) and the five
      expansion noise-sigma=8 cells (XGB+MLP via probe_through, LoRA via
      lora_channel rng(100+s), LRT via gaussian_lrt_channel; seeds [0,1,2])
  b1  the 24 baseline_gauntlet combos: uncertified combos at their 'closest'
      sweep label (train seed 0, probe seeds [0,1], attackers exactly as the
      original row: XGB+MLP+LRT, LoRA only where present); the certified
      combo (VFAE beta=1 sampled-z, easy, T1) over train seeds [0,1,2] x
      probe seeds [0,1,2] with the full battery
  b2  the 6 Obliviator combos (iter 15, 3 cells) — requires OBLIVIATOR_ROOT
  b3  the 6 official-LAFTR combos (g=1/1/8, seed 0) — requires the official
      run dirs (LAFTR_RUNS) with npz exposures
  merge  TPR blocks + reproduction gate -> tpr_extension.json (header kept)

Outputs: analysis/tpr_ext_scores/<group>.npz,
tpr_ext_partial_<section>.json, and (merge) tpr_extension.json.
"""
from __future__ import annotations


import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401  (import before torch on purpose)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
# Generated intermediates; gitignored and regenerable.
# Override the location with DG_ARTIFACTS.
ARTIFACTS = Path(os.environ.get("DG_ARTIFACTS", REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.diagnostic import (  # noqa: E402
    LORA_ALPHA, LORA_RANK, LORA_STEPS, _split, load_cell,
)
from experiments.falsification_attack import FreshLoRAAttacker  # noqa: E402
from experiments.hardening_test import _probe  # noqa: E402
from experiments.sealed_channel import train_model  # noqa: E402
from experiments.targeted_noise import iso_channel  # noqa: E402
from utils.battery import _lrt_scores, _split_idx, iso_noise_cov  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

HERE = Path(__file__).resolve().parent
SCORE_DIR = ARTIFACTS / "tpr_ext_scores"
SCORE_DIR.mkdir(exist_ok=True)
RESULTS = REPO_ROOT / "results"

FPRS = [0.001, 0.01]
TOL = 0.01
BAR = 0.55

CELL_SLUG = {
    "hmda/race/loan_decision": "easy",
    "hmda/race/loan_amount_band": "middle",
    "adult/sex/income": "hard",
}


def bg_slug(cell, baseline, label):
    s = f"bg_{CELL_SLUG[cell]}_{baseline}_{label}"
    s = (s.replace("β", "b").replace("γ", "g").replace("λ", "l")
         .replace("=", "").replace(" ", "_"))
    return "".join(ch if (ch.isalnum() or ch in "_-") else "-" for ch in s)


# --------------------------------------------------------------------------- #
#  Capture primitives (attacker configs verbatim from the stored pipeline)     #
# --------------------------------------------------------------------------- #
def _auc(yte, prob, n_classes):
    from sklearn.metrics import roc_auc_score
    if n_classes == 2:
        return float(roc_auc_score(yte, prob[:, 1]))
    return float(roc_auc_score(yte, prob, multi_class="ovr", average="macro"))


def probe_capture(arch, Xtr, ytr, Xte, yte, seed, n_classes):
    """hardening_test._probe, capturing predict_proba before the reduction."""
    if arch == "MLP":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(256, 256), activation="relu",
                            alpha=1e-4, solver="adam", learning_rate_init=1e-3,
                            max_iter=100, early_stopping=True,
                            validation_fraction=0.1, n_iter_no_change=8,
                            random_state=seed, batch_size=256)
    else:
        from xgboost import XGBClassifier
        obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
        clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                            objective=obj, tree_method="hist",
                            eval_metric="logloss", random_state=seed, n_jobs=2,
                            verbosity=0)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte).astype(np.float32)
    return prob, _auc(yte, prob, n_classes)


def lora_capture(Hn, y, n_classes, seed, device):
    """diagnostic.probe_lora, one seed, verbatim loop, capturing softmax."""
    Xtr, Xte, ytr, yte = _split(Hn, y, seed)
    torch.manual_seed(seed)
    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    Xte_t = torch.from_numpy(Xte).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    atk = FreshLoRAAttacker(repr_dim=Hn.shape[1], n_classes=n_classes,
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
    prob = (np.exp(z) / np.exp(z).sum(1, keepdims=True)).astype(np.float32)
    return prob, yte, _auc(yte, prob, n_classes)


def lrt_capture(H_clean, X_exposed_te, y, tr, te, noise_cov, n_classes):
    """utils.battery._lrt_scores on pre-split data, capturing posteriors."""
    post = _lrt_scores(H_clean[tr].astype(np.float64), y[tr],
                       X_exposed_te.astype(np.float64),
                       np.asarray(noise_cov, dtype=np.float64), n_classes)
    return post.astype(np.float32), _auc(y[te], post, n_classes)


# --------------------------------------------------------------------------- #
#  Store plumbing                                                              #
# --------------------------------------------------------------------------- #
def store(group_arrays, group, attacker, ts, ps, prob, yte, records, auc,
          extra=None):
    d = group_arrays.setdefault(group, {})
    d[f"{attacker}_ts{ts}_ps{ps}_prob"] = prob
    ykey = f"y_ts{ts}_ps{ps}"
    if ykey not in d:
        d[ykey] = np.asarray(yte, dtype=np.int16)
    rec = {"group": group, "attacker": attacker, "train_seed": ts,
           "probe_seed": ps, "auc": auc}
    if extra:
        rec.update(extra)
    records.append(rec)


def save_section(section, group_arrays, records, gates, minutes):
    for group, arrs in group_arrays.items():
        np.savez_compressed(SCORE_DIR / f"{group}.npz", **arrs)
    out = {"section": section, "records": records, "gates": gates,
           "minutes": minutes}
    (ARTIFACTS / f"tpr_ext_partial_{section}.json").write_text(json.dumps(out))
    n_fail = sum(1 for g in gates if not g["pass"])
    print(f"\n[{section}] saved {len(group_arrays)} npz groups, "
          f"{len(records)} auc records, gate {len(gates) - n_fail}/{len(gates)} "
          f"pass ({minutes:.1f} min)", flush=True)
    for g in gates:
        if not g["pass"]:
            print(f"  GATE FAIL {g['group']} {g['metric']}: stored="
                  f"{g['stored']:.4f} rerun={g['rerun']:.4f}", flush=True)


def load_ckpt(section):
    p = ARTIFACTS / f"tpr_ext_ckpt_{section}.json"
    if p.exists():
        d = json.loads(p.read_text())
        return d["done"], d["records"], d["gates"], p
    return [], [], [], p


def save_ckpt(p, done, records, gates, group_arrays, slug):
    np.savez_compressed(SCORE_DIR / f"{slug}.npz", **group_arrays[slug])
    done.append(slug)
    p.write_text(json.dumps({"done": done, "records": records,
                             "gates": gates}))


def gate(gates, group, metric, stored, rerun, src):
    gates.append({"group": group, "metric": metric, "stored": stored,
                  "rerun": rerun, "delta": rerun - stored, "source": src,
                  "pass": bool(abs(rerun - stored) <= TOL)})


def mean_auc(records, group, attacker, ps_set=None, ts=None):
    v = [r["auc"] for r in records
         if r["group"] == group and r["attacker"] == attacker
         and (ps_set is None or r["probe_seed"] in ps_set)
         and (ts is None or r["train_seed"] == ts)]
    return float(np.mean(v)) if v else None


# --------------------------------------------------------------------------- #
#  Section A — the 8 survivors                                                 #
# --------------------------------------------------------------------------- #
def section_a():
    t0 = time.time()
    device = get_device()
    group_arrays, records, gates = {}, [], []
    SEEDS = [0, 1, 2]

    # ---- A1: honest_reaudit adult/sex noise sigma_rel 4 and 8 ---------------
    from utils.pcrl_io import (INCOME_PREDICTION_PURPOSE_IDX,
                               build_adult_train_loader, extract_attr,
                               extract_frozen_reps, load_adult_encoder)
    hr = json.loads((RESULTS / "honest_reaudit.json").read_text())
    master = {(r["experiment"], r["method"]): r for r in hr["master"]}
    encoder, _ = load_adult_encoder()
    _, _, loader = build_adult_train_loader()
    H = extract_frozen_reps(encoder, loader, INCOME_PREDICTION_PURPOSE_IDX,
                            device)
    sex = extract_attr(loader, "sex")
    sigma_h = float(H.std(0).mean())
    n_sex = int(sex.max()) + 1
    print(f"[a1] PCRL adult frozen rep n={len(sex)} sigma_h={sigma_h:.4f}",
          flush=True)
    from sklearn.model_selection import train_test_split
    for sr in (4.0, 8.0):
        group = f"hr_noise_s{sr:g}"
        sa = sr * sigma_h
        for s in SEEDS:
            X = H.astype(np.float64) + np.random.default_rng(s).normal(
                0.0, sa, size=H.shape)
            Xtr, Xte, ytr, yte = train_test_split(
                X, sex, test_size=0.25, random_state=s, stratify=sex)
            for arch in ("XGB", "MLP"):
                prob, auc = probe_capture(arch, Xtr, ytr, Xte, yte, s, n_sex)
                store(group_arrays, group, arch, 0, s, prob, yte, records, auc)
        row = master[("E2 noise", f"noise σ={sr}")]
        for arch, key in (("XGB", "xgb_auc"), ("MLP", "mlp_auc")):
            gate(gates, group, arch, row[key],
                 mean_auc(records, group, arch),
                 "honest_reaudit.json master E2 noise")
        print(f"[a1 {group}] XGB={mean_auc(records, group, 'XGB'):.4f} "
              f"MLP={mean_auc(records, group, 'MLP'):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- A2: five expansion noise sigma=8 survivors --------------------------
    from experiments.expansion_pipeline import load_cell_by_tag
    ra = json.loads((RESULTS / "expansion_reaudit.json").read_text())
    ra_rows = {(r["cell"], r["method"]): r for r in ra["master"]}
    SURVIVOR_CELLS = ["diabetes_hospital/race/readmit_30_days",
                      "folktables/race/employment",
                      "folktables/race/public_coverage",
                      "folktables/race/income",
                      "lawschool/sex/pass_bar"]
    for cell in SURVIVOR_CELLS:
        tc = time.time()
        _, _, _, X, attr, task, n_attr, n_task, _, _ = load_cell_by_tag(cell)
        X_t = torch.from_numpy(X).to(device)
        P, _, _ = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                              alpha=0.0, device=device, pub_noise=0.0, seed=0)
        sigma_h = float(P.std(0).mean())
        sigma_abs = 8.0 * sigma_h
        d = P.shape[1]
        ch = iso_channel(sigma_abs)
        ncov = iso_noise_cov(d, sigma_abs)
        group = "exp_" + cell.replace("/", "_")
        P64 = P.astype(np.float64)
        for s in SEEDS:
            # XGB/MLP: probe_through convention (noise then split, rng(s))
            Xn = ch(P64, np.random.default_rng(s))
            Xtr, Xte, ytr, yte = train_test_split(
                Xn, attr, test_size=0.25, random_state=s, stratify=attr)
            for arch in ("XGB", "MLP"):
                prob, auc = probe_capture(arch, Xtr, ytr, Xte, yte, s, n_attr)
                store(group_arrays, group, arch, 0, s, prob, yte, records, auc)
            # LoRA: lora_channel convention (rng(100+s), probe_lora seed s)
            Hn = ch(P64, np.random.default_rng(100 + s)).astype(np.float32)
            prob, yte_l, auc = lora_capture(Hn, attr, n_attr, s, device)
            store(group_arrays, group, "LoRA", 0, s, prob, yte_l, records, auc)
            # LRT: gaussian_lrt_channel convention (rng(s), _split_idx)
            Hn2 = ch(P64, np.random.default_rng(s))
            tr, te = _split_idx(attr, s)
            prob, auc = lrt_capture(P64, Hn2[te], attr, tr, te, ncov, n_attr)
            store(group_arrays, group, "LRT", 0, s, prob, attr[te], records,
                  auc)
        row = ra_rows[(cell, "noise σ=8")]
        for arch, key in (("XGB", "xgb_auc"), ("MLP", "mlp_auc"),
                          ("LoRA", "lora_auc"), ("LRT", "lrt_auc")):
            gate(gates, group, arch, row[key],
                 mean_auc(records, group, arch),
                 "expansion_reaudit.json master noise σ=8")
        print(f"[a2 {cell}] XGB={mean_auc(records, group, 'XGB'):.4f} "
              f"MLP={mean_auc(records, group, 'MLP'):.4f} "
              f"LoRA={mean_auc(records, group, 'LoRA'):.4f} "
              f"LRT={mean_auc(records, group, 'LRT'):.4f} "
              f"({time.time() - tc:.0f}s)", flush=True)

    save_section("a", group_arrays, records, gates, (time.time() - t0) / 60)


# --------------------------------------------------------------------------- #
#  Section B1 — the 24 baseline_gauntlet combos                                #
# --------------------------------------------------------------------------- #
def rep_capture_fixed(group_arrays, records, group, P, h_clean, ncov, y,
                      n_classes, ps_seeds, device, with_lora, ts=0):
    """full_point's rep-surface members on a fixed exposed matrix, captured."""
    for s in ps_seeds:
        Xtr, Xte, ytr, yte = _split(P, y, s)
        for arch in ("XGB", "MLP"):
            prob, auc = probe_capture(arch, Xtr, ytr, Xte, yte, s, n_classes)
            store(group_arrays, group, arch, ts, s, prob, yte, records, auc)
        if with_lora:
            prob, yte_l, auc = lora_capture(P.astype(np.float32), y, n_classes,
                                            s, device)
            store(group_arrays, group, "LoRA", ts, s, prob, yte_l, records,
                  auc)
        tr, te = _split_idx(y, s)
        P64 = P.astype(np.float64)
        prob, auc = lrt_capture(h_clean.astype(np.float64), P64[te], y, tr, te,
                                ncov, n_classes)
        store(group_arrays, group, "LRT", ts, s, prob, y[te], records, auc)


def section_b1():
    t0 = time.time()
    device = get_device()
    from experiments.baseline_gauntlet import (CERT_SEEDS, SWEEP_SEEDS,
                                               make_exposures)
    bg = json.loads((RESULTS / "baseline_gauntlet.json").read_text())
    cells_json = {c["cell"]: c for c in bg["cells"]}
    group_arrays = {}
    done, records, gates, ckpt_path = load_ckpt("b1")

    # One job per distinct exposure slug; cert and sweep scorings of the same
    # exposure merge into one capture (union of train/probe seeds, both gates).
    jobs = {}
    for cell_name, cj in cells_json.items():
        for b in cj["baselines"]:
            for tk in ("tier1", "tier2"):
                t = b[tk]
                if t["certified"]:
                    lbl, knob = t["cert"]["label"], t["cert"]["knob"]
                    tss, ps, lora = [0, 1, 2], list(CERT_SEEDS), True
                    gate_spec = ("cert", t["cert"])
                else:
                    r = t["closest"]
                    lbl, knob = r["label"], r["knob"]
                    tss, ps = [0], list(SWEEP_SEEDS)
                    lora = "lora" in r and r.get("lora") is not None
                    gate_spec = ("sweep", r)
                slug = bg_slug(cell_name, b["baseline"], lbl)
                j = jobs.setdefault(slug, dict(
                    cell=cell_name, baseline=b["baseline"], label=lbl,
                    knob=knob, train_seeds=set(), ps=set(), with_lora=False,
                    gate_specs={}))
                j["train_seeds"] |= set(tss)
                j["ps"] |= set(ps)
                j["with_lora"] = j["with_lora"] or lora
                j["gate_specs"][gate_spec[0]] = gate_spec[1]
    JOBS = [dict(slug=s, **j) for s, j in jobs.items()]
    for j in JOBS:
        j["train_seeds"] = sorted(j["train_seeds"])
        j["ps"] = sorted(j["ps"])

    for job in JOBS:
        if job["slug"] in done:
            print(f"[b1 {job['slug']}] checkpointed, skipping", flush=True)
            continue
        tc = time.time()
        cell = job["cell"]
        ds, at, tk_ = cell.split("/")
        X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk_)
        ctx = dict(name=cell, X_t=torch.from_numpy(X).to(device), attr=attr,
                   task=task, n_attr=n_attr, n_task=n_task, task_maj=task_maj)
        slug = job["slug"]
        for ts in job["train_seeds"]:
            exps = make_exposures(job["baseline"], job["knob"], ctx, device,
                                  ts)
            exp = next(e for e in exps if e["label"] == job["label"])
            rep_capture_fixed(group_arrays, records, slug, exp["P"],
                              exp["h_clean"], exp["ncov"], attr, n_attr,
                              job["ps"], device, job["with_lora"], ts=ts)
        # reproduction gates (cert per-seed and/or sweep-row means)
        if "cert" in job["gate_specs"]:
            per = job["gate_specs"]["cert"]["per_seed"]
            for ts in job["train_seeds"]:
                for arch, key in (("XGB", "rep_xgb"), ("MLP", "rep_mlp"),
                                  ("LoRA", "rep_lora"), ("LRT", "rep_lrt")):
                    gate(gates, slug, f"{arch}_ts{ts}", per[key][ts],
                         mean_auc(records, slug, arch, ts=ts),
                         "baseline_gauntlet.json cert per_seed")
        if "sweep" in job["gate_specs"]:
            r = job["gate_specs"]["sweep"]
            pairs = [("XGB", "xgb"), ("MLP", "mlp"), ("LRT", "lrt")]
            if "lora" in r and r.get("lora") is not None:
                pairs.append(("LoRA", "lora"))
            for arch, key in pairs:
                gate(gates, slug, arch, r[key],
                     mean_auc(records, slug, arch, ps_set={0, 1}, ts=0),
                     "baseline_gauntlet.json sweep row")
        save_ckpt(ckpt_path, done, records, gates, group_arrays, slug)
        print(f"[b1 {slug}] "
              + " ".join(f"{a}={mean_auc(records, slug, a):.4f}"
                         for a in ("XGB", "MLP", "LoRA", "LRT")
                         if mean_auc(records, slug, a) is not None)
              + f" ({time.time() - tc:.0f}s)", flush=True)

    save_section("b1", group_arrays, records, gates, (time.time() - t0) / 60)


# --------------------------------------------------------------------------- #
#  Section B2 — the 6 Obliviator combos (iter 15, 3 cells)                     #
# --------------------------------------------------------------------------- #
def section_b2():
    t0 = time.time()
    device = get_device()
    from experiments.obliviator_gauntlet import run_obliviator
    ob = json.loads((RESULTS / "obliviator_gauntlet.json").read_text())
    group_arrays = {}
    done, records, gates, ckpt_path = load_ckpt("b2")
    for cj in ob["cells"]:
        cell = cj["cell"]
        if f"ob_{CELL_SLUG[cell]}_iter15" in done:
            print(f"[b2 {cell}] checkpointed, skipping", flush=True)
            continue
        tc = time.time()
        ds, at, tk_ = cell.split("/")
        X, attr, task, n_attr, n_task, _, _ = load_cell(ds, at, tk_)
        X_t = torch.from_numpy(X).to(device)
        H, _, _ = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                              alpha=0.0, device=device, pub_noise=0.0, seed=0)
        exps, stop_it, _ = run_obliviator(H, attr, task, n_attr, n_task, 0,
                                          str(device), want_iters={15})
        e = next(x for x in exps if x["iter"] == 15)
        z = e["z"]
        d = z.shape[1]
        slug = f"ob_{CELL_SLUG[cell]}_iter15"
        rep_capture_fixed(group_arrays, records, slug, z, z,
                          np.zeros((d, d)), attr, n_attr, [0, 1], device,
                          with_lora=False)
        r = cj["tier1"]["closest"]
        assert r["label"] == "iter 15", r["label"]
        for arch, key in (("XGB", "xgb"), ("MLP", "mlp"), ("LRT", "lrt")):
            gate(gates, slug, arch, r[key], mean_auc(records, slug, arch),
                 "obliviator_gauntlet.json closest iter 15")
        save_ckpt(ckpt_path, done, records, gates, group_arrays, slug)
        print(f"[b2 {slug}] z_dim={d} stop_it={stop_it} "
              f"XGB={mean_auc(records, slug, 'XGB'):.4f} "
              f"MLP={mean_auc(records, slug, 'MLP'):.4f} "
              f"LRT={mean_auc(records, slug, 'LRT'):.4f} "
              f"({(time.time() - tc) / 60:.1f} min)", flush=True)
    save_section("b2", group_arrays, records, gates, (time.time() - t0) / 60)


# --------------------------------------------------------------------------- #
#  Section B3 — the 6 official-LAFTR combos                                    #
# --------------------------------------------------------------------------- #
def section_b3():
    t0 = time.time()
    device = get_device()
    from experiments.laftr_official import (CELLS as LO_CELLS, load_exposure,
                                            run_official)
    lo = json.loads((RESULTS / "laftr_official.json").read_text())
    lo_cells = {c["cell"]: c for c in lo["cells"]}
    group_arrays, records, gates = {}, [], []
    WANT = {"hmda/race/loan_decision": 1.0,
            "hmda/race/loan_amount_band": 1.0,
            "adult/sex/income": 8.0}
    for cell_cfg in LO_CELLS:
        cell = cell_cfg["name"]
        g = WANT[cell]
        tc = time.time()
        ds, at, tk_ = cell.split("/")
        X, attr, task, n_attr, n_task, _, _ = load_cell(ds, at, tk_)
        exp_dir = run_official(cell_cfg["tag"], cell_cfg["xdim"], g, 0)
        exp, attr_n, task_n = load_exposure(exp_dir, attr, task, g)
        Z = exp["P"]
        d = Z.shape[1]
        slug = f"lo_{CELL_SLUG[cell]}_g{g:g}"
        rep_capture_fixed(group_arrays, records, slug, Z, Z,
                          np.zeros((d, d)), attr_n, n_attr, [0, 1], device,
                          with_lora=False)
        r = lo_cells[cell]["tier1"]["closest"]
        assert r["label"] == f"γ={g:g}", r["label"]
        for arch, key in (("XGB", "xgb"), ("MLP", "mlp"), ("LRT", "lrt")):
            gate(gates, slug, arch, r[key], mean_auc(records, slug, arch),
                 "laftr_official.json closest row")
        print(f"[b3 {slug}] XGB={mean_auc(records, slug, 'XGB'):.4f} "
              f"MLP={mean_auc(records, slug, 'MLP'):.4f} "
              f"LRT={mean_auc(records, slug, 'LRT'):.4f} "
              f"({(time.time() - tc) / 60:.1f} min)", flush=True)
    save_section("b3", group_arrays, records, gates, (time.time() - t0) / 60)


# --------------------------------------------------------------------------- #
#  Merge — TPR blocks, prediction scoring, final JSON                          #
# --------------------------------------------------------------------------- #
def tpr_at_fpr(y_bin, score, fpr_target):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_bin, score)
    ok = fpr <= fpr_target + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def tpr_block(prob, yte, n_classes):
    out = {}
    if n_classes == 2:
        for f in FPRS:
            out[f"tpr@{f}"] = tpr_at_fpr((yte == 1).astype(int), prob[:, 1], f)
        out["n_pos_eval"] = int((yte == 1).sum())
        out["headline"] = {f: out[f"tpr@{f}"] for f in FPRS}
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
        out["headline"] = {f: out[f"worst_class_tpr@{f}"] for f in FPRS}
    return out


def combo_tpr(arrs, records, group, suite, ps_set, train_seeds):
    """Best attacker in `suite` by mean AUC over the original grid, then TPR
    per (ts, ps) from its captured scores."""
    means = {a: mean_auc(records, group, a, ps_set=ps_set) for a in suite}
    means = {a: v for a, v in means.items() if v is not None}
    best = max(means, key=means.get)
    per_seed, h1, h001 = {}, [], []
    n_classes = None
    for r in records:
        if (r["group"] != group or r["attacker"] != best
                or r["probe_seed"] not in ps_set
                or r["train_seed"] not in train_seeds):
            continue
        ts, ps = r["train_seed"], r["probe_seed"]
        prob = arrs[f"{best}_ts{ts}_ps{ps}_prob"]
        yte = arrs[f"y_ts{ts}_ps{ps}"]
        n_classes = prob.shape[1]
        blk = tpr_block(prob, yte, n_classes)
        per_seed[f"ts{ts}_ps{ps}"] = {"auc": r["auc"], **blk}
        h001.append(blk["headline"][0.001])
        h1.append(blk["headline"][0.01])
    return {"best_attacker": best, "best_auc_mean": means[best],
            "suite_auc_means": means, "n_classes": n_classes,
            "tpr@0.001_mean": float(np.mean(h001)),
            "tpr@0.001_worst": float(np.max(h001)),
            "tpr@0.01_mean": float(np.mean(h1)),
            "tpr@0.01_worst": float(np.max(h1)),
            "per_seed": per_seed}


def merge():
    header = json.loads((RESULTS / "tpr_extension.json").read_text())
    records, gates = [], []
    sections_done = []
    for sec in ("a", "b1", "b2", "b3"):
        p = ARTIFACTS / f"tpr_ext_partial_{sec}.json"
        if p.exists():
            d = json.loads(p.read_text())
            records.extend(d["records"])
            gates.extend(d["gates"])
            sections_done.append(sec)
    arrs_cache = {}

    def arrs(group):
        if group not in arrs_cache:
            arrs_cache[group] = dict(np.load(SCORE_DIR / f"{group}.npz"))
        return arrs_cache[group]

    have = {r["group"] for r in records}
    configs = []

    # ---- population (a) ------------------------------------------------------
    A_HR = [("honest_reaudit|adult/sex|E2 noise σ=4.0", "hr_noise_s4", None),
            ("honest_reaudit|adult/sex|E2 noise σ=8.0", "hr_noise_s8", None),
            ("honest_reaudit|adult/sex|E4S1 noise sigma=8.0", "hr_noise_s8",
             "shares the E2 σ=8 measurement (identical stored AUCs)")]
    for cid, group, note in A_HR:
        if group not in have:
            continue
        c = {"config": cid, "population": "a_survivors",
             "suite": "honest (XGB+MLP)",
             **combo_tpr(arrs(group), records, group, ["XGB", "MLP"],
                         {0, 1, 2}, {0})}
        if note:
            c["note"] = note
        configs.append(c)
    for cell in ["diabetes_hospital/race/readmit_30_days",
                 "folktables/race/employment",
                 "folktables/race/public_coverage",
                 "folktables/race/income",
                 "lawschool/sex/pass_bar"]:
        group = "exp_" + cell.replace("/", "_")
        if group not in have:
            continue
        base = dict(config=f"expansion|{cell}|noise σ=8",
                    population="a_survivors", suite="tier1 (XGB+MLP+LoRA)")
        configs.append({**base,
                        **combo_tpr(arrs(group), records, group,
                                    ["XGB", "MLP", "LoRA"], {0, 1, 2}, {0})})
        configs.append({**base, "config": base["config"] + "|tier2",
                        "suite": "tier2 (+LRT)", "tier2_block": True,
                        **combo_tpr(arrs(group), records, group,
                                    ["XGB", "MLP", "LoRA", "LRT"],
                                    {0, 1, 2}, {0})})

    # ---- population (b) ------------------------------------------------------
    bg = json.loads((RESULTS / "baseline_gauntlet.json").read_text())
    for cj in bg["cells"]:
        cell = cj["cell"]
        for b in cj["baselines"]:
            for tk in ("tier1", "tier2"):
                t = b[tk]
                if t["certified"]:
                    lbl, ps, tss = t["cert"]["label"], {0, 1, 2}, {0, 1, 2}
                    with_lora, certified = True, True
                else:
                    r = t["closest"]
                    lbl, ps, tss = r["label"], {0, 1}, {0}
                    with_lora = "lora" in r and r.get("lora") is not None
                    certified = False
                slug = bg_slug(cell, b["baseline"], lbl)
                if slug not in have:
                    continue
                suite = ["XGB", "MLP"] + (["LoRA"] if with_lora else [])
                if tk == "tier2":
                    suite = suite + ["LRT"]
                configs.append({
                    "config": f"gauntlet|{cell}|{b['baseline']}|{tk}|{lbl}",
                    "population": "b_gauntlet36", "certified": certified,
                    "suite": "+".join(suite),
                    **combo_tpr(arrs(slug), records, slug, suite, ps, tss)})
    for src, prefix, jf in (("obliviator", "ob", "obliviator_gauntlet.json"),
                            ("laftr_official", "lo", "laftr_official.json")):
        d = json.loads((RESULTS / jf).read_text())
        for cj in d["cells"]:
            cell = cj["cell"]
            for tk in ("tier1", "tier2"):
                r = cj[tk]["closest"]
                lbl = r["label"]
                slug = (f"{prefix}_{CELL_SLUG[cell]}_"
                        + lbl.replace(" ", "").replace("γ=", "g"))
                if prefix == "ob":
                    slug = f"ob_{CELL_SLUG[cell]}_iter15"
                if slug not in have:
                    continue
                suite = ["XGB", "MLP"] + (["LRT"] if tk == "tier2" else [])
                configs.append({
                    "config": f"{src}|{cell}|{tk}|{lbl}",
                    "population": "b_gauntlet36", "certified": False,
                    "suite": "+".join(suite),
                    **combo_tpr(arrs(slug), records, slug, suite, {0, 1},
                                {0})})

    n_fail = sum(1 for g in gates if not g["pass"])
    header["results"] = {
        "sections_completed": sections_done,
        "n_configs": len(configs),
        "configs": configs,
        "auc_records": records,
        "reproduction_gate": {"tolerance": TOL, "n_comparisons": len(gates),
                              "n_fail": n_fail, "comparisons": gates},
    }
    (RESULTS / "tpr_extension.json").write_text(json.dumps(header, indent=1))
    print(f"[merge] {len(configs)} configs, gate {len(gates) - n_fail}/"
          f"{len(gates)} pass -> tpr_extension.json", flush=True)
    for c in configs:
        print(f"  {c['config']:<62} best={c['best_attacker']:<4} "
              f"auc={c['best_auc_mean']:.4f} tpr@1%={c['tpr@0.01_mean']:.4f} "
              f"tpr@0.1%={c['tpr@0.001_mean']:.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", required=True,
                    choices=["a", "b1", "b2", "b3", "merge"])
    args = ap.parse_args()
    {"a": section_a, "b1": section_b1, "b2": section_b2, "b3": section_b3,
     "merge": merge}[args.section]()
