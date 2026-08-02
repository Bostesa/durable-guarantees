"""Baseline dual scoring, v2 — same measurement, ~1.8x less work.

v1 trained and scored each (method, cell) twice, once per tier, even though
tier-1 and tier-2 use the SAME operating point in 7 of the 8 combinations.
v2 trains once per unique knob and scores the representation once with all four
battery members, then reads tier-1 as the max over XGB/MLP/LoRA and tier-2 as
the max over those plus the Gaussian LRT. Identical numbers, half the compute.

Method order is by decision relevance, so a partial run is still useful:
VFAE and DANN-scrub can plausibly move a verdict (macro 0.53-0.66, near the
bar); LEACE and LAFTR read 0.99+ and 0.85-1.00 and cannot. FARE and LAFTR on
the easy cell are already measured and are carried in from _baseline_partial.json.

Results are written after every (method, cell) so an interrupted run leaves a
usable file.
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
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results"
# Generated intermediates; gitignored and regenerable.
# Override the location with DG_ARTIFACTS.
ARTIFACTS = Path(os.environ.get("DG_ARTIFACTS", REPO / "analysis"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from experiments.baseline_gauntlet import make_exposures  # noqa: E402
from experiments.diagnostic import load_cell  # noqa: E402
from run_multiclass_dual_report import MIN_CLASS_ROWS, macro_ovr, wp  # noqa: E402
from run_worstpair_sweep import PROBE_SEEDS, lora_probs, probe_probs  # noqa: E402
from utils.battery import _lrt_scores, _split_idx  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

BAR = 0.55
TRAIN_SEEDS = [0, 1, 2]
CELLS = {"easy": ("hmda", "race", "loan_decision"),
         "middle": ("hmda", "race", "loan_amount_band")}
KNOBS = {("easy", "LAFTR"): {1: 8.0, 2: 2.0},
         ("easy", "VFAE"): {1: 1.0, 2: 1.0},
         ("easy", "DANN-scrub"): {1: 20.0, 2: 20.0},
         ("easy", "LEACE"): {1: 0.0, 2: 0.0},
         ("middle", "LAFTR"): {1: 2.0, 2: 2.0},
         ("middle", "VFAE"): {1: 1.0, 2: 1.0},
         ("middle", "DANN-scrub"): {1: 20.0, 2: 20.0},
         ("middle", "LEACE"): {1: 0.0, 2: 0.0}}
ORDER = [("middle", "FARE"), ("middle", "VFAE"), ("easy", "VFAE"),
         ("middle", "DANN-scrub"), ("easy", "DANN-scrub"),
         ("middle", "LEACE"), ("easy", "LEACE"), ("middle", "LAFTR")]
VFAE_LABEL = "β=1 sampled-z"


def score_all(H, attr, n_attr, keep, device, h_clean, ncov):
    """One scoring pass; returns tier-1 and tier-2 (macro, all, supported)."""
    t1 = {"mac": [], "all": [], "sup": []}
    lrt = {"mac": [], "all": [], "sup": []}

    def add(b, y, p):
        b["mac"].append(macro_ovr(y, p)); b["all"].append(wp(y, p, None))
        b["sup"].append(wp(y, p, keep))

    for s in PROBE_SEEDS:
        for arch in ("XGB", "MLP"):
            y, p = probe_probs(arch, np.asarray(H, np.float64), attr, n_attr, s)
            add(t1, y, p)
        y, p = lora_probs(np.asarray(H, np.float32), attr, n_attr, s, device)
        add(t1, y, p)
        if h_clean is not None:
            tr, te = _split_idx(attr, s)
            post = _lrt_scores(np.asarray(h_clean, np.float64)[tr], attr[tr],
                               np.asarray(H, np.float64)[te], ncov, n_attr)
            add(lrt, attr[te], post)
    out1 = {k: float(max(v)) for k, v in t1.items()}
    out2 = {k: float(max(t1[k] + lrt[k])) if lrt[k] else out1[k] for k in t1}
    return out1, out2


def lr_logits(Z, task, seed):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    Zd = np.asarray(Z, np.float64)
    lr.fit(Zd, task)
    return np.log(np.clip(lr.predict_proba(Zd), 1e-12, None)).astype(np.float32)


def emit(rows, cell, method, tier, surface, blob, knob, refit):
    r = dict(cell=cell, method=method, tier=tier, surface=surface,
             operating_point=knob, refit=refit, macro=blob["mac"],
             all_pairs=blob["all"], supported=blob["sup"],
             passes_macro=bool(blob["mac"] <= BAR),
             passes_supported=bool(blob["sup"] <= BAR))
    rows.append(r)
    print("  %-12s T%d %-4s macro=%.4f all=%.4f sup=%.4f  %s"
          % (method, tier, surface, r["macro"], r["all_pairs"], r["supported"],
             "passes" if r["passes_supported"] else "FAILS"), flush=True)


def main():
    t0 = time.time()
    device = get_device()
    rows = json.loads((ARTIFACTS / "_baseline_partial.json").read_text())
    for r in rows:
        r.setdefault("refit", r["method"] != "FARE")
        r.setdefault("passes_macro", r["macro"] <= BAR)
        r.setdefault("passes_supported", r["supported"] <= BAR)
    print(f"[setup] device={device} carried in {len(rows)} rows "
          f"(FARE/easy, LAFTR/easy)\n", flush=True)

    fare = json.loads((RESULTS / "fare_gauntlet.json").read_text())
    fare_tag = {c["key"]: ((c.get("tier1") or {}).get("cert") or {}).get("tag")
                for c in fare["cells"]}
    cache = {}

    for cellk, method in ORDER:
        ds, at, tk = CELLS[cellk]
        if cellk not in cache:
            X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk)
            counts = np.bincount(attr, minlength=n_attr)
            cache[cellk] = (X, attr, task, n_attr, n_task, task_maj, counts,
                            [bool(v >= MIN_CLASS_ROWS) for v in counts])
            print(f"### {cellk} counts={counts.tolist()} "
                  f"supported={[i for i, v in enumerate(cache[cellk][7]) if v]}",
                  flush=True)
        X, attr, task, n_attr, n_task, task_maj, counts, keep = cache[cellk]
        t1 = time.time()

        if method == "FARE":
            emb = np.load(ARTIFACTS / "fare_cells" / f"{cellk}_fare_embeddings.npz")
            base = fare_tag[cellk].rsplit("_ts", 1)[0]
            accR, accO = [], []
            for ts in TRAIN_SEEDS:
                ck, mk = f"{base}_ts{ts}_cells", f"{base}_ts{ts}_medians"
                if ck not in emb:
                    continue
                Z = emb[mk][emb[ck]].astype(np.float32)
                L = lr_logits(Z, task, ts)
                accR.append(score_all(Z, attr, n_attr, keep, device, Z,
                                      np.zeros((Z.shape[1],) * 2)))
                accO.append(score_all(L, attr, n_attr, keep, device, L,
                                      np.zeros((L.shape[1],) * 2)))
            for tier in (1, 2):
                i = tier - 1
                for surf, acc in (("rep", accR), ("out", accO)):
                    blob = {k: max(a[i][k] for a in acc) for k in ("mac", "all", "sup")}
                    emit(rows, cellk, "FARE", tier, surf, blob, base, False)
        else:
            knobs = KNOBS[(cellk, method)]
            uniq = sorted(set(knobs.values()))
            per_knob = {}
            X_t = torch.from_numpy(X).to(device)
            ctx = dict(X_t=X_t, attr=attr, task=task, n_attr=n_attr,
                       n_task=n_task, task_maj=task_maj, name=f"{ds}/{at}/{tk}")
            for kb in uniq:
                accR, accO = [], []
                for ts in TRAIN_SEEDS:
                    exps = make_exposures(method, kb, ctx, device, ts)
                    exp = (next((e for e in exps if e["label"] == VFAE_LABEL), exps[0])
                           if method == "VFAE" else exps[0])
                    P, L = exp["P"], exp["L"]
                    accR.append(score_all(P, attr, n_attr, keep, device,
                                          exp["h_clean"], exp["ncov"]))
                    accO.append(score_all(L, attr, n_attr, keep, device, L,
                                          np.zeros((L.shape[1],) * 2)))
                per_knob[kb] = (accR, accO)
            for tier in (1, 2):
                accR, accO = per_knob[knobs[tier]]
                i = tier - 1
                for surf, acc in (("rep", accR), ("out", accO)):
                    blob = {k: max(a[i][k] for a in acc) for k in ("mac", "all", "sup")}
                    emit(rows, cellk, method, tier, surf, blob, knobs[tier], True)
        print(f"    ({(time.time() - t1) / 60:.1f} min)", flush=True)

        flips = [r for r in rows if r["passes_macro"] != r["passes_supported"]]
        (RESULTS / "baseline_dual_score.json").write_text(json.dumps(dict(
            experiment="baseline_dual_score_v2", bar=BAR,
            min_class_rows=MIN_CLASS_ROWS, train_seeds=TRAIN_SEEDS,
            probe_seeds=PROBE_SEEDS,
            nulls=dict(all_pairs=0.5556, supported=0.5217),
            note=("macro/all/supported are MAX over 3 train x 3 probe seeds, "
                  "stricter than the paper's mean; compare supported vs macro "
                  "within this run"),
            rows=rows, n_flips=len(flips), flips=flips,
            not_covered=["Obliviator (separate code path, not run here)"],
            minutes=round((time.time() - t0) / 60, 1)), indent=1, default=float))

    print(f"\n  rows {len(rows)}  pass macro {sum(r['passes_macro'] for r in rows)}  "
          f"pass supported {sum(r['passes_supported'] for r in rows)}")
    print(f"[save] {RESULTS / 'baseline_dual_score.json'} "
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
