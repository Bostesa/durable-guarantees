"""Dual scoring (macro-OVR and supported-pair) for the published baselines on
the two multiclass HMDA/race cells — the entries Table 1 reports.

Motivation: our own configurations are dual-scored in multiclass_dual_report.json,
but no baseline had stored per-class scores, so Table 1 could not be given the
same treatment. This produces it.

Two paths, both at the operating point Table 1 reports:
  * FARE  — its release is a deterministic leaf-median embedding, and both the
    per-row leaf assignment and the per-leaf medians are already stored in
    fare_cells/*_fare_embeddings.npz. The release is rebuilt exactly
    (z = medians[cells]); NO tree is refit.
  * LAFTR / VFAE / DANN-scrub / LEACE — retrained at their reported knob via
    baseline_gauntlet.make_exposures, 3 training seeds.

Scoring matches multiclass_dual_report.py: macro-OVR, all-pairs worst-pair, and
supported-pair worst-pair (both classes >= 3000 rows), MAX over 3 training x 3
probe seeds, on the representation and the output surface. Obliviator is handled
separately (different code path) and is reported as not covered here.
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
REPO = HERE.parent.parent
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
FARE_CELLKEY = {"easy": "easy", "middle": "middle"}
# operating points Table 1 reports (baseline_gauntlet.json cert/closest)
KNOBS = {
    ("easy", "LAFTR"): {1: 8.0, 2: 2.0},
    ("easy", "VFAE"): {1: 1.0, 2: 1.0},
    ("easy", "DANN-scrub"): {1: 20.0, 2: 20.0},
    ("easy", "LEACE"): {1: 0.0, 2: 0.0},
    ("middle", "LAFTR"): {1: 2.0, 2: 2.0},
    ("middle", "VFAE"): {1: 1.0, 2: 1.0},
    ("middle", "DANN-scrub"): {1: 20.0, 2: 20.0},
    ("middle", "LEACE"): {1: 0.0, 2: 0.0},
}
VFAE_LABEL = "β=1 sampled-z"


def score_matrix(H, attr, n_attr, keep, device, tier, h_clean=None, ncov=None):
    mac, allp, sup = [], [], []

    def add(y, p):
        mac.append(macro_ovr(y, p)); allp.append(wp(y, p, None))
        sup.append(wp(y, p, keep))

    for s in PROBE_SEEDS:
        for arch in ("XGB", "MLP"):
            y, p = probe_probs(arch, np.asarray(H, np.float64), attr, n_attr, s)
            add(y, p)
        y, p = lora_probs(np.asarray(H, np.float32), attr, n_attr, s, device)
        add(y, p)
        if tier == 2 and h_clean is not None:
            tr, te = _split_idx(attr, s)
            post = _lrt_scores(np.asarray(h_clean, np.float64)[tr], attr[tr],
                               np.asarray(H, np.float64)[te], ncov, n_attr)
            add(attr[te], post)
    return max(mac), max(allp), max(sup)


def lr_logits(Z, task, seed):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    Zd = np.asarray(Z, np.float64)
    lr.fit(Zd, task)
    return np.log(np.clip(lr.predict_proba(Zd), 1e-12, None)).astype(np.float32)


def main():
    t0 = time.time()
    device = get_device()
    fare = json.loads((HERE / "fare_gauntlet.json").read_text())
    fare_tag = {}
    for c in fare["cells"]:
        for t in ("tier1", "tier2"):
            x = c.get(t) or {}
            if x.get("certified"):
                fare_tag[(c["key"], int(t[-1]))] = x["cert"]["tag"]
    rows = []
    print(f"[setup] device={device} seeds={TRAIN_SEEDS} probes={PROBE_SEEDS} "
          f"MIN_CLASS_ROWS={MIN_CLASS_ROWS}\n", flush=True)

    for cellk, (ds, at, tk) in CELLS.items():
        X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk)
        counts = np.bincount(attr, minlength=n_attr)
        keep = [bool(v >= MIN_CLASS_ROWS) for v in counts]
        X_t = torch.from_numpy(X).to(device)
        print(f"### {cellk} {ds}/{at}/{tk}  counts={counts.tolist()} "
              f"supported={[i for i, v in enumerate(keep) if v]}", flush=True)

        # ---- FARE: rebuild the stored release, no refit ---------------------
        emb = np.load(HERE / "fare_cells" / f"{FARE_CELLKEY[cellk]}_fare_embeddings.npz")
        for tier in (1, 2):
            tag = fare_tag.get((cellk, tier))
            if tag is None:
                continue
            base = tag.rsplit("_ts", 1)[0]
            mac_a, all_a, sup_a, mac_o, all_o, sup_o = [], [], [], [], [], []
            for ts in TRAIN_SEEDS:
                ck, mk = f"{base}_ts{ts}_cells", f"{base}_ts{ts}_medians"
                if ck not in emb:
                    continue
                Z = emb[mk][emb[ck]].astype(np.float32)
                L = lr_logits(Z, task, ts)
                d = Z.shape[1]
                a, b, c = score_matrix(Z, attr, n_attr, keep, device, tier, Z,
                                       np.zeros((d, d)))
                mac_a.append(a); all_a.append(b); sup_a.append(c)
                dl = L.shape[1]
                a, b, c = score_matrix(L, attr, n_attr, keep, device, tier, L,
                                       np.zeros((dl, dl)))
                mac_o.append(a); all_o.append(b); sup_o.append(c)
            if not mac_a:
                continue
            for surf, (m, al, sp) in (("rep", (mac_a, all_a, sup_a)),
                                      ("out", (mac_o, all_o, sup_o))):
                r = dict(cell=cellk, method="FARE", tier=tier, surface=surf,
                         operating_point=base, refit=False,
                         macro=float(max(m)), all_pairs=float(max(al)),
                         supported=float(max(sp)),
                         passes_macro=bool(max(m) <= BAR),
                         passes_supported=bool(max(sp) <= BAR))
                rows.append(r)
                print("  %-12s T%d %-4s macro=%.4f all=%.4f sup=%.4f  %s"
                      % ("FARE", tier, surf, r["macro"], r["all_pairs"],
                         r["supported"],
                         "passes" if r["passes_supported"] else "FAILS"), flush=True)

        # ---- retrained baselines -------------------------------------------
        ctx = dict(X_t=X_t, attr=attr, task=task, n_attr=n_attr, n_task=n_task,
                   task_maj=task_maj, name=f"{ds}/{at}/{tk}")
        for method in ("LAFTR", "VFAE", "DANN-scrub", "LEACE"):
            for tier in (1, 2):
                knob = KNOBS[(cellk, method)][tier]
                acc = {k: [] for k in ("mr", "ar", "sr", "mo", "ao", "so")}
                for ts in TRAIN_SEEDS:
                    exps = make_exposures(method, knob, ctx, device, ts)
                    exp = (next((e for e in exps if e["label"] == VFAE_LABEL), exps[0])
                           if method == "VFAE" else exps[0])
                    P, L = exp["P"], exp["L"]
                    a, b, c = score_matrix(P, attr, n_attr, keep, device, tier,
                                           exp["h_clean"], exp["ncov"])
                    acc["mr"].append(a); acc["ar"].append(b); acc["sr"].append(c)
                    dl = L.shape[1]
                    a, b, c = score_matrix(L, attr, n_attr, keep, device, tier, L,
                                           np.zeros((dl, dl)))
                    acc["mo"].append(a); acc["ao"].append(b); acc["so"].append(c)
                for surf, ks in (("rep", ("mr", "ar", "sr")), ("out", ("mo", "ao", "so"))):
                    r = dict(cell=cellk, method=method, tier=tier, surface=surf,
                             operating_point=knob, refit=True,
                             macro=float(max(acc[ks[0]])),
                             all_pairs=float(max(acc[ks[1]])),
                             supported=float(max(acc[ks[2]])),
                             passes_macro=bool(max(acc[ks[0]]) <= BAR),
                             passes_supported=bool(max(acc[ks[2]]) <= BAR))
                    rows.append(r)
                    print("  %-12s T%d %-4s macro=%.4f all=%.4f sup=%.4f  %s"
                          % (method, tier, surf, r["macro"], r["all_pairs"],
                             r["supported"],
                             "passes" if r["passes_supported"] else "FAILS"), flush=True)
                (HERE / "baseline_dual_score.json").write_text(
                    json.dumps(dict(rows=rows), indent=1, default=float))

    flips = [r for r in rows if r["passes_macro"] != r["passes_supported"]]
    print("\n" + "=" * 84)
    print(f"  rows: {len(rows)}   pass macro {sum(r['passes_macro'] for r in rows)}   "
          f"pass supported {sum(r['passes_supported'] for r in rows)}")
    print(f"  verdict flips (macro vs supported): {len(flips)}")
    for r in flips:
        print("    %-12s %-7s T%d %-4s macro=%.4f -> supported=%.4f (%s)"
              % (r["method"], r["cell"], r["tier"], r["surface"], r["macro"],
                 r["supported"],
                 "gains" if r["passes_supported"] else "loses"))
    out = dict(experiment="baseline_dual_score", bar=BAR,
               min_class_rows=MIN_CLASS_ROWS, train_seeds=TRAIN_SEEDS,
               probe_seeds=PROBE_SEEDS, nulls=dict(all_pairs=0.5556, supported=0.5217),
               rows=rows, n_flips=len(flips), flips=flips,
               not_covered=["Obliviator (separate code path, "
                            "obliviator_gauntlet.py; not run here)"],
               minutes=round((time.time() - t0) / 60, 1))
    (HERE / "baseline_dual_score.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {HERE / 'baseline_dual_score.json'} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
