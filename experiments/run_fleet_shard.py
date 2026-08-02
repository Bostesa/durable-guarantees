"""One fleet work unit: (method, cell, knob, training seed).

Writes fleet_shards/<method>_<cell>_k<knob>_ts<seed>.json holding the FULL
per-(surface, criterion, architecture, probe seed) AUC matrix -- nothing is
collapsed -- so the paper convention (max over architectures of the mean over
seeds) and the strict convention (max over everything) are both computable
downstream from one pass. Scoring is ~4% of unit cost, so this is free.

FARE is rebuilt exactly from its stored leaf assignments and per-leaf medians
(no tree is refit). LAFTR / VFAE / DANN-scrub / LEACE are retrained at their
Table-1 knob via baseline_gauntlet.make_exposures.

Usage: run_fleet_shard.py <method> <cell> <knob> <train_seed>
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
from experiments.diagnostic import lift, load_cell  # noqa: E402
from run_gate_shard import ARCHS, PROBE_SEEDS, score_surface  # noqa: E402
from run_worstpair_sweep import CELLS as HMDA_CELLS  # noqa: E402
from run_worstpair_supported_sweep import MIN_CLASS_ROWS, supported_mask  # noqa: E402
from utils.battery import _lrt_scores, _split_idx  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

# the hard cell is binary, so supported-pair coincides with macro-OVR there by
# construction; it is included so the seed-convention rescoring covers every
# Table-1 entry, not only the multiclass ones.
CELLS = dict(HMDA_CELLS, hard=("adult", "sex", "income"))

VFAE_LABEL = "β=1 sampled-z"


def lr_logits(Z, task, seed):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    Zd = np.asarray(Z, np.float64)
    lr.fit(Zd, task)
    return np.log(np.clip(lr.predict_proba(Zd), 1e-12, None)).astype(np.float32)


def lrt_matrix(h_clean, H, attr, ncov, n_attr, keep):
    """Per-probe-seed Gaussian-LRT AUCs, same three criteria. Tier-2 member."""
    from run_gate_shard import macro_ovr, worst_pair, worst_pair_supported
    out = {c: [] for c in ("macro", "all", "sup")}
    for s in PROBE_SEEDS:
        tr, te = _split_idx(attr, s)
        post = _lrt_scores(np.asarray(h_clean, np.float64)[tr], attr[tr],
                           np.asarray(H, np.float64)[te], ncov, n_attr)
        y = attr[te]
        out["macro"].append(float(macro_ovr(y, post)))
        out["all"].append(float(worst_pair(y, post)[0]))
        out["sup"].append(float(worst_pair_supported(y, post, keep)[0]))
    return out


def main():
    method, cellk, knob, ts = (sys.argv[1], sys.argv[2],
                               float(sys.argv[3]), int(sys.argv[4]))
    t0 = time.time()
    device = get_device()
    ds, at, tk = CELLS[cellk]
    X, attr, task, n_attr, n_task, _, task_maj = load_cell(ds, at, tk)
    keep, counts = supported_mask(attr, n_attr)

    if method == "FARE":
        fare = json.loads((RESULTS / "fare_gauntlet.json").read_text())
        tag = {c["key"]: ((c.get("tier1") or {}).get("cert") or {}).get("tag")
               for c in fare["cells"]}[cellk]
        base = tag.rsplit("_ts", 1)[0]
        emb = np.load(ARTIFACTS / "fare_cells" / f"{cellk}_fare_embeddings.npz")
        ck, mk = f"{base}_ts{ts}_cells", f"{base}_ts{ts}_medians"
        if ck not in emb:
            print(f"[skip] FARE {cellk} ts{ts}: no stored release", flush=True)
            return
        P = emb[mk][emb[ck]].astype(np.float32)
        L = lr_logits(P, task, ts)
        h_clean, ncov = P, np.zeros((P.shape[1],) * 2)
        op = base
    else:
        X_t = torch.from_numpy(X).to(device)
        ctx = dict(X_t=X_t, attr=attr, task=task, n_attr=n_attr, n_task=n_task,
                   task_maj=task_maj, name=f"{ds}/{at}/{tk}")
        exps = make_exposures(method, knob, ctx, device, ts)
        exp = (next((e for e in exps if e["label"] == VFAE_LABEL), exps[0])
               if method == "VFAE" else exps[0])
        P, L = exp["P"], exp["L"]
        h_clean, ncov = exp["h_clean"], exp["ncov"]
        op = knob
    t_train = time.time() - t0

    res = dict(method=method, cell=cellk, knob=knob, operating_point=op,
               train_seed=ts, class_counts=counts, min_class_rows=MIN_CLASS_ROWS,
               supported_classes=[i for i, v in enumerate(keep) if v],
               probe_seeds=PROBE_SEEDS, archs=list(ARCHS),
               rep=score_surface(P, attr, n_attr, keep, device),
               out=score_surface(L, attr, n_attr, keep, device),
               rep_lrt=lrt_matrix(h_clean, P, attr, ncov, n_attr, keep),
               out_lrt=lrt_matrix(L, L, attr, np.zeros((L.shape[1],) * 2),
                                  n_attr, keep),
               lift=float(lift(L, task, task_maj)), task_maj=float(task_maj),
               refit=method != "FARE",
               train_minutes=round(t_train / 60, 2),
               minutes=round((time.time() - t0) / 60, 2))

    sh = ARTIFACTS / "fleet_shards"
    sh.mkdir(exist_ok=True)
    (sh / f"{method}_{cellk}_k{knob:g}_ts{ts}.json").write_text(json.dumps(res, indent=1))
    print(f"[{method} {cellk} k={knob:g} ts{ts}] train={res['train_minutes']}min "
          f"total={res['minutes']}min "
          f"rep_sup={max(max(v['sup']) for v in res['rep'].values()):.4f} "
          f"out_sup={max(max(v['sup']) for v in res['out'].values()):.4f}", flush=True)


if __name__ == "__main__":
    main()
