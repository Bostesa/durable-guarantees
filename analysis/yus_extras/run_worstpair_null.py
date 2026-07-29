"""Null calibration for the worst-pair max-over-seeds statistic.

Unregistered addition, prompted by the sweep itself: on the easy cell the
representation-side worst-pair reading is NON-MONOTONIC in sigma (0.5793 at
sigma=16, 0.5963 at sigma=512, with negative utility) while macro-OVR falls
monotonically to 0.51. Attribute recovery cannot increase with noise, so the
statistic is measuring something other than leakage. HMDA race is severely
imbalanced — class 4 is 505 of 63,747 rows (0.79%), 126 in a 25% test split —
so the rarest pair's AUC has a large null standard error, and worst-pair takes
a MAX over 10 pairs and 9 seeds of an orientation-folded quantity.

This measures the null directly: features carrying ZERO information about the
attribute (isotropic Gaussian, same shape as a released representation), the
real HMDA race label distribution, and the identical scoring pipeline
(XGB + MLP + LoRA, 3 "training" draws x 3 probe seeds, worst-pair max over all).
Whatever this reads is the floor below which the statistic cannot distinguish a
protected channel from a random one.
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
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from experiments.diagnostic import load_cell  # noqa: E402
from run_worstpair_sweep import (  # noqa: E402
    BAR, PROBE_SEEDS, TRAIN_SEEDS, lora_probs, macro_ovr, probe_probs, worst_pair,
)
from utils.pcrl_io import get_device  # noqa: E402

D = 64
SEED = 20260729


def main():
    t0 = time.time()
    device = get_device()
    rng = np.random.default_rng(SEED)
    X, attr, task, n_attr, _, _, _ = load_cell("hmda", "race", "loan_decision")
    counts = np.bincount(attr, minlength=n_attr)
    print(f"[setup] device={device} n={len(attr)} classes={n_attr} "
          f"counts={counts.tolist()}")
    print(f"        features are pure noise: zero information about the "
          f"attribute by construction\n", flush=True)

    wp_all, mo_all, per = [], [], []
    for draw in TRAIN_SEEDS:
        H = rng.standard_normal((len(attr), D)).astype(np.float32)
        for s in PROBE_SEEDS:
            for arch in ("XGB", "MLP"):
                yte, p = probe_probs(arch, H.astype(np.float64), attr, n_attr, s)
                w, pair = worst_pair(yte, p)
                wp_all.append(w); mo_all.append(macro_ovr(yte, p))
                per.append(dict(draw=draw, probe_seed=s, attacker=arch,
                                worst_pair=w, pair=list(pair) if pair else None,
                                macro=mo_all[-1]))
            yte, p = lora_probs(H, attr, n_attr, s, device)
            w, pair = worst_pair(yte, p)
            wp_all.append(w); mo_all.append(macro_ovr(yte, p))
            per.append(dict(draw=draw, probe_seed=s, attacker="LoRA",
                            worst_pair=w, pair=list(pair) if pair else None,
                            macro=mo_all[-1]))
        print(f"  draw {draw}: worst-pair so far max={max(wp_all):.4f} "
              f"macro max={max(mo_all):.4f}", flush=True)

    wp = np.array(wp_all); mo = np.array(mo_all)
    from collections import Counter
    pairs = Counter(tuple(r["pair"]) for r in per if r["pair"])
    out = dict(experiment="worstpair_null_calibration",
               note=("features carry zero attribute information by construction; "
                     "any reading above 0.5 is sampling noise"),
               n_rows=len(attr), n_classes=int(n_attr),
               class_counts=counts.tolist(),
               test_split_counts=[int(0.25 * c) for c in counts],
               n_measurements=len(wp),
               worstpair_max=float(wp.max()), worstpair_mean=float(wp.mean()),
               worstpair_p95=float(np.percentile(wp, 95)),
               macro_max=float(mo.max()), macro_mean=float(mo.mean()),
               exceeds_bar=bool(wp.max() > BAR),
               argmax_pair_counts={str(k): v for k, v in pairs.items()},
               per_measurement=per,
               minutes=round((time.time() - t0) / 60, 1))
    print("\n" + "=" * 78)
    print("NULL CALIBRATION — zero-information features, real HMDA race labels")
    print("=" * 78)
    print(f"  measurements: {len(wp)} (3 draws x 3 probe seeds x 3 attackers)")
    print(f"  worst-pair : max {wp.max():.4f}   p95 {np.percentile(wp, 95):.4f}   "
          f"mean {wp.mean():.4f}")
    print(f"  macro-OVR  : max {mo.max():.4f}   mean {mo.mean():.4f}")
    print(f"  worst-pair max exceeds the {BAR} bar: "
          f"{'YES' if wp.max() > BAR else 'no'}")
    print(f"  pair achieving the max, by frequency: {dict(pairs)}")
    (HERE / "worstpair_null.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[save] {HERE / 'worstpair_null.json'} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
