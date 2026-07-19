"""Expansion Phase 0 — label-coupling scan + rule-based cell auto-selection
over the four additional public sources (folktables/ACS CA-2018, LSAC Law
School, Dutch Census 2001, Diabetes 130-US Hospitals via fairlearn).

Same predictor construction as diagnostic.py (XGB AUC recovering the
attribute from the 1-dim task label alone; 75/25 stratified held-out, 3
seeds). Reports per cell: n (train partition), group sizes, positive-label
rate per group, coupling AUC — then applies the fixed auto-selection rules:

  R1  folktables cells capped at 4, preferring coverage of distinct
      0.52-0.60 bands over duplicates in the same band;
  R2  every non-folktables cell with train n >= 10,000 is included;
      cells below 10,000 are excluded;
  R3  any cell with coupling AUC > 0.90 is excluded as degenerate
      near-saturation (noted);
  R4  target 6-10 selected cells total.

Writes results/expansion_coupling_scan.json (new file; touches nothing
else). macOS OpenMP guard: xgboost before torch.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: F401,E402  (import before torch on purpose)

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.folktables_coupling_scan import BAND, band_flag, label_predictor  # noqa: E402
from utils.expansion_io import SOURCES, cell_provenance, load_expansion_cell  # noqa: E402
from utils.folktables_io import ATTR_NAMES as FT_ATTRS  # noqa: E402
from utils.folktables_io import TASK_NAMES as FT_TASKS  # noqa: E402
from utils.folktables_io import load_folktables_cell  # noqa: E402

RESULTS_PATH = REPO_ROOT / "results" / "expansion_coupling_scan.json"
FT_CAP = 4          # R1
MIN_N = 10_000      # R2
DEGEN_AUC = 0.90    # R3
TARGET = (6, 10)    # R4


def candidates():
    for t in FT_TASKS:
        for a in FT_ATTRS:
            yield "folktables", a, t
    for src, cfg in SOURCES.items():
        for t in cfg["tasks"]:
            for a in cfg["attrs"]:
                yield src, a, t


def scan_cell(source, attr_name, task_name):
    t0 = time.time()
    if source == "folktables":
        X, attr, task, n_attr, n_task, attr_maj, task_maj = \
            load_folktables_cell(task_name, attr_name)
    else:
        X, attr, task, n_attr, n_task, attr_maj, task_maj = \
            load_expansion_cell(source, attr_name, task_name)
    auc, std = label_predictor(attr, task, n_attr)
    groups = {}
    for g in (0, 1):
        m = attr == g
        groups[str(g)] = {"n": int(m.sum()),
                          "pos_rate": float(task[m].mean())}
    return {"cell": f"{source}/{attr_name}/{task_name}", "source": source,
            "attr": attr_name, "task": task_name, "n": int(len(attr)),
            "x_dim": int(X.shape[1]), "attr_majority": attr_maj,
            "task_majority": task_maj, "groups": groups,
            "predictor_auc": auc, "predictor_std": std,
            "band": band_flag(auc), "seconds": round(time.time() - t0, 1)}


def select(rows):
    """Apply R1-R4; return (selected rows, per-cell admitting rule, notes)."""
    notes = []
    eligible, why = [], {}
    degen = [r for r in rows if r["predictor_auc"] > DEGEN_AUC]
    for r in degen:
        notes.append(f"R3 EXCLUDED {r['cell']}: coupling AUC "
                     f"{r['predictor_auc']:.3f} > {DEGEN_AUC} (degenerate near-saturation)")
    pool = [r for r in rows if r not in degen]

    # R1 — folktables cap with band-coverage preference
    ft = [r for r in pool if r["source"] == "folktables"]
    by_band = {"below": sorted([r for r in ft if r["band"] == "below"],
                               key=lambda r: r["predictor_auc"]),
               "inside": sorted([r for r in ft if r["band"] == "inside"],
                                key=lambda r: -r["predictor_auc"]),
               "above": sorted([r for r in ft if r["band"] == "above"],
                               key=lambda r: -r["predictor_auc"])}
    ft_pick = []
    while len(ft_pick) < FT_CAP and any(by_band.values()):
        for b in ("below", "inside", "above"):
            if by_band[b] and len(ft_pick) < FT_CAP:
                ft_pick.append(by_band[b].pop(0))
    for r in ft_pick:
        eligible.append(r)
        why[r["cell"]] = (f"R1 folktables (cap {FT_CAP}, band-coverage round-robin; "
                          f"band={r['band']})")
    dropped_ft = [r for r in ft if r not in ft_pick]
    for r in dropped_ft:
        notes.append(f"R1 dropped {r['cell']} (folktables cap; duplicate of band "
                     f"'{r['band']}')")

    # R2 — non-folktables inclusion by n
    for r in pool:
        if r["source"] == "folktables":
            continue
        if r["n"] >= MIN_N:
            eligible.append(r)
            why[r["cell"]] = f"R2 non-folktables, n={r['n']:,} >= {MIN_N:,}"
        else:
            notes.append(f"R2 EXCLUDED {r['cell']}: n={r['n']:,} < {MIN_N:,}")

    # R4 — target count
    if len(eligible) < TARGET[0]:
        notes.append(f"R4: only {len(eligible)} eligible cells (< {TARGET[0]}) — "
                     "running all eligible")
    if len(eligible) > TARGET[1]:
        notes.append(f"R4: {len(eligible)} > {TARGET[1]} — trimming would be needed "
                     "(not expected with current sources)")
    spread = {"below_052": sum(r["predictor_auc"] < BAND[0] for r in eligible),
              "inside_band": sum(r["band"] == "inside" for r in eligible),
              "above_060": sum(r["predictor_auc"] > BAND[1] for r in eligible)}
    return eligible, why, notes, spread


def main():
    print(f"[setup] expansion coupling scan | predictor = XGB AUC(attr | task label "
          f"alone), 3 seeds, 75/25 stratified | band {BAND[0]}-{BAND[1]}")
    rows = []
    hdr = (f"{'cell':<42} {'n':>8} {'grp0(n,pos%)':>16} {'grp1(n,pos%)':>16} "
           f"{'AUC(attr|label)':>16} {'band':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for source, attr_name, task_name in candidates():
        try:
            r = scan_cell(source, attr_name, task_name)
        except Exception as e:  # noqa: BLE001 — a failed source is skipped, noted
            print(f"{source}/{attr_name}/{task_name:<20} FAILED: {e}")
            rows.append({"cell": f"{source}/{attr_name}/{task_name}",
                         "source": source, "failed": str(e)})
            continue
        rows.append(r)
        g0, g1 = r["groups"]["0"], r["groups"]["1"]
        print(f"{r['cell']:<42} {r['n']:>8} "
              f"{g0['n']:>9},{100*g0['pos_rate']:>5.1f}% "
              f"{g1['n']:>9},{100*g1['pos_rate']:>5.1f}% "
              f"{r['predictor_auc']:>10.4f} ±{r['predictor_std']:.3f} "
              f"{r['band']:>7}   [{r['seconds']:.1f}s]")
    print("-" * len(hdr))

    ok = [r for r in rows if "failed" not in r]
    failed = [r for r in rows if "failed" in r]
    selected, why, notes, spread = select(ok)
    selected.sort(key=lambda r: r["predictor_auc"])

    print(f"\nSELECTED {len(selected)} cells (target {TARGET[0]}-{TARGET[1]}):")
    for r in selected:
        print(f"  {r['predictor_auc']:.4f}  {r['band']:<7} {r['cell']:<42} "
              f"<- {why[r['cell']]}")
    print(f"\nspread: {spread['below_052']} below {BAND[0]}, "
          f"{spread['inside_band']} inside band, {spread['above_060']} above {BAND[1]}")
    for n in notes:
        print(f"  note: {n}")
    for r in failed:
        print(f"  FAILED source: {r['cell']}: {r['failed']}")

    out = {"experiment": "expansion_coupling_scan", "phase": 0,
           "predictor": "XGB AUC(attr | task label alone), 75/25 stratified, 3 seeds",
           "band": list(BAND),
           "rules": {"R1": f"folktables cap {FT_CAP}, band-coverage preference",
                     "R2": f"non-folktables included iff n >= {MIN_N}",
                     "R3": f"excluded iff coupling AUC > {DEGEN_AUC}",
                     "R4": f"target {TARGET[0]}-{TARGET[1]} cells"},
           "provenance": {s: cell_provenance(s) for s in
                          ["folktables"] + sorted(SOURCES)},
           "cells": rows, "failed_sources": [r["cell"] for r in failed],
           "selected": [r["cell"] for r in selected],
           "selection_rule": why, "selection_notes": notes, "spread": spread}
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {RESULTS_PATH}")


if __name__ == "__main__":
    main()
