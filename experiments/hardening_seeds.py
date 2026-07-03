"""Experiment 15 (Part D of the final block) — bring every headline number to >= 5
training seeds, with mean ± std and honest worst-seed reporting.

Hardened here:
  1. The Experiment-10 utility-kept headline (currently 3 seeds): both e2e arms
     (C blunt, D surgical) of the 2x2, at the published winner configs, retrained
     over 5 seeds with the full Tier-1 battery on representation AND output.
     The e2e clean lift baseline is also recomputed over the same 5 seeds.
  2. The hard-cell edge case (adult/sex/income e2e surgical r=32 sigma=8 lam=100):
     per-seed rep recovery, the exact bar the WORST seed clears, and whether any
     seed breaches 0.55 — reported plainly, not averaged away.
  3. The Experiment-8 HMDA constructive headline (hmda/race/loan_decision noised
     channel sigma=8, currently 3 seeds): clean + noised variants over 5 seeds,
     full battery on P and on the output logits, utility kept.

Any headline number that moves materially vs the stored 3-seed result is flagged
loudly (bar crossings; utility-kept shifts > 5pp of clean lift).

Protocol identical to Exp 8/10 (probe seeds [0,1,2], bar 0.55, never R^2).
Local MPS/CPU; never AWS. macOS OpenMP guard: xgboost before torch.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import experiments.end_to_end_surgical as e2e  # noqa: E402
from experiments.diagnostic import (  # noqa: E402
    PROBE_SEEDS, battery, lift, load_cell,
)
from experiments.sealed_channel import train_model  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
BAR = 0.55
SEEDS5 = [0, 1, 2, 3, 4]
e2e.TRAIN_SEEDS = SEEDS5          # finalize_e2e now runs 5 training seeds

# Exp-10 winner configs (results/end_to_end_surgical.json arms C/D "best").
CELLS = [
    dict(key="easy hmda/race/loan_decision", dataset="hmda", attr="race",
         task="loan_decision",
         C={"sigma": 8.0}, D={"rank": 32, "sigma": 16.0, "lam": 10.0}),
    dict(key="middle hmda/race/loan_amount_band", dataset="hmda", attr="race",
         task="loan_amount_band",
         C={"sigma": 16.0}, D={"rank": 16, "sigma": 16.0, "lam": 100.0}),
    dict(key="hard adult/sex/income", dataset="adult", attr="sex",
         task="income",
         C={"sigma": 32.0}, D={"rank": 32, "sigma": 8.0, "lam": 100.0}),
]
HARD_KEY = "hard adult/sex/income"
PART_A_SIGMA = 8.0                # Exp-8 Part A headline sigma


# --------------------------------------------------------------------------- #
#  1+2 — Exp-10 e2e winners at 5 seeds                                          #
# --------------------------------------------------------------------------- #
def harden_exp10(device):
    stored = json.load(open(RESULTS_DIR / "end_to_end_surgical.json"))
    stored_cells = {c["cell"]: c for c in stored["cells"]}
    out = []
    for cfg in CELLS:
        print("\n" + "#" * 96 + f"\n# EXP-10 @5 seeds: {cfg['key']}\n" + "#" * 96,
              flush=True)
        t0 = time.time()
        X, attr, task, n_attr, n_task, _, task_maj = load_cell(
            cfg["dataset"], cfg["attr"], cfg["task"])
        X_t = torch.from_numpy(X).to(device)

        clean_lifts = []
        for ts in SEEDS5:
            _, _, L0 = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                                   alpha=0.0, device=device, pub_noise=0.0, seed=ts)
            clean_lifts.append(lift(L0, task, task_maj))
        clean = float(np.mean(clean_lifts))
        print(f"[clean e2e] lift={clean:+.4f}±{np.std(clean_lifts):.4f} "
              f"(Exp-10 seed-0 value {stored_cells[cfg['key']]['clean_lift_e2e']:+.4f})",
              flush=True)

        row = {"cell": cfg["key"], "clean_lift_5seed": clean,
               "clean_lift_5seed_std": float(np.std(clean_lifts)),
               "clean_lift_per_seed": [float(x) for x in clean_lifts]}
        for arm, kind in (("C", "e2e_blunt"), ("D", "e2e_surgical")):
            print(f"  [{arm} {kind} {cfg[arm]}] 5-seed retrain", flush=True)
            fin = e2e.finalize_e2e(kind, dict(cfg[arm]), X_t, attr, task, n_attr,
                                   n_task, task_maj, device)
            fin["kept_pct"] = 100.0 * fin["lift_mean"] / max(clean, 1e-9)
            fin["kept_pct_std"] = 100.0 * fin["lift_std"] / max(clean, 1e-9)
            row[arm] = fin
            # movement vs the stored 3-seed headline
            old = stored_cells[cfg["key"]]["arms"][f"{arm}_e2e_blunt"
                                                   if arm == "C" else "D_e2e_surgical"]
            old_fin = old.get("final")
            if old_fin:
                old_kept = 100.0 * old_fin["lift_mean"] / max(
                    stored_cells[cfg["key"]]["clean_lift_e2e"], 1e-9)
                moved = abs(fin["kept_pct"] - old_kept) > 5.0
                bar_flip = (old_fin["rep_max_mean"] <= BAR) != (fin["rep_max_mean"] <= BAR)
                row[f"{arm}_vs_stored"] = {"old_kept_pct": old_kept,
                                           "new_kept_pct": fin["kept_pct"],
                                           "moved_gt_5pp": bool(moved),
                                           "bar_flip": bool(bar_flip)}
                if moved or bar_flip:
                    print(f"  *** FLAG: arm {arm} moved — kept {old_kept:.0f}% (3 seeds) "
                          f"→ {fin['kept_pct']:.0f}% (5 seeds)"
                          + ("; BAR VERDICT FLIPPED" if bar_flip else ""), flush=True)
        row["minutes"] = round((time.time() - t0) / 60, 1)
        out.append(row)
    return out


def hard_cell_worst_seed(rows):
    hard = next(r for r in rows if r["cell"] == HARD_KEY)
    per = hard["D"]["per_seed"]
    rep_max_per_seed = [max(x, m, l) for x, m, l in
                        zip(per["rep_xgb"], per["rep_mlp"], per["rep_lora"])]
    worst = float(max(rep_max_per_seed))
    print("\n" + "=" * 96)
    print(f"HARD-CELL EDGE CASE — {HARD_KEY}, e2e surgical winner, per-seed rep recovery")
    print("=" * 96)
    for i, s in enumerate(SEEDS5):
        print(f"  seed {s}: XGB={per['rep_xgb'][i]:.3f} MLP={per['rep_mlp'][i]:.3f} "
              f"LoRA={per['rep_lora'][i]:.3f}  max={rep_max_per_seed[i]:.3f} "
              f"{'≤0.55' if rep_max_per_seed[i] <= BAR else 'BREACHES 0.55'}  "
              f"lift={per['lift'][i]:+.4f}")
    n_breach = sum(1 for v in rep_max_per_seed if v > BAR)
    print(f"  worst seed max = {worst:.3f} → all 5 seeds hold at bar "
          f"{np.ceil(worst * 100) / 100:.2f}; {n_breach}/5 seeds breach 0.55")
    return {"rep_max_per_seed": rep_max_per_seed, "worst_seed_max": worst,
            "n_seeds_breaching_055": int(n_breach),
            "bar_all_seeds_hold": float(np.ceil(worst * 100) / 100)}


# --------------------------------------------------------------------------- #
#  3 — Exp-8 HMDA Part A at 5 seeds                                             #
# --------------------------------------------------------------------------- #
def harden_exp8(device):
    print("\n" + "#" * 96 + "\n# EXP-8 PART A @5 seeds: hmda/race/loan_decision "
          f"noised channel σ={PART_A_SIGMA:g}\n" + "#" * 96, flush=True)
    X, attr, task, n_attr, n_task, _, task_maj = load_cell(
        "hmda", "race", "loan_decision")
    X_t = torch.from_numpy(X).to(device)
    rows = []
    for vname, s_noise in (("clean σ=0", 0.0), (f"noised σ={PART_A_SIGMA:g}",
                                                PART_A_SIGMA)):
        per = {"P": {"XGB": [], "MLP": [], "LoRA32": []},
               "output": {"XGB": [], "MLP": [], "LoRA32": []}, "lift": []}
        for ts in SEEDS5:
            P, _, L = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                                  alpha=0.0, device=device, pub_noise=s_noise, seed=ts)
            bP = battery(P, attr, n_attr, device, PROBE_SEEDS)
            bO = battery(L, attr, n_attr, device, PROBE_SEEDS)
            for a in ("XGB", "MLP", "LoRA32"):
                per["P"][a].append(bP[a]["auc"])
                per["output"][a].append(bO[a]["auc"])
            per["lift"].append(lift(L, task, task_maj))
            print(f"  [{vname:<12} seed {ts}] P max={bP['max_auc']:.3f} "
                  f"out max={bO['max_auc']:.3f} lift={per['lift'][-1]:+.4f}", flush=True)
        row = {"variant": vname, "sigma": s_noise,
               "lift_mean": float(np.mean(per["lift"])),
               "lift_std": float(np.std(per["lift"])),
               "lift_per_seed": [float(x) for x in per["lift"]]}
        for tgt in ("P", "output"):
            for a in ("XGB", "MLP", "LoRA32"):
                row[f"{tgt}_{a}_mean"] = float(np.mean(per[tgt][a]))
                row[f"{tgt}_{a}_std"] = float(np.std(per[tgt][a]))
                row[f"{tgt}_{a}_per_seed"] = [float(x) for x in per[tgt][a]]
            row[f"{tgt}_max_mean"] = float(max(
                np.mean(per[tgt][a]) for a in ("XGB", "MLP", "LoRA32")))
        rows.append(row)
    clean, noised = rows
    kept = 100.0 * noised["lift_mean"] / max(clean["lift_mean"], 1e-9)
    print(f"\n>>> EXP-8 hardened (5 seeds): P_max={noised['P_max_mean']:.3f} "
          f"out_max={noised['output_max_mean']:.3f} lift {noised['lift_mean']:+.4f}"
          f"±{noised['lift_std']:.4f} = {kept:.0f}% of clean "
          f"{clean['lift_mean']:+.4f}±{clean['lift_std']:.4f} "
          f"(3-seed headline: ~0.53 both channels, 96%)", flush=True)
    return {"rows": rows, "kept_pct": kept}


def main():
    device = get_device()
    print(f"[setup] device={device} — hardening to 5 training seeds "
          f"(probe seeds {PROBE_SEEDS}, bar {BAR})", flush=True)
    exp10 = harden_exp10(device)
    hard_edge = hard_cell_worst_seed(exp10)
    exp8 = harden_exp8(device)

    print("\n" + "=" * 100)
    print("HARDENED HEADLINE TABLE — utility kept at full-battery ≤0.55 "
          "(5 training seeds, mean±std)")
    print("=" * 100)
    print(f"{'cell':<40} {'e2e blunt (C)':>18} {'e2e surgical (D)':>18} "
          f"{'D rep_max':>10} {'D out_max':>10}")
    print("-" * 100)
    for r in exp10:
        print(f"{r['cell']:<40} "
              f"{r['C']['kept_pct']:>13.0f}%±{r['C']['kept_pct_std']:<3.0f} "
              f"{r['D']['kept_pct']:>13.0f}%±{r['D']['kept_pct_std']:<3.0f} "
              f"{r['D']['rep_max_mean']:>10.3f} {r['D']['out_max_mean']:>10.3f}")
    print("-" * 100)

    (RESULTS_DIR / "hardening_seeds.json").write_text(json.dumps(
        {"experiment": "hardening_seeds", "seeds": SEEDS5, "bar": BAR,
         "exp10_5seed": exp10, "hard_cell_edge": hard_edge, "exp8_5seed": exp8},
        indent=2, default=float))
    print(f"\n[save] {RESULTS_DIR / 'hardening_seeds.json'}", flush=True)


if __name__ == "__main__":
    main()
