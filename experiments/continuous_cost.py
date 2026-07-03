"""Experiment 9 — GOAL 1: fix the circular diagnostic by predicting a CONTINUOUS
cost, not a pass/fail at the same 0.55 bar that defines "success".

The Exp-8 diagnostic predicted a binary outcome ("durable removal possible?")
whose definition shared the 0.55 threshold with the predictor's claimed decision
boundary — partly circular. Here the OUTCOME is a continuous utility cost with no
0.55-defined success in it:

  PREDICTOR (x): XGB AUC recovering the attribute from the 1-dim task LABEL alone
    (one-feature fit; no representation, no trained model).
  OUTCOME (y, continuous): fraction of clean task lift LOST at the noise level
    where the strong-attacker attribute AUC first drops to <= 0.55, linearly
    interpolated between swept sigma points, clipped to [0, 1]. Reported for
      * cost_rep     — hide the representation P only;
      * cost_durable — hide P AND the task output (the durable-removal cost).
    The 0.55 crossing only *locates* the noise level; the outcome value itself is
    a utility fraction on a 0-1 scale, so a predictor/outcome correlation is not
    threshold-against-itself.

Mechanism: the same from-raw-x noised channel as Exp 8 (sealed_channel.train_model,
E_pub -> BatchNorm -> +N(0,sigma^2) -> head, fresh noise per forward), trained per
sigma. Attribute recovery is measured with the strong battery (XGBoost + deep MLP;
the crossing uses max(XGB, MLP) — stricter than the XGB-only literal definition,
which is also recorded). NEVER linear R^2. The sigma grid extends adaptively (up to
192) until both channels are hidden, so every cell gets a finite crossing or is
flagged censored (cost = 1.0).

Sharded by dataset (--shard adult|hmda|diabetes) so the three shards run as
parallel processes; --merge combines the shard JSONs, computes the predictor vs
cost correlation (Pearson + Spearman, degenerate cells excluded) and renders the
plot. Local MPS/CPU; never AWS. macOS OpenMP guard: xgboost before torch.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401  (import before torch on purpose)

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.diagnostic import (  # noqa: E402
    HONEST_AUC_BAR, SWEEP_PROBE_SEEDS, battery, label_predictor, lift, load_cell,
)
from experiments.sealed_channel import train_model  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
BAR = HONEST_AUC_BAR                       # 0.55 — locates the crossing only
SIGMA_GRID = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0]
DEGENERATE_LIFT = 0.01                     # clean lift below this -> cost undefined

# 20 cells spanning the predictor axis (scouted live over all 30 attr x task
# pairs; readmission cells dropped as known-degenerate). Predictor re-measured.
CELLS = [
    ("hmda", "sex", "loan_decision"),                      # ~0.503
    ("diabetes", "gender", "medication_change_outcome"),   # ~0.509
    ("hmda", "race", "loan_decision"),                     # ~0.514
    ("hmda", "race", "tract_denial_high"),                 # ~0.518
    ("diabetes", "age_bucket", "medication_change_outcome"),  # ~0.522
    ("adult", "sex", "education_level"),                   # ~0.539
    ("diabetes", "gender", "primary_diagnosis_category"),  # ~0.543
    ("diabetes", "race", "primary_diagnosis_category"),    # ~0.544
    ("adult", "marital_status", "education_level"),        # ~0.555
    ("adult", "race", "income"),                           # ~0.556
    ("hmda", "sex", "loan_amount_band"),                   # ~0.563
    ("adult", "age_group", "income"),                      # ~0.568
    ("hmda", "race", "loan_amount_band"),                  # ~0.584
    ("adult", "race", "education_level"),                  # ~0.589
    ("adult", "sex", "income"),                            # ~0.603
    ("adult", "marital_status", "occupation_group"),       # ~0.607
    ("hmda", "ethnicity", "loan_amount_band"),             # ~0.624
    ("diabetes", "age_bucket", "primary_diagnosis_category"),  # ~0.665
    ("adult", "sex", "occupation_group"),                  # ~0.676
    ("adult", "marital_status", "income"),                 # ~0.688
]


def crossing_cost(sweep, key, clean_lift):
    """Continuous cost at the first sigma where sweep[i][key] <= BAR.

    Linear interpolation of (AUC, lift) between the bracketing sigma points; cost
    = 1 - lift*/clean_lift, clipped to [0,1]. Returns (cost, sigma*, lift*,
    censored). If the grid is exhausted without a crossing -> cost 1.0, censored.
    If already <= BAR at sigma=0 -> cost 0.0.
    """
    if clean_lift <= DEGENERATE_LIFT:
        return None, None, None, False
    if sweep[0][key] <= BAR:
        return 0.0, 0.0, clean_lift, False
    for i in range(1, len(sweep)):
        a0, a1 = sweep[i - 1][key], sweep[i][key]
        if a1 <= BAR:
            frac = (a0 - BAR) / max(a0 - a1, 1e-9)
            s = sweep[i - 1]["sigma"] + frac * (sweep[i]["sigma"] - sweep[i - 1]["sigma"])
            lf = sweep[i - 1]["lift"] + frac * (sweep[i]["lift"] - sweep[i - 1]["lift"])
            return float(np.clip(1.0 - lf / clean_lift, 0.0, 1.0)), float(s), float(lf), False
    return 1.0, None, None, True


def measure_cell(dataset, attr_name, task_name, device, X=None, attr=None, task=None,
                 tag=None):
    """Predictor + adaptive noise sweep -> continuous removal costs for one cell.

    Pass (X, attr, task) to override the natural data (constructed cells, Goal 2)."""
    cell = tag or f"{dataset}/{attr_name}/{task_name}"
    print("\n" + "#" * 92 + f"\n# CELL: {cell}\n" + "#" * 92, flush=True)
    if X is None:
        X, attr, task, n_attr, n_task, attr_maj, task_maj = load_cell(dataset, attr_name, task_name)
    else:
        n_attr = int(attr.max()) + 1
        n_task = int(task.max()) + 1
        attr_maj = float(np.bincount(attr).max() / len(attr))
        task_maj = float(np.bincount(task).max() / len(task))
    X_t = torch.from_numpy(X).to(device)
    pred_auc, pred_std = label_predictor(attr, task, n_attr)
    print(f"[setup] N={X.shape[0]} {attr_name}={n_attr}cls task '{task_name}' "
          f"maj={task_maj:.3f} | PREDICTOR={pred_auc:.4f}±{pred_std:.3f}", flush=True)

    sweep = []
    for s_noise in SIGMA_GRID:
        P, _, L = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                              alpha=0.0, device=device, pub_noise=s_noise, seed=0)
        bP = battery(P, attr, n_attr, device, SWEEP_PROBE_SEEDS, archs=("XGB", "MLP"))
        bO = battery(L, attr, n_attr, device, SWEEP_PROBE_SEEDS, archs=("XGB", "MLP"))
        tl = lift(L, task, task_maj)
        pt = {"sigma": s_noise, "P_max": bP["max_auc"], "out_max": bO["max_auc"],
              "both_max": max(bP["max_auc"], bO["max_auc"]),
              "P_xgb": bP["XGB"]["auc"], "out_xgb": bO["XGB"]["auc"],
              "P_mlp": bP["MLP"]["auc"], "out_mlp": bO["MLP"]["auc"], "lift": tl}
        sweep.append(pt)
        print(f"  σ={s_noise:>6.1f}  P_max={pt['P_max']:.3f}  out_max={pt['out_max']:.3f}  "
              f"lift={tl:+.4f}", flush=True)
        if pt["P_max"] <= BAR and pt["out_max"] <= BAR:
            break

    clean_lift = sweep[0]["lift"]
    cost_rep, s_rep, l_rep, cens_rep = crossing_cost(sweep, "P_max", clean_lift)
    cost_dur, s_dur, l_dur, cens_dur = crossing_cost(sweep, "both_max", clean_lift)
    # The literal Goal-1 definition (XGB-only crossing on the representation).
    cost_xgb, _, _, _ = crossing_cost(sweep, "P_xgb", clean_lift)

    row = {"cell": cell, "dataset": dataset, "attr": attr_name, "task": task_name,
           "predictor": pred_auc, "predictor_std": pred_std,
           "attr_majority": attr_maj, "task_majority": task_maj,
           "clean_lift": clean_lift, "degenerate": clean_lift <= DEGENERATE_LIFT,
           "cost_rep": cost_rep, "sigma_rep": s_rep, "lift_at_rep": l_rep,
           "censored_rep": cens_rep,
           "cost_durable": cost_dur, "sigma_durable": s_dur, "lift_at_durable": l_dur,
           "censored_durable": cens_dur,
           "cost_rep_xgb_only": cost_xgb, "sweep": sweep}
    cr = "None" if cost_rep is None else f"{cost_rep:.3f}"
    cd = "None" if cost_dur is None else f"{cost_dur:.3f}"
    print(f"  → predictor={pred_auc:.3f} clean_lift={clean_lift:+.4f} "
          f"cost_rep={cr}{' (censored)' if cens_rep else ''} "
          f"cost_durable={cd}{' (censored)' if cens_dur else ''}"
          + ("  [DEGENERATE]" if row["degenerate"] else ""), flush=True)
    return row


# --------------------------------------------------------------------------- #
#  Merge: correlation + plot                                                   #
# --------------------------------------------------------------------------- #
def merge():
    rows = []
    for shard in ("adult", "hmda", "diabetes"):
        p = RESULTS_DIR / f"continuous_cost_{shard}.json"
        rows.extend(json.loads(p.read_text())["rows"])
    rows.sort(key=lambda r: r["predictor"])
    good = [r for r in rows if not r["degenerate"]]

    from scipy.stats import pearsonr, spearmanr
    stats = {}
    for key in ("cost_rep", "cost_durable"):
        x = np.array([r["predictor"] for r in good])
        y = np.array([r[key] for r in good])
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        stats[key] = {"pearson_r": float(pr), "pearson_p": float(pp),
                      "spearman_r": float(sr), "spearman_p": float(sp), "n": len(good)}

    print("\n" + "=" * 108)
    print("GOAL 1 — PREDICTOR vs CONTINUOUS DURABLE-REMOVAL COST (all cells, sorted by predictor)")
    print("=" * 108)
    print(f"{'cell':<46} {'pred':>6} {'clean':>8} {'cost_rep':>9} {'σ_rep':>6} "
          f"{'cost_durable':>13} {'σ_dur':>6}")
    print("-" * 108)
    for r in rows:
        cr = "  —  " if r["cost_rep"] is None else f"{r['cost_rep']:.3f}" + ("c" if r["censored_rep"] else " ")
        cd = "  —  " if r["cost_durable"] is None else f"{r['cost_durable']:.3f}" + ("c" if r["censored_durable"] else " ")
        sr = "—" if r["sigma_rep"] is None else f"{r['sigma_rep']:.1f}"
        sd = "—" if r["sigma_durable"] is None else f"{r['sigma_durable']:.1f}"
        print(f"{r['cell']:<46} {r['predictor']:>6.3f} {r['clean_lift']:>+8.4f} {cr:>9} {sr:>6} "
              f"{cd:>13} {sd:>6}" + ("  DEGEN" if r["degenerate"] else ""))
    print("-" * 108)
    for key, s in stats.items():
        print(f"{key}: Pearson r={s['pearson_r']:+.3f} (p={s['pearson_p']:.1e})  "
              f"Spearman ρ={s['spearman_r']:+.3f} (p={s['spearman_p']:.1e})  n={s['n']}")
    print("('c' = censored: no crossing within σ≤192, cost set to 1.0. Crossings use "
          "max(XGB,MLP); XGB-only costs in JSON.)")

    out = {"experiment": "continuous_cost", "bar": BAR, "sigma_grid": SIGMA_GRID,
           "stats": stats, "rows": rows}
    (RESULTS_DIR / "continuous_cost.json").write_text(json.dumps(out, indent=2))
    _plot(rows, stats, RESULTS_DIR / "continuous_cost.png")
    print(f"\n[save] {RESULTS_DIR/'continuous_cost.json'} and .png")


def _plot(rows, stats, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    for ax, key, title in [
            (axes[0], "cost_rep", "hide the REPRESENTATION"),
            (axes[1], "cost_durable", "hide rep AND output (durable removal)")]:
        for r in rows:
            if r[key] is None:
                continue
            cens = r["censored_rep" if key == "cost_rep" else "censored_durable"]
            ax.scatter(r["predictor"], r[key], s=80, zorder=3,
                       marker="^" if cens else "o",
                       color={"adult": "C0", "hmda": "C1", "diabetes": "C2"}[r["dataset"]])
            ax.annotate(r["cell"].replace("/", "\n"), (r["predictor"], r[key]),
                        fontsize=5, xytext=(4, 3), textcoords="offset points")
        s = stats[key]
        ax.set_title(f"Goal 1 — predictor vs continuous cost to {title}\n"
                     f"Pearson r={s['pearson_r']:+.3f} (p={s['pearson_p']:.1e}), "
                     f"Spearman ρ={s['spearman_r']:+.3f} (n={s['n']})")
        ax.set_xlabel("PREDICTOR: XGB AUC(attribute | task label alone)")
        ax.set_ylabel("cost: fraction of clean task lift lost at the ≤0.55 crossing")
        ax.set_ylim(-0.05, 1.1)
        ax.axvline(BAR, color="green", ls="--", lw=1, alpha=0.7, label="0.55 (old binary bar)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=d)
               for d, c in [("adult", "C0"), ("hmda", "C1"), ("diabetes", "C2")]]
    handles.append(plt.Line2D([], [], marker="^", ls="", color="gray", label="censored (σ>192)"))
    axes[0].legend(handles=handles + axes[0].get_legend_handles_labels()[0], fontsize=8,
                   loc="upper left")
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


def main():
    if "--merge" in sys.argv:
        merge()
        return
    shard = sys.argv[sys.argv.index("--shard") + 1] if "--shard" in sys.argv else None
    device = get_device()
    cells = [c for c in CELLS if shard is None or c[0] == shard]
    print(f"[setup] device={device} shard={shard} cells={len(cells)} bar≤{BAR} grid={SIGMA_GRID}",
          flush=True)
    rows = [measure_cell(d, a, t, device) for d, a, t in cells]
    out_path = RESULTS_DIR / f"continuous_cost_{shard or 'all'}.json"
    out_path.write_text(json.dumps({"experiment": "continuous_cost", "shard": shard,
                                    "bar": BAR, "rows": rows}, indent=2))
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
