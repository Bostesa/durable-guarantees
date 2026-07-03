"""Experiment 12 (scan only) — CelebA coupling scan: pick the two vision cells.

The paper needs one LOW-coupling and one HIGH-coupling CelebA task/attribute pair
as the non-tabular modality check. Per the project's standing methodology (Exp 8/9),
the cells are picked by MEASUREMENT, not intuition: the PREDICTOR is the XGBoost AUC
recovering the protected attribute from the task LABEL alone (one 1-feature fit —
no model, no images, no representations). This scan therefore needs only the CelebA
attribute labels, loaded from the same files PCRL's CelebA infrastructure
(`pcrl/data/celeba.py`) reads: `list_attr_celeba.csv` + `list_eval_partition.csv`,
train partition, -1/1 → 0/1, identically to `CelebADataset._load_data`.

For a single binary task label f predicting a binary attribute a, the population
AUC has a closed form: AUC = 1/2 + |P(f=1|a=1) − P(f=1|a=0)| / 2 (a 1-feature
XGBoost can only reproduce this ordering). We use the closed form to rank ALL
(task, protected) pairs instantly, then verify the shortlist with the project's
actual XGB predictor protocol (probe-style held-out fits, 3 split seeds) so the
reported numbers are produced by the same instrument as Exp 8/9.

Outputs: ranked table, class-balance sanity checks (skip attrs <5% or >95%
positive), near-duplicate flags (gender-coded near-synonyms of Male etc., plus a
|phi|>0.45 statistical flag), and the two recommended cells.

NO removal pipeline is run here. The full CelebA run will additionally need:
encoder forward passes (PCRL's vision encoder, 64x64 images) to get representations,
then the standard channel + battery pipeline.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import xgboost  # noqa: E402,F401  (OpenMP guard: xgboost before torch, project rule)

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
PCRL_ROOT = Path(os.environ.get("PCRL_ROOT", "/Users/nathansamson/PCRL"))
CELEBA_ROOT = PCRL_ROOT / "data" / "celeba"

PROTECTED = ["Male", "Young", "Pale_Skin"]
BALANCE_LO, BALANCE_HI = 0.05, 0.95   # skip attrs with <5% positive (either side)
PHI_DUP_FLAG = 0.45                   # |phi| above this → near-duplicate flag
SHORTLIST_XGB = 24                    # verify this many top/spread pairs with real XGB

# Always XGB-verify these (recommendation candidates near the low-coupling end,
# which a top-heavy shortlist would otherwise miss).
ALWAYS_VERIFY = [("Smiling", "Young"), ("High_Cheekbones", "Young"),
                 ("Mouth_Slightly_Open", "Young"), ("Smiling", "Male")]

# Gender-coded near-synonyms / direct grooming proxies of Male — a pair like
# (Wearing_Lipstick, Male) is the attribute itself in disguise, not a task.
MALE_SYNONYMS = {
    "Wearing_Lipstick", "Heavy_Makeup", "No_Beard", "Goatee", "Mustache",
    "Sideburns", "5_o_Clock_Shadow", "Wearing_Necklace", "Wearing_Earrings",
    "Rosy_Cheeks", "Bald",
}


def load_labels():
    """Load CelebA train-partition attribute labels exactly as PCRL's loader does."""
    attr_df = pd.read_csv(CELEBA_ROOT / "list_attr_celeba.csv")
    part_df = pd.read_csv(CELEBA_ROOT / "list_eval_partition.csv")
    df = attr_df.merge(part_df, on="image_id")
    df = df[df["partition"] == 0].reset_index(drop=True)   # train split
    attrs = [c for c in attr_df.columns if c != "image_id"]
    for c in attrs:
        df[c] = ((df[c] + 1) // 2).astype(np.int64)        # -1/1 → 0/1 (PCRL convention)
    return df, attrs


def closed_form_auc(task, attr):
    """Population AUC of the 1-binary-feature predictor: 1/2 + |q1 − q0|/2."""
    q1 = task[attr == 1].mean()
    q0 = task[attr == 0].mean()
    return 0.5 + abs(q1 - q0) / 2.0


def phi(a, b):
    """Phi (Pearson) correlation between two binary vectors."""
    return float(np.corrcoef(a, b)[0, 1])


def xgb_predictor(task, attr, seeds=(0, 1, 2)):
    """The project's actual predictor instrument (diagnostic.label_predictor):
    held-out XGB AUC recovering attr from the 1-dim task label, 3 split seeds."""
    from experiments.hardening_test import _probe
    from sklearn.model_selection import train_test_split
    aucs = []
    X = task.reshape(-1, 1).astype(np.float32)
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X.astype(np.float64), attr,
                                              test_size=0.25, random_state=s,
                                              stratify=attr)
        _, auc = _probe("XGB", Xtr, ytr, Xte, yte, s, 2)
        aucs.append(auc)
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    df, attrs = load_labels()
    n = len(df)
    print(f"[setup] CelebA train partition: {n} rows, {len(attrs)} attributes "
          f"(from {CELEBA_ROOT})")

    pos_rate = {c: float(df[c].mean()) for c in attrs}
    balanced = {c for c in attrs if BALANCE_LO <= pos_rate[c] <= BALANCE_HI}
    skipped = sorted(set(attrs) - balanced)
    print(f"[balance] skipped (<{BALANCE_LO:.0%} or >{BALANCE_HI:.0%} positive): "
          + ", ".join(f"{c} ({pos_rate[c]:.3f})" for c in skipped))

    # ---- full closed-form scan: every balanced task x protected pair ----
    rows = []
    for prot in PROTECTED:
        if prot not in balanced:
            print(f"[balance] protected attribute {prot} is itself unbalanced "
                  f"({pos_rate[prot]:.3f} positive) — scanned but flagged UNBALANCED")
        a = df[prot].to_numpy()
        for task_name in attrs:
            if task_name == prot:
                continue
            t = df[task_name].to_numpy()
            cf = closed_form_auc(t, a)
            ph = phi(t, a)
            dup = (prot == "Male" and task_name in MALE_SYNONYMS) or abs(ph) > PHI_DUP_FLAG
            rows.append({
                "task": task_name, "protected": prot,
                "closed_form_auc": round(cf, 4), "phi": round(ph, 4),
                "task_pos_rate": round(pos_rate[task_name], 4),
                "prot_pos_rate": round(pos_rate[prot], 4),
                "task_balanced": task_name in balanced,
                "prot_balanced": prot in balanced,
                "near_duplicate_flag": bool(dup),
            })
    rows.sort(key=lambda r: -r["closed_form_auc"])

    # ---- XGB verification on a shortlist: top pairs + a spread down to ~0.5 ----
    eligible = [r for r in rows if r["task_balanced"] and r["prot_balanced"]]
    # take the top half of the shortlist from the top of the ranking, the rest
    # spread evenly across the remaining range so the low-coupling end is verified too
    top = eligible[: SHORTLIST_XGB // 2]
    rest = eligible[SHORTLIST_XGB // 2:]
    stride = max(1, len(rest) // (SHORTLIST_XGB - len(top)))
    spread = rest[::stride][: SHORTLIST_XGB - len(top)]
    shortlist = top + spread
    keyed = {(r["task"], r["protected"]): r for r in eligible}
    for pair in ALWAYS_VERIFY:
        if pair in keyed and keyed[pair] not in shortlist:
            shortlist.append(keyed[pair])
    print(f"\n[xgb] verifying {len(shortlist)} pairs with the real XGB predictor "
          f"(3 held-out split seeds each) ...")
    for r in shortlist:
        m, s = xgb_predictor(df[r["task"]].to_numpy(), df[r["protected"]].to_numpy())
        r["xgb_auc"] = round(m, 4)
        r["xgb_auc_std"] = round(s, 4)
        print(f"  {r['task']:<22} -> {r['protected']:<10} closed-form={r['closed_form_auc']:.4f} "
              f"XGB={m:.4f}±{s:.4f}  phi={r['phi']:+.3f}"
              + ("  [NEAR-DUP]" if r["near_duplicate_flag"] else "")
              )

    # ---- ranked table (balanced pairs only) ----
    print("\n" + "=" * 100)
    print("RANKED COUPLING TABLE — predictor = AUC(protected | task label alone), CelebA train split")
    print("=" * 100)
    print(f"{'task':<22} {'protected':<10} {'predictor':>9} {'xgb':>8} {'phi':>7} "
          f"{'task+%':>7} {'prot+%':>7}  flags")
    print("-" * 100)
    for r in eligible:
        flags = []
        if r["near_duplicate_flag"]:
            flags.append("NEAR-DUP")
        xgb_s = f"{r['xgb_auc']:.4f}" if "xgb_auc" in r else "   —  "
        print(f"{r['task']:<22} {r['protected']:<10} {r['closed_form_auc']:>9.4f} {xgb_s:>8} "
              f"{r['phi']:>+7.3f} {r['task_pos_rate']:>7.3f} {r['prot_pos_rate']:>7.3f}  "
              + ",".join(flags))
    print("-" * 100)

    out = {"experiment": "celeba_coupling_scan", "n_train_rows": n,
           "protected_candidates": PROTECTED, "balance_window": [BALANCE_LO, BALANCE_HI],
           "skipped_unbalanced_attrs": {c: pos_rate[c] for c in skipped},
           "pos_rates": pos_rate, "pairs": rows}
    (RESULTS_DIR / "celeba_coupling_scan.json").write_text(json.dumps(out, indent=2))
    _plot(eligible, RESULTS_DIR / "celeba_coupling_scan.png")
    print(f"[save] {RESULTS_DIR / 'celeba_coupling_scan.json'} and .png")


RECOMMENDED = [("Smiling", "Young"), ("Attractive", "Young")]


def _plot(eligible, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(eligible, key=lambda r: r["closed_form_auc"])
    fig, ax = plt.subplots(figsize=(8.5, 0.22 * len(rows) + 1.6))
    names = [f"{r['task']} → {r['protected']}" for r in rows]
    vals = [r["closed_form_auc"] for r in rows]
    colors = []
    for r in rows:
        if (r["task"], r["protected"]) in RECOMMENDED:
            colors.append("C2")
        elif r["near_duplicate_flag"]:
            colors.append("C7")
        else:
            colors.append({"Male": "C0", "Young": "C1"}.get(r["protected"], "C4"))
    ax.barh(range(len(rows)), [v - 0.5 for v in vals], left=0.5, color=colors, height=0.75)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=6)
    ax.axvline(0.55, color="green", ls="--", lw=1, label="0.55 (the wall, from the tabular ramp)")
    ax.set_xlabel("predictor: AUC(protected | task label alone)")
    ax.set_title("CelebA coupling scan — green = recommended cells, gray = near-duplicate pairs",
                 fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0.5, 0.92)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
