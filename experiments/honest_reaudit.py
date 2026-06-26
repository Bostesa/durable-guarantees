"""Experiment 5 — HONEST RE-AUDIT: re-measure every R²-"stopped" verdict with
strong nonlinear probes (XGBoost + deep MLP), not just the linear R² certificate.

The Stage-2 hardening test showed the linear R² / dominant-axis certificate is blind
to nonlinear recovery: a from-scratch encoder read R²=0.023 ("stopped") while XGBoost
recovered race at 0.91 AUC. That means EVERY "stopped"/"durable" verdict measured only
with R² in Experiments 2-4 is suspect. This script re-audits them.

Two re-audits, both against the SAME held-out XGBoost + MLP probe battery used in the
hardening test (PCRL-auditor configs):

  PART 1 — Experiment 2 noise channel (Adult / income_prediction / sex). For each σ in
    the sweep, measure XGB and MLP sex recovery (acc + AUC) through a fresh noise
    channel — not just R². Find the σ (if any) that drives XGBoost to chance, and
    report the REAL income utility cost there.

  PART 2 — Experiment 4 Stage 1 smart erasers. Re-measure every method that PASSED the
    R² certificate (noise σ1/2/8, LEOPARD-MMD and HSIC projections at the ranks that
    stopped) on Adult/sex with XGB + MLP. Re-check HMDA/race (noise σ8, the only R²
    stopper there) for completeness.

Output: a master table — experiment | cell | method | R² verdict (old) | XGB AUC
(honest) | MLP AUC | does the honest probe agree the attribute is stopped? — and a
verdict on how many "durable" results survive honest measurement.

Reuses fit_projection/apply_projection (Stage 1), task_accuracy (Exp 2), and the
multiclass PCRL-auditor probe (_probe, Stage-2 hardening). macOS OpenMP guard: xgboost
before torch. Local MPS/CPU; never AWS.
"""

from __future__ import annotations

import os

# OpenMP guard MUST precede torch/xgboost import (macOS duplicate-runtime crash).
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

from experiments.hardening_test import _probe  # noqa: E402  (multiclass PCRL-auditor probe)
from experiments.noise_channel_test import (  # noqa: E402
    SEEDS, TAU, task_accuracy,
)
from experiments.smart_erasure import apply_projection, fit_projection  # noqa: E402
from utils.pcrl_io import (  # noqa: E402
    INCOME_PREDICTION_PURPOSE_IDX, build_adult_train_loader, build_train_loader,
    extract_attr, extract_frozen_reps, extract_task_label, get_device,
    load_adult_encoder, load_encoder,
)

RESULTS_DIR = REPO_ROOT / "results"

# A probe "agrees stopped" only if BOTH strong probes are within this margin of
# chance AUC (0.5). 0.55 is a generous bar (a real auditor would be stricter).
HONEST_AUC_BAR = 0.55
ADULT_SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def probe_battery(arch, H, y, n_classes, seeds, sigma_abs=0.0):
    """Held-out race/sex recovery (acc + AUC) for one strong probe, averaged over
    seeds. If sigma_abs>0 a fresh Gaussian channel is drawn per seed (matching the
    Exp-2 noise channel: test rows carry independent channel noise)."""
    from sklearn.model_selection import train_test_split
    accs, aucs = [], []
    for seed in seeds:
        X = H.astype(np.float64)
        if sigma_abs > 0.0:
            X = X + np.random.default_rng(seed).normal(0.0, sigma_abs, size=X.shape)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y)
        acc, auc = _probe(arch, Xtr, ytr, Xte, yte, seed, n_classes)
        accs.append(acc)
        aucs.append(auc)
    return {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs))}


def _honest_stopped(xgb_auc, mlp_auc):
    return bool(max(xgb_auc, mlp_auc) <= HONEST_AUC_BAR)


# --------------------------------------------------------------------------- #
#  PART 1 — Experiment 2 noise channel (Adult/sex)                            #
# --------------------------------------------------------------------------- #
def part1_noise_channel(master):
    device = get_device()
    encoder, _ = load_adult_encoder()
    purposes, train_ds, loader = build_adult_train_loader()
    H = extract_frozen_reps(encoder, loader, INCOME_PREDICTION_PURPOSE_IDX, device)
    sex = extract_attr(loader, "sex")
    income = extract_task_label(loader, "income")
    sigma_h = float(H.std(0).mean())
    sex_majority = float(np.bincount(sex).max() / len(sex))
    income_majority = float(np.bincount(income).max() / len(income))
    n_sex = int(sex.max()) + 1

    # Old R² verdicts from the saved Experiment-2 sweep.
    e2 = {r["sigma_rel"]: r for r in json.load(open(RESULTS_DIR / "noise_channel_test.json"))["rows"]}

    print("\n" + "=" * 92)
    print("PART 1 — Experiment 2 noise channel (Adult/income/sex): R² vs honest XGB/MLP")
    print("=" * 92)
    print(f"[setup] sigma_h={sigma_h:.4f}  sex majority={sex_majority:.4f}  "
          f"income majority={income_majority:.4f}  chance AUC=0.5")
    print(f"\n{'σ':>5} {'oldR²':>7} {'oldVerdict':>10} | {'XGB acc':>8} {'XGB AUC':>8} | "
          f"{'MLP acc':>8} {'MLP AUC':>8} | {'income acc':>10} {'lift':>7} | {'honest?':>9}")
    print("-" * 92)

    rows = []
    first_chance_sigma = None
    for sr in ADULT_SIGMAS:
        sa = sr * sigma_h
        xgb = probe_battery("XGB", H, sex, n_sex, SEEDS, sigma_abs=sa)
        mlp = probe_battery("MLP", H, sex, n_sex, SEEDS, sigma_abs=sa)
        income_acc = float(np.mean([task_accuracy(H, income, sa, s) for s in SEEDS]))
        lift = income_acc - income_majority
        old = e2.get(sr, {})
        old_r2 = old.get("attacked_linear_r2_mean", float("nan"))
        old_stopped = not old.get("breaches_tau", True)
        honest = _honest_stopped(xgb["auc"], mlp["auc"])
        if honest and first_chance_sigma is None and sr > 0:
            first_chance_sigma = sr
        rows.append({"sigma_rel": sr, "old_r2": old_r2, "old_stopped": old_stopped,
                     "xgb_acc": xgb["acc"], "xgb_auc": xgb["auc"],
                     "mlp_acc": mlp["acc"], "mlp_auc": mlp["auc"],
                     "income_acc": income_acc, "income_lift": lift, "honest_stopped": honest})
        print(f"{sr:>5} {old_r2:>7.4f} {('STOP' if old_stopped else 'breach'):>10} | "
              f"{xgb['acc']:>8.4f} {xgb['auc']:>8.4f} | {mlp['acc']:>8.4f} {mlp['auc']:>8.4f} | "
              f"{income_acc:>10.4f} {lift:>+7.4f} | {('STOP' if honest else 'LEAKS'):>9}")

        master.append({"experiment": "E2 noise", "cell": "adult/sex",
                       "method": f"noise σ={sr}", "old_r2": old_r2, "old_verdict":
                       ("stopped" if old_stopped else "breach"), "xgb_auc": xgb["auc"],
                       "mlp_auc": mlp["auc"], "honest_stopped": honest})

    # Real utility cost at the σ where XGBoost is first driven to ~chance.
    print("-" * 92)
    if first_chance_sigma is not None:
        r = next(x for x in rows if x["sigma_rel"] == first_chance_sigma)
        print(f"[answer] XGBoost (and MLP) first reach ~chance (AUC≤{HONEST_AUC_BAR}) at σ={first_chance_sigma}: "
              f"XGB AUC={r['xgb_auc']:.4f}, MLP AUC={r['mlp_auc']:.4f}. REAL income utility there: "
              f"acc={r['income_acc']:.4f} (lift {r['income_lift']:+.4f} over majority {income_majority:.4f}).")
    else:
        r = rows[-1]
        print(f"[answer] NO σ in the sweep drives the strong probes to chance "
              f"(σ=8: XGB AUC={r['xgb_auc']:.4f}, MLP AUC={r['mlp_auc']:.4f}). Even σ=8, which "
              f"crushes income to lift {r['income_lift']:+.4f}, leaves sex tree-recoverable above chance.")
    return rows, first_chance_sigma, income_majority


# --------------------------------------------------------------------------- #
#  PART 2 — Experiment 4 Stage 1 smart erasers                                #
# --------------------------------------------------------------------------- #
def _audit_static(label, H_erased, attr, n_attr, master, exp, cell, old_r2):
    """Probe a FIXED (projected/erased) repr with XGB + MLP; record + print a row."""
    xgb = probe_battery("XGB", H_erased, attr, n_attr, SEEDS)
    mlp = probe_battery("MLP", H_erased, attr, n_attr, SEEDS)
    honest = _honest_stopped(xgb["auc"], mlp["auc"])
    print(f"  {label:<26} oldR²={old_r2:.4f} STOP | XGB AUC={xgb['auc']:.4f} acc={xgb['acc']:.4f}"
          f" | MLP AUC={mlp['auc']:.4f} | {'STOP(honest)' if honest else 'LEAKS'}")
    master.append({"experiment": exp, "cell": cell, "method": label, "old_r2": old_r2,
                   "old_verdict": "stopped", "xgb_auc": xgb["auc"], "mlp_auc": mlp["auc"],
                   "honest_stopped": honest})
    return {"method": label, "old_r2": old_r2, "xgb_auc": xgb["auc"], "xgb_acc": xgb["acc"],
            "mlp_auc": mlp["auc"], "honest_stopped": honest}


def part2_smart_erasers(master):
    device = get_device()
    print("\n" + "=" * 92)
    print("PART 2 — Experiment 4 Stage 1 smart erasers: R²-stoppers vs honest XGB/MLP")
    print("=" * 92)
    out = {"adult": [], "hmda": []}

    # ---- Adult/sex (fragile cell): re-audit every R²-stopper -----------------
    encoder, _ = load_adult_encoder()
    purposes, train_ds, loader = build_adult_train_loader()
    H = extract_frozen_reps(encoder, loader, INCOME_PREDICTION_PURPOSE_IDX, device)
    sex = extract_attr(loader, "sex")
    n_sex = int(sex.max()) + 1
    sigma_h = float(H.std(0).mean())
    H_t = torch.from_numpy(H).to(device)
    saved = {(_mk(r)): r for r in json.load(open(RESULTS_DIR / "smart_erasure_adult.json"))["rows"]}

    print("\n--- Adult/income/sex (fragile cell) ---")
    # noise σ1/2/8 (re-uses the channel; static probe with sigma applied)
    for sr in (1.0, 2.0, 8.0):
        key = f"noise sigma={sr}"
        old_r2 = saved[key]["attacked_r2"]
        xgb = probe_battery("XGB", H, sex, n_sex, SEEDS, sigma_abs=sr * sigma_h)
        mlp = probe_battery("MLP", H, sex, n_sex, SEEDS, sigma_abs=sr * sigma_h)
        honest = _honest_stopped(xgb["auc"], mlp["auc"])
        print(f"  {key:<26} oldR²={old_r2:.4f} STOP | XGB AUC={xgb['auc']:.4f} acc={xgb['acc']:.4f}"
              f" | MLP AUC={mlp['auc']:.4f} | {'STOP(honest)' if honest else 'LEAKS'}")
        master.append({"experiment": "E4S1 noise", "cell": "adult/sex", "method": key,
                       "old_r2": old_r2, "old_verdict": "stopped", "xgb_auc": xgb["auc"],
                       "mlp_auc": mlp["auc"], "honest_stopped": honest})
        out["adult"].append({"method": key, "old_r2": old_r2, "xgb_auc": xgb["auc"],
                             "xgb_acc": xgb["acc"], "mlp_auc": mlp["auc"], "honest_stopped": honest})

    # LEOPARD-MMD / HSIC projections at the ranks that stopped (re-fit, re-apply, probe)
    for fam, kind in (("LEOPARD-MMD proj", "mmd"), ("HSIC proj", "hsic")):
        for r in [row for row in saved.values()
                  if row["method"] == fam and row["stops_attack"]]:
            rank = r["rank"]
            Q = fit_projection(H_t, sex, rank, kind, device, seed=0)
            H_er = apply_projection(H, Q)
            out["adult"].append(_audit_static(f"{fam} r={rank}", H_er, sex, n_sex, master,
                                               "E4S1 proj", "adult/sex", r["attacked_r2"]))

    # ---- HMDA/race (robust cell): only R²-stopper was noise σ8 ----------------
    purposes_h, train_ds_h, loader_h = build_train_loader("hmda")
    enc_h, _ = load_encoder("hmda", train_ds_h.info.num_features, len(purposes_h))
    Hh = extract_frozen_reps(enc_h, loader_h, 0, device)
    race = extract_attr(loader_h, "race")
    n_race = int(race.max()) + 1
    sigma_hh = float(Hh.std(0).mean())
    saved_h = {(_mk(r)): r for r in json.load(open(RESULTS_DIR / "smart_erasure.json"))["rows"]}

    print(f"\n--- HMDA/underwriting/race (robust cell, {n_race} race classes) ---")
    key = "noise sigma=8.0"
    old_r2 = saved_h[key]["attacked_r2"]
    xgb = probe_battery("XGB", Hh, race, n_race, SEEDS, sigma_abs=8.0 * sigma_hh)
    mlp = probe_battery("MLP", Hh, race, n_race, SEEDS, sigma_abs=8.0 * sigma_hh)
    honest = _honest_stopped(xgb["auc"], mlp["auc"])
    print(f"  {key:<26} oldR²={old_r2:.4f} STOP | XGB AUC={xgb['auc']:.4f} acc={xgb['acc']:.4f}"
          f" | MLP AUC={mlp['auc']:.4f} | {'STOP(honest)' if honest else 'LEAKS'}")
    master.append({"experiment": "E4S1 noise", "cell": "hmda/race", "method": key,
                   "old_r2": old_r2, "old_verdict": "stopped", "xgb_auc": xgb["auc"],
                   "mlp_auc": mlp["auc"], "honest_stopped": honest})
    out["hmda"].append({"method": key, "old_r2": old_r2, "xgb_auc": xgb["auc"],
                        "xgb_acc": xgb["acc"], "mlp_auc": mlp["auc"], "honest_stopped": honest})
    return out


def _mk(r):
    return r["method"] + (f" r={r['rank']}" if r.get("rank") is not None else "")


# --------------------------------------------------------------------------- #
#  Master table + verdict                                                     #
# --------------------------------------------------------------------------- #
def master_table_and_verdict(master):
    print("\n" + "=" * 96)
    print("MASTER TABLE — every R²-'stopped' result re-measured with honest XGBoost/MLP probes")
    print("=" * 96)
    print(f"{'experiment':<12} {'cell':<11} {'method':<22} {'oldR²':>7} {'oldVerd':>8} "
          f"{'XGB AUC':>8} {'MLP AUC':>8} {'honest agrees?':>15}")
    print("-" * 96)
    survived, total = 0, 0
    for r in master:
        total += 1
        survived += int(r["honest_stopped"])
        print(f"{r['experiment']:<12} {r['cell']:<11} {r['method']:<22} {r['old_r2']:>7.4f} "
              f"{r['old_verdict']:>8} {r['xgb_auc']:>8.4f} {r['mlp_auc']:>8.4f} "
              f"{('YES stopped' if r['honest_stopped'] else 'NO — leaks'):>15}")
    print("-" * 96)
    print(f"honest-probe bar: max(XGB,MLP) AUC ≤ {HONEST_AUC_BAR} (chance=0.5)")
    print(f"\nSURVIVED honest measurement: {survived}/{total} R²-'stopped' verdicts.")

    leaks = [r for r in master if not r["honest_stopped"]]
    survivors = [r for r in master if r["honest_stopped"]]
    print("\n================== EXPERIMENT 5 VERDICT ==================")
    if survived == 0:
        v = (f"ALL {total} R²-'stopped' results COLLAPSE under honest measurement: every method "
             f"that passed the linear R²/dominant-axis certificate leaves the attribute recoverable "
             f"by XGBoost/MLP above chance (max honest AUC "
             f"{max(r['xgb_auc'] for r in master):.3f}). The linear certificate systematically "
             f"misleads — it certified 'durable' in 0 genuine cases. This is the strongest possible "
             f"form of the central finding.")
    elif survived < total:
        surv_names = ", ".join(f"{r['experiment']}:{r['method']}" for r in survivors)
        v = (f"{survived}/{total} R²-'stopped' results SURVIVE honest measurement; {len(leaks)} "
             f"collapse. Genuine durability cases (honest XGB+MLP also ≤ {HONEST_AUC_BAR} AUC): "
             f"{surv_names}. The rest were R²-blindness like the Stage-2 win. The real set of durable "
             f"results is the smaller, honest-probe-verified one — and it costs whatever utility the "
             f"surviving σ/rank demands.")
    else:
        v = (f"All {total} R²-'stopped' results also pass the honest XGB/MLP probe. The linear "
             f"certificate did not mislead on these particular cells — but note the bar is generous "
             f"(AUC ≤ {HONEST_AUC_BAR}) and the residual is non-zero.")
    print(v)
    return v, survived, total


def main():
    print(f"[setup] device={get_device()}  (OpenMP guard set; xgboost before torch)")
    master = []
    p1_rows, chance_sigma, income_majority = part1_noise_channel(master)
    p2 = part2_smart_erasers(master)
    verdict, survived, total = master_table_and_verdict(master)

    out = {"experiment": "honest_reaudit", "tau": TAU, "honest_auc_bar": HONEST_AUC_BAR,
           "attacker_seeds": SEEDS, "part1_exp2_noise_adult": p1_rows,
           "exp2_first_chance_sigma": chance_sigma, "income_majority": income_majority,
           "part2_smart_erasers": p2, "master": master,
           "survived": survived, "total": total, "verdict": verdict}
    (RESULTS_DIR / "honest_reaudit.json").write_text(json.dumps(out, indent=2))
    _plot(p1_rows, RESULTS_DIR / "honest_reaudit.png")
    print(f"\n[save] {RESULTS_DIR/'honest_reaudit.json'} and .png")


def _plot(p1_rows, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sig = [r["sigma_rel"] for r in p1_rows]
    fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
    ax1.plot(sig, [r["old_r2"] for r in p1_rows], "^-", color="C0", lw=1.6,
             label="LoRA attacked R² (old certificate)")
    ax1.plot(sig, [r["xgb_auc"] for r in p1_rows], "s-", color="C3", lw=2,
             label="XGBoost sex AUC (honest)")
    ax1.plot(sig, [r["mlp_auc"] for r in p1_rows], "o-", color="C1", lw=2,
             label="MLP sex AUC (honest)")
    ax1.axhline(TAU, color="C0", ls="--", lw=1, label=f"τ={TAU} (R² bar)")
    ax1.axhline(0.5, color="gray", ls=":", lw=1, label="chance AUC=0.5")
    ax1.axhline(HONEST_AUC_BAR, color="C3", ls="--", lw=0.9, label=f"honest bar AUC={HONEST_AUC_BAR}")
    ax1.set_xlabel("noise σ (relative to repr per-dim std)")
    ax1.set_ylabel("R²  /  AUC")
    ax1.set_title("Honest re-audit of the Exp-2 noise channel (Adult/sex)\n"
                  "R² says 'stopped' from σ=1; XGBoost keeps recovering sex well past it")
    ax2 = ax1.twinx()
    ax2.plot(sig, [r["income_acc"] for r in p1_rows], "D-", color="C2", lw=1.4,
             label="income acc (utility)")
    ax2.set_ylabel("income accuracy", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
