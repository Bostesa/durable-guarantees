"""Experiment 4 — STAGE 2 HARDENING: stress-test the from-scratch winner.

Stage 2 found that a representation trained FROM SCRATCH on raw HMDA features,
jointly with task CE + λ·HSIC(E(·), race) at λ=100, is the only intervention that
makes HMDA/underwriting/race durable (attacked R²≈0.023 ≤ τ) while keeping ~92% of
task lift. But that headline rested on a SINGLE training seed and a SINGLE rank-8
ReLU attacker. This script hardens it two ways:

  PART A — error bars: re-train the winning config (from-scratch MLP on raw x,
    λ=100) across 3 training seeds. Report attacked R² and task lift as mean±std.
    Is the below-τ, utility-keeping result stable, or a lucky seed?

  PART B — stronger attackers: against the trained-from-scratch FROZEN encoder
    (seed 0), run the full Experiment-3 attacker battery — rank-8 ReLU LoRA
    (control), rank-32 ReLU LoRA, deep MLP probe (256-256), XGBoost probe — not
    just the rank-8 ReLU. Does durability hold against stronger adversaries, or
    does one break it? As a reference we also run the MLP/XGB probes on the RAW
    frozen PCRL repr, where race IS recoverable, to confirm the probes can find
    race when it is present.

Reuses train_two_component (Stage 2), run_attack/task_accuracy (Exp 2), evaluate
(Stage 1), and the PCRL-auditor MLP/XGB probe configs (Exp 3). macOS OpenMP guard:
xgboost imported before torch. Local MPS/CPU; never AWS.
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

from experiments.noise_channel_test import SEEDS, TAU, run_attack  # noqa: E402
from experiments.smart_erasure import evaluate  # noqa: E402  (attacked R² + task lift, rank-8)
from experiments.training_time_erasure import (  # noqa: E402
    ATTR, PURPOSE_IDX, TASK, train_two_component,
)
from utils.pcrl_io import (  # noqa: E402
    build_train_loader, extract_attr, extract_frozen_reps, extract_task_label,
    get_device, load_encoder,
)

RESULTS_DIR = REPO_ROOT / "results"

WIN_ARCH = "mlp_x"        # from-scratch MLP on raw features
WIN_LAMBDA = 100.0        # the λ that won Stage 2
TRAIN_SEEDS = [0, 1, 2]   # error-bar seeds for the from-scratch training
CANON_SEED = 0            # the encoder used for the stronger-attacker battery


# --------------------------------------------------------------------------- #
#  PCRL-auditor MLP / XGB probes (race recovery), binary or multiclass.       #
# --------------------------------------------------------------------------- #
def _probe(arch, Xtr, ytr, Xte, yte, seed, n_classes):
    """Fit one PCRL-auditor classifier; return (accuracy, AUC) on held-out.

    MLP/XGB configs copied from run_eval_multi.py::attack_acc, generalised to the
    multiclass race case (HMDA race has >2 classes).
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    if arch == "MLP":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(256, 256), activation="relu",
                            alpha=1e-4, solver="adam", learning_rate_init=1e-3,
                            max_iter=100, early_stopping=True, validation_fraction=0.1,
                            n_iter_no_change=8, random_state=seed, batch_size=256)
    elif arch == "XGB":
        from xgboost import XGBClassifier
        obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
        clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                            objective=obj, tree_method="hist", eval_metric="logloss",
                            random_state=seed, n_jobs=2, verbosity=0)
    else:
        raise ValueError(arch)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    prob = clf.predict_proba(Xte)
    try:
        if n_classes == 2:
            auc = float(roc_auc_score(yte, prob[:, 1]))
        else:
            auc = float(roc_auc_score(yte, prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return acc, auc


def probe_battery(arch, H, y, n_classes, seeds):
    """Average a probe's held-out race accuracy/AUC over seeds."""
    from sklearn.model_selection import train_test_split
    accs, aucs = [], []
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(
            H.astype(np.float64), y, test_size=0.25, random_state=seed, stratify=y)
        acc, auc = _probe(arch, Xtr, ytr, Xte, yte, seed, n_classes)
        accs.append(acc)
        aucs.append(auc)
    return {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs))}


def lora_battery(H_t, race_t, race_np, device, rank, alpha, seeds):
    """Average the ReLU LoRA attacker's R²/acc/AUC over attacker seeds (no noise)."""
    runs = [run_attack(H_t, race_t, race_np, device, s, 0.0, rank=rank, alpha=alpha)
            for s in seeds]
    return {"r2": float(np.mean([r["final_linear_r2"] for r in runs])),
            "r2_std": float(np.std([r["final_linear_r2"] for r in runs])),
            "acc": float(np.mean([r["acc"] for r in runs])),
            "auc": float(np.mean([r["auc"] for r in runs]))}


def main() -> None:
    device = get_device()
    print(f"[setup] device={device}  (OpenMP guard set; xgboost before torch)")
    purposes, train_ds, loader = build_train_loader("hmda")
    encoder, _ = load_encoder("hmda", train_ds.info.num_features, len(purposes))
    H = extract_frozen_reps(encoder, loader, PURPOSE_IDX, device)
    X = np.concatenate([b["features"].numpy() for b in loader]).astype(np.float32)
    race = extract_attr(loader, ATTR)
    task = extract_task_label(loader, TASK)
    n_race = int(race.max()) + 1
    n_task = int(task.max()) + 1
    task_majority = float(np.bincount(task).max() / len(task))
    race_majority = float(np.bincount(race).max() / len(race))
    chance_auc = 0.5
    print(f"[setup] HMDA/{purposes[PURPOSE_IDX].name}/{ATTR}: N={H.shape[0]} "
          f"x_dim={X.shape[1]} race={n_race}cls (majority {race_majority:.4f}) "
          f"task '{TASK}' majority={task_majority:.4f}")

    X_t = torch.from_numpy(X).to(device)
    h_cpu = torch.from_numpy(H).float().cpu()

    # ----------------------------------------------------------------------- #
    #  PART A — error bars over 3 training seeds                              #
    # ----------------------------------------------------------------------- #
    print(f"\n===== PART A: from-scratch {WIN_ARCH} λ={WIN_LAMBDA:g} over "
          f"{len(TRAIN_SEEDS)} training seeds =====")
    seed_rows, e_outs = [], {}
    for ts in TRAIN_SEEDS:
        E_out = train_two_component(X_t, race, task, n_race, n_task, WIN_ARCH,
                                    WIN_LAMBDA, h_cpu, device, seed=ts)
        e_outs[ts] = E_out
        ev = evaluate(E_out, race, task, task_majority, device)   # rank-8 attack, 3 attacker seeds
        seed_rows.append({"train_seed": ts, "attacked_r2": ev["attacked_r2"],
                          "attacked_r2_std": ev["attacked_r2_std"],
                          "task_lift": ev["task_lift"]})
        print(f"  train_seed={ts}: attacked R²={ev['attacked_r2']:.4f}"
              f"±{ev['attacked_r2_std']:.4f}  task_lift={ev['task_lift']:+.4f}  "
              f"{'STOPS' if ev['attacked_r2'] <= TAU else 'breaches'}")

    r2_vals = np.array([r["attacked_r2"] for r in seed_rows])
    lift_vals = np.array([r["task_lift"] for r in seed_rows])
    a_stats = {"attacked_r2_mean": float(r2_vals.mean()), "attacked_r2_std": float(r2_vals.std()),
               "task_lift_mean": float(lift_vals.mean()), "task_lift_std": float(lift_vals.std()),
               "all_below_tau": bool((r2_vals <= TAU).all())}

    print("\n  --- TABLE A: from-scratch winner across training seeds ---")
    print(f"  {'train seed':>10} {'attacked R²':>12} {'task lift':>12}")
    print("  " + "-" * 36)
    for r in seed_rows:
        print(f"  {r['train_seed']:>10} {r['attacked_r2']:>12.4f} {r['task_lift']:>+12.4f}")
    print("  " + "-" * 36)
    print(f"  {'mean±std':>10} {a_stats['attacked_r2_mean']:>7.4f}±{a_stats['attacked_r2_std']:.4f}"
          f"  {a_stats['task_lift_mean']:>+8.4f}±{a_stats['task_lift_std']:.4f}")
    print(f"  τ={TAU}; all seeds below τ: {a_stats['all_below_tau']}")

    # ----------------------------------------------------------------------- #
    #  PART B — stronger attackers vs the frozen from-scratch encoder (seed 0) #
    # ----------------------------------------------------------------------- #
    print(f"\n===== PART B: stronger attackers vs frozen from-scratch encoder "
          f"(train seed {CANON_SEED}) =====")
    E0 = e_outs[CANON_SEED]
    E0_t = torch.from_numpy(E0).to(device)
    race_t = torch.from_numpy(race).long().to(device)

    battery = []

    def add_lora(name, rank, alpha):
        m = lora_battery(E0_t, race_t, race, device, rank, alpha, SEEDS)
        below = m["r2"] <= TAU
        battery.append({"attacker": name, "kind": "lora", "target": "from-scratch E",
                        "r2": m["r2"], "r2_std": m["r2_std"], "acc": m["acc"], "auc": m["auc"],
                        "below_tau": bool(below)})
        print(f"  {name:<22} R²={m['r2']:.4f}±{m['r2_std']:.4f}  race_acc={m['acc']:.4f}"
              f"  race_AUC={m['auc']:.4f}  {'≤τ STOPS' if below else 'BREACHES τ'}")

    def add_probe(name, arch, H_src, target):
        m = probe_battery(arch, H_src, race, n_race, SEEDS)
        battery.append({"attacker": name, "kind": "probe", "target": target,
                        "r2": None, "acc": m["acc"], "acc_std": m["acc_std"],
                        "auc": m["auc"], "auc_std": m["auc_std"], "below_tau": None})
        print(f"  {name:<22} (no R²)        race_acc={m['acc']:.4f}±{m['acc_std']:.4f}"
              f"  race_AUC={m['auc']:.4f}  (majority={race_majority:.3f}, chance AUC={chance_auc})")

    print("  -- attackers on the from-scratch encoder E(x) --")
    add_lora("rank8 ReLU LoRA", 8, 16.0)
    add_lora("rank32 ReLU LoRA", 32, 64.0)
    add_probe("MLP probe (256-256)", "MLP", E0, "from-scratch E")
    add_probe("XGB probe", "XGB", E0, "from-scratch E")

    print("  -- reference: same probes on the RAW frozen PCRL repr (race present) --")
    add_probe("MLP probe (raw h)", "MLP", H, "raw frozen h")
    add_probe("XGB probe (raw h)", "XGB", H, "raw frozen h")

    # A probe "recovers race" if it beats majority acc by a margin OR AUC clears
    # ~chance by a margin. Use modest margins (acc +0.02, AUC +0.05).
    def recovers(row):
        if row["kind"] == "lora":
            return not row["below_tau"]
        return (row["acc"] > race_majority + 0.02) or (row["auc"] > chance_auc + 0.05)

    for row in battery:
        row["recovers_race"] = bool(recovers(row))

    _print_table_b(battery, race_majority, chance_auc)
    verdict = _verdict(a_stats, battery, race_majority)

    out = {"experiment": "hardening_test",
           "cell": f"hmda/{purposes[PURPOSE_IDX].name}/{ATTR}", "task": TASK, "tau": TAU,
           "winning_config": {"arch": WIN_ARCH, "lambda": WIN_LAMBDA},
           "task_majority": task_majority, "race_majority": race_majority,
           "train_seeds": TRAIN_SEEDS, "attacker_seeds": SEEDS,
           "part_a_rows": seed_rows, "part_a_stats": a_stats,
           "part_b_battery": battery, "verdict": verdict}
    (RESULTS_DIR / "hardening_test.json").write_text(json.dumps(out, indent=2))
    _plot(seed_rows, a_stats, RESULTS_DIR / "hardening_test.png")
    print(f"\n[save] {RESULTS_DIR/'hardening_test.json'} and .png")


def _print_table_b(battery, race_majority, chance_auc):
    print("\n  --- TABLE B: stronger attackers vs from-scratch encoder ---")
    print(f"  {'attacker':<22} {'target':<16} {'attacked R²':>12} {'race acc':>9} "
          f"{'race AUC':>9} {'below τ?':>10}")
    print("  " + "-" * 82)
    for r in battery:
        r2 = f"{r['r2']:.4f}" if r["r2"] is not None else "   n/a  "
        bt = ("yes" if r["below_tau"] else "NO") if r["below_tau"] is not None else \
             ("—" if not r["recovers_race"] else "RECOVERS")
        print(f"  {r['attacker']:<22} {r['target']:<16} {r2:>12} {r['acc']:>9.4f} "
              f"{r['auc']:>9.4f} {bt:>10}")
    print("  " + "-" * 82)
    print(f"  τ={TAU} (R² certificate, LoRA attackers); race majority acc={race_majority:.4f}; "
          f"chance AUC={chance_auc}")


def _verdict(a_stats, battery, race_majority) -> str:
    # Part A: stable iff all seeds below τ and lift std is small relative to mean.
    a_stable = a_stats["all_below_tau"]
    # Part B: durable iff every LoRA attacker stays ≤ τ and no probe recovers race
    # from the from-scratch encoder above the margins.
    e_attackers = [r for r in battery if r["target"] == "from-scratch E"]
    loras_hold = all(r["below_tau"] for r in e_attackers if r["kind"] == "lora")
    probes_clean = all(not r["recovers_race"] for r in e_attackers if r["kind"] == "probe")
    ref_probes = [r for r in battery if r["target"] == "raw frozen h" and r["kind"] == "probe"]
    ref_finds = any(r["recovers_race"] for r in ref_probes)
    b_robust = loras_hold and probes_clean

    print("\n  ================== HARDENING VERDICT ==================")
    breakers = [r["attacker"] for r in e_attackers if r["recovers_race"]]
    if a_stable and b_robust:
        ref_note = (" The same probes DO recover race from the raw frozen repr "
                    "(reference rows), so they are capable of finding race when it is "
                    "present — the from-scratch encoder genuinely removed it."
                    if ref_finds else "")
        v = (f"HOLDS. (A) The from-scratch winner is STABLE across {len(TRAIN_SEEDS)} "
             f"training seeds: attacked R²={a_stats['attacked_r2_mean']:.4f}"
             f"±{a_stats['attacked_r2_std']:.4f} (all ≤ τ={TAU}) and task lift "
             f"{a_stats['task_lift_mean']:+.4f}±{a_stats['task_lift_std']:.4f} — not a lucky seed. "
             f"(B) Durability is ROBUST to stronger attackers: rank-8 and rank-32 ReLU LoRA both "
             f"stay ≤ τ, and deep-MLP and XGBoost probes recover race no better than majority "
             f"({race_majority:.3f}) / chance from the frozen from-scratch encoder.{ref_note} "
             f"The constructive Stage-2 result is solid.")
    elif not a_stable and b_robust:
        v = (f"PARTIAL — seeds vary. Stronger attackers do not break the encoder, but the "
             f"from-scratch result is NOT stable across training seeds (attacked R²="
             f"{a_stats['attacked_r2_mean']:.4f}±{a_stats['attacked_r2_std']:.4f}; not all ≤ τ). "
             f"The headline depends on the training seed — report with this caveat.")
    elif a_stable and not b_robust:
        v = (f"BREAKS under a stronger attacker. Across seeds the rank-8 result is stable "
             f"(R²={a_stats['attacked_r2_mean']:.4f}±{a_stats['attacked_r2_std']:.4f} ≤ τ), but "
             f"a STRONGER adversary recovers race from the from-scratch encoder: {breakers}. "
             f"Durability is attacker-specific — the rank-8 R² certificate understated the leak, "
             f"exactly the failure mode Experiment 1/3 warned about. Report plainly: the "
             f"constructive result does NOT survive the full battery.")
    else:
        v = (f"FAILS both checks: from-scratch result is seed-unstable "
             f"(R²={a_stats['attacked_r2_mean']:.4f}±{a_stats['attacked_r2_std']:.4f}) AND a "
             f"stronger attacker breaks it ({breakers}). The Stage-2 headline does not hold up.")
    print("  " + v)
    return v


def _plot(seed_rows, a_stats, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    xs = [r["attacked_r2"] for r in seed_rows]
    ys = [r["task_lift"] for r in seed_rows]
    ax.scatter(xs, ys, s=130, color="C3", marker="^", zorder=5,
               label="from-scratch winner, per training seed")
    for r in seed_rows:
        ax.annotate(f"seed {r['train_seed']}", (r["attacked_r2"], r["task_lift"]),
                    fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.errorbar([a_stats["attacked_r2_mean"]], [a_stats["task_lift_mean"]],
                xerr=[a_stats["attacked_r2_std"]], yerr=[a_stats["task_lift_std"]],
                fmt="D", color="black", ms=8, capsize=4, zorder=6,
                label=f"mean±std (R²={a_stats['attacked_r2_mean']:.3f}"
                      f"±{a_stats['attacked_r2_std']:.3f})")
    ax.axvline(TAU, color="black", ls="--", lw=1.2, label=f"τ={TAU} (stops ←)")
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.set_xlabel("attacked R²(repr, race)  — lower = attack stopped")
    ax.set_ylabel("task lift over majority (underwriting)  — higher = utility kept")
    ax.set_title("Hardening Part A: from-scratch winner across 3 training seeds\n"
                 "(want all points LEFT of τ AND above 0)")
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
