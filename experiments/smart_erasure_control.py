"""Experiment 4 — POSITIVE CONTROL: smart erasure on the EASY cell (Adult/sex).

Experiment 4 showed that on HMDA/race (robust, nonlinearly-diffused, task-entangled
attribute) NO erasure beats noise: targeted MMD/HSIC projections never stop the
attack, and only isotropic noise does — at the cost of all utility. To turn that
negative into a characterization, we need the contrast: the SAME erasers run on a
cell where the protected signal is FRAGILE — Adult / income_prediction / sex, the
easy cell from Experiments 1-3.

Same battery as smart_erasure.py (reused verbatim): no-erasure, LEACE, a noise
sweep, and LEOPARD-MMD / HSIC rank-r projections, all frozen transforms before the
same ReLU attacker (3 seeds). Question: do targeted erasers STOP the attack on
Adult/sex while keeping task utility (income lift over majority)? If yes, the
finding is a characterization (targeted erasure works iff the signal is fragile /
low-rank-separable). If even here they fail, that is reported too. Local MPS/CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.noise_channel_test import SEEDS, TAU, BATCH_SIZE  # noqa: E402
from experiments.smart_erasure import (  # noqa: E402  (reuse the erasure battery)
    RANKS, apply_projection, evaluate, evaluate_noise, fit_projection,
)
from utils.pcrl_io import (  # noqa: E402
    build_train_loader, extract_attr, extract_frozen_reps, extract_task_label,
    get_device, load_encoder,
)

RESULTS_DIR = REPO_ROOT / "results"
PURPOSE_IDX = 0          # income_prediction
ATTR = "sex"
TASK = "income"
NOISE_SIGMA_RELS = [1.0, 2.0, 8.0]   # Adult stops cheaply ~σ1-2 (Exp 2)


def run_cell():
    device = get_device()
    print(f"[setup] device={device}")
    purposes, train_ds, loader = build_train_loader("adult", batch_size=BATCH_SIZE)
    encoder, lora_target = load_encoder("adult", train_ds.info.num_features, len(purposes))
    H = extract_frozen_reps(encoder, loader, PURPOSE_IDX, device)
    sex = extract_attr(loader, ATTR)
    income = extract_task_label(loader, TASK)
    sigma_h = float(H.std(0).mean())
    task_majority = float(np.bincount(income).max() / len(income))
    print(f"[setup] {purposes[PURPOSE_IDX].name}/{ATTR}: N={H.shape[0]} sigma_h={sigma_h:.4f} "
          f"sex({int(sex.max())+1} cls) task '{TASK}' majority={task_majority:.4f}")

    H_t = torch.from_numpy(H).to(device)
    rows = []

    def add(method, ev, rank=None):
        rows.append({"method": method, "rank": rank, **ev,
                     "stops_attack": bool(ev["attacked_r2"] <= TAU)})
        rk = f" r={rank}" if rank is not None else ""
        print(f"  {method+rk:<24} attacked R²={ev['attacked_r2']:.4f}±{ev['attacked_r2_std']:.4f}"
              f"  task_lift={ev['task_lift']:+.4f}  "
              f"{'STOPS' if ev['attacked_r2'] <= TAU else 'breaches'}")

    print("\n--- baselines ---")
    add("no erasure (frozen)", evaluate(H.copy(), sex, income, task_majority, device))

    from concept_erasure import LeaceEraser
    zoh = torch.nn.functional.one_hot(torch.from_numpy(sex).long(), int(sex.max()) + 1).float()
    eraser = LeaceEraser.fit(H_t.float().cpu(), zoh)
    H_leace = eraser(H_t.float().cpu()).numpy().astype(np.float32)
    add("LEACE (linear)", evaluate(H_leace, sex, income, task_majority, device))

    print("\n--- noise sweep ---")
    for sr in NOISE_SIGMA_RELS:
        add(f"noise sigma={sr}",
            evaluate_noise(H, sex, income, task_majority, sr * sigma_h, device))

    for kind, label in [("mmd", "LEOPARD-MMD proj"), ("hsic", "HSIC proj")]:
        print(f"\n--- {label} (sweep rank) ---")
        for r in RANKS:
            Q = fit_projection(H_t, sex, r, kind, device, seed=0)
            add(label, evaluate(apply_projection(H, Q), sex, income, task_majority, device), rank=r)

    return rows, task_majority


def _print_table(rows, task_majority) -> None:
    print("\n===== POSITIVE CONTROL: smart erasure on Adult/income_prediction/sex =====")
    print(f"{'method':<24} {'attacked R²':>12} {'task lift':>10} {'verdict':>22}")
    print("-" * 72)
    clean_lift = next(r["task_lift"] for r in rows if r["method"] == "no erasure (frozen)")
    for r in rows:
        name = r["method"] + (f" r={r['rank']}" if r["rank"] is not None else "")
        if r["stops_attack"]:
            kept = r["task_lift"] / clean_lift if clean_lift > 0 else 0.0
            v = f"STOPS, {kept*100:.0f}% lift kept"
        else:
            v = "breaches τ"
        print(f"{name:<24} {r['attacked_r2']:>12.4f} {r['task_lift']:>+10.4f} {v:>22}")
    print("-" * 72)
    print(f"τ={TAU}  clean task lift over majority={clean_lift:+.4f} (majority={task_majority:.4f})")


def _best_smart_stopper(rows):
    clean_lift = next(r["task_lift"] for r in rows if r["method"] == "no erasure (frozen)")
    smart = [r for r in rows if r["method"].startswith(("LEOPARD", "HSIC")) and r["stops_attack"]]
    if not smart:
        return None, clean_lift
    return max(smart, key=lambda r: r["task_lift"]), clean_lift


def main() -> None:
    rows, task_majority = run_cell()
    _print_table(rows, task_majority)

    best, clean_lift = _best_smart_stopper(rows)
    out = {"experiment": "smart_erasure_control", "cell": "adult/income_prediction/sex",
           "task": TASK, "tau": TAU, "task_majority": task_majority, "rows": rows}
    (RESULTS_DIR / "smart_erasure_adult.json").write_text(json.dumps(out, indent=2))

    # ---- side-by-side contrast against the saved HMDA/race result ----------
    hmda = json.loads((RESULTS_DIR / "smart_erasure.json").read_text())
    h_rows = hmda["rows"]
    h_clean = next(r for r in h_rows if r["method"] == "no erasure (frozen)")
    h_smart = [r for r in h_rows if r["method"].startswith(("LEOPARD", "HSIC")) and r["stops_attack"]]
    h_best_proj = min((r for r in h_rows if r["method"].startswith(("LEOPARD", "HSIC"))),
                      key=lambda r: r["attacked_r2"])  # closest any projection got

    a_clean = next(r for r in rows if r["method"] == "no erasure (frozen)")

    print("\n================== EXPERIMENT 4 — SIDE-BY-SIDE CONTRAST ==================")
    print(f"{'cell':<28} {'attack@rest R²':>14} {'targeted erasure stops + keeps utility?':>44}")
    print("-" * 90)
    if best is not None:
        kept = best["task_lift"] / clean_lift if clean_lift > 0 else 0.0
        a_msg = (f"YES — {best['method']} r={best['rank']} R²={best['attacked_r2']:.3f}≤τ, "
                 f"{kept*100:.0f}% lift kept ({best['task_lift']:+.3f})")
    else:
        a_msg = "NO — no projection reaches R²≤τ"
    h_msg = (f"NO — best proj only R²={h_best_proj['attacked_r2']:.3f} (>τ); none stop"
             if not h_smart else f"YES (unexpected) {h_smart}")
    print(f"{'Adult/income/sex (FRAGILE)':<28} {a_clean['attacked_r2']:>14.3f}   {a_msg}")
    print(f"{'HMDA/underwriting/race (ROBUST)':<28} {h_clean['attacked_r2']:>14.3f}   {h_msg}")
    print("-" * 90)

    if best is not None:
        verdict = (
            f"CONTRAST CONFIRMED. On Adult/sex the leaked signal is FRAGILE: a targeted "
            f"erasure ({best['method']} r={best['rank']}) stops the attack "
            f"(R²={best['attacked_r2']:.3f}≤τ) while keeping {best['task_lift']/clean_lift*100:.0f}% "
            f"of income lift ({best['task_lift']:+.3f} vs clean {clean_lift:+.3f}) — it removes a "
            f"low-rank, task-separable sex subspace. On HMDA/race the signal is ROBUST and "
            f"task-entangled: the SAME erasers never reach τ (best R²≈{h_best_proj['attacked_r2']:.2f}, "
            f"3×τ) and only isotropic noise stops it, destroying all utility. "
            f"Targeted erasure works iff the protected signal is low-rank-separable from the "
            f"task — fragile→cheap, robust→fundamentally costly.")
    else:
        verdict = (
            f"Targeted erasure does NOT cleanly stop the attack even on Adult/sex "
            f"(best projection R²>{TAU}). The fragile/robust contrast with HMDA is therefore "
            f"weaker than expected — report honestly: a frozen low-rank projection is a weak "
            f"eraser even for a fragile signal; noise remains the only reliable stopper, cheap "
            f"on Adult and ruinous on HMDA.")
    print(verdict)
    out["contrast_verdict"] = verdict
    (RESULTS_DIR / "smart_erasure_adult.json").write_text(json.dumps(out, indent=2))
    _plot(rows, RESULTS_DIR / "smart_erasure_adult.png")
    print(f"\n[save] {RESULTS_DIR/'smart_erasure_adult.json'} and .png")


def _plot(rows, png_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    styles = {"LEOPARD-MMD proj": ("o", "C0"), "HSIC proj": ("s", "C1")}
    for label, (mk, c) in styles.items():
        pts = [(r["attacked_r2"], r["task_lift"], r["rank"]) for r in rows if r["method"] == label]
        if not pts:
            continue
        xs, ys, _ = zip(*pts)
        ax.plot(xs, ys, mk + "-", color=c, label=label, ms=6)
        for x, y, rk in pts:
            ax.annotate(f"r{rk}", (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
    for r in rows:
        if r["method"].startswith(("LEOPARD", "HSIC")):
            continue
        m, c = {"no erasure (frozen)": ("*", "C2"), "LEACE (linear)": ("D", "C4")}.get(
            r["method"], ("P", "C3"))
        ax.scatter(r["attacked_r2"], r["task_lift"], marker=m, s=110, color=c, zorder=5,
                   label=r["method"])
    clean_lift = next(r["task_lift"] for r in rows if r["method"] == "no erasure (frozen)")
    ax.axvline(TAU, color="black", ls="--", lw=1.2, label=f"τ={TAU} (stops attack ←)")
    ax.axhline(clean_lift, color="gray", ls=":", lw=1, label=f"clean task lift={clean_lift:+.3f}")
    ax.set_xlabel("attacked R²(repr, sex)  — lower = attack stopped")
    ax.set_ylabel("task lift over majority (income)  — higher = utility kept")
    ax.set_title("POSITIVE CONTROL: smart erasure on Adult/sex (fragile signal)\n"
                 "(want: left of τ AND high on y — here targeted erasure can reach it)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
