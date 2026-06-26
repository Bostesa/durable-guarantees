"""Experiment 4 — STAGE 2 (decisive): training-time two-component erasure on HMDA/race.

Stage 1 established that NO frozen post-hoc erasure (LEACE, Gaussian noise,
LEOPARD-MMD / HSIC projections) makes HMDA/underwriting/race durable while keeping
utility — the robust, nonlinearly-diffused, task-entangled race signal is not
removable after the representation is fixed.

This stage tests the remaining lever: a TRAINING-TIME intervention. We build a
two-component model M = D(E(·)) trained JOINTLY with task loss + an erasure
constraint at E's output, so E is shaped to be task-compatible instead of bolted
onto a frozen representation. After training we FREEZE E and run the same ReLU
attacker on E's output to recover race.

To make the test decisive (and not rigged to fail by the affine-invariance of R²
established in Experiment 1), E is run at three power levels, all jointly trained
with task CE + λ·HSIC(E(·), race):
  1. affine on the frozen PCRL repr h (LEACE-initialized)  — the spec'd primary,
  2. MLP on h (nonlinear, same frozen rep),
  3. MLP on the raw HMDA features x (a from-scratch durable encoder) — the
     strongest shot training-time has.
λ sweeps the durability-utility trade-off. The same FreshLoRAAttacker, certificate,
run_attack/task_accuracy and HSIC code from Stages 1 / Exp 2-3 are reused. The
post-hoc baselines are pulled from results/smart_erasure.json for a single table.
Local MPS/CPU; never AWS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.noise_channel_test import BATCH_SIZE, SEEDS, TAU  # noqa: E402
from experiments.smart_erasure import evaluate, hsic_loss  # noqa: E402  (reuse attacker eval + HSIC)
from utils.pcrl_io import (  # noqa: E402
    build_train_loader, extract_attr, extract_frozen_reps, extract_task_label,
    get_device, load_encoder,
)

RESULTS_DIR = REPO_ROOT / "results"
PURPOSE_IDX = 0          # underwriting
ATTR = "race"
TASK = "loan_decision"
LAMBDAS = [0.0, 100.0, 1000.0, 10000.0]
TRAIN_STEPS = 800
TRAIN_BS = 1024


def _leace_affine_init(linear, h_cpu, race_np, n_race):
    """Initialise a 64->64 nn.Linear to exactly reproduce the LEACE eraser of race."""
    from concept_erasure import LeaceEraser
    zoh = torch.nn.functional.one_hot(torch.from_numpy(race_np).long(), n_race).float()
    er = LeaceEraser.fit(h_cpu, zoh)
    with torch.no_grad():
        c = er(torch.zeros(1, 64)).squeeze(0)               # bias
        W = er(torch.eye(64)) - c                            # out = x @ W + c
        linear.weight.copy_(W.T)
        linear.bias.copy_(c)


def build_E(arch, d_in, h_cpu, race_np, n_race):
    if arch == "affine_h":
        E = nn.Linear(d_in, 64)
        _leace_affine_init(E, h_cpu, race_np, n_race)
    elif arch in ("mlp_h", "mlp_x"):
        E = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 64))
    else:
        raise ValueError(arch)
    return E


def train_two_component(inp_t, race_np, task_np, n_race, n_task, arch, lam,
                        h_cpu, device, steps=TRAIN_STEPS, seed=0):
    """Train M=D(E(inp)) with task CE + λ·HSIC(E(inp), race). Returns frozen E(inp) (np)."""
    torch.manual_seed(seed)
    E = build_E(arch, inp_t.shape[1], h_cpu, race_np, n_race).to(device)
    D = nn.Linear(64, n_task).to(device)
    opt = torch.optim.Adam(list(E.parameters()) + list(D.parameters()), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    race_t = torch.from_numpy(race_np).long().to(device)
    task_t = torch.from_numpy(task_np).long().to(device)
    n = inp_t.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    E.train(); D.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (TRAIN_BS,), generator=g).to(device)
        z = E(inp_t[idx])
        loss = ce(D(z), task_t[idx])
        if lam > 0:
            roh = torch.nn.functional.one_hot(race_t[idx], n_race).float()
            loss = loss + lam * hsic_loss(z, roh)
        opt.zero_grad(); loss.backward(); opt.step()

    E.eval()
    with torch.no_grad():
        out = []
        for i in range(0, n, 4096):
            out.append(E(inp_t[i:i + 4096]).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    device = get_device()
    print(f"[setup] device={device}")
    purposes, train_ds, loader = build_train_loader("hmda", batch_size=BATCH_SIZE)
    encoder, _ = load_encoder("hmda", train_ds.info.num_features, len(purposes))
    H = extract_frozen_reps(encoder, loader, PURPOSE_IDX, device)
    X = np.concatenate([b["features"].numpy() for b in loader]).astype(np.float32)  # raw features
    race = extract_attr(loader, ATTR)
    task = extract_task_label(loader, TASK)
    n_race = int(race.max()) + 1
    n_task = int(task.max()) + 1
    task_majority = float(np.bincount(task).max() / len(task))
    print(f"[setup] HMDA/{purposes[PURPOSE_IDX].name}/{ATTR}: N={H.shape[0]} "
          f"x_dim={X.shape[1]} race={n_race}cls task '{TASK}' majority={task_majority:.4f}")

    H_t = torch.from_numpy(H).to(device)
    X_t = torch.from_numpy(X).to(device)
    h_cpu = H_t.float().cpu()
    clean_lift = None

    rows = []

    def add(method, ev, extra=None):
        rows.append({"method": method, **ev, "stops_attack": bool(ev["attacked_r2"] <= TAU),
                     **(extra or {})})
        print(f"  {method:<30} attacked R²={ev['attacked_r2']:.4f}±{ev['attacked_r2_std']:.4f}"
              f"  task_lift={ev['task_lift']:+.4f}  "
              f"{'STOPS' if ev['attacked_r2'] <= TAU else 'breaches'}")

    # Training-time sweep over architectures × λ.
    ARCHS = [("affine_h", H_t, "affine on h (LEACE-init)"),
             ("mlp_h",    H_t, "MLP on h"),
             ("mlp_x",    X_t, "MLP on raw x (from scratch)")]
    for arch, inp_t, label in ARCHS:
        print(f"\n--- training-time: {label} ---")
        for lam in LAMBDAS:
            E_out = train_two_component(inp_t, race, task, n_race, n_task, arch, lam,
                                        h_cpu, device)
            ev = evaluate(E_out, race, task, task_majority, device)
            add(f"{label} λ={lam:g}", ev, extra={"arch": arch, "lambda": lam})

    _print_table(rows, task_majority, clean_lift)
    verdict = _verdict(rows, task_majority)

    out = {"experiment": "training_time_erasure",
           "cell": f"hmda/{purposes[PURPOSE_IDX].name}/{ATTR}", "task": TASK, "tau": TAU,
           "task_majority": task_majority, "seeds": SEEDS, "lambdas": LAMBDAS,
           "rows": rows, "verdict": verdict}
    (RESULTS_DIR / "training_time_erasure.json").write_text(json.dumps(out, indent=2))
    _plot(rows, task_majority, RESULTS_DIR / "training_time_erasure.png")
    print(f"\n[save] {RESULTS_DIR/'training_time_erasure.json'} and .png")


def _posthoc_baselines():
    """Pull the post-hoc HMDA/race rows from Stage-1 results for the combined table."""
    p = RESULTS_DIR / "smart_erasure.json"
    if not p.exists():
        return []
    sj = json.loads(p.read_text())
    keep = {}
    for r in sj["rows"]:
        m = r["method"]
        if m in ("no erasure (frozen)", "LEACE (linear)") or m.startswith("noise"):
            keep[m] = r
        elif m.startswith(("LEOPARD", "HSIC")):
            # keep the strongest (lowest attacked R²) of each family
            fam = m.split()[0]
            if fam not in keep or r["attacked_r2"] < keep[fam]["attacked_r2"]:
                keep[fam] = {**r, "method": f"{fam} proj (best post-hoc)"}
    order = ["no erasure (frozen)", "LEACE (linear)", "noise sigma=8.0", "LEOPARD", "HSIC"]
    return [keep[k] for k in order if k in keep]


def _print_table(rows, task_majority, _):
    base = _posthoc_baselines()
    clean = next((r for r in base if r["method"] == "no erasure (frozen)"), None)
    clean_lift = clean["task_lift"] if clean else 0.019
    print("\n===== TRAINING-TIME vs POST-HOC erasure on HMDA/underwriting/race =====")
    print(f"{'method':<34} {'attacked R²':>12} {'task lift':>10} {'stops & keeps?':>16}")
    print("-" * 78)
    for r in base + [{"_sep": True}] + rows:
        if r.get("_sep"):
            print("-" * 78); continue
        stops = r["attacked_r2"] <= TAU
        kept = r["task_lift"] / clean_lift if clean_lift > 0 else 0.0
        tag = (f"YES ({kept*100:.0f}% lift)" if stops and r["task_lift"] > 0.4 * clean_lift
               else ("stops, util~0" if stops else "breaches τ"))
        print(f"{r['method']:<34} {r['attacked_r2']:>12.4f} {r['task_lift']:>+10.4f} {tag:>16}")
    print("-" * 78)
    print(f"τ={TAU}  clean task lift over majority={clean_lift:+.4f} (majority={task_majority:.4f})")


def _verdict(rows, task_majority) -> str:
    base = _posthoc_baselines()
    clean = next((r for r in base if r["method"] == "no erasure (frozen)"), None)
    clean_lift = clean["task_lift"] if clean else 0.019
    # A win = stops the attack (R²≤τ) AND keeps a meaningful share of task lift.
    wins = [r for r in rows if r["attacked_r2"] <= TAU and r["task_lift"] >= 0.4 * clean_lift]
    stoppers = [r for r in rows if r["attacked_r2"] <= TAU]
    print("\n================== STAGE 2 VERDICT ==================")
    if wins:
        b = max(wins, key=lambda r: r["task_lift"])
        v = (f"YES — training-time intervention achieves durability where every post-hoc "
             f"method failed: {b['method']} stops the attack (R²={b['attacked_r2']:.4f}≤τ) "
             f"AND keeps {b['task_lift']/clean_lift*100:.0f}% of task lift "
             f"({b['task_lift']:+.4f} vs clean {clean_lift:+.4f}). Shaping E during training "
             f"to be task-compatible — instead of bolting it onto a frozen rep — breaks the "
             f"post-hoc impossibility on HMDA/race.")
    elif stoppers:
        b = max(stoppers, key=lambda r: r["task_lift"])
        v = (f"NO (partial) — training-time CAN drive attacked R² below τ "
             f"({b['method']}, R²={b['attacked_r2']:.4f}) but only by destroying task utility "
             f"(best stopper keeps lift {b['task_lift']:+.4f} = {b['task_lift']/clean_lift*100:.0f}% "
             f"of clean {clean_lift:+.4f}). Like noise, it trades all utility for durability — "
             f"the cost is FUNDAMENTAL even at training time on HMDA/race.")
    else:
        best = min(rows, key=lambda r: r["attacked_r2"])
        v = (f"NO — no training-time config even reaches τ (best attacked R²="
             f"{best['attacked_r2']:.4f} via {best['method']}). Joint task+HSIC training does "
             f"not make race unrecoverable to the ReLU attacker; the robust, task-entangled "
             f"race signal survives. Cost is FUNDAMENTAL even at training time.")
    print(v)
    return v


def _plot(rows, task_majority, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = _posthoc_baselines()
    clean_lift = next((r["task_lift"] for r in base if r["method"] == "no erasure (frozen)"), 0.019)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    arch_style = {"affine on h (LEACE-init)": ("o", "C0"), "MLP on h": ("s", "C1"),
                  "MLP on raw x (from scratch)": ("^", "C3")}
    for label, (mk, c) in arch_style.items():
        pts = sorted([(r["attacked_r2"], r["task_lift"], r["lambda"])
                      for r in rows if r["method"].startswith(label)], key=lambda t: t[2])
        if not pts:
            continue
        xs, ys, lams = zip(*pts)
        ax.plot(xs, ys, mk + "-", color=c, ms=6, label=f"train-time: {label}")
        for x, y, lm in pts:
            ax.annotate(f"λ{lm:g}", (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
    for r in base:
        m, c = {"no erasure (frozen)": ("*", "C2"), "LEACE (linear)": ("D", "C4"),
                "noise sigma=8.0": ("P", "C5")}.get(r["method"], ("x", "C6"))
        ax.scatter(r["attacked_r2"], r["task_lift"], marker=m, s=120, color=c, zorder=5,
                   label=f"post-hoc: {r['method']}")
    ax.axvline(TAU, color="black", ls="--", lw=1.2, label=f"τ={TAU} (stops ←)")
    ax.axhline(clean_lift, color="gray", ls=":", lw=1, label=f"clean task lift={clean_lift:+.3f}")
    ax.axhline(0.4 * clean_lift, color="green", ls=":", lw=0.8,
               label="40% lift (keep-utility bar)")
    ax.set_xlabel("attacked R²(repr, race)  — lower = attack stopped")
    ax.set_ylabel("task lift over majority (underwriting)  — higher = utility kept")
    ax.set_title("Training-time vs post-hoc erasure on HMDA/race\n"
                 "(want a point LEFT of τ AND ABOVE the green bar)")
    ax.legend(fontsize=6.5, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
