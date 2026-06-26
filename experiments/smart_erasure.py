"""Experiment 4: can a SMARTER erasure beat blunt noise on the hard HMDA/race cell?

Experiment 3 found that on HMDA/underwriting/race the protected attribute is
robustly, NONLINEARLY encoded: the ReLU attacker recovers R²=0.72 while the
LINEAR certificate reads only 0.040 (it massively understates the leakage), and
the only noise level that stops the attack (sigma=8) destroys all task utility
(lift +0.019 -> +0.000). HMDA/race is the hard case.

Question: can a targeted erasure remove the robust nonlinear race signal WITHOUT
the utility collapse that isotropic Gaussian noise caused? The crux is whether
the race signal lives in a low-rank subspace SEPARABLE from the underwriting task
(a targeted projection can win) or is entangled with it (the cost is fundamental).

Erasure modules (each a FROZEN transform applied to the 64-dim repr BEFORE the
same ReLU attacker), all task-agnostic for a fair comparison to noise:

 1. LEOPARD-style: a learned orthogonal rank-r projection that REMOVES the r-dim
    subspace minimizing the pairwise class-conditional MMD of race (Gaussian
    kernel) in the projected space — matches higher moments, not just the linear
    mean LEACE removes. Sweep r.
 2. HSIC-style (Obliviator): the same learned rank-r projection, but the removed
    subspace minimizes HSIC(projected repr, race) (statistical-independence
    target). Sweep r.

Baselines: no erasure (the frozen, linear-compliant repr), explicit LEACE (linear
mean removal), and the Gaussian noise channel at sigma=8 (stops the attack but
kills utility). The same FreshLoRAAttacker + PCRL certificate + run_attack /
task_accuracy from Experiments 2-3 are reused verbatim. Local MPS/CPU; never AWS.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.noise_channel_test import (  # noqa: E402
    BATCH_SIZE, SEEDS, TAU, run_attack, task_accuracy,
)
from utils.pcrl_io import (  # noqa: E402
    build_train_loader, extract_attr, extract_frozen_reps, extract_task_label,
    get_device, load_encoder,
)

RESULTS_DIR = REPO_ROOT / "results"
PURPOSE_IDX = 0          # underwriting
ATTR = "race"
TASK = "loan_decision"
RANKS = [1, 2, 4, 8, 16, 32]
FIT_STEPS = 500
FIT_BS_PER_CLASS = 256
NOISE_SIGMA_REL = 8.0     # the Exp-3 level that stopped the HMDA attack


# ---- kernels ---------------------------------------------------------------
def _sq_dists(a, b):
    return (a * a).sum(1, keepdim=True) - 2 * a @ b.T + (b * b).sum(1)[None, :]


def _median_gamma(H):
    with torch.no_grad():
        d2 = _sq_dists(H, H)
        med = torch.median(d2[d2 > 0])
    return 1.0 / (med + 1e-8)


def _gauss(a, b, gamma):
    return torch.exp(-gamma * _sq_dists(a, b).clamp_min(0.0))


def _mmd2(X, Y, gamma):
    return _gauss(X, X, gamma).mean() + _gauss(Y, Y, gamma).mean() - 2 * _gauss(X, Y, gamma).mean()


def mmd_class_loss(Hproj, y, classes):
    """Sum of pairwise class-conditional MMD² (Gaussian kernel)."""
    gamma = _median_gamma(Hproj)
    groups = [Hproj[y == k] for k in classes]
    loss, pairs = Hproj.new_zeros(()), 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if groups[i].shape[0] >= 2 and groups[j].shape[0] >= 2:
                loss = loss + _mmd2(groups[i], groups[j], gamma)
                pairs += 1
    return loss / max(pairs, 1)


def hsic_loss(Hproj, y_onehot):
    """Empirical HSIC(repr, race): Gaussian kernel on repr, linear on one-hot."""
    n = Hproj.shape[0]
    gamma = _median_gamma(Hproj)
    K = _gauss(Hproj, Hproj, gamma)
    L = y_onehot @ y_onehot.T
    Hc = torch.eye(n, device=Hproj.device) - 1.0 / n
    return (Hc @ K @ Hc * (Hc @ L @ Hc)).sum() / ((n - 1) ** 2)


# ---- learned rank-r projection --------------------------------------------
def fit_projection(H_t, y_np, rank, loss_kind, device, seed=0):
    """Learn an orthonormal d×rank basis Q whose REMOVAL minimizes the erasure
    loss (MMD class-matching or HSIC independence). Returns frozen Q."""
    torch.manual_seed(seed)
    d = H_t.shape[1]
    classes = sorted(np.unique(y_np).tolist())
    by_class = {k: np.where(y_np == k)[0] for k in classes}
    n_classes = int(max(classes)) + 1

    M = torch.nn.Parameter(torch.randn(d, rank, device=device) * 0.1)
    opt = torch.optim.Adam([M], lr=1e-2)
    rng = np.random.default_rng(seed)

    for _ in range(FIT_STEPS):
        # Balanced minibatch: up to FIT_BS_PER_CLASS per race class.
        idx = np.concatenate([
            rng.choice(by_class[k], size=min(FIT_BS_PER_CLASS, len(by_class[k])),
                       replace=False)
            for k in classes])
        hb = H_t[torch.as_tensor(idx, device=device)]
        yb = torch.as_tensor(y_np[idx], device=device)
        Q, _ = torch.linalg.qr(M)                  # orthonormal columns
        hb_proj = hb - (hb @ Q) @ Q.T
        if loss_kind == "mmd":
            loss = mmd_class_loss(hb_proj, yb, classes)
        else:
            yoh = torch.nn.functional.one_hot(yb, n_classes).float()
            loss = hsic_loss(hb_proj, yoh)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        Q, _ = torch.linalg.qr(M)
    return Q.detach()


def apply_projection(H, Q):
    """h' = h - (h@Q)@Qᵀ — remove span(Q). H np, Q tensor; returns np."""
    Ht = torch.as_tensor(H, device=Q.device)
    return (Ht - (Ht @ Q) @ Q.T).cpu().numpy().astype(np.float32)


# ---- evaluation ------------------------------------------------------------
def evaluate(H_erased, race_np, task_np, task_majority, device):
    """Attacked R² (does it stop the attack?) + task lift over majority."""
    Ht = torch.from_numpy(H_erased).to(device)
    race_t = torch.from_numpy(race_np).long().to(device)
    attacks = [run_attack(Ht, race_t, race_np, device, s, 0.0, rank=8, alpha=16.0)
               for s in SEEDS]
    r2 = [a["final_linear_r2"] for a in attacks]
    lift = [task_accuracy(H_erased, task_np, 0.0, s) - task_majority for s in SEEDS]
    return {"attacked_r2": float(np.mean(r2)), "attacked_r2_std": float(np.std(r2)),
            "task_lift": float(np.mean(lift))}


def evaluate_noise(H, race_np, task_np, task_majority, sigma_abs, device):
    """Noise channel baseline: noise applied inside the attack / utility eval."""
    Ht = torch.from_numpy(H).to(device)
    race_t = torch.from_numpy(race_np).long().to(device)
    attacks = [run_attack(Ht, race_t, race_np, device, s, sigma_abs, rank=8, alpha=16.0)
               for s in SEEDS]
    r2 = [a["final_linear_r2"] for a in attacks]
    lift = [task_accuracy(H, task_np, sigma_abs, s) - task_majority for s in SEEDS]
    return {"attacked_r2": float(np.mean(r2)), "attacked_r2_std": float(np.std(r2)),
            "task_lift": float(np.mean(lift))}


def main() -> None:
    device = get_device()
    print(f"[setup] device={device}")
    purposes, train_ds, loader = build_train_loader("hmda", batch_size=BATCH_SIZE)
    encoder, lora_target = load_encoder("hmda", train_ds.info.num_features, len(purposes))
    H = extract_frozen_reps(encoder, loader, PURPOSE_IDX, device)
    race = extract_attr(loader, ATTR)
    task = extract_task_label(loader, TASK)
    sigma_h = float(H.std(0).mean())
    task_majority = float(np.bincount(task).max() / len(task))
    print(f"[setup] {purposes[PURPOSE_IDX].name}/{ATTR}: N={H.shape[0]} sigma_h={sigma_h:.4f} "
          f"race({int(race.max())+1} cls) task '{TASK}' majority={task_majority:.4f}")

    H_t = torch.from_numpy(H).to(device)
    rows = []

    def add(method, ev, rank=None):
        stops = ev["attacked_r2"] <= TAU
        rows.append({"method": method, "rank": rank, **ev,
                     "stops_attack": bool(stops)})
        rk = f" r={rank}" if rank is not None else ""
        print(f"  {method+rk:<26} attacked R²={ev['attacked_r2']:.4f}"
              f"±{ev['attacked_r2_std']:.4f}  task_lift={ev['task_lift']:+.4f}  "
              f"{'STOPS' if stops else 'breaches'}")

    # Baselines.
    print("\n--- baselines ---")
    add("no erasure (frozen)", evaluate(H.copy(), race, task, task_majority, device))

    from concept_erasure import LeaceEraser
    zoh = torch.nn.functional.one_hot(torch.from_numpy(race).long(),
                                      int(race.max()) + 1).float()
    eraser = LeaceEraser.fit(H_t.float().cpu(), zoh)
    H_leace = eraser(H_t.float().cpu()).numpy().astype(np.float32)
    add("LEACE (linear)", evaluate(H_leace, race, task, task_majority, device))

    add(f"noise sigma={NOISE_SIGMA_REL}",
        evaluate_noise(H, race, task, task_majority, NOISE_SIGMA_REL * sigma_h, device))

    # Smart erasures, swept over rank.
    for kind, label in [("mmd", "LEOPARD-MMD proj"), ("hsic", "HSIC proj")]:
        print(f"\n--- {label} (sweep rank) ---")
        for r in RANKS:
            Q = fit_projection(H_t, race, r, kind, device, seed=0)
            H_er = apply_projection(H, Q)
            add(label, evaluate(H_er, race, task, task_majority, device), rank=r)

    _print_table(rows, task_majority)
    verdict = _verdict(rows)

    out = {"experiment": "smart_erasure", "cell": f"hmda/{purposes[PURPOSE_IDX].name}/{ATTR}",
           "task": TASK, "tau": TAU, "sigma_h": sigma_h, "task_majority": task_majority,
           "seeds": SEEDS, "ranks": RANKS, "rows": rows, "verdict": verdict}
    (RESULTS_DIR / "smart_erasure.json").write_text(json.dumps(out, indent=2))
    _plot(rows, RESULTS_DIR / "smart_erasure.png")
    print(f"\n[save] {RESULTS_DIR/'smart_erasure.json'} and .png")


def _print_table(rows, task_majority) -> None:
    print("\n===== SMART ERASURE on HMDA/underwriting/race =====")
    print(f"{'method':<26} {'attacked R²':>12} {'task lift':>10} {'verdict':>22}")
    print("-" * 74)
    clean_lift = next(r["task_lift"] for r in rows if r["method"] == "no erasure (frozen)")
    for r in rows:
        name = r["method"] + (f" r={r['rank']}" if r["rank"] is not None else "")
        if r["stops_attack"]:
            kept = r["task_lift"] / clean_lift if clean_lift > 0 else 0.0
            v = f"STOPS, {kept*100:.0f}% lift kept"
        else:
            v = "breaches τ"
        print(f"{name:<26} {r['attacked_r2']:>12.4f} {r['task_lift']:>+10.4f} {v:>22}")
    print("-" * 74)
    print(f"τ={TAU}  clean task lift over majority={clean_lift:+.4f} "
          f"(majority={task_majority:.4f})")


def _verdict(rows) -> str:
    clean_lift = next(r["task_lift"] for r in rows if r["method"] == "no erasure (frozen)")
    # Best stopper among the SMART erasures (max retained lift among those that stop).
    smart = [r for r in rows if r["method"].startswith(("LEOPARD", "HSIC")) and r["stops_attack"]]
    noise = next(r for r in rows if r["method"].startswith("noise"))
    print("\n================== EXPERIMENT 4 VERDICT ==================")
    if not smart:
        v = (f"No smart erasure (LEOPARD-MMD or HSIC, any rank) stops the attack "
             f"(R²≤τ). The robust nonlinear race signal is NOT removable by a "
             f"low-rank linear projection; only isotropic noise stops it, and that "
             f"costs all utility (noise σ8 lift {noise['task_lift']:+.4f}). The cost "
             f"appears FUNDAMENTAL on this cell.")
        print(v)
        return v
    best = max(smart, key=lambda r: r["task_lift"])
    kept = best["task_lift"] / clean_lift if clean_lift > 0 else 0.0
    beats_noise = best["task_lift"] > noise["task_lift"] + 0.005
    if kept >= 0.5 and beats_noise:
        v = (f"YES: {best['method']} r={best['rank']} stops the attack "
             f"(R²={best['attacked_r2']:.4f}≤τ) while keeping {kept*100:.0f}% of task "
             f"lift ({best['task_lift']:+.4f} vs noise σ8 {noise['task_lift']:+.4f}). "
             f"A targeted erasure beats blunt noise — the race signal is low-rank "
             f"separable from the task.")
    elif beats_noise:
        v = (f"PARTIAL: best smart erasure ({best['method']} r={best['rank']}) stops the "
             f"attack and keeps only {kept*100:.0f}% of task lift "
             f"({best['task_lift']:+.4f}), better than noise σ8 "
             f"({noise['task_lift']:+.4f}) but still a large utility hit.")
    else:
        v = (f"NO: even the best smart erasure that stops the attack "
             f"({best['method']} r={best['rank']}, lift {best['task_lift']:+.4f}) does "
             f"NOT beat blunt noise σ8 ({noise['task_lift']:+.4f}). Cost appears "
             f"FUNDAMENTAL — race and the underwriting task are entangled.")
    print(v)
    return v


def _plot(rows, png_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    styles = {"LEOPARD-MMD proj": ("o", "C0"), "HSIC proj": ("s", "C1")}
    for label, (mk, c) in styles.items():
        pts = [(r["attacked_r2"], r["task_lift"], r["rank"]) for r in rows if r["method"] == label]
        xs, ys, rks = zip(*pts)
        ax.plot(xs, ys, mk + "-", color=c, label=label, ms=6)
        for x, y, rk in pts:
            ax.annotate(f"r{rk}", (x, y), fontsize=6, xytext=(2, 2),
                        textcoords="offset points")
    for r in rows:
        if r["method"].startswith(("LEOPARD", "HSIC")):
            continue
        mk = {"no erasure (frozen)": ("*", "C2"), "LEACE (linear)": ("D", "C4")}
        m, c = mk.get(r["method"], ("P", "C3"))
        ax.scatter(r["attacked_r2"], r["task_lift"], marker=m, s=120, color=c,
                   zorder=5, label=r["method"])
    clean_lift = next(r["task_lift"] for r in rows if r["method"] == "no erasure (frozen)")
    ax.axvline(TAU, color="black", ls="--", lw=1.2, label=f"τ={TAU} (stops attack ←)")
    ax.axhline(clean_lift, color="gray", ls=":", lw=1, label=f"clean task lift={clean_lift:+.3f}")
    ax.axhline(0.0, color="gray", lw=0.6)
    ax.set_xlabel("attacked R²(repr, race)  — lower = attack stopped")
    ax.set_ylabel("task lift over majority (underwriting)  — higher = utility kept")
    ax.set_title("Smart erasure vs blunt noise on HMDA/race\n"
                 "(want: bottom-left of τ AND high on y — stop attack, keep utility)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
