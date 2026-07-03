"""Experiment 14 (Part B of the final block) — re-certify the key operating points
at BOTH battery tiers.

Experiment 11 showed the standing battery (XGBoost + deep MLP + rank-32 LoRA) is
not the ceiling: a channel-aware Gaussian-LRT attacker with clean-representation
side information sits 0.02–0.05 above it at operating sigma. utils/battery.py
productionizes that attacker and defines the two certification tiers:

  TIER 1 (black-box)  XGB + MLP + LoRA-32          (the standing battery)
  TIER 2 (informed)   Tier 1 + channel-aware Gaussian-LRT

This experiment finds, for each headline cell, TWO operating points — the
config where Tier-1 recovery <= 0.55 (should match the published results) and
the config where Tier-2 <= 0.55 — and reports utility kept at both, side by
side. The two-tier table is a headline artifact of the paper.

Sections:
  1. E2E blunt noised channels (the Exp-8 constructive mechanism), 5 training
     seeds + error bars at each operating point:
       - hmda/race/loan_decision   (easy;   published Tier-1 sigma=8, 96% utility)
       - hmda/race/loan_amount_band (middle; published Tier-1 sigma=16, 39%)
       - adult/sex/income          (hard;   published Tier-1 sigma~25-32)
     The LRT here knows sigma: Sigma_noise = sigma^2 I.
  2. Post-hoc surgical (HSIC-subspace) winners of Exp-9/10 on the frozen clean
     trained P: Tier-1 re-check of the winner config, then a Tier-2 search over
     wider/stronger surgical configs (and blunt iso as fallback). The LRT knows
     the channel: Sigma_noise = sigma^2 Q Q^T (zero noise in the complement — an
     informed adversary sees the surviving directions at full strength).
  3. Exp-10 e2e surgical winners: Tier-2 status of the published winner configs
     (retrained, seed 0), with a bounded sigma-escalation search if they breach.

Everything measured with held-out probes; never R^2. Local MPS/CPU; never AWS.
macOS OpenMP guard: xgboost before torch.
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

from experiments.diagnostic import (  # noqa: E402
    lift, load_cell, probe_lora, probe_sklearn,
)
from experiments.end_to_end_surgical import (  # noqa: E402
    lora_channel, train_e2e_surgical,
)
from experiments.mi_ceiling import train_channel  # noqa: E402  (e2e blunt, returns pre-noise h)
from experiments.sealed_channel import train_model  # noqa: E402
from experiments.smart_erasure import fit_projection  # noqa: E402
from experiments.targeted_noise import (  # noqa: E402
    iso_channel, probe_through, subspace_channel, utility_through,
)
from utils.battery import (  # noqa: E402
    gaussian_lrt, gaussian_lrt_channel, iso_noise_cov, subspace_noise_cov,
)
from utils.pcrl_io import get_device  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
BAR = 0.55
SWEEP_SEEDS = [0, 1]            # probe seeds inside the sigma sweep
CERT_SEEDS = [0, 1, 2]          # probe seeds for certification batteries
TRAIN_SEEDS = [0, 1, 2, 3, 4]   # 5 training seeds at each certified operating point

E2E_CELLS = [
    dict(name="hmda/race/loan_decision", dataset="hmda", attr="race",
         task="loan_decision", grid=[6.0, 8.0, 12.0, 16.0, 20.0, 24.0]),
    dict(name="hmda/race/loan_amount_band", dataset="hmda", attr="race",
         task="loan_amount_band", grid=[12.0, 16.0, 24.0, 32.0, 48.0]),
    dict(name="adult/sex/income", dataset="adult", attr="sex",
         task="income", grid=[16.0, 24.0, 32.0, 48.0, 64.0]),
]

# Exp-9/10 post-hoc surgical winners on the frozen clean trained P
# (winner (rank, sigma_rel) from results/end_to_end_surgical.json arm B).
POSTHOC_CELLS = [
    dict(name="easy hmda/race/loan_decision", dataset="hmda", attr="race",
         task="loan_decision", winner=(16, 16.0)),
    dict(name="middle hmda/race/loan_amount_band", dataset="hmda", attr="race",
         task="loan_amount_band", winner=(8, 16.0)),
    dict(name="hard adult/sex/income", dataset="adult", attr="sex",
         task="income", winner=(32, 16.0)),
]
# Tier-2 search grid for post-hoc: wider/noisier surgical configs, then blunt iso.
POSTHOC_T2_SUR = [(None, 32.0), (None, 64.0), (48, 16.0), (48, 32.0),
                  (64, 16.0), (64, 32.0), (64, 64.0)]   # None = winner rank
POSTHOC_T2_ISO = [16.0, 32.0, 64.0]

# Exp-10 e2e surgical winners (from results/end_to_end_surgical.json arm D).
E2E_SUR_WINNERS = [
    dict(name="easy hmda/race/loan_decision", dataset="hmda", attr="race",
         task="loan_decision", rank=32, sigma=16.0, lam=10.0),
    dict(name="middle hmda/race/loan_amount_band", dataset="hmda", attr="race",
         task="loan_amount_band", rank=16, sigma=16.0, lam=100.0),
    dict(name="hard adult/sex/income", dataset="adult", attr="sex",
         task="income", rank=32, sigma=8.0, lam=100.0),
]
MAX_ESCALATIONS = 3   # bounded sigma-doubling search for section 3


# --------------------------------------------------------------------------- #
#  Shared measurement                                                          #
# --------------------------------------------------------------------------- #
def full_point(h, P, y, n_attr, noise_cov, device, seeds, with_lora=True):
    """XGB + MLP (+ LoRA32) + LRT on a fixed exposed matrix. Returns dict of AUCs."""
    out = {}
    out["xgb"], _ = probe_sklearn("XGB", P, y, n_attr, seeds)
    out["mlp"], _ = probe_sklearn("MLP", P, y, n_attr, seeds)
    if with_lora:
        out["lora"], _ = probe_lora(P, y, n_attr, seeds, device)
    out["lrt"], _ = gaussian_lrt(h, P, y, noise_cov, n_attr, seeds)
    t1 = [out["xgb"], out["mlp"]] + ([out["lora"]] if with_lora else [])
    out["tier1_max"] = float(max(t1))
    out["tier2_max"] = float(max(out["tier1_max"], out["lrt"]))
    return out


def fmt_point(pt):
    lora = f" LoRA={pt['lora']:.3f}" if "lora" in pt else ""
    return (f"XGB={pt['xgb']:.3f} MLP={pt['mlp']:.3f}{lora} LRT={pt['lrt']:.3f} "
            f"| T1={pt['tier1_max']:.3f} T2={pt['tier2_max']:.3f}")


# --------------------------------------------------------------------------- #
#  Section 1 — e2e blunt channels, both tiers, 5 seeds                         #
# --------------------------------------------------------------------------- #
def certify_e2e(cell, sigma, X_t, attr, task, n_attr, n_task, task_maj, device):
    """5-training-seed certification at one sigma: full battery incl. LRT + lift."""
    per = {k: [] for k in ("xgb", "mlp", "lora", "lrt", "lift")}
    d = None
    for ts in TRAIN_SEEDS:
        h, P, L = train_channel(X_t, attr, task, n_attr, n_task, sigma, device, seed=ts)
        d = h.shape[1]
        pt = full_point(h, P, attr, n_attr, iso_noise_cov(d, sigma), device, CERT_SEEDS)
        for k in ("xgb", "mlp", "lora", "lrt"):
            per[k].append(pt[k])
        per["lift"].append(lift(L, task, task_maj))
        print(f"    [seed {ts}] {fmt_point(pt)} lift={per['lift'][-1]:+.4f}", flush=True)
    out = {f"{k}_mean": float(np.mean(v)) for k, v in per.items()}
    out.update({f"{k}_std": float(np.std(v)) for k, v in per.items()})
    out["per_seed"] = per
    out["tier1_max_mean"] = float(max(out["xgb_mean"], out["mlp_mean"], out["lora_mean"]))
    out["tier2_max_mean"] = float(max(out["tier1_max_mean"], out["lrt_mean"]))
    out["sigma"] = sigma
    return out


def run_e2e_cell(cell, device):
    print("\n" + "#" * 96 + f"\n# E2E BLUNT: {cell['name']}\n" + "#" * 96, flush=True)
    t0 = time.time()
    X, attr, task, n_attr, n_task, _, task_maj = load_cell(
        cell["dataset"], cell["attr"], cell["task"])
    X_t = torch.from_numpy(X).to(device)

    # clean lift baseline over the same 5 training seeds
    clean_lifts = []
    for ts in TRAIN_SEEDS:
        _, _, L0 = train_channel(X_t, attr, task, n_attr, n_task, 0.0, device, seed=ts)
        clean_lifts.append(lift(L0, task, task_maj))
    clean_lift = float(np.mean(clean_lifts))
    print(f"[clean] lift={clean_lift:+.4f}±{np.std(clean_lifts):.4f} over 5 seeds",
          flush=True)

    # sigma sweep (train seed 0): XGB+MLP+LRT everywhere; LoRA on tier candidates
    sweep = []
    sigma_t1 = sigma_t2 = None
    for sg in cell["grid"]:
        h, P, L = train_channel(X_t, attr, task, n_attr, n_task, sg, device, seed=0)
        pt = full_point(h, P, attr, n_attr, iso_noise_cov(h.shape[1], sg), device,
                        SWEEP_SEEDS, with_lora=False)
        pt["sigma"] = sg
        pt["lift"] = lift(L, task, task_maj)
        need_lora = max(pt["xgb"], pt["mlp"]) <= BAR
        if need_lora:
            pt["lora"], _ = probe_lora(P, attr, n_attr, SWEEP_SEEDS, device)
            pt["tier1_max"] = float(max(pt["xgb"], pt["mlp"], pt["lora"]))
            pt["tier2_max"] = float(max(pt["tier1_max"], pt["lrt"]))
        sweep.append(pt)
        print(f"  σ={sg:>5.1f} {fmt_point(pt)} lift={pt['lift']:+.4f}", flush=True)
        if sigma_t1 is None and "lora" in pt and pt["tier1_max"] <= BAR:
            sigma_t1 = sg
        if sigma_t2 is None and "lora" in pt and pt["tier2_max"] <= BAR:
            sigma_t2 = sg
            break   # both points found (tier2 implies tier1 here)
    print(f"  → sweep operating points: tier1 σ={sigma_t1}, tier2 σ={sigma_t2}",
          flush=True)

    # 5-seed certification at each operating point (escalate along grid if the
    # 5-seed mean breaches — the sweep was one training seed)
    def certify_at(sigma0, tier_key):
        grid = [s for s in cell["grid"] if s >= (sigma0 or cell["grid"][-1])]
        for sg in grid:
            print(f"  [certify {tier_key} @ σ={sg:g}]", flush=True)
            c = certify_e2e(cell, sg, X_t, attr, task, n_attr, n_task, task_maj, device)
            if c[f"{tier_key}_max_mean"] <= BAR:
                return c
            print(f"    5-seed mean {tier_key} = {c[f'{tier_key}_max_mean']:.3f} > "
                  f"{BAR} — escalating", flush=True)
        return c   # last one, breaching — reported honestly

    cert1 = certify_at(sigma_t1, "tier1") if sigma_t1 is not None else None
    cert2 = (cert1 if (cert1 is not None and cert1["tier2_max_mean"] <= BAR
                       and sigma_t2 == cert1["sigma"])
             else (certify_at(sigma_t2, "tier2") if sigma_t2 is not None else None))

    out = {"cell": cell["name"], "clean_lift": clean_lift,
           "clean_lift_std": float(np.std(clean_lifts)),
           "sweep": sweep, "tier1": cert1, "tier2": cert2,
           "minutes": round((time.time() - t0) / 60, 1)}
    for tier, c in (("TIER 1", cert1), ("TIER 2", cert2)):
        if c is None:
            print(f"  >>> {tier}: NO operating point on the grid", flush=True)
        else:
            print(f"  >>> {tier}: σ={c['sigma']:g}  lift {c['lift_mean']:+.4f}"
                  f"±{c['lift_std']:.4f} ({100 * c['lift_mean'] / max(clean_lift, 1e-9):.0f}% "
                  f"of clean) T1={c['tier1_max_mean']:.3f} T2={c['tier2_max_mean']:.3f}",
                  flush=True)
    return out


# --------------------------------------------------------------------------- #
#  Section 2 — post-hoc surgical winners, both tiers                           #
# --------------------------------------------------------------------------- #
def posthoc_point(label, method, r, sr, H, H_t, attr, n_attr, task, task_maj,
                  sigma_h, device, Qcache):
    """Full two-tier battery + utility through a post-hoc channel."""
    sigma_abs = sr * sigma_h
    if method == "surgical":
        if r not in Qcache:
            Qcache[r] = fit_projection(H_t, attr, r, "hsic", device,
                                       seed=0).cpu().numpy().astype(np.float64)
        Q = Qcache[r]
        ch = subspace_channel(Q, sigma_abs)
        ncov = subspace_noise_cov(Q, sigma_abs)
    else:
        ch = iso_channel(sigma_abs)
        ncov = iso_noise_cov(H.shape[1], sigma_abs)
    _, xauc = probe_through("XGB", H, attr, n_attr, ch, SWEEP_SEEDS)
    _, mauc = probe_through("MLP", H, attr, n_attr, ch, SWEEP_SEEDS)
    lauc = lora_channel(H, attr, n_attr, ch, device, SWEEP_SEEDS)
    lrta, _ = gaussian_lrt_channel(H, attr, ch, ncov, n_attr, SWEEP_SEEDS)
    tl = utility_through(H, task, ch, SWEEP_SEEDS, task_maj)
    pt = {"label": label, "method": method, "rank": r, "sigma_rel": sr,
          "xgb": xauc, "mlp": mauc, "lora": lauc, "lrt": lrta, "lift": tl,
          "tier1_max": float(max(xauc, mauc, lauc)),
          "tier2_max": float(max(xauc, mauc, lauc, lrta))}
    print(f"  {label:<22} {fmt_point(pt)} lift={tl:+.4f}", flush=True)
    return pt


def run_posthoc_cell(cell, device):
    print("\n" + "#" * 96 + f"\n# POST-HOC SURGICAL: {cell['name']}\n" + "#" * 96,
          flush=True)
    X, attr, task, n_attr, n_task, _, task_maj = load_cell(
        cell["dataset"], cell["attr"], cell["task"])
    X_t = torch.from_numpy(X).to(device)
    H, _, _ = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                          alpha=0.0, device=device, pub_noise=0.0, seed=0)
    sigma_h = float(H.std(0).mean())
    clean_lift = utility_through(H, task, lambda Xm, rng: Xm, SWEEP_SEEDS, task_maj)
    H_t = torch.from_numpy(H.astype(np.float32)).to(device)
    Qcache = {}
    r_win, sr_win = cell["winner"]

    # Tier-1 point: the published winner config
    win = posthoc_point(f"winner sub r={r_win} s={sr_win:g}", "surgical", r_win,
                        sr_win, H, H_t, attr, n_attr, task, task_maj, sigma_h,
                        device, Qcache)

    # Tier-2 point: the winner if it already holds, else search wider surgical
    # configs, else blunt iso
    rows = [win]
    t2 = win if win["tier2_max"] <= BAR else None
    if t2 is None:
        for (r, sr) in POSTHOC_T2_SUR:
            r = r_win if r is None else r
            pt = posthoc_point(f"sub r={r} s={sr:g}", "surgical", r, sr, H, H_t,
                               attr, n_attr, task, task_maj, sigma_h, device, Qcache)
            rows.append(pt)
        for sr in POSTHOC_T2_ISO:
            pt = posthoc_point(f"iso s={sr:g}", "isotropic", None, sr, H, H_t,
                               attr, n_attr, task, task_maj, sigma_h, device, Qcache)
            rows.append(pt)
        holders = [p for p in rows if p["tier2_max"] <= BAR]
        t2 = max(holders, key=lambda p: p["lift"]) if holders else None

    out = {"cell": cell["name"], "clean_lift": clean_lift, "sigma_h": sigma_h,
           "winner_tier1": win, "rows": rows, "tier2": t2}
    print(f"  >>> TIER 1 (winner): {win['label']} lift {win['lift']:+.4f} "
          f"({100 * win['lift'] / max(clean_lift, 1e-9):.0f}%) "
          f"T1={win['tier1_max']:.3f} {'holds' if win['tier1_max'] <= BAR else 'BREACHES'}; "
          f"LRT={win['lrt']:.3f} → Tier-2 "
          f"{'holds' if win['tier2_max'] <= BAR else 'BREACHES'}", flush=True)
    if t2 is not None:
        print(f"  >>> TIER 2: {t2['label']} lift {t2['lift']:+.4f} "
              f"({100 * t2['lift'] / max(clean_lift, 1e-9):.0f}%) "
              f"T2={t2['tier2_max']:.3f}", flush=True)
    else:
        print("  >>> TIER 2: NO config on the search grid holds (reported honestly)",
              flush=True)
    return out


# --------------------------------------------------------------------------- #
#  Section 3 — Exp-10 e2e surgical winners at Tier 2                           #
# --------------------------------------------------------------------------- #
def run_e2e_surgical_winner(cfg, device):
    print("\n" + "#" * 96 + f"\n# E2E SURGICAL WINNER: {cfg['name']} "
          f"(r={cfg['rank']} σ={cfg['sigma']:g} λ={cfg['lam']:g})\n" + "#" * 96,
          flush=True)
    X, attr, task, n_attr, n_task, _, task_maj = load_cell(
        cfg["dataset"], cfg["attr"], cfg["task"])
    X_t = torch.from_numpy(X).to(device)
    _, _, L0 = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                           alpha=0.0, device=device, pub_noise=0.0, seed=0)
    clean_lift = lift(L0, task, task_maj)

    rows = []
    sigma = cfg["sigma"]
    for _ in range(MAX_ESCALATIONS + 1):
        P, L, h, Q = train_e2e_surgical(
            X_t, attr, task, n_attr, n_task, rank=cfg["rank"], sigma=sigma,
            lam=cfg["lam"], device=device, seed=0, return_clean=True)
        pt = full_point(h, P, attr, n_attr, subspace_noise_cov(Q, sigma), device,
                        SWEEP_SEEDS)
        pt["sigma"] = sigma
        pt["lift"] = lift(L, task, task_maj)
        rows.append(pt)
        print(f"  σ={sigma:>5.1f} {fmt_point(pt)} lift={pt['lift']:+.4f} "
              f"({100 * pt['lift'] / max(clean_lift, 1e-9):.0f}%)", flush=True)
        if pt["tier2_max"] <= BAR:
            break
        sigma *= 2
    t2 = rows[-1] if rows[-1]["tier2_max"] <= BAR else None
    print(f"  >>> winner config Tier-2 "
          f"{'HOLDS' if rows[0]['tier2_max'] <= BAR else 'BREACHES'} "
          f"(LRT={rows[0]['lrt']:.3f})"
          + (f"; Tier-2 point found at σ={t2['sigma']:g} "
             f"({100 * t2['lift'] / max(clean_lift, 1e-9):.0f}% utility)" if t2 and t2 is not rows[0]
             else ("" if t2 else f"; NO Tier-2 point within σ≤{sigma:g} "
                   "(subspace noise cannot cap the informed adversary — "
                   "consistent with Exp-11: surgical channels have no DPI ceiling)")),
          flush=True)
    return {"cell": cfg["name"], "config": {k: cfg[k] for k in ("rank", "sigma", "lam")},
            "clean_lift": clean_lift, "rows": rows, "tier2": t2}


# --------------------------------------------------------------------------- #
def main():
    device = get_device()
    print(f"[setup] device={device} bar≤{BAR} train_seeds={TRAIN_SEEDS} "
          f"sweep_seeds={SWEEP_SEEDS} cert_seeds={CERT_SEEDS}", flush=True)
    out = {"experiment": "two_tier_certification", "bar": BAR,
           "tier1": "XGB + MLP(256-256) + rank-32 ReLU LoRA (black-box)",
           "tier2": "tier1 + channel-aware Gaussian-LRT (informed: clean-rep side "
                    "information + known channel)"}

    out["e2e_cells"] = [run_e2e_cell(c, device) for c in E2E_CELLS]
    out["posthoc_cells"] = [run_posthoc_cell(c, device) for c in POSTHOC_CELLS]
    out["e2e_surgical_winners"] = [run_e2e_surgical_winner(c, device)
                                   for c in E2E_SUR_WINNERS]

    # ---------- the two-tier table ----------
    print("\n" + "=" * 108)
    print("TWO-TIER CERTIFICATION TABLE — operating point + utility kept, per tier")
    print("=" * 108)
    print(f"{'cell':<40} {'tier-1 point':>14} {'utility@T1':>16} "
          f"{'tier-2 point':>14} {'utility@T2':>16}")
    print("-" * 108)
    for c in out["e2e_cells"]:
        cl = c["clean_lift"]

        def u(cert):
            if cert is None:
                return "—", "—"
            return (f"σ={cert['sigma']:g}",
                    f"{cert['lift_mean']:+.4f} ({100 * cert['lift_mean'] / max(cl, 1e-9):.0f}%)")
        p1, u1 = u(c["tier1"])
        p2, u2 = u(c["tier2"])
        print(f"{'e2e ' + c['cell']:<40} {p1:>14} {u1:>16} {p2:>14} {u2:>16}")
    for c in out["posthoc_cells"]:
        cl = c["clean_lift"]
        w, t2 = c["winner_tier1"], c["tier2"]
        u1 = f"{w['lift']:+.4f} ({100 * w['lift'] / max(cl, 1e-9):.0f}%)"
        p2 = t2["label"] if t2 else "none found"
        u2 = (f"{t2['lift']:+.4f} ({100 * t2['lift'] / max(cl, 1e-9):.0f}%)"
              if t2 else "—")
        print(f"{'post-hoc ' + c['cell']:<40} {w['label']:>14} {u1:>16} {p2:>14} {u2:>16}")
    print("-" * 108)

    (RESULTS_DIR / "two_tier_certification.json").write_text(json.dumps(out, indent=2))
    _plot(out, RESULTS_DIR / "two_tier_certification.png")
    print(f"\n[save] {RESULTS_DIR / 'two_tier_certification.json'} and .png", flush=True)


def _plot(out, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = out["e2e_cells"]
    fig, axes = plt.subplots(1, len(cells), figsize=(5.6 * len(cells), 5.2),
                             squeeze=False)
    for ax, c in zip(axes[0], cells):
        sw = c["sweep"]
        sig = [p["sigma"] for p in sw]
        ax.plot(sig, [p["xgb"] for p in sw], "o-", label="XGB")
        ax.plot(sig, [p["mlp"] for p in sw], "s-", label="MLP")
        ax.plot(sig, [p["lrt"] for p in sw], "^-", color="C3",
                label="Gaussian-LRT (informed)")
        lor = [(p["sigma"], p["lora"]) for p in sw if "lora" in p]
        if lor:
            ax.plot(*zip(*lor), "D-", color="C4", label="LoRA-32")
        ax.axhline(BAR, color="green", ls="--", lw=1, label=f"bar {BAR}")
        for cert, col, nm in ((c["tier1"], "C0", "T1"), (c["tier2"], "C3", "T2")):
            if cert:
                ax.axvline(cert["sigma"], color=col, ls=":", lw=1.5,
                           label=f"{nm} σ={cert['sigma']:g} "
                                 f"({100 * cert['lift_mean'] / max(c['clean_lift'], 1e-9):.0f}% util)")
        ax.set_xlabel("σ")
        ax.set_ylabel("attribute recovery AUC")
        ax.set_title(f"e2e {c['cell']}", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Two-tier certification — the informed (LRT) tier costs more σ, "
                 "hence more utility", fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
