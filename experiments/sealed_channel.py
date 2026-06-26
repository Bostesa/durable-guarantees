"""Experiment 7 — the FOOTPRINT hypothesis: does sealing a race channel hide race,
or does it leak through the task output regardless?

The wall we keep hitting: on HMDA, race is entangled with the loan decision, so any
task-useful representation keeps race recoverable. A natural escape is an
ARCHITECTURE: split the model into

  * a PUBLIC channel  P = E_pub(x)  — trained so race is unrecoverable from it,
  * a SEALED channel  S = E_seal(x) — holds the race-related signal, used by the
    task head but NEVER exposed to the attacker,
  * a task head        D([P, S])    — makes the loan decision from both.

The footprint hypothesis predicts this cannot work: because the decision depends on
race (via S), the OUTPUT (logits) leaks race to a strong probe even if P is perfectly
scrubbed — the footprint is in the output, not the representation. The alternative
(breakthrough) is that sealing genuinely hides race while keeping utility.

We give the architecture its BEST shot: P is scrubbed with an ADVERSARIAL
gradient-reversal race discriminator (the strongest practical scrubber, not the linear
HSIC that the project already showed is weak). Then we attack from two angles with
XGBoost + MLP (never R²):
  1. probe the PUBLIC channel P for race,
  2. probe the TASK OUTPUTS (logits) for race.

Three models isolate the mechanism:
  * PLAIN      (single channel, no scrub) — unprotected footprint ceiling.
  * PUB-ONLY   (scrub P, head uses only P) — privacy floor, but utility is sacrificed.
  * SEALED     (scrub P, head uses P+S)    — the test: does it recover utility, and if
    so does the output leak race (footprint) the way the hypothesis predicts?

Reference: how much race does the TRUE loan label alone encode (XGB on the 1-feature
label)? That is the floor the output footprint is tied to. Local MPS/CPU; never AWS.
macOS OpenMP guard: xgboost before torch.
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
import torch.nn as nn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.hardening_test import _probe  # noqa: E402  (PCRL-auditor XGB/MLP probe)
from utils.pcrl_io import (  # noqa: E402
    build_train_loader, extract_attr, extract_task_label, get_device,
)

RESULTS_DIR = REPO_ROOT / "results"
SEEDS = [0, 1, 2]
PUB_DIM = 64
SEAL_DIM = 16
TRAIN_STEPS = 1500
TRAIN_BS = 1024
ALPHA = 1.5            # adversarial gradient-reversal strength (scrub P)
PUB_NOISE = 6.0        # BN-unit isotropic noise on the public channel (provably hides race)
HONEST_AUC_BAR = 0.55  # a probe "fails to recover" iff AUC ≤ this (chance 0.5)


# ----- gradient reversal (DANN) -------------------------------------------- #
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.alpha * g, None


def grl(x, alpha):
    return _GRL.apply(x, alpha)


def _mlp(d_in, d_hid, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(), nn.Linear(d_hid, d_out))


def train_model(X_t, race_np, task_np, n_race, n_task, *, use_seal, alpha, device,
                pub_noise=0.0, steps=TRAIN_STEPS, seed=0):
    """Train D([P_used,(S)]) for the task. The public channel can be hidden two ways:
      * alpha>0      — adversarial gradient-reversal race discriminator on P (DANN);
      * pub_noise>0  — a BatchNorm'd isotropic NOISE channel on P (the mechanism Exp 5
        proved actually hides race; BN fixes P's scale so the net can't game the noise).
    The exposed public channel is P_used (BN+noise if pub_noise>0). Returns frozen
    P_used, S(or None), and logits over all rows (np)."""
    torch.manual_seed(seed)
    d_in = X_t.shape[1]
    E_pub = _mlp(d_in, 128, PUB_DIM).to(device)
    bn = nn.BatchNorm1d(PUB_DIM).to(device) if pub_noise > 0 else None
    E_seal = _mlp(d_in, 64, SEAL_DIM).to(device) if use_seal else None
    head_in = PUB_DIM + (SEAL_DIM if use_seal else 0)
    D = nn.Linear(head_in, n_task).to(device)
    Cd = _mlp(PUB_DIM, 64, n_race).to(device)        # adversarial race discriminator on P

    params = list(E_pub.parameters()) + list(D.parameters()) + list(Cd.parameters())
    if use_seal:
        params += list(E_seal.parameters())
    if bn is not None:
        params += list(bn.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    ce = nn.CrossEntropyLoss()
    race_t = torch.from_numpy(race_np).long().to(device)
    task_t = torch.from_numpy(task_np).long().to(device)
    n = X_t.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    def public(xb):
        P = E_pub(xb)
        if bn is not None:
            P = bn(P) + pub_noise * torch.randn_like(P)   # fresh channel noise each forward
        return P

    for m in (E_pub, D, Cd):
        m.train()
    if use_seal:
        E_seal.train()
    if bn is not None:
        bn.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (TRAIN_BS,), generator=g).to(device)
        xb = X_t[idx]
        P = public(xb)
        z = torch.cat([P, E_seal(xb)], 1) if use_seal else P
        loss = ce(D(z), task_t[idx])
        if alpha > 0:
            loss = loss + ce(Cd(grl(E_pub(xb), alpha)), race_t[idx])  # adversary on clean E_pub
        opt.zero_grad()
        loss.backward()
        opt.step()

    E_pub.eval()
    if use_seal:
        E_seal.eval()
    if bn is not None:
        bn.eval()
    D.eval()
    with torch.no_grad():
        P_all, S_all, L_all = [], [], []
        for i in range(0, n, 4096):
            xb = X_t[i:i + 4096]
            P = public(xb)                            # exposed public channel (one noise draw/row)
            S = E_seal(xb) if use_seal else None
            z = torch.cat([P, S], 1) if use_seal else P
            P_all.append(P.cpu().numpy())
            if use_seal:
                S_all.append(S.cpu().numpy())
            L_all.append(D(z).cpu().numpy())
    P_np = np.concatenate(P_all).astype(np.float32)
    S_np = np.concatenate(S_all).astype(np.float32) if use_seal else None
    L_np = np.concatenate(L_all).astype(np.float32)
    return P_np, S_np, L_np


def probe_race(H, race, n_race, seeds=SEEDS):
    """Held-out race recovery (acc + AUC) by XGBoost and MLP on a fixed feature matrix."""
    from sklearn.model_selection import train_test_split
    res = {}
    for arch in ("XGB", "MLP"):
        accs, aucs = [], []
        for s in seeds:
            Xtr, Xte, ytr, yte = train_test_split(
                H.astype(np.float64), race, test_size=0.25, random_state=s, stratify=race)
            acc, auc = _probe(arch, Xtr, ytr, Xte, yte, s, n_race)
            accs.append(acc)
            aucs.append(auc)
        res[arch] = {"acc": float(np.mean(accs)), "auc": float(np.mean(aucs))}
    return res


def task_lift(logits, task_np, task_majority):
    """Argmax task accuracy over majority (utility indicator, same protocol all models)."""
    acc = float((logits.argmax(1) == task_np).mean())
    return acc - task_majority


CELLS = [("hmda", "race", "loan_decision"), ("adult", "sex", "income")]


def run_cell(dataset, attr_name, task_name, device):
    print("\n" + "#" * 90)
    print(f"# CELL: {dataset} / {attr_name} / {task_name}")
    print("#" * 90)
    purposes, train_ds, loader = build_train_loader(dataset)
    X = np.concatenate([b["features"].numpy() for b in loader]).astype(np.float32)
    attr = extract_attr(loader, attr_name)
    task = extract_task_label(loader, task_name)
    n_attr = int(attr.max()) + 1
    n_task = int(task.max()) + 1
    attr_majority = float(np.bincount(attr).max() / len(attr))
    task_majority = float(np.bincount(task).max() / len(task))
    X_t = torch.from_numpy(X).to(device)
    print(f"[setup] N={X.shape[0]} x_dim={X.shape[1]} {attr_name}={n_attr}cls "
          f"(majority {attr_majority:.4f}) task '{task_name}' majority={task_majority:.4f} "
          f"PUB_DIM={PUB_DIM} SEAL_DIM={SEAL_DIM}")

    rows = []

    def record(model, target, H, kind="rep", logits=None):
        pr = probe_race(H, attr, n_attr)
        lift = task_lift(logits, task, task_majority) if logits is not None else None
        row = {"cell": f"{dataset}/{attr_name}", "model": model, "target": target, "kind": kind,
               "xgb_auc": pr["XGB"]["auc"], "xgb_acc": pr["XGB"]["acc"],
               "mlp_auc": pr["MLP"]["auc"], "mlp_acc": pr["MLP"]["acc"], "task_lift": lift,
               "attr_recovered": bool(max(pr["XGB"]["auc"], pr["MLP"]["auc"]) > HONEST_AUC_BAR)}
        rows.append(row)
        lift_s = f" | task_lift={lift:+.4f}" if lift is not None else ""
        print(f"  [{model:<10} · {target:<14}] XGB AUC={row['xgb_auc']:.4f}  MLP AUC={row['mlp_auc']:.4f}"
              f"  {'RECOVERS '+attr_name if row['attr_recovered'] else 'hidden (≤0.55)'}{lift_s}")
        return row

    print(f"\n--- reference: {attr_name} from the TRUE label (1 feature) ---")
    record("label", "true label y", task.reshape(-1, 1).astype(np.float32))

    print("\n--- PLAIN model (single channel, no scrub) ---")
    P0, _, L0 = train_model(X_t, attr, task, n_attr, n_task, use_seal=False, alpha=0.0,
                            device=device, seed=0)
    record("plain", "public P", P0)
    record("plain", "output logits", L0, kind="output", logits=L0)

    print(f"\n--- SEALED-adv (adversarial gradient-reversal scrub α={ALPHA}, head P+S) ---")
    Pa, _, La = train_model(X_t, attr, task, n_attr, n_task, use_seal=True, alpha=ALPHA,
                            device=device, seed=0)
    record("sealed-adv", "public P", Pa)
    record("sealed-adv", "output logits", La, kind="output", logits=La)

    print(f"\n--- PUB-ONLY-noised (BN+noise σ={PUB_NOISE} on P, head uses only P) ---")
    Pb, _, Lb = train_model(X_t, attr, task, n_attr, n_task, use_seal=False, alpha=0.0,
                            pub_noise=PUB_NOISE, device=device, seed=0)
    record("pub-noised", "public P", Pb)
    record("pub-noised", "output logits", Lb, kind="output", logits=Lb)

    print(f"\n--- SEALED-noised (AIRTIGHT: BN+noise σ={PUB_NOISE} on P, sealed S, head P+S) ---")
    Ps, Ss, Ls = train_model(X_t, attr, task, n_attr, n_task, use_seal=True, alpha=0.0,
                             pub_noise=PUB_NOISE, device=device, seed=0)
    record("sealed", "public P", Ps)
    record("sealed", "sealed S", Ss)               # sanity: does S actually hold the attribute?
    record("sealed", "output logits", Ls, kind="output", logits=Ls)

    _table(rows, attr_name, attr_majority, task_majority)
    verdict = _verdict(rows, attr_name)
    return rows, {"attr_majority": attr_majority, "task_majority": task_majority,
                  "attr_name": attr_name, "task_name": task_name, "verdict": verdict}


def main():
    device = get_device()
    print(f"[setup] device={device}  (OpenMP guard set; xgboost before torch)  α={ALPHA} σ={PUB_NOISE}")
    all_rows, meta = [], {}
    for dataset, attr_name, task_name in CELLS:
        rows, m = run_cell(dataset, attr_name, task_name, device)
        all_rows.extend(rows)
        meta[f"{dataset}/{attr_name}"] = m

    out = {"experiment": "sealed_channel", "cells": [f"{d}/{a}" for d, a, _ in CELLS],
           "honest_auc_bar": HONEST_AUC_BAR, "alpha": ALPHA, "pub_noise": PUB_NOISE,
           "pub_dim": PUB_DIM, "seal_dim": SEAL_DIM, "seeds": SEEDS,
           "meta": meta, "rows": all_rows}
    (RESULTS_DIR / "sealed_channel.json").write_text(json.dumps(out, indent=2))
    _plot(all_rows, RESULTS_DIR / "sealed_channel.png")
    print(f"\n[save] {RESULTS_DIR/'sealed_channel.json'} and .png")


def _table(rows, attr_name, attr_majority, task_majority):
    print("\n" + "=" * 90)
    print(f"TABLE [{rows[0]['cell']}] — {attr_name} recovery (XGB/MLP) per channel and from the output")
    print("=" * 90)
    print(f"{'model':<11} {'target':<15} {'XGB AUC':>8} {'MLP AUC':>8} {'task lift':>10} {attr_name+' hidden?':>14}")
    print("-" * 90)
    for r in rows:
        lift = f"{r['task_lift']:+.4f}" if r["task_lift"] is not None else "   —  "
        print(f"{r['model']:<11} {r['target']:<15} {r['xgb_auc']:>8.4f} {r['mlp_auc']:>8.4f} "
              f"{lift:>10} {('NO — recovers' if r['attr_recovered'] else 'yes (≤0.55)'):>14}")
    print("-" * 90)
    print(f"{attr_name} majority acc={attr_majority:.4f}; chance AUC=0.5; honest bar≤{HONEST_AUC_BAR}; "
          f"task majority={task_majority:.4f}")


def _verdict(rows, attr):
    g = lambda m, t: next((r for r in rows if r["model"] == m and r["target"] == t), None)
    mx = lambda r: max(r["xgb_auc"], r["mlp_auc"])
    bar = HONEST_AUC_BAR
    label = g("label", "true label y")
    plain_out = g("plain", "output logits")
    adv_P = g("sealed-adv", "public P")
    pub_P = g("pub-noised", "public P"); pub_out = g("pub-noised", "output logits")
    sealed_P = g("sealed", "public P"); sealed_S = g("sealed", "sealed S"); sealed_out = g("sealed", "output logits")

    plain_P = g("plain", "public P")
    p_dropped = mx(plain_P) - mx(sealed_P)            # how far we hid the representation
    out_floor = mx(sealed_out)                        # output leak with the most-hidden public channel
    out_follows = mx(plain_out) - out_floor           # did the output leak fall with the representation?
    seal_util = (sealed_out["task_lift"] or 0) - (pub_out["task_lift"] or 0)   # does the task need attr?
    adv_failed = mx(adv_P) > bar

    print(f"\n========== EXPERIMENT 7 VERDICT [{rows[0]['cell']}] ==========")
    head = (f"Adversarial scrub {'FAILED' if adv_failed else 'worked'} (P {attr} XGB "
            f"{adv_P['xgb_auc']:.3f}); only a NOISE channel hides the public channel, driven from "
            f"max-AUC {mx(plain_P):.3f} (plain) down to {mx(sealed_P):.3f} (sealed). Sealed S "
            f"genuinely holds {attr} (XGB {sealed_S['xgb_auc']:.3f}); label↔{attr} floor "
            f"(XGB on the true label) = {label['xgb_auc']:.3f}. ")
    if out_floor > bar + 0.05:
        body = (f"FOOTPRINT CONFIRMED (strong): the representation was hidden by {p_dropped:.2f} AUC "
                f"but the TASK OUTPUT still leaks {attr} at {out_floor:.3f} — it does NOT follow the "
                f"representation down to chance. The decision genuinely depends on {attr} "
                f"(sealing buys task lift {seal_util:+.4f} over public-only), and that utility comes "
                f"bundled with the output leak. You cannot make this decision accurately without the "
                f"decision encoding {attr}; sealing the internal channel is irrelevant.")
    elif out_floor <= bar + 0.01:
        body = (f"NO footprint here: as the representation is hidden, the OUTPUT leak FOLLOWS it down "
                f"to {out_floor:.3f} (≈ chance/label floor). The sealed channel adds only "
                f"{seal_util:+.4f} task lift — the task barely needs {attr}, so the decision encodes "
                f"almost none and a single noised channel already hides {attr} in both representation "
                f"AND output while keeping utility. The footprint exists only insofar as the decision "
                f"USES the attribute; on this cell it doesn't.")
    else:
        body = (f"BORDERLINE: output floor {out_floor:.3f} sits just above the bar; the footprint is "
                f"weak, consistent with a task that uses {attr} only a little (sealed util "
                f"{seal_util:+.4f}).")
    v = head + body
    print(v)
    return v


def _plot(rows, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = sorted({r["cell"] for r in rows})
    fig, axes = plt.subplots(len(cells), 1, figsize=(11, 5.4 * len(cells)))
    if len(cells) == 1:
        axes = [axes]
    for ax, cell in zip(axes, cells):
        cr = [r for r in rows if r["cell"] == cell]
        labels = [f"{r['model']}\n{r['target']}" for r in cr]
        x = np.arange(len(cr))
        ax.bar(x - 0.2, [r["xgb_auc"] for r in cr], 0.4, label="XGBoost AUC", color="C3")
        ax.bar(x + 0.2, [r["mlp_auc"] for r in cr], 0.4, label="MLP AUC", color="C1")
        ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance 0.5")
        ax.axhline(HONEST_AUC_BAR, color="green", ls="--", lw=1.1, label=f"hidden bar {HONEST_AUC_BAR}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_ylabel("attribute recovery AUC")
        ax.set_title(f"Sealed-channel / footprint test — {cell}\n"
                     "(noise can hide the public channel; does the OUTPUT still leak?)")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
