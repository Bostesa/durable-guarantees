"""Experiment 10 — can SURGICAL destruction be TRAINED END-TO-END, and does it
beat the post-hoc version?

Exp 9 (Goal 3) showed post-hoc surgical noise — Gaussian noise confined to the
rank-r most-attribute-carrying HSIC subspace of a FIXED representation — beats
blunt isotropic noise at matched full-battery protection on every cell tested.
But post-hoc surgery is limited to whatever directions the fixed representation
happened to use. Here we test whether a representation trained FROM SCRATCH to
build the surgery in does better: shape the representation so attribute-carrying
and task-carrying directions are maximally separated, then destroy the attribute
directions — all trained jointly.

The comparison is a 2x2 factorial, {blunt, surgical} x {post-hoc, end-to-end},
so any end-to-end-surgical gain can be attributed (e2e-ness vs surgical-ness):

  A  post-hoc blunt     isotropic noise on the frozen clean trained P
                        (Exp-9 arm, re-run here on a targeted grid);
  B  post-hoc surgical  HSIC-subspace noise on the frozen clean P
                        (the Exp-9 winner, re-run on a targeted grid);
  C  e2e blunt          the Exp-8 noised channel — isotropic noise INSIDE
                        training (E_pub -> BN -> +noise -> head) — the control
                        that separates "end-to-end helps" from "surgical helps";
  D  e2e surgical (NEW) E(x) -> BN -> h; learned orthonormal rank-r basis Q
                        (differentiable QR); exposed rep h' = h + sigma*(z@Q^T)
                        (fresh noise per forward, confined to span(Q)); task head
                        on h'; joint loss CE(head(h'), y) + lambda*HSIC(h_perp, attr)
                        with h_perp = h - (h@Q)@Q^T — the HSIC term pushes
                        attribute information OUT of the surviving directions,
                        the task CE (trained through the noise) pulls task
                        information INTO them, and the noise destroys span(Q)
                        attacker-agnostically. lambda is piloted per cell over
                        {0, 10, 100} (lambda=0 doubles as the shaping ablation).

Protocol (Exp-9 Goal-3, unchanged): per config, XGBoost + deep MLP recover the
attribute from the exposed representation (NEVER R^2); configs with
max(XGB, MLP) <= 0.55 are candidates; the best-utility candidate per arm must
also hold a fresh rank-32 ReLU LoRA <= 0.55 or it is discarded and the next
candidate is tried. The e2e winners (C and D) are then retrained with 3 training
seeds and the FULL battery (XGB + MLP + LoRA-32) is run per seed on the exposed
representation AND on the task-output logits — error bars over training seeds.
Post-hoc winners also get the full battery on rep and output.

Utility: post-hoc arms = lift of a logistic head retrained through the channel
(Exp-9 protocol; there is no "own" head for a bolted-on channel); e2e arms =
lift of the model's own trained head (Exp-8 protocol), with the retrained-LR
lift recorded too as a cross-check (both appear in the JSON).

Footprint check: the label<->attribute predictor (XGB AUC of attr from the task
label alone) is the output-leak floor. If any arm's OUTPUT hides the attribute
(<= 0.55) on a cell whose predictor is ABOVE 0.55 while utility is kept, that
is a likely measurement leak, not a win — flagged loudly, not celebrated.

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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.diagnostic import (  # noqa: E402
    label_predictor, lift, load_cell, probe_lora, probe_sklearn,
)
from experiments.sealed_channel import PUB_DIM, TRAIN_BS, TRAIN_STEPS, _mlp, train_model  # noqa: E402
from experiments.smart_erasure import fit_projection, hsic_loss  # noqa: E402
from experiments.targeted_noise import (  # noqa: E402
    iso_channel, probe_through, subspace_channel, utility_through,
)
from utils.pcrl_io import get_device  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
BAR = 0.55
PROBE_SEEDS = [0, 1]          # probe/utility split seeds everywhere (Exp-9 grid seeds)
TRAIN_SEEDS = [0, 1, 2]       # training seeds for the e2e winners (error bars)
MAX_SIGMA = 64.0              # adaptive-extension ceiling for every arm

# e2e surgical grid
E2E_SUR_RANKS = [8, 16, 32]
E2E_SUR_SIGMAS = [4.0, 8.0, 16.0]
PILOT_LAMBDAS = [0.0, 10.0, 100.0]
PILOT_RANK, PILOT_SIGMA = 16, 8.0
# e2e blunt sweep (BN units — the Exp-8 channel), early break once clear with margin
E2E_ISO_SIGMAS = [2.0, 4.0, 8.0, 16.0, 32.0]

# Exp-9 cells with targeted post-hoc grids seeded by the Exp-9 winners
# (iso winners: easy sigma=16, middle 8, hard 16; sub winners: easy r16 s16,
#  middle r8 s16, hard r32 s16). Adaptive extension covers re-train drift.
CELLS = [
    dict(key="easy hmda/race/loan_decision", dataset="hmda", attr="race",
         task="loan_decision", iso_sigmas=[8.0, 16.0, 32.0],
         sub_grid=[(16, 8.0), (16, 16.0), (16, 32.0), (8, 16.0)]),
    dict(key="middle hmda/race/loan_amount_band", dataset="hmda", attr="race",
         task="loan_amount_band", iso_sigmas=[4.0, 8.0, 16.0],
         sub_grid=[(8, 8.0), (8, 16.0), (8, 32.0), (16, 16.0)]),
    dict(key="hard adult/sex/income", dataset="adult", attr="sex",
         task="income", iso_sigmas=[8.0, 16.0, 32.0],
         sub_grid=[(32, 8.0), (32, 16.0), (32, 32.0), (16, 16.0)]),
]


# --------------------------------------------------------------------------- #
#  Arm D — the end-to-end trained surgical channel                             #
# --------------------------------------------------------------------------- #
def train_e2e_surgical(X_t, attr_np, task_np, n_attr, n_task, *, rank, sigma, lam,
                       device, seed=0, steps=TRAIN_STEPS):
    """Jointly train encoder + learned attribute-subspace Q + task head through
    subspace noise. Returns (exposed rep h', own-head logits), both np over all rows."""
    torch.manual_seed(seed)
    d_in = X_t.shape[1]
    E = _mlp(d_in, 128, PUB_DIM).to(device)
    bn = nn.BatchNorm1d(PUB_DIM).to(device)
    M = nn.Parameter(torch.randn(PUB_DIM, rank, device=device) * 0.1)
    D = nn.Linear(PUB_DIM, n_task).to(device)
    opt = torch.optim.Adam(
        list(E.parameters()) + list(bn.parameters()) + [M] + list(D.parameters()), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    attr_t = torch.from_numpy(attr_np).long().to(device)
    task_t = torch.from_numpy(task_np).long().to(device)
    n = X_t.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    E.train(); bn.train()
    for _ in range(steps):
        idx = torch.randint(0, n, (TRAIN_BS,), generator=g).to(device)
        xb = X_t[idx]
        h = bn(E(xb))
        Q, _ = torch.linalg.qr(M)                       # orthonormal columns, d x r
        z = torch.randn(h.shape[0], rank, device=device)
        h_noised = h + sigma * (z @ Q.T)                # fresh channel noise, span(Q) only
        loss = ce(D(h_noised), task_t[idx])
        if lam > 0:
            h_perp = h - (h @ Q) @ Q.T                  # surviving directions (clean)
            yoh = F.one_hot(attr_t[idx], n_attr).float()
            loss = loss + lam * hsic_loss(h_perp, yoh)
        opt.zero_grad()
        loss.backward()
        opt.step()

    E.eval(); bn.eval()
    with torch.no_grad():
        Q, _ = torch.linalg.qr(M)
        P_all, L_all = [], []
        for i in range(0, n, 4096):
            h = bn(E(X_t[i:i + 4096]))
            z = torch.randn(h.shape[0], rank, device=device)
            hn = h + sigma * (z @ Q.T)                  # one fresh draw per row (exposed)
            P_all.append(hn.cpu().numpy())
            L_all.append(D(hn).cpu().numpy())
    return (np.concatenate(P_all).astype(np.float32),
            np.concatenate(L_all).astype(np.float32))


# --------------------------------------------------------------------------- #
#  Shared measurement helpers                                                  #
# --------------------------------------------------------------------------- #
def probe_fixed(H, y, n_classes, seeds=PROBE_SEEDS):
    """XGB + MLP AUC on a fixed feature matrix (e2e exposed reps / logits)."""
    xauc, _ = probe_sklearn("XGB", H, y, n_classes, seeds)
    mauc, _ = probe_sklearn("MLP", H, y, n_classes, seeds)
    return xauc, mauc


def lora_fixed(H, y, n_classes, device, seeds=PROBE_SEEDS):
    m, _ = probe_lora(H, y, n_classes, seeds, device)
    return float(m)


def lora_channel(H, y, n_classes, channel, device, seeds=PROBE_SEEDS):
    """Rank-32 LoRA through a post-hoc channel (fresh noise draw per seed)."""
    aucs = []
    for s in seeds:
        Hn = channel(H.astype(np.float64), np.random.default_rng(100 + s)).astype(np.float32)
        m, _ = probe_lora(Hn, y, n_classes, [s], device)
        aucs.append(m)
    return float(np.mean(aucs))


def lr_logits(H, task, seed=0):
    """Logistic head trained on H (all rows) -> class-probability outputs over
    all rows (finite, unlike log-proba which saturates to -inf). This is the
    deployed-output analogue for a post-hoc channel (mirrors the e2e arms,
    whose own head is trained on all rows); probing attr from probabilities is
    information-equivalent to probing the logits."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    clf.fit(H.astype(np.float64), task)
    return clf.predict_proba(H.astype(np.float64)).astype(np.float32)


def output_battery(logits, attr, n_attr, device):
    xauc, mauc = probe_fixed(logits, attr, n_attr)
    lauc = lora_fixed(logits, attr, n_attr, device)
    return {"out_xgb": xauc, "out_mlp": mauc, "out_lora": lauc,
            "out_max": float(max(xauc, mauc, lauc))}


# --------------------------------------------------------------------------- #
#  Post-hoc arms (A blunt, B surgical) — Exp-9 protocol on the clean trained P #
# --------------------------------------------------------------------------- #
def eval_posthoc(method, label, channel, H, attr, n_attr, task, task_maj, params):
    _, xauc = probe_through("XGB", H, attr, n_attr, channel, PROBE_SEEDS)
    _, mauc = probe_through("MLP", H, attr, n_attr, channel, PROBE_SEEDS)
    tl = utility_through(H, task, channel, PROBE_SEEDS, task_maj)
    row = {"method": method, "label": label, **params, "xgb_auc": xauc, "mlp_auc": mauc,
           "lift": tl, "candidate": bool(max(xauc, mauc) <= BAR)}
    print(f"  {label:<26} XGB={xauc:.4f} MLP={mauc:.4f} lift={tl:+.4f} "
          f"{'candidate(<=0.55)' if row['candidate'] else 'leaks'}", flush=True)
    return row


def run_posthoc_arm(method, H, H_t, attr, n_attr, task, task_maj, sigma_h, cell,
                    device, Qcache):
    rows = []
    if method == "isotropic":
        sigmas = list(cell["iso_sigmas"])
        i = 0
        while i < len(sigmas):
            sr = sigmas[i]
            rows.append(eval_posthoc("isotropic", f"iso s={sr:g}", iso_channel(sr * sigma_h),
                                     H, attr, n_attr, task, task_maj, {"sigma_rel": sr}))
            if (i == len(sigmas) - 1 and not any(r["candidate"] for r in rows)
                    and sr * 2 <= MAX_SIGMA):
                sigmas.append(sr * 2)               # adaptive extension
            i += 1
    else:
        grid = list(cell["sub_grid"])
        i = 0
        while i < len(grid):
            r, sr = grid[i]
            if r not in Qcache:
                Qcache[r] = fit_projection(H_t, attr, r, "hsic", device,
                                           seed=0).cpu().numpy().astype(np.float64)
            rows.append(eval_posthoc("subspace", f"sub r={r} s={sr:g}",
                                     subspace_channel(Qcache[r], sr * sigma_h),
                                     H, attr, n_attr, task, task_maj,
                                     {"rank": r, "sigma_rel": sr}))
            if (i == len(grid) - 1 and not any(row["candidate"] for row in rows)
                    and sr * 2 <= MAX_SIGMA):
                grid.append((r, sr * 2))            # adaptive extension
            i += 1

    # best-lift candidate that also holds the rank-32 LoRA
    chosen = None
    for c in sorted([r for r in rows if r["candidate"]], key=lambda r: -r["lift"]):
        if method == "isotropic":
            ch = iso_channel(c["sigma_rel"] * sigma_h)
        else:
            ch = subspace_channel(Qcache[c["rank"]], c["sigma_rel"] * sigma_h)
        la = lora_channel(H, attr, n_attr, ch, device)
        c["lora_auc"] = la
        print(f"  [LoRA check] {c['label']}: LoRA32={la:.4f} "
              f"{'holds' if la <= BAR else 'BREACHES — discarded'}", flush=True)
        if la <= BAR:
            chosen = c
            break
    if chosen is not None:                          # output footprint for the winner
        ch = (iso_channel(chosen["sigma_rel"] * sigma_h) if method == "isotropic"
              else subspace_channel(Qcache[chosen["rank"]], chosen["sigma_rel"] * sigma_h))
        Hn = ch(H.astype(np.float64), np.random.default_rng(7)).astype(np.float32)
        L = lr_logits(Hn, task)
        chosen["output"] = output_battery(L, attr, n_attr, device)
        print(f"  [output] {chosen['label']}: attr from logits "
              f"XGB={chosen['output']['out_xgb']:.3f} MLP={chosen['output']['out_mlp']:.3f} "
              f"LoRA={chosen['output']['out_lora']:.3f}", flush=True)
    return rows, chosen


# --------------------------------------------------------------------------- #
#  End-to-end arms (C blunt, D surgical)                                       #
# --------------------------------------------------------------------------- #
def eval_e2e(label, P, L, attr, n_attr, task, task_maj, params):
    xauc, mauc = probe_fixed(P, attr, n_attr)
    own = lift(L, task, task_maj)
    row = {"label": label, **params, "xgb_auc": xauc, "mlp_auc": mauc,
           "lift": own, "candidate": bool(max(xauc, mauc) <= BAR)}
    print(f"  {label:<26} XGB={xauc:.4f} MLP={mauc:.4f} own-lift={own:+.4f} "
          f"{'candidate(<=0.55)' if row['candidate'] else 'leaks'}", flush=True)
    return row


def _train_e2e(kind, X_t, attr, task, n_attr, n_task, params, device, seed):
    if kind == "e2e_blunt":
        P, _, L = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                              alpha=0.0, device=device,
                              pub_noise=params["sigma"], seed=seed)
        return P, L
    return train_e2e_surgical(X_t, attr, task, n_attr, n_task,
                              rank=params["rank"], sigma=params["sigma"],
                              lam=params["lam"], device=device, seed=seed)


def finalize_e2e(kind, chosen, X_t, attr, task, n_attr, n_task, task_maj, device):
    """3-training-seed retrain of an e2e winner with the FULL battery on the
    exposed rep AND the output logits, plus own-head and retrained-LR lifts."""
    per = {k: [] for k in ("rep_xgb", "rep_mlp", "rep_lora", "out_xgb", "out_mlp",
                           "out_lora", "lift", "lr_lift")}
    for ts in TRAIN_SEEDS:
        P, L = _train_e2e(kind, X_t, attr, task, n_attr, n_task, chosen, device, ts)
        xauc, mauc = probe_fixed(P, attr, n_attr)
        lauc = lora_fixed(P, attr, n_attr, device)
        ob = output_battery(L, attr, n_attr, device)
        per["rep_xgb"].append(xauc); per["rep_mlp"].append(mauc); per["rep_lora"].append(lauc)
        per["out_xgb"].append(ob["out_xgb"]); per["out_mlp"].append(ob["out_mlp"])
        per["out_lora"].append(ob["out_lora"])
        per["lift"].append(lift(L, task, task_maj))
        per["lr_lift"].append(utility_through(P, task, lambda X, rng: X,
                                              PROBE_SEEDS, task_maj))
        print(f"  [seed {ts}] rep XGB={xauc:.3f} MLP={mauc:.3f} LoRA={lauc:.3f} | "
              f"out XGB={ob['out_xgb']:.3f} MLP={ob['out_mlp']:.3f} LoRA={ob['out_lora']:.3f} | "
              f"lift={per['lift'][-1]:+.4f} (LR-through {per['lr_lift'][-1]:+.4f})", flush=True)
    final = {f"{k}_mean": float(np.mean(v)) for k, v in per.items()}
    final.update({f"{k}_std": float(np.std(v)) for k, v in per.items()})
    final["rep_max_mean"] = float(max(final["rep_xgb_mean"], final["rep_mlp_mean"],
                                      final["rep_lora_mean"]))
    final["out_max_mean"] = float(max(final["out_xgb_mean"], final["out_mlp_mean"],
                                      final["out_lora_mean"]))
    final["holds_full_battery"] = bool(final["rep_max_mean"] <= BAR)
    return final


def run_e2e_blunt(X_t, attr, task, n_attr, n_task, task_maj, device):
    rows = []
    sigmas = list(E2E_ISO_SIGMAS)
    i = 0
    while i < len(sigmas):
        s = sigmas[i]
        P, L = _train_e2e("e2e_blunt", X_t, attr, task, n_attr, n_task,
                          {"sigma": s}, device, 0)
        rows.append(eval_e2e(f"e2e-iso s={s:g}", P, L, attr, n_attr, task, task_maj,
                             {"sigma": s}))
        if rows[-1]["candidate"] and max(rows[-1]["xgb_auc"], rows[-1]["mlp_auc"]) <= BAR - 0.01:
            break                                   # clear with margin — stop the sweep
        if i == len(sigmas) - 1 and not any(r["candidate"] for r in rows) and s * 2 <= MAX_SIGMA:
            sigmas.append(s * 2)
        i += 1
    return rows


def run_e2e_surgical_grid(X_t, attr, task, n_attr, n_task, task_maj, device):
    rows = []

    def one(rank, sigma, lam):
        P, L = _train_e2e("e2e_surgical", X_t, attr, task, n_attr, n_task,
                          {"rank": rank, "sigma": sigma, "lam": lam}, device, 0)
        return eval_e2e(f"e2e-sur r={rank} s={sigma:g} l={lam:g}", P, L, attr, n_attr,
                        task, task_maj, {"rank": rank, "sigma": sigma, "lam": lam})

    print("  -- lambda pilot (r=16, s=8) --", flush=True)
    pilot = [one(PILOT_RANK, PILOT_SIGMA, lam) for lam in PILOT_LAMBDAS]
    rows.extend(pilot)
    cands = [r for r in pilot if r["candidate"]]
    lam_star = (max(cands, key=lambda r: r["lift"]) if cands
                else min(pilot, key=lambda r: max(r["xgb_auc"], r["mlp_auc"])))["lam"]
    print(f"  -- lambda* = {lam_star:g}; grid r x sigma --", flush=True)
    for rank in E2E_SUR_RANKS:
        for sigma in E2E_SUR_SIGMAS:
            if rank == PILOT_RANK and sigma == PILOT_SIGMA:
                continue                            # already run in the pilot
            rows.append(one(rank, sigma, lam_star))
    # adaptive extension: if nothing qualifies, push sigma up on the least-leaky (r, lam)
    if not any(r["candidate"] for r in rows):
        best = min(rows, key=lambda r: max(r["xgb_auc"], r["mlp_auc"]))
        s = max(E2E_SUR_SIGMAS)
        while s * 2 <= MAX_SIGMA and not rows[-1]["candidate"]:
            s *= 2
            rows.append(one(best["rank"], s, best["lam"]))
    return rows


def vet_e2e(kind, rows, X_t, attr, task, n_attr, n_task, device):
    """Best-lift candidate that also holds the rank-32 LoRA (retrain seed 0)."""
    for c in sorted([r for r in rows if r["candidate"]], key=lambda r: -r["lift"]):
        P, _ = _train_e2e(kind, X_t, attr, task, n_attr, n_task, c, device, 0)
        la = lora_fixed(P, attr, n_attr, device)
        c["lora_auc"] = la
        print(f"  [LoRA check] {c['label']}: LoRA32={la:.4f} "
              f"{'holds' if la <= BAR else 'BREACHES — discarded'}", flush=True)
        if la <= BAR:
            return c
    return None


# --------------------------------------------------------------------------- #
def run_cell(cell, device):
    key = cell["key"]
    print("\n" + "#" * 96 + f"\n# {key}\n" + "#" * 96, flush=True)
    t0 = time.time()
    X, attr, task, n_attr, n_task, _, task_maj = load_cell(
        cell["dataset"], cell["attr"], cell["task"])
    X_t = torch.from_numpy(X).to(device)
    pred, _ = label_predictor(attr, task, n_attr)

    # clean trained P (post-hoc base rep + e2e clean baseline, one model)
    H, _, L0 = train_model(X_t, attr, task, n_attr, n_task, use_seal=False,
                           alpha=0.0, device=device, pub_noise=0.0, seed=0)
    sigma_h = float(H.std(0).mean())
    clean_lift_ph = utility_through(H, task, lambda Xm, rng: Xm, PROBE_SEEDS, task_maj)
    clean_lift_e2e = lift(L0, task, task_maj)
    _, clean_xgb = probe_through("XGB", H, attr, n_attr, lambda Xm, rng: Xm, [0])
    print(f"[setup] N={X.shape[0]} d={H.shape[1]} predictor={pred:.4f} sigma_h={sigma_h:.3f} "
          f"clean lift: LR-through={clean_lift_ph:+.4f} own-head={clean_lift_e2e:+.4f} "
          f"clean XGB attr AUC={clean_xgb:.3f}", flush=True)

    H_t = torch.from_numpy(H.astype(np.float32)).to(device)
    Qcache = {}
    arms = {}

    print("\n--- A: post-hoc blunt (isotropic on frozen P) ---", flush=True)
    rows_a, best_a = run_posthoc_arm("isotropic", H, H_t, attr, n_attr, task, task_maj,
                                     sigma_h, cell, device, Qcache)
    print("\n--- B: post-hoc surgical (HSIC-subspace noise on frozen P) ---", flush=True)
    rows_b, best_b = run_posthoc_arm("subspace", H, H_t, attr, n_attr, task, task_maj,
                                     sigma_h, cell, device, Qcache)

    print("\n--- C: e2e blunt (Exp-8 noised channel, trained through) ---", flush=True)
    rows_c = run_e2e_blunt(X_t, attr, task, n_attr, n_task, task_maj, device)
    best_c = vet_e2e("e2e_blunt", rows_c, X_t, attr, task, n_attr, n_task, device)
    final_c = None
    if best_c is not None:
        print(f"  [finalize C] 3-seed retrain of {best_c['label']}", flush=True)
        final_c = finalize_e2e("e2e_blunt", best_c, X_t, attr, task, n_attr, n_task,
                               task_maj, device)

    print("\n--- D: e2e SURGICAL (learned subspace + noise + HSIC shaping) ---", flush=True)
    rows_d = run_e2e_surgical_grid(X_t, attr, task, n_attr, n_task, task_maj, device)
    best_d = vet_e2e("e2e_surgical", rows_d, X_t, attr, task, n_attr, n_task, device)
    final_d = None
    if best_d is not None:
        print(f"  [finalize D] 3-seed retrain of {best_d['label']}", flush=True)
        final_d = finalize_e2e("e2e_surgical", best_d, X_t, attr, task, n_attr, n_task,
                               task_maj, device)

    arms = {"A_posthoc_blunt": {"rows": rows_a, "best": best_a, "clean_lift": clean_lift_ph},
            "B_posthoc_surgical": {"rows": rows_b, "best": best_b, "clean_lift": clean_lift_ph},
            "C_e2e_blunt": {"rows": rows_c, "best": best_c, "final": final_c,
                            "clean_lift": clean_lift_e2e},
            "D_e2e_surgical": {"rows": rows_d, "best": best_d, "final": final_d,
                               "clean_lift": clean_lift_e2e}}

    out = {"cell": key, "dataset": cell["dataset"], "attr": cell["attr"],
           "task": cell["task"], "predictor": pred, "sigma_h": sigma_h,
           "clean_lift_posthoc": clean_lift_ph, "clean_lift_e2e": clean_lift_e2e,
           "clean_xgb_auc": clean_xgb, "arms": arms,
           "minutes": round((time.time() - t0) / 60, 1)}
    _cell_summary(out)
    return out


def _pct(best, clean, use_final=None):
    if best is None:
        return None
    l = use_final if use_final is not None else best["lift"]
    return 100.0 * l / max(clean, 1e-9)


def _cell_summary(c):
    print(f"\n>>> {c['cell']} (predictor {c['predictor']:.3f}) — utility kept at "
          f"full-battery <=0.55:", flush=True)
    for name, arm in c["arms"].items():
        b = arm["best"]
        if b is None:
            print(f"    {name:<20} none reaches full-battery <=0.55", flush=True)
            continue
        fin = arm.get("final")
        if fin:
            print(f"    {name:<20} {b['label']:<24} lift {fin['lift_mean']:+.4f}"
                  f"±{fin['lift_std']:.4f} ({_pct(b, arm['clean_lift'], fin['lift_mean']):.0f}%) "
                  f"rep_max {fin['rep_max_mean']:.3f} out_max {fin['out_max_mean']:.3f} "
                  f"{'HOLDS' if fin['holds_full_battery'] else 'BREACHES on retrain'}", flush=True)
        else:
            o = b.get("output", {})
            print(f"    {name:<20} {b['label']:<24} lift {b['lift']:+.4f} "
                  f"({_pct(b, arm['clean_lift']):.0f}%) LoRA {b.get('lora_auc', float('nan')):.3f} "
                  f"out_max {o.get('out_max', float('nan')):.3f}", flush=True)


# --------------------------------------------------------------------------- #
def main():
    device = get_device()
    print(f"[setup] device={device} bar<={BAR} probe_seeds={PROBE_SEEDS} "
          f"train_seeds={TRAIN_SEEDS}", flush=True)
    cells = [run_cell(c, device) for c in CELLS]

    print("\n" + "=" * 104)
    print("EXPERIMENT 10 — 2x2 {blunt,surgical} x {post-hoc,e2e} at full-battery honest "
          "protection (XGB+MLP+LoRA32 <= 0.55)")
    print("=" * 104)
    print(f"{'cell':<36} {'pred':>6} {'A post-blunt':>13} {'B post-surg':>12} "
          f"{'C e2e-blunt':>12} {'D e2e-surg':>15}")
    print("-" * 104)
    for c in cells:
        vals = []
        for name in ("A_posthoc_blunt", "B_posthoc_surgical", "C_e2e_blunt", "D_e2e_surgical"):
            arm = c["arms"][name]
            b, fin = arm["best"], arm.get("final")
            if b is None:
                vals.append("—")
            elif fin:
                vals.append(f"{_pct(b, arm['clean_lift'], fin['lift_mean']):.0f}%"
                            f"±{100 * fin['lift_std'] / max(arm['clean_lift'], 1e-9):.0f}")
            else:
                vals.append(f"{_pct(b, arm['clean_lift']):.0f}%")
        print(f"{c['cell']:<36} {c['predictor']:>6.3f} {vals[0]:>13} {vals[1]:>12} "
              f"{vals[2]:>12} {vals[3]:>15}")
    print("-" * 104)
    print("utility kept, % of each family's clean lift (post-hoc: retrained-LR head; "
          "e2e: own head, mean±std over 3 training seeds)")

    # footprint check
    print("\nFOOTPRINT CHECK — attr from the OUTPUT logits at each winner (floor = predictor):")
    for c in cells:
        for name, arm in c["arms"].items():
            b, fin = arm["best"], arm.get("final")
            om = (fin["out_max_mean"] if fin else
                  (b or {}).get("output", {}).get("out_max"))
            if om is None:
                continue
            flag = ""
            if c["predictor"] > BAR and om <= BAR:
                kept = _pct(b, arm["clean_lift"], fin["lift_mean"] if fin else None)
                flag = ("  <-- SUSPICIOUS if utility is kept (beats the footprint law?)"
                        if kept and kept > 30 else "  (utility mostly gone — consistent with law)")
            print(f"  {c['cell']:<36} {name:<20} out_max={om:.3f} "
                  f"predictor={c['predictor']:.3f}{flag}")

    (RESULTS_DIR / "end_to_end_surgical.json").write_text(json.dumps(
        {"experiment": "end_to_end_surgical", "bar": BAR, "probe_seeds": PROBE_SEEDS,
         "train_seeds": TRAIN_SEEDS, "cells": cells}, indent=2))
    _plot(cells, RESULTS_DIR / "end_to_end_surgical.png")
    print(f"\n[save] {RESULTS_DIR/'end_to_end_surgical.json'} and .png")


def _plot(cells, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["A_posthoc_blunt", "B_posthoc_surgical", "C_e2e_blunt", "D_e2e_surgical"]
    short = ["post-hoc\nblunt", "post-hoc\nsurgical", "e2e\nblunt", "e2e\nsurgical"]
    colors = ["C0", "C1", "C2", "C3"]
    fig, axes = plt.subplots(1, len(cells), figsize=(5.4 * len(cells), 5.4), squeeze=False)
    for ax, c in zip(axes[0], cells):
        kept, err = [], []
        for name in names:
            arm = c["arms"][name]
            b, fin = arm["best"], arm.get("final")
            if b is None:
                kept.append(0.0); err.append(0.0)
            elif fin:
                kept.append(_pct(b, arm["clean_lift"], fin["lift_mean"]))
                err.append(100 * fin["lift_std"] / max(arm["clean_lift"], 1e-9))
            else:
                kept.append(_pct(b, arm["clean_lift"])); err.append(0.0)
        x = np.arange(len(names))
        ax.bar(x, kept, yerr=err, capsize=4, color=colors)
        for xi, k in zip(x, kept):
            ax.text(xi, k + 1.5, f"{k:.0f}%", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=8)
        ax.set_ylim(0, 115)
        ax.set_ylabel("utility kept at full-battery <=0.55 (%)")
        ax.set_title(f"{c['cell']}\npredictor={c['predictor']:.3f}", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Experiment 10 — can surgical destruction be trained end-to-end? "
                 "(e2e arms: mean±std over 3 training seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
