"""Experiment 16 (FINAL) — THE BASELINE GAUNTLET: the field's published
fair-representation / erasure methods under OUR evaluation standard.

Every prior experiment compared our mechanisms against our own reconstructions
of the competing ideas (probe-shaped projections, gradient-reversal scrubbing,
blunt noise). This experiment runs the actual named methods the field ships,
faithfully at published-recommended settings, on the three headline tabular
cells, and certifies them with the SAME two-tier battery our methods face:

  TIER 1 (black-box)  XGBoost + deep MLP(256-256) + rank-32 ReLU LoRA
  TIER 2 (informed)   Tier 1 + channel-aware Gaussian-LRT (utils/battery.py)

Baselines (knob swept per cell; best honest operating point per tier is
certified over 3 training seeds — the baselines get their best shot):

  1. LAFTR (Madras et al. 2018) — encoder + task head + adversary on the
     protected attribute, alternating min-max, adversary objective
     group-balanced (their group-normalized L1, generalized to the 5-class
     race attribute as macro-averaged per-group CE). Knob: adversary weight
     gamma in {0.5, 1, 2, 4, 8} (paper sweeps up to 4; 8 added so "leaks at
     every knob" cannot be a sweep artifact).
  2. VFAE (Louizos et al. 2016) — VAE with the protected attribute factored
     out: q(z|x,s), p(x|z,s), q(y|z), + MMD penalty on q(z) across s groups.
     Knob: MMD weight beta in {1, 10, 100, 1000}; two deployment exposures
     evaluated per beta — the SAMPLED z (the stochastic encoder is a
     heteroscedastic noise channel, VFAE's only attacker-agnostic component)
     and the posterior MEAN mu (the common deterministic deployment).
  3. Adversarial forgetting / gradient-reversal scrubbing at full published
     strength — deep 256-256 adversary, proper alternating schedule (5
     adversary steps per encoder step), encoder trained task-CE − lambda *
     adversary-CE. Knob: lambda in {1, 2, 5, 10, 20}. (Strictly stronger than
     the minimal single-step GRL scrub of Exp 7, which XGB read at 1.00.)
  4. LEACE (Belrose et al. 2023) — closed-form linear concept erasure
     (concept_erasure.LeaceEraser, as published) applied post-hoc to the clean
     trained representation P; utility and logits from a retrained LR head.
     No knob (the eraser is parameter-free); 3 seeds = 3 trained P's.

Faithful-minimal deviations, all favorable or neutral to the baselines:
project-standard encoder capacity (MLP 128->64, the same as our channels —
LAFTR/VFAE's published nets are smaller); Gaussian-kernel median-heuristic MMD
(smart_erasure.mmd_class_loss) instead of VFAE's random-feature MMD; single
stochastic layer for VFAE (no z2/y-prior stage — the fair-representation
penalty is unaffected); utility reported as the BEST of the method's own head
and a retrained-LR head through the exposed representation.

Tier-2 LRT side information: for deterministic representations (LAFTR, scrub,
LEACE, VFAE-mean) the exposed rep IS the clean rep, so Sigma_noise = 0 and the
LRT degenerates to an informed Gaussian (QDA) probe — the tier collapse is
real, not an artifact. For VFAE-sampled z the channel is heteroscedastic
per-row noise; the LRT gets H_clean = mu and Sigma_noise = diag(E[sigma^2])
(moment-matched approximation of the known channel).

Protocol per baseline x cell: knob sweep at train seed 0 (XGB+MLP+LRT on the
rep; LoRA added on Tier-1 candidates; XGB+MLP on the output logits); per tier,
candidates (tier max <= 0.55) are certified best-utility-first over 3 training
seeds with the full battery on representation AND output (up to 2 candidates
tried); the verdict is the 3-seed mean. Utility normalized by the cell's clean
e2e lift from results/two_tier_certification.json — the same denominator as
our methods' table. Never R^2. Local MPS/CPU; never AWS. macOS OpenMP guard:
xgboost before torch.
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
from experiments.sealed_channel import _mlp, train_model  # noqa: E402
from experiments.smart_erasure import mmd_class_loss  # noqa: E402
from experiments.targeted_noise import utility_through  # noqa: E402
from experiments.two_tier_certification import fmt_point, full_point  # noqa: E402
from utils.pcrl_io import get_device  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
PARTIAL = RESULTS_DIR / "baseline_gauntlet_partial.json"
BAR = 0.55
SWEEP_SEEDS = [0, 1]        # probe seeds inside the knob sweep
CERT_SEEDS = [0, 1, 2]      # probe seeds for certification batteries
TRAIN_SEEDS = [0, 1, 2]     # training seeds at each certified operating point
REP_DIM = 64
STEPS = 1500
BS = 1024
MAX_CERT_CANDIDATES = 2     # per tier: certify at most this many sweep candidates

LAFTR_GAMMAS = [0.5, 1.0, 2.0, 4.0, 8.0]
SCRUB_LAMBDAS = [1.0, 2.0, 5.0, 10.0, 20.0]
SCRUB_ADV_STEPS = 5
VFAE_BETAS = [1.0, 10.0, 100.0, 1000.0]
VFAE_ALPHA = 10.0           # supervised q(y|z) weight (fixed; utility cross-checked by LR)

CELLS = [
    dict(name="hmda/race/loan_decision", dataset="hmda", attr="race",
         task="loan_decision"),
    dict(name="hmda/race/loan_amount_band", dataset="hmda", attr="race",
         task="loan_amount_band"),
    dict(name="adult/sex/income", dataset="adult", attr="sex", task="income"),
]


# --------------------------------------------------------------------------- #
#  Shared bits                                                                 #
# --------------------------------------------------------------------------- #
def group_ce(logits, y):
    """Macro-averaged per-group CE — LAFTR's group-normalized adversary
    objective, generalized to a multiclass protected attribute."""
    ce = F.cross_entropy(logits, y, reduction="none")
    per = [ce[y == k].mean() for k in torch.unique(y)]
    return torch.stack(per).mean()


def _eval_chunks(fn, X_t, n, chunk=4096):
    outs = []
    with torch.no_grad():
        for i in range(0, n, chunk):
            outs.append(fn(X_t[i:i + chunk]))
    return [np.concatenate([o[j].cpu().numpy() for o in outs]).astype(np.float32)
            for j in range(len(outs[0]))]


def _batch_gen(n, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def nxt():
        return torch.randint(0, n, (min(BS, n),), generator=g)
    return nxt


# --------------------------------------------------------------------------- #
#  Baseline 1 — LAFTR                                                          #
# --------------------------------------------------------------------------- #
def train_laftr(X_t, attr_np, task_np, n_attr, n_task, gamma, device, seed):
    torch.manual_seed(seed)
    d_in = X_t.shape[1]
    enc = _mlp(d_in, 128, REP_DIM).to(device)
    clf = nn.Linear(REP_DIM, n_task).to(device)
    adv = _mlp(REP_DIM, 64, n_attr).to(device)
    opt_fg = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=1e-3)
    opt_h = torch.optim.Adam(adv.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    attr_t = torch.from_numpy(attr_np).long().to(device)
    task_t = torch.from_numpy(task_np).long().to(device)
    n = X_t.shape[0]
    nxt = _batch_gen(n, seed)
    for m in (enc, clf, adv):
        m.train()
    for _ in range(STEPS):
        # adversary ascent step (on the current encoder, detached)
        idx = nxt().to(device)
        with torch.no_grad():
            z = enc(X_t[idx])
        la = group_ce(adv(z), attr_t[idx])
        opt_h.zero_grad()
        la.backward()
        opt_h.step()
        # encoder-classifier descent step against the current adversary
        idx = nxt().to(device)
        z = enc(X_t[idx])
        loss = ce(clf(z), task_t[idx]) - gamma * group_ce(adv(z), attr_t[idx])
        opt_fg.zero_grad()
        loss.backward()
        opt_fg.step()
    enc.eval()
    clf.eval()
    Z, L = _eval_chunks(lambda xb: (enc(xb), clf(enc(xb))), X_t, n)
    return Z, L


# --------------------------------------------------------------------------- #
#  Baseline 3 — adversarial forgetting / full-strength GR scrubbing            #
# --------------------------------------------------------------------------- #
def train_scrub(X_t, attr_np, task_np, n_attr, n_task, lam, device, seed):
    torch.manual_seed(seed)
    d_in = X_t.shape[1]
    enc = _mlp(d_in, 128, REP_DIM).to(device)
    clf = nn.Linear(REP_DIM, n_task).to(device)
    adv = nn.Sequential(nn.Linear(REP_DIM, 256), nn.ReLU(),
                        nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, n_attr)).to(device)
    opt_fg = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=1e-3)
    opt_h = torch.optim.Adam(adv.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    attr_t = torch.from_numpy(attr_np).long().to(device)
    task_t = torch.from_numpy(task_np).long().to(device)
    n = X_t.shape[0]
    nxt = _batch_gen(n, seed)
    for m in (enc, clf, adv):
        m.train()
    for _ in range(STEPS):
        # proper alternating schedule: adversary trained hard between encoder steps
        for _ in range(SCRUB_ADV_STEPS):
            idx = nxt().to(device)
            with torch.no_grad():
                z = enc(X_t[idx])
            la = ce(adv(z), attr_t[idx])
            opt_h.zero_grad()
            la.backward()
            opt_h.step()
        idx = nxt().to(device)
        z = enc(X_t[idx])
        loss = ce(clf(z), task_t[idx]) - lam * ce(adv(z), attr_t[idx])
        opt_fg.zero_grad()
        loss.backward()
        opt_fg.step()
    enc.eval()
    clf.eval()
    Z, L = _eval_chunks(lambda xb: (enc(xb), clf(enc(xb))), X_t, n)
    return Z, L


# --------------------------------------------------------------------------- #
#  Baseline 2 — VFAE                                                           #
# --------------------------------------------------------------------------- #
def train_vfae(X_t, attr_np, task_np, n_attr, n_task, beta, device, seed):
    """Returns (mu, sigma2, z_sampled, L_sampled, L_mean) over all rows."""
    torch.manual_seed(seed)
    d_in = X_t.shape[1]
    enc = nn.Sequential(nn.Linear(d_in + n_attr, 128), nn.ReLU()).to(device)
    mu_h = nn.Linear(128, REP_DIM).to(device)
    lv_h = nn.Linear(128, REP_DIM).to(device)
    dec = nn.Sequential(nn.Linear(REP_DIM + n_attr, 128), nn.ReLU(),
                        nn.Linear(128, d_in)).to(device)
    clf = nn.Linear(REP_DIM, n_task).to(device)
    params = (list(enc.parameters()) + list(mu_h.parameters()) + list(lv_h.parameters())
              + list(dec.parameters()) + list(clf.parameters()))
    opt = torch.optim.Adam(params, lr=1e-3)
    ce = nn.CrossEntropyLoss()
    attr_t = torch.from_numpy(attr_np).long().to(device)
    task_t = torch.from_numpy(task_np).long().to(device)
    classes = sorted(np.unique(attr_np).tolist())
    n = X_t.shape[0]
    nxt = _batch_gen(n, seed)
    for m in (enc, mu_h, lv_h, dec, clf):
        m.train()
    for _ in range(STEPS):
        idx = nxt().to(device)
        xb = X_t[idx]
        sb = attr_t[idx]
        soh = F.one_hot(sb, n_attr).float()
        h = enc(torch.cat([xb, soh], 1))
        mu, lv = mu_h(h), lv_h(h).clamp(-8, 8)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        recon = F.mse_loss(dec(torch.cat([z, soh], 1)), xb, reduction="none").sum(1).mean()
        kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
        loss = (recon + kl + VFAE_ALPHA * ce(clf(z), task_t[idx])
                + beta * mmd_class_loss(z, sb, classes))
        opt.zero_grad()
        loss.backward()
        opt.step()
    for m in (enc, mu_h, lv_h, clf):
        m.eval()
    attr_oh = F.one_hot(torch.from_numpy(attr_np).long(), n_attr).float().to(device)

    def fwd(xb):
        i0 = fwd.pos
        soh = attr_oh[i0:i0 + xb.shape[0]]
        fwd.pos = i0 + xb.shape[0]
        h = enc(torch.cat([xb, soh], 1))
        mu, lv = mu_h(h), lv_h(h).clamp(-8, 8)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)   # one fresh draw per row
        return mu, lv.exp(), z, clf(z), clf(mu)
    fwd.pos = 0
    torch.manual_seed(seed + 10_000)   # deterministic exposure draw
    mu, sig2, z, Lz, Lmu = _eval_chunks(fwd, X_t, n)
    return mu, sig2, z, Lz, Lmu


# --------------------------------------------------------------------------- #
#  Baseline 4 — LEACE                                                          #
# --------------------------------------------------------------------------- #
def train_leace(X_t, attr_np, task_np, n_attr, n_task, device, seed):
    """Clean trained P (the project's standard from-raw-x encoder), LEACE-erased
    as published, LR head retrained on the erased rep for utility + logits."""
    from concept_erasure import LeaceEraser
    from sklearn.linear_model import LogisticRegression
    H, _, _ = train_model(X_t, attr_np, task_np, n_attr, n_task, use_seal=False,
                          alpha=0.0, device=device, pub_noise=0.0, seed=seed)
    zoh = F.one_hot(torch.from_numpy(attr_np).long(), n_attr).float()
    eraser = LeaceEraser.fit(torch.from_numpy(H).float(), zoh)
    H_er = eraser(torch.from_numpy(H).float()).numpy().astype(np.float32)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    lr.fit(H_er.astype(np.float64), task_np)
    L = np.log(np.clip(lr.predict_proba(H_er.astype(np.float64)), 1e-12, None)
               ).astype(np.float32)
    return H_er, L


# --------------------------------------------------------------------------- #
#  Exposure adapters — every baseline yields uniform "exposure" dicts          #
# --------------------------------------------------------------------------- #
def _exposure(label, knob, P, L, h_clean=None, ncov=None):
    d = P.shape[1]
    return dict(label=label, knob=knob, P=P, L=L,
                h_clean=P if h_clean is None else h_clean,
                ncov=np.zeros((d, d)) if ncov is None else ncov)


def make_exposures(baseline, knob, ctx, device, seed):
    X_t, attr, task = ctx["X_t"], ctx["attr"], ctx["task"]
    n_attr, n_task = ctx["n_attr"], ctx["n_task"]
    if baseline == "LAFTR":
        Z, L = train_laftr(X_t, attr, task, n_attr, n_task, knob, device, seed)
        return [_exposure(f"γ={knob:g}", knob, Z, L)]
    if baseline == "AdvScrub":
        Z, L = train_scrub(X_t, attr, task, n_attr, n_task, knob, device, seed)
        return [_exposure(f"λ={knob:g}", knob, Z, L)]
    if baseline == "VFAE":
        mu, sig2, z, Lz, Lmu = train_vfae(X_t, attr, task, n_attr, n_task, knob,
                                          device, seed)
        return [
            _exposure(f"β={knob:g} sampled-z", knob, z, Lz, h_clean=mu,
                      ncov=np.diag(sig2.mean(0).astype(np.float64))),
            _exposure(f"β={knob:g} mean-z", knob, mu, Lmu),
        ]
    if baseline == "LEACE":
        H_er, L = train_leace(X_t, attr, task, n_attr, n_task, device, seed)
        return [_exposure("closed-form", 0.0, H_er, L)]
    raise ValueError(baseline)


# --------------------------------------------------------------------------- #
#  Measurement                                                                 #
# --------------------------------------------------------------------------- #
def measure_sweep(exp, ctx, device):
    """Cheap sweep measurement: XGB+MLP+LRT on the rep (LoRA on Tier-1
    candidates), XGB+MLP on the output logits, own-head + retrained-LR lift."""
    attr, task = ctx["attr"], ctx["task"]
    n_attr, task_maj = ctx["n_attr"], ctx["task_maj"]
    pt = full_point(exp["h_clean"], exp["P"], attr, n_attr, exp["ncov"], device,
                    SWEEP_SEEDS, with_lora=False)
    if max(pt["xgb"], pt["mlp"]) <= BAR:
        pt["lora"], _ = probe_lora(exp["P"], attr, n_attr, SWEEP_SEEDS, device)
        pt["tier1_max"] = float(max(pt["xgb"], pt["mlp"], pt["lora"]))
        pt["tier2_max"] = float(max(pt["tier1_max"], pt["lrt"]))
    out_x, _ = probe_sklearn("XGB", exp["L"], attr, n_attr, SWEEP_SEEDS)
    out_m, _ = probe_sklearn("MLP", exp["L"], attr, n_attr, SWEEP_SEEDS)
    lift_own = lift(exp["L"], task, task_maj)
    lift_lr = utility_through(exp["P"], task, lambda X, rng: X, SWEEP_SEEDS, task_maj)
    row = dict(label=exp["label"], knob=exp["knob"], **pt,
               out_xgb=out_x, out_mlp=out_m, out_max=float(max(out_x, out_m)),
               lift_own=lift_own, lift_lr=lift_lr,
               lift_best=float(max(lift_own, lift_lr)))
    print(f"    {exp['label']:<20} {fmt_point(pt)} out_max={row['out_max']:.3f} "
          f"lift own={lift_own:+.4f} LR={lift_lr:+.4f}", flush=True)
    return row


def certify(baseline, label, knob, ctx, device):
    """3-training-seed certification: full battery (XGB+MLP+LoRA+LRT) on the
    representation AND the output logits, own-head + LR lift."""
    attr, task = ctx["attr"], ctx["task"]
    n_attr, task_maj = ctx["n_attr"], ctx["task_maj"]
    keys = ("xgb", "mlp", "lora", "lrt")
    per = {f"rep_{k}": [] for k in keys}
    per.update({f"out_{k}": [] for k in keys})
    per["lift_own"], per["lift_lr"] = [], []
    for ts in TRAIN_SEEDS:
        exps = make_exposures(baseline, knob, ctx, device, ts)
        exp = next(e for e in exps if e["label"] == label)
        rp = full_point(exp["h_clean"], exp["P"], attr, n_attr, exp["ncov"],
                        device, CERT_SEEDS)
        d_out = exp["L"].shape[1]
        op = full_point(exp["L"], exp["L"], attr, n_attr, np.zeros((d_out, d_out)),
                        device, CERT_SEEDS)
        for k in keys:
            per[f"rep_{k}"].append(rp[k])
            per[f"out_{k}"].append(op[k])
        per["lift_own"].append(lift(exp["L"], task, task_maj))
        per["lift_lr"].append(utility_through(exp["P"], task, lambda X, rng: X,
                                              CERT_SEEDS, task_maj))
        print(f"      [seed {ts}] rep {fmt_point(rp)} | out T2={op['tier2_max']:.3f} "
              f"| lift own={per['lift_own'][-1]:+.4f} LR={per['lift_lr'][-1]:+.4f}",
              flush=True)
    c = {f"{k}_mean": float(np.mean(v)) for k, v in per.items()}
    c.update({f"{k}_std": float(np.std(v)) for k, v in per.items()})
    c["per_seed"] = per
    c["rep_tier1_max"] = float(max(c[f"rep_{k}_mean"] for k in ("xgb", "mlp", "lora")))
    c["rep_tier2_max"] = float(max(c["rep_tier1_max"], c["rep_lrt_mean"]))
    c["out_tier1_max"] = float(max(c[f"out_{k}_mean"] for k in ("xgb", "mlp", "lora")))
    c["out_tier2_max"] = float(max(c["out_tier1_max"], c["out_lrt_mean"]))
    c["lift_best"] = float(max(c["lift_own_mean"], c["lift_lr_mean"]))
    c["label"], c["knob"] = label, knob
    return c


# --------------------------------------------------------------------------- #
#  Per-baseline driver                                                         #
# --------------------------------------------------------------------------- #
KNOBS = {"LAFTR": LAFTR_GAMMAS, "VFAE": VFAE_BETAS, "AdvScrub": SCRUB_LAMBDAS,
         "LEACE": [0.0]}


def run_baseline(baseline, ctx, device):
    print(f"\n  --- {baseline} on {ctx['name']} (knob sweep, train seed 0) ---",
          flush=True)
    rows = [measure_sweep(exp, ctx, device)
            for knob in KNOBS[baseline]
            for exp in make_exposures(baseline, knob, ctx, device, seed=0)]

    cert_cache = {}

    def tier_pick(tkey):
        cands = sorted([r for r in rows if r[f"{tkey}_max"] <= BAR],
                       key=lambda r: -r["lift_best"])
        tried = []
        for r in cands[:MAX_CERT_CANDIDATES]:
            if r["label"] not in cert_cache:
                print(f"    [certify {baseline} {r['label']} over {TRAIN_SEEDS}]",
                      flush=True)
                cert_cache[r["label"]] = certify(baseline, r["label"], r["knob"],
                                                 ctx, device)
            c = cert_cache[r["label"]]
            tried.append(r["label"])
            if c[f"rep_{tkey}_max"] <= BAR:
                return dict(certified=True, cert=c, tried=tried)
        # nothing certifies: report the sweep row closest to the bar, honestly
        best = min(rows, key=lambda r: r[f"{tkey}_max"])
        return dict(certified=False, cert=None, tried=tried, closest=best)

    t1, t2 = tier_pick("tier1"), tier_pick("tier2")
    for nm, t in (("TIER 1", t1), ("TIER 2", t2)):
        if t["certified"]:
            c = t["cert"]
            print(f"  >>> {baseline} {nm}: CERTIFIES at {c['label']} — "
                  f"lift {c['lift_best']:+.4f}, rep T1={c['rep_tier1_max']:.3f} "
                  f"T2={c['rep_tier2_max']:.3f}, out T2={c['out_tier2_max']:.3f}",
                  flush=True)
        else:
            b = t["closest"]
            print(f"  >>> {baseline} {nm}: does NOT certify — closest "
                  f"{b['label']} {nm.replace(' ', '').lower()}_max="
                  f"{b[f'tier{nm[-1]}_max']:.3f} at lift {b['lift_best']:+.4f}",
                  flush=True)
    return dict(baseline=baseline, rows=rows, tier1=t1, tier2=t2,
                certs=cert_cache)


# --------------------------------------------------------------------------- #
def load_ours():
    """Our methods' certified points, for the master table."""
    p = RESULTS_DIR / "two_tier_certification.json"
    d = json.loads(p.read_text())
    ours = {}
    for c in d["e2e_cells"]:
        ours[c["cell"]] = dict(
            clean_lift=c["clean_lift"],
            t1=c["tier1"] and dict(point=f"σ={c['tier1']['sigma']:g}",
                                   lift=c["tier1"]["lift_mean"]),
            t2=c["tier2"] and dict(point=f"σ={c['tier2']['sigma']:g}",
                                   lift=c["tier2"]["lift_mean"]))
    for c in d["posthoc_cells"]:
        key = c["cell"].split(" ", 1)[1]
        w, t2 = c["winner_tier1"], c["tier2"]
        ours[key]["surgical_t1"] = dict(point=w["label"], lift=w["lift"],
                                        holds=w["tier1_max"] <= d["bar"],
                                        clean_lift=c["clean_lift"])
        ours[key]["surgical_t2"] = t2 and dict(point=t2["label"], lift=t2["lift"],
                                               clean_lift=c["clean_lift"])
    return ours


def master_table(results, ours):
    print("\n" + "=" * 118)
    print("MASTER GAUNTLET TABLE — utility kept (% of the cell's clean e2e lift) "
          "at each method's best honest operating point, per tier")
    print("=" * 118)
    hdr = (f"{'cell / method':<46} {'op. point':>18} {'util kept':>10} "
           f"{'T1 rep':>7} {'T2 rep':>7} {'out T2':>7} {'verdict':>16}")
    for cell_res in results:
        name = cell_res["cell"]
        cl = ours[name]["clean_lift"]
        print(f"\n{name}  (predictor {cell_res['predictor']:.3f}, "
              f"clean e2e lift {cl:+.4f})")
        print(hdr)
        print("-" * 118)

        def pct(x):
            return f"{100 * x / max(cl, 1e-9):.0f}%"
        for b in cell_res["baselines"]:
            for nm, t in (("T1", b["tier1"]), ("T2", b["tier2"])):
                if t["certified"]:
                    c = t["cert"]
                    print(f"  {b['baseline'] + ' [' + nm + ']':<44} "
                          f"{c['label']:>18} {pct(c['lift_best']):>10} "
                          f"{c['rep_tier1_max']:>7.3f} {c['rep_tier2_max']:>7.3f} "
                          f"{c['out_tier2_max']:>7.3f} "
                          f"{'CERTIFIES @' + nm:>16}")
                else:
                    r = t["closest"]
                    tv = r["tier1_max"] if nm == "T1" else r["tier2_max"]
                    print(f"  {b['baseline'] + ' [' + nm + ']':<44} "
                          f"{'(' + r['label'] + ')':>18} {pct(r['lift_best']):>10} "
                          f"{r['tier1_max']:>7.3f} {r['tier2_max']:>7.3f} "
                          f"{r['out_max']:>7.3f} "
                          f"{'fails: ' + f'{tv:.3f}':>16}")
        o = ours[name]
        if o["t1"]:
            print(f"  {'OURS e2e noised channel [T1]':<44} {o['t1']['point']:>18} "
                  f"{pct(o['t1']['lift']):>10} {'≤.55':>7} {'—':>7} {'—':>7} "
                  f"{'CERTIFIES @T1':>16}")
        if o["t2"]:
            print(f"  {'OURS e2e noised channel [T2]':<44} {o['t2']['point']:>18} "
                  f"{pct(o['t2']['lift']):>10} {'≤.55':>7} {'≤.55':>7} {'—':>7} "
                  f"{'CERTIFIES @T2':>16}")
        s1 = o.get("surgical_t1")
        if s1:
            scl = s1["clean_lift"]
            print(f"  {'OURS post-hoc surgical [T1]':<44} {s1['point']:>18} "
                  f"{100 * s1['lift'] / max(scl, 1e-9):>9.0f}% {'≤.55':>7} "
                  f"{'breach':>7} {'—':>7} {'CERTIFIES @T1':>16}")
    print("-" * 118)
    print("util kept = best(own-head, retrained-LR) lift / the cell's clean e2e "
          "lift (our table's denominator; surgical rows use their own clean-P lift).")
    print("T1 = max(XGB, MLP, LoRA-32) on the representation; T2 adds the "
          "informed Gaussian-LRT; out T2 = the same battery on the task logits "
          "(the footprint channel). verdict is on the representation at ≤0.55, "
          "3-seed mean.")


def _plot(results, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    markers = {"LAFTR": "o", "VFAE": "s", "AdvScrub": "^", "LEACE": "D"}
    fig, axes = plt.subplots(1, len(results), figsize=(5.8 * len(results), 5.0),
                             squeeze=False)
    for ax, cr in zip(axes[0], results):
        for b in cr["baselines"]:
            xs1 = [r["tier1_max"] for r in b["rows"]]
            xs2 = [r["tier2_max"] for r in b["rows"]]
            ys = [r["lift_best"] for r in b["rows"]]
            ax.scatter(xs1, ys, marker=markers[b["baseline"]], s=55,
                       label=f"{b['baseline']} (T1)", alpha=0.85)
            ax.scatter(xs2, ys, marker=markers[b["baseline"]], s=55,
                       facecolors="none", edgecolors="gray",
                       label=f"{b['baseline']} (T2)", alpha=0.7)
        ax.axvline(BAR, color="green", ls="--", lw=1.2, label=f"bar {BAR}")
        ax.set_xlabel("attribute recovery AUC (max over battery members)")
        ax.set_ylabel("task lift (best of own head / retrained LR)")
        ax.set_title(cr["cell"], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("The baseline gauntlet — published methods under the two-tier "
                 "battery (left of the bar = certifies)", fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()
                if k not in ("P", "L", "h_clean", "ncov")}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


def main():
    device = get_device()
    ours = load_ours()
    partial = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    print(f"[setup] device={device} bar≤{BAR} steps={STEPS} bs={BS} "
          f"train_seeds={TRAIN_SEEDS} sweep_seeds={SWEEP_SEEDS} "
          f"cert_seeds={CERT_SEEDS}", flush=True)
    results = []
    for cell in CELLS:
        t0 = time.time()
        print("\n" + "#" * 100 + f"\n# CELL: {cell['name']}\n" + "#" * 100, flush=True)
        X, attr, task, n_attr, n_task, attr_maj, task_maj = load_cell(
            cell["dataset"], cell["attr"], cell["task"])
        pred, _ = label_predictor(attr, task, n_attr)
        ctx = dict(name=cell["name"], X_t=torch.from_numpy(X).to(device),
                   attr=attr, task=task, n_attr=n_attr, n_task=n_task,
                   task_maj=task_maj)
        print(f"[setup] N={X.shape[0]} x_dim={X.shape[1]} attr={n_attr}cls "
              f"task={n_task}cls maj={task_maj:.3f} predictor={pred:.4f} "
              f"clean e2e lift={ours[cell['name']]['clean_lift']:+.4f}", flush=True)
        brs = []
        for b in ("LAFTR", "VFAE", "AdvScrub", "LEACE"):
            key = f"{cell['name']}|{b}"
            if key in partial:
                print(f"  [resume] {key} from checkpoint", flush=True)
                brs.append(partial[key])
                continue
            r = _clean(run_baseline(b, ctx, device))
            partial[key] = r
            PARTIAL.write_text(json.dumps(partial))
            brs.append(r)
        results.append(dict(cell=cell["name"], predictor=pred,
                            clean_lift=ours[cell["name"]]["clean_lift"],
                            baselines=brs,
                            minutes=round((time.time() - t0) / 60, 1)))

    master_table(results, ours)

    out = {"experiment": "baseline_gauntlet", "bar": BAR, "steps": STEPS,
           "knobs": KNOBS, "vfae_alpha": VFAE_ALPHA,
           "scrub_adv_steps": SCRUB_ADV_STEPS,
           "tier1": "XGB + MLP(256-256) + rank-32 ReLU LoRA (black-box)",
           "tier2": "tier1 + Gaussian-LRT (informed; Σ_noise=0 for deterministic "
                    "reps, diag(E[σ²]) for VFAE sampled-z)",
           "cells": _clean(results), "ours": _clean(load_ours())}
    (RESULTS_DIR / "baseline_gauntlet.json").write_text(json.dumps(out, indent=2))
    _plot(results, RESULTS_DIR / "baseline_gauntlet.png")
    PARTIAL.unlink(missing_ok=True)
    print(f"\n[save] {RESULTS_DIR / 'baseline_gauntlet.json'} and .png", flush=True)


if __name__ == "__main__":
    main()
