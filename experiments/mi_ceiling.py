"""Experiment 11 — is the noise-channel (DPI) recovery bound NON-VACUOUS at our
operating sigmas, or vacuous certificate theater?

We planned a proposition instantiating the data-processing-inequality guarantee for
the Gaussian noise channel  h~ = h + sigma*Z,  Z ~ N(0, I_d), fresh per forward
(A — h — h~ is a Markov chain). This experiment derives the tightest clean,
DISTRIBUTION-FREE version of that proposition, computes its ceiling numerically at
the exact sigmas our experiments ran, and compares it against the strong battery's
empirical recovery. Honest question: does the theorem separate us from full
recovery at the sigmas we actually operate at, or only at sigmas nobody can afford?

------------------------------------------------------------------------------
THE PROPOSITION (what the DPI guarantee cashes out to, distribution-free)
------------------------------------------------------------------------------
Let s_i^2 = Var(h_i) (per-dim variance of the PRE-noise representation; for the
BatchNorm'd end-to-end channels s_i ~= 1 by construction). Then, with NO
assumption on the distribution of h beyond those second moments:

  (i)  MI ceiling:   I(h~; A)  <=  I_cap(sigma) = 1/2 * sum_i log(1 + s_i^2/sigma^2).
       Proof: I(h~;A) <= sum_a pi_a KL(P~_a || Q) for ANY fixed Q (variational form
       of MI); take Q = N(mu_bar, diag(s_i^2 + sigma^2)); the cross-entropy term is
       a pure second-moment expression, and the entropy term uses
       h(P~_a) >= h(N(0, sigma^2 I)) (adding independent noise cannot reduce
       entropy). The mixture algebra collapses to I_cap.
       [Anisotropic full-rank noise: replace s_i^2/sigma^2 by s_i^2/sigma_i^2.
        SURGICAL noise (Exp 6/9/10: sigma=0 on the complement of a rank-r
        subspace) makes the bound DIVERGE — the DPI proposition covers only
        full-rank noise; a surgical channel has NO distribution-free ceiling.
        Its guarantee is, and remains, the empirical battery + HSIC shaping.]

  (ii) Accuracy ceiling (Fano):  any attacker's attribute accuracy
       <= 1 - pe*,  where pe* is the smallest p with
       h_b(p) + p*log(K-1) >= H(A) - I_cap.

  (iii) AUC ceiling (Pinsker chain): for each class k (binary attr, or
       one-vs-rest for K-class),  I(h~; 1{A=k}) <= I_cap  and
       TV(P~_k, P~_-k) <= sqrt(I_cap/8) / (pi_k(1-pi_k)),  hence for ANY scorer
       AUC_k <= 1/2 + TV - TV^2/2   (from ROC dominance TPR <= FPR + TV).
       Macro-OVR AUC ceiling = mean over classes.

  (iv) TIGHTNESS FLOOR (why no better distribution-free bound exists): the
       two-point distribution h = +-mu with ||mu||^2 = sum_i s_i^2 matches the
       moments, and there the OPTIMAL attacker genuinely achieves
       AUC = Phi(sqrt(2 * sum_i s_i^2) / sigma).  So ANY certificate that sees
       only second moments must concede at least this AUC — the vacuousness
       measured below is not sloppy accounting in our chain; most of it is
       irreducible without distributional assumptions.

Beyond the certificate we also compute an honest ESTIMATE of the true
information ceiling (not a theorem): the Gaussian-LRT attacker — fit
N(mu_k, Sigma_k + sigma^2 I) per class on a train split, score held-out rows
with fresh channel noise. If this lands near the battery, the battery is close
to information-optimal and the empirical protection is real; the gap to the
certificate is then the price of distribution-freeness.

------------------------------------------------------------------------------
CHANNELS EVALUATED (exactly the ones the experiments ran)
------------------------------------------------------------------------------
 1. adult/sex on the FROZEN PCRL representation + isotropic noise (Exp 2/5):
    sigma_abs = sigma_rel * mean per-dim std (~0.371); empirical XGB/MLP AUC from
    results/honest_reaudit.json (battery driven to ~0.53-0.55 at sigma_rel=4-8).
 2. hmda/race/loan_decision END-TO-END noised channel (Exp 8, the constructive
    win): BN units, sigma in {2,...,48}; empirical from results/diagnostic.json
    (incl. the 3-seed full-battery ~0.53 at sigma=8); sigma=24/48 batteries run
    fresh here (with rank-32 LoRA at 24/48, since "operate where the theorem
    bites" is a verdict-bearing claim).
 3. adult/sex/income END-TO-END noised channel (Exp 9 hard cell): sigma up to 48;
    empirical from results/continuous_cost_adult.json.
For the e2e channels the per-sigma model is retrained here (same trainer,
architecture, steps and seed as sealed_channel.train_model) and the PRE-noise
representation h = BN(E_pub(x)) extracted to measure s_i^2 per deployed model.

Local MPS/CPU; never AWS. OpenMP guard: xgboost before torch. Never R^2.
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

from experiments.diagnostic import (  # noqa: E402
    SWEEP_PROBE_SEEDS, _split, battery, lift, load_cell,
)
from experiments.sealed_channel import PUB_DIM, TRAIN_BS, TRAIN_STEPS, _mlp  # noqa: E402
from utils.pcrl_io import (  # noqa: E402
    INCOME_PREDICTION_PURPOSE_IDX, build_adult_train_loader, extract_attr,
    extract_frozen_reps, extract_task_label, get_device, load_adult_encoder,
)

RESULTS_DIR = REPO_ROOT / "results"
MEANINGFUL_AUC = 0.70   # ceiling below this = clearly separates us from full recovery
FROZEN_SIGMA_RELS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
HMDA_SIGMAS = [2.0, 4.0, 6.0, 8.0, 12.0, 24.0, 48.0]
ADULT_E2E_SIGMAS = [2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0]
FRESH_BATTERY_SIGMAS = {24.0, 48.0}   # hmda sigmas with no stored empirical numbers


# --------------------------------------------------------------------------- #
#  The certificate: closed-form ceilings from second moments                   #
# --------------------------------------------------------------------------- #
def cert_mi_nats(s2, sigma):
    """(i) distribution-free MI ceiling, nats."""
    return float(0.5 * np.sum(np.log1p(s2 / sigma ** 2)))


def _hb(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log(p) - (1 - p) * np.log(1 - p))


def entropy_nats(priors):
    return float(-np.sum([p * np.log(p) for p in priors if p > 0]))


def fano_acc_ceiling(mi_nats, priors):
    """(ii) accuracy ceiling via Fano. 1.0 = vacuous."""
    K = len(priors)
    budget = entropy_nats(priors) - mi_nats
    if budget <= 0:
        return 1.0
    lo, hi = 0.0, (K - 1) / K
    f = lambda p: _hb(p) + p * np.log(max(K - 1, 1))  # noqa: E731
    if f(hi) < budget:        # cannot happen (f(hi)=log K >= H(A)); guard anyway
        return 1.0 - hi
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(mid) >= budget:
            hi = mid
        else:
            lo = mid
    return float(1.0 - hi)


def auc_ceiling_macro(mi_nats, priors):
    """(iii) macro-OVR AUC ceiling via joint-Pinsker + ROC dominance."""
    aucs = []
    for pk in priors:
        tv = min(1.0, np.sqrt(mi_nats / 8.0) / (pk * (1 - pk)))
        aucs.append(min(1.0, 0.5 + tv - tv * tv / 2.0))
    return float(np.mean(aucs))


def worst_case_auc(s2, sigma):
    """(iv) moment-matched two-point construction: AUC an optimal attacker
    ACHIEVES on some distribution with these moments — the floor no
    second-moment-only certificate can go below."""
    from scipy.stats import norm
    return float(norm.cdf(np.sqrt(2.0 * np.sum(s2)) / sigma))


def sigma_star(s2, priors, target_auc):
    """Smallest sigma at which the certificate's macro AUC ceiling <= target."""
    for sigma in np.geomspace(0.5, 4096, 2000):
        if auc_ceiling_macro(cert_mi_nats(s2, sigma), priors) <= target_auc:
            return float(sigma)
    return float("inf")


# --------------------------------------------------------------------------- #
#  The estimate: Gaussian-LRT attacker (how close is the battery to optimal?)  #
# --------------------------------------------------------------------------- #
def gaussian_lrt_auc(h, y, sigma, n_classes, seed=0):
    """Fit N(mu_k, Sigma_k + sigma^2 I) per class on a train split; score
    held-out rows + fresh channel noise by class posterior; AUC like the battery."""
    from sklearn.metrics import roc_auc_score
    Xtr, Xte, ytr, yte = _split(h.astype(np.float64), y, seed)
    d = h.shape[1]
    rng = np.random.default_rng(seed)
    Xte_n = Xte + rng.normal(0.0, sigma, Xte.shape)
    logps = []
    for k in range(n_classes):
        Hk = Xtr[ytr == k]
        mu = Hk.mean(0)
        C = np.cov(Hk.T) + (sigma ** 2 + 1e-4) * np.eye(d)
        # Cholesky path: C is PD by construction; also sidesteps a spurious
        # numpy/Accelerate slogdet RuntimeWarning on macOS (values identical).
        from scipy.linalg import solve_triangular
        Lc = np.linalg.cholesky(C)
        logdet = 2.0 * float(np.log(np.diag(Lc)).sum())
        z = solve_triangular(Lc, (Xte_n - mu).T, lower=True)
        logp = -0.5 * ((z ** 2).sum(0) + logdet + d * np.log(2 * np.pi))
        logps.append(logp + np.log(len(Hk) / len(Xtr)))
    L = np.stack(logps, 1)
    L -= L.max(1, keepdims=True)
    post = np.exp(L) / np.exp(L).sum(1, keepdims=True)
    if n_classes == 2:
        return float(roc_auc_score(yte, post[:, 1]))
    return float(roc_auc_score(yte, post, multi_class="ovr", average="macro"))


# --------------------------------------------------------------------------- #
#  End-to-end channel replica (sealed_channel.train_model + pre-noise h)       #
# --------------------------------------------------------------------------- #
def train_channel(X_t, attr_np, task_np, n_attr, n_task, sigma, device, seed=0):
    """Identical to sealed_channel.train_model(use_seal=False, alpha=0.0,
    pub_noise=sigma) — same module creation order, optimizer, steps, batch and
    seed handling — but additionally returns the PRE-noise representation
    h = BN(E_pub(x)) (eval mode), which the certificate needs."""
    torch.manual_seed(seed)
    d_in = X_t.shape[1]
    E_pub = _mlp(d_in, 128, PUB_DIM).to(device)
    bn = nn.BatchNorm1d(PUB_DIM).to(device) if sigma > 0 else None
    D = nn.Linear(PUB_DIM, n_task).to(device)
    Cd = _mlp(PUB_DIM, 64, n_attr).to(device)   # unused (alpha=0); mirrors train_model
    params = list(E_pub.parameters()) + list(D.parameters()) + list(Cd.parameters())
    if bn is not None:
        params += list(bn.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    ce = nn.CrossEntropyLoss()
    task_t = torch.from_numpy(task_np).long().to(device)
    n = X_t.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)

    def public(xb):
        P = E_pub(xb)
        if bn is not None:
            P = bn(P) + sigma * torch.randn_like(P)
        return P

    E_pub.train()
    D.train()
    Cd.train()
    if bn is not None:
        bn.train()
    for _ in range(TRAIN_STEPS):
        idx = torch.randint(0, n, (TRAIN_BS,), generator=g).to(device)
        loss = ce(D(public(X_t[idx])), task_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    E_pub.eval()
    D.eval()
    if bn is not None:
        bn.eval()
    with torch.no_grad():
        H_all, P_all, L_all = [], [], []
        for i in range(0, n, 4096):
            xb = X_t[i:i + 4096]
            hb = bn(E_pub(xb)) if bn is not None else E_pub(xb)  # pre-noise h (no RNG)
            P = hb + sigma * torch.randn_like(hb) if bn is not None else hb
            H_all.append(hb.cpu().numpy())
            P_all.append(P.cpu().numpy())
            L_all.append(D(P).cpu().numpy())
    return (np.concatenate(H_all).astype(np.float32),
            np.concatenate(P_all).astype(np.float32),
            np.concatenate(L_all).astype(np.float32))


# --------------------------------------------------------------------------- #
#  Per-channel evaluation                                                      #
# --------------------------------------------------------------------------- #
def eval_point(h, attr, n_attr, sigma, emp, device=None, fresh_battery=False,
               task=None, task_maj=None):
    """All ceilings + estimate at one (h, sigma); emp = stored empirical numbers."""
    s2 = h.var(0).astype(np.float64)
    priors = np.bincount(attr, minlength=n_attr) / len(attr)
    mi = cert_mi_nats(s2, sigma)
    row = {
        "sigma": sigma,
        "mean_dim_std": float(np.sqrt(s2.mean())),
        "total_var": float(s2.sum()),
        "cert_mi_nats": mi,
        "H_A_nats": entropy_nats(priors),
        "cert_acc_ceiling": fano_acc_ceiling(mi, priors),
        "cert_auc_ceiling": auc_ceiling_macro(mi, priors),
        "worst_case_auc": worst_case_auc(s2, sigma),
        "gauss_lrt_auc": gaussian_lrt_auc(h, attr, sigma, n_attr),
    }
    row.update(emp)
    if fresh_battery:
        assert device is not None
        P = (h + np.random.default_rng(0).normal(0.0, sigma, h.shape)).astype(np.float32)
        b = battery(P, attr, n_attr, device, SWEEP_PROBE_SEEDS)   # XGB + MLP + LoRA32
        row["emp_xgb"] = b["XGB"]["auc"]
        row["emp_mlp"] = b["MLP"]["auc"]
        row["emp_lora32"] = b["LoRA32"]["auc"]
    return row


def print_channel_table(name, rows, priors):
    print("\n" + "=" * 118)
    print(f"CHANNEL: {name}   (K={len(priors)}, priors={[round(p,3) for p in priors]}, "
          f"H(A)={rows[0]['H_A_nats']:.3f} nats)")
    print("=" * 118)
    print(f"{'sigma':>7} {'s_bar':>6} | {'I_cap(nats)':>11} {'acc ceil':>8} "
          f"{'AUC CEIL':>8} {'wc AUC':>7} | {'gaussLRT':>8} | "
          f"{'emp XGB':>8} {'emp MLP':>8} {'emp LoRA':>8} {'lift':>8}")
    print("-" * 118)
    for r in rows:
        emp = lambda k: f"{r[k]:>8.3f}" if r.get(k) is not None else f"{'—':>8}"  # noqa: E731
        print(f"{r['sigma']:>7.2f} {r['mean_dim_std']:>6.3f} | {r['cert_mi_nats']:>11.3f} "
              f"{r['cert_acc_ceiling']:>8.3f} {r['cert_auc_ceiling']:>8.3f} "
              f"{r['worst_case_auc']:>7.3f} | {r['gauss_lrt_auc']:>8.3f} | "
              f"{emp('emp_xgb')} {emp('emp_mlp')} {emp('emp_lora32')} "
              + (f"{r['lift']:>+8.4f}" if r.get("lift") is not None else f"{'—':>8}"))
    print("-" * 118)
    print("I_cap = certificate MI bound; acc/AUC CEIL = provable ceilings from I_cap "
          "(Fano / Pinsker+ROC); wc AUC = moment-matched worst case (floor for ANY "
          "2nd-moment certificate); gaussLRT = Gaussian-LRT attacker (estimate, not a bound).")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    device = get_device()
    print(f"[setup] device={device}; meaningful ceiling bar = AUC ≤ {MEANINGFUL_AUC}")
    out = {"experiment": "mi_ceiling", "meaningful_auc": MEANINGFUL_AUC, "channels": {}}

    # ---------- Channel 1: adult/sex, frozen PCRL rep + isotropic noise ----------
    print("\n[channel 1] adult/sex on the frozen PCRL representation (Exp 2/5 sigmas)")
    encoder, _ = load_adult_encoder()
    _, _, loader = build_adult_train_loader()
    H = extract_frozen_reps(encoder, loader, INCOME_PREDICTION_PURPOSE_IDX, device)
    sex = extract_attr(loader, "sex")
    sigma_h = float(H.std(0).mean())
    hr = json.load(open(RESULTS_DIR / "honest_reaudit.json"))
    stored = {r["sigma_rel"]: r for r in hr["part1_exp2_noise_adult"]}
    rows = []
    for sr in FROZEN_SIGMA_RELS:
        s = stored.get(sr, {})
        emp = {"sigma_rel": sr, "emp_xgb": s.get("xgb_auc"), "emp_mlp": s.get("mlp_auc"),
               "emp_lora32": None, "lift": s.get("income_lift")}
        rows.append(eval_point(H, sex, 2, sr * sigma_h, emp))
        print(f"  σ_rel={sr:g} done")
    pri = list(np.bincount(sex) / len(sex))
    print_channel_table("adult/sex — frozen PCRL rep (σ_abs = σ_rel × "
                        f"{sigma_h:.3f})", rows, pri)
    s2 = H.var(0).astype(np.float64)
    ss = {t: sigma_star(s2, pri, t) for t in (MEANINGFUL_AUC, 0.65, 0.60)}
    print(f"  σ* where the CERTIFICATE first reaches AUC ceiling ≤ 0.70/0.65/0.60: "
          f"{ss[MEANINGFUL_AUC]:.1f} / {ss[0.65]:.1f} / {ss[0.60]:.1f} "
          f"(= σ_rel {ss[MEANINGFUL_AUC]/sigma_h:.0f} / {ss[0.65]/sigma_h:.0f} / "
          f"{ss[0.60]/sigma_h:.0f}; battery was at ~0.53 by σ_rel=4-8)")
    out["channels"]["adult_frozen"] = {"cell": "adult/sex (frozen PCRL rep)",
                                       "sigma_h": sigma_h, "priors": pri,
                                       "rows": rows, "sigma_star": ss}

    # ---------- Channel 2: hmda/race/loan_decision e2e channel (Exp 8) ----------
    print("\n[channel 2] hmda/race/loan_decision — end-to-end noised channel (Exp 8)")
    X, attr, task, n_attr, n_task, _, task_maj = load_cell("hmda", "race", "loan_decision")
    X_t = torch.from_numpy(X).to(device)
    diag = json.load(open(RESULTS_DIR / "diagnostic.json"))
    dsweep = {p["sigma"]: p for r in diag["part_b_c"]
              if r["cell"] == "hmda/race/loan_decision" for p in r["sweep"]}
    rows = []
    for sg in HMDA_SIGMAS:
        h, P, L = train_channel(X_t, attr, task, n_attr, n_task, sg, device, seed=0)
        sp = dsweep.get(sg)
        emp = {"emp_xgb": sp["P_xgb"] if sp else None,
               "emp_mlp": sp["P_mlp"] if sp else None,
               "emp_lora32": None,
               "lift": sp["lift"] if sp else lift(L, task, task_maj)}
        r = eval_point(h, attr, n_attr, sg, emp, device=device,
                       fresh_battery=(sg in FRESH_BATTERY_SIGMAS))
        rows.append(r)
        print(f"  σ={sg:g} trained+evaluated (lift {r['lift']:+.4f})")
    pri = list(np.bincount(attr) / len(attr))
    print_channel_table("hmda/race/loan_decision — e2e noised channel (BN units)",
                        rows, pri)
    pa = diag["part_a"]["rows"][-1]
    print(f"  [context] Exp-8 3-seed full battery at σ=8: P XGB/MLP/LoRA32 = "
          f"{pa['P_XGB_mean']:.3f}/{pa['P_MLP_mean']:.3f}/{pa['P_LoRA32_mean']:.3f}, "
          f"96% utility kept")
    s2 = rows[-1]["total_var"] / PUB_DIM * np.ones(PUB_DIM)  # BN keeps s_i^2 ~ const
    ss = {t: sigma_star(s2, pri, t) for t in (MEANINGFUL_AUC, 0.65, 0.60)}
    print(f"  σ* where the certificate reaches AUC ceiling ≤ 0.70/0.65/0.60: "
          f"{ss[MEANINGFUL_AUC]:.1f} / {ss[0.65]:.1f} / {ss[0.60]:.1f} "
          f"(battery is at ~0.53 from σ=8)")
    out["channels"]["hmda_e2e"] = {"cell": "hmda/race/loan_decision (e2e channel)",
                                   "priors": pri, "rows": rows, "sigma_star": ss}

    # ---------- Channel 3: adult/sex/income e2e channel (Exp 9 hard cell) ----------
    print("\n[channel 3] adult/sex/income — end-to-end noised channel (hard cell)")
    X, attr, task, n_attr, n_task, _, task_maj = load_cell("adult", "sex", "income")
    X_t = torch.from_numpy(X).to(device)
    cc = json.load(open(RESULTS_DIR / "continuous_cost_adult.json"))
    csweep = {p["sigma"]: p for r in cc["rows"]
              if r["cell"] == "adult/sex/income" for p in r["sweep"]}
    rows = []
    for sg in ADULT_E2E_SIGMAS:
        h, P, L = train_channel(X_t, attr, task, n_attr, n_task, sg, device, seed=0)
        sp = csweep.get(sg)
        emp = {"emp_xgb": sp["P_xgb"] if sp else None,
               "emp_mlp": sp["P_mlp"] if sp else None,
               "emp_lora32": None,
               "lift": sp["lift"] if sp else lift(L, task, task_maj)}
        rows.append(eval_point(h, attr, n_attr, sg, emp))
        print(f"  σ={sg:g} trained+evaluated (lift {rows[-1]['lift']:+.4f})")
    pri = list(np.bincount(attr) / len(attr))
    print_channel_table("adult/sex/income — e2e noised channel (BN units)", rows, pri)
    s2 = rows[-1]["total_var"] / PUB_DIM * np.ones(PUB_DIM)
    ss = {t: sigma_star(s2, pri, t) for t in (MEANINGFUL_AUC, 0.65, 0.60)}
    print(f"  σ* where the certificate reaches AUC ceiling ≤ 0.70/0.65/0.60: "
          f"{ss[MEANINGFUL_AUC]:.1f} / {ss[0.65]:.1f} / {ss[0.60]:.1f} "
          f"(empirical lift is ≈0 from σ=32)")
    out["channels"]["adult_e2e"] = {"cell": "adult/sex/income (e2e channel)",
                                    "priors": pri, "rows": rows, "sigma_star": ss}

    # ---------- Verdict ----------
    print("\n" + "=" * 118)
    print("VERDICT — is the theoretical ceiling meaningful at the operating sigmas?")
    print("=" * 118)
    verdicts = []
    for key, op_sigmas in (("adult_frozen", [4 * sigma_h, 8 * sigma_h]),
                           ("hmda_e2e", [8.0]), ("adult_e2e", [24.0, 32.0])):
        ch = out["channels"][key]
        for osg in op_sigmas:
            r = min(ch["rows"], key=lambda r: abs(r["sigma"] - osg))
            meaningful = r["cert_auc_ceiling"] <= MEANINGFUL_AUC
            verdicts.append(meaningful)
            emp_max = max(v for v in (r.get("emp_xgb"), r.get("emp_mlp"),
                                      r.get("emp_lora32")) if v is not None)
            print(f"  {ch['cell']:<42} σ={r['sigma']:>6.2f}: empirical max AUC "
                  f"{emp_max:.3f} vs certificate ceiling {r['cert_auc_ceiling']:.3f} "
                  f"(I_cap={r['cert_mi_nats']:.2f} nats vs H(A)={r['H_A_nats']:.2f}) → "
                  + ("MEANINGFUL" if meaningful else "VACUOUS"))
    out["verdict_meaningful_at_operating_sigma"] = bool(any(verdicts))
    (RESULTS_DIR / "mi_ceiling.json").write_text(json.dumps(out, indent=2))
    _plot(out, RESULTS_DIR / "mi_ceiling.png")
    print(f"\n[save] {RESULTS_DIR / 'mi_ceiling.json'} and .png")


def _plot(out, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True)
    for ax, (key, ch) in zip(axes, out["channels"].items()):
        rows = ch["rows"]
        sig = [r["sigma"] for r in rows]
        ax.plot(sig, [r["cert_auc_ceiling"] for r in rows], "o-", color="C3",
                label="certificate AUC ceiling (provable)")
        ax.plot(sig, [r["worst_case_auc"] for r in rows], "s--", color="C1",
                label="moment-matched worst case (floor for any 2nd-moment cert)")
        ax.plot(sig, [r["gauss_lrt_auc"] for r in rows], "^-", color="C2",
                label="Gaussian-LRT attacker (estimate)")
        es = [(r["sigma"], max(v for v in (r.get("emp_xgb"), r.get("emp_mlp"),
                                           r.get("emp_lora32")) if v is not None))
              for r in rows if r.get("emp_xgb") is not None]
        if es:
            ax.plot(*zip(*es), "D-", color="C0", label="empirical battery max AUC")
        ax.axhline(0.55, color="green", ls=":", lw=1, label="honest bar 0.55")
        ax.axhline(MEANINGFUL_AUC, color="gray", ls="--", lw=1,
                   label=f"'meaningful' ceiling {MEANINGFUL_AUC}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("σ")
        ax.set_title(ch["cell"], fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("attribute recovery AUC (macro-OVR for multiclass)")
    axes[0].set_ylim(0.45, 1.05)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Experiment 11 — DPI/noise-channel certificate vs empirical recovery: "
                 "the provable ceiling is vacuous at operating σ", fontsize=12)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
