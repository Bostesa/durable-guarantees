# durable-guarantees

Empirical tests of whether ML "compliance guarantees" are *durable* — i.e.
whether they survive realistic downstream use — or are merely **training-time
artifacts** that an adapter can dissolve.

## Experiment 1 — Falsification attack on PCRL's compliance certificate

PCRL (Purpose-Conditioned Representation Learning) ships a per-purpose
representation with a **compliance certificate**: the linear R² between the
frozen 64-dim representation `h_p` and a forbidden sensitive attribute is low
(below τ = 0.05). This certificate is checked at training time on the frozen
pipeline.

**Claim under test:** the certificate does *not* survive downstream LoRA
fine-tuning.

**Setup** (`experiments/falsification_attack.py`):

1. Load the audited Adult seed-0 model (`checkpoints/v2_adult_s0/final.pt`,
   `weights_only=False`) and the Adult train data under purpose
   `income_prediction` (`purpose_idx = 0`). The PCRL pipeline stays **frozen**.
2. Compute the frozen 64-dim representation (before the task head) for every
   training sample.
3. **Baseline:** measure the certificate — linear R² and dominant-axis R² —
   between the frozen representation and `sex`, using **PCRL's own certificate
   code** (`pcrl.vision.r2_helper.linear_r2` and
   `pcrl.evaluation.certificates.compute_dominant_axis_r2`). This is the
   certificate at rest; it is low.
4. Attach a **fresh, trainable rank-8 LoRA** (PCRL's own `LoRAAdapter`) strictly
   *on top of* the frozen representation, plus a small linear head, and train
   300 steps to predict `sex`. The attacker only ever touches the certified
   representation — the same per-purpose-LoRA mechanism PCRL itself uses.
5. Every 10 steps, re-measure the certificate on the **LoRA-adapted**
   representation.
6. Save the R²-vs-step curves to `results/falsification_attack.json` and a plot
   to `results/falsification_attack.png` (baseline + τ marked).

**Expected:** baseline R² low, then R² climbs sharply within tens of steps —
the certificate is a training-time-only guarantee.

> Note: this backbone never reaches `per_dim_std >= 0.5` (documented, expected),
> so the frozen representation is partially low-variance. That is not a bug.

### Result (Adult s0, MPS)

| quantity | linear R² | dominant-axis R² | vs τ=0.05 |
|---|---|---|---|
| frozen repr (certificate at rest) | **0.035** | 0.035 | **PASS** |
| + strict-linear LoRA (control, trained) | 0.035 | 0.035 | PASS (invariant) |
| + post-ReLU LoRA (attack, 300 steps) | **0.104** | 0.104 | **FAIL** |

The certificate flips **PASS → FAIL** (≈3× baseline, breaching τ at the first
measured step) once a fresh rank-8 adapter is fit downstream — while the
PCRL pipeline never changes. See `results/falsification_attack.png`.

**Key subtlety (why the attack needs a nonlinearity — not fudging).** Linear
R² and dominant-axis R² are *invariant under any invertible affine map of the
features*. A strictly-linear LoRA on top of the frozen repr is exactly such a
map, so it leaves the certificate pinned at its baseline value forever — we
verified this empirically (the `CONTROL` curve is flat at 0.035 for all 300
steps). The certificate is **blind to linear re-bases**. The breach comes from
the single downstream nonlinearity (ReLU) that every real LoRA stack sits next
to — in PCRL's own backbone each LoRA-adapted `Linear` is followed by
`BatchNorm → ReLU`. The attacker mirrors that exactly (PCRL's own `LoRAAdapter`,
then a ReLU), and we keep the linear (pre-ReLU) output as a logged control so
the invariance stays visible. The takeaway: the certified low-R² geometry is a
property of the *frozen* representation only; one downstream nonlinear adapter
surfaces the nonlinearly-encoded sensitive information the linear certificate
never bounded. The certificate does not survive downstream fine-tuning.

## Experiment 2 — Does a noise-channel erasure module survive the same attack?

Experiment 1 defeated *linear* (LEACE-style) erasure. A **Gaussian noise
channel** is the one erasure approach with a provable information-theoretic
guarantee: `h_noisy = h + N(0, σ²I)` caps the mutual information `I(h_noisy; A)`
no matter how the downstream attacker is built (data-processing inequality).
`experiments/noise_channel_test.py` tests whether that guarantee actually stops
the **same ReLU attacker**, and at what cost to task utility.

- **Stage 1 (σ=0, seeds 0/1/2):** the Experiment-1 breach reproduces tightly —
  attacked linear R² = **0.104 ± 0.001** (baseline 0.035, breaches τ=0.05).
- **Stage 2:** the noise is part of the **frozen** module and is **resampled
  every forward pass** (the attacker faces the *channel*, not a fixed offset it
  could invert). Sweep σ relative to the repr per-dim std (≈0.37); per σ, run the
  full 3-seed ReLU attack, the certificate-at-rest (raw linear R²), and task
  utility (a linear logistic head on `h_noisy` → income).

### Result (Adult s0, MPS) — `results/noise_channel_test.png`

| σ (rel) | task acc (income) | certificate at rest | attacked R² (mean±std) | vs τ=0.05 |
|---|---|---|---|---|
| 0.0  | 0.803 | 0.035 | 0.104 ± 0.001 | **BREACH** |
| 0.25 | 0.801 | 0.034 | 0.087 ± 0.001 | **BREACH** |
| 0.5  | 0.801 | 0.033 | 0.068 ± 0.001 | **BREACH** |
| 1.0  | 0.798 | 0.031 | 0.047 ± 0.001 | **STOPPED** |
| 2.0  | 0.793 | 0.026 | 0.034 ± 0.000 | STOPPED |
| 4.0  | 0.781 | 0.017 | 0.026 ± 0.000 | STOPPED |
| 8.0  | 0.762 | 0.008 | 0.010 ± 0.000 | STOPPED |

*(income majority-class baseline = 0.752)*

**The noise channel survives the attack — and cheaply.** Attacked R² falls
monotonically with σ and drops below τ at **σ ≈ 1.0** (noise std ≈ signal std),
robustly so by σ=2. This is the opposite of the linear-erasure result: the
information-theoretic guarantee holds empirically against the nonlinear attacker
that broke LEACE. The **tradeoff here is favourable** — the leaked sex signal the
attacker exploits is fragile and drowns in noise (0.104 → 0.034 by σ=2), while
income is encoded redundantly and is noise-robust, so task accuracy barely
moves (0.803 → 0.793 at σ=2). Utility only erodes toward the majority baseline at
large σ=8 (0.762). So there is a **wide operating window (σ ≈ 1–2)** that stops
this attacker for ≲1 accuracy point.

Caveat (honest scope): "STOPPED" means *this* rank-8 ReLU attacker is held below
τ; the DPI bound is attacker-agnostic but τ is a specific operating point. The
fragile-signal/robust-task separation is a property of this representation and
attribute, not a guarantee it holds for every dataset. **Experiment 3 tests both
of those caveats directly.**

## Experiment 3 — Is the cheap-durability result robust and general?

Two questions decide whether Experiment 2's result is general or specific to
Adult/sex.

### Test A — stronger attacker (`experiments/stronger_attackers.py`)

At fixed σ=2 on Adult/sex, a battery of attackers tries to recover sex from the
noisy frozen repr: the rank-8 ReLU LoRA (control), a rank-32 LoRA, a deep MLP
probe (PCRL's 256-256 auditor), and an XGBoost probe (PCRL's auditor — imported
and verified to match `attack_acc` exactly, |Δacc|=0.0000).

| attacker | σ=0 R² | σ=0 AUC | σ=2 R² | σ=2 AUC |
|---|---|---|---|---|
| rank-8 ReLU LoRA (control) | 0.104 | 0.685 | 0.034 | 0.575 |
| rank-32 ReLU LoRA | 0.104 | 0.687 | 0.034 | 0.575 |
| MLP probe (256-256) | n/a | 0.686 | n/a | 0.562 |
| XGBoost probe | n/a | 0.683 | n/a | 0.550 |

*(sex majority acc = 0.677; chance AUC = 0.5; τ = 0.05 on R²)*

**Verdict: ATTACKER-ROBUST.** Extra capacity does **not** break through — rank-32
LoRA, deep MLP, and XGBoost recover sex no better than the rank-8 control (max
stronger AUC 0.575 vs control 0.575; all LoRA R²≈0.034<τ; accuracy at majority
for all). "STOPPED" is not attacker-specific.

Honest caveat: at σ=2 a faint residual rank-signal remains (AUC≈0.575, a few SE
above chance) that the R²-certificate scores as stopped and that *no* attacker
exceeds. Closing it to ~chance (AUC≈0.52) needs σ≈8 (income acc 0.79 → 0.76).

### Test B — does the favourable tradeoff generalize? (`experiments/generalization_test.py`)

Same full σ-sweep + ReLU attack (3 seeds) on cells where the sensitive signal may
be as robust as the task signal. **Utility cost is measured as lift over the
majority baseline retained**, not absolute accuracy — these tasks have very high
majority baselines, so a ~2pp absolute drop can erase the entire useful signal.

| cell | attacked R² @σ=0 | σ to stop | task lift over majority: clean → stopped | verdict |
|---|---|---|---|---|
| Adult / income / sex (Exp 2) | 0.104 | σ≈1–2 | +0.051 → +0.042 (82% kept) | **CHEAP** |
| HMDA / underwriting / race | **0.720** | σ=8 | +0.019 → +0.000 (**1% kept**) | **EXPENSIVE** |
| Diabetes / quality_research / age_bucket | 0.101 | σ=2 | +0.000 → +0.000 | **DEGENERATE** |

**Verdict: the cheap-durability result does NOT generalize — it was specific to
Adult/sex.**

- **HMDA/race:** the attack is far more severe (R²=**0.72** at rest vs Adult's
  0.10 — race is robustly, nonlinearly encoded, and the certificate-at-rest of
  0.040 drastically understates it). Stopping it needs σ=8, by which point task
  accuracy has fallen to the majority baseline (0.912 → 0.893 = majority): **~all
  of the representation's task utility is destroyed.** Here the sensitive signal
  is *more* robust than the task signal, so noise cannot separate them.
- **Diabetes/age_bucket:** a degenerate cell — the certificate is **already
  breached at rest** (R²=0.103 > τ, i.e. the frozen repr was never compliant for
  age_bucket), and the linear task head never beats majority (lift ≈ 0), so noise
  looks "free" only because there is no utility to lose.

**Overall:** noise-channel erasure is genuinely attacker-robust (Test A), but its
*cheapness* depends entirely on the sensitive signal being more fragile than the
task signal. When they are comparably robust (HMDA/race), durability costs all the
utility. The Adult/sex win is the exception, not the rule.

### Layout

```
experiments/falsification_attack.py   # Experiment 1: linear erasure is defeated
experiments/noise_channel_test.py     # Experiment 2: noise-channel durability vs utility (Adult)
experiments/stronger_attackers.py     # Experiment 3 Test A: attacker-robustness at fixed σ
experiments/generalization_test.py    # Experiment 3 Test B: generalization to HMDA / Diabetes
utils/pcrl_io.py                       # PCRL-specific glue (imports the read-only PCRL repo)
results/                               # JSON curves + plots
requirements.txt
```

> macOS note: XGBoost and PyTorch each bundle an OpenMP runtime; loading both in
> one process segfaults. `stronger_attackers.py` imports `xgboost` before torch
> and sets `KMP_DUPLICATE_LIB_OK=TRUE` + `OMP_NUM_THREADS=1` (run it with those
> env vars).

The PCRL repo is **never modified**. `utils/pcrl_io.py` adds it to `sys.path`
(default `/Users/nathansamson/PCRL`, override with the `PCRL_ROOT` env var) and
imports its code, including mirroring the `lora_target` auto-detection from
`scripts/crosspurp/run_eval_multi.py`.

### Run

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python experiments/falsification_attack.py   # Experiment 1
.venv/bin/python experiments/noise_channel_test.py     # Experiment 2
```

Runs locally on **MPS/CPU** only — never AWS.
