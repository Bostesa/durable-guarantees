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

## Experiment 4 — Can a *smarter* erasure beat blunt noise on HMDA/race?

HMDA/race is the hard case from Experiment 3: race is robustly, nonlinearly
encoded (attack R²=0.72 at rest while the linear certificate reads 0.040), and the
only noise level that stops the attack (σ=8) destroys all task utility. Question:
can a *targeted* erasure remove the nonlinear race signal **without** that utility
collapse? `experiments/smart_erasure.py` tests two non-noise erasure modules as a
frozen transform before the same ReLU attacker (3 seeds), all task-agnostic:

1. **LEOPARD-style** — a learned orthogonal **rank-r projection** that removes the
   subspace minimizing the **class-conditional MMD** of race (Gaussian kernel;
   higher moments, not just LEACE's linear mean). Swept over r.
2. **HSIC-style (Obliviator)** — the same rank-r projection, removing the subspace
   minimizing **HSIC(repr, race)** (independence target). Swept over r.

### Result (HMDA/underwriting/race) — `results/smart_erasure.png`

| method | attacked R² | task lift over majority | stops attack (R²≤τ)? |
|---|---|---|---|
| no erasure (frozen repr) | 0.720 | +0.019 | no |
| LEACE (linear) | 0.651 | +0.019 | no |
| **noise σ=8 (baseline)** | **0.005** | **+0.000** | yes — but kills utility |
| LEOPARD-MMD proj, r=8 | 0.393 | +0.019 | no |
| LEOPARD-MMD proj, r=16 | 0.168 | +0.010 | no |
| LEOPARD-MMD proj, r=32 | 0.171 | +0.012 | no |
| HSIC proj, r=8 | 0.386 | +0.019 | no |
| HSIC proj, r=16 | 0.174 | +0.010 | no |
| HSIC proj, r=32 | 0.166 | +0.012 | no |

**Verdict: the cost is FUNDAMENTAL on this cell — no smart erasure beats noise.**
No targeted projection (LEOPARD-MMD or HSIC, any rank up to 32 of 64 dims) gets the
attacked R² below τ=0.05; the best they reach is ~0.17 (still 3× τ), and pushing
that far already starts eroding task lift (+0.019 → +0.010). LEACE barely dents it
(0.72 → 0.65). **Only isotropic noise stops the attack, and it costs all utility.**

Why: the race signal is **not** in a low-rank subspace separable from the task. The
two objectives (MMD higher-moment matching and HSIC independence) find essentially
the *same* directions and plateau at R²≈0.17 — after removing the ~16 most
race-discriminative dimensions, a nonlinear attacker still reconstructs race from
whatever remains. Removing more linear directions stops helping on race while it
keeps hurting the task. On the frontier plot, the entire "stop the attack **and**
keep utility" region (left of τ, high on y) is **empty** — every method that
crosses τ sits at zero utility. The robust-nonlinear-encoding + task-entanglement
that made HMDA/race the hard case is not something a cleverer *frozen, post-hoc*
erasure removes; it would have to be addressed in how the representation is trained.

### Positive control — the SAME erasers on the EASY cell (Adult/sex)

To turn that negative into a characterization, `experiments/smart_erasure_control.py`
runs the identical battery on **Adult / income_prediction / sex**, where the leaked
signal is fragile (Experiments 1-3). Result (`results/smart_erasure_adult.png`):

| method | attacked R² | task lift | stops attack? |
|---|---|---|---|
| no erasure (frozen) | 0.104 | +0.051 | no |
| LEACE (linear) | 0.095 | +0.049 | no |
| noise σ=1 | 0.047 | +0.046 | yes (91% lift kept) |
| LEOPARD-MMD proj, r=16 | 0.036 | +0.032 | **yes (63% kept)** |
| HSIC proj, r=16 | 0.037 | +0.042 | **yes (83% kept)** |

Here the targeted projections **do** stop the attack — they drop attacked R² to
~0.036 (well below τ) — while keeping most of the income lift (best 83%). Removing a
**low-rank, task-separable sex subspace** suffices, because the sex signal is
fragile and concentrated. (Honest nuance: lift varies with rank — e.g. r=1 removes a
direction shared by sex *and* income and keeps only 1% — but good ranks clearly
reach the stop-and-keep region. Noise is also cheap here, σ=1 keeps 91%.)

### The characterization

| cell | attack@rest R² | targeted erasure stops + keeps utility? |
|---|---|---|
| Adult / income / sex (**fragile**) | 0.104 | **YES** — HSIC r=16, R²=0.037≤τ, 83% lift kept |
| HMDA / underwriting / race (**robust**) | 0.720 | **NO** — best projection only R²≈0.17 (3×τ); none stop |

**Targeted erasure works iff the protected signal is low-rank-separable from the
task.** When the attribute is fragile and concentrated (Adult/sex), an MMD/HSIC
projection removes it cheaply. When it is robustly, nonlinearly diffused and
entangled with the task (HMDA/race), no frozen post-hoc erasure — blunt or smart —
can remove it without destroying utility; the cost is fundamental and has to be paid
at representation-training time, not bolted on afterward.

### Layout

```
experiments/falsification_attack.py   # Experiment 1: linear erasure is defeated
experiments/noise_channel_test.py     # Experiment 2: noise-channel durability vs utility (Adult)
experiments/stronger_attackers.py     # Experiment 3 Test A: attacker-robustness at fixed σ
experiments/generalization_test.py    # Experiment 3 Test B: generalization to HMDA / Diabetes
experiments/smart_erasure.py          # Experiment 4: targeted (MMD/HSIC) erasure vs noise on HMDA/race
experiments/smart_erasure_control.py  # Experiment 4 positive control: same erasers on Adult/sex + contrast
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
