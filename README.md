# durable-guarantees

Empirical tests of whether ML "compliance guarantees" are *durable* — i.e.
whether they survive realistic downstream use — or are merely **training-time
artifacts** that an adapter can dissolve.

> **Bottom line (after the Experiment 5 honest re-audit).** PCRL's linear R²
> certificate is blind to nonlinear recovery, and that blindness propagated into our
> *own* "durable" verdicts. Re-measured with an honest XGBoost/MLP probe, **only 3 of
> 21 R²-"stopped" results survive — all of them isotropic noise at high σ.** Every
> targeted projection (LEOPARD-MMD, HSIC) and the training-time HSIC encoder leak the
> attribute at near-unprotected levels to a tree probe. The single robust lesson:
> **a compliance guarantee is only as strong as the strongest attacker used to certify
> it; only an attacker-agnostic mechanism (noise / a data-processing bound) genuinely
> hides an attribute, and only at a real utility cost.** Read the experiments below in
> order — several "wins" are explicitly overturned by later sections.

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

> ⚠️ **Corrected by Experiment 5 (honest re-audit).** The "STOPPED from σ≈1" and
> "cheap, ≲1 accuracy point" claims above were measured with the R²/linear
> certificate. Re-measured with an **XGBoost** probe, sex is still recovered at
> **0.59 AUC at σ=1 and 0.55 AUC at σ=2** — *not* stopped. The honest stop point
> (XGB+MLP driven to ~chance) is **σ≈4**, where income lift falls from +0.051 to
> **+0.029 (≈57% kept)**, and only σ=8 (lift +0.010, ~20% kept) is comfortably
> clear. The noise channel *does* genuinely survive — it is the only method that
> does — but it is **meaningfully more expensive** than the R² certificate implied.
> See Experiment 5 below.

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

> **Note (Experiment 5).** This caveat was correct and is the seed of the later
> correction: the σ=2 "STOPPED" verdict is relative-to-control (no *stronger*
> attacker beats the rank-8 control), but the **absolute** residual (XGB AUC ≈0.55
> at σ=2) means sex is not actually hidden at σ=2. The honest re-audit makes this
> precise — durability requires σ≈4–8, not σ=2.

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

By the R² certificate the targeted projections appear to **stop** the attack — they
drop attacked R² to ~0.036 (well below τ) — while keeping most of the income lift
(best 83%). This looked like the cleanest win in the project.

> ⛔ **Overturned by Experiment 5 — this was the worst case of R²-blindness.** The
> projections do **not** hide sex from an honest probe. Re-measured with XGBoost,
> the projected representations leak sex at **~0.68 AUC — identical to the
> *unprotected* representation (σ=0 XGB AUC = 0.683)**, across *every* family and
> rank (MMD and HSIC, r=1…32). In other words, the MMD/HSIC projections removed
> only the linear/kernel-aligned subspace the R² certificate can see, and did
> **essentially nothing** to a gradient-boosted attacker. The "targeted erasure
> wins cheaply on Adult/sex" result was pure certificate theater. The **only**
> Adult/sex method that survives honest measurement is isotropic **noise at σ≥4**
> (see Experiment 5) — because noise is the only one with an attacker-agnostic
> (information-theoretic) guarantee; the projections have none.

### The characterization (R²-certificate view — see Experiment 5 for the honest correction)

| cell | attack@rest R² | targeted erasure stops + keeps utility? |
|---|---|---|
| Adult / income / sex (**fragile**) | 0.104 | R² said **YES** (HSIC r=16) — **but XGB still recovers sex at 0.68 AUC; honest NO** |
| HMDA / underwriting / race (**robust**) | 0.720 | **NO** — best projection only R²≈0.17 (3×τ); none stop |

The original lesson — "targeted erasure works iff the protected signal is
low-rank-separable from the task" — was an artifact of the linear certificate. The
**corrected** lesson is stronger and simpler: a *targeted* eraser (projection, HSIC,
MMD) only removes what its own statistic can see, so it never beats the honest probe;
**only an attacker-agnostic mechanism (isotropic noise / a DPI bound) genuinely
hides the attribute**, and even then only at a σ that costs real utility. The cost is
fundamental, and it must be paid by an attacker-agnostic guarantee — not bolted on by
a probe-shaped projection.

## Experiment 4 — Stage 2 (decisive): can a *training-time* intervention pay the bill?

Stage 1 said the cost must be paid at training time. Stage 2
(`experiments/training_time_erasure.py`) tests whether it actually *can* be. We
train a two-component model M = D(E(·)) **jointly** with task loss + λ·HSIC(E(·),
race), so E is shaped to be task-compatible instead of bolted onto a frozen rep.
After training E is frozen and hit with the **same** ReLU attacker (3 seeds). To
avoid rigging the test, E runs at three power levels and λ sweeps the trade-off:

| method (best λ shown) | attacked R² | task lift | stops attack AND keeps utility? |
|---|---|---|---|
| *post-hoc* best (HSIC proj) | 0.166 | +0.012 | no (breaches τ) |
| *post-hoc* noise σ=8 | 0.005 | +0.000 | no (utility destroyed) |
| train-time: **affine on h** (LEACE-init), λ=10⁴ | 0.133 | +0.018 | no (breaches τ) |
| train-time: **MLP on h**, λ=10⁴ | 0.087 | +0.019 | no (breaches τ) |
| train-time: **MLP on raw x** (from scratch), λ=10³ | 0.011 | +0.000 | stops, but utility gone |
| train-time: **MLP on raw x** (from scratch), **λ=100** | **0.023** | **+0.018** | apparent **YES** — *overturned by hardening, see below* |

**Apparent verdict (rank-8 ReLU attacker / R² certificate): YES.** Measured the same
way as every prior experiment, the from-scratch encoder at λ=100 drives the attacked
race R² to 0.023 (below τ) while keeping +0.018 of the +0.019 task lift (92%) — the
only point in the "left of τ AND high utility" region that was empty for every
Stage-1 method. **But this verdict does not survive a stronger attacker — see the
hardening test below, which overturns it.**

Crucially, **neither** training-time fix of the *frozen* PCRL repr works: an affine
E (even LEACE-initialized + jointly trained) plateaus at R²≈0.13, and a nonlinear
MLP on h plateaus at R²≈0.087 — both still breach τ. The race↔task entanglement is
baked into `h`; no transform *of `h`*, trained or not, removes it. Only learning a
fresh representation from raw features — where the encoder is free to find a
task-good basis that never encodes race — pays the bill (λ must be tuned: λ=100
wins, λ≥10³ over-erases and zeroes utility).

## Experiment 4 — Stage 2 *hardening*: the from-scratch "win" does not survive a real attacker

The Stage-2 headline rested on one training seed and one rank-8 ReLU attacker — the
same R²-certificate-shaped instrument that Experiment 1 already proved is blind to
nonlinearly-encoded signal. `experiments/hardening_test.py` stress-tests it two ways.

**Part A — error bars (3 training seeds).** The result is *reproducible*: all three
seeds land below τ with essentially identical utility.

| train seed | attacked R² (rank-8) | task lift |
|---|---|---|
| 0 | 0.0226 | +0.0175 |
| 1 | 0.0187 | +0.0177 |
| 2 | 0.0194 | +0.0174 |
| **mean ± std** | **0.0202 ± 0.0017** | **+0.0175 ± 0.0001** |

So it is *not* a lucky seed. But Part A only confirms the **rank-8 R² number** is stable — it says nothing about whether that number measures the right thing.

**Part B — stronger attackers (vs the frozen from-scratch encoder, seed 0).** Replace
the rank-8 ReLU probe with the full Experiment-3 battery. The R²-certificate attackers
still read "stopped" — but an honest nonlinear classifier recovers race outright:

| attacker | attacked R² | race acc | race AUC | verdict |
|---|---|---|---|---|
| rank-8 ReLU LoRA | 0.023 | 0.645 | 0.55 | ≤ τ (stops) |
| rank-32 ReLU LoRA | 0.021 | 0.645 | 0.56 | ≤ τ (stops) |
| **MLP probe (256-256)** | n/a | **0.730** | **0.74** | **recovers race** |
| **XGBoost probe** | n/a | **0.860** | **0.91** | **recovers race** |
| *ref:* MLP on raw `h` | n/a | 0.976 | 1.00 | (race present) |
| *ref:* XGB on raw `h` | n/a | 0.984 | 1.00 | (race present) |

*(race majority acc = 0.645; chance AUC = 0.5; the reference rows confirm the probes
can find race when it is present, so they are not broken.)*

**Verdict: the constructive Stage-2 result DOES NOT hold.** Against the rank-8 ReLU /
R² certificate the from-scratch encoder looks durable; against a gradient-boosted
probe, race is recovered at **86% accuracy (0.91 AUC)** from the *same* encoder. The
HSIC training penalty drove down the kernel-dependence statistic and the linear R²,
but left race trivially decodable by a tree ensemble. The apparent win was the
certificate's blindness — the exact failure mode Experiment 1 falsified — reappearing
in our *own* constructive result. The from-scratch encoder did remove *some* race
(XGB AUC 1.00→0.91), but nowhere near hiding it.

This completes the arc, and it is more honest and more decisive than the apparent
Stage-2 win: **the durability cost on HMDA/race is fundamental even at training time,
once you measure it with an honest attacker.** Race is entangled with the loan
decision itself, so any underwriting-useful representation keeps enough signal for a
strong probe to recover race; and pushing the HSIC penalty harder (λ ≥ 10³) only
trades that leak for zero utility. The durability/utility frontier on HMDA/race is not
escaped by post-hoc erasure, by trained transforms of `h`, **or** by a from-scratch
encoder — the bill is fundamental, and the lesson loops back to Experiment 1: a
guarantee is only as good as the strongest attacker you measure it against, and an
R²/linear certificate is not that attacker. (Honest caveats: one HSIC kernel and one
λ=100 setting; a different training objective — adversarial, or HSIC over a richer
kernel family — might do better, but the burden of proof now sits with it, measured
against XGBoost, not against the linear certificate.)

## Experiment 5 — Honest re-audit: how many "stopped" verdicts were real?

The hardening test proved the linear R²/dominant-axis certificate is blind to
nonlinear recovery. That indicts **every** R²-only "stopped" verdict in Experiments
2–4. `experiments/honest_reaudit.py` re-measures all 21 of them with the honest probe
battery (XGBoost + deep MLP, PCRL-auditor configs). A result "survives" only if
**both** strong probes are driven to ~chance (max AUC ≤ 0.55; chance = 0.5).

**Master table (every R²-"stopped" verdict, honest re-measurement):**

| experiment | cell | method | old R² (verdict) | XGB AUC | MLP AUC | honest agrees stopped? |
|---|---|---|---|---|---|---|
| E2 noise | adult/sex | noise σ=1.0 | 0.047 (stop) | 0.589 | 0.599 | **NO — leaks** |
| E2 noise | adult/sex | noise σ=2.0 | 0.034 (stop) | 0.550 | 0.562 | **NO — leaks** |
| E2 noise | adult/sex | noise σ=4.0 | 0.026 (stop) | 0.534 | 0.547 | ✅ yes |
| E2 noise | adult/sex | noise σ=8.0 | 0.010 (stop) | 0.514 | 0.518 | ✅ yes |
| E4S1 proj | adult/sex | LEOPARD-MMD r=1…16 | ~0.036 (stop) | **0.681** | 0.55–0.59 | **NO — leaks** |
| E4S1 proj | adult/sex | HSIC r=1…32 | ~0.036 (stop) | **0.683** | 0.54–0.58 | **NO — leaks** |
| E4S1 noise | adult/sex | noise σ=8.0 | 0.010 (stop) | 0.514 | 0.518 | ✅ yes |
| E4S1 noise | hmda/race | noise σ=8.0 | 0.005 (stop) | 0.549 | 0.551 | **NO — leaks** (and utility already 0) |

*(projection rows condensed — all 9 ranks behave identically; full per-rank numbers
in `results/honest_reaudit.json`. Reference: the **unprotected** Adult/sex repr has
XGB AUC 0.683 — so the projections leak sex at the unprotected level.)*

**Verdict: only 3 of 21 R²-"stopped" results survive honest measurement — and all 3
are isotropic noise at high σ.** The 18 that collapse split into two groups:

- **Targeted projections (all of them).** LEOPARD-MMD and HSIC projections leak sex
  to XGBoost at ~0.68 AUC — *the unprotected level*. They removed only what their own
  linear/kernel statistic could see and nothing a tree exploits. Every "smart erasure
  wins" verdict was R²-blindness.
- **Low-σ noise.** σ=1 and σ=2 — the "wide cheap operating window" of Experiment 2 —
  leak at 0.55–0.60 AUC. Honest durability on Adult/sex starts at **σ≈4** (lift +0.029,
  ~57% kept), not σ=1.

The pattern is sharp and explains *which* mechanisms are real: **only the noise
channel has an attacker-agnostic (data-processing-inequality) guarantee, and only the
noise channel survives an attacker-agnostic probe.** Every probe-shaped eraser (R²
projection, MMD, HSIC, training-time HSIC) only fools the matching probe. This is the
central finding in its strongest form: **a compliance guarantee is exactly as strong
as the attacker used to certify it, and a linear/R² certificate certifies almost
nothing** — here, 0 of the post-hoc *targeted* methods and only the bluntest, most
expensive noise settings. (Honest scope: the 0.55 AUC bar is generous; the survivors
still carry a small residual 0.51–0.53 AUC, i.e. "≈chance," not "provably zero." And
HMDA/race has *no* row that is both honest-stopped and utility-positive at all.)

## Experiment 6 — Can targeted (anisotropic) noise buy honest durability cheaper?

Experiment 5 left isotropic noise as the only honest stopper, and an expensive one. But
noise works by *destroying* information (a DPI bound), so a natural question
(`experiments/targeted_noise.py`): can we destroy the sex information **more surgically**
— pour noise only into the directions/dims that carry sex, sparing task-only directions
— and pay less utility for the same XGBoost-proof protection? Two targeted variants vs the
isotropic floor, all measured with XGBoost + MLP (never R²):

1. **Subspace noise** — Gaussian noise confined to the learned rank-r *most-sex-carrying
   subspace* (the same HSIC subspace the failed projection used — but we **destroy** it with
   noise instead of deleting it). Sweep r, σ.
2. **Importance noise** — Gaussian noise only on the top-k repr dims **XGBoost itself uses**
   to predict sex (axis-aligned targeting matched to the honest attacker). Sweep k, σ.

**Head-to-head — cheapest config reaching honest durability (XGB *and* MLP ≤ 0.55 AUC):**

| method | best honest config | XGB AUC | MLP AUC | income lift | % of clean |
|---|---|---|---|---|---|
| isotropic | σ=4 | 0.534 | 0.546 | +0.0289 | 57% |
| **subspace (HSIC)** | r=16, σ=16 | 0.536 | 0.548 | **+0.0434** | **86%** |
| importance (XGB-dims) | — none reached ≤0.55 — | 0.642 | 0.550 | (leaks) | — |

*(clean income lift = +0.0508; chance AUC 0.5.)*

**Mixed verdict — one targeted variant genuinely helps, the other leaks like the projections.**

- **Subspace noise is a genuine (but modest, fragile) improvement.** Confining heavy noise
  (σ=16) to the 16-dim most-sex-carrying subspace drives XGB to 0.536 while keeping **86%**
  of income lift — vs isotropic's **57%** at matched protection. Unlike *deleting* that
  subspace (the Exp-5 projection, which leaked at XGB 0.68), *noising* it with fresh
  per-sample noise destroys the tree-recoverable signal irreducibly, while income — which
  lives in the orthogonal complement — is barely touched (lift is nearly flat in σ). This is
  the **first method in the project to beat blunt noise** on the durability/utility frontier.
- **Importance noise leaks like the projections.** Noising the top-k axis-aligned dims one
  XGB used does *not* protect: a retrained tree simply routes around them and recovers sex at
  0.64–0.65 AUC even at k=48, σ=16 (and barely dents income, ~98% kept — because it barely
  dents sex). Axis-aligned targeting fails for the same reason projections did: the
  tree-exploitable signal is spread across directions/interactions the target misses.

**Honest caveat — the win is bar-dependent and cannot reach near-chance.** The subspace
advantage exists only at the *generous* 0.55 bar. The MLP residual for subspace noise
plateaus around **0.54–0.55**; at a stricter **0.53** bar **no** targeted config qualifies,
and only blunt isotropic noise (σ=8) drives *both* probes to near-chance — at 20% utility.
So targeted subspace noise buys a *cheaper approximate* protection (down to ~0.54 AUC at 86%
utility), but **for true near-chance erasure, blunt isotropic noise remains the floor.** The
direction of information-destruction can be made surgical; the *amount* needed for near-zero
recovery still costs what isotropic noise costs.

## Experiment 7 — Sealed channel vs the footprint: does race leak through the output?

A natural escape from the wall is *architecture*: split the model into a **public**
channel P (race must be unrecoverable), a **sealed** channel S (holds race, used by the
task head but never exposed), and a head D([P,S]). The footprint hypothesis says this
cannot work — because the decision depends on race via S, the **output** leaks race even
if P is perfectly hidden. `experiments/sealed_channel.py` tests it on both cells,
probing race/sex from the public channel **and** from the task outputs with XGBoost+MLP
(never R²). We give the architecture its best shot: P is hidden with the only mechanism
that works (a BatchNorm'd noise channel — adversarial gradient-reversal **failed** to
hide P from a tree, 0.997 AUC, the same probe-shaped-erasure lesson).

**Race/sex recovery (max of XGB,MLP AUC) as the representation is progressively hidden:**

| model (public channel) | HMDA: P | HMDA: **output** | Adult: P | Adult: **output** |
|---|---|---|---|---|
| plain (no scrub) | 1.00 | 0.65 | 1.00 | 0.83 |
| adversarial scrub | 1.00 | 0.67 | 1.00 | 0.76 |
| noised, public-only | 0.55 | 0.53 | 0.69 | 0.73 |
| **noised + sealed S** | **0.55** | **0.55** | **0.57** | **0.73** |
| *label↔attribute floor* | — | 0.51 | — | 0.60 |

*(sealed S holds the attribute by construction: HMDA 0.99, Adult 0.95 AUC. Task lift kept:
HMDA +0.023, Adult +0.131.)*

**Verdict — the footprint is real, and its size = how much the task actually uses the
attribute (≈ the label↔attribute correlation):**

- **Adult/sex — FOOTPRINT CONFIRMED (strong).** As the public channel is hidden from 1.00
  down to **0.57**, the output leak does **not** follow — it plateaus at **0.73** (XGB
  0.71). Income genuinely depends on sex-correlated signal (the true label alone leaks sex
  at 0.60), so an accurate income decision *must* encode sex; the sealed channel even
  *gains* utility (+0.131 vs +0.121) bundled with that leak. Sealing the internal channel
  is irrelevant — you cannot make the decision without the decision carrying sex.
- **HMDA/race — no real footprint, for the opposite reason.** Here the output leak
  **follows** the hidden representation down to **0.55 ≈ chance** (label↔race floor 0.51),
  and the sealed channel buys **zero** utility (−0.0005). The HMDA loan label barely
  encodes race, so the achievable task lift doesn't need it — a single noised channel
  hides race in *both* representation and output while keeping utility, and sealing adds
  nothing.

**The unifying answer to "can XGBoost recover the attribute from the task output alone?":
yes, to exactly the extent the task label is correlated with the attribute.** That
correlation is an irreducible floor on output leakage for any *useful* model — you can
scrub every internal representation to chance, but the decision itself betrays the
attribute whenever the decision genuinely depends on it. Sealing relocates the signal
(public→sealed→output); it never removes it. (Side result, consistent with Exp 5–6:
adversarial scrubbing fooled its own discriminator but not a fresh tree; only the noise
channel hid the public representation — and on Adult only the *sealed* architecture, which
frees P from carrying sex, pushed P near the bar.)

### Layout

```
experiments/falsification_attack.py   # Experiment 1: linear erasure is defeated
experiments/noise_channel_test.py     # Experiment 2: noise-channel durability vs utility (Adult)
experiments/stronger_attackers.py     # Experiment 3 Test A: attacker-robustness at fixed σ
experiments/generalization_test.py    # Experiment 3 Test B: generalization to HMDA / Diabetes
experiments/smart_erasure.py          # Experiment 4: targeted (MMD/HSIC) erasure vs noise on HMDA/race
experiments/smart_erasure_control.py  # Experiment 4 positive control: same erasers on Adult/sex + contrast
experiments/training_time_erasure.py  # Experiment 4 Stage 2: training-time two-component erasure
experiments/hardening_test.py         # Experiment 4 Stage 2 hardening: seeds + stronger attackers (overturns the win)
experiments/honest_reaudit.py         # Experiment 5: re-audit all R²-"stopped" verdicts with XGBoost/MLP (3/21 survive)
experiments/targeted_noise.py         # Experiment 6: anisotropic vs isotropic noise (subspace noise beats blunt, at the 0.55 bar)
experiments/sealed_channel.py         # Experiment 7: sealed-channel / footprint test (output leaks attr ∝ task's use of it)
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
