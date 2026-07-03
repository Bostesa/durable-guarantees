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
>
> **Solution half (Experiment 8).** The cost is not arbitrary — it is *predictable*. The
> task output leaks the attribute to the extent the task **label** correlates with it
> (the footprint law, now confirmed on 6 cells across 3 datasets). So a one-number,
> **model-free** diagnostic — XGBoost AUC recovering the attribute from the task label
> alone — predicts *in advance* whether durable removal is cheap (predictor ≤ 0.55) or
> impossible at positive utility (predictor > 0.55). In the achievable case a from-raw-x
> **noised channel** delivers it: on HMDA it hides race from XGBoost + MLP + rank-32 LoRA
> on **both** representation and output (1.00 → ~0.53) at **96% utility**, over 3 seeds.
> The honest story is problem + solution: *certificates lie and hiding fails* (problem);
> *durable removal is achievable and verifiable in the label-independent case, and a
> diagnostic predicts when* (solution).
>
> **Experiment 9 hardens the solution.** The diagnostic survives de-circularization — it
> predicts a *continuous* removal cost (Pearson r≈0.80, Spearman ρ≈0.85 over 20 natural
> cells); the cost landscape is a **steep ramp, not a cliff** (confirmed causally with
> constructed cells that tune only the label↔attribute coupling); and **surgical
> (HSIC-subspace) noise beats blunt isotropic noise at matched full-battery protection on
> every cell tested** (89/88/81/53% vs 57/1/36/3% utility kept, LoRA-32 vetted) — blunt
> noise is not the post-hoc floor. The floor that remains is the footprint law.
>
> **Experiment 10: the surgery is better built in than bolted on.** Training the
> surgical channel **end-to-end** — a learned attribute subspace, destroyed by noise,
> with HSIC shaping pushing the attribute into it — beats post-hoc surgical noise on
> every cell (104/100/90% vs 88/81/53% utility kept at full-battery ≤0.55, 3 seeds),
> collapsing the representation-channel cost to ≈0 across the tested band. The λ=0
> ablation shows the shaping term earns it, the rank-32 LoRA catches configs XGB+MLP
> miss (first LoRA-only breaches), and the **output** still leaks the attribute at the
> footprint floor (0.71/0.61 on above-the-wall cells) — the wall stands.
>
> **Experiment 11 (theory check): the DPI bound cannot be featured as a numeric
> proposition.** The distribution-free MI certificate for the Gaussian channel is
> **vacuous at every operating σ** (ceiling = 1.0 AUC where the battery reads ~0.53),
> becomes meaningful only far past the utility cliff, and provably cannot be tightened
> from second moments alone. The DPI stays as the qualitative *reason* noise survives;
> the guarantee itself is, and remains, empirical. A channel-aware Gaussian-LRT
> attacker also sits 0.02–0.05 *above* the battery at operating σ (HMDA σ=8: 0.574 —
> over the 0.55 bar), sharpening the honest scope of Exp 8/10: the verdicts are
> battery-relative. Experiment 12 picks the two CelebA vision cells by measurement:
> **Smiling→Young (predictor 0.522, low) and Attractive→Young (0.734, high)**.
>
> **The final block (Experiments 13–15) closes the loop.** The Gaussian-LRT joins the
> battery as a standard member and the certification splits into **two explicit tiers**
> (Tier 1 black-box = XGB+MLP+LoRA-32; Tier 2 informed = +LRT). Re-certified at both
> tiers (Exp 14): Tier 2 costs real σ and utility on the blunt e2e channels (HMDA
> loan_decision 95%→55%, loan_amount_band 57%→22%), and **every surgical channel —
> post-hoc and end-to-end — holds Tier 1 but is demolished at Tier 2** (LRT 0.66–0.94):
> with no noise in the surviving subspace, an informed adversary reads the complement
> at full strength, and no in-family σ-escalation fixes it. Surgical destruction is a
> black-box-tier method; only full-rank noise buys the informed tier. The CelebA run
> (Exp 13) confirms the LOW cell prediction (Smiling→Young durable removal is cheap:
> 99% utility, everything ≤0.55) but **falsifies the HIGH cell's numeric gauge**:
> Attractive→Young cost 0.19 (predicted ≈1) with output floor ≈0.53 (predicted 0.73) —
> because PCRL's vision representation only supports 0.565 task accuracy, and the
> *achieved output's* coupling with Young (0.536–0.540) matches the measured floor
> exactly. The footprint law survives in its causal (achieved-output) form; the
> label-based predictor is an upper bound that is loose off the high-accuracy regime.
> Hardening to 5 seeds (Exp 15) reproduces every headline except one, flagged: the
> hard-cell e2e-surgical winner (r=32 σ=8) **flips to breaching at 5 seeds** (rep max
> 0.552, 3/5 seeds over 0.55 via LoRA; all hold at 0.57); the robust operating point
> is σ=16 — ≤0.55 on all 5 seeds at **85%±2** utility (vs the quoted 90% at the bar's
> edge). Exp-8's HMDA headline hardens cleanly (95% at 5 seeds).

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

## Experiment 8 — The diagnostic: does label↔attribute correlation predict when durable removal is possible? (the *solution* half)

Experiments 1–7 are the **problem**: R²/linear certificates are blind to nonlinear
recovery (Exp 1, 5); every probe-shaped eraser only fools its own probe (Exp 4, 5);
only isotropic noise (a data-processing bound) genuinely hides an attribute (Exp 5),
and even then the **output** leaks it to the extent the task uses it (Exp 7 footprint
law). Exp 7 measured that footprint on two cells — two points. Experiment 8
(`experiments/diagnostic.py`) turns it into a **predictive diagnostic** and tests the
constructive case rigorously.

The construction under test is the one mechanism that survives honest measurement,
trained *end-to-end from raw features x* (not bolted onto the frozen PCRL rep): a
single **noised channel** `E_pub(x) → BatchNorm → + N(0,σ²) → head`, with fresh
per-forward channel noise. Everything is measured with **XGBoost + deep MLP +
rank-32 LoRA — never R²**. Candidate cells were scouted with
`experiments/_scout_predictor.py` and 6 were chosen to span the predictor axis and
cross the 0.55 bar, balanced across all three datasets.

### Part A — the constructive result, hardened (HMDA/underwriting/race)

3 training seeds, full attacker battery on the **public representation P** *and* the
**task output** (logits):

| variant | target | XGBoost | MLP (256-256) | rank-32 LoRA | task lift |
|---|---|---|---|---|---|
| clean σ=0 | public P | 1.000 | 1.000 | 1.000 | **+0.0223 ± 0.0003** |
| clean σ=0 | **output** | 0.687 | 0.685 | 0.602 | |
| **noised σ=8** | public P | **0.532 ± 0.005** | **0.527 ± 0.003** | **0.528 ± 0.003** | **+0.0213 ± 0.0004** |
| **noised σ=8** | **output** | **0.530 ± 0.003** | **0.535 ± 0.007** | **0.534 ± 0.004** | |

*(race majority acc 0.645; 5-class macro-OVR AUC; chance 0.5; bar ≤ 0.55; label↔race
predictor = 0.514.)*

**Verdict: DURABLE — and it holds up.** The noised channel drives race recovery from
**1.00 (rep) / 0.69 (output)** down to **~0.53 on both channels**, below the 0.55 bar
for *every* attacker including the rank-32 LoRA and the tree that broke every earlier
"win", while keeping **96%** of the loan-decision lift, reproducibly over 3 seeds
(tight error bars). This is the **first result in the project that survives the full
honest battery *and* keeps utility.** It works precisely because HMDA's loan label
barely needs race (predictor 0.514) — so the encoder can find a task-good basis that
drops race without touching the decision. (Honest scope: 0.55 is a generous bar; ~0.53
is near-chance for a 5-class attribute, not provably zero. There is headroom — the
sweep keeps utility flat, +0.019 to +0.024, all the way to σ=12 where P falls to 0.522.)

### Part B — the diagnostic is predictive

**Predictor** = XGBoost AUC recovering the attribute from the **task label alone** (one
1-feature fit, *no model trained*). **Outcome** = utility cost of driving attribute
recovery ≤ 0.55 (fraction of clean task lift lost), for the representation, and for
honest durable removal (representation **and** output):

| cell | predictor | clean lift | cost: hide rep | cost: hide rep **+ output** (durable) |
|---|---|---|---|---|
| diabetes / gender / medication_change | 0.509 | +0.448 | 0.00 (σ=2) | **0.00** |
| hmda / race / loan_decision | 0.514 | +0.022 | −0.06 (σ=6) | **−0.06** |
| diabetes / gender / primary_diagnosis | 0.543 | +0.137 | 0.07 (σ=2) | **0.20 (σ=4)** |
| hmda / race / loan_amount_band | 0.584 | +0.409 | 0.44 (σ=12) | **unreachable** |
| adult / sex / income | 0.603 | +0.142 | unreachable | **unreachable** |
| adult / sex / occupation_group | 0.676 | +0.716 | unreachable | **unreachable** |

**Verdict: the predictor predicts durability cost, with a sharp threshold at the bar.**
Below 0.55, durable removal is free-to-cheap — both channels hidden for ≤ 0.20 of the
lift (often 0). Above 0.55, the **output cannot be driven ≤ 0.55 at any positive
utility** (durable removal would require collapsing the task → cost 1.0), and for the
two highest cells even the *representation* cannot be hidden within σ ≤ 12. A single
XGBoost fit on the task label — computed **before training anything** — tells you which
side of the wall a cell is on. (Cells with tiny clean lift, e.g. hmda/race/loan_decision
+0.022, have noisier cost estimates; the negative cost is noise-as-regularizer.)

### Part C — the footprint law confirmed across 6 cells

Output-leak **floor** (attribute recovered from the logits once the representation is
hidden) vs the label↔attribute predictor tracks **y = x**:

| cell | predictor | output-leak floor |
|---|---|---|
| diabetes / gender / medication_change | 0.509 | 0.509 |
| hmda / race / loan_decision | 0.514 | 0.533 |
| diabetes / gender / primary_diagnosis | 0.543 | 0.557 |
| hmda / race / loan_amount_band | 0.584 | 0.571 |
| adult / sex / income | 0.603 | 0.679 |
| adult / sex / occupation_group | 0.676 | 0.629 |

**Verdict: it is a law, not two points.** The output betrays the attribute to almost
exactly the extent the task label needs it — six cells along the diagonal, spanning
three datasets. (Caveat: for the two adult cells the representation could not be fully
hidden within σ ≤ 12, so their floors are measured with the rep only partly scrubbed —
upper-ish estimates; the qualitative law is unaffected. See `results/diagnostic.png`.)

### Bottom line — an honest problem-plus-solution story

- **Problem (Exp 1–7):** a compliance guarantee is exactly as strong as the attacker
  used to certify it; R²/linear certificates certify almost nothing; every
  probe-shaped eraser only fools its own probe; and the decision itself leaks the
  attribute whenever it genuinely uses it.
- **Solution (Exp 8):** durable removal *is* achievable and verifiable in the
  label-independent case — a from-raw-x noised channel hides race on HMDA from
  XGBoost + MLP + rank-32 LoRA, on **both** representation and output, at 96% utility
  (Part A) — **and** a cheap, model-free diagnostic (XGB AUC of the attribute from the
  task label alone) predicts, *before any model is trained*, whether a cell sits on the
  achievable side (predictor ≤ 0.55) or the impossible side (predictor > 0.55) of the
  wall (Part B), because output leakage obeys the footprint law (Part C).

The solution is a **characterization of *when* durable removal is possible plus a
constructive method for the possible case** — not a way to beat the wall when the task
genuinely needs the attribute. When predictor > 0.55, no representation scrubbing hides
the attribute in a *useful* output, exactly as the footprint law predicts. What Exp 8
adds is that this impossibility, and the cost when removal *is* possible, are now
**predictable in advance** from one number you can compute without training a model.
(Scope: one noise mechanism and one architecture family; the diagnostic threshold
coincides with the recovery bar by construction — its content is the *law* tying output
leakage to label leakage, which is mechanism-agnostic because it is a property of the
label, not the model.)

## Experiment 9 — de-circularizing the diagnostic, mapping the cost landscape, and surgical destruction

Experiment 8 left three loose ends: the diagnostic predicted a pass/fail defined at the
same 0.55 bar it thresholds on (partly circular); the cheap-vs-impossible split looked
like a cliff but the middle was barely sampled; and Exp 6's "surgical noise beats blunt"
result existed on one cell, without a LoRA attacker, and was bar-fragile. Experiment 9
resolves all three. Everything is measured with the strong battery (XGBoost + deep MLP,
plus rank-32 LoRA on every winning Goal-3 config) — never R².

### Goal 1 — the diagnostic predicts a CONTINUOUS cost (non-circular) (`experiments/continuous_cost.py`)

**Fix:** the outcome is now a *continuous* utility cost — the fraction of clean task lift
lost at the noise level where strong-attacker attribute recovery first reaches ≤ 0.55
(linearly interpolated between swept σ, clipped to [0,1]; the 0.55 crossing only *locates*
the noise level, the outcome value is a utility fraction, so predictor and outcome no
longer share a threshold). Mechanism: the Exp-8 from-raw-x noised channel, retrained per σ
on an adaptive grid extended to σ ≤ 192 so *every* cell gets a finite crossing. Measured
on **20 natural cells** spanning predictor 0.503–0.688 across all three datasets
(scouted live over all 30 attr×task pairs).

| outcome | Pearson r | Spearman ρ | n |
|---|---|---|---|
| cost to hide the representation | **+0.755** (p=1.2e-4) | +0.767 (p=7.8e-5) | 20 |
| cost of durable removal (rep **and** output) | **+0.795** (p=2.8e-5) | **+0.848** (p=2.3e-6) | 20 |

**Verdict: genuinely predictive, and non-circular.** A one-number, model-free label
statistic explains most of the ordering of a continuous utility cost measured by training
dozens of models (`results/continuous_cost.png`). The extended grid also resolves Exp 8's
"unreachable" verdicts into finite numbers: adult/sex/income's *representation* does hide,
but only at σ≈25 for 89% of the lift (durable: cost 1.0 at σ≈36).

**Honest scatter — the predictor is not the whole story.** Residuals are real and
systematic: tasks with huge clean lift stay cheap past their predictor
(adult/race/education_level: predictor 0.589, durable cost 0.018 — education is encoded so
redundantly that the channel keeps it long after race is gone; the 5-class macro-OVR
predictor is also inflated by one small race class), while low-lift tasks are expensive
early (hmda/race/tract_denial_high: predictor 0.518, durable cost 0.47). The predictor
sets the *trend*; the task's own signal robustness sets a family-specific offset.

### Goal 2 — cliff or ramp? A (steep) RAMP (`experiments/cliff_or_ramp.py`)

To sample the middle densely *and causally*, 9 **constructed** cells (clearly labelled,
not natural) subsample three real cells to tune the (attr, label) joint to
`(1−t)·p(a,y) + t·p(a)p(y)` — same features, same task, only the label↔attribute coupling
moves. Combined with the 20 natural cells (`results/cliff_or_ramp.png`):

| family (constructed, t: natural→independent) | predictor → durable cost |
|---|---|
| adult/sex/income | 0.526→0.25, 0.554→0.64, 0.567→0.96, 0.603 (nat)→1.00 |
| adult/marital_status/income | 0.546→0.69, 0.571→0.84, 0.622→1.00, 0.688 (nat)→1.00 |
| hmda/ethnicity/loan_amount_band | 0.534→0.26, 0.576→0.68, 0.623 (nat)→0.80 |

**Verdict: a RAMP, not a cliff — but a steep one.** 8/17 middle-zone cells
(predictor 0.55–0.68) land at intermediate durable cost (0.25–0.85), and within each
constructed family the cost rises smoothly and monotonically with the predictor. There
*is* a usable middle ground — roughly predictor 0.52–0.60, where durable removal is real
but costs ~25–85% of the lift — and by predictor ≈0.62 the cost saturates at ~1 in the
income families. The constructed families also upgrade the diagnostic from correlation to
**causation**: moving *only* the label↔attribute coupling moves the removal cost. (The
Exp-8 "cliff" was an artifact of 6 cells and a σ≤12 grid; the ramp's steepness varies by
family — the vertical scatter of Goal 1 applies here too.)

### Goal 3 — surgical destruction beats blunt noise EVERYWHERE tested (`experiments/surgical_vs_blunt.py`)

Exp 6's best surgical method (Gaussian noise confined to the rank-r most-attribute-
carrying HSIC subspace — destroying, not deleting) head-to-head against isotropic noise,
post-hoc on a fixed representation, across the spectrum. Utility = lift of a logistic head
retrained through the channel; every winning config must hold **XGBoost + MLP + rank-32
LoRA ≤ 0.55** (candidates that fail the LoRA check are discarded — none did).

| cell (base representation) | predictor | utility kept @ full-battery ≤0.55: **blunt** | **surgical** |
|---|---|---|---|
| anchor: adult/sex/income (frozen PCRL rep — the Exp-6 setting + LoRA) | 0.603 | 57% (iso σ=4) | **89%** (r=16, σ=32) |
| easy: hmda/race/loan_decision (clean trained channel P) | 0.514 | 1% (iso σ=16) | **88%** (r=16, σ=16) |
| middle: hmda/race/loan_amount_band (clean trained P) | 0.584 | 36% (iso σ=8) | **81%** (r=8, σ=16) |
| hard: adult/sex/income (clean trained P) | 0.603 | 3% (iso σ=16) | **53%** (r=32, σ=16) |

**Verdict: surgical destruction earns its keep — blunt noise is NOT the post-hoc floor.**
The advantage generalizes across datasets, attributes (binary sex, 5-class race), task
types, and base representations, and survives the LoRA attacker Exp 6 never ran. Exp 6's
bar-fragility caveat also dissolves off the frozen rep: at a stricter **0.53** bar,
surgical still keeps 88% / 81% / 52% vs blunt's 1% / 18% / 3% on the three trained-P cells
(all stricter-bar winners LoRA-vetted too, ≤0.524 —
`results/surgical_vs_blunt_lora_extra.json`); only the frozen-PCRL anchor loses its edge
there (23% vs 19%), which is where the fragility lived. On the clean trained
representations — which leak the attribute at XGB AUC ≈1.00, far above the frozen rep's
0.68 — blunt noise must drown *everything* to reach the bar, while noise aimed at the
attribute subspace pays a fraction.

**Honest scope.** (1) This protects the *representation* channel of a fixed encoder; the
output footprint law (Exp 7/8) still governs the decision itself — surgical noise does not
(and cannot) reduce output leakage, so above-the-wall cells stay above it. (2) "Hidden"
means ~0.50–0.53 residual AUC against this battery, not provably zero; the HSIC subspace
is fit on the same data. (3) Where the end-to-end noised channel applies (below-the-wall
cells), it can still be cheaper on *both* channels (HMDA loan_decision: 96% at σ=8
end-to-end); surgical post-hoc noise is the tool for representations you cannot retrain —
and there it beats blunt noise everywhere we looked.

### Bottom line after Experiment 9

1. **An honest, non-circular predictive diagnostic — yes.** One pre-training number (XGB
   AUC of attr from the task label) predicts the *continuous* utility cost of honest
   durable removal at r≈0.8 / ρ≈0.85 over 20 natural cells, with known, systematic
   residuals (task-signal redundancy).
2. **A clear cost landscape — yes, and it is a steep ramp.** Cost rises smoothly through a
   usable middle (predictor ~0.52–0.60, cost ~25–85%) and saturates near ~0.62; confirmed
   causally by constructed families that tune only the label↔attribute coupling.
3. **A real surgical-destruction method — yes, for the representation channel.** HSIC-
   subspace noise beats blunt isotropic noise at matched full-battery protection on all
   four cells tested (89/88/81/53% vs 57/1/36/3% kept), robust to a stricter bar off the
   frozen rep. Blunt noise is *not* the floor; the floor that remains is the footprint
   law: no representation-side mechanism, surgical or blunt, hides an attribute the
   decision itself needs.

## Experiment 10 — can surgical destruction be trained END-TO-END, and does it beat post-hoc?

Post-hoc surgical noise (Exp 9's winner) is limited to whatever directions a *fixed*
representation happened to use. Experiment 10 (`experiments/end_to_end_surgical.py`)
tests whether a representation trained from scratch to *build the surgery in* does
better: an encoder → BatchNorm → **learned** orthonormal rank-r subspace Q
(differentiable QR) → Gaussian noise confined to span(Q) (fresh every forward) → task
head, all trained **jointly** with a λ·HSIC(h⊥, attr) term that pushes attribute
information out of the surviving directions while the task CE (trained through the
noise) pulls task information into them. The noise still does the destruction —
attacker-agnostically; HSIC only decides *where* information sits, so the probe-shaped
-eraser trap of Exp 4/5 does not apply (and the honest battery checks anyway).

Design: a 2×2 factorial {blunt, surgical} × {post-hoc, end-to-end} on the three Exp-9
trained-P cells, so any gain is attributable to e2e-ness vs surgical-ness. Protocol
unchanged from Exp 9: XGB + deep MLP candidacy at ≤ 0.55, best-utility candidate must
hold the rank-32 LoRA, never R²; e2e winners retrained over 3 seeds with the full
battery on representation **and** output; λ piloted per cell over {0, 10, 100} (λ=0 =
the shaping ablation).

**Utility kept at full-battery honest protection (XGB + MLP + LoRA-32 ≤ 0.55):**

| cell | predictor | post-hoc blunt | post-hoc surgical (Exp-9) | e2e blunt (Exp-8) | **e2e surgical (new)** |
|---|---|---|---|---|---|
| easy hmda/race/loan_decision | 0.514 | 1% | 88% | 98%±2 | **104%±3** |
| middle hmda/race/loan_amount_band | 0.584 | 36% | 81% | 39%±1 | **100%±0** |
| hard adult/sex/income | 0.603 | 3% | 53% | 0%±1 | **90%±3** |

*(e2e arms: mean±std over 3 training seeds; % of each family's clean lift — post-hoc:
retrained-LR head, e2e: own head. The retrained-LR cross-check on the e2e-surgical
winners matches or exceeds the own-head numbers — e.g. hard cell +0.135 LR-through vs
+0.128 own-head — so the gap is not an estimator artifact.)*

**Verdict: YES — end-to-end surgical training beats post-hoc surgical noise everywhere,
and by a lot where it matters.** On the middle cell it keeps **100%** of the task lift
(+0.411 vs clean +0.409, rep hidden at ≤0.535 incl. LoRA) vs post-hoc's 81%; on the
hard cell **90%** vs 53%. The factorial attributes the gain: it is the *combination* —
e2e blunt (the Exp-8 channel) collapses on the middle (39%) and hard (0%) cells, and
post-hoc surgical can only noise the subspace the frozen encoder used. Letting the
encoder *choose* a basis where attribute and task separate, knowing the attribute
subspace will be destroyed, moves the representation-channel ramp down to
near-zero cost across the tested band (predictor 0.51–0.60).

Three supporting findings:

1. **The shaping term is what earns it (ablation).** On the hard cell with λ=0 the
   exposed rep leaks sex at XGB 0.87 / MLP 0.99; λ=10 still leaks (MLP 0.74); only
   λ=100 concentrates the attribute into the destroyed subspace (≤ 0.55). The
   architecture alone does nothing — noise-in-a-learned-subspace without HSIC pressure
   leaves the attribute in the surviving directions.
2. **The LoRA attacker earned its place in the battery.** On the hard cell, the three
   best-utility e2e-surgical candidates passed XGB + MLP but **breached the rank-32
   LoRA** (0.571–0.575) — the first LoRA-only breaches in the project. Widening the
   destroyed subspace (r=32, σ=8) closes the adapter's route (0.547). A two-probe
   battery would have shipped a leaking config.
3. **The footprint law still bounds the output — no method beats the wall.** On the
   above-the-wall cells the winning e2e-surgical model's *output* still leaks the
   attribute at 0.710 (middle) and 0.605 (hard), at/above the label predictor (0.584,
   0.603), exactly as the law demands of a model that keeps the utility. On the easy
   (below-the-wall) cell everything including the output sits ≤ 0.54. Representation
   protection got dramatically cheaper; output leakage did not move.

Honest scope: (1) hard-cell protection sits at the bar's edge — rep LoRA 0.549±0.009
over training seeds, so the *worst* seed grazes ~0.55–0.56; call it "at the bar", not
comfortably under it. (2) The easy cell's 104% is noise-as-regularizer (within ~2σ of
100%). (3) "Hidden" still means ~0.50–0.55 residual AUC against this battery, not
provably zero, and Q/HSIC are fit on the same data. (4) This is the representation
channel only: for a *deployed decision* on an above-the-wall cell, the output floor is
untouched — the honest offer remains "utility-free representation scrubbing, output
leaks what the label needs" (middle cell: 100% utility, rep ≤ 0.535, output 0.71).

## Experiment 11 — is the DPI recovery bound a usable *theorem*, or only a mechanism? (VACUOUS at operating σ)

The noise channel is the one mechanism in this project with a principled guarantee:
`h̃ = h + N(0, σ²I)` caps `I(h̃; A)` by the data-processing inequality, no matter how
the attacker is built. We planned to feature a proposition instantiating that
guarantee numerically — σ in, attacker-agnostic recovery ceiling out. Experiment 11
(`experiments/mi_ceiling.py`) derives the tightest clean **distribution-free** version
of that proposition and computes it at the exact σ our experiments ran. Honest
question: does the *theorem* separate us from full recovery at the operating σ, or
only the *measurement*?

**The proposition (distribution-free, needs only per-dim variances s²ᵢ of the
pre-noise representation):**

1. `I(h̃;A) ≤ I_cap = ½ Σᵢ log(1 + s²ᵢ/σ²)` (variational MI bound with a Gaussian
   reference + the entropy floor `h(P̃ₐ) ≥ h(N(0,σ²I))`; for anisotropic full-rank
   noise replace `s²ᵢ/σ²` by `s²ᵢ/σ²ᵢ`).
2. Accuracy ceiling via Fano; AUC ceiling per class via
   `TV ≤ √(I_cap/8)/(π_k(1−π_k))` (joint-Pinsker) and `AUC ≤ ½ + TV − TV²/2`
   (ROC dominance); macro-OVR average for multiclass.
3. **Tightness floor:** the two-point distribution `h = ±μ`, `‖μ‖² = Σᵢs²ᵢ` matches
   the same moments, and there an optimal attacker genuinely achieves
   `AUC = Φ(√(2Σᵢs²ᵢ)/σ)`. So the vacuousness below is not sloppy accounting — *any*
   certificate that sees only second moments concedes at least this.
4. **Surgical channels have no such certificate at all**: with σ=0 in the surviving
   subspace the bound diverges. The Exp-9/10 surgical results' guarantee is, and
   remains, the empirical battery (+ HSIC shaping) — consistent with the LoRA-only
   breaches Exp 10 caught.

**σ vs empirical battery vs theoretical ceiling** (condensed; full tables in
`results/mi_ceiling.json`, plot in `results/mi_ceiling.png`):

| channel | σ | battery max AUC | Gaussian-LRT (est.) | **cert AUC ceiling** | moment worst case |
|---|---|---|---|---|---|
| adult/sex frozen rep (Exp 2/5) | σ_rel=4 | 0.547 | 0.555 | **1.000** (I_cap=2.02 ≥ H(A)=0.63) | 0.998 |
| adult/sex frozen rep | σ_rel=8 | 0.518 | 0.543 | **1.000** (I_cap=0.52) | 0.926 |
| hmda/race/loan_decision e2e (Exp 8) | σ=8 | 0.530 (LoRA 0.528) | 0.574 | **1.000** (I_cap=1.48 ≥ H(A)=0.92) | 0.993 |
| hmda e2e | σ=24 | 0.504 (LoRA 0.501) | 0.531 | 0.983 | 0.804 |
| hmda e2e | σ=48 | 0.501 (LoRA 0.494) | 0.510 | 0.888 | 0.618 |
| adult/sex/income e2e (Exp 9) | σ=24 | 0.554 | 0.611 | 0.957 | 0.809 |
| adult/sex/income e2e | σ=32 | 0.528 | 0.580 | 0.856 | 0.717 |

**Verdict: VACUOUS at every operating σ — cite the DPI principle qualitatively; do
not feature a numeric proposition.** At the σ where the strong battery is empirically
at ~0.53 (σ_rel=4–8 frozen; σ=8 e2e), the provable ceiling is exactly 1.0 — the MI
bound doesn't even drop below H(A). The certificate first reaches a *meaningful*
ceiling (≤0.7 AUC) at σ\* ≈ 15.4 (σ_rel≈42) on the frozen rep, σ\* ≈ 54 on the adult
e2e channel (where task lift is already ≈0 from σ=32), and σ\* ≈ 334 on HMDA
(macro-OVR is dominated by the rare race classes, π=0.008–0.06, which Pinsker cannot
bound tightly). Everywhere, the theorem becomes non-vacuous only far past the utility
cliff. The reason is structural, not fixable by better algebra: a distribution-free
bound must budget for **all** 64 BN-normalized dimensions of representation variance
(`I_cap ≈ d/2σ²`) because it cannot tell task signal from attribute signal — and the
two-point construction shows a moment-matched distribution where an attacker really
does recover the attribute at 0.93–1.00 AUC at our operating σ. A meaningful theorem
would need distributional assumptions, at which point it is an estimate, not a
certificate.

**Side finding — the battery is not the ceiling (honest sharpening of Exp 8/10).**
The Gaussian-LRT column is a channel-aware attacker: fit `N(μ_k, Σ_k + σ²I)` per
class on the *clean* representation, score noised rows by posterior. At every
operating point it beats the trained battery: **0.574 vs 0.530 on HMDA at σ=8 — above
the 0.55 bar the battery certifies**; 0.580 vs 0.528 on adult e2e at σ=32. Threat-model
caveat: it uses clean-representation side information (the vendor themselves, or an
attacker who can average repeated queries of the same row through the fresh-noise
channel), which the standing deployed-dataset battery does not grant. But as an
estimate of the true information ceiling it says: the honest-bar verdicts are
battery-relative, an informed attacker sits ~0.02–0.05 higher at operating σ, and
driving *it* below 0.55 on HMDA needs σ between 12 and 24 (utility 83% → 35%). This
is the project's own thesis biting our constructive result — a guarantee is only as
strong as the strongest attacker used to certify it — and since Experiment 11 shows
the theory cannot take over, the practical consequence is that the battery should
grow a channel-aware LRT member, not that the mechanism is wrong. (The gap also
confirms the battery sits close to, not at, the ceiling: at σ≥24 all attackers and
the LRT converge to ~0.50–0.53.)

## Experiment 12 — CelebA coupling scan: picking the two vision cells (scan only)

The paper adds one low-coupling and one high-coupling CelebA task/attribute pair as
the non-tabular modality check. Per the standing methodology, the cells are picked by
**measurement**: the predictor = AUC recovering the protected attribute from the task
LABEL alone, computed on the CelebA train partition (162,770 rows) loaded exactly as
PCRL's `pcrl/data/celeba.py` loads it (labels only — no images, no representations,
no removal pipeline yet). For a 1-binary-feature predictor the population AUC is
closed-form (`½ + |P(f=1|a=1) − P(f=1|a=0)|/2`), used to rank **all** (task ×
{Male, Young, Pale_Skin}) pairs; the shortlist is then re-measured with the project's
actual XGB predictor instrument (3 held-out splits) — agreement is ±0.003 everywhere.
`experiments/celeba_coupling_scan.py`, ranked table in
`results/celeba_coupling_scan.json`/`.png`.

Sanity screens: **Pale_Skin is excluded as a protected attribute** (4.3% positive,
below the 5% floor — as are Bald/Double_Chin/Gray_Hair/Mustache/Wearing_Hat as
tasks); near-duplicate pairs are flagged (|φ|>0.45 or gender-coded grooming
synonyms): **Wearing_Lipstick→Male (0.899) and Heavy_Makeup→Male (0.828) top the
ranking but are the attribute in disguise** — flagged, not eligible.

| task → protected (eligible, condensed) | predictor (XGB) | φ | verdict |
|---|---|---|---|
| Attractive → Young | **0.734** | +0.39 | **HIGH cell (recommended)** |
| Attractive → Male | 0.704 | −0.40 | high backup |
| Arched_Eyebrows → Male | 0.681 | −0.41 | grooming-adjacent |
| Big_Nose → Male | 0.661 | +0.37 | high alternative |
| Smiling → Male | 0.569 | −0.14 | middle |
| **Smiling → Young** | **0.522** | −0.03 | **LOW cell (recommended)** |
| High_Cheekbones → Young | 0.508 | −0.01 | low alternative (Smiling-correlated task) |

**Recommended cells: task=Smiling (predictor 0.522) and task=Attractive (predictor
0.734), both with protected=Young.** Reasons: (1) both tasks are canonical, balanced
CelebA tasks (48%/51% positive; Young 78% positive, within the balance window);
(2) sharing the protected attribute makes the pair a controlled contrast — only the
task changes, exactly like Exp 9's constructed families; (3) both pairs are already
PCRL CelebA purposes (`smile_detection` and `attractiveness_prediction` both list
Young as a disallowed attribute), so the full run can reuse PCRL's purpose specs
unchanged; (4) neither is a near-synonym pair (φ = −0.03 / +0.39). From the tabular
ramp (Exp 9), the prediction to test: Smiling/Young durable removal should be cheap
(predictor ≈ 0.52), Attractive/Young should cost ≈ all the lift (0.734 is far above
the ≈0.62 saturation) with an output-leak floor near 0.73. The full CelebA run needs:
PCRL's vision encoder forward passes over the 64×64 images (202k) to extract
representations, then the standard pipeline — end-to-end noised channel + full
battery (XGBoost + MLP + rank-32 LoRA) on representation and output.

## Experiment 13 — the full CelebA pipeline: does the story survive a change of modality?

The two scan-picked cells (Exp 12) get the complete standard pipeline
(`experiments/celeba_pipeline.py`). X is the **frozen PCRL vision representation**
(train partition, N=162,770, 128-d per purpose) extracted through PCRL's trained
`celeba_v2` CNN encoder by `experiments/celeba_extract.py` (checkpointed forward pass
over all 202,599 images, deterministic eval transform, both purposes in one pass).
Protocol identical to Exp 10: the 2×2 {blunt, surgical} × {post-hoc, e2e}, Tier-1
vetted (XGB + MLP + LoRA-32 ≤ 0.55, e2e winners over 3 training seeds, rep AND output
batteries), plus the Tier-2 (LRT) add-on and the output footprint.

**Scope fact discovered up front: the vision input is already partially scrubbed.**
Both purposes list Young as a disallowed attribute, so PCRL's purpose training already
suppressed it: Young is recoverable from raw X at only XGB **0.571 / 0.608**
(smile / attractiveness purpose), and the tasks themselves are only weakly present
(Smiling 0.674, Attractive 0.633). This is the realistic vision regime — and it is
exactly what breaks the numeric gauge below.

**Utility kept at Tier-1 (≤0.55) protection, % of each family's clean lift:**

| cell (predictor) | post-hoc blunt | post-hoc surgical | e2e blunt | e2e surgical |
|---|---|---|---|---|
| Smiling→Young (0.522) | 23% | **98%** (r=8, σ=16) | 29% | **99%** (r=8, σ=4, λ=100) |
| Attractive→Young (0.734) | 28% | **81%** (r=32, σ=16) | 51% | **78%** (r=32, σ=8, λ=10) |

*(all winners LoRA-vetted; outputs of every winner ≤ 0.515 (low cell) / ≤ 0.553 (high
cell). Own-head e2e lift is unstable on CelebA (some seeds' heads undertrain, e.g. e2e
blunt lift std ±0.03–0.04); the retrained-LR cross-check is stable and agrees with the
post-hoc numbers, so the arm comparison uses both.)*

**Prediction verdicts:**

1. **Smiling→Young removal is CHEAP — HELD.** 98–99% of the lift kept with the
   representation AND output at ≈ chance (≤ 0.515), reproducing the tabular
   below-the-wall pattern on vision.
2. **Attractive→Young costs ≈1 with an output floor near 0.73 — DID NOT HOLD.**
   Measured Tier-1 cost is **0.19** (81% kept) and the output floor is **0.526–0.553**,
   nowhere near 0.73. Even the *clean, unprotected* model's output leaks Young at only
   0.561. **Why (reconciliation, measured):** the footprint law ties output leakage to
   how much the *decision* carries the attribute. The label couples with Young at 0.734,
   but the achievable model is weak (accuracy 0.565 vs majority 0.514 — the purpose rep
   was never a strong Attractive predictor), and the coupling of the model's **achieved
   predictions** with Young is **0.536 (clean) / 0.540 (winner)** — matching the
   measured output floor almost exactly (low cell: 0.502–0.505 vs floor ≈ 0.51). So the
   law survives in its causal form — *the output leaks what the output actually encodes
   of the label* — but the label-based predictor is an **upper bound** on the floor and
   the cost, and it is loose whenever the task is only weakly learnable from the
   deployed representation. This is the block's most important scope discovery: the
   diagnostic's impossibility half ("predictor > 0.55 ⇒ durable removal costs ~all
   utility") presumes the model can approach the label; on an already-scrubbed,
   hard-task vision representation it can't, and durable removal gets *cheaper* than
   the gauge says, never more expensive.

**Tabular patterns checked on vision:** **surgical-beats-blunt generalizes** (98 vs 23,
81 vs 28 post-hoc; 99 vs 29, 78 vs 51 e2e) — the biggest arm gaps in the project,
because blunt noise must drown a 0.57-leak rep to reach the bar while the HSIC subspace
is small. **e2e-beats-post-hoc does NOT generalize: it ties** (99 vs 98, 78 vs 81) —
on a frozen, already-shaped vision rep there is little left for end-to-end shaping to
re-route (deviation noted as scope, consistent with the mechanism). **Tier 2 behaves
exactly as in Exp 14:** the blunt arms reach Tier 2 (low cell: at the Tier-1 winner
itself, LRT 0.521–0.522; high cell: iso σ=16 / e2e σ=16 at 10–41% utility), the
surgical arms breach it (LRT 0.552–0.621) with no in-family fix — though far more
mildly than tabular (0.56–0.62 vs 0.79–0.94) because the complement only ever held a
0.57–0.61 leak.

## Experiment 14 — two-tier certification: the battery grows the LRT, and every key operating point is re-certified

Experiment 11's side finding (the channel-aware Gaussian-LRT beats the battery at
operating σ) is now institutionalized (`utils/battery.py`,
`experiments/two_tier_certification.py`): the LRT — fit class Gaussians
`N(μ_k, Σ_k + Σ_noise)` on the **clean pre-noise representation**, integrate the known
channel analytically (`Σ_noise = σ²I` blunt; `σ²QQᵀ` surgical), score held-out noised
rows — joins XGB + MLP + LoRA-32 as a standard member, and every verdict now names its
tier:

* **TIER 1 (black-box):** XGB + MLP + LoRA-32 — the attacker sees only the deployed
  noised representation (the standing battery of Exp 5–10).
* **TIER 2 (informed):** + Gaussian-LRT — the attacker also has clean-representation
  side information and knows the channel (the vendor themselves, a pipeline auditor, or
  an attacker who can average repeated fresh-noise queries).

**The two-tier table** (e2e cells: 5 training seeds, mean±std; utility = own-head lift,
% of the 5-seed clean lift):

| cell | Tier-1 point | utility @ T1 | Tier-2 point | utility @ T2 |
|---|---|---|---|---|
| e2e hmda/race/loan_decision (pred 0.514) | σ=8 | +0.0211±0.0005 (**95%**) | σ=20 | +0.0124±0.0010 (**55%**) |
| e2e hmda/race/loan_amount_band (0.584) | σ=12 | +0.2315±0.0029 (**57%**) | σ=24 | +0.0889±0.0020 (**22%**) |
| e2e adult/sex/income (0.603) | σ=32 | −0.0008±0.0021 (−1%) | σ=64 | −0.0100±0.0022 (−7%) |
| post-hoc surgical easy (winner r=16 σ=16) | winner | +0.0184 (88%) | sub r=64 σ=32 | −0.0000 (0%) |
| post-hoc surgical middle (r=8 σ=16) | winner | +0.3327 (81%) | sub r=64 σ=16 | +0.0760 (19%) |
| post-hoc surgical hard (r=32 σ=16) | winner | +0.0778 (53%) | iso σ=32 | +0.0001 (0%) |

*(Tier-1 points match the published results — Exp-8's σ=8/96% reproduces as 95%±;
at those points the LRT reads 0.578–0.585, i.e. every Tier-1 headline is genuinely
Tier-1-only. "sub r=64" of a 64-dim rep is full-rank anisotropic noise — the surgical
family collapses onto blunt noise to buy Tier 2.)*

**The structural finding — surgical channels are a Tier-1 technology.** The Exp-9
post-hoc surgical winners hold Tier 1 exactly as published, but the informed LRT
recovers the attribute from them at **0.923 / 0.944 / 0.793** — near-clean recovery —
because subspace noise leaves the complement noise-free and an informed adversary reads
it at full strength. No surgical σ-escalation helps (σ_rel up to 64: LRT unchanged).
The Exp-10 **e2e** surgical winners breach Tier 2 the same way (LRT **0.841 / 0.855 /
0.663**, no fix within σ ≤ 128–256): HSIC shaping concentrated everything the *battery*
could see into the destroyed subspace, but not the Gaussian class structure the LRT
reads. This is Exp-11's "surgical channels have no DPI ceiling" made empirical, and the
honest scope of Exp 9/10 sharpened: **surgical destruction's near-free protection is
real at the black-box tier and does not exist at the informed tier; at Tier 2 the only
mechanism is full-rank noise, at full-rank prices** (55% easy / 22% middle / ~0% hard).

## Experiment 15 — hardening: every headline number at 5 training seeds

`experiments/hardening_seeds.py` re-runs the Exp-10 e2e winners (both arms), the
hard-cell edge case, and Exp-8's HMDA Part A at **5 training seeds**
(`results/hardening_seeds.json`, per-seed values stored):

| cell | e2e blunt (C) | e2e surgical (D) | D rep_max | D out_max |
|---|---|---|---|---|
| easy hmda/race/loan_decision | 95%±2 | **102%±3** | 0.509 | 0.525 |
| middle hmda/race/loan_amount_band | 39%±1 | **100%±1** | 0.531 | **0.700** |
| hard adult/sex/income | −1%±1 | **90%±3** | **0.552** | 0.613 |

* **Everything reproduces** (3-seed headline 104/100/90 → 5-seed 102/100/90; clean
  lifts and the footprint floors move by ≤0.01) — **except one number, flagged loudly:**
* **The hard-cell winner (r=32 σ=8 λ=100) flips its bar verdict at 5 seeds.** Per-seed
  rep max: 0.547 / 0.540 / **0.561** / **0.554** / **0.558** — 3/5 seeds breach 0.55
  (always via the LoRA), mean 0.552; all 5 hold at a 0.57 bar. Exp-10's "at the bar,
  not comfortably under it" caveat was the right call, and 5 seeds settle it: **at the
  0.55 bar the σ=8 config does not certify.** The robust operating point is **σ=16**:
  rep max per seed 0.537/0.536/0.537/0.528/0.540 (worst 0.540 — holds on every seed),
  utility **85%±2** (lift +0.1225±0.0031 of clean +0.1435). The honest hard-cell
  headline is therefore **85% at a robust ≤0.55, or 90% at a 0.57 bar** — not 90% at 0.55.
* **Exp-8's HMDA constructive headline hardens cleanly:** at 5 seeds, P max 0.535 /
  output max 0.539 (all attackers 0.529–0.539), lift +0.0211±0.0005 = **95%** of clean
  (3-seed: ~0.53 / 96%).

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
experiments/diagnostic.py             # Experiment 8: the diagnostic — label↔attr predicts durable-removal cost; footprint law across 6 cells
experiments/_scout_predictor.py       # Experiment 8 cell-selection scout: XGB AUC(attr|label) across candidate cells
experiments/continuous_cost.py        # Experiment 9 Goal 1: predictor vs CONTINUOUS removal cost, 20 cells (non-circular)
experiments/cliff_or_ramp.py          # Experiment 9 Goal 2: constructed middle-zone cells — the cost curve is a steep RAMP
experiments/surgical_vs_blunt.py      # Experiment 9 Goal 3: HSIC-subspace vs isotropic noise, full battery incl. LoRA-32
experiments/end_to_end_surgical.py    # Experiment 10: end-to-end trained surgical channel vs post-hoc (2x2 factorial)
experiments/mi_ceiling.py             # Experiment 11: the DPI certificate is VACUOUS at operating σ (theory check)
experiments/celeba_coupling_scan.py   # Experiment 12 (scan): label-only coupling scan to pick the two CelebA cells
experiments/celeba_extract.py         # Experiment 13 (step 1): checkpointed CelebA forward pass through PCRL's vision encoder
experiments/celeba_pipeline.py        # Experiment 13: the full CelebA pipeline on both cells (2x2 + tiers + footprint)
experiments/two_tier_certification.py # Experiment 14: two-tier (black-box vs informed-LRT) re-certification of the key operating points
experiments/hardening_seeds.py        # Experiment 15: every headline number at 5 training seeds (hard-cell edge case settled)
utils/battery.py                       # the standing battery: Tier 1 (XGB+MLP+LoRA32) and Tier 2 (+ channel-aware Gaussian-LRT)
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
