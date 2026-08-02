# Worst-case leakage: TPR at low FPR for the surviving operating points (Job 2), with the reproduction gate (Job 3)

2026-07-23. Script: `experiments/run_tpr.py` (new file; no existing script,
figure, or result file modified). Raw per-example scores:
`analysis/tpr_scores/*.npz` (14 files). All aggregates:
`results/tpr_capture.json`. Wall time: **41.7 min** on MPS.

## Process disclosure — the strongest finding here is unregistered

This is a slip against the project's own registered-prediction discipline, disclosed
up front rather than in a footnote: **no bet was committed before this run** — not
for the TPR readings, and not for the middle/hard averaging generalization. The only
registered bet in git history is the easy-cell averaging prediction (`ccbb53e`,
quoted below), whose scope does not cover these cells or these metrics. The
strongest result of the day — the middle subspace point collapsing to T1 ≈ 0.93
under N=16 averaging, breaching on 5/5 seeds already at N=4 — is therefore an
**exploratory, post-hoc finding**, not a confirmed prediction. Nothing below is
invalidated by this (the 230/230 bit-exact reproduction gate stands on its own),
but the epistemic status differs: the easy-cell averaging breach was predicted in
writing before measurement; the middle/hard breaches were not. Any paper use should
label them accordingly (e.g. "we then tested whether the effect generalizes",
not "as predicted").

## Precondition: the registered-bet commit

Verified with `git log`. Commit `ccbb53e`, quoted verbatim:

> averaging attack: registered prediction (committed before implementation) —
> easy-cell e2e surgical winner verified as r=32 sigma=16 lambda=10 (task brief's
> sigma=8 is the hard cell's / e2e-blunt point); predict sqrt(N) law: N-draw
> averaging ~ single draw at sigma/sqrt(N), N=4 near the s=8 grid row (XGB 0.522),
> N=16 near the s=4 row (0.541), monotone in N, bar 0.55 likely uncrossed at N=16
> but breached by extrapolation ~N=64; N=1 must reproduce winner numbers
> (0.5060/0.5017/0.4994) as the sanity gate

**Flag:** this (with its file `results/averaging_attack_prediction.md`) is the ONLY
registered bet in git history. No TPR-specific bet was ever committed; the
instruction "the registered bet is scored against worst-class" therefore has no
committed TPR prediction to score against — worst-class numbers are reported below
as specified, but only the averaging bet is scorable. The averaging bet's registered
scope is the **easy** cell; the middle/hard averaging runs here are generalization
tests of the same claims, marked as such.

## Protocol notes (deviations flagged, none silent)

- **60k/seed-42 subsample rule:** that cap belongs to the expansion pipeline
  (`expansion_pipeline.MAX_N`). Every stored number for the points below was
  measured on the FULL train partitions (adult 24,145 / hmda 63,747), so this run
  uses full partitions too. Vindicated by the gate: 230/230 stored AUCs reproduce
  **bit-exactly** (every delta 0.00000).
- Splits: 75/25 attribute-stratified per probe seed (`diagnostic._split`); probe
  seeds [0,1] for sections A/B/D/E (the e2e / continuous-cost convention), [0,1,2]
  for the full-rank certifications (two_tier `CERT_SEEDS`). Five training seeds
  [0..4] everywhere. Attacker configs verbatim from `hardening_test._probe`,
  `diagnostic.probe_lora`, `utils.battery._lrt_scores`. Single fresh noise draw per
  input (the training functions' own convention).
- TPR convention: max TPR at FPR ≤ target (step function, no interpolation).
  "Worst seed" = the (train seed × probe seed) with the **highest** TPR (most
  leakage). HMDA race is 5-class (n_attr=5): per-class one-vs-rest for all five
  classes + worst-class + macro; per-class positive-eval-count shown, and any class
  with < 500 positive eval examples flagged noisy (classes 3: n=336, 4: n=126 —
  every worst-class HMDA number driven by those two classes should be read with
  that flag).
- Source for every number below: `results/tpr_capture.json`
  (`tpr`, `averaging`, `reproduction` blocks) and the named stored-reference JSONs.

## TPR at low FPR — surviving operating points, best attacker per surface

Representation surface ("rep") is each certificate's certified surface; logits
("out") shown alongside. Mean over all (train seed × probe seed); worst seed in
parentheses. Binary cells: n_pos(eval) = 4,088.

### Unprotected baselines (no removal; reference: `continuous_cost.json` sweep σ=0)

| cell | surface | best | AUC | TPR@0.01 | TPR@0.05 |
|---|---|---|---|---|---|
| easy hmda/loan_decision | rep | MLP | 1.0000 | worst-class 1.000 (1.000) | 1.000 (1.000) |
| easy | out | XGB | 0.6807 | worst-class 0.068 (0.093) | 0.247 (0.304) |
| middle hmda/loan_amount_band | rep | MLP | 1.0000 | worst-class 1.000 (1.000) | 1.000 (1.000) |
| middle | out | MLP | 0.8291 | worst-class 0.347 (0.476) | 0.566 (0.714) |
| hard adult/income | rep | MLP | 0.9998 | 0.998 (1.000) | 1.000 (1.000) |
| hard | out | MLP | 0.7632 | 0.108 (0.320) | 0.289 (0.498) |

The unprotected representation identifies attribute members essentially perfectly
even at FPR 0.01.

### Subspace-confined Tier-1 surviving points (reference: `hardening_seeds.json`)

| point | surface | best | AUC | TPR@0.01 | TPR@0.05 |
|---|---|---|---|---|---|
| easy r=32 σ=16 λ=10 | rep | XGB | 0.5091 | worst-class 0.016 (0.021) | 0.064 (0.083) |
| easy | out | XGB | 0.5252 | worst-class 0.033 (0.056)ⁿ | 0.108 (0.143)ⁿ |
| middle r=16 σ=16 λ=100 | rep | LoRA | 0.5312 | worst-class 0.026 (0.031) | 0.084 (0.095) |
| middle | out | MLP | **0.7003** | worst-class **0.106 (0.167)**ⁿ | **0.253 (0.373)**ⁿ |
| hard r=32 σ=16 λ=100 (σ=16 alt.) | rep | LoRA | 0.5358 | 0.025 (0.033) | 0.080 (0.089) |
| hard | out | MLP | 0.5841 | 0.038 (0.043) | 0.107 (0.124) |

ⁿ worst-class driven by the n=336 / n=126 classes — noisy per the < 500 flag
(macro values: easy out 0.017/0.069; middle out 0.071/0.197).

At the certified (rep) surface all three points hold TPR@0.01 at 1.6–2.6% —
essentially the false-positive budget, i.e. no worst-case identification power.
The standout is the **middle point's output surface** (AUC 0.7003, the known
out_max leak): worst-class TPR@0.01 ≈ 10.6%, and even the well-populated class 2
(n=4,242) reads 8.4% — an attacker watching the logits identifies 1-in-12 members
of that race class at a 1% false-positive cost. The certificate covers the
representation only; this quantifies what that exclusion costs on this cell.

### Full-rank noise points (reference: `two_tier_certification.json` e2e_cells)

Rep surface; the LRT (Tier-2 member) is measured at Tier-1 points too — both shown.

| point | attacker set | best | AUC | TPR@0.01 | TPR@0.05 |
|---|---|---|---|---|---|
| easy T1 σ=8 | T1 suite | XGB | 0.5351 | worst-class 0.024 (0.039) | 0.103 (0.127) |
| easy T1 σ=8 | +LRT | LRT | 0.5834 | worst-class 0.037 (0.071)ⁿ | 0.121 (0.167)ⁿ |
| middle T1 σ=12 | T1 suite | LoRA | 0.5408 | worst-class 0.023 (0.032) | 0.089 (0.103) |
| middle T1 σ=12 | +LRT | LRT | 0.5846 | worst-class 0.028 (0.040) | 0.105 (0.113) |
| hard T1 σ=32 | T1 suite | LoRA | 0.5311 | 0.015 (0.026) | 0.066 (0.085) |
| hard T1 σ=32 | +LRT | LRT | 0.5782 | 0.026 (0.034) | 0.094 (0.107) |
| easy T2 σ=20 | full | LRT | 0.5377 | worst-class 0.023 (0.040)ⁿ | 0.096 (0.107)ⁿ |
| middle T2 σ=24 | full | LRT | 0.5496 | worst-class 0.023 (0.040)ⁿ | 0.082 (0.111)ⁿ |
| hard T2 σ=64 | full | LRT | 0.5339 | 0.014 (0.018) | 0.064 (0.073) |

Logits surfaces at these points are at or below the rep readings (all ≤ 0.028
TPR@0.01 mean except middle T1 out at 0.024; full table in `tpr_capture.json`).
The Tier-1 points' known LRT excess (AUC 0.578–0.585) translates to only ~1pp of
extra worst-case TPR@0.01 over the T1 suite — the informed attacker's advantage is
diffuse, not concentrated in a high-precision regime. At every certified point, on
its certified attacker set, TPR@0.01 stays within ~3pp of the 1% FPR budget.

### Adult floor points (Job 1's longer-grid first-hidden σ; reference: `continuous_cost.json` sweep)

| point | surface | best | AUC | TPR@0.01 | TPR@0.05 |
|---|---|---|---|---|---|
| income σ=32 | rep | LoRA | 0.5290 | 0.016 (0.026) | 0.065 (0.085) |
| income σ=32 | out | MLP | 0.5445 | 0.017 (0.028) | 0.074 (0.110) |
| occupation_group σ=16 | rep | LoRA | 0.5524 | 0.020 (0.027) | 0.082 (0.092) |
| occupation_group σ=16 | out | LoRA | 0.5877 | 0.028 (0.039) | 0.109 (0.126) |

So if Job 1's longer-grid floors are adopted, the corrected-floor worst-case
numbers exist and are mild: output-surface TPR@0.01 ≤ 2.8% mean (3.9% worst seed).

## Averaging attack — middle and hard subspace points (Tier-1 suite, probe seeds [0,1], nested draws, draw seed 20260723+ts)

T1 = max(XGB, MLP, LoRA) on the averaged representation, per train seed.

**middle r=16 σ=16 λ=100** (retrained-at-σ_eff anchors from
`end_to_end_surgical.json` arm D rows r=16 λ=100: σ=8 → 0.5606 max, σ=4 → 0.8285 max):

| N | σ_eff | T1 mean | per-seed | worst | breach (0.55) |
|---|---|---|---|---|---|
| 1 | 16.0 | 0.5312 | 0.5294 0.5332 0.5422 0.5241 0.5270 | 0.5422 | 0/5 |
| 4 | 8.0 | **0.6219** | 0.6640 0.5581 0.6449 0.5835 0.6589 | 0.6640 | **5/5** |
| 8 | 5.7 | **0.8084** | 0.8322 0.8086 0.7860 0.7815 0.8336 | 0.8336 | **5/5** |
| 16 | 4.0 | **0.9293** | 0.9419 0.9232 0.9151 0.9227 0.9436 | 0.9436 | **5/5** |

**hard r=32 σ=16 λ=100, the σ=16 alternative** (anchors r=32 λ=100: σ=8 → 0.5473
max — the original stored winner row; σ=4 → 0.5563 max):

| N | σ_eff | T1 mean | per-seed | worst | breach (0.55) |
|---|---|---|---|---|---|
| 1 | 16.0 | 0.5358 | 0.5374 0.5362 0.5371 0.5285 0.5399 | 0.5399 | 0/5 |
| 4 | 8.0 | 0.5466 | 0.5514 0.5342 0.5501 0.5391 0.5582 | 0.5582 | **3/5** |
| 8 | 5.7 | 0.5498 | 0.5527 0.5384 0.5512 0.5505 0.5564 | 0.5564 | **4/5** |
| 16 | 4.0 | 0.5591 | 0.5582 0.5442 0.5659 0.5635 0.5636 | 0.5659 | **4/5** |

## Registered bets — held or contradicted

Scored against commit `ccbb53e` / `results/averaging_attack_prediction.md`. The
bet's registered scope is the easy cell (already scored in
`results/averaging_attack.json`); middle/hard are generalization tests of the same
claims:

1. **"Monotone in N": HELD.** Easy (stored run): monotone. Middle: monotone on
   every seed. Hard: monotone in the 5-seed mean (0.5358 → 0.5466 → 0.5498 →
   0.5591); one seed (ts1) dips 0.5362 → 0.5342 at N=4, a 0.002 wobble inside the
   bet's own ~0.01 seed-noise scale.
2. **"√N law — N-draw ≈ single-draw at σ/√N (retrained anchors)": HELD on hard,
   CONTRADICTED (defense-breaking direction) on middle.** Hard tracks the anchors
   closely: N=4 mean 0.5466 vs σ=8 anchor 0.5473; N=16 mean 0.5591 vs σ=4 anchor
   0.5563. Middle blows through them: N=4 reads 0.6219 vs the 0.5606 anchor and
   N=16 reads 0.9293 vs 0.8285 — the frozen r=16 model's clean rep retains far
   more in-span attribute content than a model retrained at the lower σ, the same
   direction as the easy-cell finding (0.5596 vs 0.5411) but an order of magnitude
   larger in AUC excess. The prediction file's registered caveat ("training sigma
   also shapes the learned rep — the match is approximate") anticipated the
   direction but not the size.
3. **"Bar likely uncrossed at N=16, breach ~N=64": already CONTRADICTED on easy
   (N=16 breached at 0.5596); the new cells contradict it harder** — middle
   breaches at N=4 on 5/5 seeds, hard at N=4 on 3/5 seeds (N=16: 4/5, mean over
   the bar). No registered claim existed for these cells; reported as
   generalization outcomes.
4. **TPR bet: NOT SCORABLE — no TPR prediction was ever committed.** The
   worst-class numbers above are what such a bet would be scored against.

Certificate implication, now demonstrated on all three subspace points: the
subspace-confined Tier-1 certificates are valid **only under single-draw
exposure**. Query averaging converts the surviving points into breaches — mildly
on hard (σ≈0.56 plateau, the LRT-style ceiling for r=32 λ=100), catastrophically
on middle (0.93 at N=16, approaching the unprotected reading). Deterministic
per-input noise (fixed ε per row) would neutralize the attack; the paper's threat
model should state the assumption.

## Job 3 — reproduction gate (appended per instruction)

Every re-run AUC at every previously stored point was compared against the stored
value; tolerance 0.01. **230/230 comparisons pass, and every single delta is
exactly 0.00000** — the re-run reproduces the stored pipeline bit-exactly (same
machine, same MPS/CPU RNG paths), so all TPR and averaging numbers above are valid
for the paper by this gate.

| point group | comparisons | max \|Δ\| | stored reference |
|---|---|---|---|
| subspace easy / middle (exp10 D, per train seed, rep+out × XGB/MLP/LoRA) | 30 + 30 | 0.00000 | `hardening_seeds.json` `exp10_5seed[].D.per_seed` |
| subspace hard (σ=16 alternative, same layout) | 30 | 0.00000 | `hardening_seeds.json` `hard_cell_sigma16_alternative.per_seed` |
| full-rank T1+T2 × 3 cells (per train seed, rep × XGB/MLP/LoRA/LRT) | 120 | 0.00000 | `two_tier_certification.json` `e2e_cells[].tier1/tier2.per_seed` |
| unprotected σ=0 × 3 cells (train seed 0, rep+out × XGB/MLP) | 12 | 0.00000 | `continuous_cost.json` `rows[].sweep` σ=0 |
| Adult floor points σ=32 / σ=16 (train seed 0, rep+out × XGB/MLP) | 8 | 0.00000 | `continuous_cost.json` `rows[].sweep` σ=32 / σ=16 |

(The N=1 averaging rows equal the subspace rep AUCs by construction — same exposed
matrix — and match the same stored per-seed values; the two Adult floor points and
the unprotected baselines had stored train-seed-0 references only, so their train
seeds 1–4 and all LoRA/floor-LoRA values are new, gate-exempt data.)
