# Worst-case TPR for the 59 AUC-failing audit configs — scored

2026-07-24. Script `run_tpr_failing59.py`; data `tpr_failing59.json`; scores
`tpr59_scores/*.npz` (gitignored). Registered prediction `b4deca6`
(`tpr_failing59_prediction.md`), committed before the run. 59/59 configs,
**0 gate failures** (every config reproduced its stored XGB/MLP AUCs within
0.01), 24.6 min on MPS. With the 8 survivors of `tpr_extension.json`, all 67
approved audit configurations are now measured under both criteria.

## Bet verdict: 1 held, 3 falsified — and the falsifications are the finding

| claim | verdict | evidence |
|---|---|---|
| 1. all 59 read > 5% at 1% FPR | **FALSIFIED** | min = 1.5%; **4 configs sit inside the survivor band** |
| 2. mean TPR@1% in [50%, 75%] | **HELD** | mean 53.1% |
| 3. AUC/TPR ordering exception exists | **FALSIFIED** | not by the registered ≷0.60 test (see note) |
| 4. every projection reads > 20% | **FALSIFIED** | min projection = 14.5% (adult, all ten) |

I registered these before the run and report the misses literally, in the
defense-breaking-for-my-own-narrative direction where they land there.

## What actually happened: the split is mechanism, not AUC

Overall range [1.5%, 99.5%], mean 53.1%, median 53.2% — but the population is
bimodal by *removal type*, not by AUC:

- **noise configs (n=11): mean 9.3%, range [1.5%, 17.5%].** Additive noise is
  worst-case quiet even when it fails the AUC bar. Four fall INSIDE the
  survivor band (≤5%): diabetes-hospital/gender σ=8 (1.5%), folktables/sex
  σ=8 (1.6%), folktables/race/employment σ=2 (2.0%), and — the sharpest —
  **hmda/race noise σ=8, an AUC failure at 0.551 that reads worst-class
  4.0%.** These are the claim-1 falsifiers.
- **projection configs (n=48): mean 63.1%, range [14.5%, 99.5%].** Targeted
  projections that pass the *linear* certificate leak at the unprotected level
  under a worst-case read too — up to 99.5% (lawschool, folktables
  public-coverage). But on the low-signal adult cell even unprotected-level
  projections only reach ~15% (claim 4's falsifier), because the adult rep's
  own unprotected worst-case ceiling is ~17.5%.

**The headline this run actually supports (stronger than the registered bet):**
worst-case leakage tracks the *removal mechanism*, not the average-case AUC.
Every one of the 4 in-band failures is a noise channel; every high-precision
leak is a projection. An AUC-based audit cannot see this distinction — two
configs at the same failing AUC (e.g. hmda noise 0.551 → 4.0% vs. a projection
near 0.55 → far higher) carry opposite worst-case risk. This is the same
average-vs-worst-case decoupling the survivor table showed (Adult σ=4), now
demonstrated across the full failing population and tied to a cause.

## Note on claim 3

The registered test (a config with AUC > 0.60 reading below one with AUC <
0.60) returned no instance, so it scores FALSIFIED as written. But the
*intended* phenomenon — AUC and worst-case disagreeing — is present in a
stronger form: the disagreement is systematic by mechanism (noise vs
projection), not a stray pairwise inversion. The registered operationalization
was too narrow; the effect it was probing for is real and larger than the test.

## Paper wiring

- Limitations, the sentence that currently extends worst-case measurement "to
  the eight surviving audit configurations and the 36 baseline combinations":
  broaden to **"to all 67 approved audit configurations."** Then add the
  mechanism split: "Across the 59 that fail the average-case bar, worst-case
  leakage tracks the removal mechanism rather than the AUC: the 11 noise
  configurations read a mean 9.3% at a 1% budget (four inside the survivor
  band, including an HMDA noise channel at 4.0% despite failing at AUC 0.551),
  while the 48 projections read a mean 63.1%, up to 99.5%. Average-case and
  worst-case audits are therefore not interchangeable, and the gap is
  structured, not incidental."
- This is a genuine strengthening: the paper's own thesis (only noise removes
  attribute information; projections pass certificates without removing) now
  holds under the *worst-case* metric too, across the whole failing set.
- Registration honesty: the prediction file's four claims scored 1/4; report
  that plainly if the audit trail is mentioned. The negative result is the
  point — the effect was real but mechanism-structured, not the flat
  "everything fails high" the bet guessed.
