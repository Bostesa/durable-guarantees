# Worst-pair sigma sweep — results and scoring

Run 2026-07-29, 60.8 min sweep + 1.3 min null, local MPS. Registered at
**50d7537 before implementation** (`worstpair_sweep_prediction.md`). Data:
`worstpair_sweep.json`, `worstpair_null.json`, `worstpair_supported.json`.
No paper edits.

## Headline: no passing operating point exists, and the reason is not the one predicted

**0 of 6 (channel, cell) combinations found a sigma clearing 0.55 on worst-pair
on both surfaces.** Sweeps terminated either on negative utility or at the
sigma ceiling. But the sweep also showed the criterion itself is unusable as
specified, which changes what the result means.

## The instrument is broken at these class sizes

HMDA race is severely imbalanced:

| class | rows | share | rows in a 25% test split |
|---|---|---|---|
| 0 | 41,134 | 64.5% | 10,283 |
| 1 | 3,796 | 6.0% | 949 |
| 2 | 16,968 | 26.6% | 4,242 |
| 3 | 1,344 | 2.1% | 336 |
| 4 | **505** | **0.79%** | **126** |

Two independent signs that the all-pairs statistic measures sampling noise:

1. **Non-monotonicity in sigma.** easy/subspace representation worst-pair reads
   0.5793 at sigma=16 and **0.5963 at sigma=512** — higher after 32x more noise,
   at utility −42.6%. Attribute recovery cannot increase with noise.
2. **A direct null.** With features carrying zero attribute information by
   construction (isotropic Gaussian, real label distribution, identical
   pipeline), worst-pair max-over-seeds reads **0.5556 — above the 0.55 bar**,
   while macro-OVR correctly reads 0.5073.

Per-pair null decomposition (zero signal, so every value above 0.5 is noise):

| pair | test n | null max | | pair | test n | null max |
|---|---|---|---|---|---|---|
| (0,1) | 10283/949 | 0.5190 | | (1,3) | 949/336 | 0.5434 |
| (0,2) | 10283/4242 | 0.5129 | | (1,4) | 949/126 | 0.5435 |
| (0,3) | 10283/336 | 0.5417 | | (2,3) | 4242/336 | 0.5334 |
| (0,4) | 10283/126 | **0.5556** | | (2,4) | 4242/126 | **0.5551** |
| (1,2) | 949/4242 | 0.5217 | | (3,4) | 336/126 | 0.5475 |

Every pair involving class 3 or 4 has a null at or near the bar. Restricted to
the three classes with >= 3000 rows, the null max falls to **0.5217**, leaving
usable headroom.

**So the all-pairs worst-pair bar cannot be passed by any mechanism, including
a random matrix.** No sweep could have succeeded.

## But the leak is real, just smaller than the all-pairs number suggested

Rescoring our stored configurations under the support-restricted criterion
(pairs among classes 0/1/2 only; null 0.5217):

| config | surface | macro | worst (all pairs) | worst (supported) | verdict |
|---|---|---|---|---|---|
| subspace_easy | rep | 0.5162 | 0.5793 | **0.5412** | passes |
| subspace_easy | out | 0.5489 | 0.6441 | 0.6068 | fails |
| subspace_middle | rep | 0.5435 | 0.6420 | 0.5688 | fails |
| subspace_middle | out | 0.7218 | 0.8770 | 0.7923 | fails |
| fullrank_tier1_easy | rep | 0.5493 | 0.6449 | 0.6449 | fails |
| fullrank_tier1_easy | out | 0.5585 | 0.6759 | 0.6759 | fails |
| fullrank_tier1_middle | rep | 0.5579 | 0.6314 | 0.5912 | fails |
| fullrank_tier1_middle | out | 0.5866 | 0.7033 | 0.6243 | fails |
| fullrank_tier2_easy | rep | 0.5557 | 0.6145 | 0.6145 | fails |
| fullrank_tier2_easy | out | 0.5439 | 0.6084 | 0.6084 | fails |
| fullrank_tier2_middle | rep | 0.5590 | 0.6296 | 0.5812 | fails |
| fullrank_tier2_middle | out | 0.5386 | 0.6220 | 0.5748 | fails |

Pass counts over the 12 pairs: **macro-OVR 6, worst-pair all-pairs 0,
worst-pair supported 1.** The rare classes inflate the reading (subspace_easy
rep 0.5793 -> 0.5412) but do not create it: against a 0.5217 null, readings of
0.54–0.79 are genuine pair-level leakage that macro-OVR averaging hides.

## Prediction scoring

- **P1 — PARTIAL, 3 of 6 correct.** All three middle-cell NOs were right; all
  three easy-cell YESs were wrong. I predicted a passing sigma would exist on
  easy because utility decays slowly there; it does decay slowly (subspace
  still +49.7% at sigma=256), but no sigma passes because the criterion's null
  sits above the bar.
- **P2 — NOT APPLICABLE.** Scored only where P1's YES was correct; none were.
- **P3 — FALSIFIED as stated.** I predicted the worst-pair label-coupling
  predictor would exceed 0.55 on **both** cells. Measured: **easy 0.5444
  (below), middle 0.6858 (above)**. So the output-side floor explanation holds
  for middle and not for easy. The milder second failure mode I named did hold
  where it applied — on middle the output surface binds throughout (out
  worst-pair 0.88 -> 0.60 while rep reaches 0.56). The actual mechanism on easy
  was one I did not name: the statistic's own null exceeds the bar.

## Terminating point per combination

| cell | channel | stop sigma | rep worst | out worst | utility | why it stopped |
|---|---|---|---|---|---|---|
| easy | subspace | 512 | 0.5963 | 0.5732 | −42.6% | utility negative |
| middle | subspace | 2048 | 0.5922 | 0.6036 | +73.1% | sigma ceiling |
| easy | full-rank T1 | 64 | 0.5595 | 0.5586 | −0.7% | utility negative |
| middle | full-rank T1 | 96 | 0.5592 | 0.5655 | −1.5% | utility negative |
| easy | full-rank T2 | 40 | 0.5739 | 0.5801 | −0.4% | utility negative |
| middle | full-rank T2 | 96 | 0.5592 | 0.5874 | −1.5% | utility negative |

## Gate — mis-specified by me, reported plainly

Three gates are recorded as failures (easy/fullrank_t1 0.5457 vs 0.5351;
middle/fullrank_t1 0.5579 vs 0.5408; easy/fullrank_t2 0.5557 vs 0.5377). These
are **not** reproduction failures. My gate compares this run's
**max-over-seeds** macro-OVR against a **mean-over-seeds** stored value, so it
is not like-for-like and is expected to read high; the three "failures" exceed
tolerance by 0.011–0.018, consistent with the max/mean gap seen elsewhere
(the max-over-seeds diagnostic moved readings by 0.004–0.021). The two gates
that pass do so despite carrying the stricter statistic. A correct gate would
compare means; that was an error in my instrument, not evidence about the
channels, and no cell should be invalidated on this basis.

## What this means for the paper

The worst-pair objection does not have a fix by adding noise, on either cell,
for any of the three channels. Two separable facts should be reported together:

1. All-pairs worst-pair on a 0.79%-prevalence class is not a usable criterion —
   its no-signal null is 0.5556, above the bar. A reviewer proposing it should
   be shown the null.
2. Support-restricted worst-pair **is** usable (null 0.5217), and under it 11 of
   our 12 (config, surface) pairs still fail, at 0.54–0.79. Macro-OVR hides a
   real pair-level leak on the HMDA cells.

The binary Adult cell is unaffected throughout, since worst-pair equals
macro-OVR there.
