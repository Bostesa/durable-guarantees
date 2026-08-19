# Preregistration record

Every entry here is dated and tied to a commit. The convention:

* `<run>_prediction.md` states what we expected the run to show. It was written
  and committed **before** the run existed, so the scoring rule is fixed up
  front and the verdict cannot be argued after the fact.
* `<run>_results.md` states what the run actually showed, scored against that
  prediction. Falsified predictions stay in the record and are marked falsified.

The remaining files are standalone analyses. They derive paper numbers from
already committed JSON, or audit stored values. They re-run nothing.

Raw numbers live in `results/`. These files say what was expected, what was
found, and where the two disagreed.

## Baselines under the two tier gauntlet

| Method | Prediction | Results |
|---|---|---|
| FARE | [fare_prediction.md](fare_prediction.md) | [fare_results.md](fare_results.md) |
| FNF, Fair Normalizing Flows | [fnf_prediction.md](fnf_prediction.md) | [fnf_results.md](fnf_results.md) |
| Fair PCA | [fairpca_prediction.md](fairpca_prediction.md) | [fairpca_results.md](fairpca_results.md) |

## Our channel, attacks and hardening

| Run | Prediction | Results |
|---|---|---|
| Isolate then noise against full rank noise | [isolate_vs_fullrank_prediction.md](isolate_vs_fullrank_prediction.md) | [isolate_vs_fullrank_results.md](isolate_vs_fullrank_results.md) |
| Averaging attack on the subspace channel | [averaging_attack_prediction.md](averaging_attack_prediction.md) | `results/averaging_attack.json` |
| Knows Q attacker | [knows_q_prediction.md](knows_q_prediction.md) | `results/knows_q.json`, `results/knows_q_highsigma.json` |
| TPR at low FPR | none registered, disclosed in the results file | [tpr_results.md](tpr_results.md) |
| TPR on the 59 failing points | [tpr_failing59_prediction.md](tpr_failing59_prediction.md) | [tpr_failing59_results.md](tpr_failing59_results.md) |
| Worst pair sigma sweep, all pairs | [worstpair_sweep_prediction.md](worstpair_sweep_prediction.md) | [worstpair_sweep_results.md](worstpair_sweep_results.md) |
| Worst pair sigma sweep, supported pairs | [worstpair_supported_sweep_prediction.md](worstpair_supported_sweep_prediction.md) | `results/worstpair_supported_sweep.json` |
| 5 seed recertification of two operating points | [gate_5seed_prediction.md](gate_5seed_prediction.md) | `results/gate_5seed.json` |
| Generalization to a fresh partition | [fresh_partition_prediction.md](fresh_partition_prediction.md) | [fresh_partition_results.md](fresh_partition_results.md) |

## Hybrid mechanisms

| Run | Prediction | Results |
|---|---|---|
| Isolate plus coarsen | [hybrid_coarsen_prediction.md](hybrid_coarsen_prediction.md) | [hybrid_coarsen_results.md](hybrid_coarsen_results.md) |
| Isolate plus randomized response | [hybrid_rr_prediction.md](hybrid_rr_prediction.md) | [hybrid_rr_results.md](hybrid_rr_results.md) |

## Differential privacy

| File | What it is |
|---|---|
| [dp_feasibility.md](dp_feasibility.md) | Whether the released representation admits a per record DP guarantee at all |
| [dp_bet.md](dp_bet.md) | The registered bet on the clipped full rank channel |
| [dp_bet_results.md](dp_bet_results.md) | That bet, scored |
| [proposition3_dp_guarantee.md](proposition3_dp_guarantee.md) | Proposition 3 and its proof, with the revision that followed the verification audit |

## Standalone analyses

| File | What it is |
|---|---|
| [imputation_baseline.md](imputation_baseline.md) | The label coupling predictor against a pure marginals attacker, plus the floor against predictor table over 27 natural cells |
| [floor_censoring.md](floor_censoring.md) | Audit of the stored output floor readings on the two Adult cells, and which sigma grid produced each |
