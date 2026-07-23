# Imputation baseline — label-coupling predictor vs a pure-marginals attacker, and the floor-vs-predictor table

Report-only analysis (2026-07-23). No experiment was re-run; every number below is
read from (or arithmetically derived from) already-committed results JSONs, with the
source named per number. "The 27 natural cells" = the 20 cells of
`results/continuous_cost.json` plus the 7 non-degenerate external cells of the
expansion (`results/expansion_cost.json` / `expansion_floor.json`); the 2 degenerate
diabetes-hospital readmission cells are listed at the bottom for completeness but are
excluded from the n=27 statistics, as in the paper.

## 1. Is the per-cell predictor stored? — Yes, for all 27 (and the 2 degenerate)

The predictor is defined in `experiments/diagnostic.py::label_predictor` (line 209):
`probe_sklearn("XGB", task.reshape(-1, 1), attr, n_attr, seeds=[0, 1, 2])` — an
XGBoost classifier (100 trees, depth 6, lr 0.3, `tree_method="hist"`; config in
`experiments/hardening_test.py::_probe`) fit on the **1-dimensional task label alone**,
75/25 attribute-stratified held-out split, AUC (macro-OvR when the attribute is
multiclass), mean ± std over probe seeds {0, 1, 2}.

Storage:

| cells | primary source | duplicate copies |
|---|---|---|
| 20 original | `continuous_cost.json` → `rows[].predictor` (+ `predictor_std`) | 6 of them also in `diagnostic.json` → `part_b_c[].predictor` |
| 9 external (7 usable + 2 degenerate) | `expansion_cost.json` → `rows[].predictor` (+ `predictor_std`) | `expansion_floor.json` → `rows[].predictor`; `expansion_analysis.json` → `new_cell_pairs` / `summary_table.coupling`; per-cell shards in `results/expansion_shards/*.json` |

## 2. Would a pure-marginals attacker (empirical P(s|y) lookup) differ from XGB-on-y-alone?

**No, not by anything above seed noise — and for binary-y cells the two are exactly
equivalent in AUC.**

The argument, which needs only facts checked in the repo:

- Any model fitted on the single column y produces a score that is a **function of y**
  — piecewise-constant over the K distinct task values. AUC depends only on the
  *ranking* those K score values induce.
- The pure-marginals attacker scores each example by the empirical P(s=c | y) lookup.
  If the lookup is built on the same 75% train split the XGB sees, both attackers rank
  the K values by (an estimate of) the train-split conditional rate.
- **Binary y (K=2)** — 16 of the 27 cells (all loan_decision, tract_denial_high,
  medication_change_outcome, income, employment, public_coverage, pass_bar,
  dutch occupation cells): the ROC of any y-alone scorer has a single interior point,
  fully determined by which of the two y values gets the higher score. Any
  non-degenerate fitted model matches the empirical direction, so held-out AUC is
  **identical** to the lookup's — exact equivalence, not approximate.
- **Multi-valued y (K ∈ {4, 5, 6, 9})** — the other 11 cells (education_level K=4,
  loan_amount_band K=5, occupation_group K=6, primary_diagnosis_category K=9):
  100 depth-6 trees on ≤ 9 distinct scalar values give every value its own leaf; the
  boosted score converges to the train-split conditional log-odds per value, i.e. the
  same ranking as the lookup. The only possible divergence is XGB's regularization
  (`min_child_weight=1`, L2 λ=1 defaults) shrinking rare-value estimates enough to
  swap two *near-tied* values — and a swap of near-tied values changes AUC by less
  than the tie gap itself, which is below the stored per-cell `predictor_std`
  (0.0005–0.0145). Sample sizes here (16.6k–63.7k rows) make per-value counts large,
  so even that is unlikely.
- If instead the lookup were built on the **full sample** (train+test) rather than the
  75% split, the difference is the sampling noise of P(s|y) between the two estimates
  — again second-order relative to `predictor_std`.

Task-label cardinality per cell (computed read-only via the repo's own loaders,
`experiments/diagnostic.py::load_cell` and
`experiments/expansion_pipeline.py::load_cell_by_tag`):

| task | K (n_task) | rows |
|---|---|---|
| adult income | 2 | 24,145 |
| adult education_level | 4 | 24,145 |
| adult occupation_group | 6 | 24,145 |
| hmda loan_decision / tract_denial_high | 2 | 63,747 |
| hmda loan_amount_band | 5 | 63,747 |
| diabetes medication_change_outcome | 2 | 50,053 |
| diabetes primary_diagnosis_category | 9 | 50,053 |
| folktables employment / public_coverage / income | 2 | ~60,000 |
| lawschool pass_bar | 2 | 16,638 |
| dutch occupation | 2 | 48,336 |
| diabetes_hospital readmit_30_days (degenerate cells) | 2 | 60,000 |

**Conclusion:** y is low-cardinality (K ≤ 9) in every cell; the XGB-on-y-alone
predictor **is** the imputation/marginals baseline — exactly for the 16 binary-y
cells, and up to sub-seed-noise tie effects for the 11 multi-valued-y cells. A
separate "pure marginals" run would not produce a distinguishable number.

## 3. Are per-cell floor readings stored? — Partially; the rest derive from stored sweeps

Floor = best-attacker AUC recovering the attribute **from the task outputs (logits)**
at the first swept σ where the representation is hidden (P_max ≤ 0.55). Storage:

- **Stored directly (7 external + 2 degenerate):** `expansion_floor.json` →
  `rows[].floor` (3 probe seeds {0,1,2}, re-measured at the floor σ).
- **Stored directly (6 of the 20 originals):** `diagnostic.json` → `part_b_c[].out_floor`
  (from the Part-B sweep, 2 probe seeds {0,1}). **Caveat:** diagnostic's σ grid capped
  at 12; for `adult/sex/income` and `adult/sex/occupation_group` the rep was *not yet
  hidden* at σ=12 (`sigma_rep: null`), so the stored `out_floor` is the grid-censored
  reading at σ=12 — this is what fig2 plots for those two points.
- **Not stored as a named key (remaining 14 originals):** derivable from
  `continuous_cost.json` → `rows[].sweep` (per-σ `P_max` / `out_max`, grid up to
  σ=192, 2 probe seeds {0,1}) by applying the identical rule the expansion pipeline
  uses (`expansion_pipeline.py::measure_floor`: first sweep entry with P_max ≤ 0.55).
  Marked **(d)** below. On the extended grid every one of the 20 cells does hide its
  rep, so no derived value is censored. For the 4 uncensored diagnostic-overlap cells
  the derived value reproduces the stored `out_floor` to all printed digits (same
  sweep, same seeds), which validates the derivation.

## 4. Two-column table — floor vs predictor (leakage above the prior)

Sorted by predictor. Floor source: **[E]** = `expansion_floor.json` (stored, 3 seeds),
**[D]** = `diagnostic.json` `out_floor` (stored, 2 seeds), **(d)** = derived from
`continuous_cost.json` `sweep` (2 seeds). Predictor source: `continuous_cost.json`
for originals, `expansion_cost.json` for externals. Δ = floor − predictor.

| cell | K | predictor ± std | floor | Δ (floor − predictor) |
|---|---|---|---|---|
| hmda/sex/loan_decision | 2 | 0.5031 ± 0.0008 | 0.5300 (d) | +0.0269 |
| folktables/race/employment | 2 | 0.5049 ± 0.0022 | 0.5358 [E] | +0.0309 |
| diabetes/gender/medication_change_outcome | 2 | 0.5093 ± 0.0030 | 0.5088 [D] | −0.0005 |
| lawschool/sex/pass_bar | 2 | 0.5094 ± 0.0026 | 0.5034 [E] | −0.0060 |
| folktables/race/public_coverage | 2 | 0.5122 ± 0.0006 | 0.5536 [E] | +0.0414 |
| hmda/race/loan_decision | 2 | 0.5138 ± 0.0042 | 0.5333 [D] | +0.0195 |
| hmda/race/tract_denial_high | 2 | 0.5176 ± 0.0026 | 0.5551 (d) | +0.0375 |
| diabetes/age_bucket/medication_change_outcome | 2 | 0.5215 ± 0.0041 | 0.5270 (d) | +0.0055 |
| adult/sex/education_level | 4 | 0.5386 ± 0.0033 | 0.5582 (d) | +0.0196 |
| folktables/race/income | 2 | 0.5420 ± 0.0030 | 0.5569 [E] | +0.0149 |
| diabetes/gender/primary_diagnosis_category | 9 | 0.5431 ± 0.0012 | 0.5569 [D] | +0.0138 |
| diabetes/race/primary_diagnosis_category | 9 | 0.5443 ± 0.0014 | 0.5606 (d) | +0.0163 |
| adult/marital_status/education_level | 4 | 0.5551 ± 0.0008 | 0.5672 (d) | +0.0121 |
| adult/race/income | 2 | 0.5560 ± 0.0075 | 0.5497 (d) | −0.0063 |
| folktables/sex/income | 2 | 0.5568 ± 0.0017 | 0.5865 [E] | +0.0297 |
| hmda/sex/loan_amount_band | 5 | 0.5630 ± 0.0013 | 0.5807 (d) | +0.0177 |
| adult/age_group/income | 2 | 0.5677 ± 0.0059 | 0.5537 (d) | −0.0140 |
| hmda/race/loan_amount_band | 5 | 0.5835 ± 0.0027 | 0.5709 [D] | −0.0126 |
| adult/race/education_level | 4 | 0.5886 ± 0.0145 | 0.5311 (d) | −0.0575 |
| lawschool/race/pass_bar | 2 | 0.5912 ± 0.0057 | 0.5664 [E] | −0.0248 |
| adult/sex/income | 2 | 0.6034 ± 0.0012 | 0.6787 [D]† / 0.5612 (d) | +0.0753† / −0.0422 |
| adult/marital_status/occupation_group | 6 | 0.6069 ± 0.0041 | 0.5922 (d) | −0.0147 |
| hmda/ethnicity/loan_amount_band | 5 | 0.6235 ± 0.0005 | 0.5552 (d) | −0.0683 |
| dutch/sex/occupation | 2 | 0.6499 ± 0.0072 | 0.5761 [E]‡ | −0.0738 |
| diabetes/age_bucket/primary_diagnosis_category | 9 | 0.6647 ± 0.0044 | 0.5161 (d) | −0.1486 |
| adult/sex/occupation_group | 6 | 0.6762 ± 0.0026 | 0.6286 [D]† / 0.5896 (d) | −0.0476† / −0.0866 |
| adult/marital_status/income | 2 | 0.6884 ± 0.0026 | 0.5482 (d) | −0.1402 |
| *diabetes_hospital/gender/readmit_30_days* (degenerate) | 2 | 0.4984 ± 0.0031 | 0.5223 [E] | +0.0239 |
| *diabetes_hospital/race/readmit_30_days* (degenerate) | 2 | 0.5010 ± 0.0009 | 0.5065 [E] | +0.0055 |

† Grid-censored stored value: diagnostic's sweep stopped at σ=12 with the rep still
exposed, so its `out_floor` is the σ=12 reading (this is the value fig2 plots). The
derived value applies the honest first-σ-hidden rule on continuous_cost's longer grid
(rep hides at σ=32 for adult/sex/income, σ=16 for adult/sex/occupation_group).

‡ The paper already annotates dutch as a loose-upper-bound floor point (fig2).

Reading: at low coupling the floor sits at or slightly above the predictor (outputs
leak roughly what the label gives away — the footprint-law diagonal, with the small
positive Δ being genuine "leakage above the prior" from the representation channel
that survives into the logits). At high coupling the derived floors fall *below* the
predictor: the σ needed to hide the representation on those cells is so large it also
degrades the output channel (and the task), so the first-hidden reading undershoots
the prior — an upper-bound-of-the-mechanism reading, not evidence the prior itself is
removed (the label prior is not removable by any representation channel).

Note: the two degenerate diabetes-hospital cells are excluded from n=27 because the
task is not learned there (clean lift ≈ +0.0005), so their "floors" have no
upper-bound interpretation. CelebA cells (`celeba_pipeline.json`) also store floors
but are the vision extension, not natural cells, and are out of scope here.
