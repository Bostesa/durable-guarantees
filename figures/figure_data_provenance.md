# Figure data provenance (AAAI 2027 submission)

Regenerate any figure with one command (repo venv; matplotlib only, no
experiments are run):

```bash
.venv/bin/python figures/fig1_demolition.py
.venv/bin/python figures/fig2_footprint.py
.venv/bin/python figures/fig3_cost_ramp.py
.venv/bin/python figures/fig4_two_tier.py
```

Shared style: `figures/_style.py` — Times New Roman serif (newtx/Times
body-font match), 8pt minimum, Okabe-Ito colorblind-safe palette (validated
with the six-check palette validator; light hues carry direct labels/hatch as
secondary encoding), 3.3in single-column width, vector PDF with embedded
TrueType (fonttype 42), no figure titles, tight bounding boxes. Color roles
consistent across figures: blue `#0072B2` = our mechanism (Tier 1 solid;
Tier 2 = lighter `#56B4E9` + hatch), vermillion `#D55E00` = probe-shaped
baselines, orange `#E69F00` = CelebA accent, gray `#767676` = the weak
certificate instrument (own panel; identity never color-alone).

| Figure | PDF | Generator | Source JSON(s) | Experiment |
|---|---|---|---|---|
| 1 The demolition | `fig1_demolition.pdf` | `fig1_demolition.py` | `results/honest_reaudit.json` (`part2_smart_erasers.adult` + `master` "E2 noise" rows) | Exp 5 |
| 2 Footprint diagonal | `fig2_footprint.pdf` | `fig2_footprint.py` | `results/diagnostic.json` (`part_b_c`), `results/celeba_pipeline.json` (`cells[*].verdicts`), `results/expansion_floor.json` (`rows`, non-degenerate) | Exps 8, 13 + expansion |
| 3 Cost ramp | `fig3_cost_ramp.pdf` | `fig3_cost_ramp.py` | `results/continuous_cost.json` (`rows`), `results/cliff_or_ramp_constructed.json` (`rows`), `results/expansion_cost.json` (`rows`, non-degenerate; combined r/ρ recomputed live in-script) | Exp 9 + expansion |
| 4 Two-tier certification | `fig4_two_tier.pdf` | `fig4_two_tier.py` | `results/two_tier_certification.json` (`e2e_cells[*].tier1/tier2`) | Exp 14 (5-seed cross-check: Exp 15) |

## Source-mapping notes vs the figure brief

- The brief's `results/footprint_law*.json` and `results/celeba_full_pipeline.json`
  do not exist; the footprint-law data lives in `diagnostic.json` (Exp 8 Part C)
  and `celeba_pipeline.json` (Exp 13). The brief's Fig-3 source
  (`diagnostic.json`) holds Exp 8; the Exp-9 continuous-cost data is in
  `continuous_cost.json` + `cliff_or_ramp_constructed.json`.

## Figure 1 — flags

- **Cell is adult/sex, not HMDA/race as the brief suggested.** The
  certificate-passes/honest-fails contrast only exists on adult/sex: on HMDA
  the projections never passed the certificate (attacked R² 0.17–0.65 > τ),
  so Exp 5 never re-audited them and no honest-XGB reading exists for them in
  any JSON. On adult/sex the certificate reads ≈0.036 (PASS at τ=0.05) for
  every projection while XGBoost reads 0.681–0.683 = the unprotected level
  (0.683).
- **LEACE omitted** (brief asked for it): never R²-stopped on this
  representation → no honest-XGB value recorded in the JSONs. Not
  approximated. **LEOPARD-MMD r=32 omitted**: the re-audit swept MMD r ≤ 16.
- Noise σ=4 row comes from the `master` list (the σ-sweep re-audit);
  σ=1/2/8 and all projections from `part2_smart_erasers.adult`.

## Figure 2 — flags

- CelebA points: marker = the certified deployment arm's output floor
  (`out_floor_best_arm`), whisker = min–max floor across the four arms
  (README's quoted 0.505–0.515 / 0.526–0.553 ranges).
- **The achieved-output-coupling arrow the brief asked for is omitted**: the
  values (0.536 clean / 0.540 winner) appear in the README (Exp 13
  reconciliation) but are stored in **no results JSON** (verified by key
  search). Per the no-approximation rule the reconciliation stays in the
  caption; the point is annotated as the loose-upper-bound outlier.
- README scope note carried over: the two adult tabular floors are upper-ish
  estimates (representation only partly scrubbed within σ ≤ 12).
- **External cells (expansion swap, commit 80b6cdd)**: the 7 usable expansion
  cells from `expansion_floor.json` plot as vermillion open circles ("external
  cells" legend); the 2 degenerate diabetes-hospital cells (clean lift ≈ 0, so
  the upper-bound reading fails) are excluded, as recorded in the JSON.
  dutch/sex/occupation is annotated as a loose-upper-bound point (Δ = −0.074
  below the diagonal, same reading as Attractive→Young). The pre-expansion
  figure lives in git history; `fig2_footprint_v2.*` are the development
  copies of the same content.

## Figure 3 — flags

- Cost plotted = `cost_durable` (representation **and** output ≤ 0.55 under
  the Exp-9 black-box battery). Headline correlation after the expansion swap:
  **r = 0.799, ρ = 0.828, n = 27** natural cells (20 original + 7 external),
  recomputed live in-script from `continuous_cost.json` + `expansion_cost.json`
  and quoted in the caption (the in-plot stats text was removed in commit
  6787270). The pre-expansion values (r = 0.795, ρ = 0.848, n = 20) remain in
  `continuous_cost.json` `stats.cost_durable`. No censored rows plotted; the
  2 degenerate diabetes-hospital expansion cells are excluded (recorded in
  `expansion_cost.json`). `fig3_cost_ramp_v2.*` are the development copies.

## Figure 4 — flags

- Error bars are the **3-seed** spread from Exp 14's certification (the
  brief asked for 5-seed bars; Exp 15's 5-seed replicate exists for only one
  of the six points — easy-cell Tier 1, σ=8: 5-seed lift 0.0211 ± 0.0005 =
  94.6% kept, identical to the 3-seed value plotted).
- Cell difficulty labels follow the repo's canonical assignment (easy =
  hmda/loan_decision 0.514, middle = hmda/loan_amount_band 0.584, hard =
  adult/income 0.603); the brief's parenthetical had them permuted.

## Final plotted numbers (sanity-checked against the README)

Figure 1 (method: certificate R² / honest XGB AUC) — all match the README
Exp-5 table: no erasure 0.104/0.683 · MMD r=8 0.036/0.681 · MMD r=16
0.036/0.681 · HSIC r=8 0.036/0.683 · HSIC r=16 0.037/0.681 · HSIC r=32
0.035/0.683 · noise σ=1 0.046/0.589 · σ=2 0.034/0.550 · σ=4 0.026/0.534 ·
σ=8 0.010/0.514.

Figure 2 (predictor, floor) — tabular/CelebA points match README Exp-8
Part C exactly: (0.509, 0.509) (0.514, 0.533) (0.543, 0.557) (0.584, 0.571)
(0.603, 0.679) (0.676, 0.629); CelebA (0.522, 0.510 [0.505–0.515]) and
(0.734, 0.526 [0.526–0.553]). External points match `expansion_floor.json`
exactly (the README predates the expansion): (0.5049, 0.5358)
(0.5094, 0.5034) (0.5122, 0.5536) (0.5420, 0.5569) (0.5568, 0.5865)
(0.5912, 0.5664) (0.6499, 0.5761).

Figure 3 — combined natural r = 0.799, ρ = 0.828, n = 27 (recomputed
live; pre-expansion stored values r = 0.795, ρ = 0.848, n = 20 ✓ README);
20 original + 7 external + 9 constructed points plotted as stored. External
(predictor, cost_durable) match `expansion_cost.json` exactly:
(0.5049, 0.0354) (0.5094, 0.1791) (0.5122, 0.1052) (0.5420, 0.2149)
(0.5568, 0.7133) (0.5912, 0.9976) (0.6499, 0.8763).

Figure 4 (% clean lift kept ± std, σ) — README quotes 95→55, 57→22, −1/−7:
easy 94.6±2.2 (σ=8) → 55.5±4.4 (σ=20); middle 56.6±0.7 (σ=12) → 21.7±0.5
(σ=24); hard −0.5±1.5 (σ=32) → −7.0±1.5 (σ=64). ✓ (rounding only).

No numeric mismatches between the JSONs and the README were found.
