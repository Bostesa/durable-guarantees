# Floor censoring verification — adult/sex/income and adult/sex/occupation_group

Job 1 report (2026-07-23). Report-only; nothing modified. Feeds Job 2's capture
list (the two first-hidden σ values in §b).

## (a) Grid behind the stored floor readings 0.6787 / 0.6286

Source: `results/diagnostic.json` → `part_b_c[].out_floor`, with the per-σ readings
in `part_b_c[].sweep`. The grid is `noise_sweep` in the same file, set by
`experiments/diagnostic.py:72`:

```python
NOISE_SWEEP = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0]   # BN-unit isotropic noise on the public channel
```

On that grid neither Adult cell's representation is hidden: at σ=12,
P_max = 0.6297 (income) and 0.5793 (occupation_group), both > 0.55, and the stored
`sigma_rep` field is `null` for both cells.

## (b) Grid behind the longer-grid readings 0.5612 (σ=32) / 0.5896 (σ=16)

Source: `results/continuous_cost.json` → `sigma_grid` and `rows[].sweep`, set by
`experiments/continuous_cost.py:60`:

```python
SIGMA_GRID = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0]
```

(the sweep breaks adaptively once both P_max and out_max ≤ 0.55; income was swept to
σ=48, occupation_group to σ=32). On this grid:
income first hides at σ=32 (P_max 0.5279; σ=24 reads 0.5542 > 0.55), out_max there
= **0.5612**; occupation_group first hides at σ=16 (P_max 0.5393), out_max there =
**0.5896**. The shared grid prefix (σ ≤ 12) is numerically identical between the two
files — same `train_model` channel, same probe seeds {0,1}, train seed 0 — so the
two sweeps are the same measurement, differing only in grid extent.

## (c) The operative rule, quoted from code

`experiments/diagnostic.py:321-336` (the code that produced the stored readings):

```python
rep_pts = [p for p in sweep if p["P_max"] <= HONEST_AUC_BAR]
...
sigma_rep = min(rep_pts, key=lambda p: p["sigma"]) if rep_pts else None
...
# Footprint-law point: output-leak FLOOR = out_max at the σ that hides the rep
# (representation gone; whatever remains in the output is the label's own leak).
out_floor = sigma_rep["out_max"] if sigma_rep else sweep[-1]["out_max"]
```

The expansion pipeline's version (`experiments/expansion_pipeline.py:102-105`,
`measure_floor`) is the same rule with an explicit censoring flag:

```python
hid = [p for p in sweep if p["P_max"] <= BAR]
sigma_floor = hid[0]["sigma"] if hid else sweep[-1]["sigma"]
rep_hidden = bool(hid)
```

**Do the stored readings satisfy the rule on the full grid that existed when they
were stored?** Yes. Git history: `results/diagnostic.json` was committed at
`9fe28a0` 2026-07-03 01:36:38 −0400 ("Experiment 8: the diagnostic"); the longer
grid first exists in `results/continuous_cost.json`, committed at `583c85f`
2026-07-03 02:23:34 −0400 ("Experiment 9") — **47 minutes later**. At storage time
the only grid in existence was NOISE_SWEEP (σ ≤ 12), on which `rep_pts` is empty for
both cells, so the rule's explicit `else sweep[-1]` branch applies and yields exactly
the stored 0.6787 / 0.6286. `diagnostic.json` was never regenerated after Exp 9 (its
only commit is 9fe28a0).

## (d) Verdict per cell

- **adult/sex/income: convention difference, not a bug.** The rule was applied
  correctly on the grid available at the time; the `else`-branch is an explicit,
  deliberate censoring fallback, and the JSON records the censoring
  (`sigma_rep: null`). The σ=32 / 0.5612 reading comes from a grid created later by
  a different experiment.
- **adult/sex/occupation_group: convention difference, not a bug.** Same reasoning;
  censoring recorded as `sigma_rep: null`; the σ=16 / 0.5896 reading postdates the
  stored value.

Corroborating consistency check: `continuous_cost.json` itself reports both cells as
*uncensored* on its longer grid (`censored_rep: false`, interpolated
`sigma_rep` = 25.28 / 14.93), i.e. the codebase's later, longer-grid measurement
already knows the rep hides — it simply never wrote an `out_floor` key.

## (e) Consequences — what would change if the longer-grid floors were adopted

Not triggered as a *correction* (verdict: convention, not bug), but for completeness,
adopting the honest longer-grid floors would change exactly this and nothing else:

- **fig2_footprint plotted points** (`paper/figures/fig2_footprint.py` reads
  `diagnostic.json` `part_b_c[].out_floor` unfiltered):
  (0.6034, 0.6787) → (0.6034, 0.5612) and (0.6762, 0.6286) → (0.6762, 0.5896).
  Both points would move from on/near the identity diagonal to visibly **below**
  it — same side as the dutch loose-bound point.
- **The identity diagonal itself is data-independent** (`ax.plot(lims, lims)` with
  fixed `lims = (0.48, 0.78)`): unchanged. Any *prose* that quotes the last data
  point as (0.676, 0.629) would need the 0.629 → 0.590 update.
- **r = 0.799, ρ = 0.828 (fig3, n=27): unchanged.** Those are the
  predictor-vs-`cost_durable` correlations; `cost_durable`/`cost_rep` are computed
  from `P_max`/`both_max` bar-crossings on the continuous_cost grid
  (`crossing_cost`), which never consume `out_floor`. No stored cost value consumes
  the floor.
- **expansion_analysis floor_points / the 7 external floors: unchanged** (different
  cells, different file).

One inconsistency worth flagging even under the "convention" verdict: the expansion
pipeline annotates its one rep-not-hidden cell (dutch/sex/occupation,
`rep_hidden_within_grid: false`) as a loose upper bound in fig2, while the two Adult
cells — in exactly the same censored situation on the diagnostic grid — are plotted
without any such distinction. If the paper keeps the stored values, the two Adult
points deserve the same loose-bound annotation; if it adopts the longer-grid values,
they become ordinary (uncensored) floor points at 0.5612 / 0.5896. Job 2 captures
attacker scores at σ=32 / σ=16 for these two cells so corrected-floor worst-case
(TPR@low-FPR) numbers exist either way.
