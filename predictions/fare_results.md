# FARE under the two-tier gauntlet — results and scoring of the registered claims

Run: 2026-07-24. Prediction committed at ab76813 BEFORE any FARE run. Pipeline:
fare_stage0_export.py (cells) -> fare_stage_a.py (225 tree fits, FARE venv,
one process per config) -> fare_stage_b.py (attacker battery, project venv).
Backing data: fare_gauntlet.json, fare_cells/*_fare_manifest.json. FARE code:
eth-sri/fare @ 89cb1b6, patched fair-Gini sklearn v1.2.dev0 @ fd60379f.

## Reproduction gate — HELD (claim 1)

ACSIncome-CA-2014, k=50 ni=100 alpha=0.9 s=0.3, their entry point + eval:
dp_ub = 0.1571 vs shipped 0.15712 (bit-exact to 4dp); eval acc over 5 reps
0.796/0.792/0.798/0.797/0.797, mean 0.796 vs shipped mean 0.7959. Both within
the 0.01 gate. Nothing is borrowed — all cell numbers below are our runs.

## The Table-2 row (baseline conventions: best certified operating point per
## tier, 3 training seeds, probe seeds [0,1,2], bar 0.55 on 3-seed mean,
## utility = lift_best / cell clean_lift from baseline_gauntlet.json)

| cell | tier | operating point | utility kept | rep T1 / T2 | out T1 / T2 | verdict | FARE dp_ub |
|---|---|---|---|---|---|---|---|
| easy hmda/race/loan_decision | T1 | k=100 a=0.75 | **0.0%** (lift +0.0000) | 0.539 / 0.539 | 0.539 / 0.539 | PASS | 0.000 |
| easy | T2 | same point | 0.0% | 0.539 / 0.539 | 0.539 / 0.539 | PASS | 0.000 |
| middle hmda/race/loan_amount_band | T1 | k=100 a=0.975 | **28.3%** (lift +0.1156) | 0.535 / 0.535 | 0.536 / 0.536 | PASS | unavailable* |
| middle | T2 | same point | 28.3% | 0.535 / 0.535 | 0.536 / 0.536 | PASS | unavailable* |
| hard adult/sex/income | T1 | k=5 a=0.75 | **39.0%** (lift +0.0560) | 0.546 / 0.546 | 0.546 / 0.546 | PASS | 0.185 |
| hard | T2 | same point | 39.0% | 0.546 / 0.546 | 0.546 / 0.546 | PASS | 0.185 |

*their shipped certificate machinery (AlphaBetaAdversary) asserts every leaf
cell appears in every split of every race pair and aborts on sparse minority
races; dp_ub is null at 20/75 easy and 34/75 middle configs, including the
middle certified point. We did not patch their bound; verdicts never use dp_ub.

FARE passes BOTH tiers on BOTH surfaces on all three cells — the only baseline
in the table to do so — but the frontier is expensive: 0% utility on easy
(every config in the 25-point grid is majority-degenerate: 0/25 rows with
lift_best > 0.005; the fair-Gini tree finds no split worth keeping on the
near-zero-lift cell), 28% on middle, 39% on hard. Highest-utility FAILING
sweep rows for scale: middle k=25 a=0.75 keeps 81% at rep T1 = 0.606; hard
k=100 a=0.5 keeps 63% at 0.637.

Accounting update: published-method combinations go 36 -> 42; passes go 1 -> 7
(VFAE easy T1 at incidental noise + FARE 6/6 at 0/28/39% utility). Under the
floor partition, FARE's middle and hard passes sit on floor-infeasible cells
and clear the output surface precisely by paying the predicted utility price
(out readings 0.536 and 0.546, below the cells' coupling floors 0.584/0.603).

## Claim scoring

2. **Frontier by coupling — MIXED.** Middle HELD (28.3% <= 50%). Easy
   FALSIFIED in the interesting direction: predicted a pass at >= 40% utility;
   actual passes exist only at 0% — FARE is degenerate on the entire easy grid,
   so it joins the paper's only-noise-or-degenerate pattern rather than beating
   it. Hard FALSIFIED by 4pp (39.0% vs predicted <= 35%).
3. **Tiers collapse — HELD.** Tier-2 max equals Tier-1 max at every certified
   point (delta 0.000 <= 0.02; the informed QDA never beats XGB on a
   finite-support deterministic z). No config passes one tier and fails the
   other.
4. **Certificate coherence — MIXED, with a finding.** Hard cell: Spearman
   rho(dp_ub, rep T1) = 0.652 (n=25, p=4e-4) >= 0.6, HELD; threshold form
   (dp_ub <= 0.10 -> rep T1 <= 0.60) holds 25/25. Easy: all 25 dp_ub = 0.000
   (rho undefined); threshold form holds. Middle: FALSIFIED — the 14
   certificate-available configs all read dp_ub = 0.000, yet three of them
   (k10 a0.5, k25 a0.5, k25 a0.75) leak race pairs at rep T1 = 0.603-0.610.
   Precise scope note: dp_ub bounds downstream-classifier demographic parity,
   not attribute recovery, so 0.000-vs-0.61 is not a contradiction of FARE's
   theorem — it is the paper's verification-relativity point appearing again
   (a certificate scoped to one property says nothing about the adjacent
   property an attacker actually exploits). The multiclass Bonferroni path
   drives the per-pair balanced-accuracy bound to 0 on heavily imbalanced race
   pairs; on the binary cell, where FARE's certificate is native, coherence
   held exactly as predicted.

## Operational caveats (for the repro package)

- eth-sri/fare has NO license file — run-and-cite; do not redistribute code.
- The patched sklearn build corrupts memory across sequential in-process fits
  (segfault/abort on the 2nd+ fit; clean in a fresh process). Stage A runs one
  process per config; nothing in their code was modified.
- Compiled-in printf debugging in their Cython splitter floods fd 1; routed to
  /dev/null at the OS level in run_one.
