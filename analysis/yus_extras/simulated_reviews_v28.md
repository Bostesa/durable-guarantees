# Simulated AAAI-27 reviews — v28 ("Outputs Leak What They Use", draft 28)

Generated 2026-07-23 by three independent reviewer agents, each reading only the
PDF, with distinct personas. Full texts below, verbatim. Meta-synthesis at the
end was written by the coordinating session.

Scores: R1 (erasure insider) 6/10 conf 5 — borderline accept.
R2 (security/privacy) 7/10 conf 4 — accept w/ conditions.
R3 (generalist/statistics) 7/10 conf 4 — accept w/ conditions.

---

## Reviewer 1 — fair-representation / concept-erasure insider (6, conf 5)

[Full text as returned by the agent]

SUMMARY: The paper audits the standard practice of certifying attribute removal
from learned representations via a linear probe (R²_lin ≤ τ = 0.05). The authors
collect 67 removal configurations that pass this check across six tabular
datasets (HMDA, Adult, Diabetes, Folktables/ACS, LSAC, Dutch Census) plus two
CelebA cells, and re-attack them with a two-tier suite: Tier 1 black-box
attackers (XGBoost, a 256–256 MLP, a rank-32 LoRA probe) and Tier 2 an informed
Gaussian likelihood-ratio attacker that knows the defense. 59 of 67 verdicts
fail; only eight survive, all via high additive noise (σ ≥ 4). Two propositions
formalize why: linear certificates are invariant under any invertible affine map
(Prop. 1), and any certificate that depends only on class means/covariances can
be matched by a "two-sheet" distribution on which the Bayes-optimal attacker
nearly achieves noise-free recovery (Prop. 2). The paper then argues that even
perfect representation cleaning cannot help when the task label itself couples
to the attribute: an "output-leak floor" measured by attacking the model's
predictions, which is predicted before training by a label-coupling XGBoost fit
on labels alone (r = 0.799, ρ = 0.828 over 27 natural cells; r = 0.872
out-of-sample on external datasets; causally confirmed on 9 constructed cells).
Finally, an isolate-then-noise procedure (train a QR-orthonormalized subspace M
jointly with the encoder, HSIC-penalize the complement, add Gaussian noise only
in the subspace) passes the Tier-1 suite at 102%/100%/85±2% retained utility on
the three headline cells, while Tier 2 requires full-rank noise and renders the
hard Adult/sex/income cell infeasible (−0.5% to −7% utility). A re-evaluation of
five published methods (LAFTR, VFAE, DANN-scrub, LEACE, Obliviator) under the
same protocol finds 1 of 36 configurations passing — VFAE, and only via its
incidental sampling noise.

STRENGTHS:
- The two-tier attacker suite (Table 1) is well constructed, and the informed
  Gaussian LRT attacker in Tier 2 is the right escalation — most "stronger
  probe" audits stop at nonlinear black-box probes. The finding that
  subspace-confined noise holds the black-box bar everywhere but the LRT reads
  the untouched complement at 0.66–0.94 is a genuinely useful negative result
  that separates the two threat models.
- The output-side floor is the paper's most original quantitative contribution.
  Figure 3 and Figure 4 tell a coherent, decision-relevant story. Crucially, the
  predictor is validated out-of-sample on three external datasets selected
  before results existed (r = 0.872, ρ = 0.929) and causally via 9 constructed
  cells — much better epistemic hygiene than the typical correlational plot.
- The paper is unusually honest about its own method's limits: the averaging
  attack (Appendix B) shows the easy cell's surviving point breaches at N = 16
  fresh queries and the hard cell at N = 8, following the predicted σ/√N law;
  Figure 5 openly reports the hard cell as infeasible under Tier 2 (−7%).
  Prop. 2 is even turned against the authors' own certificates. This is how
  adversarial evaluation papers should be written.
- The empirical instrument-invariance check in Appendix A (certificate pinned
  within [0.03493, 0.03494] across 31 checkpoints, jumping to 0.104 after a
  single ReLU) is a nice concretization of Prop. 1, and the computed
  instantiation of Prop. 2 at the actual operating points (Table 3) closes the
  theory–experiment loop.

WEAKNESSES:
1. (Baseline fairness) The headline framing is harsher on prior methods than
   the paper's own scoping discussion warrants. LEACE explicitly certifies only
   against linear adversaries — counting such configurations among "59 of 67
   fail" and Table 2's failures conflates methods that overclaim with methods
   audited outside their stated contract. DANN-scrub is run from "generic" code;
   for LAFTR it is unclear whether the training-time adversary's capacity was
   swept in tandem with the attacker suite. Per-method sweep details are
   deferred to an appendix not in the submitted pages.
2. (Novelty positioning) The core empirical claim — nonlinear/informed
   attackers recover what linear probes certify away — is well established
   (Ravfogel 2022; probing literature; Song & Shmatikov 2020). The delta over
   Ravfogel 2022 is quantitative breadth, not conceptual surprise. The
   output-side floor is anticipated by Stadler et al. 2024; the new part is the
   pre-training label-coupling measurement.
3. (Prop. 1 is a dressed-up triviality) Affine invariance of R² is a three-line
   covariance cancellation; elevating it to a headline contribution oversells
   it. Prop. 2's core is a standard two-point indistinguishability construction.
4. (Method novelty) Isolate-then-noise combines known ingredients, and the
   Tier-2 result undercuts the novel component: subspace confinement protects
   only against Tier 1; surviving Tier-2 configurations use full-rank noise —
   the part that is not new. The DP comparison is dismissed in two sentences of
   Future Work; ε ≈ 5–6 is not "weak" by deployed-DP standards and deserves a
   real experiment.
5. (Scale) All evidence is tabular (two-layer MLPs, minutes on a laptop) plus a
   frozen-encoder vision check — "exactly the regime where the end-to-end
   advantage disappears." The intro motivates with LLM safeguard failures, but
   nothing at that scale is tested; the 27-cell evidence is modest and not fully
   independent.
6. (Utility accounting) Conventions differ between arms; the honest uniform
   held-out comparison (Table 4) lives in the appendix while mixed-convention
   numbers headline the abstract; "102%" retained is above clean baseline.
7. (Thresholds) 0.55 and τ = 0.05 drive all discrete verdicts; no sensitivity
   curve in the main text (only the 0.57 relaxed point).

QUESTIONS: (1) LAFTR/DANN sweep budgets and adversary-capacity scaling; (2)
what exactly are the 67 configurations — own pipelines, published methods, or
both, and does 59/67 audit "straw" configurations; (3) does Prop. 2 apply to
FARE's certificate class, and how does FARE fare under the suite; (4) what
realistic deployment enforces a per-input query cap, and report a DP baseline
at matched utility on all three cells; (5) is the predictor's upper-bound
property provable or purely empirical, and how much of the Adult gap is the
first-grid-level convention; (6) can isolate-then-noise train through even a
small fine-tuned vision/text encoder.

WRITING: genuinely coherent story with an unusual and effective structure; each
section sets up the next; figures load-bearing. Weaker: Results prose dense
with inline numbers; the two-arm design easy to lose; utility-convention split
forces back-and-forth. Distinctive declarative voice; does not read
machine-generated; occasionally too aphoristic for its own precision.

SCORE 6 (borderline accept), CONFIDENCE 5.
RECOMMENDATION: Borderline accept — merits publication if the authors temper
the baseline framing, clarify the 67-configuration provenance, and add a
serious DP comparison.

---

## Reviewer 2 — security & privacy (7, conf 4)

SUMMARY: [as above, from the security angle; identifies the two-proposition
spine, the 59/67 audit, the coupling predictor (r = 0.799/ρ = 0.828, r = 0.872
out of sample), isolate-then-noise surviving Tier 1 at 102/100/85±2 and Tier 2
requiring full-rank noise at 55.5/21.7/negative, and the 1-of-36 baseline
result traced to incidental noise.]

STRENGTHS:
- The paper attacks itself: the single-release assumption is priced (averaging
  attacker crosses the bar at N=16/4/8 per cell; per-query AUCs 0.506/0.520/
  0.530/0.560 matching σ/√N to 0.001) and the query cap is named "the informal
  version of the composition budget that differential privacy formalizes."
- Worst-case metrics actually reported: TPR at 1% FPR = 1.4–3.7% at surviving
  points vs near-perfect identification unprotected, middle-cell output surface
  honestly flagged as the exception.
- The propositions are small, checkable, load-bearing; the Appendix A LoRA
  trajectory demo is elegant, with author-stated caveats.
- Exemplary overclaiming discipline: "measured resistance rather than a proven
  guarantee"; "surviving operating points, not guarantees"; "the 0.55 bar and
  the two tiers are choices, defensible but not canonical." Threat models
  tabulated.
- The baseline autopsy is diagnosis, not a leaderboard (VFAE's pass traced to
  sampling noise; β sweep worsens protection 0.536 → 0.743; LEACE reads
  0.995–0.998 "exactly as its linear scope predicts").
- The pre-training cost predictor is practically useful independent of the
  defense.

WEAKNESSES:
1. 59/67 is measured in average-case AUC only; TPR@1%FPR is computed only at
   the paper's own six surviving points, not for the 67 audited configurations
   or the 36 baselines. A defense at AUC 0.53 can still flag a subgroup at high
   precision; the worst-case audit of the field's methods is not done.
2. No sensitivity analysis of verdicts to the 0.55 bar and τ = 0.05; at least
   one result is visibly bar-fragile (85%±2 at 0.55; 90% at 0.57; worst seed
   0.561 breaching at five seeds). One paragraph at bars 0.52/0.60 would settle
   robustness.
3. The DP comparison is two unsupported sentences ("ε ≈ 5–6 per release"); no
   table, no accounting, no utility-vs-ε curve. This comparison decides whether
   the empirical mechanism earns its keep over a provable one; it needs Fig.-5
   rigor, not a drive-by.
4. "Per-method sweeps are in the appendix" — but the submitted appendices
   contain no per-method sweep table; the mapping from 5 method rows to 36
   configurations is not in the document. Ironic for a paper whose thesis is
   that verification hides behind under-specified evaluation.
5. Attacker suite fixed and non-adaptive; no learned attacker trained on
   released noisy samples; no auxiliary-feature attacker (the Jayaraman–Evans
   imputation point the paper cites); fresh-partition evaluation deferred.
6. Scale: two-layer MLPs, minutes on a laptop, frozen vision check where the
   end-to-end advantage disappears; no text experiment despite LLM motivation.
7. The causal claim rests on 9 constructed cells reusing two datasets; supports
   sufficiency, leaned on harder than warranted. Predictor is an upper bound
   with loose slack exactly in the interesting cases; band classification more
   defensible than point predictions.
   [Notes several usual complaints the paper already handles: single-release
   composition, "certificate" language, Prop-2 scoping, utility-convention
   disclosure.]

QUESTIONS: (1) TPR at 0.1%/1% FPR for the 67 configurations (or 8 survivors +
Table 2 baselines) — does 59/67 grow or shrink under a worst-case criterion;
(2) verdict counts at bar 0.52/0.60 and τ 0.01/0.10; (3) details behind
"ε ≈ 5–6": mechanism, clipping norm, accounting, utility-vs-ε on all three
cells, and the ε at which DP matches Tier-1 points; (4) averaging across
correlated records (siblings, repeated visits) — the Dinur–Nissim regime cited
but not tested; (5) a trained non-Gaussian Tier-2 attacker vs the LRT at
σ = 20/24/64; (6) where exactly are the per-method sweeps.

WRITING: unusually well written; threat model as a table; every protection
claim scoped to a tier; narrative arc coherent; nothing machine-generated;
distinct, occasionally blunt voice. Minor: Fig. 5 negative-utility axis; mixed
conventions confuse first-pass Table-2 reading; dangling appendix references.

SCORE 7 (accept), CONFIDENCE 4.
RECOMMENDATION: Accept, contingent on worst-case TPR numbers for audited
configurations, a bar-sensitivity check, and a substantiated DP comparison in
the camera-ready.

---

## Reviewer 3 — generalist ML / statistics (7, conf 4)

SUMMARY: [accurate three-part reconstruction: audit (59/67), floor + predictor
(r = 0.799/ρ = 0.828 on 27 natural cells + 9 constructed), isolate-then-noise
(102/100/85±2 vs full-rank Tier-2 costs), 1-of-36 baselines.]

STRENGTHS:
- Honest, well-ordered threat modeling; crisp passing criterion; clean
  theory–experiment linkage (affine tap unchanged; ReLU lifts to 0.104; control
  trajectory [0.03493, 0.03494], max |ΔR²| = 1.7×10⁻⁵).
- Unusually good seed/error-bar hygiene: three seeds full suite, five at every
  headline point, and the paper reports a result that gets WORSE under more
  seeds (worst seed 0.561 breaching at five) — a credibility signal.
- Negative results surfaced, not buried (hard cell infeasible at Tier 2;
  averaging voids single-release; VFAE survivor diagnosed as accidental).
- Figures self-explanatory (Fig. 3 plain-language axes; Fig. 4 legend n's;
  Fig. 2 two-panel contrast readable at a glance).
- Above-average reproducibility posture (Appendix C/D pins data, architecture,
  optimizer, grid procedure, named artifacts).

WEAKNESSES:
1. Abstract says "The prediction is an upper bound" flatly; Results concede
   three external cells sit 0.03–0.04 ABOVE the diagonal. Qualify the abstract
   or explain the violations more tightly.
2. r = 0.799 (n = 27) reported with no CI; cells are not independent (shared
   datasets/rows/architecture); effective n < 27. Out-of-sample r on n = 7 is
   nearly anecdotal at three significant digits. Report Fisher/bootstrap CIs
   and discuss clustering by dataset.
3. "Supports a causal relationship" from 9 constructed cells reusing Adult and
   HMDA is a thin basis for "causal"; accept "consistent with a causal effect
   of coupling on these two bases."
4. Number bookkeeping across Tables 3/5 and Fig. 5: Table 5 lists σ = 16 for
   all cells (subspace channel) while Table 3/Fig. 5 use σ = 8–64 (full-rank);
   captions do not say so. Reconciling 102/100/85 vs 94.6/56.6/55.5/21.7 vs
   108.8/101.6/87.3 costs a careful reader substantial time. One reconciliation
   table (arm × channel × convention × σ) is needed.
5. Text reports hard-cell Tier points "−0.5% and −7.0%"; Fig. 5 labels the bars
   "−1%" and "−7%". Make them agree.
6. The arithmetic behind "36" is never shown (Table 2 shows 5 × 3 × 2 = 30).
7. Post-hoc arm measured once (single frozen representation, no seed variance)
   while end-to-end gets five seeds — asymmetry favors the +20.9/+20.3/+34.6pp
   advantage; acknowledge or fix with frozen-encoder seeds.
8. Multiple-comparisons hygiene: hundreds of AUC readings against one bar; AUC
   uncertainty and split sizes for smaller datasets unstated; four of five
   surviving external cells read 0.55–0.58 — outcomes sit right at the bar.
9. MMD used without expansion (HSIC is properly defined; MMD is the one lapse).
10. Title/claims generality outruns tabular-scale evidence; scope accordingly.

QUESTIONS: (1) show the arithmetic behind 36; (2) 95% CI on r = 0.799 and
cluster-bootstrap by dataset; (3) confirm which channel Table 5 vs Table 3/
Fig. 5 refer to and which σ produced each headline number; (4) do the three
above-diagonal external cells fall to the line under a finer noise grid;
(5) evaluation-split sizes for LSAC/Dutch and the SE of an attacker AUC at the
bar — could any of the 8 survivors be bar-crossings within noise; (6) 2–3
frozen-encoder seeds for the post-hoc arm.

WRITING: storytelling genuinely good and chapter-like; each section answers
the question the previous one raised; terms defined before use (MMD the one
lapse); Fig. 1 an excellent opening image. "In plain words" appears ≥3 times
and starts to read as a template; the uniformly aphoristic cadence is polished
but occasionally slick — not machine-generated, but a few aphorisms could be
traded for the reconciliation table the results badly need. The one real
clarity failure is structural: the arm × channel × convention matrix forced two
rereads of pages 5–6 and Appendix B.

SCORE 7 (accept), CONFIDENCE 4.
RECOMMENDATION: Accept, conditional on reconciling the utility/σ matrix,
adding CIs, and softening "causal" and "upper bound" phrasing.

---

## Meta-synthesis (coordinating session)

Scores 6 / 7 / 7, confidences 5 / 4 / 4, no detractor. At AAAI this profile is
a likely accept, decided in discussion; R1 is the swing reviewer and R1's asks
are answerable. Convergent asks across all three: (a) substantiate the DP
comparison (data exists: analysis/yus_extras/dp_fullrank.json); (b) bar/τ
sensitivity of the 59/67 and 1/36 counts (recomputable from stored per-config
AUCs, no retraining); (c) worst-case TPR beyond the six surviving points
(extend run_tpr.py); (d) show the 36 = 6 methods (LAFTR reimpl + LAFTR official
+ VFAE + DANN + LEACE + Obliviator) × 3 cells × 2 tiers arithmetic; (e) CI /
cluster-bootstrap on r = 0.799; (f) reconciliation table for arm × channel ×
convention; (g) Fig. 5 "−1%" vs text "−0.5%"; (h) define MMD; (i) soften
"causal" and abstract's "upper bound"; (j) per-method sweep table into the
appendix or an explicit reproducibility-package pointer. Residual from the
claims audit: Appendix B's 60k-subsample sentence still misdescribes the
headline pipeline.
