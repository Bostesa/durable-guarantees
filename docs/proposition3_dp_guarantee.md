# Proposition 3 — a per-release DP guarantee for the clipped full-rank channel

Drafted 2026-07-24; REVISED same day after the eight-point verification audit
(session log; artifacts `prop3_exact_eps_check.json`, `prop3_reaccounting.json`).
The four audit flags are fixed in this revision:
(1) adjacency stated explicitly (replace-one);
(2) certified values are EXACT analytic epsilons — the round labels 0.5/1/3/6
are demoted to design knobs, since the classical calibration is only valid for
ε ≤ 1;
(3) δ moved to 10⁻⁶, comfortably below 1/n on every cell (15.7× on HMDA,
41× on Adult), at zero utility cost — same C, same runs, re-accounted;
(4) the non-private encoder is stated outright.
No mechanism, run, or utility number changed: `dp_fullrank.json` rows are
byte-identical; only the accounting statement about them is sharpened.

## The proposition (LaTeX, drop into the appendix after the Prop. 2 proof)

```latex
\begin{proposition}[The clipped full-rank channel carries a per-release
guarantee]
\label{prop:dp}
Fix any encoder $\phi$ and clip radius $C>0$, and let the released
representation of an input $x$ be
\[
M(x) \;=\; \Pi_C\!\big(\phi(x)\big) \;+\; \sigma Z,
\qquad Z \sim \mathcal N(0, I_d),
\]
with a fresh draw $Z$ per release, where $\Pi_C$ is projection onto the
$\ell_2$ ball of radius $C$. Under replace-one adjacency --- indistinguishability
of any two inputs $x, x'$ --- the release is $\mu$-GDP with
$\mu = 2C/\sigma$ (Dong, Roth, and Su 2022), hence
$(\varepsilon,\delta)$-indistinguishable for every $\varepsilon \ge 0$ with
\[
\delta(\varepsilon) \;=\; \Phi\!\big(-\tfrac{\varepsilon}{\mu} + \tfrac{\mu}{2}\big)
 \;-\; e^{\varepsilon}\,\Phi\!\big(-\tfrac{\varepsilon}{\mu} - \tfrac{\mu}{2}\big)
\]
(Balle and Wang 2018); for $\varepsilon \le 1$ this recovers the classical
calibration $\varepsilon = \tfrac{2C}{\sigma}\sqrt{2\ln(1.25/\delta)}$
(Dwork and Roth 2014). Any function of the release, including the deployed
head's prediction $\hat y = f(M(x))$, inherits the guarantee by
post-processing, and $k$ adaptive releases of the same input compose to
$\sqrt{k}\,\mu$-GDP.
\end{proposition}

\begin{proof}
Both clipped points lie in the $C$-ball, so
$\lVert \Pi_C(\phi(x)) - \Pi_C(\phi(x')) \rVert_2 \le 2C$ for every pair
$(x, x')$: under replace-one adjacency the map $x \mapsto \Pi_C(\phi(x))$ has
global $\ell_2$ sensitivity $2C$ (the ball's diameter; add--remove adjacency
would give $C$), and $M$ is the Gaussian mechanism at noise scale $\sigma$
applied to it. The $\mu$-GDP characterization with $\mu = 2C/\sigma$, the
exact $(\varepsilon,\delta)$ conversion, the $\varepsilon \le 1$ classical
corollary, the post-processing property, and $\sqrt{k}$ composition are the
standard statements for this mechanism (Dong, Roth, and Su 2022; Balle and
Wang 2018; Dwork and Roth 2014).
\end{proof}
```

## Corollary — certified recovery ceiling (LaTeX, immediately after the proof)

```latex
\begin{corollary}[Certified recovery ceiling]
\label{cor:ceiling}
For the mechanism of Proposition~\ref{prop:dp}, the AUC of \emph{any}
attacker distinguishing the attribute from the released representation, or
from any function of it, is at most $\Phi\!\big(\mu/\sqrt{2}\big)$ with
$\mu = 2C/\sigma$: the class-conditional release distributions are mixtures
of pairwise $\mu$-GDP-indistinguishable distributions, and trade-off
functions are preserved under mixtures (Dong, Roth, and Su 2022), so the
discrimination is at most as easy as separating two unit-variance Gaussians
$\mu$ apart. At our operating points the ceilings are $0.529$, $0.558$,
$0.669$, and $0.809$: the smallest certified point bounds every attacker,
present or future, \emph{below the paper's $0.55$ bar} --- the only
adversary-independent protection statement in this work --- at the utility
cost Table~5 reports for that point.
\end{corollary}
```

## Scope remarks (LaTeX, immediately after the proposition)

```latex
Three remarks fix the scope. \emph{First}, the guarantee is per-release input
indistinguishability at serving time, conditional on the trained release map:
the encoder, its normalization statistics, and the head are trained
\emph{non-privately} by ordinary gradient descent on the full training
partition, so the proposition bounds what any attacker learns about an input
from its released representation or the model's prediction, and says nothing
about leakage of the training set through the choice of $\phi$ --- that would
require private training. \emph{Second}, the ceiling of
Corollary~\ref{cor:ceiling} binds only where utility dies: at the two
certified points below or near the bar ($0.529$, $0.558$) the channel
retains essentially no task utility (Table~5), and at any budget preserving
floor-level utility the certified ceiling ($0.669$ and above) sits
\emph{above} the measured output-leak floor --- so the floor, not the
guarantee, remains the binding constraint, and the non-private trained map
is what carries the population correlations the floor measures.
\emph{Third}, no analogous statement exists for the subspace-confined
channel: its complement $h_\perp$ is released with zero noise, so two inputs
differing in $h_\perp$ are perfectly distinguishable and $\varepsilon$ is
infinite at every $\sigma$; assumption-based routes to certifying it reduce
to second-moment certificates, which Proposition~2 shows are vacuous. The
subspace channel is measured resistance by construction; the full-rank
channel is the guarantee-bearing arm, and Table~5 is its price.
```

## Certified operating points (verified; `prop3_reaccounting.json`)

The clip radii were chosen before the run via the classical δ=10⁻⁵
calibration (registered bet e87332b) — those round targets are DESIGN KNOBS,
not the certified values, because the classical formula is valid only for
ε ≤ 1. The certified statement per point is the exact analytic ε at
**δ = 10⁻⁶**, which is comfortably below 1/n on every cell (HMDA n=63,747:
15.7×; Adult n=24,145: 41×):

| design label | C (σ=8 / σ=32) | μ = 2C/σ | **ε certified (δ=10⁻⁶)** |
|---|---|---|---|
| "0.5" | 0.41 / 1.65 | 0.103 | **0.41** |
| "1" | 0.83 / 3.30 | 0.206 | **0.86** |
| "3" | 2.48 / 9.91 | 0.619 | **2.85** |
| "6" | 4.95 / 19.82 | 1.238 | **6.25** |

Utility numbers are unchanged — same C, same runs (`dp_fullrank.json`), the
re-accounting is computation, not measurement.

Composition ledger at δ=10⁻⁶ (exact, √k·μ-GDP): the ε=0.86 point queried
4 times is (1.83, 10⁻⁶)-DP and queried 16 times is (3.93, 10⁻⁶)-DP; the
ε=2.85 point queried 16 times is (14.3, 10⁻⁶)-DP. The same √k algebra is
what the averaging attack measured empirically — on the subspace channel
(Appendix B) — so the averaging curve is the attacker's side of exactly this
ledger, priced there without a guarantee and bounded here with one.

## Wiring into the paper

- Appendix: proposition + proof + remarks after the Prop. 2 proof (or at the
  head of the "formally private variant" section, renaming it "A
  guarantee-bearing variant (Proposition 3)").
- Table 5: relabel the ε columns with the certified values —
  "ε = 0.41 / 0.86 / 2.85 / 6.25 (exact, δ = 10⁻⁶)" — and adjust the caption:
  "clip radii set by the conservative classical calibration; each point's
  certified (ε, δ) is the exact analytic value of Proposition 3, with
  δ = 10⁻⁶ < 1/(15n) on every cell." Update any main-text "ε ≈ 5–6" mention
  to "ε ≈ 6" (the middle cell reaches Tier-2 utility between the ε=2.85 and
  ε=6.25 points).
- Main text, one sentence where Table 5 is introduced: "The clipped full-rank
  variant is not merely measured: each point carries an exact per-release
  $(\varepsilon, 10^{-6})$ guarantee (Proposition 3), so Table~5 prices the
  paper's only guarantee-bearing mechanism against its measured-resistance
  one."
- Future work: replace "A formally private variant exists…" with a pointer to
  Prop. 3 and the corollary, e.g.: "A certified variant exists
  (Proposition 3): its recovery ceiling binds every future attacker below our
  bar at one operating point, at the cost of essentially all task utility
  (Table 5); at utility-preserving budgets the ceiling sits above the
  output-leak floor, which no release mechanism crosses. Certified
  composition replaces the informal query cap."
  Do NOT adopt the uncorrected "DP does not bound attribute inference" —
  false for this construction (Corollary 1 is exactly such a bound); the
  correct statement is that the bound is vacuous at utility-preserving
  budgets.

## New references required

- Balle, B.; and Wang, Y.-X. 2018. Improving the Gaussian Mechanism for
  Differential Privacy: Analytical Calibration and Optimal Denoising. In
  ICML 2018.
- Dong, J.; Roth, A.; and Su, W. J. 2022. Gaussian Differential Privacy.
  Journal of the Royal Statistical Society: Series B, 84(1): 3–37.
  (Dwork and Roth 2014 is already in the bibliography.)

## What this changes about the paper's claims (and what it must not)

Gains: a true guarantee-bearing mechanism with exact accounting at a δ that
clears the 1/n convention with room, an honest price tag, and a composition
ledger whose attacker side the paper already measured. The reviewer question
"why not just use DP?" is answered with a proposition, a table, and a curve.

Must not change: the abstract and contributions still describe
isolate-then-noise as measured resistance — Proposition 3 certifies the
full-rank arm only. Do not let any sentence drift toward "our defense is
differentially private"; the defense the paper recommends at Tier 1 remains
uncertified by construction (third remark). And keep the design-knob/
certified-value distinction: quoting "ε = 3" as a guarantee would be the
classical formula outside its validity range — the certified value is 2.85.
