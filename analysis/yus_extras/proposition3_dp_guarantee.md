# Proposition 3 — a per-release DP guarantee for the clipped full-rank channel

Drafted 2026-07-24 for the camera-ready/rebuttal. Pure mathematics + accounting
on the mechanism already measured in Table 5 (`dp_fullrank.json`, registered
bet e87332b); no new experiments. All numeric values below verified by
computation (session log); the composition algebra is the same law the
averaging attack confirmed empirically to 0.001.

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
$\ell_2$ ball of radius $C$. Then for every pair of inputs $x, x'$ and every
measurable set $S$,
\[
\Pr[M(x) \in S] \;\le\; e^{\varepsilon}\, \Pr[M(x') \in S] \;+\; \delta
\]
holds for exactly the $(\varepsilon,\delta)$ pairs of the Gaussian mechanism
with sensitivity $2C$: the release is $\mu$-GDP with $\mu = 2C/\sigma$
(Dong, Roth, and Su 2022), hence $(\varepsilon,\delta)$-indistinguishable for
every $\varepsilon \ge 0$ with
$\delta(\varepsilon) = \Phi(-\varepsilon/\mu + \mu/2)
 - e^{\varepsilon}\,\Phi(-\varepsilon/\mu - \mu/2)$
(Balle and Wang 2018), and in particular
$(\varepsilon,\delta)$-indistinguishable for
$\varepsilon = \tfrac{2C}{\sigma}\sqrt{2\ln(1.25/\delta)}$
(Dwork and Roth 2014). Any function of the release, including the deployed
head's prediction $\hat y = f(M(x))$, inherits the guarantee by
post-processing, and $k$ adaptive releases of the same input compose to
$\sqrt{k}\,\mu$-GDP.
\end{proposition}

\begin{proof}
Both clipped points lie in the $C$-ball, so
$\lVert \Pi_C(\phi(x)) - \Pi_C(\phi(x')) \rVert_2 \le 2C$ for every pair
$(x, x')$; the map $x \mapsto \Pi_C(\phi(x))$ therefore has global $\ell_2$
sensitivity $2C$, and $M$ is the Gaussian mechanism at noise scale $\sigma$
applied to it. The $\mu$-GDP characterization with $\mu = 2C/\sigma$, the
exact $(\varepsilon,\delta)$ conversion, the classical corollary, the
post-processing property, and $\sqrt{k}$ composition are the standard
statements for this mechanism (Dong, Roth, and Su 2022; Balle and Wang 2018;
Dwork and Roth 2014).
\end{proof}
```

## Scope remarks (LaTeX, immediately after the proposition)

```latex
Three remarks fix the scope. \emph{First}, the guarantee is per-release input
indistinguishability at serving time, conditional on the trained release map:
it bounds what any attacker, present or future, learns about an input from
its released representation or the model's prediction, but it says nothing
about leakage of the \emph{training} set through the choice of $\phi$, which
would require private training. \emph{Second}, it protects participation, not
correlated traits: population-level inference of the attribute from the task
output is exactly the floor of Fig.~3, which no release mechanism crosses.
\emph{Third}, no analogous statement exists for the subspace-confined
channel: its complement $h_\perp$ is released with zero noise, so two inputs
differing in $h_\perp$ are perfectly distinguishable and $\varepsilon$ is
infinite at every $\sigma$; assumption-based routes to certifying it reduce
to second-moment certificates, which Proposition~2 shows are vacuous. The
subspace channel is measured resistance by construction; the full-rank
channel is the guarantee-bearing arm, and Table~5 is its price.
```

## Numeric addendum (verified; optional table footnote for Table 5)

The Table-5 grid sets C = ε_nom·σ/K with K = 2√(2 ln(1.25/δ)) = 9.6896 at
δ = 10⁻⁵, so every labeled point is certified by Proposition 3 **with room to
spare** — the classical calibration is conservative and the exact analytic
values are lower:

| ε_nominal | μ = 2C/σ | ε_exact (δ=10⁻⁵) | slack |
|---|---|---|---|
| 0.5 | 0.103 | 0.35 | 29% |
| 1 | 0.206 | 0.75 | 25% |
| 3 | 0.619 | 2.53 | 16% |
| 6 | 1.238 | 5.62 | 6% |

Composition ledger (exact, √k·μ-GDP — the same algebra the averaging attack
measured): a nominal-ε=1 release queried 16 times is (3.51, 10⁻⁵)-DP; a
nominal-ε=3 release queried 16 times is (13.05, 10⁻⁵)-DP. Suggested footnote:
"Each labeled point satisfies Proposition 3; nominal ε is the conservative
classical calibration, and k queries of one input compose exactly to
√k · (2C/σ)-GDP — the ledger whose attacker side is the averaging curve of
Appendix B."

## Wiring into the paper

- Appendix: proposition + proof + remarks after the Prop. 2 proof (or at the
  head of the "formally private variant" section, renaming it "A
  guarantee-bearing variant (Proposition 3)").
- Main text, one sentence where Table 5 is introduced: "The clipped full-rank
  variant is not merely measured: it carries a per-release
  $(\varepsilon,\delta)$ guarantee (Proposition 3), so Table~5 prices the
  paper's only guarantee-bearing mechanism against its
  measured-resistance one."
- Future work: replace "A formally private variant exists…" with a pointer to
  Prop. 3, and adopt the sentence: "Differential privacy protects
  participation, not correlated traits: it composes past the averaging attack
  but cannot cross the floor."
- Contributions bullet 4 (optional, if space allows): append "; a clipped
  variant carries a per-release guarantee (Prop. 3) at the utility cost of
  Table 5."

## New references required

- Balle, B.; and Wang, Y.-X. 2018. Improving the Gaussian Mechanism for
  Differential Privacy: Analytical Calibration and Optimal Denoising. In
  ICML 2018.
- Dong, J.; Roth, A.; and Su, W. J. 2022. Gaussian Differential Privacy.
  Journal of the Royal Statistical Society: Series B, 84(1): 3–37.
  (Dwork and Roth 2014 is already in the bibliography.)

## What this changes about the paper's claims (and what it must not)

Gains: the paper now contains a true guarantee-bearing mechanism with exact
accounting and an honest price tag; the averaging attack upgrades from "an
attack our mechanism suffers" to "the attacker's side of a ledger the DP arm
prices exactly"; the reviewer question "why not just use DP?" is answered
with a proposition, a table, and a composition curve.

Must not change: the abstract and contributions still describe
isolate-then-noise as measured resistance — Proposition 3 certifies the
full-rank arm only. Do not let any sentence drift toward "our defense is
differentially private"; the defense the paper recommends at Tier 1 remains
uncertified by construction (third remark).
