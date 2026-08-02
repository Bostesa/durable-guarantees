# Registered prediction — Fair PCA under the two-tier gauntlet

Registered 2026-07-24 at repo HEAD eb989a1, BEFORE any run on our cells and
before implementation. The cheap linear data point: a closed-form, published
fair-representation method, minutes of compute, filling the "linear
dimensionality reduction" slot next to LEACE in the Table-2 baseline set.

## Pinned provenance

- Code: official amazon-science/fair-pca @
  18fac69ed9237b567eb7aad94774d501a13a04bf.
  LICENSE verified: **Apache License 2.0** (plus NOTICE and
  THIRD-PARTY-LICENSES). Freely usable; their `src/fair_pca/fair_PCA.py` is
  called directly, unmodified.
- Paper: "Efficient fair PCA for fair representation learning", AISTATS 2023
  (arXiv 2302.13319).
- Cell denominators: clean e2e lift from two_tier_certification.json
  (sha256 e380d61f): easy 0.02228, middle 0.40912, hard 0.14349.

## Structural constraint, recorded before the run

`FairPCA.fit` and `check_inputs` REQUIRE a binary protected attribute
(`np.unique(prot_attribute) == [0, 1]`, else `ValueError`). This is the same
structural restriction FNF has. Therefore, exactly as in the FNF run, the
5-class-race cells (easy, middle) are **out of scope and are not binarized** —
binarizing would change the cell definition and break comparability with every
other Table-2 row. Only the hard cell (adult/sex/income, binary sex) is run
against their code. If their code accepts a multiclass attribute in some path
we have not found, that is checked first and the scope is widened accordingly.

## a. Gate

Their pure-Python shipped example is `illustration_Figure1/illustration_Figure1.py`
(the Matlab/R paths for Tables 1–4 are not reproducible on this machine and are
not used). That script generates its data with an UNSEEDED
`np.random.normal`, so it publishes no fixed number to match within 0.01 —
stated plainly rather than papered over. The gate is therefore on the
method's DEFINING INVARIANT, which is exact and seed-independent:

1. Run their shipped script unmodified; it must complete and print its two
   diagnostics (reconstruction error; deviation in group-conditional
   covariances).
2. Seeded invariant check via their public API: after `FairPCA.fit`, the
   projected group-conditional MEANS must coincide — this is precisely the
   constraint their construction enforces (they project onto
   `null_space(Z^T X)` for centered `Z`) — to ≤ 1e-6 in max-norm relative to
   the projected data scale.
3. The Section-3.6 variant (`FairPCAEqualizeCovariance`) must show a strictly
   smaller group-conditional covariance deviation than the Section-2 variant
   on the same data, i.e. their claimed improvement reproduces in direction.

Gate failure on any of the three = stop and report; borrow nothing.

## b. Sweep and conventions (identical to the FARE / FNF runs)

Two exposure surfaces, both cheap and both reported, so "you applied it wrong"
is not available as an objection:
- **raw-X** (their native use: a preprocessing method applied to the input
  features) — the from-raw-x input every other trained baseline consumes;
- **post-hoc on the clean trained representation P** (the LEACE convention
  from baseline_gauntlet.py), utility and logits from a retrained LR head.

Knobs: `target_dim` over a grid spanning their recommended range up to
d − 1, and `tradeoff_param` ∈ {0, 0.25, 0.5} (their fairness/accuracy knob;
note `transform` appends `tradeoff_param³ ×` the standard-PCA representation,
so larger values are strictly LESS protective by construction).
3 training seeds; probe seeds [0,1,2]; bar 0.55 on the 3-seed mean; both
surfaces (representation and output). The representation is deterministic, so
Tier-2 follows the LEACE convention: Σ_noise = 0, the informed Gaussian (QDA)
on clean group-conditional statistics.

## Registered prediction

**P1 (the main claim).** Fair PCA FAILS Tier 1 on every cell it can be run on,
at or near the unprotected level (**Tier-1 max ≥ 0.90**), exactly like LEACE
(0.995–0.998), because its constraint removes a SINGLE linear direction — the
one along which the centered protected attribute correlates with X — from a
representation that encodes the attribute nonlinearly across many directions.
Removing one linear direction cannot stop XGBoost, a 256-256 MLP, or a rank-32
LoRA. Corollary: Tier 2 also fails everywhere (a superset of Tier 1), and
utility stays high, so this lands in the paper's "≈ unprotected at ≈ full
utility" quadrant with LEACE.

**P1b.** Protection does not improve monotonically as `target_dim` shrinks:
even at the smallest target dimension the surviving components remain
nonlinearly attribute-predictive, so Tier-1 max stays ≥ 0.80 at every swept
dimension. (Distinct from P1 so a partial effect is scoreable.)

**If it passes anywhere, that is the surprise worth knowing before a reviewer
finds it** — and it would be reported as a genuine pass, with the operating
point and utility stated, not explained away.

*Failure mode for P1:* if the PCRL-preprocessed adult features happen to
encode sex mostly along one linear direction, fair PCA's single constraint
could cut recovery substantially — P1 would be FALSIFIED and the honest
reading is that the cell's attribute encoding is more linear than assumed,
which would also weaken our own framing of why linear certificates fail.
*Failure mode for the run:* `target_dim < d` is enforced, and the projected
representation has `target_dim + target_dim` columns (their `transform`
hstacks the trade-off block), which must not be mistaken for extra protection.

## Adoption note

Adds at most one Table-2 family (hard cell, two surfaces) plus a shared scope
sentence with FNF on the binary-s restriction. Accounting delta reported, not
applied; any paper use requires author sign-off.
