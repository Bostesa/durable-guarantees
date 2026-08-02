# Registered prediction — FNF (Fair Normalizing Flows) under the two-tier gauntlet

Registered 2026-07-24 at repo HEAD eb989a1, BEFORE any FNF run on our cells and
before any implementation. Adds the closest *provable-guarantee* competitor to
the Table-2 baseline set: FNF certifies an upper bound on the accuracy of ANY
adversary, which is a strictly stronger claim than FARE's downstream-DP bound.

## Pinned provenance

- Code: official eth-sri/fnf @ f57401262dc12b258353b26b308ec02f618433d6.
  LICENSE present and verified: **Apache License 2.0** (unlike FARE, which
  ships no license) — freely usable, so this run may reuse their training
  code directly rather than reimplementing.
- Paper: Balunović, Ruoss, Vechev, "Fair Normalizing Flows", ICLR 2022
  (arXiv 2106.05937v2).
- Reference numbers for the gate, from their Table 1 (Adult), FNF row:
  **Acc 84.43, Adv Acc (g ∉ G) 59.56, Max Adv Acc (their certified bound)
  61.12**. The bound is (1 + Δ)/2 on the adversary's BALANCED accuracy, so
  61.12 implies their reported Δ ≈ 0.2224 at that operating point.
- Our cell denominators: clean e2e lift from two_tier_certification.json
  (sha256 e380d61f) — hard adult/sex/income = 0.14349.

## Environment shim (no algorithmic change; laftr_shim.py precedent)

Their tree is NOT edited. Two environment-level aliases are required to run
2022-era code on this machine, both semantically identity operations:
`np.object`/`np.bool` → builtin `object`/`bool` (removed in numpy ≥ 1.24; these
were exact aliases), and the module-level `device = 'cuda'` globals in
train_fnf.py rebound to cpu/mps (no CUDA on this Mac). Any further deviation
is a gate failure, not a shim.

## a. LICENSE + GATE FIRST

Gate = reproduce their Table-1 Adult FNF row using their own scripts
(`train_gen_categorical.py --dataset adult` then their
`train_enc_categorical.py` at the shipped `scripts/run_adult_gamma.py`
settings), at their shipped seeds, sweeping their shipped gamma grid
[0, 0.02, 0.1, 0.2, 0.9]. PASS iff some gamma in that grid reproduces
accuracy within 0.01 of 0.8443 AND adversarial accuracy within 0.01 of 0.5956.
The paper does not state which gamma produced the Table-1 row, so the gate
searches their own grid and reports the matching gamma; failure to match at
ANY gamma = gate failure, stop and report, borrow nothing.

## b. Applicability check — reported, not hacked around

FNF trains two group-conditional bijective encoders f_0, f_1 and requires a
BINARY sensitive attribute; it also requires s at inference time to select the
encoder. Therefore:
- easy (hmda/race/loan_decision) and middle (hmda/race/loan_amount_band) have
  5-class race and are **structurally out of scope** for FNF. We do NOT
  binarize race — that would change the cell definition and make the row
  incomparable to every other Table-2 row. Reported as a structural
  constraint, with the s-at-inference requirement noted as a second
  deployment-scope caveat.
- hard (adult/sex/income) has binary sex and IS run.
Our features are continuous preprocessed floats, so we use their CONTINUOUS
path (RealNVP `FlowEncoder` + GMM priors + `train_fnf.train_flow`), the
faithful analogue of their crime/law configuration, called with our cell data.

## c. Sweep and conventions (identical to the FARE run)

Their shipped continuous gamma grid [0, 0.02, 0.1, 0.2, 0.9]; 3 training
seeds; sweep at seed 0, certify candidates at all three; probe seeds [0,1,2];
bar 0.55 on the 3-seed mean; both surfaces (release z and the task output).
Release is deterministic and continuous, so the Tier-2 attacker follows the
FARE/LEACE convention: Σ_noise = 0, i.e. the informed Gaussian (QDA) fit on
clean group-conditional statistics, plus the strongest defense-aware read
available — here also FNF's own optimal adversary μ* from their Lemma 5.1
(compare exact log-densities under both trained flows), which is the
strongest honest attacker their own theory admits.

BOTH metric conventions are recorded on every reading: AUC (our standing bar,
so the row is comparable to the rest of Table 2) and BALANCED ACCURACY (their
bound's metric, so certificate coherence is an apples-to-apples comparison).

## d. Certificate coherence

Per config, log FNF's own reported statistical distance Δ and the implied
bound (1+Δ)/2 next to our measured best-attacker balanced accuracy. Report
every config where measured recovery EXCEEDS the certified bound. Their bound
is stated w.r.t. estimated densities (Theorem 5.5 adds an intractable ε for
the true-density gap); our attackers see the true test split, so any excess is
a statement about density-estimation error on OUR features, not a refutation
of their theorem — this scope distinction must be stated precisely in the
results.

## Registered predictions

**P1 (Tier-1 verdict, uncertain — range named).** FNF passes the Tier-1 bar on
the hard cell at 1–2 of the 5 swept gammas (the high-fairness end, γ ∈ {0.2,
0.9}), and at every gamma where it passes, utility kept is **≤ 40%** of clean
lift; at every gamma retaining > 60% utility it FAILS Tier 1. Stated as a
range because FNF's bound at their own Adult operating point (Δ≈0.22 → 0.611
balanced accuracy) sits ABOVE our bar, so passing requires driving Δ well
below their published operating range, which their own accuracy/Δ curve says
costs accuracy.
  *Failure mode:* the bijective encoder is information-preserving, so
  protection comes ONLY from distribution matching; if the GMM density
  estimate on our 100+-dim PCRL features is poor, the trained Δ is optimistic
  and Tier-1 fails at every gamma — the whole sweep could read ≥ 0.6.

**P2 (Tier-2).** Tier-2 fails at every operating point that retains meaningful
utility (> 40% of clean lift), because the release is a continuous,
full-precision, deterministic, information-preserving latent: the informed
attacker reads the residual density mismatch that the finite-sample flow
training leaves behind. Concretely, Tier-2 max > 0.55 at every gamma with
utility > 40%.
  *Failure mode:* if training drives Δ to near 0 at high gamma, Tier-2 could
  pass there — but P2 would still hold as stated, since those points are
  predicted to have ≤ 40% utility. P2 is falsified only by a point with BOTH
  > 40% utility and Tier-2 ≤ 0.55.

**P3 (certificate scope — the interesting one).** At least one config shows
measured adversarial recovery (balanced accuracy, their metric) ABOVE FNF's
own density-conditional bound (1+Δ)/2. Unlike FARE — whose certificate was
coherent within its own scope — FNF's bound is stated over the ESTIMATED
densities, and our attackers evaluate on the true test split, so the gap
Theorem 5.5 leaves open should be measurable.
  *Failure mode:* if the GMM density estimate happens to be good on our
  features, every reading sits under the bound and P3 is FALSIFIED — which
  would be a genuinely positive result for FNF and must be reported as such,
  strengthening rather than weakening their claim.

**P4 (failure modes).** Named inline per prediction above. Global failure
mode for the whole run: the continuous path on PCRL-preprocessed features may
be so far from their tabular setting that the flows fail to train at all
(Δ ≈ 1, accuracy at majority) — in which case the honest reading is "FNF is
not applicable to this input distribution", NOT "FNF fails", and the row is
reported as inapplicable rather than as a defeated baseline.

## Success/adoption note

This run changes no existing paper claim by itself. If adopted, it adds at
most one Table-2 row (hard cell only) plus two scope sentences (binary-s
restriction; s-required-at-inference). Accounting delta is reported, not
applied. Any paper use requires author sign-off.
