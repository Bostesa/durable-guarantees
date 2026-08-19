# Outputs Leak What They Use: Auditing Attribute Removal Beyond Its Certificates

Code, data pipelines, and results for the paper.

## What the paper is about

A model that has been debiased usually ships with a claim that some sensitive
attribute, race or sex, is no longer recoverable from its learned
representation. That claim gets verified with whatever instrument the method was
built around, either a linear R² certificate or an adversary of the same
architecture the method trained against. We re-audit those claims with a fixed
attacker battery applied identically to every method, measure what real removal
costs in downstream utility, and run seven published methods through one shared
threat model.

The battery has two tiers:

* **Tier 1**, the outside attacker. XGBoost, a deep MLP, and a rank 32 LoRA
  probe, all reading only the deployed representation.
* **Tier 2**, the informed attacker. Tier 1 plus a Gaussian likelihood ratio
  attacker that knows the noise channel and holds clean representation side
  information. This models the vendor, an auditor with pipeline access, or
  anyone who can average repeated queries of the same row.

A method certifies at a tier when no attacker in that tier recovers the
attribute above 0.55 AUC, from the representation and from the task outputs.

## What we found

**Passing a linear certificate says almost nothing.** Of 21 results the R²
certificate marked as stopped, 3 survive measurement by XGBoost and an MLP and
18 collapse. On adult/sex the dominant axis certificate reads 0.036, a pass at
τ = 0.05, on projections from which XGBoost still recovers sex at 0.68 AUC. The
unprotected level is 0.683. Every survivor is plain isotropic noise. No eraser
built on a projection survives.

**The output leaks whatever the task uses.** Hide the attribute in the
representation and the decision still betrays it, in proportion to how much the
label correlates with the attribute. Across 27 natural cells that correlation
predicts the utility cost of driving recovery down to 0.55, at Pearson r = 0.80
and Spearman ρ = 0.83. Removal is nearly free where the label barely uses the
attribute and costs nearly all utility where the label needs it.

**Under one shared threat model, published methods do not hold.** The gauntlet
table covers six methods across three cells at two tiers, and 1 of those 36
combinations certifies: VFAE at β = 1 with sampled z, Tier 1 only, on
hmda/race/loan_decision, keeping 72% of utility. LAFTR run from the authors' own
TensorFlow release, LEACE, and Obliviator all keep most of their utility and
leave the attribute recoverable at 0.84 AUC or higher on every cell. FARE, FNF,
and Fair PCA went through the same battery in separate runs, recorded under
`docs/`.

**What does certify is a noised channel trained end to end, and the footprint
law sets its price.** On hmda/race/loan_decision it keeps 95% of the clean lift
at Tier 1. On adult/sex/income, where income genuinely depends on sex, the same
certification costs the whole lift. Utility retained there is -1% at Tier 1 and
-7% at Tier 2, meaning the certified model has no lift left to keep.

## Layout

```
experiments/   experiment and analysis drivers. Each script's docstring states
               its protocol and its verdict criteria.
results/       the JSON each experiment writes. Every number in the paper is
               read from here.
figures/       figure generators, their shared style, and the built PDFs
docs/          the preregistration record. See docs/README.md.
utils/         data loaders, the attacker battery, and shims for external code
```

## Installation

Python 3.11 or newer, developed on 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the paper

Each of these reads the committed JSON, runs no experiment, and finishes in
seconds.

| Artifact | Command |
|---|---|
| Figure 1, the demolition | `python figures/fig1_demolition.py` |
| Figure 2, footprint diagonal | `python figures/fig2_footprint.py` |
| Figure 3, cost ramp | `python figures/fig3_cost_ramp.py` |
| Figure 4, two tier certification | `python figures/fig4_two_tier.py` |
| Audit scatter | `python figures/fig_audit_scatter.py` |
| Table 1, the gauntlet | `python experiments/master_gauntlet_table.py` |
| Table 2, operating points | `python -c "import json;print(json.dumps(json.load(open('results/mi_ceiling.json'))['operating_point_extension'],indent=2))"` |

`figures/figure_data_provenance.md` maps each panel to the exact JSON keys
behind it, including the places where a figure departs from what the paper text
describes.

## Where the headline numbers live

`results/` holds about 70 JSON files, one per run, and most of them are
supporting evidence. These five carry the claims above.

| File | Claim |
|---|---|
| `master_gauntlet_table.txt` | Published methods against ours, per cell, per tier. Read this one first. |
| `honest_reaudit.json` | 3 of the 21 results that passed the certificate survive honest probes |
| `continuous_cost.json` | Label coupling predicts removal cost. See the `stats` key. |
| `two_tier_certification.json` | Our operating points at both tiers, over 5 seeds |
| `mi_ceiling.json` | Distribution-free worst case ceilings at those operating points |

## Retraining from scratch

Full reproduction retrains every channel and re-runs every attacker suite on CPU
or Apple MPS. Budget hours per experiment. It needs the audited system's checkout
on `$PCRL_ROOT`, and the baseline comparisons additionally need each external
method's repository, pinned to an exact commit inside the script that uses it.
`utils/laftr_shim.py` and `utils/fnf_shim.py` document the environment aliases
that let those trees run unmodified. Input datasets cache under `data_cache/`
and intermediates under `analysis/`. Both rebuild on demand and neither is
tracked.

## Preregistration

For most runs `docs/` holds a prediction committed before the run existed,
alongside the outcome scored against it afterward. Falsified predictions stay in
the record and are marked falsified. `docs/README.md` indexes them.

## License

MIT. See [LICENSE](LICENSE).
