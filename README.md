# Outputs Leak What They Use: Auditing Attribute Removal Beyond Its Certificates

Code, data pipelines, and results for the paper.

## The short version

Say a bank trains a model to approve loans, and wants to promise that the
model's internal view of an applicant carries no information about race.
Several published methods claim to deliver exactly that, and each ships a test
showing the race signal is gone.

We rebuilt those tests with stronger attackers. In most cases the race signal
was still sitting there, visible to an attacker the original test never tried.
We then measured what it costs to remove the signal for real, and found the
price is set by something no method can engineer away: how much the decision
itself depends on race.

## Terms used here

| Term | What it means |
|---|---|
| Representation | The list of numbers a model computes about a person before it decides anything. 64 numbers, in our case. This is what gets shared or reused, so this is what the removal claims are about. |
| Attacker, or probe | A second model that reads the representation and tries to guess the protected attribute from it. If it succeeds, the attribute was never removed. |
| AUC | How well an attacker does. 0.50 means it is guessing. 1.00 means it is always right. We treat an attribute as hidden only at 0.55 or below. |
| Cell | One dataset, one protected attribute, one prediction task. `hmda/race/loan_decision` means HMDA mortgage records, protecting race, while predicting the loan decision. |
| Clean lift | How much accuracy the unprotected model gains over always answering with the most common label. Utility percentages below are shares of that number, so 100% means protection was free and 0% means the model became no better than always guessing the common answer. |
| Certificate | The test a removal method uses to declare itself finished. |

## The problem, in one measurement

On Adult data protecting sex, the certificate reads 0.036 against its own pass
mark of 0.05. It passes. Point XGBoost at the very same representation and it
recovers sex at 0.68 AUC. A completely unprotected model sits at 0.683.

The protection removed almost nothing, and the certificate could not tell,
because it only looks for a straight-line relationship between the
representation and the attribute. XGBoost does not need a straight line.

## How we test instead

One fixed battery of attackers, run identically against every method, in two
tiers.

* **Tier 1 is the outside attacker.** XGBoost, a deep neural net, and a rank 32
  LoRA probe. They see only what gets deployed.
* **Tier 2 is the informed attacker.** Everything in Tier 1, plus an attacker
  that knows how the protection works and has seen the representation before
  noise was added. This is the vendor, an auditor with pipeline access, or
  anyone who can query the same row repeatedly and average the answers.

A method certifies at a tier when nothing in that tier gets above 0.55 AUC,
both from the representation and from the model's own outputs. Checking the
outputs as well as the representation is what most of these methods fail.

## What we found

**A passing certificate is close to worthless.** Of 21 results the certificate
marked as protected, 3 survived measurement by XGBoost and a neural net, and 18
collapsed. All 3 survivors were plain random noise added to the
representation. The more sophisticated methods, which delete the attribute
along a chosen direction, all failed once attacked by something that worked
differently from the probe they were tuned against.

**Hiding the attribute inside the model does not hide it.** Even with the
representation fully scrubbed, the model's decision still gives the attribute
away, in proportion to how much the label depends on it. We tested 27 real
cells, and across them a single cheap measurement predicts the whole cost: how
well the protected attribute can be guessed from the label alone. That number
tracks how much accuracy you have to burn to hide the attribute properly, at a
correlation of 0.80. Where the label barely uses the attribute, removal is
nearly free. Where the label genuinely needs it, removal costs nearly
everything.

**Under one shared threat model, the published methods do not hold.** The main
table covers six methods across three cells at two tiers. Of those 36
combinations, 1 certifies: VFAE, at Tier 1 only, on `hmda/race/loan_decision`,
keeping 72% of its accuracy gain. LAFTR run from the authors' own TensorFlow
release, LEACE, and Obliviator all keep most of their accuracy and leave the
attribute recoverable at 0.84 AUC or higher on every cell they were tried on.
FARE, FNF, and Fair PCA went through the same battery separately, with each
prediction and outcome recorded under `docs/`.

**What does work is expensive, and the paper says exactly how expensive.**
Adding noise to the representation during training, rather than bolting
protection on afterward, certifies at both tiers. On `hmda/race/loan_decision`,
where the loan decision barely depends on race, it keeps 95% of the accuracy
gain. On `adult/sex/income`, where income genuinely depends on sex, the same
certification consumes the entire gain. Utility retained there is -1% at Tier
1 and -7% at Tier 2, so the protected model lands at or just below the accuracy
of always guessing the common answer.

## Layout

```
experiments/   the experiment scripts. Each one's docstring states its protocol
               and what would count as success or failure.
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

Each command reads the committed JSON, runs no experiment, and finishes in
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

`results/` holds about 70 JSON files, one per run, and most are supporting
evidence you will never need to open. These five carry the claims above.

| File | Claim |
|---|---|
| `master_gauntlet_table.txt` | Every method against ours, per cell, per tier. Start here. It is plain text and reads like a table. |
| `honest_reaudit.json` | 3 of the 21 results that passed the certificate survive honest probes |
| `continuous_cost.json` | Label dependence predicts removal cost. See the `stats` key. |
| `two_tier_certification.json` | Our operating points at both tiers, over 5 seeds |
| `mi_ceiling.json` | Worst case limits at those operating points, assuming nothing about the attacker |

## Retraining from scratch

Full reproduction retrains every model and re-runs every attacker suite on CPU
or Apple MPS. Budget hours per experiment. It needs the audited system's
checkout on `$PCRL_ROOT`, and the baseline comparisons additionally need each
external method's repository, pinned to an exact commit inside the script that
uses it. `utils/laftr_shim.py` and `utils/fnf_shim.py` document the environment
aliases that let those older codebases run unmodified. Input datasets cache
under `data_cache/` and intermediates under `analysis/`. Both rebuild on demand
and neither is tracked.

## Preregistration

For most runs, `docs/` holds a prediction that was committed before the run
existed, alongside the outcome scored against it afterward. Predictions that
turned out wrong stay in the record and are marked wrong. `docs/README.md`
indexes them.

## License

MIT. See [LICENSE](LICENSE).
