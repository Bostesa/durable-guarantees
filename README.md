# Durable guarantees for attribute removal

This repository is the code and results artifact for a study of attribute-removal
methods — techniques that claim to strip a sensitive attribute out of a learned
representation. We re-audit those claims under attackers stronger than the ones
used to verify them, measure what genuine removal costs in downstream utility,
and evaluate seven published methods under a single shared threat model.

## Installation

Python 3.11 or newer (developed on 3.13).

```bash
pip install -r requirements.txt
```

## Layout

- `experiments/` — experiment and analysis drivers; each script's docstring gives its protocol and verdict criteria.
- `results/` — the JSON each experiment writes; every number in the paper is read from here.
- `predictions/` — the pre-registration record (see below).
- `paper/` — style files, bibliography, figure generators, and the figure PDFs.
- `notes/` — standalone analysis writeups that support the paper without being part of it.
- `utils/` — shared data loaders, the attacker battery, and compatibility shims for external code.

## Reproducing the paper

| Artifact | Command |
|---|---|
| Figure 1 | `python paper/figures/fig1_demolition.py` |
| Figure 2 | `python paper/figures/fig2_footprint.py` |
| Figure 3 | `python paper/figures/fig3_cost_ramp.py` |
| Figure 4 | `python paper/figures/fig4_two_tier.py` |
| Audit scatter | `python paper/figures/fig_audit_scatter.py` |
| Table 1 | `python experiments/master_gauntlet_table.py` |
| Table 2 | `python -c "import json;print(json.dumps(json.load(open('results/mi_ceiling.json'))['operating_point_extension'],indent=2))"` |

## Retraining

Full reproduction retrains every channel and re-runs every attacker suite from
scratch on CPU or Apple MPS; budget hours per experiment rather than minutes.
It needs the audited system's checkout on `$PCRL_ROOT`, and the baseline
comparisons additionally need the external method repositories, each pinned to
an exact commit in the script that uses it.

## Pre-registration

`predictions/` holds, for each run, the prediction committed before that run
took place and the outcome scored against it afterwards. Predictions that were
falsified are kept and scored as falsified rather than removed.

## License

MIT — see [LICENSE](LICENSE).
