# durable-guarantees — reproducibility package

Code and results for the paper's central claim: linear-R² compliance certificates
for attribute removal are training-time artifacts that do not survive a strong
attacker suite, while an attacker-agnostic noise channel — priced by a two-tier
certification protocol — provides removal that does. (Anonymous submission.)

## Tier A — regenerate all paper figures and tables from shipped results

Every figure and table regenerates from the JSONs checked into `results/` —
no training, no external data, no GPU. Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib        # Tier A needs only these two
```

Commands (run from the repo root):

```bash
python paper/figures/fig1_demolition.py       # Figure 1
python paper/figures/fig2_footprint.py        # Figure 2
python paper/figures/fig3_cost_ramp.py        # Figure 3
python paper/figures/fig4_two_tier.py         # Figure 4
python experiments/master_gauntlet_table.py   # Table 1 (writes results/master_gauntlet_table.{txt,json})
```

Table 2's values are read directly from `results/mi_ceiling.json` (see the
provenance table below); no script run is required to inspect them.

## Artifact → command → backing results

| Paper artifact | Regenerate with | Backing JSON(s) |
|---|---|---|
| Figure 1 | `python paper/figures/fig1_demolition.py` | `results/honest_reaudit.json` |
| Figure 2 | `python paper/figures/fig2_footprint.py` | `results/diagnostic.json`, `results/celeba_pipeline.json` |
| Figure 3 | `python paper/figures/fig3_cost_ramp.py` | `results/continuous_cost.json`, `results/cliff_or_ramp_constructed.json` |
| Figure 4 | `python paper/figures/fig4_two_tier.py` | `results/two_tier_certification.json` |
| Table 1 | `python experiments/master_gauntlet_table.py` | `results/baseline_gauntlet.json`, `results/laftr_official.json`, `results/obliviator_gauntlet.json`, `results/two_tier_certification.json`; the "Ours" rows trace to `results/end_to_end_surgical.json`, the VFAE provenance footnote to `results/vfae_calibration.json` |
| Table 2 (constructed-attacker AUC / certificate ceiling at the six operating (cell, σ) points) | values in `results/mi_ceiling.json`, key `operating_point_extension` (four points are verbatim stored channel rows; σ=20 and σ=64 are computed there with the same formulas — see the key's `note`) | `results/mi_ceiling.json` |
| Utility-kept triple 102% / 100% / 85±2% | read from `results/hardening_seeds.json`: `exp10_5seed[*].D` for the two HMDA cells, `hard_cell_sigma16_alternative` for the Adult cell (the σ=16 point at which all 5 seeds hold Tier-1 ≤ 0.55) | `results/hardening_seeds.json` (5-seed re-run of the `end_to_end_surgical.json` winners) |

**Evaluation convention.** All attacker AUCs (XGBoost, MLP, rank-32 LoRA, and
the Tier-2 Gaussian-LRT) are held-out: a 75/25 stratified split per probe seed,
attacker fit on the 75%, AUC reported on the 25% (`experiments/diagnostic.py::_split`,
`utils/battery.py::_split_idx`).

## Tier B — full reproduction from scratch

The experiment scripts under `experiments/` retrain every channel/eraser and
re-run every attacker suite locally (CPU/Apple-MPS; no cloud, no multi-GPU).
Install `pip install -r requirements.txt`, then note these external
prerequisites:

1. **PCRL repo (required by 26 of 28 experiment scripts).** The audited system
   under test. Set the environment variable — there is no default:
   ```bash
   export PCRL_ROOT=/path/to/PCRL   # needs code + checkpoints/v2_adult_s0/ + data/
   ```
   All access goes through `utils/pcrl_io.py`, which fails with a clear error
   if `PCRL_ROOT` is unset or wrong.
2. **VectorInstitute/laftr (only `experiments/laftr_official.py`).** Clone the
   official TF1-era release as a sibling directory (commit pinned in the
   script) and install `tensorflow` (see the optional-extras note in
   `requirements.txt`); `utils/laftr_shim.py` provides the TF1 compatibility
   layer.
3. **Obliviator (only `experiments/obliviator_gauntlet.py`).** Clone the
   official repo (URL + commit pinned in `results/obliviator_gauntlet.json`
   provenance block) and make it importable.
4. **CelebA (only the two CelebA experiments).** Place CelebA under
   `$PCRL_ROOT/data/celeba`, then run `python experiments/celeba_extract.py`
   once to build the frozen-representation caches in `data_cache/`.

Each experiment script is self-documenting (module docstring = protocol,
provenance, and verdict criteria) and writes its JSON + PNG into `results/`.
The scripts corresponding to the paper's experiment numbering are listed in
each docstring header; `results/master_gauntlet_table.txt` is the human-readable
provenance-annotated Table-1 source.
