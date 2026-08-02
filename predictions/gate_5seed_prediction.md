# Registered prediction — 5-seed recertification of the two new operating points

Registered 2026-07-29 BEFORE implementation and before any 5-seed point was
computed. Prior art: worstpair_supported_sweep.json (commit 5e47b8b), which
found these two operating points at **3** training seeds.

## Why this run exists

The supported-pair sweep certified two new operating points:

| cell | sigma | rep supported | out supported | margin under 0.55 |
|---|---|---|---|---|
| easy (hmda/race/loan_decision) | 256 | 0.5267 | 0.5459 | 0.0041 |
| middle (hmda/race/loan_amount_band) | 1024 | 0.5275 | 0.5460 | 0.0040 |

Both were scored as MAX over 3 training x 3 probe seeds. The paper certifies at
**5** training seeds (`two_tier_certification.py:76`) and aggregates as
`max over architectures of the mean over seeds`
(`two_tier_certification.py:152-156`), not max over everything. Two seeds are
missing and the aggregation differs, on margins of four thousandths. This run
supplies the missing seeds and records the per-(architecture, seed) matrix so
BOTH conventions are computable from one pass.

## Design

Subspace-confined channel, unchanged: easy rank=32 lam=10.0 sigma=256; middle
rank=16 lam=100.0 sigma=1024. Tier 1 (these channels are Tier-1-only, as in the
sweep). Training seeds [0,1,2,3,4], probe seeds [0,1,2], architectures
XGB / MLP / LoRA-32, both surfaces, macro-OVR + all-pairs + supported-pair.
Per-(arch, train seed, probe seed) AUCs are stored, not collapsed.

Two aggregations reported:
  * **paper** — `max_arch( mean over the 15 (train, probe) values )`
  * **strict** — `max over all 45 values`

## Registered predictions

**G1 — under the PAPER convention, both operating points hold (supported-pair
<= 0.55 on both surfaces).** Mean is weakly below max, so a configuration that
passed under max-over-3 should pass under mean-over-5 unless the two new seeds
are severe outliers. Confidence high.

**G2 — under the STRICT convention at 5 seeds, at least one of the two
operating points fails.** The margins are 0.0041 and 0.0040 on the output
surface; adding two draws to a maximum can only push it up, and per-seed spread
in this codebase has already flipped a winner on the hard cell. Predicted: both
slip; scored as correct if at least one does.

**G3 — the convention gap (strict minus paper) is larger on the OUTPUT surface
than on the representation**, on both cells. The output is a 2-5 dim logit
vector whose probe variance across seeds is larger than the 32/16-dim
representation's.

**G4 — utility at 5 seeds stays within 3 percentage points of the 3-seed
reading** (+49.7% easy, +76.2% middle), since lift is already a mean over seeds
and its published std is small (clean_lift_std 0.00031 on easy).

## Scoring

G1, G2 per cell and surface against the 0.55 bar. G3 by direct comparison of
the two aggregations. G4 against the stored sweep utilities. This run makes no
paper edits and does not by itself license any paper claim; a PASS under the
paper convention is a necessary condition for the operating points to be
reportable, not a sufficient one.

Outputs: gate_shards/*.json (one per cell x train seed), gate_5seed.json.
