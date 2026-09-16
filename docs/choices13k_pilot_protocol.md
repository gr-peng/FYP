# Choices13k human–LLM behavioral-gap pilot protocol

This experiment measures whether a frozen prompt reduces the item-level gap between
DeepSeek-V4-Flash choice frequencies and Choices13k human choice rates. It is isolated
from the CDA market and does not change market agents.

## Fixed design

- Source: official `jcpeterson/choices13k` commit
  `821ae7e88386b508ebb46fae76fac63cb62ec876`.
- Unique benchmark unit: CSV row index, which is the key in `c13k_problems.json`.
  The 14,568 rows must not be deduplicated by the repeated internal `Problem` ID.
- Pilot sample: 300 unique rows, stratified by feedback, ambiguity, loss, and tertile
  of absolute expected-value gap. Seed: `20260916`.
- Frozen split: 200 calibration items and 100 held-out items. Existing split files
  are verified and reused; the runner refuses a hash mismatch.
- Sampling: 10 responses/item/prompt at temperature 0.7. Each item/prompt has exactly
  five A-first and five B-first presentations, deterministically shuffled.
- Calibration: P0–P3 run on the 200 calibration items. Select minimum item-level MAE;
  exact ties use the preregistered order P0, P1, P2, P3. Save and hash the selection.
- Held-out: after prompt freeze, run P0 and the selected prompt once. If P0 wins,
  it is run only once and the conclusion is that no candidate beat vanilla.
- Primary unit is the item after aggregating repeats into `p_llm(B)`.

The primary held-out contrast is paired item-level
`MAE_selected - MAE_vanilla`. Secondary outputs are RMSE, mean Bernoulli
Jensen–Shannon divergence, Pearson and Spearman correlation, first-position choice
rate, loss/ambiguity/feedback subgroup metrics, calibration curves, riskier-option
choice rate, higher-EV choice rate, and 2,000 paired item bootstraps.
Jensen–Shannon divergence uses base-2 logarithms. Bootstrap artifacts include MAE
and JS intervals for each prompt and paired selected-minus-vanilla intervals.

## Leakage controls

Prompts receive only the displayed alternatives. They never receive `bRate`, `n`,
EV, variance, riskier-option labels, subgroup labels, or prompt-selection results.
For `Amb=True`, the official expanded JSON contains latent probabilities that were
not shown to participants. The prompt lists the B outcomes but says their fixed
probabilities are undisclosed; exposing those probabilities would give the model
more information than humans had.

The selected prompt records hashes of the calibration split, prompt registry,
config, and calibration item-rate artifact. Held-out execution verifies all four.
Evaluation is a separate stage; the runner does not use held-out metrics to select
or modify a prompt.

## Interpretation limits

Choices13k feedback rows summarize people who made repeated choices and received
realized feedback. This pilot has no participant-level outcome trajectories, so
independent LLM samples cannot reproduce learning from realized feedback. The
feedback subgroup is a diagnostic gap, not a like-for-like feedback-learning test.
The public benchmark may also be present in model pretraining. Position swapping
reduces display bias and exact-format recall but cannot eliminate contamination.

Passing this pilot supports only a claim about aggregate risky binary choices. It
does not validate financial herding, trading frequency, strategy switching, or
news response.

## Execution stages

Nothing runs when the code is installed. The master command requires an explicit
stage:

```bash
python scripts/run_choices13k_pilot.py --stage prepare
python scripts/run_choices13k_pilot.py --stage calibration --provider siliconflow
python scripts/run_choices13k_pilot.py --stage freeze
python scripts/run_choices13k_pilot.py --stage heldout --provider siliconflow
python scripts/run_choices13k_pilot.py --stage evaluate
```

`--stage plan` is offline and prints planned calls and prerequisite status. A live
stage refuses a dirty Git worktree and is resumable from exact request identities
and response caches. Invalid JSON gets one repair attempt and then remains invalid;
it is never converted to A or B. Held-out execution is blocked when calibration
invalid rate exceeds 1%.
