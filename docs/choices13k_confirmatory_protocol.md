# Choices13k confirmatory human–LLM benchmark protocol

This experiment tests whether the frozen P1 prompt reduces DeepSeek-V4-Flash's
item-level error against human choice rates on untouched Choices13k rows. The
protocol is confirmatory: the prompts, model settings, population, split, endpoint,
and decision rule are fixed before any confirmatory API request.

## Frozen inputs

- Official dataset commit:
  `821ae7e88386b508ebb46fae76fac63cb62ec876`.
- Excluded pilot split: all 300 prior calibration and held-out rows, frozen parquet
  hash `f2aa1129a864f93a4b7b49d65bb7257b3fe62a7302c24bc77b8d64cfe9712472`.
- Confirmatory seed: `20260917`.
- Primary population: 1,000 previously unused rows with `Feedback=False`.
- Stratification variables: `has_loss`, `ambiguity`, and tertile of absolute EV gap.
  Human choice rate is never used for sampling.
- Frozen confirmatory parquet hash:
  `a339149e44014ec1a0ca8404070256c2798973fef5aa27f0aead5b08307ab659`.

The deterministic sampler operates on the frozen `all_items.parquet`, excludes every
row in `pilot_items.parquet`, filters to no-feedback rows, constructs the three
preregistered stratification variables, and proportionally samples 1,000 rows. The
runner refuses to proceed if either the exclusion hash or confirmatory split hash
changes.

## Frozen treatments

- Model: `DeepSeek-V4-Flash`.
- P0: `Choose the option you prefer.`
- P1: `Respond as an ordinary adult participant making the choice for themselves.`
- Temperature: `0.7`.
- Thinking mode: disabled.
- Repeats: 10 per item and prompt.
- Presentation: exactly five A-first and five B-first repeats per item and prompt.
- Total planned responses: `1,000 × 2 × 10 = 20,000`.

No prompt selection or parameter tuning occurs in this experiment. The code validates
the exact P0/P1 texts and all frozen settings before preparation or execution.

## Outcomes and decision rule

For each item, responses are aggregated into an LLM probability of choosing original
option B. The primary endpoint is the paired item-level contrast:

```text
ΔMAE = MAE(P1, human B-rate) - MAE(P0, human B-rate)
```

Uncertainty is estimated with 5,000 paired bootstrap replicates, resampling benchmark
items rather than individual API responses. Strong confirmatory evidence requires the
upper bound of the two-sided 95% bootstrap interval for ΔMAE to be below zero.

Secondary metrics are RMSE, mean Bernoulli Jensen–Shannon divergence, Pearson and
Spearman correlations, position bias, B-position effect, degenerate-item rate,
riskier-option choice rate, and higher-EV choice rate. Subgroups are loss/no-loss,
ambiguous/non-ambiguous, and small/medium/large EV gap. Secondary analyses do not
change the primary conclusion.

## Leakage and provenance controls

Prompts receive only displayed alternatives. Human B-rate, human sample size, expected
values, variance, risk labels, EV-gap labels, subgroup labels, and prior pilot results
never enter the prompt. Ambiguous-option probabilities remain hidden. Every live stage
requires a clean committed worktree and records config, source, dataset-manifest, prompt
registry, Git commit, model, request, response, retry, and cache provenance. API keys are
loaded from the environment and are not written to output artifacts.

The primary experiment uses no-feedback rows because independent LLM calls cannot
reproduce participant-level realized feedback trajectories. Public benchmark
contamination remains possible. Even a successful result supports only a small,
replicable reduction in aggregate risky-choice error; it does not establish general
human likeness or financial-market validity.

## Execution

The offline stages must finish and be committed before the live run:

```bash
python scripts/run_choices13k_confirmatory.py --stage plan
python scripts/run_choices13k_confirmatory.py --stage prepare
```

After the protocol commit, run the frozen matrix and evaluate it:

```bash
python scripts/run_choices13k_confirmatory.py --stage run --provider aigc_relay
python scripts/run_choices13k_confirmatory.py --stage evaluate
```

All stages are explicit and resumable. Existing response identities are reused; invalid
JSON receives one repair request and otherwise remains invalid.
