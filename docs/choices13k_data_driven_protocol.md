# Choices13k data-driven calibration protocol

This experiment asks whether examples of actual human decisions reduce the aggregate
Human–LLM choice-distribution gap more reliably than vanilla prompting or the frozen
persona baseline. It is a new experiment and does not alter the completed pilot,
confirmatory experiment, or CDA market.

## Feasible untouched population

After excluding all 300 pilot rows and all 1,000 confirmatory rows, Choices13k contains
only 1,331 untouched `Feedback=False` rows. The suggested 2,000/500/1,000 split is
therefore impossible without reusing observed items or adding feedback rows. This first
version preserves the primary no-feedback population and freezes:

- calibration pool: 600 items;
- development validation: 300 items;
- final test: 400 items;
- 31 untouched no-feedback rows left unused.

All three splits are proportionally stratified by `has_loss`, `ambiguity`, and tertile
of absolute EV gap using seed `20260918`. Human B-rate, human sample size, majority
choice, and target error are not used for sampling.

Frozen hashes:

| Split | SHA256 |
|---|---|
| calibration pool | `15e9304cfc7bd9292c22c3c747873bfca0aa03417c606bd5247dfc391489fb35` |
| development validation | `09a96481d4bfd58acf3d04555dbf6b1f99ea21fac76e9433f22a0a0b97514acb` |
| final test | `9854bb96695335761e3ec40ac146fe2544bedd8918d3fbf8d79eaa767734abd9` |

The runner verifies these hashes and the excluded pilot/confirmatory hashes before any
live stage.

## Methods

- B0 Vanilla: `Choose the option you prefer.`
- B1 Persona: `Respond as an ordinary adult participant making the choice for themselves.`
- B2 Few-shot Human-Calibrated: demonstrations selected only from the calibration pool.

For each demonstration item, its human B-rate is converted deterministically into 10
pseudo-participant A/B choices using nearest-integer allocation and a seeded shuffle.
The prompt presents the gamble and individual choices. It never states B-rate, EV,
variance, riskier-option identity, subgroup, or an explanation. The target item's human
rate, human majority, sample size, subgroup metadata, and benchmark identifiers never
enter its prompt.

Two B2 retrieval strategies are tested:

- Random: deterministic seeded ranking of calibration items.
- Matched: deterministic matching on `has_loss`, `ambiguity`, EV-gap bin, payoff scale,
  visible probability structure, and number of outcomes.

Retrieval never uses a target or calibration human choice rate. For each strategy,
development compares K = 4, 8, and 16. Demonstration option order is deterministically
counterbalanced by item identity so examples do not always attach human choices to the
same displayed position.

## Frozen model and sampling settings

- Provider/model: `aigc_relay / DeepSeek-V4-Flash`.
- Temperature: 0.7.
- Thinking: disabled.
- Maximum output: 32 tokens.
- Repeats: 10 per item/method.
- Target presentation: exactly five A-first and five B-first repeats.
- Invalid JSON: one repair request, then retained as invalid.

Development runs B0, B1, and six B2 candidates, for 24,000 planned responses. Only the
six B2 candidates are eligible for method selection. The primary development metric is
item-level MAE. The frozen tie order is:

```text
B2_random_k4
B2_random_k8
B2_random_k16
B2_matched_k4
B2_matched_k8
B2_matched_k16
```

After development, `selected_method.json` records the selected B2 method, K, strategy,
split hashes, prompt-template hash, development item-rate hash, source hash, code commit,
and model settings. This file must be committed before final-test execution. The final
stage refuses a dirty worktree, missing/untracked selection, changed split, changed
config, changed source, or changed prompt template.

## Final test and inference

The untouched 400-item final test runs exactly once with B0, B1, and the frozen B2,
for 12,000 planned responses. It is not read during method selection.

Primary contrasts:

```text
ΔMAE_B2_vs_B0 = MAE(B2) - MAE(B0)
ΔMAE_B2_vs_B1 = MAE(B2) - MAE(B1)
```

Each contrast uses 5,000 paired bootstrap replicates with benchmark item as the
statistical unit. The main success criterion is an upper 95% interval bound below zero
for B2 versus B0.

Secondary metrics are RMSE, mean Bernoulli Jensen–Shannon divergence, Pearson and
Spearman correlation, position bias, B-position effect, degenerate-item rate,
riskier-option choice rate, and higher-EV choice rate. Subgroups are loss/no-loss,
ambiguous/non-ambiguous, and small/medium/large EV gap. Human consensus is defined only
for analysis: strong consensus means B-rate ≤ 0.2 or ≥ 0.8; all other rows are mixed.
Consensus is not used for splitting, retrieval, method selection, or prompting.

## Interpretation and limitations

- If B2 beats B0 with the full interval below zero, human demonstrations provide a
  useful calibration signal in this benchmark.
- If B2 improves on development but not final test, few-shot calibration overfits or
  fails to generalize.
- If B2 does not improve development, few-shot prompting is insufficient and learned
  calibration can be considered in a separate protocol.

Choices13k is public and may exist in model pretraining. Demonstrations intentionally
contain calibration labels, so this experiment tests in-context calibration, not clean
zero-shot generalization. A successful result would concern aggregate risky binary
choices only; it would not establish financial-market or human-agent fidelity.

## Explicit stages

```bash
python scripts/run_choices13k_data_driven.py --stage plan
python scripts/run_choices13k_data_driven.py --stage prepare
python scripts/run_choices13k_data_driven.py --stage development --provider aigc_relay
python scripts/run_choices13k_data_driven.py --stage freeze
# Commit outputs/human_benchmark/choices13k_data_driven/selected_method.json here.
python scripts/run_choices13k_data_driven.py --stage final_test --provider aigc_relay
python scripts/run_choices13k_data_driven.py --stage evaluate
```

No stage is implicit. API response identities, retry logs, local cache, secret handling,
and clean-tree gates reuse the existing benchmark infrastructure.
