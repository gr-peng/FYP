# Calibration V2 development ablation

Protocol date: 2026-09-22. This is a new, exploratory development experiment. It
reuses previously observed development data and does not claim independent
confirmation. The existing final test is not read by preparation, retrieval,
selection, or evaluation. Historical experiments and their source remain intact.

The question is whether correctly paired human demonstrations improve risky-choice
fidelity beyond neutral examples, shuffled labels, and simple probability shrinkage.

**Inputs and sample**

Use the frozen 600-item calibration pool and old 300-item development set. Their
hashes are fixed in `config/choices13k_calibration_v2.json`. Deduplicate the pool
and exclude pool-equivalent target inputs using a canonical fingerprint of the
visible gamble. Collapse equal known payoffs, remove known zero-probability
outcomes, and permit A/B reversal. Ambiguous option probabilities, response labels,
participant counts, row/problem IDs, and undisplayed correlation are excluded from
the fingerprint. Do not filter ambiguous payoffs using hidden probabilities.

Select 200 eligible old development records proportionally stratified by visible
loss and ambiguity only, seed 20260922. The remaining old development records are
not an untouched test. Store exact target/pool IDs, demonstrations, thresholds,
and the label permutation in a committed `config/frozen/choices13k_calibration_v2.json`
before live execution.

**Five methods**

- B0: the frozen vanilla wording.
- C_true: matched K=4, correct calibration labels.
- C_shuffled: the same matched examples, with pool labels cyclically permuted in
  a seeded order, with no fixed donor IDs. Preserve the pool label multiset.
- C_neutral: the same examples, with five A/five B choice records for each.
- C_random: random K=4, correct calibration labels.

Matched retrieval first matches ambiguity, then loss, then visible EV-gap bin,
outcome count, probability structure, and payoff scale, with stable hash ties.
EV-gap exists only when both options' probabilities are visible. Its two tertile
thresholds are fitted exclusively on non-ambiguous pool items. Neither target nor
pool response labels enter retrieval. Random selection uses only fingerprints and
the seed. Identical visible target/demonstration gambles are prohibited.

Use ten synthetic choice records per example, rounding the relevant label rate to
the nearest tenth. Call them example records, not actual participant trajectories.
Keep the template identical across four demonstration methods. The template and
retrieval changes mean C_true is a new version, not a rerun of the original B2.

**Sampling, transport, and freeze**

Use AIGC relay / DeepSeek-V4-Flash, temperature 0.7, thinking disabled, maximum
32 output tokens, ten responses per item/method with exact five/five option-order
balance. Interleave jobs with a seeded shuffle, independent of labels. The matrix
is 200 × 5 × 10 = 10,000 responses. Keep concurrency 16 and maximum four requests
per second, use the existing retry/repair and credential-redaction transport.

First run an engineering smoke check: first two frozen targets, five methods, two
orientations = 20 extra responses in a separate directory. These are development
items. Do not select methods from the smoke responses. Require complete, valid
smoke responses before the live development matrix.

Separate mock and real providers and smoke/development outputs. Use an exclusive
process lock, unique response identities, input/config/source hashes, a committed
plan and clean worktree before live runs. Resume only missing identities, retaining
invalid responses after the existing repair policy. Validate the full matrix and
counterbalancing before evaluation. Cache identities include the plan hash and stage.

**Inference and decisions**

Report item MAE, RMSE, JS, correlation, position effects, degenerate-item rate, and
invalid rate. Compute paired item bootstrap intervals with 5,000 replicates and
seed 20260923 for C_true minus B0/shuffled/neutral/random. The mechanistic criterion
requires both C_true−C_shuffled and C_true−C_neutral upper 95% bounds below zero;
it is an intersection criterion, and remains exploratory on reused development.
Do not treat the minimum point estimate as proof of superiority over every method.

Add constant 0.5, training mean, training median (a natural absolute-error baseline),
and matched-label mean. Report probability-prediction error separately from the
analytical expected error of ten independent draws from each predicted probability.
Ten synthetic draws are not ten real people. Compare computation time and tokens
without an unverifiable monetary estimate.

Select a candidate for subsequent external validation only among C_true/C_random,
by lowest development MAE, breaking ties in that order. All five methods must meet
invalid rate ≤1%. Passing selection is not passing independent validation. If
true labels do not beat neutral/shuffled controls, prioritize probability calibration
and narrow the mechanism claim. Freeze any externally tested policy in a separate
selection artifact and commit before using external labels.

**Execution**

```bash
.venv/bin/python scripts/run_calibration_v2.py --stage prepare
# Commit the protocol, code, configuration, and frozen plan.
.venv/bin/python scripts/run_calibration_v2.py --stage smoke
.venv/bin/python scripts/run_calibration_v2.py --stage development
.venv/bin/python scripts/run_calibration_v2.py --stage evaluate
```

The experiment lives in `experiments/` to avoid changing source hashes of the
historical package. Its own freeze includes the shared package, its implementation,
and entry point. Real output is under
`outputs/human_benchmark/choices13k_calibration_v2/aigc_relay/`.
