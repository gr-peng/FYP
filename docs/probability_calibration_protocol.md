# Probability calibration development protocol

Frozen before the first call of this experiment. The preceding Calibration V2
result motivated this design; the 200 target problems are reused development data.
No Choices13k previous final-test or CPC18 response labels are read.

## Design and inputs

Reuse the exact target IDs, deduplicated pool, matched four-example identities,
example orientation seed and label permutation from the frozen Calibration V2
plan. P0 has no examples; P_true has correct labels; P_shuffled substitutes the
frozen permuted labels; P_neutral assigns 5/5 counts to every matched example.
For the three example conditions, only labels change. Counts are rounded to the
same ten-choice scale as V2, with their orientation mapped correctly. They are
example aggregates, not ten identifiable humans or persistent trajectories.

The new task forecasts which option a randomly selected adult experimental
participant would choose without outcome feedback. The state shows visible
gamble information, optional example counts and no target response rate, identity,
hidden ambiguity probability or latent statistics. This changes both task and
response representation relative to own-choice V2; cross-experiment differences
cannot identify a pure elicitation or model effect.

Use DeepSeek-V4-Flash via the already configured AIGC route, temperature 0,
thinking disabled, max output 80 tokens, max concurrency 16, max 4 requests/sec.
Version drift is checked against the configured model string; the service may
still change a model behind that string. Cache namespaces and response identities
include this frozen plan. Do not claim deterministic generation from temperature 0.

Each of 200 targets × four methods × two orientations receives one forecast:
1,600 requests before transport/schema retries. A separate 16-request smoke uses
the first two targets and both orders, with separate identities. Smoke must be
entirely valid before the main run. Shuffle dispatch by the frozen seed. Preserve
all attempts, raw outputs, repairs, usage and code/source hashes. Mock stages use
separate paths and provide engineering evidence only.

Validate finite numeric probabilities on exactly A/B, total within 0.001 of one;
record any normalization. Permit one format repair. Missing or invalid forecasts
are not replaced. Every item prediction requires both orientations, mapped back
to original B and averaged. A single orientation is not silently substituted.

## Outcomes, controls and selection

Primary outcome: problem-mean MAE of the symmetrized probability versus human B
rate. Also report RMSE, Bernoulli JS, Pearson/Spearman, signed and absolute
original-B order effects, displayed-first predicted mass, subgroup MAE by loss
and ambiguity, and higher-EV/riskier preference on observable non-ambiguous
problems with nontied options. Constants have undefined discrimination.

Use the intersection of complete valid item pairs across all four methods for
comparisons. Report omitted IDs and each method's valid-pair count. An external
candidate requires invalid request rate ≤1% in every method and common coverage
≥99%. Report diagnostics even if the gate fails, without selecting a candidate.

Paired problem bootstrap: 5,000 draws, seed from config + 1. Prespecified contrasts
are P_true minus P0, P_shuffled, P_neutral in MAE. Correct labels are supported
only if both label-control intervals have upper bounds below zero. These are
exploratory intervals on reused development, not confirmatory inference.

Retain exact-probability predictions for constant .5, pool mean/median and the
visible matched-example mean. For every method and baseline, also integrate the
expected MAE of ten independent actions analytically. This exposes sampling noise;
it is not a persistent-person model. Use the same comparison population for all.

Evaluate the complete shrinkage grid `0.5 + alpha*(p - 0.5)` for alpha in
{0, .25, .5, .75, 1}. Selection minimizes development MAE across all four methods
and all alphas; exact ties prefer smaller alpha, then P0/P_true/P_neutral/P_shuffled.
Alpha zero is explicitly a constant policy requiring no model. Keep the complete
grid and raw outputs. This selection is exploratory; freeze it before external
human labels are opened. Report shrunk predictions' rank correlations as well:
shrinking a forecast does not itself establish a better conditional mechanism.

Jev is a subsequent model comparison on the same state/task and probability
scoring after access and contract validation. No real Jev result can be inferred
from these LLM-only or mocked stages. Financial behavior, joint portfolio
feasibility, persistent heterogeneity and human-reference market dynamics remain
separate required tests of the project question.
