# Next probability experiment: design proposal after Calibration V2

Historical design proposal. The experiment is now complete; see
`probability_calibration_results_20260922.md`. The next external design is in
`cpc18_external_next.md`.
Calibration V2 failed its correct-label mechanism gate; its results are in
`calibration_v2_results_20260922.md`. Avoid another large action-sampling matrix
until the probability-prediction formulation has been tested.

Reuse the existing 200 development problems, frozen visible-information retrieval
pool, demonstration identities and shuffled-label mapping. All remain development
data. Compare four generative forecasts: no examples, correct matched labels,
neutral matched labels and shuffled matched labels. Each forecasts the choice
distribution of human experimental participants. Use the same visible gamble
information and rounded ten-choice example counts across demonstration methods;
change only the labels in the label-control contrasts. This is a new prediction
task, so do not call its difference from own-choice sampling a pure model effect.

Give each target both option orientations: 200 × 4 × 2 = 1,600 forecasts, plus
a separately frozen small contract smoke. Use the existing DeepSeek route and
`LLMProbabilityClient`, with explicitly frozen temperature/output budget, cache
namespace and request identities. Schema repair can fix formatting once; invalid
forecasts remain missing/fail the gate, with no implicit HOLD or 0.5 replacement.
Archive every transport attempt and usage, including repair calls.

Report continuous-probability MAE/RMSE, correlation and conditional behavior,
mean signed/absolute order effects, and analytic expected error from ten action
draws. The option-symmetrized original-B forecast is the mean of both orientations.
Retain constant 0.5, pool mean/median and retrieval means. Correct labels must
beat both neutral and shuffled controls for the mechanism claim. Exact comparisons,
bootstrap, invalid-response threshold and source hashes must be frozen before calls.

Inspect a small predeclared shrinkage family toward 0.5, for example
`p_alpha = 0.5 + alpha * (p - 0.5)` with alpha in {0, .25, .5, .75, 1}; fit/select
only on development, reporting the full curve. Alpha 0 is the constant baseline,
not a successful condition-sensitive human model. Preserve raw forecasts, and
freeze a final policy before opening CPC18 labels. These reused-development
comparisons cannot be relabeled independent validation.

Jev can join the same human-prediction design after access and its contract smoke
pass. Its argmax choice and confidence are distinct from sampled actions and
population choice frequencies. Keep the model comparison matched on state,
demonstrations, action set, probability scoring and external sampling.

In parallel, use the already built scopic pre-ranking states for a financial
probability comparison. Confirm the practice/main-period mapping and any
announced treatment framing in the original instructions; the current statistical
benchmarks intentionally use a restricted feature set. Do not open test sessions
until models and evaluation are frozen. A conditional sell-any model still needs
a coherent joint portfolio policy and persistent-state validation before the
central micro-to-macro experiment.
