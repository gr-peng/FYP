# Next financial probability comparison: execution design

This is a concrete design for implementation, not yet the frozen executable
protocol. No development quote states, current labels or scores have been opened.
Complete the final state adapter, runner, tests and freeze before model calls.

The train-only quote prototype now has 63,143 eligible active submissions from
51 sessions. Five-fold session CV favors both personal-history and logistic
controls over the training constant. This is a stronger, institution-related
financial bridge than reusing lottery probabilities as arbitrary trading actions.
However, partial observation and active-event conditioning remain explicit limits.

## Implementation order

1. Add summaries of the person's **earlier** submissions in previous periods and
   previous repetitions. Keep account resets separate from retained private memory.
   Test current/future action invariance across period and repetition boundaries.
   Preserve the current training CV as version-1 exploratory evidence; name any
   extended-history result separately. Do not claim that recorded human history is
   the same as autonomous generated history.
2. Fit fixed train-only statistical baselines to that final observation schema:
   constant, history prior, ridge .01, and the mean of the exact demonstration
   labels. Freeze fitted parameters and feature scaling. No finance test access.
3. Implement a dedicated source-checked development preparation path and freeze its
   eligibility, sample IDs, demonstration pairs, source hashes and output schema.
   Select 25 eligible events per development session by seeded hash, giving 400
   targets across 16 sessions if all quotas are available. Decide quotas before
   scoring; never select by prediction error or target BUY/SELL label.
4. Compare four LLM methods: no examples; four matched real-label examples;
   the same states with neutral .5 labels; the same states with shuffled training
   labels. Retrieve on visible state features only, with distinct training people.
   Fix demo identities and label permutation before calls. The one-hot observed
   labels are individual choices, not population probabilities or ten fake people.
5. Ask for a random participant's BUY/SELL probabilities **given an active order**.
   Both menu orders are required, mapped back and averaged: 400×4×2 = 3,200 main
   forecasts, plus a separate smoke. Use the existing strict probability adapter,
   frozen route/model settings, no trading fallback and complete repair accounting.
   Every model receives the same computed features and partial information.

## Scoring and interpretation to freeze

Primary: session-then-person-balanced Brier on the common valid sample. Retain
log loss, probability calibration, within-session results, option-order effects,
validity, latency and complete usage. Label controls test information-specific
benefit; statistical/history controls test added LLM value. Predeclare paired
session-bootstrap contrasts; the 400 events do not create 400 independent markets.

Validity requires a complete response matrix, no hidden fallback, and a frozen
common-coverage rule before scores. Development history necessarily includes
earlier observed human actions; no current target label may enter its own prompt
or example retrieval. Report recorded-history evaluation accurately. Do not use
the development response labels to refit the training statistical controls.

Freeze the chosen candidate, constraints and independent evaluation before the
16 test sessions. A direction forecaster is only one policy component. Before
market claims, supply coherent feasible quantity/price/type actions, arrival and
cancel timing, private generated history and a verified institution adapter.
Use a fixed common execution controller when isolating the effect of calibrated
side probabilities; later sensitivity tests can vary that controller explicitly.

Jev can enter the same bounded probability task after its pinned-version contract
probe succeeds with real access. Compare probability forecasts, sampled actions
and argmax as separate outputs. Freeze any Jev adaptation before independent test
access. Lack of a Jev key does not prevent completing the financial LLM experiment.
