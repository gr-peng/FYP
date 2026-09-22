# CPC18 external probability evaluation

Freeze executable, config, input hashes, exact 175 GameIDs, demonstrations,
statistical checkpoint and all control predictions before the first external call.
No human CPC18 choice labels informed this design or the selected policies.
This is external to our calibration data, not a guarantee against model pretraining
contamination. It remains the same broad lottery-choice family, not a market test.

## Fixed policies and input access

Carry the probability-development selection P_true, alpha=1. Keep P0, P_neutral
and P_shuffled with identical visible-state construction, matched K=4 retrieval,
rounded example counts and label permutation. Reuse the training pool and its
visible-feature thresholds. Both target orders are required and averaged in
original-B coordinates. No prompt, shrinkage or retrieval adjustment after labels.

Use the existing DeepSeek-V4-Flash route at temperature 0, thinking off, max output
80, concurrency 16 and 4 requests/second. Exactly 175×4×2=1,400 main forecasts plus
a separate 16-call smoke on the first two targets; repair/retry rules are inherited.
Require all smoke forecasts valid. Freeze cache namespace and full identities;
preserve all response/transport records. Mock paths must never read human labels.

Read workbook features through an explicit allowlist excluding B.1–B.5. Retain
only non-ambiguous problems, remove all five overlaps with the complete Choices13k
source, and keep smaller IDs in duplicate pairs 12/13 and 19/20. Validate visible
fingerprints against the already completed metadata-only audit. Ambiguous cases
remain outside this experiment. Source IDs never appear in model state.

Freeze the fitted 487-item statistical model (ridge .01, k16) and predictions for
symmetric fractional logistic and aligned nearest neighbors. Retain constant .5
and pool mean/median. No statistical model is refitted on CPC18. The development
mixture selected zero LLM weight, so the statistical policy remains standalone.

## Label-opening gate

After the full response matrix, require ≤1% invalid forecasts in each method and
≥99% of target items to have valid forecasts in both orders for all four methods.
Do not substitute one order, 0.5 or an action fallback. If the gate fails, leave
human outcomes closed and report a failed execution. Never tune on partial scores.

Before opening outcomes, require a clean committed worktree and record plan,
response, source and code hashes plus UTC time. Repeated evaluation must match
the original opening record. Read only required CSV columns, filter eligible
GameIDs and no-feedback trials 1–5 before outcome conversion/analysis. Validate
one five-trial block per participant/game and binary B responses. Do not use
payoffs, reaction time, feedback-stage decisions or excluded-game outcomes.

## Scoring and uncertainty

Human B rate is the mean of the eligible five repeated decisions per participant;
each participant/game has the same count. Primary outcome is item-mean MAE.
Also report RMSE, mean per-item Bernoulli JS in bits, Pearson/Spearman, analytical
expected MAE of ten independent simulated actions, signed/absolute order effects,
and human/model preference for the higher-EV and riskier options when not tied.
The ten-action calculation is not a model of persistent individual trajectories.

Prespecified MAE contrasts: P_true minus P0, P_neutral, P_shuffled, constant .5,
symmetric fractional logistic and aligned kNN; also logistic minus constant .5.
Use 5,000 resamples and the frozen seed plus one. Report three 95% intervals:

1. Problem bootstrap, holding observed human rates fixed.
2. Participant cluster bootstrap, holding the problem set fixed. Resample whole
   anonymous participants within Set×Location×Condition strata. All their eligible
   problems and five repeated choices travel together; never resample trial rows.
3. Joint bootstrap using those participant draws and independently resampled
   problems. This is the primary uncertainty summary for the fixed contrasts.

Conditional/joint intervals refer to this design and its observed problem family;
seven source sets and two collection sites do not identify arbitrary populations.
No extra tuning, optional stopping or selecting between interval definitions.
Correct labels pass the mechanism gate only when BOTH P_true−P_neutral and
P_true−P_shuffled joint upper bounds are below zero. Failure is not equivalence.
Keep all model/control results even if the selected LLM candidate loses.

Human labels stay unavailable to every preparation/inference/mock function.
Record selected and omitted IDs, participant counts, strata, every prediction,
per-item human rates, fixed comparisons and usage. Financial test sessions remain
unopened; this evaluation cannot establish micro-to-macro or Jev superiority.
