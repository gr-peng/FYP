# Independent CPC18 evaluation: next execution design

Draft for implementation; no CPC18 human responses have been read and no external
forecast run has started. Freeze the complete executable/config/plan before calls.

Inputs: the 175 risk-only unique GameIDs in the audited source screen, excluding
five visible overlaps with all Choices13k source gambles and the second member
of duplicate pairs 12/13 and 19/20. Use complete distributions from the workbook,
with explicit response-column exclusion. Record workbook, raw CSV, source screen,
training pool and all code hashes. Reuse the visible-information retrieval rules.

The probability study selected P_true, alpha 1. Preserve that candidate and carry
P0, P_neutral and P_shuffled as controls with the same forecast task and two orders.
This implies 175 × 4 × 2 = 1,400 main forecasts plus separate contract smoke.
The nominal old action-experiment C_random selection remains in its own report;
this is the subsequently selected probability task, not a relabeling of that run.

Include constant .5, pool mean/median, symmetric fractional logistic and aligned
16-neighbor controls. Freeze the fitted statistical model/checkpoint from the
487 risk-only training cases before opening external responses. Avoid using the
misaligned raw-B retrieval mean as the only learned non-LLM comparator.
A post hoc development mixture grid selected zero LLM weight; retain that negative
result. Any different mixture must be defined/calibrated before reading CPC18
outcomes, and cannot silently replace the selected pure statistical comparator.

Evaluation should open only no-feedback trials 1–5 of eligible GameIDs after all
forecasts and their validity audit are complete. Raw archive metadata already
confirms 686 unique participants, 84,860 relevant repeated trials and 80–125 people
per item. Each five-trial block belongs to one person; it is not five independent
people. Freeze and log response hashes and label-opening time.

Primary probability MAE and contrasts should retain P_true-versus-both-label-controls
as the mechanism test, and compare P_true with the strong statistical controls.
Also report exact versus analytically sampled-ten errors, discrimination and
option-order effects. Use corrected per-item JS in bits. Document that a
problem-bootstrap interval conditions on the observed human rates; add participant
cluster resampling within original set/site/order-condition strata if claiming
uncertainty in those rates. Freeze the exact uncertainty method before label access.

This is near-family cross-source validation, not evidence against pretraining
contamination or a financial-market result. Subsequent financial micro validation,
persistent joint actions and a matched human market reference remain required.
