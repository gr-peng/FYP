# Calibration V2: completed development ablation

The frozen experiment completed 10,000/10,000 valid responses, 2,000 per method,
on 200 previously used development problems, ten choices per method/problem with
five presentations in each option order. No new independent-test claim is made.
Code/protocol/plan were frozen at `26893b7` before live calls. The transport recovered
one HTTP 502; there were zero invalid decisions and zero cache hits. Successful
response usage totals 6,234,050 input tokens and 70,006 output tokens.

| Method | MAE ↓ | RMSE ↓ | Pearson correlation ↑ |
|---|---:|---:|---:|
| B0: no examples | 0.269923 | 0.333528 | 0.3352 |
| C_true: matched, correct labels | 0.238663 | 0.289520 | 0.4655 |
| C_shuffled: matched, shuffled labels | 0.253136 | 0.301630 | 0.4779 |
| C_neutral: matched, 50% labels | **0.235829** | **0.286068** | **0.4835** |
| C_random: random examples, correct labels | 0.238482 | 0.293169 | 0.4106 |

Paired problem bootstrap, 5,000 resamples, prespecified seed:

| Contrast in MAE | Difference | 95% interval |
|---|---:|---:|
| C_true − B0 | −0.031260 | [−0.062347, +0.001257] |
| C_true − C_shuffled | −0.014472 | [−0.040799, +0.011060] |
| C_true − C_neutral | +0.002835 | [−0.022638, +0.027916] |
| C_true − C_random | +0.000181 | [−0.028817, +0.029104] |

The prespecified correct-label mechanism gate fails: C_true does not outperform
both label controls with an interval below zero. The point improvement over B0
also has an interval crossing zero. This does not establish equivalence or prove
labels have no effect; the data do not support the proposed mechanism in this run.

The prespecified selection among C_true and C_random picks **C_random**, mechanically,
by only 0.000181 MAE. That tiny difference is not evidence of random retrieval
superiority. Preserve this selection record; do not retrospectively rename C_neutral
as the prespecified winner.

## Statistical controls and interpretation

| Statistical prediction | Exact-probability MAE | Expected MAE of 10 sampled actions |
|---|---:|---:|
| Constant 0.5 | 0.181091 | 0.216108 |
| Pool mean | 0.181125 | 0.216127 |
| Pool median | 0.180734 | 0.215918 |
| Visible matched-example mean | 0.192954 | 0.225692 |

Every LLM condition has larger observed MAE than the expected MAE of ten fair-coin
draws on these targets. That expectation is analytically integrated over binomial
draws, not a lucky random seed. Exact probabilities have a different sampling-noise
budget and are shown separately. No significance interval for a superiority claim
against the analytic control was prespecified here. A constant also has no problem
discrimination; lower MAE alone does not make a complete investor more humanlike.

Option-order dependence remains material: displayed-first choices range from
35.35% (B0) to 42.45% (C_true), despite equal order counts. Counterbalancing is
necessary, but it does not demonstrate an invariant policy. C_neutral's stronger
correlation also cautions against attributing the example effect specifically to
human response labels.

This is a changed visible-information protocol on reused development problems,
not a direct repeat of the earlier B2 final test. Keep the historical final-test
result and its documented sensitivity analyses intact. The new result narrows
the mechanism claim and changes our next experimental priority.

## Next research decision

Do not promote the current demonstration method into the main market study as
validated human calibration. First compare direct human-choice probability
forecasts with true/neutral/shuffled demonstrations, measure order sensitivity,
and fit any simple probability shrinkage on development only. Retain constants
and conditional-response metrics. Then freeze independent CPC18 evaluation on
the 175 unique, nonoverlapping risk problems while its human labels remain closed.
Finance-specific prediction and persistent rollout remain additional requirements.

Full machine-readable output:
`outputs/human_benchmark/choices13k_calibration_v2/aigc_relay/development/evaluation/summary.json`.
Plan SHA256: `85dc2d2df61c29c1c0070ab8362a3165ddf5c856b24436d5d56150bc420b8c14`.
Source SHA256: `24a9487a43dc3e1e3868d409302dae310592b45f1c37527fe1e2ca5dee4f7659`.
