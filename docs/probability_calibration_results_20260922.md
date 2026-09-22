# Probability calibration: completed reused-development study

Frozen at `660f9c0` before calls. The real smoke completed 16/16 valid responses;
the main matrix completed 1,600/1,600 valid forecasts on all 200 problems, with
no format repairs or cache hits. Successful main-call usage: 837,866 input and
46,392 output tokens. Every estimate below uses both option orientations.

| Probability forecast | MAE ↓ | RMSE ↓ | Pearson ↑ | Mean JS, bits ↓ |
|---|---:|---:|---:|---:|
| No examples (P0) | 0.179454 | 0.219151 | 0.2904 | 0.038538 |
| Correct labels (P_true) | **0.163021** | **0.198329** | **0.4322** | **0.031912** |
| Neutral labels (P_neutral) | 0.170884 | 0.204953 | 0.3514 | 0.034107 |
| Shuffled labels (P_shuffled) | 0.166544 | 0.205630 | 0.3665 | 0.034078 |
| Constant 0.5 | 0.181091 | 0.211486 | undefined | 0.036662 |

The prespecified grid selects **P_true, alpha = 1**, without shrinkage. This is
a development-selected candidate, not evidence of independent performance.

| Paired MAE contrast | Difference | 95% problem-bootstrap interval |
|---|---:|---:|
| P_true − P0 | −0.016433 | [−0.031692, −0.001773] |
| P_true − P_neutral | −0.007863 | [−0.018233, +0.002395] |
| P_true − P_shuffled | −0.003523 | [−0.014091, +0.006872] |

Correct labels improve the point estimate over the no-example prompt, and that
exploratory interval excludes zero. The true-versus-both-label-controls gate still
fails. Do not claim that the experiment identifies learning the correct human
label relationship. Do not call these reused development problems independent test.

Order dependence persists: P_true's original-B forecast changes by 0.144 on average
in absolute value when options swap; mean displayed-first mass is 0.438. Averaging
both orientations is part of the frozen policy, not proof that a single forecast
is invariant. On observable non-ambiguous items the human higher-EV choice rate
is 0.6022, versus 0.5348 for P_true. Lower global MAE does not imply all conditional
behaviors are better matched.

## Reporting correction

During review, the frozen evaluator's `mean_js` was found to be a sum: it supplied
arrays to a scalar Bernoulli-JS helper. The pre-call frozen source, raw responses
and original report are preserved. `scripts/correct_probability_js.py` reconstructs
predictions from raw responses and produces `evaluation_corrected/summary.json`
and corrected subgroup JS values. The correction records source/response/report
hashes. Tests check the one-bit bound and invariance to duplicating the dataset.
All MAE, RMSE, correlation, bootstrap contrasts and selection remain unchanged.
An independent raw-response MAE calculation also matches all four methods.

Use the corrected summary as authoritative for JS. The original `--stage evaluate`
is retained for frozen reproducibility and still reproduces its documented defect;
run the correction script afterwards. New studies must use a per-item JS mean.

## A stronger control changes the next step

The risk-only subset has 161 development items. Two additional fixed statistical
controls, fitted solely on 487 non-ambiguous training-pool items, obtain:

| Method on the same 161 risk items | MAE ↓ | Pearson ↑ |
|---|---:|---:|
| P_true probability forecast | 0.158909 | 0.4265 |
| Constant 0.5 | 0.174865 | undefined |
| Symmetric fractional logistic regression | **0.086638** | **0.8530** |
| Economically aligned 16-neighbor mean | 0.100011 | 0.8057 |

These are exploratory post-probability-study controls, not retrospectively
preregistered results. Their hyperparameters were fixed (.01 ridge, 16 neighbors),
without a search. Features use disclosed gamble distributions only. No target
labels enter prediction; pool/development share zero IDs and zero visible gamble
structures. Swapping options complements their probabilities exactly. This corrects
an important weakness of the earlier raw-B retrieval mean, which did not align
economic option roles before averaging labels.

This is stronger evidence that compact, train-fitted behavioral structure matters
than that a general LLM already models human risk choice well. Independent external
validation must include these controls. A post hoc mixture diagnostic also combined
the fitted statistical prior and P_true with LLM weights 0/.25/.5/.75/1. The MAEs
were .086638/.095707/.113204/.134790/.158909: the grid selects zero LLM weight.
This diagnostic does not support claiming added LLM value from a simple blend.
Preserve the pure statistical policy as a main external comparator. More elaborate
hybrids would need a separate rationale and independent validation.

Full outputs are under `outputs/human_benchmark/probability_calibration/aigc_relay/`
and `outputs/risk_statistical_baselines_20260922/`. CPC18 human labels and financial
test behavior remain unopened. Jev still has no real comparison result.
