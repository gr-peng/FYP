# Financial baseline development results

Code, metrics and model definitions were frozen in `067b27c` before the first
development behavior read. Source bytes and complete session partitions were
frozen earlier in `0ecd73f`. Test sessions remain unopened.

The training sample contains 30 people and 433 owned-asset opportunities in two
sessions. The development sample contains 27 people and, coincidentally, also
433 opportunities in two different sessions. Only the first five main periods
are included, preceding the first peer ranking. The target is any sale of an
already-owned asset, conditional on the recorded human state.

| Fixed method | Person-balanced Brier ↓ | Person-balanced log loss ↓ |
|---|---:|---:|
| Always HOLD | 0.204230 | 2.821545 |
| Probability 0.5 | 0.250000 | 0.693147 |
| Train constant | **0.163528** | **0.509100** |
| Gain/loss/flat bins | 0.173082 | 0.532486 |
| Ridge logistic | 0.170205 | 0.526269 |

The train constant predicts a sale with probability 0.235977; the development
person-balanced human rate is 0.204230. Always HOLD would obtain high classification
accuracy but cannot reproduce selling, which illustrates why accuracy is unsuitable.
No LLM or Jev was evaluated in this financial experiment.

The logistic model improves on the constant in session `45zaunj0`
(0.169139 versus 0.176882) and worsens in `utt3zm8h`
(0.171057 versus 0.152845). Training gain/loss bins predict approximately 0.369
for gains and 0.146 for losses, yet that conditioned model also underperforms the
constant overall. This cautions against assuming that a plausible disposition
effect description improves out-of-session human prediction.

Two development sessions support descriptive comparisons, not a precise population
superiority claim. The many asset rows do not create hundreds of independent
markets. No IID significance interval was calculated. The raw-round mapping,
exogenous common price paths, short pre-ranking window and absence of a complete
joint action policy further limit interpretation.

Next comparison: plain LLM and training-calibrated LLM probability forecasts on
the same information, retaining all statistical controls. Jev can join this design
after real access is configured. Freeze any development-selected calibration before
test sessions; do not tune these five reported baselines and relabel the result as
the same frozen comparison. A micro gain must subsequently survive persistent
portfolio rollout before any market-realism claim.

Reproduce with `.venv/bin/python scripts/run_scopic_baselines.py --stage evaluate`
from a clean worktree with unchanged frozen code. Full numerical output and
coefficients are in `outputs/scopic_baselines_20260922/results.json`.
