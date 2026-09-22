# Jev first real behavioral comparison — 2026-09-23

**Jev did not improve human-choice fidelity in this first comparison.** On 200
already-observed Choices13k development problems, Jev with human examples had
MAE 0.208199 versus 0.163021 for the same-condition DeepSeek-V4-Flash forecasts.
This is an exploratory reused-development result, not an independent test or
evidence about financial trading or market realism.

## Execution and fair comparison

The user supplied TypeSafe access and explicitly resumed Jev experiments.
Credentials are only in ignored local `.env` with mode 0600. Direct connectivity
timed out; the environment proxy reached the official endpoint with TLS enabled.
Explicit proxy support was committed at `7a5d9c8`. Four real contract probes passed.
The endpoint/version were rechecked against the official
[API reference](https://docs.typesafe.ai/api) and [models](https://docs.typesafe.ai/models).

Protocol, runner and plan were frozen at **a4500ba** before the comparison:
`docs/jev_development_protocol_20260923.md` and
`config/frozen/jev_development_20260923.json`. Model: **jev-1.13.0**.
Plan SHA256: `38f5bbab64bd6b89f10ef2015f21761f837b952e42c12f915ed20f49f99a85f8`.

All 16 comparison-smoke and 1,600 main forecasts passed probability and identity
checks. Four conditions × 200 items × two option orders. The exact state,
instructions and criteria of every planned request were compared to all 1,600
historical LLM request payloads: all matched. The LLM additionally uses its existing
JSON-output system wrapper; interfaces/tokenizers are not identical. Historical
forecasts were reused without new LLM calls. No prompt tuning during this run.

One main HTTP 503 recovered under the predefined retry policy: 1,601 main attempts,
1,600 valid forecasts, zero cache hits. Run/evaluation process exited successfully.
Completion: 2026-09-22 19:35:51 UTC / 2026-09-23 03:35:51 China time.
Full response SHA256:
`8314dace646a334ba879848fbc7888c7da7852e8afb302f8a26c81d494aef63a`.

## Results

MAE is measured against human choice proportions after averaging original-B
probabilities over both option orders; lower is better.

| Condition, all 200 items | Jev MAE | LLM MAE |
|---|---:|---:|
| No examples | 0.193593 | 0.179454 |
| Matched human examples | 0.208199 | 0.163021 |
| Neutral example counts | 0.216882 | 0.170884 |
| Shuffled human counts | 0.200251 | 0.166544 |

Constant probability 0.5 has MAE **0.181091**. Jev does not beat it descriptively
in any of the four conditions. The main fixed contrasts use 5,000 paired item
bootstrap draws, seed 20260923; intervals are exploratory and unadjusted:

| Contrast | MAE difference | 95% item-bootstrap interval |
|---|---:|---:|
| Jev plain − LLM plain | +0.014138 | [−0.009154, +0.037075] |
| Jev human examples − LLM human examples | +0.045178 | [+0.023124, +0.067821] |
| Jev human examples − Jev plain | +0.014606 | [+0.001006, +0.028708] |
| Jev human examples − Jev neutral | −0.008683 | [−0.017619, +0.000102] |
| Jev human examples − Jev shuffled | +0.007948 | [−0.000687, +0.017081] |

The plain-model contrast is inconclusive. With human examples, Jev's error is
about **27.7% higher** than the corresponding LLM. Giving Jev the examples worsens
its MAE versus plain Jev on this development set. Correct-label superiority over
both controls is unsupported; this does not prove examples never help Jev.

Risk-only comparison on the **same 161 nonambiguous items**:

| Method | MAE |
|---|---:|
| Fixed logistic statistical model | 0.086638 |
| Fixed aligned kNN | 0.100011 |
| LLM with human examples | 0.158909 |
| Constant 0.5 | 0.174865 |
| Plain Jev | 0.199434 |
| Jev with human examples | 0.221078 |

Jev-with-examples has higher correlation than LLM-with-examples (0.527 vs 0.432
over 200 items), but worse MAE/RMSE/JS. Correlation is not probability calibration.
Whether a separately fitted calibration can help is an untested follow-up, not
evidence of a successful hybrid. Absolute option-order effects remain substantial:
Jev plain 0.14095, Jev with examples 0.11865, LLM with examples 0.14400.

## Runtime and resource accounting

| Stage | Recorded input tokens | Recorded output tokens |
|---|---:|---:|
| Initial contract probe, 4 requests | 1,514 | 124 |
| Comparison smoke, 16 requests | 11,342 | 496 |
| Main, 1,600 successful responses | 1,260,860 | 49,600 |
| Total recorded | **1,273,716** | **50,220** |

Main successful-response latency: median **0.356 s**, p95 **0.495 s**; it excludes
queue time and failed attempts. The 503 response returned no usage, so its possible
billing cannot be reconstructed. The aborted initial direct connection also did
not return usage. No monetary estimate is made without a confirmed Jev tariff;
the earlier DeepSeek tariff does not apply to Jev. Historical LLM latency is not a
controlled speed comparison. Native token counts need not share a tokenizer.

## What this changes

1. Access, structured probability outputs and real logging now work. Jev is no
   longer blocked by missing credentials.
2. The current evidence does **not** support replacing the LLM investor with Jev
   to improve human fidelity. Keep statistical controls as the strongest baseline.
3. Any probability calibration or hybrid gate needs separate development checks
   and a new frozen evaluation. Do not tune against the already-observed CPC18
   outcomes and present them as an untouched external test.
4. Real financial choices, persistent private memory and autonomous market feedback
   remain untested for Jev. Financial development/test behavior was not opened here.

Artifacts: `outputs/jev/development_20260923/` contains smoke/main request,
attempt/response logs, manifests, completion, `api_usage.json` and
`development/evaluation/{summary,independent_audit}.json`.
An independent reconstruction checked all 1,600 raw probability mappings and
reproduced every primary Jev MAE to 1e-12; frozen source/data hashes still match.
Validation: existing 110-test suite passed, plus two new resumption/mapping and
smoke-gating tests passed; changed-file Ruff checks passed.

Offline reproduction (no model calls):

```bash
.venv/bin/python scripts/run_jev_development.py --stage evaluate
```
