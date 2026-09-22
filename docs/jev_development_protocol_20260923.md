# Jev reused-development comparison

Freeze before model calls: 200 existing Choices13k development items, all four
conditions P0 / matched human examples / neutral example counts / shuffled counts,
both display orientations, 1,600 main forecasts plus 16 smoke forecasts. Reuse
the exact state, instructions, criteria, retrieval identities and mappings of the
completed LLM probability experiment. No target human rates enter requests.
The prior four-call access probe is engineering evidence only.

Pin jev-1.13.0 at the official TypeSafe endpoint. Explicitly enable the machine's
configured environment proxy: direct unauthenticated connectivity timed out,
whereas proxied connectivity succeeded. Keep TLS verification and host allowlist.
Use 2 requests/second, concurrency 4, existing capped retry policy; stop on
authentication/credit errors or invalid distributions. Do not substitute models.
Commit protocol, runner, hashes of every payload and historical comparator, and
source hashes before calls. File lock, identity checks and cache permit resuming.

Primary outcome: MAE of human choice rates, averaging original-B forecasts over
both option orders. Secondary: RMSE, corrected mean Bernoulli JS, expected error
of ten independent draws, and absolute orientation effect. Compare each Jev
condition to its historical LLM counterpart; compare true examples to plain,
neutral and shuffled Jev. Report paired item-bootstrap intervals (5,000 samples,
seed 20260923), with no participant-population or confirmatory claims. These
development human outcomes were already observed before this new experiment.
No prompt/hyperparameter selection is performed within this run.

Retain constant .5 and the pre-existing fixed logistic/kNN models. Statistical
models only support nonambiguous risk items; compare all policies on that same
subset separately, never compare 161-item scores directly to 200-item scores.
Jev's argmax/confidence is not a sample or a calibrated human frequency. Main
probability evaluation makes no claim about autonomous persistent agents.

Record raw responses, request IDs, timing, native input/output token usage and
transport attempts, retaining smoke and main separately. No monetary cost claim
without a confirmed Jev tariff. Historical latency cannot establish a controlled
speed advantage. No new LLM calls are needed for this first comparison. Financial
development/test remains closed; Jev development gains alone cannot justify a
market-realism claim or a hybrid gate. Subsequent studies require separate freeze.
