# Jev agent comparison: hypotheses and staged validation

2026-09-22. Draft: no Jev requests or performance results have been obtained.
The likely intended model is TypeSafe AI Jev, released 2026-09-15. The user has
not yet confirmed the name. Official documents currently identify `jev-1.13.0`;
pin that version rather than a moving alias when the access route allows it.
[Release](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
[models](https://docs.typesafe.ai/models), [API](https://docs.typesafe.ai/api).

Official API, model, confidence, choice, and limitation documentation is archived
under `data/raw/jev_docs_20260922/`, with URL and SHA256 records. Current local
credential checks found only the existing AIGC key; no TypeSafe, Vercel Gateway,
or OpenRouter key. The AIGC model registry returned HTTP 200 and does not list Jev.
Real Jev comparisons need authorized access. Mock results cannot answer whether
Jev improves on LLM agents.

**What is being compared**

Jev accepts text/structured state and returns typed choices, their probabilities,
and confidence. Its API returns the maximum-probability choice; that is different
from drawing a simulated participant's action from the returned distribution.
The latter must be an explicitly named stochastic policy implemented in code.
Confidence and probability describe model judgments; they are not established
human population choice frequencies. Format correctness also does not establish
semantic correctness or human behavioral fidelity.
[Choice specification](https://docs.typesafe.ai/primitives/choice),
[confidence](https://docs.typesafe.ai/confidence).

The provider documents weaknesses in counting and numeric precision, recommending
arithmetic outside the model. This makes a short, structured action decision a
plausible use in this project; numerical risk-choice superiority is an empirical
question. Any computed features must be given to all compared policies and must
not disclose ambiguity-hidden information.
[Documented limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

**Hypotheses**

| Hypothesis | Intervention | Falsifiable evidence |
|---|---|---|
| J1: simpler bounded decisions reduce execution overhead | Jev A/B or discrete trade action vs a generative LLM with the same action set | Measured invalid/constraint rate, per-item wall time, server call latency, tokens and verified cost; cache hits excluded |
| J2: probability outputs improve aggregate human-choice fidelity | Jev probabilities, with/without human demonstrations, vs LLM probability outputs and sampled actions | Independent human-data MAE/RMSE/JS and conditional response effects; constant and learned statistical baselines retained |
| J3: a hybrid retains fidelity while reducing cost | Jev routes to LLM using a development-fitted gate | Held-out fidelity noninferiority and lower measured resource use, including every gating/fallback call |
| J4: micro improvements survive market feedback | Validated Jev/hybrid policy substituted into the same CDA execution layer | Run-level distance to independent human market targets, with paired environment seeds and fixed liquidity rules |

J1 can succeed while J2/J4 fail. Report efficiency, semantic performance, human
fidelity, and market fidelity separately. An accurate financial optimizer is not
necessarily a realistic human investor.

**Stages**

1. Access and contract smoke check on two already-used development items, both
   orientations. Log requested/resolved version, server IDs, raw structured result,
   probability normalization, response timing and token usage. Reject missing,
   non-finite, out-of-range, or schema-inconsistent values rather than silently
   substituting an LLM result. Protect credentials with a strict host allowlist.
2. Development comparison on the frozen calibration-v2 targets. Compare a 2×2
   model × human-demonstration design where possible, plus the existing statistical
   baselines. Jev's question asks which displayed option a randomly selected adult
   participant would choose, with visible information only; generative probability
   comparators receive the same behavioral prediction task. Keep the original
   action-sampling baseline as a distinct estimand.
3. For continuous probability predictions, average original-B probabilities over
   both display orientations and report the orientation effect. For actions, report
   both argmax and externally seeded stochastic sampling as separate policies.
   Compare exact-probability error separately from matched 10-action sample error;
   do not attribute removal of Monte Carlo noise to better behavioral understanding.
4. Fit any probability calibration or hybrid gate exclusively on development,
   freezing its parameters, retrieval pool, prompt schema, sampling rule and
   criteria before an external evaluation. A maximum-probability threshold is not
   a guaranteed threshold on human prediction error. Include matched LLM-only
   budget baselines and assess the quality/latency tradeoff.
5. Only a policy that passes the financial behavior bridge proceeds to the main
   market comparison. Give every policy the same observations, persistent agent
   state, admissible discrete actions, and quantity/limit-price execution rule.
   If Jev uses a bounded action menu, compare against a bounded-action LLM too;
   otherwise interface redesign would be confounded with model choice.

Financial state should contain code-computed account values and pre-decision
information, never realized future outcomes. Account validation stays in Python.
Do not make model-side type guarantees replace portfolio/market invariants.
Avoid simultaneous changes to persona, memory, background liquidity and model.

The external data, exact request matrix, noninferiority margin, macro targets,
budget and run count are not yet frozen. They depend on access, calibration-v2
results and the selected human financial dataset. This document records testable
applications; it is not evidence of Jev superiority.
