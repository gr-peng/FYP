# Active research objective and continuation state

Updated 2026-09-22. The user authorized continued experimentation, independent
gap analysis against the micro-to-macro research question, iterative improvements,
and a real comparison of the recently released Jev with LLM agents. This objective
is active and is not satisfied by a new runner or a single calibration experiment.

**Authoritative progress**

- Implemented the isolated `experiments/calibration_v2.py` experiment and explicit
  `scripts/run_calibration_v2.py` stages, with config and frozen plan.
- New semantic tests cover hidden-probability invariance, target/pool label
  isolation, visible structural duplicates, shuffled-label marginal preservation,
  neutral prompts, and response identity/counterbalance checks.
- Full suite after financial adapters and probability elicitation: **89 passed**
  in the project `.venv`; new-file Ruff checks passed.
- The separate mock provider completed its 20-response smoke, 10,000-response
  development matrix, and evaluation. This is engineering evidence only.
- Protocol/code/config/plan committed as `26893b7` before real calls. Real smoke
  completed 20/20 valid responses. The real development job and evaluation are
  **complete**, both exited 0; former exec sessions 43945/16961 are closed.
  Exactly 10,000 valid responses, zero invalid/cache hits, 2,000 per condition.
  One HTTP 502 recovered; successful usage: 6,234,050 input and 70,006 output tokens.
- Calibration V2 MAE: B0 .269923, C_true .238663, C_shuffled .253136,
  C_neutral .235829, C_random .238482. C_true−B0 interval [-.062347,+.001257];
  neither true−neutral nor true−shuffled excludes zero. The prespecified
  correct-label mechanism gate **fails**. The mechanical external candidate is
  C_random by only .000181 versus C_true, not evidence of retrieval superiority.
  Constant .5 exact MAE .181091, analytic expected ten-draw MAE .216108.
  See `docs/calibration_v2_results_20260922.md`; all are reused-development results.
- Process log:
  `outputs/human_benchmark/choices13k_calibration_v2/aigc_relay/development_process.log`.
  The actual output/response directory is the sibling `development/` directory.
- AIGC model registry: HTTP 200, configured DeepSeek-V4-Flash available, no Jev
  entry. TypeSafe/Vercel/OpenRouter credentials absent at check time.
- Archived official Jev docs; application hypotheses and comparison boundaries
  are in `docs/jev_agent_protocol_draft.md`. No Jev empirical results yet.
- Implemented version-pinned direct Jev transport and a four-request probe entry
  point. Mock HTTP tests validate response contracts, model drift, action sampling,
  caching and key redaction. `scripts/run_jev_probe.py --stage status` confirmed
  that direct/Gateway credentials are still absent; do not call this real validation.
- Implemented `experiments/probability_elicitation.py` so generative LLMs can be
  compared with Jev probability outputs directly. Strict validation, explicit
  normalization, one schema repair and complete per-call usage; no real probability
  calls yet. Do not use the shared trading-specific `decide()` fallback for this task.
- Downloaded and checksum-verified CPC18 raw records and dictionary. Feature-only
  audit yields 177 risk candidates after excluding five overlaps with the entire
  Choices13k source, then **175 unique candidates** after keeping the smaller IDs
  of duplicate pairs 12/13 and 19/20. All 210 workbook gamble definitions match raw
  definitions; the full raw archive has 270 games. Eligible data: 84,860 no-feedback
  trial rows from 686 people, five repeated trials per person/game. Response labels
  remain unread. See `docs/external_data_screen_20260922.md`.
- Downloaded two financial human datasets: a 21-session single-asset market workbook
  (25,042 quote rows) and a panel with risk elicitation and six-asset trading fields.
- Financial whole-session partitions frozen in **0ecd73f**, before training
  behavior: CDA 11/5/5 sessions; scopic 2/2/2 sessions, both treatments in each split.
  `experiments/finance_data.py` rejects test reads and source-hash drift.
- Training audit: scopic has 30 people, 510 rows, exact share conservation and
  nonnegative reconstructed cash. Raw round 4 resets holdings/cash for 14 main
  periods. `result_final` equals next-price-marked wealth in **390/390** nonterminal
  main records: exclude it from current inputs. Terminal future prices unavailable.
- `experiments/scopic_states.py` builds only pre-ranking main periods 1–5:
  150 training decisions, 433 owned-asset sell opportunities. It separates targets
  and IDs from past-only financial state. Do not edit this now-frozen module casually.
- Frozen financial statistical comparison **067b27c** completed on development:
  27 people, 433 opportunities, 2 sessions. Person-balanced Brier: training constant
  **0.163528**, ridge logistic **0.170205**, gain/loss bins **0.173082**; logistic
  improves in one session and worsens in the other. No IID significance claim;
  no financial LLM/Jev comparison yet. Test behavior remains unopened.
- CDA training archive is not a full interleaved event log: eventual acceptance
  annotates quotes; quote/trade chronology differs in 88/128 trading rounds.
  Exact public order book/cash/opportunity replay is unresolved. Do not treat
  each filled quote row as an immediate trade or Market Price as a pre-quote field.
  See `docs/financial_training_audit_20260922.md` and `docs/scopic_baseline_results_20260922.md`.
- The financial herding replication package has a verified public file listing;
  a local direct page fetch returned HTTP 403. Raw individual data not obtained.

**Next actions**

1. Implement and freeze the probability-prediction experiment outlined in
   `docs/probability_calibration_next.md` (not yet running/frozen). Reuse development
   IDs and retrieval maps, compare no/true/neutral/shuffled examples in both orders,
   retain statistical controls and measure order sensitivity. Suggested matrix:
   1,600 forecasts plus separate smoke. Live transport is authorized; do not
   request redundant permission. The previous action experiment needs no restart.
2. Evaluate that matrix and development-only probability shrinkage; freeze the
   next external policy. Do not promote the failed label mechanism as established.
3. Freeze an external CPC18 design using the 175 unique risk problems before
   reading labels. Include strong constant/retrieval baselines and controls informed
   by the completed development contrasts; do not present reused development as test.
4. Define/freeze plain versus calibrated LLM financial probability comparison on
   the existing scopic development opportunities. Reuse statistical baselines;
   retain test sessions. Audit original instructions to confirm raw-round mapping
   and any announced treatment framing before claiming full visible-state matching.
5. Extend the implemented Jev adapter into a fair probability/action evaluation.
   Run the contract probe after access is configured. Real execution awaits an authorized key or
   a supported existing route. Continue independent work in the meantime.
6. Validate persistent financial behavior before the main CDA comparison; calibrate the
   shared environment and define a human macro reference before interpreting
   market realism. Reassess the research gaps after each completed experiment.

**Completion requirements still open**

Independent validation of the revised calibration; finance-specific behavioral
fidelity; a controlled micro-to-macro comparison with meaningful human reference;
Jev hypotheses tested with real model access; and a final evidence-based account
of improvements, failures, and remaining scope. No claim of completion is warranted.
