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
- Full suite after CPC18 implementation: **99 passed**
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
  normalization, one schema repair and complete per-call usage. Do not use the
  shared trading-specific `decide()` fallback for this task.
- Probability experiment frozen at **660f9c0**, real smoke 16/16 and main 1,600/1,600
  valid, no repairs/cache hits. Run/evaluation exited 0; sessions 97360/8888 closed.
  P_true MAE .163021 versus P0 .179454 and constant .181091 (200 reused dev items).
  True−P0 CI [-.031692,-.001773], but true−neutral and true−shuffled cross zero:
  correct-label mechanism still unsupported. Selected candidate **P_true, alpha=1**.
  Main successful usage: 837,866 input / 46,392 output tokens.
- JS erratum: the frozen probability evaluator reports a sum as `mean_js` by calling
  a scalar helper with arrays. Original source/output preserved. Run
  `scripts/correct_probability_js.py`; authoritative JS is in `evaluation_corrected/`.
  New studies must use `experiments.probability_analysis.mean_bernoulli_js`. Primary
  MAE/selection unchanged; independently recalculated from raw responses.
- Added stronger risk-only statistical controls, fitted on 487 pool items and
  evaluated on 161 reused development items. Zero shared IDs/visible structures.
  MAE: symmetric fractional logistic **.086638**, aligned 16-neighbor **.100011**,
  P_true **.158909**, constant **.174865**. Fixed ridge .01/k16, no hyperparameter
  search. These are post-probability-study exploratory controls, not preregistered.
- Simple post hoc statistical/LLM mixture selects **zero LLM weight** on development;
  weights 0/.25/.5/.75/1 give MAE .086638/.095707/.113204/.134790/.158909.
  Do not present this as a successful hybrid or added LLM value. Reproducible with
  `scripts/evaluate_risk_statistical_baselines.py`; model/results under
  `outputs/risk_statistical_baselines_20260922/`.
  See `docs/probability_calibration_results_20260922.md`.
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
- Original author/registered instructions obtained and archived: three practice
  periods/reset confirmed; announced ranking rules belong in even pre-ranking
  context. Four train and four dev decisions use sales proceeds to fund purchases
  exceeding start-of-period cash. Do not impose a stricter gross-purchase cap or
  remove these records silently. Original source OSF node returns 410; registration
  preserves instructions but no task code found. See `financial_instructions_audit_20260922.md`.
- The financial herding replication package has a verified public file listing;
  a local direct page fetch returned HTTP 403. Raw individual data not obtained.

**Next actions**

1. CPC18 runner, explicit outcome gate and participant cluster bootstrap are
   implemented. All 99 tests and new-file Ruff pass; mock smoke 16/main 1,400 plus
   synthetic-only evaluation pass without human outcome access. Freeze before
   real smoke/main calls. See `docs/cpc18_external_protocol.md`. Use
   175 unique risk problems, four probability methods × two orders = 1,400 main
   forecasts plus smoke. Preserve selected P_true/alpha1 and include BOTH strong
   statistical controls fitted on the 487 pool cases. Keep their fitted checkpoint
   fixed; do not tune on CPC18. Live existing-route calls are already authorized.
2. Execute the frozen 1,400 main forecasts after the 16-call smoke; then score raw
   no-feedback trials 1–5 only. The implemented protocol records response/source
   hashes before label opening, and reports problem-only, within-stratum participant
   cluster and joint bootstrap intervals. Joint uncertainty is primary. Use corrected JS.
3. Reassess external results before choosing any calibration/policy intervention.
   Preserve negative mixture and mechanism findings. No current real job is running;
   completed Choices13k runs need no restart. Do not modify their frozen modules
   merely to add a new study; use separate experiment code and hashes.
4. Define/freeze plain versus calibrated LLM financial probability comparison on
   the existing scopic development opportunities. Reuse statistical baselines;
   retain test sessions. Include confirmed announced treatment framing and document
   remaining within-period funding/settlement ambiguity. A sell-any predictor is
   still not a coherent joint portfolio agent.
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
