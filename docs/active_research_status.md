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
- Full suite after CPC18 and new-market metadata partitions: **102 passed**
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
- CPC18 study **complete**, frozen at `0d41d6c`: smoke 16/16, main 1,400/1,400
  valid; 175 common cases, 686 people, 84,860 no-feedback trials, 24 strata.
  One format repair, no failed HTTP attempts/cache hits. Successful main usage:
  769,596 input / 40,664 output tokens, including repair. Real run/evaluation
  sessions 76439/34022 closed successfully; do not restart completed calls.
  Human labels opened only after all forecasts and hashes, at 09:32:49 UTC.
- External MAE: P0 **.192307**, selected P_true **.173899**, neutral **.179818**,
  shuffled **.176063**, logistic **.113514**, aligned kNN **.119651**, constant
  **.207766**. True−P0 joint CI [-.033610,-.003089]; true−logistic
  [+.040580,+.077892]. Both correct-label contrasts cross zero under the primary
  joint bootstrap: mechanism gate fails. No CPC18 tuning or mixture refit.
  Raw-response reconstruction independently reproduced MAE/RMSE/corrected JS.
  See `docs/cpc18_external_results_20260922.md` and
  `outputs/human_benchmark/cpc18_external/aigc_relay/test/evaluation/summary.json`.
- Located/downloaded a stronger human-market candidate via the authors' OSF
  project `https://osf.io/5ypku/`: original software/instructions plus 50,478,080-byte
  reproduction kit, SHA256 matching OSF
  `642898545b74210f1d999eac2c2b501ae3f08074ec45ab61bcb19081dcfff6ae`.
  Archives and source manifest are in `data/raw/finance_replication_20260922/`.
  Only README, source code, CSV headers and explicit processed-metadata columns
  read; no order/trade/account/price/skill outcomes or macro charts opened.
- Author eligibility metadata yields 83 sessions, 166 markets, 1,544 people.
  New plan committed as **041a0ac** before behavioral reads:
  `config/frozen/market_replication_partitions_20260922.json`:
  51 train /16 development /16 test sessions, 102/32/32 markets, 948/300/296 people.
  All paired conditions and repetitions remain in the same session split.
  No change to previous financial partitions. Source/software caveats and the
  author's eligibility filtering are recorded in `docs/market_replication_data_protocol.md`.
  Order/Trade have integer-second creation times; cancellation has no timestamp.
  Browser prohibits crossing limit quotes; explicit market orders execute trades.
  Exact historical book replay remains unproven, and core institution differs.

**Next actions**

1. Audit ONLY the new replication archive's frozen 51 training sessions. Implement
   a source-checked filtered loader; resolve cumulative export duplicates via UUID;
   reconstruct trade/dividend ledgers and period boundaries; quantify cancellation
   and same-second ambiguities. Original code inspected at
   `/tmp/fyp_replication_20260922/`, but authoritative bytes are the archived ZIP.
   Do not execute the external package or inspect held-out outcomes. Decide which
   micro targets and human macro references are identifiable before new model calls.
2. Preserve CPC18 as completed external evidence. Stop lottery prompt expansion;
   numerical risk choices currently favor statistical policies. Do not assume an
   arbitrary mapping from gamble probabilities to stock buy/sell actions.
3. Define/freeze plain versus calibrated LLM financial probability comparison on
   the existing scopic development opportunities. Reuse statistical baselines;
   retain test sessions. Include confirmed announced treatment framing and document
   remaining within-period funding/settlement ambiguity. A sell-any predictor is
   still not a coherent joint portfolio agent. First use the new training audit to
   determine whether an institution-matched target can avoid a cross-dataset bridge.
4. Extend the implemented Jev adapter into a fair probability/action evaluation.
   Run the contract probe after access is configured. Real execution awaits an authorized key or
   a supported existing route. Continue independent work in the meantime.
5. Validate persistent financial behavior before the main CDA comparison; calibrate the
   shared environment and define a human macro reference before interpreting
   market realism. Reassess the research gaps after each completed experiment.

**Completion requirements still open**

CPC18 provides limited external probability evidence and a failed label-mechanism
gate; it does not close the remaining requirements: finance-specific behavioral
fidelity, coherent persistent policies, a controlled micro-to-macro comparison;
Jev hypotheses tested with real model access; and a final evidence-based account
of improvements, failures, and remaining scope. No claim of completion is warranted.
