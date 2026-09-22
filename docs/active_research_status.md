# Active research objective and continuation state

Updated 2026-09-23. The user authorized continued experimentation, independent
gap analysis against the micro-to-macro research question, iterative improvements,
and a real comparison of the recently released Jev with LLM agents. This objective
is active and is not satisfied by a new runner or a single calibration experiment.

**Jev experiments resumed by explicit user instruction on 2026-09-23.** The user
provided TypeSafe access; the credential is stored only in ignored local `.env`.
The four-request real contract probe and 16-request comparison smoke both passed
with pinned `jev-1.13.0`. A 1,600-request reused-development comparison is running,
frozen at `a4500ba`; see `docs/jev_development_protocol_20260923.md`.
All 1,600 state/instruction/criteria payloads matched the historical LLM requests.
The machine needs its configured proxy (direct connection timed out); explicit
proxy support was committed at `7a5d9c8`. Earlier no-access statements below are
historical. Financial validation and market experiments have not been resumed.
The broad research objective remains incomplete.

**Authoritative progress**

- Implemented the isolated `experiments/calibration_v2.py` experiment and explicit
  `scripts/run_calibration_v2.py` stages, with config and frozen plan.
- New semantic tests cover hidden-probability invariance, target/pool label
  isolation, visible structural duplicates, shuffled-label marginal preservation,
  neutral prompts, and response identity/counterbalance checks.
- Full suite after financial training audit, quote states and statistical controls: **110 passed**
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
  Initial inspection read only README, source, CSV headers and processed metadata;
  subsequently the frozen 51 TRAIN sessions were opened as described below.
  New-market development/test behavior and skill responses remain unopened.
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
- Training audit completed successfully. Exact copies removed: 5,493 orders,
  1,849 trades, 800 person-period rows; zero conflicting identities. Unique totals:
  112,118 orders, 36,007 trades, 18,960 person-period accounts, 2,040 market-periods.
  Of the orders, 18,232 are automatic partial-fill replacements, leaving 93,886
  participant submissions. Every period cash/share/dividend/next-cash ledger checks;
  all order/trade/replacement quantity and identity links reconcile.
- Source anomalies retained: 17 self-trades across 14 sessions; 22 filled-and-cancelled
  limits exactly explain the 21 person-period available-balance discrepancies
  (7 cash, 14 shares). No cancellation time exists to identify the actual race.
  Three negative second-boundary balances occur despite nonnegative period ends.
  177 trades and 392 order records occur after nominal 120 seconds. Do not import
  source execution defects into the core or call reconstructed book states exact.
- Reproduced author training macro measures for 204 market-repetitions (paired,
  not independent): RD/RAD maximum error 6.96e-8, peak deviation 1.15e-7; average
  bias 3.05e-6 price points and exact at author's float32 precision. Two no-trade
  periods remain missing-price observations. Core inputs/results and full report:
  `docs/market_replication_training_audit_20260922.md` and
  `outputs/market_replication_training_20260922/summary.json`.
- New partial-observation quote prototype: 63,143 training states, 51 sessions,
  948 people; 38,882 limit/24,261 market submissions. Exclusions account for all
  93,886 submissions: 30,439 simultaneous seconds, 300 outside nominal time,
  3 negative reconstructed accounts, 1 unrelated trade in the same second.
  Current target fields and all future events are isolated from inputs by tests.
  Unknown book/available funds remain omitted; this is conditional BUY/SELL given
  an active order, not a HOLD rate, full information set or autonomous policy.
- Five-fold session CV entirely INSIDE TRAIN: session/person-balanced Brier
  constant .5 **.250000**, train mean **.247114**, own current-period history
  **.230750**, ridge logistic (17 features, lambda .01) **.227505**.
  These are exploratory training estimates, not financial LLM/Jev results.
  State/model/CV artifacts: `outputs/market_quote_training_20260922/`.
  State build session 78770 and CV session 46570 completed successfully. Final
  audit session 39024 completed successfully. No restart is needed.
- Jev status rechecked: direct and Gateway credentials absent, no real probe
  summary. The probe status process exited successfully without model calls.

**Backlog after the user resumes**

1. Extend the quote prototype's known private history across earlier periods and
   repetitions. Current own-history features cover only the current period. Keep
   account resets separate from memory, verify past-only invariance and preserve
   the current version-1 training CV in a separate artifact directory before changes.
2. Implement/freeze the financial comparison described in
   `docs/market_quote_probability_next.md`: proposed 25 targets/session on 16
   development sessions, four methods, two menu orders = 3,200 forecasts plus
   smoke. This is a design draft, not a frozen or started experiment. Retain
   statistical/history and neutral/shuffled-label controls; score balanced Brier.
3. Preserve all completed lottery evidence and current financial partitions.
   Scopic is a supplementary/fallback bridge with frozen statistical controls;
   no need to restart it merely because a richer market archive is now available.
   Do not map gamble probabilities directly to stock buy/sell actions. Before the
   main market comparison, build a separately tested institution/quantity/price/
   arrival/cancel controller and replace recorded human history with generated
   private history. Exact order-book replay remains unavailable.
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
