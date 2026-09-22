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
- Full suite after the Jev adapter: 74 passed in the project `.venv`; new-file Ruff checks passed.
- The separate mock provider completed its 20-response smoke, 10,000-response
  development matrix, and evaluation. This is engineering evidence only.
- Protocol/code/config/plan committed as `26893b7` before real calls. Real smoke
  completed 20/20 valid responses. The 10,000-call real development job is running:
  Python PID **382694**, exec session **43945**. At the latest inspected record
  count it had 3,040 valid responses and zero invalid responses. This is a partial
  execution count, not a behavioral result or completion claim.
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
- Downloaded and checksum-verified CPC18 raw records and dictionary. Feature-only
  audit of the 210-problem workbook yields 177 non-ambiguous candidates after
  removing five structural overlaps against the full Choices13k source. No external
  human choice rates were used for selection. See `docs/external_data_screen_20260922.md`.
- Downloaded two financial human datasets: a 21-session single-asset market workbook
  (25,042 event rows) and a panel with risk elicitation and six-asset trading fields.
  Metadata and source manifests are archived; decision-time reconstruction and
  group splits are not yet implemented.
- The financial herding replication package has a verified public file listing;
  a local direct page fetch returned HTTP 403. Raw individual data not obtained.

**Next actions**

1. Poll exec session 43945 / inspect PID 382694 and the log before taking action on
   calibration-v2. If terminal, inspect the exit status and exact matrix count;
   resume missing identities only when stopped. Do not start a second live job.
2. Evaluate the completed matrix, inspect true/shuffled/neutral contrasts and
   statistical baselines, then freeze the next externally evaluated policy.
3. Continue auditing the acquired CPC18 and financial datasets; reconstruct human
   decision-time information and split by independent source/group as appropriate.
4. Extend the implemented Jev adapter into a fair probability/action evaluation.
   Run the contract probe after access is configured. Real execution awaits an authorized key or
   a supported existing route. Continue independent work in the meantime.
5. Validate financial behavior before the main CDA comparison; calibrate the
   shared environment and define a human macro reference before interpreting
   market realism. Reassess the research gaps after each completed experiment.

**Completion requirements still open**

Independent validation of the revised calibration; finance-specific behavioral
fidelity; a controlled micro-to-macro comparison with meaningful human reference;
Jev hypotheses tested with real model access; and a final evidence-based account
of improvements, failures, and remaining scope. No claim of completion is warranted.
