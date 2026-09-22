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
- Full suite: 64 passed in the project `.venv`; new-file Ruff checks passed.
- The separate mock provider completed its 20-response smoke, 10,000-response
  development matrix, and evaluation. This is engineering evidence only.
- AIGC model registry: HTTP 200, configured DeepSeek-V4-Flash available, no Jev
  entry. TypeSafe/Vercel/OpenRouter credentials absent at check time.
- Archived official Jev docs; application hypotheses and comparison boundaries
  are in `docs/jev_agent_protocol_draft.md`. No Jev empirical results yet.
- The financial herding replication package has a verified public file listing;
  a local direct page fetch returned HTTP 403. Raw individual data not obtained.

**Next actions**

1. Commit protocol, code, config, and plan; run real smoke, then the real 10,000-call
   calibration-v2 development matrix. Preserve process handle/PID and monitor the
   actual process before resuming or restarting. No duplicate job launches.
2. Evaluate the completed matrix, inspect true/shuffled/neutral contrasts and
   statistical baselines, then freeze the next externally evaluated policy.
3. Obtain and audit CPC18 and a human financial dataset; reconstruct human
   decision-time information and split by independent source/group as appropriate.
4. Implement Jev access adapter and fair probability/action evaluation after
   verifying the official interface. Real execution awaits an authorized key or
   a supported existing route. Continue independent work in the meantime.
5. Validate financial behavior before the main CDA comparison; calibrate the
   shared environment and define a human macro reference before interpreting
   market realism. Reassess the research gaps after each completed experiment.

**Completion requirements still open**

Independent validation of the revised calibration; finance-specific behavioral
fidelity; a controlled micro-to-macro comparison with meaningful human reference;
Jev hypotheses tested with real model access; and a final evidence-based account
of improvements, failures, and remaining scope. No claim of completion is warranted.
