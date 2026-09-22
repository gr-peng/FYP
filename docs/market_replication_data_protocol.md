# Human market replication archive: partition and access protocol

This protocol freezes data eligibility and session partitions before we read any
order, trade, price, account, skill response or macro result in this new archive.
It does not freeze a behavioral model or establish exact event replay.

## Source and initial inspection

The [authors' materials link](https://chr-huber.com/research.html) leads to
[the OSF project](https://osf.io/5ypku/). Its
[data component](https://osf.io/kw8mn/) provides `_reproduction_kit.zip` and the
[software component](https://osf.io/bm2dx/) provides the original oTree software
and instructions. Downloaded originals are retained, never executed in place.
Reproduction archive SHA256 is
`642898545b74210f1d999eac2c2b501ae3f08074ec45ab61bcb19081dcfff6ae`, independently
matching the OSF checksum (MD5 also verified). A source manifest records downloads.

Initial reads were limited to the README, archive paths, CSV headers, original
software and explicit metadata columns of author-produced `_sessions.dta`,
`_markets.dta` and `_participants.dta`. This yields 83 sessions, 166 markets,
1,544 anonymous participants and no participant appearing in multiple sessions.
The 31 study-1 sessions contain 31 calm and 31 excitement markets; the 52 study-2
sessions contain 52 markets in each self-control condition.

These are the authors' processed eligibility memberships. Their cleaning code
uses completed market payment presence to omit empty sessions, removes duplicate
exports, and notes one participant's illness. We adopt that documented population
without inspecting payments or selecting it on our own model results. Do not claim
we independently chose the population from raw pre-experiment metadata alone.

The software shows limit-order submission, explicit market-order execution,
reserved cash/assets, partial fills via replacement orders, and cancellation.
The browser prohibits crossing limit quotes and self-trades. This differs from
our core's automatic matching of marketable limit orders. A faithful comparison
needs a separately tested institution adapter, preserving the existing core.

Order and trade exports contain UUIDs and integer-second creation times.
Cancellation stores `deleted` but no deletion timestamp. Same-second event order,
cancel timing and exported row order therefore need careful audit; never use
eventual `filled`/`deleted` flags as pre-decision information or infer a perfect
historical book from final order records. None of these behaviors has yet been
measured in the human rows. Metadata richness alone does not establish replay.

## Whole-session split

Stratify by study and sort session codes by SHA256 of
`20260926:replication-session:<session_code>`. Within study 1 assign 19/6/6 sessions
to train/development/test; within study 2 assign 32/10/10. Total: **51/16/16**.
Both market conditions, all traders and all market repetitions of a session stay
together. Do not treat the two repetitions as independent markets or split a
participant's repeated decisions across partitions. Verify exact coverage,
disjointness and metadata relations before writing the frozen JSON.

Hash the source archive, three metadata members, software, instructions, protocol
and executable. Commit the plan before reading training behavior. The loader for
subsequent audits must filter by frozen session membership before analyzing rows;
test access needs a later, explicitly frozen policy/evaluation protocol. Reading
headers and metadata is allowed; outcome-based exclusions are not silently added.

## First training audit and go/no-go criteria

1. Resolve repeated cumulative exports using UUID and source provenance. Check
   market membership and repetitions against the frozen metadata. Preserve an
   audit of conflicting records rather than accepting the last file silently.
2. Reconstruct trades, dividends and period-boundary cash/share ledgers. Check
   conservation and original starting endowments against the actual software.
   Do not expose a current-period dividend before its scheduled revelation.
3. Quantify same-second events, cancellation prevalence and unresolved book states
   on training sessions only. Establish which conditional decision targets are
   identifiable. Activity-only data cannot supply an unconditional HOLD rate.
4. Reproduce the authors' training-scope price/volume and mispricing definitions;
   keep price weighting, empty periods and repeated markets explicit. Freeze the
   human macro reference before policy selection on test sessions.
5. If cancellation/timing prevents exact state reconstruction, report that limit.
   Restrict the micro target or use this archive as a macro reference with an
   explicit cross-dataset financial bridge; do not label either as exact replay.

No new simulator/model call follows from this partition freeze alone. Existing
scopic and 2017 market partitions remain unchanged, with their tests unopened.
No source data, demographic records or third-party software are added to Git.
