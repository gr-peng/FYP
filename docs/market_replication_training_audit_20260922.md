# Human market training audit and financial observation prototype

The new archive supports exact period-boundary accounting and reproduction of
the authors' main training-scope mispricing statistics. It does **not** support
an exact historical order book: cancellation times are missing and many events
share a recorded second. We implemented a train-only loader and a conservative
partial-observation financial target, retaining the source anomalies explicitly.

## Access and source boundaries

Whole-session partitions were committed as `041a0ac` before training access.
Only the 51 training sessions (102 markets, 948 people) were opened. Both market
repetitions stay together; 204 market-repetitions are not 204 independent groups.
The new archive's 16 development and 16 test sessions remain unopened for analysis.
Previous scopic/2017 test partitions are unchanged and unopened.

The loader checks the original ZIP, frozen metadata/code hashes and committed
partition identity. It filters raw CSV rows by training session before reading
behavioral fields, uses explicit column allowlists and rejects development/test
overrides. The author's Stata outcome file is decoded in chunks; only training
rows are retained and analyzed. Source bytes passing through a parser are not
claimed to be unread bytes. Every raw access records the scope and hashes.

Primary sources are the [authors' OSF data](https://osf.io/kw8mn/) and
[original software/instructions](https://osf.io/bm2dx/). The archived software was
inspected as source, not executed. It confirms 10 periods per repetition, 120
seconds per period, 3,000 cash/20 shares or 1,000 cash/60 shares, and end-of-period
dividends of 0 or 10. The screen states equal probabilities and shows remaining
expected dividends `5*(11-period)`. The server shuffles five high and five low
dividends; that hidden sequence constraint and future dividends must not enter
agent inputs. Repetition 2 resets accounts, while the people remain the same.

## Duplicate exports and accounting

| Table | Selected exported rows | Unique semantic records | Exact copies removed |
|---|---:|---:|---:|
| Orders | 117,611 | 112,118 | 5,493 |
| Trades | 37,856 | 36,007 | 1,849 |
| Person-period accounts | 19,760 | 18,960 | 800 |

There were no conflicting records for the same identity. Provenance retains
all source export members. The order table contains 18,232 automatic replacement
records after partial fills, leaving **93,886 participant submissions**. Automatic
replacements must not be treated as new human choices.

All 18,960 period account records satisfy cash/share flow, dividend payment and
next-period cash equations exactly. All 1,896 initial person-repetition accounts
reconstruct to a permitted endowment: 948 of each type. The full set of 2,040
market-periods conserves cash within trading and conserves shares. All 36,007
trades link to the correct orders/market/period and exactly one market order;
prices, quantities and all 18,232 remainder-order identities/quantities reconcile.
These checks concern recorded period accounts, not every instant of execution.

## Source anomalies and unavailable information

- **14,254 cancelled limit records** have final cancellation flags but no cancel
  timestamp. A final flag is not a public observation at an earlier decision.
- **30,476 of 93,886 participant submissions** fall in a second containing more
  than one submission. There are 5,406 trades in seconds with multiple trades.
  UUID sorting is not a valid substitute for missing event order.
- **17 self-trades** occur across 14 training sessions. The browser checks against
  self-trading, but the inspected server execution function has no equivalent
  check. A stale browser/server race is possible; it is not identified by these data.
- **22 limit orders are both filled and cancelled** (8 bids, 14 asks). The final
  available-cash ledger differs in 7 person-period records and available shares
  in 14, affecting 21 records total. Summing the nominal resources of these
  cancelled-and-filled orders explains **every difference exactly**. This is
  consistent with extra reservation release; cancellation chronology is unobserved.
- Aggregating trades at recorded second boundaries produces one negative-cash
  and two negative-share boundaries among 70,472 person-second boundaries, despite
  nonnegative period-end accounts. This is not proof of feasibility at every instant.
- 392 order records and 177 trades occur after the nominal 120 seconds. Trade
  times range from 3 to 123 seconds; order times from 2 to 126 seconds. These
  recorded actions remain in the source accounting and author-matching macro audit.

No source rows were silently repaired or removed from the primary accounting or
macro reproduction. Our existing engine's self-trade and resource constraints
remain intact. A future institution adapter should reproduce the intended action
rules while disclosing discrepancies between the interface and recorded execution.

## Macro reference reproduced on training only

Each market-period's **unweighted** trade-price mean matches the raw interface
summary. For the authors' macro measures we separately compute **volume-weighted**
prices. RD and RAD divide price deviation from displayed fundamental value by
27.5. Two of the 2,040 periods have no trades; their price-based measures remain
missing, rather than becoming zero deviation or an invented carry-forward price.

The 204 market-repetition outcomes match the author's processed results within
floating-point storage differences: maximum absolute differences are 6.96e-8 for
RD/RAD, 1.15e-7 for peak relative deviation, and 3.05e-6 price points for average
bias. All 204 average-bias and peak-deviation values match exactly after conversion
to the author's float32 storage precision. We did not round data to force equality.

Training mean RAD is .731690/.341939 in study 1 for repetitions 1/2, and
.673182/.435593 in study 2. These are descriptive training summaries, not our own
independent confirmation of an experience effect or a simulator result.

## Conditional quote-state prototype

`experiments/replication_quote_states.py` creates **63,143** training observations
from all 51 sessions and 948 participants: 38,882 limit and 24,261 market orders.
The target is BUY versus SELL **conditional on an active submitted order**.
It does not supply an unconditional action probability or a HOLD label.

Inputs contain reconstructed total cash/shares, displayed fundamental value,
elapsed time, repetition, earlier-period public price/dividend summaries,
strictly earlier current-period trade prices, the original pre-market manipulation,
and the person's strictly earlier current-period buy/sell submission counts.
Source identities are kept in provenance. Current side, kind, price and quantity
are separated targets. Book depth, best quotes, outstanding orders and available
resources are explicitly unavailable. Same-second fills caused by the current
submission do not enter its input balances.

Eligibility accounts for every original participant submission, in fixed rule
order: 30,439 excluded for simultaneous submissions, 300 outside nominal time,
3 for negative reconstructed balances and 1 for an unrelated trade in the same
second. These counts differ from the unconditional timing totals above because
the exclusion categories are mutually exclusive. Selection limits generalization
to burst activity; it is not a real-time rule that could know future arrivals.

Counterfactual tests change current target side/price/quantity, future trades,
future dividends and period-end balances without changing the input state.
Separate tests reject unresolved simultaneous events and confirm scoring weights.
This is a **partial-observation, recorded-history** task, not complete human
information or an autonomous policy. The current prototype summarizes own history
within the current period; cross-period/repetition private history needs an explicit
past-only extension before the forthcoming LLM comparison is frozen.

## Exploratory statistical controls inside training

Five study-stratified session folds stay entirely inside the 51 training sessions.
All records from a session share a fold. Models normalize features and fit only
the other folds. Ridge is fixed at .01; no hyperparameter grid was searched.
Scores first weight sessions equally, then people equally within session, then
their eligible submissions equally. Repeated high-frequency actors do not dominate.

| Control | Out-of-fold Brier ↓ | Out-of-fold log loss ↓ |
|---|---:|---:|
| Constant .5 | .250000 | .693147 |
| Training mean | .247114 | .687365 |
| Own current-period history, four-observation prior | .230750 | .652395 |
| Ridge logistic, 17 visible-state features | .227505 | .645956 |

These are training CV results. They do not constitute an independent financial
LLM/Jev comparison. The small increment over the history baseline motivates
keeping a strong history control and explicitly testing persistent feedback in
rollout; it does not prove that a statistical agent already reproduces the market.

## Reproduction and next decision

```bash
.venv/bin/python scripts/audit_market_replication_training.py
.venv/bin/python scripts/build_market_quote_training.py
.venv/bin/python scripts/evaluate_market_quote_training_baselines.py
```

Outputs are in `outputs/market_replication_training_20260922/` and
`outputs/market_quote_training_20260922/`. They retain source/code hashes, complete
account and period tables, eligibility, separated states/targets, CV fold assignment
and the fitted training checkpoint. Source-code drift during a run rejects results.

Next: extend known private history across periods without using future actions;
then freeze a bounded financial LLM probability comparison on the still-closed
development sessions. Preserve all statistical controls and the source-quality
limits. Exact book replay remains unavailable; quantity, pricing, arrival/cancel
policies and a separately verified market institution are still required before
the main micro-to-macro experiment. Jev access remains absent; no real Jev result
can be inferred from the LLM adapter or contract tests.

Full project tests: **110 passed**; changed-file Ruff passed. Audit, state building
and training CV processes all exited successfully, and their recorded source/data
hashes were verified. The user requested stopping after this iteration; the next
comparison remains unstarted until work is resumed.
