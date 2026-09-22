# Training-only financial data audit

Partitions were committed at `0ecd73f` before financial behavior was inspected.
`scripts/audit_financial_training.py` reproduces the findings and writes its
machine-readable report under `outputs/financial_training_audit_20260922/`.
Development and test behavior remain unopened.

## Scopic panel: a usable restricted micro task

The two training sessions contain 30 people and 510 rows. All share transitions
conserve quantity; the reconstructed cash ledger remains nonnegative (minimum 14).
All 30 participants start raw round 4 with zero shares. Raw rounds 4–17 therefore
provide a 14-period sequence with a fresh 10,000 balance. Interpreting the first
three rounds as the practice phase is supported by that reset, but the public
data description does not explicitly label these rows. This mapping is provisional
until the original instruction/code timing is independently confirmed.

For all 390 nonterminal main-period records, the following holds exactly:

```
cash_after[t] = cash_before[t] + sum((sell[t] - buy[t]) * price[t])
result_final[t] = cash_after[t] + sum(holdings_after[t] * price[t + 1])
```

Only 20 of those rows match valuation at the current prices. `result_final` thus
contains information about the next price realization and must never be a current
decision input. The terminal next-price vector is absent; do not solve for it
using cross-participant wealth outcomes and pass that into a model. Only five of
30 terminal wealth values match current-price marking, so terminal accounting is
not fully identified by the archive.

`experiments/scopic_states.py` constructs main periods 1–5 only, before the first
peer ranking. It exposes current cash, holdings, prices, visible price history,
past main-task trades and an explicitly calculated weighted-average cost basis.
Cost basis is an analyst-derived past-only feature, not a claim about the original
interface or the authors' realization-accounting convention. Later periods need
the actual ranking display, including its profitable-trade definition.

This produces 150 six-asset decisions and 433 owned-asset opportunities, including
101 decisions to sell some of that asset. These are training observations only.
Conditional sell-any prediction can test a finance-specific behavior. It does
not supply a joint six-asset policy, identify risk traits, or establish endogenous
market realism. Participant identities and current targets are separate from state;
future prices, current trades, post-trade holdings, final wealth, risk answers and
later ranking fields cannot enter model inputs. Tests poison these fields and
verify that eligible earlier states are unchanged.

## Single-asset markets: quotation archive, not complete event replay

The 11 training markets contain 13,619 quote rows and 2,133 accepted quotes.
Quote and trade IDs are separately unique within session/round. Quote rows follow
contract order, but in 88 of 128 rounds containing trades their trade IDs do not
follow that order. Every filled quote's `Market Price` equals its eventual execution
price. Accepted status, accepter identity and trade ID are annotations on quote
rows, not demonstrated contemporaneous inputs at quote submission.

Consequences for our adapter:

- A table row is not one immediate submission-and-execution event. Replaying each
  filled row as an immediate trade would change inventory and public information.
- Separate quote and trade order does not identify their joint interleaving.
  Unknown dividends further prevent exact cash reconstruction after a period.
- Macro prices and completed-trade counts can still be summarized after separating
  executions. Pre-quote order books, feasible opportunities and HOLD rates cannot
  be claimed reconstructed from this table alone.
- A simulator with explicit acceptance can reproduce the specified institution,
  but must label any timing/arrival assumptions and cannot claim a full human replay.

## Research decision

Prioritize the restricted scopic liquidation task for the next micro benchmark,
with a constant-probability and a regularized past-information statistical baseline.
Freeze metrics and models before inspecting development behavior, then reserve test
sessions until final policies are frozen. Retain the market archive for human macro
reference and quote-distribution analyses, while continuing to seek a richer event
log if exact micro-to-macro state matching is required.

Our central question still requires three distinct steps: better conditional human
decisions, a viable persistent financial policy, and improved independently held-out
market properties under the same institution. The current data audit completes none
of those empirical claims; it identifies a valid starting point and prevents two
forms of future-information leakage.
