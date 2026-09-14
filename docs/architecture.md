# Architecture (MVP-0)

```text
scripted independent agents
    -> Exchange.submit_order (resource checks)
    -> MatchingEngine + OrderBook (price/time priority)
    -> Trade (resting price)
    -> Portfolio settlement
    -> market price, volume, CSV recorder
```

`Exchange` owns order IDs, logical timestamps, day boundaries, portfolios and transaction
records. `OrderBook` stores only open quantities. `MatchingEngine` matches an incoming order
against the best opposite resting order and calls the exchange's settlement callback before
marking the fill. The exchange validates the full limit notional or share quantity against
uncommitted balances; resting orders reserve only their remaining quantities. This permits
partial fills and price improvement without margin or short positions.

The order submission sequence is the deterministic time priority. No live clock, external
data or model output enters the matching decision. A day ends by expiring all remaining
orders. The last transaction price persists if no trade occurs.

Next boundaries follow the TDD: agent/persona and counterfactual state, historical replay,
validated LLM decisions, lagged population mean field, then endogenous experiments.
