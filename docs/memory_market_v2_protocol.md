# Memory-market V2 experiment protocol

## Research assets

The primary assets are **MU** and **SNDK**. MU is the listed DRAM/HBM/NAND memory
producer used as the closest public DRAM/AI-memory proxy. SNDK is a NAND/flash
storage company and is treated as a related memory asset, not as a DRAM producer.
“DRAM” is an industry/product category rather than a US-listed ticker, so the
experiment does not invent a synthetic DRAM security.

## Liquidity correction

There is no external market maker. Each 30-agent session uses a seeded random
arrival order split into batches of five. Agents in the first batch see an empty
same-session book. Later batches see the real resting `best_bid`, `best_ask`,
`spread`, depths, and `last_trade` created by earlier batches. Orders are submitted
after each batch, so this context can create natural crossings without injecting
outside liquidity.

The prompt explains marketability and restricts limit prices to 95%–105% of the
session-opening last trade. A violation becomes a recorded hold with
`price_band_violation`; it is never silently clipped. Every order records its
decision batch, visible book, distance to the opposite quote, validation result,
and fallback state.

## Event isolation

Every retained treatment is run under three predeclared conditions:

- `no_event`: neither event is visible;
- `e1_only`: only the July 1 event, first tradable on July 2;
- `e1_e2`: both July 1 and July 9 events, first tradable on July 2 and July 10.

Results are grouped by asset and `event_condition`. The old Boolean secondary-event
field is retained only for backward compatibility and must not be used to combine
`no_event` with `e1_only`.

## Herding manipulation check

The primary manipulation check is
`P(action = lagged majority action | a lagged buy/sell majority exists)`. Holds
remain in the denominator and therefore count as not following. The report also
shows the conditional rate among active buy/sell orders. The lagged majority is
computed from the prior session using a 0.10 signed-imbalance threshold, preventing
same-decision leakage. The expected ordering is high-herding above low-herding.
Switch rate remains a separate strategy-review outcome; it is not the herding
manipulation check.

## Predeclared liquidity gate

Before expanding the real API matrix, run seed 42, `e1_only`, endogenous mode, and
low/high herding for MU and SNDK. An asset/treatment passes when all are true:

- at least 6 of 23 sessions contain both buy and sell orders;
- at least 20 trades occur in the run;
- invalid/rejected orders plus API fallbacks stay below 5%.

The screen is descriptive and cannot establish the herding hypothesis. If the
liquidity gate passes, expand the same design to `no_event` and `e1_e2`, then add
seeds 43 and 44. If it fails, inspect `liquidity_diagnostics.csv` before changing
the mechanism; do not add a strong market maker merely to manufacture trades.

## Commands

```bash
python scripts/run_meta_compute_suite.py \
  --config config/memory_market_2026.json --provider aigc_relay \
  --modes endogenous --symbols MU SNDK --event-conditions e1_only \
  --treatments low_herding high_herding --seeds 42 --shocks 0 \
  --parallel-runs 4 --max-rps 4

python scripts/diagnose_liquidity.py \
  outputs/meta_compute_2026/aigc_relay/memory_market_v2
```
