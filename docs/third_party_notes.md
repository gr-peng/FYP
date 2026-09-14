# Reference review

Reference revisions are the shallow-clone HEADs in `third_party/`; run
`git -C third_party/NAME rev-parse HEAD` to record them for a study run.
These directories were not modified by this project.

| Repository | Cloned revision (short SHA) |
| --- | --- |
| TwinMarket | `de5f2446fcba` |
| Mean-Field-LLM | `d23df0155330` |
| ACL24-EconAgent | `bfada091eaa1` |
| LAMP | `16fda6db84b0` |
| StockAgent | `e2a9c052b816` |

| Reference | Inspected / intended role | Our implementation choice |
| --- | --- | --- |
| TwinMarket `trader/matching_engine.py` | Its `Order` model, price/time sorting, transaction record concepts and daily processing | Own `market/order.py`, `order_book.py`, `matching_engine.py`, `trade.py`, and `exchange.py`; removed multi-stock processing, random timestamps, batch clearing-price selection and A-share price limits. |
| TwinMarket `trader/trading_agent.py`, `simulation.py` | Agent action and orchestration boundaries | Future agent layer; MVP-0 uses independent scripted proposals. |
| Mean-Field-LLM | Population-signal design reference | Future deterministic lagged aggregation; no model-written aggregate in MVP. |
| ACL24-EconAgent | Recent-experience and reflection pattern | Future bounded memory layer. |
| LAMP | Long-term retrieval pattern | Deferred beyond MVP. |
| StockAgent | Run loop and structured logging pattern | Separate scripted loop and CSV/JSON records; its matching logic is not used. |

No source file from these repositories is copied into our package. The behavioral switching
paper has no verified implementation repository in the TDD, so its behavior layer will be
implemented independently in a later phase.
