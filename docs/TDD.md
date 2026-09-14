# Technical Design Document
## Behaviorally Aligned Adaptive LLM Agents for Endogenous Financial Market Simulation

**Document purpose:**  
This document is the implementation specification for Codex / software agents that will build the research prototype described in the project proposal.

**Primary research goal:**  
Build a reproducible LLM-driven agent-based financial market simulator that connects:

**micro-level behavioral fidelity → adaptive investor decisions → endogenous order interaction → macro-level market dynamics**

The system is **not a stock prediction system**. Its primary research use is **counterfactual behavioral stress testing** and **micro-to-macro validation**.

---

# 1. Project Background

Existing LLM financial simulation research has established several pieces of the problem:

- LLMs can act as economic agents.
- LLM agents can submit trading orders and participate in simulated markets.
- Multi-agent interactions can generate endogenous market phenomena.
- Social interaction can influence collective market dynamics.
- However, recent work shows that out-of-the-box LLM traders may be too rational and may fail to reproduce human behavioral biases consistently.

The central research gap of this project is therefore:

> If we improve the behavioral fidelity of adaptive LLM investors, does the endogenous market they collectively generate become more realistic?

The intended contribution is **not another larger financial simulator**. The contribution is a behavioral alignment and validation pipeline that can be inserted into an endogenous market simulator.

The system should support two experimental stages:

1. **Stage 1 — Behavioral Validation**
   - historical/exogenous market path
   - isolate individual behavioral response
   - measure strategy switching under controlled behavioral drivers
   - calibrate LLM behavior

2. **Stage 2 — Endogenous Market Simulation**
   - agents submit orders
   - exchange matches orders
   - prices emerge from transactions
   - behavior affects price and price affects future behavior
   - evaluate whether better micro-level behavioral fidelity improves macro-level market realism

---

# 2. Core Research Questions

## RQ1 — Micro Behavioral Fidelity

Do LLM investors exhibit theory-consistent strategy switching under:

- loss aversion
- herding
- relative-performance / wealth differentiation
- price misalignment

The key action is switching between:

- **Fundamental**
- **Technical**

The system should measure not only whether agents switch, but whether the **direction, magnitude, and statistical stability** of behavioral effects match behavioral-finance theory or human benchmarks.

## RQ2 — Micro-to-Macro Link

When behaviorally calibrated agents interact through an endogenous market, does improved micro-level fidelity generate more realistic macro-level financial dynamics?

Relevant macro phenomena:

- bubbles / price deviation from fundamental value
- volatility
- fat-tailed returns
- volatility clustering
- turnover
- liquidity
- crash magnitude
- maximum drawdown
- recovery time
- order execution rate

## RQ3 — Population Feedback

How does the current population state affect individual strategy switching?

Example population signals:

- Fundamental trader share
- Technical trader share
- population sentiment
- buy/sell order imbalance
- average PnL

The mechanism is:

```text
individual decisions
        ↓
population distribution
        ↓
mean-field signal
        ↓
individual decisions
        ↓
...
```

The mean-field layer should avoid unnecessary O(N²) pairwise agent communication.

---

# 3. Research Scope

The first working system MUST deliberately remain narrow.

## MVP scope

- one risky asset
- cash
- heterogeneous retail investor agents
- Fundamental / Technical style switching
- continuous double auction
- limit orders
- buy / sell / hold
- 20–50 agents initially
- configurable to 100 agents
- 30 endogenous trading periods initially
- 10-day strategy reassessment cycle for Stage 1
- deterministic accounting outside the LLM

## Explicitly out of scope for MVP

- multi-asset portfolio optimization
- options/futures/derivatives
- short selling unless later required
- margin / leverage
- loans
- bankruptcy mechanics
- full social-media network
- pairwise natural-language chat among every agent
- paid full-depth real-world LOB replay
- high-frequency trading
- market prediction
- RL policy optimization
- local fine-tuning / LoRA before the prompt-based prototype is stable

These can become future extensions.

---

# 4. Critical Design Principle

## LLMs decide behavior. Code maintains reality.

The LLM MUST NOT be trusted to maintain:

- cash balance
- positions
- average cost
- realized PnL
- unrealized PnL
- transaction prices
- market price
- order-book state
- counterfactual returns
- mean-field statistics
- portfolio constraints

These values must be calculated deterministically in Python.

The LLM is only responsible for behavioral decisions such as:

- interpreting observations
- reasoning about the market
- choosing Fundamental vs Technical
- deciding stay/switch
- buy/sell/hold
- limit price
- order quantity
- optional short rationale

This separation is essential for reproducibility and debugging.

---

# 5. Multi-Agent Interpretation

The architecture is **not**:

```text
one central LLM
    ↓
centrally controls all traders
```

Instead:

```text
Shared LLM backend/API
        ↓
Agent 1 independent context/state
Agent 2 independent context/state
Agent 3 independent context/state
...
Agent N independent context/state
```

Agents may use the same model checkpoint/API model, but each agent must have an independent:

- persona
- state
- wealth
- position
- memory
- current style
- counterfactual ledger
- random seed / decision history

This is equivalent to many independent simulated investors sharing a common cognitive backbone.

---

# 6. Reference Projects and Reuse Strategy

The project should reuse mature infrastructure where possible, but should **not blindly merge multiple external repositories into one codebase**.

Reference repositories should be cloned under:

```text
third_party/
```

and treated as implementation references.

## 6.1 TwinMarket — Primary Market Infrastructure Reference

Paper role:

- financial market simulation
- BDI-style agents
- order-driven trading
- scalable multi-agent simulation
- social interaction
- price formation

### Official repository

```bash
git clone https://github.com/FreedomIntelligence/TwinMarket.git third_party/TwinMarket
```

Repository:

```text
https://github.com/FreedomIntelligence/TwinMarket
```

Important source files currently visible in the repository include:

```text
Agent.py
simulation.py
trader/
    matching_engine.py
    trading_agent.py
    prompts.py
    recommender.py
    utility.py
    init_belief.py
```

### What we should reuse / adapt

Highest priority:

- `trader/matching_engine.py`
- order representation ideas
- transaction records
- position/cash settlement patterns
- market simulation orchestration
- trading-agent interface patterns

### What we should NOT preserve unchanged

TwinMarket's exchange rules are closely connected to its original A-share simulation design, including mechanisms such as price limits and auction-specific assumptions.

Our MVP specification is:

> **Continuous Double Auction (CDA)**

Therefore:

- retain useful data structures
- retain settlement/logging concepts
- rewrite/simplify the actual matching core where needed
- avoid unnecessary A-share-specific constraints in the first MVP

## 6.2 Behavioral Consistency Validation — Primary Behavioral Logic Reference

Paper:

**Behavioral Consistency Validation for LLM Agents: An Analysis of Trading-Style Switching through Stock-Market Simulation**

This paper is the most important conceptual source for our Stage 1 behavioral system.

The project should reproduce/adapt:

- Fundamental ↔ Technical style switching
- 10-trading-day evaluation blocks
- loss aversion factor
- herding factor
- wealth differentiation / relative performance
- price misalignment
- counterfactual alternative-style return
- explicit stay/switch decision
- behavioral-alignment metrics
- Mann–Whitney U
- effect-size reporting

### Important note for Codex

No verified public implementation repository is assumed in this TDD.

Therefore:

> Reimplement this behavioral layer from the paper specification rather than depending on an unverified GitHub repository.

The first implementation should be simple, auditable, and heavily unit-tested.

## 6.3 MF-LLM — Mean-Field Design Reference

Official repository:

```bash
git clone https://github.com/Miracle1207/Mean-Field-LLM.git third_party/Mean-Field-LLM
```

Repository:

```text
https://github.com/Miracle1207/Mean-Field-LLM
```

Relevant repository structure includes:

```text
IB-Tune/
mf_llm/
    data/
    evaluate/
    mean_field_utils/
    scripts/
```

MF-LLM uses evolving population-level signals to replace explicit pairwise interaction.

### What we should reuse

Conceptual design:

```text
agent state + population signal
        ↓
agent action
        ↓
aggregate population behavior
        ↓
updated population signal
```

### MVP simplification

DO NOT use an additional LLM to summarize the population in V1.

For the financial market MVP, calculate the mean field deterministically:

```python
MeanFieldState(
    fundamental_share=...,
    technical_share=...,
    sentiment=...,
    buy_sell_imbalance=...,
    avg_pnl=...,
)
```

This makes the signal:

- cheap
- reproducible
- inspectable
- easy to ablate

Later versions may experiment with LLM-generated population summaries or IB-Tune-style alignment.

## 6.4 EconAgent — Memory / Reflection Reference

Official repository:

```bash
git clone https://github.com/tsinghua-fib-lab/ACL24-EconAgent.git third_party/ACL24-EconAgent
```

Repository:

```text
https://github.com/tsinghua-fib-lab/ACL24-EconAgent
```

Default branch:

```text
master
```

EconAgent is not a stock-market simulator, so do not reuse its environment wholesale.

### Reuse only the pattern

The relevant idea is:

```text
recent environment
+ recent decisions
+ periodic reflection
→ future decision context
```

Our V1 implementation should maintain:

- recent observations
- recent actions
- recent realized outcomes
- periodic compressed summary

## 6.5 LAMP — Future Advanced Memory Reference

Official repository:

```bash
git clone https://github.com/hey0223/LAMP.git third_party/LAMP
```

Repository:

```text
https://github.com/hey0223/LAMP
```

Potentially useful later:

- short-term experience memory
- long-term experience pool
- retrieval of high-value historical reasoning trajectories
- separation of reasoning and decision modules

### MVP rule

Do not implement LAMP-style long-term retrieval in V1.

Only introduce it after:

- base exchange is stable
- Stage 1 behavior is reproducible
- mean-field feedback works
- baseline experiments are available

## 6.6 StockAgent — Orchestration / Logging Reference

Official repository:

```bash
git clone https://github.com/MingyuJ666/Stockagent.git third_party/StockAgent
```

Repository:

```text
https://github.com/MingyuJ666/Stockagent
```

Useful references:

- simulation loop
- agent creation
- market sessions
- JSON trading actions
- logging
- records
- prompt organization

### Do not reuse its matching mechanism as the final exchange

The public implementation uses a relatively simple matching process and should not be treated as the authoritative CDA implementation for this project.

Use it as an orchestration reference only.

---

# 7. Suggested Repository Layout

Create our own repository.

Suggested project name:

```text
behavioral-llm-market
```

Recommended structure:

```text
behavioral-llm-market/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
│
├── docs/
│   ├── TDD.md
│   ├── architecture.md
│   ├── experiment_protocol.md
│   └── third_party_notes.md
│
├── config/
│   ├── base.yaml
│   ├── stage1_behavioral.yaml
│   ├── stage2_endogenous.yaml
│   ├── agents.yaml
│   └── models.yaml
│
├── src/
│   └── behavioral_market/
│       │
│       ├── data/
│       │   ├── market_provider.py
│       │   ├── fundamental_provider.py
│       │   ├── macro_provider.py
│       │   ├── feature_builder.py
│       │   └── timestamp_alignment.py
│       │
│       ├── agents/
│       │   ├── agent.py
│       │   ├── state.py
│       │   ├── persona.py
│       │   ├── memory.py
│       │   ├── counterfactual.py
│       │   ├── behavioral_policy.py
│       │   └── strategy_switcher.py
│       │
│       ├── population/
│       │   └── mean_field.py
│       │
│       ├── market/
│       │   ├── order.py
│       │   ├── trade.py
│       │   ├── order_book.py
│       │   ├── matching_engine.py
│       │   ├── exchange.py
│       │   └── portfolio.py
│       │
│       ├── environments/
│       │   ├── base.py
│       │   ├── historical_replay.py
│       │   └── endogenous_market.py
│       │
│       ├── llm/
│       │   ├── client.py
│       │   ├── prompts.py
│       │   ├── schemas.py
│       │   ├── retry.py
│       │   └── cache.py
│       │
│       ├── calibration/
│       │   ├── behavioral_metrics.py
│       │   ├── calibrator.py
│       │   └── prompt_variants.py
│       │
│       ├── evaluation/
│       │   ├── micro.py
│       │   ├── macro.py
│       │   ├── stylized_facts.py
│       │   └── statistics.py
│       │
│       ├── simulation/
│       │   ├── engine.py
│       │   ├── scheduler.py
│       │   ├── events.py
│       │   └── recorder.py
│       │
│       └── utils/
│           ├── seed.py
│           ├── logging.py
│           └── serialization.py
│
├── scripts/
│   ├── run_stage1.py
│   ├── run_stage2.py
│   ├── evaluate_stage1.py
│   ├── evaluate_stage2.py
│   └── download_public_data.py
│
├── tests/
│   ├── test_order_book.py
│   ├── test_matching_engine.py
│   ├── test_portfolio.py
│   ├── test_counterfactual.py
│   ├── test_mean_field.py
│   ├── test_stage1_env.py
│   └── test_endogenous_env.py
│
├── outputs/
│   └── .gitkeep
│
└── third_party/
    ├── README.md
    ├── TwinMarket/
    ├── Mean-Field-LLM/
    ├── ACL24-EconAgent/
    ├── LAMP/
    └── StockAgent/
```

---

# 8. Core Domain Schemas

Prefer Python dataclasses or Pydantic models.

Pydantic is recommended for LLM-facing structured outputs because validation is important.

## 8.1 AgentPersona

```python
class AgentPersona(BaseModel):
    agent_id: str

    risk_tolerance: float         # [0, 1]
    loss_aversion: float          # [0, 1]
    herding: float                # [0, 1]
    wealth_sensitivity: float     # [0, 1]
    mispricing_sensitivity: float # [0, 1]

    initial_style: Literal["fundamental", "technical"]
    persona_summary: str
```

V1 should use deterministic/specified distributions.

Do not ask the LLM to invent the full population every run unless explicitly testing persona generation.

## 8.2 PortfolioState

```python
class PortfolioState(BaseModel):
    cash: float
    position: int
    avg_cost: float
    realized_pnl: float
    unrealized_pnl: float
    total_equity: float
```

All fields are computed by code.

## 8.3 AgentState

```python
class AgentState(BaseModel):
    agent_id: str
    trading_day: int

    current_style: Literal["fundamental", "technical"]
    last_switch_day: int | None

    portfolio: PortfolioState

    recent_return: float
    rolling_return: float
    drawdown: float

    memory_summary: str | None

    actual_strategy_return: float
    counterfactual_strategy_return: float
    counterfactual_gap: float
```

## 8.4 MarketObservation

```python
class MarketObservation(BaseModel):
    timestamp: str

    price: float
    previous_close: float
    return_1d: float
    volume: float

    macd: float | None
    macd_signal: float | None
    volatility_20: float | None
    volume_trend_20: float | None

    current_ratio: float | None
    leverage: float | None
    operating_cash_flow: float | None
    free_cash_flow: float | None
    fundamental_value: float | None

    interest_rate: float | None
    cpi: float | None
    unemployment: float | None

    news_summary: str | None
    shock_type: str | None
```

## 8.5 MeanFieldState

```python
class MeanFieldState(BaseModel):
    fundamental_share: float
    technical_share: float

    sentiment: float
    buy_sell_imbalance: float
    avg_pnl: float

    submitted_buy_share: float | None = None
    submitted_sell_share: float | None = None
```

All values must be calculated from the population.

No hallucinated values.

## 8.6 OrderDecision

LLM output:

```python
class OrderDecision(BaseModel):
    action: Literal["buy", "sell", "hold"]
    quantity: int
    limit_price: float | None
    rationale: str
```

Validation:

- hold → quantity must be 0
- buy/sell → quantity > 0
- limit price must be positive
- sell quantity cannot exceed position
- buy order cannot exceed available cash at limit price
- invalid LLM output must be repaired or rejected

The exchange must never directly execute an unvalidated LLM response.

## 8.7 StyleDecision

```python
class StyleDecision(BaseModel):
    decision: Literal["stay", "switch"]
    target_style: Literal["fundamental", "technical"]
    rationale: str

    perceived_current_advantage: float | None = None
    perceived_social_pressure: float | None = None
```

If decision = stay:

```text
target_style == current_style
```

If decision = switch:

```text
target_style != current_style
```

## 8.8 Order

```python
class Order(BaseModel):
    order_id: str
    agent_id: str

    side: Literal["buy", "sell"]
    price: float
    quantity: int
    remaining_quantity: int

    timestamp: int
    status: Literal[
        "open",
        "partially_filled",
        "filled",
        "cancelled",
        "expired"
    ]
```

## 8.9 Trade

```python
class Trade(BaseModel):
    trade_id: str

    buyer_id: str
    seller_id: str

    price: float
    quantity: int

    buy_order_id: str
    sell_order_id: str

    timestamp: int
```

---

# 9. Exchange Design

The exchange is one of the most important deterministic components.

It must have exhaustive unit tests before LLM integration.

## 9.1 Continuous Double Auction

Maintain:

```text
bid book
ask book
```

Priority:

### Buy side

```text
higher price first
then earlier timestamp
```

### Sell side

```text
lower price first
then earlier timestamp
```

Matching condition:

```python
best_bid.price >= best_ask.price
```

If false:

```text
no transaction
```

If true:

```text
execute trade
```

## 9.2 Trade Price Rule

Choose one explicit rule and keep it fixed across experiments.

Recommended MVP:

```text
maker / resting-order price
```

Alternative:

```text
earlier order's price
```

Do NOT allow LLM to determine transaction price after matching.

Document the exact choice in the experiment metadata.

## 9.3 Partial Fills

Example:

```text
Buy:  100 shares @ 10.50
Sell:  40 shares @ 10.40
```

Result:

```text
40 shares executed
buy order remains with 60
sell order filled
```

Required states:

- open
- partially filled
- filled

## 9.4 Order Expiration

MVP recommendation:

```text
DAY orders
```

At period/day end:

```text
all unfilled orders expire
```

This avoids complex persistent-book effects until explicitly studied.

Later experiments may use GTC.

---

# 10. Portfolio Accounting

Portfolio logic must be deterministic and independent from LLM calls.

Core operations:

```python
apply_buy(trade)
apply_sell(trade)
mark_to_market(price)
```

Every transaction must update:

```text
cash
position
average cost
realized PnL
unrealized PnL
equity
```

Invariant tests:

```text
cash cannot become negative unless margin is enabled
position cannot become negative unless short selling is enabled
sell quantity <= current position
```

MVP:

```text
no margin
no shorting
```

---

# 11. Fundamental vs Technical Strategies

A key research feature is that **style changes what information is emphasized**, not that the LLM becomes a completely unrelated model.

## 11.1 Fundamental Agent Input

Primary inputs:

```text
current ratio
leverage
operating cash flow
free cash flow
fundamental value / valuation gap
position
PnL
recent return
macro variables
```

The prompt should clearly say:

> Base your decision primarily on company fundamentals, valuation, and mean reversion toward fundamental value.

## 11.2 Technical Agent Input

Primary inputs:

```text
price
return
MACD
MACD signal
volatility
volume trend
recent momentum
position
PnL
```

Prompt:

> Base your decision primarily on market trend, momentum, price-volume behavior, and technical signals.

---

# 12. Counterfactual Ledger

This module is critical and must be implemented in code, not delegated to the LLM.

Purpose:

> Estimate what would have happened if the same agent had used the alternative style over the same recent evaluation block.

Example:

```text
actual style: Fundamental
actual block return: +2%

alternative style: Technical
counterfactual block return: +8%

counterfactual gap = +6%
```

The agent then receives:

```text
Your actual recent strategy return: +2%
Alternative strategy counterfactual return: +8%
Relative performance gap: +6%
```

This supports relative-performance / wealth-differentiation switching.

## 12.1 V1 Counterfactual Design

Do not run a second full parallel LLM-agent world.

Instead, use a lightweight shadow-strategy evaluation.

Recommended implementation:

```text
actual style
→ actual LLM trading decisions
→ realized portfolio result

alternative style
→ calculate hypothetical signal-based benchmark return
OR
→ periodically call the alternative style policy in shadow mode
```

Two supported modes:

```yaml
counterfactual:
  mode: "rule_proxy"
```

or:

```yaml
counterfactual:
  mode: "shadow_llm"
```

Start with `rule_proxy` for low cost and deterministic debugging.

Later use `shadow_llm` for stronger fidelity.

---

# 13. Memory System

V1 memory should be simple.

Each agent stores recent entries:

```python
MemoryEntry:
    day
    observation_summary
    action
    style
    executed_or_not
    pnl_change
    market_return
```

Limit:

```text
last L periods
```

Recommended:

```yaml
memory:
  rolling_window: 10
```

Every strategy-evaluation block:

```text
recent memories
→ summary
→ retained memory summary
```

The summary may be LLM-generated, but raw numerical state remains stored separately.

Never replace raw state with natural-language memory.

---

# 14. Population Mean Field

The population mean field is a public aggregate state.

V1:

```python
def compute_mean_field(agents, submitted_orders):
    ...
```

Recommended metrics:

```text
Fundamental share
Technical share
mean recent PnL
positive/negative sentiment share
buy quantity
sell quantity
order imbalance
```

Example order imbalance:

```python
imbalance = (buy_qty - sell_qty) / (buy_qty + sell_qty + eps)
```

## 14.1 Information Timing

Avoid future leakage.

At trading period `t`:

Agents should see:

```text
mean_field_(t-1)
```

or a clearly defined current pre-order population signal.

They must NOT see:

```text
other agents' current unsubmitted decisions
```

Recommended timeline:

```text
start of period t
    ↓
publish mean field from t-1
    ↓
agents decide
    ↓
orders submitted
    ↓
exchange clears
    ↓
compute t statistics
    ↓
mean_field_t stored for next period
```

This is clean and causal.

---

# 15. LLM Layer

The rest of the system must be model-provider agnostic.

Interface:

```python
class LLMClient(Protocol):
    async def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        ...
```

Implement adapters later:

```text
OpenAI-compatible
local vLLM-compatible
mock deterministic model
```

The core simulator must be able to run without a live API using a mock model.

---

# 16. LLM Prompt Structure

Prompts should be assembled from explicit blocks.

Recommended structure:

```text
SYSTEM ROLE
PERSONA
CURRENT STYLE
BEHAVIORAL TRAITS
PORTFOLIO STATE
MARKET OBSERVATION
MEMORY SUMMARY
COUNTERFACTUAL LEDGER
POPULATION MEAN FIELD
DECISION INSTRUCTION
OUTPUT SCHEMA
```

Example:

```text
You are investor agent A017.

Current strategy:
Technical.

Behavioral traits:
Loss aversion: high
Herding: medium
Relative-performance sensitivity: high
Mispricing sensitivity: low

Portfolio:
Cash = ...
Shares = ...
PnL = ...

Population:
Technical share = 72%
Fundamental share = 28%
Order imbalance = -0.41

Return JSON matching the required schema.
```

---

# 17. Behavioral Calibration

The project must separate:

```text
persona prompting
```

from:

```text
behavioral calibration
```

Persona prompting alone is a baseline.

Calibration means:

```text
measure behavioral response
→ quantify mismatch
→ modify behavioral policy
→ rerun
→ measure again
```

## 17.1 V1 Calibration Mechanism

Do not start with LoRA.

V1 should support:

```text
prompt parameter calibration
behavioral constraints
decision-threshold language
few-shot behavioral examples
```

Example:

```yaml
calibration:
  herding_prompt_weight: 0.6
  loss_aversion_prompt_weight: 0.8
```

These parameters do not have to be literal numeric weights inside the LLM.

They may select among structured prompt templates.

## 17.2 Calibration Objective

For behavioral factor `b`:

```text
target_behavior_effect_b
vs
observed_behavior_effect_b
```

Error:

```python
error_b = observed_effect_b - target_effect_b
```

Possible target sources:

1. behavioral finance theory direction
2. published human experiments
3. our optional oTree + Prolific study

Important:

> If no quantitative human ground truth exists, do not invent one.

Use:

```text
directional theoretical alignment
```

until human quantitative targets are collected.

---

# 18. Stage 1 — Historical Behavioral Validation Environment

Class:

```python
HistoricalReplayEnvironment
```

Key property:

> price path is exogenous

The agents do NOT move the market price.

Purpose:

> isolate behavioral consistency

## 18.1 Stage 1 Loop

```text
load historical day t
        ↓
build market/fundamental/technical observation
        ↓
agent decides trading action
        ↓
simulate accounting against historical path
        ↓
update memory
        ↓
update counterfactual ledger
        ↓
every 10 days:
    agent evaluates stay/switch
        ↓
record behavioral metrics
```

Candidate stocks from the proposal:

```text
MSFT
ICE
VRTX
CAT
CLX
```

Use one stock per simulation run.

## 18.2 Stage 1 Behavioral Factors

We need explicit controlled variants.

Example:

```yaml
behavior_traits:
  loss_aversion:
    - low
    - high
  herding:
    - low
    - high
  wealth_sensitivity:
    - low
    - high
  mispricing_sensitivity:
    - low
    - high
```

A factorial design may generate 2^k conditions.

Start smaller during software debugging.

---

# 19. Stage 2 — Endogenous Market Environment

Class:

```python
EndogenousMarketEnvironment
```

The historical price series is NOT replayed after initialization.

Historical data may initialize:

- starting price
- starting fundamental state
- technical lookback
- agent priors

After simulation starts:

```text
price must be generated by transactions
```

## 19.1 Stage 2 Loop

```text
initialize market
initialize agents
initialize fundamental state
initialize mean field
        ↓
for each period t:

    publish market observation
    publish mean_field_(t-1)

    ↓

    each agent reasons independently

    ↓

    each agent returns structured order

    ↓

    validate orders

    ↓

    exchange matches orders

    ↓

    trades update portfolios

    ↓

    market price / volume update

    ↓

    update PnL + memory + counterfactual state

    ↓

    periodically evaluate style switching

    ↓

    compute mean_field_t

    ↓

    record micro + macro state
```

---

# 20. Fundamental Value Process

To evaluate bubbles and mispricing, the simulator needs an explicit fundamental value `F_t`.

MVP options:

## Option A — Exogenous deterministic fundamental process

Recommended first.

```python
F_t = F_{t-1} * exp(fundamental_growth + shock_t)
```

Advantages:

- transparent
- controllable
- repeatable
- ideal for counterfactual shocks

## Option B — Fundamental value derived from real company data

Future extension.

For the first endogenous prototype, use Option A.

---

# 21. Shock Engine

Required Stage 2 scenarios:

```text
No Shock
Positive Fundamental Shock
Negative Fundamental Shock
```

Example:

```yaml
shock:
  period: 10
  type: negative
  magnitude: -0.10
```

Same shock must be used across behavioral conditions.

This enables the primary counterfactual experiment:

```text
same fundamental shock
        ↓
different investor population
        ↓
different market outcome
```

---

# 22. Experimental Population Conditions

At minimum:

```text
Low-Herding Population
High-Herding Population
Behaviorally Calibrated Population
```

Additional:

```text
Mixed/Heterogeneous Population
```

The simulator must make population configuration reproducible via YAML.

Example:

```yaml
population:
  n_agents: 30
  initial_fundamental_share: 0.5

  herding_distribution:
    type: beta
    alpha: 2
    beta: 5
```

---

# 23. Baselines

The project should implement four baseline families.

## Baseline 1 — Rule-Based ABM

No LLM.

Simple deterministic/stochastic rules.

Purpose:

> Compare against traditional ABM.

## Baseline 2 — Vanilla LLM

Out-of-the-box LLM.

Minimal instruction.

No behavioral calibration.

Purpose:

> Determine what the base model naturally does.

## Baseline 3 — Prompted LLM

Behavior/personality traits added through prompt.

No measured calibration loop.

Purpose:

> Determine whether persona prompting alone is enough.

## Baseline 4 — Calibrated LLM (Ours)

Behavior tested in Stage 1.

Prompt/policy calibrated using observed behavioral mismatch.

Purpose:

> Main proposed method.

---

# 24. Ablations

Required:

```text
w/o calibration
w/o counterfactual ledger
w/o population mean field
fixed strategy
adaptive switching
```

Potential later:

```text
w/o memory
w/o fundamentals
w/o macro signals
```

---

# 25. Micro-Level Evaluation

Core metric:

## Switch Rate

```text
SR = number of switching decisions / number of eligible decisions
```

Behavior effect:

```text
Δ_b = P(switch | b = 1) - P(switch | b = 0)
```

This metric alone does NOT establish human likeness.

It must be compared against:

- theoretical direction
- quantitative human benchmark where available

## Statistical tests

Use:

```text
Mann–Whitney U
Cliff's delta
```

Interpretation:

```text
Mann–Whitney U → whether groups differ statistically
Cliff's delta   → how large the difference is
```

Store:

```text
U
p-value
Cliff's delta
confidence interval if implemented
```

---

# 26. Macro-Level Evaluation

Required metrics:

```text
log returns
volatility
turnover
order execution rate
maximum drawdown
price-fundamental deviation
crash magnitude
recovery time
```

## 26.1 Return

```text
r_t = log(P_t / P_(t-1))
```

## 26.2 Price-Fundamental Deviation

```text
MSE_F = (1/T) Σ(P_t - F_t)^2
```

Possible normalized alternative:

```text
relative deviation = |P_t - F_t| / F_t
```

## 26.3 Maximum Drawdown

```text
MDD = max_t ((peak_t - P_t) / peak_t)
```

## 26.4 Order Execution Rate

Define unambiguously.

Recommended:

```text
executed quantity / submitted quantity
```

or:

```text
fully/partially executed orders / submitted orders
```

Do not mix definitions.

Recommended V1:

```text
executed share quantity / submitted share quantity
```

## 26.5 Stylized Facts

Later-stage checks:

```text
fat-tailed returns
volatility clustering
autocorrelation structure
turnover distribution
```

Do not claim realistic stylized facts unless statistical comparisons support the claim.

---

# 27. Data Layer

## Market Data

MVP:

- daily OHLCV
- volume
- dividends
- splits

Preferred sources:

```text
Alpha Vantage
Yahoo Finance public historical download
```

Role:

```text
technical features
historical replay state
initialization
```

## Fundamentals

Source:

```text
SEC EDGAR Company Facts
```

Candidate features:

```text
leverage
current ratio
operating cash flow
free cash flow
```

## Macro

Source:

```text
FRED
```

Candidate features:

```text
interest rate
CPI
unemployment
```

---

# 28. Timestamp Alignment and Leakage Prevention

This is a hard requirement.

Every source must have:

```text
observation timestamp
availability timestamp
```

Agents may only access information that had been publicly available at that simulation time.

Example:

```text
10-Q period end: March 31
filing date: May 8
```

The agent on April 15 must NOT see the May 8 filing.

Implement:

```python
get_available_information(simulation_timestamp)
```

Do not simply merge financial statements by fiscal quarter.

---

# 29. Data Feature Pipeline

```text
raw data
    ↓
cleaning
    ↓
timestamp normalization
    ↓
disclosure lag
    ↓
feature computation
    ↓
observation builder
    ↓
agent input
```

Every feature must be traceable to its source timestamp.

---

# 30. Recorder / Experiment Logging

Each run must generate:

```text
config snapshot
git commit hash
random seed
model name
prompt version
timestamp
```

Store:

```text
agents.csv / parquet
orders.csv / parquet
trades.csv / parquet
market.csv / parquet
mean_field.csv / parquet
switching.csv / parquet
metrics.json
run_metadata.json
```

Recommended output:

```text
outputs/
  run_2026xxxx_xxxxxx/
```

---

# 31. LLM Cost Control

Support:

```text
async requests
bounded concurrency
response caching
retry
structured validation
```

Cache key should include:

```text
model
system prompt
user prompt
temperature
schema version
```

During software debugging:

```text
use mock LLM
```

During behavioral debugging:

```text
use cheap model
```

Only final robustness experiments should use expensive models.

---

# 32. Reproducibility

Set and record:

```text
Python random seed
NumPy seed
simulation scheduler seed
persona generation seed
order timing seed
```

LLM APIs may remain nondeterministic.

Therefore record:

```text
temperature
model ID
prompt
raw response
validated response
```

All final experiments should use multiple seeds.

Target:

```text
10–20 seeds
```

---

# 33. Testing Strategy

Do not integrate LLMs until the exchange passes deterministic tests.

## 33.1 Matching Tests

Test:

```text
best bid < best ask → no trade
best bid == best ask → trade
best bid > best ask → trade
```

Test price priority.

Test time priority.

Test partial fill.

Test multi-order fill.

Test order expiration.

## 33.2 Portfolio Tests

Test:

```text
buy decreases cash
buy increases position
sell increases cash
sell decreases position
realized PnL correct
unrealized PnL correct
```

Invariants:

```text
no negative cash
no negative shares
```

## 33.3 Counterfactual Tests

Given fixed synthetic price path:

```text
actual Fundamental return
alternative Technical return
```

must reproduce expected values exactly.

## 33.4 Mean-Field Tests

Example:

```text
7 Technical
3 Fundamental
```

Expected:

```text
technical_share = 0.7
fundamental_share = 0.3
```

Order imbalance tests must be deterministic.

## 33.5 Historical Environment Tests

Historical price path must never be modified by simulated agent orders.

## 33.6 Endogenous Environment Tests

Endogenous price must change only because of:

```text
executed transactions
```

not because of historical future price leakage.

---

# 34. Configuration Example

`config/stage2_endogenous.yaml`

```yaml
experiment:
  name: stage2_negative_shock
  seed: 42

market:
  asset: SYNTHETIC_A
  initial_price: 100.0
  initial_fundamental_value: 100.0
  periods: 30

exchange:
  type: continuous_double_auction
  order_ttl: day
  allow_short: false
  allow_margin: false
  transaction_price_rule: resting_order

population:
  n_agents: 30
  initial_fundamental_share: 0.5

strategy_switching:
  enabled: true
  evaluation_interval: 10

mean_field:
  enabled: true
  lag_periods: 1
  fields:
    - fundamental_share
    - technical_share
    - sentiment
    - buy_sell_imbalance
    - avg_pnl

counterfactual:
  enabled: true
  mode: rule_proxy

shock:
  enabled: true
  type: negative_fundamental
  period: 10
  magnitude: -0.10

llm:
  provider: mock
  model: mock-v1
  temperature: 0.0

logging:
  save_prompts: true
  save_raw_responses: true
```

---

# 35. Development Phases for Codex

Codex should execute the project incrementally.

## Phase 0 — Repository Setup

Create our repository skeleton.

Add:

```text
pyproject.toml
pytest
ruff
mypy optional
pre-commit optional
```

Create `third_party/README.md`.

Clone external repos:

```bash
mkdir -p third_party

git clone --depth 1 https://github.com/FreedomIntelligence/TwinMarket.git third_party/TwinMarket

git clone --depth 1 https://github.com/Miracle1207/Mean-Field-LLM.git third_party/Mean-Field-LLM

git clone --depth 1 https://github.com/tsinghua-fib-lab/ACL24-EconAgent.git third_party/ACL24-EconAgent

git clone --depth 1 https://github.com/hey0223/LAMP.git third_party/LAMP

git clone --depth 1 https://github.com/MingyuJ666/Stockagent.git third_party/StockAgent
```

Do not edit the reference repos.

## Phase 1 — Deterministic Exchange

Implement:

```text
Order
Trade
OrderBook
MatchingEngine
Portfolio
Exchange
```

No LLM.

Create random/scripted agents.

Acceptance condition:

```text
pytest passes for all matching/accounting tests
```

## Phase 2 — Agent State

Implement:

```text
AgentPersona
AgentState
Memory
Strategy
CounterfactualLedger
```

No live LLM yet.

Acceptance:

```text
scripted Fundamental/Technical agents can run through synthetic data
```

## Phase 3 — Historical Replay / Stage 1

Implement:

```text
HistoricalReplayEnvironment
feature building
10-day switching scheduler
behavioral metric recorder
```

Use mock decisions first.

Acceptance:

```text
historical price is immutable
behavioral events are reproducible
```

## Phase 4 — LLM Integration

Implement:

```text
LLMClient
Pydantic structured outputs
retry/repair
prompt templates
response cache
```

Start with one model provider.

Acceptance:

```text
LLM can generate valid StyleDecision and OrderDecision
```

## Phase 5 — Mean Field

Implement deterministic population aggregation.

Acceptance:

```text
mean-field values match unit-test population states
```

## Phase 6 — Endogenous Environment / Stage 2

Connect:

```text
agents
→ orders
→ exchange
→ trades
→ new price
→ PnL
→ mean field
→ next period
```

Acceptance:

```text
30-period synthetic endogenous market completes without data leakage
```

## Phase 7 — Behavioral Calibration

Implement:

```text
baseline prompt
measure behavior
prompt-policy calibration
rerun
compare
```

No LoRA yet.

## Phase 8 — Experiment Suite

Implement:

```text
Rule-Based ABM
Vanilla LLM
Prompted LLM
Calibrated LLM
```

Ablations.

Multiple seeds.

---

# 36. First Deliverable

The first deliverable should NOT attempt the full research system.

Codex should first deliver:

```text
MVP-0: Deterministic Market Core
```

Contents:

```text
1 asset
N scripted agents
cash
position
limit orders
CDA
price-time priority
partial fill
portfolio accounting
transaction logs
market price
unit tests
```

No API calls.

This creates the stable substrate for all later experiments.

---

# 37. Second Deliverable

```text
MVP-1: Behavioral Agent Prototype
```

Add:

```text
Fundamental / Technical agent state
10-period switching
counterfactual ledger
mean-field state
mock LLM / rule policy
```

Still no expensive experiment.

---

# 38. Third Deliverable

```text
MVP-2: LLM Behavioral Market
```

Add:

```text
structured LLM decisions
prompted behavioral traits
historical Stage 1
endogenous Stage 2
```

Only after this should calibration experiments begin.

---

# 39. Codex Implementation Rules

Codex MUST follow these rules.

## DO

- inspect TwinMarket before implementing exchange logic
- document which external concepts/files inspired each module
- write our own clean interfaces
- add tests before integration
- maintain deterministic market/accounting code
- use structured LLM outputs
- log every run
- enforce causal timestamps
- separate Stage 1 and Stage 2 environments
- keep configuration external to source code
- keep code model-provider agnostic

## DO NOT

- modify cloned reference repositories
- copy an entire external simulator wholesale
- let an LLM generate or overwrite portfolio state
- replay future historical prices in Stage 2
- expose future filings/news to agents
- introduce O(N²) dialogue in MVP
- add RL before base behavioral experiments are stable
- add LoRA before prompt calibration baselines exist
- invent human ground-truth behavioral effect sizes
- claim market realism based only on visual plots

---

# 40. Architecture Decision Records

Codex should create short ADRs for decisions that materially affect results.

Required ADRs:

```text
ADR-001 transaction-price rule
ADR-002 order expiration rule
ADR-003 counterfactual return method
ADR-004 mean-field timing
ADR-005 fundamental-value process
ADR-006 LLM output schema
ADR-007 behavioral calibration target
```

This prevents silent implementation choices from becoming hidden experimental assumptions.

---

# 41. Scientific Validity Requirements

Every major simulator mechanism should answer:

```text
What assumption does this encode?
Can we ablate it?
Can we reproduce it?
Can we log it?
Can we compare it against a baseline?
```

If not, redesign it.

---

# 42. Final Intended Research Pipeline

```text
                  REAL / PUBLIC DATA
                         │
                         ▼
               Historical Replay
                         │
                         ▼
            Behavioral Validation
                         │
                  Behavioral Error
                         │
                         ▼
                  Calibration
                         │
                         ▼
               Calibrated Agents
                         │
                         ▼
          ┌─────────────────────────┐
          │   Endogenous Exchange   │
          │       CDA Matching      │
          └───────────┬─────────────┘
                      │
                      ▼
                Market Dynamics
                      │
                      ▼
              Population Mean Field
                      │
                      └───────────────┐
                                      │
                                      ▼
                                Agent Decisions
                                      │
                                      └────→ Exchange
```

Evaluation:

```text
Micro behavioral fidelity
            +
Macro market fidelity
            +
Micro-to-macro relationship
```

---

# 43. Minimal Starting Command for Codex

When Codex starts implementation, the intended first instruction is:

```text
Read docs/TDD.md completely.

Do not start by integrating an LLM.

First:
1. create the repository structure,
2. clone the five reference repositories under third_party/,
3. inspect TwinMarket/trader/matching_engine.py,
4. design our own minimal CDA interfaces,
5. implement Order, Trade, OrderBook, MatchingEngine, Portfolio and Exchange,
6. write pytest tests for price priority, time priority, partial fills and portfolio accounting,
7. run the tests,
8. only after the deterministic market core passes tests, proceed to the agent layer.

Do not modify files under third_party/.
```

---

# 44. Reference Repository Summary

| Project | Role in our system | Repository |
|---|---|---|
| TwinMarket | Primary exchange + financial simulation engineering reference | https://github.com/FreedomIntelligence/TwinMarket |
| MF-LLM | Mean-field population feedback reference | https://github.com/Miracle1207/Mean-Field-LLM |
| EconAgent | Memory/reflection design reference | https://github.com/tsinghua-fib-lab/ACL24-EconAgent |
| LAMP | Advanced experience-memory reference | https://github.com/hey0223/LAMP |
| StockAgent | Simulation loop / logging / structured action reference | https://github.com/MingyuJ666/Stockagent |
| Behavioral Consistency Validation | Main behavioral switching specification | Reimplement from paper; no verified repo assumed here |
| ASFM | Continuous/order matching methodological reference | Use paper specification unless a verified complete repo is identified |

---

# 45. Recommended Immediate Next Task

The next coding task after reading this document is:

> **Build and test the deterministic single-asset continuous-double-auction market core.**

Do not build the calibration layer first.

The dependency order is:

```text
Exchange correctness
        ↓
Agent state correctness
        ↓
Counterfactual correctness
        ↓
Stage 1 behavioral validation
        ↓
Mean-field feedback
        ↓
Stage 2 endogenous market
        ↓
Calibration
        ↓
Full experiments
```

That dependency order should be preserved unless a documented architectural reason requires changing it.
