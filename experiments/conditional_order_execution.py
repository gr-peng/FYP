"""Opt-in, invented two-period execution demo; no model or human-data loader.

An external opportunity gate requires BOTH order directions to be feasible.
The predictor supplies only P(BUY | an active order opportunity); it does not
choose participation, price, quantity, priority, or cancellation. Local memory
counts accepted submissions, including unfilled orders, not fills or rejections.
"""

import math
import random
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from decimal import Decimal

from behavioral_market.evaluation.rp_metrics import market_metrics, unknown
from behavioral_market.market.exchange import Exchange
from behavioral_market.market.portfolio import Portfolio
from experiments import conditional_memory_prompt as prompt

Predictor = Callable[[list[dict[str, str]]], float]
AGENTS = ("invented_alpha", "invented_beta")
TICKS = tuple((period, second) for period in (1, 2) for second in (0, 40, 80))
PRICE = Decimal("50")


@dataclass(frozen=True)
class DemoConfig:
    cash: tuple[int, int] = (500, 500)
    shares: tuple[int, int] = (4, 4)
    quantities: tuple[int, int] = (2, 3)
    dividends: tuple[int, int] = (10, 0)
    opportunities: tuple[bool, ...] = (True,) * 12
    seed: int = 20261107

    def validate(self):
        for values, minimum in ((self.cash, 0), (self.shares, 0), (self.quantities, 1)):
            if (
                not isinstance(values, tuple)
                or len(values) != 2
                or any(type(value) is not int or value < minimum for value in values)
            ):
                raise ValueError("two nonnegative integer balances/positive quantities required")
        if (
            not isinstance(self.dividends, tuple)
            or len(self.dividends) != 2
            or any(type(value) is not int or value not in (0, 10) for value in self.dividends)
        ):
            raise ValueError("two realized dividends in {0,10} required")
        if (
            not isinstance(self.opportunities, tuple)
            or len(self.opportunities) != 12
            or any(type(value) is not bool for value in self.opportunities)
        ):
            raise ValueError("12 explicit external opportunity switches required")
        if type(self.seed) is not int:
            raise ValueError("integer sampling seed required")


DEFAULT_CONFIG = DemoConfig()


def probability(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("predictor must return one finite native numeric BUY probability")
    return float(value)


def counts(rows):
    return {side: sum(row["side"] == side for row in rows) for side in ("buy", "sell")}


class ConditionalExecutionDemo:
    def __init__(
        self, predictors: Mapping[str, Predictor], arm: str, config: DemoConfig = DEFAULT_CONFIG
    ):
        config.validate()
        if (
            arm not in prompt.ARMS
            or set(predictors) != set(AGENTS)
            or any(not callable(predictors[agent]) for agent in AGENTS)
        ):
            raise ValueError(
                "one callback per invented account and a supported memory arm required"
            )
        self.config, self.arm = config, arm
        self.predictors = dict(predictors)
        self.exchange = Exchange("invented_dividend_share", PRICE)
        for i, agent in enumerate(AGENTS):
            self.exchange.register(
                agent, Portfolio(config.cash[i], config.shares[i], PRICE if config.shares[i] else 0)
            )
        # All potential opportunities are allocated before any gate or prediction.
        rng = random.Random(config.seed)
        self.uniforms = tuple(rng.random() for _ in range(12))
        self.next_tick = 0
        self.failed = False
        self.accepted = []
        self.trade_history = []
        self.public_periods = []
        self.events = []
        self.dividend_history = []

    def accounts(self):
        return {
            agent: {
                "cash": str(portfolio.cash),
                "shares": portfolio.position,
                "available_cash": str(self.exchange._available_cash(agent)),
                "available_shares": self.exchange._available_shares(agent),
                "marked_equity": str(portfolio.total_equity),
            }
            for agent, portfolio in self.exchange.portfolios.items()
        }

    def observation(self, agent, period, second):
        """Fresh allowlisted state from strictly earlier events, never caller observations."""
        if self.failed or self.next_tick >= len(TICKS) or (period, second) != TICKS[self.next_tick]:
            raise ValueError("observations can only describe the next scheduled decision")
        account = self.exchange.portfolios[agent]
        trades = [r for r in self.trade_history if r["period"] == period and r["second"] < second]
        last_second = max((r["second"] for r in trades), default=None)
        last = [r for r in trades if r["second"] == last_second]

        def mean(rows):
            # Per execution record, not volume-weighted, matching the prompt field.
            return (
                float(sum((Decimal(r["price"]) for r in rows), Decimal(0)) / len(rows))
                if rows
                else None
            )

        own = [r for r in self.accepted if r["agent_id"] == agent]
        current = [r for r in own if r["period"] == period and r["second"] < second]
        completed = [r for r in own if r["period"] < period]
        last_period = max((r["period"] for r in completed), default=None)
        return {
            "cash": float(account.cash),
            "shares": account.position,
            "period": period,
            "repetition": 1,
            "elapsed_seconds": second,
            "displayed_expected_remaining_dividends_per_share": 5 * (11 - period),
            "pre_market_task": {"task": "video", "condition": "calm"},
            "earlier_current_period_trades": {
                "count": len(trades),
                "mean_price": mean(trades),
                "last_completed_trade_second": last_second,
                "mean_price_in_last_completed_trade_second": mean(last),
            },
            "earlier_periods": deepcopy([r for r in self.public_periods if r["period"] < period]),
            prompt.memory.CURRENT: counts(current),
            prompt.memory.COMPLETED: {
                "earlier_periods_this_repetition": counts(completed),
                "previous_repetitions": {"buy": 0, "sell": 0},
                "last_completed_period_with_own_orders": (
                    counts([r for r in completed if r["period"] == last_period])
                    if last_period is not None
                    else None
                ),
            },
        }

    def gate(self, agent, quantity):
        reasons = []
        if self.exchange._available_cash(agent) < PRICE * quantity:
            reasons.append("insufficient_available_cash_for_buy")
        if self.exchange._available_shares(agent) < quantity:
            reasons.append("insufficient_available_shares_for_sell")
        if any(
            order.agent_id == agent
            and (
                (order.side.value == "buy" and order.price >= PRICE)
                or (order.side.value == "sell" and order.price <= PRICE)
            )
            for order in self.exchange.book.resting_orders()
        ):
            reasons.append("one_direction_would_cross_own_resting_order")
        return reasons

    def order_states(self):
        return [
            {
                "order_id": order.order_id,
                "agent_id": order.agent_id,
                "side": order.side.value,
                "price": str(order.price),
                "quantity": order.quantity,
                "remaining_quantity": order.remaining_quantity,
                "filled_quantity": order.quantity - order.remaining_quantity,
                "status": order.status.value,
                "accepted": True,
            }
            for order in self.exchange.orders
        ]

    def step(self):
        if self.failed or self.next_tick >= len(TICKS):
            raise ValueError("failed or completed demo cannot advance")
        try:
            return self._step()
        except BaseException:
            self.failed = True
            raise

    def _step(self):
        period, second = TICKS[self.next_tick]
        entries = []
        before = self.accounts()
        # Freeze every account and gate before invoking any callback or matching order.
        for i, agent in enumerate(AGENTS):
            slot = self.next_tick * 2 + i
            external = self.config.opportunities[slot]
            reasons = self.gate(agent, self.config.quantities[i]) if external else []
            state = self.observation(agent, period, second)
            entry = {
                "slot": slot,
                "agent_id": agent,
                "period": period,
                "second": second,
                "external_opportunity": external,
                "gate_reasons": reasons,
                "uniform": self.uniforms[slot],
                "state_before": state,
                "status": "not_offered"
                if not external
                else "blocked_external_opportunity"
                if reasons
                else "awaiting_probability",
                "question": None,
            }
            if external and not reasons:
                entry["question"] = prompt.build_question(state, self.arm, "buy_first")
            entries.append(entry)
        # Complete and validate all probabilities before this tick changes the exchange.
        for entry in entries:
            if entry["question"] is not None:
                p_buy = probability(
                    self.predictors[entry["agent_id"]](deepcopy(entry["question"]["messages"]))
                )
                entry.update(
                    probability_buy=p_buy, side="buy" if entry["uniform"] < p_buy else "sell"
                )
        # Fixed alpha then beta priority; independent of sampling and predictor call counts.
        new_trades = []
        for i, entry in enumerate(entries):
            if entry["question"] is None:
                continue
            agent, quantity = entry["agent_id"], self.config.quantities[i]
            # A changed feasibility check is an error, not a direction switch or another query.
            if self.gate(agent, quantity):
                raise RuntimeError("pre-batch feasible opportunity became infeasible")
            try:
                order, trades = self.exchange.submit_order(agent, entry["side"], PRICE, quantity)
            except ValueError as exc:
                entry.update(status="rejected", rejection_reason=str(exc))
                continue
            entry.update(status="accepted", order_id=order.order_id)
            self.accepted.append(
                {
                    "order_id": order.order_id,
                    "agent_id": agent,
                    "period": period,
                    "second": second,
                    "side": order.side.value,
                    "quantity": quantity,
                }
            )
            for trade in trades:
                row = {
                    "trade_id": trade.trade_id,
                    "buyer_id": trade.buyer_id,
                    "seller_id": trade.seller_id,
                    "buy_order_id": trade.buy_order_id,
                    "sell_order_id": trade.sell_order_id,
                    "price": str(trade.price),
                    "quantity": trade.quantity,
                    "period": period,
                    "second": second,
                }
                self.trade_history.append(row)
                new_trades.append(row)
        event = {
            "period": period,
            "second": second,
            "accounts_before": before,
            "opportunities": entries,
            "trades": deepcopy(new_trades),
            "orders_after_trading": self.order_states(),
            "accounts_after_trading": self.accounts(),
            "expired_order_ids": [],
            "dividends": [],
            "mark_price": str(self.exchange.market_price),
            "expected_remaining_dividends": 5 * (11 - period),
            "period_close_second": 120 if second == 80 else None,
        }
        after = event["accounts_after_trading"]
        if sum(Decimal(r["cash"]) for r in before.values()) != sum(
            Decimal(r["cash"]) for r in after.values()
        ) or sum(r["shares"] for r in before.values()) != sum(r["shares"] for r in after.values()):
            raise RuntimeError("trade settlement did not conserve cash and shares")
        if second == 80:
            period_trades = [r for r in self.trade_history if r["period"] == period]
            dividend = self.config.dividends[period - 1]
            self.public_periods.append(
                {
                    "period": period,
                    "mean_trade_price": (
                        float(sum(Decimal(r["price"]) for r in period_trades) / len(period_trades))
                        if period_trades
                        else None
                    ),
                    "closing_price": float(period_trades[-1]["price"]) if period_trades else None,
                    "dividend": dividend,
                }
            )
            event["expired_order_ids"] = [o.order_id for o in self.exchange.end_day()]
            for agent in AGENTS:
                account = self.exchange.portfolios[agent]
                amount = Decimal(dividend) * account.position
                account.cash += amount
                account.mark_to_market(self.exchange.market_price)
                row = {
                    "period": period,
                    "agent_id": agent,
                    "shares": account.position,
                    "per_share": dividend,
                    "cash_created": str(amount),
                }
                self.dividend_history.append(row)
                event["dividends"].append(row)
        event["accounts_end"] = self.accounts()
        event["period_close"] = (
            {
                "period": period,
                "second": 120,
                "expired_order_ids": deepcopy(event["expired_order_ids"]),
                "dividends": deepcopy(event["dividends"]),
                "accounts_before": deepcopy(after),
                "accounts_after": deepcopy(event["accounts_end"]),
            }
            if second == 80
            else None
        )
        cash_created = sum(Decimal(row["cash_created"]) for row in event["dividends"])
        if (
            sum(Decimal(r["cash"]) for r in event["accounts_end"].values())
            != sum(Decimal(r["cash"]) for r in after.values()) + cash_created
        ):
            raise RuntimeError("dividend accounting differs from explicit cash creation")
        self.events.append(event)
        self.next_tick += 1
        return deepcopy(event)

    def result(self):
        if self.failed or self.next_tick != len(TICKS):
            raise ValueError("only a complete six-tick demo has a result")
        opportunities = [entry for event in self.events for entry in event["opportunities"]]
        orders = self.order_states()
        return {
            "scope": "invented_two_period_execution_mapping_callback_provenance_not_attested",
            "arm": self.arm,
            "direct_model_POSTs": 0,
            "callback_external_POSTs": None,
            "config": asdict(self.config),
            "priority": list(AGENTS),
            "preallocated_uniforms": list(self.uniforms),
            "memory_semantics": "accepted_order_submissions_not_fills_or_rejected_intents",
            "denominators": {
                "potential_opportunities": 12,
                "externally_offered": sum(r["external_opportunity"] for r in opportunities),
                "blocked": sum(
                    r["status"] == "blocked_external_opportunity" for r in opportunities
                ),
                "probability_callbacks": sum(r["question"] is not None for r in opportunities),
                "accepted_submissions": len(orders),
                "rejected_submissions": sum(r["status"] == "rejected" for r in opportunities),
            },
            "events": deepcopy(self.events),
            "orders": orders,
            "trades": deepcopy(self.trade_history),
            "accounts_final": self.accounts(),
            "dividends": deepcopy(self.dividend_history),
            "rp_metrics": market_metrics(
                [float(PRICE)] + [float(event["mark_price"]) for event in self.events],
                [50] + [event["expected_remaining_dividends"] for event in self.events],
                accepted_orders=orders,
                trades=self.trade_history,
                evidence_kind="invented_execution_demo_fixed_quote_carry_forward_mark",
            ),
            "human_probability_error": unknown("no_human_targets_in_invented_demo"),
            "human_alignment": unknown("no_human_targets_in_invented_demo"),
            "limits": [
                "Only two accounts and first two of ten periods; shares do not expire early.",
                "Opportunities, price50, quantities, priority and dividends are externally fixed.",
                "The adapter itself loads no human rows, fitted predictor or static I responses.",
                "Caller-supplied callback provenance and external calls are not attested here.",
                "Predictor is trusted local code receiving only fresh filtered messages.",
                "Mark price carries forward when no trade; not an observed market transaction.",
                "A legal-order mapping is not a validated autonomous human trading policy.",
            ],
        }


def run_demo(predictors: Mapping[str, Predictor], arm: str, config: DemoConfig = DEFAULT_CONFIG):
    demo = ConditionalExecutionDemo(predictors, arm, config)
    for _ in TICKS:
        demo.step()
    return demo.result()
