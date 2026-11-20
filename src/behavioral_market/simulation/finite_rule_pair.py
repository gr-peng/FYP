"""Conditional rule comparisons with assigned quotes, separate from the actual market.

Each review block clones the observed account into two independent rule accounts.
Their counterparties are finite and persist within that block. Orders use only the
opening observation; the realized market close is supplied afterwards for marking.
This is not a counterfactual price path or an estimate of real-world execution.
"""

import copy
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from types import SimpleNamespace

from behavioral_market.agents.policy import enforce_portfolio, rule_order
from behavioral_market.market import Exchange, Portfolio

LEGACY_METHOD = "long_cash_signal_proxy"
METHOD = "finite_rule_pair_assigned_quotes_v1"
PROMPT_VERSION = "memory-market-finite-pair-v1"


def decimal_value(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not result.is_finite() or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def settings(behavior):
    """Validate before creating clients, outputs or reading research datasets."""
    method = behavior.get("strategy_return_method", LEGACY_METHOD)
    if method not in (LEGACY_METHOD, METHOD):
        raise ValueError("unsupported behavior.strategy_return_method")
    if method == LEGACY_METHOD:
        if "finite_rule_pair" in behavior:
            raise ValueError("finite_rule_pair requires its explicit strategy_return_method")
        return method, None
    spec = behavior.get("finite_rule_pair")
    if not isinstance(spec, dict) or set(spec) != {
        "half_spread",
        "counterparty_cash",
        "counterparty_shares",
    }:
        raise ValueError("finite_rule_pair requires explicit spread and counterparty endowment")
    interval = behavior.get("strategy_switch_interval")
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("strategy_switch_interval must be a positive integer")
    shares = spec["counterparty_shares"]
    if isinstance(shares, bool) or not isinstance(shares, int) or shares < 0:
        raise ValueError("counterparty_shares must be a nonnegative integer")
    return method, {
        "half_spread": str(decimal_value(spec["half_spread"], "half_spread")),
        "counterparty_cash": str(decimal_value(spec["counterparty_cash"], "counterparty_cash")),
        "counterparty_shares": shares,
    }


class ZeroNoise:
    def gauss(self, *_):
        return 0.0


def portfolio_record(portfolio):
    return {
        key: str(value) if isinstance(value, Decimal) else value
        for key, value in asdict(portfolio).items()
    }


class FiniteRulePair:
    def __init__(self, agent, reference, spec, block_index):
        reference = decimal_value(reference, "block starting mark", positive=True)
        self.spec = copy.deepcopy(spec)
        self.agent_id = agent.persona.agent_id
        self.block_index = block_index
        self.initial_portfolio = copy.deepcopy(agent.portfolio)
        self.initial_portfolio.mark_to_market(reference)
        self.start_equity = self.initial_portfolio.total_equity
        if self.start_equity <= 0:
            raise ValueError("finite rule comparison needs positive starting equity")
        self.branches = {}
        other = "technical" if agent.current_style == "fundamental" else "fundamental"
        for name, style in (("current_rule", agent.current_style), ("alternative_rule", other)):
            account = copy.deepcopy(self.initial_portfolio)
            counterparty = Portfolio(
                spec["counterparty_cash"],
                spec["counterparty_shares"],
                reference if spec["counterparty_shares"] else 0,
            )
            self.branches[name] = {
                "agent": SimpleNamespace(
                    persona=agent.persona, portfolio=account, current_style=style
                ),
                "counterparty": counterparty,
                "total_cash": account.cash + counterparty.cash,
                "total_shares": account.position + counterparty.position,
            }
        self.pending = None
        self.completed_sessions = 0
        self.last_session = None

    def open_session(self, observation):
        if self.pending is not None:
            raise ValueError("previous finite replay session has no closing mark")
        session = observation["session_date"]
        if observation["history_last_session"] >= session or (
            self.last_session is not None and session <= self.last_session
        ):
            raise ValueError(
                "finite replay requires prior completed history and increasing sessions"
            )
        reference = decimal_value(observation["last_close"], "opening reference", positive=True)
        # Copy only pre-open rule inputs. Never pass a closing price to rule_order.
        inputs = {
            key: copy.deepcopy(observation[key])
            for key in ("last_close", "fundamental_value", "momentum_5")
        }
        decimal_value(inputs["fundamental_value"], "fundamental_value", positive=True)
        momentum = inputs["momentum_5"]
        if momentum is not None and not Decimal(str(momentum)).is_finite():
            raise ValueError("momentum_5 must be finite or unavailable")
        band = observation["permitted_price_band"]
        spread = Decimal(self.spec["half_spread"])
        pending = []
        for name, branch in self.branches.items():
            agent, counterparty = branch["agent"], branch["counterparty"]
            before = portfolio_record(agent.portfolio)
            exchange = Exchange("CONDITIONAL_RULE_REPLAY", reference)
            exchange.register("RULE", agent.portfolio)
            exchange.register("ASSIGNED_COUNTERPARTY", counterparty)
            requested = rule_order(agent, inputs, ZeroNoise())
            decision, violation = enforce_portfolio(
                requested, agent.portfolio, band["minimum"], band["maximum"]
            )
            quote, quantity, reason = None, 0, "no_order"
            if decision.action != "hold":
                side = "sell" if decision.action == "buy" else "buy"
                assigned = reference + spread if side == "sell" else reference - spread
                if assigned > 0:
                    capacity = (
                        counterparty.position
                        if side == "sell"
                        else int(counterparty.cash // assigned)
                    )
                    quantity = min(decision.quantity, capacity)
                    if quantity:
                        quote, reason = str(assigned), "assigned_quote_submitted"
                        exchange.submit_order("ASSIGNED_COUNTERPARTY", side, assigned, quantity)
                    else:
                        reason = "counterparty_exhausted"
                else:
                    reason = "no_positive_assigned_quote"
                exchange.submit_order(
                    "RULE", decision.action, decision.limit_price, decision.quantity
                )
            orders = [
                {**asdict(o), "price": str(o.price), "side": o.side.value, "status": o.status.value}
                for o in exchange.orders
            ]
            trades = [{**asdict(t), "price": str(t.price)} for t in exchange.trades]
            expired = len(exchange.end_day())
            if (
                agent.portfolio.cash + counterparty.cash != branch["total_cash"]
                or agent.portfolio.position + counterparty.position != branch["total_shares"]
            ):
                raise RuntimeError("finite replay failed cash/share conservation")
            pending.append(
                {
                    "agent_id": self.agent_id,
                    "block_index": self.block_index,
                    "session_date": session,
                    "history_last_session": observation["history_last_session"],
                    "method": METHOD,
                    "branch": name,
                    "style": agent.current_style,
                    "opening_observation": inputs,
                    "permitted_price_band": copy.deepcopy(band),
                    "portfolio_before_order": before,
                    "requested_order": requested.model_dump(),
                    "validated_order": decision.model_dump(),
                    "constraint_violation": violation,
                    "assigned_quote": quote,
                    "assigned_quantity": quantity,
                    "quote_status": reason,
                    "orders_before_expiry": orders,
                    "trades": trades,
                    "expired_orders": expired,
                    "cash_and_shares_conserved": True,
                }
            )
        self.pending = pending

    def close_session(self, close):
        if self.pending is None:
            raise ValueError("no finite replay orders to close")
        close = decimal_value(close, "closing mark", positive=True)
        rows = self.pending
        for row in rows:
            branch = self.branches[row["branch"]]
            branch["agent"].portfolio.mark_to_market(close)
            branch["counterparty"].mark_to_market(close)
            row.update(
                market_close_used_after_order=str(close),
                portfolio_after_close=portfolio_record(branch["agent"].portfolio),
                counterparty_after_close=portfolio_record(branch["counterparty"]),
            )
        self.last_session = rows[0]["session_date"]
        self.completed_sessions += 1
        self.pending = None
        return rows

    def summary(self, observed_portfolio):
        if self.pending is not None:
            raise ValueError("cannot review an unclosed replay session")
        returns = {
            name: float(branch["agent"].portfolio.total_equity / self.start_equity - 1)
            for name, branch in self.branches.items()
        }
        return {
            "method": METHOD,
            "block_index": self.block_index,
            "completed_sessions": self.completed_sessions,
            "block_start_portfolio": portfolio_record(self.initial_portfolio),
            "current_rule_style": self.branches["current_rule"]["agent"].current_style,
            "alternative_rule_style": self.branches["alternative_rule"]["agent"].current_style,
            "current_rule_block_return": returns["current_rule"],
            "alternative_rule_block_return": returns["alternative_rule"],
            # Subtract exact account equities before converting to float; subtracting
            # 4% and 3% as floats can incorrectly cross the strict 1% switch threshold.
            "gap": float(
                (
                    self.branches["alternative_rule"]["agent"].portfolio.total_equity
                    - self.branches["current_rule"]["agent"].portfolio.total_equity
                )
                / self.start_equity
            ),
            "observed_account_return": float(
                observed_portfolio.total_equity / self.start_equity - 1
            ),
            "assumptions": {
                **self.spec,
                "fees": "0",
                "rule_noise": "0",
                "quote_source": "assigned_from_opening_reference_not_observed_LOB",
                "counterparty_reset": "each_review_block",
                "mark_source": "realized_market_path_shared_by_both_replays",
                "orders_expire": "each_session",
            },
            "interpretation": "Conditional rule comparison on a fixed realized price path; "
            "not a generated alternative market or guaranteed attainable return. "
            "Observed account return is separate and is not used in the paired gap.",
        }


def return_columns(agent):
    if agent.finite_review is None:
        return {
            "actual_block_return": float(agent.portfolio.total_equity) / agent.block_start_equity
            - 1,
            "counterfactual_block_return": agent.shadow_growth - 1,
        }
    review = agent.finite_review
    return {
        "strategy_return_method": METHOD,
        "review_block_index": review["block_index"],
        "observed_account_return": review["observed_account_return"],
        "current_rule_block_return": review["current_rule_block_return"],
        "alternative_rule_block_return": review["alternative_rule_block_return"],
        "paired_return_gap": review["gap"],
    }
