"""Read-only accounting audit of a completed finite-rule run; never calls a model.

Checks local records against their hashes and independently recomputes cash, shares,
marks, review returns and block resets. This cannot authenticate external liquidity,
model inference or human behavior. Missing or inconsistent evidence fails closed.
"""

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd

from behavioral_market.data.calendar import sessions
from behavioral_market.simulation.finite_rule_pair import METHOD, settings


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(
        not isinstance(value, bool) and isinstance(value, (str, int, float, Decimal)),
        "invalid accounting number",
    )
    result = Decimal(str(value))
    require(result.is_finite(), "nonfinite accounting number")
    return result


def near(actual, expected, tolerance="0.000000000001"):
    require(
        abs(number(actual) - number(expected)) <= Decimal(tolerance),
        f"accounting mismatch: {actual} versus {expected}",
    )


def count(value, name):
    require(
        not isinstance(value, bool) and isinstance(value, (int, float, Decimal)),
        name + " must be an integer count",
    )
    parsed = number(value)
    require(
        parsed >= 0 and parsed == parsed.to_integral_value(),
        name + " must be a nonnegative integer count",
    )
    return int(parsed)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_order_contract(row, before, cp_before, spec):
    requested, validated = row["requested_order"], row["validated_order"]
    for decision in [requested, validated]:
        require(decision["action"] in {"buy", "sell", "hold"}, "invalid replay action")
        require(type(decision["quantity"]) is int, "invalid replay order quantity")
        if decision["action"] == "hold":
            require(
                decision["quantity"] == 0 and decision["limit_price"] is None,
                "HOLD cannot contain an order",
            )
        else:
            require(
                decision["quantity"] > 0 and number(decision["limit_price"]) > 0,
                "invalid replay order",
            )
    violation = None
    if requested["action"] != "hold":
        price = number(requested["limit_price"])
        band = row["permitted_price_band"]
        if not number(band["minimum"]) <= price <= number(band["maximum"]):
            violation = "price_band_violation"
        elif (
            requested["action"] == "buy" and price * requested["quantity"] > number(before["cash"])
        ) or (requested["action"] == "sell" and requested["quantity"] > before["position"]):
            violation = "constraint_violation"
    require(row["constraint_violation"] == violation, "incorrect replay constraint result")
    if violation:
        require(validated["action"] == "hold", "invalid replay order was not held")
    else:
        require(
            all(validated[k] == requested[k] for k in ("action", "quantity", "limit_price")),
            "validated replay order differs without a constraint violation",
        )
    orders = row["orders_before_expiry"]
    for order in orders:
        require(count(order["quantity"], "order quantity") > 0, "accepted order must be positive")
        count(order["remaining_quantity"], "remaining quantity")
    count(row["assigned_quantity"], "assigned quantity")
    count(row["expired_orders"], "expired orders")
    focal = [o for o in orders if o["agent_id"] == "RULE"]
    quotes = [o for o in orders if o["agent_id"] == "ASSIGNED_COUNTERPARTY"]
    require(len(orders) == len(focal) + len(quotes), "unexpected replay order owner")
    quote, quantity, status = None, 0, "no_order"
    if validated["action"] == "hold":
        require(not focal, "HOLD has an accepted order")
    else:
        require(len(focal) == 1, "active replay decision needs exactly one accepted order")
        order = focal[0]
        require(
            order["side"] == validated["action"] and order["quantity"] == validated["quantity"],
            "accepted replay order differs from validated decision",
        )
        near(order["price"], validated["limit_price"])
        reference, spread = (
            number(row["opening_observation"]["last_close"]),
            number(spec["half_spread"]),
        )
        assigned = reference + spread if validated["action"] == "buy" else reference - spread
        if assigned > 0:
            capacity = (
                cp_before["position"]
                if validated["action"] == "buy"
                else int(number(cp_before["cash"]) // assigned)
            )
            quantity = min(validated["quantity"], capacity)
            quote, status = (
                (assigned, "assigned_quote_submitted")
                if quantity
                else (None, "counterparty_exhausted")
            )
        else:
            status = "no_positive_assigned_quote"
    require(
        row["assigned_quantity"] == quantity and row["quote_status"] == status,
        "assigned quote size/status mismatch",
    )
    if quote is None:
        require(row["assigned_quote"] is None and not quotes, "unexpected assigned quote")
    else:
        near(row["assigned_quote"], quote)
        require(len(quotes) == 1, "assigned quote not accepted exactly once")
        order = quotes[0]
        require(
            order["side"] == ("sell" if validated["action"] == "buy" else "buy")
            and order["quantity"] == quantity,
            "assigned order side/size mismatch",
        )
        near(order["price"], quote)
    crossing = quote is not None and (
        quote <= number(validated["limit_price"])
        if validated["action"] == "buy"
        else quote >= number(validated["limit_price"])
    )
    require(
        len(row["trades"]) == int(bool(crossing)), "matching result contradicts accepted quotes"
    )
    if crossing:
        require(
            row["trades"][0]["quantity"] == quantity,
            "matching quantity differs from available quote",
        )


def verify_day(row, previous, spec):
    """Recompute both sides' cash and shares from linked fills, without Exchange."""
    before, after = row["portfolio_before_order"], row["portfolio_after_close"]
    counterparty = row["counterparty_after_close"]
    if previous:
        require(before == previous["portfolio_after_close"], "replay account discontinuity")
        cp_before = previous["counterparty_after_close"]
    else:
        cp_before = {"cash": spec["counterparty_cash"], "position": spec["counterparty_shares"]}
    verify_order_contract(row, before, cp_before, spec)
    cash, shares = number(before["cash"]), count(before["position"], "replay starting shares")
    cp_cash, cp_shares = (
        number(cp_before["cash"]),
        count(cp_before["position"], "counterparty starting shares"),
    )
    after_shares = count(after["position"], "replay closing shares")
    after_cp_shares = count(counterparty["position"], "counterparty closing shares")
    cost = number(before["avg_cost"])
    realized = number(before["realized_pnl"])
    orders = {o["order_id"]: o for o in row["orders_before_expiry"]}
    require(len(orders) == len(row["orders_before_expiry"]), "duplicate order identity")
    filled = dict.fromkeys(orders, 0)
    trade_ids = set()
    for trade in row["trades"]:
        require(trade["trade_id"] not in trade_ids, "duplicate trade identity")
        trade_ids.add(trade["trade_id"])
        buy, sell = orders[trade["buy_order_id"]], orders[trade["sell_order_id"]]
        price, qty = number(trade["price"]), trade["quantity"]
        require(type(qty) is int and qty > 0 and price > 0, "invalid fill size/price")
        require(buy["side"] == "buy" and sell["side"] == "sell", "fill order sides mismatch")
        require(
            buy["agent_id"] == trade["buyer_id"] and sell["agent_id"] == trade["seller_id"],
            "fill account identity mismatch",
        )
        require(number(sell["price"]) <= price <= number(buy["price"]), "fill outside limit prices")
        near(price, row["assigned_quote"])
        for order in [buy, sell]:
            filled[order["order_id"]] += qty
            require(filled[order["order_id"]] <= order["quantity"], "overfilled order")
        if trade["buyer_id"] == "RULE" and trade["seller_id"] == "ASSIGNED_COUNTERPARTY":
            cost = (cost * shares + price * qty) / (shares + qty)
            cash -= price * qty
            shares += qty
            cp_cash += price * qty
            cp_shares -= qty
        elif trade["seller_id"] == "RULE" and trade["buyer_id"] == "ASSIGNED_COUNTERPARTY":
            realized += (price - cost) * qty
            cash += price * qty
            shares -= qty
            cost = cost if shares else Decimal(0)
            cp_cash -= price * qty
            cp_shares += qty
        else:
            raise ValueError("unexpected replay participant")
        require(min(cash, shares, cp_cash, cp_shares) >= 0, "negative replay balance")
    for key, expected in [
        ("cash", cash),
        ("avg_cost", cost),
        ("realized_pnl", realized),
    ]:
        near(after[key], expected)
    require(after_shares == shares, "replay closing shares mismatch")
    expiries = 0
    for oid, order in orders.items():
        remaining = order["quantity"] - filled[oid]
        require(order["remaining_quantity"] == remaining, "order remainder mismatch")
        expected_status = (
            "filled" if not remaining else ("partially_filled" if filled[oid] else "open")
        )
        require(order["status"] == expected_status, "order status differs from fills")
        expiries += int(remaining > 0)
    require(row["expired_orders"] == expiries, "expired order count mismatch")
    near(counterparty["cash"], cp_cash)
    require(after_cp_shares == cp_shares, "counterparty closing shares mismatch")
    mark = number(row["market_close_used_after_order"])
    require(mark > 0, "invalid mark")
    near(after["total_equity"], cash + shares * mark)
    near(after["unrealized_pnl"], (mark - cost) * shares)
    near(counterparty["total_equity"], cp_cash + cp_shares * mark)
    # A declared conservation flag cannot substitute for these computations.
    near(cash + cp_cash, number(before["cash"]) + number(cp_before["cash"]))
    require(
        shares + cp_shares == before["position"] + cp_before["position"], "shares not conserved"
    )


def verify_actual_days(config, dates, market, actual, trace_rows, ids, orders, trades):
    """Rebuild observed portfolios from the actual run's trades, independently."""
    order_map = {(r["agent_id"], r["session_date"]): r for r in orders}
    for order in orders:
        count(order["quantity"], "actual order quantity")
        count(order["filled_quantity"], "actual filled quantity")
    require(
        len(order_map) == len(orders) and set(order_map) == set(actual),
        "actual order evidence missing or duplicate",
    )
    require(len({t["trade_id"] for t in trades}) == len(trades), "duplicate actual trade")
    require(all(t["session_date"] in dates for t in trades), "trade outside run schedule")
    balances = {}
    for agent_id in ids:
        initial = trace_rows[agent_id, dates[0], "current_rule"]["portfolio_before_order"]
        balances[agent_id] = {
            "cash": number(config["population"]["initial_cash"]),
            "position": count(config["population"]["initial_shares"], "initial shares"),
            "avg_cost": number(initial["avg_cost"]),
            "realized_pnl": Decimal(0),
        }
    computed = {}
    for i, date in enumerate(dates):
        filled = dict.fromkeys(ids, 0)
        for trade in [t for t in trades if t["session_date"] == date]:
            price, qty = number(trade["price"]), trade["quantity"]
            require(price > 0 and type(qty) is int and qty > 0, "invalid actual trade")
            for side, key in [("buy", "buyer_id"), ("sell", "seller_id")]:
                agent_id = trade[key]
                if agent_id not in balances:
                    require(
                        config["run"]["mode"] == "historical" and agent_id == "external_market",
                        "unrecorded actual trade participant",
                    )
                    continue
                order = order_map[agent_id, date]
                require(order["action"] == side, "actual trade does not match submitted side")
                limit = number(order["limit_price"])
                require(
                    price <= limit if side == "buy" else price >= limit,
                    "actual trade outside order limit",
                )
                filled[agent_id] += qty
                require(filled[agent_id] <= order["quantity"], "actual order overfilled")
                balance = balances[agent_id]
                if side == "buy":
                    balance["avg_cost"] = (
                        balance["avg_cost"] * balance["position"] + price * qty
                    ) / (balance["position"] + qty)
                    balance["cash"] -= price * qty
                    balance["position"] += qty
                else:
                    balance["realized_pnl"] += (price - balance["avg_cost"]) * qty
                    balance["cash"] += price * qty
                    balance["position"] -= qty
                    if not balance["position"]:
                        balance["avg_cost"] = Decimal(0)
                require(
                    balance["cash"] >= 0 and balance["position"] >= 0,
                    "negative actual cash/holdings",
                )
        for agent_id, balance in balances.items():
            require(
                filled[agent_id] == order_map[agent_id, date]["filled_quantity"],
                "actual order fill count mismatch",
            )
            mark = number(market[i]["close"])
            balance["total_equity"] = balance["cash"] + balance["position"] * mark
            balance["unrealized_pnl"] = (mark - balance["avg_cost"]) * balance["position"]
            exported = actual[agent_id, date]
            for key, value in balance.items():
                if key == "position":
                    require(
                        count(exported[key], "actual closing shares") == value,
                        "actual closing shares mismatch",
                    )
                else:
                    near(exported[key], value, "0.000001")
            computed[agent_id, date] = dict(balance)
    return computed


def verify_hybrid_reviews(directory, reviews, ids, config, metadata, actual, trace, market):
    """Bind applied choices to the ordered local review log, not remote provenance."""
    entries = [
        json.loads(line)
        for line in (directory / "hybrid_review_batches.jsonl").read_text().splitlines()
    ]
    dates = sorted({date for _, date in reviews})
    all_dates = [row["session_date"] for row in market]
    calendar = sessions(
        config["data"]["experiment_start"],
        config["data"]["experiment_end"],
        config["data"].get("calendar", "XNYS"),
    )
    interval = config["behavior"]["strategy_switch_interval"]
    _, spec = settings(config["behavior"])
    cursor = 0

    def take(phase, date):
        nonlocal cursor
        require(cursor < len(entries), "hybrid review log truncated")
        entry = entries[cursor]
        cursor += 1
        require(
            entry.get("phase") == phase and entry.get("batch") == date,
            "hybrid review log order/phase mismatch",
        )
        return entry

    for date in dates:
        prepared = take("prepared", date)["requests"]
        require(
            [item["identity"]["agent_id"] for item in prepared] == ids,
            "hybrid prepared identities mismatch",
        )
        for index, item in enumerate(prepared):
            identity = item["identity"]
            review = reviews[identity["agent_id"], date]
            payload = json.loads(item["user"])
            expected_identity = {
                "agent_id": ids[index],
                "session_date": date,
                "treatment": "hybrid_style_llm",
                "kind": "style",
                "mode": "endogenous",
                "seed": config["experiment"]["seed"],
                "shock": config["fundamental_shock"]["magnitude"],
                "event_condition": metadata["event_condition"],
                "decision_batch": None,
                "strategy_return_method": METHOD,
                "prompt_version": "memory-market-finite-pair-v1",
                "order_source": "rule",
                "style_source": "model_interface",
            }
            if config["behavior"].get("mean_field_visibility", "visible") == "hidden":
                expected_identity["mean_field_visibility"] = "hidden"
            require(
                identity == expected_identity
                and payload["simulation_seed"] == config["experiment"]["seed"]
                and payload["agent_state"]["current_style"] == review["from_style"]
                and payload["agent_state"]["persona"] == {"agent_id": identity["agent_id"]},
                "hybrid frozen prompt identity/style mismatch",
            )
            state, observation = payload["agent_state"], payload["observation"]
            account = actual[ids[index], date]
            wallet = state["portfolio"]
            keys = {
                "cash",
                "position",
                "avg_cost",
                "realized_pnl",
                "unrealized_pnl",
                "total_equity",
            }
            require(
                set(wallet) == keys
                and type(wallet["position"]) is int
                and wallet["position"] == account["position"],
                "hybrid prompt holdings mismatch",
            )
            for key in keys - {"position"}:
                near(wallet[key], account[key], "0.000001")
            day = all_dates.index(date)
            block_start = all_dates[day + 1 - interval]
            cf = state["counterfactual"]
            start = trace[ids[index], block_start, "current_rule"]["portfolio_before_order"]
            require(
                cf["method"] == METHOD
                and cf["block_index"] == day // interval
                and cf["completed_sessions"] == interval
                and cf["current_rule_style"] == review["from_style"]
                and cf["alternative_rule_style"]
                == trace[ids[index], date, "alternative_rule"]["style"]
                and set(cf["block_start_portfolio"]) == keys,
                "hybrid prompt finite block/style mismatch",
            )
            require(
                type(cf["block_start_portfolio"]["position"]) is int
                and cf["block_start_portfolio"]["position"] == start["position"],
                "hybrid prompt starting holdings mismatch",
            )
            for key in keys - {"position"}:
                near(cf["block_start_portfolio"][key], start[key])
            for key in (
                "current_rule_block_return",
                "alternative_rule_block_return",
                "observed_account_return",
            ):
                near(cf[key], review[key])
            near(cf["gap"], review["paired_return_gap"])
            require(
                cf["assumptions"]
                == {
                    **spec,
                    "fees": "0",
                    "rule_noise": "0",
                    "quote_source": "assigned_from_opening_reference_not_observed_LOB",
                    "counterparty_reset": "each_review_block",
                    "mark_source": "realized_market_path_shared_by_both_replays",
                    "orders_expire": "each_session",
                },
                "hybrid prompt finite assumptions mismatch",
            )
            require(
                observation["review_block"]
                == {
                    "start_session": block_start,
                    "end_session": date,
                    "completed_sessions": interval,
                }
                and observation["phase"] == "post_close_style_review"
                and observation["history_last_session"] == observation["session_date"] == date
                and observation["decision_at_utc"] == calendar.loc[date].market_close.isoformat()
                and review["decision_at_utc"] == observation["decision_at_utc"]
                and "permitted_price_band" not in observation,
                "hybrid prompt review timing mismatch",
            )
            near(observation["last_close"], market[day]["close"])
            if config["behavior"].get("mean_field_visibility", "visible") == "hidden":
                require(
                    "mean_field" not in payload["observation"]
                    and payload["observation"].get("mean_field_availability")
                    == "withheld_by_experiment",
                    "hybrid hidden population leaked",
                )
            require(take("attempt", date)["index"] == index, "hybrid attempt order mismatch")
            returned = take("returned_unapplied", date)
            require(returned["index"] == index, "hybrid response order mismatch")
            decision, meta = returned["decision"], returned["metadata"]
            require(
                meta["status"] == "ok"
                and meta["fallback"] is False
                and meta["cache_hit"] is False
                and type(meta["repair_count"]) is int
                and meta["repair_count"] == 0,
                "hybrid response was repaired/cached/invalid",
            )
            require(
                all(
                    decision[key] == review[key]
                    for key in ("decision", "target_style", "rationale")
                )
                and meta.get("request_hash") == review["request_hash"],
                "hybrid pending and applied response differ",
            )
        require(
            take("validated_unapplied", date)["count"] == len(ids)
            and take("applied", date)["count"] == len(ids),
            "hybrid incomplete review committed",
        )
    require(cursor == len(entries), "hybrid review log contains unexpected entries")


def audit(directory):
    metadata = load(directory / "run_metadata.json")
    config = load(directory / "config_snapshot.json")
    require(metadata["status"] == "complete", "run is incomplete")
    count(metadata["sessions"], "sessions")
    count(metadata["n_agents"], "agents")
    count(config["population"]["initial_shares"], "initial shares")
    count(config["population"]["n_agents"], "configured agents")
    require(
        metadata.get("strategy_return_method") == METHOD
        and config.get("run", {}).get("strategy_return_method") == METHOD,
        "explicit matching finite method declarations required",
    )
    method, spec = settings(config["behavior"])
    declarations = [metadata, config["behavior"], config["run"]]
    hybrid = any(
        item.get("treatment") == "hybrid_style_llm"
        or "order_source" in item
        or "style_source" in item
        for item in (metadata, config["run"])
    )
    if hybrid:
        require(
            all(
                item.get("treatment") == "hybrid_style_llm"
                and item.get("mode") == "endogenous"
                and item.get("order_source") == "rule"
                and item.get("style_source") == "model_interface"
                and item.get("review_protocol") == "sequential_no_retry_v1"
                and item.get("review_provenance") in {"scripted", "live_strict"}
                for item in (metadata, config["run"])
            )
            and metadata["review_provenance"] == config["run"]["review_provenance"],
            "hybrid source declarations mismatch",
        )
    adaptation = None
    if any("rule_style_adaptation" in item for item in declarations):
        adaptation = config["behavior"].get("rule_style_adaptation")
        require(
            adaptation in ("fixed", "adaptive")
            and all(item.get("rule_style_adaptation") == adaptation for item in declarations)
            and config["run"]["treatment"] == "rule_abm",
            "rule adaptation declarations mismatch",
        )
    require(
        method == METHOD and metadata.get("finite_rule_pair") == spec,
        "execution assumptions mismatch",
    )
    require(
        hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        == metadata["config_hash"],
        "config hash mismatch",
    )
    required = {
        "agents.parquet",
        "style_switches.parquet",
        "market.parquet",
        "finite_rule_replay.jsonl",
        "orders.parquet",
        "trades.parquet",
    }
    if hybrid:
        required.add("hybrid_review_batches.jsonl")
    manifest = metadata["artifact_hashes"]
    require(required <= set(manifest), "required evidence hash missing")
    for name, expected in manifest.items():
        require(Path(name).name == name and name not in {".", ".."}, "invalid artifact name")
        require(
            (directory / name).is_file() and sha(directory / name) == expected,
            "artifact missing or hash mismatch: " + name,
        )
    require(
        isinstance(metadata.get("source_hash"), str) and len(metadata["source_hash"]) == 64,
        "source hash missing",
    )
    agents = pd.read_parquet(directory / "agents.parquet").to_dict("records")
    switches = pd.read_parquet(directory / "style_switches.parquet").to_dict("records")
    market = pd.read_parquet(directory / "market.parquet").to_dict("records")
    orders = pd.read_parquet(directory / "orders.parquet").to_dict("records")
    trades = pd.read_parquet(directory / "trades.parquet").to_dict("records")
    trace = [
        json.loads(line)
        for line in (directory / "finite_rule_replay.jsonl").read_text().splitlines()
    ]
    dates = [r["session_date"] for r in market]
    require(
        dates == sorted(set(dates)) and len(dates) == metadata["sessions"],
        "market dates missing, unordered or duplicated",
    )
    expected_dates = list(
        sessions(
            config["data"]["experiment_start"],
            config["data"]["experiment_end"],
            config["data"].get("calendar", "XNYS"),
        ).index
    )
    require(dates == expected_dates, "market dates do not cover configured schedule")
    ids = sorted({row["agent_id"] for row in agents})
    require(
        len(ids) == metadata["n_agents"] == config["population"]["n_agents"], "agent count mismatch"
    )
    actual = {(r["agent_id"], r["session_date"]): r for r in agents}
    expected_actual = {(agent, day) for agent in ids for day in dates}
    require(
        len(actual) == len(agents) and set(actual) == expected_actual,
        "actual daily account evidence missing or duplicate",
    )
    rows = {(r["agent_id"], r["session_date"], r["branch"]): r for r in trace}
    expected_rows = {
        (agent, day, branch)
        for agent, day in expected_actual
        for branch in ("current_rule", "alternative_rule")
    }
    require(
        len(rows) == len(trace) and set(rows) == expected_rows,
        "replay daily evidence missing or duplicate",
    )
    computed_actual = verify_actual_days(config, dates, market, actual, rows, ids, orders, trades)
    reviews = {(r["agent_id"], r["session_date"]): r for r in switches}
    interval = config["behavior"]["strategy_switch_interval"]
    expected_reviews = {
        (agent, date) for agent in ids for i, date in enumerate(dates) if (i + 1) % interval == 0
    }
    require(
        len(reviews) == len(switches) and set(reviews) == expected_reviews,
        "review evidence missing or duplicate",
    )
    if hybrid:
        require(
            all(
                row["validation_status"] == "rule"
                and not row["fallback"]
                and row["prompt_mean_field_supplied"] is None
                for row in orders
            ),
            "hybrid orders must come from rules without model prompts",
        )
        verify_hybrid_reviews(
            directory, reviews, ids, config, metadata, computed_actual, rows, market
        )
    summaries = []
    for agent_id in ids:
        previous, starts, styles = {}, {}, {}
        for i, date in enumerate(dates):
            block = i // interval
            account = actual[agent_id, date]
            require(
                account["strategy_return_method"] == METHOD
                and account["review_block_index"] == block,
                "daily method/block mismatch",
            )
            daily = [
                rows[agent_id, date, branch] for branch in ("current_rule", "alternative_rule")
            ]
            require(
                daily[0]["opening_observation"] == daily[1]["opening_observation"]
                and daily[0]["permitted_price_band"] == daily[1]["permitted_price_band"],
                "paired rules used different opening information",
            )
            for row in daily:
                branch = row["branch"]
                require(
                    row["method"] == METHOD and row["block_index"] == block,
                    "trace method/block mismatch",
                )
                require(row["history_last_session"] < date, "future replay information")
                near(row["market_close_used_after_order"], market[i]["close"])
                if i:
                    near(row["opening_observation"]["last_close"], market[i - 1]["close"], "1e-8")
                if i % interval == 0:
                    previous[branch] = None
                    starts[branch] = row["portfolio_before_order"]
                    styles[branch] = row["style"]
                    if i:
                        before_actual = actual[agent_id, dates[i - 1]]
                        for key in starts[branch]:
                            near(starts[branch][key], before_actual[key], "0.000001")
                    else:
                        near(starts[branch]["cash"], config["population"]["initial_cash"])
                        require(
                            starts[branch]["position"] == config["population"]["initial_shares"],
                            "initial shares mismatch",
                        )
                        near(
                            starts[branch]["avg_cost"],
                            row["opening_observation"]["last_close"]
                            if starts[branch]["position"]
                            else 0,
                            "1e-8",
                        )
                    # Opening features round price to eight decimals. The reset
                    # wealth uses the full previous close (or initial cost mark).
                    mark = market[i - 1]["close"] if i else starts[branch]["avg_cost"]
                    near(
                        starts[branch]["total_equity"],
                        number(starts[branch]["cash"]) + starts[branch]["position"] * number(mark),
                    )
                require(row["style"] == styles[branch], "replay style changed inside a block")
                verify_day(row, previous[branch], spec)
                previous[branch] = row
            require(starts["current_rule"] == starts["alternative_rule"], "paired starts differ")
            require(
                styles["current_rule"] == account["current_style"]
                and styles["alternative_rule"] in {"fundamental", "technical"}
                and styles["alternative_rule"] != styles["current_rule"],
                "paired styles mismatch",
            )
            start = number(starts["current_rule"]["total_equity"])
            require(start > 0, "nonpositive block starting wealth")
            current = number(daily[0]["portfolio_after_close"]["total_equity"])
            alternative = number(daily[1]["portfolio_after_close"]["total_equity"])
            calculated = {
                "observed_account_return": computed_actual[agent_id, date]["total_equity"] / start
                - 1,
                "current_rule_block_return": current / start - 1,
                "alternative_rule_block_return": alternative / start - 1,
                "paired_return_gap": (alternative - current) / start,
            }
            for key, value in calculated.items():
                # Actual account export rounds currency to six decimal places.
                tolerance = max(Decimal("1e-12"), Decimal("0.000001") / start)
                near(account[key], value, str(tolerance))
            if (agent_id, date) in reviews:
                review = reviews[agent_id, date]
                if hybrid:
                    require(
                        review.get("order_source") == "rule"
                        and review.get("style_source") == "model_interface"
                        and review.get("review_provenance") == metadata["review_provenance"]
                        and review.get("review_protocol") == metadata["review_protocol"]
                        and review["decision_source"] == "injected_client"
                        and review["choice_valid"]
                        and not review["fallback"]
                        and review["validation_status"] == "ok"
                        and review["cache_hit"] is False
                        and review["repair_count"] == 0,
                        "hybrid review source/validity mismatch",
                    )
                if adaptation is not None:
                    require(
                        review.get("rule_style_adaptation") == adaptation
                        and review["decision_source"]
                        == ("fixed_rule" if adaptation == "fixed" else "rule"),
                        "review rule adaptation/source mismatch",
                    )
                    if adaptation == "fixed":
                        require(review["decision"] == "stay", "fixed rule policy switched strategy")
                require(
                    review["strategy_return_method"] == METHOD
                    and review["review_block_index"] == block,
                    "review method/block mismatch",
                )
                require(
                    review["from_style"] == styles["current_rule"], "review origin style mismatch"
                )
                require(
                    review["decision"] in {"stay", "switch"}
                    and review["target_style"] in {"fundamental", "technical"},
                    "invalid applied strategy choice",
                )
                require(
                    (review["decision"] == "stay")
                    == (review["target_style"] == review["from_style"]),
                    "applied decision and target style contradict each other",
                )
                require(
                    type(review["choice_valid"]) is bool and type(review["fallback"]) is bool,
                    "invalid review validity markers",
                )
                expected_valid = (
                    review["requested_decision"] in {"stay", "switch"}
                    and review["requested_target_style"] in {"fundamental", "technical"}
                    and (
                        (review["requested_decision"] == "stay")
                        == (review["requested_target_style"] == review["from_style"])
                    )
                    and review["validation_status"] in {"ok", "mock", "rule"}
                    and not review["fallback"]
                )
                require(
                    review["choice_valid"] == expected_valid,
                    "choice_valid does not match requested choice and response status",
                )
                if not review["choice_valid"] or review["fallback"]:
                    require(
                        not review["choice_valid"] and review["decision"] == "stay",
                        "invalid/fallback review changed the applied strategy",
                    )
                else:
                    require(
                        review["validation_status"] in {"ok", "mock", "rule"},
                        "valid review has invalid response status",
                    )
                    require(
                        review["requested_decision"] == review["decision"]
                        and review["requested_target_style"] == review["target_style"],
                        "valid requested choice differs from applied choice",
                    )
                for key in calculated:
                    near(review[key], account[key])
                if i + 1 < len(dates):
                    require(
                        actual[agent_id, dates[i + 1]]["current_style"] == review["target_style"],
                        "next block style does not match applied review",
                    )
                summaries.append(
                    {
                        "agent_id": agent_id,
                        "block_index": block,
                        "start_session": dates[i + 1 - interval],
                        "end_session": date,
                        "block_start_cash": starts["current_rule"]["cash"],
                        "block_start_shares": starts["current_rule"]["position"],
                        "block_start_equity": str(start),
                        **{key: str(value) for key, value in calculated.items()},
                        "decision_source": review["decision_source"],
                        "decision": review["decision"],
                        "target_style": review["target_style"],
                    }
                )
    return {
        "status": "accounting_checks_passed",
        "method": METHOD,
        "source_hash": metadata["source_hash"],
        "config_hash": metadata["config_hash"],
        "run_metadata_sha256": sha(directory / "run_metadata.json"),
        "checked_agent_days": len(agents),
        "checked_branch_days": len(trace),
        "checked_reviews": len(summaries),
        "reviews": summaries,
        "incomplete_final_block_sessions": len(dates) % interval,
        "new_model_POSTs": 0,
        "verification_scope": "cash/shares/cost/marks, declared quote and accepted-order "
        "contracts, applied review and reset consistency; not rule-signal reproduction "
        "or authentication of original model dispatches",
        "limits": "Local accounting consistency only. Hashes bind supplied local files; "
        "they do not authenticate external data or independent model inference. "
        "Assigned quotes and fixed realized prices do not establish human or market realism.",
    }


def describe(result):
    lines = [
        "完整模拟的逐笔账本复核（本次核对在本地运行，没有调用模型）",
        "",
        f"核对了{result['checked_agent_days']}份每日模拟账户、"
        f"{result['checked_branch_days']}份每日规则账户和{result['checked_reviews']}次复盘。",
        "现金、股数、成交价格、收盘估值和下一段的起点均可对上。",
        "",
    ]
    for row in result["reviews"]:
        lines.append(
            f"账户{row['agent_id']}，{row['start_session']}至{row['end_session']}："
            f"本账户收益{number(row['observed_account_return']):.4%}；"
            f"当前规则{number(row['current_rule_block_return']):.4%}；"
            f"替代规则{number(row['alternative_rule_block_return']):.4%}；"
            f"两规则相差{number(row['paired_return_gap']):.4%}。"
        )
    lines += [
        "",
        "这份报告只核对账目。预设程序的选择不能写成AI回答；指定的交易对手报价"
        "不能写成真实盘口。尚未证明投资者更像真人或市场更真实。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.output / "REPORT.txt").write_text(describe(result))
    print(json.dumps({"status": result["status"], "reviews": result["checked_reviews"]}))


if __name__ == "__main__":
    main()
