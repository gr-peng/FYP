"""Read-only descriptive diagnosis of the sealed six-run rule study; no model or simulation."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import pandas as pd

PINNED_SEAL_SHA = "eee9488282794856b5bef404cb14898163834fe916083296a2e4e0c0711fe6e3"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value):
    result = Decimal(str(value))
    require(result.is_finite(), "nonfinite value")
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_study(study):
    require(sha(study / "final_freeze.json") == PINNED_SEAL_SHA, "study seal differs")
    seal = json.loads((study / "final_freeze.json").read_text())
    for name, expected in seal["files"].items():
        require(sha(study / name) == expected["sha256"], "sealed study file differs: " + name)
    return seal


def nullable_records(frame):
    # Optional numeric/categorical columns can deserialize as NaN. They are
    # unknown, not zero, and two unknowns should not break a paired comparison.
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def reconstruct_personas(config):
    """Reconstruct only the recorded seed's persona draws, not policy noise or orders.

    This is the population algorithm in the pinned packet_v2 source. Every
    initial style and risk value is labeled reconstructed, not an observed log.
    """
    rng = random.Random(config["experiment"]["seed"])
    spec = config["population"]
    n = spec["n_agents"]
    styles = ["fundamental"] * round(n * spec["initial_fundamental_share"])
    styles += ["technical"] * (n - len(styles))
    rng.shuffle(styles)
    output = {}
    for i, style in enumerate(styles):
        risk, *_ = [round(rng.uniform(0.2, 0.8), 5) for _ in range(4)]
        rng.gauss(0, 1)  # Herding draw consumes this separate population RNG.
        output[f"A{i:03}"] = {"risk_tolerance": risk, "initial_style": style}
    return output


def capacity_detail(order, before, previous_close, risk):
    nominal = max(1, round(1 + 8 * risk))
    buy_limit = number(round(float(previous_close) * 1.003, 2))
    sell_limit = number(round(float(previous_close) * 0.997, 2))
    buy_capacity = int(number(before["cash"]) // buy_limit)
    sell_capacity = int(before["position"])
    side = order["requested_action"]
    requested = int(order["requested_quantity"])
    known_clipped = None
    if side in {"buy", "sell"}:
        expected = min(nominal, buy_capacity if side == "buy" else sell_capacity)
        require(
            requested == expected and requested > 0, "requested size differs from rule capacity"
        )
        known_clipped = requested < nominal
        if order["action"] == side:
            require(
                number(order["limit_price"]) == (buy_limit if side == "buy" else sell_limit),
                "rule limit differs",
            )
    else:
        require(side == "hold" and requested == 0, "invalid HOLD")
    return {
        "opening_cash": str(number(before["cash"])),
        "opening_shares": sell_capacity,
        "reconstructed_risk_tolerance": risk,
        "nominal_quantity_if_active": nominal,
        "conditional_buy_capacity_at_derived_limit": buy_capacity,
        "conditional_sell_capacity": sell_capacity,
        "derived_buy_limit": str(buy_limit),
        "derived_sell_limit": str(sell_limit),
        "desired_side_before_clipping": side if side != "hold" else None,
        "raw_policy_noise": None,
        "known_active_size_clipped": known_clipped,
        "hold_reason": order["rationale"] if side == "hold" else None,
        "later_validation_rejected": order["action"] != side,
    }


def verify_fills(orders, trades, market):
    by_id = {o["order_id"]: o for o in orders if isinstance(o["order_id"], str)}
    require(len(by_id) == sum(isinstance(o["order_id"], str) for o in orders), "duplicate order id")
    filled, volume = Counter(), Counter()
    for trade in trades:
        qty = int(trade["quantity"])
        require(qty == trade["quantity"] and qty > 0, "invalid trade size")
        volume[trade["session_date"]] += qty
        for side, who in [("buy", "buyer_id"), ("sell", "seller_id")]:
            oid = trade[side + "_order_id"]
            require(oid in by_id, "missing trade order")
            order = by_id[oid]
            require(
                order["agent_id"] == trade[who]
                and order["action"] == side
                and order["session_date"] == trade["session_date"],
                "trade order identity differs",
            )
            filled[oid] += qty
    for order in orders:
        require(filled[order["order_id"]] == order["filled_quantity"], "fill count differs")
        require(0 <= order["filled_quantity"] <= order["quantity"], "overfilled order")
    for row in market:
        require(volume[row["session_date"]] == row["volume"], "single-side volume differs")
    require(set(volume) <= {r["session_date"] for r in market}, "trade date outside market")
    return volume


def valuation(accounts, initial_cash, initial_shares, initial_price, close):
    cash = sum(number(a["cash"]) for a in accounts)
    shares = sum(int(a["position"]) for a in accounts)
    equity = sum(number(a["total_equity"]) for a in accounts)
    realized = sum(number(a["realized_pnl"]) for a in accounts)
    unrealized = sum(number(a["unrealized_pnl"]) for a in accounts)
    expected = number(initial_cash) + initial_shares * number(close)
    require(
        cash == number(initial_cash) and shares == initial_shares, "market endowment not conserved"
    )
    tolerance = Decimal("0.000001") * len(accounts)
    require(abs(equity - expected) <= tolerance, "aggregate marked wealth differs")
    change = initial_shares * (number(close) - number(initial_price))
    require(abs(realized + unrealized - change) <= tolerance, "profit decomposition differs")
    return {
        "total_cash": str(cash),
        "total_shares": shares,
        "total_marked_wealth": str(equity),
        "wealth_change_from_price_mark": str(change),
        "sum_realized_pnl": str(realized),
        "sum_unrealized_pnl": str(unrealized),
        "net_cash_creation": "0",
    }


def analyze_run(directory):
    config = json.loads((directory / "config_snapshot.json").read_text())
    tables = {
        name: nullable_records(pd.read_parquet(directory / f"{name}.parquet"))
        for name in ["market", "agents", "orders", "trades"]
    }
    volume = verify_fills(tables["orders"], tables["trades"], tables["market"])
    personas = reconstruct_personas(config)
    spec = config["population"]
    initial_price = tables["market"][0]["previous_close"]
    before = {
        agent_id: {"cash": spec["initial_cash"], "position": spec["initial_shares"]}
        for agent_id in personas
    }
    orders_by_day, accounts_by_day = defaultdict(list), defaultdict(list)
    for row in tables["orders"]:
        orders_by_day[row["session_date"]].append(row)
    for row in tables["agents"]:
        accounts_by_day[row["session_date"]].append(row)
    details, daily, by_style = [], [], []
    for i, market in enumerate(tables["market"]):
        date = market["session_date"]
        orders, accounts = orders_by_day[date], accounts_by_day[date]
        require(len(orders) == len(accounts) == len(personas), "incomplete daily accounts")
        if i == 0:
            require(
                all(o["style"] == personas[o["agent_id"]]["initial_style"] for o in orders),
                "reconstructed styles differ",
            )
        day_details = []
        for order in orders:
            agent_id = order["agent_id"]
            detail = {
                **order,
                **capacity_detail(
                    order,
                    before[agent_id],
                    market["previous_close"],
                    personas[agent_id]["risk_tolerance"],
                ),
            }
            detail["unfilled_quantity"] = order["quantity"] - order["filled_quantity"]
            day_details.append(detail)
        details.extend(day_details)
        for style in ["fundamental", "technical"]:
            rows = [r for r in day_details if r["style"] == style]
            by_style.append(
                {
                    "session_date": date,
                    "style": style,
                    "decisions": dict(Counter(r["action"] for r in rows)),
                    "active_policy_clipping_count": sum(
                        r["known_active_size_clipped"] is True for r in rows
                    ),
                    "hold_reasons": dict(
                        Counter(r["hold_reason"] for r in rows if r["hold_reason"])
                    ),
                    "quantities": {
                        side: {
                            "nominal_if_active": sum(
                                r["nominal_quantity_if_active"]
                                for r in rows
                                if r["requested_action"] == side
                            ),
                            "requested_after_policy": sum(
                                r["requested_quantity"]
                                for r in rows
                                if r["requested_action"] == side
                            ),
                            "accepted": sum(r["quantity"] for r in rows if r["action"] == side),
                            "filled_order_side": sum(
                                r["filled_quantity"] for r in rows if r["action"] == side
                            ),
                            "unfilled": sum(
                                r["unfilled_quantity"] for r in rows if r["action"] == side
                            ),
                        }
                        for side in ["buy", "sell"]
                    },
                }
            )
        daily.append(
            {
                "session_date": date,
                "close": market["close"],
                "single_side_traded_quantity": volume[date],
                "opening_zero_share_accounts": sum(r["opening_shares"] == 0 for r in day_details),
                "opening_cash_cannot_buy_one": sum(
                    r["conditional_buy_capacity_at_derived_limit"] == 0 for r in day_details
                ),
                "closing_zero_share_accounts": sum(a["position"] == 0 for a in accounts),
                **valuation(
                    accounts,
                    number(spec["initial_cash"]) * spec["n_agents"],
                    spec["initial_shares"] * spec["n_agents"],
                    initial_price,
                    market["close"],
                ),
            }
        )
        before = {a["agent_id"]: a for a in accounts}
    return {
        "run_id": directory.name,
        "seed": config["experiment"]["seed"],
        "arm": config["behavior"]["rule_style_adaptation"],
        "reconstructed_personas": personas,
        "daily": daily,
        "by_day_style": by_style,
        "account_order_details": details,
    }


def execute(study, output):
    verify_study(study)
    require(
        not output.resolve().is_relative_to(study.resolve()), "output must be outside sealed study"
    )
    runs = [
        analyze_run(study / "results_v2" / f"seed{seed}_{arm}")
        for seed in [17, 29, 43]
        for arm in ["fixed", "adaptive"]
    ]
    for i in range(0, len(runs), 2):
        require(
            all(
                runs[i][key] == runs[i + 1][key]
                for key in [
                    "reconstructed_personas",
                    "daily",
                    "by_day_style",
                    "account_order_details",
                ]
            ),
            "paired diagnostics differ",
        )
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "new_model_POSTs": 0,
        "source_seal_sha256": PINNED_SEAL_SHA,
        "runs": runs,
        "limits": "Descriptive only. Nominal risk sizes reconstructed from pinned persona "
        "algorithm; no hidden policy noise or pre-clipping HOLD side recovered. Identical "
        "arms are paired controls, not independent samples. Capacity asymmetry and rising "
        "marked wealth do not identify the cause of the price path.",
    }
    (output / "diagnostic.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (output / "REPORT.txt").write_text(describe(result))
    verify_study(study)
    return result


def describe(result):
    lines = [
        "已有六次市场模拟的数量与估值检查（只读原记录，零模型调用）",
        "",
        "两种策略更新设置的记录完全相同，下面每个随机起点只讲一组，不能把两组当成独立样本。",
        "起初每户有1000现金、2股。按100.30的买入限价，资金最多买9股；库存最多卖2股。"
        "具体想下多少股还取决于规则中的风险参数。风险参数是按原随机起点和原算法复原的。",
        "等待记录没有保存原始随机扰动和被裁量前的买卖方向，这些字段保留未知。",
        "",
    ]
    for run in result["runs"]:
        if run["arm"] != "fixed":
            continue
        rows = run["account_order_details"]
        actions = Counter(r["action"] for r in rows)
        clipped = Counter(r["requested_action"] for r in rows if r["known_active_size_clipped"])
        quantities = {
            side: sum(r["quantity"] for r in rows if r["action"] == side)
            for side in ["buy", "sell"]
        }
        reasons = Counter(r["hold_reason"] for r in rows if r["hold_reason"])
        last = run["daily"][-1]
        lines.append(
            f"起点{run['seed']}：买/卖/等待分别{actions['buy']}/{actions['sell']}/{actions['hold']}次；"
            f"买卖申报量分别{quantities['buy']}/{quantities['sell']}股；"
            f"成交按单边共{sum(d['single_side_traded_quantity'] for d in run['daily'])}股。"
            f"已知买/卖请求被资金或持仓限制缩小的次数为{clipped['buy']}/{clipped['sell']}；"
            f"另有{reasons['rule portfolio bound']}次因账户约束而等待。"
            f"期末{last['closing_zero_share_accounts']}户没有股票。"
        )
        lines.append(
            f"收盘价{last['close']}，全体账面财富较起初增加{last['wealth_change_from_price_mark']}；"
            f"其中已实现盈亏合计{last['sum_realized_pnl']}、未实现盈亏合计{last['sum_unrealized_pnl']}。"
            "现金合计仍为20000，股票合计仍为40；总账面增加等于40股按新价格重新估值，不能叫现金净赚。"
        )
    lines += [
        "",
        "这确认了初始买卖能力不对称，以及库存会限制部分卖出。"
        "它还没有证明这就是价格上涨的原因。没有改旧设置或补跑来追求不同走势。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.study, args.output)
    print(json.dumps({"checked_runs": len(result["runs"]), "new_model_POSTs": 0}))


if __name__ == "__main__":
    main()
