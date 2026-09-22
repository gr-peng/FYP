"""Training-only accounting, chronology and macro-reference diagnostics."""

import numpy as np
import pandas as pd

from experiments.replication_training_data import deduplicate

GROUP = ["session", "repeat", "group", "period"]
PERSON = GROUP + ["player"]


def integral(frame, columns):
    for column in columns:
        value = pd.to_numeric(frame[column], errors="raise")
        if value.isna().any() or not np.isfinite(value).all() or (value != np.floor(value)).any():
            raise ValueError(f"invalid integer in {column}")
        frame[column] = value.astype("int64")


def normalized_tables(frames):
    rename = {
        "session_code": "session",
        "repetition": "repeat",
        "group_id": "group",
        "round_number": "period",
        "player_id": "player",
    }
    result, copies = {}, {}
    for name in ("orders", "trades"):
        frame, copies[name] = deduplicate(frames[name], ["session_code", "uuid"])
        frame = frame.rename(columns=rename)
        integers = ["repeat", "group", "period", "quantity", "price", "created"]
        if name == "orders":
            integers += ["player", "filled", "is_replacement", "deleted"]
        integral(frame, integers)
        result[name] = frame
    raw_keys = [
        "session.code",
        "subsession.repetition",
        "group.id_in_subsession",
        "subsession.round_number",
        "player.id_in_group",
    ]
    frame, copies["rounds"] = deduplicate(frames["rounds"], raw_keys)
    columns = {
        "session.code": "session",
        "subsession.repetition": "repeat",
        "group.id_in_subsession": "group",
        "subsession.round_number": "period",
        "player.id_in_group": "player",
        "participant.code": "participant",
        "player.cash": "end_cash",
        "player.assets": "end_assets",
        "player.available_cash": "available_cash",
        "player.available_assets": "available_assets",
        "player.dividend_payment": "dividend_payment",
        "player.next_cash": "next_cash",
        "group.dividend": "dividend",
        "group.closing_price": "closing_price",
        "group.average_price": "average_price",
        "group.starting_timestamp": "starting_timestamp",
        "subsession.num_rounds": "num_rounds",
        "subsession.practice": "practice",
    }
    frame = frame.rename(columns=columns)
    integral(
        frame,
        [
            v
            for v in columns.values()
            if v not in {"session", "participant", "closing_price", "average_price"}
        ],
    )
    for column in ("closing_price", "average_price"):
        frame[column] = pd.to_numeric(frame[column].replace("", np.nan), errors="raise")
    result["rounds"] = frame
    return result, copies


def attach_trade_actors(orders, trades):
    if orders.duplicated(["session", "uuid"]).any():
        raise ValueError("nonunique order reference")
    linked = trades.copy()
    for side in ("ask", "bid"):
        fields = GROUP + ["uuid", "player", "kind", "side", "quantity", "price", "created"]
        other = (
            orders[fields]
            .rename(columns={c: f"{side}_{c}" for c in fields if c not in {"session", "uuid"}})
            .rename(columns={"uuid": f"{side}_uuid"})
        )
        linked = linked.merge(other, on=["session", f"{side}_uuid"], how="left", validate="m:1")
        if linked[f"{side}_player"].isna().any():
            raise ValueError("trade references an unknown order")
        if any((linked[c] != linked[f"{side}_{c}"]).any() for c in GROUP if c != "session"):
            raise ValueError("trade references another market/period")
        if (linked[f"{side}_side"] != side).any():
            raise ValueError("trade order side mismatch")
    return linked


def account_ledger(rounds, linked):
    flows = []
    for side, sign in (("ask", -1), ("bid", 1)):
        part = linked[GROUP].copy()
        part["player"] = linked[f"{side}_player"].to_numpy()
        part["asset_flow"] = sign * linked.quantity.to_numpy()
        part["cash_flow"] = -sign * (linked.quantity * linked.price).to_numpy()
        flows.append(part)
    net = pd.concat(flows).groupby(PERSON)[["asset_flow", "cash_flow"]].sum().reset_index()
    if len(
        net.merge(rounds[PERSON], on=PERSON, how="left", indicator=True).query(
            '_merge == "left_only"'
        )
    ):
        raise ValueError("trade actor is missing from participant period records")
    ledger = rounds.merge(net, on=PERSON, how="left", validate="1:1")
    ledger[["asset_flow", "cash_flow"]] = ledger[["asset_flow", "cash_flow"]].fillna(0)
    ledger = ledger.sort_values(["session", "repeat", "group", "player", "period"])
    history = ledger.groupby(["session", "repeat", "group", "player"])
    ledger["start_cash"] = history.next_cash.shift()
    ledger["start_assets"] = history.end_assets.shift()
    first = ledger.period == 1
    # Independently check the inferred first-period account against allowed endowments.
    ledger.loc[first, "start_cash"] = ledger.loc[first, "end_cash"] - ledger.loc[first, "cash_flow"]
    ledger.loc[first, "start_assets"] = (
        ledger.loc[first, "end_assets"] - ledger.loc[first, "asset_flow"]
    )
    ledger["cash_residual"] = ledger.end_cash - ledger.start_cash - ledger.cash_flow
    ledger["asset_residual"] = ledger.end_assets - ledger.start_assets - ledger.asset_flow
    ledger["dividend_residual"] = ledger.dividend_payment - ledger.end_assets * ledger.dividend
    ledger["next_cash_residual"] = ledger.next_cash - ledger.end_cash - ledger.dividend_payment
    return ledger


def macro_panels(rounds, linked, markets):
    source = (
        rounds.groupby(GROUP)
        .agg(
            dividend=("dividend", "first"),
            reported_average=("average_price", "first"),
            reported_close=("closing_price", "first"),
            shares=("end_assets", "sum"),
        )
        .reset_index()
    )
    trade_frame = linked.assign(value=linked.quantity * linked.price)
    grouped = trade_frame.groupby(GROUP)
    stats = grouped.agg(
        transactions=("uuid", "size"),
        volume=("quantity", "sum"),
        value=("value", "sum"),
        unweighted_mean=("price", "mean"),
        median_price=("price", "median"),
    ).reset_index()
    stats["vwap"] = stats.value / stats.volume
    periods = source.merge(stats, on=GROUP, how="left", validate="1:1")
    periods[["transactions", "volume", "value"]] = periods[
        ["transactions", "volume", "value"]
    ].fillna(0)
    periods["fundamental_value"] = 5 * (11 - periods.period)
    periods["rd"] = (periods.vwap - periods.fundamental_value) / 27.5
    periods["rad"] = periods.rd.abs()
    periods["average_bias"] = periods.median_price - periods.fundamental_value
    periods["turnover"] = periods.volume / periods.shares
    outcomes = (
        periods.groupby(["session", "repeat", "group"])
        .agg(
            periods=("period", "size"),
            priced_periods=("vwap", "count"),
            rd=("rd", "mean"),
            rad=("rad", "mean"),
            rdmax=("rd", "max"),
            average_bias=("average_bias", "mean"),
            volume=("volume", "sum"),
            transactions=("transactions", "sum"),
            turnover=("turnover", "sum"),
        )
        .reset_index()
    )
    meta = markets.rename(columns={"slab": "session", "mid": "group", "mlab": "market"})
    outcomes = outcomes.merge(
        meta[["session", "group", "market", "study", "cond"]],
        on=["session", "group"],
        how="left",
        validate="m:1",
    )
    return periods, outcomes


def audit_training(frames, metadata):
    tables, copies = normalized_tables(frames)
    orders, trades, rounds = (tables[k] for k in ("orders", "trades", "rounds"))
    if any(set(frame["repeat"]) != {1, 2} for frame in tables.values()):
        raise ValueError("unexpected repetition scope")
    if any(not frame.period.between(1, 10).all() for frame in tables.values()):
        raise ValueError("unexpected period scope")
    if (
        not rounds.groupby(["session", "repeat", "group", "player"])
        .period.apply(lambda x: set(x) == set(range(1, 11)) and len(x) == 10)
        .all()
    ):
        raise ValueError("incomplete participant histories")
    human = metadata["_participants"].rename(
        columns={"slab": "session", "mid": "group", "pid": "player", "plab": "participant"}
    )
    people = rounds[["session", "group", "player", "participant"]].drop_duplicates()
    membership = people.merge(
        human,
        on=["session", "group", "player", "participant"],
        how="outer",
        indicator=True,
        validate="1:1",
    )
    if not membership._merge.eq("both").all():
        raise ValueError("training participant membership differs from frozen metadata")
    for column in ("dividend", "starting_timestamp", "average_price", "closing_price"):
        if rounds.groupby(GROUP)[column].nunique(dropna=False).max() != 1:
            raise ValueError(f"participants disagree about market-level {column}")
    linked = attach_trade_actors(orders, trades)
    ledger = account_ledger(rounds, linked)
    periods, outcomes = macro_panels(rounds, linked, metadata["_markets"])
    unique_orders = orders.set_index(["session", "uuid"])
    referenced = pd.concat(
        [
            linked[["session", "ask_uuid"]].rename(columns={"ask_uuid": "uuid"}),
            linked[["session", "bid_uuid"]].rename(columns={"bid_uuid": "uuid"}),
        ]
    )
    actual = (
        pd.MultiIndex.from_frame(referenced)
        .value_counts()
        .reindex(unique_orders.index, fill_value=0)
    )
    first = ledger[ledger.period == 1]
    endowments = first.groupby(["start_cash", "start_assets"]).size()
    valid_endowments = first.apply(
        lambda x: (x.start_cash, x.start_assets) in {(3000, 20), (1000, 60)}, axis=1
    )
    actual_orders = orders[orders.is_replacement == 0]
    order_seconds = actual_orders.groupby(GROUP + ["created"]).size()
    trade_seconds = trades.groupby(GROUP + ["created"]).size()
    limits = orders[orders.kind == "limit"]
    replacement = orders[orders.replaced_by != ""].merge(
        orders,
        left_on=["session", "replaced_by"],
        right_on=["session", "uuid"],
        how="left",
        suffixes=("_parent", "_child"),
        validate="1:1",
    )
    filled_quantities = (
        pd.concat(
            [
                linked[["session", "ask_uuid", "quantity"]].rename(columns={"ask_uuid": "uuid"}),
                linked[["session", "bid_uuid", "quantity"]].rename(columns={"bid_uuid": "uuid"}),
            ]
        )
        .groupby(["session", "uuid"])
        .quantity.sum()
    )
    parent_keys = pd.MultiIndex.from_frame(
        replacement[["session", "uuid_parent"]].rename(columns={"uuid_parent": "uuid"})
    )
    parent_fills = filled_quantities.reindex(parent_keys).to_numpy()
    live_limits = limits[(limits.filled == 0) & (limits.deleted == 0)].copy()
    live_limits["reserved_cash"] = np.where(
        live_limits.side == "bid", live_limits.quantity * live_limits.price, 0
    )
    live_limits["reserved_assets"] = np.where(live_limits.side == "ask", live_limits.quantity, 0)
    reserves = live_limits.groupby(PERSON)[["reserved_cash", "reserved_assets"]].sum().reset_index()
    ledger = ledger.merge(reserves, on=PERSON, how="left", validate="1:1")
    ledger[["reserved_cash", "reserved_assets"]] = ledger[
        ["reserved_cash", "reserved_assets"]
    ].fillna(0)
    ledger["available_cash_residual"] = (
        ledger.available_cash - ledger.end_cash + ledger.reserved_cash
    )
    ledger["available_assets_residual"] = (
        ledger.available_assets - ledger.end_assets + ledger.reserved_assets
    )
    cancelled_fills = limits[(limits.filled == 1) & (limits.deleted == 1)].copy()
    cancelled_fills["extra_cash_release"] = np.where(
        cancelled_fills.side == "bid", cancelled_fills.quantity * cancelled_fills.price, 0
    )
    cancelled_fills["extra_asset_release"] = np.where(
        cancelled_fills.side == "ask", cancelled_fills.quantity, 0
    )
    releases = cancelled_fills.groupby(PERSON)[["extra_cash_release", "extra_asset_release"]].sum()
    ledger = ledger.merge(releases.reset_index(), on=PERSON, how="left", validate="1:1")
    ledger[["extra_cash_release", "extra_asset_release"]] = ledger[
        ["extra_cash_release", "extra_asset_release"]
    ].fillna(0)
    flows = []
    for side, sign in (("ask", -1), ("bid", 1)):
        flow = linked[GROUP + ["created"]].copy()
        flow["player"] = linked[f"{side}_player"].to_numpy()
        flow["cash_flow"] = -sign * (linked.quantity * linked.price).to_numpy()
        flow["asset_flow"] = sign * linked.quantity.to_numpy()
        flows.append(flow)
    boundaries = (
        pd.concat(flows)
        .groupby(PERSON + ["created"])[["cash_flow", "asset_flow"]]
        .sum()
        .reset_index()
        .sort_values(PERSON + ["created"])
    )
    cumulative = boundaries.groupby(PERSON)[["cash_flow", "asset_flow"]].cumsum()
    boundaries[["cumulative_cash", "cumulative_assets"]] = cumulative
    boundaries = boundaries.merge(
        ledger[PERSON + ["start_cash", "start_assets"]], on=PERSON, validate="m:1"
    )
    boundaries["cash_at_second_end"] = boundaries.start_cash + boundaries.cumulative_cash
    boundaries["assets_at_second_end"] = boundaries.start_assets + boundaries.cumulative_assets
    errors = {
        "cash_period_residuals": int(ledger.cash_residual.ne(0).sum()),
        "asset_period_residuals": int(ledger.asset_residual.ne(0).sum()),
        "dividend_payment_residuals": int(ledger.dividend_residual.ne(0).sum()),
        "next_cash_residuals": int(ledger.next_cash_residual.ne(0).sum()),
        "invalid_initial_endowments": int((~valid_endowments).sum()),
        "negative_end_cash": int(ledger.end_cash.lt(0).sum()),
        "negative_end_assets": int(ledger.end_assets.lt(0).sum()),
        "negative_available_cash": int(ledger.available_cash.lt(0).sum()),
        "negative_available_assets": int(ledger.available_assets.lt(0).sum()),
        "available_cash_residuals": int(ledger.available_cash_residual.ne(0).sum()),
        "available_assets_residuals": int(ledger.available_assets_residual.ne(0).sum()),
        "period_cash_flow_not_conserved": int(ledger.groupby(GROUP).cash_flow.sum().ne(0).sum()),
        "period_asset_flow_not_conserved": int(ledger.groupby(GROUP).asset_flow.sum().ne(0).sum()),
        "self_trades": int(linked.ask_player.eq(linked.bid_player).sum()),
        "trade_nonpositive_price_or_quantity": int(
            ((linked.price <= 0) | (linked.quantity <= 0)).sum()
        ),
        "filled_reference_disagreement": int((actual != unique_orders.filled).sum()),
        "missing_replacement_children": int(replacement.uuid_child.isna().sum()),
        "replacement_flag_errors": int(replacement.is_replacement_child.ne(1).sum()),
        "replacement_quantity_errors": int(
            (
                replacement.quantity_parent.to_numpy() - parent_fills
                != replacement.quantity_child.to_numpy()
            ).sum()
        ),
        "replacement_identity_errors": int(
            np.column_stack(
                [
                    replacement[f"{c}_parent"] != replacement[f"{c}_child"]
                    for c in ["repeat", "group", "period", "player", "side", "kind", "price"]
                ]
            )
            .any(axis=1)
            .sum()
        ),
        "unlinked_replacement_flags": int(
            orders.is_replacement.sum() - replacement.uuid_child.nunique()
        ),
        "orders_nonpositive_price_or_quantity": int(
            ((orders.price <= 0) | (orders.quantity <= 0)).sum()
        ),
        "trades_without_exactly_one_market_order": int(
            (
                (
                    linked.ask_kind.eq("market").astype(int)
                    + linked.bid_kind.eq("market").astype(int)
                )
                != 1
            ).sum()
        ),
        "trade_price_order_disagreements": int(
            ((linked.price != linked.ask_price) | (linked.price != linked.bid_price)).sum()
        ),
        "trade_exceeds_either_order_quantity": int(
            (
                (linked.quantity > linked.ask_quantity) | (linked.quantity > linked.bid_quantity)
            ).sum()
        ),
        "trade_precedes_order": int(
            ((linked.created < linked.ask_created) | (linked.created < linked.bid_created)).sum()
        ),
        "reported_mean_price_disagreement": int(
            (
                ~np.isclose(
                    periods.reported_average,
                    periods.unweighted_mean,
                    equal_nan=True,
                    atol=1e-9,
                    rtol=1e-10,
                )
            ).sum()
        ),
    }
    report = {
        "scope": "51 frozen training sessions only; development/test behavior unopened",
        "deduplication": copies,
        "population": {
            "sessions": rounds.session.nunique(),
            "markets": len(metadata["_markets"]),
            "people": len(people),
            "market_repetitions": len(outcomes),
            "market_periods": len(periods),
            "person_periods": len(ledger),
        },
        "orders": {
            "total": len(orders),
            "participant_submissions": len(actual_orders),
            "replacement_records": int(orders.is_replacement.sum()),
            "cancelled_limit_records": int(limits.deleted.sum()),
            "limit_records": len(limits),
            "market_records": int(orders.kind.eq("market").sum()),
            "cancelled_and_filled": int(((orders.deleted == 1) & (orders.filled == 1)).sum()),
            "outside_0_to_120_seconds": int((~orders.created.between(0, 120)).sum()),
        },
        "trades": {
            "count": len(trades),
            "volume": int(trades.quantity.sum()),
            "outside_0_to_120_seconds": int((~trades.created.between(0, 120)).sum()),
        },
        "chronology": {
            "submission_seconds": len(order_seconds),
            "seconds_with_multiple_submissions": int(order_seconds.gt(1).sum()),
            "submissions_in_tied_seconds": int(order_seconds[order_seconds > 1].sum()),
            "trade_seconds": len(trade_seconds),
            "seconds_with_multiple_trades": int(trade_seconds.gt(1).sum()),
            "trades_in_tied_seconds": int(trade_seconds[trade_seconds > 1].sum()),
            "cancellation_timestamps_available": False,
        },
        "endowments": [
            {"cash": int(a), "assets": int(b), "person_repetitions": int(n)}
            for (a, b), n in endowments.items()
        ],
        "accounting_and_reference_errors": errors,
        "empty_market_periods": int(periods.transactions.eq(0).sum()),
        "available_balance_diagnostic": {
            "affected_person_periods": int(
                (
                    ledger.available_cash_residual.ne(0) | ledger.available_assets_residual.ne(0)
                ).sum()
            ),
            "cash_residuals_not_explained_by_cancelled_filled_orders": int(
                (ledger.available_cash_residual - ledger.extra_cash_release).ne(0).sum()
            ),
            "asset_residuals_not_explained_by_cancelled_filled_orders": int(
                (ledger.available_assets_residual - ledger.extra_asset_release).ne(0).sum()
            ),
            "interpretation": "Exact arithmetic agreement with extra release of filled orders; "
            "cancellation timing and race sequence are not observed.",
        },
        "second_boundary_diagnostic": {
            "person_second_boundaries": len(boundaries),
            "negative_cash": int(boundaries.cash_at_second_end.lt(0).sum()),
            "negative_assets": int(boundaries.assets_at_second_end.lt(0).sum()),
            "interpretation": "Grouping by recorded seconds does not resolve intra-second events.",
        },
        "mean_rad_by_study_repetition": outcomes.groupby(["study", "repeat"])
        .rad.mean()
        .reset_index()
        .to_dict("records"),
        "limitations": [
            "Period ledgers do not establish within-second order or available balances.",
            "Cancel flags give eventual states, not decision-time order books.",
            "Replacement orders are bookkeeping, not additional human decisions.",
            "Missing-price periods remain missing, not zero-mispricing periods.",
        ],
    }
    return report, {
        "ledger": ledger,
        "market_periods": periods,
        "market_outcomes": outcomes,
        "linked_trades": linked,
        "second_boundaries": boundaries,
    }
