"""Exact reconstruction of one Bybit aggregated depth stream.

This is a read-only historical book, not a counterfactual matching engine.
Update-ID jumps invalidate the segment conservatively. Absence of jumps alone
does not establish exchange-side completeness or identify individual orders.
"""

from bisect import bisect_left, insort
from dataclasses import dataclass
from decimal import Decimal


class BookIntegrityError(ValueError):
    pass


class BookGap(BookIntegrityError):
    pass


def units(text: str, places: int) -> int:
    """Convert a decimal string exactly; never round to the storage precision."""
    if not isinstance(text, str) or places < 0:
        raise BookIntegrityError("Expected a decimal string and nonnegative precision")
    negative = text.startswith("-")
    body = text[1:] if negative else text
    parts = body.split(".")
    if len(parts) > 2 or not all(part.isascii() and part.isdigit() for part in parts):
        raise BookIntegrityError(f"Invalid decimal representation: {text!r}")
    whole, fraction = parts[0], parts[1] if len(parts) == 2 else ""
    if any(char != "0" for char in fraction[places:]):
        raise BookIntegrityError("Value exceeds exact storage precision")
    value = int(whole) * 10**places + int(fraction[:places].ljust(places, "0") or "0")
    return -value if negative else value


class _Side:
    def __init__(self):
        self.quantities: dict[int, int] = {}
        self.prices: list[int] = []  # Ascending for both sides.

    def assign(self, price, quantity):
        present = price in self.quantities
        if quantity == 0:
            if present:
                del self.quantities[price]
                self.prices.pop(bisect_left(self.prices, price))
        else:
            if not present:
                insort(self.prices, price)
            self.quantities[price] = quantity

    def levels(self, descending=False):
        prices = reversed(self.prices) if descending else iter(self.prices)
        return ((price, self.quantities[price]) for price in prices)


@dataclass(frozen=True)
class BookUpdate:
    kind: str
    update_id: int
    sequence: int
    engine_ms: int
    system_ms: int
    positive_depth_change_units: int
    negative_depth_change_units: int


class BybitBook:
    def __init__(self, symbol="BTCUSDT", depth=500, price_places=2, quantity_places=3):
        if depth <= 0 or price_places < 0 or quantity_places < 0:
            raise ValueError("Invalid book representation")
        self.symbol, self.depth = symbol, depth
        self.price_places, self.quantity_places = price_places, quantity_places
        self.bids, self.asks = _Side(), _Side()
        self.valid = False
        self.last_update: BookUpdate | None = None
        self.last_signature = None

    def apply(self, message) -> BookUpdate:
        try:
            return self._apply(message)
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            self.valid = False
            if isinstance(exc, BookIntegrityError):
                raise
            raise BookIntegrityError(f"Malformed book message: {type(exc).__name__}") from exc

    def _apply(self, message):
        kind = message["type"]
        data = message["data"]
        if (
            kind not in {"snapshot", "delta"}
            or data["s"] != self.symbol
            or message["topic"] != f"orderbook.{self.depth}.{self.symbol}"
        ):
            raise BookIntegrityError("Wrong stream or message type")
        update_id, sequence = data["u"], data["seq"]
        engine_ms, system_ms = message["cts"], message["ts"]
        if not all(type(v) is int and v > 0 for v in (update_id, sequence, engine_ms, system_ms)):
            raise BookIntegrityError("Invalid sequence or timestamp")
        if not (10**12 <= engine_ms < 10**14 and 10**12 <= system_ms < 10**14):
            raise BookIntegrityError("Expected millisecond epoch timestamps")

        parsed = []
        for name in ("b", "a"):
            levels = [
                (units(p, self.price_places), units(q, self.quantity_places)) for p, q in data[name]
            ]
            if len({p for p, _ in levels}) != len(levels):
                raise BookIntegrityError("Duplicate price within one side of a message")
            if any(p <= 0 or q < 0 or (kind == "snapshot" and q == 0) for p, q in levels):
                raise BookIntegrityError("Nonpositive price or invalid quantity")
            if kind == "snapshot":
                prices = [p for p, _ in levels]
                if prices != sorted(prices, reverse=name == "b"):
                    raise BookIntegrityError("Unsorted snapshot")
            parsed.append(levels)
        signature = (
            kind,
            update_id,
            sequence,
            engine_ms,
            system_ms,
            tuple(parsed[0]),
            tuple(parsed[1]),
        )
        if self.valid and signature == self.last_signature:
            return BookUpdate("duplicate", update_id, sequence, engine_ms, system_ms, 0, 0)
        previous = self.last_update
        if kind == "delta":
            if not self.valid or previous is None:
                raise BookGap("Delta without a valid snapshot segment")
            if update_id != previous.update_id + 1:
                raise BookGap(f"Update ID discontinuity: {previous.update_id} -> {update_id}")
            if sequence <= previous.sequence or engine_ms < previous.engine_ms:
                raise BookIntegrityError("Non-monotone sequence or matching-engine time")
        elif previous is not None and engine_ms < previous.engine_ms:
            raise BookIntegrityError("Snapshot goes backwards in matching-engine time")

        if kind == "snapshot":
            if (
                self.valid
                and previous is not None
                and update_id == previous.update_id
                and sequence == previous.sequence
                and (
                    dict(parsed[0]) != self.bids.quantities
                    or dict(parsed[1]) != self.asks.quantities
                )
            ):
                raise BookIntegrityError("Same-version snapshot disagrees with reconstructed book")
            # Snapshots replace all old levels, including after a service restart.
            self.bids, self.asks = _Side(), _Side()
        positive, negative = 0, 0
        for side, levels in zip((self.bids, self.asks), parsed, strict=True):
            for price, quantity in levels:
                change = quantity - side.quantities.get(price, 0)
                positive += max(change, 0)
                negative += max(-change, 0)
                side.assign(price, quantity)
            if not 0 < len(side.prices) <= self.depth:
                raise BookIntegrityError("Empty side or more levels than subscribed depth")
        if self.bids.prices[-1] >= self.asks.prices[0]:
            raise BookIntegrityError("Locked or crossed reconstructed book")
        result = BookUpdate(kind, update_id, sequence, engine_ms, system_ms, positive, negative)
        self.last_update, self.last_signature, self.valid = result, signature, True
        return result

    def levels(self, side):
        if not self.valid:
            raise BookIntegrityError("Book is invalid until a new snapshot")
        if side == "bid":
            return self.bids.levels(descending=True)
        if side == "ask":
            return self.asks.levels()
        raise ValueError("side must be bid or ask")

    def features(self):
        bids, asks = list(self.levels("bid")), list(self.levels("ask"))
        bid, bid_size = bids[0]
        ask, ask_size = asks[0]
        mid_twice = bid + ask
        result = {
            "cts_ms": self.last_update.engine_ms,
            "ts_ms": self.last_update.system_ms,
            "update_id": self.last_update.update_id,
            "sequence": self.last_update.sequence,
            "bid_units": bid,
            "ask_units": ask,
            "bid_size_units": bid_size,
            "ask_size_units": ask_size,
            "visible_bid_levels": len(bids),
            "visible_ask_levels": len(asks),
            "outer_bid_units": bids[-1][0],
            "outer_ask_units": asks[-1][0],
            "spread_bps": (ask - bid) * 20000 / mid_twice,
            "imbalance_l1": (bid_size - ask_size) / (bid_size + ask_size),
        }
        for label, levels in (("bid", bids), ("ask", asks)):
            for depth in (1, 5, 10, 50):
                result[f"{label}_depth_{depth}_units"] = sum(q for _, q in levels[:depth])
            for bps in (1, 5, 10, 25):
                outer_distance = abs(2 * levels[-1][0] - mid_twice) * 10000
                result[f"{label}_covers_{bps}bps"] = outer_distance >= bps * mid_twice
                result[f"{label}_depth_{bps}bps_units"] = sum(
                    q for p, q in levels if abs(2 * p - mid_twice) * 10000 <= bps * mid_twice
                )
        return result

    def sweep(self, side, quantity, limit_price=None):
        """Cost against this visible book only; never mutate historical liquidity.

        Quantity is in base-asset units (BTC), price in quote per base (USDT/BTC).
        Remainder beyond the visible book is unresolved, not an assumed fill.
        Fees, latency and the response of other participants are not identified.
        """
        if side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell")
        requested = units(quantity, self.quantity_places)
        if requested <= 0:
            raise ValueError("Quantity must be positive")
        limit = None if limit_price is None else units(limit_price, self.price_places)
        if limit is not None and limit <= 0:
            raise ValueError("Limit price must be positive")
        left, cost, fills = requested, 0, []
        for price, size in self.levels("ask" if side == "buy" else "bid"):
            if limit is not None and (price > limit if side == "buy" else price < limit):
                break
            take = min(size, left)
            fills.append({"price_units": price, "quantity_units": take})
            cost += price * take
            left -= take
            if left == 0:
                break
        filled = requested - left
        return {
            "requested_units": requested,
            "filled_units": filled,
            "unfilled_units": left,
            "quote_notional": str(Decimal(cost) / 10 ** (self.price_places + self.quantity_places)),
            "average_price": None
            if not filled
            else str(Decimal(cost) / filled / 10**self.price_places),
            "fills": fills,
            "interpretation": "Read-only visible-book sweep; no verified real or passive fill",
        }
