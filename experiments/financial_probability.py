"""Frozen visible-state retrieval and conditional human-order probability task."""

import copy
import hashlib
import json
from collections import defaultdict

import numpy as np

from experiments.market_quote_memory_baselines import FEATURES, features

METHODS = ("no_examples", "real_labels", "neutral_labels", "shuffled_labels")
INSTRUCTIONS = (
    "Predict the BUY versus SELL side of an active order submitted by a human experimental "
    "participant in the supplied partial state. The participant has already decided to submit "
    "an order; HOLD is not an option. This is a probability forecast, not investment advice or "
    "an optimal-trading recommendation. The experiment uses a continuous double auction, "
    "10 periods of 120 seconds per repetition, and two repetitions with reset accounts. "
    "The screen describes a per-period dividend of 0 or 10 with equal probabilities and "
    "displays expected remaining dividends. Shares expire after period 10; no terminal resale "
    "value is guaranteed. Cash and shares are total balances, not necessarily available balances: "
    "current resting orders, cancellations and the current order book are unobserved. "
    "All observed personal histories are actual earlier human submissions; never infer future "
    "choices from missing information. Demonstrations, when present, contain a single observed "
    "choice encoded as 0/1, or a control label; these are not population probabilities. "
    "Use only the supplied observations to estimate the probability for a participant in this "
    "state. Return probabilities for the two supplied menu labels."
)


def key(record):
    p = record["provenance"]
    return p["session"] + ":" + p["order_id"]


def select_targets(records, sessions, quota=25, seed=20261004):
    grouped = defaultdict(list)
    for record in records:
        session = record["provenance"]["session"]
        if session not in sessions:
            raise ValueError("target outside declared partition")
        grouped[session].append(record)
    selected = []
    for session in sorted(sessions):
        group = sorted(
            grouped[session],
            key=lambda r: hashlib.sha256(f"{seed}:target:{key(r)}".encode()).hexdigest(),
        )
        if len(group) < quota:
            raise ValueError(f"insufficient eligible targets in {session}")
        selected.extend(group[:quota])
    return selected


def visible(state):
    return {
        "partial_observation": copy.deepcopy(state),
        "computed_features": dict(zip(FEATURES, features(state).tolist(), strict=True)),
    }


class Demonstrations:
    def __init__(self, records, model, seed=20261004):
        self.records = sorted(records, key=key)
        if len({key(r) for r in self.records}) != len(records):
            raise ValueError("duplicate training example")
        self.x = np.array([features(r["state"]) for r in self.records])
        self.mean = np.array(model["mean"])
        self.scale = np.array(model["scale"])
        self.z = (self.x - self.mean) / self.scale
        labels = np.array([r["target"]["buy"] for r in self.records])
        self.shuffled = np.random.default_rng(seed).permutation(labels)

    def retrieve(self, state, n=4):
        target = (features(state) - self.mean) / self.scale
        distances = np.sum((self.z - target) ** 2, axis=1)
        ranking = np.argsort(distances, kind="stable")
        chosen, people = [], set()
        for index in ranking:
            p = self.records[index]["provenance"]
            person = (p["session"], p["participant"])
            if person in people:
                continue
            people.add(person)
            chosen.append(int(index))
            if len(chosen) == n:
                return chosen
        raise ValueError("not enough distinct training people")

    def bundle(self, state):
        indices = self.retrieve(state)
        result = {"state": visible(state), "demonstrations": [], "example_ids": []}
        for index in indices:
            record = self.records[index]
            result["example_ids"].append(key(record))
            result["demonstrations"].append(
                {
                    "state": visible(record["state"]),
                    "real_buy": record["target"]["buy"],
                    "shuffled_buy": int(self.shuffled[index]),
                }
            )
        return result


def question(bundle, method, order):
    if method not in METHODS or order not in ("buy_first", "sell_first"):
        raise ValueError("unknown method or menu order")
    sides = ("buy", "sell") if order == "buy_first" else ("sell", "buy")
    criteria = dict(
        zip(("A", "B"), [f"Submit an active {s.upper()} order." for s in sides], strict=True)
    )
    state = {"target": copy.deepcopy(bundle["state"]), "examples": []}
    if method != "no_examples":
        for example in bundle["demonstrations"]:
            p = (
                example["real_buy"]
                if method == "real_labels"
                else example["shuffled_buy"]
                if method == "shuffled_labels"
                else 0.5
            )
            mapped = {
                label: p if side == "buy" else 1 - p
                for label, side in zip(("A", "B"), sides, strict=True)
            }
            state["examples"].append(
                {"state": copy.deepcopy(example["state"]), "observed_choice_encoding": mapped}
            )
    return state, INSTRUCTIONS, criteria, "A" if order == "buy_first" else "B"


def serializable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
