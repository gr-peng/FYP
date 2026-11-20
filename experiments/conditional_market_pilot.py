"""Dynamic invented-state pilot, paired menus, deterministic replay and diagnostics."""

import math
from copy import deepcopy
from dataclasses import replace

from experiments import conditional_order_execution as execution
from experiments import hypothetical_memory_probe as probability

canonical, digest, require = probability.canonical, probability.digest, probability.require
MODEL, OPTIONS = probability.MODEL, probability.OPTIONS
ARMS = execution.prompt.ARMS
RUN_ID = "conditional-market-20261115-001"
PRIOR, MAX_POSTS = 397, 72
CONFIG = replace(execution.DEFAULT_CONFIG, seed=20261115)


def request_pair(messages, arm, agent, tick):
    """Only choice strings move; local identity never enters the model payload."""
    require(
        arm in ARMS and agent in execution.AGENTS and type(tick) is int and 0 <= tick < 6,
        "invalid local opportunity identity",
    )
    require(
        isinstance(messages, list)
        and len(messages) == 2
        and messages[0] == {"role": "system", "content": execution.prompt.INSTRUCTIONS},
        "unexpected prompt",
    )
    original = probability.strict_json(messages[1]["content"])
    require(
        set(original) == {"observation", "choices"}
        and original["choices"]
        == {"A": "Submit an active BUY order.", "B": "Submit an active SELL order."},
        "expected canonical BUY-first choices",
    )
    menus = list(execution.prompt.MENUS)
    slot = tick * 2 + execution.AGENTS.index(agent)
    if (ARMS.index(arm) + slot) % 2:
        menus.reverse()
    requests = []
    for menu in menus:
        value = deepcopy(original)
        if menu == "sell_first":
            value["choices"] = {"A": original["choices"]["B"], "B": original["choices"]["A"]}
        payload = {
            **deepcopy(OPTIONS),
            "messages": [deepcopy(messages[0]), {"role": "user", "content": canonical(value)}],
        }
        identity = {
            "run_id": RUN_ID,
            "arm": arm,
            "agent_id": agent,
            "tick": tick,
            "slot": slot,
            "menu_order": menu,
            "buy_label": "A" if menu == "buy_first" else "B",
        }
        requests.append(
            {
                **identity,
                "payload": payload,
                "wire_sha256": digest(payload),
                "request_hash": digest({"identity": identity, "payload": payload}),
            }
        )
    return requests


def pair_probability(values):
    require(len(values) == 2, "both menus required before choosing a direction")
    return math.fsum(execution.probability(x) for x in values) / 2


def replay(requests, responses):
    """Regenerate all states and exact wires before admitting recorded probabilities."""
    require(len(requests) == len(responses) <= MAX_POSTS, "request/response coverage differs")
    cursor = 0
    demos = {}
    for arm in ARMS:

        def callback(messages, *, arm=arm):
            nonlocal cursor
            demo = demos[arm]
            # The engine calls alpha then beta; only eligible accounts call back.
            agent = active[arm].pop(0)
            pair = request_pair(messages, arm, agent, demo.next_tick)
            values = []
            for expected in pair:
                require(cursor < len(requests), "missing recorded response")
                expected = {**expected, "position": cursor + 1}
                require(canonical(expected) == canonical(requests[cursor]), "dynamic wire mismatch")
                row = responses[cursor]
                require(
                    row["position"] == cursor + 1
                    and row["request_hash"] == expected["request_hash"],
                    "response identity mismatch",
                )
                values.append(
                    probability.parse_response(row["response"], expected)["probability_buy"]
                )
                cursor += 1
            return pair_probability(values)

        demos[arm] = execution.ConditionalExecutionDemo(
            dict.fromkeys(execution.AGENTS, callback), arm, CONFIG
        )
    active = {}
    for tick in range(6):
        for arm, demo in demos.items():
            active[arm] = [
                agent
                for i, agent in enumerate(execution.AGENTS)
                if CONFIG.opportunities[tick * 2 + i] and not demo.gate(agent, CONFIG.quantities[i])
            ]
            demo.step()
            require(not active[arm], "unconsumed eligible opportunity")
    require(cursor == len(requests), "extra recorded responses")
    return {arm: demo.result() for arm, demo in demos.items()}


def summarize(results, requests, responses):
    require(set(results) == set(ARMS), "all three branches required")
    pairs = []
    require(len(requests) == len(responses) and len(requests) % 2 == 0, "paired response coverage")
    for index in range(0, len(requests), 2):
        rr = requests[index : index + 2]
        require({r["menu_order"] for r in rr} == set(execution.prompt.MENUS), "missing menu")
        values = {
            r["menu_order"]: probability.parse_response(s["response"], r)["probability_buy"]
            for r, s in zip(rr, responses[index : index + 2], strict=True)
        }
        identity = {k: rr[0][k] for k in ("arm", "agent_id", "tick", "slot")}
        require(all(all(r[k] == v for k, v in identity.items()) for r in rr), "crossed pair")
        event = results[identity["arm"]]["events"][identity["tick"]]
        entry = event["opportunities"][execution.AGENTS.index(identity["agent_id"])]
        mean = pair_probability(list(values.values()))
        require(entry["probability_buy"] == mean, "selected probability not menu average")
        pairs.append(
            {
                **identity,
                **values,
                "mean_buy_probability": mean,
                "buy_first_minus_sell_first": values["buy_first"] - values["sell_first"],
                "uniform": entry["uniform"],
                "menu_directions_disagree_at_observed_uniform": (
                    entry["uniform"] < values["buy_first"]
                )
                != (entry["uniform"] < values["sell_first"]),
            }
        )
    comparisons = []
    for left, right in zip(ARMS[:-1], ARMS[1:], strict=True):
        rows = []
        for tick in range(6):
            for i, agent in enumerate(execution.AGENTS):
                a, b = (results[arm]["events"][tick]["opportunities"][i] for arm in (left, right))
                both = a["status"] == b["status"] == "accepted"
                rows.append(
                    {
                        "tick": tick,
                        "agent_id": agent,
                        "same_full_predecision_state": a["state_before"] == b["state_before"],
                        "left_status": a["status"],
                        "right_status": b["status"],
                        "both_accepted": both,
                        "directions_differ": a["side"] != b["side"] if both else None,
                    }
                )
        comparisons.append({"left": left, "right": right, "opportunities": rows})
    return {
        "run_id": RUN_ID,
        "model": MODEL,
        "new_model_POSTs": len(requests),
        "data_origin": "invented_accounts_and_simulated_history_no_human_targets",
        "branches": {
            arm: {
                "denominators": r["denominators"],
                "rp_metrics": r["rp_metrics"],
                "accounts_final": r["accounts_final"],
            }
            for arm, r in results.items()
        },
        "menu_pairs": pairs,
        "adjacent_branch_comparisons": comparisons,
        "independent_generation_count": None,
        "limits": [
            "One common seed, two invented accounts, first two of ten task periods.",
            "Fixed quote means prices cannot reveal market realism or price discovery.",
            "After paths diverge, probability differences mix information and state changes.",
            "Two-menu average is a declared aggregation, not proof of order-bias removal.",
            "No human targets, accuracy, psychology, significance or performance claim.",
        ],
    }


def fixed_probability_sampling_diagnostic(summary):
    """A new reference process over a frozen list, not a counterfactual market path."""
    result = {}
    for arm in ARMS:
        pairs = [r for r in summary["menu_pairs"] if r["arm"] == arm]
        distribution = [1.0]
        for row in pairs:
            p = row["mean_buy_probability"]
            next_distribution = [0.0] * (len(distribution) + 1)
            for k, mass in enumerate(distribution):
                next_distribution[k] += mass * (1 - p)
                next_distribution[k + 1] += mass * p
            distribution = next_distribution
        result[arm] = {
            "fixed_opportunity_count": len(pairs),
            "reference_expected_buy_count": math.fsum(r["mean_buy_probability"] for r in pairs),
            "reference_buy_count_variance": math.fsum(
                r["mean_buy_probability"] * (1 - r["mean_buy_probability"]) for r in pairs
            ),
            "reference_buy_count_distribution": distribution,
            "recorded_buy_count": sum(r["uniform"] < r["mean_buy_probability"] for r in pairs),
        }
    return {
        "reference_process": (
            "Freeze each observed branch's p-list; draw fresh independent uniforms "
            "once per listed opportunity."
        ),
        "branches": result,
        "new_model_POSTs": 0,
        "limits": (
            "Observed states/gates depend on earlier draws. This is not the conditional law "
            "of the original market, a confidence interval or changed-state forecasts. "
            "No alternative trades or profits are generated."
        ),
    }
