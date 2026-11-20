"""Exact fictional lotteries and in-memory HTTP; no real data or credentials."""

import asyncio
import json
from copy import deepcopy
from fractions import Fraction

import httpx
import pytest

from behavioral_market.benchmarks import tradeoff_comparison as study
from behavioral_market.evaluation import matched_choice_collection as collection
from behavioral_market.evaluation.style_mechanism_design import canonical
from scripts import choices13k_tradeoff as cli
from scripts.style_fact_check import fingerprint, save


def text(*outcomes):
    return "\n".join(f"- {p}% chance of receiving {x}" for p, x in outcomes)


def candidates():
    return [
        {
            "item_id": 166,
            "option_a_text": text((90, -5), (10, 40)),
            "option_b_text": text((50, 3), (50, 4)),
        },
        {"item_id": 211, "option_a_text": text((100, 4)), "option_b_text": text((50, 0), (50, 10))},
        {
            "item_id": 384,
            "option_a_text": text((100, 1), (0, -100)),
            "option_b_text": text((90, -10), (10, -2)),
        },
        {"item_id": 542, "option_a_text": text((100, 6)), "option_b_text": text((100, -1))},
    ]


@pytest.fixture
def materials():
    by_id = {r["item_id"]: r for r in candidates()}
    return {
        "target": by_id[166],
        "references": [
            {**by_id[i], "human_b_rate": p}
            for i, p in zip((211, 384, 542), (0.4, 0.1, 0.7), strict=True)
        ],
    }


def body(choice="A", n=1):
    return {
        "id": f"fictional-{n}",
        "created": n,
        "model": study.MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps({"choice": choice})},
            }
        ],
    }


def logs(packet, policy=None):
    requests, attempts, responses = [], [], []
    for req in packet["requests"]:
        original = policy(req) if policy else ("A" if req["repeat_id"] == 0 else "B")
        letter = req["letter_order"][req["content_order"].index(original)]
        ident = collection.identity(req)
        requests.append(deepcopy(req))
        attempts.append({**ident, "dispatched_at": "2026-10-22T00:00:00+00:00", "retry_count": 0})
        responses.append(
            {
                **ident,
                "requested_at": "2026-10-22T00:00:00+00:00",
                "response_received_at": "2026-10-22T00:00:00+00:00",
                "retry_count": 0,
                "cached": False,
                "http_status": 200,
                "latency_seconds": 0,
                "response": body(letter, req["position"]),
                "cache_key_header": {"present": False, "value_sha256": None},
            }
        )
    return requests, attempts, responses


def test_exact_parse_merges_and_removes_zero():
    assert study.parse_option(text(("12.5", 1), ("87.5", "1.0"), (0, -999))) == (
        (Fraction(1), Fraction(1)),
    )
    assert study.parse_option(text(("33.33", 1), ("33.33", 2), ("33.34", 3)))[-1][1] == Fraction(
        1667, 5000
    )


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "- hidden probabilities: 1, 2",
        text((100, 1)) + "\nextra",
        text((100, 1)) + "\n\n",
        text((100, 1)) + "\n- probabilities undisclosed",
        text((99.99, 1)),
        text((100.0000000000001, 1)),
        text((-1, 0), (101, 2)),
        text((100, "NaN")),
        text(("inf", 1)),
        text((100, "1e309")),
        "junk " + text((100, 1)),
        text((100, 1)) + " trailing",
    ],
)
def test_full_option_rejects_any_unparsed_line_or_invalid_probability(value):
    with pytest.raises(ValueError):
        study.parse_option(value)


def test_diagnostic_selection_is_text_only_and_order_invariant():
    rows = candidates()
    result = study.select_tradeoff(rows)
    assert result["selected_item_id"] == 166 and result["eligible_item_ids"] == [166, 211]
    assert result == study.select_tradeoff(rows[::-1])
    dominant = next(x for x in result["items"] if x["item_id"] == 384)
    assert (
        dominant["strict_outcome_A_better"]
        and dominant["classification"] == "A_distribution_dominates"
    )
    assert dominant["support_A"] == [["1", "1"]]
    crossing = result["items"][0]["CDF_points"]
    assert any(Fraction(x["A_minus_B"]) > 0 for x in crossing)
    assert any(Fraction(x["A_minus_B"]) < 0 for x in crossing)
    for field in ["human_b_rate", "choice", "old_error"]:
        polluted = deepcopy(rows)
        polluted[0][field] = 0.8
        with pytest.raises(ValueError, match="text-only schema"):
            study.select_tradeoff(polluted)


def test_equal_and_overlapping_distribution_dominance():
    row = {
        "item_id": 0,
        "option_a_text": text((50, 0), (50, 4)),
        "option_b_text": text((50, 0), (50, 2)),
    }
    d = study.select_tradeoff([row])
    assert (
        d["selected_item_id"] is None
        and d["items"][0]["classification"] == "A_distribution_dominates"
    )
    assert not d["items"][0]["strict_outcome_A_better"]
    row["option_a_text"] = row["option_b_text"]
    assert study.select_tradeoff([row])["items"][0]["classification"] == "equal_distributions"
    row["option_a_text"], row["option_b_text"] = text((50, 0), (50, 2)), text((50, 0), (50, 4))
    assert study.select_tradeoff([row])["items"][0]["classification"] == "B_distribution_dominates"


def test_any_candidate_parse_failure_stops_not_skips():
    rows = candidates()
    rows[-1]["option_a_text"] += "\n- unknown probability"
    with pytest.raises(ValueError):
        study.select_tradeoff(rows)


def test_packet24_common_mapping_rotating_order_and_rate_pairing(materials):
    before = deepcopy(materials)
    packet = study.make_packet(materials, "fictional")
    assert len(packet["requests"]) == 24
    assert [
        {
            k: row[k]
            for k in (
                "slot_id",
                "item_id",
                "method_id",
                "content_order",
                "letter_order",
                "repeat_id",
            )
        }
        for row in packet["requests"]
    ] == cli.schedule()
    assert [row["method_id"] for row in packet["requests"][:12]] == [
        "plain",
        "true_reference",
        "shuffled_reference",
        "shuffled_reference",
        "true_reference",
        "plain",
        "true_reference",
        "shuffled_reference",
        "plain",
        "plain",
        "shuffled_reference",
        "true_reference",
    ]
    assert study.validate_packet(json.loads(canonical(packet))) == packet
    common = {
        study.panel_tool.key(row): row["presentation"] for row in packet["panel"]["opportunities"]
    }
    for req in packet["requests"]:
        presentation = common[study.panel_tool.key(req)]
        if req["method_id"] == "plain":
            assert req["payload"]["messages"] == presentation["messages"]
        else:
            assert req["payload"]["messages"][1]["content"].endswith(
                presentation["messages"][1]["content"]
            )
        assert req["wire_sha256"] == study.digest(req["payload"])
        assert req["presentation_sha256"] == study.digest(presentation)
        wire = canonical(req["payload"])
        assert all(
            token not in wire
            for token in (
                "true_reference",
                "shuffled_reference",
                "rate_source_item_id",
                "human_b_rate",
            )
        )
    true, shuffled = [
        study.reference_cards(materials, m) for m in ("true_reference", "shuffled_reference")
    ]
    assert [r["assigned_rate"] for r in true] == [0.4, 0.1, 0.7]
    assert [r["assigned_rate"] for r in shuffled] == [0.1, 0.7, 0.4]
    assert [{k: v for k, v in r.items() if k != "assigned_rate"} for r in true] == [
        {k: v for k, v in r.items() if k != "assigned_rate"} for r in shuffled
    ]
    assert materials == before


def test_target_rate_rejected_and_numeric_coincidence_allowed(materials):
    packet = study.make_packet(materials, "fictional")
    result = study.evaluate(packet, *logs(packet), {166: 0.4})
    assert result["protocol_complete"]
    materials["target"]["human_b_rate"] = 0.4
    with pytest.raises(ValueError, match="text-only schema"):
        study.make_packet(materials, "fictional")


def test_pooled_rate_error_not_mean_of_cell_errors(materials):
    packet = study.make_packet(materials, "fictional")

    def policy(r):
        if r["method_id"] == "plain":
            return "A"
        if r["method_id"] == "shuffled_reference":
            return "B"
        return "B" if r["content_order"] == "AB" else "A"

    result = study.evaluate(packet, *logs(packet, policy), {166: 0.5})
    for comp in result["comparisons"].values():
        assert comp["planned_records"] == 24
        assert comp["primary"]["left_MAE"] == 0 and comp["primary"]["right_MAE"] == 0.5
        assert comp["primary"]["left_minus_right_MAE"] == -0.5


def test_pair_completeness_separate_from_whole_round(materials):
    packet = study.make_packet(materials, "fictional")
    req, att, resp = logs(packet)
    result = study.evaluate(packet, req, att, resp[:-1], {166: 0.5})
    assert not result["protocol_complete"]
    assert (
        result["comparisons"]["true_reference_vs_plain"]["primary"]["left_minus_right_MAE"] is None
    )
    assert (
        result["comparisons"]["true_reference_vs_shuffled_reference"]["primary"][
            "left_minus_right_MAE"
        ]
        == 0
    )
    assert all(
        c["planned_records"] == 24 and c["missing_records"] == 1
        for c in result["comparisons"].values()
    )


def test_repeats_conflicts_and_evidence_identity(materials):
    packet = study.make_packet(materials, "fictional")
    result = study.evaluate(packet, *logs(packet), {166: 0.5})
    assert len(result["same_input_variability"]) == 12
    assert all(x["different_display_choices"] for x in result["same_input_variability"])
    recorded = study.make_packet(materials, "fictional", "recorded_model_logs")
    with pytest.raises(ValueError, match="log identity differs"):
        study.evaluate(recorded, *logs(packet), {166: 0.5})
    packet["evidence_kind"] = "recorded_model_logs"
    packet["packet_sha256"] = study.digest(
        {k: v for k, v in packet.items() if k != "packet_sha256"}
    )
    with pytest.raises(ValueError, match="rebuilt materials"):
        study.validate_packet(packet)


@pytest.mark.parametrize(
    "mutation",
    [
        "wire",
        "method",
        "duplicate",
        "reorder",
        "bool_id",
        "local_cache",
        "retry",
        "orphan",
        "after_invalid",
    ],
)
def test_log_authentication_rejects_forged_conditions(materials, mutation):
    packet = study.make_packet(materials, "fictional")
    req, att, resp = logs(packet)
    if mutation == "wire":
        req[0]["payload"]["messages"][1]["content"] += "extra reference"
    elif mutation == "method":
        resp[0]["method_id"] = "true_reference"
    elif mutation == "duplicate":
        req.append(req[0])
    elif mutation == "reorder":
        req[0], req[1] = req[1], req[0]
    elif mutation == "bool_id":
        att[0]["position"] = True
    elif mutation == "local_cache":
        resp[0]["cached"] = True
    elif mutation == "retry":
        att[0]["retry_count"] = 1
    elif mutation == "orphan":
        att.pop()
    else:
        resp[0]["response"]["choices"][0]["message"]["content"] = '{"choice":"C"}'
    with pytest.raises(ValueError):
        study.evaluate(packet, req, att, resp, {166: 0.5})


@pytest.fixture
def source_workspace(tmp_path, monkeypatch):
    workspace = tmp_path
    folder = workspace / cli.ITERATION
    folder.mkdir(parents=True)
    old_paths = {name: workspace / desc["path"] for name, desc in cli.SOURCES.items()}
    for path in old_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {row["item_id"]: row for row in candidates()}
    old = {
        "target": by_id[384],
        "references": [
            {**by_id[i], "human_b_rate": p}
            for i, p in zip((166, 211, 542), (0.375, 0.4, 0.7), strict=True)
        ],
    }
    save(old_paths["source_materials.json"], old)
    save(old_paths["local_target.json"], {"item_id": 384, "human_b_rate": 0.1})
    sources = {
        name: {"path": str(path.relative_to(workspace)), "fingerprint": fingerprint(path)}
        for name, path in old_paths.items()
    }
    previous = workspace / cli.PREVIOUS
    save(
        previous / "pre_execution_freeze.json",
        {"files": {desc["path"]: desc["fingerprint"] for desc in sources.values()}},
    )
    save(previous / "independent_review.json", {"status": "accepted"})
    grant = workspace / "iterations/style_fact_execution_20261022/user_budget_authorization.json"
    grant.parent.mkdir(parents=True)
    save(
        grant,
        {
            "global_cumulative_completion_POST_cap": 100,
            "endpoint": study.ENDPOINT,
            "model": study.MODEL,
            "specific_user_reply": "fictional grant; not real authorization",
            "prior_used_completion_POSTs": 25,
        },
    )
    save(
        previous / "budget_ledger.json",
        {
            "used_completion_POSTs": 91,
            "reserved_next_completion_POSTs": 0,
            "active_run_id": None,
            "global_authorized_completion_POST_cap": 100,
            "status": "fictional_accepted",
            "global_authorization_source": str(grant.resolve()),
        },
    )
    metadata = {
        name: {
            "path": str((previous / name).relative_to(workspace)),
            "fingerprint": fingerprint(previous / name),
        }
        for name in ("pre_execution_freeze.json", "independent_review.json", "budget_ledger.json")
    }
    pairing = {
        "true_reference": {"211": 211, "384": 384, "542": 542},
        "shuffled_reference": {"211": 384, "384": 542, "542": 211},
    }
    comparisons = [
        {"left": a, "right": b, "difference": "left_minus_right_absolute_error"}
        for a, b in study.COMPARISONS
    ]
    origin = previous / "next_iteration_plan.json"
    save(
        origin,
        {
            "data_sources": sources,
            "metadata_sources": metadata,
            "reference_pairing": pairing,
            "comparisons_fixed_in_advance": comparisons,
        },
    )
    origin_descriptor = {
        "path": str(origin.relative_to(workspace)),
        "fingerprint": fingerprint(origin),
    }
    monkeypatch.setattr(cli, "ORIGIN", deepcopy(origin_descriptor))
    monkeypatch.setattr(cli, "SOURCES", deepcopy(sources))
    plan = {
        "origin_plan": origin_descriptor,
        "data_sources": sources,
        "metadata_sources": metadata,
        "target_item_id": 166,
        "reference_item_ids": [211, 384, 542],
        "reference_pairing": pairing,
        "selection_input_fields": ["item_id", "option_a_text", "option_b_text"],
        "comparisons": comparisons,
        "schedule": cli.schedule(),
        "frozen_run_id": cli.RUN_ID,
        "fixed_execution_output": str((folder / "live_execution").resolve()),
        "global_grant": {
            "path": str(grant.relative_to(workspace)),
            "fingerprint": fingerprint(grant),
        },
        "budget": {
            "used_POSTs": 91,
            "authorized_cap": 100,
            "reserved_POSTs": 0,
            "planned_future_batch": 24,
            "required_future_cumulative_cap": 115,
            "new_POSTs_authorized_now": 0,
        },
    }
    save(folder / "plan.json", plan)
    return workspace


def test_actual_offline_prepare_rebuilds_only_two_sources(source_workspace):
    w = source_workspace
    result = cli.prepare(w)
    assert result == {
        "prepared_not_sent": True,
        "opportunities": 24,
        "used_POSTs": 91,
        "current_cap": 100,
        "required_future_cap": 115,
    }
    packet, targets = cli.load_packet(w)
    assert targets == {166: 0.375}
    package = w / cli.ITERATION / "package"
    role = json.loads((package / "rate_role_change.json").read_text())
    assert role["old_local_rate_becomes_reference"] and role["new_target_rate_local_only"]
    assert packet["materials"]["references"][1]["human_b_rate"] == 0.1
    assert "human_b_rate" not in packet["materials"]["target"]
    assert "0.375" not in canonical([r["payload"] for r in packet["requests"]])
    pending = json.loads((package / "pending_evaluation.json").read_text())
    assert pending["logged_dispatch_attempts"] == 0 and not pending["protocol_complete"]
    assert all(
        c["planned_records"] == 24 and c["primary"]["left_MAE"] is None
        for c in pending["comparisons"].values()
    )
    assert not (w / cli.ITERATION / "live_execution").exists()
    with pytest.raises(ValueError, match="overwrite"):
        cli.prepare(w)


@pytest.mark.parametrize(
    "attack",
    [
        "derived_rate",
        "local_target",
        "receipt_path",
        "source_plan_resign",
        "source_origin_plan_resign",
    ],
)
def test_joint_source_and_derived_resigning_rejected(source_workspace, attack):
    w = source_workspace
    cli.prepare(w)
    folder = w / cli.ITERATION
    package = folder / "package"
    receipt = json.loads((package / "source_read_receipt.json").read_text())
    if attack == "receipt_path":
        receipt["derived_files"]["../MUST_NOT_OPEN"] = {"bytes": 0, "sha256": "f" * 64}
    elif attack in ("derived_rate", "local_target"):
        if attack == "derived_rate":
            mats = json.loads((package / "source_materials.json").read_text())
            mats["references"][0]["human_b_rate"] = 0.99
            save(package / "source_materials.json", mats)
            changed = study.make_packet(mats, cli.RUN_ID, "recorded_model_logs")
            save(package / "packet.json", changed)
            (package / "expected_requests.jsonl").write_text(
                "".join(canonical(r) + "\n" for r in changed["requests"])
            )
        else:
            save(package / "local_target.json", {"item_id": 166, "human_b_rate": 0.987})
        receipt["derived_files"] = {name: fingerprint(package / name) for name in cli.DERIVED}
    else:
        plan = json.loads((folder / "plan.json").read_text())
        source = w / cli.SOURCES["source_materials.json"]["path"]
        old = json.loads(source.read_text())
        old["references"][0]["human_b_rate"] = 0.99
        save(source, old)
        plan["data_sources"]["source_materials.json"]["fingerprint"] = fingerprint(source)
        if attack == "source_origin_plan_resign":
            origin = w / cli.ORIGIN["path"]
            changed_origin = json.loads(origin.read_text())
            changed_origin["data_sources"] = plan["data_sources"]
            save(origin, changed_origin)
            plan["origin_plan"]["fingerprint"] = fingerprint(origin)
        save(folder / "plan.json", plan)
        receipt["plan_fingerprint"] = fingerprint(folder / "plan.json")
        receipt["data_sources"] = plan["data_sources"]
    save(package / "source_read_receipt.json", receipt)
    with pytest.raises(ValueError):
        cli.load_packet(w)


def test_execute_and_collection_reject_current_cap_before_credentials_or_network(
    source_workspace, materials, tmp_path
):
    with pytest.raises(ValueError, match="new explicit grant required"):
        asyncio.run(cli.execute(source_workspace))
    packet = study.make_packet(materials, "fictional")

    class NeverNetwork:
        async def get(self, *args, **kwargs):
            pytest.fail("must not connect")

        async def post(self, *args, **kwargs):
            pytest.fail("must not connect")

    output = tmp_path / "unapproved"
    with pytest.raises(ValueError, match="exceeds authorized cap"):
        asyncio.run(
            collection.collect(
                packet,
                output,
                NeverNetwork(),
                {166: 0.5},
                study.evaluate,
                {"prior_POSTs": 91, "global_cap": 100, "grant_source": "fictional"},
            )
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "attack",
    [
        None,
        "cap100",
        "bool_cap",
        "bool_prior",
        "short_batch",
        "endpoint",
        "model",
        "empty_reply",
        "packet",
        "ledger",
        "synthetic",
        "not_approved",
    ],
)
def test_new_explicit_grant_is_bound_to_complete_packet(source_workspace, attack):
    w = source_workspace
    cli.prepare(w)
    packet, _ = cli.load_packet(w)
    folder = w / cli.ITERATION
    plan = json.loads((folder / "plan.json").read_text())
    grant = {
        "approved": True,
        "specific_user_reply": "fictional test grant, not real permission",
        "question_item_id": "fictional-question",
        "endpoint": study.ENDPOINT,
        "model": study.MODEL,
        "prior_POSTs": 91,
        "global_cumulative_completion_POST_cap": 115,
        "maximum_new_completion_POSTs": 24,
        "packet_fingerprint": fingerprint(folder / "package/packet.json"),
        "prior_ledger_fingerprint": plan["metadata_sources"]["budget_ledger.json"]["fingerprint"],
    }
    if attack == "cap100":
        grant["global_cumulative_completion_POST_cap"] = 100
    elif attack == "bool_cap":
        grant["global_cumulative_completion_POST_cap"] = True
    elif attack == "bool_prior":
        grant["prior_POSTs"] = True
    elif attack == "short_batch":
        grant["maximum_new_completion_POSTs"] = 9
    elif attack == "endpoint":
        grant["endpoint"] = "https://unapproved.invalid/"
    elif attack == "model":
        grant["model"] = "other"
    elif attack == "empty_reply":
        grant["specific_user_reply"] = ""
    elif attack == "packet":
        grant["packet_fingerprint"]["sha256"] = "f" * 64
    elif attack == "ledger":
        grant["prior_ledger_fingerprint"] = {"bytes": 0, "sha256": "f" * 64}
    elif attack == "synthetic":
        packet = study.make_packet(packet["materials"], cli.RUN_ID)
    elif attack == "not_approved":
        grant["approved"] = False
    save(folder / "execution_authorization.json", grant)
    if attack is None:
        actual = cli.checked_authorization(w, packet)
        assert actual["prior_POSTs"] == 91 and actual["global_cap"] == 115
    else:
        with pytest.raises(ValueError):
            cli.checked_authorization(w, packet)


@pytest.mark.parametrize("stop_at", [1, 12, 24])
@pytest.mark.parametrize(
    "failure", ["http", "transport", "outer_duplicate", "inner_duplicate", "mode", "cancel"]
)
def test_collector_stops_without_retry_and_accounts_cancel(materials, tmp_path, stop_at, failure):
    packet = study.make_packet(materials, "fictional")
    sent = []

    async def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": study.MODEL}]})
        assert str(request.url) == study.ENDPOINT
        sent.append(json.loads(request.content))
        if len(sent) == stop_at:
            if failure == "http":
                return httpx.Response(502)
            if failure == "transport":
                raise httpx.ReadError("fictional", request=request)
            if failure == "cancel":
                raise asyncio.CancelledError
            if failure == "outer_duplicate":
                return httpx.Response(200, text=json.dumps(body())[:-1] + ',"model":"other"}')
            bad = body(n=len(sent))
            if failure == "inner_duplicate":
                bad["choices"][0]["message"]["content"] = '{"choice":"A","choice":"B"}'
            else:
                bad["choices"][0]["message"]["reasoning_content"] = "fictional reasoning"
            return httpx.Response(200, json=bad)
        return httpx.Response(200, json=body(n=len(sent)))

    output = tmp_path / "run"

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await collection.collect(
                packet,
                output,
                client,
                {166: 0.5},
                study.evaluate,
                {
                    "prior_POSTs": 91,
                    "global_cap": 115,
                    "grant_source": "fictional; not real permission",
                },
            )

    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run())
    else:
        result = asyncio.run(run())
        assert not result["protocol_complete"]
        failed_method = packet["requests"][stop_at - 1]["method_id"]
        for comp in result["comparisons"].values():
            assert comp["planned_records"] == 24
            if failed_method in (comp["left_method"], comp["right_method"]):
                assert comp["primary"]["left_minus_right_MAE"] is None
    assert len(sent) == stop_at and sent == [r["payload"] for r in packet["requests"][:stop_at]]
    ledger = json.loads((output / "budget_ledger.json").read_text())
    assert ledger["used_completion_POSTs"] == 91 + stop_at
    assert ledger["active_run_id"] is None and ledger["reserved_next_completion_POSTs"] == 0
    assert ledger["available_unreserved_POSTs"] == 24 - stop_at
    assert len(collection.read_jsonl(output / collection.LOGS[2])) == stop_at
    with pytest.raises(FileExistsError):
        asyncio.run(run())


def test_complete_repeated_returns_retained(materials, tmp_path):
    packet = study.make_packet(materials, "fictional")

    async def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": study.MODEL}]})
        return httpx.Response(200, json=body("A", 1))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await collection.collect(
                packet,
                tmp_path / "complete",
                client,
                {166: 0.5},
                study.evaluate,
                {"prior_POSTs": 91, "global_cap": 115, "grant_source": "fictional"},
            )

    result = asyncio.run(run())
    assert result["protocol_complete"] and result["used_completion_POSTs"] == 115
    assert len(result["repeated_strong_signature_groups"][0]) == 24
    assert result["independent_generation_count"] is None
