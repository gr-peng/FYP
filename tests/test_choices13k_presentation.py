"""Hypothetical in-memory HTTP only; never use real credentials or datasets."""

import asyncio
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from behavioral_market.evaluation import choice_presentation as study
from behavioral_market.evaluation.style_mechanism_design import canonical
from scripts import choices13k_presentation as cli
from scripts.style_fact_check import fingerprint, read_jsonl, save


@pytest.fixture
def sample(tmp_path):
    w = tmp_path
    index_path = "iterations/choices13k_p0_full_calibration_reuse_20261022/index/index.json"
    selection_path = "iterations/choices13k_p0_full_calibration_reuse_20261022/selection.json"
    prompts_path = "behavioral-llm-market/src/behavioral_market/benchmarks/prompts.py"
    grant_path, ledger_path = "iterations/grant.json", "iterations/accepted_ledger.json"
    for relative in [
        cli.RAW,
        cli.ORIGIN,
        cli.ITERATION + "source_offsets.json",
        index_path,
        selection_path,
        prompts_path,
        grant_path,
        ledger_path,
    ]:
        (w / relative).parent.mkdir(parents=True, exist_ok=True)
    entries, selectors, blob = [], [], b""
    for item in [166, 211]:
        originals = {
            "A": f"- 100% chance of receiving {item}",
            "B": "- receiving one of these payoffs, "
            "with fixed but undisclosed probabilities: 0, 500",
        }
        for repeat, order in enumerate(["A_first", "B_first"]):
            content = "AB" if order == "A_first" else "BA"
            user = (
                "P0: preserve A/B letters in this instruction."
                + "".join(
                    f"\n\nOption {label}:\n{originals[original]}"
                    for label, original in zip("AB", content, strict=True)
                )
                + study.SUFFIX
            )
            payload = {
                "model": study.MODEL,
                "temperature": 0.7,
                "max_tokens": 32,
                "enable_thinking": False,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "fictional choose one"},
                    {"role": "user", "content": user},
                ],
            }
            ident = {
                "run_id": "choices13k_deepseek_pilot",
                "split": "calibration",
                "prompt_id": "P0",
                "item_id": item,
                "problem_id": 777,
                "repeat_id": repeat,
                "display_order": order,
                "model": study.MODEL,
                "temperature": 0.7,
                "repair": 0,
            }
            hashed = hashlib.sha256(
                json.dumps([payload, ident, "choices13k-v1"], sort_keys=True).encode()
            ).hexdigest()
            meta = {**ident, "request_hash": hashed}
            line = (json.dumps({**meta, "payload": payload}) + "\n").encode()
            entries.append(
                {
                    "line_number": len(entries) + 1,
                    "offset": len(blob),
                    "length": len(line),
                    "sha256": hashlib.sha256(line).hexdigest(),
                    "metadata": meta,
                }
            )
            selectors.append({k: ident[k] for k in ["item_id", "repeat_id", "display_order"]})
            blob += line
    # Hashable but invalid body outside the four indexed requests: must never decode it.
    blob += b'{"prompt_id":"P1","payload":{"MUST_NOT_DECODE":broken}}\n'
    (w / cli.RAW).write_bytes(blob)
    save(w / index_path, {"files": {cli.RAW: {"selected_entries": entries}}})
    save(w / selection_path, {"selected_item_ids": [166, 211, 384]})
    (w / prompts_path).write_text("fictional frozen prompt source")

    def desc(relative):
        return {"path": relative, "fingerprint": fingerprint(w / relative)}

    save(
        w / grant_path,
        {
            "global_cumulative_completion_POST_cap": 100,
            "endpoint": study.ENDPOINT,
            "model": study.MODEL,
            "specific_user_reply": "fictional grant, not real permission",
            "prior_used_completion_POSTs": 25,
        },
    )
    save(
        w / ledger_path,
        {
            "used_completion_POSTs": 59,
            "global_authorized_completion_POST_cap": 100,
            "reserved_next_completion_POSTs": 0,
            "active_run_id": None,
            "status": "fictional_accepted",
            "global_authorization_source": str(w / grant_path),
        },
    )
    scope = {
        "raw_request_file": desc(cli.RAW),
        "frozen_index": desc(index_path),
        "accepted_selection": desc(selection_path),
        "original_prompt_source": desc(prompts_path),
    }
    origin = {
        "schedule": study.SCHEDULE,
        "authorization_source": desc(grant_path),
        "latest_accepted_ledger": desc(ledger_path),
        "material_source_scope": scope,
        "allowed_request_payload_bodies": selectors,
    }
    save(w / cli.ORIGIN, origin)
    save(
        w / (cli.ITERATION + "source_offsets.json"),
        {**scope, "entries": entries, "allowed_request_payload_bodies": selectors},
    )
    plan = {
        "origin_plan": desc(cli.ORIGIN),
        "source_offsets": desc(cli.ITERATION + "source_offsets.json"),
        "selected_item_ids": [166, 211],
        "schedule": deepcopy(study.SCHEDULE),
        "fixed_execution_output": str(w / cli.ITERATION / "live_execution"),
        "frozen_run_id": "fictional-presentation",
        "evidence_kind": "synthetic_tool_test",
        "endpoint": study.ENDPOINT,
        "model": study.MODEL,
        "prior_POSTs": 59,
        "batch_cumulative_stop": 75,
        "max_new_completion_POSTs": 16,
        "compatibility_calls_outside_allocation": 0,
        "budget_sources": {"grant": desc(grant_path), "prior_ledger": desc(ledger_path)},
        "technical_delta": {"chat_template_kwargs": study.MODE_CONTROL, "cache": study.CACHE},
    }
    return SimpleNamespace(
        workspace=w,
        plan=plan,
        package=cli.packet_for(w, plan),
        output=Path(plan["fixed_execution_output"]),
    )


def body(choice="B", identifier="mock-id"):
    return {
        "id": identifier,
        "created": 123,
        "model": study.MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"choice": choice}),
                    "reasoning_content": None,
                },
            }
        ],
        "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
    }


def run(
    sample,
    *,
    policy="original_B",
    failure=None,
    fail_at=8,
    calls=None,
    alter=None,
    missing_id=False,
):
    calls = calls if calls is not None else []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": study.MODEL}]})
        calls.append(request)
        slot = sample.package["requests"][len(calls) - 1]
        assert json.loads(request.content) == slot["payload"]
        if policy == "original_B":
            choice = slot["letter_order"][slot["content_order"].index("B")]
        elif policy == "first":
            choice = slot["letter_order"][0]
        elif policy == "second":
            choice = slot["letter_order"][1]
        elif policy == "opposite_repeats":
            choice = "A" if slot["repeat_id"] == 0 else "B"
        else:
            choice = "B"
        value = body(choice)
        if missing_id:
            value.pop("id")
            value.pop("created")
        if len(calls) == fail_at:
            if alter:
                alter(sample.output)
            if failure and failure.startswith("outer_"):
                raw = json.dumps(value)
                if failure == "outer_model":
                    raw = '{"model":"WRONG_MODEL",' + raw[1:]
                elif failure == "outer_choices":
                    raw = '{"choices":[],' + raw[1:]
                elif failure == "outer_reasoning":
                    raw = raw.replace(
                        '"reasoning_tokens": 0', '"reasoning_tokens": 999, "reasoning_tokens": 0'
                    )
                else:
                    number = "NaN" if failure == "outer_nan" else "1e999"
                    raw = '{"unexpected_nonfinite":' + number + "," + raw[1:]
                return httpx.Response(200, text=raw)
            if failure == "http":
                return httpx.Response(503)
            if failure == "transport":
                raise httpx.ReadTimeout("fictional", request=request)
            if failure == "cancel":
                raise asyncio.CancelledError
            if failure == "json":
                value["choices"][0]["message"]["content"] = "{"
            if failure == "schema":
                value["choices"][0]["message"]["content"] = '{"choice":"A","extra":1}'
            if failure == "finish":
                value["choices"][0]["finish_reason"] = "length"
            if failure == "model":
                value["model"] = "wrong"
            if failure == "mode":
                value["usage"]["completion_tokens_details"]["reasoning_tokens"] = 1
        return httpx.Response(
            200,
            json=value,
            headers={"x-litellm-cache-key": "opaque", "x-sensitive": "IN_MEMORY_SECRET"},
        )

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await cli.collect(
                sample.package, sample.workspace, client, secret="IN_MEMORY_SECRET"
            )

    return asyncio.run(go())


@pytest.mark.parametrize(
    "policy,position,label",
    [("original_B", 0, 0), ("letter_B", 0, 1), ("first", 1, 0), ("second", -1, 0)],
)
def test_disentangles_content_letter_and_position(sample, policy, position, label):
    calls = []
    result = run(sample, policy=policy, calls=calls)
    assert len(calls) == 16 and result["protocol_complete"]
    assert all(
        e["difference"] == (position if e["factor"].startswith("position") else label)
        for e in result["contrasts"]
    )
    assert result["valid_returns"] == 16
    assert len(result["repeated_id_created_text_groups"]) >= 1
    ledger = json.loads((sample.output / "budget_ledger.json").read_text())
    assert ledger["used_completion_POSTs"] == 75 and ledger["reserved_next_completion_POSTs"] == 0
    text = "".join(p.read_text() for p in sample.output.iterdir() if p.is_file())
    assert "IN_MEMORY_SECRET" not in text and "x-sensitive" not in text


def test_opposite_same_input_answers_kept_and_wire_inputs_identical(sample):
    result = run(sample, policy="opposite_repeats")
    assert all(
        c["received_original_B_rate"] == 0.5 and c["valid_returns"] == 2 for c in result["cells"]
    )
    for cell in result["cells"]:
        a, b = [r for r in sample.package["requests"] if r["slot_id"] in cell["slot_ids"]]
        assert a["payload"] == b["payload"] and a["request_hash"] != b["request_hash"]
    assert result["independent_generation_count"] is None


@pytest.mark.parametrize("at", [1, 8, 16])
@pytest.mark.parametrize(
    "failure", ["http", "transport", "json", "schema", "finish", "model", "mode"]
)
def test_first_anomaly_stops_without_extra_compatibility_or_retry(sample, at, failure):
    calls = []
    result = run(sample, fail_at=at, failure=failure, calls=calls)
    assert len(calls) == at and result["logged_dispatch_attempts"] == at
    assert not result["protocol_complete"] and result["valid_returns"] == at - 1
    assert any(e["difference"] is None for e in result["contrasts"])
    ledger = json.loads((sample.output / "budget_ledger.json").read_text())
    assert ledger["used_completion_POSTs"] == 59 + at
    assert json.loads((sample.output / "execution_state.json").read_text())["status"] == "stopped"
    with pytest.raises(FileExistsError):
        run(sample, calls=calls)
    assert len(calls) == at


@pytest.mark.parametrize("at", [8, 16])
def test_cancel_counts_dispatch_and_never_resumes(sample, at):
    calls = []
    with pytest.raises(asyncio.CancelledError):
        run(sample, fail_at=at, failure="cancel", calls=calls)
    assert len(calls) == at
    assert (
        json.loads((sample.output / "budget_ledger.json").read_text())["used_completion_POSTs"]
        == 59 + at
    )
    assert json.loads((sample.output / "execution_state.json").read_text())["status"] == "cancelled"
    assert len(read_jsonl(sample.output / study.LOGS[2])) == at
    with pytest.raises(FileExistsError):
        run(sample, calls=calls)


@pytest.mark.parametrize(
    "failure,at",
    [
        ("outer_model", 1),
        ("outer_model", 8),
        ("outer_model", 16),
        ("outer_choices", 1),
        ("outer_reasoning", 1),
        ("outer_nan", 1),
        ("outer_overflow", 1),
    ],
)
def test_outer_json_ambiguity_or_nonfinite_stops_and_preserves_original(sample, failure, at):
    calls = []
    result = run(sample, failure=failure, fail_at=at, calls=calls)
    assert len(calls) == at and result["valid_returns"] == at - 1
    assert not result["protocol_complete"]
    response = read_jsonl(sample.output / study.LOGS[2])[-1]
    assert response["http_status"] == 200 and set(response["response"]) == {"_invalid_json_text"}
    ledger = json.loads((sample.output / "budget_ledger.json").read_text())
    assert ledger["used_completion_POSTs"] == 59 + at
    assert all(r["status"] == "unsubmitted" for r in result["rows"][at:])


def test_missing_return_metadata_is_unknown_not_reuse_or_retry(sample):
    result = run(sample, missing_id=True)
    assert result["valid_returns"] == 16
    assert result["repeated_id_created_text_groups"] == []
    assert len(result["unknown_strong_signature_slots"]) == 16


@pytest.mark.parametrize("bad", ["duplicate", "missing", "order", "bool_retry", "clock"])
def test_corrupt_prefix_stops_before_next_post(sample, bad):
    def corrupt(output):
        name = study.LOGS[1] if bad == "bool_retry" else study.LOGS[2]
        rows = read_jsonl(output / name)
        if bad == "duplicate":
            rows.append(deepcopy(rows[0]))
        elif bad == "missing":
            rows.pop(0)
        elif bad == "order":
            rows[0], rows[1] = rows[1], rows[0]
        elif bad == "bool_retry":
            rows[0]["retry_count"] = False
        else:
            rows[0]["requested_at"] = "2000-01-01T00:00:00+00:00"
        (output / name).write_text("".join(canonical(r) + "\n" for r in rows))

    calls = []
    with pytest.raises(ValueError):
        run(sample, calls=calls, alter=corrupt, fail_at=8)
    assert len(calls) == 8
    assert (
        json.loads((sample.output / "budget_ledger.json").read_text())["used_completion_POSTs"]
        == 67
    )


def test_packet_mutation_and_budget_change_refuse_before_network(sample):
    changed = deepcopy(sample.package)
    changed["requests"][0]["payload"]["messages"][1]["content"] += " altered payoff"
    with pytest.raises(ValueError):
        cli.checked_packet(sample.workspace, changed)
    plan = deepcopy(sample.plan)
    plan["prior_POSTs"] = 60
    with pytest.raises(ValueError):
        cli.packet_for(sample.workspace, plan)
    ledger = sample.workspace / sample.plan["budget_sources"]["prior_ledger"]["path"]
    value = json.loads(ledger.read_text())
    value["used_completion_POSTs"] = 60
    save(ledger, value)
    with pytest.raises(ValueError):
        cli.checked_packet(sample.workspace, sample.package)


def test_dangling_next_request_stops_before_its_post(sample):
    def ghost(output):
        with (output / study.LOGS[0]).open("a") as handle:
            handle.write(canonical(sample.package["requests"][8]) + "\n")

    calls = []
    result = run(sample, calls=calls, alter=ghost, fail_at=8)
    assert len(calls) == 8 and result["logged_dispatch_attempts"] == 8
    assert result["rows"][8]["status"] == "request_record_only"
    assert json.loads((sample.output / "execution_state.json").read_text())["status"] == "stopped"


def test_exact_transform_preserves_all_option_text_and_original_wire_pair(sample):
    for request in sample.package["requests"]:
        item, content, labels = (
            str(request["item_id"]),
            request["content_order"],
            request["letter_order"],
        )
        base = sample.package["bases"][item]["A_first"]
        head, original_a, original_b = study.split_user(base["messages"][1]["content"])
        user = request["payload"]["messages"][1]["content"]
        assert user.startswith(head) and user.endswith(study.SUFFIX)
        assert original_a in user and original_b in user
        if labels == "AB":
            old = sample.package["bases"][item]["A_first" if content == "AB" else "B_first"]
            stripped = deepcopy(request["payload"])
            stripped.pop("chat_template_kwargs")
            stripped.pop("cache")
            assert stripped == old


def test_synthetic_package_cannot_read_real_credentials(sample, monkeypatch):
    plan_path = sample.workspace / "iterations/plan.json"
    save(plan_path, sample.plan)
    folder = sample.workspace / "iterations/package"
    cli.prepare(SimpleNamespace(workspace=sample.workspace, plan=plan_path, output=folder))
    import dotenv

    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **k: pytest.fail("credential access"))
    with pytest.raises(ValueError, match="synthetic"):
        asyncio.run(cli.execute(SimpleNamespace(workspace=sample.workspace, package=folder)))
