"""Fictional fixtures and in-memory HTTP only; no real data or credentials."""

import asyncio
import hashlib
import json
from copy import deepcopy

import httpx
import pytest

from behavioral_market.benchmarks import reference_comparison as study
from behavioral_market.evaluation.style_mechanism_design import canonical
from scripts import choices13k_reference_comparison as cli
from scripts.style_fact_check import fingerprint, read_jsonl, save


@pytest.fixture
def materials():
    return {
        "target": {
            "item_id": 384,
            "option_a_text": "fictional target low",
            "option_b_text": "fictional target high",
        },
        "references": [
            {
                "item_id": item,
                "option_a_text": f"fictional ref {n} low",
                "option_b_text": f"fictional ref {n} high",
                "human_b_rate": rate,
            }
            for n, (item, rate) in enumerate(zip(study.REFERENCES, (0.2, 0.4, 0.8), strict=True))
        ],
    }


def body(choice="A", number=0):
    return {
        "id": f"fictional-{number}",
        "created": 1 + number,
        "model": study.MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps({"choice": choice})},
            }
        ],
    }


def logs(packet):
    requests, attempts, responses = [], [], []
    for req in packet["requests"]:
        ident = study.identity(req)
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
                "response": body("A" if req["repeat_id"] == 0 else "B", req["position"]),
                "cache_key_header": {"present": False, "value_sha256": None},
            }
        )
    return requests, attempts, responses


def test_pairing_and_no_condition_disclosure(materials):
    before = deepcopy(materials)
    true = study.reference_cards(materials, "true_reference")
    shuffled = study.reference_cards(materials, "shuffled_reference")
    assert [r["assigned_rate"] for r in true] == [0.2, 0.4, 0.8]
    assert [r["assigned_rate"] for r in shuffled] == [0.4, 0.8, 0.2]
    assert [{k: v for k, v in r.items() if k != "assigned_rate"} for r in true] == [
        {k: v for k, v in r.items() if k != "assigned_rate"} for r in shuffled
    ]
    packet = study.make_packet(materials, "fictional")
    for req in packet["requests"]:
        text = canonical(req["payload"])
        assert not any(
            token in text
            for token in ("true_reference", "shuffled_reference", "rate_source_item_id", "shuffled")
        )
    assert materials == before


def test_exact_schedule_and_mapping(materials):
    packet = study.make_packet(materials, "fictional")
    assert [
        (r["content_order"], r["letter_order"], r["repeat_id"], r["method_id"])
        for r in packet["requests"]
    ] == [
        (content, letters, repeat, method)
        for content, letters in [("AB", "AB"), ("AB", "BA"), ("BA", "AB"), ("BA", "BA")]
        for repeat in (0, 1)
        for method in (
            ("true_reference", "shuffled_reference")
            if repeat == 0
            else ("shuffled_reference", "true_reference")
        )
    ]
    expected = {
        study.panel_tool.key(r): r["presentation"] for r in packet["panel"]["opportunities"]
    }
    for req in packet["requests"]:
        common = expected[study.panel_tool.key(req)]
        assert req["payload"]["messages"][1]["content"].endswith(common["messages"][1]["content"])
        assert req["presentation_sha256"] == study.digest(common)
        assert req["wire_sha256"] == study.digest(req["payload"])
        assert "Original Option B:" in req["payload"]["messages"][1]["content"]
    assert study.validate_packet(json.loads(canonical(packet))) == packet


@pytest.mark.parametrize(
    "field", ["human_b_rate", "target_rate", "rate_source_item_id", "metadata"]
)
def test_target_extra_fields_rejected(materials, field):
    materials["target"][field] = 0.313159
    with pytest.raises(ValueError, match="source schema"):
        study.make_packet(materials, "fictional")


def test_local_target_changes_no_wire(materials):
    packet = study.make_packet(materials, "fictional")
    req, att, resp = logs(packet)
    a = study.evaluate(packet, req, att, resp, {384: 0.1})
    b = study.evaluate(packet, req, att, resp, {384: 0.9})
    assert (
        a["comparison"]["item_pairs"][0]["human_b_rate"]
        != b["comparison"]["item_pairs"][0]["human_b_rate"]
    )
    assert packet == study.make_packet(materials, "fictional")


@pytest.mark.parametrize("swap", [False, True])
def test_target_reference_duplicate_rejected(materials, swap):
    a, b = (materials["target"][k] for k in ("option_a_text", "option_b_text"))
    materials["references"][1].update(
        option_a_text=b if swap else a, option_b_text=a if swap else b
    )
    with pytest.raises(ValueError, match="reference equals target"):
        study.make_packet(materials, "fictional")


def test_equal_rates_rejected_but_target_numeric_coincidence_allowed(materials):
    packet = study.make_packet(materials, "fictional")
    assert study.evaluate(packet, *logs(packet), {384: 0.2})["protocol_complete"]
    for r in materials["references"]:
        r["human_b_rate"] = 0.2
    with pytest.raises(ValueError, match="identical"):
        study.make_packet(materials, "fictional")


@pytest.mark.parametrize("mutation", ["method", "wire", "order", "mapping", "rate", "hidden_field"])
def test_packet_tamper_rejected(materials, mutation):
    packet = study.make_packet(materials, "fictional")
    if mutation == "method":
        packet["requests"][0]["method_id"] = "shuffled_reference"
    elif mutation == "wire":
        packet["requests"][0]["payload"]["messages"][1]["content"] += "extra"
    elif mutation == "order":
        packet["requests"].reverse()
    elif mutation == "mapping":
        packet["panel"]["opportunities"][0]["presentation"]["mapping"]["A"]["original_option"] = "B"
    elif mutation == "rate":
        packet["materials"]["references"][0]["human_b_rate"] = 0.33
    else:
        packet["materials"]["target"]["human_b_rate"] = 0.31
    with pytest.raises(ValueError):
        study.validate_packet(packet)


def test_authentication_before_generic_and_repeats_kept(materials):
    packet = study.make_packet(materials, "fictional")
    result = study.evaluate(packet, *logs(packet), {384: 0.3})
    assert result["protocol_complete"]
    assert result["logged_dispatch_attempts"] == 16
    assert result["used_completion_POSTs"] == 91
    assert all(r["different_display_choices"] for r in result["same_input_variability"])
    assert result["comparison"]["primary"]["left_MAE"] == pytest.approx(0.2)
    assert result["comparison"]["primary"]["left_minus_right_MAE"] == 0
    assert result["independent_generation_count"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "wire",
        "method",
        "duplicate",
        "reorder",
        "bool_identity",
        "cached",
        "retry",
        "orphan",
        "after_invalid",
    ],
)
def test_log_attacks_rejected(materials, mutation):
    packet = study.make_packet(materials, "fictional")
    req, att, resp = logs(packet)
    if mutation == "wire":
        req[0]["payload"]["messages"][1]["content"] += "changed"
    elif mutation == "method":
        resp[0]["method_id"] = "shuffled_reference"
    elif mutation == "duplicate":
        req.append(req[0])
    elif mutation == "reorder":
        req[0], req[1] = req[1], req[0]
    elif mutation == "bool_identity":
        att[0]["position"] = True
    elif mutation == "cached":
        resp[0]["cached"] = True
    elif mutation == "retry":
        att[0]["retry_count"] = 1
    elif mutation == "orphan":
        att.pop()
    else:
        resp[0]["response"]["choices"][0]["message"]["content"] = '{"choice":"C"}'
    with pytest.raises(ValueError):
        study.evaluate(packet, req, att, resp, {384: 0.3})


def test_missing_response_keeps_denominator(materials):
    packet = study.make_packet(materials, "fictional")
    req, att, resp = logs(packet)
    result = study.evaluate(packet, req, att, resp[:-1], {384: 0.3})
    assert result["comparison"]["planned_records"] == 16
    assert result["comparison"]["missing_records"] == 1
    assert result["comparison"]["primary"]["left_minus_right_MAE"] is None
    assert result["comparison"]["primary"]["left_MAE"] is None


def test_synthetic_logs_cannot_be_relabelled_recorded(materials):
    fake = study.make_packet(materials, "same-run", "synthetic_tool_test")
    real = study.make_packet(materials, "same-run", "recorded_model_logs")
    assert [r["payload"] for r in fake["requests"]] == [r["payload"] for r in real["requests"]]
    assert fake["requests"][0]["request_hash"] != real["requests"][0]["request_hash"]
    with pytest.raises(ValueError, match="log identity differs"):
        study.evaluate(real, *logs(fake), {384: 0.3})
    fake["evidence_kind"] = "recorded_model_logs"
    core = {k: v for k, v in fake.items() if k != "packet_sha256"}
    fake["packet_sha256"] = study.digest(core)
    with pytest.raises(ValueError, match="rebuilt materials"):
        study.validate_packet(fake)


def test_execute_rejects_synthetic_before_credentials(materials, tmp_path, monkeypatch):
    folder = tmp_path / cli.ITERATION
    folder.mkdir(parents=True)
    save(folder / "pre_execution_freeze.json", {"files": {}})
    save(folder / "pre_execution_review.json", {"status": "accepted"})
    fake = study.make_packet(materials, "fictional")
    monkeypatch.setattr(cli, "load_packet", lambda *args: (fake, {384: 0.3}))
    with pytest.raises(ValueError, match="synthetic cannot execute live"):
        asyncio.run(cli.execute(tmp_path, folder / "package"))


@pytest.mark.parametrize("stop_at", [1, 8, 16])
@pytest.mark.parametrize(
    "failure", ["http", "transport", "outer_duplicate", "inner_json", "mode", "cancel"]
)
def test_actual_collector_stops_and_accounts(materials, tmp_path, stop_at, failure):
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
                raise httpx.ReadError("fictional transport", request=request)
            if failure == "cancel":
                raise asyncio.CancelledError
            if failure == "outer_duplicate":
                text = json.dumps(body(number=len(sent)))
                return httpx.Response(200, text=text[:-1] + ',"model":"other"}')
            bad = body(number=len(sent))
            if failure == "inner_json":
                bad["choices"][0]["message"]["content"] = '{"choice":"A","choice":"B"}'
            else:
                bad["choices"][0]["message"]["reasoning_content"] = "fictional reasoning"
            return httpx.Response(200, json=bad)
        return httpx.Response(200, json=body(number=len(sent)))

    output = tmp_path / "run"

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await cli.collect(
                packet, output, client, {384: 0.3}, tmp_path / "fictional_grant"
            )

    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run())
    else:
        result = asyncio.run(run())
        assert result["comparison"]["planned_records"] == 16
        assert result["comparison"]["primary"]["left_minus_right_MAE"] is None
    assert len(sent) == stop_at
    assert sent == [r["payload"] for r in packet["requests"][:stop_at]]
    ledger = json.loads((output / "budget_ledger.json").read_text())
    assert ledger["used_completion_POSTs"] == 75 + stop_at
    assert ledger["active_run_id"] is None and ledger["reserved_next_completion_POSTs"] == 0
    assert ledger["available_unreserved_POSTs"] == 25 - stop_at
    assert len(read_jsonl(output / study.LOGS[2])) == stop_at
    with pytest.raises(FileExistsError):
        asyncio.run(run())


def test_collector_complete_and_registry_failure(materials, tmp_path):
    packet = study.make_packet(materials, "fictional")
    sent = []

    async def run(output, registry_ok):
        async def handler(request):
            if request.method == "GET":
                return httpx.Response(
                    200 if registry_ok else 500, json={"data": [{"id": study.MODEL}]}
                )
            sent.append(json.loads(request.content))
            return httpx.Response(200, json=body("B", number=1))  # Identical signatures retained.

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await cli.collect(
                packet, output, client, {384: 0.3}, tmp_path / "fictional_grant"
            )

    result = asyncio.run(run(tmp_path / "complete", True))
    assert result["protocol_complete"] and len(sent) == 16
    assert result["comparison"]["primary"]["left_minus_right_MAE"] == 0
    assert len(result["repeated_strong_signature_groups"][0]) == 16
    sent.clear()
    failed = asyncio.run(run(tmp_path / "registry_failure", False))
    assert failed["logged_dispatch_attempts"] == 0 and not sent
    assert failed["comparison"]["primary"]["left_MAE"] is None


def test_bounded_source_prepare_and_frozen_reload(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    workspace = tmp_path
    folder = workspace / cli.ITERATION
    folder.mkdir(parents=True)
    save(folder / "plan.json", {"fictional": True})

    def desc(path):
        return {"path": str(path.relative_to(workspace)), "fingerprint": fingerprint(path)}

    def payload(item, order):
        content = "AB" if order == "A_first" else "BA"
        result = {
            k: deepcopy(v)
            for k, v in study.CONTROLS.items()
            if k not in {"cache", "chat_template_kwargs"}
        }
        result["messages"] = study.panel_tool.render_presentations(
            f"fictional block {item} original A", f"fictional block {item} original B"
        )[0 if content == "AB" else 2]["messages"]
        return result

    old_path, raw_path, index_path = (
        folder / "old.json",
        folder / "raw.jsonl",
        folder / "index.json",
    )
    save(
        old_path,
        {"bases": {str(i): {o: payload(i, o) for o in ("A_first", "B_first")} for i in (166, 211)}},
    )
    entries, raw = [], b""
    for item in (384, 542):
        for repeat, order in enumerate(("A_first", "B_first")):
            pay = payload(item, order)
            ident = {
                "run_id": "choices13k_deepseek_pilot",
                "split": "calibration",
                "prompt_id": "P0",
                "item_id": item,
                "problem_id": item,
                "repeat_id": repeat,
                "display_order": order,
                "model": study.MODEL,
                "temperature": 0.7,
                "repair": 0,
            }
            hashed = hashlib.sha256(
                json.dumps([pay, ident, "choices13k-v1"], sort_keys=True).encode()
            ).hexdigest()
            meta = {**ident, "request_hash": hashed}
            blob = (json.dumps({**meta, "payload": pay}) + "\n").encode()
            entries.append(
                {
                    "offset": len(raw),
                    "length": len(blob),
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "metadata": meta,
                }
            )
            raw += blob
    raw += b'{"prompt_id":"P1","payload":MUST_NOT_DECODE}\n'
    raw_path.write_bytes(raw)
    save(
        index_path, {"files": {str(raw_path.relative_to(workspace)): {"selected_entries": entries}}}
    )
    source_offsets = folder / "source_offsets.json"
    save(
        source_offsets,
        {"raw_request_file": desc(raw_path), "frozen_index": desc(index_path), "entries": entries},
    )
    parquet = folder / "fictional.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"item_id": i, "human_b_rate": rate, "individual_PRIVATE": "MUST_NOT_LOAD"}
                for i, rate in zip(
                    (166, 211, 384, 542, 999), (0.2, 0.4, 0.317, 0.8, 0.99), strict=True
                )
            ]
        ),
        parquet,
    )
    plan = {
        "sources": {
            "source_offsets": desc(source_offsets),
            "frozen_index": desc(index_path),
            "raw_requests": desc(raw_path),
            "existing_presentation_packet": desc(old_path),
            "human_calibration": desc(parquet),
        },
        "human_read": {
            "columns": ["item_id", "human_b_rate"],
            "filter_item_ids": [166, 211, 384, 542],
        },
        "frozen_run_id": "fictional-source-check",
    }
    monkeypatch.setattr(cli, "checked_plan", lambda w: deepcopy(plan))
    calls, original = [], pq.read_table

    def projection(path, **kwargs):
        calls.append(kwargs)
        return original(path, **kwargs)

    monkeypatch.setattr(pq, "read_table", projection)
    output = folder / "package"
    result = cli.prepare(workspace, output)
    assert result["prepared_not_sent"]
    assert calls == [
        {
            "columns": ["item_id", "human_b_rate"],
            "filters": [("item_id", "in", [166, 211, 384, 542])],
        }
    ]
    receipt = json.loads((output / "source_read_receipt.json").read_text())
    assert receipt["selected_request_body_decodes"] == 4 and receipt["human_rows_materialized"] == 4
    packet, targets = cli.load_packet(workspace, output)
    assert targets == {384: 0.317} and len(calls) == 2
    wire = canonical([r["payload"] for r in packet["requests"]])
    assert "0.317" not in wire and "MUST_NOT_LOAD" not in wire
    assert set(packet["materials"]["target"]) == study.TEXT_FIELDS
    # A changed local source cannot be legitimized by the packet's own checksum.
    originals = {p.name: p.read_bytes() for p in output.iterdir()}
    changed = json.loads((output / "source_materials.json").read_text())
    changed["target"]["option_b_text"] += "tampered"
    save(output / "source_materials.json", changed)
    with pytest.raises(ValueError, match="derived source fingerprint"):
        cli.load_packet(workspace, output)

    def restore():
        for name, content in originals.items():
            (output / name).write_bytes(content)

    # Re-signing all derived artifacts cannot replace the frozen original sources.
    restore()
    changed = json.loads(originals["source_materials.json"])
    changed["references"][0]["human_b_rate"] = 0.91
    changed_packet = study.make_packet(changed, plan["frozen_run_id"], "recorded_model_logs")
    save(output / "source_materials.json", changed)
    save(output / "packet.json", changed_packet)
    (output / "expected_requests.jsonl").write_text(
        "".join(canonical(r) + "\n" for r in changed_packet["requests"])
    )
    receipt["derived_files"] = {
        name: fingerprint(output / name) for name in receipt["derived_files"]
    }
    save(output / "source_read_receipt.json", receipt)
    with pytest.raises(ValueError, match="materials differ from original sources"):
        cli.load_packet(workspace, output)
    restore()
    save(output / "local_target.json", {"item_id": 384, "human_b_rate": 0.987})
    receipt = json.loads(originals["source_read_receipt.json"])
    receipt["derived_files"]["local_target.json"] = fingerprint(output / "local_target.json")
    save(output / "source_read_receipt.json", receipt)
    with pytest.raises(ValueError, match="local target differs from original source"):
        cli.load_packet(workspace, output)
    restore()
    receipt = json.loads(originals["source_read_receipt.json"])
    receipt["derived_files"]["../MUST_NOT_OPEN"] = {"bytes": 0, "sha256": "f" * 64}
    save(output / "source_read_receipt.json", receipt)
    with pytest.raises(ValueError, match="fixed derived file names"):
        cli.load_packet(workspace, output)
