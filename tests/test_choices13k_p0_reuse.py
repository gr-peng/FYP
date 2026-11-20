"""Selected AI logs only, using fictional files; never access real answers."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from behavioral_market.benchmarks.sampling import orientations
from behavioral_market.evaluation.style_mechanism_design import canonical
from scripts import audit_choices13k_p0_reuse as mod
from scripts.style_fact_check import fingerprint


def make_request(item, repeat, order, repair=0):
    identity = {
        "run_id": "choices13k_deepseek_pilot",
        "split": "calibration",
        "prompt_id": "P0",
        "item_id": item,
        "problem_id": 777,
        "repeat_id": repeat,
        "display_order": order,
        "model": "DeepSeek-V4-Flash",
        "temperature": 0.7,
        "repair": repair,
    }
    payload = {
        "model": "DeepSeek-V4-Flash",
        "temperature": 0.7,
        "max_tokens": 32,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": f"fictional {item} {order} repair{repair}"}],
    }
    hashed = hashlib.sha256(
        json.dumps([payload, identity, "choices13k-v1"], sort_keys=True).encode()
    ).hexdigest()
    return {**identity, "request_hash": hashed, "payload": payload}


def source_line(record, body_key):
    # Preserve the historical body-last layout rather than canonical key sorting.
    value = {k: v for k, v in record.items() if k != body_key}
    if body_key in record:
        value[body_key] = record[body_key]
    return (json.dumps(value) + "\n").encode()


@pytest.fixture
def sample(tmp_path, request):
    item_count = getattr(request, "param", 20)
    opportunities = [
        {"item_id": i, "repeat_id": j, "display_order": o}
        for i in range(item_count)
        for j, o in enumerate(orientations(i, 10, 20260916))
    ]
    rows = {name: [] for name in mod.NAMES}
    for op in opportunities:
        req = make_request(op["item_id"], op["repeat_id"], op["display_order"])
        ident = {k: v for k, v in req.items() if k != "payload"}
        body = {
            "model": "DeepSeek-V4-Flash",
            "id": f"id-{op['item_id']}-{op['display_order']}",
            "created": 123,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"choice":"A"}',
                    },
                }
            ],
        }
        rows[mod.NAMES[0]].append(req)
        rows[mod.NAMES[1]].append(
            {
                **ident,
                "retry_count": 0,
                "dispatched_at": "2026-09-16T00:00:00+00:00",
            }
        )
        rows[mod.NAMES[2]].append(
            {
                **ident,
                "cached": False,
                "http_status": 200,
                "retry_count": 0,
                "requested_at": "2026-09-16T00:00:00+00:00",
                "response_received_at": "2026-09-16T00:00:01+00:00",
                "latency_seconds": 1.0,
                "response": body,
            }
        )
    for path in mod.SOURCES:
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fictional metadata; no human data")
    for name in mod.NAMES:
        body_key = "payload" if name == mod.NAMES[0] else "response"
        (tmp_path / mod.CAL / name).write_bytes(
            b"".join(source_line(r, body_key) for r in reversed(rows[name]))
        )
    if item_count == 200:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = (
            tmp_path
            / mod.ROOT
            / "data/processed/human_benchmarks/choices13k/calibration_items.parquet"
        )
        pq.write_table(
            pa.table(
                {
                    "item_id": list(range(200)),
                    "human_b_rate": ["FORBIDDEN_LABEL"] * 200,
                }
            ),
            path,
        )
        manifest = tmp_path / mod.ROOT / "data/manifests/choices13k_manifest.json"
        manifest.write_text(
            canonical({"splits": {"calibration_sha256": fingerprint(path)["sha256"]}})
        )
    selection = {
        "selected_item_ids": list(range(item_count)),
        "projected_columns": ["item_id"],
        "opportunities": opportunities,
        "source_files": {p: fingerprint(tmp_path / p) for p in mod.SOURCES},
    }
    if item_count == 200:
        legacy = {
            **selection,
            "selected_item_ids": list(range(20)),
            "opportunities": opportunities[:200],
        }
        old_records = {
            name: [
                {"line_number": i + 1, "record": deepcopy(row)}
                for i, row in enumerate(values)
                if row["item_id"] < 20
            ]
            for name, values in rows.items()
        }
        folder = tmp_path / "iterations/previous20"
        folder.mkdir(parents=True)
        (folder / "selection.json").write_text(canonical(legacy))
        (folder / "evaluation.json").write_text(canonical(mod.audit(legacy, old_records)))
        selection.update(
            {
                "expansion_classification": "coverage_expansion_after_observing20",
                "previously_inspected_item_ids": list(range(20)),
                "additional_item_ids": list(range(20, 200)),
                **{
                    key: {
                        "path": str((folder / name).relative_to(tmp_path)),
                        "fingerprint": fingerprint(folder / name),
                    }
                    for key, name in [
                        ("prior_inspected_selection", "selection.json"),
                        ("prior_inspected_result", "evaluation.json"),
                    ]
                },
            }
        )
    return SimpleNamespace(workspace=tmp_path, selection=selection, rows=rows)


def records(sample):
    return {
        name: [{"line_number": i + 1, "record": deepcopy(row)} for i, row in enumerate(rows)]
        for name, rows in sample.rows.items()
    }


def test_200_opportunities_keep_orientation_payload_and_reuse_separate(sample):
    index = mod.make_index(sample.workspace, sample.selection)
    parsed = mod.read_index(sample.workspace, sample.selection, index)
    result = mod.audit(sample.selection, parsed)
    s = result["summary"]
    assert s["planned_opportunities"] == s["valid_primary_returns"] == 200
    assert s["full_payload_groups"] == 40
    assert s["repeated_noncached_id_created_text_groups"] == 40
    assert result["orientation_counts"] == {"A_first": 100, "B_first": 100}
    assert all(g["opportunities"] == 5 for g in result["payload_groups"])
    assert all(g["observed_noncached_strong_signatures"] == 1 for g in result["payload_groups"])
    assert (
        result["independent_generation_count"] is result["deduplicated_choice_probability"] is None
    )
    assert {row["item_id"] for row in result["rows"]} == set(range(20))
    # All items deliberately have the same internal Problem ID.
    assert len(result["rows"]) == 200 and result["new_model_POSTs"] == 0


def test_other_prompt_and_unselected_bodies_are_never_fully_decoded(sample, monkeypatch):
    path = sample.workspace / mod.CAL / mod.NAMES[0]
    for prompt, item in [("P1", 0), ("P0", 9000)]:
        req = make_request(item, 0, "A_first")
        req["prompt_id"] = prompt
        meta = {k: v for k, v in req.items() if k != "payload"}
        blob = json.dumps(meta)[:-1].encode() + b', "payload": {"POISON": invalid-json}}\n'
        with path.open("ab") as handle:
            handle.write(blob)
    sample.selection["source_files"][str(path.relative_to(sample.workspace))] = fingerprint(path)
    index = mod.make_index(sample.workspace, sample.selection)
    assert index["files"][mod.CAL + mod.NAMES[0]]["bodies_not_decoded"] == 2
    original, seen = mod.decode_selected, []

    def guarded(blob):
        assert b"POISON" not in blob
        seen.append(blob)
        return original(blob)

    monkeypatch.setattr(mod, "decode_selected", guarded)
    mod.read_index(sample.workspace, sample.selection, index)
    assert len(seen) == 600


@pytest.mark.parametrize("mutation", ["duplicate", "post_body", "nonscalar", "unknown", "missing"])
def test_boundary_rejects_ambiguous_identity_before_whole_row_decode(mutation, monkeypatch):
    req = make_request(0, 0, "A_first")
    blob = source_line(req, "payload")
    if mutation == "duplicate":
        blob = blob.replace(b'"prompt_id": "P0"', b'"prompt_id": "P0", "prompt_id": "P1"')
    elif mutation == "post_body":
        blob = blob.rstrip()[:-1] + b', "prompt_id":"P1"}'
    elif mutation == "nonscalar":
        blob = blob.replace(b'"prompt_id": "P0"', b'"prompt_id": {"POISON":123}')
    elif mutation == "unknown":
        blob = b'{"human_b_rate": 0.9,' + blob[1:]
    else:
        blob = blob.replace(b'"prompt_id": "P0", ', b"")
    monkeypatch.setattr(mod, "decode_selected", lambda x: pytest.fail("forbidden whole decode"))
    with pytest.raises(ValueError):
        mod.prefix(blob, mod.NAMES[0])


def test_resigned_index_cannot_smuggle_selected_offsets_or_change_scope(sample):
    index = mod.make_index(sample.workspace, sample.selection)
    index["files"][mod.CAL + mod.NAMES[0]]["selected_entries"][0]["offset"] += 1
    with pytest.raises(ValueError, match="index differs"):
        mod.read_index(sample.workspace, sample.selection, index)
    selection = deepcopy(sample.selection)
    selection["source_files"]["behavioral-llm-market/outputs/heldout/answers.jsonl"] = {}
    with pytest.raises(ValueError, match="allowlist"):
        mod.make_index(sample.workspace, selection)


def test_two_valid_returns_for_one_hash_never_selects_preferred_answer(sample):
    raw = records(sample)
    duplicate = deepcopy(raw[mod.NAMES[2]][0])
    duplicate["record"]["response"]["choices"][0]["message"]["content"] = '{"choice":"B"}'
    raw[mod.NAMES[2]].append(duplicate)
    result = mod.audit(sample.selection, raw)
    assert result["summary"]["valid_primary_returns"] == 199
    assert result["rows"][0]["choice"] is None and result["rows"][0]["issues"]
    assert len(result["rows"]) == 200


def test_missing_identity_metadata_keeps_strong_signature_unknown(sample):
    raw = records(sample)
    for entry in raw[mod.NAMES[2]]:
        entry["record"]["response"].pop("id")
        entry["record"]["response"].pop("created")
    result = mod.audit(sample.selection, raw)
    assert result["summary"]["valid_primary_returns"] == 200
    assert result["summary"]["unknown_primary_strong_signatures"] == 200
    assert result["repeated_noncached_id_created_text_groups"] == []
    assert len(result["text_only_repetition_groups"]) == 1


def test_local_replay_is_linked_but_never_claimed_as_new_attempt(sample):
    raw = records(sample)
    raw[mod.NAMES[1]].pop(0)
    response = raw[mod.NAMES[2]][0]["record"]
    body = response["response"]
    raw[mod.NAMES[2]][0]["record"] = {
        **{k: response[k] for k in mod.BASE},
        "cached": True,
        "status": "ok",
        "response": body,
    }
    result = mod.audit(sample.selection, raw)
    assert result["summary"]["valid_primary_returns"] == 200
    assert result["summary"]["local_replay_opportunities"] == 1
    assert result["summary"]["recorded_primary_attempts"] == 199
    assert all(
        {"item_id": 0, "repeat_id": 0} not in group
        for group in result["repeated_noncached_id_created_text_groups"]
    )


def test_historical_transport_retry_is_auxiliary_not_an_extra_opportunity(sample):
    raw = records(sample)
    initial = deepcopy(raw[mod.NAMES[2]][0])
    initial["record"].update(http_status=503, response=None)
    initial["record"]["response_received_at"] = "2026-09-16T00:00:00.100000+00:00"
    raw[mod.NAMES[2]].append(initial)
    retry = raw[mod.NAMES[2]][0]["record"]
    retry.update(retry_count=1, requested_at="2026-09-16T00:00:00.200000+00:00")
    attempt = deepcopy(raw[mod.NAMES[1]][0])
    attempt["record"].update(retry_count=1, dispatched_at=retry["requested_at"])
    raw[mod.NAMES[1]].append(attempt)
    result = mod.audit(sample.selection, raw)
    assert result["summary"]["valid_primary_returns"] == 200
    assert result["summary"]["recorded_primary_attempts"] == 201
    assert len(result["rows"]) == 200


def test_repair_after_valid_primary_is_not_silently_ignored(sample):
    raw = records(sample)
    repair = make_request(0, 0, sample.selection["opportunities"][0]["display_order"], repair=1)
    raw[mod.NAMES[0]].append({"line_number": 201, "record": repair})
    result = mod.audit(sample.selection, raw)
    assert result["summary"]["valid_primary_returns"] == 199
    assert "repair_after_valid_primary" in result["rows"][0]["issues"]
    assert result["summary"]["repair_request_groups"] == 1


def test_plan_binds_selection_before_sources_or_outcomes_are_read(sample):
    path = sample.workspace / "iterations/selection.json"
    path.parent.mkdir()
    path.write_text(canonical(sample.selection))
    plan = sample.workspace / "plan.json"
    plan.write_text(
        canonical(
            {
                "selection_file": "iterations/selection.json",
                "selection_fingerprint": fingerprint(path),
            }
        )
    )
    assert mod.load_selection(sample.workspace, plan) == sample.selection
    changed = deepcopy(sample.selection)
    changed["selected_item_ids"][-1] = 9000
    path.write_text(canonical(changed))
    with pytest.raises(ValueError, match="frozen selection differs"):
        mod.load_selection(sample.workspace, plan)


@pytest.mark.parametrize(
    "side,value",
    [
        ("response", False),
        ("response", 0.0),
        ("response", None),
        ("response", 6),
        ("attempt", "1"),
        ("attempt", {"bad": 0}),
    ],
)
def test_invalid_retry_indices_on_either_side_keep_opportunity_unknown(sample, side, value):
    raw = records(sample)
    if side == "response":
        raw[mod.NAMES[2]][0]["record"]["retry_count"] = value
    else:
        extra = deepcopy(raw[mod.NAMES[1]][0])
        extra["record"]["retry_count"] = value
        raw[mod.NAMES[1]].append(extra)
    result = mod.audit(sample.selection, raw)
    assert len(result["rows"]) == 200
    assert result["summary"]["valid_primary_returns"] == 199
    assert result["rows"][0]["choice"] is None
    assert "invalid_retry_index" in result["rows"][0]["issues"]


@pytest.mark.parametrize("failure", [None, "orphan", "clock", "hash"])
def test_repair_choice_requires_unique_valid_transport_link(sample, failure):
    raw = records(sample)
    raw[mod.NAMES[2]][0]["record"]["response"]["choices"][0]["message"]["content"] = "{}"
    order = sample.selection["opportunities"][0]["display_order"]
    request = make_request(0, 0, order, repair=1)
    if failure == "hash":
        request["request_hash"] = "invalid-hash"
    ident = {k: v for k, v in request.items() if k != "payload"}
    attempt = {**ident, "retry_count": 0, "dispatched_at": "2026-09-16T00:00:02+00:00"}
    body = deepcopy(raw[mod.NAMES[2]][0]["record"]["response"])
    body["choices"][0]["message"]["content"] = '{"choice":"B"}'
    response = {
        **ident,
        "cached": False,
        "retry_count": 0,
        "http_status": 200,
        "requested_at": "2026-09-16T00:00:02+00:00",
        "response_received_at": "2026-09-16T00:00:03+00:00",
        "latency_seconds": 1.0,
        "response": body,
    }
    if failure == "clock":
        response["requested_at"] = "2026-09-16T00:00:01+00:00"
    raw[mod.NAMES[0]].append({"line_number": 201, "record": request})
    if failure != "orphan":
        raw[mod.NAMES[1]].append({"line_number": 201, "record": attempt})
    raw[mod.NAMES[2]].append({"line_number": 201, "record": response})
    result = mod.audit(sample.selection, raw)
    assert len(result["rows"]) == 200 and result["summary"]["valid_primary_returns"] == 199
    assert result["rows"][0]["choice"] is None
    assert result["repair_requests"][0]["choice"] == ("B" if failure is None else None)


@pytest.mark.parametrize("sample", [200], indirect=True)
def test_full200_projects_only_IDs_derives_counts_and_preserves_old20(sample, monkeypatch):
    import pyarrow.parquet as pq

    original, columns = pq.read_table, []

    def projected(path, *args, **kwargs):
        assert kwargs.get("columns") == ["item_id"]
        columns.append(kwargs["columns"])
        result = original(path, *args, **kwargs)
        assert result.column_names == ["item_id"]
        return result

    monkeypatch.setattr(pq, "read_table", projected)
    index = mod.make_index(sample.workspace, sample.selection)
    result = mod.audit(sample.selection, mod.read_index(sample.workspace, sample.selection, index))
    assert columns and result["summary"]["items"] == 200
    assert result["summary"]["planned_opportunities"] == 2000
    assert result["summary"]["valid_primary_returns"] == 2000
    assert result["orientation_counts"] == {"A_first": 1000, "B_first": 1000}
    prior = sample.workspace / sample.selection["prior_inspected_result"]["path"]
    assert [r for r in result["rows"] if r["item_id"] < 20] == json.loads(prior.read_text())["rows"]
    assert "已看20题＋新增180题" in mod.describe(result)


@pytest.mark.parametrize("sample", [200], indirect=True)
def test_full200_payload_group_count_is_measured_not_assumed400(sample):
    raw = records(sample)
    replacement = {
        entry["record"]["display_order"]: entry["record"]["payload"]
        for entry in raw[mod.NAMES[0]]
        if entry["record"]["item_id"] == 0
    }
    hashes = {}
    for entry in raw[mod.NAMES[0]]:
        row = entry["record"]
        if row["item_id"] == 1:
            old = row["request_hash"]
            row["payload"] = deepcopy(replacement[row["display_order"]])
            ident = {k: row[k] for k in mod.BASE - {"request_hash"}}
            row["request_hash"] = hashlib.sha256(
                json.dumps([row["payload"], ident, "choices13k-v1"], sort_keys=True).encode()
            ).hexdigest()
            hashes[old] = row["request_hash"]
    for name in mod.NAMES[1:]:
        for entry in raw[name]:
            row = entry["record"]
            row["request_hash"] = hashes.get(row["request_hash"], row["request_hash"])
    result = mod.audit(sample.selection, raw)
    assert result["summary"]["full_payload_groups"] == 398
    assert result["summary"]["valid_primary_returns"] == 2000
    merged = [g for g in result["payload_groups"] if g["items"] == [0, 1]]
    assert len(merged) == 2 and all(g["opportunities"] == 10 for g in merged)


@pytest.mark.parametrize("sample", [200], indirect=True)
@pytest.mark.parametrize("mutation", ["wrong_IDs", "concealed_expansion", "seen_IDs", "new_IDs"])
def test_full200_cannot_pass_on_length_alone_or_hide_discovery(sample, mutation):
    selection = deepcopy(sample.selection)
    if mutation == "wrong_IDs":
        selection["selected_item_ids"][-1] = 9000
    elif mutation == "concealed_expansion":
        selection["expansion_classification"] = "all200_outcome_blind"
    elif mutation == "seen_IDs":
        selection["previously_inspected_item_ids"] = list(range(20, 40))
    else:
        selection["additional_item_ids"] = list(range(21, 200))
    with pytest.raises(ValueError):
        mod.make_index(sample.workspace, selection)


@pytest.mark.parametrize("sample", [200], indirect=True)
def test_full200_missing_return_keeps2000_denominator_and_unknown(sample):
    raw = records(sample)
    raw[mod.NAMES[2]].pop()
    result = mod.audit(sample.selection, raw)
    assert result["summary"]["planned_opportunities"] == len(result["rows"]) == 2000
    assert result["summary"]["valid_primary_returns"] == 1999
    assert result["summary"]["unresolved_primary_opportunities"] == 1
    assert result["rows"][-1]["choice"] is None
