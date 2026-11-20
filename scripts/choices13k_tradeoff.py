"""Prepare a frozen24-request comparison offline; live use needs a new explicit grant."""

import argparse
import asyncio
import math
from datetime import UTC, datetime
from pathlib import Path

from behavioral_market.benchmarks import tradeoff_comparison as study
from behavioral_market.evaluation import matched_choice_collection as collection
from behavioral_market.evaluation.style_mechanism_design import canonical
from behavioral_market.evaluation.style_probe_intake import strict_json
from scripts.style_bank_square import checked_path, verify_budget_sources
from scripts.style_fact_check import fingerprint, save

ITERATION = "iterations/choices13k_tradeoff_20261022/"
PREVIOUS = "iterations/choices13k_reference_comparison_20261022/"
ORIGIN = {
    "path": PREVIOUS + "next_iteration_plan.json",
    "fingerprint": {
        "bytes": 8336,
        "sha256": "b7a067987cf2db875de1151b71d84e3b798b93e704b897bd62bf4aad13a5af52",
    },
}
SOURCES = {
    "source_materials.json": {
        "path": PREVIOUS + "package/source_materials.json",
        "fingerprint": {
            "bytes": 1095,
            "sha256": "5e589e67676845464a6f628a461f6822de61ab0405087d07715f827583a95588",
        },
    },
    "local_target.json": {
        "path": PREVIOUS + "package/local_target.json",
        "fingerprint": {
            "bytes": 46,
            "sha256": "924ad871befffb5e757bca45af05bbae6e92b8a6755243cb867e3a42f75ef08a",
        },
    },
}
DERIVED = {
    "source_materials.json",
    "local_target.json",
    "diagnostic.json",
    "packet.json",
    "expected_requests.jsonl",
    "pending_evaluation.json",
    "rate_role_change.json",
}
RUN_ID = "choices13k-tradeoff-20261022"


def schedule():
    rows = []
    for (content, letters), order in zip(study.panel_tool.FORMATS, study.ORDERS, strict=True):
        for repeat in (0, 1):
            for method in order if repeat == 0 else order[::-1]:
                rows.append(
                    {
                        "slot_id": f"slot_{len(rows) + 1:02d}",
                        "item_id": 166,
                        "method_id": method,
                        "content_order": content,
                        "letter_order": letters,
                        "repeat_id": repeat,
                    }
                )
    return rows


def checked_plan(workspace):
    plan = strict_json((workspace / ITERATION / "plan.json").read_text())
    study.require(canonical(plan["origin_plan"]) == canonical(ORIGIN), "fixed origin binding")
    origin = strict_json(checked_path(workspace, ORIGIN).read_text())
    study.require(
        canonical(plan["data_sources"]) == canonical(SOURCES) == canonical(origin["data_sources"]),
        "fixed two source bindings",
    )
    study.require(
        canonical(plan["metadata_sources"]) == canonical(origin["metadata_sources"]),
        "fixed metadata sources",
    )
    study.require(
        set(plan["metadata_sources"])
        == {"pre_execution_freeze.json", "independent_review.json", "budget_ledger.json"},
        "metadata names",
    )
    metadata = {
        name: strict_json(checked_path(workspace, desc).read_text())
        for name, desc in plan["metadata_sources"].items()
    }
    # Inspect only these two entries in the manifest; never follow its other source paths.
    for desc in SOURCES.values():
        study.require(
            metadata["pre_execution_freeze.json"]["files"][desc["path"]] == desc["fingerprint"],
            "accepted frozen source binding",
        )
    study.require(
        metadata["independent_review.json"]["status"] == "accepted", "prior stage not accepted"
    )
    study.require(
        plan["target_item_id"] == 166
        and plan["reference_item_ids"] == [211, 384, 542]
        and plan["selection_input_fields"] == ["item_id", "option_a_text", "option_b_text"]
        and canonical(plan["reference_pairing"]) == canonical(origin["reference_pairing"])
        and canonical(plan["comparisons"]) == canonical(origin["comparisons_fixed_in_advance"])
        and canonical(plan["schedule"]) == canonical(schedule())
        and plan["frozen_run_id"] == RUN_ID
        and plan["fixed_execution_output"]
        == str((workspace / ITERATION / "live_execution").resolve())
        and plan["budget"]
        == {
            "used_POSTs": 91,
            "authorized_cap": 100,
            "reserved_POSTs": 0,
            "planned_future_batch": 24,
            "required_future_cumulative_cap": 115,
            "new_POSTs_authorized_now": 0,
        },
        "fixed offline protocol",
    )
    study.require(
        plan["global_grant"]["path"]
        == "iterations/style_fact_execution_20261022/user_budget_authorization.json",
        "fixed grant path",
    )
    verify_budget_sources(
        workspace,
        {
            "budget_sources": {
                "grant": plan["global_grant"],
                "prior_ledger": plan["metadata_sources"]["budget_ledger.json"],
            },
            "endpoint": study.ENDPOINT,
            "model": study.MODEL,
            "prior_POSTs": 91,
            "batch_cumulative_stop": 100,
        },
    )
    return plan


def read_sources(workspace, plan):
    """Read only two accepted derived files; no original logs or dataset loaders."""
    study.require(
        canonical(plan["data_sources"]) == canonical(SOURCES), "fixed two source bindings"
    )
    old = strict_json(checked_path(workspace, SOURCES["source_materials.json"]).read_text())
    local = strict_json(checked_path(workspace, SOURCES["local_target.json"]).read_text())
    study.require(
        isinstance(old, dict) and set(old) == {"target", "references"}, "old materials schema"
    )
    study.text_row(old["target"], 384)
    study.require(
        isinstance(old["references"], list) and len(old["references"]) == 3, "old references"
    )
    for row, item in zip(old["references"], (166, 211, 542), strict=True):
        study.text_row(row, item, ("human_b_rate",))
    study.require(
        set(local) == {"item_id", "human_b_rate"}
        and type(local["item_id"]) is int
        and local["item_id"] == 384,
        "old local target",
    )
    rows = [old["target"], *old["references"]]
    text = [{k: row[k] for k in study.TEXT_FIELDS} for row in rows]
    diagnostic = study.select_tradeoff(text)
    study.require(diagnostic["selected_item_id"] == 166, "fixed selected target differs")
    by_id = {r["item_id"]: r for r in text}
    rates = {r["item_id"]: r["human_b_rate"] for r in old["references"]}
    rates[384] = local["human_b_rate"]
    study.require(
        all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in rates.values()),
        "finite source rates",
    )
    materials = {
        "target": by_id[166],
        "references": [{**by_id[i], "human_b_rate": rates[i]} for i in study.REFERENCES],
    }
    role_change = {
        "old_target_item_id": 384,
        "old_local_rate_becomes_reference": True,
        "new_target_item_id": 166,
        "new_target_rate_local_only": True,
        "reference_item_ids": list(study.REFERENCES),
        "exploratory_after_observing_prior_results": True,
        "new_test_set": False,
    }
    return materials, {166: rates[166]}, diagnostic, role_change


def prepare(workspace):
    plan = checked_plan(workspace)
    output = workspace / ITERATION / "package"
    study.require(not output.exists(), "never overwrite prepared package")
    materials, targets, diagnostic, role = read_sources(workspace, plan)
    packet = study.make_packet(materials, RUN_ID, "recorded_model_logs")
    pending = study.evaluate(packet, [], [], [], targets)
    checked_plan(workspace)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in {
        "source_materials.json": materials,
        "local_target.json": {"item_id": 166, "human_b_rate": targets[166]},
        "diagnostic.json": diagnostic,
        "rate_role_change.json": role,
        "packet.json": packet,
        "pending_evaluation.json": pending,
    }.items():
        save(output / name, value)
    (output / "expected_requests.jsonl").write_text(
        "".join(canonical(r) + "\n" for r in packet["requests"])
    )
    save(
        output / "source_read_receipt.json",
        {
            "recorded_utc": datetime.now(UTC).isoformat(),
            "data_sources": SOURCES,
            "plan_fingerprint": fingerprint(workspace / ITERATION / "plan.json"),
            "derived_files": {name: fingerprint(output / name) for name in sorted(DERIVED)},
            "data_files_decoded": 2,
            "existing_item_ids": [166, 211, 384, 542],
            "old_raw_or_parquet_reads": 0,
            "new_human_data_sources": 0,
            "credentials_read": False,
            "HTTP_requests": 0,
            "new_model_POSTs": 0,
        },
    )
    return {
        "prepared_not_sent": True,
        "opportunities": 24,
        "used_POSTs": 91,
        "current_cap": 100,
        "required_future_cap": 115,
    }


def load_packet(workspace):
    plan = checked_plan(workspace)
    package = workspace / ITERATION / "package"
    receipt = strict_json((package / "source_read_receipt.json").read_text())
    study.require(canonical(receipt["data_sources"]) == canonical(SOURCES), "receipt sources")
    study.require(
        receipt["plan_fingerprint"] == fingerprint(workspace / ITERATION / "plan.json"),
        "receipt plan",
    )
    study.require(set(receipt["derived_files"]) == DERIVED, "fixed derived filenames")
    for name, expected in receipt["derived_files"].items():
        study.require(fingerprint(package / name) == expected, "derived file differs")
    materials, targets, diagnostic, role = read_sources(workspace, plan)
    rebuilt = study.make_packet(materials, RUN_ID, "recorded_model_logs")
    expected = {
        "source_materials.json": materials,
        "local_target.json": {"item_id": 166, "human_b_rate": targets[166]},
        "diagnostic.json": diagnostic,
        "rate_role_change.json": role,
        "packet.json": rebuilt,
        "pending_evaluation.json": study.evaluate(rebuilt, [], [], [], targets),
    }
    for name, value in expected.items():
        study.require(
            canonical(strict_json((package / name).read_text())) == canonical(value),
            "derived content differs from accepted sources: " + name,
        )
    study.require(
        canonical(collection.read_jsonl(package / "expected_requests.jsonl"))
        == canonical(rebuilt["requests"]),
        "expected wire differs",
    )
    return rebuilt, targets


def checked_authorization(workspace, packet):
    path = workspace / ITERATION / "execution_authorization.json"
    study.require(path.exists(), "new explicit grant required: current100cap cannot fit91+24")
    grant = strict_json(path.read_text())
    fields = {
        "approved",
        "specific_user_reply",
        "question_item_id",
        "endpoint",
        "model",
        "prior_POSTs",
        "global_cumulative_completion_POST_cap",
        "maximum_new_completion_POSTs",
        "packet_fingerprint",
        "prior_ledger_fingerprint",
    }
    study.require(
        isinstance(grant, dict) and set(grant) == fields and grant["approved"] is True,
        "explicit grant schema",
    )
    study.require(
        all(
            isinstance(grant[k], str) and grant[k].strip()
            for k in ("specific_user_reply", "question_item_id")
        ),
        "actual user reply absent",
    )
    study.require(
        grant["endpoint"] == study.ENDPOINT
        and grant["model"] == study.MODEL
        and type(grant["prior_POSTs"]) is int
        and grant["prior_POSTs"] == 91
        and type(grant["maximum_new_completion_POSTs"]) is int
        and grant["maximum_new_completion_POSTs"] == 24
        and type(grant["global_cumulative_completion_POST_cap"]) is int
        and grant["global_cumulative_completion_POST_cap"] >= 115,
        "complete24 authorization required",
    )
    plan = checked_plan(workspace)
    study.require(
        grant["packet_fingerprint"] == fingerprint(workspace / ITERATION / "package/packet.json"),
        "grant packet differs",
    )
    study.require(
        grant["prior_ledger_fingerprint"]
        == plan["metadata_sources"]["budget_ledger.json"]["fingerprint"],
        "grant prior ledger differs",
    )
    study.require(packet["evidence_kind"] == "recorded_model_logs", "synthetic cannot execute live")
    return {
        "prior_POSTs": 91,
        "global_cap": grant["global_cumulative_completion_POST_cap"],
        "grant_source": str(path.resolve()),
    }


def describe(result):
    lines = [
        "# 三组单题对照",
        "",
        f"实际发送{result['logged_dispatch_attempts']}次，累计{result['used_completion_POSTs']}次。",
        "",
    ]
    for name, value in result["comparisons"].items():
        lines.append(f"{name}: {canonical(value['primary'])}")
    lines.extend(["", "单题探索；不足完整回答时保留未知，两项比较都报告，不按结果挑选。", ""])
    return "\n".join(lines)


async def execute(workspace):
    # This check happens before loading credentials or constructing an HTTP client.
    grant_path = workspace / ITERATION / "execution_authorization.json"
    study.require(grant_path.exists(), "new explicit grant required: current100cap cannot fit91+24")
    freeze = strict_json((workspace / ITERATION / "prepared_freeze.json").read_text())
    for relative, expected in freeze["files"].items():
        study.require(fingerprint(workspace / relative) == expected, "prepared frozen file differs")
    review = strict_json((workspace / ITERATION / "pre_execution_review.json").read_text())
    study.require(review["status"] == "accepted", "independent packet review required")
    packet, targets = load_packet(workspace)
    study.require(
        review["packet_fingerprint"] == fingerprint(workspace / ITERATION / "package/packet.json"),
        "reviewed packet differs",
    )
    authorization = checked_authorization(workspace, packet)
    output = workspace / ITERATION / "live_execution"
    study.require(not output.exists(), "never retry or resume")
    import httpx
    from dotenv import dotenv_values

    secret = dotenv_values(workspace / "behavioral-llm-market/.env").get("AIGC_API_KEY")
    study.require(bool(secret), "configured school credential missing")
    async with httpx.AsyncClient(
        headers={"Authorization": "Bearer " + secret},
        timeout=120,
        trust_env=False,
        transport=httpx.AsyncHTTPTransport(retries=0),
    ) as client:
        result = await collection.collect(
            packet, output, client, targets, study.evaluate, authorization, secret=secret
        )
    (output / "REPORT.md").write_text(describe(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "execute", "evaluate"])
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if args.action == "prepare":
        result = prepare(workspace)
    elif args.action == "execute":
        result = asyncio.run(execute(workspace))
        result = {
            "protocol_complete": result["protocol_complete"],
            "used_POSTs": result["used_completion_POSTs"],
        }
    else:
        packet, targets = load_packet(workspace)
        output = workspace / ITERATION / "live_execution"
        result = study.evaluate(
            packet, *(collection.read_jsonl(output / n) for n in collection.LOGS), targets
        )
        save(output / "evaluation.json", result)
        (output / "REPORT.md").write_text(describe(result))
        result = {
            "protocol_complete": result["protocol_complete"],
            "used_POSTs": result["used_completion_POSTs"],
        }
    print(canonical(result))


if __name__ == "__main__":
    main()
