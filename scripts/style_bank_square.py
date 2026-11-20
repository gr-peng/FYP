"""Prepare/execute/evaluate a frozen four-cell square without changing its prompts."""

import argparse
import asyncio
from pathlib import Path

from behavioral_market.evaluation.style_bank_square import (
    NAMESPACE,
    authorization_template,
    check_authorization,
    collect,
    evaluate,
    execution_config,
    expected_request,
    historical_diagnostics,
    make_package,
    require,
    validate_package,
)
from behavioral_market.evaluation.style_mechanism_design import canonical, digest
from behavioral_market.evaluation.style_probe_intake import strict_json
from behavioral_market.evaluation.style_reference_origin import BANK_PATH
from scripts.style_fact_check import fingerprint, read_jsonl, save

LOGS = ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")


def checked_path(workspace, descriptor):
    root = workspace.resolve()
    path = (root / descriptor["path"]).resolve()
    require(path.is_relative_to(root), "source resolves outside workspace")
    require(fingerprint(path) == descriptor["fingerprint"], "source bytes differ")
    return path


def verify_budget_sources(workspace, package):
    sources = package["budget_sources"]
    grant_path = checked_path(workspace, sources["grant"])
    grant = strict_json(grant_path.read_text())
    ledger = strict_json(checked_path(workspace, sources["prior_ledger"]).read_text())
    require(
        type(grant.get("global_cumulative_completion_POST_cap")) is int
        and grant["global_cumulative_completion_POST_cap"] == 100,
        "global grant differs",
    )
    require(
        grant.get("endpoint") == package["endpoint"] and grant.get("model") == package["model"],
        "global grant route differs",
    )
    require(
        isinstance(grant.get("specific_user_reply"), str) and grant["specific_user_reply"].strip(),
        "actual global reply absent",
    )
    require(
        type(ledger.get("used_completion_POSTs")) is int
        and ledger["used_completion_POSTs"] == package["prior_POSTs"]
        and type(ledger.get("global_authorized_completion_POST_cap")) is int
        and ledger["global_authorized_completion_POST_cap"] == 100,
        "prior used count/cap differs",
    )
    require(
        type(grant.get("prior_used_completion_POSTs")) is int
        and 0 <= grant["prior_used_completion_POSTs"] <= package["prior_POSTs"],
        "used count predates global grant",
    )
    require(
        type(ledger.get("reserved_next_completion_POSTs")) is int
        and ledger["reserved_next_completion_POSTs"] == 0
        and ledger.get("active_run_id") is None
        and isinstance(ledger.get("status"), str)
        and ledger["status"].endswith("accepted"),
        "prior batch unfinished or budget reserved",
    )
    require(
        Path(ledger["global_authorization_source"]).resolve() == grant_path,
        "ledger refers to different global grant",
    )
    require(package["batch_cumulative_stop"] <= 100, "global cap exceeded")
    return grant


def load_history(workspace, package):
    anchor = package["historical_anchor"]
    paths = {
        Path(p).name: checked_path(workspace, {"path": p, "fingerprint": fp})
        for p, fp in anchor["sources"].items()
    }
    rows = [[r for r in read_jsonl(paths[n]) if r["case_id"] == anchor["case_id"]] for n in LOGS]
    require(all(len(part) == 1 for part in rows), "historical anchor absent/duplicate")
    require(len({part[0]["request_hash"] for part in rows}) == 1, "historical request hash differs")
    require(
        rows[2][0]["http_status"] == 200 and rows[2][0]["cached"] is False,
        "historical anchor not a live successful return",
    )
    return {"body": rows[2][0]["response"], "payload_sha256": digest(rows[0][0]["payload"])}


def allocate_authorization(package, grant):
    return {
        **authorization_template(package),
        "decision": "explicit_user_granted",
        "additional_POST_limit": 4,
        "cumulative_POST_cap": package["batch_cumulative_stop"],
        "budget_sources": package["budget_sources"],
        "specific_user_reply": grant["specific_user_reply"],
        "scope_note": (
            "Fixed four-cell allocation under the existing100 user cap; no new confirmation."
        ),
    }


def describe(result):
    styles = {"fundamental": "基本面", "technical": "技术"}
    factors = {
        "current_style": "原标签基本面减技术",
        "reference_condition": "参考价105减95",
        "benchmark_favored_style": "基本面基准较好减技术较好",
        "majority_style": "基本面多数减技术多数",
    }
    flags = {
        "provider_id": "编号相同",
        "body_sha256": "完整返回相同",
        "text_sha256": "正文相同",
        "same_id_created_text": "编号、时间与正文均相同",
    }
    text = ["# 固定四种组合的实际状态", ""]
    if result["evidence_kind"] == "synthetic_tool_test":
        text += ["**本地工具测试，不能算模型回答。**", ""]
    text += [
        f"记录到{result['logged_dispatch_attempts']}次发送尝试，可评分{result['valid_returns']}条。",
        "",
        "| 场景 | 原策略 | 参考价 | 较好基准 | 群体多数 | 最终策略 |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        f, c = row["factors"], row["choice"]
        ref = 95 if f["reference_condition"] == "neutral" else 105
        text.append(
            f"| {row['case_id']} | {styles[f['current_style']]} | {ref} | "
            f"{styles[f['benchmark_favored_style']]} | {styles[f['majority_style']]} | "
            f"{styles[c['target_style']] if c else '未知'} |"
        )
    text += ["", "| 单因素比较 | 两端 | 选择基本面的差异 |", "|---|---|---|"]
    for edge in result["contrasts"]:
        d = edge["target_fundamental_difference"]
        text.append(
            f"| {factors[edge['factor']]} | {edge['positive_case']} / {edge['negative_case']} | "
            f"{d if d is not None else '未知'} |"
        )
    text += ["", "当前批次的重复组："]
    found = False
    for key, groups in result["return_identity_diagnostics"].items():
        for group in groups:
            found = True
            text.append(f"- {flags[key]}：{', '.join(group)}")
    if not found:
        text.append("没有记录到；不能据此证明独立生成。")
    history = result.get("historical_anchor_diagnostics")
    text += ["", "与旧锚点的重复线索："]
    if history:
        text.append(f"旧锚点是{history['old_anchor_case_id']}，仅供诊断，不计入本轮四条。")
        for cid, matches in history["matches_by_new_case"].items():
            text.append(f"- {cid}：{'、'.join(flags[k] for k in matches) or '未记录到'}")
    text += [
        "",
        "相同目标策略也全部保留；缺失或无效结果保留未知，不补成保持或零差异。",
        "四组对比共享四条返回，不能作为独立实验或一般规律；不证明跟风、后悔、真人相似度或市场反馈。",
        "原问题和行情不改；不重试、不修复，约定异常即停，重复也不补发。",
        "",
    ]
    return "\n".join(text)


def scored(package, workspace, logs):
    result = evaluate(package, *(read_jsonl(logs / n) for n in LOGS))
    old = load_history(workspace, package)
    result["historical_anchor_diagnostics"] = historical_diagnostics(
        result, old["body"], old["payload_sha256"], package
    )
    return result


def prepare(args):
    plan = strict_json(args.plan.read_text())
    require(
        all(fingerprint(args.workspace / p) == fp for p, fp in plan["source_files"].items()),
        "declared bank/contrast sources differ",
    )
    package = make_package(read_jsonl(args.workspace / BANK_PATH), plan)
    grant = verify_budget_sources(args.workspace, package)
    old = load_history(args.workspace, package)
    pending = evaluate(package, [], [], [])
    pending["historical_anchor_diagnostics"] = historical_diagnostics(
        pending, old["body"], old["payload_sha256"], package
    )
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "package.json", package)
    (args.output / "scheduled_inputs.jsonl").write_text(
        "".join(canonical(c) + "\n" for c in package["cases"])
    )
    (args.output / "expected_requests.jsonl").write_text(
        "".join(canonical(expected_request(c, package)) + "\n" for c in package["cases"])
    )
    save(args.output / "authorization_template.json", authorization_template(package))
    save(args.output / "allocated_authorization.json", allocate_authorization(package, grant))
    save(args.output / "pending_evaluation.json", pending)
    save(
        args.output / "input_receipt.json",
        {
            "bank_sources": plan["source_files"],
            "budget_sources": package["budget_sources"],
            "history_sources": package["historical_anchor"],
            "prompt_changes": 0,
            "new_model_calls": 0,
        },
    )
    (args.output / "REPORT.md").write_text(describe(pending))
    return {"status": "prepared_not_sent", "cases": 4, "prior_used_POSTs": package["prior_POSTs"]}


def load_package(folder):
    package = strict_json((folder / "package.json").read_text())
    validate_package(package)
    require(
        canonical(read_jsonl(folder / "scheduled_inputs.jsonl")) == canonical(package["cases"]),
        "scheduled cases differ",
    )
    require(
        canonical(read_jsonl(folder / "expected_requests.jsonl"))
        == canonical([expected_request(c, package) for c in package["cases"]]),
        "expected requests differ",
    )
    return package


async def execute(args):
    package = load_package(args.package)
    auth = strict_json(args.authorization.read_text())
    check_authorization(package, auth)
    require(package["evidence_kind"] == "recorded_model_logs", "tool test cannot execute live")
    grant = verify_budget_sources(args.workspace, package)
    require(auth["specific_user_reply"] == grant["specific_user_reply"], "user reply differs")
    load_history(args.workspace, package)
    output = Path(package["execution_output"])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "package.json", package)
    save(output / "authorization_receipt.json", auth)
    from dotenv import dotenv_values

    from behavioral_market.llm.client import OpenAICompatibleClient

    key = dotenv_values(args.workspace / "behavioral-llm-market/.env").get("AIGC_API_KEY")
    if not key:
        save(
            output / "execution_state.json",
            {"status": "stopped", "new_POSTs": 0, "reason": "AIGC_API_KEY missing; no resume"},
        )
        return {"status": "stopped", "new_POSTs": 0}
    client = OpenAICompatibleClient(
        execution_config(package), key, output, output / "fresh_cache", cache_namespace=NAMESPACE
    )
    try:
        await collect(package, auth, client, output)
    finally:
        await client.close()
    result = scored(package, args.workspace, output)
    save(output / "evaluation.json", result)
    (output / "REPORT.md").write_text(describe(result))
    return {
        "complete": result["complete"],
        "dispatches": result["logged_dispatch_attempts"],
        "targets": [r["choice"]["target_style"] if r["choice"] else None for r in result["rows"]],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "execute", "evaluate"])
    p.add_argument("--workspace", type=Path, required=True)
    for key in ["plan", "package", "authorization", "logs", "output"]:
        p.add_argument("--" + key, type=Path)
    args = p.parse_args()
    if args.action == "prepare":
        result = prepare(args)
    elif args.action == "execute":
        result = asyncio.run(execute(args))
    else:
        package = load_package(args.package)
        result = scored(package, args.workspace, args.logs)
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / "evaluation.json", result)
        (args.output / "REPORT.md").write_text(describe(result))
        result = {"complete": result["complete"]}
    print(canonical(result))


if __name__ == "__main__":
    main()
