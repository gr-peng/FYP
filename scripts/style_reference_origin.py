"""Prepare, run and describe the fixed origin-label/reference four-cell choice task."""

import argparse
import asyncio
from pathlib import Path

from behavioral_market.evaluation.style_mechanism_design import canonical, digest
from behavioral_market.evaluation.style_probe_intake import strict_json
from behavioral_market.evaluation.style_reference_origin import (
    BANK_PATH,
    NAMESPACE,
    SELECTED,
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
from scripts.style_fact_check import fingerprint, read_jsonl, save

LOGS = ("llm_requests.jsonl", "llm_attempts.jsonl", "llm_responses.jsonl")
HISTORY_ROOT = "iterations/style_conflict_package_20261022/live_execution"
HISTORY_FINGERPRINTS = {
    "llm_requests.jsonl": {
        "bytes": 21362,
        "sha256": "fef6522e94129d7df1200a8c0163f759572e4046e9d8033468ee30d7042edd35",
    },
    "llm_attempts.jsonl": {
        "bytes": 1512,
        "sha256": "b26a0e34c4e1cb3a7bb25901218ff56b8fd2e7c4777fff15a7d7047b150f284d",
    },
    "llm_responses.jsonl": {
        "bytes": 6046,
        "sha256": "bc630cb6e1eb06321292b83a96887c017b9fbc53dbd9cdb4b5c2a418ed5a5144",
    },
}


def load_history(workspace):
    """Re-read the fixed original bytes; no caller-supplied old answer is trusted."""
    root = workspace / HISTORY_ROOT
    require(
        all(fingerprint(root / n) == fp for n, fp in HISTORY_FINGERPRINTS.items()),
        "historical source bytes differ",
    )
    requests, attempts, responses = (read_jsonl(root / n) for n in LOGS)
    anchor = [
        [r for r in records if r["case_id"] == SELECTED[0]]
        for records in (requests, attempts, responses)
    ]
    require(all(len(rows) == 1 for rows in anchor), "historical anchor missing/duplicate")
    return {"body": anchor[2][0]["response"], "payload_sha256": digest(anchor[0][0]["payload"])}


def describe(result):
    names = {"fundamental": "基本面", "technical": "技术"}
    factors = {"reference_condition": "参考价105减95", "current_style": "原标签基本面减技术"}
    text = ["# 原策略与参考价：四种组合", ""]
    if result["evidence_kind"] == "synthetic_tool_test":
        text += ["**本地工具测试，不是模型回答。**", ""]
    text += [
        f"记录到{result['logged_dispatch_attempts']}次实际发送，可评分{result['valid_returns']}条。",
        "",
        "| 场景 | 原策略 | 参考价 | 最终策略 | 保持/切换 |",
        "|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        f, c = row["factors"], row["choice"]
        ref = 95 if f["reference_condition"] == "neutral" else 105
        text.append(
            f"| {row['case_id']} | {names[f['current_style']]} | {ref} | "
            f"{names[c['target_style']] if c else '未知（' + row['status'] + '）'} | "
            f"{({'stay': '保持', 'switch': '切换'}[c['decision']]) if c else '未知'} |"
        )
    text += ["", "| 单因素比较 | 两端 | 选择基本面的差异 |", "|---|---|---|"]
    for edge in result["contrasts"]:
        d = edge["target_fundamental_difference"]
        text.append(
            f"| {factors[edge['factor']]} | {edge['positive_case']} / {edge['negative_case']} | "
            f"{d if d is not None else '未知'} |"
        )
    text += [
        "",
        "相同目标策略全部保留；缺失、无效与协议异常保留未知，不补成保持或零差异。",
        "当前批次返回重复组：",
        canonical(result["return_identity_diagnostics"]),
        "",
        "与旧锚点的重复线索（旧回答不计入本轮四条）：",
        canonical(result.get("historical_anchor_diagnostics")),
        "",
        "重复线索照实保留，不加随机文本、不补发；没有重复也不证明独立生成。",
        "主要比较最终选哪种策略；保持/切换会随原策略机械变化，不能用来冒充心理机制。",
        "原标签与十期历史标签一起改变；95/105只是指定参考数。每格一个返回，四对共享四条记录。",
        "只描述指定条件下的选择，不能证明习惯、真实价值、人类相似度、一般规律或市场反馈。",
        "",
    ]
    return "\n".join(text)


def scored(package, workspace, logs):
    result = evaluate(package, *(read_jsonl(logs / n) for n in LOGS))
    old = load_history(workspace)
    result["historical_anchor_diagnostics"] = historical_diagnostics(
        result, old["body"], old["payload_sha256"], package
    )
    return result


def prepare(args):
    plan = strict_json(args.plan.read_text())
    require(
        all(fingerprint(args.workspace / p) == v for p, v in plan["source_files"].items()),
        "declared source fingerprints differ",
    )
    bank = read_jsonl(args.workspace / BANK_PATH)
    edges_path = next(p for p in plan["source_files"] if p.endswith("/contrasts.json"))
    package = make_package(
        bank,
        strict_json((args.workspace / edges_path).read_text()),
        plan["frozen_run_id"],
        plan["fixed_execution_output"],
    )
    validate_package(package)
    old = load_history(args.workspace)
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
    save(args.output / "pending_evaluation.json", pending)
    save(
        args.output / "input_receipt.json",
        {
            "sources": plan["source_files"],
            "historical_sources": {
                HISTORY_ROOT + "/" + n: fp for n, fp in HISTORY_FINGERPRINTS.items()
            },
            "new_prompt_changes": 0,
            "cases": 4,
            "contrasts": 4,
            "new_model_calls": 0,
        },
    )
    (args.output / "REPORT.md").write_text(describe(pending))
    return {"status": "prepared_not_sent", "cases": 4}


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
    load_history(args.workspace)
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
    p.add_argument("--plan", type=Path)
    p.add_argument("--package", type=Path)
    p.add_argument("--authorization", type=Path)
    p.add_argument("--logs", type=Path)
    p.add_argument("--output", type=Path)
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
