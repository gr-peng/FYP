"""Build and audit hypothetical materials from the four declared, frozen sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

from behavioral_market.evaluation.style_mechanism_design import (
    ENVELOPE,
    FACTORS,
    build_bank,
    canonical,
    changed_paths,
    contrast_edges,
    design_specification,
    flatten,
    semantic_payload,
    validate_case,
)
from behavioral_market.evaluation.style_probe_intake import strict_json
from behavioral_market.evaluation.style_probe_materials import payload_digest

SOURCE_NAMES = (
    "iterations/review_driven_20261022/probe_materials/inputs.jsonl",
    "iterations/review_driven_20261022/probe_materials/source_receipt.json",
    "iterations/style_probe_mode_v2_20261022/materials/candidate_inputs.jsonl",
    "iterations/style_probe_mode_v2_20261022/materials/variant_receipt.json",
)


def fingerprint(path: Path) -> dict:
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def field_handling(path: str) -> tuple[str, str]:
    if path == "simulation_seed":
        return "来源标识", "移除旧运行种子；新场景另行标为完整假设材料"
    if path.startswith("agent_state.persona"):
        return "账户身份", "换成所有格共用的假设账户；原两人不作策略惯性对照"
    if path == "agent_state.current_style" or re.fullmatch(r"agent_state.memory.\d+.style", path):
        return "策略标签", "当前及十条历史策略标签联动；只检查声明标签影响，不能证明习惯"
    if path.startswith("agent_state.portfolio") or path == "agent_state.drawdown":
        return "账户账本", "全部换成同一个可复算的假设持仓和回撤，不沿用旧账户值"
    if path.startswith("agent_state.memory"):
        return "交易记忆", "换成同一十期持有历史；损益、权益和收益按新路径计算；空事件移除"
    if path.startswith("agent_state.counterfactual"):
        return (
            "原相对收益代理",
            "移除个人实际/另一策略代理比较，改为具名假设基准收益；不等价于个人反事实",
        )
    if path.startswith("observation.mean_field"):
        return "群体信息", "仅保留明确滞后一格的策略人数及占比；情绪、委托方向和群体收益不继承"
    if path in {"observation.fundamental_value", "observation.fundamental_method"}:
        return "参考估值", "参考价95/105，统一改称外部指定假设输入；不沿用旧价格锚点方法"
    if path in {
        "observation.last_close",
        "observation.return_1d",
        "observation.return_5d",
        "observation.momentum_5",
        "observation.ma5",
        "observation.ma20",
        "observation.volatility_20",
        "observation.volume_ratio_5_20",
        "observation.macd",
        "observation.macd_signal",
        "observation.macd_hist",
    }:
        return "行情与指标", "只保留新路径、收盘、一日/五日收益和五日均价并重新计算；其余指标移除"
    if path.startswith("observation."):
        return "时间或其他市场背景", "改用假设资产及期数；旧财报、背景收益、事件和时钟均不继承"
    if path == "decision_instruction":
        return "任务说明", "统一只要求策略选择；明确参考价、基准收益、策略标签和群体时点的假设性质"
    if path.startswith("output_schema"):
        return "回答格式", "保留决定/目标策略/理由三项；格式正确不代表机制或真人相似度成立"
    raise ValueError(f"unclassified source field: {path}")


def inspect_sources(
    base: list[dict], receipt: dict, candidates: list[dict], variant: dict
) -> tuple[dict, dict]:
    require(len(base) == 4 and len(candidates) == 4, "retain all four source inputs")
    bmap, cmap = ({r["case_id"]: r for r in rows} for rows in (base, candidates))
    require(len(bmap) == 4 and bmap.keys() == cmap.keys(), "duplicate or missing source case")
    payloads = receipt["source_payloads"]
    require(set(payloads) == {"A000", "A001"}, "unexpected source accounts")
    pairs, inventory, facts, documents = {}, {}, {}, {}
    for row in base:
        agent = row["source"]["agent_id"]
        original = deepcopy(payloads[agent])
        d = strict_json(original["messages"][1]["content"])
        current = d["agent_state"]["current_style"]
        require(
            row["condition"] in {"own_style_majority", "other_style_majority"}, "unknown condition"
        )
        own = 0.8 if row["condition"] == "own_style_majority" else 0.2
        d["observation"]["mean_field"][f"{current}_share"] = own
        other = "technical" if current == "fundamental" else "fundamental"
        d["observation"]["mean_field"][f"{other}_share"] = 0.8 if own == 0.2 else 0.2
        expected = semantic_payload(original)
        expected["messages"][1]["content"] = d
        require(semantic_payload(row["payload"]) == expected, "hidden source-pair modification")
        require(payload_digest(row["payload"]) == row["payload_sha256"], "legacy base hash differs")
        candidate = deepcopy(cmap[row["case_id"]])
        require(
            payload_digest(candidate["payload"]) == candidate["payload_sha256"],
            "legacy variant hash differs",
        )
        mode = candidate["payload"].pop("chat_template_kwargs")
        require(
            canonical(mode) == canonical(ENVELOPE["chat_template_kwargs"]),
            "unexpected mode override",
        )
        candidate["payload_sha256"] = row["payload_sha256"]
        require(canonical(candidate) == canonical(row), "variant changed another field")
        pairs.setdefault(agent, []).append(row)
    for agent, rows in pairs.items():
        require(
            len(rows) == 2 and len({r["condition"] for r in rows}) == 2, "incomplete majority pair"
        )
        changes = changed_paths(
            semantic_payload(rows[0]["payload"]), semantic_payload(rows[1]["payload"])
        )
        require(
            changes
            == [
                "messages.1.content.observation.mean_field.fundamental_share",
                "messages.1.content.observation.mean_field.technical_share",
            ],
            "unexpected majority contrast",
        )
        pairs[agent] = changes
        d = strict_json(payloads[agent]["messages"][1]["content"])
        documents[agent] = d
        inventory[agent] = [
            {"path": p, "category": field_handling(p)[0], "new_handling": field_handling(p)[1]}
            for p in sorted(flatten(d))
        ]
        a, o = d["agent_state"], d["observation"]
        facts[agent] = {
            "current_style": a["current_style"],
            "portfolio": a["portfolio"],
            "counterfactual": a["counterfactual"],
            "drawdown": a["drawdown"],
            "reference_value": o["fundamental_value"],
            "close": o["last_close"],
            "reference_above_price_fraction": o["fundamental_value"] / o["last_close"] - 1,
            "return_5d": o["return_5d"],
            "review_session": o["session_date"],
            "population_as_of_session": o["mean_field"]["as_of_session"],
        }
        require(o["mean_field"]["as_of_session"] < o["session_date"], "source clock not lagged")
    require(variant["checks"]["candidate_count"] == 4, "variant receipt scope differs")
    assessment = {
        "old_within_account_majority_changed_paths": pairs,
        "old_accounts": facts,
        "cross_account_differing_paths": changed_paths(documents["A000"], documents["A001"]),
        "old_observations_equal": documents["A000"]["observation"]
        == documents["A001"]["observation"],
        "old_system_order_task_ambiguity": all(
            "one limit order" in p["messages"][0]["content"] for p in payloads.values()
        ),
        "cross_account_style_or_return_identified": False,
        "original_sources_followed_beyond_declared_four_files": False,
    }
    return assessment, inventory


def envelope_inventory(base: list[dict], candidates: list[dict]) -> dict:
    output = {}
    for name, rows in (("original", base), ("mode_variant", candidates)):
        envelope = deepcopy(rows[0]["payload"])
        envelope["messages"][1].pop("content")
        output[name] = [
            {
                "path": path,
                "handling": "统一换成策略选择说明，修正下单歧义"
                if path == "messages.0.content"
                else "在所有新格中固定；模型参数沿用已冻结模式设置，不构成调用授权",
            }
            for path in sorted(flatten(envelope))
        ]
    return output


def report(rows: list[dict], inventory: dict) -> str:
    names = {"fundamental": "基本面", "technical": "技术"}
    lines = [
        "# 下一次先分清：AI到底在看哪条信息",
        "",
        "本轮完成了16份全新假设输入和32对比较的离线设计。没有问模型，所以还没有新的行为结果。16份材料不是16次调用授权。",
        "",
        "## 为什么不能直接比较原来的两个账户",
        "",
        "两个账户看见的行情相同：收盘约194.97，系统参考价约205.19，五日跌幅约6.56%。但一个持100股，另一个空仓；亏损、替代收益、回撤和交易经历都不同。两者都看见‘另一策略收益更好’，因此跨账户差异不能说明原策略惯性或追收益。",
        "",
        "原来每个账户内部的两份材料，确实只改了两种策略的群体占比，可以检查该账户在那个情境下是否对多数信息敏感。它们不能单独解释540次旧选择为何几乎都转向基本面。新模板共同修正旧说明里要求下单、回答格式却要求选策略的歧义；旧文件保留。",
        "",
        "## 新材料如何避免把几件事混在一起",
        "",
        "全部使用同一假设账户：起初现金10000、100股，每股100；十期后价格95，始终不交易，权益19500，亏损和回撤都是2.5%。每日损益都是−50，每日收益按前一期价格计算。行情、动作、财富和亏损在16份材料里保持相同。",
        "",
        "| 改变的信息 | 两种情况 | 能检查什么 |",
        "|---|---|---|",
        "| 策略标签 | 基本面／技术；历史标签一起对应 | "
        "对当前及声明历史标签的反应，不能证明真实习惯 |",
        "| 参考估值 | 95／105，当前价固定95 | "
        "对这组无价差／参考价较高信息的反应，不是真实公司价值 |",
        "| 策略收益信息 | 基本面较好／技术较好，−0.5%与−4.5%互换 | "
        "对明确给定的具名基准收益反应，不是个人换策略后的收益 |",
        "| 群体多数 | 另外100位代理中基本面80人／技术80人 | "
        "对策略人数多数的反应，不是买卖方向或真人跟风 |",
        "",
        "基准的初始股票仓位10%／90%、现金和持股构造只留在离线凭据，提示仅展示假设收益及其差距，避免同时给出变化的仓位风险线索。两种策略的收益始终按名字固定，改变当前策略标签时不会把收益排名也倒过来。",
        "",
        "16格覆盖四项信息的全部二选一组合。每项可配成8对，每对其余三项相同；合计32对共享这16个端点，不能当成32次独立实验。群体信息统一截至第9期，策略复盘在第10期。",
        "",
        "| 输入编号 | 原策略标签 | 参考价 | 收益较好的策略 | 群体多数 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        f = row["factors"]
        lines.append(
            f"| {row['case_id']} | {names[f['current_style']]} | "
            f"{95 if f['reference_condition'] == 'neutral' else 105} | "
            f"{names[f['benchmark_favored_style']]} | {names[f['majority_style']]} |"
        )
    lines += [
        "",
        "## 怎样判断结果，哪些结论仍然不能下",
        "",
        "预先统一记录最终选基本面还是技术，再辅助记录保持或切换。当前策略标签一换，同一个目标可能从保持变成切换，所以不能只看总切换率。逐对展示结果；缺回答、无效回答和自相矛盾回答都保留为未知，不能补成保持或零差异。",
        "",
        "这是检查模型对指定输入的反应，不足以证明人类心理机制、盈亏不对称、真实习惯或市场价格反馈。无反应也不能排除其他强度、其他行情下的作用。所有人这轮都亏2.5%，因此没有检验盈利与亏损的差异。",
        "",
        "当前新增请求上限为0，原授权21/21已用完。实际执行前还需明确新增授权，固定顺序、次数及失败处理，并解决怎样识别响应复用。换提示、换编号或收到不同ID，都不能单独证明得到了新的独立样本。",
        "",
        "本轮没有新增模型回答、真人记录或市场运行；没有继续SCIT优化。机器校验只说明材料算术和对照边界一致，不能替代行为验证。",
        "",
        f"已逐字段登记原两账户输入，共{sum(len(v) for v in inventory.values())}个字段位置，"
        "仅统计用户正文，见 source_field_inventory.json；共同system、角色和模型参数另见 "
        "source_envelope_inventory.json。全部四来源在生成前后核对指纹。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = strict_json(args.plan.read_text())
    declared = plan["additional_prompt_material_sources"]
    require(set(declared) == set(SOURCE_NAMES), "exact four-source scope required")
    before = {p: fingerprint(args.workspace / p) for p in declared}
    require(before == declared, "frozen source changed")
    base, candidates = (
        [strict_json(line) for line in (args.workspace / SOURCE_NAMES[i]).read_text().splitlines()]
        for i in (0, 2)
    )
    receipt, variant = (strict_json((args.workspace / SOURCE_NAMES[i]).read_text()) for i in (1, 3))
    assessment, inventory = inspect_sources(base, receipt, candidates, variant)
    rows = build_bank()
    edges = contrast_edges(rows)
    require(all(not validate_case(r) for r in rows), "invalid generated material")
    after = {p: fingerprint(args.workspace / p) for p in declared}
    require(before == after, "source changed during design")
    args.output.mkdir(parents=True, exist_ok=False)
    results = {
        "design_specification.json": design_specification(),
        "contrasts.json": edges,
        "source_assessment.json": assessment,
        "source_field_inventory.json": inventory,
        "source_envelope_inventory.json": envelope_inventory(base, candidates),
        "input_receipt.json": {
            "sources_before": before,
            "sources_after": after,
            "all_unchanged": True,
            "scope": (
                "Only four declared prompt/material files; "
                "no referenced raw sources or human final data opened."
            ),
        },
        "validation.json": {
            "eligible_cells": len(rows),
            "unique_payloads": len({r["payload_sha256"] for r in rows}),
            "matched_edges": len(edges),
            "edges_per_factor": {f: sum(e["factor"] == f for e in edges) for f in FACTORS},
            "model_answers_present": 0,
            "new_model_calls": 0,
            "private_ledger_not_in_payload": True,
        },
    }
    for name, value in results.items():
        (args.output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    (args.output / "inputs.jsonl").write_text("".join(canonical(r) + "\n" for r in rows))
    (args.output / "REPORT.md").write_text(report(rows, inventory))
    print(canonical(results["validation.json"]))


if __name__ == "__main__":
    main()
