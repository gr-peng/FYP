"""Run six invented local execution demonstrations, without model or data access."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from experiments import conditional_order_execution as demo
from scripts.validate_memory_information import save

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "experiments/conditional_order_execution.py",
    "experiments/conditional_memory_prompt.py",
    "experiments/conditional_order_memory.py",
    "src/behavioral_market/market/exchange.py",
    "src/behavioral_market/market/matching_engine.py",
    "src/behavioral_market/market/order_book.py",
    "src/behavioral_market/market/order.py",
    "src/behavioral_market/market/trade.py",
    "src/behavioral_market/market/portfolio.py",
    "src/behavioral_market/market/money.py",
    "src/behavioral_market/evaluation/rp_metrics.py",
    "scripts/demo_conditional_order_execution.py",
    "scripts/validate_memory_information.py",
    "tests/test_conditional_order_execution.py",
    "tests/test_conditional_order_execution_demo.py",
)


def run(output):
    # Explicit new directory; never overwrite or silently continue an old demo.
    output.mkdir(parents=True, exist_ok=False)
    fingerprints = {}
    for name in SOURCES:
        data = (ROOT / name).read_bytes()
        copied = output / "sources" / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(data)
        fingerprints[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    save(output / "source_manifest.json", fingerprints)
    jobs = [("half_" + arm, arm, 0.5, 0.5, demo.DEFAULT_CONFIG) for arm in demo.prompt.ARMS]
    jobs.extend(
        [
            ("partial_fill_witness", demo.prompt.ARMS[2], 1, 0, demo.DEFAULT_CONFIG),
            ("all_buy_zero_trades", demo.prompt.ARMS[2], 1, 1, demo.DEFAULT_CONFIG),
            (
                "no_external_opportunities",
                demo.prompt.ARMS[2],
                0.5,
                0.5,
                replace(demo.DEFAULT_CONFIG, opportunities=(False,) * 12),
            ),
        ]
    )
    summaries = {}
    for name, arm, a, b, config in jobs:
        callbacks = dict(zip(demo.AGENTS, (lambda _, p=a: p, lambda _, p=b: p), strict=True))
        result = demo.run_demo(callbacks, arm, config)
        result.update(model_POSTs=0, callback_external_POSTs=0, predictor_kind="fixed_constants")
        save(output / (name + ".json"), result)
        summaries[name] = {
            "arm": arm,
            "callback_buy_probabilities": [a, b],
            "denominators": result["denominators"],
            "execution": result["rp_metrics"]["execution"],
            "accounts_final": result["accounts_final"],
            "human_alignment": result["human_alignment"],
        }
    for name, fingerprint in fingerprints.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != fingerprint["sha256"]:
            raise RuntimeError("source changed during local demonstration")
    save(
        output / "summary.json",
        {
            "status": "complete_pending_independent_review",
            "model_POSTs": 0,
            "human_rows_read": 0,
            "demonstrations": summaries,
            "scope": "Execution accounting only; callbacks are constants, not language models.",
            "future": "Do not reuse I static scenario probabilities as new-state predictions.",
        },
    )
    (output / "REPORT.txt").write_text(
        "概率、下单、成交和下一次记忆：本地工程演示\n\n"
        "两个虚构账户，各500现金、4股；两期各三个机会。这里用固定概率程序，"
        "没有调用语言模型，也没有真人答案。\n"
        "三档记忆都用买卖各一半的同一程序与同一随机数；"
        "另保留部分成交、全部买入零成交、完全不提供下单机会三种检查。\n"
        "部分成交例：先成交2股，再成交1股；甲变成350现金、7股，乙650现金、1股。"
        "第一期结束每股分红10，甲420、乙660，新增80现金单独记账，股票总数仍8。\n"
        "没成交的已接受订单也会进入提交历史；被阻断的机会、被拒绝的意图和成交分别记录。"
        "第80秒最后一个机会结束后，另记第120秒的挂单到期和分红，再进入下一期。\n"
        "价格与数量均人为固定，市场结果不能说明模型表现好或市场更真实。"
        "真人一致率和概率误差为未知，不填写零分。\n",
        encoding="utf-8",
    )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"demonstrations": len(result), "model_POSTs": 0}))


if __name__ == "__main__":
    main()
