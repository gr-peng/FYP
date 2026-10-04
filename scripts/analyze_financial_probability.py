# ruff: noqa: E402
"""Score a complete frozen development matrix with paired whole-session uncertainty."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from behavioral_market.data.archive import digest, write_json
from experiments.financial_probability import METHODS, key
from experiments.market_quote_baselines import metrics, weights
from experiments.probability_elicitation import validate_probabilities
from experiments.replication_development_data import FREEZE, checked_freeze

OUT = "outputs/financial_probability_development_20261004"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def paired_interval(session_differences, seed=20261004, n=10000):
    values = np.array(session_differences, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("invalid independent-cluster contrasts")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return {
        "difference": float(values.mean()),
        "percentile_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "independent_sessions": len(values),
        "bootstrap_draws": n,
    }


def main(root):
    checked_freeze(root)
    out = root / OUT
    prep = read(out / "preparation.json")
    assert prep["freeze_sha256"] == digest((root / FREEZE).read_bytes())
    for path, expected in prep["files"].items():
        assert digest((out / path).read_bytes()) == expected, path
    meta = read(out / "main/metadata.json")
    assert meta["status"] == "complete" and meta["valid"] == meta["expected_responses"] == 3200
    assert meta["preparation_sha256"] == digest((out / "preparation.json").read_bytes())
    targets = rows(out / "targets.jsonl")
    assert [key(r) for r in targets] == prep["target_ids"]
    responses = rows(out / "main/responses.jsonl")
    indexed = {(r["target_id"], r["method"], r["order"]): r for r in responses}
    expected = {
        (key(r), m, o) for r in targets for m in METHODS for o in ("buy_first", "sell_first")
    }
    assert len(indexed) == len(responses) and set(indexed) == expected
    transport_rows = rows(out / "main/transport/llm_responses.jsonl")
    literal = {
        row["request_hash"]: row["response"]
        for row in transport_rows
        if row.get("response") is not None
    }
    # Check recorded model probabilities against literal responses and both menu maps.
    for (_, _, order), row in indexed.items():
        body = literal[row["metadata"]["calls"][-1]["request_hash"]]
        choice = body["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert not choice["message"].get("reasoning_content")
        assert choice["message"]["content"] == row["metadata"]["raw_response"]
        p, _ = validate_probabilities(row["metadata"]["raw_response"], {"A": "", "B": ""})
        assert p["A" if order == "buy_first" else "B"] == row["probability_buy"]
        assert row["metadata"]["status"] == "ok"
    predictions = {k: np.array(v) for k, v in read(out / "baselines.json").items()}
    menu_effect = {}
    w = weights(targets)
    for method in METHODS:
        first = np.array(
            [indexed[(key(r), method, "buy_first")]["probability_buy"] for r in targets]
        )
        second = np.array(
            [indexed[(key(r), method, "sell_first")]["probability_buy"] for r in targets]
        )
        predictions[method] = (first + second) / 2
        menu_effect[method] = {
            "mean_absolute_buy_probability_difference": float(w @ np.abs(first - second)),
            "signed_buy_first_minus_sell_first": float(w @ (first - second)),
        }
    session_scores = {}
    for session in sorted({r["provenance"]["session"] for r in targets}):
        indices = [i for i, r in enumerate(targets) if r["provenance"]["session"] == session]
        session_scores[session] = metrics(
            [targets[i] for i in indices], {k: v[indices] for k, v in predictions.items()}
        )
    contrasts = {}
    for reference in (
        "no_examples",
        "neutral_labels",
        "shuffled_labels",
        "memory_ridge_logistic",
        "demonstration_mean",
    ):
        deltas = [
            score["methods"]["real_labels"]["brier_session_person_balanced"]
            - score["methods"][reference]["brier_session_person_balanced"]
            for score in session_scores.values()
        ]
        contrasts[f"real_labels-minus-{reference}"] = paired_interval(deltas)
    usage, status = Counter(), Counter()
    for r in transport_rows:
        status[str(r.get("http_status", r.get("status")))] += 1
        if not r.get("cached") and r.get("response") is not None:
            for k, v in r["response"].get("usage", {}).items():
                if isinstance(v, (int, float)):
                    usage[k] += v
    probabilities = []
    y = np.array([r["target"]["buy"] for r in targets])
    calibration = {}
    for method, p in predictions.items():
        bins = []
        for index in range(10):
            mask = (p >= index / 10) & ((p < (index + 1) / 10) if index < 9 else (p <= 1))
            mass = float(w[mask].sum())
            bins.append(
                {
                    "lower": index / 10,
                    "upper": (index + 1) / 10,
                    "weight": mass,
                    "predicted": float(w[mask] @ p[mask] / mass) if mass else None,
                    "observed": float(w[mask] @ y[mask] / mass) if mass else None,
                }
            )
        calibration[method] = bins
    result = {
        "scope": "Independent development sessions after train-only fitting; test remains closed",
        "overall": metrics(targets, predictions),
        "per_session": session_scores,
        "contrasts": contrasts,
        "menu_order_effects": menu_effect,
        "calibration": calibration,
        "responses": len(responses),
        "repairs": sum(r["metadata"]["repair_count"] for r in responses),
        "latency_including_queue_seconds": dict(
            zip(
                ["median", "p95"],
                np.quantile(
                    [r["metadata"]["elapsed_including_queue_seconds"] for r in responses],
                    [0.5, 0.95],
                ).tolist(),
                strict=True,
            )
        ),
        "successful_uncached_usage": dict(usage),
        "transport_statuses": dict(status),
        "response_sha256": digest((out / "main/responses.jsonl").read_bytes()),
        "preparation_sha256": digest((out / "preparation.json").read_bytes()),
        "analysis_source_sha256": digest(Path(__file__).read_bytes()),
    }
    write_json(out / "analysis.json", result)
    for i, r in enumerate(targets):
        probabilities.append(
            {
                "target_id": key(r),
                "provenance": r["provenance"],
                "observed_buy": r["target"]["buy"],
                "predictions": {m: float(p[i]) for m, p in predictions.items()},
            }
        )
    write_json(out / "predictions.json", probabilities)
    lines = [
        "# 真人金融行为：独立开发集检验",
        "",
        "先固定51场训练数据、特征和统计模型，再打开另16场开发实验。每场按订单标识的哈希选25个时点，共400个时点。这里检验的是已经下单时买还是卖；不能判断是否交易、如何报价，也不能直接证明真实市场价格更像现实。",
        "",
        "| 方法 | Brier误差，越低越好 | 对数损失 | 预测买入率 |",
        "|---|---:|---:|---:|",
    ]
    names = {
        "training_constant": "训练买入率常数",
        "constant_half": "各一半",
        "lifetime_history_prior4": "个人历史统计",
        "memory_ridge_logistic": "带个人历史的统计模型",
        "demonstration_mean": "四个示例的均值",
        "no_examples": "LLM无示例",
        "real_labels": "LLM真人标签示例",
        "neutral_labels": "LLM中性标签对照",
        "shuffled_labels": "LLM打乱标签对照",
    }
    for method, score in result["overall"]["methods"].items():
        lines.append(
            f"| {names[method]} | {score['brier_session_person_balanced']:.5f} | "
            f"{score['log_loss_session_person_balanced']:.5f} | "
            f"{score['predicted_buy_rate_session_person_balanced']:.2%} |"
        )
    lines.extend(
        [
            "",
            "真人标签示例的误差减去对照误差；负值才表示真人标签更好。以下区间按16个完整场次重采样，不把400次决策当成400个独立市场。多个比较为描述性，不能挑一个区间宣布成功。",
            "",
            "| 对照 | 误差差值 | 95%区间 |",
            "|---|---:|---|",
        ]
    )
    for name, value in contrasts.items():
        reference = name.removeprefix("real_labels-minus-")
        lo, hi = value["percentile_95"]
        lines.append(
            f"| {names[reference]} | {value['difference']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |"
        )
    lines.extend(
        [
            "",
            f"四种提示均交换买卖选项位置再取平均，全部{len(responses)}条响应有效；修复次数{result['repairs']}。成功新调用token用量：{dict(usage)}。",
            "",
            "开发阶段使用真人过去已发生的交易记录，尚非代理自己生成的历史。样本只代表满足时间及账户可重建条件的主动下单时点。当前盘口、挂单占款和撤单时点缺失。独立测试集继续封存；这些结果不能代替股票/加密货币事件的外部验证。",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"overall": result["overall"], "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main(ROOT)
