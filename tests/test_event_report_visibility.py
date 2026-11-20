"""Run the existing report on freshly created fictional inputs only."""

import json
import sys

import httpx
import numpy as np
import pandas as pd
import pytest
import requests
from matplotlib.axes import Axes

from behavioral_market.data.archive import digest, write_json
from behavioral_market.evaluation import event_report
from behavioral_market.evaluation.metrics import path_metrics
from behavioral_market.simulation import event_runner


@pytest.fixture(autouse=True)
def offline_boundary(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Fictional report must not access credentials, datasets or HTTP")

    async def forbidden_async(*args, **kwargs):
        forbidden()

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(event_runner, "load_dataset", forbidden)
    monkeypatch.setattr(event_runner, "load_dotenv", forbidden)
    monkeypatch.setattr(event_report, "load_dataset", forbidden)


@pytest.fixture
def report_fixture(tmp_path, monkeypatch):
    dates = pd.bdate_range("2026-04-01", "2026-07-17").strftime("%Y-%m-%d")
    rows = []
    for offset, symbol in enumerate(["FIXTURE", "QQQ", "SPY"]):
        for i, session in enumerate(dates):
            rows.append(
                {
                    "symbol": symbol,
                    "session_date": session,
                    "close": 100 + i * 0.1 + np.sin(i / 4 + offset),
                    "volume": 1000 + i,
                }
            )
    bars = pd.DataFrame(rows)
    root = tmp_path / "workspace"
    root.mkdir()
    write_json(root / "fictional_manifest.json", {"issues": ["Entirely fictional test data."]})
    directory = root / "mock"
    directory.mkdir()
    config = {
        "data": {
            "primary_symbol": "FIXTURE",
            "robustness_symbols": [],
            "manifest": "fictional_manifest.json",
        },
        "population": {"n_agents": 2, "initial_cash": "1000", "initial_shares": 2},
        "suite": {"treatments": ["low_herding", "high_herding"], "seeds": [1], "shocks": []},
    }
    monkeypatch.setattr(event_report, "ROOT", root)
    monkeypatch.setattr(
        event_report,
        "load_dataset",
        lambda _: (bars.copy(deep=True), [], None, "fictional-manifest"),
    )
    actual = bars[(bars.symbol == "FIXTURE") & (bars.session_date >= "2026-06-12")]
    initial = float(actual.close.iloc[0])
    sessions = actual.session_date.tolist()
    curves = {}

    def add_run(name, visibility, treatment, final, *, seed=1, source="code-a", legacy=False):
        path = directory / name
        path.mkdir()
        normalized = np.linspace(100, final, len(sessions))
        market = pd.DataFrame(
            {"session_date": sessions[1:], "close": normalized[1:] / 100 * initial}
        )
        market.to_parquet(path / "market.parquet", index=False)
        equity = (1000 + 2 * initial) * final / 100
        pd.DataFrame(
            {
                "session_date": [sessions[-1]] * 2,
                "total_equity": [equity] * 2,
                "realized_pnl": [0.0, 0.0],
            }
        ).to_parquet(path / "agents.parquet", index=False)
        metrics = {
            **path_metrics(normalized),
            "hold_rate": 0.1,
            "switch_rate": 0.2,
            "order_execution_rate": 0.3,
            "fallback_rate": 0.0,
            "trades": 3,
        }
        write_json(path / "metrics.json", metrics)
        metadata = {
            "status": "complete",
            "n_agents": 2,
            "data_manifest_hash": "fictional-manifest",
            "mode": "endogenous",
            "treatment": treatment,
            "seed": seed,
            "shock": 0.0,
            "quality": "engineering_mock",
            "include_secondary": True,
            "source_hash": source,
            "symbol": "FIXTURE",
            "event_condition": "e1_e2",
            "artifact_hashes": {p.name: digest(p.read_bytes()) for p in path.glob("*.parquet")},
        }
        snapshot = {"behavior": {}, "run": {"treatment": treatment}}
        if not legacy:
            supplied = None if treatment == "rule_abm" else visibility == "visible"
            metadata.update(mean_field_visibility=visibility, prompt_mean_field_supplied=supplied)
            snapshot["behavior"]["mean_field_visibility"] = visibility
            snapshot["run"].update(
                mean_field_visibility=visibility, prompt_mean_field_supplied=supplied
            )
        write_json(path / "run_metadata.json", metadata)
        write_json(path / "config_snapshot.json", snapshot)
        curves[(visibility, source, treatment, seed)] = normalized
        return path

    return config, directory, root / "report", add_run, curves


@pytest.fixture
def plotted_curves(monkeypatch):
    captured = {}
    plot = Axes.plot

    def record(axis, x, y, *args, **kwargs):
        lines = plot(axis, x, y, *args, **kwargs)
        label = kwargs.get("label", "")
        if label.startswith(("low_herding /", "high_herding /", "rule_abm /")):
            assert label not in captured
            captured[label] = (list(x), np.asarray(y).copy())
        return lines

    monkeypatch.setattr(Axes, "plot", record)
    return captured


def read_comparisons(destination):
    table = pd.read_csv(destination / "treatment_comparisons.csv")
    return table[table.metric == "return"]


def assert_paths_and_plot(destination, expected, plotted):
    paths = pd.read_csv(destination / "normalized_paths.csv")
    groups = paths.groupby(["mean_field_visibility", "source_hash", "treatment", "seed"])
    assert len(groups) == len(expected) == len(plotted)
    for (visibility, source, treatment, seed), values in expected.items():
        recorded = groups.get_group((visibility, source, treatment, seed))
        np.testing.assert_allclose(recorded.normalized_price, values)
        dates, curve = plotted[f"{treatment} / {visibility} / {source}"]
        assert dates == recorded.session_date.tolist()
        np.testing.assert_allclose(curve, values)
    assert (destination / "normalized_paths.png").stat().st_size > 0
    assert (destination / "normalized_paths.svg").stat().st_size > 0


def test_report_keeps_opposite_visibility_effects_and_all_four_curves(
    report_fixture, plotted_curves
):
    config, directory, destination, add, expected = report_fixture
    add("visible_low", "visible", "low_herding", 100)
    add("visible_high", "visible", "high_herding", 120)
    add("hidden_low", "hidden", "low_herding", 140)
    add("hidden_high", "hidden", "high_herding", 110)
    event_report.evaluate(config, directory, destination)
    means = pd.read_csv(destination / "group_means.csv").set_index(
        ["mean_field_visibility", "treatment"]
    )
    assert len(means) == 4 and set(means.runs) == {1}
    for key, value in {
        ("visible", "low_herding"): 0.0,
        ("visible", "high_herding"): 0.2,
        ("hidden", "low_herding"): 0.4,
        ("hidden", "high_herding"): 0.1,
    }.items():
        assert means.loc[key, "return"] == pytest.approx(value)
    comparisons = read_comparisons(destination).set_index("mean_field_visibility")
    assert len(comparisons) == 2
    assert set(comparisons.n_pairs) == set(comparisons.n_low) == set(comparisons.n_high) == {1}
    assert comparisons.loc["visible", "paired_mean_high_minus_low"] == pytest.approx(0.2)
    assert comparisons.loc["hidden", "paired_mean_high_minus_low"] == pytest.approx(-0.3)
    uncertainty = pd.read_csv(
        destination / "group_uncertainty.csv", header=[0, 1], index_col=list(range(7))
    )
    assert len(uncertainty) == 4
    assert set(uncertainty.index.get_level_values("mean_field_visibility")) == {"visible", "hidden"}
    assert_paths_and_plot(destination, expected, plotted_curves)
    runs = pd.read_parquet(destination / "run_metrics.parquet")
    assert set(runs.mean_field_visibility_origin) == {"metadata"}
    assert runs.loc[runs.mean_field_visibility == "visible", "prompt_mean_field_supplied"].all()
    assert not runs.loc[runs.mean_field_visibility == "hidden", "prompt_mean_field_supplied"].any()
    evaluation = json.loads((destination / "evaluation_metadata.json").read_text())
    assert evaluation["mean_field_visibilities"] == ["hidden", "visible"]
    print(
        json.dumps(
            {
                "fixture": "four_fictional_runs",
                "visible_paired_difference": 0.2,
                "hidden_paired_difference": -0.3,
                "pairs_per_condition": 1,
                "preserved_curves": 4,
            }
        )
    )


def test_report_isolates_same_seed_by_source_in_tables_pairs_and_plot(
    report_fixture, plotted_curves
):
    config, directory, destination, add, expected = report_fixture
    for source, low, high in [("code-a", 100, 120), ("code-b", 140, 110)]:
        add(source + "_low", "visible", "low_herding", low, source=source)
        add(source + "_high", "visible", "high_herding", high, source=source)
    event_report.evaluate(config, directory, destination)
    means = pd.read_csv(destination / "group_means.csv")
    assert len(means) == 4 and set(means.source_hash) == {"code-a", "code-b"}
    comparisons = read_comparisons(destination).set_index("source_hash")
    assert len(comparisons) == 2 and set(comparisons.n_pairs) == {1}
    assert comparisons.loc["code-a", "paired_mean_high_minus_low"] == pytest.approx(0.2)
    assert comparisons.loc["code-b", "paired_mean_high_minus_low"] == pytest.approx(-0.3)
    assert_paths_and_plot(destination, expected, plotted_curves)


@pytest.mark.parametrize("difference", ["visibility", "source", "seed"])
def test_report_never_borrows_an_unmatched_counterpart(report_fixture, difference):
    config, directory, destination, add, _ = report_fixture
    add("low", "visible", "low_herding", 100)
    add(
        "high",
        "hidden" if difference == "visibility" else "visible",
        "high_herding",
        120,
        source="code-b" if difference == "source" else "code-a",
        seed=2 if difference == "seed" else 1,
    )
    event_report.evaluate(config, directory, destination)
    comparisons = read_comparisons(destination)
    assert set(comparisons.n_pairs) == {0}
    assert comparisons.paired_mean_high_minus_low.isna().all()
    assert set(comparisons.comparison_status) == {
        "no_common_seed" if difference == "seed" else "missing_treatment"
    }


def test_legacy_default_stays_visible_and_does_not_invent_prompt_exposure(report_fixture):
    config, directory, destination, add, _ = report_fixture
    add("old", "visible", "low_herding", 100, legacy=True)
    add("new", "visible", "high_herding", 120)
    event_report.evaluate(config, directory, destination)
    runs = pd.read_parquet(destination / "run_metrics.parquet").set_index("treatment")
    assert set(runs.mean_field_visibility) == {"visible"}
    assert runs.loc["low_herding", "mean_field_visibility_origin"] == "legacy_default_visible"
    assert pd.isna(runs.loc["low_herding", "prompt_mean_field_supplied"])
    assert runs.loc["high_herding", "mean_field_visibility_origin"] == "metadata"
    comparison = read_comparisons(destination).iloc[0]
    assert comparison.n_pairs == 1 and comparison.paired_mean_high_minus_low == pytest.approx(0.2)


def test_duplicate_full_run_identity_fails_instead_of_repeating_seed(report_fixture):
    config, directory, destination, add, _ = report_fixture
    add("first", "visible", "low_herding", 100)
    add("duplicate", "visible", "low_herding", 150)
    with pytest.raises(ValueError, match="duplicate run/seed"):
        event_report.evaluate(config, directory, destination)


@pytest.mark.parametrize("value", [None, True, 0, [], {}, "invalid"])
def test_report_rejects_invalid_explicit_visibility(report_fixture, value):
    config, directory, destination, add, _ = report_fixture
    path = add("bad", "visible", "low_herding", 100)
    metadata = json.loads((path / "run_metadata.json").read_text())
    metadata["mean_field_visibility"] = value
    write_json(path / "run_metadata.json", metadata)
    with pytest.raises(ValueError, match="mean_field_visibility"):
        event_report.evaluate(config, directory, destination)


@pytest.mark.parametrize(
    "treatment,visibility,value",
    [
        ("low_herding", "visible", None),
        ("low_herding", "visible", False),
        ("low_herding", "visible", 0),
        ("low_herding", "visible", 1),
        ("low_herding", "visible", "true"),
        ("low_herding", "hidden", True),
        ("rule_abm", "hidden", False),
        ("rule_abm", "hidden", 0),
    ],
)
def test_report_rejects_contradictory_or_nonboolean_supplied_flag(
    report_fixture, treatment, visibility, value
):
    config, directory, destination, add, _ = report_fixture
    path = add("bad", visibility, treatment, 100)
    metadata = json.loads((path / "run_metadata.json").read_text())
    metadata["prompt_mean_field_supplied"] = value
    write_json(path / "run_metadata.json", metadata)
    with pytest.raises(ValueError, match="prompt_mean_field_supplied"):
        event_report.evaluate(config, directory, destination)


@pytest.mark.parametrize("evidence", ["snapshot.behavior", "snapshot.run", "path"])
def test_legacy_missing_visibility_rejects_positive_hidden_evidence(report_fixture, evidence):
    config, directory, destination, add, _ = report_fixture
    path = add(
        "case_mean_field_hidden" if evidence == "path" else "case",
        "visible",
        "low_herding",
        100,
        legacy=True,
    )
    if evidence != "path":
        snapshot = json.loads((path / "config_snapshot.json").read_text())
        snapshot[evidence.split(".")[1]]["mean_field_visibility"] = "hidden"
        write_json(path / "config_snapshot.json", snapshot)
    with pytest.raises(ValueError, match="mean_field_visibility"):
        event_report.evaluate(config, directory, destination)


def test_snapshot_cannot_contradict_explicit_visibility(report_fixture):
    config, directory, destination, add, _ = report_fixture
    path = add("bad", "hidden", "low_herding", 100)
    snapshot = json.loads((path / "config_snapshot.json").read_text())
    snapshot["behavior"]["mean_field_visibility"] = "visible"
    write_json(path / "config_snapshot.json", snapshot)
    with pytest.raises(ValueError, match="mean_field_visibility"):
        event_report.evaluate(config, directory, destination)


def test_explicit_hidden_custom_path_and_rule_null_are_valid(report_fixture):
    config, directory, destination, add, _ = report_fixture
    add("custom_directory", "hidden", "rule_abm", 100)
    event_report.evaluate(config, directory, destination)
    row = pd.read_parquet(destination / "run_metrics.parquet").iloc[0]
    assert row.mean_field_visibility == "hidden" and pd.isna(row.prompt_mean_field_supplied)


def test_metrics_cannot_overwrite_validated_identity_or_exposure(report_fixture):
    config, directory, destination, add, _ = report_fixture
    path = add("valid", "visible", "low_herding", 100)
    metrics = json.loads((path / "metrics.json").read_text())
    metrics.update(mean_field_visibility="hidden", source_hash="other-code", seed=999)
    write_json(path / "metrics.json", metrics)
    event_report.evaluate(config, directory, destination)
    row = pd.read_parquet(destination / "run_metrics.parquet").iloc[0]
    assert (row.mean_field_visibility, row.source_hash, row.seed) == ("visible", "code-a", 1)


@pytest.mark.parametrize("value", [None, False, ""])
def test_report_rejects_unrecorded_source_identity(report_fixture, value):
    config, directory, destination, add, _ = report_fixture
    path = add("bad", "visible", "low_herding", 100)
    metadata = json.loads((path / "run_metadata.json").read_text())
    metadata["source_hash"] = value
    write_json(path / "run_metadata.json", metadata)
    with pytest.raises(ValueError, match="source_hash"):
        event_report.evaluate(config, directory, destination)


def test_report_text_uses_included_scope_and_leaves_unverified_plan_unknown(report_fixture):
    config, directory, destination, add, _ = report_fixture
    for visibility, low, high in [("visible", 100, 120), ("hidden", 140, 110)]:
        add(visibility + "_low", visibility, "low_herding", low)
        add(visibility + "_high", visibility, "high_herding", high)
    # A parameter product is not a frozen, complete set of planned run identities.
    config["suite"]["seeds"] = list(range(20))
    config["suite"]["shocks"] = [0.1, 0.2]
    event_report.evaluate(config, directory, destination)
    context = json.loads((destination / "report_context.json").read_text())
    text = (destination / "REPORT.md").read_text()
    assert context["included_runs"] == 4 and context["agents_declared"] == [2]
    assert context["seed_values"] == [1] and context["planned_completion"] is None
    assert context["calendar_declared_in_evaluation_config"] is None
    assert context["reference_prices"]["date_count"] == 26
    assert context["reference_prices"]["first_date"] == "2026-06-12"
    assert len(context["runs"]) == 4
    assert {run["market_distinct_dates"] for run in context["runs"]} == {25}
    assert {run["market_first_date"] for run in context["runs"]} == {"2026-06-15"}
    assert {run["market_last_date"] for run in context["runs"]} == {"2026-07-17"}
    assert context["recorded_noncached_responses"] == 0
    assert "本次纳入 4 次完整状态运行" in text and "模拟账户数：2" in text
    assert "随机种子：1 个（1）" in text and "计划完成度：未知" in text
    assert "记录日期数" in text and "26 个记录日期" in text
    for unsupported in [
        "48 次",
        "30-agent",
        "690",
        "3 个 seed",
        "23 个 XNYS",
        "首轮真实 API",
        "thinking 关闭",
        "没有出现 429",
        "HTTP 402",
        "9 月 14 日",
    ]:
        assert unsupported not in text


@pytest.mark.parametrize("folder", ["mock", "siliconflow", "custom_batch"])
def test_folder_name_is_not_producer_or_price_evidence(report_fixture, folder):
    config, directory, destination, add, _ = report_fixture
    add("run", "visible", "low_herding", 100)
    renamed = directory.with_name(folder)
    if renamed != directory:
        directory.rename(renamed)
    event_report.evaluate(config, renamed, destination)
    context = json.loads((destination / "report_context.json").read_text())
    row = context["runs"][0]
    assert row["provider"]["status"] == row["model"]["status"] == "unrecorded"
    assert row["provider"]["value"] is row["model"]["value"] is None
    assert context["pricing_assumption_source"] is None
    assert json.loads((destination / "api_usage.json").read_text())["estimated_cost_cny"] is None
    text = (destination / "REPORT.md").read_text()
    assert "服务来源声明：未记录" in text and "模型名称声明：未记录" in text
    assert "确定性 MockLLMClient" not in text and "首轮真实 API" not in text


@pytest.mark.parametrize("declarations", ["matching", "conflicting", "snapshot_only"])
def test_producer_declarations_and_requested_mode_are_not_verified_behavior(
    report_fixture, declarations
):
    config, directory, destination, add, _ = report_fixture
    path = add("run", "visible", "low_herding", 100)
    metadata = json.loads((path / "run_metadata.json").read_text())
    snapshot = json.loads((path / "config_snapshot.json").read_text())
    metadata["llm_parameters"] = {"enable_thinking": False}
    if declarations != "snapshot_only":
        metadata.update(provider="provider-a", model="requested-a")
    snapshot["run"].update(
        provider="provider-b" if declarations == "conflicting" else "provider-a",
        model="requested-b" if declarations == "conflicting" else "requested-a",
    )
    write_json(path / "run_metadata.json", metadata)
    write_json(path / "config_snapshot.json", snapshot)
    event_report.evaluate(config, directory, destination)
    row = json.loads((destination / "report_context.json").read_text())["runs"][0]
    text = (destination / "REPORT.md").read_text()
    if declarations == "conflicting":
        assert row["provider"]["value"] is None and row["provider"]["status"] == "conflicting"
        assert row["model"]["value"] is None and row["model"]["status"] == "conflicting"
        assert "声明冲突 1 次" in text
    else:
        assert row["provider"]["value"] == "provider-a" and row["model"]["value"] == "requested-a"
        assert row["provider"]["status"] == row["model"]["status"] == "declared"
    assert "没有验证实际思考模式" in text and "thinking 关闭" not in text


def test_usage_scope_and_response_labels_do_not_become_run_producer_proof(report_fixture):
    config, directory, destination, add, _ = report_fixture
    path = add("run", "visible", "low_herding", 100)
    metadata = json.loads((path / "run_metadata.json").read_text())
    metadata.update(provider="declared-provider", model="requested-model")
    write_json(path / "run_metadata.json", metadata)
    precheck = directory / "outside_included_runs"
    precheck.mkdir()
    records = [
        {
            "cached": False,
            "http_status": 200,
            "latency_seconds": 0.1,
            "retry_count": 0,
            "response": {
                "model": "returned-label",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        },
        {"cached": True, "response": {"model": "cache-only-label"}},
    ]
    (precheck / "llm_responses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    event_report.evaluate(config, directory, destination)
    context = json.loads((destination / "report_context.json").read_text())
    assert context["included_runs"] == 1 and context["recorded_noncached_responses"] == 1
    assert context["recorded_local_cache_hits"] == 1
    assert context["response_model_labels"] == ["returned-label"]
    assert context["runs"][0]["model"]["value"] == "requested-model"
    text = (destination / "REPORT.md").read_text()
    assert "可能包括预检、未完成或未纳入" in text
    assert "目录内非缓存响应记录 1 条" in text and "本地缓存命中记录 1 条" in text
    assert "不构成模型身份或真实远端调用的独立证明" in text


@pytest.mark.parametrize("style_recorded", [True, False])
def test_degraded_description_separates_order_and_style_evidence(report_fixture, style_recorded):
    config, directory, destination, add, _ = report_fixture
    path = add("run", "visible", "low_herding", 100)
    metadata = json.loads((path / "run_metadata.json").read_text())
    metadata["quality"] = "degraded"
    write_json(path / "run_metadata.json", metadata)
    if style_recorded:
        metrics = json.loads((path / "metrics.json").read_text())
        metrics["invalid_style_rate"] = 1.0
        write_json(path / "metrics.json", metrics)
    event_report.evaluate(config, directory, destination)
    row = json.loads((destination / "report_context.json").read_text())["runs"][0]
    assert row["quality_recorded"] == "degraded" and row["order_fallback_rate_recorded"] == 0
    assert row["invalid_style_rate_recorded"] == (1.0 if style_recorded else None)
    text = (destination / "REPORT.md").read_text()
    assert "记录标为降级的运行：1 次" in text and "降级运行（订单 fallback >5%）" not in text
    assert "订单回退比例（记录值）" in text and "策略答复无效率（记录值）" in text
    assert "0.0%" in text and ("100.0%" in text if style_recorded else "未记录" in text)


def test_explicit_cli_pricing_is_a_uniform_assumption_with_existing_cost_math(
    report_fixture, monkeypatch
):
    config, directory, destination, add, _ = report_fixture
    add("run", "visible", "low_herding", 100)
    rates = {"input": 1.0, "cached_input": 0.5, "output": 2.0}
    pricing = {
        "source": "https://fixture.invalid/archived-rates",
        "timezone": "UTC",
        "off_peak_start_hour": 0,
        "off_peak_end_hour": 24,
        "peak": rates,
        "off_peak": rates,
    }
    pricing_path = directory.parent / "fictional_pricing.json"
    config_path = directory.parent / "fictional_config.json"
    write_json(pricing_path, pricing)
    write_json(config_path, config)
    record = {
        "cached": False,
        "http_status": 200,
        "latency_seconds": 0.1,
        "retry_count": 0,
        "requested_at": "2026-07-01T00:00:00+00:00",
        "response": {
            "model": "returned-label",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 20},
            },
        },
    }
    (directory / "llm_responses.jsonl").write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "event_report",
            "--config",
            str(config_path),
            "--directory",
            str(directory),
            "--destination",
            str(destination),
            "--pricing",
            str(pricing_path),
        ],
    )
    event_report.main()
    usage = json.loads((destination / "api_usage.json").read_text())
    assert usage["estimated_cost_cny"] == pytest.approx((80 * 1 + 20 * 0.5 + 50 * 2) / 1e6)
    context = json.loads((destination / "report_context.json").read_text())
    assert context["pricing_assumption_source"] == pricing["source"]
    text = (destination / "REPORT.md").read_text()
    assert "统一套价估算" in text and "不证明服务商匹配" in text and "不等同到账单实扣" in text


def test_no_included_runs_does_not_invent_a_completed_experiment(report_fixture):
    config, directory, destination, _, _ = report_fixture
    event_report.evaluate(config, directory, destination)
    context = json.loads((destination / "report_context.json").read_text())
    assert context["included_runs"] == 0 and context["runs"] == []
    assert context["seed_values"] == [] and context["planned_completion"] is None
    text = (destination / "REPORT.md").read_text()
    assert "当前没有符合筛选条件且状态完整的模拟运行" in text
    assert "首轮真实 API" not in text and "48 次" not in text
