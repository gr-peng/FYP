"""The local demo preserves full traces and refuses existing output directories."""

import json

import pytest
from test_conditional_order_memory import no_external_access as no_external_access

from scripts import demo_conditional_order_execution as runner


def test_local_demo_writes_six_full_traces_and_unknown_human_score(tmp_path):
    out = tmp_path / "demo"
    summaries = runner.run(out)
    assert len(summaries) == 6
    summary = json.loads((out / "summary.json").read_text())
    assert summary["model_POSTs"] == summary["human_rows_read"] == 0
    for name in summaries:
        result = json.loads((out / (name + ".json")).read_text())
        assert len(result["events"]) == 6
        assert len(result["preallocated_uniforms"]) == 12
        assert result["human_alignment"]["status"] == "unknown"
        assert result["model_POSTs"] == result["callback_external_POSTs"] == 0
        assert result["predictor_kind"] == "fixed_constants"
    for source in runner.SOURCES:
        assert (out / "sources" / source).read_bytes() == (runner.ROOT / source).read_bytes()
    assert "固定概率程序" in (out / "REPORT.txt").read_text()


def test_existing_demo_is_not_overwritten_or_resumed(tmp_path):
    marker = tmp_path / "prior.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        runner.run(tmp_path)
    assert list(tmp_path.iterdir()) == [marker]
    assert marker.read_text() == "preserve"
