import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from behavioral_market.data.archive import write_json

ROOT = Path(__file__).resolve().parents[3]


def usage_report(directory, pricing=None):
    attempts = []
    local_hits = 0
    for path in sorted(directory.rglob("llm_responses.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("cached"):
                local_hits += 1
            else:
                attempts.append(record)
    total_input = total_output = reasoning = cached_input = 0
    cost = 0.0
    models, fingerprints = set(), set()
    for record in attempts:
        body = record.get("response") or {}
        usage = body.get("usage", {})
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        cached = usage.get(
            "prompt_cache_hit_tokens",
            usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        )
        reason = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        total_input += prompt
        total_output += completion
        reasoning += reason
        cached_input += cached
        if body.get("model"):
            models.add(body["model"])
        if body.get("system_fingerprint"):
            fingerprints.add(body["system_fingerprint"])
        if pricing and usage:
            stamp = datetime.fromisoformat(record["requested_at"]).astimezone(
                ZoneInfo(pricing["timezone"])
            )
            off_peak = pricing["off_peak_start_hour"] <= stamp.hour < pricing["off_peak_end_hour"]
            rates = pricing["off_peak" if off_peak else "peak"]
            cost += (
                (prompt - cached) * rates["input"]
                + cached * rates["cached_input"]
                + completion * rates["output"]
            ) / 1e6
    latency = [r["latency_seconds"] for r in attempts]
    return {
        "calls": len(attempts),
        "successful_http_calls": sum(r.get("http_status") == 200 for r in attempts),
        "failed_calls": sum(r.get("http_status") != 200 for r in attempts),
        "retry_calls": sum(r.get("retry_count", 0) > 0 for r in attempts),
        "repair_calls": sum(r.get("repair", 0) > 0 for r in attempts),
        "local_cache_hits": local_hits,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "reasoning_tokens": reasoning,
        "provider_cached_input_tokens": cached_input,
        "average_latency_seconds_including_queue": float(np.mean(latency)) if latency else None,
        "p95_latency_seconds_including_queue": float(np.percentile(latency, 95))
        if latency
        else None,
        "estimated_cost_cny": cost if pricing else None,
        "pricing_source": pricing["source"] if pricing else None,
        "models_reported": sorted(models),
        "system_fingerprints": sorted(fingerprints),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory", type=Path, nargs="?", default=ROOT / "outputs/meta_compute_2026/siliconflow"
    )
    parser.add_argument(
        "--pricing", type=Path, default=ROOT / "config/siliconflow_pricing_20260901.json"
    )
    args = parser.parse_args()
    result = usage_report(args.directory, json.loads(args.pricing.read_text()))
    write_json(args.directory / "api_usage.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
