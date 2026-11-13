"""Validated client measurements from vLLM --save-result --save-detailed.

No profiler-derived metrics are inferred here. This module uses only the stdlib
so it can also be copied into a CCL-Bench tools directory.
"""
import json
import math
from pathlib import Path
from statistics import mean, median


METRIC_INFO = {
    "client_system_tps": ("Client system TPS", "tok/s", True, "Successful output tokens / benchmark wall duration (includes prefill)."),
    "client_user_tps": ("Client user TPS", "tok/s", True, "Mean per-request decode rate: (output tokens - 1) / (E2E - TTFT), excluding outputs of length <= 1."),
    "client_request_throughput": ("Client requests/s", "req/s", True, "Successful requests / benchmark wall duration."),
    "client_ttft_ms": ("Client mean TTFT", "ms", False, "Mean successful-request time to first token measured at the client."),
    "client_tpot_ms": ("Client mean TPOT", "ms", False, "Mean (E2E - TTFT) / (output tokens - 1), for outputs longer than one token."),
    "client_e2e_ms": ("Client mean E2E", "ms", False, "Mean successful-request end-to-end latency, including prefill."),
}


def number(value, name, *, positive=False, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or (positive and value == 0) or (integer and int(value) != value)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} {'integer' if integer else 'number'}")
    return value


def close(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=0.01, abs_tol=1e-6):
        raise ValueError(f"{name} inconsistent: {actual} vs expected {expected}")


def measure(data, *, demo=False):
    """Return metrics + validation report. Strict mode requires all request arrays.

    vLLM detailed arrays contain all attempts; errors identifies failed requests.
    Demo mode only uses supplied aggregate fields and never computes user TPS
    from shortened arrays.
    """
    if not isinstance(data, dict):
        raise ValueError("benchmark result must be an object")
    if data.get("artifact_kind") == "demo_shortened_request_arrays" and not demo:
        raise ValueError("explicitly labelled DEMO result requires --demo")
    duration = number(data.get("duration"), "duration", positive=True)
    completed = number(data.get("completed"), "completed", positive=True, integer=True)
    failed = number(data.get("failed", 0), "failed", integer=True)
    n = completed + failed
    if "num_prompts" in data and number(data["num_prompts"], "num_prompts", integer=True) != n:
        raise ValueError("num_prompts != completed + failed")
    for key in ("total_input_tokens", "total_output_tokens"):
        number(data.get(key), key, integer=True)
    metrics = {
        "client_system_tps": data["total_output_tokens"] / duration,
        "client_request_throughput": completed / duration,
        "client_user_tps": None, "client_ttft_ms": None,
        "client_tpot_ms": None, "client_e2e_ms": None,
    }
    for key, expected in (("request_throughput", metrics["client_request_throughput"]),
                          ("output_throughput", metrics["client_system_tps"]),
                          ("total_token_throughput", (data["total_input_tokens"] + data["total_output_tokens"]) / duration)):
        if key in data:
            close(number(data[key], key), expected, key)
    if demo:
        for field, metric in (("mean_ttft_ms", "client_ttft_ms"), ("mean_tpot_ms", "client_tpot_ms")):
            if data.get(field) is not None:
                metrics[metric] = number(data[field], field)
        return metrics, {"demo": True, "warning": "DEMO: request arrays are shortened; aggregate fields only; user TPS and E2E unavailable.", "completed": completed, "failed": failed}

    required = ("input_lens", "output_lens", "ttfts", "latencies", "errors")
    for key in required:
        if not isinstance(data.get(key), list) or len(data[key]) != n:
            raise ValueError(f"{key} must contain {n} entries; shortened data requires --demo")
    if any(not isinstance(e, str) for e in data["errors"]):
        raise ValueError("errors must contain strings")
    success = [i for i, error in enumerate(data["errors"]) if not error]
    if len(success) != completed:
        raise ValueError("errors array disagrees with completed/failed counts")
    for key in required[:-1]:
        for value in data[key]:
            number(value, key, integer=key.endswith("lens"))
    for key, total in (("input_lens", "total_input_tokens"), ("output_lens", "total_output_tokens")):
        if sum(data[key][i] for i in success) != data[total]:
            raise ValueError(f"{total} disagrees with successful request arrays")
    tpots, rates = [], []
    for i in success:
        tokens, ttft, latency = data["output_lens"][i], data["ttfts"][i], data["latencies"][i]
        if tokens < 1 or latency < ttft:
            raise ValueError(f"invalid successful request {i}: output length or E2E < TTFT")
        if tokens > 1:
            decode = latency - ttft
            if decode <= 0:
                raise ValueError(f"nonpositive decode time for request {i}")
            tpots.append(decode / (tokens - 1))
            rates.append((tokens - 1) / decode)
    metrics.update(client_user_tps=mean(rates) if rates else None,
                   client_tpot_ms=1000 * mean(tpots) if tpots else None,
                   client_ttft_ms=1000 * mean(data["ttfts"][i] for i in success),
                   client_e2e_ms=1000 * mean(data["latencies"][i] for i in success))
    for key, metric in (("mean_ttft_ms", "client_ttft_ms"), ("mean_tpot_ms", "client_tpot_ms")):
        if data.get(key) is not None and metrics[metric] is not None:
            close(number(data[key], key), metrics[metric], key)
    return metrics, {"demo": False, "completed": completed, "failed": failed,
                     "decode_requests": len(rates), "request_arrays_validated": True}


def metric_cal(trace_dir, metric):
    """CCL-Bench adapter; raw benchmark and manifest are unambiguous filenames."""
    path = Path(trace_dir)
    manifest = json.loads((path / "manifest.json").read_text())
    data = json.loads((path / "benchmark.json").read_text())
    metrics, _ = measure(data, demo=manifest["validation"]["demo"])
    if metric not in METRIC_INFO:
        raise ValueError(f"unsupported serving metric: {metric}")
    return metrics[metric]


def pareto_front(points):
    """Maximize both axes; equal points are retained. Return original objects."""
    return [p for p in points if not any(
        q["user_tps"] >= p["user_tps"] and q["system_tps"] >= p["system_tps"]
        and (q["user_tps"] > p["user_tps"] or q["system_tps"] > p["system_tps"])
        for q in points)]


def median_points(rows):
    """Reduce repeated measurements at each concurrency without mixing runs."""
    groups = {}
    for row in rows:
        if row["validation"]["demo"]:
            continue
        m = row["metrics"]
        if m["client_user_tps"] is None:
            continue
        key = (row["run"], row["concurrency"])
        groups.setdefault(key, []).append(row)
    return [dict(run=run, concurrency=c, repeats=len(rs),
                 user_tps=median(r["metrics"]["client_user_tps"] for r in rs),
                 system_tps=median(r["metrics"]["client_system_tps"] for r in rs))
            for (run, c), rs in sorted(groups.items())]
