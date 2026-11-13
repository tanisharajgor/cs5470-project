"""Plot per-run median points and nondominated user/system output TPS.

Only combine compatible workload/model/phase and equal hardware budget exports.
Demo rows and one-token prefill runs have no user decode rate and are excluded.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from serving_metrics import median_points, pareto_front


def plot(exports, out):
    rows = []
    for directory in exports:
        rows.extend(json.loads((Path(directory) / "summary.json").read_text())["rows"])
    if len({json.dumps(r["comparison"], sort_keys=True) for r in rows}) > 1:
        raise ValueError("incompatible comparisons: use one model/workload/precision/load at equal device count")
    seen = set()
    for row in rows:
        key = (row["run"], row["concurrency"], row["repeat"])
        if key in seen:
            raise ValueError(f"duplicate measurement: {key}")
        seen.add(key)
    points = median_points(rows)
    if not points:
        raise ValueError("no validated decode points; DEMO and one-token outputs are excluded")
    front = pareto_front(points)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in sorted({p["run"] for p in points}):
        ps = sorted((p for p in points if p["run"] == run), key=lambda p: p["concurrency"])
        ax.plot([p["user_tps"] for p in ps], [p["system_tps"] for p in ps], "o-", label=run, alpha=.8)
        for p in ps:
            ax.annotate(f'c={p["concurrency"]}', (p["user_tps"], p["system_tps"]), xytext=(4, 5), textcoords="offset points", fontsize=8)
    frontier = sorted(front, key=lambda p: p["user_tps"])
    ax.plot([p["user_tps"] for p in frontier], [p["system_tps"] for p in frontier], "k--", label="Pareto frontier")
    ax.set(xlabel="Mean user decode TPS (output tok/s)", ylabel="System output TPS (tok/s)",
           title="Client-measured serving throughput — medians across repeats", xlim=(0, None), ylim=(0, None))
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(out / f"pareto.{extension}", dpi=160)
    plt.close(fig)
    with (out / "pareto.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "concurrency", "repeats", "user_tps", "system_tps", "pareto"])
        writer.writeheader()
        writer.writerows({**p, "pareto": p in front} for p in points)
    return points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exports", type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    args = p.parse_args()
    try:
        points = plot(args.exports, args.out)
    except (ValueError, OSError, KeyError) as e:
        p.exit(2, f"plot failed: {e}\n")
    print(f"Plotted {len(points)} points: {args.out}/pareto.{{png,pdf,csv}}")


if __name__ == "__main__":
    main()
