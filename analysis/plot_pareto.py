"""Plot the user-TPS vs system-TPS Pareto frontier and latency curves from `vllm bench serve` results.

Reads results/<platform>/<config>/c<concurrency>_r<rep>.json (written by the sweep in
run-workload.md step 6). Repeats at the same concurrency are reduced to their median.

Usage:
    python analysis/plot_pareto.py                      # all results under ./results
    python analysis/plot_pareto.py --results results --out figures

Writes:
    <out>/pareto.png      per-user TPS (1000 / mean TPOT) vs system TPS (output tok/s)
    <out>/latency.png     TTFT and TPOT vs concurrency (two panels, shared x)
    <out>/summary.csv     one row per (platform, config, concurrency)
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

# Categorical slots in fixed order (validated reference palette); color follows the config,
# line style follows the platform so GPU vs TPU is never color-alone.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CONFIG_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6"]
PLATFORM_STYLE = {"gpu": dict(ls="-", marker="o"), "tpu": dict(ls="--", marker="s")}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

FIELDS = {
    "system_tps": "output_throughput",
    "mean_tpot_ms": "mean_tpot_ms",
    "p99_tpot_ms": "p99_tpot_ms",
    "mean_ttft_ms": "mean_ttft_ms",
    "p99_ttft_ms": "p99_ttft_ms",
}


def load(results_dir):
    """Return {(platform, config): [row, ...]} sorted by concurrency."""
    runs = defaultdict(lambda: defaultdict(list))
    for f in Path(results_dir).glob("*/*/c*_r*.json"):
        m = re.match(r"c(\d+)_r\d+\.json", f.name)
        if not m:
            continue
        platform, config = f.parts[-3], f.parts[-2]
        runs[(platform, config)][int(m.group(1))].append(json.loads(f.read_text()))

    out = {}
    for key, by_c in runs.items():
        rows = []
        for c, reps in sorted(by_c.items()):
            row = {"platform": key[0], "config": key[1], "concurrency": c, "reps": len(reps)}
            for name, field in FIELDS.items():
                row[name] = median(r[field] for r in reps)
            row["user_tps"] = 1000.0 / row["mean_tpot_ms"]
            rows.append(row)
        out[key] = rows
    return out


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def series_kwargs(platform, config):
    color = SERIES[CONFIG_ORDER.index(config) % len(SERIES)] if config in CONFIG_ORDER else MUTED
    st = PLATFORM_STYLE.get(platform, dict(ls=":", marker="^"))
    return dict(color=color, linewidth=2, markersize=7, markeredgecolor="white",
                markeredgewidth=1.5, label=f"{config} ({platform.upper()})", **st)


def sort_key(item):
    (platform, config), _ = item
    return (CONFIG_ORDER.index(config) if config in CONFIG_ORDER else 99, platform)


def plot_pareto(data, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for (platform, config), rows in sorted(data.items(), key=sort_key):
        xs = [r["system_tps"] for r in rows]
        ys = [r["user_tps"] for r in rows]
        ax.plot(xs, ys, **series_kwargs(platform, config))
        # label only the ends of each curve with their concurrency
        for r in (rows[0], rows[-1]) if len(rows) > 1 else rows:
            ax.annotate(f"c={r['concurrency']}", (r["system_tps"], r["user_tps"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8, color=MUTED)
    ax.set_xlabel("System TPS (output tok/s)")
    ax.set_ylabel("Per-user TPS (1000 / mean TPOT)")
    # headroom so end-of-curve labels aren't clipped
    all_rows = [r for rows in data.values() for r in rows]
    ax.set_xlim(0, 1.15 * max(r["system_tps"] for r in all_rows))
    ax.set_ylim(0, 1.15 * max(r["user_tps"] for r in all_rows))
    style_axes(ax)
    if len(data) > 1:
        ax.legend(frameon=False, fontsize=9)
    title = "Qwen3-30B-A3B, ShareGPT: user TPS vs system TPS"
    if len(data) == 1:
        (platform, config), = data
        title += f"\n{config} on {platform.upper()}"
    ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    fig.tight_layout()
    fig.savefig(out / "pareto.png", dpi=150)
    plt.close(fig)


def plot_latency(data, out):
    fig, (ax_ttft, ax_tpot) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for (platform, config), rows in sorted(data.items(), key=sort_key):
        cs = [r["concurrency"] for r in rows]
        kw = series_kwargs(platform, config)
        ax_ttft.plot(cs, [r["mean_ttft_ms"] for r in rows], **kw)
        ax_tpot.plot(cs, [r["mean_tpot_ms"] for r in rows], **kw)
    concurrencies = sorted({r["concurrency"] for rows in data.values() for r in rows})
    for ax, label in ((ax_ttft, "Mean TTFT (ms)"), (ax_tpot, "Mean TPOT (ms)")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(concurrencies, [str(c) for c in concurrencies])
        ax.minorticks_off()
        ax.set_xlabel("Max concurrency")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        style_axes(ax)
    if len(data) > 1:
        ax_tpot.legend(frameon=False, fontsize=9)
    fig.suptitle("Latency vs concurrency", x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.tight_layout()
    fig.savefig(out / "latency.png", dpi=150)
    plt.close(fig)


def write_summary(data, out):
    cols = ["platform", "config", "concurrency", "reps", "system_tps", "user_tps",
            "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms"]
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for _, rows in sorted(data.items(), key=sort_key):
            for r in rows:
                w.writerow({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    args = p.parse_args()

    data = load(args.results)
    if not data:
        raise SystemExit(f"no c<C>_r<R>.json files under {args.results}/<platform>/<config>/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_pareto(data, out)
    plot_latency(data, out)
    write_summary(data, out)
    for key, rows in sorted(data.items(), key=sort_key):
        print(f"{key[1]} {key[0]}: concurrency {[r['concurrency'] for r in rows]}")
    print(f"wrote {out}/pareto.png, latency.png, summary.csv")


if __name__ == "__main__":
    main()
