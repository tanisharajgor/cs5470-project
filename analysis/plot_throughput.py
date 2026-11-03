"""Plot request throughput (req/s) vs max concurrency from `vllm bench serve` results.

Reads results/<platform>/<run>/c<concurrency>_r<rep>.json (written by scripts/run_workload.py).
Each <platform>/<run> folder becomes one line; repeats at the same concurrency are
reduced to their median.

Usage:
    python analysis/plot_throughput.py                       # <repo>/results -> <repo>/figures
    python analysis/plot_throughput.py --results some/dir --out some/other/dir
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

# Categorical slots in fixed order (validated reference palette); line style follows
# the platform so GPU vs TPU is never color-alone.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PLATFORM_STYLE = {"gpu": dict(ls="-", marker="o"), "tpu": dict(ls="--", marker="s")}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ROOT = Path(__file__).resolve().parents[1]


def load(results_dir):
    """Return {(platform, run): [(concurrency, median req/s), ...]} sorted by concurrency."""
    runs = defaultdict(lambda: defaultdict(list))
    for f in Path(results_dir).glob("*/*/c*_r*.json"):
        m = re.fullmatch(r"c(\d+)_r\d+\.json", f.name)
        if m:
            platform, run = f.parts[-3], f.parts[-2]
            runs[(platform, run)][int(m.group(1))].append(json.loads(f.read_text())["request_throughput"])
    return {key: [(c, median(v)) for c, v in sorted(by_c.items())] for key, by_c in sorted(runs.items())}


def plot(data, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, ((platform, run), points) in enumerate(data.items()):
        ax.plot([c for c, _ in points], [r for _, r in points],
                color=SERIES[i % len(SERIES)], linewidth=2, markersize=7,
                markeredgecolor="white", markeredgewidth=1.5,
                label=f"{run} ({platform.upper()})", **PLATFORM_STYLE.get(platform, dict(ls=":", marker="^")))

    concurrencies = sorted({c for points in data.values() for c, _ in points})
    ax.set_xscale("log", base=2)
    ax.set_xticks(concurrencies, [str(c) for c in concurrencies])
    ax.minorticks_off()
    ax.set_xlabel("Max concurrency", color=TEXT)
    ax.set_ylabel("Request throughput (req/s)", color=TEXT)
    ax.set_ylim(bottom=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    title = "Request throughput vs concurrency"
    if len(data) > 1:
        ax.legend(frameon=False, fontsize=9)
    else:
        (platform, run), = data
        title += f"\n{run} on {platform.upper()}"
    ax.set_title(title, loc="left", fontsize=11, color=TEXT)

    fig.tight_layout()
    path = out / "request_throughput_vs_concurrency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=ROOT / "results")
    p.add_argument("--out", default=ROOT / "figures")
    args = p.parse_args()

    data = load(args.results)
    if not data:
        raise SystemExit(f"no c<C>_r<R>.json files under {args.results}/<platform>/<run>/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for (platform, run), points in data.items():
        print(f"{platform}/{run}: " + ", ".join(f"c={c}: {r:.2f} req/s" for c, r in points))
    print(f"wrote {plot(data, out)}")


if __name__ == "__main__":
    main()
