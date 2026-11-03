"""Compute ShareGPT workload-card length stats from a `vllm bench serve --save-detailed` result JSON.

Usage: python analysis/card_stats.py results/gpu/<run>/c16_r0.json
Prints a markdown table for workloads/sharegpt/README.md section 3 and writes
workloads/sharegpt/length_hist.png.
"""
import json
import sys
from pathlib import Path

import numpy as np


def row(name, x):
    p = np.percentile(x, [50, 90, 99])
    return f"| {name} | {x.mean():.0f} | {p[0]:.0f} | {p[1]:.0f} | {p[2]:.0f} | {x.max()} |"


def main(path):
    res = json.loads(Path(path).read_text())
    inp = np.array(res["input_lens"])
    out = np.array(res["output_lens"])

    print(f"n = {len(inp)} requests\n")
    print("| Tokens | mean | p50 | p90 | p99 | max |")
    print("|---|---|---|---|---|---|")
    print(row("Input (prompt)", inp))
    print(row("Output", out))
    print(row("Total (in + out)", inp + out))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, x, label in zip(axes, (inp, out), ("Input tokens", "Output tokens")):
        ax.hist(x, bins=50)
        ax.set_xlabel(label)
        ax.set_ylabel("Requests")
    fig.suptitle(f"ShareGPT, Qwen3 tokenizer, n={len(inp)}")
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parents[1] / "workloads" / "sharegpt" / "length_hist.png", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1])
