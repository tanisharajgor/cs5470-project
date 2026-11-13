# Expert-parallel MoE inference on GPUs and TPUs (CS 5470, Team 23)

How expert-parallel (EP) settings affect MoE inference throughput and latency, compared on Perlmutter A100s and Cloud TPU v6e. The primary model is Qwen3-30B-A3B.

## Layout
```
docs/run-gpu-workload.md    step-by-step GPU (Perlmutter) setup and run (start here)
docs/run-tpu-workload.md    step-by-step TPU (Cloud TPU v6e) setup and run
scripts/run_workload.py     one-command run (GPU or TPU): server, warm-up, concurrency sweep, logs
scripts/tune_moe.sh         tune vLLM's fused-MoE kernels (writes configs/moe_tuning/*.json)
configs/moe_tuning/         tuned MoE kernel configs, shared by the whole team
analysis/plot_throughput.py request throughput vs concurrency -> figures/
analysis/card_stats.py      input/output length stats for the workload description
workloads/sharegpt/         ShareGPT workload description (lengths, arrival pattern, flags)
trace_collection/           CCL-Bench workload cards (one YAML per run)
results/   (gitignored)     results/<platform>/<run>/{run_meta.json, c<C>_r<R>.json, logs/}
figures/   (gitignored)     generated plots
```

## Quick start (Perlmutter compute node or TPU VM, env active)
```bash
python scripts/run_workload.py --config C1 --concurrency 256 128 64 32 16                  # GPU
python scripts/run_workload.py --platform tpu --config C1 --concurrency 256 128 64 32 16   # TPU
python analysis/plot_throughput.py
```
Run `python scripts/run_workload.py --help` for all options (config, devices, model, workload, sweep).

## CCL-Bench analysis

Use the separate **Python 3.12 CPU analysis environment** to export full vLLM
results, preview them in the local CCL-Bench dashboard, and plot user/system TPS
Pareto frontiers. Follow [docs/ccl-bench.md](docs/ccl-bench.md) for setup and
commands. GPU dependencies remain separate in `requirements-gpu.txt`.

For separate GPU trace collection and follow-up analysis, use
[docs/gpu-profiling.md](docs/gpu-profiling.md). Serving integration supports
C1–C3; GPU profiling and C4–C6/multi-node support remain follow-up work.

`results/model.json` is an explicitly labelled **demo with shortened request
arrays**. It requires `--demo` export and is excluded from Pareto analysis.
