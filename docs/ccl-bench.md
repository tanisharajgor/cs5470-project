# CCL-Bench serving integration

## Handoff status and scope

The serving-results integration supports the current single-node C1/C2/C3 vLLM
layouts. Local checks cover complete benchmark export, CCL-Bench data generation,
dashboard rendering, and Pareto plots. The original C1 sweep consists of nine
points (`c1_r0.json` through `c256_r0.json`), with one repeat each. Its files are
local artifacts and are not committed.

Validation on 2026-11-13: all 13 local regression tests passed; the full original
C1 files passed strict export, generated all 54 dashboard values (9 rows × 6
metrics) through CCL-Bench, and produced PNG/PDF/CSV Pareto outputs. The demo
dashboard also passed the headless Chrome smoke check. C2/C3 layouts are covered
by local fixtures; they have not been validated here against real C2/C3 runs.

The separate [GPU profiling runbook](gpu-profiling.md) documents how to collect
traces and try compatible upstream metric tools. **GPU trace collection has not
been run or validated on Perlmutter as part of this integration.** Trace-derived
metrics are not registered by the serving preview. C4–C6/multi-node layouts and
custom topology overrides require further implementation. Expert histograms,
padding overhead, and per-layer dispatch/combine require additional evidence or
instrumentation; the current exporter does not generate them.

Run GPU experiments with `scripts/run_workload.py`, then copy each complete run
folder (`run_meta.json` and `c<C>_r<R>.json`) to the analysis host. No GPU, torch,
vLLM, or profiler trace is required for serving analysis.

## Separate Python 3.12 environments

Keep the GPU setup in [run-gpu-workload.md](run-gpu-workload.md); its wheel
dependency is also recorded in `requirements-gpu.txt`. Create a CPU environment
from this repo root, without installing GPU or CCL-Bench root requirements:

```bash
python3.12 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r requirements-analysis.txt
```

Alternatively, use `uv python install 3.12` and
`uv venv --python 3.12 --seed .venv-analysis`. `.python-version` records the version.

## Retrieve the original C1 results

On Perlmutter, no GPU allocation is needed to locate existing files:

```bash
cd "$PSCRATCH/cs5470-project"
ls -lh results/gpu/C1/
realpath results/gpu/C1
```

On the laptop/analysis host, from this repository root, set the account and
absolute path printed above. This example uses the GPU lead's original run:

```bash
mkdir -p results/gpu
scp -r anishkar@perlmutter.nersc.gov:/pscratch/sd/a/anishkar/cs5470-project/results/gpu/C1 \
  results/gpu/
```

Use your own account/path for another run. Copy the full JSON files rather than
terminal summaries or shortened excerpts. The old C1 folder has no
`run_meta.json`, so supply explicit layout flags:

```bash
.venv-analysis/bin/python analysis/export_ccl.py results/gpu/C1 \
  --config C1 --platform gpu --devices 4 \
  --card trace_collection/qwen3-30b-a3b-vllm-gpu-tp4-ep1-sharegpt-perlmutter-group_23/qwen3-30b-a3b-vllm-gpu-tp4-ep1-sharegpt-perlmutter-group_23.yaml \
  --out build/ccl-C1-original
.venv-analysis/bin/python analysis/preview_ccl.py \
  build/ccl-C1-original --out build/ccl-preview-C1-original
.venv-analysis/bin/python analysis/plot_pareto.py \
  build/ccl-C1-original --out figures/C1-original
.venv-analysis/bin/python -m http.server 8082 --bind 127.0.0.1 \
  --directory build/ccl-preview-C1-original
```

Open <http://127.0.0.1:8082>. Expect nine rows; `c16_smoke.json` and `summary.csv`
are automatically excluded. If an output directory already exists, reuse the
existing export/preview or choose a new output name. Ctrl+C stops the HTTP
server. Validation failures should be investigated before rerunning a sweep.

## Export a complete run

```bash
.venv-analysis/bin/python analysis/export_ccl.py \
  results/gpu/C1-sharegpt-qwen3-30b-a3b-4gpu \
  --card trace_collection/qwen3-30b-a3b-vllm-gpu-tp4-ep1-sharegpt-perlmutter-group_23/qwen3-30b-a3b-vllm-gpu-tp4-ep1-sharegpt-perlmutter-group_23.yaml \
  --out build/ccl-C1
```

Use your actual run directory name. Config, platform, device count, model, and
workload come from `run_meta.json`. For older runs without it, explicitly add
`--config C1 --platform gpu --devices 4`; dataset metadata comes from the card.
For C2/C3, use their run directory and an appropriate card. The exporter sets
TP/DP/EP for these three layouts and rejects topology overrides in
`extra_server_args` to avoid ambiguous configuration claims.

Review the card's hardware model, precision, network, framework versions, and
dataset before exporting. Those descriptions remain supplied by the card; each
manifest also preserves the original run metadata. Client concurrency is stored
separately from dynamic server batch size, which is left unspecified.

Each point gets an artifact directory with the unchanged `benchmark.json`,
`workload.yaml`, and `manifest.json` containing source SHA-256, validation,
metadata, and metrics. `summary.json` feeds plots; `benchmark_config.json` feeds
the dashboard. The whole sweep is validated before output is published.
Existing output directories are never overwritten; choose a fresh name.

Strict validation checks finite nonnegative values, success/failure counts,
request-array lengths, exact successful token totals, filename concurrency,
and consistency of reported aggregates (1% tolerance). Errors identify failed
requests, which are excluded from latency averages. Missing detailed arrays
fail clearly rather than producing partial request-derived metrics.

## Local CCL-Bench dashboard

Clone CCL-Bench beside this repo if it is absent:

```bash
git clone https://github.com/cornell-sysphotonics/ccl-bench.git ../ccl-bench
.venv-analysis/bin/python analysis/preview_ccl.py build/ccl-C1 --out build/ccl-preview-C1
.venv-analysis/bin/python -m http.server 8081 --bind 127.0.0.1 --directory build/ccl-preview-C1
```

Open <http://127.0.0.1:8081>. Select another checkout with `--ccl-repo PATH`.
The preview copies the upstream dashboard, generator, and dispatcher into
`build/`, registers six `client_*` metrics, runs the real generator, and checks
the resulting values against the export. Only this export's rows are included.
The sibling checkout remains unchanged. `preview_provenance.json` records the
upstream commit; initially tested with
`1d257f07b35b262d0e36d4ee8d5345f8abe2e647`. A changed dispatcher anchor fails
clearly. Metrics have neutral styling because universal thresholds would be
misleading.

For exact reproduction, on a fresh CCL-Bench clone use
`git -C ../ccl-bench checkout --detach 1d257f07b35b262d0e36d4ee8d5345f8abe2e647`
before generating the preview. Preserve any existing local changes in a shared
checkout; use a separate clone and `--ccl-repo` when necessary.

The commands generate a local preview and do not update the public leaderboard
or publish traces. The staged `tools/client_serving.py`, dispatcher registration,
metric definitions, and lightweight cards are available for a future upstream
review.

To browse other contributors' runs, use <https://cclbench.ai/>. Filter by model,
framework, or hardware, inspect the information button, and select rows to
compare. The local serving preview intentionally contains only the selected
export. Similar model names do not establish equal workloads or hardware budgets.

## Client metric definitions

vLLM detailed times are seconds; display values convert to milliseconds.
The formulas follow the [vLLM benchmark implementation](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/benchmarks/serve.py).

| Metric | Definition |
| --- | --- |
| `client_system_tps` | Successful output tokens / benchmark wall duration, including prefill and scheduling. |
| `client_user_tps` | Mean per-request `(output_len - 1) / (latency - TTFT)`. |
| `client_request_throughput` | Completed requests / benchmark wall duration. |
| `client_ttft_ms` | Mean successful TTFT × 1000. |
| `client_tpot_ms` | Mean `(latency - TTFT) / (output_len - 1)` × 1000. |
| `client_e2e_ms` | Mean successful end-to-end latency × 1000. |

One-token requests have no decode TPOT or user TPS. Mean user rate is not the
reciprocal of mean TPOT. System TPS counts output tokens only. ITLs can represent
streaming chunks, so calculations use output counts and E2E/TTFT instead of
assuming one token per chunk. These metrics are distinct from CCL-Bench's
trace-derived `ttft`/`tpot`. Dispatch/combine, communication, padding, imbalance,
and overlap still require separate profiler artifacts and tools.

## Pareto plots

```bash
.venv-analysis/bin/python analysis/plot_pareto.py build/ccl-C1 --out figures/C1
# After exporting C2/C3, compare compatible runs:
.venv-analysis/bin/python analysis/plot_pareto.py \
  build/ccl-C1 build/ccl-C2 build/ccl-C3 --out figures/sharegpt-4devices
```

Outputs: `pareto.png`, `pareto.pdf`, and `pareto.csv`. Points are medians of
repeat-level measurements at each concurrency, grouped by run. The CSV records
repeat counts and frontier membership. Both axes are maximized; ties remain on
the frontier. Lines connect each run's sweep, while the dashed frontier spans
all plotted median points. Duplicate measurements and mismatched
model/workload/precision/load/device counts are rejected. Equal device count
is only a chip-count budget assumption: verify comparable workload sampling,
model settings, warm-up, and hardware budget for GPU/TPU comparisons.

## Explicit demo

`results/model.json` contains only three request entries against 500 completed
requests. It is labelled in the file and rejected by strict export. Preview the
integration using its original aggregate fields:

```bash
.venv-analysis/bin/python analysis/export_ccl.py results/model.json \
  --demo --config C1 --platform gpu --devices 4 \
  --card trace_collection/qwen3-30b-a3b-vllm-gpu-tp4-ep1-sharegpt-perlmutter-group_23/qwen3-30b-a3b-vllm-gpu-tp4-ep1-sharegpt-perlmutter-group_23.yaml \
  --out build/ccl-demo
.venv-analysis/bin/python analysis/preview_ccl.py build/ccl-demo --out build/ccl-preview-demo
.venv-analysis/bin/python -m http.server 8081 --bind 127.0.0.1 --directory build/ccl-preview-demo
```

All artifacts and the dashboard banner say DEMO. No user TPS or E2E is computed
from shortened arrays; Pareto plots exclude every demo row. Use the original
full C1 run directory for experimental plots.

## Local tests

```bash
.venv-analysis/bin/python -m unittest discover -s tests -v
.venv-analysis/bin/python scripts/run_workload.py --config C1 --dry-run
```

Tests cover malformed measurements, failures, one-token outputs, demo handling,
atomic export, config layouts, repeat medians, dominance, plot artifacts, and the
real CCL-Bench generator/dispatcher. The latter skips if the sibling checkout
is absent. Local checks do not load a GPU model.

An optional browser smoke check uses a separate environment and local Chrome:

```bash
python3.12 -m venv .venv-browser
.venv-browser/bin/python -m pip install playwright==1.58.0
# With the demo preview server running:
.venv-browser/bin/python tests/check_dashboard_browser.py
```

It checks the rendered demo row, client metric labels, unavailable values, and
JavaScript errors, then writes `build/ccl-dashboard-demo.png`.

## Troubleshooting and sharing

| Symptom | Action |
| --- | --- |
| Output exists | Choose a fresh `--out`, or serve the already generated preview. |
| Request arrays missing/shortened | Retrieve the original `--save-detailed` files. `--demo` is only for explicitly labelled examples. |
| Missing run metadata | Supply `--config`, `--platform`, and `--devices`; review the source card. |
| GPU/TPU card mismatch | Select the card for the measured hardware. |
| Unsupported dispatcher version | Use the tested CCL-Bench commit or adapt/test the registration before proceeding. |
| Port in use | Select another HTTP-server port; GPU serving and the static dashboard use different servers. |
| No Pareto points | Demo rows and requests without decode tokens cannot provide a user decode rate. |

Commit source, requirements, tests, and runbooks. Keep `results/gpu/`, `build/`,
`figures/`, virtual environments, and profiler traces out of Git. Share traces
through the team's artifact storage and record a link plus exact profiler/run
configuration in the workload card. The tracked `results/model.json` is the
explicitly labelled shortened demo exception.
