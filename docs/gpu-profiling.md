# GPU profiling follow-up (Perlmutter)

**Status: documented collection procedure, not GPU-tested by this integration.**
The commands follow vLLM 0.33.0's profiler API. They need validation on a compute
node with the team's actual vLLM/CUDA environment. Do not claim profiler metrics
or a successful capture until worker JSON files contain GPU kernel events.

This is separate from the validated [serving-results runbook](ccl-bench.md).
Existing unprofiled C1 results remain the throughput baseline. Profiling affects
timing, and the capped trace below samples execution rather than covering the
whole benchmark. It is not a substitute for the concurrency sweep.

## 1. Allocate and prepare

From a Perlmutter login node:

```bash
salloc -N 1 -C "gpu&hbm40g" --gpus-per-node 4 \
  -q interactive -t 01:00:00 -A m4999
```

Use the team's valid allocation account if it differs. On the assigned `nid...`
node, activate the same GPU environment used for the baseline, with its existing
scratch/cache variables (see [GPU setup](run-gpu-workload.md)). Do not create a
new environment or install CPU analysis dependencies into it.

```bash
cd "$PSCRATCH/cs5470-project"
which python
python -c 'import vllm; print(vllm.__version__)'
nvidia-smi
```

The remaining commands target vLLM 0.33.0. If that is not the installed version,
check its profiler fields before launching; do not upgrade a working baseline
environment solely for this runbook. Keep any other GPU server stopped before
loading another model on the same devices.

## 2. Start a C1 profiling server

Run all subsequent collection blocks in the same shell, so the directory and
server PID variables persist. Port 8010 is separate from the normal port 8000.

```bash
export CCL_PROFILE_DIR="$PSCRATCH/traces/C1-c16-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$CCL_PROFILE_DIR/torch"
nvidia-smi > "$CCL_PROFILE_DIR/nvidia-smi.txt"
python -c 'import vllm, torch; print("vllm", vllm.__version__); print("torch", torch.__version__); print("cuda", torch.version.cuda)' \
  > "$CCL_PROFILE_DIR/versions.txt"

vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 4 \
  --max-model-len 4096 --max-num-seqs 256 \
  --no-enable-prefix-caching --generation-config vllm \
  --port 8010 \
  --profiler-config "{
    \"profiler\": \"torch\",
    \"torch_profiler_dir\": \"$CCL_PROFILE_DIR/torch\",
    \"torch_profiler_use_gzip\": false,
    \"torch_profiler_with_stack\": false,
    \"ignore_frontend\": true,
    \"max_iterations\": 100
  }" \
  > "$CCL_PROFILE_DIR/server.log" 2>&1 &
CCL_SERVER_PID=$!
echo "Server PID: $CCL_SERVER_PID"
echo "Artifacts: $CCL_PROFILE_DIR"
tail -f "$CCL_PROFILE_DIR/server.log"
```

Wait for `Application startup complete`; Ctrl+C stops `tail`, leaving the
background server running. If startup fails, inspect the log and stop here.
No automatic fallback changes the profiler or model configuration.

The configuration disables frontend CPU tracing and stacks, writes uncompressed
worker JSON, and caps collection at 100 worker iterations. This is a first
capture to establish compatibility; assess which phases are represented before
choosing windows for final comparisons. Keep CUDA graphs configured as in the
baseline; graph replay can affect operator-level attribution.

## 3. Warm up, then capture

Verify health, then run an unprofiled warm-up:

```bash
curl -f http://127.0.0.1:8010/health
vllm bench serve \
  --backend vllm --model Qwen/Qwen3-30B-A3B \
  --endpoint /v1/completions --port 8010 \
  --dataset-name sharegpt \
  --dataset-path "$PSCRATCH/data/ShareGPT_V3_unfiltered_cleaned_split.json" \
  --num-prompts 50 --seed 0 --ignore-eos \
  --request-rate inf --max-concurrency 16 \
  2>&1 | tee "$CCL_PROFILE_DIR/warmup.log"
```

Only proceed if warm-up succeeded. Enable pipeline error reporting, then capture:

```bash
set -o pipefail
vllm bench serve \
  --backend vllm --model Qwen/Qwen3-30B-A3B \
  --endpoint /v1/completions --port 8010 \
  --dataset-name sharegpt \
  --dataset-path "$PSCRATCH/data/ShareGPT_V3_unfiltered_cleaned_split.json" \
  --num-prompts 32 --seed 0 --ignore-eos \
  --request-rate inf --max-concurrency 16 --profile \
  --save-result --save-detailed \
  --result-dir "$CCL_PROFILE_DIR" \
  --result-filename profiled_benchmark.json \
  2>&1 | tee "$CCL_PROFILE_DIR/client.log"
```

Allow the stop-profile call and trace flushing to finish. Save this command
sequence as `run.sh` with the artifacts, including any changes. The short
profiled benchmark is not named `c16_r0.json`, so normal sweep imports cannot
accidentally mix it into the unprofiled performance curve.

## 4. Verify capture and stop the server

```bash
python - <<'PY'
import json, os
from pathlib import Path
files = sorted((Path(os.environ["CCL_PROFILE_DIR"]) / "torch").rglob("*.json"))
if not files:
    raise SystemExit("No JSON traces: inspect server.log/client.log before continuing")
gpu_total = 0
for path in files:
    data = json.loads(path.read_text())
    events = data.get("traceEvents", [])
    kernels = [e for e in events if "kernel" in str(e.get("cat", "")).lower()]
    nccl = [e for e in kernels if "nccl" in str(e.get("name", "")).lower()]
    gpu_total += len(kernels)
    print(path.name, "events=", len(events), "GPU kernels=", len(kernels), "NCCL kernels=", len(nccl))
if not gpu_total:
    raise SystemExit("No GPU kernel events found; CPU-only traces are insufficient")
print("GPU events found. Inspect worker/rank coverage and the captured time window next.")
PY

echo "$CCL_PROFILE_DIR"
kill -TERM "$CCL_SERVER_PID"
wait "$CCL_SERVER_PID"
```

A nonzero `wait` status after termination can be expected. Stop the server even
if verification failed. Preserve logs for investigation. Inspect the worker
files for all four TP ranks; CPU-only traces or missing ranks must not be
reported as complete distributed coverage.

## 5. Transfer and try upstream metrics on the analysis host

Copy the entire printed directory to the laptop, keeping traces outside Git:

```bash
mkdir -p traces
# Replace account and path with the actual capture directory:
scp -r YOUR_NERSC_USERNAME@perlmutter.nersc.gov:/ABSOLUTE/CAPTURE/DIRECTORY traces/
```

The following preparatory example uses the separate Python 3.12 analysis
environment and existing CCL-Bench checkout. Replace the capture-directory name.
It creates a dedicated metric directory containing worker JSON and a card whose
source is profiler JSON, rather than serving benchmark JSON:

```bash
export CCL_LOCAL_CAPTURE=traces/YOUR_CAPTURE_DIRECTORY
.venv-analysis/bin/python - <<'PY'
import os, shutil
from pathlib import Path
import yaml
capture = Path(os.environ["CCL_LOCAL_CAPTURE"])
out = capture / "ccl-metrics"
out.mkdir(exist_ok=False)
files = list((capture / "torch").rglob("*.json"))
if not files:
    raise SystemExit("No worker JSON files found")
for index, path in enumerate(sorted(files)):
    shutil.copy2(path, out / f"worker_{index}_{path.name}")
card_path = next(Path("trace_collection").glob("*gpu*/*.yaml"))
card = yaml.safe_load(card_path.read_text())
card["description"] = "C1 TP4 ShareGPT, c=16; capped PyTorch worker profile, separate from serving sweep."
card["workload"]["model"]["iteration"] = 1
card["workload"]["data"].update(batch_size=None, input_len=None, output_len=None)
card["metric_source"] = {"traces": ["json"]}
(out / "workload.yaml").write_text(yaml.safe_dump(card, sort_keys=False))
print(out)
PY

.venv-analysis/bin/python ../ccl-bench/tools/main.py \
  --trace "$CCL_LOCAL_CAPTURE/ccl-metrics" --metric communication_fraction
.venv-analysis/bin/python ../ccl-bench/tools/main.py \
  --trace "$CCL_LOCAL_CAPTURE/ccl-metrics" --metric moe_fraction
.venv-analysis/bin/python ../ccl-bench/tools/main.py \
  --trace "$CCL_LOCAL_CAPTURE/ccl-metrics" --metric compute_comm_overlap
```

These upstream tool invocations are a follow-up validation step, not a promise
of available values. Review the card's remaining hardware/model fields, the
tool's rank-sampling rules, kernel classification, phase ranges, units, and
missing-value behavior. `None`, `-1`, parse failures, or missing ranges do not
establish zero overhead. Do not put profiler-derived values into `client_*`
columns or label sampled kernel-time fractions as whole-run wall-time fractions.
Adding these metrics to a generated dashboard still needs a separate trace-aware
configuration/integration; `preview_ccl.py` only registers serving metrics.

## Follow-up experiments and limits

After one usable capture, repeat with C2 (`--data-parallel-size 4
--enable-expert-parallel`) or C3 (`--tensor-parallel-size 4
--enable-expert-parallel`), a new output directory, and the correct card/layout.
Warm up each server independently and keep model, workload, and capture-window
policy comparable. C1 is the TP-only baseline; EP dispatch/combine comparisons
require an EP-enabled configuration as well.

PyTorch traces can support a subset of kernel/communication metrics. Some tools
need Nsight Systems SQLite exports or additional annotations instead. Expert
token histograms, padding, and per-layer dispatch/combine are not guaranteed by
this capture. No GPU metric should be claimed merely because its dashboard
column exists.

References: [vLLM profiler configuration at v0.33.0](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/profiler.py),
[vLLM profiling guide](https://docs.vllm.ai/en/latest/contributing/profiling/),
[CCL-Bench metric interface and compatibility](https://github.com/cornell-sysphotonics/ccl-bench/blob/main/tools/README.md).
