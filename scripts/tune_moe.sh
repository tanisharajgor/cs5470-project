#!/bin/bash
# Tune vLLM's fused-MoE Triton kernel for Qwen3-30B-A3B on A100-40GB.
# Writes one JSON per expert layout into configs/moe_tuning/; vLLM picks them up via
# VLLM_TUNED_CONFIG_FOLDER (set in the conda env, see docs/run-gpu-workload.md step 1c).
#
# Run on a 4-GPU node (salloc -C "gpu&hbm40g"), from anywhere:
#   bash scripts/tune_moe.sh            # all three layouts
#   bash scripts/tune_moe.sh c1         # just one: c1 | ep4 | ep8
set -euo pipefail

MODEL=Qwen/Qwen3-30B-A3B
OUT=$(cd "$(dirname "$0")/.." && pwd)/configs/moe_tuning
BENCH=$WORKSPACE/vllm/benchmarks/kernels/benchmark_moe.py

# --tp-size here only sets how experts are sharded, so EP=8 shapes can be tuned on 4 GPUs.
declare -A LAYOUTS=(
  [c1]="--tp-size 4"                            # C1:         E=128, N=192
  [ep4]="--tp-size 4 --enable-expert-parallel"  # C2, C3, C5: E=32,  N=768
  [ep8]="--tp-size 8 --enable-expert-parallel"  # C4, C6:     E=16,  N=768
)

python -c "import ray" 2>/dev/null || python -m pip install ray

NAMES=("$@")
[ ${#NAMES[@]} -eq 0 ] && NAMES=(c1 ep4 ep8)

for n in "${NAMES[@]}"; do
  [ -n "${LAYOUTS[$n]:-}" ] || { echo "unknown layout '$n' (use c1, ep4, ep8)"; exit 1; }
  echo "=== tuning $n: ${LAYOUTS[$n]}"
  python "$BENCH" --model $MODEL ${LAYOUTS[$n]} --tune --save-dir "$OUT"
done

ls -1 "$OUT"/*.json
