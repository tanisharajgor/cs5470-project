**Applicable Settings**
Workload: ShareGPT
Devices: 4
Configuration: C1 (Baseline) - Tensor parallel only, no EP

Workload details (filtering, arrival pattern, sweep): [workloads/sharegpt/WORKLOAD_CARD.md](workloads/sharegpt/WORKLOAD_CARD.md)

**Steps to Run a Workload**

## GPU (Perlmutter, 4x A100)

### 1. One-time setup (login node)
```bash
ssh <user>@perlmutter.nersc.gov
module load conda
conda create -p /global/common/software/<project>/vllm-env python=3.12 -y
conda activate /global/common/software/<project>/vllm-env
pip install vllm            # record the version; everyone uses the same one

export HF_HOME=$SCRATCH/hf  # add to ~/.bashrc
huggingface-cli download Qwen/Qwen3-30B-A3B   # ~61 GB

mkdir -p $SCRATCH/data && cd $SCRATCH/data
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
shasum -a 256 ShareGPT_V3_unfiltered_cleaned_split.json   # record in WORKLOAD_CARD.md
```

### 2. Allocate a node
```bash
salloc -N 1 -C gpu -q interactive -t 04:00:00 -A <project> --gpus-per-node 4
```

### 3. Record versions (fill the TBDs in the YAML)
```bash
nvidia-smi | head -4
python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.nccl.version())"
```

### 4. Start the server
```bash
vllm serve Qwen/Qwen3-30B-A3B --tensor-parallel-size 4 \
  --max-model-len 4096 --max-num-seqs 256 --no-enable-prefix-caching \
  --port 8000 > server_c1.log 2>&1 &
until curl -s localhost:8000/health >/dev/null; do sleep 5; done; echo ready

# sanity check
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-30B-A3B","prompt":"Hello, my name is","max_tokens":20}'
```

### 5. Warm-up (discard results)
```bash
vllm bench serve --model Qwen/Qwen3-30B-A3B --dataset-name sharegpt \
  --dataset-path $SCRATCH/data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 50 --max-concurrency 16 --ignore-eos
```

### 6. Concurrency sweep
```bash
OUT=results/gpu/C1; mkdir -p $OUT
for C in 1 2 4 8 16 32 64 128 256; do
  for REP in 0; do   # first pass: 1 rep; final runs: 0 1 2
    vllm bench serve --backend vllm --model Qwen/Qwen3-30B-A3B --endpoint /v1/completions \
      --dataset-name sharegpt --dataset-path $SCRATCH/data/ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 500 --seed 0 --ignore-eos \
      --request-rate inf --max-concurrency $C \
      --save-result --save-detailed --result-dir $OUT --result-filename c${C}_r${REP}.json
  done
done
```
C=1 and C=2 take the longest (~15+ min each). If the sweep doesn't fit in the 4 h interactive limit, run it with `sbatch`.

### 7. Profiling run (separate from the sweep; the profiler adds overhead)
1. Stop the server and restart it with the profiler on:
   - older vLLM: `VLLM_TORCH_PROFILER_DIR=$SCRATCH/traces/gpu_c1 vllm serve ...`
   - newer vLLM: `--profiler-config` flag (check `vllm serve --help | grep -i profil`)
2. Short run: same bench command with `--num-prompts 32 --max-concurrency 16 --profile`
3. In C1, expect NCCL all-reduce kernels in every MoE layer (no all-to-all until C2+).

## TPU (Cloud TPU v6e-4)

### 1. Create the VM
```bash
gcloud compute tpus tpu-vm create c1-tpu --zone=<v6e zone, e.g. us-east5-b> \
  --accelerator-type=v6e-4 --version=v2-alpha-tpuv6e
gcloud compute tpus tpu-vm ssh c1-tpu --zone=<zone>
```

### 2. Install + download
- `pip install vllm-tpu` (or the `vllm/vllm-tpu` Docker image); pin the version
- Download the same model and dataset as the GPU steps
- Check `df -h`; the boot disk may be too small for the 61 GB model, attach a larger disk if so

### 3. Record versions
`pip list | grep -iE "vllm|tpu|jax"` and the TPU runtime version

### 4–6. Server, warm-up, sweep
Same commands as GPU steps 4–6, with `OUT=results/tpu/C1`. First startup and warm-up are much slower (XLA compilation), so don't skip the warm-up.

### 7. Profiling
tpu-inference writes a JAX/XLA trace; view in TensorBoard / xprof. Check the tpu-inference docs for the env var.

### 8. Delete the VM (it bills while it exists)
```bash
gcloud compute tpus tpu-vm delete c1-tpu --zone=<zone>
```

## After the run
1. `python workloads/sharegpt/card_stats.py results/gpu/C1/c16_r0.json`, then paste the table into WORKLOAD_CARD.md section 3 and fill `input_len` / `output_len` in the YAMLs
2. Fill the remaining TBDs in the C1 YAMLs (versions, driver, `trace_url`)
3. Upload traces to shared storage and put the link in `trace_url`
4. Save the exact commands you ran as `run.sh` in each C1 folder under `trace_collection/`
5. Commit YAMLs + scripts (results/traces are gitignored)
6. Note anything you had to change below, so C2–C6 follow the same procedure

## Notes / issues encountered
- 
