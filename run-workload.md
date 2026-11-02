**Applicable Settings**
Workload: ShareGPT
Devices: 4
Configuration: C1 (Baseline) - Tensor parallel only, no EP

Workload details (filtering, arrival pattern, sweep): [workloads/sharegpt/WORKLOAD_CARD.md](workloads/sharegpt/WORKLOAD_CARD.md)

**Steps to Run a Workload**

Run all commands from the repo root; results go to `./results/` (gitignored).

## GPU (Perlmutter, 4x A100 40GB)

### 1. One-time setup (login node)
Don't install anything into the default `nersc-python` env; it's read-only, and pip either fails with `Permission denied` or silently installs to `~/.local`.

**1a. Workspace: our repo and the vLLM source side by side**
```
~/assignments/project-home/
├── cs5470-project/   # this repo
└── vllm/             # vLLM source at the pinned version (v0.33.0)
```
```bash
mkdir -p ~/assignments/project-home && cd ~/assignments/project-home
git clone https://github.com/tanisharajgor/cs5470-project.git
git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout v0.33.0 && cd ..
ls   # cs5470-project  vllm
```

**1b. Check for leftovers in `~/.local`** (these override packages in your env)
```bash
ls ~/.local/lib/ 2>/dev/null
```
If anything is listed that you didn't install on purpose, remove it with `rm -rf ~/.local/lib/python3.*`. Also clear any pip cache in home: `rm -rf ~/.cache/pip`.

**1c. Create and configure the env**
```bash
module load conda
conda create -y --prefix $PSCRATCH/cs5470-ep python=3.12
conda activate $PSCRATCH/cs5470-ep
# ignore ~/.local, keep model + pip cache on scratch (reactivate to apply)
conda env config vars set PYTHONNOUSERSITE=1 HF_HOME=$PSCRATCH/huggingface PIP_CACHE_DIR=$PSCRATCH/pip-cache
conda deactivate && conda activate $PSCRATCH/cs5470-ep
which python                 # must be $PSCRATCH/cs5470-ep/bin/python
```
If `conda create` errors (e.g. ends with `Example: CONDA_NO_PLUGINS=true conda install <package>`), retry with `CONDA_NO_PLUGINS=true conda create ...`. If `which python` doesn't point into the env, don't continue: ask in the group chat.

**1d. Install vLLM** (pick one)

Option A, wheel (default; use the `vllm/` clone only as a reference for reading the MoE/EP code):
```bash
python -m pip install --upgrade pip
python -m pip install vllm==0.33.0   # team-pinned version; everyone uses the same one
```

Option B, editable install from `../vllm` (only if you need to patch vLLM, e.g. timing hooks or expert-load logging in the MoE layer). Python edits take effect without reinstalling; `VLLM_USE_PRECOMPILED=1` reuses the prebuilt CUDA kernels, so there's no 20-min compile like HW2:
```bash
python -m pip install --upgrade pip
cd ~/assignments/project-home/vllm
VLLM_USE_PRECOMPILED=1 python -m pip install -e .
cd ../cs5470-project
```

Either way, check:
```bash
python -c "import vllm, torch; print(vllm.__version__, vllm.__file__, torch.__version__, torch.version.cuda)"
```
`vllm.__file__` shows which one is active (`site-packages` = A, `project-home/vllm` = B). No `gxx` or `cudatoolkit` module needed for either; the wheels ship their own CUDA libraries. Check in step 3 that the node's driver supports the CUDA version torch prints here.

**1e. Download the model and dataset**
```bash
huggingface-cli download Qwen/Qwen3-30B-A3B   # not gated, no `hf auth login` needed; lands in $HF_HOME

mkdir -p $PSCRATCH/data && cd $PSCRATCH/data
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
sha256sum ShareGPT_V3_unfiltered_cleaned_split.json   # record in WORKLOAD_CARD.md
cd ~/assignments/project-home/cs5470-project
```

**Every new shell:**
```bash
module load conda && conda activate $PSCRATCH/cs5470-ep
cd ~/assignments/project-home/cs5470-project
```

### 2. Allocate a node
Pin the 40GB A100s so every teammate's GPU runs are on identical hardware:
```bash
salloc -N 1 -C "gpu&hbm40g" --gpus-per-node 4 -q interactive -t 04:00:00 -A m4999
```
After `salloc` lands you on the compute node, re-run the "every new shell" lines. Open a second terminal on the same node for the client (`ssh <nodename>` from a login node; `hostname` on the first shell shows it), as in HW2: shell 1 = server, shell 2 = benchmark client.

### 3. Record versions (fill the TBDs in the YAML)
```bash
nvidia-smi | head -4
python -c "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.nccl.version(), torch.cuda.is_available())"
```
The "CUDA Version" in the `nvidia-smi` header must be ≥ `torch.version.cuda`, and `is_available()` must be `True`. If not, the wheel's CUDA is too new for the node's driver; ask the GPU lead before running anything.

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
  --dataset-path $PSCRATCH/data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 50 --max-concurrency 16 --ignore-eos
```

### 6. Concurrency sweep
```bash
OUT=results/gpu/C1; mkdir -p $OUT
for C in 1 2 4 8 16 32 64 128 256; do
  for REP in 0; do   # first pass: 1 rep; final runs: 0 1 2
    vllm bench serve --backend vllm --model Qwen/Qwen3-30B-A3B --endpoint /v1/completions \
      --dataset-name sharegpt --dataset-path $PSCRATCH/data/ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 500 --seed 0 --ignore-eos \
      --request-rate inf --max-concurrency $C \
      --save-result --save-detailed --result-dir $OUT --result-filename c${C}_r${REP}.json
  done
done
```
C=1 and C=2 take the longest (~15+ min each). If the sweep doesn't fit in the 4 h interactive limit, run it with `sbatch`.

### 7. Profiling run (separate from the sweep; the profiler adds overhead)
1. Stop the server and restart it with the profiler on:
   - older vLLM: `VLLM_TORCH_PROFILER_DIR=$PSCRATCH/traces/gpu_c1 vllm serve ...`
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
