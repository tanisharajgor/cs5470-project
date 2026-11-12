**Applicable Settings**
Workload: ShareGPT
Devices: 4 TPU v6e chips (`ct6e-standard-4t`)
Configuration: C1 (Baseline) - Tensor parallel only, no EP

Workload details (filtering, arrival pattern, sweep): [workloads/sharegpt/README.md](../workloads/sharegpt/README.md)
GPU counterpart: [run-gpu-workload.md](run-gpu-workload.md)

**Steps to Run a Workload on TPU**

The TPU VM is temporary: it **deletes itself, disk included, when `--max-run-duration` runs out**. Have these steps ready before it starts, and copy results off before the deadline.

Laptop steps run in a local terminal; VM steps run over SSH on the TPU VM.

## 1. Request the VM (laptop)
```bash
gcloud config get-value project        # must be cs5470-fa26-team23
gcloud compute instances create <netid>-v6e \
  --zone=us-east5-a \
  --machine-type=ct6e-standard-4t \
  --provisioning-model=FLEX_START \
  --request-valid-for-duration=2h \
  --max-run-duration=4h \
  --instance-termination-action=DELETE \
  --maintenance-policy=TERMINATE \
  --image-project=ubuntu-os-accelerator-images \
  --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
  --boot-disk-size=200GB \
  --scopes=cloud-platform
```
- `ct6e-standard-4t` = TPU v6e, 4 chips. The image family name just lists which TPUs the image supports.
- `--boot-disk-size=200GB`: the model is 57 GB, and the env, caches and dataset come on top.
- Flex-start queues the request until 4 chips are free: minutes to 2 h. After 2 h the request expires; submit again.

Check status (another terminal is fine; the `create` command may block while it waits):
```bash
gcloud compute instances describe <netid>-v6e --zone=us-east5-a --format="value(status)"
```
`PENDING`/`PROVISIONING` = waiting for chips, `STAGING` = booting (1-3 min), `RUNNING` = ready. The 4 h clock starts at `RUNNING`.

**If it stays `PENDING` for a long time:**
- Shorter `--max-run-duration` (e.g. 3h) is easier to schedule.
- Try another zone with v6e: `gcloud compute machine-types list --filter="name=ct6e-standard-4t" --format="value(zone)"`. Delete any extra VM as soon as one starts.
- Request at quiet hours, not right before deadlines.
- To cancel and resubmit: check the status first, then `gcloud compute instances delete <netid>-v6e --zone=us-east5-a --quiet`. Wait until `gcloud compute instances list` no longer shows it before re-creating with the same name.

## 2. Connect (laptop)
```bash
gcloud compute ssh <netid>-v6e --zone=us-east5-a
```

## 3. Start tmux (VM)
Everything after this runs inside tmux, so a dropped SSH connection or a sleeping laptop doesn't kill the run.
```bash
tmux new -s run            # reattach later: tmux attach -t run
```

## 4. Install (VM)
The image ships Python 3.10; use `uv` to get 3.12.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv venv --python 3.12 ~/vllm-env && source ~/vllm-env/bin/activate
uv pip install vllm-tpu                          # record the version; everyone pins the same one
uv pip install -U "huggingface_hub[cli]" matplotlib
python -c "import jax; print(jax.devices())"      # must list 4 TPU v6e devices
```
If `vllm-tpu` fails to install, check the [tpu-inference README](https://github.com/vllm-project/tpu-inference) for the current install command.

## 5. Get the code, dataset and model (VM)
```bash
git clone https://github.com/tanisharajgor/cs5470-project.git && cd cs5470-project
mkdir -p ~/data && wget -P ~/data https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
hf download Qwen/Qwen3-30B-A3B                   # ~57 GB into ~/.cache/huggingface
```

## 6. Run the sweep (VM)
Same runner and flags as GPU, with `--platform tpu`:
```bash
python scripts/run_workload.py --platform tpu --config C1 --concurrency 256 128 64 32 16
```
- The first server start is slow: XLA compiles every batch shape, often 20-40 min. The runner waits up to 1 h.
- If the server dies, the runner prints the last 30 lines of `logs/server.log`. Qwen3-MoE support in tpu-inference is the most likely failure point; the fallback is MaxText (see the proposal).
- Results land in `results/tpu/C1-sharegpt-qwen3-30b-a3b-4tpu/`: `c<C>_r<R>.json`, `run_meta.json` (versions, devices), and `logs/` (server, warm-up, each point).
- Running out of time? Re-run with `--skip-existing` on a new VM to fill in only the missing points.

Time budget for a 4 h VM: ~15 min install + downloads, up to ~40 min first server start, ~20-30 min for 256 to 16.

## 7. Copy results off (laptop, before the VM expires)
```bash
cd <path to your local clone of this repo>
gcloud compute scp --recurse --zone=us-east5-a <netid>-v6e:~/cs5470-project/results/tpu ./results/
```

## 8. Plot GPU and TPU together (laptop)
```bash
python3 analysis/plot_throughput.py      # -> figures/request_throughput_vs_concurrency.png
```
Every `results/<platform>/<run>/` folder becomes one line (GPU solid, TPU dashed).

## 9. Delete the VM (laptop)
It deletes itself at the max run duration, but delete it as soon as you're done; it bills while it exists.
```bash
gcloud compute instances delete <netid>-v6e --zone=us-east5-a
```

## Profiling (separate, short run)
tpu-inference writes a JAX/XLA trace; view it in TensorBoard / xprof. Check the tpu-inference docs for the env var, and keep profiled runs short (`--num-prompts 32`) since the profiler adds overhead.

## Notes / issues encountered
- 
