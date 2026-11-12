"""Run one workload end to end on GPU (Perlmutter) or TPU (Cloud TPU VM): start vLLM, wait for it, warm up, sweep concurrency, shut down.

Run on a Perlmutter compute node (after `salloc`, with the cs5470-ep env active), from any directory.

Examples:
    python scripts/run_workload.py --config C1                                   # GPU, defaults below
    python scripts/run_workload.py --platform tpu --config C1                    # same run on TPU
    python scripts/run_workload.py --config C2 --concurrency 256 128 64 32 16
    python scripts/run_workload.py --config C3 --workload decode --reps 3
    python scripts/run_workload.py --config C1 --dry-run                         # print commands only

On GPU, run on a Perlmutter compute node (after `salloc`, cs5470-ep env active). On TPU, run on
the TPU VM with the vllm-tpu env active. Either way, from any directory.

Each run writes one folder, results/<platform>/<run-name>/:
    run_meta.json        exact commands, args, versions, hostname, devices
    c<C>_r<R>.json       vllm bench serve results (what the plots read)
    logs/server.log      vllm serve output
    logs/warmup.log      warm-up client output
    logs/c<C>_r<R>.log   client output for each point (the summary table you see in the terminal)
    logs/driver.log      everything this script printed
Plot them with `python analysis/plot_throughput.py`.
"""
import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Parallel layout per config (see the proposal's experiment table). Each maps the
# device count to vLLM serve flags. C4-C6 need 8 GPUs = 2 Perlmutter nodes (multi-node
# launch), which this script doesn't do yet.
CONFIGS = {
    "C1": lambda n: ["--tensor-parallel-size", str(n)],                                   # TP only, no EP
    "C2": lambda n: ["--data-parallel-size", str(n), "--enable-expert-parallel"],         # EP, DP attention
    "C3": lambda n: ["--tensor-parallel-size", str(n), "--enable-expert-parallel"],       # EP + TP attention
}

# Workloads and phases from workloads/sharegpt/README.md, as `vllm bench serve` flags.
SHAREGPT = ["--dataset-name", "sharegpt", "--dataset-path", "{dataset_path}"]
WORKLOADS = {
    "sharegpt": SHAREGPT,                                                  # prefill + decode co-located
    "prefill": SHAREGPT + ["--sharegpt-output-len", "1"],                  # prefill-only
    "decode": ["--dataset-name", "random", "--random-input-len", "128",    # decode-only
               "--random-output-len", "1024"],
}

# Server settings shared by every run (workloads/sharegpt/README.md section 6).
SERVER_FLAGS = ["--max-model-len", "4096", "--max-num-seqs", "256",
                "--no-enable-prefix-caching", "--generation-config", "vllm"]

ROOT = Path(__file__).resolve().parents[1]  # repo root; results always land in ROOT/results

# Perlmutter keeps data on $PSCRATCH; the TPU VM has no scratch, so default to ~/data there.
DEFAULT_DATASET = os.path.join(os.environ.get("PSCRATCH", os.path.expanduser("~")), "data",
                               "ShareGPT_V3_unfiltered_cleaned_split.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, choices=sorted(CONFIGS))
    p.add_argument("--platform", default="gpu", choices=["gpu", "tpu"])
    p.add_argument("--devices", type=int, default=4, help="GPUs / TPU chips to use (default 4)")
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--workload", default="sharegpt", choices=sorted(WORKLOADS))
    p.add_argument("--dataset-path", default=DEFAULT_DATASET)
    p.add_argument("--concurrency", type=int, nargs="+", default=[256, 128, 64, 32, 16, 8, 4, 2, 1],
                   help="sweep points, run in this order (default: high to low, fast ones first)")
    p.add_argument("--num-prompts", type=int, default=500)
    p.add_argument("--reps", type=int, default=1, help="repeats per concurrency (final runs: 3)")
    p.add_argument("--name", help="results folder name (default: <config>-<workload>-<model>-<devices><platform>)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--startup-timeout", type=int,
                   help="seconds to wait for the server (default 1200 on GPU, 3600 on TPU: XLA compiles every batch shape)")
    p.add_argument("--skip-existing", action="store_true", help="skip points whose result file exists")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--extra-server-args", default="", help='extra vllm serve flags, e.g. "--all2all-backend naive"')
    p.add_argument("--dry-run", action="store_true", help="print the commands without running anything")
    return p.parse_args()


def server_cmd(args):
    return (["vllm", "serve", args.model] + CONFIGS[args.config](args.devices) + SERVER_FLAGS
            + ["--port", str(args.port)] + shlex.split(args.extra_server_args))


def bench_cmd(args, concurrency, num_prompts, out_dir=None, filename=None):
    workload = [a.format(dataset_path=args.dataset_path) for a in WORKLOADS[args.workload]]
    cmd = (["vllm", "bench", "serve", "--backend", "vllm", "--model", args.model,
            "--endpoint", "/v1/completions", "--port", str(args.port)] + workload
           + ["--num-prompts", str(num_prompts), "--seed", "0", "--ignore-eos",
              "--request-rate", "inf", "--max-concurrency", str(concurrency)])
    if out_dir:
        cmd += ["--save-result", "--save-detailed", "--result-dir", str(out_dir), "--result-filename", filename]
    return cmd


class Tee:
    """Write to the terminal and a log file at once."""
    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, "a")

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stream.flush()


def run_logged(cmd, log_path):
    """Run cmd, streaming its output to the terminal and to log_path."""
    print(f"$ {shlex.join(cmd)}  (log: {log_path})")
    with open(log_path, "w") as log:
        log.write(f"$ {shlex.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        if proc.wait():
            sys.exit(f"command failed with exit code {proc.returncode}; see {log_path}")


def wait_for_server(proc, port, timeout, log_path):
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace").splitlines()[-30:]
            sys.exit(f"server exited with code {proc.returncode} during startup. Last log lines:\n" + "\n".join(tail))
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5):
                print(f"server ready after {time.time() - start:.0f}s")
                return
        except OSError:
            time.sleep(5)
    sys.exit(f"server not ready after {timeout}s; see {log_path}")


def stop_server(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# Runs in a child process: importing torch/vLLM/JAX can claim the accelerator (only one process
# may hold a TPU), and the driver has to leave it free for vllm serve.
VERSION_PROBE = """
import json
out = {}
def probe(fn):
    try:
        out.update(fn())
    except Exception as e:
        out.setdefault("probe_errors", []).append(repr(e))
def vllm_torch():
    import torch, vllm
    return {"vllm": vllm.__version__, "torch": torch.__version__}
def cuda():
    import torch
    return {"cuda": torch.version.cuda, "nccl": ".".join(map(str, torch.cuda.nccl.version()))}
def jax_tpu():
    import jax
    return {"jax": jax.__version__, "devices": [str(d) for d in jax.devices()]}
probe(vllm_torch)
probe(cuda)
probe(jax_tpu)
print(json.dumps(out))
"""


def versions(platform):
    """Best-effort versions and devices; whatever is missing on this platform is recorded, not fatal."""
    out = {"hostname": socket.gethostname()}
    probe = subprocess.run([sys.executable, "-c", VERSION_PROBE], capture_output=True, text=True)
    try:
        out.update(json.loads(probe.stdout.strip().splitlines()[-1]))
    except (IndexError, json.JSONDecodeError):
        out["probe_errors"] = [probe.stderr.strip()[-500:]]
    if platform == "gpu":
        try:
            out["gpus"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                         capture_output=True, text=True).stdout.strip().splitlines()
        except FileNotFoundError:
            pass
    return out


def main():
    args = parse_args()
    if args.startup_timeout is None:
        args.startup_timeout = 3600 if args.platform == "tpu" else 1200
    model_slug = args.model.split("/")[-1].lower()
    name = args.name or f"{args.config}-{args.workload}-{model_slug}-{args.devices}{args.platform}"
    out_dir = ROOT / "results" / args.platform / name
    serve = server_cmd(args)

    if args.dry_run:
        print("server:", shlex.join(serve))
        if not args.no_warmup:
            print("warmup:", shlex.join(bench_cmd(args, 16, 50)))
        for c in args.concurrency:
            for rep in range(args.reps):
                print(f"c={c} r={rep}:", shlex.join(bench_cmd(args, c, args.num_prompts, out_dir, f"c{c}_r{rep}.json")))
        return

    if args.platform == "gpu" and socket.gethostname().startswith("login"):
        sys.exit("this is a login node; run inside `salloc` on a compute node (nid...)")
    if args.workload != "decode" and not Path(args.dataset_path).exists():
        sys.exit(f"dataset not found: {args.dataset_path} (see docs/run-{args.platform}-workload.md)")

    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(sys.stdout, logs / "driver.log")
    (out_dir / "run_meta.json").write_text(json.dumps(
        {"args": vars(args), "server_cmd": serve, **versions(args.platform)}, indent=2))

    log_path = logs / "server.log"
    print(f"starting server (log: {log_path}):\n  {shlex.join(serve)}")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(serve, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_for_server(proc, args.port, args.startup_timeout, log_path)
        if not args.no_warmup:
            print("warm-up (50 prompts, c=16, not saved)")
            run_logged(bench_cmd(args, 16, 50), logs / "warmup.log")
        for c in args.concurrency:
            for rep in range(args.reps):
                filename = f"c{c}_r{rep}.json"
                if args.skip_existing and (out_dir / filename).exists():
                    print(f"skip {filename} (exists)")
                    continue
                print(f"=== c={c} rep={rep}")
                run_logged(bench_cmd(args, c, args.num_prompts, out_dir, filename), logs / f"c{c}_r{rep}.log")
    finally:
        stop_server(proc)
    print(f"done; results in {out_dir}. Plot: python analysis/plot_throughput.py")


if __name__ == "__main__":
    main()
