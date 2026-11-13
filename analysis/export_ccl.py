"""Export one vLLM sweep to validated, per-point CCL-Bench artifact folders."""
import argparse
import copy
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import yaml

from serving_metrics import METRIC_INFO, measure, number


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metric_config():
    return {key: {"label": label, "unit": unit, "higher_is_better": higher,
                  "description": description, "category": "client_serving",
                  "trace_types": ["vllm_bench_serve"], "required_update": True}
            for key, (label, unit, higher, description) in METRIC_INFO.items()}


def export(source, out, card_path, *, demo=False, config=None, platform=None, devices=None):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError(f"output exists: {out}; choose a fresh directory")
    if source.is_dir():
        files = sorted(f for f in source.glob("c*_r*.json") if re.fullmatch(r"c\d+_r\d+\.json", f.name))
        meta_path = source / "run_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    else:
        files, meta = [source], {}
    if not files:
        raise ValueError("no c<C>_r<R>.json benchmark files found")
    args = meta.get("args", {})
    config, platform, devices = config or args.get("config"), platform or args.get("platform"), devices or args.get("devices")
    if config not in ("C1", "C2", "C3") or platform not in ("gpu", "tpu"):
        raise ValueError("supply --config C1/C2/C3 and --platform gpu/tpu (or run_meta.json)")
    number(devices, "devices", positive=True, integer=True)
    # Overrides that change topology require a new explicit layout/card, rather
    # than silently claiming that a C1/C2/C3 default was used.
    extra = args.get("extra_server_args", "")
    if any(flag in extra for flag in ("--tensor-parallel", "--data-parallel", "--pipeline-parallel", "--expert-parallel")):
        raise ValueError("parallelism overrides in extra_server_args are not supported by this exporter")
    card = yaml.safe_load(Path(card_path).read_text())
    if not isinstance(card, dict) or not all(k in card for k in ("workload", "Model-executor")):
        raise ValueError("card must contain workload and Model-executor mappings")
    run = f"{platform}/{source.name if source.is_dir() else source.stem}"
    rows, artifacts = [], []
    model_ids = set()
    for file in files:
        raw = file.read_bytes()
        data = json.loads(raw)
        if file.name == "model.json" and not demo:
            raise ValueError("results/model.json is a shortened DEMO; pass --demo explicitly")
        metrics, validation = measure(data, demo=demo)
        model = data.get("model_id", data.get("model"))
        if not isinstance(model, str) or not model:
            raise ValueError(f"{file}: model_id missing")
        model_ids.add(model)
        if args.get("model") and args["model"] != model:
            raise ValueError(f"{file}: model disagrees with run_meta.json")
        c = number(data.get("max_concurrency"), "max_concurrency", positive=True, integer=True)
        match = re.fullmatch(r"c(\d+)_r(\d+)\.json", file.name)
        rep = int(match[2]) if match else 0
        if match and int(match[1]) != c:
            raise ValueError(f"{file}: concurrency disagrees with filename")
        point_name = ("DEMO-" if demo else "") + f"{config}-{platform}-c{c}-r{rep}"
        point_card = copy.deepcopy(card)
        point_card["description"] = ("DEMO — shortened request arrays; not experimental evidence. " if demo else "") + (
            f"{config}, {platform}, {model}; max client concurrency={c}, repeat={rep}. "
            "Client-measured serving results; no profiler metrics. Batch size is dynamic, not client concurrency.")
        point_card["hf_url"] = f"https://huggingface.co/{model}"
        point_card["trace_url"] = None
        workload = point_card["workload"]
        workload["model"].update(model_family=model.split("/")[-1], phase="serving", iteration=1, granularity="request")
        workload["data"].update(batch_size=None, input_len=None, output_len=None)
        if not demo:
            good = [i for i, error in enumerate(data["errors"]) if not error]
            workload["data"].update(input_len=sum(data["input_lens"][i] for i in good) / len(good),
                                    output_len=sum(data["output_lens"][i] for i in good) / len(good))
        if args.get("workload"):
            workload["data"]["dataset"] = args["workload"]
        workload["hardware"]["xpu_spec"].update(type=platform.upper(), total_count=devices,
                                                   count_per_node=devices)
        if platform.upper() != card["workload"]["hardware"]["xpu_spec"]["type"].upper():
            raise ValueError("card hardware type does not match platform")
        plan = point_card["Model-executor"]["model_plan_parallelization"]
        plan.update(tp=1 if config == "C2" else devices, dp_replicate=devices if config == "C2" else 1,
                    dp_shard=1, pp=1, cp=1, ep=1 if config == "C1" else devices)
        point_card["metric_source"] = {"traces": ["vllm_bench_serve"], "metrics_specific_trace": list(METRIC_INFO)}
        point_card["serving_benchmark"] = {"config": config, "max_concurrency": c, "repeat": rep, "demo": demo}
        manifest = {"schema_version": 1, "source": str(file), "sha256": hashlib.sha256(raw).hexdigest(),
                    "run": run, "config": config, "platform": platform, "devices": devices,
                    "concurrency": c, "repeat": rep, "model": model, "validation": validation,
                    "metrics": metrics, "run_metadata": meta,
                    "comparison": {"model": model, "dataset": workload["data"].get("dataset"),
                                   "phase": args.get("workload", workload["model"]["phase"]),
                                   "precision": workload["model"].get("precision"),
                                   "devices": devices, "num_prompts": data.get("num_prompts", data["completed"] + data.get("failed", 0)),
                                   "request_rate": data.get("request_rate"),
                                   "burstiness": data.get("burstiness")}}
        rows.append({**manifest, "artifact": point_name})
        artifacts.append((point_name, raw, point_card, manifest))
    if len(model_ids) != 1:
        raise ValueError("a sweep must contain exactly one model")
    # Validate the entire sweep before writing; publish staging only on success.
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "export"
        stage.mkdir()
        for name, raw, point_card, manifest in artifacts:
            point = stage / name
            point.mkdir()
            (point / "benchmark.json").write_bytes(raw)
            (point / "workload.yaml").write_text(yaml.safe_dump(point_card, sort_keys=False))
            write_json(point / "manifest.json", manifest)
        write_json(stage / "summary.json", {"schema_version": 1, "demo": demo, "rows": rows})
        write_json(stage / "benchmark_config.json", {
            "pairs": [{"trace": str(out / row["artifact"]), "metrics": list(METRIC_INFO), "required_update": True} for row in rows],
            "metric_categories": [{"id": "client_serving", "label": "Client serving"}],
            "metric_info": metric_config()})
        shutil.move(str(stage), str(out))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="one sweep directory, or one result JSON")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--card", type=Path, required=True, help="GPU/TPU workload card with hardware metadata")
    p.add_argument("--config", choices=["C1", "C2", "C3"])
    p.add_argument("--platform", choices=["gpu", "tpu"])
    p.add_argument("--devices", type=int)
    p.add_argument("--demo", action="store_true", help="label all output as DEMO; use aggregate fields only")
    args = p.parse_args()
    try:
        rows = export(args.source, args.out, args.card, demo=args.demo, config=args.config, platform=args.platform, devices=args.devices)
    except (ValueError, OSError, KeyError, TypeError) as e:
        p.exit(2, f"export failed: {e}\n")
    print(f"Exported {len(rows)} {'DEMO ' if args.demo else ''}points to {args.out}")


if __name__ == "__main__":
    main()
