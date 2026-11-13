"""Build an isolated local CCL-Bench dashboard using its real data generator.

The upstream checkout is read-only. Only exported serving metrics are registered
in the staged tools/main.py; public rows and profiler metrics are not included.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from serving_metrics import METRIC_INFO

ROOT = Path(__file__).resolve().parents[1]


def build_preview(export_dir, ccl_repo, out):
    export_dir, ccl_repo, out = (Path(p).resolve() for p in (export_dir, ccl_repo, out))
    if out.exists():
        raise ValueError(f"output exists: {out}; choose a fresh directory")
    for relative in ("index.html", "website/styles.css", "website/generate_data.py", "tools/main.py"):
        if not (ccl_repo / relative).is_file():
            raise ValueError(f"CCL-Bench file missing: {relative}")
    summary = json.loads((export_dir / "summary.json").read_text())
    config = json.loads((export_dir / "benchmark_config.json").read_text())
    # Resolve paths from the current export location, so copied exports work.
    config["pairs"] = [{"trace": str(export_dir / row["artifact"]), "metrics": list(METRIC_INFO),
                        "required_update": True} for row in summary["rows"]]
    main = (ccl_repo / "tools/main.py").read_text()
    anchor = '    if metric_name == "wall_time_s":'
    if main.count(anchor) != 1:
        raise ValueError("unsupported CCL-Bench dispatcher version; expected wall_time_s registration anchor")
    registration = ('    if metric_name in ' + repr(tuple(METRIC_INFO)) + ':\n'
                    '        from client_serving import metric_cal\n'
                    '        value = metric_cal(trace_directory, metric_name)\n'
                    '        print("nan" if value is None else value)\n'
                    '        sys.exit(0)\n\n')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "preview"
        (stage / "website").mkdir(parents=True)
        (stage / "tools").mkdir()
        for relative in ("index.html", "website/styles.css", "website/generate_data.py"):
            shutil.copy2(ccl_repo / relative, stage / relative)
        if (ccl_repo / "assets").is_dir():
            shutil.copytree(ccl_repo / "assets", stage / "assets")
        for relative in ("favicon.ico", "leaderboard.html"):
            if (ccl_repo / relative).is_file():
                shutil.copy2(ccl_repo / relative, stage / relative)
        html = (stage / "index.html").read_text()
        label = "DEMO — shortened request arrays; not experimental evidence" if summary["demo"] else "LOCAL PREVIEW — client-measured serving metrics"
        html = html.replace("<body>", f'<body><div style="padding:16px;background:#fff2b3;color:#222;text-align:center">{label}</div>', 1)
        # Client metrics have no universal good/bad thresholds.
        html = html.replace("  let formatted;", "  cls = 'mp-na'; // Neutral styling for client measurements\n  let formatted;", 1)
        (stage / "index.html").write_text(html)
        (stage / "tools/main.py").write_text(main.replace(anchor, registration + anchor))
        shutil.copy2(ROOT / "analysis/serving_metrics.py", stage / "tools/client_serving.py")
        (stage / "website/benchmark_config.json").write_text(json.dumps(config, indent=2))
        subprocess.run([sys.executable, "website/generate_data.py"], cwd=stage, check=True)
        generated = json.loads((stage / "website/benchmark_data.json").read_text())
        if len(generated["rows"]) != len(summary["rows"]):
            raise ValueError("dashboard generator dropped rows")
        expected = {row["artifact"]: row["metrics"] for row in summary["rows"]}
        for row in generated["rows"]:
            for metric, value in row["metrics"].items():
                want = expected[Path(row["trace"]).name][metric]
                if (want is None) != (value is None) or (want is not None and abs(want - value) > max(1e-5, abs(want) * 1e-5)):
                    raise ValueError(f"dashboard metric mismatch: {metric}")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ccl_repo, capture_output=True, text=True, check=True).stdout.strip()
        (stage / "preview_provenance.json").write_text(json.dumps({"ccl_commit": commit, "source_export": str(export_dir)}, indent=2))
        shutil.move(str(stage), str(out))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("export", type=Path)
    p.add_argument("--ccl-repo", type=Path, default=ROOT.parent / "ccl-bench")
    p.add_argument("--out", type=Path, default=ROOT / "build/ccl-preview")
    args = p.parse_args()
    try:
        out = build_preview(args.export, args.ccl_repo, args.out)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as e:
        p.exit(2, f"preview failed: {e}\n")
    print(f"Preview ready. Serve with: {sys.executable} -m http.server 8081 --bind 127.0.0.1 --directory {out}")
    print("Open http://127.0.0.1:8081")


if __name__ == "__main__":
    main()
