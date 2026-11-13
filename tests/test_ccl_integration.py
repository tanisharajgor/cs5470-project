"""CPU-only validation, export, dashboard, and plotting regression tests."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_ccl import export
from plot_pareto import plot
from preview_ccl import build_preview
from serving_metrics import measure, median_points, metric_cal, pareto_front
from run_workload import bench_cmd, parse_args
from unittest.mock import patch

CARD = next((ROOT / "trace_collection").glob("*gpu*/*.yaml"))


def benchmark():
    # Deliberately unequal decode times: mean(rate) differs from 1/mean(TPOT).
    return {"model_id": "Qwen/Qwen3-30B-A3B", "num_prompts": 3, "completed": 3,
            "failed": 0, "duration": 10., "total_input_tokens": 9,
            "total_output_tokens": 7, "request_throughput": .3,
            "output_throughput": .7, "max_concurrency": 2,
            "input_lens": [2, 3, 4], "output_lens": [3, 3, 1],
            "ttfts": [.1, .2, .3], "latencies": [1.1, 2.2, .3],
            "errors": ["", "", ""], "request_rate": "inf", "burstiness": 1.}


class MetricsTest(unittest.TestCase):
    def test_definitions_and_units(self):
        metrics, report = measure(benchmark())
        self.assertAlmostEqual(metrics["client_user_tps"], 1.5)
        self.assertAlmostEqual(metrics["client_system_tps"], .7)
        self.assertAlmostEqual(metrics["client_ttft_ms"], 200)
        self.assertAlmostEqual(metrics["client_tpot_ms"], 750)
        self.assertAlmostEqual(metrics["client_e2e_ms"], 1200)
        self.assertEqual(report["decode_requests"], 2)

    def test_failed_requests_are_excluded(self):
        data = benchmark()
        data.update(num_prompts=4, failed=1)
        for key in ("input_lens", "output_lens", "ttfts", "latencies"):
            data[key].append(0)
        data["errors"].append("timeout")
        self.assertEqual(measure(data)[0], measure(benchmark())[0])

    def test_one_token_outputs(self):
        data = benchmark()
        data.update(output_lens=[1, 1, 1], total_output_tokens=3, output_throughput=.3)
        self.assertIsNone(measure(data)[0]["client_user_tps"])
        self.assertIsNone(measure(data)[0]["client_tpot_ms"])

    def test_invalid_fields(self):
        cases = [("duration", 0), ("duration", float("nan")), ("duration", True),
                 ("completed", 0), ("failed", -1), ("total_output_tokens", 8),
                 ("total_input_tokens", 10), ("request_throughput", 20),
                 ("num_prompts", 9), ("errors", ["failure", "", ""]),
                 ("latencies", [.01, 2.2, .3]), ("latencies", [.1, 2.2, .3]),
                 ("input_lens", [2, 3]), ("output_lens", [3, float("inf"), 1]),
                 ("mean_ttft_ms", 9999)]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                data = benchmark()
                data[key] = value
                measure(data)

    def test_demo_uses_no_request_derived_values(self):
        data = json.loads((ROOT / "results/model.json").read_text())
        with self.assertRaises(ValueError):
            measure(data)
        metrics, report = measure(data, demo=True)
        self.assertTrue(report["demo"])
        self.assertIsNone(metrics["client_user_tps"])
        self.assertIsNone(metrics["client_e2e_ms"])
        self.assertAlmostEqual(metrics["client_system_tps"], data["output_throughput"])

    def test_frontier_maximizes_both_and_retains_ties(self):
        points = [dict(user_tps=x, system_tps=y) for x, y in [(1, 3), (2, 2), (3, 1), (1, 1), (1, 3)]]
        self.assertEqual(pareto_front(points), points[:3] + points[4:])


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "gpu-C1"
        self.source.mkdir()
        (self.source / "run_meta.json").write_text(json.dumps({"args": {
            "config": "C1", "platform": "gpu", "devices": 4,
            "model": "Qwen/Qwen3-30B-A3B", "workload": "sharegpt"}}))
        (self.source / "c2_r0.json").write_text(json.dumps(benchmark()))

    def do_export(self, **kwargs):
        return export(self.source, self.root / "export", CARD, **kwargs)

    def test_export_metrics_card_provenance_and_adapter(self):
        import yaml
        rows = self.do_export()
        self.assertEqual(len(rows), 1)
        point = self.root / "export" / rows[0]["artifact"]
        self.assertEqual(metric_cal(point, "client_user_tps"), 1.5)
        self.assertEqual(rows[0]["sha256"], hashlib.sha256((self.source / "c2_r0.json").read_bytes()).hexdigest())
        card = yaml.safe_load((point / "workload.yaml").read_text())
        self.assertIsNone(card["workload"]["data"]["batch_size"])
        self.assertEqual(card["metric_source"]["traces"], ["vllm_bench_serve"])
        self.assertEqual(card["Model-executor"]["model_plan_parallelization"]["tp"], 4)
        with self.assertRaises(ValueError):
            self.do_export()

    def test_c2_c3_layouts(self):
        import yaml
        for config in ("C2", "C3"):
            out = self.root / config
            rows = export(self.source, out, CARD, config=config)
            card = yaml.safe_load((out / rows[0]["artifact"] / "workload.yaml").read_text())
            plan = card["Model-executor"]["model_plan_parallelization"]
            self.assertEqual(plan["ep"], 4)
            self.assertEqual(plan["tp"], 1 if config == "C2" else 4)
            self.assertEqual(plan["dp_replicate"], 4 if config == "C2" else 1)

    def test_atomic_validation_and_filename_mismatch(self):
        (self.source / "c4_r0.json").write_text(json.dumps(benchmark()))
        with self.assertRaises(ValueError):
            self.do_export()
        self.assertFalse((self.root / "export").exists())

    def test_explicit_demo_only_and_plot_exclusion(self):
        source = ROOT / "results/model.json"
        opts = dict(config="C1", platform="gpu", devices=4)
        with self.assertRaisesRegex(ValueError, "DEMO"):
            export(source, self.root / "export", CARD, **opts)
        rows = export(source, self.root / "export", CARD, demo=True, **opts)
        self.assertIn("DEMO", rows[0]["artifact"])
        self.assertEqual(median_points(rows), [])
        with self.assertRaisesRegex(ValueError, "no validated"):
            plot([self.root / "export"], self.root / "figures")

    def test_repeats_plot_and_duplicate_rejection(self):
        data = benchmark()
        data["latencies"] = [2.1, 4.2, .3]
        (self.source / "c2_r1.json").write_text(json.dumps(data))
        rows = self.do_export()
        self.assertAlmostEqual(median_points(rows)[0]["user_tps"], 1.125)
        points = plot([self.root / "export"], self.root / "figures")
        self.assertEqual(points[0]["repeats"], 2)
        for extension in ("png", "pdf", "csv"):
            self.assertGreater((self.root / f"figures/pareto.{extension}").stat().st_size, 0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            plot([self.root / "export"] * 2, self.root / "figures")

    @unittest.skipUnless((ROOT.parent / "ccl-bench/tools/main.py").exists(), "sibling CCL-Bench checkout required")
    def test_real_ccl_generator_and_dispatcher(self):
        self.do_export()
        upstream = ROOT.parent / "ccl-bench"
        before = subprocess.check_output(["git", "status", "--porcelain"], cwd=upstream)
        out = build_preview(self.root / "export", upstream, self.root / "preview")
        dashboard = json.loads((out / "website/benchmark_data.json").read_text())
        self.assertEqual(dashboard["rows"][0]["metrics"]["client_user_tps"], 1.5)
        self.assertEqual(len(dashboard["all_metrics"]), 6)
        self.assertIn("LOCAL PREVIEW", (out / "index.html").read_text())
        self.assertEqual(before, subprocess.check_output(["git", "status", "--porcelain"], cwd=upstream))
        demo_out = self.root / "demo"
        export(ROOT / "results/model.json", demo_out, CARD, demo=True, config="C1", platform="gpu", devices=4)
        preview = build_preview(demo_out, upstream, self.root / "demo-preview")
        demo_data = json.loads((preview / "website/benchmark_data.json").read_text())
        self.assertIsNone(demo_data["rows"][0]["metrics"]["client_user_tps"])
        self.assertIn("DEMO — shortened", (preview / "index.html").read_text())


class RunnerTest(unittest.TestCase):
    def test_detailed_result_command(self):
        with patch.object(sys, "argv", ["run_workload.py", "--config", "C1"]):
            args = parse_args()
        cmd = bench_cmd(args, 16, 500, Path("out"), "c16_r0.json")
        self.assertIn("--save-detailed", cmd)
        self.assertIn("--save-result", cmd)
        self.assertEqual(cmd[cmd.index("--max-concurrency") + 1], "16")


if __name__ == "__main__":
    unittest.main()
