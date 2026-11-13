These YAML templates follow CCL-Bench conventions and are separated by model,
workload, and hardware. Review their descriptive metadata before use. The
[validated exporter](../docs/ccl-bench.md) creates per-point cards with the
actual concurrency/repeat, config layout, input/output means, and a distinct
`vllm_bench_serve` metric source. Serving JSON alone does not establish profiler
metrics or trace availability.
