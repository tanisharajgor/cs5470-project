# Workload Card: ShareGPT (chatbot)

The dataset-level card. Every config (C1–C6) and both platforms (GPU and TPU) use this exact workload, so any difference in results comes from the hardware or the parallel layout, not the requests. The per-run CCL-Bench YAML cards live under `trace_collection/`.

## 1. Source
| Field | Value |
|---|---|
| Dataset | ShareGPT V3 (`ShareGPT_V3_unfiltered_cleaned_split.json`) |
| Download | `wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json` |
| SHA-256 | _TBD: run `sha256sum` after download and record the hash here_ |
| Use case | Single-turn chatbot: short-to-medium prompts with variable-length replies |

## 2. Request construction
These are the defaults of vLLM's built-in ShareGPT sampler (`vllm bench serve --dataset-name sharegpt`). Keep them unchanged so the workload is reproducible.

| Field | Value |
|---|---|
| Turn used | First human turn → prompt; first assistant turn → target output length |
| Filtering | Drops prompt < 4 or output < 4 tokens; drops prompt > 1024 tokens or prompt + output > 2048 tokens |
| Tokenizer | `Qwen/Qwen3-30B-A3B` (same tokenizer on both platforms) |
| Endpoint | `/v1/completions` (raw text, no chat template, so Qwen3 "thinking" mode is never triggered) |
| Output length | Fixed to the reference reply length with `--ignore-eos`, so GPU and TPU generate exactly the same number of tokens |
| Sampling | `--seed 0`, `--num-prompts 500`, the same 500 requests at every concurrency |
| Decoding | Greedy, via server flag `--generation-config vllm` (otherwise Qwen3's generation_config.json sets temperature 0.6 / top_k 20 / top_p 0.95). Quality is not measured |

## 3. Length distributions
From the C1 run (`python analysis/card_stats.py results/gpu/C1/c16_r0.json`), n = 500 requests, Qwen3 tokenizer. Identical at every concurrency because of the fixed seed and `--ignore-eos`.

| Tokens | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| Input (prompt) | 207 | 76 | 611 | 804 | 975 |
| Output | 220 | 176 | 519 | 769 | 812 |
| Total (in + out) | 427 | 364 | 787 | 1220 | 1588 |

Totals: 103,437 input tokens, 109,888 output tokens. Inputs are heavily skewed (median 76 vs. mean 207): most prompts are short, with a long tail.

Histogram: [length_hist.png](length_hist.png)

## 4. Arrival pattern
| Field | Value |
|---|---|
| Mode | Closed loop: `--request-rate inf` with `--max-concurrency C` (C requests in flight at all times) |
| Concurrency sweep | C ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}. Each point is one dot on the Pareto plot |
| Warm-up | One throw-away run at C=16 with 50 prompts after the server starts (triggers CUDA graph capture / XLA compilation) |
| Repeats | 3 per point; report the median |

Why closed loop: it holds the number of concurrent requests (the batch the server sees) fixed, which is the knob that moves a run along the user-TPS vs. system-TPS frontier. Open-loop Poisson arrivals (`--request-rate R`) could be added later as a sensitivity check.

## 5. Phase variants
| Phase | How |
|---|---|
| Co-located (default) | As above |
| Prefill-only | Same prompts, `--sharegpt-output-len 1` |
| Decode-only | `--dataset-name random --random-input-len 128 --random-output-len 1024 --ignore-eos` (not ShareGPT, but isolates decode) |

## 6. Server settings shared by all configs
| Field | Value |
|---|---|
| Model | `Qwen/Qwen3-30B-A3B`, bf16 |
| `--max-model-len` | 4096 (covers the 2048 filter with headroom) |
| `--max-num-seqs` | 256 (≥ the largest concurrency, so the client sets the batch size, not the server) |
| Prefix caching | **Disabled** (`--no-enable-prefix-caching`) so repeated prefixes don't skew TTFT |
| Generation config | `--generation-config vllm` (ignore the model's sampling defaults; same decoding on GPU and TPU) |

## 7. Metrics recorded per point
System TPS (output token throughput), per-user TPS (= 1 / mean TPOT), TTFT (mean/p50/p99), TPOT (mean/p50/p99), ITL, and request throughput, all from the `vllm bench serve` result JSON.

## 8. Reference command
```bash
vllm bench serve --backend vllm --model Qwen/Qwen3-30B-A3B --endpoint /v1/completions \
  --dataset-name sharegpt --dataset-path $PSCRATCH/data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 --seed 0 --ignore-eos \
  --request-rate inf --max-concurrency $C \
  --save-result --save-detailed --result-dir results/<platform>/<config>/ \
  --result-filename c${C}_r${REP}.json
```
