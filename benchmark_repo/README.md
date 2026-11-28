# benchmark_repo

InfeGo's measurement data. Every benchmark run ends up here as one
structured JSON file. This folder is the seed of the dataset that
InfeGo's recommendations and learned performance model will be built
on. See the [work doctrine](../README.md#work-doctrine).

## Layout

``` text
benchmark_repo/
├── README.md        this file: purpose, layout, result schema
├── results/         one JSON file per run (never edited after a run)
├── workloads/       workload definitions (input/output lengths, datasets, request rates)
└── scripts/         harness scripts that run benchmarks and write results
```

## Naming results

```
results/<date>_<engine>_<model>_<quant>_<workload>_<short-id>.json
```

Example: `results/2026-12-01_vllm_qwen2.5-3b_fp16_chat_a1b2.json`

## Result schema (v0, draft)

Every result file follows this shape. Add fields when needed, and bump
`schema_version` when a change breaks existing files.

``` json
{
  "schema_version": "0.1",
  "run_id": "a1b2c3d4",
  "timestamp": "2026-12-01T10:00:00Z",

  "hardware": {
    "gpu": "NVIDIA RTX A5000 Laptop GPU",
    "gpu_memory_gb": 16,
    "gpu_count": 1,
    "driver": "",
    "cuda": "",
    "cpu": "",
    "ram_gb": 0
  },

  "software": {
    "engine": "vllm",
    "engine_version": "",
    "python": "",
    "torch": ""
  },

  "model": {
    "name": "Qwen/Qwen2.5-3B-Instruct",
    "revision": "",
    "params_b": 3.0,
    "quantization": "fp16"
  },

  "config": {
    "max_model_len": 4096,
    "max_num_seqs": null,
    "max_num_batched_tokens": null,
    "prefix_caching": false,
    "extra": {}
  },

  "workload": {
    "name": "chat",
    "dataset": "sharegpt",
    "input_len": { "mean": 0, "p50": 0, "p95": 0 },
    "output_len": { "mean": 0, "p50": 0, "p95": 0 },
    "concurrency": 8,
    "request_rate": null,
    "num_requests": 200
  },

  "method": {
    "tool": "vllm bench serve",
    "warmup_requests": 20,
    "repetitions": 3
  },

  "metrics": {
    "ttft_ms": { "p50": 0, "p95": 0, "p99": 0 },
    "itl_ms": { "p50": 0, "p95": 0, "p99": 0 },
    "e2e_latency_ms": { "p50": 0, "p95": 0, "p99": 0 },
    "output_throughput_tok_s": 0,
    "request_throughput_req_s": 0,
    "goodput": { "slo": "p95 TTFT < 300ms", "req_s": 0 },
    "variance": { "output_throughput_stdev": 0 }
  },

  "telemetry": {
    "gpu_memory_peak_gb": 0,
    "gpu_util_mean_pct": 0,
    "power_mean_w": 0,
    "temp_max_c": 0,
    "sm_clock_min_mhz": 0,
    "throttling_detected": false
  },

  "quality": {
    "eval": null,
    "score": null
  },

  "notes": ""
}
```

## Rules

-   Results are **append-only**. A wrong run gets a new run with a note,
    not an edit.
-   Never store prompts or model outputs, only their shape (lengths,
    rates).
-   A result without hardware, software version, and model revision is
    not a valid result.
