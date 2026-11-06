# InferGo

**An inference optimization layer for vLLM.**

InferGo aims to help developers get more performance from the hardware they already have. The project starts with systematic experimentation, then turns validated findings into practical optimization automation.

> Measure first. Understand the workload. Validate improvements. Automate what works.

## Roadmap

| Phase | Focus | Deliverables |
|---|---|---|
| **1. Experimentation** | Benchmark vLLM across different workloads and supported optimization techniques | Research articles, optimization cheat sheet, reproducible benchmark repository |
| **2. Automation** | Turn repeatable analysis and validated recommendations into tools | Profiler, bottleneck analysis, configuration advisor, automated experiments |

---

## Phase 1 — Experimentation

**Goal:** Understand how workload characteristics and vLLM optimization techniques affect inference performance.

### What to test

| Dimension | Examples |
|---|---|
| Workloads | Short requests, long context, generation-heavy, mixed, bursty |
| Request patterns | Input/output lengths, concurrency, arrival rate, prefix reuse |
| Optimizations | Batching, KV cache, chunked prefill, prefix caching, quantization, speculative decoding, parallelism |
| Constraints | Available GPU memory, latency targets, throughput and quality requirements |

Start with a small set of controlled experiments. Record the model, hardware, software versions, exact configuration, workload, and benchmark methodology.

### Metrics

- Time to first token (TTFT)
- Time per output token (TPOT) / inter-token latency (ITL)
- P50/P95/P99 latency
- Output tokens/sec and requests/sec
- GPU memory and utilization
- Quality impact where relevant

Evaluate trade-offs, not throughput alone. A faster configuration is not useful if it violates latency or quality requirements.

### Shareable outputs

**1. Technical articles**

Publish data-backed observations about vLLM performance under different workloads. Explain the setup, results, trade-offs, recommendations, and limitations.

**2. Actionable optimization cheat sheet**

A concise reference mapping common symptoms to investigations and possible experiments.

| Symptom | Investigate |
|---|---|
| High TTFT | Prompt length, prefill, queueing |
| High ITL / TPOT | Decode cost, memory bandwidth, contention |
| Frequent preemption | KV-cache and memory pressure |
| Low throughput | Workload, batching, utilization |
| Repeated prefixes | Prefix-cache hit rate |
| Long prompts affect short requests | Chunked prefill |

Treat these as diagnostic starting points; publish recommendations only when supported by measurements.

**3. Reproducible benchmark repository**

Publish experiment scripts, configurations, workload definitions, results, plots, environment details, and reproduction instructions.

### Suggested first six weeks

| Week | Focus |
|---|---|
| 1 | Benchmark setup and baseline |
| 2 | Workload characterization |
| 3 | Batching and KV-cache experiments |
| 4 | Prefix caching and chunked prefill |
| 5 | Quantization and speculative decoding |
| 6 | Findings, articles, and cheat sheet v1 |

---

## Phase 2 — Automation

**Goal:** Automate the analysis and optimization decisions that proved useful during Phase 1.

### Potential capabilities

1. **Workload analyzer:** characterize request lengths, concurrency, traffic patterns, and prefix reuse.
2. **Profiler:** collect vLLM, GPU, memory, and latency metrics.
3. **Bottleneck analyzer:** identify likely compute, memory, scheduling, or communication constraints.
4. **Configuration advisor:** recommend relevant vLLM settings to test.
5. **Experiment runner:** benchmark candidate configurations against a baseline.
6. **Results analyzer:** compare performance while enforcing latency, memory, and quality constraints.

### Possible API

```python
from infergo import Profiler, Optimizer

report = Profiler(backend="vllm").analyze(
    model="MODEL_NAME",
    workload="workload.json"
)

recommendations = Optimizer(report).recommend(
    objectives=["latency", "throughput"]
)

results = recommendations.benchmark()
results.compare_against_baseline()
```

This is an illustrative API; finalize it based on what the experiments demonstrate.

### Longer-term differentiation

Explore research-driven optimizations such as:

- Layer-wise low-rank compression and adaptive rank allocation.
- Sensitivity-aware mixed precision.
- Workload-aware model compression.

Validate actual end-to-end latency and quality; fewer parameters or FLOPs do not automatically mean faster inference.

---

## Immediate next steps

1. Set up the `infergo-lab` repository.
2. Choose one representative model and establish a reproducible vLLM baseline.
3. Test short, long-context, generation-heavy, mixed, and bursty workloads.
4. Capture latency, throughput, memory, and utilization metrics.
5. Start with batching and KV-cache experiments.
6. Publish the first evidence-backed findings.

## Guiding principles

- Optimize for the user's existing hardware.
- Make workload-specific recommendations, not universal claims.
- Use reproducible experiments and measured results.
- Keep vLLM as the execution engine; avoid rebuilding its capabilities.
- Start with diagnosis and recommendations; automate progressively.

**Experiments → Evidence → Articles + Cheat Sheet + Benchmarks → Automation → InferGo**
