# InfeGo

### The inference planner. Measured, not guessed.

InfeGo measures how models actually run on your hardware, learns from
every measurement, and decides the best way to serve them: which
engine, which quantization, which settings.

Developers should not need to become inference infrastructure experts
just to ship AI-powered applications, and they should not have to trust
benchmark numbers from someone else's GPU either.

> **You build AI applications. InfeGo finds the fastest, cheapest way to
> serve them, and shows you the evidence.**

------------------------------------------------------------------------

## The problem

Running models in production is more than choosing a model and calling
an API. Getting the right balance between latency, throughput, memory
usage, and cost can require substantial specialized work.

Teams may need to evaluate and combine techniques such as:

-   KV Caching
-   Quantization and low-precision execution
-   Batching and caching
-   Low-rank and other model-compression techniques
-   Speculative or other accelerated decoding methods
-   Runtime and serving-configuration tuning
-   Hardware-aware optimization

These techniques can deliver meaningful improvements, but they also
introduce complexity. The best choice depends on the model, workload,
hardware, quality requirements, and serving environment.

**InfeGo's goal is to make that complexity easier to manage.**

## The vision

### Measured, not guessed.

InfeGo is an **inference planner**. You give it a model, your hardware,
and a performance target. InfeGo measures how the model actually runs,
learns from every measurement, and decides the best way to serve it.

``` text
infego plan --model Llama-3.1-8B-Instruct --gpu "RTX A5000 16GB" \
            --workload chat --slo "p95 TTFT < 300ms"
```

> Illustrative only: this command is the target experience, not an
> available feature yet.

### Why measurement is the product

Advice about inference is cheap. Blog posts, docs, and AI assistants
will all suggest a configuration. But that advice is usually:

-   **Benchmarked on different hardware**, often data-center GPUs rather
    than the card you have.
-   **Out of date.** Inference engines ship significant performance
    changes every few weeks.
-   **Unverifiable.** A plausible number is not a measured one.

InfeGo's answers are backed by real runs on real hardware, and every
recommendation shows the measurements and trade-offs behind it.

### How it works: a learning loop

``` text
   benchmark ──> structured results ──> performance model
       ^                                       |
       |                                       v
   verify <──────── configuration <──── decide what to try
```

1.  **Benchmark:** run reproducible experiments across engines,
    quantization formats, and serving settings, with pinned versions
    and a fixed result schema.
2.  **Learn:** a performance model predicts latency, throughput, and
    memory for configurations, including ones not measured yet.
3.  **Decide:** instead of brute-forcing thousands of combinations,
    InfeGo picks the few most likely to meet the target and measures
    them.
4.  **Verify:** a recommendation is marked as *measured* or *predicted*,
    so you always know how much to trust it.

Every run makes the next recommendation better.

### Engine-neutral

InfeGo does not try to replace vLLM, SGLang, TensorRT-LLM, llama.cpp, or
other runtimes. It sits above them and helps you choose and configure
the right one for your case. Each new engine or optimization technique
widens the search space InfeGo can explore.

### Built for the hardware most people have

Most published optimization work targets large GPU clusters. InfeGo
starts with **single-GPU and small-team setups**: consumer and
workstation cards with 16–48 GB of memory, where choosing the right
configuration decides whether a model fits at all.

### Community-powered data

The long-term goal is an open, growing dataset of inference
measurements across GPUs, engines, and workloads:

-   Run `infego bench` and get a useful report on your own setup.
-   Optionally share anonymized results. Sharing is **opt-in** and
    includes only hardware, software versions, settings, workload
    *shape* (for example prompt-length distributions), and metrics.
    Prompts and outputs are never collected.
-   Aggregated results are published openly, as a public reference for
    the best measured configuration per model and GPU.

### Where it is heading

The intended progression follows the [roadmap](#roadmap):

1.  **Bench:** trustworthy, reproducible measurements.
2.  **Recommend:** a configuration for a given model, hardware, and
    target, with the evidence behind it.
3.  **Configure:** generate and validate the launch configuration for
    the chosen runtime.
4.  **Adapt:** later, keep a deployment near its optimum as traffic and
    conditions change.

InfeGo's measurements are also designed to be machine-readable, so AI
agents and coding assistants can call InfeGo for grounded answers
instead of guessing.

## Philosophy: understand first, add layers last

> *"Sometimes the best engineering isn't adding another layer."*
> [Mark Emerson](https://www.linkedin.com/posts/mark-emerson-60148312_softwareengineering-coding-cleancode-share-7530580291183091712-Qeg1/)

Every tool built to remove complexity tends to grow new abstractions on
top. InfeGo resists that:

1.  **Understand** the architecture and where its bottlenecks really are.
2.  **Measure** what existing runtimes already deliver when configured
    well.
3.  **Build** only where the evidence shows a real gap.

Often the best optimization is the right setting in a runtime you
already use, not a new layer.

## Design principles

-   **Developer-first:** Keep the developer experience simple; expose
    complexity only when it is useful.
-   **Evidence-driven:** Measure optimizations against a baseline rather
    than assuming they help.
-   **Workload-aware:** Select strategies based on the model, hardware,
    traffic pattern, and quality constraints.
-   **Composable:** Make it possible to adopt individual capabilities
    without committing to an all-or-nothing stack.
-   **Transparent:** Make performance changes and trade-offs observable.
-   **Practical:** Prioritize improvements that matter in real
    production workloads.

## What InfeGo is intended to optimize

The project is exploring an extensible set of inference optimization
capabilities, including:

  -----------------------------------------------------------------------
  Area                                Objective
  ----------------------------------- -----------------------------------
  Model and strategy selection        Find an appropriate serving
                                      strategy for a workload

  Quantization                        Reduce memory and compute
                                      requirements where quality permits

  Low-rank optimization               Explore lower-cost computation for
                                      suitable model components

  Batching and caching                Improve serving efficiency under
                                      relevant traffic patterns

  Runtime configuration               Tune execution settings for the
                                      target environment

  Performance evaluation              Compare latency, throughput,
                                      memory, and quality against a
                                      baseline
  -----------------------------------------------------------------------

These are areas of exploration, not a claim that every capability is
currently implemented.

## Reference workloads

Inference optimizations only mean something relative to a workload. A
technique that helps a chatbot can hurt a batch summarization job.
InfeGo is benchmarked against a small set of reference workloads, each
picked to stress a different bottleneck. Together they also work as a
learning path and as portfolio evidence.

The first targets fit on a single 16 GB GPU (the development machine is
an RTX A5000 Laptop GPU). Larger setups come later.

### Recommended models

| Role            | Model                                   | Why                                           |
|-----------------|-----------------------------------------|-----------------------------------------------|
| Smoke test      | `facebook/opt-125m`, `Qwen2.5-0.5B`     | Fast iteration, debugging, CI                 |
| Main target     | `Llama-3.2-3B-Instruct`, `Qwen2.5-3B`   | Realistic, FP16 fits in 16 GB with KV headroom |
| Stretch target  | `Llama-3.1-8B-Instruct` / `Qwen2.5-7B`  | Needs INT8/INT4 (AWQ, GPTQ) on 16 GB, so it makes a good quantization case study |
| Draft model     | `Qwen2.5-0.5B` paired with a 3B/7B one  | Speculative decoding experiments              |

### Workloads

| # | Workload | Traffic shape | Main bottleneck | Optimizations it showcases | Key metrics |
|---|----------|---------------|-----------------|----------------------------|-------------|
| 1 | **Interactive chat** (ShareGPT-style multi-turn) | Short/medium prompts, medium outputs, many concurrent users | Decode (memory-bandwidth bound), KV memory | KV cache, PagedAttention, continuous batching, prefix caching for multi-turn history | TTFT, ITL / TPOT, p50/p99 latency, goodput at an SLO |
| 2 | **RAG / long-context QA** | Long prompts (4k–32k tokens), short answers | Prefill (compute bound), KV cache size | Chunked prefill, prefix caching of shared documents, KV-cache quantization (FP8) | TTFT vs. prompt length, max context per GPU |
| 3 | **Shared system prompt / agent loop** | Same long system prompt and tools, many requests | Redundant prefill | Automatic prefix caching, prompt-cache hit rate | TTFT with and without caching, cache hit rate |
| 4 | **Offline batch** (summarization, classification, synthetic data) | Large batch, latency doesn't matter | Throughput, GPU utilization | Large batches, quantization, max `num_seqs`/`max_num_batched_tokens` tuning | Tokens/s, cost per 1M tokens, GPU utilization |
| 5 | **Code completion** | Short prompts, short outputs, very latency sensitive | Per-token latency at batch size 1 | Speculative decoding (draft model, n-gram, EAGLE), CUDA graphs | ITL at batch 1, acceptance rate, speedup |
| 6 | **Structured output / tool calling** | JSON schema constrained | Constrained-decoding overhead | Guided decoding backends (xgrammar, outlines) | Overhead vs. unconstrained decoding, validity rate |

### Suggested order

1. **Baseline (workload 4 + workload 1):** run vLLM with default
   settings, measure throughput and latency, and set up the benchmark
   harness (`vllm bench serve`, GuideLLM, or a custom script).
2. **KV caching from scratch:** a naive HF `generate` loop with and
   without a KV cache, showing that decode cost changes from O(n²) to
   O(n) (`playground/kv_caching`).
3. **Batching (workload 1):** static vs. continuous batching, and a
   concurrency sweep that plots throughput against p99 latency.
4. **Prefix caching (workloads 2 + 3):** TTFT reduction from shared
   prefixes.
5. **Quantization (stretch model):** FP16 vs. FP8 vs. AWQ/GPTQ INT4,
   covering memory, speed, and quality (perplexity or a small eval such
   as MMLU or GSM8K subsets).
6. **Speculative decoding (workload 5):** speedup vs. acceptance rate
   for different draft strategies.
7. **Workload-aware selection:** feed the results into InfeGo so it
   recommends a configuration from a workload profile. This step is the
   core of the product.

### Datasets

-   **ShareGPT** / **LMSYS-Chat-1M**: realistic chat prompt and output
    length distributions
-   **LongBench**, or a document set for RAG: long-context prefill
-   **HumanEval** / **MBPP** prompts: code completion
-   **CNN/DailyMail**, or arXiv abstracts: batch summarization
-   **Synthetic random-length prompts**: controlled sweeps (fixed input
    and output lengths)

### What makes it portfolio-grade

-   Every optimization is measured against the **same baseline**, with
    the hardware, model, and runtime versions pinned.
-   Report **latency percentiles and goodput under an SLO**, not just
    average tokens/s.
-   Include **quality checks** alongside any speed gain from
    quantization or speculative decoding.
-   Publish **plots and short write-ups** for each experiment (see
    [`docs/articles/`](docs/articles/README.md)).
-   Show the **trade-off curves** (throughput vs. latency, memory vs.
    quality). They show the engineering judgment InfeGo is meant to
    automate.

## How it fits into an inference stack

At a high level, the intended flow is:

``` text
              AI application
                    |
                    v
                 InfeGo
       +---------------------------+
       | Workload understanding   |
       | Optimization strategies  |
       | Evaluation and selection |
       | Performance measurement |
       +---------------------------+
                    |
                    v
          Inference runtime
                    |
                    v
               Model + GPU
```

InfeGo is intended to complement inference runtimes and serving
systems, not force developers to replace their existing infrastructure.
The integration approach and supported runtimes will be defined as the
project develops.

## Project status

**InfeGo is under active development.**

The current focus is to establish a practical foundation, validate
optimization strategies with reproducible benchmarks, and learn from
developers operating real inference workloads.

Interfaces, supported runtimes, and feature availability may change as
the project evolves. Check the repository's releases and documentation
for the current implementation status.

## Work doctrine

How InfeGo is built, right now: **learn by benchmarking, and write as
you go.** Every experiment produces three things:

1.  **Understanding:** why the numbers look the way they do.
2.  **Data:** structured results added to
    [`benchmark_repo/`](benchmark_repo/README.md).
3.  **Writing:** an article or section in
    [`docs/articles/`](docs/articles/README.md).

### Rules

-   **Measure before building.** Use existing tools (`vllm bench`,
    GuideLLM, `llama-bench`, SGLang `bench_serving`) first. Build only
    where they fall short.
-   **Every run is saved as structured JSON** with the fixed result
    schema: hardware, software versions, model, quantization, settings,
    workload shape, metrics, and telemetry. Results that live only in a
    notebook don't count.
-   **Pin everything:** GPU, driver, CUDA, engine version, and model
    revision.
-   **Trustworthy numbers only:** warmup runs, multiple repetitions, and
    reported variance. Report percentiles (p50/p95/p99), not averages.
-   **Watch the hardware:** log clocks, temperature, and power during
    runs. On a laptop GPU, thermal throttling can skew results by a lot.
-   **Speed is never reported without quality** when a technique can
    change model output (quantization, speculative decoding).
-   **Optimize for goodput:** throughput while meeting a latency target
    is the number decisions are based on.

### What to measure

| Metric | What it reflects |
|--------|------------------|
| TTFT (time to first token) | Prefill cost |
| ITL / TPOT (time per output token) | Decode speed, memory-bandwidth bound |
| Throughput (tokens/s) | Overall serving capacity |
| p50 / p95 / p99 latency | What users actually experience |
| Goodput at an SLO | Useful throughput within a latency target |
| GPU memory, utilization, power, temperature | Hardware limits and measurement validity |

### First configuration space

-   **Models:** Qwen2.5-0.5B (smoke test), Llama-3.2-3B / Qwen2.5-3B
    (main), 7–8B quantized (stretch).
-   **Engines:** vLLM and llama.cpp first, SGLang later.
-   **Quantization:** FP16, FP8, AWQ / GPTQ INT4, GGUF Q4 / Q8.
-   **Settings:** concurrency, max batched tokens, context length,
    prefix caching on/off.

### Build to understand

To understand what production engines do, InfeGo builds each technique
once by hand in [`nanoinfer/`](nanoinfer/README.md), a small
from-scratch inference engine, and then measures the real engines
against it. `nanoinfer` is a learning tool and a correctness baseline,
not a new runtime: the "measure before building" rule applies to
InfeGo the product.

### Rhythm

Phase 1 (Dec 2026 – early Mar 2027) ships one article about every two
weeks. Each article builds a technique from scratch, measures it, and
checks it against a production engine. The week-by-week schedule is in
the [workplan](workplan.md).

| Weeks | Article | Built from scratch |
|-------|---------|--------------------|
| 0 | (setup) | Benchmark harness and result schema |
| 1–2 | Transformers: a performance perspective | Llama/Qwen forward pass, loader, sampler, roofline counter |
| 3–4 | KV cache → PagedAttention | Contiguous and paged KV cache |
| 5–6 | Continuous batching | Iteration-level scheduler |
| 7 | (buffer + `v0.1` launch) | — |
| 8–9 | Quantization | int8 / int4, AWQ-style scaling |
| 10–11 | Attention kernels in Triton | Fused and Flash-style kernels |
| 12 | Speculative decoding | Draft/verify loop |
| 13 | Engine shoot-out on one 16 GB GPU | (compares `nanoinfer`, HF, vLLM, SGLang, llama.cpp) |

The plan is a guide, not a commitment. It changes when the measurements
say something unexpected.

## Roadmap

Built in short loops: **learn → build → measure → write**, one
bottleneck at a time.

| Phase | Goal | Output |
|-------|------|--------|
| **1. Foundations + Bench** (Dec 2026 – Mar 2027) | Understand each core inference technique by building it from scratch, and measure real engines on your own hardware with one command | [Article series](docs/articles/README.md), [`nanoinfer`](nanoinfer/README.md), `infego bench` harness and `infego reproduce`; see the [workplan](workplan.md) |
| **2. Recommend** | Suggest the best configuration for a given model, hardware, and workload | `infego recommend`; MoE and SSM articles |
| **3. Optimize** | Apply and auto-tune configurations, and fill the gaps found in phases 1–2 | `infego optimize` |
| **4. Production** | More runtimes, observability, managed service | Integrations, hosted offering |

Roadmap items are goals, not delivery commitments.

## Articles

Each article builds one inference technique from scratch, measures it
on a 16 GB GPU, and checks it against a production engine. See the
[full series](docs/articles/README.md).

1.  [Transformers: A Performance Perspective](docs/articles/01-transformers-a-performance-perspective.md)
2.  [The KV Cache: From Contiguous to PagedAttention](docs/articles/02-kv-cache-to-paged-attention.md)
3.  [Continuous Batching: A Scheduler from Scratch](docs/articles/03-continuous-batching-from-scratch.md)
4.  [Quantization from Scratch: int8, int4 and AWQ-style Scaling](docs/articles/04-quantization-from-scratch.md)
5.  [Attention Kernels in Triton: Naive, Fused, Flash-style](docs/articles/05-attention-kernels-in-triton.md)
6.  [Speculative Decoding from Scratch](docs/articles/06-speculative-decoding-from-scratch.md)
7.  [Engine Shoot-out on One 16 GB GPU](docs/articles/07-engine-shoot-out-on-one-16gb-gpu.md)

## Who is InfeGo for?

InfeGo is being developed with the following users in mind:

-   AI application developers who want simpler model serving
-   ML engineers responsible for inference performance
-   Startups looking to manage inference cost without building a large
    optimization team
-   Platform and infrastructure teams operating model-serving systems
-   Teams that want to benchmark and evaluate optimization strategies
    systematically

## Get involved

InfeGo is being shaped around real engineering problems. Feedback from
developers and teams working with production inference systems is
especially valuable.

If you are working on model serving, inference infrastructure, or AI
applications, I'd like to hear:

-   Which part of inference optimization consumes the most engineering
    time?
-   What are your biggest constraints: latency, throughput, GPU memory,
    cost, or output quality?
-   Which runtimes, models, and hardware environments do you use?
-   What would make an optimization layer genuinely useful in your
    workflow?

Open an issue or start a discussion in this repository. You can also get
in touch to discuss an integration or an early use case.

## The idea in one sentence

**InfeGo measures, learns, and decides the best way to serve your model
on your hardware, so you can focus on building AI applications.**

------------------------------------------------------------------------

*InfeGo is an evolving project. Features and integrations described as
goals or areas of exploration should not be interpreted as currently
available functionality.*
