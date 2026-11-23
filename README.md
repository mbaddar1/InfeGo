# InfeGo

### Hide inference optimization complexity. Let developers focus on building.

InfeGo is an intelligent inference optimization layer designed to make
AI model serving simpler, faster, and more cost-efficient.

Developers should not need to become inference infrastructure experts
just to ship AI-powered applications. InfeGo aims to take care of the
complexity behind inference optimization so teams can focus on building
products.

> **You build AI applications. InfeGo takes care of inference
> optimization.**

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

InfeGo aims to provide a smart layer between AI applications and
inference infrastructure.

Instead of requiring developers to manually investigate, select,
configure, and maintain every optimization, InfeGo will help identify
suitable strategies and make them easier to evaluate and apply.

The intended experience:

1.  **Connect** an existing model and inference setup.
2.  **Understand** the workload and its performance requirements.
3.  **Evaluate** relevant optimization opportunities.
4.  **Apply** suitable optimizations through a consistent workflow.
5.  **Measure** the impact on latency, throughput, resource usage, and
    cost.

The long-term objective is to make inference optimization an
increasingly seamless part of developing and deploying AI applications.

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

## Roadmap

Built in short loops: **learn → benchmark → write**, one bottleneck at a
time.

| Phase | Goal | Output |
|-------|------|--------|
| **0. Foundations** | Understand architectures from an inference angle | [Article series](docs/articles/README.md): Transformers, attention, autoregressive decoding, KV cache, MoE, SSMs (Mamba), and where each one hits its bottlenecks |
| **1. Bench** | Benchmark engines and optimizations on your own hardware with one command | `infego bench`, reproducible reports, one write-up per experiment |
| **2. Recommend** | Suggest the best configuration for a given model, hardware, and workload | `infego recommend` |
| **3. Optimize** | Apply and auto-tune configurations, and fill the gaps found in phases 1–2 | `infego optimize` |
| **4. Production** | More runtimes, observability, managed service | Integrations, hosted offering |

Roadmap items are goals, not delivery commitments.

## Articles

Architectures explained from an inference-performance angle. See the
[full series](docs/articles/README.md).

1.  [Transformers: A Performance Perspective](docs/articles/01-transformers-a-performance-perspective.md)

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

**InfeGo hides inference optimization complexity so developers can
focus on building AI applications.**

------------------------------------------------------------------------

*InfeGo is an evolving project. Features and integrations described as
goals or areas of exploration should not be interpreted as currently
available functionality.*
