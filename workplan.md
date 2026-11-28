# InfeGo Workplan

Short loops of **learn → build → measure → write**. Each work item ships
an article, from-scratch code, and real measurements that anyone can
reproduce with one command.

## Phase 1: build it, measure it, check it against the real engine

**When:** a setup week plus 13 weeks, Mon 2026-12-01 → Sun 2027-03-07
**Hardware:** 16 GB RTX A5000 laptop GPU
**Models:** GPT-2, Qwen2.5-0.5B, Llama-3.2-1B / 3B

### The idea

Over Phase 1, build **one small inference engine by hand:
[`nanoinfer/`](nanoinfer/README.md)**. Each article adds one layer to it.
Real engines (HF, vLLM, SGLang, llama.cpp) are **oracles only**: they
check that the output is correct and give a speed target. The engine
is never built on top of them.

`nanoinfer` is a learning engine, not a new runtime. It follows the
[philosophy](README.md#philosophy-understand-first-add-layers-last):
understand first, then measure what real engines deliver. The
engine-neutral benchmark harness (`infego/bench`) is built alongside
it and is the seed of `infego bench` and the Phase 2 planner.

### Import rules for `nanoinfer`

| Allowed | Not allowed | Reference only (tests and benchmarks) |
|---------|-------------|---------------------------------------|
| `torch` tensors and ops, `triton`, `safetensors`, `tokenizers` | `transformers` modeling code, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, `generate()`, bitsandbytes, vLLM internals | HF `transformers`, vLLM, SGLang, llama.cpp |

### Every article follows the same template

1. **Question:** a practitioner question ("why is my decode slow?").
2. **From scratch:** the technique, written by hand in `nanoinfer`,
   with tests that check it against a reference.
3. **Measure:** latency percentiles, tokens/s and peak memory on the
   A5000, saved as JSON in [`benchmark_repo/`](benchmark_repo/README.md).
4. **Validate:** the same measurement on a production engine. Does the
   from-scratch version show the real effect, and how big is the gap?
5. **Decision rule:** "use X when Y". This is what practitioners keep.
6. **Reproduce:** `infego reproduce <article>` regenerates every table
   and plot in the article.

### Schedule

| Week | Dates | Article | Built from scratch in `nanoinfer` | Validated against |
|------|-------|---------|-----------------------------------|-------------------|
| 0 | Dec 1–7 | (setup) | Benchmark harness (`infego/bench`): CUDA-event timing, warmup, p50/p95/p99, peak memory, environment capture, JSON result schema; `infego reproduce` CLI; CI; `pyproject.toml` | `torch.cuda.Event` / Nsight on a plain matmul |
| 1–2 | Dec 8–21 | [01 Transformers: a performance perspective](docs/articles/01-transformers-a-performance-perspective.md) | Llama/Qwen forward pass from raw tensors (RMSNorm, RoPE, GQA attention, SwiGLU MLP, LM head), safetensors weight loader, greedy / top-p sampler, analytic FLOP and byte counter (roofline) | Logits `allclose` to HF for Llama-3.2-1B; identical greedy text |
| 3–4 | Dec 22–Jan 4 | [02 KV cache → PagedAttention](docs/articles/02-kv-cache-to-paged-attention.md) | Contiguous KV cache, then block allocator + block table + paged attention (PyTorch gather version) | Same tokens as without cache; measured memory vs formula; HF `past_key_values` |
| 5–6 | Jan 5–18 | [03 Continuous batching](docs/articles/03-continuous-batching-from-scratch.md) | Request queue, iteration-level scheduler, prefill/decode mixing, preemption when blocks run out, load generator | Same output as sequential; throughput–latency curve vs vLLM |
| 7 | Jan 19–25 | (buffer + launch) | Polish 01–03, tag `v0.1`, submit to Towards Data Science, share on HN / Reddit / LinkedIn | — |
| 8–9 | Jan 26–Feb 8 | [04 Quantization](docs/articles/04-quantization-from-scratch.md) | int8 absmax / zero-point, int4 group-wise packing, AWQ-style scale search, dequant matmul | Perplexity vs fp16; bitsandbytes / AWQ in vLLM |
| 10–11 | Feb 9–22 | [05 Attention kernels in Triton](docs/articles/05-attention-kernels-in-triton.md) | Fused softmax → tiled online-softmax (Flash-style) forward → paged decode kernel plugged into `nanoinfer` | `allclose` vs own PyTorch attention; speed vs SDPA / FlashAttention |
| 12 | Feb 23–Mar 1 | [06 Speculative decoding](docs/articles/06-speculative-decoding-from-scratch.md) | Draft/target loop, rejection-sampling verify, KV rollback in the paged cache | Output distribution matches target-only sampling; vLLM spec decode |
| 13 | Mar 2–7 | [07 Engine shoot-out on one 16 GB GPU](docs/articles/07-engine-shoot-out-on-one-16gb-gpu.md) | Nothing new: `nanoinfer` vs HF vs vLLM vs SGLang vs llama.cpp in the same harness; tag `v0.3` | The engines themselves; gap analysis |

### Real-workload track

Micro-benchmarks show *why* something is fast. Practitioners decide on
**their** workload. So every result is also measured on application
workloads: full pipelines with realistic traffic and a quality metric,
defined in [`benchmark_repo/workloads/`](benchmark_repo/workloads/README.md).

| Workload | Pipeline (fits on 16 GB) | Speed metrics | Quality metric | Engines compared |
|----------|--------------------------|---------------|----------------|------------------|
| **Semantic search** | Embedding model → vector search → cross-encoder reranker | Queries/s, p95 query latency, indexing throughput | Recall@10 / nDCG@10 | PyTorch, ONNX Runtime, Text Embeddings Inference, vLLM (embeddings) |
| **Voice AI understanding** | Streaming ASR (Whisper family) → LLM (intent + reply) → TTS | Real-time factor, time to first transcript, **end-to-end turn latency / time to first audio** | WER, intent accuracy | HF, faster-whisper, whisper.cpp; vLLM / llama.cpp for the LLM |
| **Structured extraction** | Document or message → LLM with constrained JSON output | Docs/s, p95 latency, cost per 1k docs | JSON validity, field-level accuracy | vLLM, SGLang, llama.cpp with grammar-constrained decoding |
| **Chat / RAG answer** | Multi-turn chat, long retrieved context | TTFT, TPOT, goodput at an SLO | Small eval set | `nanoinfer`, HF, vLLM, SGLang, llama.cpp |

How it fits into Phase 1 without adding weeks:

- **Week 0:** write the workload specs (models, datasets, traffic
  shape, SLOs, quality metric) and make `infego/bench` run a pipeline,
  not only a single model.
- **Articles 02–06:** each adds at least one real-workload measurement
  where the technique matters:

  | Article | Real workload measured |
  |---------|------------------------|
  | 02 KV cache | Chat with growing history |
  | 03 Continuous batching | Chat and the voice LLM step under concurrent users |
  | 04 Quantization | Extraction accuracy and search recall vs memory |
  | 05 Triton attention | RAG with long context |
  | 06 Speculative decoding | Voice turn latency |

- **Article 07:** becomes the full shoot-out across all four workloads,
  with a "which engine and settings for which problem" table.
- `nanoinfer` stays decoder-only LLM. Search encoders, ASR and TTS
  are benchmarked on existing engines through `infego bench`, which is
  the core of the product anyway.

Phase 2 follows up with one deep-dive article per workload (see
[Moved to Phase 2](#moved-to-phase-2)).

### Rules

- **Done when:** article published, `infego reproduce <name>` runs end
  to end, from-scratch tests pass, numbers in the article match the
  committed JSON.
- **Budget:** about 12–15 h/week.
- **Slip rule:** if more than one week behind at week 7, move article
  06 (speculative decoding) to Phase 2. Protect article 07: it bridges
  to the planner.
- **Ampere note:** the A5000 has no FP8 tensor cores. Article 04 says
  so explicitly and covers int8 / int4 instead.
- **Cadence:** one article every two weeks.

### Publishing and distribution

Articles live in [`docs/articles/`](docs/articles/README.md) and are also
published where ML practitioners read. Every published version links
back to the repo and its `infego reproduce` command.

| Channel | Role | Target |
|---------|------|--------|
| [Towards Data Science](https://towardsdatascience.com/submissions/) | Main external outlet: editor-reviewed, homepage placement, newsletter, paid author program | Submit every article; **at least 3 of 7 accepted** |
| [Hugging Face community blog](https://huggingface.co/docs/hub/main/blog-articles) | Reaches people who download models and run inference | Articles 01, 02, 04 and 07 (needs a PRO account or org write access) |
| LinkedIn | Professional reach; one chart plus 3–5 takeaways and a link | Every article |
| Hacker News, r/LocalLLaMA, r/MachineLearning | Bursts of traffic and stars | Articles 02, 05 and 07 (the most "show, don't tell" ones) |
| Personal blog or Substack | Canonical copy you own; email list | Every article |

**Rules:**
- Before submitting, read TDS's current submission guide and author
  terms, especially on prior publication and rights. If TDS wants to
  publish first, submit there first and publish elsewhere after its
  decision, with a canonical link where the platform allows.
- Write the article once, in the repo. Adapt the intro and length for
  each platform, never the numbers.
- Track each article's links, views, stars gained and feedback. The "Published at" column in the
  [article index](docs/articles/README.md) lists the links.

### Phase 1 exit criteria

- 7 articles published, at least 3 of them on Towards Data Science
- Results on all four real workloads (search, voice, extraction, chat) in `benchmark_repo/`
- `infego` / `nanoinfer` `v0.3` with every number reproducible in one command
- A benchmark JSON dataset that Phase 2 (`infego recommend`) builds on
- ~50 GitHub stars

### Moved to Phase 2

- [Mixture of Experts: a performance perspective](docs/articles/backlog/mixture-of-experts-a-performance-perspective.md)
- [State space models (Mamba): a performance perspective](docs/articles/backlog/state-space-models-a-performance-perspective.md)
- [Semantic search on one GPU: embeddings, rerankers and the latency budget](docs/articles/backlog/semantic-search-on-one-gpu.md)
- [Voice AI under 800 ms: profiling an ASR → LLM → TTS turn](docs/articles/backlog/voice-ai-turn-latency.md)
- [Structured extraction at scale: constrained decoding, throughput and accuracy](docs/articles/backlog/structured-extraction-at-scale.md)

## Work items

### 0. Setup: harness and package skeleton (week 0)

- **Status:** planned
- **Code:** `infego/bench/`, `nanoinfer/`, `pyproject.toml`

**Tasks:**
- [ ] `pyproject.toml` with `infego` and `nanoinfer` packages and an `infego` CLI entry point
- [ ] Timing harness: CUDA events, warmup, repetitions, p50/p95/p99, peak memory
- [ ] Environment capture (GPU, driver, CUDA, torch, clocks, temperature)
- [ ] Write results in the [`benchmark_repo` schema](benchmark_repo/README.md)
- [ ] Workload specs for search, voice, extraction and chat in [`benchmark_repo/workloads/`](benchmark_repo/workloads/README.md); harness can time a multi-stage pipeline per stage and end to end
- [ ] `infego reproduce <article>` command skeleton
- [ ] CI: lint + CPU-only unit tests

### 1. Transformers – an inference performance perspective (weeks 1–2)

- **Status:** in progress
- **Issue:** [mbaddar1/InfeGo#11](https://github.com/mbaddar1/InfeGo/issues/11)
- **Article:** [`docs/articles/01-transformers-a-performance-perspective.md`](docs/articles/01-transformers-a-performance-perspective.md)
- **Code:** `nanoinfer/model.py`, `nanoinfer/loader.py`, `nanoinfer/sampling.py`, `infego/roofline/`

**Goal:** explain the decoder-only transformer through inference cost,
where the FLOPs, memory and latency go, by writing its forward pass by
hand and measuring it on a 16 GB RTX A5000 laptop GPU.

**Tasks:**
- [ ] Outline the post: embeddings, attention, MLP, norm, LM head, and the cost of each
- [ ] Derive parameter count, FLOPs and bytes per token for each block; prefill vs decode arithmetic intensity
- [ ] Write the Llama/Qwen forward pass from raw tensors and a safetensors loader
- [ ] Test: logits `allclose` to HF for Llama-3.2-1B; identical greedy output
- [ ] Per-layer timing with the harness; latency and memory vs sequence length and batch size
- [ ] Roofline plot: measured vs analytic
- [ ] Plots + results table in the article; `infego reproduce transformers`
- [ ] Polish, link the code, publish

**Done when:** the article is published, the code runs end to end with a
single command, and the numbers in the post are reproducible from it.

### 2–7. See the schedule above

Each gets its own section here when it starts.
