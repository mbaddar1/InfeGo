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
| 7 | Jan 19–25 | (buffer + launch) | Polish 01–03, tag `v0.1`, publish and share | — |
| 8–9 | Jan 26–Feb 8 | [04 Quantization](docs/articles/04-quantization-from-scratch.md) | int8 absmax / zero-point, int4 group-wise packing, AWQ-style scale search, dequant matmul | Perplexity vs fp16; bitsandbytes / AWQ in vLLM |
| 10–11 | Feb 9–22 | [05 Attention kernels in Triton](docs/articles/05-attention-kernels-in-triton.md) | Fused softmax → tiled online-softmax (Flash-style) forward → paged decode kernel plugged into `nanoinfer` | `allclose` vs own PyTorch attention; speed vs SDPA / FlashAttention |
| 12 | Feb 23–Mar 1 | [06 Speculative decoding](docs/articles/06-speculative-decoding-from-scratch.md) | Draft/target loop, rejection-sampling verify, KV rollback in the paged cache | Output distribution matches target-only sampling; vLLM spec decode |
| 13 | Mar 2–7 | [07 Engine shoot-out on one 16 GB GPU](docs/articles/07-engine-shoot-out-on-one-16gb-gpu.md) | Nothing new: `nanoinfer` vs HF vs vLLM vs SGLang vs llama.cpp in the same harness; tag `v0.3` | The engines themselves; gap analysis |

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
- **Cadence:** one article every two weeks, mirrored to a blog and
  shared with one chart.

### Phase 1 exit criteria

- 7 articles published
- `infego` / `nanoinfer` `v0.3` with every number reproducible in one command
- A benchmark JSON dataset that Phase 2 (`infego recommend`) builds on
- ~50 GitHub stars

### Moved to Phase 2

- [Mixture of Experts: a performance perspective](docs/articles/backlog/mixture-of-experts-a-performance-perspective.md)
- [State space models (Mamba): a performance perspective](docs/articles/backlog/state-space-models-a-performance-perspective.md)

## Work items

### 0. Setup: harness and package skeleton (week 0)

- **Status:** planned
- **Code:** `infego/bench/`, `nanoinfer/`, `pyproject.toml`

**Tasks:**
- [ ] `pyproject.toml` with `infego` and `nanoinfer` packages and an `infego` CLI entry point
- [ ] Timing harness: CUDA events, warmup, repetitions, p50/p95/p99, peak memory
- [ ] Environment capture (GPU, driver, CUDA, torch, clocks, temperature)
- [ ] Write results in the [`benchmark_repo` schema](benchmark_repo/README.md)
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
