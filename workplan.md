# InfeGo Workplan

Short loops of **learn → benchmark → write**. Each work item ships a blog post plus runnable code and real measurements.

## Work items

### 1. Transformers – an inference performance perspective

- **Type:** blog post + supporting code
- **Status:** in progress
- **Issue:** [mbaddar1/InfeGo#11](https://github.com/mbaddar1/InfeGo/issues/11)
- **Article:** [`docs/articles/01-transformers-a-performance-perspective.md`](docs/articles/01-transformers-a-performance-perspective.md)
- **Code:** `playground/transformers_perf/` (to be created)

**Goal:** explain the transformer architecture through the lens of inference cost — where the FLOPs, memory, and latency go — backed by measurements on a 16 GB RTX A5000 laptop GPU.

**Tasks:**
- [ ] Outline the post: blocks of a decoder-only transformer (embeddings, attention, MLP, norm, LM head) and the cost of each
- [ ] Derive parameter count and FLOPs per token for each block
- [ ] Write supporting code: per-layer timing/profiling of a small model (e.g. GPT-2 / a small Llama) with PyTorch profiler
- [ ] Measure latency and memory vs. sequence length and batch size
- [ ] Add plots and a results table to the article
- [ ] Polish, link the code, publish

**Done when:** the article is published, the code runs end to end with a single command, and the numbers in the post are reproducible from it.
