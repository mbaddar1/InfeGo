# InfeGo Articles

Phase 1 of the [roadmap](../../README.md#roadmap): each article builds
one inference technique from scratch in [`nanoinfer`](../../nanoinfer/README.md),
measures it on a 16 GB RTX A5000 laptop GPU, and checks it against a
production engine. Schedule: [workplan](../../workplan.md).

| # | Article | Weeks | Status |
|---|---------|-------|--------|
| 01 | [Transformers: A Performance Perspective](01-transformers-a-performance-perspective.md) | 1–2 | Draft |
| 02 | [The KV Cache: From Contiguous to PagedAttention](02-kv-cache-to-paged-attention.md) | 3–4 | Planned |
| 03 | [Continuous Batching: A Scheduler from Scratch](03-continuous-batching-from-scratch.md) | 5–6 | Planned |
| 04 | [Quantization from Scratch: int8, int4 and AWQ-style Scaling](04-quantization-from-scratch.md) | 8–9 | Planned |
| 05 | [Attention Kernels in Triton: Naive, Fused, Flash-style](05-attention-kernels-in-triton.md) | 10–11 | Planned |
| 06 | [Speculative Decoding from Scratch](06-speculative-decoding-from-scratch.md) | 12 | Planned |
| 07 | [Engine Shoot-out on One 16 GB GPU](07-engine-shoot-out-on-one-16gb-gpu.md) | 13 | Planned |

Article 01 also covers the quadratic cost of attention and prefill vs
decode, which were separate stubs before.

## Backlog (Phase 2)

- [Mixture of Experts: A Performance Perspective](backlog/mixture-of-experts-a-performance-perspective.md)
- [State Space Models (Mamba): A Performance Perspective](backlog/state-space-models-a-performance-perspective.md)

New articles use the next number: `NN-short-title.md`.
