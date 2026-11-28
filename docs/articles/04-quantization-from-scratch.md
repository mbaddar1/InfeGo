# Quantization from Scratch: int8, int4 and AWQ-style Scaling

> Status: planned (Phase 1, weeks 8–9)

**Question:** What do int8 and int4 weights buy on a 16 GB GPU, and what do they cost in quality?

**Built from scratch in [`nanoinfer`](../../nanoinfer/README.md):** int8 absmax and zero-point quantization, int4 group-wise packing, AWQ-style scale search, and a dequant matmul. (The A5000 is Ampere: no FP8 tensor cores.)

**Validated against:** perplexity vs fp16; bitsandbytes and AWQ in vLLM.

**Reproduce:** `infego reproduce quantization` (not available yet)
