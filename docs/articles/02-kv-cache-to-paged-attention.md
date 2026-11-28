# The KV Cache: From Contiguous to PagedAttention

> Status: planned (Phase 1, weeks 3–4)

**Question:** How much memory does the KV cache really cost, and why do serving engines page it?

**Built from scratch in [`nanoinfer`](../../nanoinfer/README.md):** a contiguous KV cache, then a block allocator, block table and paged attention (PyTorch gather version).

**Validated against:** identical tokens with and without the cache, measured memory vs the analytic formula, HF `past_key_values`.

**Reproduce:** `infego reproduce kv-cache` (not available yet)
