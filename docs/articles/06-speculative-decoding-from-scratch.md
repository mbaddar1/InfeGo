# Speculative Decoding from Scratch

> Status: planned (Phase 1, week 12)

**Question:** When does a draft model actually speed up decoding?

**Built from scratch in [`nanoinfer`](../../nanoinfer/README.md):** the draft/target loop, rejection-sampling verification, and KV rollback in the paged cache.

**Validated against:** output distribution matches target-only sampling (statistical test); vLLM speculative decoding.

**Reproduce:** `infego reproduce specdec` (not available yet)
