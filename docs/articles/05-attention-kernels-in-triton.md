# Attention Kernels in Triton: Naive, Fused, Flash-style

> Status: planned (Phase 1, weeks 10–11)

**Question:** Where does attention time go, and how much do tiling and fusion recover?

**Built from scratch in [`nanoinfer`](../../nanoinfer/README.md):** Triton kernels: fused softmax, a tiled online-softmax (Flash-style) forward pass, and a paged decode kernel plugged into `nanoinfer`.

**Validated against:** `allclose` against the PyTorch attention in `nanoinfer`; speed vs PyTorch SDPA and FlashAttention.

**Reproduce:** `infego reproduce kernels` (not available yet)
