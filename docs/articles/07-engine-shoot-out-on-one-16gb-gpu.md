# Engine Shoot-out on One 16 GB GPU

> Status: planned (Phase 1, week 13)

**Question:** On one 16 GB GPU, which engine wins for which workload, and why is my from-scratch engine slower?

**Built from scratch in [`nanoinfer`](../../nanoinfer/README.md):** nothing new: `nanoinfer` runs next to HF, vLLM, SGLang and llama.cpp in the same `infego` harness.

**Validated against:** the engines themselves, with a gap analysis per optimization.

**Reproduce:** `infego reproduce shootout` (not available yet)
