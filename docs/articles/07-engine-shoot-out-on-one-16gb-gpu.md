# Engine Shoot-out on One 16 GB GPU

> Status: planned (Phase 1, week 13)

**Question:** On one 16 GB GPU, which engine and settings win for which real problem (search, voice AI, structured extraction, chat / RAG), and why is my from-scratch engine slower?

**Built from scratch in [`nanoinfer`](../../nanoinfer/README.md):** nothing new: `nanoinfer` runs next to HF, vLLM, SGLang and llama.cpp in the same `infego` harness.

**Workloads:** all four [real workloads](../../benchmark_repo/workloads/README.md), each with its SLO and quality metric.

**Validated against:** the engines themselves, with a gap analysis per optimization and a "which engine and settings for which problem" table.

**Reproduce:** `infego reproduce shootout` (not available yet)
