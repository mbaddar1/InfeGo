# Structured Extraction at Scale: Constrained Decoding, Throughput and Accuracy

> Status: backlog (Phase 2)

**Question:** How many documents per second can one GPU turn into valid JSON, and what does constrained decoding cost?

**Workload:** [Structured extraction](../../../benchmark_repo/workloads/README.md)

**Measured:** vLLM vs SGLang vs llama.cpp with grammar-constrained decoding; model size and quantization vs docs/s, JSON validity and field accuracy.
