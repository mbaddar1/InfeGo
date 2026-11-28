# Semantic Search on One GPU: Embeddings, Rerankers and the Latency Budget

> Status: backlog (Phase 2)

**Question:** How should a small team serve embeddings and a reranker to hit a search latency target, and what does each choice cost in recall?

**Workload:** [Semantic search](../../../benchmark_repo/workloads/README.md)

**Measured:** embedding and reranker models across PyTorch, ONNX Runtime, Text Embeddings Inference and vLLM; batch size, sequence length and quantization vs queries/s, p95 latency and recall@10.
