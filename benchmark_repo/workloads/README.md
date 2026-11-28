# Workloads

Application workloads that InfeGo benchmarks on. Each one is a full
pipeline with realistic traffic, a latency target and a quality metric,
so results answer "what should I run for **this** problem?" rather than
"which engine is fastest on a synthetic prompt?".

Each workload gets a spec file here (`<name>.yaml`, added in Phase 1
week 0). A spec pins: models and revisions, dataset and sample, traffic
shape (concurrency or request rate, input/output lengths), the SLO, and
how quality is scored. Prompts and outputs are never stored in results,
only their shape.

Every target below fits on a single 16 GB GPU.

## 1. Semantic search

| | |
|---|---|
| **Pipeline** | Embed query → vector search → rerank top-k with a cross-encoder |
| **Candidate models** | `BAAI/bge-small-en-v1.5`, `intfloat/e5-base-v2`; reranker `BAAI/bge-reranker-base` |
| **Data** | A BEIR subset (e.g. SciFact, FiQA) |
| **SLO example** | p95 query latency < 50 ms at N queries/s |
| **Speed metrics** | Queries/s, p50/p95/p99 query latency per stage, indexing throughput (docs/s) |
| **Quality** | Recall@10, nDCG@10 |
| **Engines** | PyTorch, ONNX Runtime, Text Embeddings Inference, vLLM (embeddings) |

## 2. Voice AI understanding

| | |
|---|---|
| **Pipeline** | Streaming ASR → LLM (intent + reply) → TTS |
| **Candidate models** | Whisper small / distil-large-v3; Qwen2.5-1.5B/3B or Llama-3.2-3B; a small open TTS model |
| **Data** | LibriSpeech / Common Voice clips; a small set of spoken task requests with intent labels |
| **SLO example** | End-to-end turn latency (end of speech → first audio) < 800 ms |
| **Speed metrics** | Real-time factor, time to first transcript, LLM TTFT, time to first audio, per-stage breakdown |
| **Quality** | WER, intent accuracy |
| **Engines** | HF, faster-whisper, whisper.cpp; vLLM / llama.cpp for the LLM step |

## 3. Structured extraction

| | |
|---|---|
| **Pipeline** | Message or document → LLM with JSON-schema-constrained output |
| **Candidate models** | Qwen2.5-3B/7B (7B quantized), Llama-3.2-3B |
| **Data** | A labelled set of order-like messages or invoices with target JSON |
| **SLO example** | Throughput-oriented: max docs/s with p95 < 2 s |
| **Speed metrics** | Docs/s, p95 latency, cost per 1k docs, constrained-decoding overhead |
| **Quality** | JSON validity rate, field-level precision / recall |
| **Engines** | vLLM, SGLang, llama.cpp (grammar-constrained decoding) |

## 4. Chat / RAG answer

| | |
|---|---|
| **Pipeline** | Multi-turn chat; RAG variant adds 4k–16k tokens of retrieved context |
| **Candidate models** | Qwen2.5-0.5B (smoke), Llama-3.2-3B / Qwen2.5-3B (main) |
| **Data** | ShareGPT / LMSYS-Chat length distributions; LongBench for RAG |
| **SLO example** | p95 TTFT < 300 ms, p95 TPOT < 50 ms |
| **Speed metrics** | TTFT, TPOT, goodput at the SLO |
| **Quality** | Small eval set (e.g. MMLU / GSM8K subset) |
| **Engines** | `nanoinfer`, HF, vLLM, SGLang, llama.cpp |

Model and dataset choices are starting points. Check licenses and pin
revisions in the spec file before running.
