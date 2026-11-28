# Continuous Batching: A Scheduler from Scratch

> Status: planned (Phase 1, weeks 5–6)

**Question:** Why does continuous batching beat static batching, and what does it cost in latency?

**Built from scratch in [`nanoinfer`](../../nanoinfer/README.md):** a request queue, an iteration-level scheduler, prefill/decode mixing, preemption when KV blocks run out, and a load generator.

**Validated against:** identical output to sequential decoding; throughput vs latency curve compared with vLLM under the same load.

**Reproduce:** `infego reproduce batching` (not available yet)
