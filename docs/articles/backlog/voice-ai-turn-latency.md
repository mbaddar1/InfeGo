# Voice AI Under 800 ms: Profiling an ASR → LLM → TTS Turn

> Status: backlog (Phase 2)

**Question:** Where does the time go in one voice-agent turn, and which optimizations bring it under the budget?

**Workload:** [Voice AI understanding](../../../benchmark_repo/workloads/README.md)

**Measured:** per-stage latency (ASR, LLM TTFT, TTS first audio), streaming vs batch ASR, faster-whisper vs whisper.cpp vs HF, LLM quantization and speculative decoding, against WER and intent accuracy.
