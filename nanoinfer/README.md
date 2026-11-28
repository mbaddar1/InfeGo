# nanoinfer

A small LLM inference engine written from scratch, one layer per
article. It exists to understand where inference time and memory go,
not to replace vLLM, SGLang or llama.cpp.

See the [Phase 1 workplan](../workplan.md) for the schedule.

## Rules

- **Allowed imports:** `torch` tensors and ops, `triton`, `safetensors`,
  `tokenizers`.
- **Not allowed:** `transformers` modeling code,
  `nn.MultiheadAttention`, `F.scaled_dot_product_attention`,
  `generate()`, bitsandbytes, vLLM internals.
- **Reference only:** HF `transformers`, vLLM, SGLang and llama.cpp are
  used in tests and benchmarks to check correctness and set a speed
  target, never as building blocks.
- **Every layer is tested** against a reference (logits `allclose`,
  identical greedy output, or a statistical test for sampling).
- **Readable over clever:** small files, short functions, comments that
  explain the inference cost of what the code does.

## Planned layout

| Module | Article | What it adds |
|--------|---------|--------------|
| `model.py`, `loader.py`, `sampling.py` | 01 | Llama/Qwen forward pass, safetensors loader, sampler |
| `kv_cache.py`, `paged.py` | 02 | Contiguous and paged KV cache, block allocator |
| `scheduler.py` | 03 | Continuous batching scheduler |
| `quant.py` | 04 | int8 / int4 weight quantization, AWQ-style scaling |
| `kernels/` | 05 | Triton attention kernels |
| `specdec.py` | 06 | Speculative decoding |

The KV-cache experiments in
[`playground/kv_caching/`](../playground/kv_caching/) are the starting
point for article 02.
