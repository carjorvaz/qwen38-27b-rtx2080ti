# Qwen3.8-27B on a 22 GB RTX 2080 Ti

A fork of [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) that ports its
vLLM 0.28.0 serving stack to Turing (SM75). One RTX 2080 Ti with 22 GB runs
Qwen3.8-27B at 262,144 tokens of context out of a 5.5 GiB int4 KV pool, with
MTP speculative decoding, prefix caching and an int8 prefill path.

Measured on that card at 280 W, single stream, real prompts:

    cold 32k prefill      31.9 s
    cold 253k prefill     625 s, 0 preemptions
    decode                112 tok/s at 2k, 52.9 ms/step at 128k
    context / KV pool     262,144 tokens / 5.5 GiB int4
    quality               10.88 PPL wikitext-2, 94.5% GSM8K (200 questions)

`patches-turing/` is the port, applied after upstream's `patches/`. Each patch
carries its own measurements in its header. The series-wide numbers, including
what was measured and rejected, are in [docs/turing-2080ti.md](docs/turing-2080ti.md).
The optional prefill work, including bounded KV staging and transient int8
GEMMs, is in [docs/turing-prefill.md](docs/turing-prefill.md).

## Running it

Apply `patches/` and then `patches-turing/` in `patches-turing/series` order to
a vLLM 0.28.0 checkout, then:

```bash
vllm serve models/Qwen3.8-27B-W4A16-AutoRound \
  --served-model-name qwen3.8-27b --dtype float16 --language-model-only \
  --attention-backend TRITON_ATTN --kv-cache-dtype int4_per_token_head \
  --mamba-cache-dtype float16 --mamba-ssm-cache-dtype float16 --mamba-cache-mode align \
  --kv-cache-memory 5905580032 --max-model-len 262144 \
  --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --enable-prefix-caching --prefix-match-unit 16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"attention_backend":"TRITON_ATTN","draft_sample_method":"probabilistic"}' \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+rms_norm","+silu_and_mul"],"max_cudagraph_capture_size":8}' \
  --async-scheduling
```

Optional prefill settings, all off by default:

    VLLM_TURING_PREFILL_WINDOW=16384   bound the FP16 attention staging
    VLLM_TURING_EXTEND_MAX=32          split-KV path for short extensions
    VLLM_TURING_PREFILL_INT8=mlp       transient int8 GEMMs; +0.97% PPL

Model preparation (requantized embeddings and lm_head, int4 MTP draft) is in
[prepare/](prepare/). `single-user/` has a launcher and a systemd unit, and
`bench/` has the harnesses the numbers above came from. There is no room on
this card for batch mode's second KV pool.

## Credit

Fork of [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen), Apache-2.0.
Upstream's 3090 stack, model preparation scripts and benchmark harnesses are
what this port builds on.
