# Qwen3.8-27B on a 22 GB RTX 2080 Ti

A fork of [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) that ports its
vLLM 0.29.0 serving stack to Turing (SM75). One RTX 2080 Ti with 22 GB runs
Qwen3.8-27B at 262,144 tokens of context out of a 5.5 GiB int4 KV pool, with
MTP speculative decoding, prefix caching and an int8 prefill path.

Measured on that card at 280 W, single stream, on the vLLM 0.29.0 stack (V2
runner, `VLLM_MTP_LOOKUP=1`, MTP k=4). Decode uses chat-shaped Wikitext-2
prompts, greedy sampling and 128 output tokens (`qwen38-sm75-bench
--text-file`):

Decode:

| context | tok/s |
| ---: | ---: |
| 2k | 113.4 |
| 8k | 112.4 |
| 32k | 99.8 |
| 64k | 80.6 |
| 128k | 80.4 |
| 253k | 51.6 |

Tail prefill, extending an existing session by 2048 tokens
(`bench/turing_api_bench.py --corpus wikitext --tails 2048`):

| context depth | tok/s |
| ---: | ---: |
| 32k | 729 |
| 64k | 525 |
| 253k | ~210 |

From an empty cache, 32k took 31.9 s (1,028 tok/s) and 253k took 628 s
(404 tok/s, 0 preemptions).

Quality with the int4 KV cache: 10.88 PPL on wikitext-2 and 94.8% on GSM8K
(500 questions).

`patches-turing/` is the port, applied after upstream's `patches/series`. Each
patch carries its own measurements in its header. The
[0.29.0 port notes](docs/turing-0.29-port.md) cover validation. The older
0.28.0 results and patch-by-patch measurements are in
[docs/turing-2080ti.md](docs/turing-2080ti.md); optional prefill work is in
[docs/turing-prefill.md](docs/turing-prefill.md).

## Running it

Apply `patches/series` and then `patches-turing/series` to a vLLM 0.29.0
checkout, both at `--fuzz 0` (`patches-turing/README.md` has the loop), then:

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

The MTP history lookup is off by default and runs on both 0.29.0 runners
(`VLLM_MTP_LOOKUP=1`; on copy/edit work it drafts from the request's own
history). Its own measurements and gates are in
[docs/turing-2080ti.md](docs/turing-2080ti.md).

Model preparation (requantized embeddings and lm_head, int4 MTP draft) is in
[prepare/](prepare/). `single-user/` has a launcher and a systemd unit, and
`bench/` has the harnesses the numbers above came from. There is no room on
this card for batch mode's second KV pool.

## Credit

Fork of [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen), Apache-2.0.
Upstream's 3090 stack, model preparation scripts and benchmark harnesses are
what this port builds on.
