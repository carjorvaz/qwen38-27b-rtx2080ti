# Turing (SM75): Qwen3.8-27B on a 22 GB RTX 2080 Ti

This branch adds a Turing port to the 3090 stack. `patches-turing/` is a
24-patch series applied after `patches/`, on the same vLLM 0.28.0, and it runs the same
serving setup on a card that is 8 GB smaller and one generation older: an RTX 2080 Ti with 22 GB, running at 280 W (the card's
factory cap is 250 W).

The short version: 262,144 tokens of context out of a 5.5 GiB int4 KV pool,
112 tok/s single-stream decode at 2k and 66 tok/s at 128k, with MTP speculation
and prefix caching. Everything below is measured on that card, single stream,
with the launch flags in [Running it](#running-it). The 3090 numbers elsewhere
in this repo are upstream's and are not re-measured here.

## What SM75 is missing, and what replaces it

| 3090 (SM86) stack uses | Turing (SM75) instead |
| --- | --- |
| fp8 KV storage and bf16 attention | packed int4 KV with per-token-head scales and zero points, fp16 operands / fp32 accumulation |
| 99 KiB shared memory per SM | 64 KiB, which rules out the larger attention tile and dense Hadamard matrices at head_dim 256 |
| SM80+-targeted MMA/rotation kernels, hadacore | SM75-specific MMA layouts; a butterfly Hadamard kernel using registers instead of a dense shared-memory matrix |
| FA2-style prefill at head_dim 256 | a Cutlass memory-efficient-attention port with 256-key tiles and **32 queries per CTA**, fp32 accumulators |
| `chunk_gated_delta_rule` kernels for SM80+ | an SM75 chunk-state kernel: fp16 MMA operands, fp32 recurrent state in registers |

The port is specific to head_dim 256 and to the packed `int4_per_token_head`
layout; other shapes fall back to the paths they already had.

**Hardware correction:** Turing does support INT8 and INT4 tensor-core MMA
([NVIDIA tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#tensor-core-operations)).
The original attention port does not use it; that is a kernel limitation, not
missing hardware. The original documentation also said 64 queries per CTA,
but the shipped Cutlass instantiation uses 32.

## The series

Applied in `patches-turing/series` order. Measurements are single-stream decode
on real text unless noted; "correctness" means the patch fixes a path that
otherwise produces wrong results or does not run.

| patch | what it does | result |
| --- | --- | --- |
| `fp16-attn-verify` | prefill attention in fp16 with a packed-KV workspace | enables prefill at all on SM75 |
| `fp16-drafter-units` | power-of-two unit shift so a bf16 checkpoint runs in fp16 | drafter loads without overflow |
| `gdn-chunk-state` | chunked gated-delta-rule state update for SM75 | correctness on Turing |
| `mamba-block-retirement` | frees the null gaps mamba prefill leaves | correctness (pool leak) |
| `spec-decode-mma` | the verify attention kernel: fp16 MMA, split-KV, in-kernel dequant | the speculative path runs on Turing |
| `mtp-history-lookup` | drafts from the context when the tail is already in it | 60k "reproduce this" 78.4 -> 90 tok/s, exact |
| `mtp-prefix-checkpoint` | checkpoints recurrent state where prefix caching matched | correctness (prefix caching with a hybrid model) |
| `packed-int4-kv` | asymmetric int4 KV, per-(token, head) zero points | 262k context in 5.5 GiB |
| `prefill-attn-dispatch` | keeps short prefills off the decode graph | correctness (recurrent-state init) |
| `prefill-attn-tiles` | the SM75 tile kernel, constants measured | tail prefill 464 tok/s at 64k, 240 at 248k |
| `int4-kv-hadamard` | butterfly Hadamard for the int4 quantizer | +5.5% at 6k, +3.5% at 32k context |
| `spec-attn-nseg` | split-KV segments as a knob | tuning, default 32 |
| `spec-attn-staging` | vectorized int4 staging in the verify kernel | 64k 42.9 -> 72.6 tok/s, 128k 24.0 -> 49.1 |
| `int4-attn-fp16-dots` | fp16 dots for the int4 attention | ~2% at 64k, perplexity unchanged |
| `spec-attn-scratch-cap` | bounds the 3D-scratch workspace | correctness at large blocks |
| `spec-attn-register-softmax` | scores stay in the QK accumulators; 4-lane reductions and a 512 B (max, sum) exchange replace the 4.6 KiB score tile | layer 1.73 -> 1.22 ms at 131k; decode step 41.2 -> 39.0 / 46.6 -> 44.0 / 57.0 -> 52.9 ms at 32k/64k/128k |
| `gdn-chunk-o` | native fp16 WMMA replacement for FLA's Triton chunk-output kernel, whose `tl.dot` lowers to SIMT on SM75 | 61.3 -> 14.8 ms per layer at T=32k (4.1x); cold 32k 36.3 -> 34.0 s, +32k tail at 48k 50.8 -> 48.6 s, to 128k 145.2 -> 140.8 s |
| `gdn-state-16w` | the same state kernel at 16 warps instead of 4, same 48 KiB tile, bitwise identical output | 5.67 -> 1.98 ms per 2048-token chunk per layer (3.1x); cold 32k 34.0 -> 31.9 s, +32k tail at 48k 48.6 -> 46.3 s, to 128k 140.8 -> 135.9 s |
| `sampling-log` | logs effective sampling parameters per request | tooling; explains why greedy-only lookup drafting |
| `vllm-*` (3) | backports from vLLM after 0.28.0: SSE keep-alive, engine stall sentinel, completion log | operational |
| `prefill-memory-and-extend` | optional bounded KV staging and a separate small-query split-KV path | [follow-up measurements](turing-prefill.md) |
| `marlin-prefill-only-int8` | optional transient W4A8 repacking for large-M target GEMMs; canonical decode weights stay intact | lossy prefill mode; [quality and limits](turing-prefill.md) |

Standalone GPU tests cover the native kernels, with no model and no server:
`python bench/test_turing_prefill.py`, `bench/test_turing_marlin.py`, and
`bench/test_turing_gdn.py` (chunk state and chunk output against their Triton
references, plus timing).

## Running it

Install vLLM 0.28.0, apply both patch directories (`patches-turing/README.md`
has the command), and launch with the flags this was measured with:

```bash
vllm serve models/Qwen3.8-27B-W4A16-AutoRound \
  --served-model-name qwen3.8-27b --dtype float16 --language-model-only \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype int4_per_token_head \
  --mamba-cache-dtype float16 --mamba-ssm-cache-dtype float16 --mamba-cache-mode align \
  --kv-cache-memory 5905580032 --max-model-len 262144 \
  --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --enable-prefix-caching --prefix-match-unit 16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"attention_backend":"TRITON_ATTN","draft_sample_method":"probabilistic"}' \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["+rms_norm","+silu_and_mul"],"max_cudagraph_capture_size":8}' \
  --async-scheduling --sse-keep-alive-interval 30
```

`--kv-cache-memory 5905580032` is the whole budget the card has for KV at this
context: raise it for a longer context, lower it if the model does not load.
`--load-format runai_streamer` with `--model-loader-extra-config
'{"concurrency":2,"memory_limit":1610612736}'` is worth adding on a host with
little RAM; it made the difference between a clean load and an OOM kill on an
8 GB host.

Knobs this series adds, all optional:

| variable | default | effect |
| --- | --- | --- |
| `VLLM_MTP_LOOKUP` | `0` | enable the context-lookup drafts (greedy requests only) |
| `VLLM_MTP_LOOKUP_MIN` | `24` | shortest tail worth looking up |
| `VLLM_MTP_LOOKUP_SAMPLED` | `0` | allow the lookup on sampled requests too; task-dependent (+17% copy, −9.5% free-form prose at 23k), see [turing-prefill.md](turing-prefill.md) |
| `VLLM_TURING_PREFILL_TILE` | `256` | key-tile width for the prefill kernel; `0` selects the portable fallback |
| `VLLM_TURING_SPEC_NSEG` | `32` | split-KV partials in the verify attention |
| `VLLM_TURING_GDN_STATE` | `1` | `0` restores the upstream chunk-state path |
| `VLLM_SAMPLING_LOG` | `0` | per-request client address, prompt digest and sampling parameters |
| `VLLM_ENGINE_STALL_SENTINEL_S` | unset | seconds without an engine iteration before it logs and aborts |
| `VLLM_TURING_PREFILL_WINDOW` | `0` | bounded FP16 staging window; `16384` reduces memory, `0` retains full-context staging |
| `VLLM_TURING_EXTEND_MAX` | `0` | separate split-KV path up to this query length; `32` is the measured candidate |
| `VLLM_TURING_PREFILL_INT8` | `off` | `mlp` or `all`: transient W4A8 at M >= 1024, excluding MTP/lm_head; lossy (+0.97% / +1.66% wikitext PPL, +1.81% / +2.84% Python) for −11.7% / −16.0% cold-32k TTFT; [follow-up](turing-prefill.md) |

## Measurements

Single stream, MTP k=4 with the context lookup, real text prompts (wikitext-2
test), 128-192 output tokens, greedy. Decode rate excludes prefill.

| context | decode tok/s | ms per step |
| --- | --- | --- |
| 2k | 112.3 | 33.8 |
| 8k | 97.7 | 36.2 |
| 32k | 96.6 | 39.8 |
| 64k | 71.6 | 45.8 |
| 128k | 65.6 | 57.6 |

A step is ~24 ms of weight reads at every depth; speculation adds 4.5 ms of
draft work at 2k and 7.7 ms at 64k, and the verify attention adds ~5 ms at 64k.
The 64k and 128k rows are well above what the same configuration does without
the vectorized int4 staging (42.9 and 24.0 tok/s).

Prefill, which is what a long turn actually costs:

| context | tail prefill tok/s |
| --- | --- |
| 32k | 624 |
| 64k | 464 |
| 248k | ~240 |

A fully cached 246k-token turn takes 8.5 s. Five scenarios at 248k were run
end to end (cold prefill, cached replay of the same turn, replacement of the
cached prefix, recompute after a mismatch, and a short request); all completed
with zero preemptions. Attention runs at 32 TFLOPS effective
there, 60-70% of what fp16 with fp32 accumulation can reach on this part, so the
tail is close to what this generation of hardware does.

The optional prefill follow-up (`VLLM_TURING_PREFILL_WINDOW`,
`VLLM_TURING_PREFILL_INT8`) trades perplexity for prefill latency: `mlp` is
−11.7% cold-32k TTFT for +0.97% wikitext / +1.81% Python perplexity, `all` is
−16.0% for +1.66% / +2.84%. Both are opt-in; see
[turing-prefill.md](turing-prefill.md) for the paired quality intervals and
the longer-context rows. Independently of them, `gdn-chunk-o` replaces FLA's
Triton chunk-output kernel (whose `tl.dot` lowers to SIMT on SM75) with a
native fp16-WMMA one and takes another **6.3% off cold 32k** (36.3 s ->
34.0 s), 4.4% off a 32k tail at depth and 3.0% off the run to 128k, without
touching quality (max |diff| 4.9e-4 against the reference).

Quality: 10.8797 perplexity on wikitext-2 test (en 10.8077, da 10.938) and 94.5%
on GSM8K (n=200), with the int4 KV cache in place. To separate the KV precision
from the weights, the same server was run with an fp16 KV cache on the same
5.5 GiB pool (65k context instead of 262k): 10.8451 perplexity and 95.5% GSM8K.
The 4-bit cache costs about 0.3% perplexity, with the GSM8K difference inside
the noise of 200 questions.

## Measured and rejected

Kept here so nobody re-derives them:

- **The transient int8 repack kernel** (`turing_marlin_prefill.py`) runs at
  68 GB/s of a 616 GB/s memory floor: it is gather-bound, and the gather is
  inherent -- one output tile reads 1024 nibbles scattered over a window of up
  to N*16 nibbles (139 KB at N=17408), so there is no contiguous source block
  to stage. Warp counts 1/2/4 measure the same and 8 is worse. It costs 0.55 s
  of a 32k prefill (1.7%) and is the price of the transient layout.
- **Native ports of the remaining FLA GDN kernels** (`chunk_scaled_dot_kkt_fwd`,
  `recompute_w_u_fwd`). Both do their products with `tl.dot`, i.e. SIMT on
  SM75, but they are latency-bound well above their bandwidth floor (72 KB of
  traffic per chunk-head for 2.1 MFLOP), so tensor cores do not pay: a fp16
  WMMA kkt is exact (3.4e-7) and 1.71x (6.73 -> 3.94 ms per layer at T=32768,
  0.4% end to end), and a 16-warp w_u is 0.90x, i.e. slower than Triton. The
  same holds for solve_tril and the conv/post kernels: that whole tail is
  memory- and latency-bound, not compute-bound.
- **Longer verify blocks** (static k=8, and an adaptive block that asks the
  scheduler for more tokens when a step is fully accepted). k=8 is worth +16 to
  +38% on chat-like traffic, but wide verify at depth costs more than the extra
  tokens return: five variants of the adaptive block all lost end to end.
- **Windowing the drafter's attention** to 4096 tokens, on the theory that a
  local prediction does not need the whole context. The windowed path is 25%
  slower per step at 8k and faults at 64k, because the staged int4 kernel
  indexes the block table assuming a full-causal layout.
- **warpN=32 or 128 in the prefill tile**: the first does not compile, the
  second is 5x slower. Two blocks per SM gained 3% at 64k and nothing at depth.
- **A hand-written prefill attention built from the verify kernel's parts.**
  Run as a prefill engine (R requests of 5 queries over a 131k-token context,
  its own nseg combine) the register-softmax kernel reaches 15.5 / 20.6 /
  20.6 TFLOPS at 160 / 640 / 1280 queries, while the Cutlass port does 20.8 /
  29.6 / 31.5 on the same fixture. The simple design saturates where the
  Cutlass one keeps scaling, so a replacement would need to make up ~1.5x
  before it won anything; the missing ingredient is software pipelining
  (double-buffered staging, cp.async, deeper ILP), which is a different class
  of kernel from anything in this series.
- **Smaller key tiles in the prefill attention** (`kKeysPerBlock` 128 or 64, to
  trade tile reuse for more resident CTAs): neither compiles. The pinned
  PyTorch template keeps the fp32 output accumulator in registers only while
  `kMaxK <= kKeysPerBlock`; below that it needs a shared accumulator buffer,
  which the port static-asserts against — and that buffer would consume the
  shared memory the extra occupancy was meant to buy. The 32-query x 256-key
  tile is the only shape this card fits, so a faster prefill attention needs a
  purpose-built kernel, not a parameter change, and it is the largest item
  left: 25% of a cold 32k prefill and 67% of a long-context chunk at ~30
  TFLOPS effective.

## Known limits

- Where the verify-attention kernel's time went: compiled variants with one
  phase removed (131k keys, five queries, nseg 32, one layer, baseline
  1.726 ms) measured K/V staging at ~0.3%, page-table/scale loads at ~11%,
  QK MMA at ~10%, PV MMA at ~8%, and the score store/reload exchange plus the
  32-lane softmax at ~45% — `save_score` alone 16%, the softmax loop 29%.
  `spec-attn-register-softmax` removes the 4.6 KiB score tile and reduces over
  four lanes instead, taking the layer to 1.223 ms (-29%) and the decode step
  to 39.0 / 44.0 / 52.9 ms at 32k / 64k / 128k. What is left is latency at low
  occupancy: 2 CTAs/SM (179 registers, ~22 KiB shared, eight warps, ~12%).
  Metadata prefetching does not survive ptxas (1.4%), folding the scales into
  staging is worth 3%, and cp.async cannot pay because staging is not the
  cost. A packed-int4 shared stage (4 KiB instead of 17 KiB) to reach 3-4 CTAs
  per SM is the remaining direction.
- The GDN chunk-state kernel is native (16 warps since `gdn-state-16w`, 5.67 ->
  1.98 ms per 2048-token chunk per layer) but still one CTA per SM: its 48 KiB
  static shared tile leaves 16 warps and 103 registers. It walks its chunks
  serially with ~5 barriers and six shared round trips per chunk, and the
  state is a scalar-decay linear recurrence per head, so a shared-memory
  redesign (operand K-blocking, halved fp32 state tile) for 2 CTAs per SM and
  a parallel scan are the open directions. It is now ~4% of the cold-32k
  prefill, down from 10.6%.
- The port assumes head_dim 256 and `int4_per_token_head`. Other shapes take
  the paths they already had, which on SM75 means slower or absent.
- No native FP8 MMA or validated FP8-KV path in this setup. `int4_per_token_head` is the only validated way
  to reach 262k on this card.
- Single stream only: these numbers are `max-num-seqs=1`. Serving several
  streams needs a second KV pool, about 5.97 GiB beyond what the card has spare
  at 262k.
- The Docker and compose paths in this repo build for the 3090's CUDA arch
  list; they are not wired for Turing. Apply the series to a local vLLM 0.28.0
  environment instead.
- The int4 3D-scratch attention variant boots but is not faster here; its
  workspace is capped so a large block cannot over-allocate.
- One hardware note: an early long run at 280 W produced Xid 45 on our card.
  It has not recurred in multi-hour soaks at the same settings since, and we run
  a power-limit service and a watchdog, but the cause is unexplained. The
  numbers above are at 280 W; the unlock was measured at +1.6%, inside the
  noise of a single run, so it is not what any of them depend on.

## Credit

Fork of [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) (formerly
`syv-ai/qwen38-27b-rtx3090`), whose 3090 stack, preparation scripts and
benchmark harness this series builds on and does not replace. The patches here modify vLLM 0.28.0; both are
Apache-2.0, see `LICENSE`.
