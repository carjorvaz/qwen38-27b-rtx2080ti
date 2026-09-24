# SM75 prefill follow-up

Optional additions to the [2080 Ti port](turing-2080ti.md), tested on the
22 GB card at 280 W, vLLM 0.28.0. **All new flags default off.** These are
separate changes, not a promise that every workload improves.

## What is implemented

### Bounded KV staging

`VLLM_TURING_PREFILL_WINDOW=16384` replaces the full-context FP16 workspace
with a 16k-token window. Complete windows before the query suffix are
noncausal; the last window contains every query token and uses lower-right
causality. FP32 log-sum-exp merging combines their normalized outputs.
Applying a separate causal mask to every window would be incorrect.

For 262k maximum context, 24 query heads / 4 KV heads / D256:

- Original FP16 staging: approximately **1 GiB**.
- 16k staging plus the preallocated 2048-query FP32 accumulator/LSE:
  **112.2 MiB**.
- With the optional 32-query extend path as well: **137.5 MiB**.
- At a 4096-token batch limit, staging/merge/extend workspace: **185.7 MiB**.

All persistent scratch is created during model construction, before memory
profiling. The full prefix is still dequantized each layer, a window at a time;
no full-context FP16 copy stays resident. The sizes above exclude the KV pool
and per-call query/output tensors. Attention still covers the complete context;
this is not sliding-window attention or KV eviction.

At 253,952 keys and 2048 queries, the measured attention call was 402.1 ms
with full staging and 405.7 ms with 16k staging: about **0.9% slower**, in
exchange for the memory. A profile of the original path put 99.1% of GPU
time in Cutlass attention and only 0.8% in gathering KV. Removing the gather
alone cannot produce a large cold-prefill speedup.

Floating-point reduction order changes. The operation is mathematically
equivalent, not guaranteed token-for-token identical at greedy near-ties.

### Small-query extensions

`VLLM_TURING_EXTEND_MAX=32` allocates a separate packed split-KV attention
workspace for short prefill/extend calls. It does **not** enlarge speculative
QMAX or replay a decode model graph for a prefill. Recurrent-state
initialization and the existing short-prefill graph guard remain intact.

At 131,072 keys, representative kernel timings (milliseconds):

| new query tokens | gather + Cutlass | best split-KV in sweep |
| --- | ---: | ---: |
| 16 | 9.67 | 4.67 |
| 32 | 9.78 | 8.14 |
| 64 | 9.98 | 14.61 |
| 256 | 28.04 | 55.40 |

The narrow threshold is intentional. Turning every prefill into split-KV
would regress performance. The existing decode segment default, 32, remained
the best measured choice for the five-query verify shape across 4k–248k;
there is no new decode-segment default.

These are kernel timings, **not API TTFT speedups**. A nominal 16-token
follow-up can recompute 32 tokens because of prefix-checkpoint safety margins.
Replay can recompute substantially more. Use the API benchmark's actual
`cached_tokens`/`uncached_tokens`, not the nominal suffix length.

Early construction also exposed a dtype bug: Hadamard signs inherited the
model loader's FP16 default dtype. Their construction now explicitly uses
FP32, as required by the native rotation kernel.

### Transient W4A8 for large-M projections

`VLLM_TURING_PREFILL_INT8=mlp` selects target MLP projections; `all` selects
all eligible target projections. Both exclude MTP and lm_head. Requirements:
SM75, FP16 activations, symmetric group-128 W4 weights, no activation-order
permutation or runtime zero points.

At **M >= 1024**, one canonical W4A16 matrix is temporarily repacked for
Marlin W4A8, including the negative-scale sign correction. The INT8 GEMM then
runs and the temporary matrix can be released. There is no persistent second
model-sized weight copy. Small-M calls use the original weights and kernels.

The M-dependent decision is **inside an opaque custom op**. Specializing a
Python branch while profiling at M=2048 is unsafe when vLLM compiles a whole
range and does not evaluate dynamic-shape guards. The regression test first
compiles a large shape, then captures/replays a five-token decode graph and
requires bit-identical results against the original W4A16 operation.

The mode participates in vLLM's compilation-cache hash. Switching it cannot
silently reuse a baseline graph that omits the optimization.

Representative large MLP GEMMs were around **1.3× faster including transient
repacking**. This is not an end-to-end model speedup. Repacking lost at M=256,
which is why the dispatch does not enable it for small extensions.

**This mode is lossy.** Besides activation rounding, folding a negative scale
clips a flipped -8 to +7 rather than +8. Canonical weights remain unchanged, but the
prefill state and therefore subsequent outputs can change. Identical small-M
GEMMs do not imply identical answers or speculative acceptance.

## Quality and serving measurements

**Perplexity.** Paired non-overlapping 2048-token windows scored with
`prompt_logprobs`, on a 64k / 3 GiB KV profile, identical settings except
`VLLM_TURING_PREFILL_INT8`. Two corpora: 48 windows of the public
[PyTorch wikitext-2 test corpus](https://github.com/pytorch/examples/blob/main/word_language_model/data/wikitext-2/test.txt)
(98,256 scored tokens) and 24 windows of the vLLM Python source tree
(49,128 tokens) — the code-heavy case where activation quantization has
historically cost the most. Intervals are 95% paired over windows.

| target prefill | PPL, wikitext-2 | vs W4A16 | PPL, Python | vs W4A16 |
| --- | ---: | ---: | ---: | ---: |
| W4A16 | 6.3181 | — | 2.2424 | — |
| transient W4A8, `mlp` | 6.3796 | **+0.97%** [0.83, 1.10] | 2.2830 | **+1.81%** [1.30, 2.32] |
| transient W4A8, `all` | 6.4237 | **+1.66%** [1.37, 1.91] | 2.3061 | **+2.84%** [2.22, 3.46] |

The earlier 16-window run (W4A16 6.0594, `all` 6.1499, +1.49%) is consistent
with these; the absolute level moves with which corpus windows are scored, so
compare rows within a table only. This is a narrow perplexity check, not broad
coding, reasoning, or tool-call validation; a task-level gate was not run.

**Serving.** Normal 5.5 GiB KV pool, MTP4 with the context lookup, prefix
caching, batch limit 4096, window `16384`, extend threshold `32`, one request
at a time, client-observed TTFT (single sweeps; the earlier campaign's numbers
reproduce to ~0.5%). `+32k tail` extends the same cached 32k prompt once at
~48k depth; `+65k tail` extends it to 128k, so it is a 65k prefill at
depth.

| cold prompt | W4A16 | `mlp` | `all` |
| --- | ---: | ---: | ---: |
| 32,768 tokens (cold) | 41.37 s | 36.55 s (−11.7%) | **34.76 s (−16.0%)** |
| +32k tail at ~48k | 55.82 s | 51.04 s (−8.6%) | **49.26 s (−11.8%)** |
| +65k tail at ~98k | 155.24 s | 145.68 s (−6.2%) | **141.78 s (−8.7%)** |

The batch limit alone buys nothing here: windowed W4A16 at 2048 and at 4096
measure 41.26 s and 41.37 s on the same 32k cold prompt. The int8 flag is the
whole gain. Decode is unchanged by construction (the flag only fires at
M ≥ 1024): stream rates at 32k/128k sit within run-to-run acceptance noise.

**Reading the trade.** `mlp` gets 71–73% of `all`'s prefill gain for 58% of the
perplexity cost, so it is the better ratio (≈12% cold TTFT per 1% wikitext
perplexity, ≈6.5% per 1% Python). `all` is the bigger lever and the one used
for the capacity test below; both are opt-in and neither is enabled by
default.

**Sampled traffic and the context lookup.** The model's `generation_config`
is `do_sample: true, temperature 1.0, top_p 0.95, top_k 20`, and requests that
do not set sampling explicitly get those — which is most real client traffic.
`VLLM_MTP_LOOKUP` (`mtp-history-lookup.patch`) is greedy-only by default, so
it never fires for that traffic. Measured on 23k-token documents at
temperature 1.0, 3 reps, interleaved server boots (`VLLM_MTP_LOOKUP_SAMPLED`):

| task | lookup off, tok/s | lookup on, tok/s | effect |
| --- | ---: | ---: | ---: |
| reproduce the document verbatim | 109.8 | **129.0** | **+17%** |
| quote and explain | 89.9 | **102.0** | +13% |
| rewrite keeping commands | 92.0 | 92.0 | neutral |
| free-form summary | 65.9 | 59.7 | **−9.5%** |

A stricter gate keeps the copy win but not the prose loss. `LOOKUP_MIN` cannot
exceed `NMAX`, so the first attempt (`MIN=32`, `NMIN=8`) only reaches a
32-token match (copy +13%, summary −7%). Raising the cap too (`NMAX=64`,
`MIN=48`, `NMIN=8`) buys the rest of the copy-shaped work and still loses on
prose:

| task | off | `SAMPLED=1` | `MIN=32` | `NMAX=64, MIN=48` |
| --- | ---: | ---: | ---: | ---: |
| reproduce verbatim | 4.33 | 4.87 | 4.89 | 4.89 tok/step |
| quote and explain | 3.57 | 3.95 | 3.98 | **4.54** |
| rewrite keeping commands | 3.67 | 3.63 | 3.55 | 3.80 |
| free-form summary | 2.62 | 2.37 | 2.44 | 2.42 |

The point mass is exact but a weaker draft than the drafter's own distribution
wherever the drafter is right — the DFlash2 stack fixes that by *fusing* the two
proposals (`_fuse_draft_kernel`: take the lookup only when the match is strong
or the drafter agrees on its first tokens). The fusion does not port cleanly to
MTP: the MTP drafter sees the whole context, so on copy work the lookup's value
comes precisely from *disagreeing* with it, and an agreement gate would discard
that. Until a confidence rule that separates the two cases exists, sampled
lookup stays opt-in with the `NMAX=64` gate recommended for copy-heavy traffic.

The combined `all` arm also retrieved all three audit keys at 10%, 50%, and 90%
depth in a **253,790-token** chat prompt, then returned the same correct keys
on cached replay. Cold completion took 650.50 s; replay took 1.58 s, with
253,760 cached tokens and **30 recomputed tokens**. Both completed without
preemption. The replay time includes 26 output tokens; it is not a TTFT or a
like-for-like comparison to the older 8.5-second cached-turn measurement.

Results: [`turing-prefill-results.json`](turing-prefill-results.json).
The original production runtime/profile was restored; these flags were not
enabled persistently.

## Reproducing

Apply the full series, including `prefill-memory-and-extend.patch` and
`marlin-prefill-only-int8.patch`. Do not apply these directly to unpatched
vLLM. Start with one flag at a time; use an idle GPU for kernel benchmarks.

```bash
python bench/test_turing_prefill.py
python bench/test_turing_marlin.py
# Native GDN chunk state and chunk output against their Triton references. The
# 32k timing case and Triton's autotuning want a free GPU, not a loaded one.
python bench/test_turing_gdn.py

python bench/turing_attention_bench.py \
  --window 16384 --extend-max 32 --segments 32 \
  --lengths 4096 32768 131072 253952 --queries 16 32 64 256 2048
python bench/turing_gemm_bench.py --full --transient

# A running server with --enable-prompt-tokens-details.
# This harness never starts/stops services.
python bench/turing_api_bench.py --base http://127.0.0.1:8080 \
  --tag baseline --seed 20260923 --lengths 4096 32768 131072
# Use a fresh seed on the same server. Match seeds across fresh A/B servers.

# Use a representative UTF-8 corpus and a server with logits headroom.
python bench/turing_quality.py --base http://127.0.0.1:8080 \
  --tag baseline --corpus /path/to/wikitext-test.txt --samples 16 --window 2048
# Separately, with the full 262k serving profile:
python bench/turing_quality.py --base http://127.0.0.1:8080 \
  --tag capacity --corpus /path/to/wikitext-test.txt --samples 0 \
  --needle-lengths 253952
```

Benchmark cold input independently of cached replay/extension. Warm JIT shapes
before comparing short-prompt TTFT, repeat/interleave arms, and keep power
settings fixed. Metric deltas assume no other client is using the server.
Decode stream rates include client/stream effects; engine timing and acceptance
counters are included separately.

All **94 attention comparisons** and the Marlin compile/capture checks passed.
Regression coverage includes ragged page/window boundaries (including a
one-key prefix window followed by a query-sized causal window), an independent
FP32 attention reference, noncontiguous outputs with canaries, sign generation
under a FP16 default dtype, negative weight scales, immutable canonical weights,
compilation hashes, and dynamic-M CUDA-graph execution. A full GNU-patch pass
against pristine vLLM 0.28.0 is also required.

## Investigated but not enabled

- **Unpacking KV inside each attention tile:** not implemented as a serving
  kernel. The profile showed less than 1% gather time; bounded staging was
  chosen first to recover memory without repeatedly unpacking K/V per query
  tile. A faster fused packed-cache kernel remains future work.
- **INT8-QK / FP16-PV attention prototype:** native SM75 integer MMA worked,
  with roughly 0.35% relative RMS output error on the synthetic cache, but the
  first fused kernel was approximately 5× slower at full prefill shapes.
  It is not part of the serving path. Faster arithmetic alone did not offset
  its staging, instruction and tile-layout costs.
- **64-query Cutlass tile:** exceeds the 64 KiB shared-memory limit with the
  existing 256-key / D256 template. The shipped tile is 32 queries, despite
  what the original prose said.
- **Blindly increasing split-KV segments:** did not improve the five-query
  decode default. Larger query blocks can prefer other counts, but are a
  different workload.
- **Permanently enabling W4A8 for decode:** the eager small-M GEMM probe was
  slower. The transient path deliberately leaves small-M weights and math
  alone instead.

Turing supports INT8 and INT4 tensor cores; see the
[NVIDIA tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#tensor-core-operations).
The old claim that it has no integer tensor cores was incorrect.
