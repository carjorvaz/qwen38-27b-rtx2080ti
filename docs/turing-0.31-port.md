# Porting the Turing series to vLLM 0.31.0

The fork's sync to upstream, what retired, where the rebase stopped, and what
is verified. [← back to the main README](../README.md)

## The move

The fork was synced to `upstream/main` at `3512f0e`: the vLLM 0.30.0 pin (upstream's
`d88544b`, 2026-09-28) and the 0.31.0 pin (upstream's `87e813d` + `3512f0e`,
2026-10-11), plus the 85 commits between the fork's old base (`1cf8665`, the
0.29.0 pin) and them. `patches/` is upstream's 0.31.0 series unmodified — 49
patches in `patches/series` order plus the four KVarN files — and
`patches-turing/` is the Turing series regenerated on top of it.

The rebase itself: every vLLM-side hunk the Turing patches touch was replayed
onto a pristine v0.31.0 checkout with upstream's series applied, one patch at a
time, with GNU patch at `--fuzz 0`. Fourteen patches applied with exact context
(some at an offset) and were regenerated; seven needed hand porting, and one
retired:

| patch | what changed |
| --- | --- |
| `fp16-attn-verify` | upstream's `spec-attn-smem-fit` (#188) now owns the Triton verify attention's shared-memory limit on Turing (`_TILE_FITS`, the OutOfResources retry) and the block-M clamp, so the patch's own `small_shared_memory` guard was retired with it; the fp16/dtype-generic dots stay. |
| `fp16-drafter-units` | 0.31's `DFlashQwen3Model.__init__` now calls `_add_global_draft_layer_exclusions(...)` right where the residual scale is set, and `_build_context_kv_buffers` already spells the KV-weight line the patch's hunk carried; the `_hidden_norm_eps` alias is kept. |
| `spec-decode-mma` | the Triton `_spec_attn_partial` call is now a retry loop under `except OutOfResources`; the Turing dispatch wraps it (`if self.turing_partial ... else:`). The verify attention's `supports_quant_query_input` expression gained an `and not uses_sm90_fa4_fp8_kv_dequant` continuation, which the `_spec_attn_max_reqs` assignment moved below. The patch's `small_shared_memory` attribute is gone (its reader was retired with `fp16-attn-verify`). |
| `mtp-history-lookup` | 0.31 inserted `from vllm.model_executor.model_loader.utils import get_draft_load_config` between the hunk's context lines; the appended Triton kernel moved down with it. |
| `packed-int4-kv` | the packed-int4 preamble that rotates the query and redirects `out` now sits under 0.31's shared-memory comment and `_TILE_FITS` lookup. |
| `prefill-attn-dispatch` | `_is_uniform_decode` gained a `force_uniform_decode` parameter and upstream's body returns the shape test; the patch's state-aware body (`num_computed_tokens >= num_prompt_tokens`) is kept, driven by the new parameter first. |
| `marlin-prefill-only-int8` | 0.31 removed g_idx from the marlin stack: `MPLinearLayerConfig.has_g_idx` is gone (the prefill gate's `not c.has_g_idx` term went with it), and `apply_gptq_marlin_linear` dropped `g_idx`, `g_idx_sort_indices` and `is_k_full` — the transient W4A8 path's call sites and the `turing_marlin_prefill.linear` custom op (and its fake) were re-signed to the 13-argument form. The `padded_n, padded_k` if/else the patch's hunk carried is upstream's single `marlin_padded_nk(...)` line now. |

## Retirements

One patch is gone because upstream 0.31 carries its behaviour natively:

| retired from `patches-turing/` | now provided by |
| --- | --- |
| `mtp-prefix-checkpoint.patch` | the hybrid coordinator sets `drop_eagle_checkpoint_block` on the `MambaManager` when the scheduler drops Eagle's last hash unit (`use_eagle_block_drop`), `MambaManager` shifts `latest_prompt_hash_boundary` down one unit for it, and `is_mamba_prefill_checkpoint_valid(...)` with the `num_prefill_lookahead` plumbing enforces the "a physical-page boundary must not leave fewer known inputs than the draft lookahead" rule. See `patches-turing/README.md`. |

## What is verified

**Patch mechanics.** `patches-turing/check_vllm_series.sh` against a pristine
v0.31.0 clone: upstream's own checker is green (its GNU patch pass, the KVarN
files, the five ordered DFlash patches under `git apply`), the Turing series
then applies in order at `--fuzz 0` (21 patches, 4 at an offset, 0 with fuzz),
and `git diff --check` is clean.

**Names and calls.** The two checks upstream runs on its own ports were run on
this series against the 0.29.0 and 0.31.0 tags:

- `scripts/port-removed-names.py`'s logic (identifiers present in 0.29.0's
  Python tree and absent from 0.31.0's, met on the series' added lines): one
  hit, `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`, which is an entry in `envs.py`'s
  hash-exclusion list (a name, not an attribute read) — accepted. The first run
  of this check found the real ones: `c.has_g_idx`, `g_idx_sort_indices` and
  `is_k_full`, all in `marlin-prefill-only-int8` (upstream vLLM #54809 removed
  them, and the same class of hit once killed the 0.30 port's first int8 boot).
- `scripts/port-changed-signatures.py` (calls on the series' added lines bound
  against the new pin's definitions): 136 resolved calls, 0 breaks.

**Syntax.** All 76 Python files the series changes compile under Python 3.12.

## Validated on the card (2026-10-11, tiberius, RTX 2080 Ti)

The runtime was built from this revision (`qwen38-sm75-prepare`), and the
packaged regressions and kernel suites ran with the GPU to themselves:

| gate | result |
| --- | --- |
| `check attention` | PASS (FP16/INT8/INT4 reference and live graphs, packed int4, native selector JIT) |
| `check prefill` | PASS (all dtypes, the 262144-token case, tiles regression, graph replay) |
| `check prefix` | PASS (scheduler boundaries, allocator + lookup + COW, `CACHE_CPU_REGRESSION_PASS`) |
| `check gdn`, `check downloads` | PASS |
| `test_turing_gdn` | PASS (chunk state/output parity, escape hatches; 3.8x / 10x over the reference) |
| `test_turing_prefill` | PASS (94 comparisons) |
| `test_turing_marlin` | PASS (packing, negative scales, compile hash, dynamic-M compile, graph bit-equality) |

The first run of the port found two things the offline checks could not:

- `check attention` died in the Triton verify attention with `OutOfResources`
  (Required 81920, limit 65536). Upstream's `spec-attn-smem-fit` only halves
  the KV tile, and `_plan` can still pick `BLOCK_M=128` at D=256, whose 64 KiB
  query tile never fits sm75 however far the tile halves. The series' own
  `small_shared_memory` guard (the probe, the `BLOCK_M` clamp and the initial
  `TILE=32`) is load-bearing and is restored in `fp16-attn-verify`; it is also
  what routes the sm75/D256 shapes to the native kernel at all.
- The engine failed to load the model with
  `ModuleNotFoundError: runai_model_streamer`: the deployment loads with
  `--load-format runai_streamer`, and 0.31 lists that package only in
  `requirements/rocm.txt`. `requirements.lock` (the package's own artifact)
  carries `runai-model-streamer==0.16.1` with its `humanize` dependency.

Live-behaviour probes on the served model:

- Prompts of 1632 / 1633 / 1634 / 1636 / 3265 tokens all complete in about a
  second. The lookahead-window deferral upstream documents is not reachable in
  this configuration: without the Eagle block drop, the block-aligned split
  does not cap a chunk inside the draft window.
- `qwen38-sm75-bench --text-file`, greedy, 128 tokens: **98.0-100.0 / 96.1 /
  76.8 tok/s** at 8k / 32k / 64k (acceptance 3.54 / 3.74 / 3.31; TTFT 0.2-6.0 /
  27.2 / 44.6 s) against the 0.29 record's 112.4 / 99.8 / 80.6 (acceptance
  3.97 / 3.97 / 3.54). The 8k cell is ~12% down and the gap tracks acceptance,
  not step time; the mid and long cells are within 4-5%. That is a drafting
  follow-up (MTP acceptance on this text), not a kernel one.
- Quality is unchanged: PPL en 10.8097 / da 10.942 against 10.8096 / 10.9401,
  and GSM8K 96.0% (n=200, mean 381 tokens) against 95.5%.

**Not yet re-measured.** The remaining README and `docs/turing-2080ti.md` cells
(2k, 128k, 253k decode; the prefill tails; the copy/edit acceptance) still come
from the 0.29.0 runs.

