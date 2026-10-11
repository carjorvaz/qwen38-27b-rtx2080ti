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

**Not yet verified.** Everything that needs the card: the CUDA kernels compiling
(`turing_spec_attn.cu`, `turing_prefill_attn.cu`, the GDN kernels), the kernel
suites (`test_turing_gdn.py`, `test_turing_prefill.py`, `test_turing_marlin.py`),
the packaged regressions (`qwen38-sm75-check attention|prefill|prefix|gdn|
downloads`), quality (PPL / GSM8K), and the benchmark tables. The README and
`docs/turing-2080ti.md` tables remain the 0.29.0 runs until a re-run on this
pin replaces them.
