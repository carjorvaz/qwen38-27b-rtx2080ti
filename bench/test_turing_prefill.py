#!/usr/bin/env python3
"""GPU regression tests for the optional SM75 extend/windowed-prefill paths.

No model or server. Checks ragged/page/window boundaries, noncontiguous output,
FP32 attention oracle, deterministic signs under a FP16 default dtype, and the
bounded workspace. Requires the new Turing prefill patches.
"""

import os

import torch
from turing_attention_bench import errors, packed_cache
from vllm.v1.attention.ops.int4_per_token_head import _RHT_SIGNS_CACHE, _get_rht_signs
from vllm.v1.attention.ops.turing_prefill_attn import TuringPrefillAttention


def main():
    assert torch.cuda.get_device_capability() == (7, 5)
    torch.manual_seed(7341)
    device = torch.device("cuda:0")
    _RHT_SIGNS_CACHE.clear()
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        signs = _get_rht_signs(256, 0, device)
    finally:
        torch.set_default_dtype(old)
    assert signs.dtype == torch.float32
    _RHT_SIGNS_CACHE.clear()
    assert torch.equal(signs, _get_rht_signs(256, 0, device))
    count = 0
    for length in (513, 1025, 8197, 32768):
        k, v, ks, vs, table = packed_cache(length, block=1632)
        os.environ["VLLM_TURING_PREFILL_WINDOW"] = "0"
        os.environ["VLLM_TURING_EXTEND_MAX"] = "0"
        dense = TuringPrefillAttention(
            length, 4, 256, device, num_heads=24, max_queries=1024
        )
        os.environ["VLLM_TURING_PREFILL_WINDOW"] = "1024"
        windowed = TuringPrefillAttention(
            length, 4, 256, device, num_heads=24, max_queries=1024
        )
        os.environ["VLLM_TURING_EXTEND_MAX"] = "32"
        extended = TuringPrefillAttention(
            length, 4, 256, device, num_heads=24, max_queries=1024
        )
        assert windowed.storage.nbytes <= 2 * 4 * 1024 * 256 * 2
        for qlen in (1, 5, 16, 17, 32, 33, 64, 127, 256, 511, 512, 1024):
            if qlen > length:
                continue
            # length=1025, qlen=1024 forces a one-key noncausal window
            # followed by a window exactly as long as the query suffix.
            q = torch.randn(qlen, 24, 256, device=device, dtype=torch.float16)
            ref = torch.empty_like(q)
            # Head stride differs from the query's; test writes do not clobber
            # adjacent storage (e.g. fused/gated attention's output view).
            backing = torch.full(
                (qlen, 24, 512), 71, device=device, dtype=torch.float16
            )
            out = backing[..., :256]
            dense.run(q, k, v, ref, table, length, 1 / 16, ks, vs)
            for candidate in (windowed, extended):
                candidate.run(q, k, v, out, table, length, 1 / 16, ks, vs)
                err = errors(out, ref)
                assert torch.isfinite(out).all() and err["relative_rms"] < 0.004, (
                    length,
                    qlen,
                    err,
                )
                assert (backing[..., 256:] == 71).all()
                count += 1
            if length == 513 and qlen == 17:
                # Independent FP32 math reference over the gathered KV, not
                # another split/online-softmax implementation.
                h = torch.ones((1, 1), device=device)
                while h.shape[0] < 256:
                    h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
                rotated_q = (q.float() * signs) @ h
                keys = dense.key[:, :length].float().repeat_interleave(6, 0)
                values = dense.value[:, :length].float().repeat_interleave(6, 0)
                scores = rotated_q.transpose(0, 1) @ keys.transpose(1, 2) / (16 * 256)
                allowed = torch.arange(length, device=device)[None, :] <= (
                    length - qlen + torch.arange(qlen, device=device)[:, None]
                )
                scores.masked_fill_(~allowed, -torch.inf)
                oracle = (
                    ((scores.softmax(-1) @ values).transpose(0, 1) @ h) * signs / 256
                )
                err = errors(ref, oracle)
                assert err["relative_rms"] < 0.004, err
        del dense, windowed, extended, k, v, ks, vs, table
        torch.cuda.empty_cache()
    print(
        f"PASS: {count} attention comparisons, FP32 oracle, sign dtype, output canaries, bounded workspace"
    )


if __name__ == "__main__":
    main()
