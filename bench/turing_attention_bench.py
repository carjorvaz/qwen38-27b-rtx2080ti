#!/usr/bin/env python3
"""SM75 packed-INT4 prefill/extend and split-KV benchmark (no model/server).

Run only on an idle GPU. Uses production's interleaved 264-byte K/V head rows,
random page tables, asymmetric scale/zero metadata, ragged tails, and a fixed seed.
Compares against the existing FP16-gather/Cutlass path. JSONL on stdout.
"""

import argparse
import inspect
import json
import os
import statistics
import sys
from functools import partial

import torch
from vllm.v1.attention.ops.spec_decode_attn import SpecDecodeAttention
from vllm.v1.attention.ops.turing_prefill_attn import TuringPrefillAttention


def packed_cache(length, block=1632, heads=4):
    pages = (length + block - 1) // block + 2
    storage = torch.randint(
        0, 256, (pages, heads, block, 264), dtype=torch.uint8, device="cuda"
    ).transpose(1, 2)
    k, v = storage[..., :132], storage[..., 132:]
    ks, vs = (
        k[..., 128:].view(torch.float32).squeeze(-1),
        v[..., 128:].view(torch.float32).squeeze(-1),
    )
    for scales in (ks, vs):
        raw = torch.rand(scales.shape, device="cuda") + 1.0
        zeros = torch.randint(5, 11, scales.shape, device="cuda", dtype=torch.int32)
        scales.copy_(((raw.view(torch.int32) & -16) | zeros).view(torch.float32))
    table = torch.randperm(pages, device="cuda", dtype=torch.int32).view(1, -1)
    return k, v, ks, vs, table


def timing(fn, repetitions=5, iterations=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repetitions):
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end) / iterations)
    return statistics.median(times)


def errors(actual, reference):
    a, b = actual.float(), reference.float()
    return {
        "relative_rms": ((a - b).square().mean() / b.square().mean().clamp_min(1e-20))
        .sqrt()
        .item(),
        "max_abs": (a - b).abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0
        ).item(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lengths", nargs="+", type=int, default=[4096, 32768, 131072, 253952]
    )
    p.add_argument("--queries", nargs="+", type=int, default=[5, 16, 64, 256, 2048])
    p.add_argument("--segments", nargs="+", type=int, default=[8, 16, 32, 64])
    p.add_argument("--block", type=int, default=1632)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--tag", default="baseline")
    p.add_argument(
        "--window",
        type=int,
        default=0,
        help="Compare bounded KV staging against full-context staging",
    )
    p.add_argument(
        "--extend-max",
        type=int,
        default=0,
        help="Compare the separate small-extension dispatch",
    )
    p.add_argument(
        "--profile", help="Export a torch profiler Chrome trace for the largest prefill"
    )
    a = p.parse_args()
    assert torch.cuda.get_device_capability() == (7, 5), "SM75 benchmark"
    torch.manual_seed(1729)
    for length in a.lengths:
        k, v, ks, vs, table = packed_cache(length, a.block)
        init_kw = {}
        if "num_heads" in inspect.signature(TuringPrefillAttention).parameters:
            init_kw = {"num_heads": 24, "max_queries": max(a.queries)}
        previous = {
            name: os.environ.get(name)
            for name in ("VLLM_TURING_PREFILL_WINDOW", "VLLM_TURING_EXTEND_MAX")
        }
        try:
            os.environ["VLLM_TURING_PREFILL_WINDOW"] = "0"
            os.environ["VLLM_TURING_EXTEND_MAX"] = "0"
            prefill = TuringPrefillAttention(
                length, 4, 256, torch.device("cuda"), **init_kw
            )
            candidate = None
            if a.window or a.extend_max:
                if not init_kw:
                    raise RuntimeError(
                        "candidate flags require prefill-memory-and-extend.patch"
                    )
                os.environ["VLLM_TURING_PREFILL_WINDOW"] = str(a.window)
                os.environ["VLLM_TURING_EXTEND_MAX"] = str(a.extend_max)
                candidate = TuringPrefillAttention(
                    length, 4, 256, torch.device("cuda"), **init_kw
                )
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        used = torch.tensor([length], dtype=torch.int32, device="cuda")
        dense = fn = attn = None
        for qlen in a.queries:
            if qlen > length:
                continue
            q = torch.randn(qlen, 24, 256, device="cuda", dtype=torch.float16)
            out, ref = torch.empty_like(q), torch.empty_like(q)
            cu = torch.tensor([0, qlen], dtype=torch.int32, device="cuda")
            dense = partial(prefill.run, q, k, v, ref, table, length, 1 / 16, ks, vs)
            dense_ms = timing(dense, iterations=a.iterations)
            print(
                json.dumps(
                    {
                        "tag": a.tag,
                        "kv": length,
                        "q": qlen,
                        "path": "gather_cutlass",
                        "ms": dense_ms,
                    }
                ),
                flush=True,
            )
            if candidate is not None:
                fn = partial(candidate.run, q, k, v, out, table, length, 1 / 16, ks, vs)
                ms = timing(fn, iterations=a.iterations)
                err = errors(out, ref)
                assert torch.isfinite(out).all() and err["relative_rms"] < 0.01, err
                print(
                    json.dumps(
                        dict(
                            tag=a.tag,
                            kv=length,
                            q=qlen,
                            path="candidate",
                            window=a.window,
                            extend_max=a.extend_max,
                            workspace_bytes=candidate.workspace_bytes,
                            ms=ms,
                            speedup=dense_ms / ms,
                            **err,
                        )
                    ),
                    flush=True,
                )
            if qlen <= 256:
                for nseg in a.segments:
                    attn = SpecDecodeAttention(
                        1,
                        24,
                        256,
                        torch.device("cuda"),
                        qlen,
                        num_segments=nseg,
                        packed_int4=True,
                    )
                    fn = partial(
                        attn.run, q, k, v, out, cu, used, table, 1 / 16, 1, qlen, ks, vs
                    )
                    ms = timing(fn, iterations=a.iterations)
                    err = errors(out, ref)
                    assert torch.isfinite(out).all() and err["relative_rms"] < 0.01, err
                    print(
                        json.dumps(
                            dict(
                                tag=a.tag,
                                kv=length,
                                q=qlen,
                                path="paged_split",
                                segments=nseg,
                                ms=ms,
                                speedup=dense_ms / ms,
                                **err,
                            )
                        ),
                        flush=True,
                    )
                    del attn
            if a.profile and length == a.lengths[-1] and qlen == a.queries[-1]:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    dense()
                    torch.cuda.synchronize()
                prof.export_chrome_trace(a.profile)
                print(
                    prof.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=15
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        dense = fn = attn = None
        del k, v, ks, vs, table, prefill, candidate
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
