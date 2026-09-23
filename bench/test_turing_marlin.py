#!/usr/bin/env python3
"""Transient W4A8 packing, negative-scale, small-M and compile-hash guards."""

import os
from functools import partial

import torch
from turing_attention_bench import errors
from vllm import envs
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace,
    marlin_quantize,
)
from vllm.model_executor.layers.quantization.utils.turing_marlin_prefill import repack
from vllm.scalar_type import scalar_types


def main():
    assert torch.cuda.get_device_capability() == (7, 5)
    os.environ["VLLM_TURING_PREFILL_INT8"] = "off"
    baseline = envs.compile_factors()
    os.environ["VLLM_TURING_PREFILL_INT8"] = "all"
    changed = envs.compile_factors()
    assert baseline["VLLM_TURING_PREFILL_INT8"] == "off"
    assert changed["VLLM_TURING_PREFILL_INT8"] == "all"
    assert baseline != changed
    torch.manual_seed(1729)
    for k, n in ((512, 512), (1024, 1536)):
        w = torch.randn(k, n, device="cuda", dtype=torch.float16) / k**0.5
        ref, w16, s16, idx, sort, _ = marlin_quantize(
            w, scalar_types.uint4b8, 128, False
        )
        _, w8, _, _, _, _ = marlin_quantize(
            w, scalar_types.uint4b8, 128, False, input_dtype=torch.int8
        )
        assert torch.equal(repack(w16, s16)[0], w8), "different positive-scale packing"
        workspace = MarlinWorkspace(n, 64, 16).scratch
        empty = torch.empty(0, device="cuda", dtype=torch.int32)
        for negative in (False, True):
            if negative:
                s16[::2].neg_()
                ref.view(k // 128, 128, n)[::2].neg_()
            w_before, s_before = w16.clone(), s16.clone()
            for m in (1, 5, 128, 1024, 1632):
                x = torch.randn(m, k, device="cuda", dtype=torch.float16)

                run = partial(
                    apply_gptq_marlin_linear,
                    x,
                    w16,
                    s16,
                    empty,
                    idx,
                    sort,
                    workspace,
                    scalar_types.uint4b8,
                    n,
                    k,
                    True,
                )
                y = run(prefill_int8=True)
                if m < 1024:
                    assert torch.equal(y, run()), "small-M path changed"
                err = errors(y, x.float() @ ref.float())
                assert torch.isfinite(y).all() and err["relative_rms"] < 0.04, err
                assert torch.equal(w16, w_before) and torch.equal(s16, s_before), (
                    "canonical weights mutated"
                )

    # Profile/compile large M first, then execute and capture a decode shape.
    # This catches accidental specialization of the entire model to W4A8.
    def dispatch(inp):
        return apply_gptq_marlin_linear(
            inp,
            w16,
            s16,
            empty,
            idx,
            sort,
            workspace,
            scalar_types.uint4b8,
            n,
            k,
            True,
            prefill_int8=True,
        )

    compiled = torch.compile(
        dispatch,
        fullgraph=True,
        dynamic=True,
        options={"enable_auto_functionalized_v2": False},
    )
    large = torch.randn(1632, k, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(compiled(large), dispatch(large), rtol=0, atol=0)
    small = torch.randn(5, k, device="cuda", dtype=torch.float16)
    for _ in range(3):
        compiled(small)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(small)
    for _ in range(2):
        small.normal_()
        graph.replay()
        expected = apply_gptq_marlin_linear(
            small,
            w16,
            s16,
            empty,
            idx,
            sort,
            workspace,
            scalar_types.uint4b8,
            n,
            k,
            True,
        )
        assert torch.equal(captured, expected), "compiled/captured decode changed"
    print(
        "PASS: packing, negative scales, finite output, immutable weights, compile hash, dynamic-M compile and decode graph bit equality"
    )


if __name__ == "__main__":
    main()
