#!/usr/bin/env python3
"""Probe and benchmark the installed SM75 Marlin W4A16/W4A8 paths.

A capability probe, not a claim that INT8_ACT is supported. Keeps quantization
outside timed regions but includes activation quantization in W4A8 timings.
Uses positive synthetic scales; checkpoint negative-scale handling still needs
an end-to-end quality gate before enabling this on a model.
"""

import argparse
import json
from functools import partial

import torch
from turing_attention_bench import errors, timing
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_gptq_marlin_linear,
    marlin_act_int8_process_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace,
    marlin_quantize,
)
from vllm.scalar_type import scalar_types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--full",
        action="store_true",
        help="Include representative projection/MLP shapes",
    )
    p.add_argument(
        "--transient",
        action="store_true",
        help="Also time prefill-only repacking, including its cost",
    )
    a = p.parse_args()
    assert torch.cuda.get_device_capability() == (7, 5)
    torch.manual_seed(1729)
    shapes = [(512, 512)] + (
        [(5120, 5120), (5120, 17408), (17408, 5120)] if a.full else []
    )
    unsupported = set()
    for k, n in shapes:
        w = torch.randn(k, n, dtype=torch.float16, device="cuda") / k**0.5
        modes = [("w4a16", None), ("w4a8", torch.int8)]
        if a.transient:
            modes.append(("transient", None))
        for name, dtype in modes:
            if name in unsupported:
                continue
            try:
                wref, weight, scales, idx, sort, _ = marlin_quantize(
                    w, scalar_types.uint4b8, 128, False, input_dtype=dtype
                )
                global_scale = None
                if dtype is not None:
                    scales, global_scale = marlin_act_int8_process_scales(scales)
                workspace = MarlinWorkspace(n, 64, 16).scratch
                empty = torch.empty(0, dtype=torch.int32, device="cuda")
                for m in [5, 256, 2048]:
                    x = torch.randn(m, k, dtype=torch.float16, device="cuda")
                    fn = partial(
                        apply_gptq_marlin_linear,
                        x,
                        weight,
                        scales,
                        empty,
                        idx,
                        sort,
                        workspace,
                        scalar_types.uint4b8,
                        n,
                        k,
                        True,
                        input_global_scale=global_scale,
                        input_dtype=dtype,
                        **({"prefill_int8": True} if name == "transient" else {}),
                    )
                    y = fn()
                    torch.cuda.synchronize()
                    err = errors(y, x.float() @ wref.float())
                    assert torch.isfinite(y).all() and err["relative_rms"] < 0.05, err
                    ms = timing(fn)
                    print(
                        json.dumps(dict(path=name, m=m, n=n, k=k, ms=ms, **err)),
                        flush=True,
                    )
            except RuntimeError as exc:
                if name != "w4a8":
                    raise
                unsupported.add(name)
                print(
                    json.dumps({"path": name, "n": n, "k": k, "error": str(exc)}),
                    flush=True,
                )


if __name__ == "__main__":
    main()
