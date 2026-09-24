#!/usr/bin/env python3
"""GPU regression tests for the native SM75 GDN chunk kernels.

No model, no server. The serving path replaces two FLA Triton kernels with
native WMMA implementations -- ``chunk_fwd_o`` (chunk output, `turing_gdn_o.py`)
and ``chunk_gated_delta_rule_fwd_h`` (chunk state, `turing_gdn.py`). Both keep
the reference importable and select it per call via ``VLLM_TURING_GDN_O`` /
``VLLM_TURING_GDN_STATE``, so one process can run the same inputs through both
paths at the model's geometry (Hg=16, H=48, K=V=128, chunk 64).

Covers: the full chunk rule end to end, the two sub-kernels directly, partial
and single-row chunks, varlen sequences, the escape hatches (bitwise equality
with the reference when disabled), and timing at the 32k prefill shape when
VRAM allows. Requires gdn-chunk-state, gdn-chunk-o and gdn-state-16w.
"""

import os
import time

import torch

from vllm.third_party.flash_linear_attention.ops.chunk import (
    chunk_fwd_o,
    chunk_gated_delta_rule_fwd,
)
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h as reference_state,
)
from vllm.third_party.flash_linear_attention.ops.chunk_o import (
    chunk_fwd_o as reference_o,
)
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.third_party.flash_linear_attention.ops.turing_gdn import (
    chunk_gated_delta_rule_fwd_h,
)
from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd

HG, H, K, V, BT = 16, 48, 128, 128, 64
DEVICE = torch.device("cuda:0")
# Both kernels accumulate in fp32 and consume fp16; they fold the chunk decay at
# different points, so they are not bitwise equal. Measured relative RMS on the
# model's geometry is 3.0e-4 (chunk output, 32k); 1e-3 is a 3x margin on that,
# tight enough to catch a real indexing or layout bug.
TOLERANCE = 1e-3
# The 32k case allocates ~2 GB of fixtures plus the reference's workspace.
BIG_TOKENS = 32768
BIG_FREE_BYTES = 6 << 30


def set_native(output=True, state=True):
    os.environ["VLLM_TURING_GDN_O"] = "1" if output else "0"
    os.environ["VLLM_TURING_GDN_STATE"] = "1" if state else "0"


def rel_rms(out, ref):
    diff = out.float() - ref.float()
    scale = ref.float().square().mean().clamp_min(1e-20)
    return (diff.square().mean() / scale).sqrt().item()


def shapes(cu, T):
    indices = prepare_chunk_indices(cu, BT)
    return indices, prepare_chunk_offsets(cu, BT), len(indices)


def fixtures(T, lens=None, seed=0):
    """Random GDN inputs; ``lens`` makes a varlen batch instead of one sequence."""
    torch.manual_seed(seed)
    head = torch.tensor([0] + [sum(lens[: i + 1]) for i in range(len(lens))],
                        dtype=torch.int32) if lens else torch.tensor([0, T],
                        dtype=torch.int32)
    cu = head.to(DEVICE)
    ci, off, chunks = shapes(cu, T)
    raw = -(torch.rand(1, T, H, device=DEVICE, dtype=torch.float32) * 0.05)
    q = torch.randn(1, T, HG, K, device=DEVICE, dtype=torch.float16) * 0.2
    k = torch.randn(1, T, HG, K, device=DEVICE, dtype=torch.float16) * 0.2
    v = torch.randn(1, T, H, V, device=DEVICE, dtype=torch.float16) * 0.2
    return {
        "T": T,
        "chunks": chunks,
        "cu": cu,
        "ci": ci,
        "off": off,
        "q": q,
        "k": k,
        "v": v,
        "beta": torch.rand(1, T, H, device=DEVICE, dtype=torch.float32),
        "g_raw": raw,
        "g": chunk_local_cumsum(raw, chunk_size=BT, cu_seqlens=cu,
                                chunk_indices=ci),
        # The state pass emits h in its serving layout, [1, chunks, H, V, K].
        "h": torch.randn(1, chunks, H, V, K, device=DEVICE,
                         dtype=torch.float16) * 0.2,
        "scale": K ** -0.5,
    }


def wy(c):
    """The WY pass' w and u for fixtures -- the state kernels' real inputs.

    Random w/u are not usable here: they are not the delta-rule's transformed
    keys and values, and the recurrence then grows without bound (h reaches
    5e4 at 1k tokens and inf at 2k in *both* the native kernel and the Triton
    reference).
    """
    g = c["g"]
    A = chunk_scaled_dot_kkt_fwd(k=c["k"], beta=c["beta"], g=g,
                                 cu_seqlens=c["cu"], chunk_indices=c["ci"],
                                 output_dtype=torch.float32)
    A = solve_tril(A=A, cu_seqlens=c["cu"], chunk_indices=c["ci"],
                   output_dtype=c["k"].dtype)
    return recompute_w_u_fwd(k=c["k"], v=c["v"], beta=c["beta"], A=A,
                             g_cumsum=g, cu_seqlens=c["cu"],
                             chunk_indices=c["ci"])


def check_full_rule():
    """The entry point the model calls, references vs natives, same inputs."""
    cases = [
        ("one 512-token sequence", 512, None),
        ("partial last chunk", 200, None),
        ("single-row chunk", 65, None),
        ("two sequences", 250, [100, 150]),
        ("three ragged sequences", 300, [64, 128, 108]),
    ]
    for name, T, lens in cases:
        c = fixtures(T, lens)
        args = dict(q=c["q"], k=c["k"], v=c["v"], g=c["g_raw"], beta=c["beta"],
                    scale=c["scale"], initial_state=None,
                    output_final_state=True, cu_seqlens=c["cu"],
                    chunk_indices=c["ci"], chunk_offsets=c["off"])
        set_native(False, False)
        _, ref, _, ref_fs, _, _, _ = chunk_gated_delta_rule_fwd(**args)
        set_native()
        _, out, _, out_fs, _, _, _ = chunk_gated_delta_rule_fwd(**args)
        assert torch.isfinite(out).all(), name
        err, err_fs = rel_rms(out, ref), rel_rms(out_fs, ref_fs)
        assert err < TOLERANCE and err_fs < TOLERANCE, (name, err, err_fs)
        print(f"  {name:<24} o rel_rms {err:.2e}  final_state {err_fs:.2e}")
        del c
        torch.cuda.empty_cache()
    print("PASS: full chunk rule matches the reference on every geometry")


def check_chunk_output():
    # Correctness does not need a model-sized batch; keep the footprint small
    # enough to run against a live server with a few hundred MiB free.
    c = fixtures(1024)
    set_native()
    native = chunk_fwd_o(q=c["q"], k=c["k"], v=c["v"], h=c["h"], g=c["g"],
                         scale=c["scale"], cu_seqlens=c["cu"],
                         chunk_indices=c["ci"])
    ref = reference_o(q=c["q"], k=c["k"], v=c["v"], h=c["h"], g=c["g"],
                      scale=c["scale"], cu_seqlens=c["cu"],
                      chunk_indices=c["ci"])
    err = rel_rms(native, ref)
    assert torch.isfinite(native).all() and err < TOLERANCE, err
    # VLLM_TURING_GDN_O=0 must select the reference itself, bit for bit.
    set_native(output=False)
    fallback = chunk_fwd_o(q=c["q"], k=c["k"], v=c["v"], h=c["h"], g=c["g"],
                           scale=c["scale"], cu_seqlens=c["cu"],
                           chunk_indices=c["ci"])
    assert torch.equal(fallback, ref), "VLLM_TURING_GDN_O=0 is not the reference"
    # Unsupported geometry (bf16 activations) falls back too, not silently wrong.
    set_native()
    zeros = dict(device=DEVICE, dtype=torch.bfloat16)
    unsupported = chunk_fwd_o(q=c["q"].to(torch.bfloat16), k=c["k"].to(torch.bfloat16),
                              v=torch.zeros(1, c["T"], H, V, **zeros),
                              h=torch.zeros(1, c["chunks"], H, V, K, **zeros), g=c["g"],
                              scale=c["scale"], cu_seqlens=c["cu"],
                              chunk_indices=c["ci"])
    assert unsupported.dtype == torch.bfloat16 and torch.isfinite(unsupported).all()
    print(f"  chunk output (1024 tokens) rel_rms {err:.2e}; "
          f"escape hatch and bf16 fallback are the reference")
    del c
    torch.cuda.empty_cache()


def check_chunk_state():
    c = fixtures(1024)
    w, u = wy(c)
    args = dict(k=c["k"], w=w, u=u, g=c["g"], cu_seqlens=c["cu"],
                chunk_indices=c["ci"], chunk_offsets=c["off"],
                output_final_state=True)
    set_native()
    h, v_new, final = chunk_gated_delta_rule_fwd_h(**args)
    ref_h, ref_v_new, ref_final = reference_state(**args)
    err_h, err_v = rel_rms(h, ref_h), rel_rms(v_new, ref_v_new)
    err_final = rel_rms(final, ref_final)
    assert torch.isfinite(h).all() and err_h < TOLERANCE, err_h
    assert err_v < TOLERANCE, err_v
    assert final.dtype == torch.float32 and err_final < TOLERANCE, err_final
    assert h.shape == (1, c["chunks"], H, V, K), h.shape
    set_native(state=False)
    fb_h, fb_v_new, fb_final = chunk_gated_delta_rule_fwd_h(**args)
    assert (torch.equal(fb_h, ref_h) and torch.equal(fb_v_new, ref_v_new)
            and torch.equal(fb_final, ref_final)), \
        "VLLM_TURING_GDN_STATE=0 is not the reference"
    print(f"  chunk state (1024 tokens) h rel_rms {err_h:.2e}  v_new {err_v:.2e}  "
          f"final {err_final:.2e}; escape hatch is the reference")
    del c
    torch.cuda.empty_cache()


def timeit(fn, iters=10, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def check_timing():
    free, _ = torch.cuda.mem_get_info()
    if free < BIG_FREE_BYTES:
        print(f"  SKIP 32k chunk output: {free / 2**30:.1f} GiB free, "
              f"{BIG_FREE_BYTES / 2**30:.0f} GiB needed (stop the server to run it)")
    else:
        c = fixtures(BIG_TOKENS)
        args = dict(q=c["q"], k=c["k"], v=c["v"], h=c["h"], g=c["g"],
                    scale=c["scale"], cu_seqlens=c["cu"], chunk_indices=c["ci"])
        set_native()
        native = timeit(lambda: chunk_fwd_o(**args), iters=5)
        ref = timeit(lambda: reference_o(**args), iters=3, warmup=1)
        flops = c["chunks"] * H * (2 * BT * K * V + 2 * BT * K * BT + 2 * BT * BT * V)
        print(f"  chunk output @32k: native {native:.1f} ms "
              f"({flops / native / 1e9:.2f} TFLOPS) vs reference {ref:.1f} ms "
              f"({flops / ref / 1e9:.2f} TFLOPS) = {ref / native:.2f}x")
        del c
        torch.cuda.empty_cache()
    c = fixtures(2048)
    w, u = wy(c)
    args = dict(k=c["k"], w=w, u=u, g=c["g"], cu_seqlens=c["cu"],
                chunk_indices=c["ci"], chunk_offsets=c["off"],
                output_final_state=True)
    set_native()
    native = timeit(lambda: chunk_gated_delta_rule_fwd_h(**args), iters=20)
    ref = timeit(lambda: reference_state(**args), iters=5, warmup=1)
    print(f"  chunk state @2048: native {native:.2f} ms vs reference {ref:.2f} ms "
          f"= {ref / native:.2f}x")


def main():
    assert torch.cuda.get_device_capability() == (7, 5), "SM75 only"
    for flag in ("VLLM_TURING_GDN_O", "VLLM_TURING_GDN_STATE"):
        assert flag not in os.environ or os.environ[flag] == "1", \
            f"unset {flag}: this test drives the flags itself"
    set_native()
    free, total = torch.cuda.mem_get_info()
    print(f"  GPU free {free / 2**30:.2f} of {total / 2**30:.2f} GiB")
    check_full_rule()
    check_chunk_output()
    check_chunk_state()
    check_timing()
    print("PASS: GDN chunk kernels -- reference parity, escape hatches, timing")


if __name__ == "__main__":
    main()
