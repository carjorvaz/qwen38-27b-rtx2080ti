#!/usr/bin/env python3
"""Quality gates that actually exercise large-M prefill (not 300-token prompts).

PPL: non-overlapping token windows from a UTF-8 corpus (or wikitext parquet),
with cache salts and prompt_logprobs=0. Run with spare logits memory, e.g. the
64k / 3 GiB KV quality profile, not a VRAM-saturated production server.
Needles: three independent retrieval targets per long prompt, then exact replay.
JSONL output. A retrieval miss, missing logprob, or preemption fails the run.
"""

import argparse
import json
import math
import random
import time

from turing_api_bench import metrics, request, tokenize


def post(base, path, body):
    with request(base, path, body) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://127.0.0.1:18021")
    p.add_argument("--tag", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--needle-lengths", type=int, nargs="*", default=[])
    a = p.parse_args()
    if a.corpus.endswith(".parquet"):
        import pyarrow.parquet as pq

        corpus = "".join(pq.read_table(a.corpus, columns=["text"])["text"].to_pylist())
    else:
        with open(a.corpus) as f:
            corpus = f.read()
    tokens = tokenize(a.base, corpus)
    assert len(tokens) >= a.samples * a.window, (
        "corpus too short for disjoint PPL windows"
    )
    logprob_sum = 0.0
    scored = 0
    for i in range(a.samples):
        prompt = tokens[i * a.window : (i + 1) * a.window]
        r = post(
            a.base,
            "/v1/completions",
            {
                "model": "qwen3.8-27b",
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "prompt_logprobs": 0,
                "cache_salt": f"{a.tag}-ppl-{i}",
            },
        )
        entries = r["choices"][0]["prompt_logprobs"]
        assert len(entries) == len(prompt)
        lps = []
        for token, entry in zip(prompt[1:], entries[1:]):
            assert entry is not None and str(token) in entry, (
                "missing actual-token logprob"
            )
            lp = entry[str(token)]["logprob"]
            assert math.isfinite(lp), "nonfinite logprob"
            lps.append(lp)
        logprob_sum += sum(lps)
        scored += len(lps)
        print(
            json.dumps(
                {
                    "tag": a.tag,
                    "kind": "ppl_window",
                    "sample": i,
                    "tokens": len(lps),
                    "logprob_sum": sum(lps),
                }
            ),
            flush=True,
        )
    if scored:
        print(
            json.dumps(
                {
                    "tag": a.tag,
                    "kind": "ppl",
                    "tokens": scored,
                    "value": math.exp(-logprob_sum / scored),
                }
            ),
            flush=True,
        )
    rng = random.Random(1729)
    for length in a.needle_lengths:
        codes = [f"K{rng.getrandbits(32):08X}" for _ in range(3)]
        # Reserve room for needles, question and chat framing; report the
        # server's actual token count rather than a characters/token estimate.
        assert length > 512
        haystack = (tokens * (1 + length // len(tokens)))[: length - 256]
        original_length = len(haystack)
        for depth, name, code in reversed(
            list(zip((0.1, 0.5, 0.9), ("alpha", "beta", "gamma"), codes))
        ):
            pos = int(original_length * depth)
            needle = tokenize(a.base, f"\nThe unique {name} audit key is {code}.\n")
            haystack[pos:pos] = needle
        text = post(
            a.base, "/detokenize", {"model": "qwen3.8-27b", "tokens": haystack}
        )["prompt"]
        prompt = (
            f"Audit document {length}.\n"
            + text
            + "\nReturn the alpha, beta, and gamma audit keys, in that order. Only the three keys."
        )
        body = {
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0,
            "cache_salt": f"{a.tag}-needle-{length}",
            "chat_template_kwargs": {
                "enable_thinking": False,
                "reasoning_effort": "none",
            },
        }
        for replay in (False, True):
            before = metrics(a.base)
            start = time.perf_counter()
            r = post(a.base, "/v1/chat/completions", body)
            elapsed = time.perf_counter() - start
            answer = r["choices"][0]["message"].get("content") or ""
            ok = all(code in answer for code in codes)
            preemptions = (
                metrics(a.base)["vllm:num_preemptions_total"]
                - before["vllm:num_preemptions_total"]
            )
            print(
                json.dumps(
                    {
                        "tag": a.tag,
                        "kind": "needle",
                        "requested_length": length,
                        "replay": replay,
                        "ok": ok,
                        "answer": answer,
                        "usage": r["usage"],
                        "elapsed_s": elapsed,
                        "preemptions": preemptions,
                    }
                ),
                flush=True,
            )
            assert ok and preemptions == 0, "retrieval or capacity gate failed"
            cached = (r["usage"].get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            )
            assert cached is not None, "enable --enable-prompt-tokens-details"
            assert (cached > 0) if replay else (cached == 0), "unexpected cache state"


if __name__ == "__main__":
    main()
