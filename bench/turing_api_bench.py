#!/usr/bin/env python3
"""Seeded SM75 API benchmark: cold prefill, replay, extensions and decode.

Uses token-ID completion prompts so requested lengths are exact. Prefix caching
stays enabled; a unique leading token salt prevents cross-case cache hits. Engine
warmup is separate from KV-cache reuse. JSONL on stdout; no service management.
"""

import argparse
import hashlib
import json
import os
import random
import time
import urllib.request
from pathlib import Path


def request(base, path, payload=None):
    headers = {"Content-Type": "application/json"}
    if os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["VLLM_API_KEY"]
    data = None if payload is None else json.dumps(payload).encode()
    return urllib.request.urlopen(
        urllib.request.Request(base + path, data, headers), timeout=3600
    )


def tokenize(base, text):
    with request(base, "/tokenize", {"model": "qwen3.8-27b", "prompt": text}) as r:
        return json.load(r)["tokens"]


def metrics(base):
    names = {
        "vllm:request_decode_time_seconds_sum",
        "vllm:request_decode_time_seconds_count",
        "vllm:request_prefill_time_seconds_sum",
        "vllm:num_preemptions_total",
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    }
    values = dict.fromkeys(names, 0.0)
    with request(base, "/metrics") as response:
        for raw in response:
            line = raw.decode()
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name in names:
                values[name] += float(line.rsplit(" ", 1)[-1])
    return values


def completion(base, tokens, output):
    body = {
        "model": "qwen3.8-27b",
        "prompt": tokens,
        "temperature": 0,
        "max_tokens": output,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": 1729,
    }
    before = metrics(base)
    start = time.perf_counter()
    first = None
    text = ""
    usage = None
    with request(base, "/v1/completions", body) as r:
        for raw in r:
            if not raw.startswith(b"data: ") or raw.strip() == b"data: [DONE]":
                continue
            event = json.loads(raw[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                chunk = choice.get("text", "")
                if chunk:
                    first = first or time.perf_counter()
                    text += chunk
    end = time.perf_counter()
    if first is None or usage is None:
        raise RuntimeError("stream missing content or usage")
    after = metrics(base)
    deadline = time.perf_counter() + 3
    while (
        after["vllm:request_decode_time_seconds_count"]
        <= before["vllm:request_decode_time_seconds_count"]
        and time.perf_counter() < deadline
    ):
        time.sleep(0.05)
        after = metrics(base)
    delta = {key.removeprefix("vllm:"): after[key] - before[key] for key in before}
    details = usage.get("prompt_tokens_details") or {}
    if details.get("cached_tokens") is None:
        raise RuntimeError("server must enable --enable-prompt-tokens-details")
    cached = details["cached_tokens"]
    return {
        "ttft_s": first - start,
        "total_s": end - start,
        "usage": usage,
        "uncached_tokens": usage["prompt_tokens"] - cached,
        "metrics_delta": delta,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "stream_decode_tok_s": (usage["completion_tokens"] - 1)
        / max(end - first, 1e-9),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://127.0.0.1:8080")
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 32768, 131072])
    p.add_argument("--tails", type=int, nargs="*", default=[16, 64, 256, 2048])
    p.add_argument("--output", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--tag", default="baseline")
    p.add_argument(
        "--corpus", help="Optional representative UTF-8 text; otherwise synthetic prose"
    )
    a = p.parse_args()
    text = (
        Path(a.corpus).read_text(encoding="utf-8")
        if a.corpus
        else (
            "The engineer checked the measured latency and recorded the result in the log. "
            "The next experiment changed only one variable, preserving the reference implementation. "
            "An independent correctness check compared the output against the expected answer.\n"
        )
    )
    corpus = tokenize(a.base, text)
    rng = random.Random(a.seed)
    completion(
        a.base,
        tokenize(a.base, "Warmup: explain why measurements need a baseline."),
        32,
    )
    for length in a.lengths:
        # Salt is before the repeated body, not after it. Same seed across fresh
        # A/B servers gives the same prompts; use a new seed when reusing a server.
        salt = tokenize(
            a.base, f"Experiment {rng.getrandbits(64)}. Read this document:\n"
        )
        prompt = (salt + corpus * (1 + length // len(corpus)))[:length]
        cases = [("cold", prompt), ("replay", prompt)]
        for tail in a.tails:
            # Each extension branches directly from the original document.
            branch = tokenize(a.base, f"\nBranch {rng.getrandbits(64)}: ")
            extra = (branch + corpus * (1 + tail // len(corpus)))[:tail]
            cases.append((f"extend_{tail}", prompt + extra))
        for kind, tokens in cases:
            result = completion(a.base, tokens, a.output if kind == "cold" else 1)
            result.update(tag=a.tag, seed=a.seed, kind=kind, input_tokens=len(tokens))
            print(json.dumps(result), flush=True)
            if kind == "cold" and result["uncached_tokens"] != len(tokens):
                raise RuntimeError(
                    "cold prompt hit cache; restart the server or use a fresh --seed"
                )
            if result["metrics_delta"]["num_preemptions_total"]:
                raise RuntimeError(
                    "request was preempted; timings are not a clean A/B comparison"
                )


if __name__ == "__main__":
    main()
