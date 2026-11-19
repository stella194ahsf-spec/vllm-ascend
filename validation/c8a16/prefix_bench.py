"""Fixed-token PD workload with separate local P-prefix and D-MTP metric deltas."""

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

import httpx


def counter(text, name):
    pattern = re.compile(r"^" + re.escape(name) + r"(?:_total)?(?:\{[^\n]*\})?\s+([0-9.eE+-]+)(?:\s|$)")
    values = [float(match[1]) for line in text.splitlines() if (match := pattern.match(line))]
    if not values:
        raise ValueError(f"Required metric not exported: {name}; inspect /metrics before changing its meaning")
    return sum(values)


async def request(client, url, payload):
    start = time.perf_counter()
    first = last = None
    usage = None
    finish = None
    async with client.stream("POST", url + "/v1/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            if data.get("error"):
                raise RuntimeError(data["error"])
            for choice in data.get("choices", []):
                if choice.get("text"):
                    last = time.perf_counter()
                    first = last if first is None else first
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
            usage = data.get("usage") or usage
    assert usage is not None, "stream did not return usage; no token counts may be inferred from SSE chunk count"
    assert usage["prompt_tokens"] == len(payload["prompt"]), usage
    assert usage["completion_tokens"] == payload["max_tokens"], usage
    assert finish == "length", finish
    assert first is not None and last is not None
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "output_tokens": usage["completion_tokens"],
        "ttft_ms": (first - start) * 1000,
        "tpot_ms": (last - first) * 1000 / max(usage["completion_tokens"] - 1, 1),
        "latency_ms": (time.perf_counter() - start) * 1000,
    }


async def bench(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    unit = tokenizer.encode(
        "This is a shared document about mathematics, engineering and computing. ", add_special_tokens=False
    )
    common_len = int(args.input_tokens * 0.98)
    common = (unit * (common_len // len(unit) + 1))[:common_len]
    prompts = []
    for i in range(args.requests + 1):
        tail = tokenizer.encode(f"Unique request {i}: explain the document carefully. ", add_special_tokens=False)
        ids = common + (tail * ((args.input_tokens - common_len) // len(tail) + 1))[: args.input_tokens - common_len]
        prompts.append(ids)
    digest = hashlib.sha256(json.dumps(prompts, separators=(",", ":")).encode()).hexdigest()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:

        def payload(ids, output_tokens):
            return {
                "model": "qwen36",
                "prompt": ids,
                "temperature": 0,
                "seed": 2026,
                "max_tokens": output_tokens,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

        # Isolate the warmup from the measured metric window. The complete
        # common prefix must reside in the P cache before taking the snapshot.
        await request(client, args.proxy, payload(prompts[0], 1))
        p_before = (await client.get(args.prefill + "/metrics")).raise_for_status().text
        d_before = (await client.get(args.decode + "/metrics")).raise_for_status().text
        semaphore = asyncio.Semaphore(args.concurrency)

        async def measured(ids):
            async with semaphore:
                return await request(client, args.proxy, payload(ids, args.output_tokens))

        start = time.perf_counter()
        records = await asyncio.gather(*(measured(ids) for ids in prompts[1:]))
        elapsed = time.perf_counter() - start
        p_after = (await client.get(args.prefill + "/metrics")).raise_for_status().text
        d_after = (await client.get(args.decode + "/metrics")).raise_for_status().text
    for label, text in (("p-before", p_before), ("p-after", p_after), ("d-before", d_before), ("d-after", d_after)):
        out.with_name(out.stem + "-" + label + ".metrics").write_text(text)
    hits = counter(p_after, "vllm:prefix_cache_hits") - counter(p_before, "vllm:prefix_cache_hits")
    queries = counter(p_after, "vllm:prefix_cache_queries") - counter(p_before, "vllm:prefix_cache_queries")
    drafts = counter(d_after, "vllm:spec_decode_num_drafts") - counter(d_before, "vllm:spec_decode_num_drafts")
    accepted = counter(d_after, "vllm:spec_decode_num_accepted_tokens") - counter(
        d_before, "vllm:spec_decode_num_accepted_tokens"
    )
    rate = hits / queries if queries > 0 else 0
    result = {
        "group": args.group,
        "workload_sha256": digest,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "p_local_prefix_hits": hits,
        "p_local_prefix_queries": queries,
        "p_local_prefix_hit_rate": rate,
        "d_drafts": drafts,
        "d_accepted": accepted,
        "acceptance_length": 1 + accepted / drafts if drafts else None,
        "output_tokens_per_second": sum(row["output_tokens"] for row in records) / elapsed,
        "median_ttft_ms": statistics.median(row["ttft_ms"] for row in records),
        "median_tpot_ms": statistics.median(row["tpot_ms"] for row in records),
        "records": records,
        "gate_passed": rate > args.min_hit_rate and drafts > 0 and accepted > 0,
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    assert result["gate_passed"], "P local prefix/MTP gate failed; do not use this run as target-scenario evidence"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--group", choices=("A", "B"), required=True)
    run.add_argument("--proxy", default="http://127.0.0.1:9000")
    run.add_argument("--prefill", default="http://127.0.0.1:8100")
    run.add_argument("--decode", default="http://127.0.0.1:8200")
    run.add_argument("--input-tokens", type=int, default=131072)
    run.add_argument("--output-tokens", type=int, default=5120)
    run.add_argument("--requests", type=int, default=4)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--timeout", type=float, default=21600)
    run.add_argument("--min-hit-rate", type=float, default=0.90)
    run.add_argument("--out", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)
    args = parser.parse_args()
    if args.command == "run":
        assert args.requests > 0 and 0 < args.concurrency <= args.requests
        assert args.input_tokens > 0 and args.output_tokens > 0
        asyncio.run(bench(args))
    else:
        a, b = (json.loads(Path(p).read_text()) for p in (args.a, args.b))
        assert a["group"] == "A" and b["group"] == "B"
        for key in ("workload_sha256", "input_tokens", "output_tokens", "requests", "concurrency"):
            assert a[key] == b[key], (key, a[key], b[key])
        assert a["gate_passed"] and b["gate_passed"]
        difference = abs(a["p_local_prefix_hit_rate"] - b["p_local_prefix_hit_rate"])
        assert difference <= 0.01, f"P cache hit rate differs by {difference:.2%}"
        print(
            json.dumps(
                {
                    "comparison_gate": "pass",
                    "p_hit_rate_difference_pp": difference * 100,
                    "b_vs_a_tpot_change_percent": (b["median_tpot_ms"] / a["median_tpot_ms"] - 1) * 100,
                    "b_vs_a_throughput_change_percent": (
                        b["output_tokens_per_second"] / a["output_tokens_per_second"] - 1
                    )
                    * 100,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
