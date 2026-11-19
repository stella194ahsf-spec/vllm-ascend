"""Compare common-history greedy outputs without comparing divergent histories."""

import argparse
import json
from pathlib import Path


def compare(reference, candidate, tolerance):
    assert len(reference["outputs"]) == len(candidate["outputs"])
    records = []
    for a, b in zip(reference["outputs"], candidate["outputs"]):
        assert a["repeat"] == b["repeat"] and a["prompt_tokens"] == b["prompt_tokens"]
        assert a["finish_reason"] == b["finish_reason"] == "length"
        assert len(a["token_ids"]) == len(b["token_ids"])
        divergence = next((i for i, pair in enumerate(zip(a["token_ids"], b["token_ids"])) if pair[0] != pair[1]), None)
        if divergence is None:
            records.append({"status": "exact", "prompt_tokens": a["prompt_tokens"], "repeat": a["repeat"]})
            continue
        # At the first divergence, preceding output IDs are identical. Later
        # positions no longer condition on the same history and are excluded.
        pa, pb = a["top_logprobs"][divergence], b["top_logprobs"][divergence]
        selected = {str(a["token_ids"][divergence]), str(b["token_ids"][divergence])}
        covered = selected <= pa.keys() and selected <= pb.keys()
        common = pa.keys() & pb.keys()
        error = max((abs(pa[t] - pb[t]) for t in common), default=float("inf"))
        margins = []
        for probabilities in (pa, pb):
            ranked = sorted(probabilities.values(), reverse=True)
            margins.append(ranked[0] - ranked[1] if len(ranked) > 1 else float("inf"))
        # Report a bounded tie candidate, never an automatic full-sequence
        # correctness PASS. A teacher-forced distribution test remains separate.
        tie_candidate = covered and error <= tolerance and max(margins) <= 2 * tolerance
        records.append(
            {
                "status": "tie_candidate" if tie_candidate else "review_required",
                "prompt_tokens": a["prompt_tokens"],
                "repeat": a["repeat"],
                "first_divergent_token": divergence,
                "selected_tokens": sorted(selected),
                "selected_candidates_covered": covered,
                "topk_common_max_logprob_error": error,
                "top1_top2_margins": margins,
            }
        )
    return {
        "status": "exact" if all(row["status"] == "exact" for row in records) else "review_required",
        "tolerance": tolerance,
        "records": records,
        "note": "tie candidates require review; no comparisons beyond the first divergent token",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--logprob-atol", type=float, default=0.02)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(Path(args.reference).read_text()), json.loads(Path(args.candidate).read_text()), args.logprob_atol
    )
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "exact" else 2)
