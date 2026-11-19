"""Probe the installed FIA mode-5 multi-query causal contract on one A3 device."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu


def split_mode5_queries(
    query, key_int8, value_int8, key_scale, value_scale, lengths, common
):
    """Verify a uniform query suffix using existing single-query mode 5 calls."""
    query_length = query.shape[1]
    rows = []
    for j in range(query_length):
        visible_lengths = [length - query_length + j + 1 for length in lengths]
        row, _ = torch_npu.npu_fused_infer_attention_score(
            query[:, j : j + 1].contiguous(),
            key_int8.flatten(2),
            value_int8.flatten(2),
            key_antiquant_scale=key_scale,
            value_antiquant_scale=value_scale,
            key_antiquant_mode=5,
            value_antiquant_mode=5,
            actual_seq_lengths=[1] * query.shape[0],
            actual_seq_lengths_kv=visible_lengths,
            atten_mask=None,
            sparse_mode=0,
            **common,
        )
        rows.append(row.reshape(query.shape[0], 1, query.shape[2], query.shape[3]))
    return torch.cat(rows, dim=1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--strategy", choices=("native", "split"), default="native")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    torch.manual_seed(20261118)
    pages, block_size, kv_heads, heads, dim = 6, 128, 1, 8, 256
    lengths = [129, 133]
    table = torch.tensor([[4, 1], [3, 2]], dtype=torch.int32, device=device)
    # Quantize D independently for every physical page, offset and head.
    key = torch.randn(
        pages, block_size, kv_heads, dim, device=device, dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    key_int8, key_scale = torch_npu.npu_dynamic_quant(key, dst_type=torch.int8)
    value_int8, value_scale = torch_npu.npu_dynamic_quant(value, dst_type=torch.int8)
    key_scale = key_scale.clamp_min(torch.finfo(torch.float32).tiny)
    value_scale = value_scale.clamp_min(torch.finfo(torch.float32).tiny)
    key_dequant = (key_int8.float() * key_scale.unsqueeze(-1)).bfloat16().flatten(2)
    value_dequant = (
        (value_int8.float() * value_scale.unsqueeze(-1)).bfloat16().flatten(2)
    )
    common = {
        "num_heads": heads,
        "num_key_value_heads": kv_heads,
        "scale": dim**-0.5,
        "block_table": table,
        "block_size": block_size,
        "input_layout": "BSND",
        "inner_precise": 0,
    }
    fia = torch_npu.npu_fused_infer_attention_score
    results = []
    for query_length in (1, 2, 4):
        query = torch.randn(
            2, query_length, heads, dim, device=device, dtype=torch.bfloat16
        )
        # Independent reference for each query. Query j can read only through
        # length - query_length + j, inclusive. No multi-query mask is reused.
        expected_rows = []
        for j in range(query_length):
            visible_lengths = [length - query_length + j + 1 for length in lengths]
            expected, _ = fia(
                query[:, j : j + 1].contiguous(),
                key_dequant,
                value_dequant,
                actual_seq_lengths=[1, 1],
                actual_seq_lengths_kv=visible_lengths,
                atten_mask=None,
                sparse_mode=0,
                **common,
            )
            expected_rows.append(expected.reshape(2, 1, heads, dim))
        expected = torch.cat(expected_rows, dim=1).float().cpu()
        masks = [(0, None)]
        if args.strategy == "native":
            masks.append(
                (3, torch.ones(2048, 2048, device=device, dtype=torch.int8).triu_(1))
            )
        for sparse_mode, mask in masks:
            result = {
                "query_length": query_length,
                "strategy": args.strategy,
                "sparse_mode": sparse_mode,
                "mask": "None" if mask is None else "upper_triangular_2048",
            }
            try:
                if args.strategy == "split":
                    actual = split_mode5_queries(
                        query,
                        key_int8,
                        value_int8,
                        key_scale.transpose(1, 2).contiguous(),
                        value_scale.transpose(1, 2).contiguous(),
                        lengths,
                        common,
                    )
                else:
                    actual, _ = fia(
                        query,
                        key_int8.flatten(2),
                        value_int8.flatten(2),
                        key_antiquant_scale=key_scale.transpose(1, 2).contiguous(),
                        value_antiquant_scale=value_scale.transpose(1, 2).contiguous(),
                        key_antiquant_mode=5,
                        value_antiquant_mode=5,
                        actual_seq_lengths=[query_length, query_length],
                        actual_seq_lengths_kv=lengths,
                        atten_mask=mask,
                        sparse_mode=sparse_mode,
                        **common,
                    )
                torch.npu.synchronize()
                actual = actual.reshape(2, query_length, heads, dim).float().cpu()
                result["max_abs"] = (actual - expected).abs().max().item()
                result["per_query_max_abs"] = [
                    (actual[:, j] - expected[:, j]).abs().max().item()
                    for j in range(query_length)
                ]
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                result["status"] = "pass"
            except (RuntimeError, AssertionError) as exc:
                result["status"] = "fail"
                result["error"] = str(exc)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    passed = sum(row["status"] == "pass" for row in results)
    failed = len(results) - passed
    summary = {
        "strategy": args.strategy,
        "passed": passed,
        "failed": failed,
        "all_tested_cases_passed": failed == 0,
    }
    print(json.dumps({"summary": summary}), flush=True)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "kv_lengths": lengths,
                "contract": "query j sees KV length L-Sq+j+1",
                "results": results,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    baseline = next(
        row for row in results if row["query_length"] == 1 and row["sparse_mode"] == 0
    )
    if baseline["status"] != "pass":
        raise SystemExit(
            "Single-query baseline failed; inspect runtime before interpreting MTP."
        )
    if failed:
        raise SystemExit(
            f"FAIL: {passed} passed, {failed} failed. Inspect each case in JSON."
        )
    print(
        f"PASS: all {passed} {args.strategy} cases passed. "
        "This validates tested attention shapes; PD and MTP state handling need separate validation."
    )


if __name__ == "__main__":
    main()
