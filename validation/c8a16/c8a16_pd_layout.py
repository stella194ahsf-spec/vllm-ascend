"""Check the runner-owned C8A16 four-component cache layout on one A3 device.

Builds one combined allocation exactly as the runner now does (K, V and both
fp32 scale planes budgeted into one page), takes the four views, scatters a
dynamic-quantized KV through them, and runs the mode-5 FIA decode on the
result. Also verifies that a page-to-page copy carries all four components.
"""

import argparse
import json
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.attention.attention_c8a16 import (
    c8a16_cache_spec,
    c8a16_cache_views_for_spec,
    scatter_c8a16_scale_cache,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    torch.manual_seed(20261118)

    block_size, kv_heads, heads, dim = 128, 1, 8, 256
    num_blocks = 6
    results = {}

    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = c8a16_cache_spec(
        FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=kv_heads,
            head_size=dim,
            dtype=torch.int8,
        )
    )
    dense_page = block_size * kv_heads * 2 * dim
    results["page_bytes"] = spec.page_size_bytes
    results["dense_kv_page_bytes"] = dense_page
    results["scale_bytes_per_page"] = spec.page_size_bytes - dense_page
    assert results["scale_bytes_per_page"] == block_size * kv_heads * 2 * 4

    raw = torch.zeros(num_blocks * spec.page_size_bytes, dtype=torch.int8, device=device)
    k_cache, v_cache, k_scale, v_scale = c8a16_cache_views_for_spec(
        raw, num_blocks, spec.page_size_bytes, block_size, kv_heads, dim, dim
    )
    results["shapes"] = {
        "key": list(k_cache.shape),
        "value": list(v_cache.shape),
        "key_scale": list(k_scale.shape),
        "value_scale": list(v_scale.shape),
    }
    results["dtypes"] = {
        "key": str(k_cache.dtype),
        "key_scale": str(k_scale.dtype),
    }
    results["scale_page_pitch"] = k_scale.stride(0) * k_scale.element_size()
    results["kv_page_pitch"] = k_cache.stride(0) * k_cache.element_size()

    # Two requests, unequal lengths, out-of-order physical pages.
    lengths = [129, 133]
    table = torch.tensor([[4, 1], [3, 2]], dtype=torch.int32, device=device)
    slots = []
    for row, length in zip(table.tolist(), lengths):
        for t in range(length):
            slots.append(row[t // block_size] * block_size + t % block_size)
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=device)

    key = torch.randn(len(slots), kv_heads, dim, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    k_int8, k_sc = torch_npu.npu_dynamic_quant(key, dst_type=torch.int8)
    v_int8, v_sc = torch_npu.npu_dynamic_quant(value, dst_type=torch.int8)
    k_sc = k_sc.reshape(k_sc.shape[0], -1).clamp_min(torch.finfo(torch.float32).tiny)
    v_sc = v_sc.reshape(v_sc.shape[0], -1).clamp_min(torch.finfo(torch.float32).tiny)

    torch_npu.npu_scatter_pa_kv_cache(
        key=k_int8,
        value=v_int8,
        key_cache=k_cache,
        value_cache=v_cache,
        slot_mapping=slot_mapping,
        cache_mode="Norm",
    )
    scatter_c8a16_scale_cache(k_scale, slot_mapping, k_sc)
    scatter_c8a16_scale_cache(v_scale, slot_mapping, v_sc)
    torch.npu.synchronize()

    # Reference: dequantize the same INT8 payload with the same scales.
    k_ref = (k_int8.float() * k_sc.unsqueeze(-1)).bfloat16()
    v_ref = (v_int8.float() * v_sc.unsqueeze(-1)).bfloat16()

    query = torch.randn(2, 1, heads, dim, device=device, dtype=torch.bfloat16)
    common = {
        "num_heads": heads,
        "num_key_value_heads": kv_heads,
        "scale": dim**-0.5,
        "block_size": block_size,
        "input_layout": "BSND",
        "inner_precise": 0,
        "sparse_mode": 0,
        "atten_mask": None,
        "actual_seq_lengths": [1, 1],
        "actual_seq_lengths_kv": lengths,
    }
    actual, _ = torch_npu.npu_fused_infer_attention_score(
        query,
        k_cache.view(num_blocks, block_size, kv_heads * dim),
        v_cache.view(num_blocks, block_size, kv_heads * dim),
        key_antiquant_scale=k_scale,
        value_antiquant_scale=v_scale,
        key_antiquant_mode=5,
        value_antiquant_mode=5,
        block_table=table,
        **common,
    )
    # Same math with the explicit dequantized reference and no antiquant.
    expected_rows = []
    for j in range(2):
        visible = [lengths[j]]
        dense_k = k_ref[sum(lengths[:j]) : sum(lengths[: j + 1])].unsqueeze(0)
        dense_v = v_ref[sum(lengths[:j]) : sum(lengths[: j + 1])].unsqueeze(0)
        row, _ = torch_npu.npu_fused_infer_attention_score(
            query[j : j + 1],
            dense_k,
            dense_v,
            **{**common, "actual_seq_lengths": [1], "actual_seq_lengths_kv": visible},
        )
        expected_rows.append(row)
    expected = torch.cat(expected_rows, dim=0)
    torch.npu.synchronize()
    results["max_abs"] = (actual.float() - expected.float()).abs().max().item()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.02)
    results["numeric"] = "pass"

    # Page copy must carry all four components (prefix-cache reuse / recycle).
    src, dst = 4, 5
    for cache in (k_cache, v_cache, k_scale, v_scale):
        cache[dst] = cache[src].clone()
    torch.npu.synchronize()
    results["page_copy"] = {
        "key": bool(torch.equal(k_cache[dst], k_cache[src])),
        "value": bool(torch.equal(v_cache[dst], v_cache[src])),
        "key_scale": bool(torch.equal(k_scale[dst], k_scale[src])),
        "value_scale": bool(torch.equal(v_scale[dst], v_scale[src])),
    }
    assert all(results["page_copy"].values())

    results["status"] = "pass"
    print(json.dumps(results, ensure_ascii=False), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
