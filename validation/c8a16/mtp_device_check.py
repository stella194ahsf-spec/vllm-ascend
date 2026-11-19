"""A3 check of the actual ragged Mode-5 backend, including rejected slots."""

import argparse
import json
import math
from itertools import accumulate
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu

from vllm_ascend.attention.attention_c8a16 import (
    AscendC8A16AttentionBackendImpl,
    c8a16_cache_views_for_spec,
    dyn_quant_kv,
    scatter_c8a16_scale_cache,
)
from vllm_ascend.device.device_op import DeviceOperator


@torch.inference_mode()
def run(device, long_context=False):
    torch.npu.set_device(device)
    torch.manual_seed(20261119)
    device = f"npu:{device}"
    block, heads, dim = 128, 8, 256
    lengths, widths = [131071, 131073, 136192] if long_context else [127, 129, 133], [1, 2, 4]
    blocks_per_request = math.ceil(max(lengths) / block)
    pages = 3 * blocks_per_request + 2
    pitch = block * (2 * dim + 8)
    raw = torch.zeros(pages * pitch, dtype=torch.int8, device=device)
    caches = c8a16_cache_views_for_spec(raw, pages, pitch, block, 1, dim, dim)
    key, value, ks, vs = caches
    table = torch.arange(3 * blocks_per_request, dtype=torch.int32, device=device).flip(0).reshape(3, -1)
    rows = table.cpu().tolist()

    def scatter(k, v, slots):
        ki, sk = dyn_quant_kv(k, len(k))
        vi, sv = dyn_quant_kv(v, len(v))
        slots = torch.tensor(slots, dtype=torch.int64, device=device)
        DeviceOperator.reshape_and_cache(key=ki, value=vi, key_cache=key, value_cache=value, slot_mapping=slots)
        scatter_c8a16_scale_cache(ks, slots, sk)
        scatter_c8a16_scale_cache(vs, slots, sv)

    slots = [rows[i][t // block] * block + t % block for i, length in enumerate(lengths) for t in range(length)]
    scatter(
        torch.randn(len(slots), 1, dim, device=device).bfloat16(),
        torch.randn(len(slots), 1, dim, device=device).bfloat16(),
        slots,
    )
    impl = object.__new__(AscendC8A16AttentionBackendImpl)
    impl.use_bnsd_kv_cache = False
    impl.num_heads, impl.num_kv_heads, impl.scale = heads, 1, dim**-0.5
    impl.key_cache, impl.value_cache, impl.k_scale_cache, impl.v_scale_cache = caches

    def check(widths, lengths):
        count = sum(widths)
        query = torch.randn(count + 2, heads, dim, device=device).bfloat16()
        output = torch.full((count + 2, heads * dim), -7, device=device, dtype=torch.bfloat16)
        metadata = SimpleNamespace(
            num_actual_tokens=count,
            actual_seq_lengths_q=[*accumulate(widths), count + 2],
            seq_lens_list=[*lengths, 1],
            block_tables=torch.cat([table, table[:1]]),
        )
        impl._forward_c8a16_verification(query, metadata, output)
        references, position = [], 0
        for request, (width, length) in enumerate(zip(widths, lengths)):
            for step in range(width):
                visible = length - width + step + 1
                logical = torch.arange(visible, device=device)
                physical = table[request, logical // block].long()
                offset = logical % block
                kd = (key[physical, offset].float() * ks[physical, 0, offset, None, None]).bfloat16()
                vd = (value[physical, offset].float() * vs[physical, 0, offset, None, None]).bfloat16()
                reference, _ = torch_npu.npu_fused_infer_attention_score(
                    query[position : position + 1].unsqueeze(1).contiguous(),
                    kd.unsqueeze(0).contiguous(),
                    vd.unsqueeze(0).contiguous(),
                    input_layout="BSND",
                    num_heads=heads,
                    num_key_value_heads=1,
                    scale=dim**-0.5,
                    actual_seq_lengths=[1],
                    actual_seq_lengths_kv=[visible],
                    sparse_mode=0,
                    atten_mask=None,
                    inner_precise=0,
                )
                references.append(reference.reshape(1, -1))
                position += 1
        expected = torch.cat(references)
        torch.testing.assert_close(output[:count], expected, rtol=0.02, atol=0.02)
        assert bool((output[count:] == -7).all()), "backend wrote graph padding"
        return {
            "widths": widths,
            "lengths": lengths,
            "max_abs": (output[:count] - expected).abs().max().item(),
            "fia_calls": max(widths),
            "status": "pass",
        }

    report = [check(widths, lengths)]
    # Simulate accepting only the first draft of request 2. Rejected future
    # positions stay resident and poisoned; the replacement token overwrites
    # both INT8 payloads and both scales at the first rejected position.
    for pos in (lengths[2] - 2, lengths[2] - 1):
        page, offset = rows[2][pos // block], pos % block
        key[page, offset].fill_(127)
        value[page, offset].fill_(-127)
        ks[page, 0, offset] = 100
        vs[page, 0, offset] = 100
    replacement_position = lengths[2] - 3
    replacement = rows[2][replacement_position // block] * block + replacement_position % block
    scatter(
        torch.randn(1, 1, dim, device=device).bfloat16(),
        torch.randn(1, 1, dim, device=device).bfloat16(),
        [replacement],
    )
    report.append(check([1, 2, 1], [lengths[0], lengths[1], lengths[2] - 2]))
    torch.npu.synchronize()
    return {"status": "pass", "cases": report, "scope": "attention only; no GDN state proof"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(args.device, args.long_context)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
