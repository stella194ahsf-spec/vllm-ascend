import torch, torch_npu

torch.npu.set_device(0)
nb, bs, h, d = 6, 128, 1, 256
pitch = bs * h * 2 * d + 2 * bs * h * 4  # 66560
raw = torch.zeros(nb * pitch, dtype=torch.int8, device="npu:0")
kd = bs * h * d
k = torch.as_strided(raw, (nb, bs, h, d), (pitch, h * d, d, 1), storage_offset=0)
v = torch.as_strided(raw, (nb, bs, h, d), (pitch, h * d, d, 1), storage_offset=kd)
ks = torch.as_strided(raw.view(torch.float32), (nb, h, bs), (pitch // 4, bs, 1), storage_offset=(2 * kd) // 4)
vs = torch.as_strided(
    raw.view(torch.float32), (nb, h, bs), (pitch // 4, bs, 1), storage_offset=(2 * kd + bs * h * 4) // 4
)
print("k", k.stride(), "ks", ks.stride(), "contig", ks.is_contiguous())
slots = torch.arange(262, dtype=torch.int64, device="npu:0")
ki = torch.zeros(262, h, d, dtype=torch.int8, device="npu:0")
torch_npu.npu_scatter_pa_kv_cache(
    key=ki, value=ki, key_cache=k, value_cache=v, slot_mapping=slots, cache_mode="Norm"
)
ks[0, 0, :5] = 1.5
print("scale write ok", ks[0, 0, :3].tolist(), "raw tail", raw[2 * kd : 2 * kd + 8].tolist())
q = torch.randn(1, 1, 8, d, device="npu:0", dtype=torch.bfloat16)
try:
    out, _ = torch_npu.npu_fused_infer_attention_score(
        q,
        k.view(nb, bs, h * d),
        v.view(nb, bs, h * d),
        key_antiquant_scale=ks,
        value_antiquant_scale=vs,
        key_antiquant_mode=5,
        value_antiquant_mode=5,
        num_heads=8,
        num_key_value_heads=1,
        scale=d**-0.5,
        block_table=torch.tensor([[4, 1]], dtype=torch.int32, device="npu:0"),
        block_size=bs,
        input_layout="BSND",
        actual_seq_lengths=[1],
        actual_seq_lengths_kv=[129],
        atten_mask=None,
        sparse_mode=0,
        inner_precise=0,
    )
    print("FIA strided-scale OK", out.shape)
except Exception as e:
    print("FIA strided-scale FAIL", str(e)[:220].replace("\n", " "))
