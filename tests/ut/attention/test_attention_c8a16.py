from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport  # type: ignore
from vllm.v1.kv_cache_interface import FullAttentionSpec

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_c8a16 import (
    AscendC8A16AttentionBackendImpl,
    AscendC8A16MetadataBuilder,
    _c8a16_query_ranges,
    _kv_paged_dequant_reference,
    c8a16_cache_spec,
    c8a16_cache_views_for_spec,
    scatter_c8a16_scale_cache,
    should_use_c8a16_dynamic,
)
from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder, AscendAttentionState

NUM_KV_HEADS = 2
HEAD_SIZE = 64
BLOCK_SIZE = 4
NUM_BLOCKS = 3


def _quant_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Emulate npu_dynamic_quant on CPU: per-row INT8 + fp32 amax/127 scale."""
    scale = rows.abs().amax(dim=-1) / 127.0
    scale = torch.where(scale == 0, torch.full_like(scale, 1e-8), scale)
    int8 = (rows / scale.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    return int8, scale


class TestC8A16Gating(TestBase):
    def _config(self, cache_dtype="int8", enable_c8_quant=False):
        quant_config = SimpleNamespace(enable_c8_quant=enable_c8_quant) if enable_c8_quant else None
        return SimpleNamespace(quant_config=quant_config, cache_config=SimpleNamespace(cache_dtype=cache_dtype))

    def test_plain_int8_routes_to_dynamic(self):
        self.assertTrue(should_use_c8a16_dynamic(self._config(), "int8"))
        self.assertTrue(should_use_c8a16_dynamic(self._config(), torch.int8))

    def test_recipe_c8_and_mxfp_and_float_do_not_route(self):
        self.assertFalse(should_use_c8a16_dynamic(self._config(enable_c8_quant=True), "int8"))
        self.assertFalse(should_use_c8a16_dynamic(self._config(cache_dtype="mxfp8"), torch.int8))
        self.assertFalse(should_use_c8a16_dynamic(self._config(), torch.bfloat16))
        self.assertFalse(should_use_c8a16_dynamic(self._config(), "auto"))

    def test_missing_quant_config_is_tolerated(self):
        config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="int8"))
        self.assertTrue(should_use_c8a16_dynamic(config, "int8"))


class TestC8A16ReferenceDequant(TestBase):
    """ND page writes and physical-page scale roundtrip."""

    def setUp(self):
        torch.manual_seed(7)
        self.seq_lens = [2, 5]
        self.block_table = torch.tensor([[0, -99], [2, 1]], dtype=torch.int64)
        # Request 1 crosses from physical page 2 back to page 1.
        self.slot_mapping = torch.tensor([0, 1, 8, 9, 10, 11, 4], dtype=torch.int64)
        self.num_tokens = 7
        self.rows = torch.randn(self.num_tokens, NUM_KV_HEADS, HEAD_SIZE)
        self.k_int8, self.k_scale = _quant_rows(self.rows)

    def _write_cache(self, int8_rows: torch.Tensor) -> torch.Tensor:
        cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.int8)
        for token, slot in enumerate(self.slot_mapping.tolist()):
            block, offset = divmod(slot, BLOCK_SIZE)
            cache[block, offset] = int8_rows[token]
        return cache

    def test_roundtrip_matches_direct_dequant(self):
        key_cache = self._write_cache(self.k_int8)
        value_cache = self._write_cache(self.k_int8.flip(0))
        k_scales = torch.zeros(NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE)
        v_scales = torch.zeros_like(k_scales)
        scatter_c8a16_scale_cache(k_scales, self.slot_mapping, self.k_scale)
        scatter_c8a16_scale_cache(v_scales, self.slot_mapping, self.k_scale.flip(0))

        dense_k, dense_v = _kv_paged_dequant_reference(
            key_cache, value_cache, k_scales, v_scales, self.block_table, self.seq_lens, torch.bfloat16
        )

        expected_k = (self.k_int8.float() * self.k_scale.unsqueeze(-1)).bfloat16()
        self.assertEqual(dense_k.shape, (self.num_tokens, NUM_KV_HEADS, HEAD_SIZE))
        torch.testing.assert_close(dense_k, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(dense_v, expected_k.flip(0), rtol=0, atol=0)

    def test_empty_block_table_returns_empty(self):
        key_cache = self._write_cache(self.k_int8)
        scale_cache = torch.zeros(NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE)
        empty_table = torch.zeros(0, 2, dtype=torch.int64)
        dense_k, dense_v = _kv_paged_dequant_reference(
            key_cache, key_cache, scale_cache, scale_cache, empty_table, [], torch.float32
        )
        self.assertEqual(dense_k.shape[0], 0)
        self.assertEqual(dense_v.shape[0], 0)


class TestC8A16PagedForward(TestBase):
    def _impl(self):
        impl = object.__new__(AscendC8A16AttentionBackendImpl)
        impl.use_bnsd_kv_cache = False
        impl.num_heads = 4
        impl.num_kv_heads = NUM_KV_HEADS
        impl.head_size = HEAD_SIZE
        impl.scale = 0.125
        impl.key_cache = self.key_cache
        impl.value_cache = self.value_cache
        impl.k_scale_cache = self.scale_cache
        impl.v_scale_cache = self.scale_cache
        return impl

    def setUp(self):
        torch.manual_seed(11)
        self.seq_lens = [2]
        self.block_table = torch.tensor([[0, 1]], dtype=torch.int64)
        self.slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
        self.key_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.int8)
        self.value_cache = self.key_cache.clone()
        self.rows = torch.randn(2, NUM_KV_HEADS, HEAD_SIZE)
        int8_rows, self.scale = _quant_rows(self.rows)
        for token, slot in enumerate(self.slot_mapping.tolist()):
            self.key_cache[slot // BLOCK_SIZE, slot % BLOCK_SIZE] = int8_rows[token]
        self.scale_cache = torch.zeros(NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE)
        scatter_c8a16_scale_cache(self.scale_cache, self.slot_mapping, self.scale)

    def test_cached_prefill_runs_dense_bf16_fia_without_antiquant(self):
        impl = self._impl()
        query = torch.randn(2, 4, HEAD_SIZE)
        output = torch.zeros(2, 4 * HEAD_SIZE)
        metadata = SimpleNamespace(
            actual_seq_lengths_q=[2],
            num_actual_tokens=2,
            block_tables=self.block_table,
            seq_lens_list=self.seq_lens,
            attn_mask=None,
        )
        fia_out = torch.zeros(2, 4 * HEAD_SIZE)
        mock = MagicMock(return_value=(fia_out, None))
        with patch("vllm_ascend.attention.attention_c8a16.torch_npu") as mock_torch_npu:
            mock_torch_npu.npu_fused_infer_attention_score = mock
            result = impl._forward_c8a16_paged_attention(query, metadata, output)

        self.assertIs(result, output)
        self.assertTrue(torch.equal(output, fia_out.view(2, 4 * HEAD_SIZE)))
        kwargs = mock.call_args.kwargs
        self.assertEqual(kwargs["input_layout"], "TND")
        self.assertIsNone(kwargs["block_table"])
        self.assertEqual(kwargs["sparse_mode"], 3)
        self.assertEqual(kwargs["actual_seq_lengths_kv"], [2])
        self.assertNotIn("key_antiquant_scale", kwargs)
        self.assertNotIn("value_antiquant_scale", kwargs)
        expected_k = _quant_rows(self.rows)[0].float() * self.scale.unsqueeze(-1)
        self.assertTrue(torch.allclose(kwargs["key"], expected_k))

    def test_decode_calls_fia_mode_five_with_nd_pages_and_fp32_scales(self):
        impl = self._impl()
        query = torch.randn(2, 4, HEAD_SIZE).bfloat16()
        output = torch.zeros_like(query)
        metadata = SimpleNamespace(
            num_actual_tokens=1,
            block_tables=torch.cat((self.block_table, self.block_table)),
            seq_lens_list=[2, 1],
            attn_mask=torch.ones(1, 1, dtype=torch.int8),
        )
        impl.v_scale_cache = self.scale_cache.clone()
        with patch(
            "vllm_ascend.attention.attention_c8a16.torch_npu",
        ) as mock_npu:
            fia = mock_npu.npu_fused_infer_attention_score
            fia.return_value = (query[:1].unsqueeze(1), None)
            impl._forward_c8a16_decode(query, metadata, output)

        args, kwargs = fia.call_args
        self.assertEqual(args[0].shape, (1, 1, 4, HEAD_SIZE))
        self.assertEqual(args[0].dtype, torch.bfloat16)
        self.assertEqual(args[1].shape, (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS * HEAD_SIZE))
        self.assertEqual(args[1].dtype, torch.int8)
        self.assertEqual(kwargs["input_layout"], "BSND")
        self.assertIsNone(kwargs["atten_mask"])
        self.assertEqual(kwargs["sparse_mode"], 0)
        self.assertEqual(kwargs["key_antiquant_mode"], 5)
        self.assertEqual(kwargs["value_antiquant_mode"], 5)
        self.assertIs(kwargs["key_antiquant_scale"], self.scale_cache)
        self.assertIs(kwargs["value_antiquant_scale"], impl.v_scale_cache)
        self.assertEqual(kwargs["actual_seq_lengths"], [1])
        self.assertEqual(kwargs["actual_seq_lengths_kv"], [2])
        torch.testing.assert_close(kwargs["block_table"], self.block_table)
        torch.testing.assert_close(output[:1], query[:1])
        self.assertEqual(output[1].count_nonzero(), 0)


class TestC8A16GraphSupport(TestBase):
    def test_piecewise_supported_full_rejected(self):
        piecewise = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE))
        self.assertEqual(
            AscendC8A16MetadataBuilder.get_cudagraph_support(piecewise, None),
            AttentionCGSupport.UNIFORM_BATCH,
        )
        full = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY))
        self.assertEqual(
            AscendC8A16MetadataBuilder.get_cudagraph_support(full, None),
            AttentionCGSupport.NEVER,
        )


class TestC8A16Verification(TestC8A16PagedForward):
    """Exercise real packed request boundaries, not a fixed draft width."""

    def _metadata(self):
        return SimpleNamespace(
            num_actual_tokens=7,
            actual_seq_lengths_q=[1, 3, 7, 9],
            seq_lens_list=[127, 129, 133, 1],
            block_tables=torch.tensor([[0, 1], [1, 2], [2, 0], [0, 0]]),
        )

    def test_ragged_queries_batch_by_position_and_preserve_padding(self):
        metadata = self._metadata()
        query = torch.randn(9, 4, HEAD_SIZE).bfloat16()
        output = torch.full((9, 4 * HEAD_SIZE), -7, dtype=torch.bfloat16)
        with patch("vllm_ascend.attention.attention_c8a16.torch_npu") as npu:
            fia = npu.npu_fused_infer_attention_score
            fia.side_effect = lambda q, *args, **kwargs: (q.clone(), None)
            self._impl()._forward_c8a16_verification(query, metadata, output)
        self.assertEqual(fia.call_count, 4)
        expected = [([0, 1, 3], [127, 128, 130]), ([2, 4], [129, 131]), ([5], [132]), ([6], [133])]
        for call, (tokens, lengths) in zip(fia.call_args_list, expected):
            args, kwargs = call
            torch.testing.assert_close(args[0].squeeze(1), query[tokens])
            self.assertEqual(kwargs["actual_seq_lengths_kv"], lengths)
            self.assertEqual(kwargs["actual_seq_lengths"], [1] * len(tokens))
            self.assertEqual(kwargs["key_antiquant_mode"], 5)
            self.assertEqual(kwargs["value_antiquant_mode"], 5)
            self.assertEqual(kwargs["sparse_mode"], 0)
            self.assertIsNone(kwargs["atten_mask"])
        torch.testing.assert_close(output[:7], query[:7].flatten(1))
        self.assertTrue(torch.all(output[7:] == -7))
        torch.testing.assert_close(fia.call_args_list[0].kwargs["block_table"], metadata.block_tables[:3])
        torch.testing.assert_close(fia.call_args_list[1].kwargs["block_table"], metadata.block_tables[1:3])

    def test_malformed_boundaries_and_short_kv_fail_before_fia(self):
        query = torch.randn(9, 4, HEAD_SIZE).bfloat16()
        with patch("vllm_ascend.attention.attention_c8a16.torch_npu") as npu:
            for ends, lengths in (([1, 3], [3, 5]), ([3, 1, 7], [3, 5, 9]), ([1, 3, 7], [3, 5, 2])):
                metadata = self._metadata()
                metadata.actual_seq_lengths_q = ends
                metadata.seq_lens_list = lengths
                with self.assertRaises(ValueError):
                    self._impl()._forward_c8a16_verification(query, metadata, torch.zeros_like(query))
            npu.npu_fused_infer_attention_score.assert_not_called()

    def test_forward_routes_mixed_batch_without_using_short_prefill_as_decode(self):
        impl = self._impl()
        impl.kv_sharing_target_layer_name = None
        from vllm.v1.attention.backend import AttentionType

        impl.attn_type = AttentionType.DECODER
        impl.sliding_window = impl.alibi_slopes = impl.sinks = None
        impl.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1)
        )
        metadata = self._metadata()
        metadata.attn_state = AscendAttentionState.ChunkedPrefill
        metadata.attn_mask = torch.ones(1, 1)
        metadata.c8a16_decode_requests = (0, 2)
        query = torch.zeros(9, 4, HEAD_SIZE, dtype=torch.bfloat16)
        with (
            patch("vllm_ascend.attention.attention_c8a16._EXTRA_CTX", capturing=False),
            patch.object(impl, "_bind_c8a16_caches"),
            patch.object(impl, "_forward_c8a16_verification") as verification,
            patch.object(impl, "_forward_c8a16_paged_attention") as prefill,
        ):
            impl.forward(SimpleNamespace(layer_name="layer.0"), query, None, None, None, metadata, query.clone())
        self.assertEqual(verification.call_args.args[3], (0, 2))
        self.assertEqual(prefill.call_args.args[3], (1,))

    def test_partial_prefill_output_scatter_keeps_other_requests(self):
        metadata = self._metadata()
        # Use small valid page lengths for this CPU cache fixture.
        metadata.seq_lens_list = [1, 2, 4, 1]
        query = torch.randn(9, 4, HEAD_SIZE).bfloat16()
        output = torch.full_like(query, -7)
        metadata.attn_mask = None
        with patch("vllm_ascend.attention.attention_c8a16.torch_npu") as npu:
            npu.npu_fused_infer_attention_score.side_effect = lambda **kw: (kw["query"], None)
            self._impl()._forward_c8a16_paged_attention(query, metadata, output, (1,))
        torch.testing.assert_close(output[1:3], query[1:3])
        self.assertTrue(torch.all(output[[0, 3, 4, 5, 6, 7, 8]] == -7))

    def test_pad_slot_does_not_overwrite_last_scale(self):
        scales = torch.full((3, 1, 4), 11.0)
        scatter_c8a16_scale_cache(scales, torch.tensor([0, -1, 5]), torch.tensor([[2.0], [99.0], [3.0]]))
        self.assertEqual(scales[0, 0, 0], 2)
        self.assertEqual(scales[1, 0, 1], 3)
        self.assertEqual(scales[2, 0, 3], 11)

    def test_split_matches_independent_full_causal_attention(self):
        impl = self._impl()
        impl.key_cache = torch.randint(-127, 128, self.key_cache.shape, dtype=torch.int8)
        impl.value_cache = torch.randint(-127, 128, self.key_cache.shape, dtype=torch.int8)
        impl.k_scale_cache = torch.rand_like(self.scale_cache) * 0.01
        impl.v_scale_cache = torch.rand_like(self.scale_cache) * 0.01
        metadata = SimpleNamespace(
            num_actual_tokens=7,
            actual_seq_lengths_q=[1, 3, 7],
            seq_lens_list=[2, 4, 8],
            block_tables=torch.tensor([[2, 0], [1, 0], [0, 2]]),
        )
        query = torch.randn(7, 4, HEAD_SIZE).bfloat16()

        def dense_rows(cache, scale, table, length):
            cache = cache.reshape(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
            positions = torch.arange(length)
            pages, offsets = table[positions // BLOCK_SIZE], positions % BLOCK_SIZE
            return (cache[pages, offsets].float() * scale[pages, :, offsets].unsqueeze(-1)).repeat_interleave(2, 1)

        def fake_fia(q, key, value, **kwargs):
            rows = []
            for request, length in enumerate(kwargs["actual_seq_lengths_kv"]):
                table = kwargs["block_table"][request]
                k = dense_rows(key, impl.k_scale_cache, table, length)
                v = dense_rows(value, impl.v_scale_cache, table, length)
                scores = torch.einsum("hd,lhd->hl", q[request, 0].float(), k) * impl.scale
                rows.append(torch.einsum("hl,lhd->hd", scores.softmax(-1), v))
            return torch.stack(rows).unsqueeze(1).bfloat16(), None

        output = torch.zeros_like(query)
        with patch("vllm_ascend.attention.attention_c8a16.torch_npu") as npu:
            npu.npu_fused_infer_attention_score.side_effect = fake_fia
            impl._forward_c8a16_verification(query, metadata, output)
        expected = []
        for request, (start, end) in enumerate(zip([0, 1, 3], [1, 3, 7])):
            length = metadata.seq_lens_list[request]
            k = dense_rows(impl.key_cache, impl.k_scale_cache, metadata.block_tables[request], length)
            v = dense_rows(impl.value_cache, impl.v_scale_cache, metadata.block_tables[request], length)
            scores = torch.einsum("qhd,lhd->hql", query[start:end].float(), k) * impl.scale
            causal = torch.arange(length)[None, :] <= length - (end - start) + torch.arange(end - start)[:, None]
            scores.masked_fill_(~causal.unsqueeze(0), -torch.inf)
            expected.append(torch.einsum("hql,lhd->qhd", scores.softmax(-1), v))
        torch.testing.assert_close(output, torch.cat(expected).bfloat16(), rtol=0.02, atol=0.02)


class TestC8A16SpecMetadata(TestBase):
    def test_phase_flags_distinguish_short_prefill_from_verification(self):
        builder = object.__new__(AscendC8A16MetadataBuilder)
        common = SimpleNamespace(is_prefilling=torch.tensor([False, True, False]))
        result = builder._build_backend_metadata(common, query_lens=torch.tensor([1, 2, 4]), num_decodes=3)
        self.assertEqual(result["c8a16_decode_requests"], (0, 2))

    def test_builder_uses_exact_device_lengths_instead_of_upper_bounds(self):
        builder = object.__new__(AscendC8A16MetadataBuilder)
        builder.speculative_config = SimpleNamespace(num_speculative_tokens=3)
        common = SimpleNamespace(num_reqs=3, seq_lens=torch.tensor([127, 129, 133]))
        built = SimpleNamespace(seq_lens_list=[131, 133, 137, 1])
        with patch.object(AscendAttentionMetadataBuilder, "build", return_value=built):
            result = builder.build(0, common)
        self.assertEqual(result.seq_lens_list, [127, 129, 133, 1])
        self.assertEqual(result.seq_lens_cpu.tolist(), [127, 129, 133])

    def test_zero_length_request_keeps_following_request_indices(self):
        metadata = SimpleNamespace(
            num_actual_tokens=3,
            actual_seq_lengths_q=[0, 1, 3, 8],
            seq_lens_list=[0, 5, 6, 1],
            block_tables=torch.zeros(4, 1),
        )
        self.assertEqual(_c8a16_query_ranges(metadata), ([0, 0, 1], [0, 1, 3]))


class TestC8A16RunnerOwnedCaches(TestBase):
    """The runner budgets the scale planes and hands over four components."""

    def _spec(self):
        return FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            dtype=torch.int8,
        )

    def test_spec_budgets_both_fp32_scale_planes(self):
        spec = self._spec()
        dense_page = spec.page_size_bytes
        scaled = c8a16_cache_spec(spec)
        expected_extra = BLOCK_SIZE * NUM_KV_HEADS * 2 * 4
        self.assertEqual(scaled.page_size_bytes, dense_page + expected_extra)
        self.assertEqual(scaled.dtype, torch.int8)
        self.assertEqual(scaled.head_size, HEAD_SIZE)
        self.assertEqual(scaled.real_page_size_bytes, scaled.page_size_bytes)

    def test_views_split_one_allocation_into_four_components(self):
        spec = c8a16_cache_spec(self._spec())
        num_blocks = 3
        raw = torch.zeros(num_blocks * spec.page_size_bytes, dtype=torch.int8, device="cpu")
        k, v, ks, vs = c8a16_cache_views_for_spec(
            raw,
            num_blocks,
            spec.page_size_bytes,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_SIZE,
            HEAD_SIZE,
        )
        self.assertEqual(k.shape, (num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE))
        self.assertEqual(v.shape, k.shape)
        self.assertEqual(k.dtype, torch.int8)
        self.assertEqual(ks.shape, (num_blocks, NUM_KV_HEADS, BLOCK_SIZE))
        self.assertEqual(vs.shape, ks.shape)
        self.assertEqual(ks.dtype, torch.float32)
        # Scale planes are page-strided like K/V so one page mapping covers
        # all four components; mode-5 FIA accepts that layout.
        self.assertEqual(ks.stride(0) * ks.element_size(), spec.page_size_bytes)
        self.assertEqual(vs.stride(0) * vs.element_size(), spec.page_size_bytes)

    def test_views_share_one_page_mapping(self):
        spec = c8a16_cache_spec(self._spec())
        num_blocks = 2
        raw = torch.zeros(num_blocks * spec.page_size_bytes, dtype=torch.int8, device="cpu")
        k, v, ks, vs = c8a16_cache_views_for_spec(
            raw,
            num_blocks,
            spec.page_size_bytes,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_SIZE,
            HEAD_SIZE,
        )
        # Writing page 1 of every component must land inside that page's own
        # region, so a PD page-1 -> page-1 mapping carries all four together.
        k[1].fill_(7)
        v[1].fill_(9)
        ks[1].fill_(1.5)
        vs[1].fill_(2.5)
        self.assertEqual(int(k[1].float().abs().sum().item()), 7 * k[1].numel())
        self.assertEqual(int(v[1].float().abs().sum().item()), 9 * v[1].numel())
        self.assertTrue(torch.all(ks[1] == 1.5))
        self.assertTrue(torch.all(vs[1] == 2.5))
        self.assertEqual(int(k[0].abs().sum().item()), 0)
        self.assertEqual(int(ks[0].abs().sum().item()), 0)

    def test_bind_requires_four_components(self):
        impl = object.__new__(AscendC8A16AttentionBackendImpl)
        impl.use_bnsd_kv_cache = False
        impl.kv_cache_layout = "NHD"
        impl.key_cache = None
        impl.value_cache = None
        impl.k_scale_cache = None
        impl.v_scale_cache = None
        # The mode-5 kernel is fixed at a 128-token block.
        key = torch.zeros(NUM_BLOCKS, 128, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.int8)
        scale = torch.zeros(NUM_BLOCKS, NUM_KV_HEADS, 128)
        with self.assertRaises(NotImplementedError):
            impl._bind_c8a16_caches((key, key.clone()))
        impl._bind_c8a16_caches((key, key.clone(), scale, scale.clone()))
        self.assertIs(impl.key_cache, key)
        self.assertIs(impl.k_scale_cache, scale)


class TestC8A16PaddedPage(TestBase):
    """Hybrid models pad the attention page up to the Mamba page."""

    def test_padded_page_keeps_all_four_components_in_place(self):
        base = FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            dtype=torch.int8,
        )
        scaled = c8a16_cache_spec(base)
        padded_bytes = scaled.page_size_bytes + 4096
        padded = replace(scaled, page_size_padded=padded_bytes)
        self.assertEqual(padded.page_size_bytes, padded_bytes)
        num_blocks = 2
        raw = torch.zeros(num_blocks * padded_bytes, dtype=torch.int8, device="cpu")
        k, v, ks, vs = c8a16_cache_views_for_spec(
            raw,
            num_blocks,
            padded_bytes,
            BLOCK_SIZE,
            NUM_KV_HEADS,
            HEAD_SIZE,
            HEAD_SIZE,
        )
        self.assertEqual(k.shape, (num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE))
        self.assertEqual(ks.shape, (num_blocks, NUM_KV_HEADS, BLOCK_SIZE))
        # All four components advance by the padded page pitch, so block 1 of
        # every component lives in page 1 -- the property a PD page mapping
        # relies on.
        for cache in (k, v, ks, vs):
            self.assertEqual(cache.stride(0) * cache.element_size(), padded_bytes)
        # Every component of block 1 lies inside page 1 and nowhere else.
        page_start = raw.data_ptr() + padded_bytes
        page_end = page_start + padded_bytes
        for cache in (k, v, ks, vs):
            block = cache[1]
            span = block.element_size() + sum(
                (size - 1) * stride * block.element_size() for size, stride in zip(block.shape, block.stride())
            )
            self.assertGreaterEqual(block.data_ptr(), page_start)
            self.assertLessEqual(block.data_ptr() + span, page_end)


class TestC8A16SharedBackingOffset(TestBase):
    """Hybrid layers are slices of one backing; views must keep that offset."""

    def test_views_respect_layer_storage_offset(self):
        spec = c8a16_cache_spec(
            FullAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=NUM_KV_HEADS,
                head_size=HEAD_SIZE,
                dtype=torch.int8,
            )
        )
        num_blocks = 2
        layer_bytes = num_blocks * spec.page_size_bytes
        backing = torch.zeros(layer_bytes * 3, dtype=torch.int8, device="cpu")
        # Layer 1 lives in the middle of the shared backing.
        layer = backing[layer_bytes : 2 * layer_bytes]
        self.assertGreater(layer.storage_offset(), 0)
        k, v, ks, vs = c8a16_cache_views_for_spec(
            layer, num_blocks, spec.page_size_bytes, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE
        )
        # K starts at the layer base; V, K scale and V scale follow inside
        # the same page. All stay inside this layer.
        base = layer.storage_offset()
        kv_page = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE
        scale_page = BLOCK_SIZE * NUM_KV_HEADS * 4
        self.assertEqual(k.storage_offset(), base)
        self.assertEqual(v.storage_offset(), base + kv_page)
        self.assertEqual(ks.storage_offset() * 4, base + 2 * kv_page)
        self.assertEqual(vs.storage_offset() * 4, base + 2 * kv_page + scale_page)
        # Writes must land in this layer's slice, never in layer 0 or 2.
        k.fill_(5)
        self.assertEqual(int(backing[:layer_bytes].abs().sum().item()), 0)
        self.assertEqual(int(backing[2 * layer_bytes :].abs().sum().item()), 0)
        self.assertGreater(int(layer.abs().sum().item()), 0)
