#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Dynamic C8A16 KV cache with fused FIA mode-5 decode.

New prefill computes attention with BF16 Q/K/V and caches INT8 K/V.
Single-token decode reads the ND paged INT8 cache directly through the
run-package extension to FIA. Other paged states use compact BF16 TND K/V.
"""

import math
from bisect import bisect_right
from dataclasses import replace
from itertools import accumulate
from typing import TYPE_CHECKING

import torch
import torch_npu
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import (  # type: ignore
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
)

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.utils import notify_kv_cache_written
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.utils import C8_MXFP_KV_CACHE_DTYPE

if TYPE_CHECKING:
    from vllm.config import VllmConfig

C8A16_BLOCK_SIZE = 128
C8A16_ANTIQUANT_MODE = 5
C8A16_DECODE_SPARSE_MODE = 0
C8A16_PREFILL_SPARSE_MODE = 3
C8A16_SCALE_DTYPE = torch.float32


def should_use_c8a16_dynamic(vllm_config: "VllmConfig", kv_cache_dtype: str | torch.dtype) -> bool:
    """True for plain ``--kv-cache-dtype int8`` without recipe C8 or MXFP.

    Dynamic per-token quantization needs no checkpoint scales, so INT8 KV
    cache without a quantization recipe routes to the C8A16 backend. Callers
    pass either the ``cache_config.cache_dtype`` string or the resolved
    ``torch.dtype``.
    """
    if kv_cache_dtype not in ("int8", torch.int8):
        return False
    quant_config = getattr(vllm_config, "quant_config", None)
    if quant_config is not None and getattr(quant_config, "enable_c8_quant", False):
        return False
    cache_config = getattr(vllm_config, "cache_config", None)
    return not (cache_config is not None and cache_config.cache_dtype == C8_MXFP_KV_CACHE_DTYPE)


def c8a16_cache_spec(spec: "FullAttentionSpec") -> "FullAttentionSpec":
    """Budget both dynamic scale planes inside one physical page.

    ``FullAttentionSpec`` sizes a page from ``(head_size + head_size_v)``
    bytes per KV head and token. The dynamic C8A16 cache also stores one fp32
    K scale and one fp32 V scale per KV head and token, so without this the
    planner would reserve only the INT8 payload and the runner would have to
    allocate the scales after initialization (risking OOM and leaving them
    outside the range Mooncake registers).

    ``state_content_bytes`` is the upstream hook that lets a page carry more
    than dense K/V content; ``dtype`` stays INT8 so the K/V views keep their
    real width. ``page_size_padded`` (when the hybrid planner already padded
    this layer) must remain at least the enlarged page.
    """
    if spec.tokens_per_state != 1 or spec.num_heads != spec.num_kv_heads:
        raise ValueError("C8A16 dynamic cache requires one stored state per token and one slot per KV head.")
    kv_bytes = spec.head_size + spec.head_size_v
    scale_bytes = 2 * get_dtype_size(C8A16_SCALE_DTYPE)
    result = replace(spec, state_content_bytes=kv_bytes + scale_bytes)
    if result.page_size_padded is not None and result.page_size_padded < result.unpadded_page_size_bytes:
        raise ValueError("C8A16 padded page is smaller than its INT8 K/V and fp32 scale payload.")
    return result


def c8a16_cache_views_for_spec(
    raw_tensor: torch.Tensor,
    kernel_num_blocks: int,
    kernel_pitch_bytes: int,
    kernel_block_size: int,
    num_kv_heads: int,
    k_dim: int,
    v_dim: int,
):
    """Split the runner's allocation into K, V and the two fp32 scale planes.

    One physical page holds ``[K][V][K_scale][V_scale]`` and is
    ``kernel_pitch_bytes`` wide, which may exceed the payload when the hybrid
    planner padded the page up to the Mamba page. All four views advance by
    that same pitch, so kernel block ``j`` of every component lives in page
    ``j``; that is what makes one PD page mapping (P page 10 -> D page 25)
    carry all four components.

    The fp32 scale views are therefore page-strided rather than contiguous.
    Mode-5 FIA accepts a strided antiquant scale as long as its element layout
    is ``[blocks, kv_heads, block_size]``, which this keeps.
    """
    if raw_tensor.ndim != 1 or raw_tensor.dtype != torch.int8 or raw_tensor.stride(0) != 1:
        raise ValueError("C8A16 raw cache must be a contiguous 1-D int8 view.")
    if kernel_num_blocks <= 0 or kernel_block_size <= 0:
        raise ValueError("C8A16 cache requires at least one kernel block.")

    dtype_size = get_dtype_size(C8A16_SCALE_DTYPE)
    kv_page_bytes = kernel_block_size * num_kv_heads * (k_dim + v_dim)
    scale_page_bytes = kernel_block_size * num_kv_heads * dtype_size
    payload_bytes = kv_page_bytes + 2 * scale_page_bytes
    if kernel_pitch_bytes < payload_bytes:
        raise ValueError(
            f"C8A16 physical page is too small for its K/V and scale payload: "
            f"pitch={kernel_pitch_bytes}, need at least {payload_bytes}."
        )
    if kernel_pitch_bytes % dtype_size:
        raise ValueError("C8A16 physical page must be aligned to the fp32 scale dtype.")
    required = kernel_num_blocks * kernel_pitch_bytes
    if raw_tensor.numel() < required:
        raise ValueError(
            f"C8A16 allocation is smaller than its paged payload: {raw_tensor.numel()} < {required}."
        )

    # Keep the layer's own offset inside a shared hybrid backing allocation.
    base_offset = raw_tensor.storage_offset()
    kv_shape = (2, kernel_num_blocks, kernel_block_size, num_kv_heads, k_dim)
    kv_dense_strides = [math.prod(kv_shape[dim + 1 :]) for dim in range(len(kv_shape))]
    combined = torch.as_strided(
        raw_tensor,
        size=kv_shape,
        stride=(kernel_block_size * num_kv_heads * k_dim, kernel_pitch_bytes, *kv_dense_strides[2:]),
        storage_offset=base_offset,
    )
    scale_shape = (kernel_num_blocks, num_kv_heads, kernel_block_size)
    scale_strides = (kernel_pitch_bytes // dtype_size, kernel_block_size, 1)
    scale_base = base_offset // dtype_size + kv_page_bytes // dtype_size
    scale_tensor = raw_tensor.view(C8A16_SCALE_DTYPE)
    k_scale = torch.as_strided(
        scale_tensor, size=scale_shape, stride=scale_strides, storage_offset=scale_base
    )
    v_scale = torch.as_strided(
        scale_tensor,
        size=scale_shape,
        stride=scale_strides,
        storage_offset=scale_base + scale_page_bytes // dtype_size,
    )
    return combined[0], combined[1], k_scale, v_scale


def dyn_quant_kv(kv: torch.Tensor, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic INT8 quantization of K/V.

    ``npu_dynamic_quant`` reduces the last dimension per row: for K/V of
    shape (T, H, D) it yields INT8 (T, H, D) plus an fp32 per-token-per-head
    scale, normalized here to (T, H).
    """
    out, scale = torch_npu.npu_dynamic_quant(kv[:num_tokens].contiguous(), dst_type=torch.int8)
    return out, scale.reshape(scale.shape[0], -1).clamp_min(torch.finfo(torch.float32).tiny)


def scatter_c8a16_scale_cache(scale_cache: torch.Tensor, slot_mapping: torch.Tensor, scale: torch.Tensor) -> None:
    """Write FP32 scales using the unpadded slots of the INT8 write.

    ``scale_cache`` is a page-strided view over the packed physical page, so it
    is not contiguous and cannot be flattened. Index the three logical axes
    directly instead of computing flat offsets.
    """
    _, num_heads, block_size = scale_cache.shape
    slots = slot_mapping.long()
    pages = slots // block_size
    offsets = slots % block_size
    heads = torch.arange(num_heads, device=scale_cache.device)
    scale_cache[pages[:, None], heads[None, :], offsets[:, None]] = scale


def _kv_paged_dequant_reference(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int] | torch.Tensor,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather ND pages in request order and dequantize in FP32."""
    if block_table.numel() == 0:
        empty = torch.empty(0, key_cache.shape[2], key_cache.shape[3], dtype=out_dtype, device=key_cache.device)
        return empty, empty.clone()

    block_size = key_cache.shape[1]
    seq_lens_t = torch.as_tensor(seq_lens, dtype=torch.long, device=key_cache.device)
    positions = torch.arange(block_table.shape[1] * block_size, device=key_cache.device)
    requests, tokens = torch.where(positions.unsqueeze(0) < seq_lens_t.unsqueeze(1))
    blocks = block_table[requests, tokens // block_size].long()
    offsets = tokens % block_size

    def _gather_and_dequant(cache: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        rows = cache[blocks, offsets]
        row_scales = scales.transpose(1, 2)[blocks, offsets]
        return (rows.float() * row_scales.unsqueeze(-1)).to(out_dtype)

    return _gather_and_dequant(key_cache, k_scale_cache), _gather_and_dequant(value_cache, v_scale_cache)


class AscendC8A16MetadataBuilder(AscendAttentionMetadataBuilder):
    """Metadata builder for the dynamic C8A16 backend.

    The read path allocates its dense gather output and consumes
    data-dependent per-request lengths, so it must run outside captured
    regions: PIECEWISE is advertised (attention runs as a plain call between
    compiled pieces) and FULL graph modes are rejected outright instead of
    capturing an incorrect graph.
    """

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport:
        mode = vllm_config.compilation_config.cudagraph_mode
        if mode.has_piecewise_cudagraphs() and not mode.has_full_cudagraphs():
            return AttentionCGSupport.UNIFORM_BATCH
        return AttentionCGSupport.NEVER


class AscendC8A16AttentionBackendImpl(AscendAttentionBackendImpl):
    """Attention implementation for dynamic C8A16 (INT8 cache, BF16 math).

    See the module docstring for the scheme and scope limits. Installed via
    ``self.__class__`` assignment in ``AscendAttentionBackendImpl.__init__``
    for plain ``--kv-cache-dtype int8``, which does not re-run constructors.
    """

    # The class swap above bypasses ``__init__``, so the scale caches need
    # defaults here rather than assignments in a constructor.
    k_scale_cache: torch.Tensor | None = None
    v_scale_cache: torch.Tensor | None = None

    def _bind_c8a16_caches(self, kv_cache) -> None:
        """Bind the runner-owned INT8 K/V and fp32 scale caches.

        The runner allocates all four components as views over one physical
        page (K, V, K scale, V scale) and hands them here as a 4-tuple. Binding
        the runner's tensors instead of allocating scales in the first forward
        keeps the scale addresses visible to the KV connector at registration
        time, which PD disaggregation requires.
        """
        if kv_cache is None:
            return
        if self.key_cache is None and isinstance(kv_cache, (list, tuple)):
            if len(kv_cache) != 4:
                raise NotImplementedError(
                    "C8A16 dynamic attention requires the runner to provide "
                    f"(key_cache, value_cache, key_scale, value_scale); got {len(kv_cache)} components."
                )
            self.key_cache, self.value_cache, self.k_scale_cache, self.v_scale_cache = kv_cache
        if self.key_cache is None:
            return
        if self.use_bnsd_kv_cache:
            raise NotImplementedError(
                "C8A16 dynamic attention requires the NHD KV cache layout "
                f"(VLLM_KV_CACHE_LAYOUT), got {self.kv_cache_layout}."
            )
        if self.key_cache.shape[1] != C8A16_BLOCK_SIZE:
            raise NotImplementedError(f"C8A16 mode 5 requires block size {C8A16_BLOCK_SIZE}.")

    def _quant_and_scatter_kv(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata) -> None:
        """Dynamic-quantize K/V and scatter INT8 payload + fp32 scales."""
        num_actual_tokens = attn_metadata.num_actual_tokens
        k_int8, k_scale = dyn_quant_kv(key, num_actual_tokens)
        v_int8, v_scale = dyn_quant_kv(value, num_actual_tokens)

        slots = attn_metadata.slot_mapping[:num_actual_tokens]
        DeviceOperator.reshape_and_cache(
            key=k_int8,
            value=v_int8,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            slot_mapping=slots,
        )
        scatter_c8a16_scale_cache(self.k_scale_cache, slots, k_scale)
        scatter_c8a16_scale_cache(self.v_scale_cache, slots, v_scale)
        notify_kv_cache_written()

    def _forward_c8a16_decode(
        self, query: torch.Tensor, attn_metadata: AscendMetadata, output: torch.Tensor
    ) -> torch.Tensor:
        """Use the installed mode-5 FIA kernel directly on INT8 paged KV."""
        batch_size = attn_metadata.num_actual_tokens
        block_table = attn_metadata.block_tables[:batch_size].contiguous()
        seq_lens = attn_metadata.seq_lens_list[:batch_size]
        key_cache, value_cache, block_size = self._get_kv_cache_view(self.key_cache, self.value_cache)
        # Each query is the last valid token, so the paged lengths provide the
        # causal boundary. Mode 5 on A3 produced incorrect results with the mask.
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(1).contiguous(),
            key_cache,
            value_cache,
            key_antiquant_scale=self.k_scale_cache,
            value_antiquant_scale=self.v_scale_cache,
            key_antiquant_mode=C8A16_ANTIQUANT_MODE,
            value_antiquant_mode=C8A16_ANTIQUANT_MODE,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            scale=self.scale,
            block_table=block_table,
            block_size=block_size,
            input_layout="BSND",
            actual_seq_lengths=[1] * batch_size,
            actual_seq_lengths_kv=seq_lens,
            atten_mask=None,
            sparse_mode=C8A16_DECODE_SPARSE_MODE,
            inner_precise=0,
        )
        output_view = output[:batch_size]
        output_view.copy_(attn_output.reshape_as(output_view))
        return output

    def _forward_c8a16_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-token cached attention: gather+dequant, BF16 TND.

        One dequant and one FIA call per step, mirroring the static C8
        ``_forward_c8_fused_infer_attention`` dense tail (sparse_mode=3 with
        the shared causal mask; block_table dropped after dequantization).
        """
        num_tokens = attn_metadata.num_actual_tokens
        q_ends = attn_metadata.actual_seq_lengths_q
        actual_seq_qlen = q_ends[: bisect_right(q_ends, num_tokens)]
        query = query[:num_tokens]

        seq_lens = attn_metadata.seq_lens_list[: len(actual_seq_qlen)]
        block_table = attn_metadata.block_tables[: len(seq_lens)]
        dense_k, dense_v = _kv_paged_dequant_reference(
            self.key_cache,
            self.value_cache,
            self.k_scale_cache,
            self.v_scale_cache,
            block_table,
            seq_lens,
            query.dtype,
        )
        actual_seq_lengths_kv = list(accumulate(seq_lens))

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=dense_k.contiguous(),
            value=dense_v.contiguous(),
            atten_mask=attn_metadata.attn_mask,
            block_table=None,
            input_layout="TND",
            block_size=self.key_cache.shape[1],
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=C8A16_PREFILL_SPARSE_MODE,
        )
        output_view = output[:num_tokens]
        output_view.copy_(attn_output.reshape_as(output_view))
        return output

    def _forward_c8a16_prefill(
        self,
        query: torch.Tensor,
        float_key: torch.Tensor | None,
        float_value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill states: float KV for all-new prefill, dequant for cache hit.

        ``PrefillNoCache`` attends over the fresh BF16 K/V directly (no
        cache roundtrip, matching the static C8 ``all_new_prefill``
        convention); ``PrefillCacheHit`` gathers + dequantizes the paged
        INT8 prefix. Both run BF16 TND without antiquant parameters.
        """
        num_tokens = attn_metadata.num_actual_tokens
        q_ends = attn_metadata.actual_seq_lengths_q
        actual_seq_qlen = q_ends[: bisect_right(q_ends, num_tokens)]
        query = query[:num_tokens]

        if attn_metadata.attn_state != AscendAttentionState.PrefillNoCache:
            return self._forward_c8a16_paged_attention(query, attn_metadata, output)
        key = float_key[:num_tokens]
        value = float_value[:num_tokens]

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key.contiguous(),
            value=value.contiguous(),
            atten_mask=attn_metadata.attn_mask,
            block_table=None,
            input_layout="TND",
            block_size=self.key_cache.shape[1],
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_qlen,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=C8A16_PREFILL_SPARSE_MODE,
        )
        output_view = output[:num_tokens]
        output_view.copy_(attn_output.reshape_as(output_view))
        return output

    def forward(  # type: ignore[override]
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for AscendC8A16AttentionBackendImpl"
            )
        if attn_metadata is None or attn_metadata.num_actual_tokens == 0:
            return output.fill_(0)
        if self.kv_sharing_target_layer_name is not None:
            raise NotImplementedError("C8A16 dynamic attention requires independent KV and scale caches per layer.")
        if self.attn_type != AttentionType.DECODER or attn_metadata.attn_mask is None:
            raise NotImplementedError("C8A16 dynamic attention requires causal decoder attention.")
        if query.dtype != torch.bfloat16:
            raise NotImplementedError("C8A16 mode 5 requires BF16 query.")
        if self.sliding_window is not None or self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("C8A16 dynamic attention requires full attention without ALiBi or sinks.")
        if _EXTRA_CTX.capturing:
            raise NotImplementedError(
                "C8A16 dynamic attention does not support FULL/ACLGraph capture: the dequant read path "
                "allocates and reads data-dependent lengths inside the captured region. Run with "
                "--cudagraph-mode PIECEWISE (or --enforce-eager)."
            )
        # PD disaggregation is supported: the runner allocates K/V and both
        # fp32 scale planes as one packed page, so the connector registers and
        # transfers all four components under a single page mapping.
        parallel_config = self.vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size > 1 or parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError("C8A16 dynamic attention does not support PCP/DCP yet.")

        self._layer_name = layer.layer_name
        self._bind_c8a16_caches(kv_cache)

        if key is not None and value is not None:
            self._quant_and_scatter_kv(key, value, attn_metadata)

        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            tokens = attn_metadata.num_actual_tokens
            if attn_metadata.actual_seq_lengths_q[:tokens] == list(range(1, tokens + 1)):
                return self._forward_c8a16_decode(query, attn_metadata, output)
        return self._forward_c8a16_prefill(query, key, value, attn_metadata, output)
