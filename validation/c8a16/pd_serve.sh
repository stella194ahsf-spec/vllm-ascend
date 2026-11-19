#!/usr/bin/env bash
# 1P1D PD launch for the dynamic C8A16 INT8 KV cache.
# P: container devices 0,1 (NPU0).  D: container devices 2,3 (NPU2).
set -euo pipefail
export LAB=/root/remote-dev/dev-env-20261117
_ORIG_PYTHONPATH="${PYTHONPATH:-}"
source "$LAB/env.sh"
export PYTHONPATH="$PYTHONPATH${_ORIG_PYTHONPATH:+:$_ORIG_PYTHONPATH}"
export TMPDIR="$LAB/tmp"
export TRITON_CACHE_DIR="$LAB/tmp/triton_cache"
export VLLM_CACHE_ROOT="$LAB/tmp/vllm_cache"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$LAB/results/v2/pd"
export VLLM_KV_CACHE_LAYOUT=NHD
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_MOONCAKE_CONNECTOR_TIMEOUT=1800
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
MC_LIB=$(python -c 'import mooncake,pathlib;print(pathlib.Path(mooncake.__file__).parent)' 2>/dev/null || true)
if [[ -n "$MC_LIB" ]]; then export LD_LIBRARY_PATH="$MC_LIB:${LD_LIBRARY_PATH:-}"; fi

MODEL="$LAB/models/Qwen3.6-35B-A3B"
LOGS="$LAB/results/v2/pd"
ROLE=$1   # prefill | decode

common=(
  --served-model-name qwen36
  --host 127.0.0.1 --distributed-executor-backend mp
  --tensor-parallel-size 2 --enable-expert-parallel
  --language-model-only --dtype bfloat16 --kv-cache-dtype int8
  --block-size 128 --max-model-len 8192 --max-num-batched-tokens 8192
  --max-num-seqs 16 --gpu-memory-utilization 0.80
  --enforce-eager --no-enable-prefix-caching --seed 2026
)

if [[ "$ROLE" == "prefill" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES=0,1
  KV=$(printf '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":"30001","engine_id":"0","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}')
  exec vllm serve "$MODEL" --port 8100 "${common[@]}" --kv-transfer-config "$KV"
else
  export ASCEND_RT_VISIBLE_DEVICES=2,3
  KV=$(printf '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_port":"30101","engine_id":"1","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}')
  exec vllm serve "$MODEL" --port 8200 "${common[@]}" --kv-transfer-config "$KV"
fi
