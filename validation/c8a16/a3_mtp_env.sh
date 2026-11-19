#!/usr/bin/env bash
# Source inside the existing A3 container; do not source the old env.sh.
export LAB=/root/remote-dev/dev-env-20261117
export ASCEND_REPO=/vllm-workspace/vllm-ascend-private
export VLLM_REPO=/vllm-workspace/vllm
export PYTHONPATH="$ASCEND_REPO:$VLLM_REPO:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$LAB/tmp"
export TRITON_CACHE_DIR="$LAB/tmp/triton_cache"
export VLLM_CACHE_ROOT="$LAB/tmp/vllm_cache"
export VLLM_KV_CACHE_LAYOUT=NHD
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_MOONCAKE_CONNECTOR_TIMEOUT=1800
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="$no_proxy"
export MODEL="$LAB/models/Qwen3.6-35B-A3B"
export RESULTS="$LAB/results/mtp-v2/${RUN_LABEL:-manual}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$RESULTS"
MC_LIB=$(python -c 'import mooncake,pathlib;print(pathlib.Path(mooncake.__file__).parent)' 2>/dev/null || true)
if [[ -n "$MC_LIB" ]]; then export LD_LIBRARY_PATH="$MC_LIB:${LD_LIBRARY_PATH:-}"; fi
unset MC_LIB
