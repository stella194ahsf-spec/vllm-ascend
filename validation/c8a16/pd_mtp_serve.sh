#!/usr/bin/env bash
# Same MTP geometry on producer and consumer, BF16 weights for both A/B.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/a3_mtp_env.sh"
GROUP=${1:?usage: pd_mtp_serve.sh A|B prefill|decode smoke|long}
ROLE=${2:?missing role}
PRESET=${3:-smoke}
case "$GROUP" in A) DTYPE=auto ;; B) DTYPE=int8 ;; *) echo 'group must be A or B' >&2; exit 2 ;; esac
case "$PRESET" in smoke) MAX_LEN=8192 ;; long) MAX_LEN=139264 ;; *) echo 'preset must be smoke or long' >&2; exit 2 ;; esac
case "$ROLE" in
  prefill) export ASCEND_RT_VISIBLE_DEVICES=0,1; PORT=8100; KV_ROLE=kv_producer; KV_PORT=30001; ENGINE=0 ;;
  decode) export ASCEND_RT_VISIBLE_DEVICES=2,3; PORT=8200; KV_ROLE=kv_consumer; KV_PORT=30101; ENGINE=1 ;;
  *) echo 'role must be prefill or decode' >&2; exit 2 ;;
esac

python - <<'PY'
import os
from pathlib import Path
import acl
import vllm
import vllm_ascend
for module, directory in ((vllm, "VLLM_REPO"), (vllm_ascend, "ASCEND_REPO")):
    assert Path(module.__file__).resolve().is_relative_to(Path(os.environ[directory]).resolve())
    print(module.__name__, module.__file__)
print("acl", acl.__file__)
print("runtime devices", os.environ["ASCEND_RT_VISIBLE_DEVICES"])
print("vendors", os.getenv("ASCEND_CUSTOM_OPP_PATH"), os.getenv("LD_LIBRARY_PATH"))
PY

KV=$(printf '{"kv_connector":"MooncakeConnectorV2","kv_role":"%s","kv_port":"%s","engine_id":"%s","kv_connector_extra_config":{"use_ascend_direct":true,"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}' "$KV_ROLE" "$KV_PORT" "$ENGINE")
exec vllm serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --served-model-name qwen36 \
  --distributed-executor-backend mp --tensor-parallel-size 2 --enable-expert-parallel \
  --language-model-only --dtype bfloat16 --kv-cache-dtype "$DTYPE" \
  --block-size 128 --max-model-len "$MAX_LEN" --max-num-batched-tokens 8192 \
  --max-num-seqs 8 --gpu-memory-utilization 0.80 --mamba-cache-mode align \
  --enforce-eager --enable-prefix-caching --enable-chunked-prefill --seed 2026 \
  --generation-config vllm \
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --kv-transfer-config "$KV"
