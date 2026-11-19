# C8A16 MTP 接入与 A3 第二版验证方案

日期：2026-11-19。开发基线：vllm-ascend `ec5cecf3dceedbd301a199596c7dd886b3e92034`，配对 vLLM `544f182d9f0c9bfe8bd7645525951d5e3692a542`。本分支属于待 A3 验证的实现，不能沿用基础 PD 的通过结论替代本轮验收。

## 1. 实现与边界

原有代码只把每请求一个 token 的 DecodeOnly 路由到 Mode 5；多 token 验证会回退到显式 gather/dequant + BF16 TND FIA。旧的 split 探针证明的是 attention 数学路径，没有接入真实 metadata。

本次修改位于 `vllm_ascend/attention/attention_c8a16.py`：

- 使用后端 metadata 扩展点，根据真实 `is_prefilling` 判断请求阶段。MRV2 中的 ChunkedPrefill 可以包含 MTP verification；短 prefill 也可能被原来的 query 长度阈值分类为 decode。
- MTP 构建 metadata 时从设备端有效 `seq_lens` 取得精确长度，防止 CPU 上界包含已拒绝 draft。每个 metadata build 有一次同步，不能宣称零开销；复用同一份 metadata 的 attention 层不重复同步。
- 真实 token 由累积 query 边界描述。对于请求 i，query 数 Q_i、最终有效 KV 长度 L_i，第 j 个 query 的可见长度为 `L_i-Q_i+j+1`。
- 按验证位置合批：Q=[1,2,4] 的 batch 用 4 次单 query FIA，而不是 7 次。每次仍是 BSND、Sq=1、Mode 5、sparse_mode=0、atten_mask=None。不同请求可见长度分别设置，结果回填原始 token 位置。
- 混合 batch 中 decode/verification 使用 Mode 5，prefill 使用原 BF16 attention 路径。新 prefill 保留原始 BF16 K/V 计算；命中缓存的 prefill 仍显式反量化历史 INT8 KV。padding 不参与 verification，也不回填输出。
- 动态量化、INT8 payload 缓存、FP32 scale 缓存和 PD 四组件页布局不变。额外修复 scale scatter 对 PAD_SLOT_ID=-1 的处理，与 payload scatter 的跳过语义一致。布尔筛选可能带来动态 shape/同步开销，需要 profile。

拒绝的 draft 数据可以暂留在 HBM，attention 通过下轮精确 KV 长度屏蔽；重算该位置时 K/V 和两个 scale 都必须覆盖。GDN 是递归状态，不能仅靠缩短 KV 长度回滚。本次不另造一套 GDN 回滚：继续使用仓库 `AscendMambaHybridModelState`、`postprocess_sampled`、`num_accepted_tokens` 和 GDN spec conv/recurrent 路径。

本次未修改 vLLM、C++、AscendC、Mooncake 协议或 run 包，不需要重编算子。原生 Sq>1 Mode 5 仍不支持；FULL graph 仍不支持。本方案采用 eager。

## 2. 固定环境与实验设计

| 项目 | 固定值 |
| --- | --- |
| A3 主机 | node-17-122 / 80.5.17.122 |
| 容器 | w00953746-dev-env-20261117 |
| 镜像 | harbor.computing.lab.rnd.huawei.com/hisecr/vllm-ascend:nightly-ci-main-a3-cann9.2.1-20261111-r92 |
| 私仓源码 | /vllm-workspace/vllm-ascend-private |
| vLLM 源码 | /vllm-workspace/vllm |
| 项目与模型 | /root/remote-dev/dev-env-20261117；其 models/Qwen3.6-35B-A3B |
| P | 物理 NPU0，当前容器逻辑设备 0,1；TP=2，8100 |
| D | 物理 NPU2，当前容器逻辑设备 2,3；TP=2，8200 |
| Proxy | 127.0.0.1:9000 |
| A / B | BF16 权重 + auto KV / BF16 权重 + 动态 INT8 KV |
| 公共配置 | 1P1D、TP2/EP、eager、Prefix Cache、MTP 3 draft tokens |
| 最终工作负载 | 每请求输入 131072 tokens、输出 5120 tokens，默认并发 1 |

128K 按 131072，5K 按 5120。max-model-len=139264，给完整输入输出及 draft 留余量。先检查模型 config 的上下文上限，不通过时停止，不盲目设置长度覆盖环境变量。

为保持 connector 的目标层、draft 层、GDN 状态几何一致，初版在 P/D 两端都加载同样 MTP 配置。P 仍只生成 proxy 预处理所需的短输出，真正连续验证主要在 D 执行。暂不把“仅 D 开 MTP”作为已支持的非对称配置。A/B 必须使用同一份本分支代码和相同参数，只改变 KV dtype。

## 3. 进入容器与切换分支

以下只有第一条在 A3 宿主机执行，其余在容器内执行。不重建容器、不重新拉镜像、不覆盖 env.sh。

```bash
docker exec -it w00953746-dev-env-20261117 bash
```

工作区必须干净；有未提交修改时先保留，不执行 reset/clean。拉取本次专用分支：

```bash
cd /vllm-workspace/vllm-ascend-private
git status --short
git fetch origin c8a16/mtp-ragged-validation-20261119
git switch --track origin/c8a16/mtp-ragged-validation-20261119
git rev-parse HEAD
git -C /vllm-workspace/vllm rev-parse HEAD
```

若已创建同名本地分支，改为 git switch 到该分支，再 git pull --ff-only。不要对旧的已验收分支强推。

每个新终端进入容器后加载：

```bash
export RUN_LABEL=B-smoke-r1
source /vllm-workspace/vllm-ascend-private/validation/c8a16/a3_mtp_env.sh
cd "$ASCEND_REPO"
set -o pipefail
```

RUN_LABEL 是本轮结果子目录名，同一轮 P/D/proxy/请求终端必须相同。不要再 source 项目旧 env.sh；它会前置旧源码。

## 4. 前置检查与单测

确认 NPU0/NPU2 已预留、无其他进程占用；不要停止他人的任务。确认 8100/8200/9000 端口空闲。

```bash
npu-smi info
python - <<'PY' 2>&1 | tee "$RESULTS/environment.log"
import importlib, json, os, subprocess
from pathlib import Path
import torch, torch_npu, acl, vllm, vllm_ascend
for module, root in ((vllm, "VLLM_REPO"), (vllm_ascend, "ASCEND_REPO")):
    print(module.__name__, module.__file__)
    assert Path(module.__file__).resolve().is_relative_to(Path(os.environ[root]).resolve())
expected = "544f182d9f0c9bfe8bd7645525951d5e3692a542"
assert subprocess.check_output(["git", "-C", os.environ["VLLM_REPO"], "rev-parse", "HEAD"], text=True).strip() == expected
extension = importlib.import_module("vllm_ascend.vllm_ascend_C")
print("extension", extension.__file__, "acl", acl.__file__)
print("torch", torch.__version__, "torch_npu", torch_npu.__version__)
for key in ("ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH"):
    print(key, os.getenv(key))
config = json.loads((Path(os.environ["MODEL"]) / "config.json").read_text())
text = config.get("text_config", config)
print("context configuration", {k: text.get(k) for k in ("max_position_embeddings", "rope_scaling", "rope_parameters")})
assert text.get("max_position_embeddings", 0) >= 139264, "需先确认模型的长上下文配置，禁止盲目绕过限制"
print("PASS environment")
PY
pytest -q tests/ut/attention/test_attention_c8a16.py 2>&1 | tee "$RESULTS/unit-attention.log"
pytest -q tests/ut/ops/test_gdn_attn_builder.py tests/ut/ops/test_gdn_mixed_batch.py   tests/ut/ops/test_gdn_recurrent_state_indices.py   2>&1 | tee "$RESULTS/unit-gdn.log"
```

attention 预期 25 passed，GDN 不硬编码数量，以实际全通过为准。若模型 context 检查失败，可以继续 8K 正确性阶段，但不能进行 long 阶段。保留已验收的系统 vendor，不因本次 Python 修改切换 run 包。记录实际包 checksum 和加载路径。

## 5. 真实后端设备检查

只在 P/D 服务均未运行时执行，先用 NPU0 的两个逻辑设备中的第一个。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
python validation/c8a16/c8a16_pd_layout.py --device 0 --out "$RESULTS/layout.json"   2>&1 | tee "$RESULTS/layout.log"
python validation/c8a16/mtp_device_check.py --device 0 --out "$RESULTS/ragged-device.json"   2>&1 | tee "$RESULTS/ragged-device.log"
python validation/c8a16/mtp_device_check.py --device 0 --long-context \
  --out "$RESULTS/ragged-device-long.json" 2>&1 | tee "$RESULTS/ragged-device-long.log"
```

判据：布局四组件仍通过；新脚本的 [1,2,4] 与模拟拒绝后 [1,2,1] 两例均 pass，rtol/atol=0.02，padding 原值保留。新脚本调用本次真正后端方法，并用同一份 INT8 payload/scale 显式反量化后的 BF16 attention 为参考；还会污染已拒绝未来位置，验证有效长度的隔离。长上下文版本再覆盖 KV 长度 131071、131073、136192，先检测交付内核是否支持最终输入输出长度；失败时停止长上下文压测。此处仍不证明 GDN 数值回滚。

## 6. 小规模真实模型 MTP 验证

保持 P/D 停止，三条命令顺序执行，每条退出后确认进程释放设备。这里的 hooks 会同步设备、保存诊断，禁止拿耗时做性能结论。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
python -m validation.c8a16.mtp_validation --model "$MODEL" --kv-cache-dtype int8 --mtp off   --out "$RESULTS/int8-no-mtp.json" 2>&1 | tee "$RESULTS/int8-no-mtp.log"
python -m validation.c8a16.mtp_validation --model "$MODEL" --kv-cache-dtype int8 --mtp on   --out "$RESULTS/int8-mtp.json" 2>&1 | tee "$RESULTS/int8-mtp.log"
python -m validation.c8a16.mtp_validation --model "$MODEL" --kv-cache-dtype auto --mtp on   --out "$RESULTS/bf16-mtp.json" 2>&1 | tee "$RESULTS/bf16-mtp.log"
python validation/c8a16/compare_mtp.py --reference "$RESULTS/int8-no-mtp.json"   --candidate "$RESULTS/int8-mtp.json" --logprob-atol 0.02 --out "$RESULTS/mtp-comparison.json"
```

实际 prompts 长度为 127/128/129/4000，输出长度各不相同，重复运行两次以覆盖请求退出与 slot 复用。真实 MTP 的单 query draft 和多 query target verification 共用新路由，不伪造 sampler 的接受/拒绝。

检查 int8-mtp.json：

- functional_status=pass；各 rank mode5_calls>0，invalid_mode5_calls=0。
- metadata 中实际 decode query 不只固定 4，应检查首次步、尾步、不同请求进度。若只观察到均匀 query，非均匀设备测试虽通过，真实 scheduler 的非均匀覆盖仍需补测。
- metrics_delta 的 drafts>0、accepted>0，不接受“仅配置 MTP 但实际没运行”。
- gdn_postprocess_calls>0，observed_sampler_coverage 中完整接受、部分接受、全部拒绝三项均应有记录；某项为 0 就记为未覆盖，增加 prompt 多样性或输出长度补测，不写成通过。受诊断上限限制，postprocess 最多记录前 512 次调用。
- compare_mtp.py 返回 0 仅表示这些生成序列严格一致，且三类拒绝/接受覆盖齐全时，可作为本轮 GDN 回滚的端到端行为回归证据。它不是逐层状态等价证明。
- 返回 2 时检查首次分叉的共同历史 logprob、top1/top2 margin。tie_candidate 只提示近似并列，仍需复核；不比较分叉后不同上下文的 logprob，也不自动放宽容差。0.02 是本方案预设的诊断阈值，不能据此宣称全分布数值等价。

如果真实 MTP 测试未通过，不进入最终性能验收，先保留 JSON 和报错在 A3。

## 7. 小规模 1P1D + MTP + Prefix Cache

使用四个容器终端，全部 export RUN_LABEL=B-smoke-r1 后 source 同一个环境脚本。这里 P/D 两端均开 MTP 与 Prefix Cache，保留基础 PD 已支持的相同页几何；新规划产生的注册总字节不必等于旧的 13940326400，应按本次实际计划重新核对。

P 终端：

```bash
bash validation/c8a16/pd_mtp_serve.sh B prefill smoke 2>&1 | tee "$RESULTS/prefill.log"
```

D 终端：

```bash
bash validation/c8a16/pd_mtp_serve.sh B decode smoke 2>&1 | tee "$RESULTS/decode.log"
```

Proxy 终端：

```bash
python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py   --host 127.0.0.1 --port 9000 --prefiller-hosts 127.0.0.1 --prefiller-ports 8100   --decoder-hosts 127.0.0.1 --decoder-ports 8200 2>&1 | tee "$RESULTS/proxy.log"
```

请求终端：

```bash
curl -fsS http://127.0.0.1:8100/health
curl -fsS http://127.0.0.1:8200/health
python validation/c8a16/prefix_bench.py run --model "$MODEL" --group B   --input-tokens 4096 --output-tokens 128 --requests 4 --concurrency 2 --min-hit-rate 0   --out "$RESULTS/pd-smoke.json" 2>&1 | tee "$RESULTS/pd-smoke.log"
```

smoke 的 min-hit-rate=0 仅为短上下文功能检查：仍要求实际命中率大于 0、D 端 MTP drafts/accepted>0，不能作为 90% 达标证明。检查 D 日志中出现 `C8A16_MODE5_VERIFICATION`（A 组应无该标记）。它说明新分支已提交 Mode 5 调用；结合请求完成和设备测试作为路由证据，不能仅凭一条可能出现在 warmup 的日志证明所有 decode 调用都命中。精确逐调用覆盖使用第 6 节诊断，性能跑次只保留一次性日志。检查 Mooncake 日志中成功注册、目标 attention 四组件和 GDN 状态传输、无异构页映射错误、无超时；不要将 D 外部 KV 命中当作 P 本地 APC 命中。

对 A 组也完整运行一遍，RUN_LABEL=A-smoke-r1，P/D 启动命令中的 B 换成 A。切组前在各服务终端 Ctrl+C 结束本轮实例，确认其 worker 已退出，勿使用全局 pkill -9。

## 8. 最终 128K + 5K 高命中 A/B

先确认前面各项 gate 已通过、上下文配置允许、单个 128K cache 可以保留。在本轮全新 P/D 实例上运行，不复用上一组缓存。推荐顺序 A1、B1、B2、A2、A3、B3，降低时间漂移；每次都使用相同 warmup。

以下以 A1 为例，四个终端均设置 RUN_LABEL=A-long-r1 后 source 环境。在对应 P/D 终端分别运行：

```bash
bash validation/c8a16/pd_mtp_serve.sh A prefill long 2>&1 | tee "$RESULTS/prefill.log"
```

```bash
bash validation/c8a16/pd_mtp_serve.sh A decode long 2>&1 | tee "$RESULTS/decode.log"
```

Proxy 命令与前面相同。两端 ready 后在请求终端执行：

```bash
python validation/c8a16/prefix_bench.py run --model "$MODEL" --group A   --input-tokens 131072 --output-tokens 5120 --requests 4 --concurrency 1   --out "$RESULTS/benchmark.json" 2>&1 | tee "$RESULTS/benchmark.log"
```

B1 使用 RUN_LABEL=B-long-r1，P/D 和客户端 group 全部换成 B。其余参数相同。客户端先做一次不计入测量窗口的 warmup，共享 98% token 前缀，尾部不同，输入严格为 131072 个 token；warmup 生成 1 token，测量请求生成 5120 token。共享前缀比例不等于实测 APC 命中率，hybrid 的管理页对齐/淘汰可能降低命中。

脚本从 P 的 /metrics 取 `prefix_cache_hits` / `prefix_cache_queries` 增量，从 D 另取 MTP 增量；保留前后完整 metrics。要求 P 本地 token 命中率 >90%，MTP 实际接受 token，所有请求输出长度准确，否则非零退出。若当前版本指标名称或定义不同，先查实际 /metrics 与实现，不能拿 connector/external 指标替换。

每对跑次比较：

```bash
python validation/c8a16/prefix_bench.py compare   --a "$LAB/results/mtp-v2/A-long-r1/benchmark.json"   --b "$LAB/results/mtp-v2/B-long-r1/benchmark.json"
```

比较脚本要求 token workload hash、输入输出、请求数、并发相同，两组各自 gate_passed，且 P 本地命中率相差不超过 1 个百分点。任一不满足，不发布该对性能结论。先修复容量/缓存复用，再以相同参数重跑；不得只对一组减少并发或缩短输出。

报告三轮的中位数及范围：TTFT、TPOT、输出 tokens/s、P 命中率、MTP 接受长度、KV 容量、实际注册页几何。脚本的 TPOT 按流式首末内容时间及真实输出 token 数计算；MTP 一个 SSE chunk 可携带多个 token，因此不将 chunk 间隔冒充逐 token ITL。当前 4 请求的小规模结果不代表饱和吞吐。

128K 输入只需预热一次，但 5K 输出可能耗时很长，timeout 默认为每次网络读等待 21600 秒。可先用相同 128K 输入、128 输出试跑容量与 APC；它不属于正式 128K+5K 结果。OOM 时保留日志和原配置，不擅自放宽 A/B 条件。

## 9. Profile 与性能归因

正式跑次不安装 worker diagnostic hooks。若 B 仍慢，单独短跑 profile 以下内容：

1. 每个 verification step 的 Mode 5 FIA 调用数，应约为活动请求的最大 Q_i，非请求数乘 Q_i；普通单 token batch 一次 FIA。
2. metadata 精确长度 D2H 同步、scale scatter 的动态索引、query/table index_select 和 output index_copy。
3. 非连续 K/V/scale 是否触发 torch_npu/CANN 的 AutoContiguous、隐式 Clone 或整池复制。能接受跨步视图不等于没有拷贝，也不等于每次只读取有效 KV。
4. Q=4 是否反复读同一长 KV，INT8 带宽节省能否抵消四次 FIA、提交开销与 MTP draft 开销。
5. GDN causal_conv1d/recurrent 是否仍回退 PyTorch，同步/MoE 是否主导端到端时间。
6. APC 命中、PD 传输字节、MTP 接受率是否在 A/B 间相近。接受率变化也会影响 TPOT，不能全部归因于 FIA。

## 10. 结果与回退

结果保存在 `$LAB/results/mtp-v2/<RUN_LABEL>/`，不要求上传文件或推送日志。你后续同步每项通过情况、首次错误和摘要即可。

本轮新增：`a3_mtp_env.sh`、`pd_mtp_serve.sh`、`mtp_device_check.py`、`mtp_validation.py`、`compare_mtp.py`、`prefix_bench.py`，以及本手册；修改 backend 和单测。仓库原有基础 PD 脚本保留。

回退时先停掉本轮自己的服务，确认源码工作区干净，然后切回 `c8a16/pd-verified-20261119` 的 ec5cecf3d 基线。无需重装 Python 包或 run 包。

本机完成的是 CPU 隔离测试与静态检查。实际 NPU、真实 MTP/GDN、PD+MTP、长上下文 APC 和性能均待按本文在 A3 执行。发布合入前须将 release-note-draft 的内容放入正式 PR 编号对应的 release note fragment，并补齐硬件证据。
