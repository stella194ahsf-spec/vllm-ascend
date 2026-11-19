# C8A16 动态 INT8 KV 缓存 PD 改造 —— 环境与源码基线

记录时间：2026-11-19（CST）
主机：node-17-122（80.5.17.122）
容器：`w00953746-dev-env-20261117`

## 1. 已验收的旧环境（PD 验收结果来自这里）

| 仓库 | URL | 完整 commit ID |
| --- | --- | --- |
| vLLM | `https://github.com/vllm-project/vllm` | `4ba787ec9959e566d3b587a796dff07673324da5` |
| vLLM Ascend | `https://github.com/vllm-project/vllm-ascend` | `c251a6a5ed5cc10f7ec6daa5b87d410d731bf6fd` |

工作目录：`/root/remote-dev/dev-env-20261117/src/{vllm,vllm-ascend}`
（`env.sh` 把 `PYTHONPATH` 前置到这两个目录，因此运行时用的是这一份）

### 未提交修改（`git status --short`，vllm-ascend）

```
 M vllm_ascend/attention/attention_v1.py
 M vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/base_worker.py
 M vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
 M vllm_ascend/worker/v2/attn_utils.py
?? benchmark/
?? tests/ut/attention/test_attention_c8a16.py
?? vllm_ascend/attention/attention_c8a16.py
```

vllm 仓库无未提交修改。

## 2. 新私仓环境（仅 import / help / matmul 通过，**未跑 PD 验收**）

| 仓库 | URL | 完整 commit ID |
| --- | --- | --- |
| vLLM | `https://github.com/vllm-project/vllm` | `544f182d9f0c9bfe8bd7645525951d5e3692a542` |
| vLLM Ascend（私仓） | `https://github.com/stella194ahsf-spec/vllm-ascend.git` | `e63a3c19795dd70eb7a12bcbd2a661c37d3bf097`（main） |

- vllm commit 取自私仓 `.github/vllm-main-verified.commit`（注意：与旧仓该文件的
  `24420b7a341af37a9bc3186e6fc6062512e0aa58` 不同，必须用私仓自己的值）。
- 安装方式：两边均 `pip install -v -e . --no-build-isolation --no-deps`
  （build-system 钉死 `torch==2.14.0` / `torch-npu==2.10.0.post4`，不加 `--no-deps`
  会覆盖运行时 torch，破坏 NPU 环境）。
- 安装位置：`/vllm-workspace/vllm`、`/vllm-workspace/vllm-ascend-private`。
- 版本号：`vllm 0.1.dev23637+g544f182d9.empty`、`vllm_ascend 0.1.dev6325+ge63a3c197`。

### 新私仓安装后的 `git status --short`（切到本分支前）

```
(clean，仅 benchmark/ 未跟踪)
```

## 3. 本分支基线

```
branch : c8a16/pd-verified-20261119
tip    : b574ec12589427a74cccb98f152dc62ad90e9ec3
base   : e63a3c19795dd70eb7a12bcbd2a661c37d3bf097   (= 私仓 origin/main)
```

```
b574ec125  [C8A16] Add PD validation scripts, logs and environment baseline
86cd64a37  [C8A16] Make dynamic INT8 KV scale planes PD-transferable
e63a3c197  [Doc][Misc] Revise deployment config details and accuracy evaluation methods
```

### 为什么最终 rebase 到 origin/main

最初按 `c251a6a5e`（已验收旧环境的基线）建分支，但该 commit 不在私仓远端历史中
（私仓 main 为 `e63a3c197`，两者共同祖先只到 `ffca1c99`），推送需要携带 52 MB 对象，
被公司出口网关 `netentsec` 拦截（该网关拒绝大于约 64 KB 的上传 POST）。

改用 `git rebase --onto origin/main c251a6a5e` 重放两个提交后，推送量降到 110 KB + 52 KB，
分两次推送成功。

### rebase 的代价（重要）

- 代码改动内容未变（仍是 949 insertions / 9 deletions 的六个文件），但**父提交变了**，
  因此 `attention_v1.py` / `attn_utils.py` 等文件的上下文与旧基线不同。
- `attn_utils.py` 有一处冲突：origin/main 新增了 EAGLE `is_draft` 标注逻辑，
  与本改动的 `elif` 分支相邻。解决方式为**两者都保留**——C8A16 的 `elif` 在前，
  EAGLE 的 `if` 在其后独立执行。
- 因此**本分支尚未在 rebase 后的基线上跑过单测和 PD 验收**。旧基线上已验收的版本
  保存在容器内 `backup/c8a16-validated-on-c251a6a5e` 分支，可用于对照。

### 行尾

所有代码与验证文件统一为 LF（仓库约定）。最初从 Windows 上传时带入了 CRLF，
已在重新提交前全部规范化；`pd-*.log` 中残留的少量 `\r` 是 tqdm 进度条的真实输出
（`\n\r`），非行尾污染。

## 4. 运行环境

| 项 | 值 |
| --- | --- |
| 镜像 ID | `sha256:d382b9c79679c3790455244ffbc1a4ba01d5c4807963c95b06cb47b0b6483ddb` |
| 镜像名 | `harbor.computing.lab.rnd.huawei.com/hisecr/vllm-ascend:nightly-ci-main-a3-cann9.2.1-20261111-r92` |
| Python | 3.12.16 |
| torch | `2.10.0+cpu` |
| torch_npu | `2.10.0.post7` |
| CANN | 9.2.1（`/usr/local/Ascend/cann-9.2.1`） |
| npu-smi | 26.1.0.b087 |
| SoC | Ascend910（SOC_VERSION=`ascend910_9391`） |
| 容器 CPU | `nproc=1`（编译耗时的直接原因） |

## 5. Mode 5 自定义算子包

| 项 | 值 |
| --- | --- |
| 新私仓构建产物 | `/vllm-workspace/vllm-ascend-private/csrc/build/cann-ops-transformer-custom_linux-aarch64.run` |
| SHA256（新私仓） | `86aa152ea1d1eb1de8c51b9c37d45d0336a8637c3852853febbafa6ab3478a95` |
| SHA256（旧仓产物） | `d5b06023767600a439524f90132e6fef5adf61696ad713eae6d2d50b736c495d` |
| 安装位置 | `<repo>/vllm_ascend/_cann_ops_custom/vendors/custom_transformer` |
| vendor 加载方式 | `export CUSTOM_OPS_VENDOR=/usr/local/Ascend/cann-9.2.1/opp/vendors/custom_transformer`<br>`export LD_LIBRARY_PATH=$CUSTOM_OPS_VENDOR/op_api/lib:$LD_LIBRARY_PATH` |

旧环境实际加载的是系统 vendor：
`/usr/local/Ascend/cann-9.2.1/opp/vendors/custom_transformer`（`version.info` = `custom_opp_compiler_version=9.2.0`）。
新私仓产物编译版本为 `9.2.1`，安装到仓库内 `_cann_ops_custom`，**尚未在 PD 场景下验证**。

## 6. 设备映射

| 角色 | 容器逻辑设备 | 物理 NPU | 服务端口 | KV 端口 | engine_id |
| --- | --- | --- | --- | --- | --- |
| P（prefill） | 0, 1 | NPU0 | 8100 | 30001 | 0 |
| D（decode） | 2, 3 | NPU2 | 8200 | 30101 | 1 |
| 代理 | — | — | 9000 | — | — |

容器启动映射：`/dev/davinci0,1` → NPU0，`/dev/davinci4,5` → NPU2；
`ASCEND_VISIBLE_DEVICES=0,1,4,5`，`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`。
TP=2（每 rank Nq=8 / Nkv=1 / D=256）。

## 7. 验收状态（重要）

| 项 | 状态 |
| --- | --- |
| 旧环境 PD 验收 | **通过**（注册字节精确吻合、740/1420 ranges 传输、PD 与单机 md5 一致、prefix 命中 35.3%） |
| 新私仓 import / `vllm serve --help` / NPU matmul | **通过** |
| 新私仓 PD 验收 | **未执行** |

因此：本分支锁定的是**旧源码环境**下已验证通过的代码与布局。迁移到新私仓后，
必须先用 `validation/c8a16/` 下的脚本复跑小规模 PD 验收，才能把新私仓提交定为后续开发基线。

## 8. 复跑方法

```bash
# 1. 起 1P1D（P=NPU0, D=NPU2）
bash validation/c8a16/pd_serve.sh prefill   # 端口 8100
bash validation/c8a16/pd_serve.sh decode    # 端口 8200
python validation/c8a16/../pd_proxy.py      # 或既有 load_balance_proxy_server_example，端口 9000

# 2. 布局与数值校验（设备级）
python validation/c8a16/c8a16_pd_layout.py

# 3. prefix cache 复用校验
python validation/c8a16/c8a16_prefix_reuse.py

# 4. mode-5 FIA 跨步 scale 契约
python validation/c8a16/scale_stride_test.py

# 5. mode5 + MTP 探针
python validation/c8a16/v2-mode5-mtp-probe.py
```

判据（与旧环境一致）：
- 注册字节数 = `num_layers × num_blocks × page_size_bytes`，精确相等；
- PD 输出完整 md5 与单机非 PD 基线一致（旧基线 `e5203f00926d`）；
- `c8a16_pd_layout.py` 的 `max_abs` < 0.02（旧基线 0.00390625）。

## 9. 已知注意事项

1. **`env.sh` 会遮蔽私仓**：`source env.sh` 后 `import vllm_ascend` 指向
   `/root/remote-dev/dev-env-20261117/src/vllm-ascend`（旧副本）。用私仓需 `unset PYTHONPATH`。
2. **hybrid 模型 prefix cache 命中率**：planner 把注意力 block 提升到 2176 token，
   短 prompt 凑不满一个 block 恒为 0%，需 ≥5218 token 的 prompt。
3. **未配 `--reasoning-parser` 时思考内容进 content**，数值比对请用完整输出 md5，勿看开头片段。
4. **`/tmp` 为 noexec**，安装器自解压后需 `TMPDIR=<overlay目录>`；
   `$HOME/log` 目录在 overlay 中不可写，需 `HOME=/tmp/fakehome`。
