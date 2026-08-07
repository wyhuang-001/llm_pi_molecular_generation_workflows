# Simple Molecular Agent

一个从零实现的最小蛋白质-配体改造工作流。主工作流只读取当前项目中显式指定的任务和完整复合物 PDB，不扫描父目录，也不依赖原有工作流。配体的三维坐标和化学图从 PDB 的 `HETATM/CONECT` 记录解析；主工作流运行时不要求独立配体 SDF。

## 主工作流

```text
任务 + 完整共晶复合物 PDB
  -> 从 PDB 识别蛋白和配体
  -> LLM 自主选择结构或化学查询工具
  -> LLM 选择有位点证据的带氢编辑原子和单连接片段
  -> RDKit 生成候选并保留原始骨架坐标
  -> 价态、净电荷、描述符和刚性蛋白碰撞检查
  -> 输出候选 SDF
```

主工作流没有固定 6 Å 口袋输入。`ComplexContext` 解析完整 PDB，后续如何选择、压缩或向 LLM 暴露结构上下文属于主工作流后续设计。

首版不运行 docking 或 AsyncFEP。两者通过 adapter 明确返回 `not_configured`，避免把几何检查冒充活性预测。

## LLM 何时停止调用工具

LLM 每轮可以返回 `QUERY`、`READY` 或 `PROPOSE_TOOL`，并自主决定查询顺序和查询内容。工具目录是能力菜单，不是固定查询流程。宿主当前只保留必要的确定性安全限制：

- 最终编辑原子必须查询过局部蛋白环境；
- 同一个最终编辑原子必须查询过增长空间；
- 编辑原子必须存在可替换氢；
- 候选必须通过 RDKit 解析、价态、净电荷和刚性碰撞检查。

宿主还会阻止完全重复的工具调用并限制最大上下文轮数。若现有工具不足，`PROPOSE_TOOL` 只生成待审核提案，不直接执行任意代码。

## 示例体系

示例为 `1H1Q`：2.5 A 的 CDK2/cyclin A-NU6094 共晶结构。NU6094 是真正的共晶优化起点，不是从其他配体对齐得到的姿势。

```bash
cd simple_molecular_agent
NO_PROXY='*' no_proxy='*' mamba run -n molecular-agent \
  python scripts/prepare_1h1q.py --output input

mamba run -n molecular-agent python -m molecular_agent.cli \
  --task input/task.json --check-input

mamba run -n molecular-agent python -m molecular_agent.cli \
  --task input/task.json --scripted-demo --run-dir runs/demo
```

脚本化演示只验证状态机和化学工具，不包含隐藏的 NU6102 答案。

## 调用第三方 OpenAI-compatible API

```bash
cp config.example.json config.json
export OPENAI_API_KEY='...'
mamba run -n molecular-agent python -m molecular_agent.cli \
  --task input/task.json --config config.json --run-dir runs/live
```

默认示例端点为 `https://api.p1-103n1x.com/v1`，客户端调用 Responses API 的 `/responses`。

## 独立工具预算对比实验

> 本节的 6 Å 规则仅用于额外对比测试，不属于主工作流输入或处理逻辑。

`scripts/compare_tool_budgets.py` 是主工作流之外的独立 ablation 实验，不会修改 `Workflow`、`ComplexContext`、主 CLI、主输入契约或证据门。它固定任务、模型和 API 配置，只改变每次实验允许的工具调用次数。

该测试默认发送：配体坐标 + 配体周围 `6.0 Å` 内命中的蛋白残基的完整坐标。这样可以避免在每组实验请求中重复发送整个大型 PDB。完整复合物坐标仍可用 `--coordinate-scope full` 显式启用。

预算 `k` 的含义是：该组最多执行 `k` 次工具调用。预算耗尽后，脚本要求 LLM 直接返回 `READY`；如果模型仍要求工具，该组记录为未完成，不会超预算执行。每组保存输入坐标、LLM 决策、工具结果、最终改造计划、候选 SDF、RDKit/刚性碰撞验证和 `result.json`。其中 `budget-00` 是只给坐标、不调用工具的直接输入基线。

显式使用完整复合物坐标模式：

```bash
cd /mnt/f/doctoral_period_huangwy/PhD_project/external_model/context_learn/test/simple_molecular_agent
KEY=$(python3 -c 'import json; print(json.load(open("/home/hwy/.pi/agent/auth.json"))["openai"]["key"])')
OPENAI_API_KEY="$KEY" mamba run -n molecular-agent \
  python scripts/compare_tool_budgets.py \
  --task input/task.json \
  --config config.json \
  --output-root runs/ablation-tool-budget-full \
  --budgets 0 1 2 3 4 5 \
  --coordinate-scope full
```

默认的 6 Å 口袋坐标模式如下，仍然只传坐标，不传工具结果或额外配体文件：

```bash
OPENAI_API_KEY="$KEY" mamba run -n molecular-agent \
  python scripts/compare_tool_budgets.py \
  --task input/task.json \
  --config config.json \
  --output-root runs/ablation-tool-budget-pocket-6A \
  --budgets 0 1 2 3 4 5 \
  --coordinate-scope pocket \
  --pocket-radius 6.0
```

建议先做单组检查，再跑完整对比：

```bash
OPENAI_API_KEY="$KEY" mamba run -n molecular-agent \
  python scripts/compare_tool_budgets.py \
  --output-root runs/ablation-tool-budget-smoke \
  --budgets 0
```

也可以直接使用项目根目录的一键脚本运行 DeepSeek 官方 API 测试。脚本默认使用 `deepseek-v4-pro` 和官方 `https://api.deepseek.com/v1`，默认从 `$HOME/.deepseek_api_key` 读取纯文本 key，不再要求每次执行前导出环境变量。该文件位于项目目录之外，不会被 Git 跟踪：

```bash
./run_ablation.sh
```

如果要使用其他独立 key 文件，把 key 单独写入指定文件，然后运行：

```bash
DEEPSEEK_KEY_FILE=.deepseek_api_key ./run_ablation.sh
```

`.deepseek_api_key` 已加入 `.gitignore`，不会提交到 GitHub。也可以用 `DEEPSEEK_API_KEY` 环境变量临时覆盖文件读取。

默认输出到 `runs/ablation-deepseek-pocket-6A/`，等价于运行预算 `0 1 2 3 4 5` 的 6 Å 口袋坐标实验。常用覆盖方式：

```bash
BUDGETS="0 1 2" OUTPUT_ROOT=runs/ablation-deepseek-smoke ./run_ablation.sh
ABLATION_MODEL=deepseek-v4-pro ./run_ablation.sh
COORDINATE_SCOPE=pocket POCKET_RADIUS=6.0 ./run_ablation.sh
```

测试脚本使用官方 DeepSeek Chat Completions API：

```text
https://api.deepseek.com/v1/chat/completions
```

如果需要切换到兼容 DeepSeek 模型的其他网关，可以只给测试脚本覆盖 URL：

```bash
DEEPSEEK_API_KEY='...' \
ABLATION_BASE_URL="https://your-deepseek-compatible-endpoint/v1" \
ABLATION_MODEL=deepseek-v4-pro \
./run_ablation.sh
```

主工作流仍读取 `config.json` 中的模型和端点，不受 `ABLATION_MODEL` 或 `ABLATION_BASE_URL` 影响。

比较结果时读取各目录的 `result.json` 和根目录的 `summary.json`，重点比较 `status`、`tool_call_count`、`decision_count`、`result.decision`、`result.validation.property_delta`、`result.validation.structure_change` 和 `error`。这项实验只能比较信息预算与工具使用对方案的影响，不能直接比较活性；后续接入 docking/RBFE 时，应在相同候选评估协议下追加结果。

## 测试

```bash
mamba run -n molecular-agent pytest -q
```

## 后续增量

1. 在候选通过当前化学/碰撞检查后加入 docking adapter。
2. docking 姿势稳定后，显式连接 `/mnt/f/doctoral_period_huangwy/PhD_project/external_model/AsyncFEP`。
3. 将 docking/FEP 结果作为新 observation 返回 LLM，复用同一 `QUERY/READY` 方式开始下一轮改造。
