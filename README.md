# Simple Molecular Agent

一个从零实现的最小蛋白质-配体改造工作流。它只读取当前项目中显式指定的任务、复合物 PDB 和配体 SDF，不扫描父目录，也不依赖原有工作流。

## 当前闭环

```text
任务 + 共晶复合物 PDB + 同一配体 3D SDF
  -> LLM 每轮选择一个结构查询工具
  -> 宿主证据门判断是否允许 READY
  -> LLM 选择一个已查询的带氢编辑位点和单连接片段
  -> RDKit 生成候选并保留原始骨架坐标
  -> 价态、净电荷、描述符和刚性蛋白碰撞检查
  -> 输出候选 SDF
```

首版不运行 docking 或 AsyncFEP。两者通过 adapter 明确返回 `not_configured`，避免把几何检查冒充活性预测。

## LLM 何时停止调用工具

LLM 每轮只能返回 `QUERY` 或 `READY`。`READY` 必须同时满足宿主硬门槛：

- `ligand_identity`：已读取配体结构、净电荷和原子索引；
- `pocket_environment`：已查询口袋残基；
- `key_interactions`：已查询基础相互作用；
- `edit_site_environment`：已查询最终编辑原子的局部环境；
- `edit_site_geometry`：已检查同一编辑原子的增长方向。

此外，宿主阻止重复工具调用，并限制最大查询轮数。即使 LLM 自称信息足够，只要硬门槛没通过就不能设计。

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

`scripts/compare_tool_budgets.py` 是主工作流之外的独立 ablation 实验，不会修改 `Workflow` 的证据门或状态逻辑。它固定任务、模型和 API 配置，只改变每次实验允许的工具调用次数。

默认输入是：配体坐标 + 配体周围 `6.0 Å` 内的蛋白残基坐标。这样可以保留局部结合环境，避免把整个大型 PDB 文本塞入每次 LLM 请求。完整复合物坐标仍可用 `--coordinate-scope full` 显式启用。

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

也可以直接使用项目根目录的一键脚本。它会优先使用已导出的 `OPENAI_API_KEY`；如果未导出，则临时读取 `$HOME/.pi/agent/auth.json` 中的 key，不写入项目文件：

```bash
./run_ablation.sh
```

默认等价于运行预算 `0 1 2 3 4 5` 的 6 Å 口袋坐标实验。常用覆盖方式：

```bash
BUDGETS="0 1 2" OUTPUT_ROOT=runs/ablation-smoke ./run_ablation.sh
COORDINATE_SCOPE=pocket POCKET_RADIUS=6.0 ./run_ablation.sh
```

比较结果时读取各目录的 `result.json` 和根目录的 `summary.json`，重点比较 `status`、`tool_call_count`、`decision_count`、`result.decision`、`result.validation.property_delta`、`result.validation.structure_change` 和 `error`。这项实验只能比较信息预算与工具使用对方案的影响，不能直接比较活性；后续接入 docking/RBFE 时，应在相同候选评估协议下追加结果。

## 测试

```bash
mamba run -n molecular-agent pytest -q
```

## 后续增量

1. 在候选通过当前化学/碰撞检查后加入 docking adapter。
2. docking 姿势稳定后，显式连接 `/mnt/f/doctoral_period_huangwy/PhD_project/external_model/AsyncFEP`。
3. 将 docking/FEP 结果作为新 observation 返回 LLM，复用同一 `QUERY/READY` 方式开始下一轮改造。
