# Simple Molecular Agent

一个从零实现的最小蛋白质-配体改造工作流。主工作流只读取当前项目中显式指定的任务和完整复合物 PDB，不扫描父目录，也不依赖原有工作流。配体三维坐标来自 PDB 的 `HETATM` 记录，化学键级、芳香性、电荷和氢数来自项目内对应的标准化学组件 CIF；运行时不把独立配体 SDF 作为输入契约。PDB `CONECT` 只用于校验原子连接集合，不再被当作完整键级定义。

## 主工作流

```text
任务 + 完整共晶复合物 PDB
  -> 从 PDB 识别蛋白和配体坐标
  -> 用本地化学组件 CIF 恢复并校验配体拓扑
  -> LLM 自主选择结构或化学查询工具
  -> LLM 选择有位点证据的带氢编辑原子和单连接片段
  -> RDKit 生成候选并保留原始骨架坐标
  -> 价态、净电荷、描述符和刚性蛋白碰撞检查
  -> 输出候选 SDF
```

主工作流没有固定 6 Å 口袋输入。`ComplexContext` 解析完整 PDB，后续如何选择、压缩或向 LLM 暴露结构上下文属于主工作流后续设计。

docking 和 AsyncFEP/RBFE 已接入为配置驱动 adapter，但不会自动伪造分数。没有配置外部 docking 命令、AsyncFEP target/依赖或 GPU 时，adapter 明确返回 `not_configured`；只有外部程序成功退出并写出审计结果后，工作流才记录 `complete`。

## LLM 何时停止调用工具

LLM 每轮可以返回 `QUERY`、`READY` 或 `PROPOSE_TOOL`，并自主决定查询顺序和查询内容。工具目录是能力菜单，不是固定查询流程。宿主当前只保留必要的确定性安全限制：

- 最终编辑原子必须查询过局部蛋白环境；
- 同一个最终编辑原子必须查询过增长空间；
- 编辑原子必须存在可替换氢；
- `validate_candidate_geometry` 必须针对最终的 `edit_atom_index + fragment_smiles` 执行；主工作流会在 READY 后确定性验证，最终 ablation 组还要求该工具结果为 `accepted`。
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

预算 `k` 的含义是：`budget-00` 到 `budget-05` 各组最多执行 `k` 次工具调用。预算耗尽后，脚本要求 LLM 直接返回 `READY`；如果模型仍要求工具，该组记录为未完成，不会超预算执行。这六组保持原有 ablation 协议，不启用主工作流的严格位点证据门。随后追加的 `budget-06` 是最终验证组，不限制工具调用次数；它允许 LLM 继续查询，直到自行返回 READY，并要求 READY 选择的同一个 `edit_atom_index` 同时有 `get_atom_environment`、`check_growth_space`，以及同一个 `fragment_smiles` 的 `validate_candidate_geometry` 记录，且该具体候选几何结果必须为 `accepted`，否则状态为 `site_evidence_gate_failed`，不生成候选。环境/增长空间工具只是解释性探查；实际候选几何工具复用最终 RDKit/UFF 构象和碰撞检查。

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

也可以直接使用项目根目录的一键脚本运行 AI 智算云官方 API 对比测试。脚本默认使用 `GLM-5.2`、配体周围 6 Å 口袋坐标和宿主生成的 `ligand_atom_map`，默认从 `$HOME/.aicloud_api_key` 读取纯文本 key，不再要求每次执行前导出环境变量。这里不会把完整 PDB 发送给 LLM。该 key 文件位于项目目录之外，不会被 Git 跟踪：

```bash
./run_ablation.sh
```

如果要使用其他独立 key 文件，把 key 单独写入指定文件，然后运行：

```bash
AICLOUD_KEY_FILE=/path/to/aicloud.key ./run_ablation.sh
```

当前默认 key 文件是空模板。填入 key 后验证：

```bash
printf '%s\n' '你的AI智算云API_KEY' > /home/hwy/.aicloud_api_key
chmod 600 /home/hwy/.aicloud_api_key
pi --list-models 'aicloud/*'
```

`.aicloud_api_key` 已加入 `.gitignore`，不会提交到 GitHub。也可以用 `AICLOUD_API_KEY` 环境变量临时覆盖文件读取。

默认输出到 `runs/ablation-aicloud-pocket-6A-mapped-01/`，等价于运行 `budget-00` 到 `budget-05` 的 6 Å 口袋坐标 + 原子映射 ablation，再追加不限制工具调用且启用严格位点证据门的 `budget-06` 最终验证组。原子映射是固定输入元数据，不计入工具预算。每个 `budget-XX` 子目录在该预算开始时会清理，避免旧决策文件污染本次结果。常用覆盖方式：

```bash
BUDGETS="0 1 2" OUTPUT_ROOT=runs/ablation-aicloud-smoke-01 ./run_ablation.sh
ABLATION_MODEL=GLM-5.2 ./run_ablation.sh
COORDINATE_SCOPE=pocket POCKET_RADIUS=6.0 OUTPUT_ROOT=runs/ablation-aicloud-pocket-6A-mapped-rerun ./run_ablation.sh
```

测试脚本使用 AI 智算云 OpenAI-compatible Chat Completions API：

```text
https://llmapi.blsc.cn/v1/chat/completions
```

如果需要切换到其他兼容网关，可以只给测试脚本覆盖 URL：

```bash
AICLOUD_API_KEY='...' \
ABLATION_BASE_URL="https://your-compatible-endpoint/v1" \
ABLATION_MODEL=GLM-5.2 \
./run_ablation.sh
```

主工作流仍读取 `config.json` 中的模型和端点，不受 `ABLATION_MODEL` 或 `ABLATION_BASE_URL` 影响。若在 `config.json` 或 `config.aicloud.json` 中配置 `docking.command` 和 `rbfe.script`，候选通过几何检查后会写出 protein-only receptor、reference-ligand、候选 pose，并调用对应 adapter；所有命令、stdout/stderr 和返回码会写入运行目录。AI 智算云 key 未填入前，pi 不会显示 `aicloud/GLM-5.2` 为可用模型；填入后重新打开 pi 或重新执行模型列表即可。

严格对比结果时读取各目录的 `result.json` 和根目录的 `summary.json`。其中只有 `budget-06` 的 `state.site_evidence_gate_required` 应为 `true`；`budget-00` 到 `budget-05` 应为 `false`。每组的 `input.json` 还保存了固定的 `ligand_atom_map`，可检查 `rdkit_index`、PDB serial 和配体原子名的映射。重点比较 `status`、`tool_call_count`、`decision_count`、`result.ready_gate`、`result.decision`、`result.validation.property_delta`、`result.validation.structure_change` 和 `error`。这项实验只能比较信息预算与工具使用对方案的影响，不能直接比较活性；后续接入 docking/RBFE 时，应在相同候选评估协议下追加结果。

## Pose、Docking 和 RBFE

当前候选首先使用共晶配体的受限骨架坐标生成 constrained candidate pose，并在 READY 前用同一 RDKit/UFF/刚性碰撞逻辑做具体片段验证。这是几何预筛选 pose，不是 docking pose。docking adapter 应生成一个或多个 receptor-compatible docked poses，并保留原始候选 pose、dock score、pose rank 和输出文件；不能直接覆盖共晶 pose。

RBFE 需要的中间准备不是“再加一个任意 pose”这么简单，至少包括：protein-only receptor、共晶 reference ligand、候选 ligand、reference/candidate 原子映射、质子化/电荷、力场参数、ligand alignment，以及可接受的初始 complex pose。当前 adapter 会生成 protein-only receptor 和 reference-ligand，并创建 AsyncFEP reference-target YAML；实际 AsyncFEP 仍需要其 force-field、OpenMM、GPU 和参数化依赖。

推荐的 pose 层次是：

1. 共晶 constrained pose：主工作流几何预筛选和 reference 起点。
2. docking pose 集合：候选的独立姿势评估，保留 top-N，不直接替换 reference pose。
3. RBFE 对齐/最小化 pose：在 reference-candidate 原子映射和力场参数就绪后生成；若 docking pose 与共晶约束明显冲突，再选择经过 restrained minimization 的 pose。

在有可靠 docking pose、参数化和对齐之前，不应把 `candidate_geometry_accepted` 解读成 affinity 或 activity 结果。

## 测试

```bash
mamba run -n molecular-agent pytest -q
```

## 当前环境限制

当前机器检查未发现 `gnina`、`vina`、`smina`、`OpenMM`、`perses` 等可直接执行依赖，也没有可用 GPU。因此 adapter 代码已接入但真实 docking/RBFE 会返回 `not_configured`，不会生成虚假的 docking score 或 `DeltaDeltaG`。
