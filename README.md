# Simple Molecular Agent

一个从零实现的最小蛋白质-配体改造工作流。主工作流只读取当前项目中显式指定的任务和完整复合物 PDB，不扫描父目录，也不依赖原有工作流。配体三维坐标来自 PDB 的 `HETATM` 记录，化学键级、芳香性、电荷和氢数来自项目内对应的标准化学组件 CIF；运行时不把独立配体 SDF 作为输入契约。PDB `CONECT` 只用于校验原子连接集合，不再被当作完整键级定义。

## 主工作流

```text
任务 + 完整共晶复合物 PDB
  -> 从 PDB 识别蛋白和配体坐标
  -> 用本地化学组件 CIF 恢复并校验配体拓扑
  -> LLM 自主选择结构或化学查询工具
  -> LLM 选择有位点证据的带氢取代或非环侧链片段替换
  -> RDKit 生成候选并保留参考配体中未替换骨架的坐标
  -> 价态、净电荷、描述符和刚性蛋白碰撞检查
  -> docking 生成并解析 top-N pose/score
  -> docking 结果、完整历史和历史最佳反馈给 LLM，继续查询知识并改造
  -> 达到 docking 平台期且 LLM 同意停止，或触发硬安全上限
  -> 输出历史最佳候选和完整审计；RBFE 暂不进入实际循环
```

主工作流没有固定 6 Å 口袋输入。`ComplexContext` 解析完整 PDB，后续如何选择、压缩或向 LLM 暴露结构上下文属于主工作流后续设计。

docking 和 AsyncFEP/RBFE 保留为配置驱动 adapter，但当前实际设计循环只运行到 docking。docking 为 `complete` 时，top-N pose 属性和外部程序审计会反馈给 LLM；LLM 可查询新证据并更换位点或片段。若 LLM 不修改候选，循环以 `no_candidate_revision_after_docking` 停止，避免重复 docking。RBFE 配置暂时保留但不会被主工作流或 ablation 调用，结果明确记录为 `deferred`，不会伪造分数。

## LLM 何时停止调用工具

LLM 每轮可以返回 `QUERY`、`READY` 或 `PROPOSE_TOOL`，并自主决定查询顺序和查询内容。工具目录是能力菜单，不是固定查询流程。宿主当前只保留必要的确定性安全限制：

- 最终编辑原子必须查询过局部蛋白环境；
- `replace_hydrogen` 的最终编辑原子必须查询过增长空间，并且存在可替换氢；
- `replace_fragment` 必须先调用 `list_fragment_replacement_sites`，再选择宿主返回的 `replacement_site_id`；该操作不使用氢取代式增长探针；
- `validate_candidate_geometry` 必须针对最终完整 transformation 执行；主工作流会在 READY 后再次确定性验证，最终 ablation 组还要求该工具结果为 `accepted`。
- 候选必须通过 RDKit 解析、价态、净电荷和刚性碰撞检查。

宿主还会阻止完全重复的工具调用并限制最大上下文轮数。重复按规范化的 `tool + 完整 arguments` 判断，不是只按原子判断。单个重复调用会复用已有观察结果并留下审计记录；LLM 可用 `QUERY_BATCH` 一次请求多个互不依赖的工具，批内重复项会跳过，其他新调用继续执行。依赖前一个结果的查询仍应使用多轮 `QUERY`。若现有工具不足，`PROPOSE_TOOL` 只生成待审核提案，不直接执行任意代码。

## 编辑操作和片段库

主工作流支持两种受控 transformation，并继续兼容旧的 `edit_atom_index + fragment_smiles` READY 格式：

```json
{"action":"READY","operation":"replace_hydrogen","edit_atom_index":10,"fragment_id":"fluoro","fragment_smiles":"[*:1]F","understanding":"...","edit_hypothesis":"..."}
```

`replace_hydrogen` 要求锚点有可替换氢。`replace_fragment` 不要求锚点有氢。LLM 不能自由猜测 `cut_bond`：它先调用 `list_fragment_replacement_sites`，宿主只枚举合法的定向非环单键切割，并为每个选项返回固定的 `replacement_site_id`、保留骨架、删除侧、连接原子、原子集合、片段 SMILES 和 attachment vector。默认排除删除超过原始重原子 40% 的方向，并可通过 `protected_core_atom_indices` 保护指定核心原子：

```json
{"action":"READY","operation":"replace_fragment","replacement_site_id":"replacement-site-005","fragment_id":"fluoro","fragment_smiles":"[*:1]F","understanding":"...","edit_hypothesis":"..."}
```

宿主由 `replacement_site_id` 恢复切键和方向，不接受 LLM 自由指定切键或删除集合。两种操作都必须针对完整 transformation 调用 `validate_candidate_geometry`，并在实际 design 阶段再次执行相同确定性构建和碰撞检查。当前仍是单步搜索：每个候选都只对原始共晶配体执行一次 transformation，不会在上一轮候选上叠加第二处改造。

离线种子库位于 `molecular_agent/data/fragments.json`，可用 `search_fragment_library` 和 `get_fragment_record` 查询。常见化学名称通过 SMARTS 子结构匹配，其他词使用元数据文本匹配；结果严格遵守记录的 `operation` 或 `allowed_operations`，不会把仅标记为 `substitute` 的记录伪装成 `replace_fragment` 候选。项目提供 ChEMBL 导入器；ChEMBL 提供公开 REST API 和官方 FTP 下载，数据采用 CC BY-SA 3.0，并要求保留 ChEMBL ID、release 和署名。

小规模抽样可继续使用 REST 模式。全量构建应使用官方 `chemreps` FTP 快照：下载支持 `curl` 断点续传和官方 SHA256 校验，BRICS 派生状态保存在 SQLite checkpoint 中，中断后重新执行相同命令即可恢复：

```bash
mamba run -n molecular-agent python scripts/download_chembl_fragments.py \
  --source ftp \
  --molecules 0 \
  --max-parent-mw 350 \
  --max-fragment-heavy-atoms 12 \
  --cache-dir cache \
  --checkpoint molecular_agent/data/chembl_fragments.checkpoint.sqlite \
  --output molecular_agent/data/chembl_fragments.json
```

原始全量派生库保留所有合法单连接点片段。工作流建议使用保守精炼子集：至少两个 ChEMBL 来源分子支持，仅保留常见药化元素、中性且无自由基的片段，并去除 RDKit PAINS/Brenk 警示：

```bash
mamba run -n molecular-agent python scripts/filter_fragment_library.py \
  --input molecular_agent/data/chembl_fragments.json \
  --output molecular_agent/data/chembl_fragments_working.json \
  --min-source-molecules 2
```

在任务 JSON 中用 `fragment_library_path` 指向工作库。联网只发生在显式执行导入脚本时，设计工作流始终读取本地快照。ChEMBL 来源：`https://www.ebi.ac.uk/chembl/`；许可与署名：`https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/`。

项目还提供统一片段库，由 10 条 CC0 curated seed 片段和 ChEMBL working 子集合并生成：

```bash
mamba run -n molecular-agent-docking python scripts/build_unified_fragment_library.py \
  --seed molecular_agent/data/fragments.json \
  --working molecular_agent/data/chembl_fragments_working.json \
  --output molecular_agent/data/fragments_unified.json
```

统一库为每条记录保留 `fragment_id`、`name`、`smiles`、来源记录、`allowed_operations` 和 ChEMBL provenance，并写入确定性的 `size_class` 与 `chemical_tags`。尺寸层级为 `minimal`（1 个重原子）、`small`（2-4）、`medium`（5-8）和 `large`（9-12）；同时保存电荷、分子量、LogP、HBD、HBA、TPSA、环数和可旋转键等基础性质。`size_class` 是 LLM 可选择的动作空间，不是宿主强制的小到大执行顺序。`search_fragment_library` 和 `generate_site_candidate_batch` 支持用 `size_class`、`chemical_tag` 过滤，最终候选仍必须经过几何验证和 docking。

## Docking 趋势和收敛

`docking_optimization` 配置主指标、显著改善阈值、seed 稳定性和硬安全上限。每轮同时记录原始 attempt score 与单调不下降的 best-so-far 轨迹；允许探索候选变差，不会伪造成每轮都改善。以 `minimizedAffinity` 为主指标时，candidate-reference delta 越负越好。候选必须达到 `minimum_seed_win_fraction` 才进入历史最佳竞争，quality 还会按 `seed_stddev_penalty * seed标准差` 扣分，避免由单一 seed 驱动选择。

工作流支持 `search_policy.mode=adaptive` 的证据驱动搜索。启用 `site_lock_enabled` 后，LLM 先通过 `assess_edit_sites` 工具提交 host 校验过的位点优先级和位点类型（`core_anchor`、`pocket_extension`、`solvent_exposed`、`linker_or_sidechain` 或 `uncertain`）。宿主按该策略锁定当前最高优先级的开放位点，并将 `active_target`、`site_search` 和局部统计反馈给 LLM；`minimum_local_attempts` 和 `minimum_local_families` 是显式关闭前的证据下限，`local_patience` 只作为要求 LLM 重新评估的信号，不会自动把位点标为 plateau 或切换到下一位点。位点只有在 LLM 使用有证据理由的 `MARK_UNMODIFIABLE` 关闭后才切换。这样可形成不设局部尝试上限的位点内局部 SAR，而不是每轮在所有位点之间跳转。

`generate_site_candidate_batch` 可由 LLM 调用，为一个锁定位点从 operation-compatible 片段库中批量取出候选，并执行确定性构建和刚性蛋白碰撞预筛选；它不执行 docking，也不替代 READY 的完整证据门。宿主会在发送给 LLM 的位点摘要中计算 `geometry_feasible_not_docked`：它是批次或确定性几何检查已接受、但尚未出现在 docking history 中的 transformation；这不是对整个片段库的穷举，未被查询的片段仍只是潜在候选。完整 observation、GNINA 原始输出、pose、候选结构和 provenance 只保存在运行目录，LLM 输入使用去重后的基线/当前位点/最近窗口和结构化指标摘要。`minimum_distinct_transformations_per_target` 仍是 adaptive 模式的最低多样性门槛。LLM 必须读取每个位点的化学环境、空间方向、已有相互作用、片段性质、attachment-centered 3D profile、docking 分数、seed 稳定性、pose 共识和相互作用变化，再决定继续提出新的化学上不同的 transformation，或使用有证据理由的 `MARK_UNMODIFIABLE` 关闭该位点。几何拒绝也会作为后续搜索证据反馈给 LLM。所有尝试由独立的 `exploration_attempts` 审计账本记录，不会把几何拒绝误认为成功 docking。每次 docking 的主指标、相对参考的表现、历史最佳、seed 稳定性、pose 一致性、相互作用变化和失败原因都会反馈给 LLM。候选变差不会单独触发停止；当前没有局部 `maximum_local_attempts`，`hard_max_attempts` 仍只作为防止进程失控的全局安全上限，而不是科学收敛条件。重复 transformation、重复工具调用和连续无进展决策会被拦截或要求 LLM 修正；最终 `candidate_path` 指向历史最佳候选，而不是最后一次尝试。

这类收敛只表示固定 docking 协议下的搜索平台，不等价于实验活性或真实结合自由能收敛。

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

默认示例端点为 `https://api.p1-103n1x.com/v1`，客户端调用 Responses API 的 `/responses`。配置可用 `api_key_file` 指定纯文本 key 文件；环境变量优先于该文件。CLI 默认实时打印 LLM 决策、工具调用、候选几何检查、docking 命令、相对分数趋势和停止原因，并将完整审计 JSON 写入 `--run-dir`；使用 `--quiet` 可关闭实时事件，使用 `--full-json` 可在结束时额外打印完整结果。

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

也可以直接使用项目根目录的一键脚本运行 OpenAI-compatible API 对比测试。脚本默认使用 `gpt-5.6-sol`、配体周围 6 Å 口袋坐标和宿主生成的 `ligand_atom_map`。主工作流直接从 `~/.codex/config.toml` 读取 provider、Responses API endpoint 和模型，并从 `~/.codex/auth.json` 的 `OPENAI_API_KEY` 读取认证信息；这里不会把完整 PDB 发送给 LLM。Codex 配置和认证文件位于项目目录之外，不会被 Git 跟踪：

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

默认输出到 `runs/ablation-aicloud-pocket-6A-mapped-02/`，等价于运行 `budget-00` 到 `budget-05` 的 6 Å 口袋坐标 + 原子映射 ablation，再追加不限制工具调用且启用严格位点证据门的 `budget-06` 最终验证组。原子映射是固定输入元数据，不计入工具预算。每个 `budget-XX` 子目录在该预算开始时会清理，避免旧决策文件污染本次结果。常用覆盖方式：

```bash
BUDGETS="0 1 2" OUTPUT_ROOT=runs/ablation-aicloud-smoke-01 ./run_ablation.sh
ABLATION_MODEL=gpt-5.6-sol ./run_ablation.sh
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
ABLATION_MODEL=gpt-5.6-sol \
./run_ablation.sh
```

主工作流读取配置中的模型和端点；`config.aicloud.json` 已配置为从 `~/.codex` 读取 `gpt-5.6-sol` 和 Responses API endpoint，不受独立 ablation 的 `ABLATION_MODEL` 或 `ABLATION_BASE_URL` 影响。若在 `config.json` 或 `config.aicloud.json` 中启用 `docking.command`，候选通过几何检查后会写出 protein-only receptor、reference-ligand、候选 constrained pose。系统默认使用 `[17, 29, 43]` 三个固定 seed；对每个 seed，先在 `docking-reference-baseline/seed-*/` 中用同一 receptor、同一 reference autobox 和同一 GNINA 参数独立重对接参考分子，再在 `docking-attempt-XX/seed-*/` 中对接候选，并按相同 seed 配对比较。结果包含每个 seed 的 rank-1 分数、差值、均值、样本标准差、范围、候选胜出次数和胜率。每个 seed 还分别记录 GNINA rank-1 和按各评分指标选择的最优 pose；三个 seed 的 rank-1 pose 会在共享受体坐标系中计算重原子 RMSD 共识，并与参考配体比较跨 seed 残基接触共识。对于 `minimizedAffinity`，更负表示相对更好；对于 `CNNscore`、`CNNaffinity` 和 `CNN_VS`，更正表示相对更好。命令、stdout/stderr、返回码、docked SDF 和 pose 属性摘要会分别写入各 seed 目录。该差值和 pose 共识只是同一协议下的排序与稳定性指标，不是实验亲和力或活性结论。`rbfe` 配置仅保留供未来阶段使用，本轮不执行。Codex 配置有效后，主工作流会使用 `gpt-5.6-sol`；不需要在项目中保存 API key。

严格对比结果时读取各目录的 `result.json` 和根目录的 `summary.json`。其中只有 `budget-06` 的 `state.site_evidence_gate_required` 应为 `true`；`budget-00` 到 `budget-05` 应为 `false`。每组的 `input.json` 还保存了固定的 `ligand_atom_map`，可检查 `rdkit_index`、PDB serial 和配体原子名的映射。重点比较 `status`、`tool_call_count`、`decision_count`、`result.ready_gate`、`result.decision`、`result.validation.property_delta`、`result.validation.structure_change` 和 `error`。这项实验只能比较信息预算与工具使用对方案的影响，不能直接比较活性；后续接入 docking/RBFE 时，应在相同候选评估协议下追加结果。

## Pose、Docking 和 RBFE

当前候选首先使用共晶配体的受限骨架坐标生成 constrained candidate pose，并在 READY 前用同一 RDKit/UFF/刚性碰撞逻辑做具体片段验证。这是几何预筛选 pose，不是 docking pose。docking adapter 应生成一个或多个 receptor-compatible docked poses，并保留原始候选 pose、dock score、pose rank 和输出文件；不能直接覆盖共晶 pose。

RBFE 需要的中间准备不是“再加一个任意 pose”这么简单，至少包括：protein-only receptor、共晶 reference ligand、候选 ligand、reference/candidate 原子映射、质子化/电荷、力场参数、ligand alignment，以及可接受的初始 complex pose。当前 adapter 会生成 protein-only receptor 和 reference-ligand，并创建 AsyncFEP reference-target YAML；实际 AsyncFEP 仍需要其 force-field、OpenMM、GPU 和参数化依赖。

推荐的 pose 层次是：

1. 共晶 constrained pose：主工作流几何预筛选和 reference 起点。
2. docking pose 集合：候选的独立姿势评估，保留 top-N，不直接替换 reference pose。
3. RBFE 对齐/最小化 pose：在 reference-candidate 原子映射和力场参数就绪后生成；若 docking pose 与共晶约束明显冲突，再选择经过 restrained minimization 的 pose。

在有可靠 docking pose、参数化和对齐之前，不应把 `candidate_geometry_accepted` 解读成 affinity 或 activity 结果。

## HTML Docking 报告

对一次已经完成的 LLM + GNINA 运行生成静态 HTML 报告：

```bash
mamba run -n molecular-agent-docking \
  python scripts/generate_docking_report.py \
  runs/docking-loop-real-multiseed-10b/real/result.json
```

报告包含每个 LLM 设计循环的流程行、候选/参考多 seed 分数比较、最终候选和参考基准。生成的 `docking-report.html` 不会重新调用 API 或 GNINA，可直接在浏览器打开。

## 测试

```bash
mamba run -n molecular-agent pytest -q
```

## 计算节点一键安装（推荐）

当前设计、docking 以及后续 AsyncFEP/RBFE 统一使用一个 Python 3.11 计算环境。它包含 OpenMM、AmberTools、OpenFF、Vina 及完整分析栈，并默认下载经过校验的 GNINA，是计算节点的主安装入口。Conda 依赖清单在 `environment.compute.yml`，pip-only 包在 `requirements.compute.txt`，一键安装脚本为 `scripts/install_compute_env.sh`：

```bash
cd /absolute/path/to/simple_molecular_agent
ASYNCFEP_ROOT=/absolute/path/to/AsyncFEP \
  ./scripts/install_compute_env.sh
```

脚本会自动选择 `mamba`、`micromamba` 或 `conda`，创建或更新名为 `molecular-agent` 的环境，并以 editable 方式安装当前项目、AsyncFEP core 和 `bloom_prepare`。重复执行是安全的。先查看将执行的命令而不改动环境：

```bash
./scripts/install_compute_env.sh --dry-run
```

安装完成后可单独复查环境：

```bash
mamba run -n molecular-agent \
  python scripts/check_compute_env.py \
  --asyncfep-root /absolute/path/to/AsyncFEP
mamba run -n molecular-agent pytest -q
```

计算节点需要 Linux x86_64、可用的 Miniforge/Conda，以及能够访问 conda-forge 和 PyPI（或配置好的内部镜像）。OpenMM 的 CUDA 平台还需要节点镜像提供匹配的 NVIDIA 驱动和 `libcuda.so`；安装器只安装用户态 CUDA 依赖，不安装内核驱动。默认按 AsyncFEP Dockerfile 的 CUDA 12.0 基线求解，可用 `COMPUTE_CUDA_VERSION=12.6` 等覆盖；没有 GPU 的登录节点可先用 `REQUIRE_GPU=0` 安装和检查，生产节点建议：

```bash
REQUIRE_GPU=1 REQUIRE_GNINA=1 \
  ASYNCFEP_ROOT=/absolute/path/to/AsyncFEP \
  ./scripts/install_compute_env.sh
```

GNINA 是可选的官方二进制下载项，默认使用 GNINA `v1.3.2` 的官方 Linux release asset 和 SHA256 校验，下载失败只给出 warning；设置 `REQUIRE_GNINA=1` 会把失败变成错误。可用 `GNINA_URL` 指向集群内部镜像，并同步设置对应的 `GNINA_SHA256`。Vina 已由 conda 安装，但它不是 GNINA 命令行的完全 drop-in replacement；当前配置里的 docking/RBFE 仍保持 `enabled: false`，安装不会自动启动真实计算。启用 docking 前应先确认 receptor、reference ligand、候选 pose、原子映射和 docking 输入预检已通过；RBFE 的 target YAML 和参数化检查属于未来阶段。

环境中 NumPy 固定为 `1.26.x`，因为 AsyncFEP 当前声明 `numpy~=1.26`；不要在这个共享环境里执行会升级到 NumPy 2 的普通 `pip install`。安装器对本地包和 `requirements.compute.txt` 使用 `pip --no-deps`，由 conda 统一管理底层科学栈。

如果某个节点只做设计和 docking、确认不会运行 RBFE，可使用轻量的 `molecular-agent-docking` 环境。它不会安装 OpenMM、AmberTools、OpenFF、JAX 或 AsyncFEP；完整说明见 [`DOCKING_INSTALL.md`](DOCKING_INSTALL.md)：

```bash
./scripts/install_docking_env.sh --dry-run
./scripts/install_docking_env.sh
```

## Docking 安装依赖

当前示例命令是 GNINA 命令，不是通用 Vina 命令。启用 `docking.enabled=true` 前，计算节点至少需要：

- 可执行的 `gnina`，以及它所需的 Linux x86_64 运行库；GNINA 可使用 CPU，GPU 加速还需要匹配的 NVIDIA 驱动和 CUDA 运行时。
- RDKit，用于生成和读取候选/参考 SDF；项目环境已声明该依赖。
- 一个只含蛋白 `ATOM` 记录的 receptor PDB，以及共晶 reference ligand SDF；adapter 使用 reference ligand 和 `--autobox_add 4` 自动定义搜索框。
- 可写的运行目录和足够的磁盘空间，用于 `docked.sdf`、日志和命令审计。

Docking-only 安装器会安装 Vina、Open Babel、Meeko 和 ProDy 作为结构准备与 QC 工具，并可选下载 GNINA：

```bash
REQUIRE_GNINA=1 ./scripts/install_docking_env.sh
mamba run -n molecular-agent-docking \
  python scripts/check_docking_env.py --require-gnina
```

Vina 需要单独的 PDBQT receptor/ligand 准备流程，不能直接替换示例中的 GNINA 命令；若使用 Vina，应同时修改 `docking.command`、输入格式和输出文件约定。安装完成后，先用对应环境的检查脚本和一个小规模候选验证 GNINA 输出包含可读 pose，再把配置设为 `enabled=true`。本轮 workflow 到 docking 为止，RBFE 不会启动。

## 当前机器限制

当前开发机不具备计算节点的 GPU 驱动和完整外部程序链时，docking adapter 返回 `not_configured`，RBFE 固定返回 `deferred`，不会生成虚假的 docking score 或 `DeltaDeltaG`。请在计算节点完成上述安装和对应环境检查后再开启 docking；RBFE 即使配置为 `enabled: true`，当前实际循环也不会调用。
