# Simple Molecular Agent

一个从零实现的最小蛋白质-配体改造工作流。主工作流只读取当前项目中显式指定的任务和完整复合物 PDB，不扫描父目录，也不依赖原有工作流。配体三维坐标来自 PDB 的 `HETATM` 记录，化学键级、芳香性、电荷和氢数来自项目内对应的标准化学组件 CIF；运行时不把独立配体 SDF 作为输入契约。PDB `CONECT` 只用于校验原子连接集合，不再被当作完整键级定义。

## 当前工作流：顺序式、位点锁定的局部优化

当前只保留顺序式的单候选优化流程，不保留 portfolio/batch 设计模式。每次 LLM 决策只提出一个单点 transformation；Host 完成确定性构建、几何检查、docking、pose/native-like 评价和相互作用分析，再把结果反馈给 LLM。

```text
任务 + 完整共晶复合物 PDB
  -> Host 从 PDB/CIF 恢复蛋白、配体坐标与化学拓扑
  -> LLM 通过 QUERY 获取配体、口袋、相互作用和合法位点证据
  -> LLM 用 assess_edit_sites 给所有可行位点排序并定义 site_type
  -> Host 锁定当前最高优先级 active_target
  -> LLM 一次提出一个新 transformation
  -> Host 用 RDKit 构建、检查价态/电荷/碰撞/重复结构
  -> 通过候选执行 docking、native-like pose gate、RMSD 和相互作用分析
  -> Host 更新 candidate_history、docking_history 和 sar_memory
  -> LLM 根据累计证据继续当前位点，或用 MARK_UNMODIFIABLE 关闭当前位点
  -> Host 自动推进到下一个未关闭位点
  -> 所有位点完成证据驱动搜索后，LLM STOP；RBFE 暂不进入主循环
```

LLM 不批量提交多个候选，也不要求把整个片段库全部尝试一遍。搜索是按“位点—化学假设—单个候选—反馈”的闭环进行的。完整审计仍写入 run directory；docking 是固定协议下的排序和稳定性信号，不等价于实验活性或真实结合自由能。

## LLM 与 Host 的边界

当前有效动作包括：

- `QUERY` / `QUERY_BATCH`：获取尚未提供且能改变下一步决策的 Host 证据；`QUERY_BATCH` 只用于独立证据查询，不用于批量生成候选；
- `READY`：提交一个具体的、单个位点的 transformation；
- `MARK_UNMODIFIABLE`：在当前位点的证据驱动搜索结束后关闭该位点或一个明确的 modification family；
- `STOP`：在全局搜索完成或安全预算耗尽时停止。

LLM 负责：

- 理解 SMILES、分子图、口袋几何和相互作用证据；
- 选择当前 active target；
- 提出可证伪的局部化学假设；
- 选择一个操作和一个片段；
- 根据历史 SAR、pose 和 docking 反馈决定继续、关闭位点或停止。

Host 负责：

- 确定合法位点和允许操作；
- 精确执行 RDKit transformation；
- 检查价态、电荷、连接性、碰撞和结构重复；
- 运行 docking、RMSD/native-like pose gate 和 PLIP；
- 保存候选历史、失败记录、SAR memory 和 provenance。

## 编辑位点和改变类型（两条正交轴）

编辑不再用一个一维操作名描述，而是 **位点类型 × 改变类型**：

```text
site_type（在哪里改）        change_type（改什么）
  atom    带可替换氢的原子      addition     成一根新键
  bond    一根定向单键          deletion     切一根键并删掉相连部分
  linker  两端保留的链原子      replacement  替换一个原子元素，或换掉相连部分
  ring    环骨架原子
```

组合出的操作标签就是 `site_type:change_type`，例如 `atom:addition`、`bond:deletion`、`bond:replacement`、`ring:replacement`。

每个位点在 dossier 中只列出**它自己允许**的改变类型（`allowed_change_types`）：

```json
{
  "target_type": "atom",
  "site_type": "atom",
  "target_id": 10,
  "allowed_change_types": ["addition"]
}
```

```json
{
  "target_type": "bond",
  "site_type": "bond",
  "target_id": "bond-site-005",
  "cut_bond": [10, 9],
  "allowed_change_types": ["deletion", "replacement"],
  "removed_reference_interactions": [
    {"kind": "hydrogen_bond", "protein_atom": "MET:A:793:N", "distance": 2.71}
  ]
}
```

各自的要求：

- `atom:addition`：位点必须有可替换氢；需要连接一个片段。
- `bond:replacement`：必须使用 Host 返回的 `bond_site_id`，LLM 不能猜测 `cut_bond`、删除集合或连接方向。
- `bond:deletion`：同样使用 `bond_site_id`，但**不需要片段**；Host 会移除该侧并让锚点补氢。
- `atom:replacement` / `ring:replacement`：换一个原子的元素，需要 `element`，不需要片段。环骨架上的元素替换会被报告为 T2。

`deletion` 的关键差别是：它的后果是 **Host 可确定计算的事实**，不是预测。位点会给出 `removed_heavy_atoms`、`removed_fragment_smiles` 以及 `removed_reference_interactions`（哪些参考接触会消失），由 LLM 判断这个代价是否值得。

`linker:addition` / `linker:deletion`（内部连接子长度调整）已经在 taxonomy 中定义，但当前阶段不发射 linker 位点、也不执行：链长变化会移动整个下游片段，需要在非刚性协议下单独定义评价方式。

这些改变类型不是 LLM 自由发明的，而是由以下信息共同决定：

```text
分子图合法性
+ 位点原子类型和氢数
+ 原始配体局部化学环境
+ 口袋接触和相互作用
+ 外向向量与空间 clearance
+ 片段的 operation compatibility
```

## 顺序式多位点搜索策略

当存在多个位点时，不是把所有位点同时提交给 LLM，也不是修改一个位点后自动把新分子再累积修改另一个位点。当前策略是：

1. Host 列出所有合法位点；
2. LLM 根据配体化学、口袋环境、相互作用和空间信息给位点排序；
3. Host 将最高优先级未关闭位点设置为 `active_target`；
4. 当前位点一次只测试一个候选；
5. 当前位点关闭后，Host 才推进到下一个位点；
6. 默认每个候选都从原始共晶配体构建，除非明确启用同一位点的 parent-child 局部优化策略；
7. 不允许把不同位点的编辑自动组合到一个分子中。

因此，多位点搜索更接近：

```text
site A: hypothesis 1 -> feedback -> hypothesis 2 -> close
site B: hypothesis 1 -> feedback -> hypothesis 2 -> close
site C: ...
```

而不是：

```text
site A + site B + site C 一次规划并行生成
```

## 如何避免穷举所有片段

LLM 不以“未使用片段数量”作为继续搜索的理由。每个新候选必须满足至少一个明确目的：

- 验证一个新的相互作用假设；
- 测试一个空间方向或体积级别；
- 比较极性、疏水性、芳香性或氢键能力变化；
- 针对上一个候选的失败原因做结构性修正；
- 检验一个局部 SAR 趋势；
- 对一个 native-like pose 中保留/丢失的 interaction 做定向补偿。

搜索优先使用：

```text
同一位点
+ 尚未尝试的化学 family
+ 与当前口袋和方向匹配的片段
+ 能区分两个竞争假设的候选
```

`search_fragment_library` 和 `get_fragment_panel` 用于按化学标签、大小和 operation compatibility 取得候选；`get_fragment_spatial_profile` 只在片段的形状或延伸范围会改变决策时调用。没有决策相关假设时，不继续查询或生成新片段。

## 位点何时停止

位点不会因为固定尝试次数自动关闭，也不会因为一个候选变差就关闭。当前停止逻辑分三层：

### 1. 硬性不可行

如果 Host 证明位点不能进行合法编辑，例如：

- 没有可替换氢；
- 没有合法 replacement site；
- 所有允许操作均违反价态或连接性；
- 所有候选都发生确定性严重碰撞；
- site table 将位点标记为 protected/hard-reject；

则可以由 LLM 用 `MARK_UNMODIFIABLE` 关闭。

### 2. 局部证据饱和

如果当前位点已经测试了若干**化学上不同**的假设，并且：

- 没有 native-like 且 docking proxy 改善的候选；
- 新候选重复已有 modification family；
- clearance、碰撞或构象问题持续出现；
- 关键 interaction 持续丢失；
- SAR memory 不再支持新的可区分方向；

LLM 可以在引用这些证据后关闭位点。

### 3. 仍有明确可检验方向

如果当前位点虽然已有失败候选，但仍然存在：

- 一个未测试的化学 family；
- 一个由口袋几何支持的方向；
- 一个保留 anchor 的 native-like 改造；
- 一个可以验证当前 SAR 假设的替代片段；

则不能仅因为 docking 变差就关闭位点，应继续提出一个新的单候选。

`local_patience` 目前是 Host 给 LLM 的复查信号，不是“达到 N 次就自动停止”的科学结论。最终的 `MARK_UNMODIFIABLE` 必须保留理由和证据。

## Docking 趋势和收敛

`docking_optimization` 配置主指标、显著改善阈值、seed 稳定性和硬安全上限。每轮同时记录原始 attempt score 与单调 best-so-far 轨迹；允许探索候选变差，不会伪造成每轮都改善。以 `minimizedAffinity` 为主指标时，candidate-reference delta 越负越好。候选必须达到 `minimum_seed_win_fraction` 才进入历史最佳竞争，quality 还会按 `seed_stddev_penalty * seed 标准差` 扣分。

只有在以下信息同时被考虑后，LLM 才应决定是否继续：

```text
docking proxy
+ seed 稳定性
+ native-like pose / RMSD
+ interaction retained/gained/lost
+ 当前位点的化学 family 覆盖
+ 仍未验证的合理假设
```

这些信号仍然不能直接解释为实验活性。

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
# 将 API key 直接填写到 config.json 的 api_key 字段
mamba run -n molecular-agent python -m molecular_agent.cli \
  --task input/task.json --config config.json --run-dir runs/live
```

默认示例端点为 `https://api.p1-103n1x.com/v1`，客户端调用 Responses API 的 `/responses`。配置可用 `api_key` 直接填写 key，也可用 `api_key_file` 指定纯文本 key 文件；配置文件中的 `api_key` 优先，其次才读取环境变量或 Codex 凭据。CLI 默认实时打印 LLM 决策、工具调用、候选几何检查、docking 命令、相对分数趋势和停止原因，并将完整审计 JSON 写入 `--run-dir`；使用 `--quiet` 可关闭实时事件，使用 `--full-json` 可在结束时额外打印完整结果。

### 一次性 Playwright MCP 研究与图片输入

任务文件可以配置原生 MCP stdio 研究。Host 会在第一次 LLM 请求前启动一次 MCP server，执行固定的 `calls`，将文本、来源、结构文件和截图保存到运行目录的 `external-research/`，后续迭代只复用保存的结果，不再浏览：

```json
{
  "external_research": {
    "enabled": true,
    "required": true,
    "transport": "mcp_stdio",
    "mcp_command": ["npx", "-y", "@playwright/mcp@0.0.82", "--headless", "--browser", "chromium", "--no-sandbox", "--image-responses", "allow"],
    "environment": {"PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS": "1"},
    "include_tool_catalog": true,
    "calls": [
      {"tool": "browser_navigate", "arguments": {"url": "https://www.rcsb.org/structure/1H1Q"}},
      {"tool": "browser_snapshot", "arguments": {}},
      {"tool": "browser_take_screenshot", "arguments": {"fullPage": true}}
    ]
  }
}
```

`--image-responses allow` 负责让 MCP 返回图片；`--caps vision` 只负责额外的坐标鼠标操作，并不是视觉模型开关。研究返回的图片会保存为 artifact，ResponsesClient 会把它们转换为 `input_image`；Chat Completions 则转换为 `image_url`。如果当前模型或供应商不支持图片输入，在 LLM 配置中设置 `"send_images": false`，此时仍会保留图片文件和文本/结构化证据。

当前输入提供了一个不覆盖原始 `input/task.json` 的单编辑配置。推荐统一使用本地 `config.single_edit.json`（已加入 `.gitignore`），填写 `base_url`、`model`、`api_key`；不设置 `codex_config_dir`，避免覆盖 endpoint/model。不需要 export。不要将真实 key 放入受版本控制的 example 文件。

```bash
# config.single_edit.json 不存在时，从 config.single_edit.example.json 复制并填写
mamba run -n molecular-agent-docking python -m molecular_agent.cli --task input/task.single_edit.json --config config.single_edit.json --run-dir "runs/single-edit-$(date +%Y%m%d-%H%M%S)"
```

**先校准，再设计：** 当前任务已启用 `pose_retention`。浏览器/LLM 运行前，Host 检查原配体全部 top-N redocking poses 对共晶的核心恢复和少数原子级锚点，要求至少两个 seed 支持同一 family，冻结 redocked 主参考。候选自由 docking，按对冻结参考的直接核心 RMSD/锚点门控后才排序；不固定骨架坐标，不回退 rank 1。受体保留 TPO 等白名单修饰残基。校准失败会明确停止，而不是继续增加候选。协议、默认阈值、离线审计工具和当前未通过的真实校准结果详见 [POSE_RETENTION.md](POSE_RETENTION.md)。

`input/task.single_edit.json` 的第一阶段外部研究只访问对应 RCSB 结构页面 `https://www.rcsb.org/structure/1H1Q` 及其 RCSB entry/chemical-component JSON 记录，随后保存 snapshot 和结构页面 screenshot。不做 PubChem、ChEMBL、BindingDB、UniProt 或在线 SDF 下载。该阶段只提取结构身份、实验质量、配体身份、结合位点/报道 anchor、构建体/突变/物种和文献元数据；原子坐标、分子图、精确距离、RMSD、native-like pose、当前候选相互作用和 docking 仍由 Host 权威生成。导航说明不再伪装为 SDF 文件。

首个 LLM 请求专门分析文本和截图，返回最多 12 条有 source_id 的结构观察和不确定性，保存 `research-memory.json`。观察区分 reported_fact、visual_observation、hypothesis；Host 只校验格式与来源引用，不宣称验证了模型解读。后续所有设计请求只带摘要，不带图片、原始网页或 MCP 工具目录。恢复运行复用已保存摘要；不完整摘要阻止进入设计。原始证据继续留作审计。

单编辑模式禁止 `parent_attempt`：每次都从原始共晶配体构造一个候选，不累积以前的修改。位点仍由 LLM 自选。best-so-far 只参与比较，不是构造基底；主 docking 指标始终与原始配体的同 seed baseline 比较。每轮保留一份原始配体完整图和性质，性质不重复在 ligand 顶层展开。候选完整图/性质保存为 `molecule-attempt-XX.json`，不把全部历史图重发给模型。

暂时不会访问 PubChem、ChEMBL、UniProt、DrugBank 或 BindingDB。原始输入中的 `input/raw/2A6.cif` 和 `input/ligand.sdf` 仍作为 Host 的本地结构依据。当前机器中 GNINA 和 PLIP 都位于 `molecular-agent-docking` 环境，因此实际运行使用该环境；MCP 客户端通过 `PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=1` 跳过 Playwright 的依赖预检，并使用已经验证可以启动的 Chromium。

PLIP 已安装并验证：

```text
PLIP 3.0.1
molecular-agent-docking/bin/plip
```

启用 `plip.enabled` 后，Host 将指定 pose 自动组装为受体复合物并调用 PLIP，保存 `evaluated-complex.pdb`、`report.xml`、`interaction-result.json`、`atom-mapping.json` 和日志。新门控流程分析通过几何预筛的各 poses，最终反馈绑定到所选 Evaluation Pose；未启用门控的旧流程仍使用 rank 1。采用 `--nohydro --nofix`，不让 PLIP 补氢、修复或优化坐标；PDB 投影精度为 0.001 Å。缺氢可能影响氢键识别，PDB 键感知也不等同于 SDF 化学权威，这些限制记录在结果中。

下一轮反馈包含按 seed 的 interaction-type/residue 保留、新增、丢失和计数，不发送原始 XML。PLIP 软反馈失败标记 unavailable，不当作零相互作用；如果显式配置了 PLIP 原子级硬锚点，其评价失败则不能放行该 pose。当前 1H1Q 硬锚点使用 Host 独立的供体/羰基受体几何检查，其余 PLIP 作用是软反馈。集成 smoke test 不是候选优化效果证明。

### 4WKQ 结构测试与盲标签小库方案

新增独立结构校准任务 `4WKQ/task.calibration.json`，不覆盖旧 1H1Q 任务。4WKQ–gefitinib 已完成真实三 seed 原配体校准，三个 seed 均通过固定核心与 MET793 原子级氢键门控。输入显式氢、CCD 的 `CL` 元素符号及 CSX797 保留策略已经处理。

78条匿名混合侧链库已冻结在 `4WKQ/design/`；正式闭集任务为 `4WKQ/task.benchmark.json`，只允许 C6 单编辑，每次展示完整78条目录，最多20个唯一候选，实验标签与成员映射仅由私有离线评估器读取。已真实跑完一轮20候选、三seed随机基线并完成独立评价，未调用付费LLM。初始构象碰撞在此 benchmark 中保留为诊断，最终 docking poses 必须通过重原子碰撞、RMSD及锚点门控。

运行命令、隔离边界、评估方式及实际验证结果见 [`4WKQ/BENCHMARK_RUNNING.md`](4WKQ/BENCHMARK_RUNNING.md)；方案依据见 [`4WKQ/BENCHMARK_PLAN.md`](4WKQ/BENCHMARK_PLAN.md)。此任务是隐藏活性标签的回顾性选择/重发现，不是结构完全留出的 de novo 发现测试。

### 选择多个 LLM（不替换原配置）

CLI 和 `run_docking_loop_test.sh` 均支持 `--llm current|gpt-5.4-mini|gpt-5.6-luna|doubao|deepseek`：

- `current`：沿用原 `--config` 中的模型/端点/Codex 配置，默认仍是原模型。
- `gpt-5.4-mini`：沿用 `current` 的端点、Codex 配置和认证信息，仅把模型覆盖为 `gpt-5.4-mini`，继续使用 Responses API。
- `gpt-5.6-luna`：沿用 `current` 的端点、Codex 配置和认证信息，仅把模型覆盖为 `gpt-5.6-luna`，继续使用 Responses API。考虑到该端点曾出现连接重置，Luna profile 使用 600 秒请求/重试窗口、最多 5 次重试和 10 秒重试间隔。客户端启用 curl `--retry-all-errors`，因此 TLS reset 等错误也会自动重试。
- `doubao`：使用火山方舟 `https://ark.cn-beijing.volces.com/api/v3/responses` 和模型 `doubao-seed-evolving`，从 `ARK_API_KEY` 或项目外的 `~/.config/simple-molecular-agent/doubao-api-key` 读取认证信息。
- `deepseek`：加载 `molecular_agent/data/llm_profiles.json` 中的独立配置，使用 Chat Completions 协议，不读取原模型的认证信息。
- 未指定时读取配置的 `llm_profile`，缺省为 `current`。脚本把显式选择写入该运行的 `runtime-config.json`；恢复时省略参数则沿用运行配置。直接 CLI 的 `--llm` 是本次调用覆盖，恢复时请再次指定。

DeepSeek 使用官方端点 `https://api.deepseek.com/v1`、模型 ID `deepseek-flash`（Chat Completions 协议）。优先读取 `DEEPSEEK_API_KEY`，否则读取项目外的 `~/.config/simple-molecular-agent/deepseek-api-key`（权限应为 `600`）。不要把明文 Key 写入配置或 Git。

新建真实测试（会调用付费 LLM 和 GNINA；使用唯一运行目录）：

```bash
RUN_ROOT=runs/current-$(date +%Y%m%d-%H%M%S) ./run_docking_loop_test.sh --real --llm current
```

```bash
RUN_ROOT=runs/gpt-5.4-mini-$(date +%Y%m%d-%H%M%S) ./run_docking_loop_test.sh --real --llm gpt-5.4-mini
```

```bash
RUN_ROOT=runs/gpt-5.6-luna-$(date +%Y%m%d-%H%M%S) ./run_docking_loop_test.sh --real --llm gpt-5.6-luna
```

```bash
RUN_ROOT=runs/doubao-$(date +%Y%m%d-%H%M%S) ./run_docking_loop_test.sh --real --llm doubao
```

```bash
RUN_ROOT=runs/deepseek-$(date +%Y%m%d-%H%M%S) ./run_docking_loop_test.sh --real --llm deepseek
```

在未完成的旧运行中显式换用 DeepSeek：

```bash
RUN_ROOT=runs/docking-loop-real-160 ./run_docking_loop_test.sh --real --resume --llm deepseek --skip-tests
```

模型选择不会修改 docking/RBFE 参数，也不做自动故障切换。每次 CLI 启动会追加非敏感的 `llm-selection.jsonl`，记录所选 profile、模型、端点和有效重试参数；同一次运行切换模型后属于混合模型运行，不能作为单模型对照实验。

若供应商的模型 ID 或端点不同，可在主配置增加覆盖（不改原模型字段）：

```json
{"llm_profiles": {"deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-flash"}}}
```

### LLM 工作记忆与完整审计

每次工具调用和候选改造仍完整保存为 observation/Event，用于审计、去重和恢复；LLM 输入则使用按全局信息、位点、片段、候选和 elite archive 归并的结构化工作记忆，并仅保留少量最近/代表性历史。不同 parent 的位点事实分开保存，避免将原始配体和子代的局部几何混淆。LLM 请求进度事件同时记录 payload 大小和 token 用量字段（若 API 返回）。

### 从中断运行继续

工作流中断后，可以使用同一运行目录恢复。恢复只读取该目录中的 `observation-*.json`、`context-final.json`、`edit-attempt-*.json`、candidate SDF 和 docking history；不会从其他运行导入候选、评分或 incumbent。已有 `result.json` 的运行视为已完成，不能用 `--resume` 覆盖。

直接使用 CLI：

```bash
mamba run -n molecular-agent-docking python -m molecular_agent.cli \
  --task runs/docking-loop-codex-gpt56-20260902-171109/runtime-task.json \
  --config runs/docking-loop-codex-gpt56-20260902-171109/runtime-config.json \
  --run-dir runs/docking-loop-codex-gpt56-20260902-171109/real \
  --resume
```

也可以使用测试脚本。`RUN_ROOT` 必须是原运行的根目录，脚本会保留原 `real` 子目录和 runtime 配置，不会先删除它：

```bash
RUN_ROOT=runs/docking-loop-codex-gpt56-20260902-171109 \
  ./run_docking_loop_test.sh --real --resume --skip-tests
```

恢复时 attempt 编号从旧 history 的最大编号之后继续；如果上次停在 parent 局部优化，恢复会重新请求尚未成功记录的工具调用，然后继续同一位点的 local child docking。恢复前应确认代码、task、fragment library、docking config 与原运行一致。

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

以下描述的是**未启用 `pose_retention` 的旧流程**；当前单编辑任务请以 [POSE_RETENTION.md](POSE_RETENTION.md) 的全 pose 门控和冻结参考协议为准。

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
