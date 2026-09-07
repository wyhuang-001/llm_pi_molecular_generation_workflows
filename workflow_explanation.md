# `molecular_agent/workflow.py` 代码说明

> 本文根据当前工作区中的 `molecular_agent/workflow.py` 编写。文件同时保留旧的 QUERY/READY 兼容路径和默认的 portfolio batch 路径；批量多位点方案见 `LLM_BATCHED_MULTISITE_OPTIMIZATION.md`。代码行号会随后续修改变化。
>
> 本文件只解释 `workflow.py` 的职责、状态流转、证据门、候选生成、docking 反馈、恢复和持久化逻辑，不重复粘贴完整源代码。完整工具的具体化学实现主要位于 `molecular_agent/tools.py`、`editing.py`、`structure.py` 和 `fragment_library.py`。

## 1. 文件的总体职责

`workflow.py` 是 molecular agent 的主控制器。它不直接实现所有化学计算，而是把以下组件组织成一个由 LLM 驱动、由 Host 确定性校验的闭环：

```text
任务 JSON + 完整复合物结构
        │
        ▼
ComplexContext 解析结构和任务
        │
        ▼
LLM 选择 QUERY / READY / QUERY_BATCH / MARK_UNMODIFIABLE / STOP
        │
        ├── QUERY：调用 ToolRegistry 获取结构、片段或几何证据
        ├── READY：提交一个具体的分子 transformation
        ├── MARK_UNMODIFIABLE：声明某个位点或化学家族不再继续搜索
        └── STOP：请求终止，但必须通过停止条件检查
        │
        ▼
确定性证据门和 transformation 校验
        │
        ▼
RDKit 构造候选 + 价态/电荷/空间碰撞检查
        │
        ▼
GNINA docking（候选与参考配体使用相同 seed 配对）
        │
        ▼
计算相对指标、seed 稳定性、pose 共识和相互作用变化
        │
        ▼
把压缩后的反馈交还给 LLM，选择新的化学假设
```

该文件的核心设计原则是：

1. **LLM 负责提出知识请求和化学假设**，但不能直接执行任意代码。
2. **Host 负责确定性事实和安全检查**，包括工具执行、结构构造、证据门、去重、几何检查和 docking 统计。
3. **完整审计信息写入运行目录**，而发送给 LLM 的上下文只使用有界的摘要。
4. **候选搜索不是简单的“每轮分数变好”循环**。候选可以变差；系统保留探索记录、拒绝记录和历史最佳。
5. **docking 结果是固定协议下的排序和稳定性信号**，不是实验活性或真实结合自由能。

---

## 2. 依赖关系和主要对象

文件开头导入了以下组件：

| 对象 | 来源 | 作用 |
|---|---|---|
| `Chem` | RDKit | 分子复制、原子检查、SMILES 规范化、SDF 读写辅助 |
| `NotConfiguredAdapter`, `configured_adapters` | `adapters.py` | docking/RBFE adapter；没有配置时使用占位 adapter |
| `EditResult`, `apply_transformation`, `write_sdf` | `editing.py` | 执行受控分子改造、输出验证报告和 SDF |
| `FragmentLibrary` | `fragment_library.py` | 读取本地片段库、检查片段来源和允许的操作 |
| `AgentState`, `ToolObservation` | `models.py` | 保存工作流状态、工具观察、历史和工作记忆 |
| `ComplexContext` | `structure.py` | 解析任务 JSON、复合物、配体、蛋白原子和输出 receptor |
| `ToolRegistry` | `tools.py` | 注册并执行 Host 化学/结构/片段工具 |

### 2.1 `DecisionClient` 协议

```python
class DecisionClient(Protocol):
    def complete_json(self, payload: dict[str, Any]) -> dict[str, Any]: ...
```

它是 LLM 客户端的最小接口。`Workflow` 不关心具体使用哪一个 API，只要求客户端接收一个字典 payload，并返回 JSON 字典。

因此可以接入：

- OpenAI-compatible 客户端；
- 测试用的 scripted client；
- 其他实现了 `complete_json` 的本地或远程客户端。

### 2.2 异常类型

文件定义了三个与决策校验有关的异常：

- `ReadyDecisionError`：READY 缺字段、操作不支持、编辑位点非法或违反某个 READY 规则。
- `DuplicateToolCallError`：为重复工具调用准备的异常类型，携带结构化 rejection。
- `ReadyEvidenceError`：READY 化学 transformation 本身不一定错误，但当前上下文缺少支持该 transformation 的工具证据。

`ReadyEvidenceError` 会计算 `requires_llm_review`：如果缺少片段库记录、片段性质或片段空间 profile，说明 LLM 还需要阅读片段知识后重新判断；如果只缺少可以自动补齐的局部证据，Host 可以尝试自动执行推荐查询。

---

## 3. `Workflow.__init__`：创建运行时环境

位置：约第 78–118 行。

构造函数接收：

```python
Workflow(
    task_path,
    client,
    run_dir,
    config_path=None,
    progress=None,
)
```

### 初始化步骤

1. `ComplexContext(task_path)` 读取任务和复合物。
2. 将 `run_dir` 转成绝对路径并创建目录。
3. 把原始配体复制为 `parent_candidates[0]`。
4. 为 parent 0 创建元数据：这是原始配体，attempt 为 0、generation 为 0。
5. 从任务中的 `fragment_library_path` 找到片段库。相对路径以任务输入目录为基准解析。
6. 创建 `ToolRegistry`，并注入 `parent_resolver=self._resolve_parent_candidate`，从而允许工具针对某个历史稳定候选查询局部环境。
7. 如果提供 `config_path`，通过 `configured_adapters` 创建 docking 和 RBFE adapter；否则使用 `NotConfiguredAdapter`。
8. 创建 `AgentState`，其中 `max_context_rounds` 默认从任务读取，默认值为 8。
9. 初始化 `reference_docking_result`，用于缓存参考配体的 docking baseline。
10. 将 `_design_phase` 设为 `False`。上下文收集阶段和实际 design 阶段对可发送给 LLM 的历史范围不同。

### parent candidate 的意义

工作流支持局部 parent 优化。普通第一代候选以原始配体为 parent；如果某次候选满足 seed 稳定性门槛，则它会被保存为新的可用 parent。后续 LLM 可以在 READY 中提供 `parent_attempt`，但 site-lock 模式下 child 必须仍然属于同一个 active target。

这可以表示为：

```text
parent 0：原始共晶配体
   │
   ├── attempt 1：generation 1
   │       │
   │       └── attempt 3：generation 2，parent_attempt=1
   │
   └── attempt 2：generation 1
```

不过 candidate 构造仍由 Host 控制，LLM 只能选择已经存在的 parent attempt，不能凭空引用不存在的候选。

---

## 4. 状态模型：`AgentState` 中保存什么

`workflow.py` 通过 `AgentState` 保存所有状态。`models.py` 中的字段可分成四层。

### 4.1 决策和工具审计

- `decisions`：LLM 返回的原始有效决策。
- `observations`：实际执行过的工具调用、arguments、result 和 evidence。
- `call_signatures`：工具调用的哈希签名，用于去重。
- `tool_rejections`：工具调用、READY 或 STOP 被拒绝时的结构化原因。

### 4.2 分子搜索历史

- `candidate_history`：每个 design attempt 的候选、验证、docking 摘要。
- `docking_history`：候选相对参考配体的 docking 统计和历史最佳轨迹。
- `exploration_attempts`：比 candidate history 更底层的探索账本。它还记录几何拒绝、批量预筛选、重复结构和 `MARK_UNMODIFIABLE`。
- `unmodifiable_targets`：LLM 已声明关闭的位点或化学家族。

`exploration_attempts` 很重要：几何失败也代表一个被探索过的化学假设，但它不代表成功 docking，也不代表有效候选。

### 4.3 site-lock 和搜索控制

- `site_strategy`：LLM 通过 `assess_edit_sites` 提交、Host 保存的位点排序。
- `active_target`：当前被锁定、必须继续搜索的最高优先级位点。
- `site_search`：每个位点的尝试数、docking 数、化学家族、最佳质量和 patience 状态。
- `convergence`：主指标、历史最佳、连续无显著改善次数和全局搜索状态。

### 4.4 可重建工作记忆

- `global_memory`：配体、口袋、相互作用等全局事实的紧凑摘要。
- `site_memory`：按 atom 或 replacement site 分组的局部事实和 docking 事件。
- `fragment_memory`：按 fragment ID 或 SMILES 保存的片段事实。
- `candidate_memory`：按 attempt 保存的候选 docking 结果。
- `elite_archive`：最多保留 10 个按质量排序的 elite 候选摘要。

这些 memory 是**可重建缓存**，不是唯一事实来源。完整 observations、candidate reports 和 docking 原始结果仍然是审计依据。

---

## 5. 运行入口：`run()`

位置：约第 3572 行。

`run(resume=False)` 的流程非常简单：

```python
if resume:
    result = self._resume_design()
else:
    first_decision = self.collect_context()
    result = self.design(first_decision)

final = {"state": self.state.compact_view(), "result": result}
_write_json("result.json", final)
return final
```

运行开始和结束时会通过 `_emit()` 发出进度事件；如果没有传入 `progress` callback，`_emit` 什么也不做。因此进度面板/实时输出并不是 `Workflow` 的强制副作用，真正的结果始终写入 `run_dir/result.json`。

---

## 6. 上下文收集阶段：`collect_context()`

位置：约第 2098–2170 行。

上下文收集阶段的目标不是立刻做 docking，而是让 LLM 获取足够证据，提交第一个合法的 READY。

### 每轮步骤

1. 构造 `_query_payload()`。
2. 调用 `client.complete_json(payload)`。
3. 使用 `_repair_decision()` 修复常见的 JSON/动作格式错误。
4. 使用 `_handle_with_recovery()` 执行 QUERY、QUERY_BATCH 或 MARK_UNMODIFIABLE。
5. 如果返回 READY，调用 `_validate_design()`。
6. 如果 READY 缺少证据，记录 rejection，并按情况自动补查询或重新请求 LLM。
7. 如果返回 STOP，收集阶段直接失败，因为此时还没有有效 design。
8. 达到 `max_context_rounds` 仍没有合法 READY，则失败。

### 上下文阶段允许的动作

有效 action 包括：

- `QUERY`
- `QUERY_BATCH`
- `READY`
- `MARK_UNMODIFIABLE`
- `PROPOSE_TOOL`
- `STOP`

`PROPOSE_TOOL` 不会执行新工具，而是把提案保存为 `tool-proposal-XX.json` 后中止，等待 Host 审核。

### 证据集合

`AgentState` 中的基础证据集合来自 `models.py`：

```python
SITE_EVIDENCE = {
    "edit_site_environment",
    "edit_site_geometry",
    "candidate_geometry",
}
```

`state.missing_evidence` 是全局视角的缺失证据；最终 `_validate_design()` 会根据具体 operation、位点、parent 和片段再进行更精确的检查。

---

## 7. LLM 决策格式修复和错误恢复

### 7.1 `_unwrap_decision()`

LLM 有时会把有效决策包在 `answer`、`decision` 或 `response` 字段中。该方法最多拆两层，只要最终对象包含 `action` 就接受。

### 7.2 `_repair_decision()`

该方法处理三类常见错误：

1. **把工具名直接当作 action**

   错误形式：

   ```json
   {"action": "get_atom_environment", "atom_index": 10}
   ```

   修复要求：

   ```json
   {
     "action": "QUERY",
     "tool": "get_atom_environment",
     "arguments": {"atom_index": 10}
   }
   ```

2. **返回了 transformation，但缺少 READY 包装**

   方法会要求返回带有 `action: READY`、`understanding`、`edit_hypothesis` 和完整 transformation 的对象。

3. **返回裸参数或普通文本**

   方法会要求返回一个 action 属于白名单的完整 JSON 对象。

最多进行两次格式修复。若模型持续返回 transformation-shaped 对象，`_normalize_transformation_action()` 只补上 `action: READY`，不会替模型猜测缺失的化学字段；之后仍必须经过正常 READY 语义校验。

所有非法决策都会写入 `invalid-decision-XX.json`，以便审计。

### 7.3 `_handle_with_recovery()`

它负责执行决策时的恢复：

- QUERY 参数非法；
- QUERY_BATCH 中包含非法项；
- MARK_UNMODIFIABLE 的 target、scope 或 active target 不合法；
- 工具调用被拒绝或重复。

发生错误后，rejection 会加入 `state.tool_rejections`，并连同当前状态重新发给 LLM。最多尝试三轮恢复；恢复失败则抛出异常。

---

## 8. 工具调用：`_execute_query()` 和 `QUERY_BATCH`

位置：约第 929–1003 行，以及 `_handle_decision()` 中的 QUERY_BATCH 分支。

### `_execute_query()` 的安全限制

执行前会检查：

1. `tool` 必须是字符串。
2. `arguments` 必须是字典。
3. `tool + arguments` 的 SHA-256 签名不能已经执行过。
4. `assess_edit_sites` 必须在 `get_edit_site_candidates` 之后执行。
5. design 阶段不能重新调用 `assess_edit_sites` 试图重排已经开始的 active target。
6. design 阶段的 `generate_site_candidate_batch` 只能针对当前 active target。

执行成功后：

- 将签名加入 `call_signatures`；
- 把结果和 evidence 加入 `observations`；
- 更新 `state.evidence`；
- 更新 working memory；
- 对特殊工具更新 `site_strategy` 或 `exploration_attempts`；
- 写入 `observation-XX.json`；
- 通过 `_emit("tool_completed", ...)` 发送小型摘要。

### 重复工具调用

重复定义是：

```text
tool 名称 + 完整 arguments 的规范化 JSON
```

不是只按 atom index 判断。因此同一个工具对不同 radius、不同 parent 或不同 fragment 参数可以是不同调用。

重复调用不会再次执行 Host 工具。系统会生成包含原观察结果的 rejection，并要求 LLM 选择新的 query、不同的 READY 或 STOP。

### QUERY_BATCH

`QUERY_BATCH` 允许一次请求多个互不依赖的查询。批处理逻辑会：

- 检查每一项格式和工具名；
- 跳过历史重复项；
- 跳过同一批内部重复项；
- 保留其他新查询继续执行；
- 检查剩余上下文预算；
- 按顺序执行可执行查询。

依赖前一个工具结果的查询不应放在同一个 batch 中，而应分多轮执行。

---

## 9. Transformation 的规范化：`_transformation()`

位置：约第 1099–1157 行。

READY 决策经过 `_transformation()` 后，转换为 Host 内部统一结构。支持两类操作：

### 9.1 `replace_hydrogen`

示例：

```json
{
  "action": "READY",
  "operation": "replace_hydrogen",
  "edit_atom_index": 10,
  "fragment_id": "fluoro",
  "fragment_smiles": "[*:1]F",
  "understanding": "该位点朝向口袋空腔。",
  "edit_hypothesis": "在该位点加入小型卤素取代基。"
}
```

要求最终编辑原子是可支持的带氢重原子。芳香 `[nH]` 由于当前单键编辑器没有显式互变异构/质子化处理，被视为 Host 不支持的位点。

### 9.2 `replace_fragment`

示例：

```json
{
  "action": "READY",
  "operation": "replace_fragment",
  "replacement_site_id": "replacement-site-005",
  "fragment_id": "fluoro",
  "fragment_smiles": "[*:1]F",
  "understanding": "该侧链位于可切割的非环单键外侧。",
  "edit_hypothesis": "替换外侧片段以探索局部空间。"
}
```

LLM 不能直接提交任意 `cut_bond` 或删除原子集合。它必须先调用 `list_fragment_replacement_sites`，再使用 Host 产生的 `replacement_site_id`。Host 根据 site ID 恢复：

- 切割键；
- 保留原子；
- 被删除的一侧；
- attachment atom；
- 片段 SMILES；
- attachment vector。

这样可以防止模型自由猜测切割方向。

### 9.3 片段库一致性

如果提供 `fragment_id`，方法会：

1. 从片段库读取记录；
2. 检查记录是否允许当前操作；
3. 如果没有 `fragment_smiles`，使用库记录中的 SMILES；
4. 如果两者都提供，则使用 RDKit 判断结构等价；
5. 保存完整 `library_record` 供后续验证和审计。

`replace_hydrogen` 在片段库权限中对应 `substitute`；`replace_fragment` 则必须明确允许 `replace_fragment`。这避免把只允许氢取代的片段误用于片段替换。

### 9.4 parent 和 generation

当 READY 提供 `parent_attempt` 时：

- parent attempt 必须是正整数；
- 必须能在 `parent_candidates` 中解析；
- generation 为 parent generation + 1；
- `replace_existing_substituent` 设为 `True`。

如果没有 parent，默认以原始配体为基础，generation 为 1。

---

## 10. transformation 去重

`_transformation_key()` 会先调用 `_exploration_transformation()`，然后生成排序后的 JSON 字符串。

规范化会保留 operation、位点、fragment SMILES、parent、generation、fragment ID、replacement site 和 cut bond 等关键字段，但用于化学重复判断时会移除：

- `fragment_id`；
- `generation`。

原因是同一个化学结构不应仅因为 fragment ID 或 generation 不同就被视为新 transformation。`cut_bond` 也会规范成排序后的二元组。

`_transformation_was_attempted()` 会检查：

- `exploration_attempts` 中由 design 提交或被判为几何拒绝、重复结构、docking 失败的 transformation；
- `candidate_history` 中已有的 transformation。

如果重复，后续 READY 会被拒绝，要求 LLM 使用新化学假设或查询新事实。

---

## 11. READY 证据门：`_validate_design()`

位置：约第 2298–2500 行。

这是 workflow 的核心安全边界。READY 并不等于立即生成候选；Host 首先验证结构字段和证据是否齐全。

### 11.1 必需的解释字段

READY 必须包含：

- `understanding`：对局部口袋/位点的理解；
- `edit_hypothesis`：本次结构改造的假设。

缺少时直接产生 `ReadyDecisionError`。

### 11.2 transformation 合法性

方法检查：

- operation 必须是 `replace_hydrogen` 或 `replace_fragment`；
- edit atom index 必须在 parent 分子范围内；
- `replace_hydrogen` 的原子必须有可替换氢，除非这是带 parent 的替换已有取代基场景；
- `replace_fragment` 必须有 Host 枚举的 `replacement_site_id`；
- 片段 SMILES 必须存在且有效。

### 11.3 位点环境证据

最终位点必须有匹配的 `get_atom_environment` 观察。匹配键包括：

```text
(atom_index, parent_attempt)
```

因此原始配体位点和子代 parent 的同一数值 atom index 不会被混淆。

### 11.4 增长空间证据

`replace_hydrogen` 还要求相同位点存在 `check_growth_space` 观察。`replace_fragment` 不要求氢增长探针，因为它使用 Host 给出的 replacement site 和 attachment vector。

### 11.5 replacement site 证据

`replace_fragment` 必须至少执行过一次 `list_fragment_replacement_sites`。最终 site ID 仍会通过 `resolve_replacement_site()` 再解析，LLM 不能只凭文本猜测切割键。

### 11.6 自适应模式中的片段知识证据

当 `search_policy.mode == "adaptive"` 时，最终片段还需要与下列观察关联：

- 选中的片段库记录，尤其是 `replace_fragment`；
- 选中的片段性质；
- 选中的片段空间 profile；
- 完整 transformation 的 accepted `validate_candidate_geometry` 结果。

如果缺少片段库记录、片段性质或空间 profile，`ReadyEvidenceError.requires_llm_review` 为真。Host 可以自动补齐部分 query，但不会跳过 LLM 对新片段证据的阅读和判断。

### 11.7 精确候选几何证据

普通位点环境和增长空间只能证明“这个位点值得研究”，不能证明“这个具体 fragment 可以接上去”。因此最终 transformation 必须有完全匹配的 `validate_candidate_geometry` 且结果为 `accepted`。

匹配基于规范化 transformation，而不是只比较 fragment 或 atom index。

---

## 12. 位点策略和 site-lock

### 12.1 site strategy

LLM 先通过：

1. `get_edit_site_candidates` 获取 Host 支持的站点 dossier；
2. `assess_edit_sites` 提交带有 priority、site_type、rationale 的位点策略。

策略项通常包含：

```json
{
  "target_type": "atom",
  "target_id": 10,
  "priority": 1,
  "site_type": "pocket_extension",
  "rationale": "该位点朝向可增长空腔，且不破坏已知锚定相互作用。"
}
```

`assess_edit_sites` 的结果会保存到：

- `state.site_strategy`；
- `site-strategy.json`；
- `state.active_target` 和 `state.site_search`。

### 12.2 `_refresh_site_search()`

该方法根据策略和已有探索记录重算每个位点的：

- status：active、pending 或 closed；
- 尝试数；
- 几何接受/拒绝数；
- docking 数；
- 已测试的局部化学家族；
- 最佳质量和最佳 attempt；
- 不改善次数；
- `local_patience` 是否达到。

其中 attempt count 是唯一 Host 记录的 transformation 数，包含 batch 几何预筛选和几何拒绝，不等于 docking 次数。

优先级最高的 pending site 会成为 active target。

### 12.3 `_site_lock_rejection()`

在 site-lock 开启且处于 design 阶段时：

- READY 必须针对 `active_target`；
- 不能跳到更低优先级位点；
- 如果使用 parent，parent 的 target 也必须属于当前 active target；
- 只有显式 `MARK_UNMODIFIABLE` 后才允许切换位点。

`local_patience` 只是反馈给 LLM 的 review 信号，不会自动切换位点，也不会自动终止搜索。

### 12.4 `MARK_UNMODIFIABLE`

`_record_unmodifiable()` 验证：

- `target_type` 为 `atom` 或 `replacement_site`；
- atom target 必须是 Host 支持的带氢重原子；
- replacement site 必须由 Host 枚举；
- `scope` 为 `site` 或 `family`；
- family 必须是 `halogen`、`non_halogen` 或 `fragment_replacement`；
- reason 非空；
- 不能重复关闭同一个目标；
- site-lock 模式中只能关闭当前 active target。

接受后，它既写入 `unmodifiable_targets`，也写入 `exploration_attempts`，但不会被当作候选成功或 docking 证据。

---

## 13. 全局搜索覆盖：`_global_search_coverage()`

位置：约第 2638–2990 行。

该方法计算搜索是否完成，以及还存在什么 pending obligations。

### 13.1 可编辑氢位点

它从原始配体中找出：

- 原子序数大于 1；
- 总氢数大于 0；
- 当前单键编辑器支持的原子。

芳香 `[nH]` 会列入 `host_ineligible_hydrogen_atoms`，作为 Host 已知不可处理的位点，而不是无限要求 LLM 尝试。

### 13.2 replacement sites

方法会通过 `list_fragment_replacement_sites(limit=100)` 获得 Host 支持的 replacement site ID，并统计每个位点的尝试记录。

### 13.3 化学家族覆盖

对于 `replace_hydrogen`，家族大致包括：

- `halogen`；
- `alkyl`；
- `polar`；
- `heteroaryl`；
- `other`。

对于 replacement operation，局部家族会加上 `fragment_replacement:` 前缀。

在非 adaptive 的 coverage 逻辑中，代码还会检查：

- 每个可编辑原子是否至少有一次尝试；
- 空间净空较小时是否至少覆盖 halogen；
- 空间更充足时是否同时覆盖 halogen 和 non-halogen；
- 每个 replacement site 是否至少测试两个不同片段；
- 全局是否见过 halogen、non-halogen 和 fragment replacement；
- 如果某个位点的 halogen 候选优于参考，是否进行了 non-halogen follow-up。

### 13.4 adaptive 模式

`mode == "adaptive"` 时，完成条件主要是所有开放 target 都已关闭；LLM 需要阅读局部证据并决定继续提出新 transformation，或者使用 `MARK_UNMODIFIABLE` 关闭位点。

### 13.5 site-lock 模式

如果 `site_lock_enabled` 且已有 `site_strategy`，完成条件变为：

1. strategy 至少有 `minimum_prioritized_sites` 个位点；
2. 策略中的所有位点状态都是 `closed`。

这时 `pending_obligations` 主要包含：

- 还没有合法 site strategy；或
- 当前 active target 的局部搜索义务。

---

## 14. STOP 门：`_stop_gate_rejection()`

位置：约第 2994–3022 行。

LLM 返回 STOP 后，是否接受由此方法决定。

- `optimization` 模式：直接允许 STOP，因为该模式强调在有限预算下维护高质量候选组合。
- 其他模式：只有 `_global_search_coverage()["complete"]` 为真时才允许 STOP。
- 如果不完整，返回 `global_search_incomplete` rejection，并把 pending obligations 发给 LLM。

因此普通模式中的 STOP 不是无条件命令。LLM 仍然是科学搜索终止的主要决策者，但 Host 可以阻止明显尚未覆盖的搜索。

> 当前代码中，`_search_policy()` 实际读取的字段包括 `mode`、`site_lock_enabled`、`site_strategy_required`、`minimum_prioritized_sites` 和 `local_patience`。README 中提到的 `minimum_local_attempts`、`minimum_local_families` 和 `minimum_distinct_transformations_per_target` 在当前 `workflow.py` 的 `_search_policy()` / `_record_unmodifiable()` 中没有作为独立硬门实现；它们如果存在于 task JSON，不会由这几个方法直接强制执行。阅读实际行为时应以代码为准。

---

## 15. design 阶段主循环：`design()`

位置：约第 3215–3570 行。

这是从第一个 READY 开始，到最终结果或停止为止的主循环。

### 15.1 初始化

方法读取：

- `primary_metric`，默认 `minimizedAffinity`；
- `minimum_improvement`，默认 0.25；
- `seed_stddev_penalty`，默认 0.25；
- `minimum_seed_win_fraction`，默认 2/3；
- `hard_max_attempts`，默认使用任务中的 `max_edit_attempts` 或 30。

随后创建：

- `history`：当前运行的 attempt report；
- `seen_candidate_smiles`：用于检测重复候选结构；
- `reference-ligand.sdf` 路径；
- `receptor-protein-only.pdb` 路径。

### 15.2 每次 attempt 的顺序

```text
1. 校验当前 READY
2. 规范化 transformation
3. 解析 parent candidate
4. 记录 exploration_attempt=submitted
5. apply_transformation
6. 写 edit-attempt-XX.sdf 和初始 report
7. 判断几何/结构验证是否通过
8. 检查 canonical SMILES 是否重复
9. 写 candidate-XX.sdf
10. 准备 reference ligand 和 protein-only receptor
11. 调用 docking adapter
12. 记录 docking 和 candidate history
13. 计算相对质量、seed 稳定性和历史最佳
14. 将 docking feedback 交给 LLM
15. 要求下一次 chemically distinct transformation、QUERY 或 STOP
```

### 15.3 候选构造失败或几何拒绝

`apply_transformation()` 抛异常，或者返回的 report 不是 `accepted` 时：

- docking 不会启动；
- report 的 failure stage 为 `deterministic_geometry_prescreen`；
- exploration record 更新为 `geometry_rejected`；
- 写入 `edit-attempt-XX.json`；
- 几何失败原因反馈给 LLM；
- LLM 只能选择新的 transformation、查询新证据或合法停止。

几何拒绝会计入搜索证据，但不会计入 docking 成功。

### 15.4 重复候选结构

即便两个 transformation 字段看起来不同，只要生成的 canonical SMILES 已经出现过，就会：

- 标记为 `duplicate_candidate_structure`；
- 不运行 docking；
- 保存首次出现的 attempt；
- 记录 exploration 状态 `duplicate_structure`；
- 要求 LLM 生成新的化学假设。

### 15.5 保存候选和参考配体

第一个可接受候选出现时：

- 写 `reference-ligand.sdf`，内容是原始共晶配体；
- 调用 `context.write_receptor_pdb()` 写出只含蛋白的 receptor；
- 候选写成 `candidate-XX.sdf`。

这确保后续 docking 使用明确、可审计的输入文件。

---

## 16. docking 调用和相对参考基线

如果 adapter 提供 `run_with_reference_baseline()`，工作流会同时运行或读取：

- 候选 docking；
- 参考配体 docking baseline；
- 候选与参考使用相同 seed 的配对结果。

参考 baseline 完成后缓存到 `self.reference_docking_result`，后续 attempt 可以复用。

如果 adapter 没有该方法，则调用普通 `run()`。

docking report 写入当前 attempt 的 JSON，并包含状态、seed 数、pose 数、candidate/reference 每 seed 结果、命令审计路径和比较指标。docking 非 `complete` 时，当前 workflow 通过 `_accepted_output(..., "docking_not_complete")` 返回已有最佳结果，不会把未完成 docking 当作成功评分。

---

## 17. docking 评分：`_record_docking_result()`

位置：约第 3024–3175 行。

### 17.1 相对指标

方法从 docking comparison 中读取主指标的：

```text
delta = candidate - reference
```

如果指标是 lower-is-better，例如 `minimizedAffinity`：

```text
raw_quality = -delta
```

如果指标是 higher-is-better，例如 CNN score：

```text
raw_quality = delta
```

因此内部统一成“越大越好”。

### 17.2 seed 稳定性

候选只有在：

```text
candidate_better_seed_fraction >= minimum_seed_win_fraction
```

时才是 `stability_eligible`。满足后，质量分数计算为：

```text
quality = raw_quality - seed_stddev_penalty * seed_stddev
```

这会降低 seed 间波动很大的候选的排名，避免单个 seed 偶然胜出就成为 parent 或历史最佳。

### 17.3 历史最佳和非改善次数

- `is_new_best`：当前 quality 是否超过历史 best。
- `is_significant_improvement`：是否超过历史 best 加上 `minimum_improvement`。
- `best_attempt`：历史最佳 attempt。
- `non_improving_attempts`：连续没有显著改善的次数。

候选只要不是显著改善，也可能仍然成为新的较小 best；代码区分“刷新 best”和“显著改善”，因此不会强行把分数轨迹做成单调改善。

### 17.4 convergence 的语义

`convergence` 会记录：

- 主指标和方向；
- 已评分 attempt 数；
- seed 稳定候选数；
- best attempt、best quality；
- best candidate 相对 reference 的 delta；
- pose/interaction 和全局搜索摘要；
- 下一次决策提示。

`converged` 在正常记录时仍是 `False`。这里的“收敛”更多是搜索状态和平台期信息，而不是实验意义上的结合自由能收敛。

---

## 18. docking 后如何要求 LLM 继续搜索

每次 docking 完成后，`design()` 生成 `feedback`，其中包括：

- docking 结果的 LLM 安全摘要；
- 最新 attempt 的相对指标；
- seed 标准差和胜率；
- 是否稳定 eligible；
- 是否是新的历史最佳；
- incumbent 的 best attempt 和 best quality；
- convergence 状态；
- 推荐继续查询的工具。

随后调用 `_retry_ready_decision()`。

### `_retry_ready_decision()` 的规则

1. 所有已经尝试过的 transformation 都禁止再次提交。
2. LLM 可以查询新事实。
3. LLM 可以提交新的 READY。
4. LLM 可以提交 `MARK_UNMODIFIABLE`。
5. LLM 可以提交 STOP，但必须通过 `_stop_gate_rejection()`。
6. 连续没有观察、探索或关闭位点进展时，计入 no-progress retry。
7. 连续重复 transformation 超过限制时，workflow 失败。

如果 LLM 在 docking 后返回与刚完成的候选完全相同的 transformation，代码会再发一次 `candidate_not_revised` rejection；第二次仍相同则抛出异常，防止无限重复 docking。

---

## 19. working memory 与完整审计

### 19.1 `_llm_safe_value()`

该方法为 LLM 上下文裁剪数据：

- 排除 `source_molecule_ids`、命令、stdout/stderr、raw HTTP body、poses 等大字段；
- `fragments` 最多保留 20 条；
- 普通 list 最多保留 50 条；
- 长字符串最多 4000 字符。

### 19.2 observation 视图

上下文收集阶段保留所有 observation 的压缩视图；design 阶段优先保留：

- baseline tools；
- active target 相关工具；
- 最近 12 条 observation。

随后按 tool + arguments 去重，最多保留 32 条 observation 发送给 LLM。

### 19.3 docking 和 candidate 历史

`_compact_docking_history()` 只发送结构化指标，例如：

- delta mean/stddev；
- candidate seed 胜率；
- quality；
- pose stability；
- interaction consensus。

`_compact_candidate_history()` 发送候选路径、验证状态、分子量、heavy atoms、canonical SMILES、docking 状态和少量 pose 摘要。

完整 docking pose、命令、日志和原始输出不进入常规 LLM payload，而保存在运行目录中。

### 19.4 `_update_working_memory()`

该方法把工具结果和 docking 结果写入分层 memory：

- 全局工具事实；
- site 的 parent-specific 上下文；
- fragment ID/SMILES 事实；
- candidate attempt 的 docking 事实；
- elite archive。

site memory 会限制同一 parent 上保留的 docking event 数，candidate memory 最多保留最近 32 个，elite archive 最多保留 10 个。这样可以在不丢失完整审计的情况下控制 LLM 输入规模。

---

## 20. 中断恢复：`_restore_run_state()` 和 `_resume_design()`

### 20.1 恢复来源

恢复只从当前 `run_dir` 读取：

- `observation-*.json`；
- 如果没有 observation，则读取 `context-final.json`；
- `edit-attempt-*.json`；
- eligible candidate SDF；
- docking history。

如果已有 `result.json`，认为该 run 已经完成，不允许覆盖恢复。

### 20.2 恢复检查

方法会验证：

- 保存的 task 与当前 task 相同；
- observation 能重建 `ToolObservation`；
- candidate/docking history 能重建；
- eligible attempt 的 edit report 和 candidate path 存在；
- candidate path 位于当前 run directory 内，避免从其他 run 注入候选；
- candidate SDF 可以被 RDKit 读取；
- 参考 docking baseline 可以重新识别。

### 20.3 恢复后重建

恢复时会重新建立：

- call signatures；
- global/site/fragment/candidate memory；
- parent candidates；
- parent metadata；
- reference docking result。

然后 `_resume_design()` 使用最后一个 report 的 decision 作为恢复起点，生成一个 `workflow_resume` rejection，要求 LLM 从下一个未尝试 transformation 继续，而不是重复已完成候选。

---

## 21. `_accepted_output()`：最终结果结构

最终结果会包含：

- `status`：通常为 `candidate_accepted`；
- `stopping_reason`；
- `best_attempt`；
- `candidate_path`：历史最佳候选，而不是最后一次候选；
- `reference_path`；
- `attempts`；
- 最佳候选 docking 摘要；
- 完整 `docking_history`；
- `elite_archive`；
- `convergence`；
- `rbfe` 和兼容字段 `fep`。

RBFE 当前明确返回：

```json
{
  "stage": "rbfe",
  "status": "deferred",
  "message": "RBFE is intentionally deferred until docking pose selection and ligand mapping are validated."
}
```

因此当前 workflow 不会伪造 RBFE 分数。

---

## 22. 审计文件

典型 run directory 中会出现：

| 文件 | 内容 |
|---|---|
| `decision-XX.json` | LLM 决策和当时的紧凑 state |
| `observation-XX.json` | 工具调用后的 state |
| `context-final.json` | 上下文收集完成时的 state |
| `site-strategy.json` | Host 接受的位点策略 |
| `invalid-decision-XX.json` | 非法 JSON/action 及修复信息 |
| `tool-proposal-XX.json` | 待 Host 审核的新工具提案 |
| `edit-attempt-XX.json` | 每次设计的决策、transformation、验证和 docking report |
| `edit-attempt-XX.sdf` | 设计阶段写出的候选或编辑结果 |
| `candidate-XX.sdf` | 通过几何和重复结构门的候选 |
| `reference-ligand.sdf` | 原始参考配体 |
| `receptor-protein-only.pdb` | 只含蛋白的 receptor |
| `docking-history.json` | docking 历史和 convergence |
| `result.json` | 最终 state 和 result |

完整工具输出和 docking 原始文件由相应工具/adapter 写入各自的目录。`workflow.py` 只负责在关键节点保存 JSON 审计和候选路径。

---

## 23. `ScriptedDemoClient`

文件末尾的 `ScriptedDemoClient` 是确定性的 smoke-test client，不包含实验 SAR 答案。

它大致按以下顺序工作：

1. 调用 `get_edit_site_candidates`；
2. 调用 `assess_edit_sites`，把 atom 10 标为一个脚本化的 pocket-extension target；
3. 查询 atom 10 的 `get_atom_environment`；
4. 查询 atom 10 的 `check_growth_space`；
5. 查询 `[*:1]F` 在 atom 10 的 exact `validate_candidate_geometry`；
6. 提交 `replace_hydrogen` 的 READY。

它的作用是测试状态机、证据门、候选构造和文件输出，而不是证明真实化学设计结论。

---

## 24. 一次完整运行的伪代码

```python
def run_workflow():
    load task and complex
    create original ligand as parent 0
    create tools and docking adapters

    if resume:
        restore state from current run directory
        ask LLM to continue from last unfinished decision
    else:
        while context budget remains:
            payload = build_context_payload()
            decision = ask_llm(payload)
            decision = repair_invalid_schema(decision)

            if decision is QUERY:
                execute_new_tool_call()
            elif decision is QUERY_BATCH:
                execute_new_nonduplicate_tool_calls()
            elif decision is MARK_UNMODIFIABLE:
                validate_and_record_closure()
            elif decision is READY:
                if ready_evidence_is_incomplete:
                    reject_and_request_evidence()
                else:
                    break
            elif decision is STOP:
                fail_because_no_candidate_was_selected()

    for attempt in range(1, hard_max_attempts + 1):
        validate READY again
        normalize transformation
        select parent

        try:
            candidate = apply_transformation(parent, transformation)
            if candidate is invalid or clashes:
                record geometry rejection
                ask LLM for a new transformation
                continue

            if canonical_smiles_was_seen:
                record duplicate structure
                ask LLM for a new transformation
                continue

            save candidate and reference inputs
            docking = run_candidate_and_reference_with_paired_seeds()
            record docking and candidate history
            update convergence and working memory

            if docking_is_incomplete:
                return best_available_result

            decision = ask LLM using docking feedback
            if decision is STOP and stop_gate_passes:
                return best_candidate
            if decision repeats the same transformation:
                reject it and ask once more

        except candidate_or_docking_error:
            record complete failure audit
            recover or stop according to the error class

    return best_candidate_or_no_candidate_result
```

---

## 25. 代码阅读时最重要的几个区分

### 25.1 evidence 和 result 不同

工具返回了某种局部事实，不代表一个完整候选已经被验证。READY 需要 exact transformation 的 accepted geometry evidence。

### 25.2 geometry accepted 和 docking good 不同

几何接受只说明当前 RDKit/UFF/碰撞检查通过。它不说明 docking 分数好，更不说明实验活性好。

### 25.3 docking score 和 quality 不同

`raw_quality_from_mean` 是根据方向统一后的相对平均指标；`quality` 还可能减去 seed 标准差惩罚，并且只有达到 seed 胜率门槛才生成。

### 25.4 candidate history 和 exploration attempts 不同

`candidate_history` 面向候选设计报告；`exploration_attempts` 面向搜索覆盖和审计，包含更多负面结果和批量预筛选。

### 25.5 working memory 和 audit trail 不同

working memory 是给 LLM 的紧凑、可重建摘要；audit trail 是磁盘上完整的决策、观察、结构、命令、日志和 docking 记录。

### 25.6 LLM stop 和 Host safety limit 不同

LLM 可以基于科学证据选择 STOP，但普通搜索模式下 Host 会检查全局覆盖；无论如何，`hard_max_attempts` 都是防止无限运行的硬安全上限。

---

## 26. 总结

`workflow.py` 实现的是一个“LLM 提出假设、Host 执行和审计”的分子改造搜索状态机：

1. 用 `ComplexContext` 和 `ToolRegistry` 获取可靠结构事实；
2. 用 `AgentState` 保留决策、观察、候选、docking 和搜索覆盖；
3. 用 `_repair_decision` 和 `_handle_with_recovery` 应对 LLM 格式错误；
4. 用 `_transformation` 把 LLM 输出转成受控的两类编辑操作；
5. 用 `_validate_design` 实施位点、片段和 exact candidate geometry 证据门；
6. 用 `_global_search_coverage`、site-lock 和 `MARK_UNMODIFIABLE` 管理局部和全局搜索；
7. 用 `design()` 执行候选构造、重复检查、docking 和反馈循环；
8. 用 `_record_docking_result` 计算相对质量、seed 稳定性和历史最佳；
9. 用 working memory 限制 LLM 上下文，用 JSON/SDF/原始 docking 文件保存完整审计；
10. 用 `_restore_run_state` 支持只从当前 run directory 恢复中断任务。

因此，这个文件不是一个单纯的“调用 LLM 再调用 docking”的脚本，而是把化学工具约束、证据完整性、候选身份、局部搜索策略、统计稳定性、错误恢复和可复现实验审计集中到一起的主工作流控制器。

相关文件：

- `molecular_agent/models.py`：状态数据结构和 `compact_view()`；
- `molecular_agent/tools.py`：Host 工具目录和工具执行；
- `molecular_agent/editing.py`：分子 transformation 和几何验证；
- `molecular_agent/fragment_library.py`：片段库检索和 provenance；
- `molecular_agent/structure.py`：复合物、配体和蛋白上下文；
- `molecular_agent/adapters.py`：docking/RBFE adapter；
- `tests/test_workflow.py`：工作流行为测试。

---

*记录文件：`workflow_explanation.md`*
*对应流程图：`workflow.mmd` / `workflow.png`*
*被解释源文件：`molecular_agent/workflow.py`*
*记录原则：以当前源代码实际行为为准。*

---

# 第二部分：按函数分组的深入说明

## 27. 初始化与恢复相关函数

### 27.1 `_resolve_parent_candidate()`

该函数是传给 `ToolRegistry` 的 parent resolver。工具只获得一个 `parent_attempt` 编号，不直接持有任意 RDKit 分子；resolver 根据编号从 `self.parent_candidates` 取出分子并复制一份：

```python
return Chem.Mol(self.parent_candidates[attempt])
```

复制的目的，是避免工具查询或后续编辑意外修改工作流保存的 parent 原对象。`None` 被解释为原始配体 attempt 0；不存在的编号会转换为明确的 `ValueError`。

### 27.2 `_parent_metadata_for()`

该函数返回指定 parent 的元数据副本。元数据包含：

- attempt 编号；
- generation；
- target type/id；
- docking quality；
- canonical SMILES；
- candidate path。

它既用于验证 LLM 的 `parent_attempt`，也用于向 LLM 展示可用 parent。函数返回 `dict()` 副本，避免调用者直接修改内部元数据。

### 27.3 `_restore_run_state()` 的恢复顺序

恢复方法不是简单读取一个总状态文件，而是先读取运行目录中的事件文件，再重建派生状态。具体顺序是：

1. 如果发现 `result.json`，立即拒绝恢复，防止覆盖已完成运行。
2. 按编号读取最后一个 `observation-XX.json`；如果没有 observation，则尝试 `context-final.json`。
3. 检查保存的 task 字符串是否与当前任务一致。
4. 重建每一条 `ToolObservation`。
5. 恢复 decisions、covered evidence、docking history、candidate history、exploration attempts、关闭声明和 site strategy。
6. 根据 observations 和 docking history 重新计算 global/site/fragment/candidate memory。
7. 读取所有 `edit-attempt-XX.json`，并验证 history 中涉及的 attempt 都有对应 report。
8. 只为 stability-eligible 的历史候选恢复 parent candidate。
9. 验证 candidate path 位于当前 run directory 内，并能被 RDKit 读取。
10. 找到历史中保存的 reference docking baseline。

这种设计避免把一个运行中的“派生缓存”当作唯一真相。即使 memory 文件没有保存，仍可由 authoritative observations 和 events 重建。

### 27.4 `_resume_design()`

恢复设计时不会直接重复最后一个候选。它会：

- 使用最后一个 attempt report；
- 构造 `workflow_resume` rejection；
- 告诉 LLM 之前完成的 transformation 不得重复；
- 允许重试上次尚未成功执行的工具调用；
- 让 LLM 决定新 transformation、继续查询或 STOP。

如果恢复后的 decision 是 STOP，则直接调用 `_accepted_output()`；否则进入普通 `design()`，并把 `start_attempt` 设置为旧 history 最大 attempt + 1。

---

## 28. 工作记忆辅助函数的调用关系

工作记忆相关代码可以按下面的依赖关系理解：

```text
_tool_result_summary()
        │
        ├── _compact_observation_result()
        │       └── _llm_safe_value()
        │
        ├── _llm_observation_view()
        └── _update_working_memory()

_compact_transformation()
        │
        ├── _compact_candidate_history()
        ├── _compact_docking_history()
        ├── _compact_exploration_attempts()
        └── _adaptive_target_summaries()

_global_search_coverage()
        │
        ├── _optimization_context()
        ├── _stop_gate_rejection()
        └── _refresh_site_search()
```

### 28.1 `_llm_safe_value()` 不是安全校验器

名字中的 safe 指的是“适合放进 LLM context”，不是化学安全。它只负责：

- 删除大体积字段；
- 限制列表长度；
- 截断长字符串；
- 递归处理嵌套字典和列表。

它不检查 SMILES、价态或 transformation 合法性。化学合法性仍由 `_transformation()`、`_validate_design()` 和 `apply_transformation()` 负责。

### 28.2 `_compact_transformation()`

该函数只保留 LLM 和审计所需的关键 transformation 字段，不把完整的 `library_record`、删除原子集合或巨大结构对象放入普通摘要。它通常保留：

- operation；
- edit atom 或 replacement site；
- fragment ID/SMILES；
- cut bond；
- parent 和 generation；
- 是否替换已有取代基。

### 28.3 `_llm_observation_view()`

它采用“基线 + active target + 最近窗口”的策略：

- 上下文阶段：保留全部已执行 observation 的压缩视图；
- design 阶段：保留基础结构工具、当前 active target 相关 observation 和最近 12 条 observation；
- 按工具签名去重；
- 最多发送 32 条。

因此磁盘上的 observation 数量和单次 LLM 请求中的 observation 数量可以不同。前者服务于完整审计，后者服务于可控上下文。

### 28.4 `_llm_state_view()`

它把 LLM 当前真正需要的状态拼成一个字典，主要字段包括：

- task、round、max rounds；
- covered/missing evidence；
- 最近 decisions；
- 压缩 observations；
- tool rejections；
- active target 和 site search；
- convergence。

`_query_payload()` 再在这个 state 上补充结构输入、工具目录、host readiness、optimization context 和当前阶段 instruction。

---

## 29. 决策记录和工具执行的细节

### 29.1 `_record_decision()`

该函数先把原始有效 decision 放入 `state.decisions`，然后写出：

```text
decision-01.json
decision-02.json
...
```

写入内容包含 decision 和当时的 `state.compact_view()`。随后根据 action 构造一个更小的 progress details：

- QUERY：工具名和参数；
- QUERY_BATCH：queries；
- READY：操作、位点、片段和 hypothesis；
- MARK_UNMODIFIABLE：target、scope、family、reason；
- STOP：reason。

这个小 details 只用于 progress callback，不替代磁盘中的完整 decision。

### 29.2 `_execute_query()` 的 observation 生命周期

一个成功的新工具查询会按如下顺序改变状态：

```text
验证参数
  -> 计算 signature
  -> 检查是否重复
  -> 执行 ToolRegistry
  -> 更新 call_signatures
  -> 更新 evidence
  -> 追加 ToolObservation
  -> 更新 working memory
  -> 处理特殊工具结果
  -> 写 observation-XX.json
  -> emit tool_completed
```

特殊工具处理包括：

- `assess_edit_sites` 完成后更新 site strategy、active target 和 `site-strategy.json`；
- `validate_candidate_geometry` 后写一条 exploration attempt；
- `generate_site_candidate_batch` 后把批量接受/拒绝的 transformation 加入探索账本。

### 29.3 `_record_candidate_batch_exploration()`

批量工具只做 Host 确定性预筛选，不做 docking。这个方法把批量结果转成 exploration records：

- candidates 记为 `batch_geometry_accepted`；
- rejected 记为 `geometry_rejected`；
- 如果结果没有完整 transformation，则根据 target type、fragment ID/SMILES 自动补上最小记录。

后续 `_geometry_feasible_not_docked()` 会根据这些记录找出“已经几何可行、但还没有进入 docking”的候选，反馈给 LLM。它们不会被错误地算作 docking hit。

---

## 30. READY 处理的完整路径

一个 READY 从 LLM 返回到正式进入 design，通常经过以下多次检查：

```text
LLM JSON
  │
  ├── _unwrap_decision()
  ├── _repair_decision()
  ├── _handle_decision() 记录 decision
  │
  ├── collect_context 中的 _validate_design()
  │       ├── 字段检查
  │       ├── _transformation()
  │       ├── site-lock 检查
  │       ├── parent/atom/operation 检查
  │       ├── 局部环境证据检查
  │       ├── 片段知识证据检查
  │       └── exact geometry 证据检查
  │
  └── design 每个 attempt 再次 _validate_design()
          └── apply_transformation() 执行最终确定性构造
```

上下文阶段的 READY 主要用于结束 context collection；design 阶段每轮开始还会再验证一次。这种重复校验是有意的，因为 state 可能在上一个 attempt 后发生变化，特别是 active target、parent 和 history。

### 30.1 `_auto_complete_ready_evidence()`

对于 READY rejection 中列出的 `recommended_queries`，方法会：

1. 检查 query 是否是合法字典；
2. 跳过已执行的相同 query；
3. 如果需要 LLM review，则跳过 `validate_candidate_geometry` 的自动执行；
4. 执行其余可以确定性补齐的 query；
5. 将自动执行失败写入 `tool_rejections`。

如果自动补证后没有需要 LLM 主动判断的知识缺口，调用方会再次运行 `_validate_design()`。因此“自动补证”不是绕过证据门，而是把可机械执行的缺口补上后重新过门。

---

## 31. exploration ledger、candidate history、docking history 三者的关系

可以把一次候选尝试表示成三张不同粒度的表：

```text
exploration_attempts
    记录：想探索什么、是否几何接受、是否重复、是否关闭

candidate_history
    记录：一次 design attempt 的候选文件、验证报告和 docking 摘要

docking_history
    记录：进入 docking 后的相对指标、seed 稳定性和历史最佳
```

### 31.1 可能只有 exploration，没有 candidate history

例如 `generate_site_candidate_batch` 产生了一个几何可行候选，但 LLM 暂时没有选择它进入正式 design。此时它可以存在于 exploration ledger 和 `geometry_feasible_not_docked`，但不应出现在 docking history。

### 31.2 可能有 candidate history，但没有 docking history

候选可能因以下原因无法进入 docking：

- 构造失败；
- 价态或 sanitization 失败；
- 蛋白碰撞严重；
- canonical SMILES 重复。

这些 attempt 仍会写 `edit-attempt-XX.json` 和 candidate history，用于防止重复和解释失败原因。

### 31.3 docking history 的前提

只有 candidate 通过确定性几何和结构门、且 docking adapter 返回可记录的结果，才会进入 docking history。docking 不完整时会保留 candidate report，但不会产生可比较的 quality。

---

## 32. 设计循环中的几条关键分支

### 分支 A：几何检查失败

```text
apply_transformation 失败
  -> report.validation.status = rejected
  -> docking.status = not_run_geometry_rejected
  -> exploration = geometry_rejected
  -> candidate_history 追加失败记录
  -> 保存 edit-attempt-XX.json
  -> LLM retry
```

### 分支 B：canonical structure 重复

```text
candidate canonical SMILES 已存在
  -> 不启动 docking
  -> 标记 duplicate_candidate_structure
  -> 保存首次出现 attempt
  -> candidate_history 追加记录
  -> LLM retry
```

### 分支 C：docking 不完整

```text
docking.status != complete
  -> 保存 docking report
  -> 不产生正常 docking quality
  -> 返回历史最佳或当前可接受结果
  -> stopping_reason = docking_not_complete
```

### 分支 D：docking 完整但没有可比较主指标

```text
raw_quality_from_mean is None
  -> 不把结果当成可排序质量
  -> 返回已有结果
  -> stopping_reason = no_candidate_revision_without_comparable_metric
```

### 分支 E：LLM 返回相同 transformation

```text
当前候选已经 docking
  -> LLM 再次返回相同 transformation
  -> candidate_not_revised rejection
  -> 再要求一次新假设
  -> 再次相同则抛出 RuntimeError
```

### 分支 F：达到 hard safety limit

达到 `hard_max_attempts` 后：

- 如果已有完成的 docking，返回历史最佳；
- 如果从未有完成 docking，返回 `no_candidate_accepted`；
- 不把安全上限称作科学收敛。

---

## 33. site-lock 的实际状态机

site-lock 开启并且 site strategy 已建立后，位点状态可以抽象为：

```text
strategy ready
     │
     ▼
active target
     │
     ├── QUERY / batch / READY：继续积累局部证据
     │
     ├── LLM STOP：通常被 global search gate 拒绝
     │
     ├── MARK_UNMODIFIABLE 当前 target
     │       └── target -> closed
     │
     └── READY 另一个 target
             └── site_lock_violation
```

当当前 target 关闭后，`_refresh_site_search()` 选择下一个 priority 最小的 pending target。只有所有策略位点都 closed，site-lock 分支下的全局搜索才会 complete。

注意：`local_patience` 只影响 `patience_reached` 字段和 LLM 看到的提示。当前代码不会因为 patience reached 自动关闭 site，也不会因为 patience reached 自动切换 target。

---

## 34. parent 优化的实际限制

当 LLM 使用 `parent_attempt` 时，代码同时约束三个方面：

1. parent 必须存在于 `parent_candidates`；
2. parent 通常来自达到 seed 稳定性门槛的候选；
3. site-lock 模式下，parent target 和 child target 必须保持一致。

这意味着 parent 优化不是任意多代遗传搜索，而是受 docking 稳定性和 active target 限制的局部延伸。一个 geometry accepted 但 seed 胜率不足的候选可以被记录，但不会自动成为后续 parent。

`parent_metadata` 还会记录 parent 的 generation 和 candidate path，以便恢复时从 SDF 重建分子，并向 LLM 说明可用 parent 的来源。

---

## 35. 代码中的“确定性”和“LLM 决策”边界

| 事项 | 负责方 |
|---|---|
| 决定下一步查询什么 | LLM |
| 返回具体化学 transformation 假设 | LLM |
| 解析任务和复合物 | Host / `ComplexContext` |
| 枚举合法 replacement site | Host / `ToolRegistry` |
| 检查 fragment ID 与 SMILES 一致性 | Host |
| 检查 operation 权限 | Host |
| 构造 RDKit candidate | Host / `editing.py` |
| 检查价态、sanitization 和碰撞 | Host |
| 判断工具调用是否重复 | Host |
| 判断 candidate canonical structure 是否重复 | Host |
| 执行 docking | Host / adapter |
| 计算相对指标和 seed 统计 | Host |
| 根据证据决定继续哪个化学假设 | LLM，但受 Host rejection 和 site-lock 约束 |
| 是否关闭位点 | LLM 提交 `MARK_UNMODIFIABLE`，Host 校验格式和目标 |
| 是否超过硬安全上限 | Host |

这种边界避免把化学事实、文件操作和外部程序执行交给不可预测的 LLM 输出。

---

## 36. 阅读和调试该文件的推荐顺序

如果需要继续修改 `workflow.py`，建议按以下顺序阅读和定位：

1. 先看 `run()`，确认当前处于新运行还是 resume。
2. 看 `collect_context()`，理解第一个 READY 如何产生。
3. 看 `_handle_decision()`、`_execute_query()` 和 `_repair_decision()`，理解 LLM action 如何进入 Host。
4. 看 `_transformation()` 和 `_validate_design()`，理解 READY 的 schema 和证据门。
5. 看 `design()`，逐个检查 candidate construction、duplicate、docking 和 retry 分支。
6. 看 `_record_docking_result()`，确认质量和历史最佳如何更新。
7. 看 `_global_search_coverage()`、`_refresh_site_search()` 和 `_stop_gate_rejection()`，确认 STOP 为什么被允许或拒绝。
8. 最后看 `_restore_run_state()`，确认新状态字段是否支持恢复。

新增状态字段时，至少要同时检查：

- `AgentState` 是否定义；
- `compact_view()` 是否持久化；
- restore 是否读取；
- working memory 是否可以重建；
- LLM view 是否需要暴露；
- 相关 JSON report 是否需要记录。

---

## 37. 最终理解框架

阅读这个文件时，可以把每次 LLM 输出分成三道门：

```text
第一道门：决策格式门
    action 是否合法？QUERY 是否有 tool/arguments？

第二道门：证据和搜索策略门
    READY 是否有正确位点、片段、环境、空间和 exact geometry 证据？
    是否违反 site-lock、重复规则或全局搜索要求？

第三道门：化学和计算门
    RDKit 能否构造？价态/电荷/碰撞是否通过？
    docking 是否完成？主指标是否可比较？
```

只有三道门逐步通过，候选才会进入可排序的 docking history。最终输出再从 history 中选择 `best_attempt`，而不是简单取最后一个 attempt。

这也是 `workflow.py` 最核心的设计：**模型可以提出很多假设，但每个假设都必须经过格式、证据、结构和计算四个层面的显式记录与校验。**