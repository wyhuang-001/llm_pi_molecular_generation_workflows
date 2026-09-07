# LLM 批量多位点分子优化

## 改动目标

将原来的“单个位点、单个候选、反复查询”流程改为受预算约束的多位点批量优化，减少 LLM 请求、重复化学信息和无效 docking，并保留必要的工具查询能力。

## 已完成改动

- 默认真实工作流启用 `batch_optimization`，单独设置批次数、候选数和 LLM 决策数，不设置总运行时间上限。
- 新增启动阶段 `design dossier`：宿主一次性整理配体、口袋、相互作用、全部合法编辑位点、空间余量、片段库统计和每个位点的初始片段面板。
- 完整片段库只在宿主侧加载；给 LLM 的片段记录包含 ID、SMILES、操作类型、尺寸、标签、电荷、MW、LogP、HBD/HBA、TPSA、环和可旋转键等紧凑化学信息。
- 新增多样化片段面板选择，优先覆盖不同尺寸和化学家族，避免把两万余条片段完整塞入每次请求。
- 新增 `PLAN_BATCH`：LLM 可一次为多个位点选择多个片段。每个候选必须说明位点证据、结构改变、预期作用、风险和成功标准。
- Portfolio 请求使用单独的精简 system prompt，不再重复发送旧的长篇 site-lock/Coverage 规则。
- 新增宿主批量几何筛选：统一完成片段合法性、结构构建、价态/电荷、碰撞和批内结构去重；通过者自动进入 docking。
- 新增位点状态面板，支持 `screening`、`promoted`、`active`、`deprioritized` 和 `discarded`。优化模式不再要求所有位点显式关闭。
- 保留 LLM 工具调用，但限制为片段面板刷新、parent-specific 环境、空间、片段三维形状和相互作用等决策相关不确定性；不允许重复查询 dossier 已提供的基础片段性质。
- 新增 `get_fragment_panel`、`get_design_dossier` 和 `screen_candidate_batch` 高层工具；每个补充查询必须说明 `why_needed` 和 `decision_impact`。
- Docking 支持 seed 子集：批量候选先使用 screening seed；LLM 可用 `CONFIRM` 选择少量候选执行完整多 seed confirmation。
- `STOP` 可依据候选质量、趋势、剩余机会和预算直接结束，不再依赖全位点覆盖门；连续若干批次没有新 best 时也会按配置停止。
- LLM 请求默认超时降为 120 秒、最多重试 2 次、常规输出上限降为 4096 token，避免供应商故障长期占用总预算。
- 原有逐候选、自适应位点锁定和 Coverage 逻辑继续保留，用于兼容旧任务和专项测试。

## 默认预算

当前 `input/task.json` 设置为：

- 每批最多 8 个候选；
- 最多 3 批、12 个 screening 候选；
- 最多确认 2 个 finalist；
- screening 使用 seed 17，confirmation 使用 17、29、43；
- 最多 8 次 LLM 决策；
- 不设置内部 wall-clock 总时间上限；运行由批次、候选、LLM 决策和停滞条件控制。

这些参数是运行安全预算，不代表科学收敛阈值。当前不设置总运行时间上限。

## 真实验证

已使用 `current / gpt-5.6-sol` 和真实 GNINA 完成一轮 portfolio 测试：8 个候选进入 screening，2 个 finalist 完成 3-seed confirmation，最终状态为 `candidate_accepted`。成功的恢复段约用时 14 分钟，实际时间仍取决于模型服务和候选数量。

## 主要文件

- `molecular_agent/fragment_library.py`：完整库摘要和多样化片段面板。
- `molecular_agent/tools.py`：design dossier、片段面板和批量几何筛选工具。
- `molecular_agent/workflow.py`：`PLAN_BATCH`、`CONFIRM`、多位点状态和预算循环。
- `molecular_agent/adapters.py`：screening/confirmation seed 子集。
- `molecular_agent/llm.py`：批量优化动作和 LLM 边界说明。
- `input/task.json`：批量优化默认预算。
