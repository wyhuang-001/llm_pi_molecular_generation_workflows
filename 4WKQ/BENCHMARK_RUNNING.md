# 4WKQ 闭集侧链 benchmark：运行与评估

## 当前已冻结的内容

- `design/fragments.json`：78 条匿名、唯一、可组装的侧链，只有 `fragment_id`、attachment `smiles`、`allowed_operations`。
- `design/pool-manifest.json`：公共库哈希、结构/CCD 身份、定向切口、78 条完整可见、最多20个唯一提议、最多60次设计层请求。
- `task.benchmark.json`：只允许原配体 C6 单编辑；禁止 parent、其他位点、库外 SMILES。化学等价的库内 SMILES 会归一化为同一个 ID。
- 库的构成为26条可达文献侧链、26条近邻背景、26条物性匹配的多样性背景。原始 gefitinib 不是可选候选。
- 公共库 SHA256：`42326333c86bcb625e96984c1ffafe728dc9f44f6e3602099b7858a0af07f05f`。

原始库缺少部分高极性侧链的合适背景，因此在结构规则下增加了一步 CH2 插入/删除、甲基添加、环 O→CH2 替换。背景候选池共599条，各文献侧链有5–147条满足固定物性/类别条件的背景选择；没有读取活性来筛选，没有静默放宽匹配阈值。所有78个分子都通过了精确重组和真实 Host 构建的图身份检查。

生成脚本是 `scripts/build_4wkq_pool.py`；会拒绝覆盖已有冻结目录。修改规则或目录顺序应显式创建新版本，不能在正式运行后调整背景以改善结果。

## 一个必要的几何处理修正

构建审计发现：原先的 `seed=17` 配体单独 UFF 构象把78个候选中的61个判为受体碰撞。把原始 gefitinib 拆开再接回去，5个构象种子也都产生碰撞；但共晶本身的最大重原子 VDW overlap 只有约0.122 Å。这说明“一个初始构象有碰撞”不能直接证明该化学结构无可行结合姿势。

因此本 benchmark 的规则是：

1. 化学构建、身份、价态错误仍然阻断。
2. **初始受体碰撞保留为诊断，允许 GNINA 搜索；不删除这些库成员。** 它不表示初始构象已通过碰撞检查。
3. **最终 docked Evaluation Pose 增加全配体重原子–受体重原子 VDW overlap ≤0.55 Å 的硬门控**，再结合原有核心 RMSD、最大位移、原子级氢键锚点和多数 seed family。
4. 不对配体重新叠合、不固定 docking 核心坐标、不回退到 rank1。
5. 氢保留供 GNINA/PLIP 使用，但重原子碰撞检查不把正常 D–H…A 距离误判成重原子重叠。

该行为仅在显式设置 `candidate_construction.initial_receptor_clash_policy=defer_to_docking` 且有最终 steric gate 的闭集任务启用；原来的非 benchmark 流程仍保留其既有初始碰撞拒绝行为。

`benchmark-private/construction-audit-v2/construction-audit.json` 保存全部78条的构建检查：78/78正确构建，其中17条初始无碰撞、61条初始碰撞交给 docking。**这些离线结果不会提前发送给 designer。**

## 设计侧能看到什么

每次请求都得到完整78条紧凑目录，不经过6条片段面板或启发式尺寸筛选。属性全部以同一 RDKit attachment-fragment 方法计算。

能看到：结构图/结构证据、唯一合法切口、匿名片段 SMILES 和统一性质、自己已经尝试过的候选的 Host/GNINA/PLIP 反馈。

看不到：文献成员标记、原文编号、ChEMBL ID、实验 IC50、强弱标签、背景匹配对象、SAR 总结。此 benchmark 的外部研究明确关闭，不加载之前保存的 web/research-memory。

每个 run 保存：

- `closed-pool-run.json`：输入/实现/对接协议身份。
- `closed-pool-library.json`：公共库快照。
- `closed-pool-request-NNN.json`：实际设计层请求、全目录可见 ID 列表和 payload 哈希。
- `closed-pool-response-NNN.json`：原始结构化响应、STOP 或失败；修复请求也计入并审计。
- 原来的构建 SDF、GNINA/PLIP 文件、pose 门控结果和最终 `result.json`。

注意：这里的“设计请求数”是 `complete_json` 层计数，不等于底层 HTTP 重试次数或 token usage；本次没有实现供应商 usage 账单审计。随机/无LLM基线也使用相同的请求审计接口，但没有调用任何模型。

## LLM 正式运行

使用已有本地 `config.single_edit.json`，API key 仍直接读取 JSON，不需要 export。使用新的 run 目录，并保持同一 GNINA 配置用于所有等预算比较。

```bash
mamba run -n molecular-agent-docking python -m molecular_agent.cli --task 4WKQ/task.benchmark.json --config config.single_edit.json --run-dir "runs/4wkq-llm-$(date +%Y%m%d-%H%M%S)"
```

首次运行先校准原配体。如果校准失败，设计不开始。模型可提前 STOP；评估器同时报告实际提议数和预设20个最大预算，不假装提前停止后仍完成了20个提议。

### 无 LLM 的等预算基线

随机、不放回提议，随后按与 LLM 相同的 Host/GNINA/pose/质量分流程处理：

```bash
mamba run -n molecular-agent-docking python scripts/run_4wkq_baseline.py --strategy random --seed 17 --task 4WKQ/task.benchmark.json --config config.single_edit.json --run-dir "runs/4wkq-random-$(date +%Y%m%d-%H%M%S)"
```

无 LLM 评分驱动策略：前3条均匀探索，此后每第4条探索，其余选当前最佳 pose-passing 平均 docking 改善候选的未尝试近邻：

```bash
mamba run -n molecular-agent-docking python scripts/run_4wkq_baseline.py --strategy score-guided --seed 17 --task 4WKQ/task.benchmark.json --config config.single_edit.json --run-dir "runs/4wkq-score-guided-$(date +%Y%m%d-%H%M%S)"
```

两者都不读实验标签，也不先对全库78个分子 docking 再选择20个。它们不是全库穷举的伪等预算对照。baseline runner 暂不支持恢复；故障后需要新的 run 目录。

## 独立评估器

以下全部是 evaluator-only 输入，**不得提供给 designer**：

- `benchmark-private/pool-v1/membership.json`：来源分组和完整分子映射。
- `benchmark-private/activity-source-v1/activities.json`：重新获取的 ChEMBL 原始记录。
- `benchmark-private/evaluator-v1/`：冻结标签、评价政策、哈希 seal。

核验结果：28条记录均为同一 assay、同一 EGFR target、精确 `=` 的 IC50、nM；与旧整理文件的28个 activity IDs 和数值一致。一个 ChEMBL potential-duplicate 标志保留在私有标签中，没有当成独立重复实验。本轮未逐表独立重录原论文全文，不能把数据库一致性冒充全文复核。

评价阈值在模型运行前冻结：同 assay 相对原始配体 IC50 改善至少3倍记为预定义命中；这不表示经过重复实验验证的统计显著改善。52个背景的实验标签为 unknown，不填0、不当负例。

仅对已经结束且有 `result.json` 的运行评价，输出必须是新文件，位于设计 run 和公共输入之外：

```bash
mamba run -n molecular-agent-docking python scripts/evaluate_4wkq_benchmark.py evaluate --run-dir runs/YOUR-COMPLETED-RUN --output "4WKQ/benchmark-private/evaluations/evaluation-$(date +%Y%m%d-%H%M%S).json"
```

结果区分：提议命中、构建成功、初始碰撞/延后、对接准入、最终 pose 通过、最终 top1/3/5，以及首次提出已知高活性分子的步数。对有实验标签且 pose 通过的观察子集计算排名相关性，不声称这是全系列无偏相关性。

另外生成1万次随机不放回抽样的**提议层**参考，同时报告实际预算和最大20预算。它不是随机 docking/pose 模拟，也不替代真实等预算基线；候选的“已知高活性命中”不意味着未知背景都不活跃。

评估器禁止用已改动的标签、政策、库快照或截断目录悄悄评价。改变冻结评估器实现也需要显式建立新版本，不覆盖已有答案键。

## 隔离边界

当前 designer 是独立 API 请求，只得到上面列出的公共 payload，没有本地文件工具，也不继承开发会话。`benchmark-private/` 已在 `.gitignore` 中，但忽略规则和目录名不是安全隔离。

如果之后使用能读文件的 coding agent 作为 designer，必须将其放在只挂载 PDB、CCD、公共目录与公开任务配置的环境；不要挂载整个 `4WKQ/` 或 `research/`，也不要把实验评价文件放进其工作区。

Geﬁtinib 和这些结构可能存在于预训练语料。因此应把结果称为**隐藏标签的闭集回顾性侧链选择/重发现**，不声称是完全未知结构的 de novo 发现。

## 已实际运行的验证

- 全库图重组与 Host 构建：78/78。
- 新 benchmark 协议下的真实随机基线：`runs/4wkq-random-baseline-v1/`，20个唯一候选、20次无LLM决策、原配体3个 seeds 校准成功，20个候选均完成多数 seed pose 门控的对接评价。
- 上述随机运行已通过独立离线评估，结果只保存在私有评估目录，没有回流到设计请求；这里不展示高活性成员或答案键。
- score-guided runner 已实现并通过单元测试，尚未实际跑完整轮。
- **尚未启动付费 LLM benchmark。** 一次随机运行只验证链路和提供一个基线实例，不是对 LLM 优势或新活性的证明。正式比较还应重复多个随机种子/模型运行。
