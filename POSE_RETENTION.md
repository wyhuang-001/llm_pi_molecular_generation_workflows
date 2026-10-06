# 共晶校准 → 冻结 docking reference → 候选 pose 门控

## 当前实现

入口仍是单编辑工作流，协议放在 `input/task.single_edit.json` 的 `pose_retention` 中。未启用该设置的旧流程保持原 rank-1 行为。

1. **先制备、先校准，再调用浏览器/LLM。** 原配体保存为 `reference-ligand.sdf`；RDKit 仅添加配体氢，不移动共晶重原子。原始 PDB/CIF 及 selector 有 SHA256 身份记录。
2. GNINA 对原配体运行所有配置的 seeds，读取每个 seed 的前 `top_n` 个输出，不只读取 rank 1。
3. 每个 pose 必须与输入分子具有相同化学身份和立体化学。枚举完整图的合法对称映射，在**受体坐标系**中计算固定核心的直接 RMSD 和最大原子位移，**不进行配体叠合**。
4. 原配体 poses 先满足对共晶的绝对几何门槛和硬锚点。然后寻找至少 `minimum_passing_seeds` 个 seed 支持的 complete-link family（两两 RMSD 都合格，不允许 A–B–C 链式聚类）。校准失败时，不以“最接近的坏 pose”凑数，也不启动候选设计。
5. 冻结 family 和其 medoid 主参考。候选的主要几何门控对照**这个固定的 redocked 主参考**，不是直接约束原始坐标，也不随 best-so-far 更新。参考 family 的其他成员保留为审计集合，不扩大候选门控区域。
6. **几何参考不等于评分 baseline。** 原配体每个有效 seed 的 baseline 是同一模式下、通过门控的 poses 中主评分最好的 Evaluation Pose，而非默认 rank 1、也非几何 medoid 的分数。
7. 候选自由 docking。所有 top-N poses 经过身份 → 核心 RMSD/最大位移 → 少数锚点门控后，各 seed 才按 `primary_metric` 选择 Evaluation Pose。候选相对共晶的 RMSD 同时记录，作为漂移诊断，不冒充坐标约束。
8. 对所选候选 Evaluation Poses 再做跨 seed complete-link family 检查。至少两个 seed 同属一个 family，且这些 seed 有对应有效 baseline，才能进入正式评分。无合格多数用 `no_eligible_pose`，绝不回退 rank 1。工具/映射/评分无法评价是 `evaluation_failed`，不是“没有相互作用”。
9. 分数、PLIP 报告、跨 seed 几何和最终导出全部绑定到同一个 Evaluation Pose。最终 `candidate_path` 指向选定 docking pose，另有 `constructed_candidate_path` 和 `evaluation_pose_paths`。

## 固定核心与化学映射

本任务的初始比较核心为原配体索引 `[0,1,13,14,16,17,18,19,20,21]`：嘌呤环九个原子 + N2 氢键供体。它是**比较的原子集合，不是固定坐标集合**，也不等价于整个 24 重原子配体。

- 原子在 Host 化学图上带 `_reference_atom_index`，图编辑复制时保留；新原子没有旧身份。
- 写 SDF 时保存 `reference_atom_indices`。对接后通过完整输入分子到输出 pose 的化学同构找回原子顺序，并处理合法对称性。
- 核心完整保留；片段替换的合法位点筛选自动保护该核心。不会每候选临时选一个更小、更有利的 MCS。
- 保留原配体的重原子比例独立报告，不使用与现有删片段规则矛盾的固定 80% 阈值。
- 对称映射数量超过显式上限时失败，不把截断枚举当作精确最小 RMSD。
- 离线分析旧增长型候选时，仅允许整个原配体图的严格子图映射；旧删除型候选没有 provenance 则拒绝，不猜 MCS。

## 硬锚点与 PLIP

默认只有一条待校准锚点：原配体 `atom 21 / N2` 作为供体 → `LEU:A:83:O` 主链羰基受体。

默认 `method=explicit_atom_geometry`：

- Host 用 RDKit 验证配体供体化学、读取 pose 的显式供体氢；验证指定蛋白 O/C 的唯一身份及羰基几何。
- D–A ≤ 3.5 Å、H–A ≤ 2.7 Å、D–H–A ≥ 100°。缺氢/原子不明确是 unavailable，不能自动通过。
- 目前这种直接几何后端只支持**配体供体 → 明确的主链羰基 O**，不假装普遍解决蛋白质子化/蛋白供体氢问题。
- 锚点必须被制备后的共晶支持，并由多数 reference seeds 恢复，才会成为有效冻结协议的一部分。

也支持 `method=plip`：严格匹配 ligand/protein 原子、供受体方向、距离、角度；PLIP 失败会阻断该硬门控。不能只按 `hydrogen_bond|LEU:A:83` 判断。

本例选择直接几何是有实际依据的：PLIP 在无新增蛋白氢模式下可能把该供体的作用分配给竞争原子，未必报告指定羰基 O；因此不能用它筛选后的残基标签替代独立的原子级锚点。所有外围/新增 PLIP 作用仍是**软反馈**。

PLIP complex 现在支持指定 rank、保留已制备受体的 TPO，并连续重编号原子。去掉了会让 PLIP `--nofix` 内部映射多加一个编号的 `TER`。完整序号 provenance 独立保存在 `atom-mapping.json`；坐标仅发生 PDB 的 0.001 Å 输出舍入。

## 受体制备及限制

- 原先 `ATOM`-only 会误删 A/C 链的 TPO160，共 22 个原子。现在默认保留经白名单审查的 TPO/SEP/PTR/MSE，碰撞检查、GNINA 和 PLIP 使用一致的受体原子集合。
- 共晶配体及其他自由配体、水/离子默认排除；其他 HETATM 必须显式审查/列入名单。
- 保留 blank/A altloc，过滤并保留受体内部 CONECT；保存 `receptor-protein-only.preparation.json`。文件名沿用旧接口，但内容不再是盲目的 ATOM-only。
- **尚未实现通用 pH/互变异构/受体加氢优化。** GNINA 使用其内部准备；PLIP 使用 `--nohydro --nofix`。校准成功也不等于完成了全面的质子化验证。

## 多 seed 评分口径

总 seed、几何/锚点通过 seed、family seed、可配对 seed、改善 seed 分别记录。失败/outlier seed 不从分母中消失。

当前质量分（不是自由能）为：

`有利方向的 paired Δ均值 − seed_stddev_penalty × Δ标准差 − missing_seed_penalty × (1 − paired_seed_fraction)`

只有整体 pose 门控通过且改善 seed/**全部配置 seeds** 满足原有胜率政策时，才有 best-so-far 资格。三个 seed 的多数条件使用整数 `minimum_passing_seeds=2`，避免 `0.6667 > 2/3` 的误差。

`no_eligible_pose` 在单编辑循环中作为失败证据反馈给 LLM，可以换改造或 STOP；不可评分的工具失败会停止本轮流程。所有候选均未合格时，最终输出 `no_candidate_accepted`，不会选最后一个失败候选充数。

## 冻结与复用

`docking-reference-baseline/calibration.json` 包含协议、源码/软件身份、输入哈希、共晶锚点校验、各 seed/各 rank 的门控明细、geometry reference/family、各 seed 评分 baseline 和 artifact 哈希。

- 同一 run 内只校准一次；后续候选从磁盘验证并复用，不重新 redock 原配体。
- 输入、协议、实现、已冻结 artifact 不匹配则拒绝复用，要求新 run 目录。
- 旧的未版本化 rank-1 baseline 不能自动升级为可信 reference。
- 失败校准也审计保存；改变制备或搜索参数必须开新目录，不自动放宽门槛。

## 工具与测试

只做参考校准（无浏览器/LLM/候选设计）：

```bash
mamba run -n molecular-agent-docking python scripts/calibrate_reference.py --task input/task.single_edit.json --config config.single_edit.json --run-dir "runs/reference-calibration-$(date +%Y%m%d-%H%M%S)"
```

只读分析已有运行的全部 top-N（旧 receptor + 旧 scores，仅诊断，不能当作修复 TPO 后的生产 baseline）：

```bash
mamba run -n molecular-agent-docking python scripts/analyze_pose_recovery.py --run-dir runs/single-edit-20260927-132739 --output-dir "runs/pose-recovery-$(date +%Y%m%d-%H%M%S)" --candidates 28 23 19
```

新测试位于 `tests/test_pose_retention.py`，覆盖不叠合 RMSD、对称/重编号、化学与手性错误、删除后的原子身份、非链式 family、不同 rank 的几何/评分参考、冻结缓存篡改、2/3 分母、无合格 pose 不回退、原子级锚点、TPO、校准失败前不调用 LLM，以及最终不接受失败候选。

## 实测状态：不要把实现完成误解为校准已经成功

- 旧运行离线审计：`runs/pose-retention-offline-20260927/summary.json`。60 个原配体 poses 中，只有 seed17/rank1 通过本协议；seed29 无满足核心几何的 pose，seed43/rank1 的核心 RMSD 约 1.981 Å，但指定锚点角度约 90.43°，不合格。因此未建立参考，候选 28/23/19 没有被强行重新排名。
- 修复 TPO 后的真实 GNINA 三 seed 校准：`runs/pose-retention-calibration-20260927/docking-reference-baseline/calibration.json`。默认搜索下仍失败：各 seed 最小固定核心 RMSD 约 2.556 / 3.049 / 2.383 Å，均未达到 2 Å。
- 这是协议正确阻止了错误模式继续被“优化”，不是已经得到更可靠的活性赢家。下一步应检查制备/搜索预算与合适的局部搜索协议，再在新目录重新校准；不得为了放行旧赢家无条件放宽阈值。
