# 闭环迭代记录

目标：在一次真实 benchmark run 中，让候选改造结果不断改善并收敛。

## iter1（失败，已修复）
- run: `runs/iter1-20261007-012941`（deepseek-flash, thinking=enabled）
- 现象：第 2 次 LLM 调用即 `workflow_failed`，`assistant_output_truncated`
  - deepseek-flash 开 thinking 后推理无界：单次决策推理 16384 token 仍未结束
    （`finish_reason=length`，`content` 为空）
- 修复（commit `b6c81e5`）：改用 `deepseek-v4-pro`（推理有界，复杂设计任务仅 ~4.7k 推理 token），
  并把 `max_output_tokens`/`repair_max_output_tokens` 提到 32768。

## iter2（达标）
- run: `runs/iter2-20261007-014118`（deepseek-v4-pro, thinking=enabled）
- 结果：
  - 26 个完成 docking 的候选（28 attempts，2 个几何拒绝）
  - `stopping_reason = no_promising_edit`（两级关闭枚举理由生效，非过早 llm_stop）
  - best_attempt=20，`best_quality=0.590`，`best_delta=-0.604`，seed_win_fraction=1.0
  - 最佳候选：`atom:addition` @ site 1 + fragment `BG-small-015`（NC1([*:1])CC1）
  - tool_rejections 6 次，全部为可恢复的无效决策（重复/无 fragment_id），无重复死循环
  - LLM 请求 35 次 / 26 候选 ≈ 1.35（上一轮 flash 是 4 候选 / 16 请求）
- best_quality 单调上升：0.116 → 0.277 → 0.283 → 0.328 → 0.43 → 0.462 → 0.59
- 收敛：attempt 20 之后 8 个候选均未超过 best，模型以 `no_promising_edit` 停止
- 对比上一轮（deepseek-flash）：候选 4→26，best_delta -0.29→-0.60，结束原因从重复死循环变为正常收敛

## iter3（苯胺可编辑 + 单元素→多原子替换）
- 文献查证（离线 ChEMBL 库 120 化合物 + 在线 ChEMBL API）：
  - 吉非替尼类似物系列 97.5%（117/120）只改 C6 侧链，苯胺高度保守；
  - 药物层面苯胺是被改的：吉非替尼(F+Cl) → 厄洛替尼(3-乙炔基) → 拉帕替尼/凡德他尼。
- 设计落地：苯胺环=核心，Cl/F 与 H 可替换，且支持「单元素→多原子」替换。
- 代码/构建改动（commit 见 git）：
  1. `closed_pool._normalize_multisite`：修 element_swap 被误当 fragment replacement 的 bug（atom/ring:replacement 走 element 路径）。
  2. `workflow`：direct 模式接入 MARK_UNMODIFIABLE（位点关闭 + completion_reason）。
  3. `build_4wkq_multisite_library.py`：派生位点表新增 C-Cl/C-F 切断位点（cut-Cl/cut-F），片段库新增 AS-ethynyl 等苯胺取代片段。
  4. 重建 fragments/sites/reachability/manifest，重新 freeze evaluator-multisite-v2。
- 验证：Cl→乙炔基、Cl→甲基、F→H(deletion) 均能构建正确产物；全量 215 测试通过。
- 注意：新位点/片段不改变文献覆盖（仍是 26 个 C6 化合物），苯胺编辑为无标签背景探索。

## iter3 分析 + iter4 修复
- iter3 真实 run（`runs/4wkq-llm-20261007-142819`，deepseek-v4-pro）：20 候选/21 请求、0 拒绝、best_delta=-0.35、
  no_promising_edit 收尾。干净但 best 不如 iter2（-0.60），且模型试了苯胺 Cl→Br 却 pose retention 失败。
- 根因：`pose_retention.core_atom_indices` 原为 [10..30]，把 Cl(26)/F(28) 也算进姿势核心，编辑它们被判 non-native。
- 另外发现：冻结表 sites-v2.json 本就有 cut-010(Cl)/cut-011(F)（aniline-halogen），上一轮加的 cut-Cl/cut-F 是重复的，已回退。
- 修复：
  1. task 的 pose core 与 protected core 排除 Cl(26)/F(28)（19 原子，苯胺环仍在核心里）；
  2. 回退 build 脚本里重复的 cut-Cl/cut-F；
  3. 重建位点表(11 cuts)/片段库(214, 含 AS-ethynyl)/reachability，重新 freeze evaluator-multisite-v3。
- 全量 215 测试通过。待新一轮真实 run 验证 Cl→Br/Cl→乙炔基 能过 pose 门。

## iter4 分析 + 位点级自动关闭
- iter4 真实 run（`runs/4wkq-llm-20261007-153604`）：30/30 候选、0 拒绝、best_delta=-0.564、best_quality=0.545、
  hard_safety_limit 收尾；Cl→Br（cut-010）pose 门通过（delta +0.096 更差，模型正确放弃）。
- 新增：direct/closed-pool 模式的位点级门控——`_auto_close_exhausted_site` 在某位点连续
  `max_consecutive_no_improvement`（默认5）个非改善 docking 候选后自动 MARK_UNMODIFIABLE（no_promising_edit）。
- task 增加显式 `termination_policy`；重新 freeze evaluator-multisite-v4。
- 全量 216 测试通过。
