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
