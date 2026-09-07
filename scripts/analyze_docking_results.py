#!/usr/bin/env python3
"""Summarize all docking attempts in a molecular-agent run directory."""
from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

METRICS = ("minimizedAffinity", "CNNscore", "CNNaffinity", "CNN_VS")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "是" if value else "否"
    spec = f"{('+' if signed else '')}.{digits}f"
    return format(float(value), spec)


def edit_path(run_dir: Path, attempt: int) -> Path:
    for name in (f"edit-attempt-{attempt:02d}.json", f"edit-attempt-{attempt}.json"):
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No edit-attempt file for attempt {attempt}")


def infer_family(fragment: str, tags: list[str]) -> str:
    lower = fragment.lower()
    if "heteroaryl" in tags:
        return "杂芳环"
    if "nitrile" in tags or "#n" in lower:
        return "腈/极性"
    if fragment in {"[*:1]Cl", "[*:1]F"}:
        return "卤素"
    if "polar" in tags or any(token in fragment for token in ("N", "O")):
        return "极性杂环"
    return "烷基/疏水"


def collect(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    history_doc = load_json(run_dir / "docking-history.json")
    history = {item["attempt"]: item for item in history_doc["history"]}
    rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    for attempt in sorted(history):
        item = history[attempt]
        edit = load_json(edit_path(run_dir, attempt))
        docking = edit["docking"]
        comparison = docking["comparison"]
        transformation = edit["transformation"]
        validation = edit.get("validation", {})
        fragment = transformation.get("fragment_smiles", "")
        tags = transformation.get("library_record", {}).get("chemical_tags", [])

        candidate_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
        reference_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
        per_seed_delta: dict[int, dict[str, float | None]] = {}

        for seed_item in comparison.get("per_seed", []):
            seed = int(seed_item["seed"])
            candidate_properties = seed_item["candidate_pose"]["properties"]
            reference_properties = seed_item["reference_pose"]["properties"]
            seed_delta: dict[str, float | None] = {}
            seed_row: dict[str, Any] = {
                "attempt": attempt,
                "parent_attempt": edit.get("parent_attempt") or 0,
                "generation": edit.get("generation"),
                "site": item.get("design_region", "atom:9"),
                "fragment_id": transformation.get("fragment_id", ""),
                "fragment_smiles": fragment,
                "seed": seed,
                "status": seed_item.get("status"),
            }
            for metric in METRICS:
                candidate = float(candidate_properties[metric])
                reference = float(reference_properties[metric])
                delta = candidate - reference
                candidate_values[metric].append(candidate)
                reference_values[metric].append(reference)
                seed_delta[metric] = delta
                seed_row[f"candidate_{metric}"] = candidate
                seed_row[f"reference_{metric}"] = reference
                seed_row[f"delta_{metric}"] = delta
            per_seed_delta[seed] = seed_delta
            seed_rows.append(seed_row)

        pose = item.get("pose_consensus", {})
        interactions = item.get("interaction_consensus", {})
        metric_summary = comparison.get("metrics", {})
        row: dict[str, Any] = {
            "attempt": attempt,
            "parent_attempt": edit.get("parent_attempt") or 0,
            "generation": edit.get("generation"),
            "site": item.get("design_region", "atom:9"),
            "operation": transformation.get("operation"),
            "fragment_id": transformation.get("fragment_id", ""),
            "fragment_smiles": fragment,
            "family": infer_family(fragment, tags),
            "canonical_smiles": validation.get("canonical_smiles"),
            "heavy_atoms": validation.get("heavy_atoms"),
            "molecular_weight": validation.get("molecular_weight"),
            "logp": validation.get("logp"),
            "hbd": validation.get("hbd"),
            "hba": validation.get("hba"),
            "tpsa": validation.get("tpsa"),
            "status": docking.get("status"),
            "completed_seed_count": comparison.get("completed_seed_count"),
            "mean_minimizedAffinity": mean(candidate_values["minimizedAffinity"]),
            "delta_minimizedAffinity": item.get("delta_candidate_minus_reference"),
            "stddev_delta_minimizedAffinity": item.get("seed_stddev"),
            "seed_win_fraction": item.get("seed_win_fraction"),
            "quality": item.get("quality"),
            "mean_CNNscore": mean(candidate_values["CNNscore"]),
            "delta_CNNscore": metric_summary.get("CNNscore", {}).get("delta_candidate_minus_reference", {}).get("mean"),
            "mean_CNNaffinity": mean(candidate_values["CNNaffinity"]),
            "delta_CNNaffinity": metric_summary.get("CNNaffinity", {}).get("delta_candidate_minus_reference", {}).get("mean"),
            "mean_CNN_VS": mean(candidate_values["CNN_VS"]),
            "delta_CNN_VS": metric_summary.get("CNN_VS", {}).get("delta_candidate_minus_reference", {}).get("mean"),
            "pose_mean_pairwise_rmsd": pose.get("mean_pairwise_rmsd"),
            "pose_max_pairwise_rmsd": pose.get("max_pairwise_rmsd"),
            "pose_stable": pose.get("stable", False),
            "gained_contacts": ";".join(interactions.get("gained_consensus_residues", [])),
            "lost_contacts": ";".join(interactions.get("lost_consensus_residues", [])),
            "gained_contact_count": len(interactions.get("gained_consensus_residues", [])),
            "lost_contact_count": len(interactions.get("lost_consensus_residues", [])),
            "is_new_best": item.get("is_new_best", False),
        }
        for seed in (17, 29, 43):
            for metric in METRICS:
                row[f"seed_{seed}_delta_{metric}"] = per_seed_delta.get(seed, {}).get(metric)
        rows.append(row)

    # Include attempted docking runs that did not enter scored docking history.
    failures: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("edit-attempt-*.json")):
        edit = load_json(path)
        attempt = int(edit["attempt"])
        docking = edit.get("docking", {})
        if docking.get("status") == "failed" and attempt not in history:
            failures.append({
                "attempt": attempt,
                "parent_attempt": edit.get("parent_attempt") or 0,
                "generation": edit.get("generation"),
                "site": f"atom:{edit.get('transformation', {}).get('edit_atom_index', 9)}",
                "fragment_smiles": edit.get("transformation", {}).get("fragment_smiles"),
                "failure_class": docking.get("failure_class"),
                "pose_count_per_seed": docking.get("pose_count_per_seed"),
            })

    return rows, seed_rows, failures, history_doc["convergence"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, seed_rows, failures, convergence = collect(run_dir)
    write_csv(output_dir / "docking_results_full.csv", rows)
    write_csv(output_dir / "docking_results_by_seed.csv", seed_rows)

    reference = {
        metric: statistics.mean(seed_row[f"reference_{metric}"] for seed_row in seed_rows[:3])
        for metric in METRICS
    }
    stable = [row for row in rows if row["pose_stable"]]
    fully_robust = [
        row for row in rows
        if row["seed_win_fraction"] == 1.0
        and row["pose_stable"]
        and row["delta_minimizedAffinity"] < 0
        and row["delta_CNNscore"] > 0
        and row["delta_CNNaffinity"] > 0
        and row["delta_CNN_VS"] > 0
    ]
    ranked = sorted((row for row in rows if row["quality"] is not None), key=lambda row: row["quality"], reverse=True)

    generation_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        generation_rows[int(row["generation"])].append(row)

    lines = [
        "# Docking 全结果与趋势分析",
        "",
        f"- 运行目录：`{run_dir}`",
        f"- 成功汇总：**{len(rows)}** 个候选（每个成功候选 3 个 seed，共 {len(seed_rows)} 个配对结果）",
        f"- 另有失败 docking：**{len(failures)}** 个",
        f"- 已实际 docking 的设计位点：**atom 9**；操作均为 `replace_hydrogen`/在后代中替换 atom 9 已有取代基",
        f"- 全局最佳：attempt **{convergence.get('best_attempt')}**，quality **{fmt(convergence.get('best_quality'))}**",
        "",
        "## 评分解释",
        "",
        "- `minimizedAffinity`：越低越好；`ΔMA = candidate - reference`，负值表示改善。",
        "- `CNNscore`、`CNNaffinity`、`CNN_VS`：越高越好；对应 Δ 为正表示改善。",
        "- `quality = -mean(ΔMA) - 0.25 × std(ΔMA)`，只在 seed 胜率达到门槛时计分；它不包含 CNN 指标或 pose RMSD。",
        "- `pose RMSD < 2 Å` 被标记为稳定。",
        "",
        "## Reference rank-1 三 seed 均值",
        "",
        f"| minimizedAffinity | CNNscore | CNNaffinity | CNN_VS |",
        "|---:|---:|---:|---:|",
        f"| {fmt(reference['minimizedAffinity'])} | {fmt(reference['CNNscore'])} | {fmt(reference['CNNaffinity'])} | {fmt(reference['CNN_VS'])} |",
        "",
        "## 每次成功 docking：主指标、稳定性与综合质量",
        "",
        "|A|G/P|片段|家族|MA均值|ΔMA|ΔSD|胜率|Quality|RMSD|稳定|接触 +/−|",
        "|---:|:---:|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"|{row['attempt']}|G{row['generation']}/P{row['parent_attempt']}|`{row['fragment_smiles']}`|{row['family']}|"
            f"{fmt(row['mean_minimizedAffinity'])}|{fmt(row['delta_minimizedAffinity'], signed=True)}|"
            f"{fmt(row['stddev_delta_minimizedAffinity'])}|{fmt(row['seed_win_fraction'], 2)}|"
            f"{fmt(row['quality'])}|{fmt(row['pose_mean_pairwise_rmsd'])}|"
            f"{'是' if row['pose_stable'] else '否'}|{row['gained_contact_count']}/{row['lost_contact_count']}|"
        )

    lines += [
        "",
        "## 每次成功 docking：CNN 各项评分",
        "",
        "|A|CNNscore|ΔCNNscore|CNNaffinity|ΔCNNaffinity|CNN_VS|ΔCNN_VS|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"|{row['attempt']}|{fmt(row['mean_CNNscore'])}|{fmt(row['delta_CNNscore'], signed=True)}|"
            f"{fmt(row['mean_CNNaffinity'])}|{fmt(row['delta_CNNaffinity'], signed=True)}|"
            f"{fmt(row['mean_CNN_VS'])}|{fmt(row['delta_CNN_VS'], signed=True)}|"
        )

    lines += ["", "## 未形成完整评分的 docking", ""]
    if failures:
        lines += ["|A|G/P|位点|片段|失败类别|各 seed pose 数|", "|---:|:---:|---|---|---|---|"]
        for failure in failures:
            lines.append(
                f"|{failure['attempt']}|G{failure['generation']}/P{failure['parent_attempt']}|{failure['site']}|"
                f"`{failure['fragment_smiles']}`|{failure['failure_class']}|`{failure['pose_count_per_seed']}`|"
            )
    else:
        lines.append("无。")

    lines += [
        "",
        "## 按 generation 的趋势",
        "",
        "|Generation|n|平均ΔMA|中位ΔMA|最佳ΔMA|3/3 seed改善|稳定pose|平均Quality*|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for generation in sorted(generation_rows):
        group = generation_rows[generation]
        qualities = [row["quality"] for row in group if row["quality"] is not None]
        lines.append(
            f"|{generation}|{len(group)}|{fmt(statistics.mean(row['delta_minimizedAffinity'] for row in group), signed=True)}|"
            f"{fmt(statistics.median(row['delta_minimizedAffinity'] for row in group), signed=True)}|"
            f"{fmt(min(row['delta_minimizedAffinity'] for row in group), signed=True)}|"
            f"{sum(row['seed_win_fraction'] == 1.0 for row in group)}|{sum(row['pose_stable'] for row in group)}|"
            f"{fmt(statistics.mean(qualities) if qualities else None)}|"
        )
    lines.append("\n\\* 平均 Quality 仅统计有 quality 的候选。")

    lines += [
        "",
        "## 最值得保留的候选",
        "",
        "以下候选同时满足：ΔMA<0、3/3 seed 改善、pose 稳定，且三个 CNN 均值均优于 reference。",
        "",
        "|排名|A|片段|ΔMA|Quality|RMSD|ΔCNNscore|ΔCNNaffinity|ΔCNN_VS|新增接触|丢失接触|",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(sorted(fully_robust, key=lambda item: item["quality"], reverse=True), 1):
        lines.append(
            f"|{rank}|{row['attempt']}|`{row['fragment_smiles']}`|{fmt(row['delta_minimizedAffinity'], signed=True)}|"
            f"{fmt(row['quality'])}|{fmt(row['pose_mean_pairwise_rmsd'])}|{fmt(row['delta_CNNscore'], signed=True)}|"
            f"{fmt(row['delta_CNNaffinity'], signed=True)}|{fmt(row['delta_CNN_VS'], signed=True)}|"
            f"{row['gained_contacts'] or '无'}|{row['lost_contacts'] or '无'}|"
        )

    lines += [
        "",
        "## Quality 前 10",
        "",
        "|排名|A|片段|G/P|ΔMA|Quality|RMSD|pose稳定|",
        "|---:|---:|---|:---:|---:|---:|---:|:---:|",
    ]
    for rank, row in enumerate(ranked[:10], 1):
        lines.append(
            f"|{rank}|{row['attempt']}|`{row['fragment_smiles']}`|G{row['generation']}/P{row['parent_attempt']}|"
            f"{fmt(row['delta_minimizedAffinity'], signed=True)}|{fmt(row['quality'])}|"
            f"{fmt(row['pose_mean_pairwise_rmsd'])}|{'是' if row['pose_stable'] else '否'}|"
        )

    lines += [
        "",
        "## 总体统计",
        "",
        f"- 42 个成功候选中，**{sum(row['delta_minimizedAffinity'] < 0 for row in rows)}** 个平均 minimizedAffinity 优于 reference。",
        f"- **{sum(row['seed_win_fraction'] == 1.0 for row in rows)}** 个在 3/3 seed 上改善；**{len(stable)}** 个 pose 稳定。",
        f"- **{len(fully_robust)}** 个同时达到主指标全 seed 改善、pose 稳定、三个 CNN 均值全部改善。",
        f"- CNNaffinity 对 **{sum(row['delta_CNNaffinity'] > 0 for row in rows)}/{len(rows)}** 个候选均给出改善，区分度较低；CNNscore 和 pose RMSD 更能暴露不稳定或异常方案。",
        "- 所有成功 docking 都集中在 atom 9；当前结果只能说明 atom 9 局部 SAR，不能代表全位点搜索已完成。",
        "",
        "## 结论",
        "",
        "1. **Attempt 18 是当前首选**：主指标改善最大、3/3 seed 胜出、pose 稳定、全部 CNN 指标改善，且没有丢失 reference 共识接触。",
        "2. **Attempts 30、14、41 是高稳健备选**：均为稳定 pose、全 seed 主指标改善、全部 CNN 指标改善，并且没有丢失共识接触。",
        "3. **Attempt 43 是优质新支系父代**：打分强且 pose 稳定，但丢失 4 个 reference 共识接触；适合作为化学多样性备选，不应替代 attempt 18。",
        "4. **Attempt 46 的 quality 很高但 pose 不稳定**：quality 公式不惩罚 RMSD，不能仅凭 quality 排为第二。",
        "5. **Attempt 48 明显退化**：CNNscore/CNN_VS 大幅下降，pose RMSD 5.732 Å；不建议继续沿双羟基/高极性方向扩展。",
        "6. 搜索从 G1 到 G2 明显改善，G3 提供多个好分支，但 G4 未超过 G2 的 attempt 18，显示 atom 9 当前化学系列开始进入平台期。",
        "",
        "完整逐 seed 数据见：`docking_results_by_seed.csv`；候选属性与汇总见：`docking_results_full.csv`。",
    ]

    (output_dir / "docking_results_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    site_clearance = {
        0: "原始配体：局部 clearance 3.917 Å；2.5 Å 外向探针最小 clearance 3.617 Å",
        14: "Parent 14：2.5 Å 外向探针最小 clearance 2.989 Å",
        18: "Parent 18：2.5 Å 外向探针 3.046 Å；3.0 Å 外向探针 3.212 Å",
        43: "Parent 43：3.0 Å 外向探针最小 clearance 3.225 Å",
    }

    def cell(value: Any, digits: int = 3, signed: bool = False) -> str:
        return html.escape(fmt(value, digits, signed))

    def score_class(value: float | None, lower_is_better: bool = False) -> str:
        if value is None:
            return "neutral"
        good = value < 0 if lower_is_better else value > 0
        if abs(value) < 1e-12:
            return "neutral"
        return "good" if good else "bad"

    robust_attempts = {row["attempt"] for row in fully_robust}
    best_attempt = convergence.get("best_attempt")
    main_body: list[str] = []
    property_body: list[str] = []
    for row in rows:
        parent = int(row["parent_attempt"])
        site_mode = "替换原始 H" if parent == 0 else f"替换 Parent {parent} 在 atom 9 的已有取代基"
        classes = ["result-row"]
        if row["attempt"] == best_attempt:
            classes.append("best-row")
        elif row["attempt"] in robust_attempts:
            classes.append("robust-row")
        elif not row["pose_stable"] and row["pose_mean_pairwise_rmsd"] is not None and row["pose_mean_pairwise_rmsd"] >= 4:
            classes.append("unstable-row")
        search = " ".join(str(row.get(key, "")) for key in ("attempt", "parent_attempt", "generation", "fragment_id", "fragment_smiles", "family", "site"))
        main_body.append(f'''<tr class="{' '.join(classes)}" data-generation="{row['generation']}" data-parent="{parent}" data-stable="{str(bool(row['pose_stable'])).lower()}" data-robust="{str(row['attempt'] in robust_attempts).lower()}" data-search="{html.escape(search.lower(), quote=True)}">
<td data-sort="{row['attempt']}"><strong>{row['attempt']}</strong></td>
<td data-sort="{row['generation']}">G{row['generation']}</td>
<td data-sort="{parent}">{'原始配体' if parent == 0 else 'A' + str(parent)}</td>
<td><span class="site-chip">Atom 9</span><div class="sub">芳香 C · pocket extension</div><div class="sub">{html.escape(site_mode)}</div></td>
<td><code>{html.escape(row['fragment_smiles'] or '')}</code><div class="sub">{html.escape(row['fragment_id'] or 'curated / 未记录 ID')}</div></td>
<td>{html.escape(row['family'])}</td>
<td data-sort="{row['mean_minimizedAffinity']}">{cell(row['mean_minimizedAffinity'])}</td>
<td class="{score_class(row['delta_minimizedAffinity'], True)}" data-sort="{row['delta_minimizedAffinity']}">{cell(row['delta_minimizedAffinity'], signed=True)}</td>
<td data-sort="{row['stddev_delta_minimizedAffinity']}">{cell(row['stddev_delta_minimizedAffinity'])}</td>
<td data-sort="{row['seed_win_fraction']}">{int(round(row['seed_win_fraction'] * 3))}/3</td>
<td data-sort="{'' if row['quality'] is None else row['quality']}">{cell(row['quality'])}</td>
<td data-sort="{'' if row['pose_mean_pairwise_rmsd'] is None else row['pose_mean_pairwise_rmsd']}">{cell(row['pose_mean_pairwise_rmsd'])}</td>
<td data-sort="{1 if row['pose_stable'] else 0}"><span class="badge {'ok' if row['pose_stable'] else 'warn'}">{'稳定' if row['pose_stable'] else '不稳定'}</span></td>
<td data-sort="{row['mean_CNNscore']}">{cell(row['mean_CNNscore'])}</td>
<td class="{score_class(row['delta_CNNscore'])}" data-sort="{row['delta_CNNscore']}">{cell(row['delta_CNNscore'], signed=True)}</td>
<td data-sort="{row['mean_CNNaffinity']}">{cell(row['mean_CNNaffinity'])}</td>
<td class="{score_class(row['delta_CNNaffinity'])}" data-sort="{row['delta_CNNaffinity']}">{cell(row['delta_CNNaffinity'], signed=True)}</td>
<td data-sort="{row['mean_CNN_VS']}">{cell(row['mean_CNN_VS'])}</td>
<td class="{score_class(row['delta_CNN_VS'])}" data-sort="{row['delta_CNN_VS']}">{cell(row['delta_CNN_VS'], signed=True)}</td>
<td><span class="good">+{row['gained_contact_count']}</span> / <span class="bad">−{row['lost_contact_count']}</span><details><summary>残基</summary><div class="details-text"><b>新增：</b>{html.escape(row['gained_contacts'] or '无')}<br><b>丢失：</b>{html.escape(row['lost_contacts'] or '无')}</div></details></td>
</tr>''')
        property_body.append(f'''<tr data-generation="{row['generation']}" data-parent="{parent}" data-stable="{str(bool(row['pose_stable'])).lower()}" data-robust="{str(row['attempt'] in robust_attempts).lower()}" data-search="{html.escape(search.lower(), quote=True)}">
<td>{row['attempt']}</td><td><span class="site-chip">Atom 9</span><div class="sub">{html.escape(site_mode)}</div></td><td><code>{html.escape(row['fragment_smiles'] or '')}</code></td>
<td>{cell(row['heavy_atoms'], 0)}</td><td>{cell(row['molecular_weight'], 2)}</td><td>{cell(row['logp'], 2)}</td><td>{cell(row['hbd'], 0)}</td><td>{cell(row['hba'], 0)}</td><td>{cell(row['tpsa'], 2)}</td><td class="canonical"><code>{html.escape(row['canonical_smiles'] or '')}</code></td>
</tr>''')

    seed_body: list[str] = []
    row_lookup = {row["attempt"]: row for row in rows}
    for seed_row in seed_rows:
        row = row_lookup[seed_row["attempt"]]
        parent = int(row["parent_attempt"])
        search = f"{row['attempt']} {row['fragment_smiles']} atom 9 {seed_row['seed']}".lower()
        seed_body.append(f'''<tr data-generation="{row['generation']}" data-parent="{parent}" data-stable="{str(bool(row['pose_stable'])).lower()}" data-robust="{str(row['attempt'] in robust_attempts).lower()}" data-search="{html.escape(search, quote=True)}">
<td>{row['attempt']}</td><td>G{row['generation']}/P{parent}</td><td><span class="site-chip">Atom 9</span></td><td><code>{html.escape(row['fragment_smiles'])}</code></td><td>{seed_row['seed']}</td>
<td>{cell(seed_row['candidate_minimizedAffinity'])}</td><td>{cell(seed_row['reference_minimizedAffinity'])}</td><td class="{score_class(seed_row['delta_minimizedAffinity'], True)}">{cell(seed_row['delta_minimizedAffinity'], signed=True)}</td>
<td>{cell(seed_row['candidate_CNNscore'])}</td><td class="{score_class(seed_row['delta_CNNscore'])}">{cell(seed_row['delta_CNNscore'], signed=True)}</td>
<td>{cell(seed_row['candidate_CNNaffinity'])}</td><td class="{score_class(seed_row['delta_CNNaffinity'])}">{cell(seed_row['delta_CNNaffinity'], signed=True)}</td>
<td>{cell(seed_row['candidate_CNN_VS'])}</td><td class="{score_class(seed_row['delta_CNN_VS'])}">{cell(seed_row['delta_CNN_VS'], signed=True)}</td>
</tr>''')

    failed_body = "".join(
        f"<tr><td>{failure['attempt']}</td><td>G{failure['generation']}/P{failure['parent_attempt']}</td><td><span class='site-chip'>Atom 9</span></td><td><code>{html.escape(failure['fragment_smiles'] or '')}</code></td><td>{html.escape(str(failure['failure_class']))}</td><td><code>{html.escape(str(failure['pose_count_per_seed']))}</code></td></tr>"
        for failure in failures
    ) or "<tr><td colspan='6'>无失败 docking</td></tr>"

    robust_cards = "".join(
        f'''<article class="candidate-card {'champion' if row['attempt'] == best_attempt else ''}"><div class="candidate-rank">A{row['attempt']}</div><code>{html.escape(row['fragment_smiles'])}</code><div class="candidate-metrics"><span>ΔMA <b>{cell(row['delta_minimizedAffinity'], signed=True)}</b></span><span>Q <b>{cell(row['quality'])}</b></span><span>RMSD <b>{cell(row['pose_mean_pairwise_rmsd'])} Å</b></span></div><div class="sub">位点：Atom 9 · G{row['generation']}/P{row['parent_attempt']}</div></article>'''
        for row in sorted(fully_robust, key=lambda item: item["quality"], reverse=True)
    )

    generation_parent = {1: "原始配体", 2: "A14", 3: "A18", 4: "A43"}
    generation_parts: list[str] = []
    for generation, group in sorted(generation_rows.items()):
        n = len(group)
        all_seed = sum(row["seed_win_fraction"] == 1.0 for row in group)
        stable_count = sum(row["pose_stable"] for row in group)
        both = sum(row["seed_win_fraction"] == 1.0 and row["pose_stable"] for row in group)
        generation_parts.append(
            f"<tr><td>G{generation}</td><td>{generation_parent.get(generation, '—')}</td><td>{n}</td>"
            f"<td>{fmt(statistics.mean(row['delta_minimizedAffinity'] for row in group), signed=True)}</td>"
            f"<td>{fmt(statistics.median(row['delta_minimizedAffinity'] for row in group), signed=True)}</td>"
            f"<td>{fmt(min(row['delta_minimizedAffinity'] for row in group), signed=True)}</td>"
            f"<td>{all_seed}/{n} ({all_seed / n:.1%})</td><td>{stable_count}/{n} ({stable_count / n:.1%})</td>"
            f"<td>{both}/{n} ({both / n:.1%})</td></tr>"
        )
    generation_body = "".join(generation_parts)

    html_document = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Docking 全结果与位点分析</title>
<style>
:root{{--bg:#f4f7fb;--panel:#fff;--text:#172033;--muted:#667085;--line:#d9e2ef;--blue:#2257d7;--blue2:#e9f0ff;--green:#087443;--green-bg:#e8f7ef;--red:#b42318;--red-bg:#fff0ee;--amber:#9a6700;--amber-bg:#fff7db;--shadow:0 8px 28px rgba(19,42,86,.08)}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 Inter,"Noto Sans SC","Microsoft YaHei",Arial,sans-serif}} .wrap{{max-width:1900px;margin:auto;padding:24px}} h1{{font-size:28px;margin:0 0 6px}} h2{{font-size:20px;margin:28px 0 12px}} h3{{font-size:16px;margin:0 0 8px}} .subtitle,.sub{{color:var(--muted)}} .sub{{font-size:12px;margin-top:2px}} .hero,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}} .hero{{padding:22px}} .stats{{display:grid;grid-template-columns:repeat(6,minmax(130px,1fr));gap:12px;margin-top:18px}} .stat{{padding:14px;border-radius:12px;background:#f8faff;border:1px solid #e4eaf4}} .stat b{{display:block;font-size:22px;color:#163b91}} .site-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .site-card{{background:white;border:1px solid var(--line);border-radius:14px;padding:16px}} .site-card.primary{{border:2px solid var(--blue);background:linear-gradient(135deg,#fff,#f0f5ff)}} .site-chip{{display:inline-block;padding:2px 8px;border-radius:999px;background:var(--blue2);color:var(--blue);font-weight:700;white-space:nowrap}} .candidate-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}} .candidate-card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px;position:relative}} .candidate-card.champion{{border:2px solid #d19a00;background:#fffaf0}} .candidate-rank{{font-size:18px;font-weight:800;color:var(--blue)}} .candidate-metrics{{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}} .filters{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:14px;background:#f8faff;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}} input,select{{border:1px solid #bcc9dc;border-radius:8px;padding:8px 10px;background:#fff;color:var(--text)}} input{{min-width:260px}} .table-wrap{{overflow:auto;max-height:76vh}} table{{border-collapse:separate;border-spacing:0;width:100%;background:#fff}} th,td{{padding:9px 10px;border-bottom:1px solid #e5eaf2;border-right:1px solid #eef2f7;text-align:left;vertical-align:top;white-space:nowrap}} th{{background:#edf3ff;color:#20345c;font-size:12px;position:sticky;top:0;z-index:3;cursor:pointer}} td code{{font-size:12px}} tr:hover td{{background:#f1f6ff!important}} .best-row td{{background:#fff8df}} .robust-row td{{background:#f0fbf5}} .unstable-row td{{background:#fffafa}} .good{{color:var(--green);font-weight:700}} .bad{{color:var(--red);font-weight:700}} .neutral{{color:var(--muted)}} .badge{{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:700}} .badge.ok{{background:var(--green-bg);color:var(--green)}} .badge.warn{{background:var(--red-bg);color:var(--red)}} .panel{{overflow:hidden;margin:12px 0 28px}} .panel-head{{padding:16px 18px;border-bottom:1px solid var(--line)}} .tabs{{display:flex;gap:6px;margin-top:22px}} .tab{{border:1px solid var(--line);background:#fff;padding:10px 16px;border-radius:10px 10px 0 0;cursor:pointer;font-weight:700}} .tab.active{{color:#fff;background:var(--blue);border-color:var(--blue)}} .tab-content{{display:none}} .tab-content.active{{display:block}} details{{white-space:normal}} summary{{cursor:pointer;color:var(--blue)}} .details-text{{min-width:260px;max-width:420px;padding:6px 0;white-space:normal}} .canonical{{max-width:650px;white-space:normal;word-break:break-all}} .legend{{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);margin:8px 0}} .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}} .foot{{color:var(--muted);font-size:12px;margin:26px 0}} @media(max-width:1100px){{.stats,.site-grid,.candidate-grid{{grid-template-columns:repeat(2,1fr)}}}} @media(max-width:650px){{.wrap{{padding:10px}}.stats,.site-grid,.candidate-grid{{grid-template-columns:1fr}}input{{min-width:100%}}}}
</style></head><body><main class="wrap">
<section class="hero"><h1>Docking 全结果与位点分析</h1><div class="subtitle">Run: {html.escape(run_dir.parent.name)} · 自动汇总 3-seed GNINA rank-1 结果</div>
<div class="stats"><div class="stat"><b>{len(rows)}</b>成功候选</div><div class="stat"><b>{len(seed_rows)}</b>配对 seed 结果</div><div class="stat"><b>Atom 9</b>唯一已 docking 位点</div><div class="stat"><b>A{best_attempt}</b>当前全局最佳</div><div class="stat"><b>{len(stable)}</b>稳定 pose</div><div class="stat"><b>{len(fully_robust)}</b>全条件稳健候选</div></div></section>
<h2>位点信息</h2><section class="site-grid"><article class="site-card primary"><h3><span class="site-chip">Atom 9</span> 已实际 Docking</h3><p><b>原子：</b>芳香碳 C，原始结构有 1 个可替换 H<br><b>策略类型：</b>pocket extension<br><b>优先级：</b>1（当前 active target）<br><b>局部 clearance：</b>3.917 Å</p><p class="sub">邻近：HIS:A:84:O 3.048 Å；ILE:A:10:CG2 3.404 Å；HIS:A:84:C 3.907 Å；GLN:A:85:CA 3.919 Å。</p></article>
<article class="site-card"><h3>原始配体 / G1</h3><p>{html.escape(site_clearance[0])}</p><p class="sub">操作：替换 atom 9 上的原始氢。</p></article><article class="site-card"><h3>Parent 14 / G2</h3><p>{html.escape(site_clearance[14])}</p><p class="sub">操作：在同一 atom 9 替换 parent 14 的取代基。</p></article><article class="site-card"><h3>Parent 18、43 / G3–G4</h3><p>{html.escape(site_clearance[18])}<br>{html.escape(site_clearance[43])}</p><p class="sub">仍是同一核心 atom 9，不是新位点。</p></article></section>
<h2>当前稳健候选</h2><section class="candidate-grid">{robust_cards}</section>
<h2>代际趋势</h2><section class="panel"><div class="panel-head"><div class="sub">所有 ΔMA 都是候选相对原始 reference 的配对差值，并非相对上一代 parent。计数只包含完成 3 个 seed 的候选。</div></div><div class="table-wrap" style="max-height:none"><table><thead><tr><th>Generation</th><th>本代 Parent</th><th>成功候选数 n</th><th title="本代每个候选的三-seed平均ΔMA，再对本代候选求平均；负值越大越好">平均 ΔMA</th><th title="本代候选ΔMA的中位数，受极端值影响较小">中位 ΔMA</th><th title="本代单个候选中最负的ΔMA">最佳 ΔMA</th><th title="候选在 seed 17、29、43 的 minimizedAffinity 均低于各自 reference">3/3 seed 改善</th><th title="候选三个 rank-1 pose 的全部两两重原子RMSD均不超过2 Å">稳定 pose</th><th title="同时达到3/3 seed主指标改善和pose稳定">同时满足两项</th></tr></thead><tbody>{generation_body}</tbody></table></div></section>
<div class="tabs"><button class="tab active" data-tab="summary">候选汇总</button><button class="tab" data-tab="seed">逐 Seed</button><button class="tab" data-tab="properties">理化性质</button><button class="tab" data-tab="failed">失败 Docking</button></div>
<section id="summary" class="tab-content active panel"><div class="panel-head"><h3>所有成功 docking</h3><div class="legend"><span><i class="dot" style="background:#fff0b8"></i>全局最佳</span><span><i class="dot" style="background:#dff5e9"></i>全条件稳健</span><span><i class="dot" style="background:#ffe8e8"></i>RMSD ≥ 4 Å</span></div></div><div class="filters"><input class="search" placeholder="搜索 attempt、片段、fragment ID、位点…"><select class="generation-filter"><option value="all">全部 Generation</option><option value="1">G1</option><option value="2">G2</option><option value="3">G3</option><option value="4">G4</option></select><select class="parent-filter"><option value="all">全部 Parent</option><option value="0">原始配体</option><option value="14">P14</option><option value="18">P18</option><option value="43">P43</option></select><select class="stable-filter"><option value="all">全部 Pose</option><option value="true">仅稳定</option><option value="false">仅不稳定</option></select><select class="robust-filter"><option value="all">全部候选</option><option value="true">仅全条件稳健</option></select><span class="visible-count"></span></div><div class="table-wrap"><table class="sortable"><thead><tr><th>A</th><th>代</th><th>Parent</th><th>位点与操作</th><th>片段</th><th>家族</th><th>MA均值</th><th>ΔMA</th><th>ΔSD</th><th>胜率</th><th>Quality</th><th>RMSD</th><th>Pose</th><th>CNNscore</th><th>ΔCNNscore</th><th>CNNaff.</th><th>ΔCNNaff.</th><th>CNN_VS</th><th>ΔCNN_VS</th><th>接触 +/−</th></tr></thead><tbody>{''.join(main_body)}</tbody></table></div></section>
<section id="seed" class="tab-content panel"><div class="panel-head"><h3>每个 attempt 的 seed 17、29、43 原始结果</h3></div><div class="filters"><input class="search" placeholder="搜索 attempt、片段、seed、位点…"><span class="visible-count"></span></div><div class="table-wrap"><table class="sortable"><thead><tr><th>A</th><th>G/P</th><th>位点</th><th>片段</th><th>Seed</th><th>MA候选</th><th>MA参考</th><th>ΔMA</th><th>CNNscore</th><th>Δ</th><th>CNNaff.</th><th>Δ</th><th>CNN_VS</th><th>Δ</th></tr></thead><tbody>{''.join(seed_body)}</tbody></table></div></section>
<section id="properties" class="tab-content panel"><div class="panel-head"><h3>候选理化性质</h3></div><div class="filters"><input class="search" placeholder="搜索 attempt、片段、位点…"><span class="visible-count"></span></div><div class="table-wrap"><table class="sortable"><thead><tr><th>A</th><th>位点与操作</th><th>片段</th><th>重原子</th><th>MW</th><th>logP</th><th>HBD</th><th>HBA</th><th>TPSA</th><th>候选 canonical SMILES</th></tr></thead><tbody>{''.join(property_body)}</tbody></table></div></section>
<section id="failed" class="tab-content panel"><div class="panel-head"><h3>未形成完整 3-seed 评分的 docking</h3></div><div class="table-wrap" style="max-height:none"><table><thead><tr><th>A</th><th>G/P</th><th>位点</th><th>片段</th><th>失败类别</th><th>各 seed pose 数</th></tr></thead><tbody>{failed_body}</tbody></table></div></section>
<p class="foot">说明：minimizedAffinity 越低越好，故 ΔMA&lt;0 为改善；CNN 指标越高越好，故 Δ&gt;0 为改善。Quality 仅由 ΔMA 均值、跨 seed 标准差和胜率门槛计算，不包含 CNN 或 pose RMSD。所有接触均为距离型共识接触，不等同于实验活性或自由能。</p>
</main><script>
function applyFilters(panel){{const q=(panel.querySelector('.search')?.value||'').toLowerCase();const g=panel.querySelector('.generation-filter')?.value||'all';const p=panel.querySelector('.parent-filter')?.value||'all';const s=panel.querySelector('.stable-filter')?.value||'all';const r=panel.querySelector('.robust-filter')?.value||'all';let n=0;panel.querySelectorAll('tbody tr').forEach(tr=>{{const show=(!q||(tr.dataset.search||tr.innerText.toLowerCase()).includes(q))&&(g==='all'||tr.dataset.generation===g)&&(p==='all'||tr.dataset.parent===p)&&(s==='all'||tr.dataset.stable===s)&&(r==='all'||tr.dataset.robust===r);tr.style.display=show?'':'none';if(show)n++;}});const out=panel.querySelector('.visible-count');if(out)out.textContent='显示 '+n+' 行';}}
document.querySelectorAll('.filters input,.filters select').forEach(el=>el.addEventListener('input',()=>applyFilters(el.closest('.tab-content'))));document.querySelectorAll('.tab-content').forEach(applyFilters);
document.querySelectorAll('.tab').forEach(btn=>btn.addEventListener('click',()=>{{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.tab-content').forEach(x=>x.classList.remove('active'));btn.classList.add('active');document.getElementById(btn.dataset.tab).classList.add('active');}}));
document.querySelectorAll('table.sortable th').forEach((th,index)=>th.addEventListener('click',()=>{{const table=th.closest('table'),body=table.tBodies[0],rows=[...body.rows],asc=th.dataset.asc!=='true';table.querySelectorAll('th').forEach(x=>delete x.dataset.asc);th.dataset.asc=String(asc);rows.sort((a,b)=>{{let av=a.cells[index]?.dataset.sort??a.cells[index]?.innerText??'',bv=b.cells[index]?.dataset.sort??b.cells[index]?.innerText??'';const an=parseFloat(av),bn=parseFloat(bv);if(!Number.isNaN(an)&&!Number.isNaN(bn))return asc?an-bn:bn-an;return asc?av.localeCompare(bv,'zh-CN'):bv.localeCompare(av,'zh-CN');}});rows.forEach(row=>body.appendChild(row));}}));
</script></body></html>'''
    (output_dir / "docking_results.html").write_text(html_document, encoding="utf-8")

    print(output_dir / "docking_results_analysis.md")
    print(output_dir / "docking_results_full.csv")
    print(output_dir / "docking_results_by_seed.csv")
    print(output_dir / "docking_results.html")


if __name__ == "__main__":
    main()
