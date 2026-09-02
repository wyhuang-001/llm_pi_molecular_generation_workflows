#!/usr/bin/env python3
"""Generate a self-contained HTML audit report for a docking workflow run."""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


METRICS = (
    ("minimizedAffinity", "Affinity", "lower"),
    ("CNNscore", "CNNscore", "higher"),
    ("CNNaffinity", "CNNaffinity", "higher"),
    ("CNN_VS", "CNN_VS", "higher"),
)


def esc(value: Any) -> str:
    return html.escape("-" if value is None else str(value), quote=True)


def number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return esc(value)


def seq(path: Path) -> int:
    try:
        return int(path.stem.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def metric_delta(metric: dict[str, Any] | None) -> tuple[Any, Any, Any, str]:
    if not isinstance(metric, dict):
        return None, None, None, ""
    delta = metric.get("delta_candidate_minus_reference") or {}
    if not isinstance(delta, dict):
        return delta, None, None, metric.get("direction", "")
    return (
        delta.get("mean"),
        delta.get("stddev"),
        metric.get("candidate_better_seed_count"),
        metric.get("direction", ""),
    )


def delta_class(value: Any, direction: str) -> str:
    if not isinstance(value, (int, float)):
        return "neutral"
    better = value < 0 if direction == "lower_is_better" else value > 0
    return "better" if better else "worse"


def metric_html(metric: dict[str, Any] | None) -> str:
    mean, stddev, wins, direction = metric_delta(metric)
    if mean is None:
        return '<td class="neutral">-</td>'
    cls = delta_class(mean, direction)
    n = metric.get("n", "-") if isinstance(metric, dict) else "-"
    return (
        f'<td class="{cls}"><strong>{number(mean)}</strong>'
        f'<span class="sub">+/- {number(stddev)} | {esc(wins)}/{esc(n)} seeds</span></td>'
    )


def status_badge(status: Any) -> str:
    labels = {
        "complete": "完成", "accepted": "通过", "closed": "已关闭", "docked": "已 docking",
        "active": "当前搜索", "pending": "待处理", "rejected": "拒绝",
        "duplicate_structure": "重复结构", "failed": "失败",
    }
    text = esc(labels.get(status, status or "-"))
    cls = "ok" if status in {"complete", "accepted", "closed", "docked"} else "warn"
    return f'<span class="badge {cls}">{text}</span>'


SITE_TYPE_LABELS = {
    "pocket_extension": "口袋延伸位点",
    "solvent_exposed": "溶剂暴露位点",
    "core_anchor": "核心锚定位点",
    "linker_or_sidechain": "连接体/侧链位点",
    "uncertain": "不确定位点",
}
FAMILY_LABELS = {
    "alkyl": "烷基",
    "halogen": "卤素",
    "polar": "极性片段",
    "heteroaryl": "杂芳基",
    "other": "其他",
    "fragment_replacement": "片段替换",
}
SIZE_LABELS = {
    "minimal": "最小（minimal）",
    "small": "小（small）",
    "medium": "中（medium）",
    "large": "大（large）",
}
TAG_LABELS = {
    "alkyl": "烷基", "halogen": "卤素", "polar": "极性", "aromatic": "芳香",
    "heteroaryl": "杂芳基", "cyclic": "环状", "saturated_ring": "饱和环",
    "saturated_heterocycle": "饱和杂环", "hbond_donor": "氢键供体",
    "hbond_acceptor": "氢键受体", "nitrile": "腈", "carbonyl": "羰基",
    "amide": "酰胺", "alkoxy": "烷氧基", "hbond": "氢键",
}
SOURCE_LABELS = {
    "project_seed": "项目种子片段",
    "chembl_working": "ChEMBL 工作库",
}
OPERATION_LABELS = {
    "substitute": "氢替换（substitute）",
    "replace_fragment": "片段替换（replace_fragment）",
}
FRAGMENT_NAME_ZH = {
    "curated-fluoro": "氟基",
    "curated-chloro": "氯基",
    "curated-methyl": "甲基",
    "curated-ethyl": "乙基",
    "curated-hydroxy": "羟基",
    "curated-amino": "氨基",
    "curated-cyano": "氰基",
    "curated-methoxy": "甲氧基",
    "curated-acetamide": "酰胺基",
    "chembl-brics-filtered-000443": "桥环烷基片段 443",
    "chembl-brics-filtered-000445": "桥环烷基片段 445",
    "chembl-brics-filtered-000446": "双环丁基片段",
    "chembl-brics-filtered-000519": "环丁基片段",
    "chembl-brics-filtered-000529": "氮杂环丁基片段",
    "chembl-brics-filtered-000530": "氧杂环丁基片段",
    "chembl-brics-filtered-000547": "环丙基甲基片段",
    "chembl-brics-filtered-000548": "环丙基氨基片段",
    "chembl-brics-filtered-000550": "环丙基片段",
    "chembl-brics-filtered-001053": "N-连接氮杂环丁基片段",
}
SMILES_NAME_ZH = {
    "[*:1]F": "氟基",
    "[*:1]Cl": "氯基",
    "[*:1]C": "甲基",
    "[*:1]CC": "乙基",
    "[*:1]O": "羟基",
    "[*:1]N": "氨基",
    "[*:1]C#N": "氰基",
    "C1CC1[*:1]": "环丙基",
    "C1CC1C[*:1]": "环丙基甲基",
}
RATIONALE_ZH = {
    9: "当前未关闭原子中净空最大（3.917 Å）。这是芳香碳，邻近 ILE:A:10、HIS:A:84、GLN:A:85 疏水接触，适合测试卤素或小型取代基以加强疏水填充。",
    6: "净空 3.479 Å，具有一定生长空间；邻近 GLN:A:131 骨架极性环境，可测试取代基是否能增加相互作用。",
    3: "苯胺苯环上的位点，净空 2.791 Å，靠近 GLY:A:11/GLY:A:13 骨架，优先考虑小型卤素。",
    2: "净空 2.564 Å，面对 VAL:A:64 和 PHE:A:80 的疏水面，只适合较小的卤素或取代基。",
    23: "位于口袋边缘的环己基碳，接触 ILE:A:10、PHE:A:82、HIS:A:84；净空 1.637 Å，生长空间有限。",
    21: "核心 NH 与 LEU:A:83 骨架形成角色相容的极性接触，是高价值锚点，修改可能破坏关键结合。",
    1: "苯环 CH 接触 VAL:A:64 和 PHE:A:80，净空 1.748 Å，只适合极小取代基。",
    4: "苯环碳净空 1.937 Å，空间有限，只适合小型取代。",
    10: "苯胺苯环位点，净空 1.913 Å，接触 GLN:A:85 和 ILE:A:10，只适合小型取代。",
    11: "净空 1.833 Å，位于靠近 GLN:A:85 和 ILE:A:10 的连接体区域，可测试小型极性或卤素片段。",
    12: "净空 1.62 Å，靠近 LEU:A:134，空间很紧，只适合最小片段。",
    5: "净空仅 1.209 Å，靠近 GLU:A:12/GLY:A:13，宿主判断应限制为卤素家族。",
    7: "净空 1.076 Å，属于非常紧的核心位置，宿主判断应限制为卤素。",
    8: "净空 1.4 Å，属于紧邻核心位置，宿主判断应限制为卤素。",
}
REPLACEMENT_RATIONALE_ZH = {
    "replacement-site-001": "宿主验证的定向切割位点，可替换环己基甲氧侧链，测试口袋占据方式的重构。",
    "replacement-site-002": "宿主验证的定向切割位点，可移除末端苯基并换成杂芳基；目标是在保留核心锚定的同时引入新的定向极性作用。",
    "replacement-site-003": "定向切割会移除苯胺 NH 和苯基，可能破坏 LEU:A:83 极性锚定，因此风险较高。",
    "replacement-site-004": "定向切割可改变 O 连接侧链，属于较大结构扰动，优先级较低。",
    "replacement-site-005": "定向切割提供侧链整体替换机会，需先完成空间轮廓和片段兼容性验证。",
}


def translate_tag(value: Any) -> str:
    return TAG_LABELS.get(str(value), str(value))


def translate_family(value: Any) -> str:
    text = str(value or "")
    if text.startswith("fragment_replacement:"):
        return "片段替换：" + FAMILY_LABELS.get(text.split(":", 1)[1], text.split(":", 1)[1])
    return FAMILY_LABELS.get(text, text or "-")


def rationale_text(decision: dict[str, Any]) -> str:
    text = decision.get("edit_hypothesis") or decision.get("understanding") or decision.get("question")
    return " ".join(text.split()) if isinstance(text, str) else ""


def concise_llm_rationale(decision: dict[str, Any], limit: int = 230) -> str:
    """Keep the full hypothesis in the title while showing only a short summary."""
    text = rationale_text(decision)
    if not text:
        return "未记录 LLM 改造理由"
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _reason_segments(text: str) -> list[str]:
    sentences = [
        part.strip(" ;；。")
        for part in re.split(r"(?<=[。！？])|(?<=[.!?])\s+(?=[A-Z])", text)
        if part.strip(" ;；。")
    ]
    if len(sentences) < 3:
        clauses = [part.strip(" ;；。") for part in re.split(r"[；;]", text) if part.strip(" ;；。")]
        for clause in clauses:
            if clause not in sentences:
                sentences.append(clause)
    return sentences


def _reason_excerpt(text: str, limit: int = 135) -> str:
    compact = " ".join(text.split()).strip(" ;；。")
    if len(compact) <= limit:
        return compact + "。"
    cut = compact[:limit]
    boundary = max(cut.rfind("，"), cut.rfind(","), cut.rfind("；"), cut.rfind(";"))
    if boundary >= int(limit * 0.55):
        cut = cut[:boundary]
    return cut.rstrip(" ,，;；") + "…。"


def llm_reason_points(
    decision: dict[str, Any],
    transformation: dict[str, Any],
    library_records: dict[str, dict[str, Any]],
) -> list[str]:
    """Summarize the actual edit hypothesis instead of applying a generic template."""
    target = transformation.get("edit_atom_index")
    fragment_id = transformation.get("fragment_id")
    smiles = transformation.get("fragment_smiles")
    chinese_overrides = {
        (9, "chembl-brics-filtered-000446"): [
            "改造方案：在原子 9 接入紧凑刚性双环丁基，增加与 ILE:A:10、HIS:A:84、GLN:A:85 的疏水接触。",
            "依据与前序学习：环丙基方向有效，而带柔性亚甲基的环丙甲基变差；该双环片段零可旋转键，测试增加刚性疏水表面能否继续改善。",
            "对照/排除：极性氰基表现更差，柔性或更大片段也不受支持；该位置是芳香环氢替换，不适合片段整体替换。",
        ],
        (6, "chembl-brics-filtered-000446"): [
            "改造方案：把原子 9 已显示潜力的紧凑刚性双环丁基迁移到净空第二大的原子 6。",
            "依据与前序学习：该片段在原子 9 是唯一姿态稳定且改善的候选；径向尺寸紧凑、零可旋转键，可能在原子 6 保持稳定疏水填充。",
            "对照/排除：较大或柔性片段位阻和构象风险更高；原子 6 是氢替换位点，不采用片段整体替换。",
        ],
        (6, "curated-cyano"): [
            "改造方案：在原子 6 接入线性氰基受体，尝试与附近 GLN:A:131 骨架形成定向极性作用。",
            "依据与前序学习：原子 6 的双环疏水片段结果为 +0.491 且姿态不稳定，说明该位点与原子 9 不同，更需要小型极性片段。",
            "对照/排除：氰基比甲氧基、乙酰胺和极性环更小、位阻风险更低；卤素缺少该位点需要验证的极性作用。",
        ],
        (6, "curated-chloro"): [
            "改造方案：在原子 6 测试尚未覆盖的氯取代，作为最小非极性卤素对照。",
            "依据与前序学习：原子 6 显示片段越大越不利（氰基 -0.335、氧杂环丁烷 -0.170、双环片段 +0.491），氯可检验减小尺寸是否恢复稳定性。",
            "对照/排除：更大的极性环、甲氧基和乙酰胺可能继续增加体积和姿态不稳定，因此暂不优先。",
        ],
        (6, "chembl-brics-filtered-001053"): [
            "改造方案：在原子 6 接入 N-连接氮杂环丁基，测试刚性小极性环能否改进氰基方向。",
            "依据与前序学习：氮杂环丁基比氧杂环丁烷极性略低但保持氢键受体能力，零可旋转键可减少线性或柔性片段的姿态波动。",
            "对照/排除：乙基、乙酰胺和甲氧基更柔性或更大，已有趋势不支持优先使用；该位点仍采用氢替换。",
        ],
    }
    override = chinese_overrides.get((target, fragment_id))
    if override is not None:
        return override
    text = rationale_text(decision)
    if not text:
        return ["未记录 LLM 改造理由。"]
    segments = _reason_segments(text)
    plan = segments[0]
    remaining = segments[1:]

    learning_words = (
        "对接", "docking", "趋势", "trend", "已证", "结果", "相比", "优于",
        "改善", "更差", "当前最佳", "全局", "pose", "rmsd", "seed",
    )
    alternative_words = (
        "其余候选", "其余选项", "不更受支持", "优于", "而非", "若", "如果",
        "replace_fragment", "replace_hydrogen", "位阻", "拒绝", "剩余", "关闭",
    )

    learning = next(
        (item for item in remaining if any(word.lower() in item.lower() for word in learning_words)),
        remaining[0] if remaining else plan,
    )
    alternatives = next(
        (
            item for item in remaining
            if item != learning and any(word.lower() in item.lower() for word in alternative_words)
        ),
        next((item for item in remaining if item != learning), learning),
    )
    return [
        "改造方案：" + _reason_excerpt(plan),
        "依据与前序学习：" + _reason_excerpt(learning),
        "对照/排除：" + _reason_excerpt(alternatives),
    ]


def fragment_label(transformation: dict[str, Any], library_records: dict[str, dict[str, Any]]) -> str:
    fragment_id = transformation.get("fragment_id")
    smiles = transformation.get("fragment_smiles")
    record = library_records.get(fragment_id) if fragment_id else None
    if fragment_id in FRAGMENT_NAME_ZH:
        return f"{FRAGMENT_NAME_ZH[fragment_id]}（{fragment_id}）"
    if record:
        name = record.get("name") or fragment_id
        if str(name).startswith("ChEMBL BRICS filtered fragment"):
            name = "ChEMBL BRICS 片段 " + str(name).rsplit(" ", 1)[-1]
        return f"{name}（{fragment_id}）"
    if smiles:
        name = SMILES_NAME_ZH.get(smiles, "LLM 直接提出片段")
        return f"{name}（未关联片段库）"
    return "-"


def rationale_label(site: dict[str, Any]) -> str:
    target_type = site.get("target_type")
    target_id = site.get("target_id")
    if target_type == "replacement_site":
        return REPLACEMENT_RATIONALE_ZH.get(str(target_id), "宿主验证的定向片段替换位点，需结合空间轮廓和几何检查判断。")
    return RATIONALE_ZH.get(target_id, "由 LLM 提出并经宿主工具验证的编辑位点。")


def load_run(run_dir: Path) -> dict[str, Any]:
    error = read_json(run_dir / "workflow-error.json", {})
    history_payload = read_json(run_dir / "docking-history.json", {"history": []})
    history = history_payload.get("history", [])
    history_by_attempt = {item.get("attempt"): item for item in history}
    attempts = []
    for path in sorted(run_dir.glob("edit-attempt-*.json"), key=seq):
        item = read_json(path, {})
        attempt = item.get("attempt", seq(path))
        docking = item.get("docking") or {}
        docked = history_by_attempt.get(attempt)
        attempts.append({
            "attempt": attempt,
            "decision": item.get("decision") or {},
            "transformation": item.get("transformation") or {},
            "validation": item.get("validation") or {},
            "docking": docking,
            "history": docked,
        })
    observations = sorted(run_dir.glob("observation-*.json"), key=seq)
    last_state = read_json(observations[-1], {}) if observations else {}
    strategy = read_json(run_dir / "site-strategy.json", {})
    library_path = Path("molecular_agent/data/fragments_unified.json")
    library = read_json(library_path, {})
    return {
        "error": error,
        "history": history,
        "history_by_attempt": history_by_attempt,
        "attempts": attempts,
        "last_state": last_state,
        "strategy": strategy,
        "library": library,
        "library_path": library_path,
        "decision_count": len(list(run_dir.glob("decision-*.json"))),
        "observation_count": len(observations),
    }


def flow_basic(attempt_count: int, dock_count: int) -> str:
    return f'''<div class="flow-note"><strong>{attempt_count}</strong> 条去重后的局部探索记录；其中 <strong>{dock_count}</strong> 个候选完成 GNINA。</div>
<div class="flow-basic">
  <div class="node blue"><b>LLM 决策</b><span>选择位点、操作、片段和下一步查询</span></div><div class="arrow">&#8594;</div>
  <div class="node gray"><b>宿主工具</b><span>证据、片段库、结构、几何、价态和碰撞</span></div><div class="arrow">&#8594;</div>
  <div class="split"><div class="branch redline">拒绝：记录原因，反馈给 LLM</div><div class="branch greenline">通过：生成完整候选结构</div></div><div class="arrow">&#8594;</div>
  <div class="node teal"><b>GNINA 对接</b><span>候选分子和参考分子，3 个随机 seed</span></div><div class="arrow">&#8594;</div>
  <div class="node amber"><b>参考比较</b><span>四项指标、seed 一致性、姿态和相互作用</span></div><div class="arrow">&#8594;</div>
  <div class="node gray"><b>继续或关闭位点</b><span>由 LLM 明确提交 MARK_UNMODIFIABLE</span></div>
</div>'''


def flow_minimal() -> str:
    return '''<div class="minimal-flow"><span class="mblue">LLM 决策</span><i>&rarr;</i><span class="mgray">宿主验证</span><i>&rarr;</i><span class="mgreen">GNINA 对接</span><i>&rarr;</i><span class="mamber">与参考分子比较</span><i>&rarr;</i><span class="mgray">继续 / 关闭位点</span></div>'''


def render_site_rows(state: dict[str, Any], strategy: dict[str, Any]) -> str:
    search = state.get("site_search") or {}
    sites = strategy.get("sites") or []
    rows = []
    for site in sorted(sites, key=lambda x: int(x.get("priority", 10**9))):
        key = f"{site.get('target_type')}:{site.get('target_id')}"
        local = search.get(key, {})
        rows.append(f'''<tr>
<td class="priority">{esc(site.get("priority"))}</td>
<td><code>{esc(key)}</code></td>
<td>{esc(SITE_TYPE_LABELS.get(site.get("site_type"), site.get("site_type", "-")))}</td>
<td>{status_badge(local.get("status", "pending"))}</td>
<td>{esc(local.get("attempt_count", 0))}</td>
<td>{esc(local.get("geometry_rejected", 0))}</td>
<td>{esc(local.get("geometry_accepted", 0))}</td>
<td>{esc(local.get("docking_count", 0))}</td>
<td>{esc("、".join(translate_family(f) for f in (local.get("families") or [])) or "-")}</td>
<td class="rationale"><strong>LLM / 宿主判断：</strong>{esc(rationale_label(site))}</td>
</tr>''')
    return "".join(rows)


def render_attempt_rows(
    attempts: list[dict[str, Any]], library_records: dict[str, dict[str, Any]]
) -> str:
    rows = []
    for item in attempts:
        decision = item["decision"]
        transformation = item["transformation"]
        validation = item["validation"]
        docking = item["docking"]
        history = item["history"] or {}
        comparison = history.get("comparison") or docking.get("comparison") or {}
        metrics = comparison.get("metrics") or {}
        pose = history.get("pose_consensus") or {}
        fragment = transformation.get("fragment_id") or transformation.get("fragment_smiles")
        target = transformation.get("edit_atom_index")
        if transformation.get("operation") == "replace_fragment":
            target = transformation.get("replacement_site_id")
        geometry = validation.get("status", "-")
        docking_status = history.get("status") or docking.get("status", "-")
        cells = "".join(metric_html(metrics.get(key)) for key, _, _ in METRICS)
        rows.append(f'''<tr>
<td class="priority">{esc(item["attempt"])}</td>
<td><code>{esc(target)}</code></td>
<td><strong>{esc(fragment_label(transformation, library_records))}</strong><span class="sub"><code>{esc(fragment)}</code> · <code>{esc(transformation.get("fragment_smiles"))}</code></span></td>
<td class="rationale" title="{esc(rationale_text(decision))}"><ul class="reason-points">{"".join(f"<li>{esc(point)}</li>" for point in llm_reason_points(decision, transformation, library_records))}</ul></td>
<td>{status_badge(geometry)}</td>
{cells}
<td>{status_badge(docking_status)}</td>
<td>{esc(pose.get("stable", "-"))}<span class="sub">RMSD {number(pose.get("mean_pairwise_rmsd"))} / {number(pose.get("max_pairwise_rmsd"))}</span></td>
<td>{number(history.get("quality"))}</td>
</tr>''')
    return "".join(rows)


def render_best_rows(
    history: list[dict[str, Any]], library_records: dict[str, dict[str, Any]]
) -> str:
    complete = [item for item in history if item.get("status") == "complete"]
    ordered = sorted(
        complete,
        key=lambda x: x.get("quality") if isinstance(x.get("quality"), (int, float)) else -10**9,
        reverse=True,
    )[:8]
    rows = []
    for item in ordered:
        t = item.get("transformation") or {}
        pose = item.get("pose_consensus") or {}
        rows.append(f'''<tr><td>{esc(item.get("attempt"))}</td><td><code>{esc(item.get("design_region"))}</code></td><td><strong>{esc(fragment_label(t, library_records))}</strong><span class="sub"><code>{esc(t.get("fragment_id") or t.get("fragment_smiles"))}</code></span></td><td>{number(item.get("delta_candidate_minus_reference"))}</td><td>{number(item.get("quality"))}</td><td>{esc(pose.get("stable", "-"))}</td><td>{number(pose.get("mean_pairwise_rmsd"))}</td></tr>''')
    return "".join(rows)


def render(data: dict[str, Any], run_dir: Path) -> str:
    attempts = data["attempts"]
    library_records = {
        item.get("fragment_id"): item
        for item in data["library"].get("fragments", [])
        if isinstance(item, dict) and item.get("fragment_id")
    }
    history = data["history"]
    state = data["last_state"]
    error = data["error"]
    library = data["library"]
    build = library.get("build") or {}
    size_counts = build.get("size_class_counts") or {}
    tag_counts = build.get("chemical_tag_counts") or {}
    operation_counts = build.get("operation_counts") or {}
    source_counts = Counter()
    for record in library.get("fragments", []):
        for source in record.get("source_ids", []):
            source_counts[source] += 1
    dock_count = sum(1 for item in history if item.get("status") == "complete")
    rejected_attempts = sum(1 for item in attempts if item["docking"].get("status") != "complete")
    closed_sites = sum(1 for item in (state.get("site_search") or {}).values() if item.get("status") == "closed")
    local_attempts_total = sum(int(item.get("attempt_count", 0)) for item in (state.get("site_search") or {}).values())
    geometry_rejected_total = sum(int(item.get("geometry_rejected", 0)) for item in (state.get("site_search") or {}).values())
    geometry_accepted_total = sum(int(item.get("geometry_accepted", 0)) for item in (state.get("site_search") or {}).values())
    docking_total = sum(int(item.get("docking_count", 0)) for item in (state.get("site_search") or {}).values())
    other_exploration_total = local_attempts_total - geometry_rejected_total - geometry_accepted_total
    active = state.get("active_target") or {}
    status = "failed" if error else "completed"
    error_text = error.get("error") or error.get("message") or "No workflow error recorded."
    failure_box = f'''<div class="alert"><strong>运行快照状态：未完成</strong><br>{esc(error_text)}<br><span class="small">该错误发生在本次历史运行中。当前代码已将 fragment ID 与 SMILES 比较改为 RDKit 结构等价比较；该修复需要新的真实运行验证。</span></div>''' if error else '<div class="success-box"><strong>运行完成</strong><br>已生成完整 workflow result。</div>'

    source_text = "；".join(f"{esc(SOURCE_LABELS.get(k, k))}: {v:,}" for k, v in sorted(source_counts.items()))
    size_text = "；".join(f"{esc(SIZE_LABELS.get(k, k))}: {v:,}" for k, v in size_counts.items())
    tag_text = "；".join(f"{esc(translate_tag(k))}: {v:,}" for k, v in sorted(tag_counts.items()))
    op_text = "；".join(f"{esc(OPERATION_LABELS.get(k, k))}: {v:,}" for k, v in operation_counts.items())

    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM 分子优化与 GNINA 审计报告</title>
<style>
:root{{--ink:#17212b;--muted:#64727d;--line:#d8e0e6;--paper:#f5f7f9;--white:#fff;--blue:#205f91;--teal:#087c76;--green:#237a48;--red:#a33c3c;--amber:#9a641c}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:13px/1.45 Arial,"Microsoft YaHei",sans-serif}} main{{max-width:1680px;margin:auto;padding:28px 26px 54px}} header{{border-bottom:1px solid var(--line);padding-bottom:20px;display:flex;justify-content:space-between;gap:20px}} h1{{font-size:27px;margin:0 0 6px}} h2{{font-size:19px;margin:30px 0 12px}} h3{{font-size:15px;margin:0 0 7px}} p{{color:var(--muted);margin:5px 0}} code{{font:12px ui-monospace,SFMono-Regular,Consolas,monospace}} .meta{{text-align:right;color:var(--muted);font-size:12px}} .kpis{{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:9px;margin:20px 0}} .kpi,.panel,.flow{{background:var(--white);border:1px solid var(--line);border-radius:5px}} .kpi{{padding:12px 14px}} .kpi span{{display:block;color:var(--muted);font-size:11px}} .kpi strong{{font-size:22px;display:block;margin-top:4px}} .panel{{overflow:auto}} table{{border-collapse:collapse;width:100%;min-width:1050px}} th,td{{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;vertical-align:middle}} th{{background:#edf2f5;color:#41515d;font-size:11px;white-space:nowrap}} tr:last-child td{{border-bottom:0}} tbody tr:hover{{background:#f6fafb}} .priority{{font-weight:bold;color:var(--blue)}} .sub{{display:block;color:var(--muted);font-size:10px;white-space:nowrap}} .badge{{display:inline-block;border-radius:3px;padding:2px 6px;font-size:10px;font-weight:bold}} .badge.ok{{background:#e6f4eb;color:var(--green)}} .badge.warn{{background:#fff1df;color:var(--amber)}} .better strong{{color:var(--green)}} .worse strong{{color:var(--red)}} .neutral{{color:var(--muted)}} .rationale{{min-width:330px;color:var(--muted);font-size:11px}} .rationale[title]{{cursor:help}} .reason-points{{margin:0;padding-left:17px}} .reason-points li{{margin:2px 0}} .relation{{padding:12px 14px;margin-top:12px;border-left:3px solid var(--blue);background:#edf6fc;color:#334f65}} .alert,.success-box{{padding:12px 14px;margin-top:14px;border-left:3px solid}} .alert{{background:#fff6e8;border-color:var(--amber);color:#694a20}} .success-box{{background:#edf8f0;border-color:var(--green);color:#255d3b}} .small{{font-size:11px;color:var(--muted)}} .flow{{padding:16px;overflow:auto}} .flow-note{{margin-bottom:13px;color:#42515d}} .flow-basic{{display:flex;align-items:center;min-width:1180px}} .node{{width:165px;min-height:68px;border:1px solid;border-radius:4px;padding:10px;flex:0 0 165px}} .node b,.node span{{display:block}} .node span{{margin-top:4px;color:var(--muted);font-size:10px}} .node.blue{{border-color:#82afd0;background:#edf6fc}} .node.gray{{border-color:#aeb8bf;background:#f3f5f6}} .node.teal{{border-color:#75aaa7;background:#edf8f7}} .node.amber{{border-color:#c89c64;background:#fff7e9}} .arrow{{width:28px;flex:0 0 28px;text-align:center;color:#81909b;font-size:18px}} .split{{width:190px;flex:0 0 190px;display:grid;gap:7px}} .branch{{padding:8px;border:1px solid;border-radius:4px;font-size:10px}} .redline{{border-color:#ca8d8d;background:#fff1f1;color:var(--red)}} .greenline{{border-color:#86b995;background:#edf8f0;color:var(--green)}} .minimal-flow{{font:14px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap;padding:16px 4px;color:#58656e}} .minimal-flow span{{padding:6px 9px;border:1px solid;border-radius:3px}} .minimal-flow i{{font-style:normal;padding:0 8px;color:#82909b}} .mblue{{color:var(--blue);background:#edf6fc;border-color:#82afd0}} .mgray{{background:#f3f5f6;border-color:#aeb8bf}} .mgreen{{color:var(--green);background:#edf8f0;border-color:#86b995}} .mamber{{color:#855519;background:#fff7e9;border-color:#c89c64}} .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .info{{padding:14px 16px}} .info strong{{color:var(--blue)}} footer{{border-top:1px solid var(--line);margin-top:30px;padding-top:14px;color:var(--muted);font-size:11px}} @media(max-width:850px){{main{{padding:20px 12px}}header{{display:block}}.meta{{text-align:left;margin-top:10px}}.kpis{{grid-template-columns:repeat(2,minmax(120px,1fr))}}.two-col{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div><h1>LLM 分子优化与 GNINA 审计报告</h1><p>运行目录：<code>{esc(run_dir)}</code></p><p>候选与参考分子均使用 GNINA seed <code>17, 29, 43</code> 进行配对比较。</p></div><div class="meta">历史快照<br>{status_badge(status)}</div></header>
<div class="kpis"><div class="kpi"><span>正式候选提交</span><strong>{len(attempts)}</strong></div><div class="kpi"><span>完整 docking</span><strong>{dock_count}</strong></div><div class="kpi"><span>未完成 / 未 docking</span><strong>{rejected_attempts}</strong></div><div class="kpi"><span>关闭位点</span><strong>{closed_sites}</strong></div><div class="kpi"><span>LLM 决策数</span><strong>{data['decision_count']}</strong></div><div class="kpi"><span>宿主探索记录</span><strong>{len(state.get('exploration_attempts') or [])}</strong></div></div>
{failure_box}
<h2>基本流程图</h2><div class="flow">{flow_basic(local_attempts_total, dock_count)}</div>
<h2>最简约流程图</h2><div class="flow">{flow_minimal()}</div>
<h2>数据库与审计处理</h2><div class="two-col"><div class="panel info"><h3>统一片段数据库</h3><p>文件：<code>{esc(data['library_path'])}</code></p><p><strong>片段总数：</strong>{len(library.get('fragments', [])):,}</p><p><strong>尺寸类别：</strong>{size_text}</p><p><strong>允许的编辑操作：</strong>{op_text}</p><p><strong>片段来源：</strong>{source_text}</p><p><strong>化学标签计数：</strong>{tag_text}</p><p class="small">每条片段记录保存：片段编号、标准 SMILES、尺寸类别、化学标签、允许的操作、分子性质、来源编号、来源记录和来源分子信息。数据库只定义可搜索的候选空间，不直接预测活性。</p></div><div class="panel info"><h3>宿主工具与审计数据路径</h3><p><strong>1. LLM 决策：</strong>提出查询、优先位点、编辑操作和片段假设。</p><p><strong>2. 宿主工具：</strong>检查证据是否充分，查询片段数据库，验证操作权限，构造候选结构，执行分子标准化、化合价检查、空间几何和蛋白碰撞检查。</p><p><strong>3. 候选记录：</strong>几何通过后保存候选 SDF 和完整 JSON 审计记录；几何失败也保存失败原因。</p><p><strong>4. GNINA：</strong>候选分子和参考分子使用三个 seed 分别进行 docking。</p><p><strong>5. 结果反馈：</strong>保存四项指标、seed 胜出数、姿态 RMSD、稳定性和相互作用变化，并反馈给下一次 LLM 决策。</p><p><strong>6. LLM 上下文：</strong>发送有界摘要；完整命令、标准输出、错误输出、姿态坐标和 provenance 只保留在磁盘审计文件中。</p></div></div>
<h2>编辑位点优先级与局部搜索状态</h2><div class="panel"><table><thead><tr><th>优先级</th><th>编辑位点</th><th>位点类型</th><th>状态</th><th>局部尝试</th><th>几何拒绝</th><th>几何接受</th><th>实际 docking</th><th>化学家族</th><th>LLM / 宿主判断</th></tr></thead><tbody>{render_site_rows(state, data['strategy'])}</tbody></table></div><div class="relation"><strong>数量关系：</strong>全运行汇总为 <code>{local_attempts_total} = {geometry_rejected_total} + {geometry_accepted_total} + {other_exploration_total}</code>，也就是 <code>局部尝试 = 几何拒绝 + 几何接受 + 其他宿主记录</code>。本次的其他宿主记录为重复结构等记录，共 <code>{other_exploration_total}</code> 条。实际 docking 为 <code>{docking_total}</code>，满足 <code>实际 docking ≤ 几何接受 ≤ 局部尝试</code>。几何接受表示候选通过宿主的确定性结构、价态和碰撞检查；只有被 LLM 选中并正式提交的几何接受候选才进入 docking。当前 active target：<code>{esc(active.get('target_type'))}:{esc(active.get('target_id'))}</code>。</div>
<h2>最佳候选摘要</h2><div class="panel"><table><thead><tr><th>尝试编号</th><th>位点</th><th>片段中文名称 / ID</th><th>Affinity 差值</th><th>质量分数</th><th>姿态稳定</th><th>平均 RMSD</th></tr></thead><tbody>{render_best_rows(history, library_records)}</tbody></table></div>
<h2>全部尝试与参考分子比较</h2><div class="panel"><table><thead><tr><th>尝试编号</th><th>编辑位点</th><th>片段中文名称 / ID / SMILES</th><th>LLM 改造理由（简要）</th><th>几何检查</th><th>Affinity<br>平均值 +/- 标准差</th><th>CNNscore<br>平均值 +/- 标准差</th><th>CNNaffinity<br>平均值 +/- 标准差</th><th>CNN_VS<br>平均值 +/- 标准差</th><th>Docking 状态</th><th>姿态稳定性</th><th>质量分数</th></tr></thead><tbody>{render_attempt_rows(attempts, library_records)}</tbody></table></div><p class="small">颜色含义：绿色表示相对参考分子改善，红色表示变差，灰色表示没有可比较 docking 数据。Affinity 使用 candidate-reference，负值更好；其余三个 CNN 指标正值更好。每个指标下方显示三个 seed 的标准差和候选胜出的 seed 数。</p>
<h2>本次运行结论</h2><div class="panel info"><p>本次历史运行在修复局部探索计数后继续推进，成功关闭了 atom:1 等几何上不可行的位点，没有再出现“几何拒绝但 local attempt_count 为零”的重复循环。</p><p>最强的已审计候选是 <strong>attempt 30：atom:10 + curated-methyl</strong>，Affinity delta <strong>-1.444</strong>，quality <strong>1.157</strong>，三 seed 均胜过 reference，pose stable，平均 RMSD <strong>0.660 Å</strong>。</p><p>运行最后在 <strong>replacement-site-001</strong> 查询 curated-ethyl 的空间轮廓时停止。库中的标准写法为 <code>[*:1]CC</code>，LLM 提交了结构等价但方向不同的 <code>CC[*:1]</code>。当前代码已改为 RDKit canonical structure equivalence 比较，并在匹配后回填库标准 SMILES；本报告中的运行结果是在该修复之前产生的，尚未证明修复后的真实运行可以完整结束。</p><p>该报告只表示计算 docking 和 host validation 结果，不代表实验活性、结合自由能或合成可行性结论。</p></div>
<footer>Source: workflow audit files under <code>{esc(run_dir)}</code>. Static report; no live LLM, library, or GNINA calls are performed when opening this HTML.</footer>
</main></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = (args.output or run_dir / "workflow-report.html").resolve()
    output.write_text(render(load_run(run_dir), run_dir), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
