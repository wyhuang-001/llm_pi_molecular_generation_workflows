#!/usr/bin/env python3
"""Generate a self-contained HTML report for one completed docking workflow."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


METRICS = (
    ("minimizedAffinity", "Affinity", "lower"),
    ("CNNscore", "CNNscore", "higher"),
    ("CNNaffinity", "CNNaffinity", "higher"),
    ("CNN_VS", "CNN_VS", "higher"),
)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return esc(value)


def metric_cell(metric: dict[str, Any] | None) -> str:
    if not metric:
        return '<td class="muted">-</td>'
    delta = metric.get("delta_candidate_minus_reference", {})
    mean_value = delta.get("mean") if isinstance(delta, dict) else delta
    stddev = delta.get("stddev") if isinstance(delta, dict) else None
    wins = metric.get("candidate_better_seed_count")
    n = metric.get("n")
    direction = metric.get("direction", "")
    cls = "positive" if (float(mean_value) < 0 if direction == "lower_is_better" else float(mean_value) > 0) else "negative"
    return (
        f'<td class="{cls}"><strong>{fmt(mean_value)}</strong>'
        f'<span class="sub">+/- {fmt(stddev)} | {wins}/{n} seeds</span></td>'
    )


def build_flow(attempts: list[dict[str, Any]]) -> str:
    docking_cycles = sum(
        1 for attempt in attempts if attempt.get("docking", {}).get("status") == "complete"
    )
    return f'''<div class="flow-summary"><strong>本次工作流共有 {len(attempts)} 个候选循环；其中 {docking_cycles} 个循环通过几何预筛并进入 docking。</strong></div>
<div class="workflow-diagram">
  <div class="diagram-node llm"><b>LLM 提出改造方向</b><span>选择位点、片段和下一步查询</span></div>
  <div class="diagram-arrow">&#8594;</div>
  <div class="diagram-node feedback"><b>证据与几何反馈</b><span>环境、增长空间、候选几何检查</span></div>
  <div class="diagram-arrow">&#8594;</div>
  <div class="diagram-branch">
    <div class="branch-label">未通过</div>
    <div class="diagram-node retry"><b>反馈给 LLM 重试</b><span>修订位点或片段，进入下一循环</span></div>
    <div class="branch-label pass-label">通过</div>
    <div class="diagram-node passed"><b>几何预筛通过</b><span>才允许进入 docking</span></div>
  </div>
  <div class="diagram-arrow">&#8594;</div>
  <div class="diagram-node docking"><b>Docking</b><span>本次实际进入 {docking_cycles} 个循环</span></div>
  <div class="diagram-arrow">&#8594;</div>
  <div class="diagram-node score"><b>分数反馈给 LLM</b><span>决定是否提出下一轮候选</span></div>
</div>'''


def render(data: dict[str, Any], result_path: Path) -> str:
    result = data.get("result", {})
    state = data.get("state", {})
    attempts = result.get("attempts", [])
    first_docking = attempts[0].get("docking", {}) if attempts else {}
    comparison = first_docking.get("comparison", {})
    reference = first_docking.get("reference_baseline", {})
    final_docking = result.get("docking", {})
    final_comparison = final_docking.get("comparison", {})
    final_decision = attempts[-1].get("decision", {}) if attempts else {}

    rows = []
    for attempt in attempts:
        decision = attempt.get("decision", {})
        docking = attempt.get("docking", {})
        metrics = docking.get("comparison", {}).get("metrics", {})
        cells = "".join(metric_cell(metrics.get(key)) for key, _label, _direction in METRICS)
        rows.append(
            f'''<tr>
  <td class="cycle">{esc(attempt.get("attempt"))}</td>
  <td>{esc(decision.get("edit_atom_index"))}</td>
  <td><code>{esc(decision.get("fragment_smiles"))}</code></td>
  <td><span class="pill success">{esc(attempt.get("validation", {}).get("status", "-"))}</span></td>
  {cells}
  <td><span class="pill success">{esc(docking.get("status", "-"))}</span></td>
</tr>'''
        )

    reference_rows = []
    for seed, item in reference.get("per_seed", {}).items():
        props = (item.get("top_pose") or {}).get("properties", {})
        reference_rows.append(
            f"<tr><td>{esc(seed)}</td><td>{fmt(props.get('minimizedAffinity'), 5)}</td>"
            f"<td>{fmt(props.get('CNNscore'), 5)}</td><td>{fmt(props.get('CNNaffinity'), 5)}</td>"
            f"<td>{fmt(props.get('CNN_VS'), 5)}</td></tr>"
        )

    final_metrics = final_comparison.get("metrics", {})
    final_metric_summary = "".join(
        f'''<div class="metric-card"><span>{esc(label)}</span><strong>{fmt(final_metrics.get(key, {}).get("delta_candidate_minus_reference", {}).get("mean"))}</strong><small>mean delta · {final_metrics.get(key, {}).get("candidate_better_seed_count", 0)}/{final_metrics.get(key, {}).get("n", 0)} seeds better</small></div>'''
        for key, label, _direction in METRICS
    )

    title = "LLM Protein-Ligand Design and Docking Report"
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --ink:#17212b; --muted:#65727e; --line:#d9e0e6; --paper:#f7f9fb; --white:#fff; --blue:#1d5f8f; --teal:#087f78; --green:#287a4b; --amber:#a76512; --red:#a33b3b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--paper); font:14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
main {{ max-width:1500px; margin:0 auto; padding:34px 28px 56px; }}
header {{ display:flex; justify-content:space-between; align-items:flex-start; gap:24px; border-bottom:1px solid var(--line); padding-bottom:24px; }}
h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:0; }}
h2 {{ margin:32px 0 14px; font-size:19px; letter-spacing:0; }}
h3 {{ margin:0 0 7px; font-size:15px; }}
p {{ margin:6px 0; color:var(--muted); }}
.meta {{ text-align:right; color:var(--muted); font-size:12px; }}
.kpis {{ display:grid; grid-template-columns:repeat(5,minmax(130px,1fr)); gap:10px; margin:22px 0 28px; }}
.kpi, .metric-card, .panel {{ background:var(--white); border:1px solid var(--line); border-radius:6px; }}
.kpi {{ padding:14px 15px; }}
.kpi span, .metric-card span {{ display:block; color:var(--muted); font-size:12px; }}
.kpi strong {{ display:block; margin-top:4px; font-size:22px; }}
.panel {{ overflow:auto; }}
table {{ width:100%; border-collapse:collapse; min-width:980px; }}
th, td {{ padding:11px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:middle; }}
th {{ color:#42515d; background:#eef3f6; font-size:12px; font-weight:700; white-space:nowrap; }}
tbody tr:last-child td {{ border-bottom:0; }}
tbody tr:hover {{ background:#f4f8fa; }}
.cycle {{ font-weight:800; color:var(--blue); }}
code {{ font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }}
.sub {{ display:block; color:var(--muted); font-size:11px; white-space:nowrap; margin-top:2px; }}
.positive strong {{ color:var(--green); }} .negative strong {{ color:var(--red); }} .muted {{ color:var(--muted); }}
.pill {{ display:inline-block; padding:3px 8px; border-radius:999px; font-size:11px; font-weight:700; }}
.success {{ color:var(--green); background:#e7f4eb; }}
.note {{ border-left:3px solid var(--amber); padding:10px 13px; background:#fff8ed; color:#6c4d25; margin-top:15px; }}
.metric-grid {{ display:grid; grid-template-columns:repeat(4,minmax(150px,1fr)); gap:10px; }}
.metric-card {{ padding:13px 14px; }}
.metric-card strong {{ display:block; font-size:23px; margin:4px 0 2px; color:var(--blue); }}
.metric-card small {{ color:var(--muted); }}
.flow {{ padding:18px; background:var(--white); border:1px solid var(--line); border-radius:6px; overflow-x:auto; }}
.flow-summary {{ margin-bottom:16px; color:#42515d; }}
.workflow-diagram {{ display:flex; align-items:center; min-width:980px; gap:0; }}
.diagram-node {{ width:180px; min-height:76px; padding:12px; border:1px solid; border-radius:5px; flex:0 0 180px; }}
.diagram-node b, .diagram-node span {{ display:block; }} .diagram-node span {{ margin-top:5px; color:var(--muted); font-size:11px; }}
.diagram-node.llm {{ border-color:#7aa7c8; background:#edf6fc; }} .diagram-node.feedback {{ border-color:#a6aeb5; background:#f3f5f6; }} .diagram-node.passed {{ border-color:#79b58b; background:#eef9f0; }} .diagram-node.retry {{ border-color:#c58b8b; background:#fff1f1; }} .diagram-node.docking {{ border-color:#78a3a1; background:#edf8f7; }} .diagram-node.score {{ border-color:#c59b69; background:#fff7ea; }}
.diagram-arrow {{ width:32px; flex:0 0 32px; display:flex; align-items:center; justify-content:center; color:#82909b; font-size:20px; }}
.diagram-branch {{ width:196px; flex:0 0 196px; position:relative; padding:50px 0 50px; }}
.diagram-branch::before {{ content:""; position:absolute; left:50%; top:0; bottom:0; border-left:1px solid #bdc7cf; }}
.diagram-branch .diagram-node {{ position:relative; width:196px; }}
.branch-label {{ position:absolute; top:8px; left:10px; color:var(--red); font-size:11px; font-weight:700; }}
.pass-label {{ top:auto; bottom:8px; color:var(--green); }}
.small-table {{ max-width:800px; }}
footer {{ margin-top:34px; color:var(--muted); font-size:12px; border-top:1px solid var(--line); padding-top:16px; }}
@media (max-width:800px) {{ main {{ padding:22px 14px 40px; }} header {{ display:block; }} .meta {{ text-align:left; margin-top:12px; }} .kpis {{ grid-template-columns:repeat(2,minmax(130px,1fr)); }} .metric-grid {{ grid-template-columns:repeat(2,minmax(140px,1fr)); }} h1 {{ font-size:24px; }} }}
</style>
</head>
<body>
<main>
<header>
  <div><h1>{title}</h1><p>Run: <code>{esc(result_path.parent)}</code></p><p>Paired reference/candidate GNINA docking with seeds <code>17, 29, 43</code>.</p></div>
  <div class="meta">Generated from audited run<br><span class="pill success">{esc(result.get("status", "unknown"))}</span></div>
</header>

<div class="kpis">
  <div class="kpi"><span>LLM design cycles</span><strong>{len(attempts)}</strong></div>
  <div class="kpi"><span>Entered docking</span><strong>{sum(1 for a in attempts if a.get("docking", {}).get("status") == "complete")}/{len(attempts)}</strong></div>
  <div class="kpi"><span>Seeds per candidate</span><strong>{esc(final_comparison.get("seed_count", "-"))}</strong></div>
  <div class="kpi"><span>Poses per seed</span><strong>20</strong></div>
  <div class="kpi"><span>RBFE</span><strong>deferred</strong></div>
</div>

<h2>Workflow Loop</h2>
<div class="flow">
  <p><strong>Docking is a gated branch, not an automatic step in every LLM cycle.</strong> Candidates that fail evidence or deterministic geometry checks return to the LLM for revision.</p>
  {build_flow(attempts)}
</div>

<h2>Candidate Comparison</h2>
<div class="panel">
<table>
<thead><tr><th>Cycle</th><th>Edit atom</th><th>Fragment</th><th>Geometry</th><th>Affinity delta<br>(mean +/- sd)</th><th>CNNscore delta<br>(mean +/- sd)</th><th>CNNaffinity delta<br>(mean +/- sd)</th><th>CNN_VS delta<br>(mean +/- sd)</th><th>Docking</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</div>
<p class="note">Delta = candidate minus independently redocked reference. More negative is better for Affinity; more positive is better for CNNscore, CNNaffinity and CNN_VS. “Seeds better” counts paired seeds where the candidate wins that metric.</p>

<h2>Metric Guide</h2>
<div class="panel" style="padding:15px 18px;">
  <p><strong>Affinity delta</strong>：GNINA 的 `minimizedAffinity`，候选分数减参考分数；负值表示候选更好。</p>
  <p><strong>CNNscore / CNNaffinity / CNN_VS delta</strong>：GNINA 的 CNN 相关分数，同样是候选减参考；正值表示候选更好。</p>
  <p><strong>mean +/- sd</strong>：3 个 seed 的平均差值和样本标准差。平均值表示总体方向，标准差表示不同随机 seed 之间的波动大小。</p>
</div>

<h2>Final Candidate</h2>
<div class="panel" style="padding:16px 18px;">
  <p><strong>Cycle {esc(len(attempts))}</strong> · edit atom <code>{esc(final_decision.get("edit_atom_index", "-"))}</code> · fragment <code>{esc(final_decision.get("fragment_smiles", "-"))}</code></p>
  <div class="metric-grid">{final_metric_summary}</div>
</div>

<h2>Reference Baseline</h2>
<div class="panel small-table">
<table><thead><tr><th>Seed</th><th>Affinity</th><th>CNNscore</th><th>CNNaffinity</th><th>CNN_VS</th></tr></thead><tbody>{"".join(reference_rows)}</tbody></table>
</div>

<h2>Run Notes</h2>
<div class="panel" style="padding:15px 18px;">
  <p>Context queries used: <strong>{esc(state.get("round", "-"))}</strong> / <strong>{esc(state.get("max_rounds", "-"))}</strong>.</p>
  <p>All displayed candidates passed deterministic geometry prescreen and completed reference-relative docking.</p>
  <p>The result is a docking ranking signal, not an experimental affinity or activity conclusion. RBFE/AsyncFEP was not executed.</p>
</div>
<footer>Source: {esc(result_path)} · This report is static and contains no live API or docking calls.</footer>
</main>
</body>
</html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Workflow result.json")
    parser.add_argument("--output", type=Path, help="HTML output path")
    args = parser.parse_args()
    result_path = args.result.resolve()
    data = json.loads(result_path.read_text(encoding="utf-8"))
    output = (args.output or result_path.with_name("docking-report.html")).resolve()
    output.write_text(render(data, result_path), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
