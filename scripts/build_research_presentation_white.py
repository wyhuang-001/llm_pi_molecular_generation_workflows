"""Build the compact white-background revision; never overwrite the original deck.

Data and chemical structures come from the same runs used by the original deck.
Workflow diagrams, plots, and all slide text are native editable PowerPoint objects.
The generated cover illustration is conceptual, not structural evidence.
"""
from pathlib import Path
import json
import math

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets' / 'ppt_white'
OUTDIR = ROOT / 'deliverables'
ASSETS.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / '科研汇报_保留骨架的分子优化智能体_白底紧凑版.pptx'
LLM_RUN = ROOT / 'runs' / '4wkq-llm-20260927-193317'
RANDOM_RUN = ROOT / 'runs' / '4wkq-random-baseline-v1'
llm = json.loads((LLM_RUN / 'result.json').read_text())
random = json.loads((RANDOM_RUN / 'result.json').read_text())
calibration = json.loads((LLM_RUN / 'docking-reference-baseline' / 'calibration.json').read_text())
history = llm['state']['docking_history']
best_attempt = llm['result']['best_attempt']
best = next(row for row in history if row['attempt'] == best_attempt)
random_best = next(row for row in random['state']['docking_history']
                   if row['attempt'] == random['result']['best_attempt'])

# Exact retained atom membership rather than MCS-derived partial highlighting.
def draw_structures():
    ref = next(m for m in Chem.SDMolSupplier(str(LLM_RUN / 'reference-ligand.sdf')) if m)
    cand = next(m for m in Chem.SDMolSupplier(str(LLM_RUN / f'candidate-{best_attempt:02d}.sdf')) if m)
    kept = best['transformation']['bond']['retained_atom_indices']
    scaffold = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(ref, atomsToUse=kept))
    ref_match = ref.GetSubstructMatch(scaffold)
    cand_match = cand.GetSubstructMatch(scaffold)
    assert len(ref_match) == len(cand_match) == 21
    rdDepictor.Compute2DCoords(ref)
    rdDepictor.GenerateDepictionMatching2DStructure(cand, ref, list(zip(ref_match, cand_match)))
    for mol, match, name in [(ref, ref_match, 'reference_ligand'), (cand, cand_match, 'candidate_06')]:
        core = set(match)
        cyan, orange = (0.83, 0.95, 0.97), (1.0, 0.86, 0.65)
        atom_colors = {a.GetIdx(): cyan if a.GetIdx() in core else orange for a in mol.GetAtoms()}
        bond_colors = {b.GetIdx(): cyan if b.GetBeginAtomIdx() in core and b.GetEndAtomIdx() in core else orange
                       for b in mol.GetBonds()}
        drawer = rdMolDraw2D.MolDraw2DCairo(1500, 820)
        opt = drawer.drawOptions()
        opt.padding = 0.065
        opt.bondLineWidth = 3.4
        opt.fixedBondLength = 66
        opt.minFontSize = 28
        opt.maxFontSize = 38
        opt.highlightBondWidthMultiplier = 12
        opt.setBackgroundColour((1, 1, 1, 1))
        drawer.DrawMolecule(mol, highlightAtoms=list(atom_colors), highlightBonds=list(bond_colors),
                            highlightAtomColors=atom_colors, highlightBondColors=bond_colors)
        drawer.FinishDrawing()
        (ASSETS / f'{name}.png').write_bytes(drawer.GetDrawingText())


draw_structures()

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333333), Inches(7.5)
FONT = 'Microsoft YaHei'
WHITE = 'FFFFFF'
BLUE = '123D79'
INK = '26384A'
MUTED = '657789'
TEAL = '129CB0'
ORANGE = 'DB8A22'
LINE = 'D9E5ED'
PALE = 'F3F8FC'
LIGHT_TEAL = 'EAF7F8'
LIGHT_ORANGE = 'FFF6E8'


def rgb(hex_value):
    return RGBColor.from_string(hex_value)


def flat(shape):
    # Avoid inherited theme shadows on shapes and lines.
    if shape._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}style') is not None:
        shape._element.remove(shape._element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}style'))
    if not shape._element.spPr.findall('{http://schemas.openxmlformats.org/drawingml/2006/main}effectLst'):
        shape._element.spPr.append(OxmlElement('a:effectLst'))
    return shape


def text(s, x, y, w, h, value, size=18, color=INK, bold=False,
         align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    sh = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = valign
    for i, line_value in enumerate(value.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = 1.1
        p.space_before = Pt(0)
        p.space_after = Pt(3)
        r = p.add_run()
        r.text = line_value
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = rgb(color)
        ea = OxmlElement('a:ea')
        ea.set('typeface', FONT)
        r._r.get_or_add_rPr().append(ea)
    return sh


def rect(s, x, y, w, h, color):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(color)
    sh.line.fill.background()
    return flat(sh)


def circle(s, x, y, d, color):
    sh = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(color)
    sh.line.fill.background()
    return flat(sh)


def line(s, x1, y1, x2, y2, color=LINE, width=1.2, arrow=False, dashed=False):
    sh = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                               Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    sh.line.color.rgb = rgb(color)
    sh.line.width = Pt(width)
    if dashed:
        from pptx.enum.dml import MSO_LINE_DASH_STYLE
        sh.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if arrow:
        end = OxmlElement('a:tailEnd')
        end.set('type', 'triangle')
        end.set('w', 'sm')
        end.set('len', 'sm')
        sh.line._get_or_add_ln().append(end)
    return flat(sh)


def picture(s, path, x, y, w, h):
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return s.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2),
                                width=Inches(pw), height=Inches(ph))


def header(section, title, subtitle, source=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(WHITE)
    text(s, .55, .24, 11.9, .23, section, 10.5, TEAL, True)
    text(s, .55, .62, 12.15, .56, title, 28, BLUE, True)
    line(s, .55, 1.31, 12.78, 1.31, LINE, 1)
    line(s, .55, 1.31, 1.43, 1.31, TEAL, 2.5)
    if subtitle:
        text(s, .58, 1.49, 12.08, .5, subtitle, 16, MUTED)
    page = len(prs.slides)
    line(s, .55, 7.02, 12.78, 7.02, LINE, .65)
    text(s, .55, 7.14, 11.6, .19, source or '近期工作进展  |  保留骨架的分子优化智能体', 8.5, MUTED)
    text(s, 12.22, 7.11, .55, .23, f'{page:02d}', 10, MUTED, align=PP_ALIGN.RIGHT)
    return s


def takeaway(s, head, body, y=6.29):
    rect(s, .58, y, .045, .46, TEAL)
    text(s, .77, y+.02, 1.45, .36, head, 17.5, BLUE, True)
    text(s, 2.21, y+.025, 10.40, .43, body, 17.5, INK)


def dotline(s, x, y, w, value, color=TEAL, size=17):
    circle(s, x, y+.105, .08, color)
    text(s, x+.20, y, w-.20, .42, value, size, INK)


def step(s, cx, y, number, title, detail, width=2.20, color=TEAL):
    circle(s, cx-.225, y, .45, LIGHT_TEAL if color == TEAL else PALE)
    text(s, cx-.225, y+.045, .45, .30, number, 12, color, True, PP_ALIGN.CENTER)
    text(s, cx-width/2, y+.67, width, .43, title, 20, BLUE, True, PP_ALIGN.CENTER)
    text(s, cx-width/2, y+1.23, width, .80, detail, 14.5, MUTED, False, PP_ALIGN.CENTER)


def notes(s, value):
    s.notes_slide.notes_text_frame.text = value


# 1 — White, compact title / framing. No dark title panel.
s = header('科研汇报  /  近期工作进展', '保留骨架的分子优化智能体',
           '结合蛋白口袋信息，让大语言模型提出可验证的局部改造方案')
text(s, .66, 2.14, 7.4, 1.02, 'LLM 提假设，计算工具做验证', 27, BLUE, True)
text(s, .69, 3.03, 6.30, .86,
     '以共晶配体为起点，只改允许的侧链；\n用结构、对接与多 seed 反馈逐步筛选。', 20, INK)
for i, (head, detail) in enumerate([
    ('研究目标', '在有限预算内，筛出可信的候选分子'),
    ('关键约束', '骨架不变、姿态保留、过程可追溯'),
    ('当前阶段', '4WKQ–EGFR / gefitinib 闭集验证'),
]):
    y = 4.26 + i*.60
    text(s, .70, y, 1.38, .37, head, 17, TEAL, True)
    text(s, 2.08, y, 5.15, .4, detail, 17, INK)
picture(s, ASSETS / 'molecular_agent_white_concept.png', 8.03, 2.02, 4.51, 4.49)
text(s, 8.14, 6.47, 4.32, .24, '概念示意图，非 4WKQ 实际结构', 10.5, MUTED, align=PP_ALIGN.CENTER)
text(s, .70, 6.48, 6.3, .28, '黄婉仪  ·  2026/9/28', 13, MUTED)
notes(s, '参考“近期工作进展.pptx”的白底蓝色标题风格。封面图由 image-gen 扩展生成，只用于概念说明，不代表真实蛋白/配体几何。研究数据沿用原汇报的 2026-09-27 运行快照。')

# 2 — Challenge-to-strategy rows, not nested cards.
s = header('01  /  研究问题', '分子优化：既要改得动，也要验证改得对',
           '真正的问题不是“生成一个分子”，而是在结构约束下有效分配计算预算。')
text(s, .73, 2.17, 5.25, .38, '优化中的困难', 20, BLUE, True)
text(s, 7.18, 2.17, 5.35, .38, '本项目的处理方式', 20, TEAL, True)
rows = [
    ('选择空间大', '侧链很多，逐一构建与对接成本高', '受控候选规划', '在合法位点与片段库内提出可检验假设'),
    ('化学合法 ≠ 结合可靠', '能接上，不代表放得进或方向正确', '保留骨架与结合姿态', '将分子图校验、口袋与锚点检查结合'),
    ('单次评分不稳定', '随机 seed 与 pose 差异会改变排序', '多 seed 配对比较', '固定参考与协议，同时考虑改善和稳定性'),
]
for i, (a,b,c,d) in enumerate(rows):
    y = 2.88 + i*1.03
    text(s, .75, y, 5.22, .34, a, 19, INK, True)
    text(s, .75, y+.41, 5.3, .38, b, 16, MUTED)
    line(s, 6.25, y+.34, 6.73, y+.34, TEAL, 1.8, arrow=True)
    text(s, 7.18, y, 5.37, .34, c, 19, BLUE, True)
    text(s, 7.18, y+.41, 5.35, .39, d, 16, INK)
    if i < 2:
        line(s, .75, y+.87, 12.54, y+.87, LINE, .7)
takeaway(s, '研究边界', '做计算筛选与排序，不把 docking 改善等同于实验活性提升。')
notes(s, '保留原版三项核心挑战，去掉钥匙配锁插图、圆角卡片和装饰边框。此页不是对实验活性或临床效果的声明。')

# 3 — High-level editable five-stage loop.
s = header('02  /  总体流程', '总体框架：从结构输入到反馈更新',
           'LLM 负责选择尝试方向；Host（宿主程序）执行化学、结构与对接验证。')
centers = [1.58, 4.12, 6.66, 9.20, 11.74]
stages = [
    ('01', '结构输入', '共晶复合物\n允许位点与片段库'),
    ('02', '候选规划', '结合口袋信息\n提出侧链编辑假设'),
    ('03', '构建验证', '精确分子图编辑\n化学与几何检查'),
    ('04', '对接评价', 'GNINA / PLIP\n姿态与多 seed 验证'),
    ('05', '反馈更新', '汇总有效与无效改动\n输出历史最佳候选'),
]
for i, (cx, values) in enumerate(zip(centers, stages)):
    step(s, cx, 2.27, *values)
    if i < 4:
        line(s, cx+.40, 2.50, centers[i+1]-.40, 2.50, TEAL, 1.8, arrow=True)
# Return line has real OOXML arrowheads, not an unsupported Python property.
line(s, 11.74, 4.27, 11.74, 4.65, TEAL, 1.6)
line(s, 11.74, 4.65, 4.12, 4.65, TEAL, 1.6)
line(s, 4.12, 4.65, 4.12, 4.23, TEAL, 1.6, arrow=True)
rect(s, 6.10, 4.47, 3.93, .34, WHITE)
text(s, 6.13, 4.49, 3.87, .29, '分数、姿态与失败原因 → 下一轮', 13.3, TEAL, True, PP_ALIGN.CENTER)
text(s, .76, 5.16, 2.0, .37, '输入有约束', 18, BLUE, True)
text(s, .76, 5.62, 3.7, .44, '固定骨架、合法切口、有限预算', 15.5, INK)
text(s, 4.87, 5.16, 2.0, .37, '执行可验证', 18, BLUE, True)
text(s, 4.87, 5.62, 3.8, .44, '不由模型直接给出结构与分数结论', 15.5, INK)
text(s, 9.16, 5.16, 2.0, .37, '输出可追溯', 18, BLUE, True)
text(s, 9.16, 5.62, 3.50, .44, '候选结构、相互作用与运行记录', 15.5, INK)
takeaway(s, '闭环要点', '只在验证结果支持时更新候选，而不是接受模型的文字判断。', y=6.39)
notes(s, '此图为项目总体逻辑，刻意省略请求类型、缓存、工具内部模块等细节。候选规划在总体框架支持批量；本次闭集 benchmark 是固定 C6 位点、逐候选提交，不宣称本次运行即为完整批量规划实验。当前主流程到 docking，RBFE 尚不是本次结果。')

# 4 — Local edit / exact structures and a short workflow.
s = header('03  /  重点流程一', '单点编辑：保留骨架，只替换允许的侧链',
           '4WKQ 示例：固定 C6 切口，在 78 条匿名侧链中选择；不做无约束骨架跳跃。')
text(s, .73, 2.07, 4.76, .39, '原始配体  /  gefitinib', 19, BLUE, True)
text(s, 7.77, 2.07, 4.87, .39, '候选 #06  /  本次运行最佳', 19, BLUE, True)
picture(s, ASSETS / 'reference_ligand.png', .61, 2.43, 4.96, 2.58)
picture(s, ASSETS / 'candidate_06.png', 7.77, 2.43, 4.96, 2.58)
text(s, 5.70, 3.14, 1.87, .4, '侧链替换', 18, TEAL, True, PP_ALIGN.CENTER)
line(s, 5.72, 3.83, 7.57, 3.83, TEAL, 2, arrow=True)
text(s, 5.72, 4.08, 1.83, .59, '21 个核心\n重原子保留', 13.3, MUTED, False, PP_ALIGN.CENTER)
circle(s, .78, 5.07, .14, 'AEDFE8')
text(s, 1.02, 5.00, 3.36, .31, '浅蓝：固定核心', 13.5, MUTED)
circle(s, 4.44, 5.07, .14, 'FFCB83')
text(s, 4.68, 5.00, 6.5, .31, '浅橙：原侧链 / 替换侧链；结构来自实际 SDF', 13.5, MUTED)
flow_x = [1.96, 5.08, 8.23, 11.38]
for i, (cx, label) in enumerate(zip(flow_x, ['确定合法切口', '选择库内片段', 'Host 精确组装', '校验结构身份'])):
    text(s, cx-1.32, 5.65, 2.64, .39, label, 18, BLUE, True, PP_ALIGN.CENTER)
    if i < 3:
        line(s, cx+1.23, 5.83, flow_x[i+1]-1.35, 5.83, TEAL, 1.7, arrow=True)
takeaway(s, '控制原则', 'LLM 选择“改哪里、接什么”；Host 决定“是否构建正确”。', y=6.36)
notes(s, '分子图来源：runs/4wkq-llm-20260927-193317/reference-ligand.sdf 与 candidate-06.sdf。保留原子集合来自 result.json 中 bond_site.retained_atom_indices，21 个重原子。使用固定核心图匹配统一二维方向；颜色仅表示图编辑区域，不表示对接姿态或实验活性。候选 #06 为此运行按稳定性资格与质量函数筛出的最佳。')

# 5 — Verification workflow with honest geometry policy.
s = header('04  /  重点流程二', '候选验证：先确认姿态可信，再比较分数',
           '参考配体先校准；候选沿同一协议评价，避免仅凭一次低分就接受。')
centers4 = [1.85, 5.04, 8.23, 11.42]
for i, (cx, vals) in enumerate(zip(centers4, [
    ('01', '化学构建', '价态、电荷\n结构身份与重复检查'),
    ('02', '对接搜索', '固定参考与协议\n多个 seed 搜索姿态'),
    ('03', '姿态门控', '核心 RMSD、碰撞\n关键氢键锚点保留'),
    ('04', '稳定性筛选', '配对分数与 seed 一致性\n更新历史最佳候选'),
])):
    step(s, cx, 2.24, *vals, width=2.55)
    if i < 3:
        line(s, cx+.43, 2.47, centers4[i+1]-.43, 2.47, TEAL, 1.8, arrow=True)
line(s, .75, 4.47, 12.58, 4.47, LINE, .9)
text(s, .76, 4.78, 5.60, .40, '失败也要保留原因', 19, BLUE, True)
dotline(s, .78, 5.35, 5.6, '构建或最终姿态不合格，不进入最佳候选')
dotline(s, .78, 5.87, 5.6, '失败记录回到下一轮，避免重复无效尝试')
text(s, 7.05, 4.78, 5.5, .40, '本次闭集 benchmark 的几何规则', 19, BLUE, True)
dotline(s, 7.07, 5.35, 5.52, '初始碰撞仅作诊断，允许 docking 搜索', size=16)
dotline(s, 7.07, 5.87, 5.52, '最终 pose 必须通过硬碰撞与姿态检查', size=16)
takeaway(s, '评价原则', '更低的 docking 分数 ≠ 可靠结合，更不等于实验活性已提升。', y=6.48)
notes(s, '闭集协议的关键限定：initial_receptor_clash_policy=defer_to_docking；初始碰撞不作为最终硬拒绝证据。最终 heavy-atom overlap ≤0.55 Å；固定核心 RMSD 阈值 2.0 Å，最大核心位移 3.0 Å，关键 MET:A:793 氢键锚点；至少 2/3 seed 同一 pose family。主评分 minimizedAffinity，配对 reference seed；稳定性资格与最终排序以 result.json 实际实现为准。避免将几何预筛描述为本轮全部候选初始碰撞均已通过。')

# 6 — Native editable plots, tightly aligned metrics and explicit limitations.
s = header('05  /  当前进展', '链路已跑通；尚不能宣称 LLM 优于基线',
           '4WKQ–EGFR / gefitinib：78 条固定侧链，每次运行最多提出 20 个候选。',
           source='数据：LLM run 20260927-193317；random-baseline-v1；原配体 calibration.json（沿用原汇报快照）')
metrics = [
    (f"{calibration['passing_seed_count']} / 3", '参考配体 seed 校准通过'),
    ('78 / 78', '匿名侧链可正确构建'),
    (f"{sum(r['pose_retention']['status'] == 'passed' for r in history)} / 20", 'LLM 候选通过姿态门控'),
    (f"{sum(bool(r.get('stability_eligible')) for r in history)} / 20", 'LLM 候选具稳定性资格'),
]
for i, (number, label) in enumerate(metrics):
    x = .72 + i*3.13
    text(s, x, 2.10, 2.85, .48, number, 28, TEAL if i != 2 else BLUE, True)
    text(s, x, 2.67, 2.96, .32, label, 13.5, INK)
    if i < 3:
        line(s, x+2.83, 2.14, x+2.83, 2.91, LINE, .8)
line(s, .73, 3.19, 12.59, 3.19, LINE, .85)
text(s, .75, 3.41, 3.67, .37, '参考校准：核心 RMSD', 18, BLUE, True)
text(s, .75, 3.88, 3.69, .31, '3 个 seed 均低于 2.0 Å 门槛', 13.3, MUTED)
for i, row in enumerate(calibration['reference_family']):
    y = 4.48 + i*.47
    value = row['crystal_core_rmsd']
    text(s, .78, y-.045, .80, .27, str(row['seed']), 12.5, MUTED)
    rect(s, 1.40, y+.035, 1.75, .11, PALE)
    rect(s, 1.40, y+.035, 1.75*value/.5, .11, TEAL)
    circle(s, 1.40+1.75*value/.5-.07, y+.02, .14, TEAL)
    text(s, 3.34, y-.06, 1.1, .31, f'{value:.3f} Å', 14.5, BLUE, True)
text(s, .77, 5.93, 3.75, .36, 'seed 17 / 29 / 43；直接受体坐标比较', 11.6, MUTED)
# Candidate bar plot: native shapes and text so all values/labels remain editable.
text(s, 4.90, 3.41, 7.7, .38, '20 个 LLM 候选相对参考的 docking 差值', 18, BLUE, True)
text(s, 4.91, 3.88, 7.60, .30, 'Δ minimizedAffinity；负值更优；橙色为最终最佳候选', 12.5, MUTED)
plot_x, plot_y, plot_w, plot_h = 5.34, 4.38, 7.09, 1.51
lo, hi = -.5, .6
map_y = lambda v: plot_y + (hi-v)/(hi-lo)*plot_h
for value in [-.4, 0, .4]:
    yy = map_y(value)
    line(s, plot_x, yy, plot_x+plot_w, yy, LINE if value else '8499A9', .7 if value else 1)
    text(s, 4.78, yy-.12, .43, .25, f'{value:.1f}', 10.5, MUTED, align=PP_ALIGN.RIGHT)
bar_pitch = plot_w/20
for i, row in enumerate(history):
    value = row['delta_candidate_minus_reference']
    x = plot_x + i*bar_pitch+.063
    top, zero = map_y(value), map_y(0)
    color = ORANGE if row['attempt'] == best_attempt else (TEAL if row['stability_eligible'] else 'B9CCD9')
    rect(s, x, min(top,zero), .225, max(abs(top-zero), .012), color)
    text(s, x-.045, 5.99, .32, .23, str(row['attempt']), 9.8,
         ORANGE if row['attempt'] == best_attempt else MUTED, row['attempt'] == best_attempt,
         PP_ALIGN.CENTER)
# Exact best summary with no inference of comparative superiority.
line(s, .74, 6.37, 12.58, 6.37, LINE, .8)
text(s, .77, 6.48, 7.30, .33,
     f"单次最佳 Δ：LLM {best['delta_candidate_minus_reference']:.3f}；随机基线 {random_best['delta_candidate_minus_reference']:.3f}",
     16, BLUE, True)
text(s, 8.05, 6.48, 4.55, .34, '需要重复实验，不能据此判断策略优劣。', 13.3, MUTED)
notes(s, '数据直接读取运行文件，不重新计算或挑选最新运行。LLM：runs/4wkq-llm-20260927-193317/result.json；Random：runs/4wkq-random-baseline-v1/result.json。78/78 来自 4WKQ/benchmark-private/construction-audit-v2/construction-audit.json 和 BENCHMARK_RUNNING.md 的构建审计结论，非所有初始构象无碰撞。20/20 为候选层 pose retention 通过，不是全部 seed 逐一均通过。16/20 为 stability_eligible。图中 Δ 是合格配对 seed 的 minimizedAffinity 平均差，不是实验自由能。Best 按原程序的资格与质量函数选定，不是任意挑选。脚注 BENCHMARK_RUNNING.md 的“尚未启动付费LLM”段落已滞后，因此以实际 result.json 为准。')

# 7 — Concrete next steps and deliverables; no decorative dark outro.
s = header('06  /  总结与计划', '下一步：从“链路可用”走向“收益可验证”',
           '现阶段完成受控编辑与可审计评价；后续重点是公平比较、重复验证与实验闭环。')
text(s, .77, 2.16, 5.65, .42, '已具备的能力', 21, BLUE, True)
text(s, 7.18, 2.16, 5.4, .42, '下一步验证重点', 21, BLUE, True)
left = [
    ('受控分子编辑', '保留骨架，在明确位点和片段库内提出候选'),
    ('可解释计算反馈', '关联侧链选择、构建、pose 与相互作用变化'),
    ('可复现实验记录', '固定参考与 seeds，保留结构、命令和日志'),
]
right = [
    ('同预算重复比较', '多次 LLM、随机与 score-guided 基线运行'),
    ('更可靠的候选排序', '评估不确定性，再推进更高精度自由能计算'),
    ('面向实验筛选', '形成优先候选清单，推进合成与活性验证'),
]
for i in range(3):
    y = 2.99 + i*1.01
    for x, entries in [(.79, left), (7.20, right)]:
        text(s, x, y, 5.41, .37, entries[i][0], 19, TEAL if x < 1 else BLUE, True)
        text(s, x, y+.44, 5.45, .40, entries[i][1], 15.5, INK)
    if i < 2:
        line(s, .78, y+.85, 12.57, y+.85, LINE, .7)
takeaway(s, '预期交付', '一份稳定、可解释、可追溯的候选清单，而不是未经验证的“新药”。', y=6.41)
notes(s, 'RBFE、更高精度自由能计算、合成和湿实验属于下一步，不是已经完成的结果。当前内容忠实沿用原汇报的数据快照与结论边界。')

prs.core_properties.title = '保留骨架的分子优化智能体｜白底紧凑版'
prs.core_properties.subject = '总体流程、单点编辑、候选验证与当前进展'
prs.core_properties.author = '黄婉仪'
prs.core_properties.comments = '白底重排；流程图和数据图为原生可编辑元素；概念插图由 image-gen 扩展生成。'
OUTDIR.mkdir(parents=True, exist_ok=True)
prs.save(OUT)

# Mechanical QC: geometry, white background, editable workflow, no old dark assets.
issues = []
for i, slide in enumerate(prs.slides, 1):
    if str(slide.background.fill.fore_color.rgb) != WHITE:
        issues.append(f'Slide {i}: nonwhite background')
    for sh in slide.shapes:
        if sh.left < -9144 or sh.top < -9144 or sh.left+sh.width > prs.slide_width+9144 or sh.top+sh.height > prs.slide_height+9144:
            issues.append(f'Slide {i}: off-canvas {sh.name}')
assert not issues, '\n'.join(issues)
summary = {
    'output': str(OUT.relative_to(ROOT)),
    'slide_count': len(prs.slides),
    'white_backgrounds': len(prs.slides),
    'workflow_slides': [3, 4, 5],
    'native_workflows_and_plots': True,
    'llm_candidate_count': len(history),
    'pose_passing': sum(r['pose_retention']['status'] == 'passed' for r in history),
    'stability_eligible': sum(bool(r.get('stability_eligible')) for r in history),
    'llm_best_delta': best['delta_candidate_minus_reference'],
    'random_best_delta': random_best['delta_candidate_minus_reference'],
    'out_of_bounds_shapes': issues,
}
(OUTDIR / 'white_revision_qc.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(OUT)
print(json.dumps(summary, ensure_ascii=False, indent=2))
