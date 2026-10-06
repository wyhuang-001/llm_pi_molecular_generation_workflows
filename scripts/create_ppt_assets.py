from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib import font_manager
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, rdFMCS
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "ppt"
OUT.mkdir(parents=True, exist_ok=True)

# Prefer an installed CJK font for deterministic chart labels.
font_candidates = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttf",
]
for fp in font_candidates:
    if Path(fp).exists():
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def trim(path: Path, pad: int = 12):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    diff = ImageChops.difference(im, bg).convert("L")
    bbox = diff.getbbox()
    if bbox:
        l, t, r, b = bbox
        l, t = max(0, l-pad), max(0, t-pad)
        r, b = min(im.width, r+pad), min(im.height, b+pad)
        im = im.crop((l, t, r, b))
    im.save(path)


def load_mol(sdf: Path):
    suppl = Chem.SDMolSupplier(str(sdf), removeHs=True)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise RuntimeError(f"Cannot read {sdf}")
    return mol


def draw_molecule(sdf: Path, out: Path, legend: str, reference=None):
    mol = load_mol(sdf)
    rdDepictor.Compute2DCoords(mol)
    highlight_atoms = []
    highlight_colors = {}
    if reference is not None:
        mcs = rdFMCS.FindMCS([reference, mol], ringMatchesRingOnly=True, completeRingsOnly=True, timeout=10)
        patt = Chem.MolFromSmarts(mcs.smartsString)
        match = set(mol.GetSubstructMatch(patt)) if patt is not None else set()
        highlight_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in match]
        highlight_colors = {idx: (0.98, 0.63, 0.20) for idx in highlight_atoms}
    options = Draw.MolDrawOptions()
    options.legendFontSize = 28
    options.baseFontSize = 0.72
    options.bondLineWidth = 3.0
    options.padding = 0.08
    img = Draw.MolToImage(
        mol,
        size=(1100, 650),
        legend=legend,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        options=options,
    )
    img.save(out)
    trim(out)


run = ROOT / "runs" / "4wkq-llm-20260927-193317"
reference_mol = load_mol(run / "reference-ligand.sdf")
draw_molecule(run / "reference-ligand.sdf", OUT / "reference_ligand.png", "原始配体（参考）")
draw_molecule(run / "candidate-06.sdf", OUT / "best_candidate.png", "候选分子（橙色为替换侧链）", reference=reference_mol)

# Candidate-score plot for the completed 20-candidate LLM run.
data = json.load(open(run / "result.json", encoding="utf-8"))
history = data["state"]["docking_history"]
attempts = [x["attempt"] for x in history]
deltas = [x.get("delta_candidate_minus_reference") for x in history]
eligible = [bool(x.get("stability_eligible")) for x in history]
best_attempt = data["result"]["best_attempt"]

fig, ax = plt.subplots(figsize=(12.6, 4.8), dpi=180)
fig.patch.set_alpha(0)
ax.set_facecolor("#F5F8FC")
colors = []
for a, d, e in zip(attempts, deltas, eligible):
    if a == best_attempt:
        colors.append("#F2A33A")
    elif d is not None and d < 0 and e:
        colors.append("#14B8A6")
    elif d is not None and d < 0:
        colors.append("#65C7BE")
    else:
        colors.append("#AAB7C6")
ax.bar(attempts, deltas, color=colors, width=0.72, edgecolor="none")
ax.axhline(0, color="#334155", linewidth=1.2)
ax.grid(axis="y", color="#DCE5EF", linewidth=0.8, alpha=0.9)
ax.set_axisbelow(True)
ax.set_xlabel("候选编号", fontsize=12, color="#334155")
ax.set_ylabel("相对参考的 docking 差值\nΔminimizedAffinity", fontsize=11, color="#334155")
ax.set_xticks(attempts)
ax.tick_params(colors="#526273", labelsize=9)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.text(0.01, 0.96, "负值更优；橙色为最终 best，绿色为通过稳定性门的改善候选", transform=ax.transAxes,
        ha="left", va="top", fontsize=11, color="#526273")
ax.annotate(
    f"best #{best_attempt}\n{deltas[best_attempt-1]:.3f}",
    xy=(best_attempt, deltas[best_attempt-1]),
    xytext=(best_attempt+1.2, deltas[best_attempt-1]-0.45),
    arrowprops=dict(arrowstyle="->", color="#C47B18", lw=1.5),
    fontsize=11, color="#9A5B0B", fontweight="bold",
)
plt.tight_layout()
fig.savefig(OUT / "llm_candidate_scores.png", transparent=True, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# Reference calibration plot.
seeds = [17, 29, 43]
rmsd = [0.349, 0.297, 0.413]
fig, ax = plt.subplots(figsize=(7.8, 3.8), dpi=180)
fig.patch.set_alpha(0)
ax.set_facecolor("#F5F8FC")
ax.bar([str(x) for x in seeds], rmsd, color=["#2DD4BF", "#14B8A6", "#0F9F96"], width=0.55)
ax.axhline(2.0, color="#F2A33A", ls="--", lw=1.4)
ax.text(2.45, 2.03, "门槛 2.0 Å", fontsize=10, color="#B56D12", va="bottom", ha="right")
for i, v in enumerate(rmsd):
    ax.text(i, v+0.05, f"{v:.3f} Å", ha="center", va="bottom", fontsize=12, color="#0F4C5C", fontweight="bold")
ax.set_ylim(0, 2.35)
ax.set_xlabel("GNINA seed", fontsize=11, color="#334155")
ax.set_ylabel("固定核心 RMSD", fontsize=11, color="#334155")
ax.grid(axis="y", color="#DCE5EF", linewidth=0.8)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(colors="#526273")
plt.tight_layout()
fig.savefig(OUT / "reference_calibration.png", transparent=True, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

print("Assets written to", OUT)
