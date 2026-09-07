from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ppt_assets"
OUT.mkdir(exist_ok=True)
RUN = ROOT / "runs/docking-loop-codex-gpt56-20260902-171109/real"

ref = Chem.SDMolSupplier(str(RUN / "reference-ligand.sdf"), removeHs=False)[0]
cand = Chem.SDMolSupplier(str(RUN / "candidate-20.sdf"), removeHs=False)[0]
ref = Chem.RemoveHs(ref)
cand = Chem.RemoveHs(cand)
Chem.rdDepictor.Compute2DCoords(ref)
Chem.rdDepictor.Compute2DCoords(cand)

# Determine atoms added relative to the reference scaffold for visual emphasis.
mcs = rdFMCS.FindMCS([ref, cand], ringMatchesRingOnly=True, completeRingsOnly=True, timeout=10)
query = Chem.MolFromSmarts(mcs.smartsString)
cand_match = set(cand.GetSubstructMatch(query)) if query is not None else set()
added = [i for i in range(cand.GetNumAtoms()) if i not in cand_match]
highlight_colors = {i: (0.98, 0.55, 0.20) for i in added}

opts = Draw.MolDrawOptions()
opts.baseFontSize = 0.62
opts.bondLineWidth = 2.0
opts.padding = 0.08
opts.legendFontSize = 30
opts.addAtomIndices = False

ref_img = Draw.MolToImage(ref, size=(1500, 760), options=opts, legend="Reference ligand")
cand_img = Draw.MolToImage(
    cand,
    size=(1500, 760),
    options=opts,
    legend="Best candidate (added group highlighted)",
    highlightAtoms=added,
    highlightAtomColors=highlight_colors,
)

# Convert white to transparent-ish output is intentionally avoided for maximum PowerPoint compatibility.
ref_img.save(OUT / "reference_ligand.png")
cand_img.save(OUT / "best_candidate.png")

# Create a small swatch legend image only if useful elsewhere.
legend = Image.new("RGB", (800, 120), "white")
legend.save(OUT / "blank_white.png")

print(OUT)
