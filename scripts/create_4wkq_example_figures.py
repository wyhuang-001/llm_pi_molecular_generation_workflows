"""Render the co-crystal ligand and a demonstrative replacement candidate.

The previous example (candidate-06) was only a single H -> OH addition, which is
a weak illustration of a side-chain replacement. This renders candidate-01 of the
same run (bond-site-005): the terminal morpholine becomes an
N-methylpiperazine. The replaced terminal ring is drawn in a stronger shade so
the change is visible at a glance.
"""
from pathlib import Path
import json

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/4wkq-llm-20260927-193317'
OUT = ROOT / 'assets/ppt_4wkq_rev'
OUT.mkdir(parents=True, exist_ok=True)
ATTEMPT = 1

llm = json.loads((RUN / 'result.json').read_text())
row = next(r for r in llm['state']['docking_history'] if r['attempt'] == ATTEMPT)
kept = row['transformation']['bond']['retained_atom_indices']

ref = next(m for m in Chem.SDMolSupplier(str(RUN / 'reference-ligand.sdf')) if m)
cand = next(m for m in Chem.SDMolSupplier(str(RUN / f'candidate-{ATTEMPT:02d}.sdf')) if m)

scaffold = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(ref, atomsToUse=kept))
ref_match = ref.GetSubstructMatch(scaffold)
cand_match = cand.GetSubstructMatch(scaffold)
assert len(ref_match) == len(cand_match) == 21

rdDepictor.Compute2DCoords(ref)
rdDepictor.GenerateDepictionMatching2DStructure(cand, ref, list(zip(ref_match, cand_match)))

CYAN = (0.83, 0.95, 0.97)      # retained core
LIGHT = (1.0, 0.90, 0.78)      # editable side-chain linker
STRONG = (0.98, 0.66, 0.30)    # terminal ring that actually changes


def side_ring_atoms(mol, core):
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    outer = [r for r in rings if not (r & core)]
    return set().union(*outer) if outer else set()


def draw(mol, match, name):
    core = set(match)
    ring = side_ring_atoms(mol, core)

    def atom_color(i):
        if i in core:
            return CYAN
        return STRONG if i in ring else LIGHT

    def bond_color(i, j):
        if i in core and j in core:
            return CYAN
        return STRONG if (i in ring and j in ring) else LIGHT

    atom_colors = {a.GetIdx(): atom_color(a.GetIdx()) for a in mol.GetAtoms()}
    bond_colors = {b.GetIdx(): bond_color(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()}
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
    (OUT / f'{name}.png').write_bytes(drawer.GetDrawingText())
    print('wrote', OUT / f'{name}.png')


draw(ref, ref_match, 'reference_ligand')
draw(cand, cand_match, 'candidate_01')
print('transformation:', row['transformation']['fragment_id'], row['transformation']['fragment_smiles'])
