from pathlib import Path

import numpy as np
from rdkit import Chem

from molecular_agent.adapters import DockingAdapter
from molecular_agent.structure import ComplexContext, parse_pdb
from scripts.audit_4wkq_benchmark import decompose, split_with_attachment

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / '4WKQ/task.calibration.json'


def test_prepared_4wkq_hydrogens_and_ccd_chlorine():
    context = ComplexContext(TASK)
    assert context.ligand.GetNumAtoms() == context.ligand.GetNumHeavyAtoms() == 31
    assert len([a for a in context.atoms if a.residue_name == 'IRE' and a.element == 'H']) == 24
    assert {a.GetSymbol() for a in context.ligand.GetAtoms()} == {'C', 'N', 'O', 'Cl', 'F'}
    assert context.ligand.GetAtomWithIdx(10).GetProp('atom_name') == 'CBA'
    assert context.ligand.GetAtomWithIdx(18).GetProp('atom_name') == 'N3'
    assert context.ligand.GetBondBetweenAtoms(10, 9).GetBondType() == Chem.BondType.SINGLE
    assert context.task['pose_retention']['anchors'][0]['ligand_atom_index'] == 18
    assert set(context.task['pose_retention']['core_atom_indices']) == set(range(10, 31))


def test_4wkq_export_preserves_oxidized_cysteine_and_protein_hydrogen(tmp_path):
    context = ComplexContext(TASK)
    path = context.write_receptor_pdb(tmp_path/'receptor.pdb')
    atoms = parse_pdb(path)
    assert len(atoms) == len(context.protein_atoms)
    assert len([a for a in atoms if a.residue_name == 'CSX' and a.residue_number == 797]) == 12
    assert any(a.residue_name == 'MET' and a.residue_number == 793 and a.name == 'H' for a in atoms)
    assert not any(a.residue_name in {'IRE', 'HOH'} for a in atoms)
    assert DockingAdapter._receptor_preflight(path, allowed_hetero=['CSX'])[0]
    assert not DockingAdapter._receptor_preflight(path)[0]  # Must explicitly review this residue.


def test_4wkq_crystal_heavy_coordinates_not_moved():
    context = ComplexContext(TASK)
    original = {a.name: a for a in parse_pdb(ROOT/'4WKQ/raw/4WKQ.original.pdb')
                if a.residue_name == 'IRE' and a.chain == 'A' and a.element != 'H'}
    for atom in context.ligand_pdb_atoms:
        assert np.array_equal(atom.xyz, original[atom.name].xyz)


def test_fixed_cut_reconstruction_and_wrong_core_rejection():
    context = ComplexContext(TASK)
    native = context.ligand
    original_fragment = decompose(native, native)
    assert len(original_fragment) == 1
    pieces = split_with_attachment(native, native.GetBondBetweenAtoms(10, 9).GetIdx())
    core = next(m for m in pieces if any(a.HasProp('_reference_atom_index') and
                a.GetIntProp('_reference_atom_index') == 10 for a in m.GetAtoms()))
    # Synthetic side chain, no assay/experimental label is involved in this unit test.
    side_chain = Chem.MolFromSmiles('[*:1]OCCF')
    compound = Chem.molzip(Chem.CombineMols(core, side_chain))
    Chem.SanitizeMol(compound)
    assert decompose(native, compound) == [Chem.MolToSmiles(side_chain, isomericSmiles=True)]
    changed = Chem.RWMol(native)
    changed.ReplaceAtom(28, Chem.Atom('Br'))
    changed = changed.GetMol()
    Chem.SanitizeMol(changed)
    assert not decompose(native, changed)  # Aniline F->Br is not a C6-only transformation.
