"""Offline structure/reachability audit. Never select compounds by experimental activity.

Detailed structure membership is evaluator-private; the public report contains counts
and preparation findings only. This is NOT the mixed fragment library or an activity evaluator.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from rdkit import Chem
from molecular_agent.structure import ComplexContext, parse_pdb
from molecular_agent.pose_retention import save_json

ROOT = Path(__file__).resolve().parents[1]


def canonical(molecule):
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def split_with_attachment(molecule, bond_index):
    cut = Chem.FragmentOnBonds(molecule, [bond_index], dummyLabels=[(0, 0)])
    for atom in cut.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(1)
    return Chem.GetMolFrags(cut, asMols=True, sanitizeFrags=True)


def decompose(native, compound, anchor=10, side=9):
    """Exact single-cut reconstruction, not a candidate-dependent MCS/R-group guess."""
    pieces = split_with_attachment(native, native.GetBondBetweenAtoms(anchor, side).GetIdx())
    core = next(m for m in pieces if any(a.HasProp('_reference_atom_index') and
                a.GetIntProp('_reference_atom_index') == anchor for a in m.GetAtoms()))
    dummy = next(a.GetIdx() for a in core.GetAtoms() if a.GetAtomicNum() == 0)
    marked_anchor = core.GetAtomWithIdx(dummy).GetNeighbors()[0].GetIdx()
    query_rw = Chem.RWMol(core)
    query_rw.RemoveAtom(dummy)
    query = query_rw.GetMol()
    Chem.SanitizeMol(query)
    if dummy < marked_anchor:
        marked_anchor -= 1
    outputs = set()
    matches = compound.GetSubstructMatches(query, uniquify=False, useChirality=True, maxMatches=4097)
    if len(matches) > 4096:
        raise ValueError('Exact mapping enumeration limit exceeded; do not silently truncate')
    for match in matches:
        retained = set(match)
        crossing = [b for b in compound.GetBonds()
                    if (b.GetBeginAtomIdx() in retained) != (b.GetEndAtomIdx() in retained)]
        if len(crossing) != 1:
            continue
        bond = crossing[0]
        if (bond.GetBondType() != Chem.BondType.SINGLE or
                match[marked_anchor] not in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())):
            continue
        fragments = split_with_attachment(compound, bond.GetIdx())
        if len(fragments) != 2:
            continue
        for i, fragment in enumerate(fragments):
            if canonical(fragment) != canonical(core):
                continue
            replacement = fragments[1-i]
            # molzip verifies attachment position, full connectivity, charges and stereochemistry.
            reconstructed = Chem.molzip(Chem.CombineMols(core, replacement))
            Chem.SanitizeMol(reconstructed)
            if canonical(reconstructed) == canonical(compound):
                outputs.add(canonical(replacement))
    return sorted(outputs)


def main():
    context = ComplexContext(ROOT / '4WKQ/task.calibration.json')
    source = ROOT / 'research/test_system_selection_egfr/gefitinib_discovery_series_28.json'
    data = json.loads(source.read_text())
    private = []
    all_fragments, novel_fragments = set(), set()
    baseline = canonical(context.ligand)
    for record in data:
        # Explicit whitelist: no potency, rank, activity class, prior r_group_smiles or SAR text.
        structure = Chem.MolFromSmiles(record['canonical_smiles'])
        if structure is None:
            raise ValueError('Unreadable literature structure')
        fragments = decompose(context.ligand, structure)
        is_reference = canonical(structure) == baseline
        private.append({'source_compound_id': record['molecule_chembl_id'],
                        'canonical_smiles': canonical(structure), 'is_reference': is_reference,
                        'reachable_single_c6_edit': len(fragments) == 1,
                        'attachment_fragments': fragments})
        if len(fragments) == 1:
            all_fragments.update(fragments)
            if not is_reference:
                novel_fragments.update(fragments)
    original = parse_pdb(ROOT / '4WKQ/raw/4WKQ.original.pdb')
    old_ligand = {a.name: a for a in original if a.residue_name == 'IRE' and a.chain == 'A'
                  and a.residue_number == 1101 and a.element not in {'H', 'D', 'T'}}
    displacements = [float(np.linalg.norm(a.xyz-old_ligand[a.name].xyz)) for a in context.ligand_pdb_atoms]
    atom_key = lambda a: (a.chain, a.residue_number, a.residue_name, a.name)
    old_protein = {atom_key(a): a for a in original if a.record == 'ATOM' and a.element not in {'H', 'D', 'T'}}
    receptor_displacements = [float(np.linalg.norm(a.xyz-old_protein[atom_key(a)].xyz)) for a in context.protein_atoms
                             if a.element not in {'H', 'D', 'T'} and atom_key(a) in old_protein]
    report = {
        'status': 'structure_and_reachability_audit_only', 'reference_pdb': '4WKQ',
        'ligand': {'residue': 'IRE:A:1101', 'heavy_atoms': context.ligand.GetNumHeavyAtoms(),
                   'chemical_identity_matches_ccd': True, 'canonical_smiles': baseline},
        'preparation': {'input_is_maestro_prepared_not_untouched_deposition': True,
            'ligand_heavy_direct_rmsd_vs_deposition': float(np.sqrt(np.mean(np.square(displacements)))),
            'ligand_max_heavy_displacement_vs_deposition': max(displacements),
            'shared_receptor_heavy_atom_count': len(receptor_displacements),
            'shared_receptor_heavy_direct_rmsd_vs_deposition': float(np.sqrt(np.mean(np.square(receptor_displacements)))),
            'retained_modified_residue': 'CSX:A:797 (S-OXY CYSTEINE); do not silently delete or convert to CYS',
            'altloc_policy': 'blank/A; subject to preparation review'},
        'edit_definition': {'chemical_position': 'quinazoline C6',
            'core_anchor_pdb_atom_name': 'CBA', 'core_anchor_reference_index': 10,
            'side_atom_pdb_atom_name': 'OAV', 'side_reference_index': 9,
            'cut_bond': [10, 9], 'fragment_includes_linker_oxygen': True,
            'warning': 'PDB atom named C6 is NOT the medicinal-chemistry C6 substitution position'},
        'dataset': {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'record_count': len(data), 'unique_structures': len({r['canonical_smiles'] for r in private}),
            'assay_counts': dict(Counter(r['assay_chembl_id'] for r in data)),
            'endpoint_counts': dict(Counter(r['standard_type'] for r in data)),
            'reachable_records_including_reference': sum(r['reachable_single_c6_edit'] for r in private),
            'unreachable_records': sum(not r['reachable_single_c6_edit'] for r in private),
            'reference_records': sum(r['is_reference'] for r in private),
            'unique_reachable_fragments_including_reference': len(all_fragments),
            'unique_nonreference_fragments': len(novel_fragments)},
        'leakage_policy': 'No experimental labels or compound membership details in this public report; private output is never designer context.'}
    save_json(ROOT / '4WKQ/structure-audit.json', report)
    save_json(ROOT / '4WKQ/benchmark-private/reachability.json', {
        'role': 'evaluator_only_do_not_send_to_designer', 'source_sha256': report['dataset']['source_sha256'],
        'records': private, 'fragments_including_reference': sorted(all_fragments)})
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
