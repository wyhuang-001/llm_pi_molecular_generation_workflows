"""Freeze a structure-only 26+26+26 candidate pool. No activity-based selection.

Public files contain anonymous chemistry only. Membership/provenance stay private.
Existing outputs cannot be overwritten; create a separately versioned pool to revise rules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from molecular_agent.closed_pool import assemble, canonical, scaffold, sha256
from molecular_agent.fragment_library import FragmentLibrary
from molecular_agent.pose_retention import save_json
from molecular_agent.structure import ComplexContext
from scripts.audit_4wkq_benchmark import decompose

ROOT = Path(__file__).resolve().parents[1]
POOL_SEED = 410627
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
BASIC_N = Chem.MolFromSmarts('[NX3;!$(N-C=O);!$(N-S(=O)=O);!$(N-[a])]')
REACTIVE = [Chem.MolFromSmarts(s) for s in ('[O]-[O]', '[N]-[N]', '[N]-[O]', '[C](=O)[F,Cl,Br,I]')]


def features(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    return {**FragmentLibrary._properties(smiles),
            'basic_n': len(molecule.GetSubstructMatches(BASIC_N)), 'fp': FP.GetFingerprint(molecule)}


def background_allowed(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1 or Chem.GetFormalCharge(molecule) != 0:
        return False
    if not 2 <= molecule.GetNumHeavyAtoms() <= 12:
        return False
    if any(a.GetAtomicNum() not in {0, 6, 7, 8, 9, 16, 17} or a.GetIsotope() for a in molecule.GetAtoms()):
        return False
    dummies = [a for a in molecule.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1 or dummies[0].GetDegree() != 1 or dummies[0].GetAtomMapNum() != 1:
        return False
    if any(a.GetAtomMapNum() for a in molecule.GetAtoms() if a.GetAtomicNum() != 0):
        return False
    oxygen = dummies[0].GetNeighbors()[0]
    if oxygen.GetAtomicNum() != 8 or oxygen.GetDegree() != 2:
        return False
    first = next(a for a in oxygen.GetNeighbors() if a.GetAtomicNum() != 0)
    if first.GetAtomicNum() != 6 or first.GetIsAromatic() or first.GetHybridization() != Chem.HybridizationType.SP3:
        return False
    if any(b.GetBondType() != Chem.BondType.SINGLE for b in oxygen.GetBonds()):
        return False
    if any(molecule.HasSubstructMatch(pattern) for pattern in REACTIVE):
        return False
    if any(len(ring) < 4 for ring in molecule.GetRingInfo().AtomRings()):
        return False
    return True


def compatible(a, b):
    return (abs(a['heavy_atoms']-b['heavy_atoms']) <= 2 and
            abs(a['hba']-b['hba']) <= 1 and abs(a['hbd']-b['hbd']) <= 1 and
            abs(a['rotatable_bonds']-b['rotatable_bonds']) <= 2 and
            abs(a['tpsa']-b['tpsa']) <= 25 and abs(a['logp']-b['logp']) <= 1.5 and
            abs(a['ring_count']-b['ring_count']) <= 1 and
            a['aromatic_ring_count'] == b['aromatic_ring_count'] and
            bool(a['basic_n']) == bool(b['basic_n']))


def structural_neighbors(smiles):
    """Label-independent, bounded medicinal-chemistry edits of every supplied structure.

    Supplement a source library that may lack polar/long-chain matched backgrounds.
    These are hypotheses, not asserted synthesizable or low-activity compounds.
    """
    molecule = Chem.MolFromSmiles(smiles)
    generated = set()

    def accept(rw):
        try:
            result = rw.GetMol()
            Chem.SanitizeMol(result)
            text = Chem.MolToSmiles(result, isomericSmiles=True)
            if background_allowed(text):
                generated.add(text)
        except (ValueError, RuntimeError):
            pass

    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if (bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE or
                a.GetAtomicNum() == 0 or b.GetAtomicNum() == 0 or a.GetIsAromatic() or b.GetIsAromatic()):
            continue
        rw = Chem.RWMol(molecule)
        rw.RemoveBond(a.GetIdx(), b.GetIdx())
        carbon = rw.AddAtom(Chem.Atom('C'))
        rw.AddBond(a.GetIdx(), carbon, Chem.BondType.SINGLE)
        rw.AddBond(carbon, b.GetIdx(), Chem.BondType.SINGLE)
        accept(rw)
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        neighbors = list(atom.GetNeighbors())
        if (atom.GetAtomicNum() == 6 and not atom.IsInRing() and atom.GetTotalNumHs() == 2
                and len(neighbors) == 2 and all(a.GetAtomicNum() in {6, 7, 8} for a in neighbors)
                and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds())):
            rw = Chem.RWMol(molecule)
            rw.AddBond(neighbors[0].GetIdx(), neighbors[1].GetIdx(), Chem.BondType.SINGLE)
            rw.RemoveAtom(index)
            accept(rw)
        if (atom.GetAtomicNum() in {6, 7} and not atom.GetIsAromatic() and atom.GetTotalNumHs() > 0):
            rw = Chem.RWMol(molecule)
            carbon = rw.AddAtom(Chem.Atom('C'))
            rw.AddBond(index, carbon, Chem.BondType.SINGLE)
            accept(rw)
        if atom.GetAtomicNum() == 8 and atom.IsInRing():
            rw = Chem.RWMol(molecule)
            rw.ReplaceAtom(index, Chem.Atom('C'))
            accept(rw)
    return generated


def select_pool(native, structures, source_fragments, seed=POOL_SEED):
    core = scaffold(native, [10, 9])
    original = Chem.MolToSmiles(native, isomericSmiles=True)
    gold = set()
    excluded = []
    # Sort STRUCTURES, never potency/order/IDs. Changing activity labels cannot change the public pool.
    for smiles in sorted(set(canonical(s) for s in structures)):
        fragments = decompose(native, Chem.MolFromSmiles(smiles))
        if smiles == original:
            continue
        if len(fragments) != 1:
            excluded.append(smiles)
        else:
            gold.add(fragments[0])
    if len(gold) != 26:
        raise ValueError(f'Expected 26 reachable nonreference structures, got {len(gold)}')
    original_fragment = decompose(native, native)[0]
    pool = {}
    augmented = set(source_fragments)
    for smiles in sorted(gold | {original_fragment}):
        augmented.update(structural_neighbors(smiles))
    for raw in sorted(augmented):
        if not background_allowed(raw):
            continue
        smiles = canonical(raw)
        if smiles in gold or smiles == original_fragment:
            continue
        try:
            product = assemble(core, smiles)
            if Chem.GetFormalCharge(product) != Chem.GetFormalCharge(native):
                continue
        except (ValueError, RuntimeError):
            continue
        pool[smiles] = features(smiles)
    properties = {s: features(s) for s in gold}
    eligible = {s: [p for p in sorted(pool) if compatible(properties[s], pool[p])] for s in sorted(gold)}
    order = sorted(gold, key=lambda s: (len(eligible[s]), s))
    selected, membership = [], {s: {'group': 'literature'} for s in gold}
    for group in ('near_neighbor', 'property_matched_diverse'):
        for target in order:
            available = [s for s in eligible[target] if s not in membership]
            if not available:
                raise ValueError(f'Insufficient matched backgrounds for a target; do not silently relax constraints ({group})')
            if group == 'near_neighbor':
                choice = min(available, key=lambda s: (-DataStructs.TanimotoSimilarity(properties[target]['fp'], pool[s]['fp']), s))
            else:
                reference_fps = [properties[s]['fp'] for s in sorted(gold)] + [pool[s]['fp'] for s in selected]
                # Max-min diversity, but only inside the same explicit chemical/descriptor envelope.
                choice = min(available, key=lambda s: (max(DataStructs.BulkTanimotoSimilarity(pool[s]['fp'], reference_fps)), s))
            membership[choice] = {'group': group, 'matched_to_fragment': target,
                                  'target_similarity': DataStructs.TanimotoSimilarity(properties[target]['fp'], pool[choice]['fp'])}
            selected.append(choice)
    all_smiles = sorted(membership)
    random.Random(seed).shuffle(all_smiles)
    records = [{'fragment_id': f'F{i:03d}', 'smiles': s, 'allowed_operations': ['replace_fragment']}
               for i, s in enumerate(all_smiles, 1)]
    products = [Chem.MolToSmiles(assemble(core, r['smiles']), isomericSmiles=True) for r in records]
    if len(records) != 78 or len(set(products)) != 78 or original in products:
        raise ValueError('Pool does not reconstruct exactly 78 distinct nonreference molecules')
    return records, products, membership, {'background_eligible_pool_count': len(pool),
            'matched_background_count_range': [min(map(len, eligible.values())), max(map(len, eligible.values()))],
            'excluded_non_c6_structures': excluded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-dir', type=Path, default=ROOT/'4WKQ/design')
    parser.add_argument('--private-dir', type=Path, default=ROOT/'4WKQ/benchmark-private/pool-v1')
    args = parser.parse_args()
    if args.public_dir.exists() or args.private_dir.exists():
        parser.error('Output directories must be new; frozen pools cannot be overwritten')
    context = ComplexContext(ROOT/'4WKQ/task.calibration.json')
    source = ROOT/'research/test_system_selection_egfr/gefitinib_discovery_series_28.json'
    source_library = ROOT/'molecular_agent/data/fragments_unified.json'
    literature = json.loads(source.read_text())
    library = json.loads(source_library.read_text())['fragments']
    records, products, membership, audit = select_pool(context.ligand,
        [row['canonical_smiles'] for row in literature], [r['smiles'] for r in library])
    public = args.public_dir.resolve()
    public.mkdir(parents=True)
    args.private_dir.mkdir(parents=True)
    save_json(public/'fragments.json', {'schema_version': 1, 'allowed_operations': ['replace_fragment'], 'fragments': records})
    manifest = {'schema_version': 1, 'library_sha256': sha256(public/'fragments.json'),
        'structure_source_identity': context.preparation_identity(), 'cut_bond': [10, 9],
        'candidate_count': 78, 'maximum_unique_candidates': 20, 'maximum_design_requests': 60,
        'catalog_order': 'frozen', 'descriptor_method': 'RDKit attachment-fragment descriptors',
        'rdkit_version': rdBase.rdkitVersion, 'allowed_operations': ['replace_fragment']}
    save_json(public/'pool-manifest.json', manifest)
    private_rows = [{**r, 'product_smiles': product, **membership[r['smiles']]}
                    for r, product in zip(records, products)]
    # Provenance and source IDs stay evaluator-private. No activity values are copied here.
    save_json(args.private_dir/'membership.json', {'role': 'evaluator_only', 'library_sha256': manifest['library_sha256'],
        'pool_manifest_sha256': sha256(public/'pool-manifest.json'), 'pool_seed': POOL_SEED,
        'source_library_sha256': sha256(source_library), 'source_structures_sha256': sha256(source),
        'builder_sha256': sha256(Path(__file__)), 'records': private_rows, 'audit': audit,
        'matching_rules': {'heavy_atoms_delta': 2, 'hba_delta': 1, 'hbd_delta': 1, 'rotatable_delta': 2,
            'tpsa_delta': 25, 'logp_delta': 1.5, 'ring_delta': 1,
            'same_aromatic_ring_count': True, 'same_basic_amine_presence': True},
        'background_pool_generation': 'source library plus one-step CH2 insertion/deletion, methyl addition and ring O->CH2 edits of ALL reachable side chains',
        'no_activity_fields_used_for_selection': True})
    print(json.dumps({'public_directory': str(public), 'candidate_count': len(records),
                      'library_sha256': manifest['library_sha256'],
                      'background_pool_size': audit['background_eligible_pool_count'],
                      'matched_background_count_range': audit['matched_background_count_range']}, indent=2))


if __name__ == '__main__':
    main()
