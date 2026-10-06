from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Geometry import Point3D

from .edit_taxonomy import EditTaxonomyError, normalize_transformation
from .structure import PDBAtom


VDW = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.8, "S": 1.8, "CL": 1.75, "BR": 1.85}


@dataclass
class EditResult:
    molecule: Chem.Mol
    report: dict[str, Any]


def _fragment(fragment_smiles: str) -> tuple[Chem.Mol, int, int]:
    fragment = Chem.MolFromSmiles(fragment_smiles)
    if fragment is None:
        raise ValueError(f"Invalid fragment SMILES: {fragment_smiles}")
    dummies = [atom for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 1 or dummies[0].GetAtomMapNum() != 1:
        raise ValueError("Fragment must contain exactly one mapped dummy atom [*:1]")
    if len(Chem.GetMolFrags(fragment)) != 1:
        raise ValueError("Fragment must be one connected component")
    neighbors = list(dummies[0].GetNeighbors())
    if len(neighbors) != 1:
        raise ValueError("Mapped dummy atom must have exactly one neighbor")
    bond = fragment.GetBondBetweenAtoms(dummies[0].GetIdx(), neighbors[0].GetIdx())
    if bond.GetBondType() != Chem.BondType.SINGLE:
        raise ValueError("Fragment attachment bond must be single")
    return fragment, dummies[0].GetIdx(), neighbors[0].GetIdx()


def _scan_clashes(
    candidate: Chem.Mol,
    protein_atoms: list[PDBAtom],
    indices: range,
) -> list[dict[str, Any]]:
    """Rigid-receptor VDW overlap scan for the listed candidate atoms."""
    conformer = candidate.GetConformer()
    clashes = []
    for index in indices:
        atom = candidate.GetAtomWithIdx(index)
        point = conformer.GetAtomPosition(index)
        xyz = np.array([point.x, point.y, point.z])
        ligand_radius = VDW.get(atom.GetSymbol().upper(), 1.7)
        for protein_atom in protein_atoms:
            protein_radius = VDW.get(protein_atom.element, 1.7)
            distance = float(np.linalg.norm(protein_atom.xyz - xyz))
            overlap = ligand_radius + protein_radius - distance
            if overlap > 0.55:
                clashes.append(
                    {
                        "candidate_atom": index,
                        "protein_atom": f"{protein_atom.residue_name}:{protein_atom.chain}:{protein_atom.residue_number}:{protein_atom.name}",
                        "distance": round(distance, 3),
                        "vdw_overlap": round(overlap, 3),
                    }
                )
    clashes.sort(key=lambda item: item["vdw_overlap"], reverse=True)
    return clashes


def _blocking_residues(clashes: list[dict[str, Any]], limit: int = 5) -> list[str]:
    residues: list[str] = []
    for clash in clashes:
        residue = clash["protein_atom"].rsplit(":", 1)[0]
        if residue not in residues:
            residues.append(residue)
        if len(residues) == limit:
            break
    return residues


def annotate_edit_layer(parent: Chem.Mol, product: Chem.Mol, report: dict[str, Any]) -> dict[str, Any]:
    """Record which edit layer the produced candidate belongs to.

    The layer is a property of the parent, the operation and the chosen fragment,
    not of the fragment alone, so it is computed from the built product and is
    stored on every candidate report.  T1 means the product differs from its
    parent by one or two heavy atoms with no ring-framework change and should be
    narrated as a minimal edit; T2 means a real fragment or ring replacement.
    """
    from .tiering import T1, classify_change

    try:
        parent_heavy = Chem.RemoveHs(Chem.Mol(parent))
        product_heavy = Chem.RemoveHs(Chem.Mol(product))
        change = classify_change(parent_heavy, product_heavy)
    except Exception as error:  # pragma: no cover - defensive, never blocks construction
        report["edit_layers"] = {"status": "unavailable", "error": str(error)}
        return report
    report["edit_layers"] = {
        "status": "complete",
        "tier": change["tier"],
        "compile_as": "minimal_edit" if change["tier"] == T1 else "whole_fragment",
        "heavy_atom_changes": change["cost"],
        "added": change["added"],
        "lost": change["lost"],
        "substitutions": change["substitutions"],
        "ring_size_change": change["ring_size_change"],
        "ring_skeleton_change": change["ring_skeleton_change"],
        "ring_change_kinds": change["ring_change_kinds"],
        "reason": change["reason"],
        "note": (
            "The host builds the requested product. This field states how large the change "
            "actually is, so a one-atom change is never reported as a whole-fragment replacement."
        ),
    }
    return report


def _molecule_properties(molecule: Chem.Mol) -> dict[str, Any]:
    return {
        "canonical_smiles": Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(molecule)), isomericSmiles=True),
        "formal_charge": Chem.GetFormalCharge(molecule),
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "molecular_weight": round(Descriptors.MolWt(molecule), 2),
        "logp": round(Crippen.MolLogP(molecule), 2),
        "hbd": Lipinski.NumHDonors(molecule),
        "hba": Lipinski.NumHAcceptors(molecule),
        "tpsa": round(rdMolDescriptors.CalcTPSA(molecule), 2),
        "rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
    }


def _property_delta(parent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: round(candidate[key] - parent[key], 2)
        for key in (
            "formal_charge", "heavy_atoms", "molecular_weight", "logp",
            "hbd", "hba", "tpsa", "rotatable_bonds",
        )
    }


def apply_atom_addition(
    parent: Chem.Mol,
    anchor_index: int,
    fragment_smiles: str,
    protein_atoms: list[PDBAtom],
    seed: int = 17,
) -> EditResult:
    """``atom:addition`` - attach a library fragment at a hydrogen-bearing atom."""
    parent = Chem.RemoveHs(Chem.Mol(parent))
    if anchor_index < 0 or anchor_index >= parent.GetNumAtoms():
        raise ValueError(f"Invalid anchor atom index: {anchor_index}")
    anchor = parent.GetAtomWithIdx(anchor_index)
    if anchor.GetTotalNumHs() < 1:
        raise ValueError("Anchor atom has no replaceable hydrogen")

    fragment, dummy_index, attachment_index = _fragment(fragment_smiles)
    combined = Chem.CombineMols(parent, fragment)
    parent_count = parent.GetNumAtoms()
    rw = Chem.RWMol(combined)
    rw.AddBond(anchor_index, parent_count + attachment_index, Chem.BondType.SINGLE)
    rw.RemoveAtom(parent_count + dummy_index)
    candidate = rw.GetMol()
    Chem.SanitizeMol(candidate)

    parent_charge = Chem.GetFormalCharge(parent)
    candidate_charge = Chem.GetFormalCharge(candidate)
    if candidate_charge != parent_charge:
        raise ValueError(f"Formal charge changed from {parent_charge} to {candidate_charge}")

    parent_conformer = parent.GetConformer()
    coordinate_map = {}
    for index in range(parent_count):
        point = parent_conformer.GetAtomPosition(index)
        coordinate_map[index] = Point3D(point.x, point.y, point.z)

    candidate = Chem.AddHs(candidate, addCoords=False)
    if AllChem.EmbedMolecule(
        candidate,
        randomSeed=seed,
        coordMap=coordinate_map,
        useRandomCoords=True,
        enforceChirality=True,
    ) != 0:
        raise ValueError("Could not generate a constrained 3D candidate")
    force_field = AllChem.UFFGetMoleculeForceField(candidate)
    for index in range(parent_count):
        force_field.AddFixedPoint(index)
    force_field.Initialize()
    force_field.Minimize(maxIts=400)

    conformer = candidate.GetConformer()
    new_heavy_indices = range(parent_count, candidate.GetNumHeavyAtoms())
    clashes = _scan_clashes(candidate, protein_atoms, new_heavy_indices)
    parent_properties = {
        "canonical_smiles": Chem.MolToSmiles(parent, isomericSmiles=True),
        "formal_charge": parent_charge,
        "heavy_atoms": parent_count,
        "molecular_weight": round(Descriptors.MolWt(parent), 2),
        "logp": round(Crippen.MolLogP(parent), 2),
        "hbd": Lipinski.NumHDonors(parent),
        "hba": Lipinski.NumHAcceptors(parent),
        "tpsa": round(rdMolDescriptors.CalcTPSA(parent), 2),
        "rotatable_bonds": Lipinski.NumRotatableBonds(parent),
    }
    candidate_properties = {
        "canonical_smiles": Chem.MolToSmiles(Chem.RemoveHs(candidate), isomericSmiles=True),
        "formal_charge": candidate_charge,
        "heavy_atoms": candidate.GetNumHeavyAtoms(),
        "molecular_weight": round(Descriptors.MolWt(candidate), 2),
        "logp": round(Crippen.MolLogP(candidate), 2),
        "hbd": Lipinski.NumHDonors(candidate),
        "hba": Lipinski.NumHAcceptors(candidate),
        "tpsa": round(rdMolDescriptors.CalcTPSA(candidate), 2),
        "rotatable_bonds": Lipinski.NumRotatableBonds(candidate),
    }
    property_delta = {
        key: round(candidate_properties[key] - parent_properties[key], 2)
        for key in (
            "formal_charge",
            "heavy_atoms",
            "molecular_weight",
            "logp",
            "hbd",
            "hba",
            "tpsa",
            "rotatable_bonds",
        )
    }
    added_atoms = []
    for index in new_heavy_indices:
        point = conformer.GetAtomPosition(index)
        added_atoms.append(
            {
                "candidate_atom_index": index,
                "element": candidate.GetAtomWithIdx(index).GetSymbol(),
                "xyz": [round(point.x, 3), round(point.y, 3), round(point.z, 3)],
            }
        )
    worst_clash = clashes[0] if clashes else None
    worst_clash_residue = worst_clash["protein_atom"].rsplit(":", 1)[0] if worst_clash else None
    blocking_residues = _blocking_residues(clashes)
    report = {
        **candidate_properties,
        "heavy_atom_delta": property_delta["heavy_atoms"],
        "parent": parent_properties,
        "candidate": candidate_properties,
        "property_delta": property_delta,
        "structure_change": {
            "edit_atom_index": anchor_index,
            "fragment_smiles": fragment_smiles,
            "added_atoms": added_atoms,
            "preserved_parent_heavy_atoms": parent_count,
        },
        "severe_clash_count": len(clashes),
        "severe_clashes": clashes[:20],
        "status": "accepted" if not clashes else "rejected",
        "failure_class": "none" if not clashes else "steric_clash",
        "anchor_atom": anchor_index,
        "fragment_smiles": fragment_smiles,
        "worst_clash_residue": worst_clash_residue,
        "worst_overlap": worst_clash["vdw_overlap"] if worst_clash else None,
        "growth_direction_blockers": blocking_residues,
        "recommended_next_queries": [] if not clashes else [
            "get_atom_environment",
            "check_growth_space",
            "validate_candidate_geometry",
        ],
        "rejection_details": {
            "failure_class": None if not clashes else "steric_clash",
            "anchor_atom": anchor_index,
            "fragment_smiles": fragment_smiles,
            "worst_clash": worst_clash,
            "worst_clash_residue": worst_clash_residue,
            "worst_overlap": worst_clash["vdw_overlap"] if worst_clash else None,
            "blocking_residues": blocking_residues,
            "growth_direction_blockers": blocking_residues,
            "recommended_next_queries": [] if not clashes else [
                "get_atom_environment",
                "check_growth_space",
                "validate_candidate_geometry",
            ],
            "message": (
                "Candidate passed deterministic geometry checks."
                if not clashes
                else "The proposed fragment creates rigid-receptor VDW overlap; query the blocking site or revise the fragment."
            ),
        },
        "limitation": "Rigid receptor and constrained parent scaffold; this is not docking or a binding-affinity prediction.",
    }
    return EditResult(molecule=candidate, report=annotate_edit_layer(parent, candidate, report))


def _retained_fragment(parent: Chem.Mol, cut_bond: tuple[int, int]) -> tuple[Chem.Mol, int, dict[int, int]]:
    """Remove one non-ring side-chain bond and return the retained 3D scaffold."""
    left, right = cut_bond
    if left == right or not (0 <= left < parent.GetNumAtoms()) or not (0 <= right < parent.GetNumAtoms()):
        raise ValueError(f"Invalid cut_bond: {cut_bond!r}")
    bond = parent.GetBondBetweenAtoms(left, right)
    if bond is None:
        raise ValueError(f"No bond exists for cut_bond: {cut_bond!r}")
    if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
        raise ValueError("A bond site only permits cutting a non-ring single bond")

    graph = Chem.RWMol(parent)
    graph.RemoveBond(left, right)
    components = Chem.GetMolFrags(graph.GetMol(), asMols=False, sanitizeFrags=False)
    if len(components) != 2:
        raise ValueError("cut_bond must split the ligand into exactly two components")
    retained_atoms = set(components[0]) if left in components[0] else set(components[1])
    removed_atoms = set(range(parent.GetNumAtoms())) - retained_atoms
    if not removed_atoms or not retained_atoms:
        raise ValueError("cut_bond cannot remove the complete ligand")
    if right in retained_atoms:
        retained_atoms, removed_atoms = removed_atoms, retained_atoms

    rw = Chem.RWMol()
    old_to_new: dict[int, int] = {}
    for old_index in sorted(retained_atoms):
        old_to_new[old_index] = rw.AddAtom(Chem.Atom(parent.GetAtomWithIdx(old_index)))
    for bond in parent.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin in retained_atoms and end in retained_atoms:
            rw.AddBond(old_to_new[begin], old_to_new[end], bond.GetBondType())
            new_bond = rw.GetBondBetweenAtoms(old_to_new[begin], old_to_new[end])
            new_bond.SetIsAromatic(bond.GetIsAromatic())
    scaffold = rw.GetMol()
    conformer = Chem.Conformer(scaffold.GetNumAtoms())
    conformer.Set3D(True)
    source_conformer = parent.GetConformer()
    for old_index, new_index in old_to_new.items():
        point = source_conformer.GetAtomPosition(old_index)
        conformer.SetAtomPosition(new_index, Point3D(point.x, point.y, point.z))
    scaffold.AddConformer(conformer)
    Chem.SanitizeMol(scaffold)
    retained_anchor = old_to_new[left]
    return scaffold, retained_anchor, old_to_new


def apply_bond_replacement(
    parent: Chem.Mol,
    cut_bond: tuple[int, int],
    fragment_smiles: str,
    protein_atoms: list[PDBAtom],
    seed: int = 17,
) -> EditResult:
    """``bond:replacement`` - swap the removed side of a cut bond for a library fragment."""
    original = Chem.RemoveHs(Chem.Mol(parent))
    scaffold, anchor_index, old_to_new = _retained_fragment(original, cut_bond)
    result = apply_atom_addition(scaffold, anchor_index, fragment_smiles, protein_atoms, seed=seed)
    original_charge = Chem.GetFormalCharge(original)
    candidate_charge = Chem.GetFormalCharge(result.molecule)
    if candidate_charge != original_charge:
        raise ValueError(
            f"Formal charge changed from original ligand {original_charge} to {candidate_charge}"
        )
    original_properties = {
        "canonical_smiles": Chem.MolToSmiles(original, isomericSmiles=True),
        "formal_charge": original_charge,
        "heavy_atoms": original.GetNumHeavyAtoms(),
        "molecular_weight": round(Descriptors.MolWt(original), 2),
        "logp": round(Crippen.MolLogP(original), 2),
        "hbd": Lipinski.NumHDonors(original),
        "hba": Lipinski.NumHAcceptors(original),
        "tpsa": round(rdMolDescriptors.CalcTPSA(original), 2),
        "rotatable_bonds": Lipinski.NumRotatableBonds(original),
    }
    candidate_properties = result.report["candidate"]
    result.report["parent"] = original_properties
    result.report["property_delta"] = {
        key: round(candidate_properties[key] - original_properties[key], 2)
        for key in (
            "formal_charge",
            "heavy_atoms",
            "molecular_weight",
            "logp",
            "hbd",
            "hba",
            "tpsa",
            "rotatable_bonds",
        )
    }
    result.report["heavy_atom_delta"] = result.report["property_delta"]["heavy_atoms"]
    result.report["operation"] = "bond:replacement"
    result.report["structure_change"]["cut_bond"] = list(cut_bond)
    result.report["structure_change"]["retained_atom_indices"] = sorted(old_to_new)
    result.report["structure_change"]["removed_atom_indices"] = sorted(
        set(range(parent.GetNumAtoms())) - set(old_to_new)
    )
    result.report["structure_change"]["edit_atom_index"] = cut_bond[0]
    result.report = annotate_edit_layer(original, result.molecule, result.report)
    return result


def _element_replacement_graph(molecule: Chem.Mol, atom_index: int, element: str) -> tuple[Chem.Mol, str, bool]:
    """Graph-only element swap used by both the 3D executor and the dedup builder."""
    molecule = Chem.RemoveHs(Chem.Mol(molecule))
    if not isinstance(atom_index, int) or not 0 <= atom_index < molecule.GetNumAtoms():
        raise ValueError(f"Invalid element-replacement atom index: {atom_index!r}")
    if not isinstance(element, str) or not element.strip():
        raise ValueError("element replacement requires an element symbol")
    table = Chem.GetPeriodicTable()
    try:
        target_number = table.GetAtomicNumber(element.strip().capitalize())
    except Exception as error:
        raise ValueError(f"Unknown element for replacement: {element!r}") from error
    atom = molecule.GetAtomWithIdx(atom_index)
    source_element = atom.GetSymbol()
    if target_number == atom.GetAtomicNum():
        raise ValueError(f"Atom {atom_index} already has element {source_element}")
    was_aromatic = bool(atom.GetIsAromatic())
    editable = Chem.RWMol(molecule)
    edited = editable.GetAtomWithIdx(atom_index)
    edited.SetAtomicNum(target_number)
    edited.SetNoImplicit(False)
    edited.SetNumExplicitHs(0)
    edited.SetNumRadicalElectrons(0)
    edited.SetIsAromatic(False)
    product = editable.GetMol()
    try:
        Chem.SanitizeMol(product)
    except Exception as error:
        raise ValueError(
            f"Swapping {source_element} for {element} at atom {atom_index} is not chemically valid: {error}"
        ) from error
    return product, source_element, was_aromatic


def apply_atom_replacement(
    parent: Chem.Mol,
    atom_index: int,
    element: str,
    protein_atoms: list[PDBAtom],
    seed: int = 17,
) -> EditResult:
    """``atom:replacement`` / ``ring:replacement`` - swap one heavy atom's element.

    Covers halogen and heteroatom exchanges such as ``Cl -> F``, chain substitutions
    like ``O -> CH2``, and aromatic ``CH -> N``.  Coordinates are inherited from the
    parent, so only the edited atom is re-scanned for clashes.
    """
    parent = Chem.RemoveHs(Chem.Mol(parent))
    if not isinstance(atom_index, int) or not 0 <= atom_index < parent.GetNumAtoms():
        raise ValueError(f"Invalid element-swap atom index: {atom_index!r}")
    if not isinstance(element, str) or not element.strip():
        raise ValueError("element replacement requires an element symbol")
    table = Chem.GetPeriodicTable()
    try:
        target_number = table.GetAtomicNumber(element.strip().capitalize())
    except Exception as error:
        raise ValueError(f"Unknown element for replacement: {element!r}") from error
    atom = parent.GetAtomWithIdx(atom_index)
    source_element = atom.GetSymbol()
    if target_number == atom.GetAtomicNum():
        raise ValueError(f"Atom {atom_index} already has element {source_element}")

    editable = Chem.RWMol(parent)
    edited = editable.GetAtomWithIdx(atom_index)
    was_aromatic = edited.GetIsAromatic()
    edited.SetAtomicNum(target_number)
    edited.SetNoImplicit(False)
    edited.SetNumExplicitHs(0)
    edited.SetNumRadicalElectrons(0)
    edited.SetIsAromatic(False)
    candidate = editable.GetMol()
    try:
        Chem.SanitizeMol(candidate)
    except Exception as error:
        raise ValueError(
            f"Swapping {source_element} for {element} at atom {atom_index} is not chemically valid: {error}"
        ) from error

    parent_charge = Chem.GetFormalCharge(parent)
    candidate_charge = Chem.GetFormalCharge(candidate)
    if candidate_charge != parent_charge:
        raise ValueError(f"Formal charge changed from {parent_charge} to {candidate_charge}")

    parent_conformer = parent.GetConformer()
    conformer = Chem.Conformer(candidate.GetNumAtoms())
    conformer.Set3D(True)
    for index in range(candidate.GetNumAtoms()):
        point = parent_conformer.GetAtomPosition(index)
        conformer.SetAtomPosition(index, Point3D(point.x, point.y, point.z))
    candidate.RemoveAllConformers()
    candidate.AddConformer(conformer, assignId=True)

    # The edited atom is a different chemical atom now, so it must not carry the
    # parent's reference identity.  Keeping it made the pose-retention provenance
    # check raise "Retained atom chemistry changed" for every element swap, which
    # made the whole edit class structurally unevaluable.
    if candidate.GetAtomWithIdx(atom_index).HasProp("_reference_atom_index"):
        candidate.GetAtomWithIdx(atom_index).ClearProp("_reference_atom_index")

    clashes = _scan_clashes(candidate, protein_atoms, range(atom_index, atom_index + 1))
    candidate_properties = {
        "canonical_smiles": Chem.MolToSmiles(candidate, isomericSmiles=True),
        "formal_charge": candidate_charge,
        "heavy_atoms": candidate.GetNumHeavyAtoms(),
        "molecular_weight": round(Descriptors.MolWt(candidate), 2),
        "logp": round(Crippen.MolLogP(candidate), 2),
        "hbd": Lipinski.NumHDonors(candidate),
        "hba": Lipinski.NumHAcceptors(candidate),
        "tpsa": round(rdMolDescriptors.CalcTPSA(candidate), 2),
        "rotatable_bonds": Lipinski.NumRotatableBonds(candidate),
    }
    parent_properties = {
        "canonical_smiles": Chem.MolToSmiles(parent, isomericSmiles=True),
        "formal_charge": parent_charge,
        "heavy_atoms": parent.GetNumHeavyAtoms(),
        "molecular_weight": round(Descriptors.MolWt(parent), 2),
        "logp": round(Crippen.MolLogP(parent), 2),
        "hbd": Lipinski.NumHDonors(parent),
        "hba": Lipinski.NumHAcceptors(parent),
        "tpsa": round(rdMolDescriptors.CalcTPSA(parent), 2),
        "rotatable_bonds": Lipinski.NumRotatableBonds(parent),
    }
    worst_clash = clashes[0] if clashes else None
    blocking_residues = _blocking_residues(clashes)
    report = {
        **candidate_properties,
        "operation": "atom:replacement",
        "heavy_atom_delta": 0,
        "parent": parent_properties,
        "candidate": candidate_properties,
        "property_delta": {
            key: round(candidate_properties[key] - parent_properties[key], 2)
            for key in (
                "formal_charge", "heavy_atoms", "molecular_weight", "logp",
                "hbd", "hba", "tpsa", "rotatable_bonds",
            )
        },
        "structure_change": {
            "edit_atom_index": atom_index,
            "from_element": source_element,
            "to_element": candidate.GetAtomWithIdx(atom_index).GetSymbol(),
            "was_aromatic": bool(was_aromatic),
            "changed_atoms": [
                {
                    "candidate_atom_index": atom_index,
                    "element": candidate.GetAtomWithIdx(atom_index).GetSymbol(),
                    "heavy_atom_count_unchanged": True,
                }
            ],
            "preserved_parent_heavy_atoms": parent.GetNumAtoms(),
        },
        "severe_clash_count": len(clashes),
        "severe_clashes": clashes[:20],
        "status": "accepted" if not clashes else "rejected",
        "failure_class": "none" if not clashes else "steric_clash",
        "edit_atom_index": atom_index,
        "worst_clash_residue": worst_clash["protein_atom"].rsplit(":", 1)[0] if worst_clash else None,
        "worst_overlap": worst_clash["vdw_overlap"] if worst_clash else None,
        "growth_direction_blockers": blocking_residues,
        "recommended_next_queries": [] if not clashes else ["validate_candidate_geometry"],
        "limitation": (
            "The edited atom keeps its parent coordinates; only that atom is re-scanned for "
            "rigid-receptor overlap. This is not docking or an affinity prediction."
        ),
    }
    return EditResult(
        molecule=candidate,
        report=annotate_edit_layer(parent, candidate, report),
    )


def apply_bond_deletion(
    parent: Chem.Mol,
    cut_bond: tuple[int, int],
    protein_atoms: list[PDBAtom],
    seed: int = 17,
) -> EditResult:
    """``bond:deletion`` - cut a non-ring side-chain bond and leave hydrogen behind.

    The removed side simply disappears; the retained scaffold keeps its parent
    coordinates and the cut anchor gains an implicit hydrogen.  No heavy atom is
    added, so this cannot introduce a new rigid-receptor clash.
    """
    original = Chem.RemoveHs(Chem.Mol(parent))
    original_properties = _molecule_properties(original)
    scaffold, retained_anchor, old_to_new = _retained_fragment(original, tuple(cut_bond))
    if Chem.GetFormalCharge(scaffold) != Chem.GetFormalCharge(original):
        raise ValueError(
            f"Formal charge changed from original ligand {Chem.GetFormalCharge(original)} "
            f"to {Chem.GetFormalCharge(scaffold)}"
        )
    candidate = Chem.AddHs(scaffold, addCoords=True)
    Chem.SanitizeMol(candidate)
    candidate_properties = _molecule_properties(candidate)
    removed_indices = sorted(set(range(original.GetNumAtoms())) - set(old_to_new))
    report = {
        **candidate_properties,
        "operation": "bond:deletion",
        "heavy_atom_delta": candidate_properties["heavy_atoms"] - original_properties["heavy_atoms"],
        "parent": original_properties,
        "candidate": candidate_properties,
        "property_delta": _property_delta(original_properties, candidate_properties),
        "structure_change": {
            "edit_atom_index": retained_anchor,
            "source_atom_index": tuple(cut_bond)[0],
            "cut_bond": list(cut_bond),
            "retained_atom_indices": sorted(old_to_new),
            "removed_atom_indices": removed_indices,
            "removed_heavy_atoms": len(removed_indices),
            "preserved_parent_heavy_atoms": len(old_to_new),
        },
        "severe_clash_count": 0,
        "severe_clashes": [],
        "status": "accepted",
        "failure_class": "none",
        "anchor_atom": retained_anchor,
        "worst_clash_residue": None,
        "worst_overlap": None,
        "growth_direction_blockers": [],
        "recommended_next_queries": [],
        "limitation": (
            "Deletion removes heavy atoms and cannot create a new rigid-receptor clash. "
            "The consequence is lost contacts, which is reported separately by the caller."
        ),
    }
    return EditResult(
        molecule=candidate,
        report=annotate_edit_layer(original, candidate, report),
    )


def _attach_fragment_graph(
    scaffold: Chem.Mol, anchor_index: int, fragment_smiles: str
) -> Chem.Mol:
    scaffold = Chem.RemoveHs(Chem.Mol(scaffold))
    fragment, dummy_index, attachment_index = _fragment(fragment_smiles)
    scaffold_count = scaffold.GetNumAtoms()
    rw = Chem.RWMol(Chem.CombineMols(scaffold, fragment))
    rw.AddBond(anchor_index, scaffold_count + attachment_index, Chem.BondType.SINGLE)
    rw.RemoveAtom(scaffold_count + dummy_index)
    product = rw.GetMol()
    Chem.SanitizeMol(product)
    return product


def transformation_product_smiles(
    parent: Chem.Mol, transformation: dict[str, Any]
) -> str:
    """Build the edited molecular graph without 3D embedding for exact deduplication."""
    site_type, change_type = resolve_edit_axes(transformation)
    original = Chem.RemoveHs(Chem.Mol(parent))
    if site_type in {"atom", "ring"} and change_type == "replacement":
        product, _source, _aromatic = _element_replacement_graph(
            original, transformation.get("edit_atom_index"), transformation.get("element")
        )
    elif site_type == "atom" and change_type == "addition":
        fragment_smiles = transformation.get("fragment_smiles")
        if not isinstance(fragment_smiles, str):
            raise ValueError("atom:addition requires fragment_smiles")
        atom_index = transformation.get("edit_atom_index")
        if not isinstance(atom_index, int) or not 0 <= atom_index < original.GetNumAtoms():
            raise ValueError("atom:addition requires a valid integer edit_atom_index")
        if original.GetAtomWithIdx(atom_index).GetTotalNumHs() < 1:
            raise ValueError("Anchor atom has no replaceable hydrogen")
        product = _attach_fragment_graph(original, atom_index, fragment_smiles)
    elif site_type == "bond" and change_type == "replacement":
        fragment_smiles = transformation.get("fragment_smiles")
        if not isinstance(fragment_smiles, str):
            raise ValueError("bond:replacement requires fragment_smiles")
        retained, removed = _resolved_cut_bond(original, transformation)
        scaffold, retained_anchor, _mapping = _retained_fragment(original, (retained, removed))
        product = _attach_fragment_graph(scaffold, retained_anchor, fragment_smiles)
    elif site_type == "bond" and change_type == "deletion":
        retained, removed = _resolved_cut_bond(original, transformation)
        scaffold, _anchor, _mapping = _retained_fragment(original, (retained, removed))
        product = scaffold
    else:
        raise ValueError(f"Unsupported transformation {site_type}:{change_type}")
    if Chem.GetFormalCharge(product) != Chem.GetFormalCharge(original):
        raise ValueError("Transformation changes the formal charge")
    return Chem.MolToSmiles(product, isomericSmiles=True)


def resolve_edit_axes(transformation: dict[str, Any]) -> tuple[str, str]:
    """Return ``(site_type, change_type)`` from the canonical two-axis schema."""
    normalized = normalize_transformation(transformation)
    return normalized["site_type"], normalized["change_type"]


def _resolved_cut_bond(molecule: Chem.Mol, transformation: dict[str, Any]) -> tuple[int, int]:
    """Return the directed ``(retained, removed)`` bond for a bond site.

    A local child carries ``replace_existing_substituent``: its cut bond is resolved
    on the parent molecule at ``edit_atom_index`` rather than copied from the
    reference ligand.
    """
    if transformation.get("replace_existing_substituent"):
        anchor_index = transformation.get("edit_atom_index")
        if not isinstance(anchor_index, int) or not 0 <= anchor_index < molecule.GetNumAtoms():
            raise ValueError("A resolved bond site requires a valid integer edit_atom_index")
        anchor = molecule.GetAtomWithIdx(anchor_index)
        removable = [
            bond for bond in anchor.GetBonds()
            if not bond.IsInRing()
            and bond.GetBondType() == Chem.BondType.SINGLE
            and bond.GetOtherAtom(anchor).GetAtomicNum() > 1
        ]
        requested_neighbor = transformation.get("remove_neighbor_index")
        if requested_neighbor is not None:
            removable = [
                bond for bond in removable
                if bond.GetOtherAtom(anchor).GetIdx() == requested_neighbor
            ]
        if not removable:
            raise ValueError("Parent has no removable substituent at the selected anchor")
        removable.sort(key=lambda bond: bond.GetOtherAtom(anchor).GetIdx(), reverse=True)
        return anchor_index, removable[0].GetOtherAtom(anchor).GetIdx()
    cut_bond = transformation.get("cut_bond")
    if not isinstance(cut_bond, list) or len(cut_bond) != 2 or not all(
        isinstance(item, int) for item in cut_bond
    ):
        raise ValueError("A bond site requires cut_bond: [retained_atom, removed_side_atom]")
    return cut_bond[0], cut_bond[1]


def apply_transformation(
    parent: Chem.Mol,
    transformation: dict[str, Any],
    protein_atoms: list[PDBAtom],
    seed: int = 17,
) -> EditResult:
    """Dispatch a validated host-side transformation on the (site_type, change_type) axes."""
    site_type, change_type = resolve_edit_axes(transformation)
    operation = f"{site_type}:{change_type}"
    if site_type in {"atom", "ring"} and change_type == "replacement":
        atom_index = transformation.get("edit_atom_index", transformation.get("atom_index"))
        if not isinstance(atom_index, int):
            raise ValueError("atom/ring replacement requires integer edit_atom_index")
        element = transformation.get("element")
        if not isinstance(element, str) or not element.strip():
            raise ValueError("atom/ring replacement requires an element symbol")
        result = apply_atom_replacement(parent, atom_index, element, protein_atoms, seed=seed)
    elif site_type == "atom" and change_type == "addition":
        fragment_smiles = transformation.get("fragment_smiles")
        if not isinstance(fragment_smiles, str):
            raise ValueError("atom:addition requires fragment_smiles")
        atom_index = transformation.get("edit_atom_index")
        if not isinstance(atom_index, int):
            raise ValueError("atom:addition requires integer edit_atom_index")
        result = apply_atom_addition(parent, atom_index, fragment_smiles, protein_atoms, seed=seed)
    elif site_type == "bond" and change_type == "replacement":
        fragment_smiles = transformation.get("fragment_smiles")
        if not isinstance(fragment_smiles, str):
            raise ValueError("bond:replacement requires fragment_smiles")
        original = Chem.RemoveHs(Chem.Mol(parent))
        retained, removed = _resolved_cut_bond(original, transformation)
        result = apply_bond_replacement(
            parent, (retained, removed), fragment_smiles, protein_atoms, seed=seed
        )
    elif site_type == "bond" and change_type == "deletion":
        original = Chem.RemoveHs(Chem.Mol(parent))
        retained, removed = _resolved_cut_bond(original, transformation)
        result = apply_bond_deletion(parent, (retained, removed), protein_atoms, seed=seed)
    else:
        raise ValueError(f"Unsupported transformation {site_type}:{change_type}")
    result.report["site_type"] = site_type
    result.report["change_type"] = change_type
    result.report["operation"] = operation
    if transformation.get("replace_existing_substituent"):
        result.report["replaced_existing_substituent"] = True
    return result


def write_sdf(result: EditResult, path: Path, name: str = "candidate") -> None:
    result.molecule.SetProp("_Name", name)
    # Molecule-level SDF property survives writing; atom provenance survives graph edits.
    # Hydrogens and new fragment atoms deliberately have no reference identity.
    import json
    result.molecule.SetProp("reference_atom_indices", json.dumps([
        atom.GetIntProp("_reference_atom_index")
        if atom.HasProp("_reference_atom_index") and atom.GetAtomicNum() != 1 else None
        for atom in result.molecule.GetAtoms()
    ]))
    writer = Chem.SDWriter(str(path))
    writer.write(result.molecule)
    writer.close()
