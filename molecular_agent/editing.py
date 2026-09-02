from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Geometry import Point3D

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


def apply_substituent(
    parent: Chem.Mol,
    anchor_index: int,
    fragment_smiles: str,
    protein_atoms: list[PDBAtom],
    seed: int = 17,
) -> EditResult:
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
    clashes = []
    new_heavy_indices = range(parent_count, candidate.GetNumHeavyAtoms())
    for index in new_heavy_indices:
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
    blocking_residues = []
    for clash in clashes:
        residue = clash["protein_atom"].rsplit(":", 1)[0]
        if residue not in blocking_residues:
            blocking_residues.append(residue)
        if len(blocking_residues) == 5:
            break
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
    return EditResult(molecule=candidate, report=report)


def _retained_fragment(parent: Chem.Mol, cut_bond: tuple[int, int]) -> tuple[Chem.Mol, int, dict[int, int]]:
    """Remove one non-ring side-chain bond and return the retained 3D scaffold."""
    left, right = cut_bond
    if left == right or not (0 <= left < parent.GetNumAtoms()) or not (0 <= right < parent.GetNumAtoms()):
        raise ValueError(f"Invalid cut_bond: {cut_bond!r}")
    bond = parent.GetBondBetweenAtoms(left, right)
    if bond is None:
        raise ValueError(f"No bond exists for cut_bond: {cut_bond!r}")
    if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
        raise ValueError("replace_fragment only permits cutting a non-ring single bond")

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


def apply_fragment_replacement(
    parent: Chem.Mol,
    cut_bond: tuple[int, int],
    fragment_smiles: str,
    protein_atoms: list[PDBAtom],
    seed: int = 17,
) -> EditResult:
    """Replace the component beyond a non-ring cut bond with a library fragment."""
    original = Chem.RemoveHs(Chem.Mol(parent))
    scaffold, anchor_index, old_to_new = _retained_fragment(original, cut_bond)
    result = apply_substituent(scaffold, anchor_index, fragment_smiles, protein_atoms, seed=seed)
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
    result.report["operation"] = "replace_fragment"
    result.report["structure_change"]["cut_bond"] = list(cut_bond)
    result.report["structure_change"]["retained_atom_indices"] = sorted(old_to_new)
    result.report["structure_change"]["removed_atom_indices"] = sorted(
        set(range(parent.GetNumAtoms())) - set(old_to_new)
    )
    result.report["structure_change"]["edit_atom_index"] = cut_bond[0]
    return result


def apply_transformation(
    parent: Chem.Mol,
    transformation: dict[str, Any],
    protein_atoms: list[PDBAtom],
    seed: int = 17,
) -> EditResult:
    """Dispatch a validated host-side transformation."""
    operation = transformation.get("operation", "replace_hydrogen")
    fragment_smiles = transformation.get("fragment_smiles")
    if not isinstance(fragment_smiles, str):
        raise ValueError("Transformation requires fragment_smiles")
    if operation == "replace_hydrogen":
        atom_index = transformation.get("edit_atom_index")
        if not isinstance(atom_index, int):
            raise ValueError("replace_hydrogen requires integer edit_atom_index")
        if transformation.get("replace_existing_substituent"):
            anchor = parent.GetAtomWithIdx(atom_index)
            removable = [
                bond for bond in anchor.GetBonds()
                if not bond.IsInRing()
                and bond.GetBondType() == Chem.BondType.SINGLE
                and bond.GetOtherAtom(anchor).GetAtomicNum() > 1
            ]
            if not removable:
                raise ValueError("Parent has no removable substituent at the selected edit atom")
            requested_neighbor = transformation.get("remove_neighbor_index")
            if requested_neighbor is not None:
                removable = [
                    bond for bond in removable
                    if bond.GetOtherAtom(anchor).GetIdx() == requested_neighbor
                ]
                if not removable:
                    raise ValueError(
                        "remove_neighbor_index is not a removable substituent at the selected edit atom"
                    )
            else:
                removable.sort(key=lambda bond: bond.GetOtherAtom(anchor).GetIdx())
            bond = removable[0]
            result = apply_fragment_replacement(
                parent,
                (atom_index, bond.GetOtherAtom(anchor).GetIdx()),
                fragment_smiles,
                protein_atoms,
                seed=seed,
            )
            result.report["operation"] = operation
            result.report["replaced_existing_substituent"] = True
            return result
        result = apply_substituent(parent, atom_index, fragment_smiles, protein_atoms, seed=seed)
        result.report["operation"] = operation
        return result
    if operation == "replace_fragment":
        cut_bond = transformation.get("cut_bond")
        if not isinstance(cut_bond, list) or len(cut_bond) != 2 or not all(isinstance(x, int) for x in cut_bond):
            raise ValueError("replace_fragment requires cut_bond: [retained_atom, removed_side_atom]")
        return apply_fragment_replacement(parent, (cut_bond[0], cut_bond[1]), fragment_smiles, protein_atoms, seed=seed)
    raise ValueError(f"Unsupported transformation operation: {operation!r}")


def write_sdf(result: EditResult, path: Path, name: str = "candidate") -> None:
    result.molecule.SetProp("_Name", name)
    writer = Chem.SDWriter(str(path))
    writer.write(result.molecule)
    writer.close()
