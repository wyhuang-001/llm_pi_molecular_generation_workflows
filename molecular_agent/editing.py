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
        "limitation": "Rigid receptor and constrained parent scaffold; this is not docking or a binding-affinity prediction.",
    }
    return EditResult(molecule=candidate, report=report)


def write_sdf(result: EditResult, path: Path, name: str = "candidate") -> None:
    result.molecule.SetProp("_Name", name)
    writer = Chem.SDWriter(str(path))
    writer.write(result.molecule)
    writer.close()
