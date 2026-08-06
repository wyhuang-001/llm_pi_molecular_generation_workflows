from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem


@dataclass(frozen=True)
class PDBAtom:
    record: str
    serial: int
    name: str
    residue_name: str
    chain: str
    residue_number: int
    element: str
    xyz: np.ndarray


def parse_pdb(path: Path) -> list[PDBAtom]:
    atoms: list[PDBAtom] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line[:6].strip() not in {"ATOM", "HETATM"}:
            continue
        if line[16:17] not in {" ", "A"}:
            continue
        name = line[12:16].strip()
        element = line[76:78].strip().upper() or name.lstrip("0123456789")[:1].upper()
        atoms.append(
            PDBAtom(
                record=line[:6].strip(),
                serial=int(line[6:11]),
                name=name,
                residue_name=line[17:20].strip(),
                chain=line[21:22].strip(),
                residue_number=int(line[22:26]),
                element=element,
                xyz=np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=float,
                ),
            )
        )
    if not atoms:
        raise ValueError(f"No atoms found in {path}")
    return atoms


def _selector_key(atom: PDBAtom) -> tuple[str, str, int]:
    return atom.chain, atom.residue_name, atom.residue_number


def _pdb_ligand_block(path: Path, selected: list[PDBAtom]) -> str:
    selected_serials = {atom.serial for atom in selected}
    source_lines = path.read_text(encoding="utf-8").splitlines()
    atom_lines = [
        line
        for line in source_lines
        if line[:6].strip() == "HETATM" and int(line[6:11]) in selected_serials
    ]
    conect_lines = []
    for line in source_lines:
        if line[:6].strip() != "CONECT":
            continue
        values = [
            int(line[index : index + 5])
            for index in range(6, len(line), 5)
            if line[index : index + 5].strip()
        ]
        if not values or values[0] not in selected_serials:
            continue
        kept = [values[0], *[value for value in values[1:] if value in selected_serials]]
        if len(kept) > 1:
            conect_lines.append("CONECT" + "".join(f"{value:5d}" for value in kept))
    return "\n".join([*atom_lines, *conect_lines, "END", ""])


class ComplexContext:
    def __init__(self, task_path: Path):
        self.task_path = task_path.resolve()
        self.input_dir = self.task_path.parent
        self.task = json.loads(self.task_path.read_text(encoding="utf-8"))
        self.complex_path = (self.input_dir / self.task["complex_path"]).resolve()
        self.atoms = parse_pdb(self.complex_path)
        self.protein_atoms = [atom for atom in self.atoms if atom.record == "ATOM"]
        self.ligand_pdb_atoms = self._select_ligand_atoms()
        self.ligand_selector = {
            "chain": self.ligand_pdb_atoms[0].chain,
            "residue_name": self.ligand_pdb_atoms[0].residue_name,
            "residue_number": self.ligand_pdb_atoms[0].residue_number,
        }
        block = _pdb_ligand_block(self.complex_path, self.ligand_pdb_atoms)
        self.ligand = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=True)
        if self.ligand is None:
            raise ValueError("Could not reconstruct ligand graph from PDB HETATM/CONECT records")
        if self.ligand.GetNumConformers() != 1 or not self.ligand.GetConformer().Is3D():
            raise ValueError("PDB ligand must contain one 3D conformer")
        if self.ligand.GetNumHeavyAtoms() != len(self.ligand_pdb_atoms):
            raise ValueError(
                "PDB ligand graph/coordinate mismatch: "
                f"{self.ligand.GetNumHeavyAtoms()} vs {len(self.ligand_pdb_atoms)}"
            )
        self.ligand_source = "PDB HETATM/CONECT; no separate ligand file required"

    def _select_ligand_atoms(self) -> list[PDBAtom]:
        selector = self.task.get("ligand_selector")
        groups: dict[tuple[str, str, int], list[PDBAtom]] = {}
        for atom in self.atoms:
            if atom.record != "HETATM" or atom.residue_name in {"HOH", "WAT", "DOD"}:
                continue
            groups.setdefault(_selector_key(atom), []).append(atom)
        if selector:
            selected = [
                atom
                for atom in self.atoms
                if atom.record == "HETATM"
                and atom.chain == selector["chain"]
                and atom.residue_name == selector["residue_name"]
                and atom.residue_number == int(selector["residue_number"])
            ]
            if not selected:
                raise ValueError(f"Ligand selector matched no atoms: {selector}")
            return selected
        candidates = [items for items in groups.values() if len(items) >= 5]
        if not candidates:
            raise ValueError("No non-water HETATM group is large enough to identify a ligand")
        candidates.sort(key=len, reverse=True)
        if len(candidates) > 1 and len(candidates[0]) == len(candidates[1]):
            choices = [
                {"chain": x[0].chain, "residue_name": x[0].residue_name, "residue_number": x[0].residue_number, "atoms": len(x)}
                for x in candidates[:8]
            ]
            raise ValueError(f"Multiple equally large ligand candidates; add ligand_selector: {choices}")
        return candidates[0]

    def protein_near(self, xyz: np.ndarray, radius: float) -> list[tuple[PDBAtom, float]]:
        nearby = []
        for atom in self.protein_atoms:
            distance = float(np.linalg.norm(atom.xyz - xyz))
            if distance <= radius:
                nearby.append((atom, distance))
        return sorted(nearby, key=lambda item: item[1])

    def ligand_atom_rows(self) -> list[dict[str, Any]]:
        conformer = self.ligand.GetConformer()
        rows = []
        for atom in self.ligand.GetAtoms():
            position = conformer.GetAtomPosition(atom.GetIdx())
            rows.append(
                {
                    "index": atom.GetIdx(),
                    "element": atom.GetSymbol(),
                    "aromatic": atom.GetIsAromatic(),
                    "replaceable_hydrogens": atom.GetTotalNumHs(),
                    "xyz": [round(position.x, 3), round(position.y, 3), round(position.z, 3)],
                }
            )
        return rows
