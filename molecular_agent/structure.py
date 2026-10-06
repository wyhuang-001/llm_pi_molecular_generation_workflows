from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import gemmi
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


def _pdb_ligand_bonds(path: Path, selected: list[PDBAtom]) -> set[frozenset[str]]:
    names_by_serial = {atom.serial: atom.name for atom in selected}
    bonds: set[frozenset[str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line[:6].strip() != "CONECT":
            continue
        serials = [
            int(line[index : index + 5])
            for index in range(6, len(line), 5)
            if line[index : index + 5].strip()
        ]
        if not serials or serials[0] not in names_by_serial:
            continue
        for neighbor in serials[1:]:
            if neighbor in names_by_serial:
                bonds.add(frozenset((names_by_serial[serials[0]], names_by_serial[neighbor])))
    if not bonds:
        raise ValueError("Selected ligand has no PDB CONECT records")
    return bonds


def _component_path(input_dir: Path, residue_name: str) -> Path:
    return input_dir / "raw" / f"{residue_name}.cif"


def _cif_charge(value: str) -> int:
    if value in {".", "?", ""}:
        return 0
    charge = float(value)
    if not charge.is_integer():
        raise ValueError(f"Chemical component charge is not integral: {value}")
    return int(charge)


def _bond_type(value_order: str) -> Chem.BondType:
    try:
        return {
            "SING": Chem.BondType.SINGLE,
            "DOUB": Chem.BondType.DOUBLE,
            "TRIP": Chem.BondType.TRIPLE,
            "AROM": Chem.BondType.AROMATIC,
        }[value_order]
    except KeyError as exc:
        raise ValueError(f"Unsupported chemical component bond order: {value_order}") from exc


def _build_ligand_from_component(
    component_path: Path,
    selected: list[PDBAtom],
    expected_bonds: set[frozenset[str]],
) -> Chem.Mol:
    if not component_path.exists():
        raise ValueError(
            "Reliable ligand topology requires the local chemical component file: "
            f"{component_path}"
        )
    block = gemmi.cif.read_file(str(component_path)).sole_block()
    atom_table = block.find(
        "_chem_comp_atom.",
        ["atom_id", "type_symbol", "charge", "pdbx_aromatic_flag"],
    )
    component_atoms = [list(row) for row in atom_table if list(row)[1] != "H"]
    component_by_name = {row[0]: row for row in component_atoms}
    selected_names = [atom.name for atom in selected]
    selected_by_name = {atom.name: atom for atom in selected}
    component_names = list(component_by_name)
    if len(selected_names) != len(set(selected_names)):
        raise ValueError("Selected ligand has duplicate atom names")
    if set(selected_names) != set(component_names):
        raise ValueError(
            "PDB ligand atoms do not match the chemical component atoms: "
            f"pdb_only={sorted(set(selected_names) - set(component_names))}, "
            f"component_only={sorted(set(component_names) - set(selected_names))}"
        )

    rw_mol = Chem.RWMol()
    atom_indices: dict[str, int] = {}
    for atom_name in selected_names:
        _, element, charge, aromatic_flag = component_by_name[atom_name]
        pdb_atom = selected_by_name[atom_name]
        if element.upper() != pdb_atom.element.upper():
            raise ValueError(
                f"Element mismatch for ligand atom {atom_name}: "
                f"PDB={pdb_atom.element}, component={element}"
            )
        # CCD commonly uses upper-case two-letter symbols (e.g. CL); RDKit needs Cl.
        rdkit_atom = Chem.Atom(element.capitalize())
        rdkit_atom.SetFormalCharge(_cif_charge(charge))
        if aromatic_flag == "Y":
            rdkit_atom.SetIsAromatic(True)
        rdkit_atom.SetProp("atom_name", atom_name)
        rdkit_atom.SetIntProp("pdb_serial", pdb_atom.serial)
        atom_indices[atom_name] = rw_mol.AddAtom(rdkit_atom)

    bond_table = block.find(
        "_chem_comp_bond.",
        ["atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag"],
    )
    seen_bonds: set[frozenset[str]] = set()
    for atom_1, atom_2, value_order, aromatic_flag in (list(row) for row in bond_table):
        if atom_1 not in atom_indices or atom_2 not in atom_indices:
            continue
        key = frozenset((atom_1, atom_2))
        if key in seen_bonds:
            raise ValueError(f"Duplicate chemical component bond: {atom_1}-{atom_2}")
        seen_bonds.add(key)
        rw_mol.AddBond(atom_indices[atom_1], atom_indices[atom_2], _bond_type(value_order))
        bond = rw_mol.GetBondBetweenAtoms(atom_indices[atom_1], atom_indices[atom_2])
        if aromatic_flag == "Y":
            bond.SetIsAromatic(True)
    if seen_bonds != expected_bonds:
        display = lambda bonds: sorted("-".join(sorted(bond)) for bond in bonds)
        raise ValueError(
            "PDB CONECT does not match the chemical component topology: "
            f"pdb_only={display(expected_bonds - seen_bonds)}, "
            f"component_only={display(seen_bonds - expected_bonds)}"
        )

    molecule = rw_mol.GetMol()
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.Set3D(True)
    for atom in selected:
        index = atom_indices[atom.name]
        conformer.SetAtomPosition(index, tuple(float(value) for value in atom.xyz))
    molecule.AddConformer(conformer)
    Chem.SanitizeMol(molecule)
    return molecule


class ComplexContext:
    def __init__(self, task_path: Path):
        self.task_path = task_path.resolve()
        self.input_dir = self.task_path.parent
        self.task = json.loads(self.task_path.read_text(encoding="utf-8"))
        self.complex_path = (self.input_dir / self.task["complex_path"]).resolve()
        self.atoms = parse_pdb(self.complex_path)
        self.retained_hetero_residues = set(self.task.get("receptor_preparation", {}).get(
            "retain_hetero_residue_names", ["TPO", "SEP", "PTR", "MSE"]
        ))
        self.protein_atoms = [atom for atom in self.atoms if atom.record == "ATOM"
                              or atom.residue_name in self.retained_hetero_residues]
        self.ligand_pdb_atoms = self._select_ligand_atoms()
        self.ligand_selector = {
            "chain": self.ligand_pdb_atoms[0].chain,
            "residue_name": self.ligand_pdb_atoms[0].residue_name,
            "residue_number": self.ligand_pdb_atoms[0].residue_number,
        }
        self.protein_atoms = [atom for atom in self.protein_atoms if _selector_key(atom) !=
                              (self.ligand_selector["chain"], self.ligand_selector["residue_name"],
                               self.ligand_selector["residue_number"])]
        self.component_path = _component_path(
            self.complex_path.parent, self.ligand_selector["residue_name"]
        )
        self.ligand = _build_ligand_from_component(
            self.component_path,
            self.ligand_pdb_atoms,
            _pdb_ligand_bonds(self.complex_path, self.ligand_pdb_atoms),
        )
        if self.ligand.GetNumConformers() != 1 or not self.ligand.GetConformer().Is3D():
            raise ValueError("Chemical component ligand must contain one 3D PDB conformer")
        if self.ligand.GetNumHeavyAtoms() != len(self.ligand_pdb_atoms):
            raise ValueError(
                "Chemical component/PDB ligand atom mismatch: "
                f"{self.ligand.GetNumHeavyAtoms()} vs {len(self.ligand_pdb_atoms)}"
            )
        # Stable provenance, independent of the RDKit index after an edit.
        for atom in self.ligand.GetAtoms():
            atom.SetIntProp("_reference_atom_index", atom.GetIdx())
        self.ligand_source = f"PDB coordinates + chemical component topology: {self.component_path}"

    def _select_ligand_atoms(self) -> list[PDBAtom]:
        selector = self.task.get("ligand_selector")
        groups: dict[tuple[str, str, int], list[PDBAtom]] = {}
        for atom in self.atoms:
            # Prepared PDBs can contain explicit ligand H/D. CCD topology and stable
            # reference indices are heavy-atom based; hydrogens are prepared separately.
            if (atom.record != "HETATM" or atom.residue_name in {"HOH", "WAT", "DOD"}
                    or atom.element in {"H", "D", "T"}):
                continue
            groups.setdefault(_selector_key(atom), []).append(atom)
        if selector:
            selected = [
                atom
                for atom in self.atoms
                if atom.record == "HETATM"
                and atom.element not in {"H", "D", "T"}
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

    def preparation_identity(self) -> dict[str, Any]:
        """Tie cached prepared inputs to the source used to construct NEW candidates."""
        return {
            "complex_sha256": hashlib.sha256(self.complex_path.read_bytes()).hexdigest(),
            "component_sha256": hashlib.sha256(self.component_path.read_bytes()).hexdigest(),
            "ligand_selector": self.ligand_selector,
            "receptor_policy": "ATOM plus reviewed HETATM, blank/A altloc, no added receptor H",
            "ligand_hydrogen_policy": "RDKit AddHs(addCoords=True), unchanged crystal heavy coordinates",
        }

    def write_receptor_pdb(self, path: Path) -> Path:
        """Keep protein and explicitly reviewed modified residues, not other ligands/water."""
        original = self.complex_path.read_text(encoding="utf-8").splitlines()
        selected = self.ligand_selector
        lines = [line for line in original
                 if (line[:6].strip() == "ATOM" or
                     (line[:6].strip() == "HETATM" and
                      line[17:20].strip() in self.retained_hetero_residues))
                 and line[16:17] in {" ", "A"}
                 and not (line[21:22].strip() == selected["chain"]
                          and line[17:20].strip() == selected["residue_name"]
                          and int(line[22:26]) == selected["residue_number"])]
        if not lines:
            raise ValueError("Complex contains no protein ATOM records")
        path.parent.mkdir(parents=True, exist_ok=True)
        kept_serials = {int(line[6:11]) for line in lines}
        conect = []
        for line in original:
            if line.startswith("CONECT"):
                ids = [int(line[i:i+5]) for i in range(6, len(line), 5) if line[i:i+5].strip()]
                if ids and ids[0] in kept_serials:
                    ids = [i for i in ids if i in kept_serials]
                    if len(ids) > 1:
                        conect.append("CONECT" + "".join(f"{i:5d}" for i in ids))
        path.write_text("\n".join([*lines, *conect, "END", ""]), encoding="utf-8")
        excluded = sorted({f"{a.residue_name}:{a.chain}:{a.residue_number}"
                           for a in self.atoms if a.serial not in kept_serials})
        manifest = {
            "source": str(self.complex_path), "source_identity": self.preparation_identity(),
            "retained_atom_count": len(lines),
            "retained_hetero_residue_names": sorted(self.retained_hetero_residues),
            "retained_hetero_atom_count": sum(line.startswith("HETATM") for line in lines),
            "excluded_residues": excluded, "altloc_policy": "blank or A",
            "hydrogen_policy": "preserve receptor input; GNINA internal preparation; PLIP nohydro",
            "limitations": "No pH/protonation optimization. Waters/metals/cofactors excluded unless explicitly listed; review before calibration.",
        }
        path.with_suffix(".preparation.json").write_text(json.dumps(manifest, indent=2))
        return path

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
