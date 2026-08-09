from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from .structure import ComplexContext


class ToolRegistry:
    def __init__(self, context: ComplexContext):
        self.context = context
        self._tools: dict[str, tuple[Callable[..., dict[str, Any]], set[str], dict[str, Any]]] = {
            "get_ligand_info": (
                self.get_ligand_info,
                {"ligand_identity"},
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
            "get_pocket_residues": (
                self.get_pocket_residues,
                {"pocket_environment"},
                {
                    "type": "object",
                    "properties": {"radius": {"type": "number", "minimum": 3, "maximum": 8}},
                    "required": ["radius"],
                },
            ),
            "detect_basic_interactions": (
                self.detect_basic_interactions,
                {"key_interactions"},
                {
                    "type": "object",
                    "properties": {"cutoff": {"type": "number", "minimum": 2.5, "maximum": 5}},
                    "required": ["cutoff"],
                },
            ),
            "get_atom_environment": (
                self.get_atom_environment,
                {"edit_site_environment"},
                {
                    "type": "object",
                    "properties": {
                        "atom_index": {"type": "integer", "minimum": 0},
                        "radius": {"type": "number", "minimum": 3, "maximum": 8},
                    },
                    "required": ["atom_index", "radius"],
                },
            ),
            "check_growth_space": (
                self.check_growth_space,
                {"edit_site_geometry"},
                {
                    "type": "object",
                    "properties": {
                        "atom_index": {"type": "integer", "minimum": 0},
                        "distance": {"type": "number", "minimum": 1.0, "maximum": 4.0},
                    },
                    "required": ["atom_index", "distance"],
                },
            ),
            "get_fragment_properties": (
                self.get_fragment_properties,
                {"fragment_properties"},
                {
                    "type": "object",
                    "properties": {
                        "smiles": {"type": "string", "minLength": 1},
                    },
                    "required": ["smiles"],
                },
            ),
            "get_ligand_fragment": (
                self.get_ligand_fragment,
                {"fragment_properties"},
                {
                    "type": "object",
                    "properties": {
                        "atom_index": {"type": "integer", "minimum": 0},
                        "radius_bonds": {"type": "integer", "minimum": 1, "maximum": 4},
                    },
                    "required": ["atom_index", "radius_bonds"],
                },
            ),
        }

    def catalog(self) -> dict[str, Any]:
        requirements = {
            "get_ligand_info": "Any successful call covers ligand identity.",
            "get_pocket_residues": "radius must be at least 5.0 A to cover pocket environment.",
            "detect_basic_interactions": "cutoff must be at least 4.0 A to cover key interactions.",
            "get_atom_environment": "radius must be at least 4.0 A for the final edit atom.",
            "check_growth_space": "probe distance must be at least 1.5 A for the final edit atom.",
            "get_fragment_properties": "Any valid fragment returns deterministic fragment properties.",
            "get_ligand_fragment": "Any valid atom and bond radius returns a local ligand fragment.",
        }
        return {
            name: {
                "input_schema": schema,
                "potential_evidence": sorted(evidence),
                "coverage_requirement": requirements[name],
            }
            for name, (_, evidence, schema) in self._tools.items()
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        handler, potential_evidence, _ = self._tools[name]
        result = handler(**arguments)
        covers = {
            "get_ligand_info": True,
            "get_pocket_residues": float(arguments.get("radius", 0)) >= 5.0,
            "detect_basic_interactions": float(arguments.get("cutoff", 0)) >= 4.0,
            "get_atom_environment": float(arguments.get("radius", 0)) >= 4.0,
            "check_growth_space": float(arguments.get("distance", 0)) >= 1.5,
            "get_fragment_properties": True,
            "get_ligand_fragment": True,
        }[name]
        return result, set(potential_evidence) if covers else set()

    def get_ligand_info(self) -> dict[str, Any]:
        molecule = self.context.ligand
        return {
            "name": molecule.GetProp("_Name") if molecule.HasProp("_Name") else "ligand",
            "canonical_smiles": Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True),
            "formal_charge": Chem.GetFormalCharge(molecule),
            "heavy_atoms": molecule.GetNumHeavyAtoms(),
            "molecular_weight": round(Descriptors.MolWt(molecule), 2),
            "logp": round(Crippen.MolLogP(molecule), 2),
            "hbd": Lipinski.NumHDonors(molecule),
            "hba": Lipinski.NumHAcceptors(molecule),
            "tpsa": round(rdMolDescriptors.CalcTPSA(molecule), 2),
            "atoms": self.context.ligand_atom_rows(),
        }

    def get_pocket_residues(self, radius: float) -> dict[str, Any]:
        conformer = self.context.ligand.GetConformer()
        minima: dict[tuple[str, str, int], float] = defaultdict(lambda: float("inf"))
        for ligand_atom in self.context.ligand.GetAtoms():
            if ligand_atom.GetAtomicNum() == 1:
                continue
            position = conformer.GetAtomPosition(ligand_atom.GetIdx())
            xyz = np.array([position.x, position.y, position.z])
            for atom, distance in self.context.protein_near(xyz, radius):
                key = (atom.residue_name, atom.chain, atom.residue_number)
                minima[key] = min(minima[key], distance)
        residues = [
            {"residue": f"{name}:{chain}:{number}", "minimum_distance": round(distance, 3)}
            for (name, chain, number), distance in sorted(minima.items(), key=lambda item: item[1])
        ]
        return {"radius": radius, "residues": residues}

    def detect_basic_interactions(self, cutoff: float) -> dict[str, Any]:
        conformer = self.context.ligand.GetConformer()
        contacts = []
        for ligand_atom in self.context.ligand.GetAtoms():
            if ligand_atom.GetAtomicNum() == 1:
                continue
            position = conformer.GetAtomPosition(ligand_atom.GetIdx())
            xyz = np.array([position.x, position.y, position.z])
            for atom, distance in self.context.protein_near(xyz, cutoff):
                ligand_element = ligand_atom.GetSymbol().upper()
                polar = ligand_element in {"N", "O", "S"} and atom.element in {"N", "O", "S"}
                hydrophobic = ligand_element in {"C", "CL", "F"} and atom.element == "C"
                if not (polar or hydrophobic):
                    continue
                contacts.append(
                    {
                        "kind": "polar_contact_candidate" if polar else "hydrophobic_contact",
                        "ligand_atom_index": ligand_atom.GetIdx(),
                        "protein_atom": f"{atom.residue_name}:{atom.chain}:{atom.residue_number}:{atom.name}",
                        "distance": round(distance, 3),
                    }
                )
        contacts.sort(key=lambda item: item["distance"])
        return {
            "cutoff": cutoff,
            "contacts": contacts[:40],
            "limitation": "Distance/element heuristics only; hydrogen-bond directionality is not assigned.",
        }

    def _ligand_atom(self, atom_index: int):
        if atom_index < 0 or atom_index >= self.context.ligand.GetNumAtoms():
            raise ValueError(f"Invalid ligand atom index: {atom_index}")
        atom = self.context.ligand.GetAtomWithIdx(atom_index)
        if atom.GetAtomicNum() == 1:
            raise ValueError("Edit-site tools require a heavy atom")
        return atom

    def get_atom_environment(self, atom_index: int, radius: float) -> dict[str, Any]:
        atom = self._ligand_atom(atom_index)
        position = self.context.ligand.GetConformer().GetAtomPosition(atom_index)
        xyz = np.array([position.x, position.y, position.z])
        nearby = self.context.protein_near(xyz, radius)
        return {
            "atom_index": atom_index,
            "element": atom.GetSymbol(),
            "aromatic": atom.GetIsAromatic(),
            "replaceable_hydrogens": atom.GetTotalNumHs(),
            "protein_atoms": [
                {
                    "atom": f"{item.residue_name}:{item.chain}:{item.residue_number}:{item.name}",
                    "element": item.element,
                    "distance": round(distance, 3),
                }
                for item, distance in nearby[:30]
            ],
        }

    def check_growth_space(self, atom_index: int, distance: float) -> dict[str, Any]:
        atom = self._ligand_atom(atom_index)
        neighbors = [item for item in atom.GetNeighbors() if item.GetAtomicNum() > 1]
        if not neighbors:
            raise ValueError("Selected atom has no heavy-atom neighbor")
        conformer = self.context.ligand.GetConformer()
        origin_point = conformer.GetAtomPosition(atom_index)
        origin = np.array([origin_point.x, origin_point.y, origin_point.z])
        neighbor_positions = []
        for item in neighbors:
            point = conformer.GetAtomPosition(item.GetIdx())
            neighbor_positions.append(np.array([point.x, point.y, point.z]))
        center = np.mean(neighbor_positions, axis=0)
        direction = origin - center
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            raise ValueError("Could not determine outward growth vector")
        probe = origin + direction / norm * distance
        nearest = sorted(
            ((item, float(np.linalg.norm(item.xyz - probe))) for item in self.context.protein_atoms),
            key=lambda item: item[1],
        )
        return {
            "atom_index": atom_index,
            "probe_distance": distance,
            "probe_xyz": [round(float(value), 3) for value in probe],
            "nearest_protein_atoms": [
                {
                    "atom": f"{item.residue_name}:{item.chain}:{item.residue_number}:{item.name}",
                    "distance": round(item_distance, 3),
                }
                for item, item_distance in nearest[:10]
            ],
            "minimum_clearance": round(nearest[0][1], 3),
            "limitation": "Rigid outward-vector probe; receptor flexibility and free energy are not modeled.",
        }

    def get_fragment_properties(self, smiles: str) -> dict[str, Any]:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid fragment SMILES: {smiles}")
        return {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
            "formal_charge": Chem.GetFormalCharge(molecule),
            "heavy_atoms": molecule.GetNumHeavyAtoms(),
            "molecular_weight": round(Descriptors.MolWt(molecule), 2),
            "logp": round(Crippen.MolLogP(molecule), 2),
            "hbd": Lipinski.NumHDonors(molecule),
            "hba": Lipinski.NumHAcceptors(molecule),
            "tpsa": round(rdMolDescriptors.CalcTPSA(molecule), 2),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(molecule),
            "rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
            "limitation": "Descriptors are calculated for the isolated fragment; they do not predict affinity.",
        }

    def get_ligand_fragment(self, atom_index: int, radius_bonds: int) -> dict[str, Any]:
        self._ligand_atom(atom_index)
        visited = {atom_index}
        frontier = {atom_index}
        for _ in range(radius_bonds):
            next_frontier = set()
            for current in frontier:
                next_frontier.update(neighbor.GetIdx() for neighbor in self.context.ligand.GetAtomWithIdx(current).GetNeighbors())
            visited.update(next_frontier)
            frontier = next_frontier
        ring_info = self.context.ligand.GetRingInfo()
        changed = True
        while changed:
            changed = False
            for ring in ring_info.AtomRings():
                ring_atoms = set(ring)
                if visited & ring_atoms and not ring_atoms <= visited:
                    visited.update(ring_atoms)
                    changed = True
        atom_indices = sorted(visited)
        atom_set = set(atom_indices)
        bond_indices = [
            bond.GetIdx()
            for bond in self.context.ligand.GetBonds()
            if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set
        ]
        editable = Chem.PathToSubmol(self.context.ligand, bond_indices, useQuery=False)
        smiles = Chem.MolToSmiles(editable, isomericSmiles=True)
        return {
            "center_atom_index": atom_index,
            "radius_bonds": radius_bonds,
            "atom_indices": atom_indices,
            "bond_indices": bond_indices,
            "smiles": smiles,
            "properties": self.get_fragment_properties(smiles),
        }
