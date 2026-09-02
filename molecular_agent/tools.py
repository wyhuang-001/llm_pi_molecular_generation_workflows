from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors

from .editing import apply_substituent, apply_transformation
from .fragment_library import FragmentLibrary, chemical_tags, size_class_for
from .structure import ComplexContext


class ToolRegistry:
    def __init__(
        self,
        context: ComplexContext,
        fragment_library: FragmentLibrary | None = None,
        parent_resolver: Callable[[int | None], Chem.Mol] | None = None,
    ):
        self.context = context
        self.fragment_library = fragment_library or FragmentLibrary()
        self.parent_resolver = parent_resolver
        self._replacement_sites = self._build_replacement_sites()
        self._tools: dict[str, tuple[Callable[..., dict[str, Any]], set[str], dict[str, Any]]] = {
            "get_ligand_info": (
                self.get_ligand_info,
                {"ligand_identity"},
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
            "get_edit_site_candidates": (
                self.get_edit_site_candidates,
                {"edit_site_candidates"},
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
            "assess_edit_sites": (
                self.assess_edit_sites,
                {"site_strategy"},
                {
                    "type": "object",
                    "properties": {
                        "sites": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "target_type": {
                                        "type": "string",
                                        "enum": ["atom", "replacement_site"],
                                    },
                                    "target_id": {},
                                    "priority": {"type": "integer", "minimum": 1},
                                    "site_type": {
                                        "type": "string",
                                        "enum": [
                                            "core_anchor",
                                            "pocket_extension",
                                            "solvent_exposed",
                                            "linker_or_sidechain",
                                            "uncertain",
                                        ],
                                    },
                                    "rationale": {"type": "string", "minLength": 1},
                                    "search_status": {
                                        "type": "string",
                                        "enum": ["hard-reject", "pilot", "active"],
                                    },
                                },
                                "required": [
                                    "target_type", "target_id", "priority", "site_type", "rationale"
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "global_rationale": {"type": "string"},
                    },
                    "required": ["sites"],
                    "additionalProperties": False,
                },
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
                        "parent_attempt": {"type": "integer", "minimum": 1},
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
                        "parent_attempt": {"type": "integer", "minimum": 1},
                    },
                    "required": ["atom_index", "distance"],
                },
            ),
            "list_fragment_replacement_sites": (
                self.list_fragment_replacement_sites,
                {"replacement_sites"},
                {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                },
            ),
            "get_replacement_site_spatial_profile": (
                self.get_replacement_site_spatial_profile,
                {"replacement_site_spatial_profile"},
                {
                    "type": "object",
                    "properties": {
                        "replacement_site_id": {"type": "string", "minLength": 1},
                        "max_distance": {"type": "number", "minimum": 1.0, "maximum": 6.0},
                        "probe_count": {"type": "integer", "minimum": 2, "maximum": 8},
                    },
                    "required": ["replacement_site_id"],
                },
            ),
            "validate_candidate_geometry": (
                self.validate_candidate_geometry,
                {"candidate_geometry"},
                {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["replace_hydrogen", "replace_fragment"]},
                        "atom_index": {"type": "integer", "minimum": 0},
                        "edit_atom_index": {"type": "integer", "minimum": 0},
                        "replacement_site_id": {"type": "string", "minLength": 1},
                        "fragment_id": {"type": "string"},
                        "fragment_smiles": {"type": "string", "minLength": 1},
                        "parent_attempt": {"type": "integer", "minimum": 1},
                        "replace_existing_substituent": {"type": "boolean"},
                    },
                    "required": ["fragment_smiles"],
                },
            ),
            "generate_site_candidate_batch": (
                self.generate_site_candidate_batch,
                {"candidate_batch"},
                {
                    "type": "object",
                    "properties": {
                        "target_type": {
                            "type": "string",
                            "enum": ["atom", "replacement_site"],
                        },
                        "target_id": {},
                        "query": {"type": "string"},
                        "max_heavy_atoms": {"type": "integer", "minimum": 1, "maximum": 30},
                        "size_class": {
                            "type": "string",
                            "enum": ["minimal", "small", "medium", "large"],
                        },
                        "chemical_tag": {"type": "string", "minLength": 1},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 32},
                        "parent_attempt": {"type": "integer", "minimum": 1},
                    },
                    "required": ["target_type", "target_id"],
                    "additionalProperties": False,
                },
            ),
            "search_fragment_library": (
                self.search_fragment_library,
                {"fragment_library"},
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_heavy_atoms": {"type": "integer", "minimum": 1, "maximum": 30},
                        "operation": {"type": "string", "enum": ["substitute", "replace_fragment"]},
                        "size_class": {
                            "type": "string",
                            "enum": ["minimal", "small", "medium", "large"],
                        },
                        "chemical_tag": {"type": "string", "minLength": 1},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                },
            ),
            "get_fragment_record": (
                self.get_fragment_record,
                {"fragment_library"},
                {
                    "type": "object",
                    "properties": {"fragment_id": {"type": "string", "minLength": 1}},
                    "required": ["fragment_id"],
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
            "get_fragment_spatial_profile": (
                self.get_fragment_spatial_profile,
                {"fragment_spatial_profile"},
                {
                    "type": "object",
                    "properties": {
                        "fragment_id": {"type": "string", "minLength": 1},
                        "fragment_smiles": {"type": "string", "minLength": 1},
                    },
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

    def catalog(self, include_candidate_geometry: bool = True) -> dict[str, Any]:
        requirements = {
            "get_ligand_info": "Any successful call covers ligand identity.",
            "get_edit_site_candidates": (
                "Returns all host-supported atom and replacement-site targets with deterministic local "
                "environment, interaction, and directional geometry summaries for strategy assessment."
            ),
            "assess_edit_sites": (
                "Submit one priority and site_type assessment for each currently plausible host target. "
                "The host validates target IDs; this is an LLM hypothesis record, not an affinity prediction."
            ),
            "get_pocket_residues": "radius must be at least 5.0 A to cover pocket environment.",
            "detect_basic_interactions": "cutoff must be at least 4.0 A to cover key interactions.",
            "get_atom_environment": "radius must be at least 4.0 A for the final edit atom.",
            "check_growth_space": "probe distance must be at least 1.5 A for the final edit atom.",
            "list_fragment_replacement_sites": "Enumerates host-validated directed side-chain cuts. Use replacement_site_id; never guess cut_bond indices.",
            "get_replacement_site_spatial_profile": "Returns deterministic attachment-vector probes and nearest protein distances for one returned replacement_site_id. It reports geometry facts, not a suitability verdict.",
            "validate_candidate_geometry": "Runs the exact deterministic candidate construction and rigid-protein clash check. replace_fragment requires a replacement_site_id returned by list_fragment_replacement_sites.",
            "generate_site_candidate_batch": (
                "For one locked atom or replacement site, retrieve an operation-compatible fragment batch "
                "and run deterministic candidate construction and rigid-protein clash prescreening. "
                "Optional size_class and chemical_tag filters expose the unified library action space."
            ),
            "search_fragment_library": (
                "Searches by one supported chemical term (for example heterocycle, pyridine, morpholine, "
                "indole, oxetane, nitrile), one valid SMILES/SMARTS pattern, or an empty query for browsing. "
                "Optional size_class filters minimal, small, medium, or large fragments; chemical_tag filters "
                "labels such as halogen, alkyl, polar, heteroaryl, nitrile, or hbond_donor. Do not send "
                "natural-language descriptions. Results include source metadata and deterministic properties."
            ),
            "get_fragment_record": "Returns one auditable library record by fragment_id.",
            "get_fragment_properties": "Any valid fragment returns deterministic fragment properties.",
            "get_fragment_spatial_profile": "Returns deterministic 3D conformer extent and attachment-centered coordinates for a fragment. It reports shape facts, not a suitability verdict.",
            "get_ligand_fragment": "Any valid atom and bond radius returns a local ligand fragment.",
        }
        catalog = {
            name: {
                "input_schema": schema,
                "potential_evidence": sorted(evidence),
                "coverage_requirement": requirements[name],
            }
            for name, (_, evidence, schema) in self._tools.items()
            if include_candidate_geometry or name != "validate_candidate_geometry"
        }
        return catalog

    def execute(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        handler, potential_evidence, _ = self._tools[name]
        result = handler(**arguments)
        covers = {
            "get_ligand_info": True,
            "get_edit_site_candidates": True,
            "assess_edit_sites": True,
            "get_pocket_residues": float(arguments.get("radius", 0)) >= 5.0,
            "detect_basic_interactions": float(arguments.get("cutoff", 0)) >= 4.0,
            "get_atom_environment": float(arguments.get("radius", 0)) >= 4.0,
            "check_growth_space": float(arguments.get("distance", 0)) >= 1.5,
            "list_fragment_replacement_sites": True,
            "get_replacement_site_spatial_profile": True,
            "validate_candidate_geometry": True,
            "generate_site_candidate_batch": True,
            "get_fragment_properties": True,
            "get_fragment_spatial_profile": True,
            "get_ligand_fragment": True,
            "search_fragment_library": True,
            "get_fragment_record": True,
        }[name]
        rejected_query = (
            name == "search_fragment_library"
            and result.get("failure_class") == "unsupported_fragment_query"
        )
        return result, set(potential_evidence) if covers and not rejected_query else set()

    def get_edit_site_candidates(self) -> dict[str, Any]:
        interactions = self.detect_basic_interactions(4.0).get("contacts", [])
        interactions_by_atom: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for contact in interactions:
            atom_index = contact.get("ligand_atom_index")
            if isinstance(atom_index, int):
                interactions_by_atom[atom_index].append(contact)
        atom_sites = []
        for atom in self.context.ligand.GetAtoms():
            if (
                atom.GetAtomicNum() <= 1
                or atom.GetTotalNumHs() < 1
                or (atom.GetSymbol() == "N" and atom.GetIsAromatic())
            ):
                continue
            atom_index = atom.GetIdx()
            environment = self.get_atom_environment(atom_index, 4.0)
            try:
                growth = self.check_growth_space(atom_index, 3.0)
            except Exception as error:
                growth = {"status": "unavailable", "error": str(error)}
            atom_sites.append({
                "target_type": "atom",
                "target_id": atom_index,
                "element": atom.GetSymbol(),
                "aromatic": atom.GetIsAromatic(),
                "replaceable_hydrogens": atom.GetTotalNumHs(),
                "nearby_protein_atoms": environment["protein_atoms"][:10],
                "current_interactions": interactions_by_atom.get(atom_index, [])[:8],
                "growth_probe": {
                    key: growth.get(key)
                    for key in (
                        "status", "probe_distance", "probe_xyz", "minimum_clearance",
                        "nearest_protein_atoms", "error",
                    )
                    if key in growth
                },
            })
        replacement_sites = []
        for site in self._replacement_sites:
            try:
                spatial = self.get_replacement_site_spatial_profile(
                    site["replacement_site_id"], max_distance=4.0, probe_count=5
                )
                directional_clearance = [
                    {
                        "label": item["label"],
                        "minimum_protein_atom_distance_along_probe": item[
                            "minimum_protein_atom_distance_along_probe"
                        ],
                    }
                    for item in spatial["direction_profiles"]
                ]
            except Exception as error:
                directional_clearance = [{"status": "unavailable", "error": str(error)}]
            replacement_sites.append({
                "target_type": "replacement_site",
                "target_id": site["replacement_site_id"],
                "retained_atom_index": site["retained_atom_index"],
                "removed_side_atom_index": site["removed_side_atom_index"],
                "removed_heavy_atoms": site["removed_heavy_atoms"],
                "removed_fraction": site["removed_fraction"],
                "removed_fragment_smiles": site["removed_fragment_smiles"],
                "attachment_vector": site["attachment_vector"],
                "retained_atom_interactions": interactions_by_atom.get(
                    site["retained_atom_index"], []
                )[:8],
                "directional_clearance": directional_clearance,
            })
        return {
            "status": "complete",
            "atom_site_count": len(atom_sites),
            "replacement_site_count": len(replacement_sites),
            "atom_sites": atom_sites,
            "replacement_sites": replacement_sites,
            "site_type_vocabulary": [
                "core_anchor", "pocket_extension", "solvent_exposed",
                "linker_or_sidechain", "uncertain",
            ],
            "limitation": (
                "These are deterministic host-supported targets and rigid-structure summaries. Priority and "
                "site type remain LLM assessments; receptor flexibility and binding free energy are not modeled."
            ),
        }

    def assess_edit_sites(
        self,
        sites: list[dict[str, Any]],
        global_rationale: str = "",
    ) -> dict[str, Any]:
        if not isinstance(sites, list) or not sites:
            raise ValueError("assess_edit_sites requires a non-empty sites array")
        editable_atoms = {
            atom.GetIdx()
            for atom in self.context.ligand.GetAtoms()
            if atom.GetAtomicNum() > 1
            and atom.GetTotalNumHs() > 0
            and not (atom.GetSymbol() == "N" and atom.GetIsAromatic())
        }
        replacement_sites = {
            site["replacement_site_id"] for site in self._replacement_sites
        }
        allowed_types = {
            "core_anchor", "pocket_extension", "solvent_exposed",
            "linker_or_sidechain", "uncertain",
        }
        normalized = []
        seen_targets = set()
        seen_priorities = set()
        for item in sites:
            if not isinstance(item, dict):
                raise ValueError("Each site assessment must be an object")
            target_type = item.get("target_type")
            target_id = item.get("target_id")
            priority = item.get("priority")
            site_type = item.get("site_type")
            rationale = item.get("rationale")
            search_status = item.get("search_status", "active")
            if target_type not in {"atom", "replacement_site"}:
                raise ValueError("site assessment target_type must be atom or replacement_site")
            valid_target = (
                target_type == "atom" and isinstance(target_id, int) and target_id in editable_atoms
            ) or (
                target_type == "replacement_site"
                and isinstance(target_id, str)
                and target_id in replacement_sites
            )
            if not valid_target:
                raise ValueError(f"Unknown or non-editable site target: {target_type}:{target_id}")
            if not isinstance(priority, int) or priority < 1:
                raise ValueError("site assessment priority must be a positive integer")
            if priority in seen_priorities:
                raise ValueError("site assessment priorities must be unique")
            if site_type not in allowed_types:
                raise ValueError(f"Unsupported site_type: {site_type!r}")
            if not isinstance(rationale, str) or not rationale.strip():
                raise ValueError("site assessment rationale must be non-empty")
            if search_status not in {"hard-reject", "pilot", "active"}:
                raise ValueError("site assessment search_status must be hard-reject, pilot, or active")
            target_key = (target_type, target_id)
            if target_key in seen_targets:
                raise ValueError(f"Duplicate site assessment: {target_type}:{target_id}")
            seen_targets.add(target_key)
            seen_priorities.add(priority)
            normalized.append({
                "target_type": target_type,
                "target_id": target_id,
                "priority": priority,
                "site_type": site_type,
                "rationale": rationale.strip(),
                "search_status": search_status,
            })
        priorities = sorted(seen_priorities)
        if priorities != list(range(1, len(priorities) + 1)):
            raise ValueError("site assessment priorities must be consecutive starting at 1")
        normalized.sort(key=lambda item: item["priority"])
        return {
            "status": "complete",
            "sites": normalized,
            "site_count": len(normalized),
            "global_rationale": global_rationale.strip() if isinstance(global_rationale, str) else "",
            "limitation": (
                "Priority and site type are LLM-generated strategic assessments validated against host targets. "
                "They are not structural annotations, docking scores, or affinity predictions."
            ),
        }

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

    @staticmethod
    def _ligand_polar_roles(atom: Chem.Atom) -> list[str]:
        roles = []
        if atom.GetSymbol() not in {"N", "O", "S"}:
            return roles
        if atom.GetTotalNumHs() > 0:
            roles.append("donor")
        molecule = atom.GetOwningMol()
        acceptor_matches = {
            index
            for match in molecule.GetSubstructMatches(
                Chem.MolFromSmarts("[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),$([nH0,o,s;+0])]"),
            )
            for index in match
        }
        if atom.GetIdx() in acceptor_matches:
            roles.append("acceptor")
        return roles

    @staticmethod
    def _protein_polar_roles(atom: Any) -> list[str]:
        residue = atom.residue_name.upper()
        name = atom.name.upper()
        roles = []
        if atom.element == "O":
            roles.append("acceptor")
            if (residue, name) in {("SER", "OG"), ("THR", "OG1"), ("TYR", "OH")}:
                roles.append("donor")
        elif atom.element == "N":
            backbone_n = name == "N"
            donor_names = {
                "ARG": {"NE", "NH1", "NH2"},
                "ASN": {"ND2"},
                "GLN": {"NE2"},
                "HIS": {"ND1", "NE2"},
                "LYS": {"NZ"},
                "TRP": {"NE1"},
            }
            if backbone_n or name in donor_names.get(residue, set()):
                roles.append("donor")
            acceptor_names = {"HIS": {"ND1", "NE2"}}
            if name in acceptor_names.get(residue, set()):
                roles.append("acceptor")
        elif atom.element == "S":
            if residue == "CYS" and name == "SG":
                roles.extend(["donor", "acceptor"])
            elif residue == "MET" and name == "SD":
                roles.append("acceptor")
        return sorted(set(roles))

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
                contact = {
                    "kind": "polar_contact_candidate" if polar else "hydrophobic_contact",
                    "ligand_atom_index": ligand_atom.GetIdx(),
                    "protein_atom": f"{atom.residue_name}:{atom.chain}:{atom.residue_number}:{atom.name}",
                    "distance": round(distance, 3),
                }
                if polar:
                    ligand_roles = self._ligand_polar_roles(ligand_atom)
                    protein_roles = self._protein_polar_roles(atom)
                    complementary = (
                        "donor" in ligand_roles and "acceptor" in protein_roles
                    ) or (
                        "acceptor" in ligand_roles and "donor" in protein_roles
                    )
                    contact.update({
                        "ligand_roles": ligand_roles,
                        "protein_roles": protein_roles,
                        "hydrogen_bond_role_compatible": complementary,
                        "role_warning": (
                            None if complementary else
                            "Polar proximity lacks a donor-acceptor role pairing and must not be called a hydrogen bond."
                        ),
                    })
                contacts.append(contact)
        contacts.sort(key=lambda item: item["distance"])
        return {
            "cutoff": cutoff,
            "contacts": contacts[:40],
            "limitation": (
                "Donor/acceptor roles and distances are screened, but hydrogen-bond angles, "
                "protonation ambiguity, water mediation, and energetics are not modeled."
            ),
        }

    def _ligand_atom(self, atom_index: int):
        if atom_index < 0 or atom_index >= self.context.ligand.GetNumAtoms():
            raise ValueError(f"Invalid ligand atom index: {atom_index}")
        atom = self.context.ligand.GetAtomWithIdx(atom_index)
        if atom.GetAtomicNum() == 1:
            raise ValueError("Edit-site tools require a heavy atom")
        return atom

    def get_atom_environment(
        self, atom_index: int, radius: float, parent_attempt: int | None = None
    ) -> dict[str, Any]:
        ligand = self._parent_ligand(parent_attempt)
        if atom_index < 0 or atom_index >= ligand.GetNumAtoms():
            raise ValueError(f"Invalid ligand atom index: {atom_index}")
        atom = ligand.GetAtomWithIdx(atom_index)
        position = ligand.GetConformer().GetAtomPosition(atom_index)
        xyz = np.array([position.x, position.y, position.z])
        nearby = self.context.protein_near(xyz, radius)
        return {
            "atom_index": atom_index,
            "parent_attempt": parent_attempt,
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

    def check_growth_space(
        self, atom_index: int, distance: float, parent_attempt: int | None = None
    ) -> dict[str, Any]:
        ligand = self._parent_ligand(parent_attempt)
        if atom_index < 0 or atom_index >= ligand.GetNumAtoms():
            raise ValueError(f"Invalid ligand atom index: {atom_index}")
        atom = ligand.GetAtomWithIdx(atom_index)
        neighbors = [item for item in atom.GetNeighbors() if item.GetAtomicNum() > 1]
        if not neighbors:
            raise ValueError("Selected atom has no heavy-atom neighbor")
        conformer = ligand.GetConformer()
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
            "parent_attempt": parent_attempt,
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

    def _build_replacement_sites(self) -> list[dict[str, Any]]:
        molecule = Chem.RemoveHs(Chem.Mol(self.context.ligand))
        total_atoms = molecule.GetNumHeavyAtoms()
        configured = self.context.task.get("fragment_replacement") or {}
        max_removed_fraction = float(configured.get("max_removed_heavy_atom_fraction", 0.4))
        protected = {
            int(index) for index in configured.get("protected_core_atom_indices", [])
            if isinstance(index, int)
        }
        ring_atoms = {
            index for ring in molecule.GetRingInfo().AtomRings() for index in ring
        }
        candidates = []
        for bond in molecule.GetBonds():
            if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            graph = Chem.RWMol(molecule)
            graph.RemoveBond(left, right)
            components = Chem.GetMolFrags(
                graph.GetMol(), asMols=False, sanitizeFrags=False
            )
            if len(components) != 2:
                continue
            component_sets = [set(component) for component in components]
            eligible = []
            for retained in component_sets:
                removed = set(range(total_atoms)) - retained
                if protected and not protected <= retained:
                    continue
                if len(removed) / total_atoms > max_removed_fraction:
                    continue
                if ring_atoms and not retained & ring_atoms:
                    continue
                eligible.append(retained)
            if not eligible:
                continue
            retained = max(
                eligible,
                key=lambda atoms: (len(atoms & ring_atoms), len(atoms)),
            )
            removed = set(range(total_atoms)) - retained
            retained_atom = left if left in retained else right
            removed_atom = right if retained_atom == left else left
            retained_point = molecule.GetConformer().GetAtomPosition(retained_atom)
            removed_point = molecule.GetConformer().GetAtomPosition(removed_atom)
            candidates.append({
                "cut_bond": [retained_atom, removed_atom],
                "retained_atom_index": retained_atom,
                "removed_side_atom_index": removed_atom,
                "retained_atom_element": molecule.GetAtomWithIdx(retained_atom).GetSymbol(),
                "removed_side_atom_element": molecule.GetAtomWithIdx(removed_atom).GetSymbol(),
                "retained_heavy_atoms": len(retained),
                "removed_heavy_atoms": len(removed),
                "removed_fraction": round(len(removed) / total_atoms, 3),
                "retained_ring_atoms": len(retained & ring_atoms),
                "removed_ring_atoms": len(removed & ring_atoms),
                "retained_atom_indices": sorted(retained),
                "removed_atom_indices": sorted(removed),
                "retained_scaffold_smiles": Chem.MolFragmentToSmiles(
                    molecule, atomsToUse=sorted(retained), isomericSmiles=True
                ),
                "removed_fragment_smiles": Chem.MolFragmentToSmiles(
                    molecule, atomsToUse=sorted(removed), isomericSmiles=True
                ),
                "attachment_vector": [
                    round(removed_point.x - retained_point.x, 3),
                    round(removed_point.y - retained_point.y, 3),
                    round(removed_point.z - retained_point.z, 3),
                ],
            })
        candidates.sort(
            key=lambda item: (
                item["removed_heavy_atoms"],
                item["retained_atom_index"],
                item["removed_side_atom_index"],
            )
        )
        for number, site in enumerate(candidates, start=1):
            site["replacement_site_id"] = f"replacement-site-{number:03d}"
        return candidates

    def list_fragment_replacement_sites(self, limit: int = 30) -> dict[str, Any]:
        return {
            "count": min(len(self._replacement_sites), limit),
            "total_count": len(self._replacement_sites),
            "sites": self._replacement_sites[:limit],
            "policy": (
                "Each site is a directed non-ring single-bond cut. The first atom and larger "
                "ring-rich scaffold are retained; the listed removed side is deleted. Sites that "
                "remove more than the configured fraction or violate protected core atoms are excluded."
            ),
        }

    def resolve_replacement_site(self, replacement_site_id: str) -> dict[str, Any]:
        for site in self._replacement_sites:
            if site["replacement_site_id"] == replacement_site_id:
                return dict(site)
        raise ValueError(
            f"Unknown replacement_site_id: {replacement_site_id}. "
            "Call list_fragment_replacement_sites and select one returned ID."
        )

    def get_replacement_site_spatial_profile(
        self,
        replacement_site_id: str,
        max_distance: float = 4.0,
        probe_count: int = 5,
    ) -> dict[str, Any]:
        site = self.resolve_replacement_site(replacement_site_id)
        conformer = self.context.ligand.GetConformer()
        point = conformer.GetAtomPosition(site["retained_atom_index"])
        origin = np.array([point.x, point.y, point.z], dtype=float)
        attachment = np.array(site["attachment_vector"], dtype=float)
        norm = float(np.linalg.norm(attachment))
        if norm < 1e-8:
            raise ValueError(f"Replacement site {replacement_site_id} has no attachment direction")
        forward = attachment / norm
        reference = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(forward, reference))) > 0.85:
            reference = np.array([0.0, 1.0, 0.0])
        lateral_a = np.cross(forward, reference)
        lateral_a /= np.linalg.norm(lateral_a)
        lateral_b = np.cross(forward, lateral_a)
        lateral_b /= np.linalg.norm(lateral_b)

        directions = [("forward", forward)]
        cone_angle = np.deg2rad(35.0)
        for number in range(max(0, probe_count - 1)):
            angle = 2.0 * np.pi * number / max(1, probe_count - 1)
            lateral = np.cos(angle) * lateral_a + np.sin(angle) * lateral_b
            vector = np.cos(cone_angle) * forward + np.sin(cone_angle) * lateral
            directions.append((f"forward_tilt_{number + 1}", vector / np.linalg.norm(vector)))

        sampled_distances = np.linspace(1.0, float(max_distance), num=7)
        direction_profiles = []
        for label, vector in directions:
            limiting = None
            for sample_distance in sampled_distances:
                probe = origin + vector * sample_distance
                nearest = sorted(
                    (
                        (protein_atom, float(np.linalg.norm(protein_atom.xyz - probe)))
                        for protein_atom in self.context.protein_atoms
                    ),
                    key=lambda item: item[1],
                )
                sample = {
                    "distance_from_attachment": round(float(sample_distance), 3),
                    "probe_xyz": [round(float(value), 3) for value in probe],
                    "minimum_protein_atom_distance": round(nearest[0][1], 3),
                    "nearest_protein_atoms": [
                        {
                            "atom": (
                                f"{atom.residue_name}:{atom.chain}:"
                                f"{atom.residue_number}:{atom.name}"
                            ),
                            "element": atom.element,
                            "distance": round(distance, 3),
                        }
                        for atom, distance in nearest[:5]
                    ],
                }
                if limiting is None or sample["minimum_protein_atom_distance"] < limiting[
                    "minimum_protein_atom_distance"
                ]:
                    limiting = sample
            endpoint = origin + vector * float(max_distance)
            direction_profiles.append({
                "label": label,
                "unit_vector": [round(float(value), 4) for value in vector],
                "endpoint_xyz": [round(float(value), 3) for value in endpoint],
                "minimum_protein_atom_distance_along_probe": limiting[
                    "minimum_protein_atom_distance"
                ],
                "limiting_sample": limiting,
            })

        return {
            "status": "complete",
            "replacement_site_id": replacement_site_id,
            "retained_atom_index": site["retained_atom_index"],
            "removed_side_atom_index": site["removed_side_atom_index"],
            "attachment_origin_xyz": [round(float(value), 3) for value in origin],
            "attachment_vector": site["attachment_vector"],
            "attachment_unit_vector": [round(float(value), 4) for value in forward],
            "max_probe_distance": float(max_distance),
            "probe_count": len(direction_profiles),
            "probe_cone_angle_degrees": 35.0,
            "direction_profiles": direction_profiles,
            "limitation": (
                "Rigid-protein directional point probes report atom-center distances. They do not "
                "include fragment van der Waals radii, conformer placement, receptor flexibility, "
                "or binding energetics; validate_candidate_geometry remains authoritative."
            ),
        }

    def _resolve_fragment_smiles(
        self, fragment_id: str | None, fragment_smiles: str | None
    ) -> tuple[str, dict[str, Any] | None]:
        record = None
        if fragment_id:
            record = self.fragment_library.get(fragment_id)
            if fragment_smiles and not self.fragment_library.smiles_equivalent(
                fragment_smiles, record["smiles"]
            ):
                raise ValueError(
                    f"fragment_id {fragment_id} does not match fragment_smiles"
                )
            fragment_smiles = record["smiles"]
        if not isinstance(fragment_smiles, str) or not fragment_smiles:
            raise ValueError("Provide fragment_id or fragment_smiles")
        return fragment_smiles, record

    def get_fragment_spatial_profile(
        self,
        fragment_id: str | None = None,
        fragment_smiles: str | None = None,
    ) -> dict[str, Any]:
        fragment_smiles, record = self._resolve_fragment_smiles(fragment_id, fragment_smiles)
        molecule = Chem.MolFromSmiles(fragment_smiles)
        if molecule is None:
            raise ValueError(f"Invalid fragment SMILES: {fragment_smiles}")
        dummy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummy_atoms) != 1 or dummy_atoms[0].GetAtomMapNum() != 1:
            raise ValueError("Fragment must contain exactly one mapped dummy atom [*:1]")
        neighbors = list(dummy_atoms[0].GetNeighbors())
        if len(neighbors) != 1:
            raise ValueError("Mapped dummy atom must have exactly one neighbor")
        dummy_index = dummy_atoms[0].GetIdx()
        attachment_index = neighbors[0].GetIdx()

        embedding_molecule = Chem.RWMol(Chem.Mol(molecule))
        # RDKit's force fields do not define a UFF type for dummy atoms. Use a
        # carbon placeholder only for conformer generation; the dummy remains
        # the attachment origin and is excluded from fragment extents.
        embedding_molecule.GetAtomWithIdx(dummy_index).SetAtomicNum(6)
        embedding_molecule.GetAtomWithIdx(dummy_index).SetFormalCharge(0)
        embedded = Chem.AddHs(embedding_molecule.GetMol(), addCoords=False)
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = 17
        parameters.useRandomCoords = True
        conformer_ids = list(AllChem.EmbedMultipleConfs(embedded, numConfs=10, params=parameters))
        if not conformer_ids:
            raise ValueError("Could not generate deterministic fragment conformers")

        heavy_indices = [
            atom.GetIdx()
            for atom in embedded.GetAtoms()
            if atom.GetAtomicNum() > 1 and atom.GetIdx() != dummy_index
        ]
        conformer_profiles = []
        representative_atoms = []
        for conformer_number, conformer_id in enumerate(conformer_ids, start=1):
            conformer = embedded.GetConformer(conformer_id)
            dummy_point = conformer.GetAtomPosition(dummy_index)
            attachment_point = conformer.GetAtomPosition(attachment_index)
            origin = np.array([dummy_point.x, dummy_point.y, dummy_point.z], dtype=float)
            attachment_xyz = np.array(
                [attachment_point.x, attachment_point.y, attachment_point.z], dtype=float
            )
            axis = attachment_xyz - origin
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1e-8:
                continue
            axis /= axis_norm
            coordinates = []
            distances = []
            axial_extents = []
            radial_extents = []
            for atom_index in heavy_indices:
                atom_point = conformer.GetAtomPosition(atom_index)
                relative = np.array(
                    [atom_point.x, atom_point.y, atom_point.z], dtype=float
                ) - origin
                axial = float(np.dot(relative, axis))
                radial = float(np.linalg.norm(relative - axial * axis))
                distance = float(np.linalg.norm(relative))
                coordinates.append(relative)
                distances.append(distance)
                axial_extents.append(axial)
                radial_extents.append(radial)
                if conformer_number == 1:
                    representative_atoms.append({
                        "atom_index": atom_index,
                        "element": embedded.GetAtomWithIdx(atom_index).GetSymbol(),
                        "relative_xyz_from_attachment_point": [
                            round(float(value), 3) for value in relative
                        ],
                        "axial_extent": round(axial, 3),
                        "radial_extent": round(radial, 3),
                    })
            coordinate_array = np.array(coordinates)
            centroid = np.mean(coordinate_array, axis=0)
            radius_of_gyration = float(
                np.sqrt(np.mean(np.sum((coordinate_array - centroid) ** 2, axis=1)))
            )
            conformer_profiles.append({
                "conformer": conformer_number,
                "max_attachment_distance": round(max(distances), 3),
                "maximum_forward_extent": round(max(axial_extents), 3),
                "maximum_radial_extent": round(max(radial_extents), 3),
                "radius_of_gyration": round(radius_of_gyration, 3),
            })
        if not conformer_profiles:
            raise ValueError("Fragment conformers did not define a usable attachment axis")

        def extent_summary(key: str) -> dict[str, float]:
            values = [float(profile[key]) for profile in conformer_profiles]
            return {
                "minimum": round(min(values), 3),
                "mean": round(float(np.mean(values)), 3),
                "maximum": round(max(values), 3),
            }

        return {
            "status": "complete",
            "fragment_id": fragment_id,
            "fragment_smiles": fragment_smiles,
            "allowed_operations": record.get("allowed_operations") if record else None,
            "dummy_atom_index": dummy_index,
            "attachment_atom_index": attachment_index,
            "attachment_atom_element": molecule.GetAtomWithIdx(attachment_index).GetSymbol(),
            "heavy_atoms": molecule.GetNumHeavyAtoms(),
            "rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
            "ring_count": rdMolDescriptors.CalcNumRings(molecule),
            "conformer_count": len(conformer_profiles),
            "max_attachment_distance": extent_summary("max_attachment_distance"),
            "maximum_forward_extent": extent_summary("maximum_forward_extent"),
            "maximum_radial_extent": extent_summary("maximum_radial_extent"),
            "radius_of_gyration": extent_summary("radius_of_gyration"),
            "representative_conformer_atoms": representative_atoms,
            "limitation": (
                "Isolated-fragment conformers are attachment-centered shape facts. They are not "
                "aligned to a protein site and do not predict steric compatibility or affinity; "
                "validate_candidate_geometry remains authoritative."
            ),
        }

    def _resolve_fragment_replacement(self, transformation: dict[str, Any]) -> None:
        site_id = transformation.get("replacement_site_id")
        if not isinstance(site_id, str):
            raise ValueError(
                "replace_fragment requires replacement_site_id from "
                "list_fragment_replacement_sites; direct cut_bond input is not accepted"
            )
        site = self.resolve_replacement_site(site_id)
        transformation["cut_bond"] = site["cut_bond"]
        transformation["edit_atom_index"] = site["retained_atom_index"]
        transformation["replacement_site"] = site

    def _parent_ligand(self, parent_attempt: int | None = None) -> Chem.Mol:
        if parent_attempt is None:
            return self.context.ligand
        if self.parent_resolver is None:
            raise ValueError("parent_attempt is not available in this tool context")
        return self.parent_resolver(parent_attempt)

    def validate_candidate_geometry(
        self, parent_attempt: int | None = None, **transformation: Any
    ) -> dict[str, Any]:
        # The host, rather than the model, resolves site IDs and fragment IDs.
        parent = self._parent_ligand(parent_attempt)
        if parent_attempt is not None:
            transformation["parent_attempt"] = parent_attempt
        if transformation.get("operation") == "replace_fragment":
            try:
                self._resolve_fragment_replacement(transformation)
            except Exception as exc:
                return {"status": "rejected", "error": str(exc), "transformation": transformation}
        if transformation.get("fragment_id"):
            record = self.fragment_library.get(str(transformation["fragment_id"]))
            requested_operation = transformation.get("operation", "replace_hydrogen")
            library_operation = "substitute" if requested_operation in {"substitute", "replace_hydrogen"} else requested_operation
            if not self.fragment_library.allows_operation(record, library_operation):
                return {
                    "status": "rejected",
                    "error": f"Fragment {transformation['fragment_id']} does not allow operation {library_operation}",
                    "transformation": transformation,
                }
            if not transformation.get("fragment_smiles"):
                transformation["fragment_smiles"] = record["smiles"]
            elif not self.fragment_library.smiles_equivalent(
                transformation["fragment_smiles"], record["smiles"]
            ):
                return {
                    "status": "rejected",
                    "error": f"fragment_id {transformation['fragment_id']} does not match fragment_smiles",
                    "transformation": transformation,
                }
            else:
                transformation["fragment_smiles"] = record["smiles"]
            transformation["library_record"] = record
            if transformation.get("operation") == "substitute":
                transformation["operation"] = "replace_hydrogen"
        if "atom_index" in transformation and "edit_atom_index" not in transformation:
            transformation["edit_atom_index"] = transformation["atom_index"]
        try:
            result = apply_transformation(
                parent,
                transformation,
                self.context.protein_atoms,
                seed=17,
            )
        except Exception as exc:
            return {"status": "rejected", "error": str(exc), "transformation": transformation}
        return {**result.report, "transformation": transformation}

    def generate_site_candidate_batch(
        self,
        target_type: str,
        target_id: Any,
        query: str = "",
        max_heavy_atoms: int = 12,
        size_class: str | None = None,
        chemical_tag: str | None = None,
        limit: int = 16,
        parent_attempt: int | None = None,
    ) -> dict[str, Any]:
        parent = self._parent_ligand(parent_attempt)
        if target_type == "atom":
            if not isinstance(target_id, int) or not 0 <= target_id < parent.GetNumAtoms():
                raise ValueError(f"Invalid ligand atom index: {target_id}")
            atom = parent.GetAtomWithIdx(target_id)
            if atom.GetAtomicNum() == 1:
                raise ValueError("Edit-site tools require a heavy atom")
            if atom.GetTotalNumHs() < 1 or (atom.GetSymbol() == "N" and atom.GetIsAromatic()):
                raise ValueError(f"Atom {target_id!r} is not supported for replace_hydrogen")
            operation = "substitute"
        elif target_type == "replacement_site":
            self.resolve_replacement_site(target_id)
            operation = "replace_fragment"
        else:
            raise ValueError("target_type must be atom or replacement_site")
        search = self.fragment_library.search(
            query=query,
            max_heavy_atoms=max_heavy_atoms,
            operation=operation,
            limit=limit,
            size_class=size_class,
            chemical_tag=chemical_tag,
        )
        if search.get("status") != "complete":
            return {
                **search,
                "target_type": target_type,
                "target_id": target_id,
                "candidates": [],
            }
        candidates = []
        rejected = []
        seen_structures = set()
        for record in search.get("fragments", []):
            transformation = {
                "operation": "replace_hydrogen" if target_type == "atom" else "replace_fragment",
                "fragment_id": record["fragment_id"],
                "fragment_smiles": record["smiles"],
                "parent_attempt": parent_attempt,
                "replace_existing_substituent": parent_attempt is not None,
            }
            if target_type == "atom":
                transformation["edit_atom_index"] = target_id
            else:
                transformation["replacement_site_id"] = target_id
                self._resolve_fragment_replacement(transformation)
            try:
                result = apply_transformation(
                    parent,
                    transformation,
                    self.context.protein_atoms,
                    seed=17,
                )
            except Exception as error:
                rejected.append({
                    "fragment_id": record["fragment_id"],
                    "fragment_smiles": record["smiles"],
                    "failure_class": "candidate_construction_or_geometry",
                    "error": str(error),
                })
                continue
            canonical = result.report["candidate"]["canonical_smiles"]
            if canonical in seen_structures:
                continue
            seen_structures.add(canonical)
            summary = {
                "fragment_id": record["fragment_id"],
                "fragment_smiles": record["smiles"],
                "transformation": transformation,
                "canonical_smiles": canonical,
                "status": result.report["status"],
                "failure_class": result.report.get("failure_class"),
                "severe_clash_count": result.report.get("severe_clash_count"),
                "property_delta": result.report.get("property_delta"),
                "fragment_properties": record.get("properties"),
            }
            if result.report["status"] == "accepted":
                candidates.append(summary)
            else:
                rejected.append(summary)
        return {
            "status": "complete",
            "target_type": target_type,
            "target_id": target_id,
            "parent_attempt": parent_attempt,
            "query": query,
            "operation": operation,
            "requested_limit": limit,
            "size_class": size_class,
            "chemical_tag": chemical_tag,
            "library_match_count": search.get("count", 0),
            "accepted_count": len(candidates),
            "rejected_count": len(rejected),
            "candidates": candidates,
            "rejected": rejected[:20],
            "limitation": (
                "Candidates passed only deterministic construction and rigid-protein clash prescreening. "
                "They are not docked, ranked by affinity, or experimental activity predictions."
            ),
        }

    def search_fragment_library(
        self,
        query: str = "",
        max_heavy_atoms: int = 12,
        operation: str = "substitute",
        size_class: str | None = None,
        chemical_tag: str | None = None,
        limit: int = 30,
    ) -> dict[str, Any]:
        return self.fragment_library.search(
            query,
            max_heavy_atoms,
            operation,
            limit,
            size_class=size_class,
            chemical_tag=chemical_tag,
        )

    def get_fragment_record(self, fragment_id: str) -> dict[str, Any]:
        return self.fragment_library.get(fragment_id)

    def get_fragment_properties(self, smiles: str) -> dict[str, Any]:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid fragment SMILES: {smiles}")
        return {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
            "formal_charge": Chem.GetFormalCharge(molecule),
            "heavy_atoms": molecule.GetNumHeavyAtoms(),
            "size_class": size_class_for(molecule.GetNumHeavyAtoms()),
            "chemical_tags": chemical_tags(molecule),
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
