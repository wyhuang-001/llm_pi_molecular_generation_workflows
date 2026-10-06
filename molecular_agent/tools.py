from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any, Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors

from .edit_taxonomy import (
    CHANGE_TYPES,
    EditTaxonomyError,
    SITE_ALLOWED_CHANGE_TYPES,
    axes_for_operation,
    change_types_from_legacy_operations,
    normalize_transformation,
    operation_label,
    taxonomy_documentation,
)
from .editing import apply_transformation
from .fragment_library import FragmentLibrary, chemical_tags, size_class_for
from .structure import ComplexContext


def apply_site_canonicalization(transformation: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a symmetry-redundant atom target to its canonical representative.

    Attaching the same fragment to any member of a symmetry class gives the identical
    molecule, so the host canonicalises instead of rejecting.  The requested index is
    kept for audit.
    """
    info = transformation.get("site_canonicalization")
    if not isinstance(info, dict):
        return transformation
    canonical = info.get("canonical_target_id")
    if isinstance(canonical, int) and transformation.get("edit_atom_index") != canonical:
        transformation.setdefault("requested_edit_atom_index", transformation.get("edit_atom_index"))
        transformation["edit_atom_index"] = canonical
    return transformation


#: RDKit's default rotatable-bond pattern (single, acyclic, non-terminal, no triple bond).
ROTATABLE_BOND_SMARTS = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")

#: Canonical change type -> historical fragment-library operation name.
#: The frozen unified library still stores ``substitute`` / ``replace_fragment``; the
#: boundary maps to those names instead of rewriting multi-megabyte generated data.
LIBRARY_OPERATION_FOR_CHANGE_TYPE = {
    "addition": "substitute",
    "replacement": "replace_fragment",
}


def _library_panel(
    library: FragmentLibrary,
    change_type: str | None,
    limit: int,
    max_heavy_atoms: int,
    size_classes: list[str] | None = None,
    chemical_tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return a library panel for a change type, or an empty panel when none applies."""
    operation = LIBRARY_OPERATION_FOR_CHANGE_TYPE.get(change_type or "")
    if operation is None:
        return []
    return library.panel(
        operation=operation,
        limit=limit,
        max_heavy_atoms=max_heavy_atoms,
        size_classes=size_classes,
        chemical_tags_any=chemical_tags,
    )


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
        self.site_table = self._load_site_table()
        self._bond_sites = self._build_bond_sites()
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
            "get_design_dossier": (
                self.get_design_dossier,
                {"design_dossier"},
                {
                    "type": "object",
                    "properties": {
                        "panel_size": {"type": "integer", "minimum": 2, "maximum": 20},
                    },
                    "additionalProperties": False,
                },
            ),
            "get_complex_geometry": (
                self.get_complex_geometry,
                {"complex_geometry"},
                {
                    "type": "object",
                    "properties": {
                        "radius": {"type": "number", "minimum": 4.0, "maximum": 8.0},
                        "max_pocket_atoms": {"type": "integer", "minimum": 32, "maximum": 400},
                    },
                    "additionalProperties": False,
                },
            ),
            "get_fragment_panel": (
                self.get_fragment_panel,
                {"fragment_panel"},
                {
                    "type": "object",
                    "properties": {
                        "target_type": {"type": "string", "enum": ["atom", "bond"]},
                        "target_id": {},
                        "size_classes": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["minimal", "small", "medium", "large"]},
                        },
                        "chemical_tags": {"type": "array", "items": {"type": "string"}},
                        "max_heavy_atoms": {"type": "integer", "minimum": 1, "maximum": 12},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 30},
                    },
                    "required": ["target_type", "target_id"],
                    "additionalProperties": False,
                },
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
                                        "enum": ["atom", "bond"],
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
            "list_bond_sites": (
                self.list_bond_sites,
                {"bond_sites"},
                {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                },
            ),
            "get_bond_site_spatial_profile": (
                self.get_bond_site_spatial_profile,
                {"bond_site_spatial_profile"},
                {
                    "type": "object",
                    "properties": {
                        "bond_site_id": {"type": "string", "minLength": 1},
                        "max_distance": {"type": "number", "minimum": 1.0, "maximum": 6.0},
                        "probe_count": {"type": "integer", "minimum": 2, "maximum": 8},
                    },
                    "required": ["bond_site_id"],
                },
            ),
            "validate_candidate_geometry": (
                self.validate_candidate_geometry,
                {"candidate_geometry"},
                {
                    "type": "object",
                    "properties": {
                        "site_type": {
                            "type": "string",
                            "enum": ["atom", "bond", "linker", "ring"],
                        },
                        "change_type": {
                            "type": "string",
                            "enum": ["addition", "deletion", "replacement"],
                        },
                        "atom_index": {"type": "integer", "minimum": 0},
                        "edit_atom_index": {"type": "integer", "minimum": 0},
                        "bond_site_id": {"type": "string", "minLength": 1},
                        "fragment_id": {"type": "string"},
                        "fragment_smiles": {"type": "string", "minLength": 1},
                        "element": {"type": "string", "minLength": 1},
                        "parent_attempt": {"type": "integer", "minimum": 1},
                        "replace_existing_substituent": {"type": "boolean"},
                    },
                    "required": ["site_type", "change_type"],
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
                        "change_type": {"type": "string", "enum": ["addition", "replacement"]},
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
            "get_design_dossier": (
                "Returns one compact deterministic dossier containing ligand, pocket, interactions, all "
                "editable sites, the complete library summary, and diverse operation-compatible panels."
            ),
            "get_complex_geometry": (
                "Returns a bounded, machine-readable 3D complex representation in receptor coordinates: "
                "ligand atoms, local pocket atoms, edit vectors, distances, roles, and interaction edges."
            ),
            "get_fragment_panel": (
                "Returns a refreshed, diverse panel with complete precomputed chemistry for one host-listed "
                "target. Use it only when the initial panel lacks a decision-relevant chemical direction."
            ),
            "assess_edit_sites": (
                "Submit one priority and site_type assessment for each currently plausible host target. "
                "The host validates target IDs; this is an LLM hypothesis record, not an affinity prediction."
            ),
            "get_pocket_residues": "radius must be at least 5.0 A to cover pocket environment.",
            "detect_basic_interactions": "cutoff must be at least 4.0 A to cover key interactions.",
            "get_atom_environment": "radius must be at least 4.0 A for the final edit atom.",
            "check_growth_space": "probe distance must be at least 1.5 A for the final edit atom.",
            "list_bond_sites": "Enumerates host-validated directed side-chain cuts. Use bond_site_id; never guess cut_bond indices. Each site lists allowed_change_types (deletion and/or replacement).",
            "get_bond_site_spatial_profile": "Returns deterministic attachment-vector probes and nearest protein distances for one returned bond_site_id. It reports geometry facts, not a suitability verdict.",
            "validate_candidate_geometry": "Runs the exact deterministic candidate construction and rigid-protein clash check. Every site lists allowed_change_types; bond sites require a bond_site_id from list_bond_sites, and bond:deletion needs no fragment.",
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
            "get_design_dossier": True,
            "get_complex_geometry": True,
            "get_fragment_panel": True,
            "assess_edit_sites": True,
            "get_pocket_residues": float(arguments.get("radius", 0)) >= 5.0,
            "detect_basic_interactions": float(arguments.get("cutoff", 0)) >= 4.0,
            "get_atom_environment": float(arguments.get("radius", 0)) >= 4.0,
            "check_growth_space": float(arguments.get("distance", 0)) >= 1.5,
            "list_bond_sites": True,
            "get_bond_site_spatial_profile": True,
            "validate_candidate_geometry": True,
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

    @staticmethod
    def _site_change_types(record: dict[str, Any]) -> list[str]:
        """Return the canonical ``change_type`` list for one site record.

        New site tables list ``allowed_change_types`` directly.  Frozen tables and
        older run artifacts list legacy ``allowed_operations``; those are projected
        onto the canonical change types so no generated library or site table has to
        be rewritten.
        """
        explicit = record.get("allowed_change_types")
        if isinstance(explicit, list) and explicit:
            return [item for item in explicit if item in CHANGE_TYPES]
        return change_types_from_legacy_operations(record.get("allowed_operations"))

    def _protected_core_indices(self) -> set[int]:
        """Atoms the pose/native-like comparison relies on and an edit must not remove.

        Combines the explicit protected core, the pose-retention comparison core, the
        calibrated anchor atoms, and any protected atoms from a frozen site table.
        Indices are in the heavy-atom (``RemoveHs``) ligand frame, the same frame used
        by ``removed_atom_indices``.
        """
        task = self.context.task
        protected = {
            int(index)
            for index in (task.get("fragment_replacement") or {}).get(
                "protected_core_atom_indices"
            ) or []
            if isinstance(index, int)
        }
        retention = task.get("pose_retention") or {}
        protected |= {
            int(index) for index in retention.get("core_atom_indices") or [] if isinstance(index, int)
        }
        for anchor in retention.get("anchors") or []:
            index = anchor.get("ligand_atom_index")
            if isinstance(index, int):
                protected.add(index)
        if self.site_table:
            protected |= {
                int(index)
                for index in self.site_table.get("protected_atom_indices") or []
                if isinstance(index, int)
            }
        return protected

    def _removal_profile(self, removed_atom_indices: Any) -> dict[str, Any]:
        """Deterministic consequence profile for the side a bond edit removes.

        The three fields answer three different questions.  ``core_overlap`` is a
        legality check: removing a pose-comparison atom would invalidate the
        native-like comparison.  The other two are the gain side of the ledger,
        because the reference-interaction list only reports what a deletion loses.
        """
        molecule = Chem.RemoveHs(Chem.Mol(self.context.ligand))
        removed = sorted(
            {int(index) for index in removed_atom_indices or [] if isinstance(index, int)}
        )
        roles: Counter[str] = Counter()
        for index in removed:
            if 0 <= index < molecule.GetNumAtoms():
                roles.update(self._ligand_polar_roles(molecule.GetAtomWithIdx(index)))
        rotatable = (
            {tuple(sorted(match)) for match in molecule.GetSubstructMatches(ROTATABLE_BOND_SMARTS)}
            if ROTATABLE_BOND_SMARTS is not None else set()
        )
        removed_set = set(removed)
        overlap = sorted(removed_set & self._protected_core_indices())
        return {
            "removed_polar_roles": {role: roles[role] for role in ("donor", "acceptor") if roles[role]},
            "removed_rotatable_bonds": sum(
                1 for begin, end in rotatable if begin in removed_set or end in removed_set
            ),
            "core_overlap": overlap,
        }

    def _load_site_table(self) -> dict[str, Any] | None:
        """Load the host-supplied edit-site table, when the task names one.

        The table is the authoritative site list: it fixes which atoms and which
        directed cuts exist, which operations each one accepts, and which region is
        protected. It never names a fragment.
        """
        configured = self.context.task.get("edit_site_table_path")
        if not configured:
            return None
        path = (self.context.input_dir / str(configured)).resolve()
        if not path.is_file():
            raise ValueError(f"edit_site_table_path does not exist: {path}")
        table = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(table, dict) or not isinstance(table.get("atom_sites"), list):
            raise ValueError(f"Edit-site table must contain atom_sites: {path}")
        table["path"] = str(path)
        table["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return table

    def _site_table_policy(
        self,
        change_type: str,
        target_type: str,
        target_id: Any,
    ) -> dict[str, Any]:
        """Return whether one change type is allowed on one host-listed site."""
        if self.site_table is None:
            return {"status": "complete", "allowed": True, "source": "enumerated_sites"}
        if target_type == "atom":
            record = next(
                (item for item in self.site_table["atom_sites"] if item["atom_index"] == target_id),
                None,
            )
        elif target_type == "bond":
            record = next(
                (item for item in self.site_table["cut_sites"] if item["site_id"] == target_id),
                None,
            )
        else:
            raise ValueError("target_type must be atom or bond_site")
        if record is None:
            return {
                "status": "rejected",
                "allowed": False,
                "failure_class": "unlisted_edit_site",
                "reason": f"{target_type} {target_id!r} is not in the host edit-site table",
                "source": "edit_site_table",
            }
        allowed = self._site_change_types(record)
        if change_type not in allowed:
            return {
                "status": "rejected",
                "allowed": False,
                "failure_class": (
                    "protected_edit_site"
                    if record.get("protection") == "protected"
                    else "change_type_not_allowed_at_site"
                ),
                "reason": (
                    f"{target_type} {target_id!r} is {record.get('protection')} and does not allow "
                    f"change_type {change_type!r}; allowed change types: {allowed or 'none'}"
                ),
                "site": record,
                "source": "edit_site_table",
            }
        canonical = record.get("canonical_atom_index")
        canonicalization = None
        if target_type == "atom" and isinstance(canonical, int) and canonical != target_id:
            canonicalization = {
                "requested_target_id": target_id,
                "canonical_target_id": canonical,
                "canonical_site_id": f"atom-{canonical:03d}",
                "reason": "symmetry_equivalent_site",
            }
        return {
            "status": "complete",
            "allowed": True,
            "site": record,
            "protection": record.get("protection"),
            "probe_verdict": record.get("probe_verdict"),
            "canonicalization": canonicalization,
            "source": "edit_site_table",
        }

    def list_edit_sites(self) -> dict[str, Any]:
        """Return the host-supplied site list when present, else the enumerated one."""
        if self.site_table is None:
            return self.get_edit_site_candidates()
        return {
            "status": "complete",
            "source": "edit_site_table",
            "site_table_path": self.site_table.get("path"),
            "site_table_sha256": self.site_table.get("sha256"),
            "protected_atom_indices": self.site_table.get("protected_atom_indices", []),
            "atom_coverage": self.site_table.get("atom_coverage"),
            "atom_sites": self.site_table["atom_sites"],
            "cut_sites": self.site_table["cut_sites"],
            "input_contract": self.site_table.get("input_contract"),
            "limitation": (
                "Probe verdicts are rigid-structure geometry facts, not free-energy verdicts. "
                "A protected site stays visible and auditable but rejects every operation."
            ),
        }

    def get_edit_site_candidates(self) -> dict[str, Any]:
        if self.site_table is not None:
            return self._site_table_candidates()
        return self._enumerated_site_candidates()

    def _site_table_candidates(self) -> dict[str, Any]:
        interactions = self.detect_basic_interactions(4.0).get("contacts", [])
        interactions_by_atom: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for contact in interactions:
            atom_index = contact.get("ligand_atom_index")
            if isinstance(atom_index, int):
                interactions_by_atom[atom_index].append(contact)
        atom_sites = []
        for record in self.site_table["atom_sites"]:
            if not record.get("allowed_operations"):
                continue
            atom_index = record["atom_index"]
            environment = self.get_atom_environment(atom_index, 4.0)
            try:
                growth = self.check_growth_space(atom_index, 3.0)
            except Exception as error:
                growth = {"status": "unavailable", "error": str(error)}
            atom_sites.append({
                "target_type": "atom",
                "site_type": "ring" if record.get("is_ring_atom") else "atom",
                "target_id": atom_index,
                "site_id": record["site_id"],
                "region": record["region"],
                "element": record["element"],
                "aromatic": record["is_aromatic"],
                "replaceable_hydrogens": record["hydrogen_count"],
                "allowed_operations": record["allowed_operations"],
                "allowed_change_types": self._site_change_types(record),
                "protection": record["protection"],
                "probe_clearance": record["probe_clearance"],
                "probe_verdict": record["probe_verdict"],
                "symmetry_equivalent_atom_indices": record.get("symmetry_equivalent_atom_indices", [atom_index]),
                "symmetry_representative": record.get("symmetry_representative", True),
                "equivalent_to": record.get("equivalent_to"),
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
        bond_sites = []
        for site in self._bond_sites:
            if not site.get("allowed_operations"):
                continue
            try:
                spatial = self.get_bond_site_spatial_profile(
                    site["bond_site_id"], max_distance=4.0, probe_count=5
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
            bond_sites.append({
                "target_type": "bond",
                "site_type": "bond",
                "target_id": site["bond_site_id"],
                "site_id": site["bond_site_id"],
                "bond_site_id": site["bond_site_id"],
                "region": site.get("region"),
                "label": site.get("label"),
                "allowed_operations": site.get("allowed_operations", []),
                "allowed_change_types": self._site_change_types(site),
                "protection": site.get("protection"),
                "retained_atom_index": site["retained_atom_index"],
                "removed_side_atom_index": site["removed_side_atom_index"],
                "removed_heavy_atoms": site["removed_heavy_atoms"],
                "removed_fraction": site["removed_fraction"],
                "removed_fragment_smiles": site["removed_fragment_smiles"],
                "attachment_vector": site["attachment_vector"],
                "retained_atom_interactions": interactions_by_atom.get(
                    site["retained_atom_index"], []
                )[:8],
                "removed_reference_interactions": self._removed_side_interactions(
                    site, interactions_by_atom
                ),
                "removed_polar_roles": site.get("removed_polar_roles", {}),
                "removed_rotatable_bonds": site.get("removed_rotatable_bonds"),
                "core_overlap": site.get("core_overlap", []),
                "directional_clearance": directional_clearance,
            })
        return {
            "status": "complete",
            "source": "edit_site_table",
            "atom_site_count": len(atom_sites),
            "bond_site_count": len(bond_sites),
            "atom_sites": atom_sites,
            "bond_sites": bond_sites,
            "protected_atom_indices": self.site_table.get("protected_atom_indices", []),
            "site_type_vocabulary": [
                "core_anchor", "pocket_extension", "solvent_exposed",
                "linker_or_sidechain", "uncertain",
            ],
            "edit_taxonomy": taxonomy_documentation(),
            "limitation": (
                "The host fixed this site list; the designer chooses which fragment goes to which site. "
                "Probe verdicts are geometry facts, not affinity predictions."
            ),
        }

    def _enumerated_site_candidates(self) -> dict[str, Any]:
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
                "allowed_operations": ["addition"],
                "allowed_change_types": ["addition"],
                "site_type": "ring" if atom.IsInRing() else "atom",
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
        bond_sites = []
        for site in self._bond_sites:
            try:
                spatial = self.get_bond_site_spatial_profile(
                    site["bond_site_id"], max_distance=4.0, probe_count=5
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
            bond_sites.append({
                "target_type": "bond",
                "site_type": "bond",
                "target_id": site["bond_site_id"],
                "site_id": site["bond_site_id"],
                "bond_site_id": site["bond_site_id"],
                "retained_atom_index": site["retained_atom_index"],
                "removed_side_atom_index": site["removed_side_atom_index"],
                "removed_heavy_atoms": site["removed_heavy_atoms"],
                "removed_fraction": site["removed_fraction"],
                "removed_fragment_smiles": site["removed_fragment_smiles"],
                "allowed_operations": ["deletion", "replacement"],
                "allowed_change_types": ["deletion", "replacement"],
                "attachment_vector": site["attachment_vector"],
                "retained_atom_interactions": interactions_by_atom.get(
                    site["retained_atom_index"], []
                )[:8],
                "removed_reference_interactions": self._removed_side_interactions(
                    site, interactions_by_atom
                ),
                "removed_polar_roles": site.get("removed_polar_roles", {}),
                "removed_rotatable_bonds": site.get("removed_rotatable_bonds"),
                "core_overlap": site.get("core_overlap", []),
                "directional_clearance": directional_clearance,
            })
        return {
            "status": "complete",
            "atom_site_count": len(atom_sites),
            "bond_site_count": len(bond_sites),
            "atom_sites": atom_sites,
            "bond_sites": bond_sites,
            "site_type_vocabulary": [
                "core_anchor", "pocket_extension", "solvent_exposed",
                "linker_or_sidechain", "uncertain",
            ],
            "edit_taxonomy": taxonomy_documentation(),
            "limitation": (
                "These are deterministic host-supported targets and rigid-structure summaries. Priority and "
                "site type remain LLM assessments; receptor flexibility and binding free energy are not modeled."
            ),
        }

    @staticmethod
    def _site_max_fragment_heavy_atoms(site: dict[str, Any]) -> int:
        if site.get("target_type") == "atom":
            clearance = (site.get("growth_probe") or {}).get("minimum_clearance")
        else:
            values = [
                item.get("minimum_protein_atom_distance_along_probe")
                for item in (site.get("directional_clearance") or [])
                if isinstance(item, dict)
            ]
            numeric = [float(value) for value in values if isinstance(value, (int, float))]
            clearance = max(numeric) if numeric else None
        if not isinstance(clearance, (int, float)):
            return 8
        if clearance < 1.5:
            return 2
        if clearance < 2.0:
            return 4
        if clearance < 3.0:
            return 8
        return 12

    def get_complex_geometry(
        self, radius: float = 5.5, max_pocket_atoms: int = 240
    ) -> dict[str, Any]:
        """Return the bounded 3D representation exposed to the LLM.

        Coordinates are authoritative host facts, not a learned embedding.  The LLM receives
        a compact receptor-frame representation with explicit atom identity, local distances,
        donor/acceptor roles, edit vectors and interaction edges.  Raw PDB/SDF files remain
        audit artifacts and are not required for a text-only model to reason about geometry.
        """
        if not 4.0 <= float(radius) <= 8.0:
            raise ValueError("radius must be between 4.0 and 8.0 angstrom")
        if not 32 <= int(max_pocket_atoms) <= 400:
            raise ValueError("max_pocket_atoms must be between 32 and 400")
        ligand = Chem.RemoveHs(Chem.Mol(self.context.ligand))
        ligand_conf = ligand.GetConformer()
        ligand_xyz = ligand_conf.GetPositions()
        centroid = np.mean(ligand_xyz, axis=0)
        ligand_atoms = []
        for atom in ligand.GetAtoms():
            point = ligand_conf.GetAtomPosition(atom.GetIdx())
            ligand_atoms.append({
                "atom_index": atom.GetIdx(),
                "reference_atom_index": (
                    atom.GetIntProp("_reference_atom_index")
                    if atom.HasProp("_reference_atom_index") else atom.GetIdx()
                ),
                "element": atom.GetSymbol(),
                "formal_charge": atom.GetFormalCharge(),
                "aromatic": atom.GetIsAromatic(),
                "hydrogens": atom.GetTotalNumHs(),
                "roles": self._ligand_polar_roles(atom),
                "xyz": [round(float(value), 3) for value in point],
                "relative_to_ligand_centroid": [
                    round(float(value), 3) for value in np.asarray(point) - centroid
                ],
            })

        pocket_by_serial: dict[int, tuple[Any, float]] = {}
        for point in ligand_xyz:
            for protein_atom, distance in self.context.protein_near(np.asarray(point), radius):
                old = pocket_by_serial.get(protein_atom.serial)
                if old is None or distance < old[1]:
                    pocket_by_serial[protein_atom.serial] = (protein_atom, distance)
        pocket_rows = []
        for atom, minimum_distance in sorted(
            pocket_by_serial.values(), key=lambda item: (item[1], item[0].serial)
        )[: int(max_pocket_atoms)]:
            pocket_rows.append({
                "source_serial": atom.serial,
                "atom": atom.name,
                "residue": f"{atom.residue_name}:{atom.chain}:{atom.residue_number}",
                "element": atom.element,
                "roles": self._protein_polar_roles(atom),
                "xyz": [round(float(value), 3) for value in atom.xyz],
                "minimum_ligand_distance": round(float(minimum_distance), 3),
            })

        interactions = self.detect_basic_interactions(min(4.5, float(radius)))
        interaction_edges = []
        for contact in interactions.get("contacts", []):
            interaction_edges.append({
                key: contact[key]
                for key in (
                    "kind", "ligand_atom_index", "protein_atom", "distance",
                    "ligand_roles", "protein_roles", "hydrogen_bond_role_compatible",
                    "role_warning",
                ) if key in contact
            })

        site_vectors = []
        for atom in ligand.GetAtoms():
            if atom.GetAtomicNum() <= 1 or atom.GetTotalNumHs() < 1:
                continue
            index = atom.GetIdx()
            neighbors = [item for item in atom.GetNeighbors() if item.GetAtomicNum() > 1]
            if not neighbors:
                continue
            origin = np.asarray(ligand_xyz[index], dtype=float)
            center = np.mean([ligand_xyz[item.GetIdx()] for item in neighbors], axis=0)
            vector = origin - center
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-8:
                continue
            unit = vector / norm
            probe = origin + unit * 3.0
            nearest = min(
                (float(np.linalg.norm(protein.xyz - probe)) for protein in self.context.protein_atoms),
                default=float("nan"),
            )
            site_vectors.append({
                "target_type": "atom",
                "target_id": index,
                "origin_xyz": [round(float(value), 3) for value in origin],
                "outward_unit_vector": [round(float(value), 4) for value in unit],
                "probe_xyz_at_3A": [round(float(value), 3) for value in probe],
                "probe_clearance": round(nearest, 3) if np.isfinite(nearest) else None,
            })
        for site in self._bond_sites:
            profile = self.get_bond_site_spatial_profile(
                site["bond_site_id"], max_distance=min(4.0, float(radius)), probe_count=5
            )
            site_vectors.append({
                "target_type": "bond",
                "target_id": site["bond_site_id"],
                "retained_atom_index": site["retained_atom_index"],
                "attachment_vector": [round(float(value), 4) for value in site["attachment_vector"]],
                "directional_clearance": [
                    {
                        "label": item.get("label"),
                        "unit_vector": item.get("unit_vector"),
                        "minimum_protein_atom_distance_along_probe": item.get(
                            "minimum_protein_atom_distance_along_probe"
                        ),
                        "limiting_sample": {
                            "distance_from_attachment": (item.get("limiting_sample") or {}).get(
                                "distance_from_attachment"
                            ),
                            "probe_xyz": (item.get("limiting_sample") or {}).get("probe_xyz"),
                            "nearest_protein_atoms": [
                                {
                                    key: row.get(key)
                                    for key in ("atom", "element", "distance")
                                }
                                for row in ((item.get("limiting_sample") or {}).get("nearest_protein_atoms") or [])[:3]
                            ],
                        },
                    }
                    for item in profile.get("direction_profiles", [])
                ],
            })

        representation = {
            "status": "complete",
            "schema_version": "simple-molecular-agent.complex-geometry.v1",
            "coordinate_frame": "input_receptor_pdb_frame",
            "units": "angstrom",
            "coordinate_policy": "direct coordinates; no ligand alignment or fitted embedding",
            "ligand_centroid_xyz": [round(float(value), 3) for value in centroid],
            "ligand_atoms": ligand_atoms,
            "pocket_atoms": pocket_rows,
            "interaction_edges": interaction_edges,
            "edit_site_vectors": site_vectors,
            "truncation": {
                "pocket_radius": float(radius),
                "max_pocket_atoms": int(max_pocket_atoms),
                "pocket_atoms_returned": len(pocket_rows),
            },
            "limitations": [
                "Coordinates are host-extracted facts, not an LLM-learned 3D embedding.",
                "Pocket rows describe proximity and roles; they do not prove energetic contribution.",
                "Hydrogen-bond claims require donor/acceptor compatibility and pose-level verification.",
            ],
        }
        representation["geometry_identity"] = hashlib.sha256(
            json.dumps(representation, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return representation

    def get_design_dossier(self, panel_size: int = 6) -> dict[str, Any]:
        """Build the bounded initial context for sequential site-locked optimization."""
        if not 2 <= int(panel_size) <= 20:
            raise ValueError("panel_size must be between 2 and 20")
        sites = self.get_edit_site_candidates()
        enriched_sites = []
        for site in sites["atom_sites"] + sites["bond_sites"]:
            change_types = site.get("allowed_change_types") or self._site_change_types(site)
            panel_change = next(
                (item for item in ("replacement", "addition") if item in change_types), None
            )
            max_heavy_atoms = self._site_max_fragment_heavy_atoms(site)
            enriched_sites.append({
                **site,
                "allowed_change_types": change_types,
                "fragment_edit_change_type": panel_change,
                "recommended_max_fragment_heavy_atoms": max_heavy_atoms,
                "fragment_panel": _library_panel(
                    self.fragment_library, panel_change, int(panel_size), max_heavy_atoms
                ),
            })
        return {
            "status": "complete",
            "ligand": self.get_ligand_info(),
            "pocket": self.get_pocket_residues(6.0),
            "reference_interactions": self.detect_basic_interactions(4.5),
            "geometry": self.get_complex_geometry(),
            "fragment_library": self.fragment_library.overview(),
            "edit_taxonomy": taxonomy_documentation(),
            "sites": enriched_sites,
            "site_count": len(enriched_sites),
            "panel_size_per_site": int(panel_size),
            "tool_policy": (
                "Base chemistry and site facts are already supplied. Tools remain available only for "
                "decision-relevant uncertainty, refreshed panels, parent-specific facts, or pose analysis."
            ),
        }


    def get_fragment_panel(
        self,
        target_type: str,
        target_id: Any,
        size_classes: list[str] | None = None,
        chemical_tags: list[str] | None = None,
        max_heavy_atoms: int | None = None,
        limit: int = 12,
    ) -> dict[str, Any]:
        dossier = self.get_edit_site_candidates()
        sites = dossier["atom_sites"] + dossier["bond_sites"]
        site = next(
            (
                item for item in sites
                if item.get("target_type") == target_type and item.get("target_id") == target_id
            ),
            None,
        )
        if site is None:
            raise ValueError(f"Unknown host target: {target_type}:{target_id}")
        change_types = site.get("allowed_change_types") or self._site_change_types(site)
        panel_change = next(
            (item for item in ("replacement", "addition") if item in change_types), None
        )
        capacity = self._site_max_fragment_heavy_atoms(site)
        effective_max = min(capacity, int(max_heavy_atoms or capacity))
        fragments = _library_panel(
            self.fragment_library,
            panel_change,
            int(limit),
            effective_max,
            size_classes=size_classes,
            chemical_tags=chemical_tags,
        )
        return {
            "status": "complete",
            "target_type": target_type,
            "target_id": target_id,
            "allowed_change_types": change_types,
            "fragment_edit_change_type": panel_change,
            "site_capacity_heavy_atoms": capacity,
            "effective_max_heavy_atoms": effective_max,
            "requested_size_classes": size_classes,
            "requested_chemical_tags": chemical_tags,
            "count": len(fragments),
            "fragments": fragments,
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
        bond_sites = {
            site["bond_site_id"] for site in self._bond_sites
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
            if target_type not in {"atom", "bond"}:
                raise ValueError("site assessment target_type must be atom or bond_site")
            valid_target = (
                target_type == "atom" and isinstance(target_id, int) and target_id in editable_atoms
            ) or (
                target_type == "bond"
                and isinstance(target_id, str)
                and target_id in bond_sites
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

    @staticmethod
    def _molecule_graph(molecule: Chem.Mol) -> dict[str, Any]:
        """Return a compact graph projection without pose coordinates.

        Atom elements remain part of the graph because topology without element
        identity is not a chemically meaningful molecular graph. Coordinates
        stay in the PDB/SDF pose artifacts and are deliberately not duplicated
        in the LLM JSON context.
        """
        atoms = []
        for atom in molecule.GetAtoms():
            atoms.append({
                "atom_id": f"atom-{atom.GetIdx()}",
                "atom_index": atom.GetIdx(),
                "element": atom.GetSymbol(),
                "atomic_number": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "isotope": atom.GetIsotope(),
                "explicit_hydrogens": atom.GetNumExplicitHs(),
                "implicit_hydrogens": atom.GetNumImplicitHs(),
                "bond_degree": atom.GetDegree(),
                "aromatic": atom.GetIsAromatic(),
                "chiral_tag": str(atom.GetChiralTag()).split(".")[-1],
            })
        bonds = []
        for bond in molecule.GetBonds():
            bonds.append({
                "bond_id": f"bond-{bond.GetIdx()}",
                "atom_ids": [
                    f"atom-{bond.GetBeginAtomIdx()}",
                    f"atom-{bond.GetEndAtomIdx()}",
                ],
                "begin_atom_index": bond.GetBeginAtomIdx(),
                "end_atom_index": bond.GetEndAtomIdx(),
                "order": str(bond.GetBondType()).split(".")[-1],
                "aromatic": bond.GetIsAromatic(),
                "stereo": str(bond.GetStereo()).split(".")[-1],
            })
        hydrogen_free = Chem.RemoveHs(molecule)
        chemical_smiles = Chem.MolToSmiles(hydrogen_free, isomericSmiles=True)
        normalized_smiles = Chem.MolToSmiles(hydrogen_free, isomericSmiles=False)
        identity_material = {
            "normalized_smiles": normalized_smiles,
            "atoms": atoms,
            "bonds": bonds,
        }
        graph_identity = hashlib.sha256(
            json.dumps(identity_material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        chemical_identity = hashlib.sha256(
            json.dumps(
                {"chemical_smiles": chemical_smiles, "formal_charge": Chem.GetFormalCharge(molecule)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return {
            "schema_version": "simple-molecular-agent.molecule-graph.v1",
            "atom_identity": "rdkit_atom_index_for_current_structure",
            "normalized_smiles": normalized_smiles,
            "chemical_smiles": chemical_smiles,
            "canonical_smiles": chemical_smiles,
            "graph_identity": graph_identity,
            "chemical_identity": chemical_identity,
            "atoms": atoms,
            "bonds": bonds,
        }

    @staticmethod
    def _molecular_properties(molecule: Chem.Mol) -> dict[str, Any]:
        return {
            "schema_version": "simple-molecular-agent.molecule-properties.v1",
            "molecular_formula": rdMolDescriptors.CalcMolFormula(molecule),
            "molecular_weight": round(float(Descriptors.MolWt(molecule)), 4),
            "exact_mass": round(float(Descriptors.ExactMolWt(molecule)), 4),
            "formal_charge": int(Chem.GetFormalCharge(molecule)),
            "heavy_atoms": int(molecule.GetNumHeavyAtoms()),
            "component_count": len(Chem.GetMolFrags(molecule)),
            "logp": round(float(Crippen.MolLogP(molecule)), 4),
            "hbd": int(Lipinski.NumHDonors(molecule)),
            "hba": int(Lipinski.NumHAcceptors(molecule)),
            "tpsa": round(float(rdMolDescriptors.CalcTPSA(molecule)), 4),
            "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(molecule)),
            "ring_count": int(rdMolDescriptors.CalcNumRings(molecule)),
            "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
            "fraction_csp3": round(float(rdMolDescriptors.CalcFractionCSP3(molecule)), 4),
        }

    def get_ligand_info(self) -> dict[str, Any]:
        molecule = self.context.ligand
        graph = self._molecule_graph(molecule)
        properties = self._molecular_properties(molecule)
        return {
            "name": molecule.GetProp("_Name") if molecule.HasProp("_Name") else "ligand",
            "canonical_smiles": graph["canonical_smiles"],
            "formal_charge": properties["formal_charge"],
            "heavy_atoms": properties["heavy_atoms"],
            "molecular_weight": properties["molecular_weight"],
            "logp": properties["logp"],
            "hbd": properties["hbd"],
            "hba": properties["hba"],
            "tpsa": properties["tpsa"],
            "molecule_graph": graph,
            "molecular_properties": properties,
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

    def _build_bond_sites(self) -> list[dict[str, Any]]:
        if self.site_table is not None:
            return self._bond_sites_from_table()
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
                "allowed_operations": ["deletion", "replacement"],
                "allowed_change_types": ["deletion", "replacement"],
                **self._removal_profile(sorted(removed)),
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
            site["bond_site_id"] = f"bond-site-{number:03d}"
        return candidates

    def _bond_sites_from_table(self) -> list[dict[str, Any]]:
        molecule = Chem.RemoveHs(Chem.Mol(self.context.ligand))
        ring_atoms = {index for ring in molecule.GetRingInfo().AtomRings() for index in ring}
        sites = []
        for record in self.site_table["cut_sites"]:
            retained = list(record["retained_atom_indices"])
            removed = list(record["removed_atom_indices"])
            sites.append({
                "bond_site_id": record["site_id"],
                "site_id": record["site_id"],
                "region": record.get("region"),
                "label": record.get("label"),
                "cut_bond": list(record["cut_bond"]),
                "retained_atom_index": record["retained_atom_index"],
                "removed_side_atom_index": record["removed_side_atom_index"],
                "retained_atom_element": molecule.GetAtomWithIdx(record["retained_atom_index"]).GetSymbol(),
                "removed_side_atom_element": molecule.GetAtomWithIdx(record["removed_side_atom_index"]).GetSymbol(),
                "retained_heavy_atoms": record["retained_heavy_atoms"],
                "removed_heavy_atoms": record["removed_heavy_atoms"],
                "removed_fraction": record["removed_fraction"],
                "retained_ring_atoms": len(set(retained) & ring_atoms),
                "removed_ring_atoms": len(set(removed) & ring_atoms),
                "retained_atom_indices": retained,
                "removed_atom_indices": removed,
                "retained_scaffold_smiles": record["retained_scaffold_smiles"],
                "removed_fragment_smiles": record["removed_fragment_smiles"],
                "attachment_vector": list(record["attachment_vector"]),
                "allowed_operations": list(record.get("allowed_operations") or []),
                "allowed_change_types": self._site_change_types(record),
                **self._removal_profile(removed),
                "protection": record.get("protection"),
            })
        return sites

    def list_bond_sites(self, limit: int = 30) -> dict[str, Any]:
        return {
            "count": min(len(self._bond_sites), limit),
            "total_count": len(self._bond_sites),
            "sites": self._bond_sites[:limit],
            "policy": (
                "Each site is a directed non-ring single-bond cut. The first atom and larger "
                "ring-rich scaffold are retained; the listed removed side is deleted. Sites that "
                "remove more than the configured fraction or violate protected core atoms are excluded."
            ),
        }

    def resolve_bond_site(self, bond_site_id: str) -> dict[str, Any]:
        for site in self._bond_sites:
            if site["bond_site_id"] == bond_site_id:
                return dict(site)
        raise ValueError(
            f"Unknown bond_site_id: {bond_site_id}. "
            "Call list_bond_sites and select one returned ID."
        )

    @staticmethod
    def _removed_side_interactions(
        site: dict[str, Any], interactions_by_atom: dict[int, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Deterministic consequence of a deletion: which reference contacts disappear.

        The removed side of a bond site is fully known, so the lost contacts are a
        host fact rather than a prediction.  It lets the designer judge the real cost
        of a truncation instead of discovering it only after docking.
        """
        seen: set[tuple[Any, ...]] = set()
        lost: list[dict[str, Any]] = []
        for index in site.get("removed_atom_indices") or []:
            for contact in interactions_by_atom.get(index) or []:
                key = (
                    contact.get("kind"),
                    contact.get("protein_atom"),
                    contact.get("ligand_atom_index"),
                )
                if key in seen:
                    continue
                seen.add(key)
                lost.append(contact)
        return lost[:12]

    def get_bond_site_spatial_profile(
        self,
        bond_site_id: str,
        max_distance: float = 4.0,
        probe_count: int = 5,
    ) -> dict[str, Any]:
        site = self.resolve_bond_site(bond_site_id)
        conformer = self.context.ligand.GetConformer()
        point = conformer.GetAtomPosition(site["retained_atom_index"])
        origin = np.array([point.x, point.y, point.z], dtype=float)
        attachment = np.array(site["attachment_vector"], dtype=float)
        norm = float(np.linalg.norm(attachment))
        if norm < 1e-8:
            raise ValueError(f"Replacement site {bond_site_id} has no attachment direction")
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
            "bond_site_id": bond_site_id,
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

    def _resolve_bond_site(self, transformation: dict[str, Any]) -> None:
        site_id = transformation.get("bond_site_id")
        if not isinstance(site_id, str):
            raise ValueError(
                "A bond site requires bond_site_id from list_bond_sites; "
                "direct cut_bond input is not accepted"
            )
        site = self.resolve_bond_site(site_id)
        transformation["cut_bond"] = site["cut_bond"]
        transformation["edit_atom_index"] = site["retained_atom_index"]
        transformation["bond"] = site

    def _parent_ligand(self, parent_attempt: int | None = None) -> Chem.Mol:
        if parent_attempt is None:
            return self.context.ligand
        if self.parent_resolver is None:
            raise ValueError("parent_attempt is not available in this tool context")
        return self.parent_resolver(parent_attempt)

    def _enforce_site_policy(self, transformation: dict[str, Any]) -> dict[str, Any] | None:
        """Return a rejection when the chosen change type is not allowed at the site."""
        if self.site_table is None:
            return None
        normalize_transformation(transformation)
        site_type = transformation["site_type"]
        change_type = transformation["change_type"]
        if site_type in {"bond", "linker"}:
            site_id = transformation.get("bond_site_id")
            if not isinstance(site_id, str):
                return None  # the normal validation path reports the missing site id
            policy = self._site_table_policy(change_type, "bond", site_id)
        else:
            atom_index = transformation.get("edit_atom_index", transformation.get("atom_index"))
            if not isinstance(atom_index, int):
                return None
            policy = self._site_table_policy(change_type, "atom", atom_index)
        if policy.get("allowed"):
            transformation["site_policy"] = {
                "status": policy.get("status"),
                "protection": policy.get("protection"),
                "probe_verdict": policy.get("probe_verdict"),
                "source": policy.get("source"),
            }
            if policy.get("canonicalization"):
                transformation["site_canonicalization"] = policy["canonicalization"]
            return None
        return {
            "status": "rejected",
            "failure_class": policy.get("failure_class"),
            "error": policy.get("reason"),
            "site_policy": policy,
            "transformation": transformation,
        }

    def validate_candidate_geometry(
        self, parent_attempt: int | None = None, **transformation: Any
    ) -> dict[str, Any]:
        # The host, rather than the model, resolves site IDs and fragment IDs.
        parent = self._parent_ligand(parent_attempt)
        if parent_attempt is not None:
            transformation["parent_attempt"] = parent_attempt
        if "atom_index" in transformation and "edit_atom_index" not in transformation:
            transformation["edit_atom_index"] = transformation["atom_index"]
        target_type = transformation.get("target_type")
        target_id = transformation.get("target_id")
        if target_type == "atom" and transformation.get("edit_atom_index") is None:
            transformation["edit_atom_index"] = target_id
        elif target_type == "bond" and transformation.get("bond_site_id") is None:
            transformation["bond_site_id"] = target_id
        try:
            normalize_transformation(transformation)
        except EditTaxonomyError as exc:
            return {"status": "rejected", "error": str(exc), "transformation": transformation}
        policy_rejection = self._enforce_site_policy(transformation)
        if policy_rejection is not None:
            return policy_rejection
        apply_site_canonicalization(transformation)
        site_type = transformation["site_type"]
        change_type = transformation["change_type"]
        if site_type in {"bond", "linker"} and not transformation.get("replace_existing_substituent"):
            try:
                self._resolve_bond_site(transformation)
            except Exception as exc:
                return {"status": "rejected", "error": str(exc), "transformation": transformation}
        resolved_bond = transformation.get("bond") or {}
        if site_type in {"bond", "linker"} and resolved_bond.get("core_overlap"):
            overlap = resolved_bond["core_overlap"]
            return {
                "status": "rejected",
                "failure_class": "pose_core_removal",
                "error": (
                    f"This edit removes pose-comparison core atoms {overlap}; the native-like "
                    "reference comparison would be invalid. Choose a site whose removed side "
                    "lies outside the comparison core."
                ),
                "core_overlap": overlap,
                "transformation": transformation,
            }
        if change_type == "replacement" and site_type in {"atom", "ring"} and not isinstance(
            transformation.get("element"), str
        ):
            return {
                "status": "rejected",
                "error": "atom/ring replacement requires an element symbol",
                "transformation": transformation,
            }
        needs_fragment = (site_type == "atom" and change_type == "addition") or (
            site_type == "bond" and change_type == "replacement"
        )
        if needs_fragment and transformation.get("fragment_id"):
            record = self.fragment_library.get(str(transformation["fragment_id"]))
            library_operation = LIBRARY_OPERATION_FOR_CHANGE_TYPE.get(change_type, change_type)
            if not self.fragment_library.allows_operation(record, library_operation):
                return {
                    "status": "rejected",
                    "error": (
                        f"Fragment {transformation['fragment_id']} does not allow a "
                        f"{change_type} edit (library operation {library_operation})"
                    ),
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
        if needs_fragment and not isinstance(transformation.get("fragment_smiles"), str):
            return {
                "status": "rejected",
                "error": "The transformation requires fragment_smiles or a valid fragment_id",
                "transformation": transformation,
            }
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


    def search_fragment_library(
        self,
        query: str = "",
        max_heavy_atoms: int = 12,
        change_type: str = "addition",
        site_type: str | None = None,
        operation: str | None = None,
        size_class: str | None = None,
        chemical_tag: str | None = None,
        limit: int = 30,
    ) -> dict[str, Any]:
        # ``operation`` is accepted for older callers and run artifacts; the
        # canonical selector is ``change_type``.
        library_operation = operation or LIBRARY_OPERATION_FOR_CHANGE_TYPE.get(
            change_type, change_type
        )
        return self.fragment_library.search(
            query,
            max_heavy_atoms,
            library_operation,
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
