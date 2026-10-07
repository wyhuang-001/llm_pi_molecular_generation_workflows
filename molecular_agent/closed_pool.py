"""Public-only contract for a frozen, label-blind candidate pool.

Two modes are supported:

``frozen_cut`` (default)
    One frozen directed cut, one site, replacement only.  Every public record is
    ``{fragment_id, smiles, allowed_operations}`` and a fragment maps to exactly one
    product.

``multisite``
    The whole host site table is exposed and any legal change type may be chosen, so
    one compound can be reachable through several (site, fragment) pairs.  The catalog
    carries ``allowed_change_types`` and hits are keyed by the **product molecule**
    instead of the fragment id.

This module NEVER imports an evaluator or opens research/private activity files.
Library membership, full visibility, chemistry and budgets are Host-enforced.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rdkit import Chem

from .edit_taxonomy import EditTaxonomyError, normalize_transformation
from .fragment_library import FragmentLibrary, canonical_operation


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Unreadable molecular SMILES")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def scaffold(native: Chem.Mol, cut_bond: list[int]) -> Chem.Mol:
    molecule = Chem.RemoveHs(Chem.Mol(native))
    bond = molecule.GetBondBetweenAtoms(*cut_bond)
    if bond is None or bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
        raise ValueError("Closed-pool cut must be a non-ring single bond")
    for atom in molecule.GetAtoms():
        atom.SetIntProp("_pool_original_index", atom.GetIdx())
    split = Chem.FragmentOnBonds(molecule, [bond.GetIdx()], dummyLabels=[(0, 0)])
    for atom in split.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(1)
    parts = Chem.GetMolFrags(split, asMols=True)
    if len(parts) != 2:
        raise ValueError("Expected exactly two components after directed cut")
    return next(part for part in parts if any(a.HasProp("_pool_original_index") and
                a.GetIntProp("_pool_original_index") == cut_bond[0] for a in part.GetAtoms()))


def assemble(core: Chem.Mol, smiles: str) -> Chem.Mol:
    fragment = Chem.MolFromSmiles(smiles)
    if fragment is None or len(Chem.GetMolFrags(fragment)) != 1:
        raise ValueError("Attachment fragment must be a single connected molecule")
    dummies = [a for a in fragment.GetAtoms() if a.GetAtomicNum() == 0]
    if (len(dummies) != 1 or dummies[0].GetAtomMapNum() != 1 or dummies[0].GetDegree() != 1
            or any(a.GetAtomMapNum() for a in fragment.GetAtoms() if a.GetAtomicNum() != 0)):
        raise ValueError("Require exactly one [*:1] and no other atom maps")
    if dummies[0].GetBonds()[0].GetBondType() != Chem.BondType.SINGLE:
        raise ValueError("Attachment bond must be single")
    molecule = Chem.molzip(Chem.CombineMols(core, fragment))
    Chem.SanitizeMol(molecule)
    if any(a.GetAtomicNum() == 0 or a.GetAtomMapNum() for a in molecule.GetAtoms()):
        raise ValueError("Unresolved attachment after reconstruction")
    return molecule


def catalog_record(record: dict) -> dict:
    # Whitelist + single calculation method: no names, source counts or curated flags.
    properties = FragmentLibrary._properties(record["smiles"])
    return {"fragment_id": record["fragment_id"], "smiles": properties["canonical_smiles"],
            **{key: properties[key] for key in ("heavy_atoms", "formal_charge", "hbd", "hba", "tpsa", "rotatable_bonds")},
            **({"allowed_change_types": sorted(str(value) for value in record["allowed_change_types"])}
               if record.get("allowed_change_types") is not None else {})}


def product_for(parent: Chem.Mol, transformation: dict) -> str:
    """Canonical SMILES of the molecule one host-side edit builds.

    A multisite pool keys hits on this value rather than on a fragment id, because the
    same compound can be reached through more than one (site, fragment) pair.
    """
    from .editing import transformation_product_smiles

    return transformation_product_smiles(parent, transformation)


class ClosedPool:
    def __init__(self, context: Any, tools: Any):
        self.settings = context.task["closed_pool"]
        self.mode = str(self.settings.get("mode", "frozen_cut"))
        self.tools = tools
        self.ligand = Chem.Mol(context.ligand)
        self.library_path = tools.fragment_library.path
        self.manifest_path = (context.input_dir / self.settings["manifest_path"]).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        expected_library = self.manifest.get("library_sha256") or self.manifest.get(
            "public_files", {}
        ).get("catalog_sha256")
        if expected_library is None or expected_library != sha256(self.library_path):
            raise ValueError("Frozen candidate library hash mismatch")
        identity = self.manifest.get("structure_source_identity")
        if identity is not None and identity != context.preparation_identity():
            raise ValueError("Frozen pool was generated for different structure/topology/preparation")
        if not context.task.get("single_edit_mode", {}).get("enabled"):
            raise ValueError("Closed-pool benchmark requires single_edit_mode")
        if context.task.get("external_research", {}).get("enabled"):
            raise ValueError("Closed-pool benchmark disables browsing/research context")
        self.construction_policy = context.task.get("candidate_construction") or {}
        if self.construction_policy.get("initial_receptor_clash_policy", "reject") not in {"reject", "defer_to_docking"}:
            raise ValueError("Unsupported initial receptor clash policy")
        if self.construction_policy.get("initial_receptor_clash_policy") == "defer_to_docking":
            retention = context.task.get("pose_retention") or {}
            if not retention.get("enabled") or not 0 < float(retention.get("maximum_heavy_atom_overlap", 0)) <= .55:
                raise ValueError("Deferring initial receptor clashes requires an explicit final-pose overlap gate <=0.55 angstrom")
        self.budget = self.manifest["maximum_unique_candidates"]
        self.max_requests = self.manifest["maximum_design_requests"]
        if self.mode == "multisite":
            self._load_multisite(context, tools)
            return
        if context.task.get("docking_optimization", {}).get("hard_max_attempts") != self.budget:
            raise ValueError("Attempt budget must match frozen closed-pool contract")
        self.cut = self.manifest["cut_bond"]
        if self.settings["cut_bond"] != self.cut:
            raise ValueError("Frozen directed cut mismatch")
        sites = [site for site in tools._bond_sites if site["cut_bond"] == self.cut]
        if len(sites) != 1:
            raise ValueError("Frozen C6 cut is not a unique Host-legal replacement site")
        self.site = sites[0]
        core = scaffold(context.ligand, self.cut)
        retained_indices = {a.GetIntProp("_pool_original_index") for a in core.GetAtoms() if a.GetAtomicNum() != 0}
        retention = context.task.get("pose_retention") or {}
        if not retention.get("enabled") or set(retention.get("core_atom_indices", [])) != retained_indices:
            raise ValueError("Closed-pool protocol requires pose gating on the entire fixed non-side-chain core")
        self.by_id, self.by_smiles, self.products = {}, {}, {}
        allowed = {"fragment_id", "smiles", "allowed_operations"}
        native_smiles = Chem.MolToSmiles(Chem.RemoveHs(context.ligand), isomericSmiles=True)
        for record in tools.fragment_library.records:
            operations = [canonical_operation(str(value)) for value in record.get("allowed_operations") or []]
            if set(record) != allowed or operations != ["bond:replacement"]:
                raise ValueError("Public records must contain only ID, attachment SMILES and bond:replacement permission")
            fragment_id, smiles = record["fragment_id"], canonical(record["smiles"])
            if fragment_id in self.by_id or smiles in self.by_smiles:
                raise ValueError("Duplicate pool fragment ID or chemical structure")
            product = assemble(core, smiles)
            product_smiles = Chem.MolToSmiles(product, isomericSmiles=True)
            if product_smiles == native_smiles or product_smiles in self.products.values():
                raise ValueError("Duplicate full molecule / original-ligand no-op in pool")
            if Chem.GetFormalCharge(product) != Chem.GetFormalCharge(context.ligand):
                raise ValueError("Pool candidate changes the original formal charge")
            self.by_id[fragment_id] = {**record, "smiles": smiles}
            self.by_smiles[smiles] = fragment_id
            self.products[fragment_id] = product_smiles
        expected_count = self.manifest.get("candidate_count") or self.manifest.get(
            "counts", {}
        ).get("catalog_total")
        if expected_count is not None and len(self.by_id) != expected_count:
            raise ValueError("Frozen candidate count mismatch")
        self.catalog = [catalog_record(record) for record in self.by_id.values()]

    # ------------------------------------------------------------------ #
    # multisite: whole site table, any legal change type, product-keyed hits
    # ------------------------------------------------------------------ #
    def _load_multisite(self, context: Any, tools: Any) -> None:
        if not context.task.get("edit_site_table_path"):
            raise ValueError("A multisite closed pool requires the task edit-site table")
        configured = context.task.get("fragment_library_path")
        if configured:
            expected = (context.input_dir / str(configured)).resolve()
            if expected != self.library_path:
                raise ValueError("Closed pool library does not match task fragment_library_path")
        self.bond_sites = [dict(site) for site in tools.list_bond_sites(limit=200)["sites"]]
        self.atom_sites = [
            dict(site) for site in tools.get_edit_site_candidates()["atom_sites"]
        ]
        self.native_smiles = canonical(
            Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(context.ligand)), isomericSmiles=True)
        )
        self.by_id: dict[str, dict] = {}
        self.by_smiles: dict[str, str] = {}
        for record in tools.fragment_library.records:
            if not {"fragment_id", "smiles", "allowed_change_types"} <= set(record):
                raise ValueError(
                    "Multisite records need fragment_id, smiles and allowed_change_types"
                )
            change_types = [str(value) for value in record["allowed_change_types"]]
            if not change_types or any(
                value not in {"addition", "replacement"} for value in change_types
            ):
                raise ValueError("A fragment may only allow addition and/or replacement")
            fragment_id, smiles = record["fragment_id"], canonical(record["smiles"])
            if fragment_id in self.by_id or smiles in self.by_smiles:
                raise ValueError("Duplicate pool fragment ID or chemical structure")
            self.by_id[fragment_id] = {**record, "smiles": smiles}
            self.by_smiles[smiles] = fragment_id
        expected_count = self.manifest.get("candidate_count") or self.manifest.get(
            "counts", {}
        ).get("catalog_total")
        if expected_count is not None and len(self.by_id) != expected_count:
            raise ValueError("Frozen candidate count mismatch")
        self.catalog = [catalog_record(record) for record in self.by_id.values()]

    def _site_for(self, site_type: str, target_id: Any) -> dict | None:
        if site_type in {"bond", "linker"}:
            return next(
                (site for site in self.bond_sites if site["bond_site_id"] == target_id), None
            )
        return next(
            (site for site in self.atom_sites if site["target_id"] == target_id), None
        )

    def _normalize_multisite(self, decision: dict) -> dict:
        if decision.get("action") == "STOP":
            return decision
        try:
            axes = normalize_transformation(dict(decision))
        except EditTaxonomyError as error:
            raise ValueError(f"Unsupported edit request: {error}") from error
        site_type, change_type = axes["site_type"], axes["change_type"]
        if "parent_attempt" in decision or "cut_bond" in decision:
            raise ValueError("Multisite pool allows one original-ligand edit, no parent or cut_bond")
        if site_type in {"bond", "linker"}:
            site_id = decision.get("bond_site_id")
            site = self._site_for(site_type, site_id)
            if site is None:
                raise ValueError("Select one bond_site_id from the exposed site list")
            if change_type not in (site.get("allowed_change_types") or []):
                raise ValueError(f"Site {site_id} does not allow change_type {change_type}")
            out = {**decision, "site_type": "bond", "change_type": change_type,
                   "cut_bond": list(site["cut_bond"]),
                   "edit_atom_index": site["retained_atom_index"]}
        else:
            atom_index = decision.get("edit_atom_index", decision.get("atom_index"))
            site = self._site_for(site_type, atom_index)
            if site is None:
                raise ValueError("Select one edit_atom_index from the exposed site list")
            if change_type not in (site.get("allowed_change_types") or []):
                raise ValueError(f"Site {atom_index} does not allow change_type {change_type}")
            out = {**decision, "site_type": site_type, "change_type": change_type,
                   "edit_atom_index": atom_index}

        needs_fragment = (
            change_type == "addition"
            or (change_type == "replacement" and site_type in {"bond", "linker"})
        )
        if needs_fragment:
            fragment_id = decision.get("fragment_id")
            smiles = decision.get("fragment_smiles")
            if smiles:
                mapped = self.by_smiles.get(canonical(smiles))
                if not mapped or (fragment_id and fragment_id != mapped):
                    raise ValueError("SMILES is outside the frozen pool or conflicts with fragment_id")
                fragment_id = mapped
            if fragment_id not in self.by_id:
                raise ValueError("Select one fragment_id from the full visible closed-pool catalog")
            record = self.by_id[fragment_id]
            if change_type not in record["allowed_change_types"]:
                raise ValueError(f"Fragment {fragment_id} does not allow change_type {change_type}")
            out["fragment_id"] = fragment_id
            out["fragment_smiles"] = record["smiles"]
        elif change_type == "replacement":
            # atom:replacement and ring:replacement are element swaps (e.g. Cl -> Br),
            # not fragment replacements; they take an element symbol, not a fragment.
            element = decision.get("element")
            if not isinstance(element, str) or not element.strip():
                raise ValueError("An atom/ring replacement needs an element symbol")
            out["element"] = element
        return out

    def product_smiles(self, decision: dict) -> str:
        """Product molecule of a normalized decision; the multisite hit key."""
        return product_for(self.ligand, decision)

    def _dossier_multisite(self, tools: Any) -> dict:
        sites = [{**site, "target_type": "bond", "site_type": "bond",
                  "target_id": site["bond_site_id"],
                  "allowed_change_types": site.get("allowed_change_types") or []}
                 for site in self.bond_sites]
        sites += [{**site, "target_type": "atom",
                   "site_type": site.get("site_type", "atom"),
                   "target_id": site["target_id"],
                   "allowed_change_types": site.get("allowed_change_types") or []}
                  for site in self.atom_sites]
        return {
            "status": "complete",
            "site_count": len(sites),
            "sites": sites,
            "fragment_catalog": self.catalog,
            "selection_contract": {
                "mode": "closed_pool_multisite",
                "candidate_count": len(self.catalog),
                "full_catalog_visible": True,
                "maximum_unique_candidates": self.budget,
                "candidate_construction": self.construction_policy,
                "instruction": (
                    "Choose one listed site and one change_type that the site allows; supply a "
                    "listed fragment_id for addition or bond replacement, an element symbol for "
                    "an atom or ring replacement, and no fragment for a bond deletion. No free "
                    "fragments, no parents and no unlisted sites."
                ),
                "descriptor_method": (
                    "RDKit attachment-fragment descriptors (dummy retained); same method for "
                    "every entry"
                ),
            },
        }

    def normalize(self, decision: dict) -> dict:
        if self.mode == "multisite":
            return self._normalize_multisite(decision)
        if decision.get("action") == "STOP":
            return decision
        try:
            axes = normalize_transformation(dict(decision))
        except EditTaxonomyError as error:
            raise ValueError(f"Closed pool requires one original-ligand bond:replacement: {error}") from error
        if (
            (axes["site_type"], axes["change_type"]) != ("bond", "replacement")
            or "parent_attempt" in decision
            or "cut_bond" in decision
        ):
            raise ValueError("Closed pool requires one original-ligand bond:replacement, no parent or direct cut_bond")
        if decision.get("bond_site_id") != self.site["bond_site_id"]:
            raise ValueError("Only the supplied C6 bond_site_id is allowed")
        if "edit_atom_index" in decision and decision["edit_atom_index"] != self.cut[0]:
            raise ValueError("Conflicting edit_atom_index for the frozen C6 site")
        fragment_id = decision.get("fragment_id")
        smiles = decision.get("fragment_smiles")
        if smiles:
            from_smiles = self.by_smiles.get(canonical(smiles))
            if not from_smiles or (fragment_id and fragment_id != from_smiles):
                raise ValueError("SMILES is outside the frozen pool or conflicts with fragment_id")
            fragment_id = from_smiles
        if fragment_id not in self.by_id:
            raise ValueError("Select one fragment_id from the full visible closed-pool catalog")
        return {**decision, "fragment_id": fragment_id,
                "fragment_smiles": self.by_id[fragment_id]["smiles"]}

    def dossier(self, tools: Any) -> dict:
        if self.mode == "multisite":
            return self._dossier_multisite(tools)
        return {"status": "complete", "site_count": 1, "sites": [{
            **self.site, "target_type": "bond", "site_type": "bond",
            "target_id": self.site["bond_site_id"],
            "allowed_change_types": ["replacement"],
            "spatial_profile": tools.get_bond_site_spatial_profile(self.site["bond_site_id"])}],
            "fragment_catalog": self.catalog,
            "selection_contract": {"mode": "closed_pool", "candidate_count": len(self.catalog),
                "full_catalog_visible": True, "cut_bond": self.cut, "maximum_unique_candidates": self.budget,
                "candidate_construction": self.construction_policy,
                "instruction": "Choose one listed fragment_id and the sole bond_site_id. No other sites, free fragments, or parents.",
                "descriptor_method": "RDKit attachment-fragment descriptors (dummy retained); same method for every entry"}}

    def bind_run(self, directory: Path, context: Any, docking_config: dict) -> None:
        manifest = {"schema_version": 1, "task_sha256": sha256(context.task_path),
                    "library_sha256": sha256(self.library_path), "pool_manifest_sha256": sha256(self.manifest_path),
                    "candidate_count": len(self.catalog), "maximum_unique_candidates": self.budget,
                    "docking_protocol": docking_config,
                    "implementation_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                        ("closed_pool.py", "workflow.py", "llm.py", "editing.py", "tools.py")}}
        path = directory / "closed-pool-run.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Closed-pool run inputs/protocol changed; use a new run directory")
        if not path.exists():
            if any(directory.glob("edit-attempt-*.json")) or (directory / "state-checkpoint.json").exists():
                raise ValueError("Cannot reuse an unversioned non-benchmark run")
            path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
            (directory / "closed-pool-library.json").write_bytes(self.library_path.read_bytes())
        elif sha256(directory / "closed-pool-library.json") != manifest["library_sha256"]:
            raise ValueError("Closed-pool run library snapshot changed")

    def audit_request(self, directory: Path, payload: dict) -> tuple[int, Path]:
        previous = list(directory.glob("closed-pool-request-*.json"))
        number = max([int(path.stem.split("-")[-1]) for path in previous] or [0]) + 1
        if number > self.max_requests:
            raise RuntimeError("Closed-pool design request budget exhausted")
        visible = payload.get("design_dossier", {}).get("fragment_catalog", [])
        if visible != self.catalog:
            raise ValueError("Designer payload truncated, reordered or changed the frozen complete catalog")
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        record = {"request_number": number, "visible_fragment_ids": [r["fragment_id"] for r in visible],
                  "payload_sha256": hashlib.sha256(encoded.encode()).hexdigest(), "payload": payload}
        path = directory / f"closed-pool-request-{number:03d}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        return number, directory / f"closed-pool-response-{number:03d}.json"
