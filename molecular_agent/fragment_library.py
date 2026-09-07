from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


DEFAULT_LIBRARY_PATH = Path(__file__).resolve().parent / "data" / "fragments.json"

SIZE_CLASSES = {
    "minimal": {"min_heavy_atoms": 1, "max_heavy_atoms": 1},
    "small": {"min_heavy_atoms": 2, "max_heavy_atoms": 4},
    "medium": {"min_heavy_atoms": 5, "max_heavy_atoms": 8},
    "large": {"min_heavy_atoms": 9, "max_heavy_atoms": 12},
}


def size_class_for(heavy_atoms: int) -> str:
    for name, limits in SIZE_CLASSES.items():
        if limits["min_heavy_atoms"] <= heavy_atoms <= limits["max_heavy_atoms"]:
            return name
    return "oversized"


def chemical_tags(molecule: Chem.Mol) -> list[str]:
    heavy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    symbols = {atom.GetSymbol() for atom in heavy_atoms}
    tags: set[str] = set()
    if symbols & {"F", "Cl", "Br", "I"}:
        tags.add("halogen")
    if symbols & {"N", "O", "S"}:
        tags.add("polar")
    if heavy_atoms and all(
        atom.GetSymbol() == "C" and not atom.GetIsAromatic() for atom in heavy_atoms
    ):
        tags.add("alkyl")
    if any(atom.GetIsAromatic() for atom in heavy_atoms):
        tags.add("aromatic")
    if any(
        atom.GetIsAromatic() and atom.GetSymbol() in {"N", "O", "S"}
        for atom in heavy_atoms
    ):
        tags.add("heteroaryl")
    ring_count = rdMolDescriptors.CalcNumRings(molecule)
    if ring_count:
        tags.add("cyclic")
    if ring_count and not any(atom.GetIsAromatic() for atom in heavy_atoms):
        tags.add("saturated_ring")
    if ring_count and symbols & {"N", "O", "S"} and "heteroaryl" not in tags:
        tags.add("saturated_heterocycle")
    if molecule.HasSubstructMatch(Chem.MolFromSmarts("[CX2]#[NX1]")):
        tags.add("nitrile")
    if molecule.HasSubstructMatch(Chem.MolFromSmarts("[CX3]=[OX1]")):
        tags.add("carbonyl")
    if molecule.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")):
        tags.add("amide")
    if molecule.HasSubstructMatch(Chem.MolFromSmarts("[OX2]-[CX4]")):
        tags.add("alkoxy")
    if Lipinski.NumHDonors(molecule):
        tags.add("hbond_donor")
    if Lipinski.NumHAcceptors(molecule):
        tags.add("hbond_acceptor")
    return sorted(tags or {"other"})

# Common medicinal-chemistry names are resolved structurally. Unknown queries
# still use auditable metadata text matching.
CHEMICAL_QUERY_SMARTS = {
    "amide": "[CX3](=[OX1])[NX3]",
    "aniline": "[NX3]-c1ccccc1",
    "aminopyridine": "[NX3]-c1ccccn1",
    "cyano": "[CX2]#[NX1]",
    "fluoro": "[F]",
    "fluorophenyl": "[F]-[c]1[c][c][c][c][c]1",
    "heterocycle": "[r;!#6]",
    "hydroxymethyl": "[CH2][OH1]",
    "indazole": "c1ccc2[nH]ncc2c1",
    "indole": "c1ccc2[nH]ccc2c1",
    "methyl": "[CH3]",
    "methylpiperazine": "CN1CCNCC1",
    "morpholine": "O1CCNCC1",
    "nitrile": "[CX2]#[NX1]",
    "oxetane": "C1COC1",
    "phenyl": "c1ccccc1",
    "polar heterocycle": "[r;!#6]",
    "pyridine": "n1ccccc1",
    "pyridylamine": "[NX3]-c1ccccn1",
    "pyrimidine": "n1ccnc(n1)",
}


class FragmentLibrary:
    """Small, auditable fragment library with optional user-supplied records."""

    def __init__(self, path: Path | None = None):
        self.path = (path or DEFAULT_LIBRARY_PATH).resolve()
        self.default_allowed_operations: set[str] = set()
        if not self.path.exists():
            self.records: list[dict[str, Any]] = []
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            self.records = payload.get("fragments", [])
            configured = payload.get("allowed_operations")
            if isinstance(configured, list):
                self.default_allowed_operations = {str(value) for value in configured}
        elif isinstance(payload, list):
            self.records = payload
        else:
            self.records = []
        if not isinstance(self.records, list):
            raise ValueError(f"Fragment library must contain a list: {self.path}")

    @staticmethod
    def _properties(smiles: str) -> dict[str, Any]:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid fragment SMILES: {smiles}")
        return {
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
            "rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
            "ring_count": rdMolDescriptors.CalcNumRings(molecule),
            "aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(molecule),
        }

    def _allowed_operations(self, record: dict[str, Any]) -> set[str]:
        configured = record.get("allowed_operations")
        if isinstance(configured, list):
            return {str(value) for value in configured}
        if self.default_allowed_operations:
            return set(self.default_allowed_operations)
        operation = record.get("operation")
        return {str(operation)} if isinstance(operation, str) else set()

    @staticmethod
    def smiles_equivalent(left: str, right: str) -> bool:
        """Compare fragment structures, not their traversal-dependent SMILES text."""
        if not isinstance(left, str) or not isinstance(right, str):
            return False
        left_molecule = Chem.MolFromSmiles(left)
        right_molecule = Chem.MolFromSmiles(right)
        if left_molecule is None or right_molecule is None:
            return False
        return Chem.MolToSmiles(left_molecule, isomericSmiles=True) == Chem.MolToSmiles(
            right_molecule, isomericSmiles=True
        )

    def allows_operation(self, record: dict[str, Any], operation: str) -> bool:
        return operation in self._allowed_operations(record)

    def search(
        self,
        query: str = "",
        max_heavy_atoms: int = 12,
        operation: str = "substitute",
        limit: int = 30,
        size_class: str | None = None,
        chemical_tag: str | None = None,
    ) -> dict[str, Any]:
        query = query.lower().strip()
        if size_class is not None and size_class not in SIZE_CLASSES:
            raise ValueError(f"Unknown size_class: {size_class}")
        chemical_tag = chemical_tag.lower().strip() if isinstance(chemical_tag, str) else None
        query_smarts = CHEMICAL_QUERY_SMARTS.get(query)
        if query_smarts is None and query and any(character in query for character in "[]()=#@1234567890"):
            query_smarts = query
        structural_query = Chem.MolFromSmarts(query_smarts) if query_smarts else None
        supported_queries = sorted(CHEMICAL_QUERY_SMARTS)
        if query and structural_query is None and any(character.isspace() for character in query):
            return {
                "status": "rejected",
                "failure_class": "unsupported_fragment_query",
                "library_path": str(self.path),
                "query": query,
                "operation": operation,
                "count": 0,
                "fragments": [],
                "supported_chemical_queries": supported_queries,
                "recommended_queries": [
                    {"query": "heterocycle", "operation": operation, "max_heavy_atoms": max_heavy_atoms, "limit": limit},
                    {"query": "pyridine", "operation": operation, "max_heavy_atoms": max_heavy_atoms, "limit": limit},
                    {"query": "morpholine", "operation": operation, "max_heavy_atoms": max_heavy_atoms, "limit": limit},
                    {"query": "", "operation": operation, "max_heavy_atoms": max_heavy_atoms, "limit": limit},
                ],
                "error": (
                    "query must be one supported chemical term, one valid SMILES/SMARTS pattern, "
                    "or an empty string for browsing; natural-language descriptions are not searchable"
                ),
            }
        match_mode = "chemical_substructure" if structural_query is not None else "metadata_text"
        if not query:
            match_mode = "unfiltered"
        matches = []
        operation_compatible_records = sum(
            isinstance(record, dict) and operation in self._allowed_operations(record)
            for record in self.records
        )
        filtered_compatible_records = 0
        size_class_counts: Counter[str] = Counter()
        chemical_tag_counts: Counter[str] = Counter()
        for record in self.records:
            if not isinstance(record, dict):
                continue
            if operation not in self._allowed_operations(record):
                continue
            smiles = record.get("smiles")
            if not isinstance(smiles, str):
                continue
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None or molecule.GetNumHeavyAtoms() > max_heavy_atoms:
                continue
            record_size_class = record.get("size_class") or size_class_for(molecule.GetNumHeavyAtoms())
            record_tags = set(record.get("chemical_tags") or chemical_tags(molecule))
            size_class_counts[record_size_class] += 1
            chemical_tag_counts.update(record_tags)
            if size_class is not None and record_size_class != size_class:
                continue
            if chemical_tag is not None and chemical_tag not in record_tags:
                continue
            filtered_compatible_records += 1
            if structural_query is not None:
                if not molecule.HasSubstructMatch(structural_query):
                    continue
            elif query and query not in json.dumps(record, ensure_ascii=False).lower():
                continue
            mappings = [
                atom.GetAtomMapNum()
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() == 0
            ]
            if len(mappings) != 1:
                continue
            matches.append({
                **record,
                "size_class": record_size_class,
                "chemical_tags": sorted(record_tags),
                "allowed_operations": sorted(self._allowed_operations(record)),
                "properties": self._properties(smiles),
                "attachment_points": mappings,
                "matched_by": match_mode,
            })
            if len(matches) >= limit:
                break
        return {
            "status": "complete",
            "library_path": str(self.path),
            "query": query,
            "query_smarts": query_smarts,
            "match_mode": match_mode,
            "operation": operation,
            "size_class": size_class,
            "chemical_tag": chemical_tag,
            "operation_compatible_records": operation_compatible_records,
            "filtered_compatible_records": filtered_compatible_records,
            "size_class_counts": dict(sorted(size_class_counts.items())),
            "chemical_tag_counts": dict(sorted(chemical_tag_counts.items())),
            "count": len(matches),
            "fragments": matches,
            "supported_chemical_queries": supported_queries,
            "limitation": (
                "Library membership and substructure matching are chemical starting-point filters, "
                "not predictions of binding or activity. Empty results mean no record explicitly "
                "allows the requested operation under the current size limit."
            ),
        }

    def _compact_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Return the chemistry needed for LLM selection without bulky provenance."""
        smiles = record.get("smiles")
        if not isinstance(smiles, str):
            raise ValueError("Fragment record is missing smiles")
        properties = self._properties(smiles)
        return {
            "fragment_id": record.get("fragment_id"),
            "name": record.get("name"),
            "smiles": smiles,
            "canonical_smiles": record.get("canonical_smiles") or properties["canonical_smiles"],
            "allowed_operations": sorted(self._allowed_operations(record)),
            "size_class": record.get("size_class") or properties["size_class"],
            "chemical_tags": record.get("chemical_tags") or properties["chemical_tags"],
            "attachment_atom_element": record.get("attachment_atom_element"),
            "formal_charge": record.get("formal_charge", properties["formal_charge"]),
            "heavy_atoms": record.get("heavy_atoms", properties["heavy_atoms"]),
            "molecular_weight": record.get("molecular_weight", properties["molecular_weight"]),
            "logp": record.get("logp", properties["logp"]),
            "hbd": record.get("hbd", properties["hbd"]),
            "hba": record.get("hba", properties["hba"]),
            "tpsa": record.get("tpsa", properties["tpsa"]),
            "rotatable_bonds": record.get("rotatable_bonds", properties["rotatable_bonds"]),
            "ring_count": record.get("ring_count", properties["ring_count"]),
            "aromatic_ring_count": record.get(
                "aromatic_ring_count", properties["aromatic_ring_count"]
            ),
            "curated": bool(record.get("curated", False)),
        }

    def overview(self) -> dict[str, Any]:
        """Summarize the complete host-side library without sending every record."""
        size_counts: Counter[str] = Counter()
        tag_counts: Counter[str] = Counter()
        operation_counts: Counter[str] = Counter()
        charge_counts: Counter[str] = Counter()
        for record in self.records:
            if not isinstance(record, dict) or not isinstance(record.get("smiles"), str):
                continue
            heavy_atoms = record.get("heavy_atoms")
            if not isinstance(heavy_atoms, int):
                molecule = Chem.MolFromSmiles(record["smiles"])
                if molecule is None:
                    continue
                heavy_atoms = molecule.GetNumHeavyAtoms()
            size_counts[str(record.get("size_class") or size_class_for(heavy_atoms))] += 1
            tag_counts.update(record.get("chemical_tags") or [])
            operation_counts.update(self._allowed_operations(record))
            charge_counts[str(record.get("formal_charge", 0))] += 1
        return {
            "library_path": str(self.path),
            "fragment_count": len(self.records),
            "size_class_counts": dict(sorted(size_counts.items())),
            "chemical_tag_counts": dict(tag_counts.most_common()),
            "operation_counts": dict(sorted(operation_counts.items())),
            "formal_charge_counts": dict(sorted(charge_counts.items())),
            "selection_contract": (
                "The complete library is loaded by the host. The LLM receives compact diverse panels "
                "and may request refreshed panels; per-fragment property queries are not required."
            ),
        }

    def panel(
        self,
        *,
        operation: str,
        limit: int = 12,
        max_heavy_atoms: int = 12,
        size_classes: list[str] | None = None,
        chemical_tags_any: list[str] | None = None,
        exclude_fragment_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Select a deterministic, chemically diverse panel from the full library."""
        if limit < 1:
            return []
        allowed_sizes = set(size_classes or SIZE_CLASSES)
        unknown_sizes = allowed_sizes - set(SIZE_CLASSES)
        if unknown_sizes:
            raise ValueError(f"Unknown size classes: {sorted(unknown_sizes)}")
        requested_tags = {
            str(value).lower().strip() for value in (chemical_tags_any or []) if str(value).strip()
        }
        excluded = exclude_fragment_ids or set()
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
        tag_priority = (
            "halogen", "alkyl", "polar", "heteroaryl", "nitrile",
            "hbond_donor", "hbond_acceptor", "cyclic", "other",
        )
        for record in self.records:
            if not isinstance(record, dict) or record.get("fragment_id") in excluded:
                continue
            if operation not in self._allowed_operations(record):
                continue
            smiles = record.get("smiles")
            if not isinstance(smiles, str):
                continue
            heavy_atoms = record.get("heavy_atoms")
            if not isinstance(heavy_atoms, int):
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    continue
                heavy_atoms = molecule.GetNumHeavyAtoms()
            size = str(record.get("size_class") or size_class_for(heavy_atoms))
            tags = {str(value).lower() for value in (record.get("chemical_tags") or [])}
            if heavy_atoms > max_heavy_atoms or size not in allowed_sizes:
                continue
            if requested_tags and not requested_tags.intersection(tags):
                continue
            primary_tag = next((tag for tag in tag_priority if tag in tags), "other")
            buckets.setdefault((size, primary_tag), []).append(record)

        for records in buckets.values():
            records.sort(
                key=lambda item: (
                    not bool(item.get("curated", False)),
                    -int(item.get("source_molecule_count", 0) or 0),
                    str(item.get("fragment_id", "")),
                )
            )
        ordered_keys = [
            (size, tag)
            for size in ("minimal", "small", "medium", "large")
            for tag in tag_priority
            if (size, tag) in buckets
        ]
        selected: list[dict[str, Any]] = []
        cursor = 0
        while len(selected) < limit and ordered_keys:
            key = ordered_keys[cursor % len(ordered_keys)]
            records = buckets[key]
            if records:
                selected.append(self._compact_record(records.pop(0)))
            if not records:
                ordered_keys.remove(key)
                if not ordered_keys:
                    break
                cursor %= len(ordered_keys)
            else:
                cursor += 1
        return selected

    def get(self, fragment_id: str) -> dict[str, Any]:
        for record in self.records:
            if isinstance(record, dict) and record.get("fragment_id") == fragment_id:
                properties = self._properties(record["smiles"])
                return {
                    **record,
                    "size_class": record.get("size_class") or properties["size_class"],
                    "chemical_tags": record.get("chemical_tags") or properties["chemical_tags"],
                    "allowed_operations": sorted(self._allowed_operations(record)),
                    "properties": properties,
                }
        raise ValueError(f"Unknown fragment_id: {fragment_id}")
