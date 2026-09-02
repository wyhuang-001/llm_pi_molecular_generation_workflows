#!/usr/bin/env python3
"""Merge curated and ChEMBL fragments into one tagged, auditable library."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


SIZE_CLASSES = {
    "minimal": {"min_heavy_atoms": 1, "max_heavy_atoms": 1},
    "small": {"min_heavy_atoms": 2, "max_heavy_atoms": 4},
    "medium": {"min_heavy_atoms": 5, "max_heavy_atoms": 8},
    "large": {"min_heavy_atoms": 9, "max_heavy_atoms": 12},
}
SIZE_RANK = {name: index for index, name in enumerate(SIZE_CLASSES)}


def size_class_for(heavy_atoms: int) -> str:
    for name, limits in SIZE_CLASSES.items():
        if limits["min_heavy_atoms"] <= heavy_atoms <= limits["max_heavy_atoms"]:
            return name
    raise ValueError(f"Fragment heavy atom count is outside configured classes: {heavy_atoms}")


def _has_match(molecule: Chem.Mol, smarts: str) -> bool:
    query = Chem.MolFromSmarts(smarts)
    return bool(query and molecule.HasSubstructMatch(query))


def chemical_tags(molecule: Chem.Mol) -> list[str]:
    heavy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    symbols = {atom.GetSymbol() for atom in heavy_atoms}
    tags: set[str] = set()
    if symbols & {"F", "Cl", "Br", "I"}:
        tags.add("halogen")
    if symbols & {"N", "O", "S"}:
        tags.add("polar")
    if all(atom.GetSymbol() == "C" and not atom.GetIsAromatic() for atom in heavy_atoms):
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
    if ring_count and symbols & {"N", "O", "S"} and not tags.intersection({"heteroaryl"}):
        tags.add("saturated_heterocycle")
    if _has_match(molecule, "[CX2]#[NX1]"):
        tags.add("nitrile")
    if _has_match(molecule, "[CX3]=[OX1]"):
        tags.add("carbonyl")
    if _has_match(molecule, "[CX3](=[OX1])[NX3]"):
        tags.add("amide")
    if _has_match(molecule, "[OX2]-[CX4]"):
        tags.add("alkoxy")
    if Lipinski.NumHDonors(molecule):
        tags.add("hbond_donor")
    if Lipinski.NumHAcceptors(molecule):
        tags.add("hbond_acceptor")
    return sorted(tags or {"other"})


def fragment_metadata(smiles: str) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid fragment SMILES: {smiles}")
    dummy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1 or dummy_atoms[0].GetAtomMapNum() != 1:
        raise ValueError(f"Fragment must contain exactly one [*:1] attachment point: {smiles}")
    neighbors = list(dummy_atoms[0].GetNeighbors())
    if len(neighbors) != 1:
        raise ValueError(f"Fragment attachment point must have one neighbor: {smiles}")
    heavy_atoms = molecule.GetNumHeavyAtoms()
    return {
        "canonical_smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
        "heavy_atoms": heavy_atoms,
        "size_class": size_class_for(heavy_atoms),
        "chemical_tags": chemical_tags(molecule),
        "attachment_atom_element": neighbors[0].GetSymbol(),
        "formal_charge": Chem.GetFormalCharge(molecule),
        "molecular_weight": round(Descriptors.MolWt(molecule), 2),
        "logp": round(Crippen.MolLogP(molecule), 2),
        "hbd": Lipinski.NumHDonors(molecule),
        "hba": Lipinski.NumHAcceptors(molecule),
        "tpsa": round(rdMolDescriptors.CalcTPSA(molecule), 2),
        "rotatable_bonds": Lipinski.NumRotatableBonds(molecule),
        "ring_count": rdMolDescriptors.CalcNumRings(molecule),
        "aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(molecule),
    }


def _record_operations(payload: dict[str, Any], record: dict[str, Any]) -> list[str]:
    configured = record.get("allowed_operations")
    if isinstance(configured, list):
        return sorted({str(value) for value in configured})
    defaults = payload.get("allowed_operations")
    if isinstance(defaults, list):
        return sorted({str(value) for value in defaults})
    operation = record.get("operation")
    return [str(operation)] if isinstance(operation, str) else []


def _source_summary(payload: dict[str, Any], source_id: str) -> dict[str, Any]:
    source = dict(payload.get("source") or {})
    return {"source_id": source_id, **source}


def _enriched_record(
    payload: dict[str, Any],
    record: dict[str, Any],
    source_id: str,
    curated: bool,
) -> dict[str, Any]:
    metadata = fragment_metadata(str(record["smiles"]))
    original_id = str(record["fragment_id"])
    enriched = {
        **record,
        **metadata,
        "fragment_id": f"curated-{original_id}" if curated else original_id,
        "allowed_operations": _record_operations(payload, record),
        "source_ids": [source_id],
        "source_records": [{"source_id": source_id, "fragment_id": original_id}],
        "curated": curated,
    }
    return enriched


def build_library(seed_payload: dict[str, Any], working_payload: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    source_order = [
        ("project_seed", seed_payload, True),
        ("chembl_working", working_payload, False),
    ]
    for source_id, payload, curated in source_order:
        for record in payload.get("fragments") or []:
            if not isinstance(record, dict):
                continue
            enriched = _enriched_record(payload, record, source_id, curated)
            key = enriched["canonical_smiles"]
            existing = merged.get(key)
            if existing is None:
                merged[key] = enriched
                continue
            existing["allowed_operations"] = sorted({
                *existing.get("allowed_operations", []),
                *enriched.get("allowed_operations", []),
            })
            existing["source_ids"] = sorted({
                *existing.get("source_ids", []),
                *enriched.get("source_ids", []),
            })
            existing["source_records"] = [
                *existing.get("source_records", []),
                *enriched.get("source_records", []),
            ]
            if enriched.get("source_molecule_ids"):
                existing["source_molecule_ids"] = enriched["source_molecule_ids"]
                existing["source_molecule_count"] = enriched.get("source_molecule_count")

    records = sorted(
        merged.values(),
        key=lambda item: (
            SIZE_RANK[item["size_class"]],
            0 if item.get("curated") else 1,
            item["canonical_smiles"],
        ),
    )
    size_counts = Counter(record["size_class"] for record in records)
    tag_counts = Counter(tag for record in records for tag in record["chemical_tags"])
    operation_counts = Counter(
        operation for record in records for operation in record["allowed_operations"]
    )
    return {
        "schema_version": 2,
        "allowed_operations": sorted(operation_counts),
        "size_classes": SIZE_CLASSES,
        "sources": [
            _source_summary(seed_payload, "project_seed"),
            _source_summary(working_payload, "chembl_working"),
        ],
        "build": {
            "description": (
                "Unified curated and ChEMBL working library with deterministic size classes, "
                "chemical tags, properties, operation permissions, and provenance."
            ),
            "selection_policy": (
                "Size classes are an LLM-selectable action space, not a mandatory execution order."
            ),
            "input_fragment_count": sum(
                len(payload.get("fragments") or []) for _, payload, _ in source_order
            ),
            "output_fragment_count": len(records),
            "deduplicated_fragment_count": sum(
                len(payload.get("fragments") or []) for _, payload, _ in source_order
            ) - len(records),
            "size_class_counts": dict(sorted(size_counts.items(), key=lambda item: SIZE_RANK[item[0]])),
            "chemical_tag_counts": dict(sorted(tag_counts.items())),
            "operation_counts": dict(sorted(operation_counts.items())),
        },
        "fragments": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("molecular_agent/data/fragments.json"),
    )
    parser.add_argument(
        "--working",
        type=Path,
        default=Path("molecular_agent/data/chembl_fragments_working.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("molecular_agent/data/fragments_unified.json"),
    )
    args = parser.parse_args()
    seed_payload = json.loads(args.seed.read_text(encoding="utf-8"))
    working_payload = json.loads(args.working.read_text(encoding="utf-8"))
    output = build_library(seed_payload, working_payload)
    output["build"]["inputs"] = {
        "seed": str(args.seed),
        "working": str(args.working),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "fragments": len(output["fragments"]),
        "size_class_counts": output["build"]["size_class_counts"],
        "deduplicated": output["build"]["deduplicated_fragment_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
