#!/usr/bin/env python3
"""Create a conservative, auditable working subset from a fragment library."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


ALLOWED_ELEMENTS = {"C", "N", "O", "S", "F", "Cl", "Br", "I"}


def filter_records(
    records: list[dict[str, Any]],
    min_source_molecules: int,
    require_neutral: bool,
    remove_alerts: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    counts: Counter[str] = Counter()
    catalog = None
    if remove_alerts:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        catalog = FilterCatalog(params)

    selected = []
    for record in records:
        counts["all"] += 1
        if int(record.get("source_molecule_count", 0)) < min_source_molecules:
            continue
        counts["minimum_source_support"] += 1
        molecule = Chem.MolFromSmiles(str(record.get("smiles", "")))
        if molecule is None:
            counts["invalid_smiles"] += 1
            continue
        heavy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 0]
        if any(atom.GetAtomicNum() == 1 for atom in heavy_atoms):
            continue
        counts["no_explicit_hydrogen"] += 1
        if any(atom.GetSymbol() not in ALLOWED_ELEMENTS for atom in heavy_atoms):
            continue
        counts["allowed_elements"] += 1
        if require_neutral and Chem.GetFormalCharge(molecule) != 0:
            continue
        counts["neutral"] += 1
        if any(atom.GetNumRadicalElectrons() for atom in heavy_atoms):
            continue
        counts["no_radicals"] += 1
        if catalog is not None and catalog.HasMatch(molecule):
            continue
        counts["no_pains_brenk_alert"] += 1
        selected.append(record)

    for index, record in enumerate(sorted(selected, key=lambda item: item["smiles"]), start=1):
        record["fragment_id"] = f"chembl-brics-filtered-{index:06d}"
        record["name"] = f"ChEMBL BRICS filtered fragment {index}"
    return selected, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-source-molecules", type=int, default=2)
    parser.add_argument("--allow-charged", action="store_true")
    parser.add_argument("--keep-alerts", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records = payload.get("fragments", [])
    selected, counts = filter_records(
        records,
        min_source_molecules=args.min_source_molecules,
        require_neutral=not args.allow_charged,
        remove_alerts=not args.keep_alerts,
    )
    output = {
        "schema_version": payload.get("schema_version", 1),
        "allowed_operations": payload.get("allowed_operations", []),
        "source": {
            **(payload.get("source") or {}),
            "derived_from": str(args.input.resolve()),
            "working_subset": True,
            "working_subset_filters": {
                "min_source_molecules": args.min_source_molecules,
                "allowed_elements": sorted(ALLOWED_ELEMENTS),
                "require_neutral": not args.allow_charged,
                "remove_pains_brenk_alerts": not args.keep_alerts,
            },
            "input_fragment_count": len(records),
            "output_fragment_count": len(selected),
        },
        "filters": {
            **(payload.get("filters") or {}),
            "working_subset": True,
            "min_source_molecules": args.min_source_molecules,
            "allowed_elements": sorted(ALLOWED_ELEMENTS),
            "require_neutral": not args.allow_charged,
            "remove_pains_brenk_alerts": not args.keep_alerts,
        },
        "filter_counts": dict(counts),
        "fragments": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "input": len(records), "output_fragments": len(selected), "filter_counts": counts}))


if __name__ == "__main__":
    main()
