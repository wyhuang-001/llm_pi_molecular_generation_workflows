"""Calibrate a reference only: no browser, LLM, or new candidate optimization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem
from molecular_agent.adapters import configured_adapters
from molecular_agent.editing import EditResult, write_sdf
from molecular_agent.structure import ComplexContext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=Path("input/task.single_edit.json"))
    parser.add_argument("--config", type=Path, default=Path("config.single_edit.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if directory.exists():
        parser.error("Use a new run directory; frozen preparations are not overwritten")
    directory.mkdir(parents=True)
    context = ComplexContext(args.task)
    adapter, _ = configured_adapters(args.config, directory)
    if not hasattr(adapter, "prepare_reference"):
        parser.error("A configured GNINA adapter is required")
    adapter.config["pose_retention"] = {**context.task["pose_retention"],
        "primary_metric": context.task.get("docking_optimization", {}).get("primary_metric", "minimizedAffinity")}
    adapter.config["retained_hetero_residue_names"] = sorted(context.retained_hetero_residues)
    adapter.config["structure_source_identity"] = context.preparation_identity()
    native = directory / "reference-ligand.sdf"
    write_sdf(EditResult(Chem.AddHs(context.ligand, addCoords=True), {}), native, "reference-ligand")
    receptor = context.write_receptor_pdb(directory / "receptor-protein-only.pdb")
    result = adapter.prepare_reference(native, receptor, directory / "docking-reference-baseline")
    print(json.dumps({key: result.get(key) for key in
                      ("status", "error", "valid_seeds", "invalid_seeds", "geometry_reference_path")}, indent=2))
    print("Manifest:", directory / "docking-reference-baseline/calibration.json")
    raise SystemExit(0 if result["status"] == "complete" else 2)


if __name__ == "__main__":
    main()
